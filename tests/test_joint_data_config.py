from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from datasets import Dataset
import numpy as np
import torch

from compression.contrec_vae.cache import array_hash
from compression.geometry_ae.pipeline import fit_input_statistics
from data.tigerdiff_data import TigerDiffTokenizer
from joint_training.config import JointConfig, parse_config
from joint_training.data import JointTokenizer, initialize_random_ae, load_data


class JointConfigTest(unittest.TestCase):
    def test_independent_defaults_and_typed_overrides(self):
        config, args = parse_config(["--set", "ae.lr=0.0002", "--set", "ae.hidden_dims=[12,8]",
                                     "--set", "training.deterministic=false", "--contrastive"])
        self.assertEqual(config.data.category, "Beauty")
        self.assertEqual(config.ae.lr, 0.0002)
        self.assertEqual(config.ae.hidden_dims, (12, 8))
        self.assertFalse(config.training.deterministic)
        self.assertTrue(config.contrastive.enabled)
        self.assertEqual(config.recommender.lr, 5e-4)
        self.assertTrue(args.override_mapping["contrastive"]["enabled"])
        self.assertEqual(JointConfig.from_dict(config.to_dict()), config)

    def test_json_partial_config_and_checkpoint_base_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.json"
            path.write_text(json.dumps({"ae": {"lr": 0.001}, "training": {"run_id": "json_run"}}))
            base = JointConfig().to_dict()
            base["training"]["epochs"] = 25
            base["contrastive"]["enabled"] = True
            config, args = parse_config(["--config", str(path), "--set", "ae.lr=0.002", "--run-id", "cli_run"],
                                        base_mapping=base)
        self.assertEqual(config.training.epochs, 25)
        self.assertEqual(config.training.run_id, "cli_run")
        self.assertEqual(config.ae.lr, 0.002)
        self.assertTrue(config.contrastive.enabled)
        self.assertNotIn("epochs", args.override_mapping["training"])

    def test_unknown_mistyped_and_invalid_fields_are_rejected(self):
        for value in ("ae.typo=1", "training.epochs=true", "ae.lr=nan", 'ae.lr="0.001"',
                      "ae.frozen=1", "ae.hidden_dims=[8,true]", "contrastive.positions=all_positions",
                      "ae.relation_mode=local", "evaluation.topk=[]", "training.run_id=../escape",
                      "recommender.hstu_attention_dim=3", "recommender.num_sampling_steps=1",
                      "recommender.rectified_flow_steps=1", "recommender.hstu_output_dropout=1",
                      "recommender.hstu_attention_dropout=1", "recommender.supervision_mode=last_item",
                      "ae.init_mode=random", "ae.init_artifact=cache/unused", "contrastive.mode=ae_only",
                      "contrastive.mode=both", "contrastive.mode=ae_hstu", "contrastive.lambda_ae=0.01",
                      "contrastive.lambda_rec=0.01", "contrastive.start_epoch_ae=1",
                      "contrastive.start_epoch_rec=1", "contrastive.ramp_epochs_ae=0",
                      "contrastive.ramp_epochs_rec=0", "contrastive.loss_type=infonce"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_config(["--set", value])

        with self.assertRaises(ValueError):
            JointConfig.from_dict({"unknown": {}})
        with self.assertRaises(ValueError):
            JointConfig.from_dict({"ae": {"lr": float("inf")}})

    def test_hstu_dropout_upper_bound_only_applies_to_hstu(self):
        config, _ = parse_config(["--set", "recommender.hstu_output_dropout=0.999",
                                  "--set", "recommender.hstu_attention_dropout=0.999"])
        self.assertEqual(config.recommender.hstu_output_dropout, 0.999)
        self.assertEqual(config.recommender.hstu_attention_dropout, 0.999)
        config, _ = parse_config(["--set", "recommender.encoder_type=transformer",
                                  "--set", "recommender.hstu_output_dropout=1",
                                  "--set", "recommender.hstu_attention_dropout=1"])
        self.assertEqual(config.recommender.hstu_output_dropout, 1.0)
        self.assertEqual(config.recommender.hstu_attention_dropout, 1.0)

    def test_resume_and_eval_modes_and_smoke_budget(self):
        config, args = parse_config(["--smoke-test", "--resume", "latest.pt"])
        self.assertEqual(args.resume, "latest.pt")
        self.assertTrue(args.smoke_test)
        self.assertEqual(config.training.epochs, 2)
        self.assertEqual(config.training.max_steps, 4)
        self.assertEqual(config.ae.hidden_dims, (1024, 512))
        with self.assertRaises(SystemExit):
            parse_config(["--resume", "latest.pt", "--eval-only", "best.pt"])
        with self.assertRaises(SystemExit):
            parse_config(["--ae-init", "cache/unused"])
        with self.assertRaises(SystemExit):
            parse_config(["--ae-random-init"])


class JointDataTest(unittest.TestCase):
    def test_tokenizer_bypasses_frozen_embedding_initialization_and_preserves_labels(self):
        dataset = SimpleNamespace(cache_dir="cache/fixture", n_items=6,
                                  item2id={str(item): item for item in range(1, 6)}, user2id={"u": 1},
                                  id_mapping={"id2item": ["[PAD]"] + [str(item) for item in range(1, 6)]})
        settings = {"max_item_seq_len": 3, "split": "leave_one_out", "num_proc": 1,
                    "accelerator": SimpleNamespace(is_main_process=True)}
        with patch.object(TigerDiffTokenizer, "_init_tokenizer", side_effect=AssertionError("frozen cache")):
            tokenizer = JointTokenizer(settings, dataset)
        result = tokenizer.tokenize_function({"item_seq": [["1", "2", "3", "4", "5"]]}, "train")
        self.assertEqual(result["labels"], [[2, 3, 4], [-100, -100, 5]])
        self.assertEqual(len(tokenizer.item2tokens), 5)
        self.assertFalse(hasattr(tokenizer, "sent_embs"))
        splits = {"train": Dataset.from_dict({"item_seq": [["1", "2", "3"]]})}
        with patch.object(TigerDiffTokenizer, "tokenize", return_value={}) as inherited:
            tokenizer.tokenize(splits)
            initial = tokenizer._tokenized_cache_path()
            tokenizer.tokenize(splits)
            self.assertEqual(initial, tokenizer._tokenized_cache_path())
            tokenizer.tokenize({"train": Dataset.from_dict({"item_seq": [["1", "2", "4"]]})})
            self.assertNotEqual(initial, tokenizer._tokenized_cache_path())
        self.assertEqual(inherited.call_count, 3)

    def test_load_data_retains_one_based_training_ids_and_raw_source(self):
        ids = ["[PAD]", "1", "2", "3", "4"]
        splits = {"train": {"item_seq": [["1", "3", "1"]]}}
        dataset = SimpleNamespace(cache_dir="cache/fixture", n_items=len(ids), item2id={v: i for i, v in enumerate(ids)},
                                  user2id={"u": 1}, id_mapping={"id2item": ids}, split=lambda: splits)
        config = JointConfig()
        config.data.sent_emb_dim = 3
        raw = np.ones((4, 3), dtype=np.float32)
        with patch("joint_training.data.AmazonReviews2014", return_value=dataset), patch(
                "joint_training.data.load_item_sentence_embeddings", return_value=raw), patch.object(
                TigerDiffTokenizer, "tokenize", side_effect=AssertionError("premature tokenization")):
            actual, tokenizer, values, train_ids, source = load_data(config, SimpleNamespace(is_main_process=True))
        self.assertIs(actual, dataset)
        self.assertIs(values, raw)
        self.assertEqual(train_ids.tolist(), [1, 3])
        self.assertEqual(source["embeddings_sha256"], array_hash(raw))
        self.assertEqual(len(tokenizer.item2tokens), 4)

    def test_random_ae_fits_training_items_and_restores_rng(self):
        config = JointConfig()
        config.ae = replace(config.ae, architecture="mlp", hidden_dims=(8, 6), latent_dim=3)
        config.training.seed = 11
        raw = np.arange(24, dtype=np.float32).reshape(6, 4)
        raw[-1] = 100
        train_ids = torch.tensor([1, 3, 4])
        torch.manual_seed(3)
        expected_noise = torch.randn(4)
        torch.manual_seed(3)
        first = initialize_random_ae(config, raw, train_ids)
        actual_noise = torch.randn(4)
        second = initialize_random_ae(config, raw, train_ids)
        mean, scale = fit_input_statistics(raw, train_ids.numpy() - 1)
        torch.testing.assert_close(first.input_mean, mean)
        torch.testing.assert_close(first.input_scale, scale)
        self.assertFalse(torch.equal(first.input_mean, torch.from_numpy(raw.mean(0).astype(np.float32))))
        torch.testing.assert_close(actual_noise, expected_noise)
        for key, value in first.state_dict().items():
            torch.testing.assert_close(second.state_dict()[key], value)
        self.assertFalse(first.training)
        self.assertFalse(first.input_mean.requires_grad)
        config.training.seed = 12
        changed = initialize_random_ae(config, raw, train_ids)
        self.assertFalse(torch.equal(changed.encoder[0].weight, first.encoder[0].weight))
        with self.assertRaisesRegex(ValueError, "unique in-range"):
            initialize_random_ae(config, raw, torch.tensor([1, 1]))
        with self.assertRaisesRegex(ValueError, "no usable variance"):
            initialize_random_ae(config, np.ones_like(raw), train_ids)


if __name__ == "__main__":
    unittest.main()
