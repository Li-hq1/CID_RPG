"""Exercise the standalone entry point with in-memory recommendation data."""

import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from joint_training import __main__ as launcher
from joint_training.checkpoint import read_checkpoint


class EpochBoundaryInterrupt(Exception):
    pass


def synthetic_data(config, accelerator):
    del config, accelerator
    sequences = [(1, 2, 3), (2, 3, 4), (3, 4, 5),
                 (4, 5, 1), (5, 1, 2), (1, 3, 5)]
    train = [{"input_ids": torch.tensor([left, middle, 0, 0]),
              "attention_mask": torch.tensor([1, 1, 0, 0]),
              "labels": torch.tensor([middle, target, -100, -100]),
              "seq_lens": torch.tensor(2)} for left, middle, target in sequences]
    validation = [{**row, "labels": row["labels"][1:2]} for row in train[:3]]
    test = [{**row, "labels": row["labels"][1:2]} for row in train[3:]]
    splits = {"train": train, "val": validation, "test": test}
    tokenizer = SimpleNamespace(eos_token=None, item2tokens=range(5),
                                _sample_source={"fixture_sequences": sequences},
                                tokenize=lambda values: values)
    dataset = SimpleNamespace(split=lambda: splits)
    raw = np.random.default_rng(93).normal(size=(5, 6)).astype(np.float32)
    return dataset, tokenizer, raw, torch.arange(1, 6), {"fixture": "joint_resume_v1"}


def fixture_config(output_dir, run_id):
    return {
        "data": {"max_item_seq_len": 4, "sent_emb_dim": 6},
        "ae": {"architecture": "linear", "hidden_dims": [8], "latent_dim": 3,
               "batch_size": 5, "update_every": 2, "lr": .001, "warmup_ratio": .25},
        "recommender": {
            "encoder_type": "transformer", "n_embd": 8, "n_layer": 1, "n_head": 2,
            "n_inner": 16, "diffloss_w": 16, "diffloss_d": 1,
            "num_sampling_steps": 2, "rectified_flow_steps": 10,
            "train_batch_size": 2, "embd_pdrop": .15, "attn_pdrop": .1,
            "tiger_uncond_prob": .2, "tiger_guidance_weight": 0.,
            "lr": .002, "warmup_ratio": .25,
        },
        "contrastive": {"enabled": True, "max_queries": 3},
        "evaluation": {"eval_sampling_mode": "knn", "eval_interval": 2,
                       "eval_batch_size": 2, "topk": [1, 3], "val_metric": "ndcg@3"},
        "training": {"epochs": 3, "seed": 47, "device": "cpu", "cpu_threads": 2,
                     "num_workers": 0, "output_dir": str(output_dir), "run_id": run_id},
    }


class JointEntryPointResumeTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        self.data_patch = patch.object(launcher, "load_data", side_effect=synthetic_data)
        self.logging_patch = patch.object(launcher, "initialize_logging")
        self.data_patch.start()
        self.logging_patch.start()
        self.addCleanup(self.logging_patch.stop)
        self.addCleanup(self.data_patch.stop)
        self.addCleanup(torch.set_num_threads, self.threads)

    def write_config(self, root, run_id):
        path = root / f"{run_id}.json"
        config = fixture_config(root / "runs", run_id)
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def assert_tree_equal(self, expected, actual):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key, value in expected.items():
                self.assert_tree_equal(value, actual[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(expected, actual):
                self.assert_tree_equal(left, right)
        else:
            self.assertEqual(actual, expected)

    def test_main_resume_matches_uninterrupted_shuffled_epochs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            uninterrupted = self.write_config(root, "uninterrupted")
            interrupted = self.write_config(root, "interrupted")
            expected_summary = launcher.main(["--config", str(uninterrupted)])
            save = launcher.save_checkpoint

            def stop_after_first_epoch(path, trainer, **metadata):
                save(path, trainer, **metadata)
                if Path(path).name == "latest.pt" and metadata["epoch"] == 1:
                    raise EpochBoundaryInterrupt()

            with patch.object(launcher, "save_checkpoint", side_effect=stop_after_first_epoch):
                with self.assertRaises(EpochBoundaryInterrupt):
                    launcher.main(["--config", str(interrupted)])
            expected_dir = root / "runs" / "Beauty" / "uninterrupted"
            resumed_dir = root / "runs" / "Beauty" / "interrupted"
            boundary = read_checkpoint(resumed_dir / "latest.pt")
            self.assertEqual(boundary["epoch"], 1)
            self.assertEqual(boundary["last_validation_epoch"], 0)
            self.assertEqual(boundary["trainer"]["global_step"], 3)
            resumed_summary = launcher.main(["--resume", str(resumed_dir / "latest.pt")])
            expected = read_checkpoint(expected_dir / "latest.pt")
            actual = read_checkpoint(resumed_dir / "latest.pt")
            self.assert_tree_equal(expected["model"], actual["model"])
            self.assert_tree_equal(expected["trainer"], actual["trainer"])
            self.assert_tree_equal(expected["rng"], actual["rng"])
            self.assertEqual(actual["trainer"]["global_step"], 9)
            self.assertEqual(actual["trainer"]["ae_steps"], 4)
            self.assertEqual(actual["trainer"]["rec_steps"], 9)
            for key in ("best_epoch", "best_score", "stale_epochs", "last_validation_epoch", "source"):
                self.assert_tree_equal(expected[key], actual[key])
            expected_rows = [json.loads(row) for row in (expected_dir / "metrics.jsonl").read_text().splitlines()]
            actual_rows = [json.loads(row) for row in (resumed_dir / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(actual_rows, expected_rows)
            self.assertEqual(resumed_summary["test_results"], expected_summary["test_results"])

    def test_main_eval_only_loads_joint_weights_and_allows_new_evaluation_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self.write_config(root, "evaluation")
            launcher.main(["--config", str(config_path)])
            directory = root / "runs" / "Beauty" / "evaluation"
            path = directory / "best.pt"
            checkpoint_bytes = path.read_bytes()
            expected = read_checkpoint(path)["model"]
            evaluate = launcher.evaluate
            captured = {}

            def observe(model, loader, config, tokenizer):
                captured["model"] = {key: value.clone() for key, value in model.state_dict().items()}
                result = evaluate(model, loader, config, tokenizer)
                captured["catalog"] = model.catalog.clone()
                return result

            with patch.object(launcher, "evaluate", side_effect=observe), \
                    patch.object(launcher, "initialize_random_ae",
                                 side_effect=AssertionError("eval requires no fresh AE")), \
                    patch.object(launcher.JointTrainer, "step", side_effect=AssertionError("eval must not train")):
                result = launcher.main(["--eval-only", str(path), "--set", "evaluation.topk=[1,2]",
                                        "--set", "evaluation.val_metric=ndcg@2",
                                        "--set", "evaluation.tiger_knn_metric=cosine",
                                        "--set", "evaluation.eval_candidate_metric=cosine",
                                        "--set", "evaluation.eval_batch_size=1"])
            self.assert_tree_equal(expected, captured["model"])
            self.assertEqual(captured["catalog"].shape, (6, 3))
            self.assertTrue(torch.equal(captured["catalog"][0], torch.zeros(3)))
            self.assertEqual(set(result), {"recall@1", "recall@2", "ndcg@1", "ndcg@2", "n_visited_items"})
            self.assertTrue(all(math.isfinite(value) for value in result.values()))
            self.assertEqual(path.read_bytes(), checkpoint_bytes)
            saved = json.loads((directory / "eval_results.json").read_text())
            self.assertEqual(saved["test_results"], result)
            self.assertEqual(saved["evaluation"]["topk"], [1, 2])
            self.assertEqual(saved["evaluation"]["eval_candidate_metric"], "cosine")
            self.assertEqual(saved["evaluation"]["tiger_knn_metric"], "cosine")

    def test_validation_only_training_never_evaluates_test_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self.write_config(root, "validation_only")
            captured = {}
            evaluate = launcher.evaluate

            def data(config, accelerator):
                result = synthetic_data(config, accelerator)
                captured["test_dataset"] = result[0].split()["test"]
                return result

            def validation_only(model, loader, config, tokenizer):
                self.assertIsNot(loader.dataset, captured["test_dataset"])
                return evaluate(model, loader, config, tokenizer)

            with patch.object(launcher, "load_data", side_effect=data), \
                    patch.object(launcher, "evaluate", side_effect=validation_only) as observed:
                summary = launcher.main(["--config", str(config_path), "--set", "training.evaluate_test=false"])
            self.assertIsNone(summary["test_results"])
            self.assertEqual(observed.call_count, 2)
            self.assertTrue(math.isfinite(summary["best_val_score"]))
            catalog = np.load(root / "runs/Beauty/validation_only/catalog_latents.npy")
            self.assertEqual(catalog.shape, (5, 3))


if __name__ == "__main__":
    unittest.main()
