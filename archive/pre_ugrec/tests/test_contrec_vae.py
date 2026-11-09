from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from config import CONFIG
from compression.contrec_vae import SigmaVae, VAEConfig, configure_loss
from compression.contrec_vae.cache import (
    artifact_directory, atomic_write, load_features, read_manifest, source_metadata, training_item_mask,
)
from compression.contrec_vae.pipeline import (
    batch_indices, export_artifact, load_best, split_items, standardize_latents, train_artifact,
)
from data.embeddings import sentence_embedding_cache_path
from data.tigerdiff_data import TigerDiffTokenizer
from engine.checkpoint import load_checkpoint, save_checkpoint
from misc.config import parse_config
from models.tigerdiff import TigerDiff
from tests.test_tigerdiff import make_batch, make_config


def fixture(directory):
    ids = ["[PAD]"] + [f"item{i}" for i in range(1, 13)]
    splits = {"train": {"item_seq": [ids[1:11]]}}
    dataset = SimpleNamespace(
        cache_dir=str(Path(directory) / "AmazonReviews2014" / "Beauty"),
        n_items=len(ids), id_mapping={"id2item": ids},
        item2id={item: i for i, item in enumerate(ids)}, user2id={"u": 1},
        split=lambda: splits,
    )
    dataset.split_data = splits
    config = {**CONFIG, "category": "Beauty", "dataset": "AmazonReviews2014", "device": "cpu",
              "sent_emb_dim": 6, "num_proc": 1, "max_item_seq_len": 4}
    embeddings = np.random.default_rng(42).normal(size=(12, 6)).astype(np.float32)
    path = Path(sentence_embedding_cache_path(config, dataset.cache_dir))
    path.parent.mkdir(parents=True)
    embeddings.tofile(path)
    return dataset, config, embeddings, training_item_mask(dataset)


class ContRecVaeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_masking_preserves_input_and_gradients_and_eval_averages_branches(self):
        model = SigmaVae(6, 8, embeds_num=10, mask_ratio=1.0)
        x = torch.randn(4, 6)
        original = x.clone()
        reconstructed = model(x)[0]
        torch.testing.assert_close(x, original, rtol=0, atol=0)
        torch.nn.functional.mse_loss(reconstructed, x).backward()
        for encoder in model.encoders:
            self.assertTrue(torch.isfinite(encoder[0].weight.grad).all())
        self.assertIsNotNone(model.pos.weight.grad)
        model.eval()
        _, encoded = model.encode(x)
        torch.testing.assert_close(model.latent(x), torch.stack(encoded).mean(0))
        torch.testing.assert_close(model.valid(x)[0], model.decode(model.latent(x)))
        with torch.no_grad():
            model.pos.weight.fill_(1e6)
        torch.testing.assert_close(model.valid(x)[0], model.decode(model.latent(x)))

    def test_configure_loss_matches_contrec_combination_and_backprops(self):
        model = SigmaVae(6, 8, embeds_num=10)
        x = torch.randn(4, 6)
        x_hat, mus, encoded = model(x)
        live = configure_loss(x, x_hat, mus, encoded, beta=0.5, gamma=0.5, twins_lambda=0.005)
        self.assertTrue(torch.isfinite(live["loss"]))
        live["loss"].backward()
        self.assertTrue(torch.isfinite(model.encoders[0][0].weight.grad).all())
        only_recon = configure_loss(
            x, x_hat.detach(), [m.detach() for m in mus],
            [e.detach() for e in encoded], beta=0.0, gamma=0.0,
        )
        torch.testing.assert_close(
            only_recon["loss"], torch.nn.functional.mse_loss(x_hat.detach(), x),
        )

    def test_noise_roundtrip_and_default_dimension(self):
        model = SigmaVae(6, dim=VAEConfig().latent_dim, embeds_num=10).eval()
        self.assertEqual(model.latent(torch.randn(2, 6)).shape, (2, VAEConfig().latent_dim))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            torch.save(model.state_dict(), path)
            restored = SigmaVae(6, dim=VAEConfig().latent_dim, seed=99).eval()
            restored.load_state_dict(torch.load(path, weights_only=True))
            x = torch.randn(3, 6)
            torch.testing.assert_close(model.latent(x), restored.latent(x), rtol=0, atol=0)
            self.assertIn("epsilon", restored.state_dict())
            self.assertIn("sigma", restored.state_dict())

    def test_atomic_failure_preserves_previous_file_and_removes_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.bin"
            path.write_bytes(b"previous")

            def fail(stream):
                stream.write(b"partial")
                raise OSError("disk full")

            with self.assertRaisesRegex(OSError, "disk full"):
                atomic_write(path, fail)
            self.assertEqual(path.read_bytes(), b"previous")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_population_splits_singleton_batches_and_statistics(self):
        mask = np.array([True] * 20 + [False] * 3)
        splits = split_items(mask, VAEConfig())
        self.assertEqual({key: len(value) for key, value in splits.items()},
                         {"test": 2, "valid": 3, "train": 15})
        all_indices = np.concatenate(list(splits.values()))
        self.assertEqual(len(np.unique(all_indices)), 20)
        self.assertTrue(mask[all_indices].all())
        batches = batch_indices(np.arange(9), 4, generator=torch.Generator().manual_seed(1), training=True)
        self.assertEqual([len(batch) for batch in batches], [4, 5])
        self.assertEqual(set(np.concatenate(batches)), set(range(9)))
        latent = np.array([[1., 2.], [3., 2.], [100., 200.]], dtype=np.float32)
        features, mean, std = standardize_latents(latent, np.array([True, True, False]))
        np.testing.assert_array_equal(mean, [2., 2.])
        np.testing.assert_allclose(std, [1., 1e-6])
        np.testing.assert_allclose(features[:2, 0], [-1., 1.])
        with self.assertRaisesRegex(ValueError, "four training items"):
            split_items(np.ones(3, dtype=bool), VAEConfig())

    def test_fingerprint_tracks_sources_settings_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, config, embeddings, mask = fixture(directory)
            source = source_metadata(embeddings, dataset, config, mask)
            settings = VAEConfig().to_dict()
            original = artifact_directory(dataset.cache_dir, source, settings)
            changed = embeddings.copy()
            changed[0, 0] += 1
            self.assertNotEqual(original, artifact_directory(dataset.cache_dir,
                source_metadata(changed, dataset, config, mask), settings))
            self.assertNotEqual(original, artifact_directory(dataset.cache_dir,
                source_metadata(embeddings, dataset, config, ~mask), settings))
            dataset.id_mapping["id2item"][1:3] = dataset.id_mapping["id2item"][1:3][::-1]
            self.assertNotEqual(original, artifact_directory(dataset.cache_dir,
                source_metadata(embeddings, dataset, config, mask), settings))
            self.assertNotEqual(original, artifact_directory(dataset.cache_dir, source, {**settings, "seed": 7}))

    def test_train_export_load_tokenizer_and_recommendation_backward(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, config, embeddings, mask = fixture(directory)
            original = embeddings.copy()
            settings = VAEConfig(epochs=2, train_batch_size=3, eval_batch_size=4)
            root = train_artifact(embeddings, dataset, config, mask, settings)
            np.testing.assert_array_equal(original, embeddings)
            manifest = read_manifest(root)
            source = source_metadata(embeddings, dataset, config, mask)
            model, checkpoint, _ = load_best(root, source)
            raw = np.load(root / "raw_latents.npy")
            with torch.no_grad():
                expected = model.latent(torch.from_numpy(embeddings)).numpy()
            np.testing.assert_allclose(raw, expected, atol=1e-6)
            self.assertEqual(raw.shape, (12, settings.latent_dim))
            self.assertIn(checkpoint["epoch"], [1, 2])
            self.assertIn("train_mu", checkpoint["history"][0])
            self.assertIn("train_twins", checkpoint["history"][0])
            with patch("compression.contrec_vae.pipeline.SigmaVae", side_effect=AssertionError("retrained")):
                self.assertEqual(train_artifact(embeddings, dataset, config, mask, settings), root)
            config.update(sent_emb_compression="contrec_vae", contrec_vae_cache=str(root))
            features, identity = load_features(root, dataset, config)
            np.testing.assert_allclose(features[mask].mean(0), 0, atol=1e-4)
            self.assertEqual(identity, manifest["artifact_id"])
            with patch.object(TigerDiffTokenizer, "log"), patch.object(
                    TigerDiffTokenizer, "_load_sent_emb", side_effect=AssertionError("PCA path")):
                tokenizer = TigerDiffTokenizer(config, dataset)
            np.testing.assert_array_equal(features, tokenizer.sent_embs)
            self.assertTrue(Path(tokenizer._tokenized_cache_path()).is_relative_to(root))
            rec_config = {**config, **make_config()}
            recommender = TigerDiff(rec_config, dataset, tokenizer)
            loss = recommender(make_batch()).loss
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNotNone(recommender.input_proj.linear.weight.grad)
            self.assertFalse(recommender.item_id2embs.requires_grad)
            export_artifact(root, embeddings, dataset, config, mask)
            reexported, reexported_identity = load_features(root, dataset, config)
            np.testing.assert_array_equal(features, reexported)
            self.assertEqual(identity, reexported_identity)

    def test_incomplete_corrupt_and_mismatched_artifacts_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, config, embeddings, mask = fixture(directory)
            root = train_artifact(embeddings, dataset, config, mask, VAEConfig(epochs=1))
            altered = {**config, "sent_emb_prompt": "changed"}
            with self.assertRaises(FileNotFoundError):
                load_features(root, dataset, altered)
            dataset.id_mapping["id2item"][1] = "changed-item"
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                load_features(root, dataset, config)
            dataset.id_mapping["id2item"][1] = "item1"
            dataset.split_data["train"]["item_seq"][0].pop()
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                load_features(root, dataset, config)
            path = Path(sentence_embedding_cache_path(config, dataset.cache_dir))
            embeddings[0, 0] += 1
            embeddings.tofile(path)
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                load_features(root, dataset, config)
            with (root / "latents.npy").open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                read_manifest(root)
            (root / "latents.npy").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Incomplete"):
                read_manifest(root)

    def test_smoke_artifact_requires_smoke_recommendation(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, config, embeddings, mask = fixture(directory)
            root = train_artifact(embeddings, dataset, config, mask, VAEConfig(epochs=1, smoke_items=8))
            self.assertIn("smoke", root.parts)
            with self.assertRaisesRegex(ValueError, "Smoke VAE"):
                load_features(root, dataset, config)
            self.assertEqual(
                load_features(root, dataset, {**config, "smoke_test": True})[0].shape,
                (12, VAEConfig().latent_dim),
            )

    def test_checkpoint_feature_identity_validated_before_weights_change(self):
        accelerator = SimpleNamespace(is_main_process=True, unwrap_model=lambda model: model)
        model = torch.nn.Linear(2, 2)
        model.config = {"sent_emb_compression": "contrec_vae", "contrec_vae_artifact_id": "artifact-a"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rec.pt"
            save_checkpoint(model, path, accelerator)
            self.assertEqual(load_checkpoint(model, path, "cpu")["contrec_vae_artifact_id"], "artifact-a")
            original = model.weight.detach().clone()
            for config in ({"contrec_vae_artifact_id": "artifact-b"}, {},
                           {"sent_emb_compression": "contrec_vae"}):
                model.config = config
                with self.assertRaisesRegex(ValueError, "VAE"):
                    load_checkpoint(model, path, "cpu")
                torch.testing.assert_close(model.weight, original)
            torch.save(model.state_dict(), path)
            model.config = {"contrec_vae_artifact_id": "artifact-a"}
            with self.assertRaisesRegex(ValueError, "artifact mismatch"):
                load_checkpoint(model, path, "cpu")

    def test_cli_requires_cache_for_contrec_vae_and_rejects_wrong_encoder_dim(self):
        self.assertEqual(parse_config([])[0]["sent_emb_compression"], CONFIG["sent_emb_compression"])
        with self.assertRaisesRegex(ValueError, "contrec_vae_cache"):
            parse_config(["--sent_emb_compression=contrec_vae", "--contrec_vae_cache="])
        config, _ = parse_config(["--config=config_transformer.py", "--sent_emb_compression=contrec_vae",
                                 "--contrec_vae_cache=cache/example", "--smoke-test"])
        self.assertEqual(config["contrec_vae_cache"], "cache/example")
        self.assertTrue(config["smoke_test"])
        self.assertEqual(parse_config(["--cache_dir=/tmp/custom-cache"])[0]["cache_dir"], "/tmp/custom-cache")
        with self.assertRaisesRegex(ValueError, "2560"):
            parse_config(["--sent_emb_compression=contrec_vae", "--contrec_vae_cache=cache/example",
                          "--sent_emb_dim=768"])


if __name__ == "__main__":
    unittest.main()
