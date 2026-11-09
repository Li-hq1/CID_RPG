"""Numerical, provenance and downstream checks for independent compression."""

from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from compression.geometry_ae import AEConfig, GeometryAE
from compression.geometry_ae.cache import artifact_directory, experiment_key, load_features, read_manifest, source_metadata
from compression.geometry_ae.experiments import (ablation_configs, baseline_metrics, recommendation_commands,
                                                 run_ablation, summarize_recommendation)
from compression.geometry_ae.metrics import geometry_metrics
from compression.geometry_ae.model import local_relation_kl, loss_terms, objective_weights, relation_kl
from compression.geometry_ae.neighborhood import LocalCandidates, latent_statistics
from compression.geometry_ae.pipeline import fit_input_statistics, load_best, split_items, train_artifact, export_artifact
from data.tigerdiff_data import TigerDiffTokenizer
from engine.checkpoint import save_checkpoint, load_checkpoint
from misc.config import parse_config
from models.tigerdiff import TigerDiff
from tests.fixtures import fixture
from tests.test_tigerdiff import make_config, make_batch


class GeometryAETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_statistics_and_subsample_keep_holdout_unchanged(self):
        mask = np.ones(100, dtype=bool)
        base = AEConfig()
        full, sub = split_items(mask, base), split_items(mask, replace(base, train_items=20))
        for key in ("valid", "test"):
            np.testing.assert_array_equal(full[key], sub[key])
        np.testing.assert_array_equal(sub["train"], full["train"][:20])
        x = np.array([[1, 3], [3, 5], [100, 100]], dtype=np.float32)
        mean, scale = fit_input_statistics(x, np.array([0, 1]))
        np.testing.assert_array_equal(mean.numpy(), [2, 4])
        self.assertAlmostEqual(scale.item(), 1.)
        with self.assertRaisesRegex(ValueError, "variance"):
            fit_input_statistics(np.ones((3, 2), dtype=np.float32), np.arange(3))

    def test_losses_prevent_constant_and_repeated_dimensions(self):
        torch.manual_seed(7)
        x = torch.randn(512, 4)
        z = torch.randn(512, 4, requires_grad=True)
        terms = loss_terms(x, x, z, torch.tensor(1.))
        repeated = x[:, :1].repeat(1, 4).requires_grad_()
        duplicate_terms = loss_terms(x, x, repeated, torch.tensor(1.))
        self.assertLess(terms["correlation"].item(), .02)
        self.assertGreater(duplicate_terms["correlation"].item(), .99)
        constant_terms = loss_terms(x, x, torch.zeros_like(z, requires_grad=True), torch.tensor(1.))
        self.assertGreater(constant_terms["variance"].item(), .99)
        sum(constant_terms.values()).backward()
        sum(terms.values()).backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        torch.testing.assert_close(terms["reconstruction"], torch.tensor(0.))
        self.assertLess(relation_kl(x, x).abs().item(), .1)
        for batch in (2, 7, 31):
            loss = relation_kl(x[:batch], z[:batch])
            self.assertTrue(torch.isfinite(loss))
        singleton = relation_kl(x[:1], z[:1])
        self.assertEqual(singleton.item(), 0.)

    def test_warmup_and_geometry_diagnostics(self):
        config = AEConfig(relation_mode="batch")
        self.assertEqual(objective_weights(config, 10)["relation"], 0)
        self.assertAlmostEqual(objective_weights(config, 11)["relation"], .005)
        self.assertAlmostEqual(objective_weights(config, 30)["relation"], .1)
        rng = np.random.default_rng(42)
        repeated = np.repeat(rng.normal(size=(100, 1)), 4, axis=1)
        self.assertAlmostEqual(geometry_metrics(repeated)["effective_rank"], 1., places=5)
        self.assertEqual(geometry_metrics(np.zeros((4, 3)))["effective_rank"], 0.)

    def test_legacy_config_identity_and_local_cli(self):
        from compression.geometry_ae.__main__ import parse_args
        config = AEConfig(relation_mode="batch")
        legacy = {key: value for key, value in asdict(config).items()
                  if not key.startswith("local_") and key != "relation_mode"}
        legacy["hidden_dims"] = list(config.hidden_dims)
        self.assertEqual(config.to_dict(), legacy)
        self.assertEqual(AEConfig.from_dict(legacy).to_dict(), legacy)
        self.assertEqual(experiment_key({}, config.to_dict()), experiment_key({}, legacy))
        local = replace(config, relation_mode="local")
        self.assertNotEqual(experiment_key({}, local.to_dict()), experiment_key({}, legacy))
        self.assertNotEqual(experiment_key({}, local.to_dict()),
                            experiment_key({}, replace(local, local_random_count=8).to_dict()))
        args = parse_args(["--relation_mode=local", "--local_random_count=4"])
        self.assertEqual(args.relation_mode, "local")
        self.assertEqual(args.local_random_count, 4)
        self.assertEqual(parse_args([]).relation_mode, AEConfig().relation_mode)
        none = replace(config, relation_mode="none")
        self.assertEqual(none.to_dict()["relation_mode"], "none")
        self.assertEqual(objective_weights(none, 100)["relation"], 0.)
        self.assertNotEqual(experiment_key({}, none.to_dict()), experiment_key({}, legacy))
        self.assertNotEqual(experiment_key({}, none.to_dict()), experiment_key({}, local.to_dict()))
        self.assertEqual(parse_args(["--relation_mode=none"]).relation_mode, "none")
        for changes in ({"relation_mode": "other"}, {"local_near_count": 33},
                        {"local_middle_pool": 32}, {"local_random_count": 0}, {"local_version": 2}):
            with self.assertRaises(ValueError):
                replace(config, **changes).validate()

    def test_local_candidate_population_order_and_sampling(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(120, 6)).astype(np.float32)
        train = rng.permutation(80)
        anchors = np.array([0, 12, 90, 100])
        config = AEConfig(relation_mode="local", local_near_pool=8, local_middle_pool=24,
                          local_near_count=3, local_middle_count=4, local_random_count=5, eval_batch_size=3)
        sampler = LocalCandidates(x, train, anchors, config)
        normalized = x / np.linalg.norm(x, axis=1, keepdims=True)
        for anchor in anchors:
            available = train[train != anchor]
            expected = available[np.argsort(-(normalized[available] @ normalized[anchor]))][:24]
            np.testing.assert_array_equal(sampler.neighbors[int(anchor)], expected)
        samples = sampler.sample(anchors, np.random.default_rng(42))
        np.testing.assert_array_equal(samples, sampler.sample(anchors, np.random.default_rng(42)))
        self.assertFalse(np.array_equal(samples, sampler.sample(anchors, np.random.default_rng(43))))
        self.assertEqual(samples.shape, (4, 12))
        for anchor, row in zip(anchors, samples):
            self.assertEqual(len(np.unique(row)), len(row))
            self.assertNotIn(anchor, row)
            self.assertTrue(set(row).issubset(set(train)))
            neighbors = sampler.neighbors[int(anchor)]
            self.assertTrue(set(row[:3]).issubset(set(neighbors[:8])))
            self.assertTrue(set(row[3:7]).issubset(set(neighbors[8:24])))
        tiny = LocalCandidates(x, np.array([0, 1]), anchors, config)
        small = tiny.sample(anchors, np.random.default_rng(42))
        self.assertEqual(small.shape, (4, 1))
        for anchor, row in zip(anchors, small):
            self.assertNotIn(anchor, row)
            self.assertTrue(set(row).issubset({0, 1}))

    def test_local_soft_kl_geometry_and_gradients(self):
        torch.manual_seed(7)
        x = torch.randn(5, 6, requires_grad=True)
        cx = torch.randn(5, 9, 6, requires_grad=True)
        mean, std = torch.zeros(6), torch.ones(6)
        same = local_relation_kl(x, x, cx, cx, mean, std)
        self.assertLess(abs(same.item()), 1e-6)
        z = torch.randn(5, 4, requires_grad=True)
        cz = torch.randn(5, 9, 4, requires_grad=True)
        mean = torch.randn(4, requires_grad=True)
        std = torch.ones(4, requires_grad=True)
        loss = local_relation_kl(x, z, cx, cz, mean, std)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for latent in (z, cz):
            self.assertTrue(torch.isfinite(latent.grad).all())
            self.assertGreater(latent.grad.abs().sum().item(), 0.)
        for teacher in (x, cx, mean, std):
            self.assertIsNone(teacher.grad)
        values = np.array([[1., 2.], [3., 2.]], dtype=np.float32)
        m, s = latent_statistics(values)
        np.testing.assert_array_equal(m, [2, 2])
        np.testing.assert_allclose(s, [1, 1e-6])

    def test_local_training_export_and_legacy_cache_separation(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, data_config, x, mask = fixture(directory)
            base = AEConfig(latent_dim=3, hidden_dims=(12, 8), epochs=3, train_batch_size=4,
                            eval_batch_size=3, warmup_epochs=1, ramp_epochs=1, neighbor_interval=1)
            legacy_root = train_artifact(x, dataset, data_config, mask, base)
            legacy_id = read_manifest(legacy_root)["artifact_id"]
            local = replace(base, relation_mode="local")
            root = train_artifact(x, dataset, data_config, mask, local)
            self.assertNotEqual(root, legacy_root)
            manifest = read_manifest(root)
            self.assertEqual(manifest["ae_config"]["relation_mode"], "local")
            model, cp, loaded = load_best(root, source_metadata(x, dataset, data_config, mask))
            self.assertEqual(loaded.to_dict(), local.to_dict())
            with (root / "loss.csv").open() as stream:
                import csv
                rows = list(csv.DictReader(stream))
            self.assertEqual(float(rows[0]["encoder_grad_relation"]), 0.)
            self.assertGreater(float(rows[1]["encoder_grad_relation"]), 0.)
            self.assertIn("valid_local_relation_kl", rows[0])
            with (root / "metrics.json").open() as stream:
                metrics = json.load(stream)
            valid = metrics["valid"]
            self.assertAlmostEqual(valid["score"], valid["normalized_mse"] +
                                   .1 * (1 - valid["cosine_similarity"]) + .1 * valid["local_relation_kl"])
            self.assertIn("relation_kl", valid)
            features, identity = load_features(root, dataset, data_config)
            export_artifact(root, x, dataset, data_config, mask)
            np.testing.assert_array_equal(load_features(root, dataset, data_config)[0], features)
            self.assertEqual(read_manifest(root)["artifact_id"], identity)
            export_artifact(legacy_root, x, dataset, data_config, mask)
            self.assertEqual(read_manifest(legacy_root)["artifact_id"], legacy_id)
            with patch("compression.geometry_ae.pipeline.GeometryAE", side_effect=AssertionError("retrain")):
                self.assertEqual(train_artifact(x, dataset, data_config, mask, base), legacy_root)

    def test_none_skips_relations_in_training_selection_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, data_config, x, mask = fixture(directory)
            config = AEConfig(relation_mode="none", relation_weight=10., latent_dim=3,
                              hidden_dims=(12, 8), epochs=3, train_batch_size=4, eval_batch_size=3,
                              warmup_epochs=0, ramp_epochs=1, neighbor_interval=1)
            with patch("compression.geometry_ae.model.relation_kl", side_effect=AssertionError("batch KL")), \
                    patch("compression.geometry_ae.pipeline.relation_kl", side_effect=AssertionError("eval KL")), \
                    patch("compression.geometry_ae.pipeline.local_relation_kl", side_effect=AssertionError("local KL")), \
                    patch("compression.geometry_ae.pipeline.LocalCandidates", side_effect=AssertionError("sampling")):
                root = train_artifact(x, dataset, data_config, mask, config)
                manifest = read_manifest(root)
                self.assertEqual(manifest["ae_config"]["relation_mode"], "none")
                _, cp, loaded = load_best(root, source_metadata(x, dataset, data_config, mask))
                self.assertEqual(loaded.to_dict(), config.to_dict())
                for row in cp["history"]:
                    for key in ("train_relation", "encoder_grad_relation", "valid_relation_kl"):
                        self.assertEqual(row[key], 0.)
                    self.assertAlmostEqual(row["valid_score"], row["valid_normalized_mse"] +
                                           .1 * (1 - row["valid_cosine_similarity"]))
                self.assertAlmostEqual(cp["valid_score"], min(r["valid_score"] for r in cp["history"]))
                with (root / "metrics.json").open() as stream:
                    metrics = json.load(stream)
                for split in ("train", "valid", "test"):
                    self.assertEqual(metrics[split]["relation_kl"], 0.)
                    self.assertNotIn("local_relation_kl", metrics[split])
                self.assertIn("neighbors", metrics)
                for diagnostic in cp["diagnostics"]:
                    self.assertEqual(diagnostic["weights"]["relation"], 0.)
                    self.assertGreater(diagnostic["weights"]["variance"], 0.)
                    self.assertGreater(diagnostic["weights"]["correlation"], 0.)
                features, identity = load_features(root, dataset, data_config)
                export_artifact(root, x, dataset, data_config, mask)
                np.testing.assert_array_equal(load_features(root, dataset, data_config)[0], features)
                self.assertEqual(read_manifest(root)["artifact_id"], identity)
                self.assertEqual(train_artifact(x, dataset, data_config, mask, config), root)
            with patch("compression.geometry_ae.experiments.relation_kl", side_effect=AssertionError("baseline KL")):
                result = baseline_metrics(x, x, x, torch.tensor(1.), config)
            self.assertEqual(result["relation_kl"], 0.)
            self.assertAlmostEqual(result["score"], result["normalized_mse"] +
                                   .1 * (1 - result["cosine_similarity"]))

    def test_linear_ae_learns_synthetic_low_rank_data(self):
        rng = np.random.default_rng(7)
        basis, _ = np.linalg.qr(rng.normal(size=(12, 3)))
        x = (rng.normal(size=(256, 3)) @ basis.T).astype(np.float32)
        mean, scale = fit_input_statistics(x, np.arange(192))
        torch.manual_seed(7)
        model = GeometryAE(12, AEConfig(architecture="linear", latent_dim=3), mean, scale)
        optimizer = torch.optim.Adam(model.parameters(), lr=.02)
        train = torch.from_numpy(x[:192])
        for _ in range(200):
            optimizer.zero_grad()
            loss = (model(train)[0] - train).square().mean() / scale.square()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            test = torch.from_numpy(x[192:])
            nmse = (model(test)[0] - test).square().mean() / scale.square()
        self.assertLess(nmse.item(), .002)

    def test_train_export_provenance_tokenizer_and_recommendation(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, data_config, x, mask = fixture(directory)
            original = x.copy()
            settings = AEConfig(latent_dim=3, hidden_dims=(12, 8), epochs=3, train_batch_size=4,
                                eval_batch_size=3, warmup_epochs=0, ramp_epochs=1, neighbor_interval=1)
            root = train_artifact(x, dataset, data_config, mask, settings)
            np.testing.assert_array_equal(x, original)
            manifest = read_manifest(root)
            source = source_metadata(x, dataset, data_config, mask)
            model, cp, config = load_best(root, source)
            raw = np.load(root / "raw_latents.npy")
            with torch.no_grad():
                np.testing.assert_allclose(raw, model.latent(torch.from_numpy(x)).numpy(), atol=1e-6)
            splits = split_items(mask, config)
            mean, scale = fit_input_statistics(x, splits["train"])
            torch.testing.assert_close(model.input_mean, mean)
            torch.testing.assert_close(model.input_scale, scale)
            self.assertAlmostEqual(cp["valid_score"], min(r["valid_score"] for r in cp["history"]))
            self.assertIn("encoder_grad_relation", cp["history"][-1])
            with patch("compression.geometry_ae.pipeline.GeometryAE", side_effect=AssertionError("retrain")):
                self.assertEqual(train_artifact(x, dataset, data_config, mask, settings), root)
            data_config.update(sent_emb_compression="geometry_ae", geometry_ae_cache=str(root))
            features, identity = load_features(root, dataset, data_config)
            np.testing.assert_allclose(features[mask].mean(0), 0, atol=1e-5)
            np.testing.assert_allclose(features[mask].std(0), 1, atol=1e-5)
            with patch.object(TigerDiffTokenizer, "log"), patch.object(
                    TigerDiffTokenizer, "_load_sent_emb", side_effect=AssertionError("PCA")):
                tokenizer = TigerDiffTokenizer(data_config, dataset)
            np.testing.assert_array_equal(tokenizer.sent_embs, features)
            self.assertTrue(Path(tokenizer._tokenized_cache_path()).is_relative_to(root))
            rec = TigerDiff({**data_config, **make_config()}, dataset, tokenizer)
            rec(make_batch()).loss.backward()
            self.assertFalse(rec.item_id2embs.requires_grad)
            self.assertIsNotNone(rec.input_proj.linear.weight.grad)
            export_artifact(root, x, dataset, data_config, mask)
            np.testing.assert_array_equal(load_features(root, dataset, data_config)[0], features)
            self.assertEqual(read_manifest(root)["artifact_id"], identity)
            changed = replace(settings, relation_weight=.2)
            self.assertNotEqual(root, artifact_directory(dataset.cache_dir, source, changed.to_dict()))
            dataset.split_data["train"]["item_seq"][0].pop()
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                load_features(root, dataset, data_config)
            dataset.split_data["train"]["item_seq"][0].append("item10")
            dataset.id_mapping["id2item"][1:3] = dataset.id_mapping["id2item"][1:3][::-1]
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                load_features(root, dataset, data_config)
            with (root / "latents.npy").open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_manifest(root)

    def test_checkpoint_rejects_wrong_method_and_artifact_before_loading(self):
        model = torch.nn.Linear(2, 2)
        accelerator = SimpleNamespace(is_main_process=True, unwrap_model=lambda model: model)
        model.config = {"sent_emb_compression": "geometry_ae", "geometry_ae_artifact_id": "a"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rec.pt"
            save_checkpoint(model, path, accelerator)
            load_checkpoint(model, path, "cpu")
            original = model.weight.detach().clone()
            for config in ({"sent_emb_compression": "geometry_ae", "geometry_ae_artifact_id": "b"},
                           {"sent_emb_compression": "geometry_ae"}, {"sent_emb_compression": "pca"},
                           {"sent_emb_compression": "contrec_vae", "contrec_vae_artifact_id": "a"}):
                model.config = config
                with self.assertRaises(ValueError):
                    load_checkpoint(model, path, "cpu")
                torch.testing.assert_close(model.weight, original)
            with self.assertRaisesRegex(ValueError, "identity"):
                model.config = {"sent_emb_compression": "geometry_ae"}
                save_checkpoint(model, path, accelerator)

    def test_cli_ablation_and_paired_recommendation_settings(self):
        with self.assertRaisesRegex(ValueError, "geometry_ae_cache"):
            parse_config(["--sent_emb_compression=geometry_ae", "--geometry_ae_cache="])
        commands = recommendation_commands("/tmp/ae", "Beauty", "/tmp/cache", "/tmp/ckpt", "paired")
        self.assertEqual(len(commands), 6)
        for method, seed, run_id, command in commands:
            config, _ = parse_config(command[2:])
            self.assertEqual(config["sent_emb_compression"], method)
            self.assertEqual(config["rand_seed"], seed)
            self.assertEqual(config["epochs"], 100)
            self.assertEqual(config["patience"], 100)
        candidates = ablation_configs(AEConfig(train_items=4096, epochs=100))
        self.assertEqual(candidates["mlp_cosine"].relation_weight, 0.)
        self.assertEqual(candidates["mlp_statistics"].correlation_weight, 1.)
        records = [{"method": method, "test_results": {"ndcg@10": score, "recall@10": score}}
                   for method, score in (("pca", .02), ("geometry_ae", .01)) for _ in range(3)]
        self.assertEqual(summarize_recommendation(records)["recommended_default"], "pca")

    def test_ablation_selection_and_smoke_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, data_config, x, mask = fixture(directory)
            config = AEConfig(latent_dim=3, hidden_dims=(8, 6), epochs=1, train_batch_size=4, eval_batch_size=3)
            path, report = run_ablation(x, dataset, data_config, mask, config)
            self.assertTrue((path / "ablation.json").is_file())
            best = min(report["candidates"], key=lambda k: report["candidates"][k]["valid"]["score"])
            self.assertEqual(report["winner"], best)
            root = train_artifact(x, dataset, data_config, mask, replace(config, smoke_items=8))
            with self.assertRaisesRegex(ValueError, "Smoke"):
                load_features(root, dataset, data_config)


if __name__ == "__main__":
    unittest.main()
