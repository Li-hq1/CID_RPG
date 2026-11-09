"""Optuna search over the v5.1 control configuration."""

import json
from itertools import product
from pathlib import Path
import tempfile
import unittest

import optuna

from joint_training.config import JointConfig, parse_config
from joint_training.search import (
    load_search_spec,
    run_search,
    suggest_parameters,
    trial_argv,
    validate_search_spec,
)
from protocol import ROOT


def _spec(**overrides):
    spec = {
        "study_name": "search_test",
        "n_trials": 1,
        "timeout": None,
        "direction": "maximize",
        "sampler": "tpe",
        "pruner": "none",
        "pruner_startup_trials": 0,
        "pruner_warmup_steps": 0,
        "enqueue_baseline": False,
        "parameters": [
            {"path": "ae.lr", "type": "float", "low": 0.00001, "high": 0.0001, "log": True},
        ],
    }
    spec.update(overrides)
    return spec


def _write_spec(directory, spec):
    path = Path(directory) / "search.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def _adopted_beauty():
    return parse_config(["--config", str(ROOT / "configs/direct_ce.json")])[0]


class SearchSpaceTest(unittest.TestCase):
    def test_first_round_space_is_hstu_epoch_and_recommender_lr(self):
        spec = load_search_spec()
        self.assertEqual([item["path"] for item in spec["parameters"]], [
            "recommender.n_layer", "recommender.n_embd", "recommender.n_head",
            "recommender.hstu_hidden_dim", "recommender.hstu_attention_dim",
            "training.epochs", "recommender.lr",
        ])
        self.assertEqual(spec["pruner"], "none")
        config = _adopted_beauty()
        self.assertEqual(config.flow_ranking.path, "direct_ce")
        self.assertEqual(config.recommender.n_layer, 1)
        validate_search_spec(spec, config)
        choices = {item["path"]: item["choices"] for item in spec["parameters"] if item["type"] == "categorical"}
        base = config.to_dict()
        for layer, width, heads, hidden, attention in product(
                choices["recommender.n_layer"], choices["recommender.n_embd"], choices["recommender.n_head"],
                choices["recommender.hstu_hidden_dim"], choices["recommender.hstu_attention_dim"]):
            base["recommender"].update(n_layer=layer, n_embd=width, n_head=heads,
                                       hstu_hidden_dim=hidden, hstu_attention_dim=attention)
            JointConfig.from_dict(base)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=1))
        for _ in range(8):
            trial = study.ask()
            suggestions = suggest_parameters(trial, spec["parameters"])
            parsed, _ = parse_config(["--config", str(ROOT / "configs/direct_ce.json"),
                                      *trial_argv([], "sample", suggestions)])
            self.assertEqual(parsed.training.run_id, "sample")
            self.assertFalse(parsed.training.evaluate_test)
            self.assertEqual(parsed.recommender.n_embd % parsed.recommender.n_head, 0)
            self.assertEqual(parsed.recommender.hstu_attention_dim % 2, 0)
            self.assertIn(parsed.training.epochs, {40, 50, 60, 70, 80})
            study.tell(trial, 0.0)

    def test_int_parameter_lands_on_its_grid(self):
        spec = _spec(parameters=[{
            "path": "training.epochs", "type": "int", "low": 40, "high": 80, "step": 10, "log": False,
        }])
        validate_search_spec(spec, _adopted_beauty())
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=0))
        trial = study.ask()
        suggestions = suggest_parameters(trial, spec["parameters"])
        parsed, _ = parse_config(["--config", str(ROOT / "configs/direct_ce.json"), *trial_argv([], "sample", suggestions)])
        self.assertIn(parsed.training.epochs, {40, 50, 60, 70, 80})

    def test_baseline_must_be_on_integer_search_grid(self):
        config = JointConfig()
        with self.assertRaisesRegex(ValueError, "outside the search space"):
            validate_search_spec(_spec(parameters=[{
                "path": "training.epochs", "type": "int", "low": 50, "high": 300,
                "step": 30, "log": False,
            }]), config)

    def test_trial_overrides_replace_user_run_id_and_set_values(self):
        argv = trial_argv(["--set", "ae.lr=0.00002", "--run-id", "user_run"], "study_t000", {"ae.lr": 0.00005})
        config, _ = parse_config(argv)
        self.assertEqual(config.training.run_id, "study_t000")
        self.assertEqual(config.ae.lr, 0.00005)

    def test_invalid_space_and_baseline_are_rejected(self):
        config, _ = parse_config([])
        with self.assertRaises(ValueError):
            load_search_spec_from({"extra": True})
        with self.assertRaisesRegex(ValueError, "Unknown search parameter"):
            validate_search_spec(_spec(parameters=[{
                "path": "ae.missing", "type": "float", "low": 0.1, "high": 0.2, "log": False,
            }]), config)
        with self.assertRaisesRegex(ValueError, "outside the search space"):
            validate_search_spec(_spec(parameters=[{
                "path": "ae.lr", "type": "float", "low": 0.001, "high": 0.01, "log": True,
            }]), config)
        with self.assertRaisesRegex(ValueError, "log scale"):
            load_search_spec_from(_spec(parameters=[{
                "path": "ae.lr", "type": "float", "low": 0.0, "high": 0.1, "log": True,
            }]))


def load_search_spec_from(spec):
    with tempfile.TemporaryDirectory() as directory:
        path = _write_spec(directory, spec if "study_name" in spec else {**_spec(), **spec})
        return load_search_spec(path)


class SearchRunTest(unittest.TestCase):
    def test_baseline_trial_uses_adopted_beauty_config(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = load_search_spec()
            spec["n_trials"] = 1
            path = _write_spec(directory, spec)
            seen = []

            def train(argv, on_validation=None):
                config, _ = parse_config(argv)
                seen.append(config)
                on_validation(5, 0.2)
                return {"best_val_score": 0.2, "test_results": {"ndcg@10": 0.4}}

            study = run_search(["--search-config", str(path), "--config", str(ROOT / "configs/direct_ce.json"),
                                "--set", f"training.output_dir={directory}/out", "--device", "cpu"], train=train)
            self.assertEqual(len(seen), 1)
            self.assertEqual(seen[0].flow_ranking.path, "direct_ce")
            self.assertEqual(seen[0].flow_ranking.draws, 1)
            self.assertEqual(seen[0].flow_ranking.noise_scale, 0)
            self.assertEqual(seen[0].recommender.n_layer, 1)
            self.assertEqual(seen[0].recommender.n_embd, 128)
            self.assertEqual(seen[0].recommender.lr, 0.001)
            self.assertEqual(seen[0].training.epochs, 40)
            self.assertTrue(seen[0].training.run_id.endswith("_t000"))
            self.assertFalse(seen[0].training.evaluate_test)
            self.assertEqual(study.best_value, 0.2)
            summary = json.loads((Path(directory) / "out" / "Beauty" / "beauty_hstu_epoch_lr" / "summary.json").read_text())
            self.assertEqual(summary["objective_metric"], "ndcg@10")
            self.assertEqual(summary["objective_split"], "validation")
            self.assertNotIn("test_results", summary["best_trial"])

    def test_training_entry_requests_subset_audit(self):
        from joint_training.search import _train
        seen = {}

        def fake_main(argv, on_validation=None):
            seen["argv"] = list(argv)
            seen["callback"] = on_validation
            return {"best_val_score": 0.1}

        import run
        original = run.main
        run.main = fake_main
        try:
            _train(["--config", "configs/final.json"], on_validation=fake_main)
        finally:
            run.main = original
        self.assertEqual(seen["argv"][-2:], ["--audit", "subset"])

    def test_resume_adds_a_trial_without_repeating_completed_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_spec(directory, _spec())
            calls = []

            def train(argv, on_validation=None):
                config, _ = parse_config(argv)
                calls.append(config.training.run_id)
                return {"best_val_score": 0.2}

            base = ["--search-config", str(path), "--set", f"training.output_dir={directory}/out", "--device", "cpu"]
            run_search(base, train=train)
            run_search([*base, "--n-trials", "1"], train=train)
            self.assertEqual(calls, ["search_test_t000", "search_test_t001"])

    def test_median_pruner_stops_a_weak_trial_and_failures_do_not_stop_the_study(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = _spec(n_trials=3, pruner="median", pruner_startup_trials=1, enqueue_baseline=False)
            path = _write_spec(directory, spec)
            calls = {"n": 0}

            def train(argv, on_validation=None):
                calls["n"] += 1
                if calls["n"] == 1:
                    on_validation(5, 0.8)
                    return {"best_val_score": 0.8}
                if calls["n"] == 2:
                    on_validation(5, 0.0)
                    raise AssertionError("pruned trial continued")
                raise RuntimeError("trial failed")

            study = run_search(["--search-config", str(path), "--set", f"training.output_dir={directory}/out",
                                "--device", "cpu"], train=train)
            states = [trial.state.name for trial in study.trials]
            self.assertEqual(states, ["COMPLETE", "PRUNED", "FAIL"])
            self.assertEqual(study.best_value, 0.8)

    def test_objective_uses_validation_even_when_test_prefers_another_trial(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_spec(directory, _spec(n_trials=2, enqueue_baseline=False))
            calls = {"n": 0}

            def train(argv, on_validation=None):
                calls["n"] += 1
                if calls["n"] == 1:
                    on_validation(5, 0.9)
                    return {"best_val_score": 0.9, "test_results": {"ndcg@10": 0.2}}
                on_validation(5, 0.4)
                return {"best_val_score": 0.4, "test_results": {"ndcg@10": 0.7}}

            study = run_search(["--search-config", str(path), "--set", f"training.output_dir={directory}/out",
                                "--device", "cpu"], train=train)
            self.assertEqual(study.best_value, 0.9)
            self.assertEqual(study.best_trial.number, 0)

    def test_changed_search_space_requires_a_new_study(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_spec(directory, _spec())

            def train(argv, on_validation=None):
                return {"best_val_score": 0.2}

            base = ["--search-config", str(path), "--set", f"training.output_dir={directory}/out", "--device", "cpu"]
            run_search(base, train=train)
            path.write_text(json.dumps(_spec(parameters=[{
                "path": "ae.lr", "type": "float", "low": 0.00002, "high": 0.0001, "log": True,
            }])), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different parameters"):
                run_search(base, train=train)

    def test_resume_and_eval_flags_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_spec(directory, _spec())
            for flag in ("--resume", "--eval-only"):
                with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "new trials"):
                    run_search(["--search-config", str(path), flag, "checkpoint.pt"], train=lambda *args, **kwargs: None)

    def test_existing_legacy_objective_cannot_be_mixed_with_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_spec(directory, _spec())
            study_directory = Path(directory) / "out/Beauty/search_test"
            study_directory.mkdir(parents=True)
            storage = f"sqlite:///{study_directory / 'optuna.db'}"
            legacy = optuna.create_study(study_name="search_test", storage=storage, direction="maximize")
            legacy.add_trial(optuna.trial.create_trial(value=.3))
            with self.assertRaisesRegex(ValueError, "legacy or unknown objective"):
                run_search(["--search-config", str(path), "--set", f"training.output_dir={directory}/out",
                            "--device", "cpu"], train=lambda *a, **kw: None)
