"""Tests for Optuna search over joint_training settings."""

import json
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


class JointSearchConfigTest(unittest.TestCase):
    def test_default_space_matches_contrastive_defaults_and_samples_valid_configs(self):
        spec = load_search_spec()
        config, _ = parse_config(["--contrastive"])
        validate_search_spec(spec, config)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=1))
        for _ in range(4):
            trial = study.ask()
            suggestions = suggest_parameters(trial, spec["parameters"])
            parsed, _ = parse_config(["--contrastive", *trial_argv([], "sample", suggestions)])
            self.assertEqual(parsed.training.run_id, "sample")
            study.tell(trial, 0.0)
            self.assertEqual(JointConfig.from_dict(parsed.to_dict()), parsed)

    def test_int_parameter_lands_on_its_grid(self):
        spec = _spec(parameters=[{
            "path": "contrastive.num_negatives", "type": "int", "low": 16, "high": 64, "step": 16, "log": False,
        }])
        config, _ = parse_config(["--contrastive"])
        validate_search_spec(spec, config)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=0))
        trial = study.ask()
        suggestions = suggest_parameters(trial, spec["parameters"])
        parsed, _ = parse_config(["--contrastive", *trial_argv([], "sample", suggestions)])
        self.assertIn(parsed.contrastive.num_negatives, {16, 32, 48, 64})

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
        with self.assertRaisesRegex(ValueError, "contrastive"):
            validate_search_spec(_spec(parameters=[{
                "path": "contrastive.weight", "type": "float", "low": 0.001, "high": 0.1, "log": True,
            }]), config)
        with self.assertRaisesRegex(ValueError, "log scale"):
            load_search_spec_from(_spec(parameters=[{
                "path": "ae.lr", "type": "float", "low": 0.0, "high": 0.1, "log": True,
            }]))


def load_search_spec_from(spec):
    with tempfile.TemporaryDirectory() as directory:
        path = _write_spec(directory, spec if "study_name" in spec else {**_spec(), **spec})
        return load_search_spec(path)


class JointSearchRunTest(unittest.TestCase):
    def test_baseline_trial_uses_current_config_and_returns_its_score(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = _spec(enqueue_baseline=True, parameters=[{
                "path": "ae.lr", "type": "float", "low": 0.00001, "high": 0.0001, "log": True,
            }])
            path = _write_spec(directory, spec)
            seen = []

            def train(argv, on_validation=None):
                config, _ = parse_config(argv)
                seen.append(config)
                on_validation(5, 0.2)
                return {"best_val_score": 0.2, "test_results": {"ndcg@10": 0.4, "recall@10": 0.5}}

            study = run_search(["--search-config", str(path), "--set", f"training.output_dir={directory}/out",
                                "--device", "cpu"], train=train)
            self.assertEqual(len(seen), 1)
            self.assertEqual(seen[0].ae.lr, 3e-5)
            self.assertTrue(seen[0].training.run_id.endswith("_t000"))
            self.assertFalse(seen[0].training.evaluate_test)
            self.assertEqual(study.best_value, 0.2)
            summary = json.loads((Path(directory) / "out" / "Beauty" / "search_test" / "summary.json").read_text())
            self.assertEqual(summary["objective_metric"], "ndcg@10")
            self.assertEqual(summary["best_trial"]["value"], 0.2)
            self.assertEqual(summary["objective_split"], "validation")
            self.assertEqual(summary["best_trial"]["validation_score"], 0.2)
            self.assertNotIn("test_results", summary["best_trial"])

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
                # The first run looks better on validation and worse on the final test.
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

    def test_resume_preserves_configured_pruner_warmup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_spec(directory, _spec(pruner="median", pruner_startup_trials=1,
                                                pruner_warmup_steps=30))
            scores = iter([.8, .1])

            def train(argv, on_validation=None):
                score = next(scores)
                on_validation(5, score)
                return {"best_val_score": score}

            base = ["--search-config", str(path), "--set", f"training.output_dir={directory}/out", "--device", "cpu"]
            run_search(base, train=train)
            study = run_search(base, train=train)
            self.assertEqual([trial.state.name for trial in study.trials], ["COMPLETE", "COMPLETE"])
            self.assertEqual(study.pruner._n_warmup_steps, 30)

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


if __name__ == "__main__":
    unittest.main()
