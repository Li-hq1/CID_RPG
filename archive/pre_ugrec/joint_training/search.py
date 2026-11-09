"""Optuna search over an existing joint_training configuration.

The search space lives in search_space.json. Each trial is one joint_training
run. Optuna maximizes the test score of the checkpoint selected on validation.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys

import torch

from compression.contrec_vae.cache import write_json

from .config import JointConfig, parse_config
from .config_parse import _coerce


DEFAULT_SEARCH_SPACE = Path(__file__).with_name("search_space.json")
_SPEC_KEYS = {
    "study_name", "n_trials", "timeout", "direction", "sampler", "pruner",
    "pruner_startup_trials", "pruner_warmup_steps", "enqueue_baseline", "parameters",
}
_LOCKED_KEYS = (
    "parameters", "sampler", "pruner", "pruner_startup_trials", "pruner_warmup_steps", "direction",
)
_FLOAT_KEYS = {"path", "type", "low", "high", "log"}
_INT_KEYS = {"path", "type", "low", "high", "step", "log"}
_CATEGORICAL_KEYS = {"path", "type", "choices"}


def load_search_spec(path=None):
    source = Path(path) if path is not None else DEFAULT_SEARCH_SPACE
    with source.expanduser().open(encoding="utf-8") as stream:
        spec = json.load(stream)
    if not isinstance(spec, dict):
        raise ValueError("Search config must be a JSON object.")
    unknown = set(spec) - _SPEC_KEYS
    if unknown:
        raise ValueError(f"Unknown search config fields: {sorted(unknown)}")
    missing = _SPEC_KEYS - set(spec)
    if missing:
        raise ValueError(f"Search config is missing fields: {sorted(missing)}")
    if not isinstance(spec["parameters"], list) or not spec["parameters"]:
        raise ValueError("Search config parameters must be a non-empty list.")
    spec["parameters"] = [_parameter(item) for item in spec["parameters"]]
    paths = [item["path"] for item in spec["parameters"]]
    if len(paths) != len(set(paths)):
        raise ValueError("Search parameters must have unique paths.")
    _name(spec["study_name"], "study_name")
    if type(spec["n_trials"]) is not int or spec["n_trials"] <= 0:
        raise ValueError("n_trials must be a positive integer.")
    if spec["timeout"] is not None and (type(spec["timeout"]) not in {int, float} or spec["timeout"] <= 0):
        raise ValueError("timeout must be null or a positive number of seconds.")
    if spec["direction"] != "maximize":
        raise ValueError("direction must be maximize; the study maximizes the test metric.")
    if spec["sampler"] != "tpe":
        raise ValueError("sampler must be tpe.")
    if spec["pruner"] not in {"median", "none"}:
        raise ValueError("pruner must be median or none.")
    for key in ("pruner_startup_trials", "pruner_warmup_steps"):
        if type(spec[key]) is not int or spec[key] < 0:
            raise ValueError(f"{key} must be a non-negative integer.")
    if type(spec["enqueue_baseline"]) is not bool:
        raise ValueError("enqueue_baseline must be a boolean.")
    return spec


def validate_search_spec(spec, config):
    for parameter in spec["parameters"]:
        _validate_parameter(parameter, config)
    contrastive = [item["path"] for item in spec["parameters"] if item["path"].startswith("contrastive.")]
    if contrastive and not config.contrastive.enabled:
        raise ValueError("Search space includes contrastive parameters. Pass --contrastive, or remove "
                         + ", ".join(contrastive) + ".")
    for parameter in spec["parameters"]:
        baseline = _baseline(config, parameter["path"])
        if not _in_space(parameter, baseline):
            raise ValueError(f"Current value of {parameter['path']} ({baseline}) is outside the search space.")
        for value in _edge_values(parameter):
            try:
                _apply(config, parameter["path"], value)
            except ValueError as error:
                raise ValueError(f"{parameter['path']}={value} is not a valid joint configuration: {error}") from error


def suggest_parameters(trial, parameters):
    suggested = {}
    for parameter in parameters:
        name, kind = parameter["path"], parameter["type"]
        if kind == "float":
            suggested[name] = trial.suggest_float(name, parameter["low"], parameter["high"], log=parameter["log"])
        elif kind == "int" and parameter["log"]:
            suggested[name] = trial.suggest_int(name, parameter["low"], parameter["high"], log=True)
        elif kind == "int":
            suggested[name] = trial.suggest_int(name, parameter["low"], parameter["high"], step=parameter["step"])
        else:
            suggested[name] = trial.suggest_categorical(name, parameter["choices"])
    return suggested


def trial_argv(base_argv, run_id, suggestions):
    overrides = []
    for path, value in suggestions.items():
        overrides.extend(("--set", f"{path}={json.dumps(value)}"))
    return [*base_argv, "--run-id", run_id, *overrides]


def run_search(argv=None, *, train=None):
    optuna = _optuna()
    args, remaining = _parse_args(argv)
    if any(item in {"--resume", "--eval-only"} for item in remaining):
        raise ValueError("Hyperparameter search starts new trials. Remove --resume and --eval-only.")
    spec = load_search_spec(args.search_config)
    if args.study_name is not None:
        _name(args.study_name, "study_name")
        spec["study_name"] = args.study_name
    if args.n_trials is not None:
        if args.n_trials <= 0:
            raise ValueError("n_trials must be a positive integer.")
        spec["n_trials"] = args.n_trials
    if args.timeout is not None:
        if args.timeout <= 0:
            raise ValueError("timeout must be a positive number of seconds.")
        spec["timeout"] = args.timeout
    if args.no_baseline:
        spec["enqueue_baseline"] = False
    config, _ = parse_config(remaining)
    validate_search_spec(spec, config)
    study_dir = Path(config.training.output_dir) / config.data.category / spec["study_name"]
    if (study_dir / "config.json").is_file():
        raise FileExistsError(f"{study_dir} is a training run. Choose another --study-name.")
    study_dir.mkdir(parents=True, exist_ok=True)
    saved_path = study_dir / "search_config.json"
    if saved_path.is_file():
        saved = load_search_spec(saved_path)
        changed = [key for key in _LOCKED_KEYS if saved[key] != spec[key]]
        if changed:
            raise ValueError("This study already exists with a different " + ", ".join(changed)
                             + ". Use a new --study-name after changing the search space.")
    write_json(saved_path, spec)
    storage = args.storage or f"sqlite:///{(study_dir / 'optuna.db').resolve()}"
    study = _study(optuna, spec, storage, config.training.seed)
    metric = config.evaluation.val_metric
    study.set_user_attr("objective_metric", metric)
    if spec["enqueue_baseline"] and len(study.trials) == 0:
        study.enqueue_trial({item["path"]: _baseline(config, item["path"]) for item in spec["parameters"]})
    log_path = study_dir / "search.log"

    def objective(trial):
        suggestions = suggest_parameters(trial, spec["parameters"])
        run_id = f"{spec['study_name']}_t{trial.number:03d}"
        trial.set_user_attr("run_id", run_id)

        trial.set_user_attr("objective_metric", metric)

        def on_validation(epoch, score):
            # Validation reports are only for the pruner. The finished trial is
            # ranked by the test metric of its best checkpoint.
            trial.report(float(score), int(epoch))
            if trial.should_prune():
                raise optuna.TrialPruned(f"Pruned at epoch {epoch} with best validation {score}.")

        try:
            summary = (train or _train)(trial_argv(remaining, run_id, suggestions), on_validation=on_validation)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results = summary.get("test_results") if isinstance(summary, dict) else None
        score = results.get(metric) if isinstance(results, dict) else None
        if type(score) not in {int, float} or isinstance(score, bool) or not math.isfinite(score):
            raise ValueError(f"Trial did not return a finite test {metric}.")
        trial.set_user_attr("test_results", {key: float(value) for key, value in results.items()
                                             if type(value) in {int, float} and not isinstance(value, bool)})
        return float(score)

    def on_trial_end(study, trial):
        message = (f"trial={trial.number} state={trial.state.name} value={trial.value} "
                   f"run_id={trial.user_attrs.get('run_id')} params={trial.params}")
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    study.optimize(objective, n_trials=spec["n_trials"], timeout=spec["timeout"],
                   catch=(Exception,), callbacks=[on_trial_end])
    write_json(study_dir / "summary.json", _summary(study))
    _print_best(study)
    return study


def main(argv=None):
    study = run_search(argv)
    return 0 if any(trial.state.name == "COMPLETE" for trial in study.trials) else 1


def _train(argv, on_validation=None):
    from .__main__ import main as train_main
    return train_main(argv, on_validation=on_validation)


def _optuna():
    try:
        import optuna
    except ImportError as error:
        raise ImportError("Optuna is required for hyperparameter search. Install project dependencies with uv sync.") from error
    return optuna


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Search joint_training hyperparameters with Optuna. "
                    "Remaining arguments are forwarded to python -m joint_training.",
        allow_abbrev=False)
    parser.add_argument("--study-name")
    parser.add_argument("--n-trials", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--search-config", type=Path)
    parser.add_argument("--storage")
    parser.add_argument("--no-baseline", action="store_true")
    return parser.parse_known_args(argv)


def _study(optuna, spec, storage, seed):
    try:
        study = optuna.load_study(study_name=spec["study_name"], storage=storage)
    except KeyError:
        study = optuna.create_study(study_name=spec["study_name"], storage=storage, direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
                                    pruner=_pruner(optuna, spec))
    if study.direction != optuna.study.StudyDirection.MAXIMIZE:
        raise ValueError("Existing study does not maximize the test metric.")
    return study


def _pruner(optuna, spec):
    if spec["pruner"] == "none":
        return optuna.pruners.NopPruner()
    return optuna.pruners.MedianPruner(n_startup_trials=spec["pruner_startup_trials"],
                                       n_warmup_steps=spec["pruner_warmup_steps"])


def _summary(study):
    trials = []
    for trial in study.trials:
        trials.append({"number": trial.number, "state": trial.state.name, "value": trial.value,
                       "params": trial.params, "run_id": trial.user_attrs.get("run_id")})
    complete = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    best = None if not complete else {
        "number": study.best_trial.number, "value": study.best_trial.value,
        "params": study.best_trial.params, "run_id": study.best_trial.user_attrs.get("run_id"),
        "test_results": study.best_trial.user_attrs.get("test_results"),
    }
    return {"study_name": study.study_name, "direction": "maximize",
            "objective_metric": study.user_attrs.get("objective_metric"),
            "best_trial": best, "trials": trials}


def _print_best(study):
    complete = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    if not complete:
        print("No completed trial.", flush=True)
        return
    best = study.best_trial
    print(f"Best trial: #{best.number} value={best.value} run_id={best.user_attrs.get('run_id')} "
          f"params={best.params}", flush=True)


def _parameter(item):
    if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("type"), str):
        raise ValueError(f"Each search parameter needs a path and type: {item}")
    path, kind = item["path"], item["type"]
    parts = path.split(".")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Search parameter path must be section.field: {path}")
    if kind == "categorical":
        if set(item) != _CATEGORICAL_KEYS:
            raise ValueError(f"{path} must define path, type, and choices.")
        choices = item["choices"]
        if not isinstance(choices, list) or len(choices) < 2 or len({repr(choice) for choice in choices}) != len(choices):
            raise ValueError(f"{path} choices must contain at least two unique values.")
        return {"path": path, "type": kind, "choices": list(choices)}
    if kind == "float":
        if set(item) != _FLOAT_KEYS:
            raise ValueError(f"{path} must define path, type, low, high, and log.")
    elif kind == "int":
        if set(item) - _INT_KEYS or not {"path", "type", "low", "high", "log"} <= set(item):
            raise ValueError(f"{path} must define path, type, low, high, and log.")
    else:
        raise ValueError(f"{path} type must be float, int, or categorical.")
    low, high, log = item["low"], item["high"], item["log"]
    if type(log) is not bool:
        raise ValueError(f"{path} log must be a boolean.")
    if kind == "float":
        if type(low) not in {int, float} or type(high) not in {int, float} or isinstance(low, bool) or isinstance(high, bool):
            raise ValueError(f"{path} low and high must be numbers.")
        low, high = float(low), float(high)
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError(f"{path} must have finite low < high.")
        if log and low <= 0:
            raise ValueError(f"{path} log scale requires low > 0.")
        return {"path": path, "type": kind, "low": low, "high": high, "log": log}
    step = item.get("step", 1)
    if type(low) is not int or type(high) is not int or type(step) is not int:
        raise ValueError(f"{path} low, high, and step must be integers.")
    if low >= high or step <= 0:
        raise ValueError(f"{path} must have low < high and a positive step.")
    if log and (low <= 0 or step != 1):
        raise ValueError(f"{path} log scale requires low > 0 and step 1.")
    return {"path": path, "type": kind, "low": low, "high": high, "step": step, "log": log}


def _validate_parameter(parameter, config):
    try:
        current = _baseline(config, parameter["path"])
    except AttributeError as error:
        raise ValueError(f"Unknown search parameter: {parameter['path']}") from error
    annotation = type(current)
    if annotation not in {bool, int, float, str}:
        raise ValueError(f"{parameter['path']} is not a scalar search parameter.")
    if parameter["type"] == "categorical":
        for choice in parameter["choices"]:
            _coerce(choice, annotation, parameter["path"])
        return
    if parameter["type"] == "float" and annotation is not float:
        raise ValueError(f"{parameter['path']} is {annotation.__name__}; use a matching search type.")
    if parameter["type"] == "int" and annotation is not int:
        raise ValueError(f"{parameter['path']} is {annotation.__name__}; use a matching search type.")


def _baseline(config, path):
    section, field = path.split(".")
    return getattr(getattr(config, section), field)


def _in_space(parameter, value):
    if parameter["type"] == "categorical":
        return value in parameter["choices"]
    if parameter["log"] and value <= 0:
        return False
    return parameter["low"] <= value <= parameter["high"]


def _edge_values(parameter):
    if parameter["type"] == "categorical":
        return parameter["choices"]
    return (parameter["low"], parameter["high"])


def _apply(config, path, value):
    data = config.to_dict()
    section, field = path.split(".")
    data[section][field] = value
    JointConfig.from_dict(data)


def _name(value, label):
    if not isinstance(value, str) or not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be a non-empty name, not a path.")


if __name__ == "__main__":
    sys.exit(main())
