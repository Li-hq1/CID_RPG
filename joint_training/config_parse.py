"""Validation and CLI parsing for the settings in config.py."""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import json
import math
from pathlib import Path
import types
from typing import Union, get_args, get_origin, get_type_hints

from .config import (
    ContrastiveConfig,
    DataConfig,
    EvaluationConfig,
    JointAEConfig,
    JointConfig,
    RecommenderConfig,
    TrainingConfig,
)


def _positive(config, *names):
    for name in names:
        if getattr(config, name) <= 0:
            raise ValueError(f"{type(config).__name__}.{name} must be positive.")


def _nonnegative(config, *names):
    for name in names:
        if getattr(config, name) < 0:
            raise ValueError(f"{type(config).__name__}.{name} must be non-negative.")


def _choice(config, name, choices):
    if getattr(config, name) not in choices:
        raise ValueError(f"{type(config).__name__}.{name} must be one of {sorted(choices)}.")


def _probability(config, *names):
    for name in names:
        if not 0 <= getattr(config, name) <= 1:
            raise ValueError(f"{type(config).__name__}.{name} must be in [0, 1].")


def _validate_types(config):
    for key, annotation in get_type_hints(type(config)).items():
        _coerce(getattr(config, key), annotation, f"{type(config).__name__}.{key}")


def validate_data(config):
    _validate_types(config)
    _choice(config, "dataset", {"AmazonReviews2014"})
    _choice(config, "metadata", {"sentence"})
    _choice(config, "split", {"leave_one_out", "last_out"})
    _choice(config, "sent_emb_dtype", {"auto", "float32", "float16", "bfloat16"})
    _positive(config, "max_item_seq_len", "num_proc", "sent_emb_dim", "sent_emb_batch_size")
    if config.sent_emb_max_length is not None:
        _positive(config, "sent_emb_max_length")
    for name in ("category", "cache_dir", "sent_emb_model"):
        if not getattr(config, name):
            raise ValueError(f"DataConfig.{name} must be non-empty.")


def validate_ae(config):
    _validate_types(config)
    _choice(config, "architecture", {"linear", "mlp"})
    _choice(config, "scheduler", {"cosine", "constant"})
    _choice(config, "relation_mode", {"none", "batch"})
    _positive(config, "latent_dim", "lr", "max_grad_norm", "batch_size", "update_every", "relation_temperature")
    _nonnegative(config, "weight_decay", "cosine_weight", "variance_weight", "correlation_weight", "relation_weight")
    _probability(config, "warmup_ratio")
    if config.batch_size < 2:
        raise ValueError("AE batch_size must be at least two for geometry regularization.")
    if not config.hidden_dims or any(value <= 0 for value in config.hidden_dims):
        raise ValueError("AE hidden_dims must contain positive integers.")


def validate_recommender(config):
    _validate_types(config)
    _choice(config, "encoder_type", {"hstu", "transformer"})
    _choice(config, "activation_function", {"relu", "gelu", "gelu_new", "gelu_fast"})
    _choice(config, "hstu_attention_scaling", {"input_length", "max_seq_len"})
    _choice(config, "ode_solver", {"euler", "midpoint", "rk4"})
    _choice(config, "scheduler", {"cosine", "constant"})
    _positive(config, "n_embd", "n_layer", "n_head", "n_inner", "layer_norm_epsilon", "hstu_hidden_dim",
              "hstu_attention_dim", "diffloss_w", "diffloss_d", "num_sampling_steps", "diffusion_batch_mul",
              "rectified_flow_steps", "lr", "max_grad_norm", "train_batch_size", "update_every")
    _nonnegative(config, "weight_decay", "tiger_guidance_weight", "diff_temperature")
    _probability(config, "embd_pdrop", "attn_pdrop", "hstu_output_dropout", "hstu_attention_dropout",
                 "tiger_uncond_prob", "warmup_ratio")
    if config.n_embd % config.n_head:
        raise ValueError("recommender.n_embd must be divisible by n_head.")
    if config.encoder_type == "hstu" and config.hstu_attention_dim % 2:
        raise ValueError("recommender.hstu_attention_dim must be even for rotary positions.")
    if config.encoder_type == "hstu" and max(config.hstu_output_dropout, config.hstu_attention_dropout) >= 1:
        raise ValueError("HSTU output and attention dropout must be less than one.")
    if min(config.num_sampling_steps, config.rectified_flow_steps) < 2:
        raise ValueError("Rectified-flow training and sampling require at least two timesteps.")
    if not config.use_rectified_flow:
        raise ValueError("Joint training currently supports rectified flow only.")


def validate_contrastive(config):
    _validate_types(config)
    _positive(config, "max_queries", "temperature", "weight", "num_negatives")


def validate_training(config):
    _validate_types(config)
    _positive(config, "epochs", "save_interval", "cpu_threads")
    _nonnegative(config, "seed", "num_workers")
    if config.max_steps is not None:
        _positive(config, "max_steps")
    if not config.device or not config.output_dir:
        raise ValueError("training.device and output_dir must be non-empty.")
    if not config.run_id or config.run_id in {".", ".."} or "/" in config.run_id or "\\" in config.run_id:
        raise ValueError("training.run_id must be a non-empty task name, not a path.")


def validate_evaluation(config):
    _validate_types(config)
    _choice(config, "eval_sampling_mode", {"knn", "diffusion_multi"})
    _choice(config, "eval_candidate_metric", {"l2", "cosine", "inner_product"})
    _choice(config, "eval_candidate_ranking", {"support", "first_seen", "min_distance"})
    _choice(config, "tiger_knn_metric", {"l2", "cosine", "inner_product"})
    _positive(config, "eval_num_samples", "eval_candidates_per_sample", "eval_support_temperature",
              "eval_batch_size", "eval_interval", "patience", "temperature")
    if not config.topk or any(value <= 0 for value in config.topk) or len(set(config.topk)) != len(config.topk):
        raise ValueError("evaluation.topk must contain unique positive integers.")
    if not config.metrics or not set(config.metrics) <= {"recall", "ndcg"} or len(set(config.metrics)) != len(config.metrics):
        raise ValueError("evaluation.metrics must contain unique recall/ndcg names.")
    names = {f"{metric}@{k}" for metric in config.metrics for k in config.topk}
    if config.val_metric not in names:
        raise ValueError("evaluation.val_metric must be one of the configured metrics.")


_SECTION_VALIDATORS = {
    DataConfig: validate_data,
    JointAEConfig: validate_ae,
    RecommenderConfig: validate_recommender,
    ContrastiveConfig: validate_contrastive,
    TrainingConfig: validate_training,
    EvaluationConfig: validate_evaluation,
}


def validate_joint(config):
    for entry in fields(config):
        section = getattr(config, entry.name)
        expected = get_type_hints(type(config))[entry.name]
        if not isinstance(section, expected):
            raise ValueError(f"{entry.name} must be its declared configuration section.")
        _SECTION_VALIDATORS[type(section)](section)


def build_config(values):
    config = _construct(JointConfig, values, "config")
    validate_joint(config)
    return config


def _coerce(value, annotation, name):
    origin, arguments = get_origin(annotation), get_args(annotation)
    if origin in {Union, types.UnionType}:
        if value is None and type(None) in arguments:
            return None
        for candidate in arguments:
            if candidate is type(None):
                continue
            try:
                return _coerce(value, candidate, name)
            except ValueError:
                pass
        raise ValueError(f"{name} has an invalid value or type.")
    if is_dataclass(annotation):
        return _construct(annotation, value, name)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{name} must be an array.")
        return tuple(_coerce(item, arguments[0], name) for item in value)
    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be an object.")
        return {_coerce(key, arguments[0], name): _coerce(item, arguments[1], name) for key, item in value.items()}
    if annotation is float:
        if type(value) not in {float, int} or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
        return float(value)
    if type(value) is not annotation:
        raise ValueError(f"{name} must be {annotation.__name__}.")
    return value


def _construct(cls, values, name):
    if not isinstance(values, dict):
        raise ValueError(f"{name} must be an object.")
    hints = get_type_hints(cls)
    unknown = set(values) - set(hints)
    if unknown:
        raise ValueError(f"Unknown configuration fields in {name}: {sorted(unknown)}")
    return cls(**{key: _coerce(value, hints[key], f"{name}.{key}") for key, value in values.items()})


def _merge(base, patch):
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def _set_patch(patch, key, value):
    parts = key.split(".")
    if len(parts) != 2 or parts[0] not in get_type_hints(JointConfig):
        raise ValueError("--set requires an existing section.field=value key.")
    section_cls = get_type_hints(JointConfig)[parts[0]]
    annotation = get_type_hints(section_cls).get(parts[1])
    if annotation is None:
        raise ValueError(f"Unknown configuration field: {key}")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value
    patch.setdefault(parts[0], {})[parts[1]] = _coerce(parsed, annotation, key)


def parse_config(argv=None, *, base_mapping=None):
    parser = argparse.ArgumentParser(description="Train the independent joint AE and rectified-flow recommender.",
                                     allow_abbrev=False)
    parser.add_argument("--config", type=Path, help="A partial or complete independent JSON configuration.")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD=VALUE")
    parser.add_argument("--category")
    parser.add_argument("--device")
    parser.add_argument("--run-id", dest="run_id")
    contrastive = parser.add_mutually_exclusive_group()
    contrastive.add_argument("--contrastive", dest="contrastive_enabled", action="store_true")
    contrastive.add_argument("--no-contrastive", dest="contrastive_enabled", action="store_false")
    parser.set_defaults(contrastive_enabled=None)
    parser.add_argument("--smoke-test", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", metavar="PATH")
    mode.add_argument("--eval-only", metavar="PATH", dest="eval_only")
    args = parser.parse_args(argv)
    config = JointConfig.from_dict(base_mapping or {}).to_dict()
    patch = {}
    if args.config is not None:
        with args.config.expanduser().open() as stream:
            loaded = json.load(stream)
        if not isinstance(loaded, dict):
            raise ValueError("--config must contain a JSON object.")
        _merge(patch, loaded)
    for item in args.set:
        if "=" not in item:
            raise ValueError("--set requires section.field=value.")
        key, value = item.split("=", 1)
        _set_patch(patch, key, value)
    for value, section, key in (
        (args.category, "data", "category"), (args.device, "training", "device"),
        (args.run_id, "training", "run_id"), (args.contrastive_enabled, "contrastive", "enabled"),
    ):
        if value is not None:
            patch.setdefault(section, {})[key] = value
    if args.smoke_test:
        _merge(patch, {"training": {"smoke_test": True, "epochs": 2, "max_steps": 4},
                       "recommender": {"train_batch_size": 8}, "ae": {"batch_size": 16},
                       "evaluation": {"eval_batch_size": 8, "eval_interval": 1, "eval_num_samples": 4}})
    args.override_mapping = patch
    return JointConfig.from_dict(_merge(config, patch)), args
