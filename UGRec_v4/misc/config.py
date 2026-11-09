import argparse
import json
from pathlib import Path
from copy import deepcopy

from config import CONFIG
from models.evaluation_sampling import validate_eval_sampling


def boolean(value):
    if value.lower() in {"true", "1"}:
        return True
    if value.lower() in {"false", "0"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def validate_config(config):
    validate_eval_sampling(config)
    positive = ("lr", "train_batch_size", "eval_batch_size", "max_item_seq_len",
                "n_embd", "n_layer", "n_head", "n_inner", "diffloss_w", "diffloss_d",
                "num_sampling_steps", "rectified_flow_steps", "diffusion_batch_mul",
                "num_proc", "eval_interval", "save_interval", "temperature")
    for key in positive:
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive.")
    if config["epochs"] < 0 or (config["warmup_steps"] is not None and config["warmup_steps"] < 0):
        raise ValueError("epochs and warmup_steps must be non-negative.")
    if not 0 <= config.get("warmup_ratio", 0.0) < 1:
        raise ValueError("warmup_ratio must be in [0, 1).")
    if config["steps"] is not None and config["steps"] < 0:
        raise ValueError("steps must be non-negative or None.")
    encoder_type = config.get("encoder_type", "transformer")
    if encoder_type not in {"transformer", "hstu"}:
        raise ValueError("encoder_type must be transformer or hstu.")
    if encoder_type == "transformer" and config["n_embd"] % config["n_head"]:
        raise ValueError("n_embd must be divisible by n_head.")
    if encoder_type == "hstu":
        for key in ("hstu_hidden_dim", "hstu_attention_dim"):
            value = config.get(key, 64)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} must be a positive integer.")
        if config.get("hstu_attention_dim", 64) % 2:
            raise ValueError("hstu_attention_dim must be even for RoPE.")
        if not isinstance(config.get("hstu_use_group_norm", True), bool):
            raise ValueError("hstu_use_group_norm must be boolean.")
        for key, default in (("hstu_output_dropout", 0.1), ("hstu_attention_dropout", 0.0)):
            if not 0 <= config.get(key, default) < 1:
                raise ValueError(f"{key} must be in [0, 1).")
        if config.get("hstu_attention_scaling", "input_length") not in {"input_length", "max_seq_len"}:
            raise ValueError("hstu_attention_scaling must be input_length or max_seq_len.")
    if not config["topk"] or any(k <= 0 for k in config["topk"]):
        raise ValueError("topk must contain positive values.")
    metric_names = {f"{metric}@{k}" for metric in config["metrics"] for k in config["topk"]}
    if config["val_metric"] not in metric_names:
        raise ValueError("val_metric must be one of the configured metrics.")
    if not set(config["metrics"]) <= {"ndcg", "recall"}:
        raise ValueError("Only ndcg and recall metrics are supported.")
    if config["metadata"] != "sentence":
        raise ValueError("TigerDiff requires sentence metadata.")
    if config["ode_solver"] not in {"euler", "midpoint", "rk4"}:
        raise ValueError("Unsupported ODE solver.")
    if config.get("supervision_mode", "last_item") not in {"last_item", "all_positions"}:
        raise ValueError("supervision_mode must be last_item or all_positions.")
    for key in ("sent_emb_dim", "sent_emb_batch_size"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive.")
    if config.get("sent_emb_max_length") is not None and config["sent_emb_max_length"] <= 0:
        raise ValueError("sent_emb_max_length must be positive or None.")
    if config.get("sent_emb_dtype", "float32") not in {"auto", "float32", "float16", "bfloat16"}:
        raise ValueError("Unsupported sent_emb_dtype.")
    if not 0 <= config["sent_emb_pca"] <= config["sent_emb_dim"]:
        raise ValueError("sent_emb_pca must be between 0 and sent_emb_dim.")
    compression = config.get("sent_emb_compression", "pca")
    if compression not in {"pca", "geometry_ae"}:
        raise ValueError("sent_emb_compression must be pca or geometry_ae.")
    if compression == "geometry_ae" and not config.get("geometry_ae_cache"):
        raise ValueError("geometry_ae_cache is required; pretrain the AE independently first.")


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description="Train the UGRec_v3 continuous recommender.", allow_abbrev=False)
    parser.add_argument("--config", default="config.py",
                        help="Select current defaults or the preserved pre-HSTU Transformer configuration.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--smoke-test", "--smoke_test", dest="smoke_test", action="store_true")
    # Accepted only for compatibility with old TigerDiff launch commands.
    parser.add_argument("--model", choices=["TigerDiff"], default="TigerDiff")
    parser.add_argument("--dataset", choices=["AmazonReviews2014"], default="AmazonReviews2014")
    parser.add_argument("--category")
    parser.add_argument("--cache_dir", help="Root containing dataset and feature caches.")
    parser.add_argument("--run_id")
    parser.add_argument("--ckpt_dir")
    parser.add_argument("--rand_seed", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--encoder_type", choices=["transformer", "hstu"])
    parser.add_argument("--hstu_hidden_dim", type=int)
    parser.add_argument("--hstu_attention_dim", type=int)
    parser.add_argument("--hstu_use_group_norm", type=boolean)
    parser.add_argument("--hstu_output_dropout", type=float)
    parser.add_argument("--hstu_attention_dropout", type=float)
    parser.add_argument("--hstu_attention_scaling", choices=["input_length", "max_seq_len"])
    parser.add_argument("--sent_emb_model")
    parser.add_argument("--sent_emb_dim", type=int)
    parser.add_argument("--sent_emb_pca", type=int)
    parser.add_argument("--sent_emb_compression", choices=["pca", "geometry_ae"])
    parser.add_argument("--geometry_ae_cache", help="Pretrained Geometry_AE experiment directory.")
    parser.add_argument("--sent_emb_batch_size", type=int)
    parser.add_argument("--sent_emb_max_length", type=int)
    parser.add_argument("--sent_emb_dtype", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--sent_emb_device", help="Embedding device, e.g. cuda:0 or cpu.")
    parser.add_argument("--sent_emb_local_files_only", type=boolean)
    parser.add_argument("--sent_emb_prompt")
    parser.add_argument("--sent_emb_normalize", type=boolean)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--eval_batch_size", type=int)
    parser.add_argument("--eval_sampling_mode", choices=["knn", "diffusion_multi"])
    parser.add_argument("--eval_num_samples", type=int)
    parser.add_argument("--eval_candidates_per_sample", type=int)
    parser.add_argument("--eval_candidate_metric", choices=["l2", "cosine", "inner_product"])
    parser.add_argument("--eval_candidate_ranking", choices=["support", "first_seen", "min_distance"])
    parser.add_argument("--eval_support_temperature", type=float)
    parser.add_argument("--eval_log_sampling_stats", type=boolean)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--warmup_ratio", type=float,
                        help="Fraction of planned training used for linear warmup (default: 0.1).")
    parser.add_argument("--warmup_steps", type=int,
                        help="Fixed warmup steps; overrides warmup_ratio, including 0 to disable warmup.")
    parser.add_argument("--save_interval", type=int,
                        help="Overwrite latest every N epochs and at training end; best saves on validation improvement.")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--diffloss_w", type=int)
    parser.add_argument("--diffloss_d", type=int)
    parser.add_argument("--num_sampling_steps", type=int)
    parser.add_argument("--diffusion_batch_mul", type=int)
    parser.add_argument("--diff_temperature", type=float)
    parser.add_argument("--use_rectified_flow", type=boolean)
    parser.add_argument("--rectified_flow_steps", type=int)
    parser.add_argument("--ode_solver", choices=["euler", "midpoint", "rk4"])
    parser.add_argument("--tiger_guidance_weight", type=float)
    parser.add_argument("--tiger_uncond_prob", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--tiger_knn_metric", choices=["cosine", "inner_product", "l2"])
    parser.add_argument("--supervision_mode", choices=["last_item", "all_positions"],
                        help="Both modes use causal attention; supervise all valid next-item labels or only the last label.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true",
                        help="Resume optimizer, scheduler and epoch state from the latest checkpoint.")
    mode.add_argument("--eval-only", "--eval_only", dest="eval_only", action="store_true",
                        help="Skip training and evaluate the best checkpoint, or --checkpoint if provided.")
    args = parser.parse_args(argv)
    if args.config == "config_transformer.py":
        from config_transformer import CONFIG as transformer_config
        config = deepcopy(transformer_config)
        config["encoder_type"] = "transformer"
    elif args.config == "config_hstu.py":
        from config_hstu import CONFIG as hstu_config
        config = deepcopy(hstu_config)
    elif args.config == "config_knn_pca.py":
        from config_knn_pca import CONFIG as knn_config
        config = deepcopy(knn_config)
    elif args.config == "config.py":
        config = deepcopy(CONFIG)
    else:
        config_path = Path(args.config)
        if config_path.suffix != ".json":
            parser.error("--config must select a built-in config.py or an explicit JSON configuration.")
        with config_path.open(encoding="utf-8") as stream:
            saved = json.load(stream)
        if not isinstance(saved, dict):
            raise ValueError("JSON configuration must be an object.")
        config = {**deepcopy(CONFIG), **saved}
    for key, value in vars(args).items():
        if (key in config or key.startswith("hstu_") or key in {
                "sent_emb_compression", "geometry_ae_cache"}) and value is not None:
            config[key] = value
    if args.warmup_ratio is not None and args.warmup_steps is None:
        config["warmup_steps"] = None
    config.update(model=args.model, dataset=args.dataset)
    if args.smoke_test:
        config.update(epochs=1, steps=None, warmup_steps=0, train_batch_size=32, eval_batch_size=32)
        config["smoke_test"] = True
    validate_config(config)
    run_id = str(config["run_id"])
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must be a non-empty task name, not a path.")
    return config, args
