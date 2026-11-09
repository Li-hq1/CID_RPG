"""Run standalone Geometry_AE pretraining with ``python -m compression.geometry_ae``."""

import argparse
from copy import deepcopy
import json
import logging
from pathlib import Path
import shlex

import torch
from accelerate import Accelerator

from config import CONFIG
from data.amazon_reviews import AmazonReviews2014
from data.embeddings import load_item_sentence_embeddings
from compression.contrec_vae.cache import training_item_mask
from .config import AEConfig
from .pipeline import export_artifact, train_artifact


def parse_args(argv=None):
    defaults = AEConfig()
    parser = argparse.ArgumentParser(description="Pretrain a geometry-preserving item autoencoder.", allow_abbrev=False)
    parser.add_argument("--category", default=CONFIG["category"])
    parser.add_argument("--cache_dir", default=CONFIG["cache_dir"])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sent_emb_device", default=None)
    parser.add_argument("--sent_emb_batch_size", type=int, default=CONFIG["sent_emb_batch_size"])
    parser.add_argument("--sent_emb_max_length", type=int, default=CONFIG["sent_emb_max_length"])
    parser.add_argument("--sent_emb_dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    for key in ("latent_dim", "seed", "train_batch_size", "eval_batch_size", "epochs", "patience",
                "warmup_epochs", "ramp_epochs", "train_items", "smoke_items", "local_near_pool",
                "local_middle_pool", "local_near_count", "local_middle_count", "local_random_count"):
        parser.add_argument(f"--{key}", type=int, default=getattr(defaults, key))
    for key in ("lr", "weight_decay", "max_grad_norm", "cosine_weight", "variance_weight",
                "correlation_weight", "relation_weight", "relation_temperature", "test_ratio", "valid_ratio"):
        parser.add_argument(f"--{key}", type=float, default=getattr(defaults, key))
    parser.add_argument("--architecture", choices=["linear", "mlp"], default=defaults.architecture)
    parser.add_argument("--relation_mode", choices=["none", "batch", "local"], default=defaults.relation_mode)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.sent_emb_batch_size <= 0 or args.sent_emb_max_length <= 0:
        parser.error("Embedding batch size and maximum length must be positive.")
    if args.export_only != bool(args.artifact_dir):
        parser.error("--export-only and --artifact-dir must be used together.")
    if args.force and args.export_only:
        parser.error("--force cannot be combined with --export-only.")
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; choose --device=cpu.")
    fields = ("architecture", "latent_dim", "seed", "train_batch_size", "eval_batch_size", "epochs", "patience",
              "warmup_epochs", "ramp_epochs", "train_items", "smoke_items", "lr", "weight_decay",
              "max_grad_norm", "cosine_weight", "variance_weight", "correlation_weight", "relation_weight",
              "relation_temperature", "test_ratio", "valid_ratio", "relation_mode", "local_near_pool",
              "local_middle_pool", "local_near_count", "local_middle_count", "local_random_count")
    settings = {key: getattr(args, key) for key in fields}
    if args.smoke_test:
        settings.update(epochs=2, train_batch_size=16, eval_batch_size=64, smoke_items=64, train_items=None)
    config = AEConfig(**settings)
    config.validate()
    data_config = deepcopy(CONFIG)
    data_config.update(category=args.category, cache_dir=args.cache_dir, dataset="AmazonReviews2014",
                       metadata="sentence", split="leave_one_out", sent_emb_model="Qwen/Qwen3-Embedding-4B",
                       sent_emb_dim=2560, sent_emb_prompt="", sent_emb_normalize=True,
                       sent_emb_max_length=args.sent_emb_max_length, sent_emb_dtype=args.sent_emb_dtype,
                       sent_emb_batch_size=args.sent_emb_batch_size, sent_emb_local_files_only=True,
                       sent_emb_device=args.sent_emb_device, device=device)
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    accelerator = Accelerator(cpu=device.type == "cpu")
    if accelerator.num_processes != 1:
        raise RuntimeError("Geometry AE pretraining requires a single process.")
    data_config["accelerator"] = accelerator
    try:
        dataset = AmazonReviews2014(data_config)
        mask = training_item_mask(dataset)
        embeddings = load_item_sentence_embeddings(dataset, data_config)
        if args.export_only:
            directory = Path(args.artifact_dir).expanduser().resolve()
            root = (Path(dataset.cache_dir) / "Geometry_AE").resolve()
            if not directory.is_relative_to(root):
                raise ValueError(f"Geometry AE artifacts must be under {root}.")
            directory = export_artifact(directory, embeddings, dataset, data_config, mask, device=device)
        else:
            directory = train_artifact(embeddings, dataset, data_config, mask, config, device=device, force=args.force)
    finally:
        accelerator.end_training()
    with (directory / "manifest.json").open() as stream:
        relation_mode = json.load(stream)["ae_config"].get("relation_mode", "batch")
    print(f"Geometry AE artifact: {directory}")
    command = ["uv", "run", "train.py", f"--category={args.category}", f"--cache_dir={args.cache_dir}",
               "--sent_emb_compression=geometry_ae", f"--geometry_ae_cache={directory}",
               f"--sent_emb_max_length={args.sent_emb_max_length}", f"--sent_emb_dtype={args.sent_emb_dtype}",
               f"--run_id=tigerdiff_{args.category.lower()}_geometry_ae" +
               (f"_{directory.name}" if relation_mode != "batch" else "")]
    if "/smoke/" in str(directory):
        command.append("--smoke-test")
    print(f"Recommendation: {shlex.join(command)}")
    return directory


if __name__ == "__main__":
    main()
