"""Run with python -m compression.contrec_vae; never starts recommendation training."""

import argparse
from copy import deepcopy
import logging
from pathlib import Path
import shlex

from accelerate import Accelerator
import torch

from config import CONFIG
from data.amazon_reviews import AmazonReviews2014
from data.embeddings import load_item_sentence_embeddings
from .cache import training_item_mask
from .config import VAEConfig
from .pipeline import export_artifact, train_artifact


def parse_args(argv=None):
    defaults = VAEConfig()
    parser = argparse.ArgumentParser(description="Pretrain ContRec SigmaVae on Qwen item features.", allow_abbrev=False)
    parser.add_argument("--category", default=CONFIG["category"])
    parser.add_argument("--cache_dir", default=CONFIG["cache_dir"])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sent_emb_device", default=None)
    parser.add_argument("--sent_emb_batch_size", type=int, default=CONFIG["sent_emb_batch_size"])
    parser.add_argument("--sent_emb_max_length", type=int, default=CONFIG["sent_emb_max_length"])
    parser.add_argument("--sent_emb_dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    for key in ("epochs", "patience", "train_batch_size", "eval_batch_size", "seed", "latent_dim"):
        parser.add_argument(f"--{key}", type=int, default=getattr(defaults, key))
    for key in (
        "lr", "weight_decay", "sigma", "mask_ratio", "test_ratio", "valid_ratio",
        "beta", "gamma", "twins_lambda",
    ):
        parser.add_argument(f"--{key}", type=float, default=getattr(defaults, key))
    parser.add_argument("--smoke-test", action="store_true", help="Two epochs on at most 64 training-population items.")
    parser.add_argument("--export-only", action="store_true", help="Reload best.pt and re-export without training.")
    parser.add_argument("--artifact-dir", help="Existing experiment directory; required for --export-only.")
    parser.add_argument("--force", action="store_true", help="Retrain and replace the same experiment's artifacts.")
    args = parser.parse_args(argv)
    if args.export_only != bool(args.artifact_dir):
        parser.error("--export-only and --artifact-dir must be used together.")
    if args.force and args.export_only:
        parser.error("--force cannot be combined with --export-only.")
    for key in ("sent_emb_batch_size", "sent_emb_max_length"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key} must be positive.")
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; choose --device=cpu.")
    settings = {key: getattr(args, key) for key in (
        "epochs", "patience", "train_batch_size", "eval_batch_size", "seed", "latent_dim",
        "lr", "weight_decay", "sigma", "mask_ratio", "test_ratio", "valid_ratio",
        "beta", "gamma", "twins_lambda",
    )}
    if args.smoke_test:
        settings.update(epochs=2, train_batch_size=16, eval_batch_size=64, smoke_items=64)
    vae_config = VAEConfig(**settings)
    vae_config.validate()
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    accelerator = Accelerator(cpu=device.type == "cpu")
    if accelerator.num_processes != 1:
        raise RuntimeError("VAE pretraining is a single-process job; launch one Python process.")
    data_config = deepcopy(CONFIG)
    data_config.update(category=args.category, cache_dir=args.cache_dir,
                       dataset="AmazonReviews2014", metadata="sentence", split="leave_one_out",
                       sent_emb_model="Qwen/Qwen3-Embedding-4B", sent_emb_dim=2560,
                       sent_emb_prompt="", sent_emb_normalize=True,
                       sent_emb_max_length=args.sent_emb_max_length, sent_emb_dtype=args.sent_emb_dtype,
                       sent_emb_batch_size=args.sent_emb_batch_size, sent_emb_local_files_only=True,
                       sent_emb_device=args.sent_emb_device, device=device, accelerator=accelerator)
    try:
        dataset = AmazonReviews2014(data_config)
        train_mask = training_item_mask(dataset)
        if args.export_only:
            directory = Path(args.artifact_dir).expanduser().resolve()
            root = (Path(dataset.cache_dir) / "ContRec_VAE").resolve()
            if not directory.is_relative_to(root):
                raise ValueError(f"VAE artifacts must be under {root}.")
        embeddings = load_item_sentence_embeddings(dataset, data_config)
        if args.export_only:
            directory = export_artifact(directory, embeddings, dataset, data_config, train_mask, device=device)
        else:
            directory = train_artifact(embeddings, dataset, data_config, train_mask,
                                       vae_config, device=device, force=args.force)
        print(f"VAE artifact: {directory}")
        command = ["uv", "run", "train.py", f"--category={args.category}",
                   f"--cache_dir={args.cache_dir}",
                   "--sent_emb_compression=contrec_vae", f"--contrec_vae_cache={directory}",
                   f"--sent_emb_max_length={args.sent_emb_max_length}",
                   f"--sent_emb_dtype={args.sent_emb_dtype}",
                   f"--run_id=tigerdiff_{args.category.lower()}_contrec_vae"]
        if "/smoke/" in str(directory):
            command.append("--smoke-test")
        print(f"Recommendation: {shlex.join(command)}")
        return directory
    finally:
        accelerator.end_training()


if __name__ == "__main__":
    main()
