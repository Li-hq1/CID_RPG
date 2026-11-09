"""Standalone launcher: python -m joint_training."""

from contextlib import nullcontext
import json
import logging
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from misc.artifacts import atomic_write, json_hash, write_json
from compression.geometry_ae.model import GeometryAE

from .checkpoint import load_checkpoint, read_checkpoint, save_checkpoint
from .config import parse_config
from .data import initialize_random_ae, load_data
from .evaluation import evaluate, latent_diagnostics
from .model import JointModel
from .tensorboard import JointTensorBoard
from .training import JointTrainer


def seed_run(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def smoke_configuration(config):
    config.data.max_item_seq_len = 8
    rec = config.recommender
    rec.n_embd, rec.n_layer, rec.n_head, rec.n_inner = 32, 1, 4, 64
    rec.hstu_hidden_dim = rec.hstu_attention_dim = 8
    rec.diffloss_w, rec.diffloss_d, rec.num_sampling_steps = 32, 1, 2
    config.validate()


def initialize_logging(directory, mode):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(directory / f"{mode}.log")],
                        force=True)


def append_metrics(path, values):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(values, allow_nan=False) + "\n")


def _fmt_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, int):
        return str(value)
    value = float(value)
    if value == 0:
        return "0"
    if value.is_integer() and abs(value) < 1e7:
        return str(int(value))
    magnitude = abs(value)
    if 1e-3 <= magnitude < 1e3:
        return f"{value:.4f}"
    return f"{value:.2e}"


def format_kv(values, keys=None):
    items = ((key, values[key]) for key in keys if key in values) if keys is not None else values.items()
    return " ".join(f"{key}={_fmt_number(value)}" for key, value in items)


def ranking_summary(results, topk=(5, 10)):
    headers, values = [], []
    for k in topk:
        for prefix, name in (("R", f"recall@{k}"), ("N", f"ndcg@{k}")):
            if name not in results:
                continue
            headers.append(f"{prefix}@{k}")
            values.append(f"{float(results[name]):.3f}")
    return "\t".join(headers), "\t".join(values)


def log_ranking_line(prefix, results, *, extra="", table=False):
    compact = format_kv(results, ("recall@5", "ndcg@5", "recall@10", "ndcg@10",
                                  "n_visited_items", "sampling_unique_candidates",
                                  "sampling_deduplication_rate", "sampling_retrieval_depth"))
    suffix = f"; {extra}" if extra else ""
    logging.info("%s%s", prefix, f"{compact}{suffix}" if compact else results)
    if table:
        headers, values = ranking_summary(results)
        if headers:
            logging.info("%s\n%s", headers, values)


def log_test_results(results):
    log_ranking_line("Test Results: ", results, table=True)


_TRAIN_LOG_KEYS = (
    "ae_loss", "rf_loss", "ae_contrastive",
    "contrastive_weight", "contrastive_negative_count", "ae_lr", "rec_lr",
    "ae/reconstruction", "ae/cosine", "ae/correlation",
    "ae_grad_norm", "ae_contrastive_grad_norm",
)


def main(argv=None, *, on_validation=None):
    config, args = parse_config(argv)
    selected = args.resume or args.eval_only
    checkpoint = read_checkpoint(selected) if selected else None
    if checkpoint is not None:
        base = {key: value for key, value in checkpoint["config"].items() if key != "normalization"}
        config, args = parse_config(argv, base_mapping=base)
    if config.training.smoke_test:
        smoke_configuration(config)
        if checkpoint is None and config.training.run_id == "ugrec_v3_beauty":
            config.training.run_id = "ugrec_v3_beauty_smoke"
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("The joint_training entry point currently supports one process; use one CPU or GPU.")
    device_name = config.training.device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)
    if device.type not in {"cpu", "cuda"} or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Choose an available CPU or CUDA device.")
    torch.set_num_threads(config.training.cpu_threads)
    configured_directory = Path(config.training.output_dir) / config.data.category / config.training.run_id
    directory = Path(selected).expanduser().resolve().parent if selected else configured_directory
    if checkpoint is not None and args.resume:
        if any(key in args.override_mapping.get("training", {}) for key in ("output_dir", "run_id")):
            if configured_directory.resolve() != directory:
                raise ValueError("Resume writes into its checkpoint directory; keep the original run_id/output_dir.")
    if checkpoint is None and directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Experiment directory is not empty: {directory}. Use a distinct --run-id or --resume.")
    directory.mkdir(parents=True, exist_ok=True)
    mode = "eval" if args.eval_only else "resume" if args.resume else "train"
    initialize_logging(directory, mode)
    logging.info("Joint experiment: %s, device=%s, contrastive=%s",
                 directory, device, config.contrastive.enabled)
    if checkpoint is None:
        write_json(directory / "config.json", config.to_dict())
    # Dataset helpers only need this local logging/cache coordination interface.
    local = SimpleNamespace(is_main_process=True, main_process_first=nullcontext)
    dataset, tokenizer, raw, train_ids, source = load_data(config, local)
    seed_run(config.training.seed, config.training.deterministic)
    ae = (GeometryAE(raw.shape[1], config.ae) if checkpoint is not None
          else initialize_random_ae(config, raw, train_ids))
    model = JointModel(config, raw, ae=ae).to(device)
    splits = dataset.split()
    if config.training.smoke_test:
        splits = {key: values.select(range(min(8, len(values)))) for key, values in splits.items()}
    samples = tokenizer.tokenize(splits)
    if config.training.smoke_test:
        samples = {key: values.select(range(min(16 if key == "train" else 8, len(values))))
                   for key, values in samples.items()}
    source = {**source, "sample_source_sha256": json_hash(tokenizer._sample_source),
              "sample_counts": {key: len(values) for key, values in samples.items()}}
    loader_generator = torch.Generator().manual_seed(config.training.seed)
    train_loader = DataLoader(samples["train"], batch_size=config.recommender.train_batch_size,
                              shuffle=True, num_workers=config.training.num_workers, generator=loader_generator)
    validation_loader = DataLoader(samples["val"], batch_size=config.evaluation.eval_batch_size,
                                   num_workers=config.training.num_workers)
    test_loader = DataLoader(samples["test"], batch_size=config.evaluation.eval_batch_size,
                             num_workers=config.training.num_workers)
    if not len(train_loader):
        raise ValueError("The recommendation training loader is empty.")
    total_steps = config.training.epochs * len(train_loader)
    if config.training.max_steps is not None:
        total_steps = min(total_steps, config.training.max_steps)
    trainer = JointTrainer(model, config, train_ids, source, total_steps, loader_generator)
    if checkpoint is not None:
        load_checkpoint(checkpoint, trainer, resume=bool(args.resume))
    diagnostics_ids = train_ids[:min(512, len(train_ids))]
    model.ae.eval()
    if trainer.diagnostics_reference is None:
        trainer.diagnostics_reference = model.encode_ids(diagnostics_ids.to(device), grad=False).cpu()
    reference = trainer.diagnostics_reference
    best_epoch, best_score, stale_epochs = 0, -math.inf, 0
    if checkpoint is not None:
        best_epoch, best_score = checkpoint["best_epoch"], checkpoint["best_score"]
        stale_epochs = checkpoint["stale_epochs"]
    tb = JointTensorBoard(directory / "tensorboard")
    try:
        if args.eval_only:
            results = evaluate(model, test_loader, config, tokenizer)
            write_json(directory / "eval_results.json", {"checkpoint": str(selected), "test_results": results,
                                                         "evaluation": config.to_dict()["evaluation"]})
            logging.info("Testing checkpoint: %s", selected)
            log_test_results(results)
            tb.log_test(results, max(best_epoch, 0))
            return results
        start_epoch = int(checkpoint["epoch"]) + 1 if checkpoint is not None else 1
        latest, best = directory / "latest.pt", directory / "best.pt"
        if args.resume and best_epoch and not best.is_file():
            raise FileNotFoundError("Resume requires the best.pt from the same experiment directory.")
        last_validation = int(checkpoint.get("last_validation_epoch", checkpoint["epoch"])) if checkpoint else 0
        for epoch in range(start_epoch, config.training.epochs + 1):
            if trainer.global_step >= total_steps:
                break
            epoch_metrics, counts = {}, {}
            epoch_steps = min(len(train_loader), total_steps - trainer.global_step)
            with tqdm(total=epoch_steps, desc=f"Training - [Epoch {epoch}]",
                      unit="batch", dynamic_ncols=True) as progress:
                for batch_number, batch in enumerate(train_loader):
                    if trainer.global_step >= total_steps:
                        break
                    batch = {key: value.to(device) for key, value in batch.items()}
                    values = trainer.step(batch, epoch, diagnostics=batch_number == 0)
                    for key, value in values.items():
                        epoch_metrics[key] = epoch_metrics.get(key, 0.) + value
                        counts[key] = counts.get(key, 0) + 1
                    progress.update(1)
            epoch_metrics = {key: value / counts[key] for key, value in epoch_metrics.items()}
            row = {"epoch": epoch, "global_step": trainer.global_step, "ae_steps": trainer.ae_steps,
                   "rec_steps": trainer.rec_steps, "training": epoch_metrics}
            logging.info("[Epoch %s] step=%s %s", epoch, trainer.global_step,
                         format_kv(epoch_metrics, _TRAIN_LOG_KEYS))
            tb.log_training(epoch_metrics, epoch)
            final = epoch == config.training.epochs or trainer.global_step >= total_steps
            stop = False
            if epoch % config.evaluation.eval_interval == 0 or final:
                validation = evaluate(model, validation_loader, config, tokenizer)
                if not all(math.isfinite(value) for value in validation.values()):
                    raise ValueError("Non-finite recommendation validation metric.")
                row["validation"] = validation
                row["latents"] = latent_diagnostics(model, diagnostics_ids, reference)
                stale_epochs += epoch - last_validation
                last_validation = epoch
                score = validation[config.evaluation.val_metric]
                if score > best_score:
                    best_epoch, best_score, stale_epochs = epoch, score, 0
                    save_checkpoint(best, trainer, epoch=epoch, best_epoch=best_epoch,
                                    best_score=best_score, stale_epochs=stale_epochs,
                                    last_validation_epoch=last_validation)
                    logging.info("[Epoch %s] Saved best checkpoint (%s=%s) to %s",
                                 epoch, config.evaluation.val_metric, _fmt_number(score), best)
                stop = stale_epochs >= config.evaluation.patience
                log_ranking_line(f"[Epoch {epoch}] Val Results: ", validation,
                                 extra=f"best_epoch={best_epoch}", table=True)
                tb.log_validation(validation, epoch)
                tb.log_latents(row["latents"], epoch)
            append_metrics(directory / "metrics.jsonl", row)
            if epoch % config.training.save_interval == 0 or final or stop:
                save_checkpoint(latest, trainer, epoch=epoch, best_epoch=best_epoch,
                                best_score=best_score, stale_epochs=stale_epochs,
                                last_validation_epoch=last_validation)
            # Report the running validation best only while another epoch will still run.
            # The finishing epoch loads best.pt and optionally evaluates test.
            if on_validation is not None and "validation" in row and not stop and not final:
                on_validation(epoch, float(best_score))
            if stop:
                logging.info("Early stopping at epoch %s", epoch)
                break
        if not best.is_file():
            raise RuntimeError("No validated best checkpoint is available.")
        chosen = read_checkpoint(best)
        load_checkpoint(chosen, trainer)
        results = evaluate(model, test_loader, config, tokenizer) if config.training.evaluate_test else None
        if results is None:
            model.refresh_catalog()
        atomic_write(directory / "catalog_latents.npy",
                     lambda stream: np.save(stream, model.catalog[1:].cpu().numpy(), allow_pickle=False))
        summary = {"best_epoch": chosen["best_epoch"], "best_val_score": chosen["best_score"],
                   "checkpoint": str(best), "test_results": results, "smoke_test": config.training.smoke_test,
                   "contrastive_enabled": config.contrastive.enabled}
        write_json(directory / "results.json", summary)
        logging.info("Best epoch: %s, Best val score: %s",
                     chosen["best_epoch"], _fmt_number(chosen["best_score"]))
        if results is not None:
            logging.info("Testing checkpoint: %s", best)
            log_test_results(results)
            tb.log_test(results, chosen["best_epoch"])
        else:
            logging.info("Validation-only run: test evaluation skipped; selected checkpoint %s", best)
        return summary
    finally:
        tb.close()


if __name__ == "__main__":
    main()
