import math
from logging import getLogger

from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from transformers.optimization import get_scheduler
from tqdm import tqdm

from engine.checkpoint import load_checkpoint, save_checkpoint
from engine.evaluate import evaluate
from misc.logging import config_for_log, get_file_name, log


def train(model, train_dataloader, val_dataloader, config, tokenizer, accelerator,
          checkpoint_path, resume_checkpoint=None):
    logger = getLogger()
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    total_steps = config["steps"] if config["steps"] is not None else len(train_dataloader) * config["epochs"]
    if total_steps == 0:
        log("No training steps needed.", accelerator, logger)
        return model, None, None
    if not len(train_dataloader):
        raise ValueError("The training dataloader is empty.")
    warmup_steps = config.get("warmup_steps")
    if warmup_steps is None:
        warmup_steps = math.ceil(total_steps * config.get("warmup_ratio", 0.0))
    log(f"Cosine scheduler: {warmup_steps} warmup steps / {total_steps} planned steps "
        f"(about {warmup_steps / len(train_dataloader):.2f} warmup epochs before sharding).",
        accelerator, logger)
    scheduler = get_scheduler(
        "cosine", optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler,
    )
    start_epoch = 1
    best_epoch, best_val_score = 0, -1
    if resume_checkpoint is not None:
        checkpoint = load_checkpoint(accelerator.unwrap_model(model), resume_checkpoint, accelerator.device)
        if not checkpoint:
            log("Legacy checkpoint contains model weights only; optimizer and epoch start fresh.",
                accelerator, logger, level="warning")
        if checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_epoch = int(checkpoint.get("best_epoch", 0))
        best_val_score = float(checkpoint.get("best_val_score", -1))
        log(f"Resumed checkpoint from {resume_checkpoint} at epoch {start_epoch}", accelerator, logger)
    accelerator.init_trackers(
        project_name=get_file_name(config), config=config_for_log(config),
        init_kwargs={"tensorboard": {"flush_secs": 60}},
    )
    # Legacy steps controls the epoch count; training completes whole epochs.
    n_epochs = math.ceil(total_steps / (len(train_dataloader) * accelerator.num_processes))
    if start_epoch > n_epochs:
        log(f"Checkpoint has already reached the target of {n_epochs} epochs.", accelerator, logger)
        return model, None, None
    for epoch in range(start_epoch, n_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_dataloader, total=len(train_dataloader), desc=f"Training - [Epoch {epoch}]"):
            optimizer.zero_grad()
            outputs = model(batch)
            loss = outputs.loss
            accelerator.backward(loss)
            if config["max_grad_norm"] is not None:
                clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_dataloader)
        accelerator.log({"Loss/train_loss": avg_loss}, step=epoch)
        log(f"[Epoch {epoch}] Train Loss: {avg_loss}", accelerator, logger)
        should_stop = False
        if epoch % config["eval_interval"] == 0:
            results = evaluate(model, val_dataloader, config, tokenizer, accelerator, split="val")
            if accelerator.is_main_process:
                accelerator.log({f"Val_Metric/{key}": value for key, value in results.items()}, step=epoch)
            log(f"[Epoch {epoch}] Val Results: {results}", accelerator, logger)
            score = results[config["val_metric"]]
            if score > best_val_score:
                best_epoch, best_val_score = epoch, score
            if config["patience"] is not None and epoch - best_epoch >= config["patience"]:
                log(f"Early stopping at epoch {epoch}", accelerator, logger)
                should_stop = True
        if epoch % config["save_interval"] == 0 or epoch == n_epochs or should_stop:
            save_checkpoint(model, checkpoint_path, accelerator, optimizer=optimizer,
                            scheduler=scheduler, epoch=epoch, best_epoch=best_epoch,
                            best_val_score=best_val_score, config=config)
            log(f"[Epoch {epoch}] Saved latest checkpoint to {checkpoint_path}", accelerator, logger)
        if should_stop:
            break
    log(f"Best epoch: {best_epoch}, Best val score: {best_val_score}", accelerator, logger)
    return model, best_epoch, best_val_score
