from pathlib import Path
import os
import torch


def save_checkpoint(model, path, accelerator, *, optimizer=None, scheduler=None,
                    epoch=0, best_epoch=0, best_val_score=-1, config=None):
    if accelerator.is_main_process:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "model": accelerator.unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_score": best_val_score,
        }
        if config is not None:
            from misc.logging import _make_loggable
            payload["config"] = {key: _make_loggable(value) for key, value in config.items()
                                 if key not in {"accelerator", "device"}}
        # A failed write must leave the previous latest checkpoint intact.
        temporary = path.with_name(path.name + ".tmp")
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def resolve_checkpoint(path):
    path = Path(path)
    if path.is_dir():
        candidates = list(path.glob("latest*.pth")) + list(path.glob("latest*.pt"))
        if not candidates:
            candidates = list(path.glob("*.pth")) + list(path.glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found in {path}")
        path = max(candidates, key=lambda candidate: (candidate.stat().st_mtime_ns, candidate.name))
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def load_checkpoint(model, path, device):
    # torch serialization accepts cpu but not Accelerate's cpu:0 tag.
    map_location = "cpu" if str(device).startswith("cpu") else device
    checkpoint = torch.load(resolve_checkpoint(path), map_location=map_location, weights_only=True)
    # Legacy checkpoints are plain model state dictionaries.
    if "format_version" in checkpoint and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        return checkpoint
    model.load_state_dict(checkpoint)
    return {}


def prepare_test_model(model, test_dataloader, accelerator, checkpoint_path):
    accelerator.wait_for_everyone()
    model = accelerator.unwrap_model(model)
    load_checkpoint(model, checkpoint_path, accelerator.device)
    return accelerator.prepare(model, test_dataloader)
