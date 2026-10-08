from pathlib import Path
import os
import torch


def save_checkpoint(model, path, accelerator, *, optimizer=None, scheduler=None,
                    epoch=0, best_epoch=0, best_val_score=-1, config=None):
    if accelerator.is_main_process:
        model_config = config if config is not None else getattr(accelerator.unwrap_model(model), "config", {})
        feature_identity = model_config.get("contrec_vae_artifact_id")
        geometry_identity = model_config.get("geometry_ae_artifact_id")
        if model_config.get("sent_emb_compression") == "geometry_ae" and not geometry_identity:
            raise ValueError("Cannot save a geometry AE checkpoint without its feature artifact identity.")
        if model_config.get("sent_emb_compression") == "contrec_vae" and not feature_identity:
            raise ValueError("Cannot save a VAE recommendation checkpoint without its feature artifact identity.")
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
        if feature_identity is not None:
            payload["contrec_vae_artifact_id"] = feature_identity
        if geometry_identity is not None:
            payload["geometry_ae_artifact_id"] = geometry_identity
        payload["sent_emb_compression"] = model_config.get("sent_emb_compression", "pca")
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


def resolve_checkpoint(path, *, prefer="latest"):
    if prefer not in {"latest", "best"}:
        raise ValueError("Checkpoint preference must be latest or best.")
    path = Path(path)
    if path.is_dir():
        candidates = list(path.glob(f"{prefer}*.pth")) + list(path.glob(f"{prefer}*.pt"))
        if not candidates and prefer == "latest":
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
    validate_feature_identity(model, checkpoint)
    # Legacy checkpoints are plain model state dictionaries.
    if "format_version" in checkpoint and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        return checkpoint
    model.load_state_dict(checkpoint)
    return {}


def validate_feature_identity(model, checkpoint):
    """Check feature provenance before reading weights or resume metadata."""
    current_config = getattr(model, "config", {})
    method = current_config.get("sent_emb_compression")
    if method is None:
        method = "geometry_ae" if current_config.get("geometry_ae_artifact_id") else (
            "contrec_vae" if current_config.get("contrec_vae_artifact_id") else "pca")
    saved_method = checkpoint.get("sent_emb_compression")
    if saved_method is None:
        saved_method = "contrec_vae" if checkpoint.get("contrec_vae_artifact_id") else "pca"
    if method != saved_method:
        raise ValueError("Recommendation checkpoint VAE/AE artifact mismatch: compression method differs.")
    current_geometry = current_config.get("geometry_ae_artifact_id")
    if method == "geometry_ae" and not current_geometry:
        raise ValueError("Geometry AE feature identity is missing.")
    if current_geometry != checkpoint.get("geometry_ae_artifact_id"):
        raise ValueError("Recommendation checkpoint geometry AE artifact mismatch.")
    current_identity = current_config.get("contrec_vae_artifact_id")
    saved_identity = checkpoint.get("contrec_vae_artifact_id")
    if current_config.get("sent_emb_compression") == "contrec_vae" and not current_identity:
        raise ValueError("VAE feature identity is missing from the current recommendation model.")
    if current_identity != saved_identity:
        raise ValueError("Recommendation checkpoint VAE artifact mismatch. Use matching features and weights.")


def prepare_test_model(model, test_dataloader, accelerator, checkpoint_path):
    accelerator.wait_for_everyone()
    model = accelerator.unwrap_model(model)
    load_checkpoint(model, checkpoint_path, accelerator.device)
    return accelerator.prepare(model, test_dataloader)
