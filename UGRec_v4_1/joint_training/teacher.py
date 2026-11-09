"""Load fixed alignment targets with the same train/source identity as the AE."""

from pathlib import Path

import torch

from misc.artifacts import array_hash


def initialize_teacher(model, config, source, checkpoint=None):
    if not config.ranking.enabled or config.ranking.teacher_weight <= 0:
        return source
    if checkpoint is not None:
        values = checkpoint["model"]["teacher"][1:]
        if not bool(checkpoint["model"]["teacher_ready"]):
            raise ValueError("Checkpoint has no initialized alignment teacher.")
        # Strict model loading below restores the exact stored teacher tensor.
        teacher_hash = array_hash(values.numpy())
    else:
        if not config.ranking.teacher_path:
            raise ValueError("teacher_weight>0 requires ranking.teacher_path for a new training run.")
        artifact = torch.load(Path(config.ranking.teacher_path), map_location="cpu", weights_only=True)
        for key in ("embeddings_sha256", "item_order_sha256", "train_mask_sha256"):
            if artifact.get("source", {}).get(key) != source.get(key):
                raise ValueError(f"Alignment teacher source mismatch: {key}")
        model.set_teacher(artifact["values"])
        teacher_hash = array_hash(model.teacher[1:].cpu().numpy())
    return {**source, "alignment_teacher_sha256": teacher_hash}
