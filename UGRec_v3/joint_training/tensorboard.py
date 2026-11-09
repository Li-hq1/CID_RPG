"""Optional TensorBoard logging for joint training (no Accelerate dependency)."""

from __future__ import annotations

import logging
from pathlib import Path


_LOSS_KEYS = {
    "ae_loss", "rf_loss", "ae_contrastive",
}
_TRAIN_KEYS = {
    "ae_lr", "rec_lr", "contrastive_weight", "contrastive_negative_count", "ae_updated", "rec_updated",
}
_GRAD_KEYS = {
    "ae_grad_norm", "ae_contrastive_grad_norm",
}


def metric_tag(key):
    if key.startswith("ae/"):
        return f"AE/{key.split('/', 1)[1]}"
    if key in _LOSS_KEYS:
        return f"Loss/{key}"
    if key in _TRAIN_KEYS:
        return f"Train/{key}"
    if key in _GRAD_KEYS:
        return f"Grad/{key}"
    return f"Train/{key.replace('/', '_')}"


def scalar_items(values, *, prefix):
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if number != number:  # NaN
            continue
        yield f"{prefix}/{key}" if prefix else metric_tag(key), number


class JointTensorBoard:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            logging.warning("TensorBoard is unavailable; install tensorboard to enable event files.")
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.directory))
        logging.info("TensorBoard log directory: %s", self.directory)

    @property
    def enabled(self):
        return self.writer is not None

    def log(self, values, step, *, prefix=""):
        if self.writer is None:
            return
        for tag, number in scalar_items(values, prefix=prefix):
            self.writer.add_scalar(tag, number, global_step=step)

    def log_training(self, metrics, epoch):
        self.log(metrics, epoch)

    def log_validation(self, metrics, epoch):
        self.log(metrics, epoch, prefix="Val_Metric")

    def log_latents(self, metrics, epoch):
        self.log(metrics, epoch, prefix="Latent")

    def log_test(self, metrics, step):
        self.log(metrics, step, prefix="Test_Metric")

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
