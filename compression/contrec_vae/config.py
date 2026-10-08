"""Defaults from ContRec/code/rec_tokenizer/vae.py, separate from recommender settings."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class VAEConfig:
    latent_dim: int = 128  # ContRec default is 2048; 128 is a compression choice.
    mask_ratio: float = 0.1
    sigma: float = 0.1
    seed: int = 42
    lr: float = 1e-4
    weight_decay: float = 1e-3
    # ContRec configure_loss weights: recon + beta*mu + gamma*twins.
    beta: float = 0.5
    gamma: float = 0.5
    twins_lambda: float = 0.005
    train_batch_size: int = 512
    eval_batch_size: int = 256
    epochs: int = 1000
    patience: int = 1000
    test_ratio: float = 0.1
    valid_ratio: float = 0.2
    smoke_items: int | None = None

    def validate(self):
        for key in ("latent_dim", "train_batch_size", "eval_batch_size", "epochs", "patience"):
            value = getattr(self, key)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} must be a positive integer.")
        if self.train_batch_size < 2:
            raise ValueError("train_batch_size must be at least 2 for BatchNorm.")
        for key in (
            "lr", "weight_decay", "sigma", "mask_ratio", "test_ratio", "valid_ratio",
            "beta", "gamma", "twins_lambda",
        ):
            if not math.isfinite(getattr(self, key)):
                raise ValueError(f"{key} must be finite.")
        if self.lr <= 0 or self.weight_decay < 0 or self.sigma < 0:
            raise ValueError("lr must be positive; weight_decay and sigma must be non-negative.")
        if self.beta < 0 or self.gamma < 0 or self.twins_lambda < 0:
            raise ValueError("beta, gamma, and twins_lambda must be non-negative.")
        if not 0 <= self.mask_ratio <= 1:
            raise ValueError("mask_ratio must be in [0, 1].")
        if not 0 < self.test_ratio < 1 or not 0 < self.valid_ratio < 1:
            raise ValueError("test_ratio and valid_ratio must be in (0, 1).")
        if self.smoke_items is not None and self.smoke_items < 4:
            raise ValueError("smoke_items must be at least 4.")

    def to_dict(self):
        self.validate()
        return asdict(self)
