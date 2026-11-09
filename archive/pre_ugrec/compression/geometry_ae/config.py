"""Independent geometry-preserving compression settings."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class AEConfig:
    architecture: str = "mlp"
    latent_dim: int = 128
    hidden_dims: tuple[int, ...] = (1024, 512)
    seed: int = 42
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    train_batch_size: int = 512
    eval_batch_size: int = 512
    epochs: int = 200
    patience: int = 30
    test_ratio: float = 0.1
    valid_ratio: float = 0.2
    train_items: int | None = None
    smoke_items: int | None = None
    cosine_weight: float = 0.1
    variance_weight: float = 0.1
    correlation_weight: float = 1.0
    relation_weight: float = 0.1
    relation_temperature: float = 0.1
    warmup_epochs: int = 10
    ramp_epochs: int = 20
    neighbor_interval: int = 5
    neighbor_candidates: int = 3000
    neighbor_queries: int = 256
    neighbor_k: int = 20
    collapse_rank: float = 64.0
    collapse_patience: int = 5
    relation_mode: str = "none" # batch | local | none     关系蒸馏方案，batch效果最好，利用了qwen-embedding的相关度
    local_near_pool: int = 32
    local_middle_pool: int = 128
    local_near_count: int = 8
    local_middle_count: int = 8
    local_random_count: int = 16
    local_version: int = 1

    def to_dict(self):
        self.validate()
        result = asdict(self)
        result["hidden_dims"] = list(self.hidden_dims)
        # Keep the exact configuration fingerprint of pre-local batch artifacts.
        if self.relation_mode == "batch":
            result.pop("relation_mode")
        if self.relation_mode != "local":
            for key in ("local_near_pool", "local_middle_pool", "local_near_count",
                        "local_middle_count", "local_random_count", "local_version"):
                result.pop(key)
        return result

    def validate(self):
        if self.relation_mode not in {"none", "batch", "local"}:
            raise ValueError("relation_mode must be none, batch or local.")
        for key in ("local_near_pool", "local_middle_pool", "local_near_count",
                    "local_middle_count", "local_random_count"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer.")
        if (self.local_near_count > self.local_near_pool or
                self.local_middle_pool <= self.local_near_pool or
                self.local_middle_count > self.local_middle_pool - self.local_near_pool):
            raise ValueError("Local sample counts must fit their ordered neighbor pools.")
        if type(self.local_version) is not int or self.local_version != 1:
            raise ValueError("Unsupported local relation version.")
        if self.architecture not in {"linear", "mlp"}:
            raise ValueError("architecture must be linear or mlp.")
        for key in ("latent_dim", "train_batch_size", "eval_batch_size", "epochs", "patience",
                    "ramp_epochs", "neighbor_interval", "neighbor_candidates", "neighbor_queries",
                    "neighbor_k", "collapse_patience"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer.")
        if min(self.train_batch_size, self.eval_batch_size, self.neighbor_candidates) < 2:
            raise ValueError("Batch and candidate sizes must be at least two.")
        if type(self.warmup_epochs) is not int or self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be a non-negative integer.")
        if not self.hidden_dims or any(type(dim) is not int or dim <= 0 for dim in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive integers.")
        for key in ("lr", "weight_decay", "max_grad_norm", "cosine_weight", "variance_weight",
                    "correlation_weight", "relation_weight", "relation_temperature", "collapse_rank",
                    "test_ratio", "valid_ratio"):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and non-negative.")
        if min(self.lr, self.max_grad_norm, self.relation_temperature, self.collapse_rank) <= 0:
            raise ValueError("Learning rate, gradient limit, temperature and rank threshold must be positive.")
        if not 0 < self.test_ratio < 1 or not 0 < self.valid_ratio < 1:
            raise ValueError("Split ratios must be in (0, 1).")
        for key in ("train_items", "smoke_items"):
            value = getattr(self, key)
            if value is not None and (type(value) is not int or value < (4 if key == "smoke_items" else 2)):
                raise ValueError(f"{key} is too small.")
