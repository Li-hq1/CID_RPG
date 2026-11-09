"""Explicit settings for online-AE FMRec; legacy FlowRec settings stay separate."""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

from .config import JointAEConfig


@dataclass
class FMRecSettings:
    method: str = "fmrec"  # fmrec | ce_only (same bridge) | history_ce
    num_blocks: int = 4
    heads: int = 4
    dropout: float = .1
    emb_dropout: float = .3
    lambda_uncertainty: float = .001
    eps: float = .001
    eps_reverse: float = .001
    sample_N: int = 30
    sampling_method: str = "mode"
    s_modsamp: float = 1.
    m_logNorm: float = 1.
    s_logNorm: float = .6
    last: int = 2
    fm_weight: float = 1.
    ce_weight: float = .2
    mse_weight: float = 0.
    mask_ratio: float = 1.
    # Retain the existing joint encoder's per-item representation, not L2 cosine.
    latent_normalization: str = "layer"


@dataclass
class FMRecData:
    category: str = "Beauty"
    cache_dir: str = "cache"
    max_item_seq_len: int = 50


@dataclass
class FMRecTraining:
    epochs: int = 500  # reference maximum; campaigns explicitly record their budget
    batch_size: int = 512
    seed: int = 2024
    device: str = "cuda:0"
    lr: float = .001
    weight_decay: float = 0.
    decay_step: int = 100
    gamma: float = .1
    local_weight: float = .1
    eval_interval: int = 20
    eval_batch_size: int = 128
    selection_users: int = 4096
    patience: int = 5
    cpu_threads: int = 2
    run_id: str = "fmrec_joint_s2024"
    output_dir: str = "outputs/fmrec/runs"
    max_steps: int | None = None
    smoke_test: bool = False


@dataclass
class FMRecConfig:
    data: FMRecData = field(default_factory=FMRecData)
    ae: JointAEConfig = field(default_factory=lambda: JointAEConfig(hidden_dims=(512, 256), lr=1e-4))
    fmrec: FMRecSettings = field(default_factory=FMRecSettings)
    training: FMRecTraining = field(default_factory=FMRecTraining)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        types = {"data": FMRecData, "ae": JointAEConfig, "fmrec": FMRecSettings, "training": FMRecTraining}
        if set(value)-set(types):
            raise ValueError("Unknown FMRec configuration section.")
        result = cls(**{key: types[key](**settings) for key, settings in value.items()})
        result.validate()
        return result

    @classmethod
    def read(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def validate(self):
        self.ae.validate()
        f, t = self.fmrec, self.training
        if f.method not in {"fmrec", "ce_only", "history_ce"}:
            raise ValueError("Unknown FMRec method.")
        if f.method != "fmrec" and f.fm_weight != 0:
            raise ValueError("CE ablations must explicitly set fm_weight=0.")
        if f.method == "history_ce" and f.mse_weight:
            raise ValueError("History-only baseline does not use the reconstruction decoder.")
        if f.heads < 1 or self.ae.latent_dim < 2 or self.ae.latent_dim % f.heads:
            raise ValueError("AE latent width must be divisible by attention heads.")
        if f.num_blocks < 1 or not 0 <= f.last < f.num_blocks:
            raise ValueError("last must select an intermediate Transformer block.")
        for name in ["fm_weight", "ce_weight", "mse_weight", "mask_ratio", "lambda_uncertainty", "s_logNorm"]:
            if not math.isfinite(getattr(f, name)) or getattr(f, name) < 0:
                raise ValueError(f"Invalid fmrec.{name}.")
        if f.ce_weight <= 0 or f.latent_normalization not in {"layer", "none"}:
            raise ValueError("Positive CE weight and explicit latent normalization required.")
        if not 0 <= f.dropout < 1 or not 0 <= f.emb_dropout < 1:
            raise ValueError("Invalid dropout.")
        if not 0 < f.eps < 1 or not 0 < f.eps_reverse < 1 or f.sample_N < 1:
            raise ValueError("Invalid FMRec time range or sampling steps.")
        if f.sampling_method not in {"mode", "uniform", "logit_normal", "cosmap"} or not -1 <= f.s_modsamp <= 1:
            raise ValueError("Invalid time sampler (mode scale restricted to its monotone range).")
        if self.data.max_item_seq_len < 1 or self.ae.frozen:
            raise ValueError("This experiment requires nonempty histories and a trainable AE encoder.")
        for key in ["epochs", "batch_size", "decay_step", "eval_interval", "eval_batch_size", "selection_users", "patience", "cpu_threads"]:
            if getattr(t, key) < 1:
                raise ValueError(f"Invalid training.{key}.")
        if t.max_steps is not None and t.max_steps < 1:
            raise ValueError("max_steps must be positive.")
        if not 0 < t.gamma <= 1 or t.lr <= 0 or t.local_weight < 0 or t.weight_decay < 0:
            raise ValueError("Invalid optimizer or AE loss settings.")
