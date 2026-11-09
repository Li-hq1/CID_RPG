"""Editable joint-training settings.

Change defaults in the dataclasses below. Type checks, value checks, and CLI
parsing live in config_parse.py so this file stays a list of settings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class DataConfig:
    dataset: str = "AmazonReviews2014"
    category: str = "Beauty"
    cache_dir: str = "cache/"
    metadata: str = "sentence"
    split: str = "leave_one_out"
    max_item_seq_len: int = 50
    num_proc: int = 1

    sent_emb_model: str = "Qwen/Qwen3-Embedding-4B"
    sent_emb_model_paths: dict[str, str] = field(default_factory=lambda: {
        "Qwen/Qwen3-Embedding-4B": "cache/models/Qwen/Qwen3-Embedding-4B",
        "sentence-transformers/sentence-t5-base": "cache/models/sentence-transformers/sentence-t5-base",
    })
    sent_emb_dim: int = 2560
    sent_emb_batch_size: int = 8
    sent_emb_max_length: int | None = 512
    sent_emb_dtype: str = "auto"
    sent_emb_device: str | None = None
    sent_emb_local_files_only: bool = True
    sent_emb_prompt: str = ""
    sent_emb_normalize: bool = True


@dataclass
class JointAEConfig:
    architecture: str = "mlp"
    hidden_dims: tuple[int, ...] = (1024, 512)
    latent_dim: int = 128

    lr: float = 3e-5
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    batch_size: int = 512
    update_every: int = 1
    frozen: bool = False
    scheduler: str = "cosine"
    warmup_ratio: float = 0.05

    cosine_weight: float = 0.1
    variance_weight: float = 0.1
    correlation_weight: float = 1.0
    relation_mode: str = "none"
    relation_weight: float = 0.1
    relation_temperature: float = 0.1

    def validate(self):
        # GeometryAE calls this on whatever config object it receives.
        from .config_parse import validate_ae
        validate_ae(self)


@dataclass
class RecommenderConfig:
    encoder_type: str = "hstu"
    n_embd: int = 512
    n_layer: int = 4
    n_head: int = 4
    n_inner: int = 1024
    activation_function: str = "gelu_new"
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-12

    hstu_hidden_dim: int = 64
    hstu_attention_dim: int = 64
    hstu_use_group_norm: bool = True
    hstu_output_dropout: float = 0.1
    hstu_attention_dropout: float = 0.0
    hstu_attention_scaling: str = "input_length"

    diffloss_w: int = 512
    diffloss_d: int = 3
    num_sampling_steps: int = 20
    diffusion_batch_mul: int = 1
    diff_temperature: float = 1.0
    grad_checkpointing: bool = False
    tiger_uncond_prob: float = 0.1
    tiger_guidance_weight: float = 2.0
    use_rectified_flow: bool = True
    rectified_flow_steps: int = 1000
    ode_solver: str = "euler"

    lr: float = 5e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    train_batch_size: int = 256
    update_every: int = 1
    scheduler: str = "cosine"
    warmup_ratio: float = 0.1


@dataclass
class ContrastiveConfig:
    enabled: bool = False
    max_queries: int = 1024
    temperature: float = 0.04
    weight: float = 0.01
    num_negatives: int = 64


@dataclass
class RankingConfig:
    enabled: bool = False
    weight: float = 1.0
    rf_weight: float = 1.0
    local_weight: float = 0.1
    temperature: float = 0.07
    prediction_weight: float = 0.0
    endpoint_weight: float = 0.0
    rollout_weight: float = 0.0
    rollout_steps: int = 4
    rollout_samples: int = 1
    rollout_objective: str = "sample_ce"
    prediction_min_time: float = 0.5
    teacher_weight: float = 0.0
    teacher_path: str = ""
    teacher_dim: int = 128
    teacher_min_time: float = 0.5
    teacher_ae_grad: bool = True
    history_ae_grad: bool = True
    catalog_ae_grad: bool = True
    prior_mode: str = "noise"
    prior_noise: float = 0.3


@dataclass
class TrainingConfig:
    epochs: int = 100
    seed: int = 2024
    device: str = "auto"
    run_id: str = "ugrec_v4_1_beauty"
    output_dir: str = "outputs/joint_training"
    save_interval: int = 1
    deterministic: bool = True
    num_workers: int = 0
    cpu_threads: int = 2
    smoke_test: bool = False
    max_steps: int | None = None
    evaluate_test: bool = True


@dataclass
class EvaluationConfig:
    eval_batched_retrieval: bool = False
    prediction_source: str = "flow"
    random_seed: int = 20261028
    eval_sampling_mode: str = "diffusion_multi"
    eval_num_samples: int = 32
    eval_candidates_per_sample: int = 1
    eval_candidate_metric: str = "l2"
    eval_candidate_ranking: str = "support"
    eval_support_temperature: float = 1.0
    eval_log_sampling_stats: bool = True
    topk: tuple[int, ...] = (5, 10)
    metrics: tuple[str, ...] = ("recall", "ndcg")
    val_metric: str = "ndcg@10"
    eval_batch_size: int = 64
    eval_interval: int = 5
    patience: int = 20
    tiger_knn_metric: str = "cosine"
    temperature: float = 0.07


@dataclass
class JointConfig:
    data: DataConfig = field(default_factory=DataConfig)
    ae: JointAEConfig = field(default_factory=JointAEConfig)
    recommender: RecommenderConfig = field(default_factory=RecommenderConfig)
    contrastive: ContrastiveConfig = field(default_factory=ContrastiveConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self):
        from .config_parse import validate_joint
        validate_joint(self)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        from .config_parse import build_config
        return build_config(values)


def parse_config(argv=None, *, base_mapping=None):
    from .config_parse import parse_config as _parse_config
    return _parse_config(argv, base_mapping=base_mapping)
