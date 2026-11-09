"""Joint model persistence, including both optimization timelines and RNGs."""

from contextlib import contextmanager
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from misc.artifacts import atomic_write


FORMAT_VERSION = 1


def random_state(device=None):
    state = np.random.get_state()
    cuda = None
    if device is not None and torch.device(device).type == "cuda":
        cuda = {"device": str(device), "state": torch.cuda.get_rng_state(device)}
    return {
        "python": random.getstate(),
        "numpy": (state[0], torch.from_numpy(state[1].astype(np.int64)), *state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": cuda,
    }


def restore_random_state(state, device=None):
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, keys.cpu().numpy().astype(np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"].cpu())
    cuda = state["cuda"]
    if cuda and device is not None and torch.device(device).type == "cuda":
        if isinstance(cuda, dict):
            value = cuda["state"]
        else:
            index = torch.device(device).index or 0
            value = cuda[index] if index < len(cuda) else cuda[0]
        torch.cuda.set_rng_state(value.cpu(), device)


@contextmanager
def evaluation_randomness(device, seed):
    """Evaluation noise that does not consume the training RNG stream."""
    device = torch.device(device)
    state = random_state(device)
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.default_generator.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.default_generators[device.index if device.index is not None else torch.cuda.current_device()].manual_seed(seed)
        yield
    finally:
        restore_random_state(state, device)


def migrate_checkpoint(checkpoint):
    """Load earlier AE-only checkpoints without exposing retired experiment settings."""
    config = checkpoint.get("config")
    if isinstance(config, dict):
        from .config import RankingConfig, EvaluationConfig
        config.setdefault("ranking", {})
        for key, value in asdict(RankingConfig()).items():
            config["ranking"].setdefault(key, value)
        if isinstance(config.get("evaluation"), dict):
            for key in ("prediction_source", "random_seed", "eval_batched_retrieval"):
                config["evaluation"].setdefault(key, getattr(EvaluationConfig(), key))
        # Fields added with the final scheme; their defaults keep older runs unchanged.
        from .config import FlowRankingConfig, FMConfig
        config.setdefault("fm", asdict(FMConfig()))
        rec = config.get("recommender", {})
        if rec.get("input_code_levels", 0):
            raise ValueError("Content-code checkpoints require archive/UGRec_v5_content_codes_20261106.")
        for key in ("input_code_levels", "input_code_size", "input_code_iterations", "input_code_seed"):
            rec.pop(key, None)
        flow = config.setdefault("flow_ranking", {})
        if flow.get("vlad_clusters", 0) and "vlad_condition" not in flow:
            raise ValueError("Residual VLAD checkpoints require archive/UGRec_v5_vlad_residual_20261106.")
        for key, value in asdict(FlowRankingConfig()).items():
            flow.setdefault(key, value)
    if isinstance(config, dict) and "bridge" in config and "contrastive" not in config:
        config["contrastive"] = config.pop("bridge")
    if isinstance(config, dict) and isinstance(config.get("contrastive"), dict):
        contrastive = config["contrastive"]
        schedules = ("lambda_ae", "lambda_rec", "start_epoch_ae", "start_epoch_rec",
                     "ramp_epochs_ae", "ramp_epochs_rec")
        mode = contrastive.get("mode", "ae_only")
        # Older scheduled configs without an explicit mode used a different gradient route.
        legacy_implicit_mode = "mode" not in contrastive and any(key in contrastive for key in schedules)
        if contrastive.get("enabled", False) and (mode != "ae_only" or legacy_implicit_mode):
            raise ValueError("Only AE-only contrastive checkpoints are supported.")
        if contrastive.get("loss_type", "infonce") != "infonce":
            raise ValueError("Only InfoNCE contrastive checkpoints are supported.")
        for key in ("mode", "loss_type", *schedules):
            contrastive.pop(key, None)
    model = checkpoint.get("model")
    if isinstance(model, dict) and any(key.startswith("bridge_head.") for key in model):
        checkpoint["model"] = {
            key.replace("bridge_head.", "contrastive_head.", 1) if key.startswith("bridge_head.") else key: value
            for key, value in model.items()
        }
    return checkpoint


def read_checkpoint(path):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != FORMAT_VERSION or checkpoint.get("method") != "joint_training":
        raise ValueError("Expected a joint_training checkpoint, not a standalone AE or TigerDiff checkpoint.")
    return migrate_checkpoint(checkpoint)


def save_checkpoint(path, trainer, *, epoch, best_epoch, best_score, stale_epochs,
                    last_validation_epoch=None):
    state = {
        "format_version": FORMAT_VERSION,
        "method": "joint_training",
        "config": trainer.config.to_dict(),
        "source": trainer.source,
        "model": trainer.model.state_dict(),
        "trainer": trainer.state_dict(),
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "stale_epochs": stale_epochs,
        "last_validation_epoch": epoch if last_validation_epoch is None else last_validation_epoch,
        "rng": random_state(trainer.device),
    }
    atomic_write(path, lambda stream: torch.save(state, stream))


def _drop_retired_fields(section, previous, requested):
    """Ignore settings removed after older checkpoints were saved."""
    if section == "recommender" and previous.get("supervision_mode", "all_positions") != "all_positions":
        raise ValueError("Joint checkpoints must supervise every valid position.")
    if section == "contrastive" and previous.get("positions", "all_positions") != "all_positions":
        raise ValueError("Joint checkpoints must use every valid contrastive position.")
    retired = {
        "ae": ("init_artifact", "init_mode"),
        "recommender": ("supervision_mode",),
        "contrastive": ("positions",),
    }
    if section == "contrastive":
        previous.setdefault("weight", 0.01)
        previous.setdefault("num_negatives", 64)
    if section == "ranking":
        # The teacher values live in model state; moving its original artifact
        # must not invalidate a checkpoint whose source identity still matches.
        previous.pop("teacher_path", None)
        requested.pop("teacher_path", None)
    for key in retired.get(section, ()):
        previous.pop(key, None)
        requested.pop(key, None)


def load_checkpoint(checkpoint, trainer, *, resume=False):
    if checkpoint["source"] != trainer.source:
        raise ValueError("Joint checkpoint source mismatch: category, raw features, item order or train items changed.")
    # Device and output paths can change; optimization and model semantics cannot.
    migrate_checkpoint(checkpoint)
    old = checkpoint["config"]
    current = trainer.config.to_dict()
    for section in ("ae", "recommender", "contrastive", "ranking", "data", "flow_ranking", "fm"):
        previous, requested = dict(old[section]), dict(current[section])
        _drop_retired_fields(section, previous, requested)
        if section == "flow_ranking" and not previous.get("vlad_clusters", 0) and not requested.get("vlad_clusters", 0):
            # Disabled VLAD tuning does not change a control checkpoint's semantics.
            for key in ("vlad_condition", "vlad_groups", "vlad_expansion", "vlad_assignment_temperature"):
                previous.pop(key, None)
                requested.pop(key, None)
        if section == "data":
            for key in ("cache_dir", "sent_emb_model_paths", "sent_emb_device", "num_proc"):
                previous.pop(key, None)
                requested.pop(key, None)
        if previous != requested:
            raise ValueError(f"Joint checkpoint {section} configuration mismatch.")
    if resume:
        if old["evaluation"] != current["evaluation"]:
            raise ValueError("Resume cannot change evaluation settings or compare scores from different protocols.")
        for key in ("epochs", "max_steps", "seed", "deterministic", "num_workers"):
            if old["training"].get(key) != current["training"].get(key):
                raise ValueError(f"Resume cannot change training.{key}; start a separate run for a new budget.")
    trainer.model.load_state_dict(checkpoint["model"], strict=True)
    trainer.model.invalidate_catalog()
    if resume:
        trainer.load_state_dict(checkpoint["trainer"])
        restore_random_state(checkpoint["rng"], trainer.device)
