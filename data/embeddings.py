"""Shared sentence encoder and raw embedding cache format."""

import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def embedding_settings(config):
    return {
        "model": config["sent_emb_model"],
        "dim": config["sent_emb_dim"],
        "max_length": config.get("sent_emb_max_length"),
        "dtype": config.get("sent_emb_dtype", "float32"),
        "prompt": config.get("sent_emb_prompt", ""),
        "normalize": config.get("sent_emb_normalize", False),
    }


def sentence_embedding_cache_path(config, cache_dir):
    settings = embedding_settings(config)
    name = os.path.basename(os.path.normpath(config["sent_emb_model"]))
    # Keep the original sentence-T5 caches available for legacy experiments.
    legacy = settings == {
        "model": "sentence-transformers/sentence-t5-base", "dim": 768,
        "max_length": None, "dtype": "float32", "prompt": "", "normalize": False,
    }
    if not legacy:
        key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
        name = f"{name}_{key}"
    return os.path.join(cache_dir, "processed", f"{name}.sent_emb")


def read_sentence_embeddings(path, dim, n_items=None):
    values = np.fromfile(path, dtype=np.float32)
    if values.size % dim or (n_items is not None and values.size != n_items * dim):
        raise ValueError(f"Invalid embedding cache {path}: expected {n_items} rows of {dim} float32 values.")
    embeddings = values.reshape(-1, dim)
    if not np.isfinite(embeddings).all():
        raise ValueError(f"Non-finite values in embedding cache {path}.")
    return embeddings


def encode_sentences(sentences, config):
    model_name = config["sent_emb_model"]
    local_path = config.get("sent_emb_model_paths", {}).get(model_name)
    source = Path(local_path or model_name).expanduser()
    if not source.is_absolute() and (local_path or (PROJECT_ROOT / source).is_dir()):
        source = PROJECT_ROOT / source
    is_local = source.is_dir()
    offline = config.get("sent_emb_local_files_only", True)
    if local_path and not is_local:
        raise FileNotFoundError(f"Configured embedding model directory does not exist: {source}")

    device = torch.device(config.get("sent_emb_device") or config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run on a GPU-enabled server or select --sent_emb_device=cpu.")
    dtype_name = config.get("sent_emb_dtype", "float32")
    if dtype_name == "auto":
        if device.type == "cuda":
            with torch.cuda.device(device):
                dtype_name = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        else:
            dtype_name = "float32"
    dtype = getattr(torch, dtype_name)
    model_kwargs = {"dtype": dtype}
    if is_local:
        with (source / "config.json").open() as f:
            is_qwen = json.load(f).get("model_type") == "qwen3"
    else:
        is_qwen = "qwen3-embedding" in model_name.lower()
    if is_qwen:
        model_kwargs["attn_implementation"] = "sdpa"
    model = SentenceTransformer(
        str(source) if is_local else model_name,
        local_files_only=offline or is_local,
        device=str(device),
        model_kwargs=model_kwargs,
    )
    try:
        if config.get("sent_emb_max_length") is not None:
            model.max_seq_length = config["sent_emb_max_length"]
        if is_qwen:
            model.tokenizer.padding_side = "left"
        model[0].auto_model.config.use_cache = False
        dim = model.get_embedding_dimension()
        if dim != config["sent_emb_dim"]:
            raise ValueError(f"Encoder produces {dim} dimensions, but sent_emb_dim={config['sent_emb_dim']}.")
        model.eval()
        with torch.inference_mode():
            embeddings = model.encode(
                sentences,
                prompt=config.get("sent_emb_prompt", ""),
                convert_to_numpy=True,
                batch_size=config["sent_emb_batch_size"],
                show_progress_bar=True,
                device=str(device),
                normalize_embeddings=config.get("sent_emb_normalize", False),
            )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.shape != (len(sentences), dim) or not np.isfinite(embeddings).all():
            raise ValueError(f"Invalid encoder output: shape={embeddings.shape}, expected {(len(sentences), dim)}.")
        if config.get("sent_emb_normalize", False):
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            if np.any(norms == 0):
                raise ValueError("Encoder produced zero vectors; cannot normalize embeddings.")
            embeddings = embeddings / norms
        return embeddings
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
