"""Train, monitor, select and export frozen geometry AE features."""

import csv
import io
import logging
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.decomposition import PCA

from compression.contrec_vae.pipeline import batch_indices, split_items as legacy_split, standardize_latents
from .cache import (ARTIFACT_FILES, FORMAT_VERSION, METHOD, artifact_directory, atomic_write,
                    experiment_key, file_hash, json_hash, read_manifest, source_metadata, write_json)
from .config import AEConfig
from .metrics import geometry_metrics, neighbor_comparison
from .model import (GeometryAE, encoder_gradient_norms, local_relation_kl, loss_terms,
                    objective_weights, relation_kl)
from .neighborhood import LocalCandidates, latent_statistics

logger = logging.getLogger(__name__)


def split_items(mask, config):
    # Subsample only after splitting so all ablations retain the original validation/test sets.
    splits = legacy_split(mask, config)
    if config.train_items is not None:
        splits["train"] = splits["train"][:config.train_items]
    return splits


def fit_input_statistics(embeddings, indices):
    selected = embeddings[indices].astype(np.float64)
    mean = selected.mean(0)
    scale = float(np.sqrt(np.mean((selected - mean) ** 2)))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError("Compression training inputs have no usable variance.")
    return torch.from_numpy(mean.astype(np.float32)), torch.tensor(scale, dtype=torch.float32)


def write_history(path, history):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(row.keys() for row in history))))
    writer.writeheader()
    writer.writerows(history)
    atomic_write(path, lambda output: output.write(stream.getvalue().encode()))


@torch.no_grad()
def encode_all(model, embeddings, batch_size, device):
    model.eval()
    result = []
    for indices in batch_indices(np.arange(len(embeddings)), batch_size):
        result.append(model.latent(torch.from_numpy(embeddings[indices]).to(device)).cpu().numpy())
    return np.concatenate(result)


def local_forward(model, embeddings, anchors, candidates, device):
    # Encode each item once; candidate popularity does not reweight reconstruction/statistics.
    unique, inverse = np.unique(np.concatenate((anchors, candidates.ravel())), return_inverse=True)
    x = torch.from_numpy(embeddings[unique]).to(device)
    z = model.latent(x)
    positions = torch.from_numpy(inverse).to(device)
    anchor_positions = positions[:len(anchors)]
    candidate_positions = positions[len(anchors):].reshape(candidates.shape)
    return (x[anchor_positions], model.decode(z[anchor_positions]), z[anchor_positions],
            x[candidate_positions], z[candidate_positions])


@torch.no_grad()
def evaluate(model, embeddings, indices, config, device, *, candidates=None, statistics=None):
    model.eval()
    if config.relation_mode == "local" and (candidates is None or statistics is None):
        raise ValueError("Local validation needs training-only candidates and latent statistics.")
    mse, cosine, relation, relation_rows, rows, local = 0., 0., 0., 0, 0, 0.
    reconstruction_variance_sum = np.zeros(embeddings.shape[1], dtype=np.float64)
    reconstruction_sum = np.zeros_like(reconstruction_variance_sum)
    for batch in batch_indices(indices, config.eval_batch_size):
        if config.relation_mode == "local":
            x, rec, z, cx, cz = local_forward(model, embeddings, batch, candidates[rows:rows + len(batch)], device)
            mean, std = (torch.from_numpy(value).to(device) for value in statistics)
            local += float(local_relation_kl(x, z, cx, cz, mean, std, config.relation_temperature)) * len(batch)
        else:
            x = torch.from_numpy(embeddings[batch]).to(device)
            rec, z = model(x)
        mse += float(F.mse_loss(rec, x)) * len(batch)
        cosine += float(F.cosine_similarity(rec, x, dim=1).sum())
        if len(batch) > 1 and config.relation_mode != "none":
            relation += float(relation_kl(x, z, config.relation_temperature)) * len(batch)
            relation_rows += len(batch)
        a = rec.cpu().numpy().astype(np.float64)
        reconstruction_sum += a.sum(0)
        reconstruction_variance_sum += (a ** 2).sum(0)
        rows += len(batch)
    mse /= rows
    cosine /= rows
    relation /= max(relation_rows, 1)
    nmse = mse / float(model.input_scale.square())
    variance = np.maximum(reconstruction_variance_sum / rows - (reconstruction_sum / rows) ** 2, 0).mean()
    input_variance = embeddings[indices].var(axis=0, dtype=np.float64).mean()
    relation_score = .1 * relation if config.relation_mode == "batch" else 0.
    result = {"mse": mse, "normalized_mse": nmse, "cosine_similarity": cosine,
              "relation_kl": relation, "score": nmse + .1 * (1 - cosine) + relation_score,
              "decoder_output_variance_ratio": float(variance / max(input_variance, 1e-12))}
    if config.relation_mode == "local":
        result["local_relation_kl"] = local / rows
        result["score"] = nmse + .1 * (1 - cosine) + .1 * result["local_relation_kl"]
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError("Non-finite geometry AE validation metrics.")
    return result


def fit_pca(embeddings, train, config):
    count = min(config.latent_dim, len(train), embeddings.shape[1])
    return PCA(n_components=count, svd_solver="randomized", random_state=config.seed).fit(embeddings[train])


def pca_features(pca, embeddings, train):
    raw = pca.transform(embeddings)
    return (raw - raw[train].mean(0)) / np.maximum(raw[train].std(0), 1e-6)


def load_best(directory, source, device="cpu"):
    cp = torch.load(Path(directory) / "best.pt", map_location="cpu", weights_only=True)
    if cp.get("method") != METHOD or cp.get("format_version") != FORMAT_VERSION or cp.get("source") != source:
        raise ValueError("Geometry AE checkpoint format or source mismatch.")
    config = AEConfig(**cp["ae_config"])
    if cp.get("experiment_key") != experiment_key(source, config.to_dict()):
        raise ValueError("Geometry AE checkpoint identity mismatch.")
    model = GeometryAE(source["input_shape"][1], config)
    model.load_state_dict(cp["model"])
    return model.to(device).eval(), cp, config


@torch.no_grad()
def export_artifact(directory, embeddings, dataset, data_config, train_mask, *, device="cpu"):
    directory = Path(directory)
    source = source_metadata(embeddings, dataset, data_config, train_mask)
    model, cp, config = load_best(directory, source, device)
    splits = {key: value.numpy() for key, value in cp["splits"].items()}
    expected = split_items(train_mask, config)
    if set(splits) != set(expected) or any(not np.array_equal(splits[k], expected[k]) for k in expected):
        raise ValueError("Geometry AE item splits mismatch.")
    directory.joinpath("manifest.json").unlink(missing_ok=True)
    raw = encode_all(model, embeddings, config.eval_batch_size, device)
    features, mean, std = standardize_latents(raw, train_mask)
    local_options = {}
    if config.relation_mode == "local":
        sampler = LocalCandidates(embeddings, splits["train"], np.concatenate(list(splits.values())), config)
        statistics = latent_statistics(raw[splits["train"]])
        local_options = {key: {"candidates": sampler.sample(indices, np.random.default_rng(config.seed)),
                               "statistics": statistics} for key, indices in splits.items()}
    metrics = {key: evaluate(model, embeddings, indices, config, device, **local_options.get(key, {}))
               for key, indices in splits.items()}
    pca = fit_pca(embeddings, splits["train"], config)
    pc = pca_features(pca, embeddings, splits["train"])
    metrics["geometry"] = {"raw": geometry_metrics(raw[train_mask]),
                           "standardized": geometry_metrics(features[train_mask])}
    metrics["neighbors"] = {key: neighbor_comparison(embeddings[indices], features[indices], pc[indices], config)
                            for key, indices in splits.items()}
    metrics["best_epoch"] = cp["epoch"]
    metrics["split_sizes"] = {key: len(indices) for key, indices in splits.items()}
    for name, values in (("raw_latents.npy", raw), ("latents.npy", features)):
        atomic_write(directory / name, lambda stream, values=values: np.save(stream, values, allow_pickle=False))
    atomic_write(directory / "statistics.npz", lambda stream: np.savez(
        stream, mean=mean, std=std, input_mean=model.input_mean.cpu().numpy(),
        input_scale=model.input_scale.cpu().numpy()))
    atomic_write(directory / "splits.npz", lambda stream: np.savez(stream, **splits))
    if not (directory / "loss.csv").exists():
        write_history(directory / "loss.csv", cp["history"])
    if not (directory / "diagnostics.json").exists():
        write_json(directory / "diagnostics.json", cp["diagnostics"])
    write_json(directory / "metrics.json", metrics)
    hashes = {name: file_hash(directory / name) for name in ARTIFACT_FILES}
    write_json(directory / "manifest.json", {
        "method": METHOD, "format_version": FORMAT_VERSION, "source": source,
        "experiment_key": cp["experiment_key"], "ae_config": config.to_dict(), "files": hashes,
        "artifact_id": json_hash({"experiment_key": cp["experiment_key"], "files": hashes}),
        "feature_shape": list(features.shape), "row_order": "row i is non-padding item ID i + 1",
        "input_statistics_population": "compression_training_items",
        "standardization_population": "recommendation_training_items",
    })
    return directory


def train_artifact(embeddings, dataset, data_config, train_mask, config=None, *, device="cpu", force=False):
    config = config or AEConfig()
    settings = config.to_dict()
    source = source_metadata(embeddings, dataset, data_config, train_mask)
    directory = artifact_directory(dataset.cache_dir, source, settings)
    if (directory / "manifest.json").exists() and not force:
        read_manifest(directory)
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "loss.csv", "diagnostics.json"):
        directory.joinpath(name).unlink(missing_ok=True)
    splits = split_items(train_mask, config)
    mean, scale = fit_input_statistics(embeddings, splits["train"])
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    model = GeometryAE(embeddings.shape[1], config, mean, scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    pc = pca_features(fit_pca(embeddings, splits["train"], config), embeddings, splits["train"])
    sampler, valid_candidates = None, None
    if config.relation_mode == "local":
        logger.info("Building Qwen neighbor pools from %d compression training items", len(splits["train"]))
        sampler = LocalCandidates(embeddings, splits["train"],
                                  np.concatenate((splits["train"], splits["valid"])), config)
        valid_candidates = sampler.sample(splits["valid"], np.random.default_rng(config.seed))
    candidate_rng = np.random.default_rng(config.seed)
    best, stale, low_rank = float("inf"), 0, 0
    history, diagnostics = [], []
    for epoch in range(1, config.epochs + 1):
        weights = objective_weights(config, epoch)
        statistics = None
        if sampler is not None and weights["relation"] > 0:
            statistics = latent_statistics(encode_all(model, embeddings[splits["train"]],
                                                      config.eval_batch_size, device))
        model.train()
        totals = {key: 0. for key in weights}
        loss_total, rows = 0., 0
        for number, indices in enumerate(batch_indices(splits["train"], config.train_batch_size,
                                                      generator=generator, training=True)):
            optimizer.zero_grad(set_to_none=True)
            if statistics is not None:
                candidates = sampler.sample(indices, candidate_rng)
                x, rec, z, cx, cz = local_forward(model, embeddings, indices, candidates, device)
            else:
                x = torch.from_numpy(embeddings[indices]).to(device)
                rec, z = model(x)
            terms = loss_terms(x, rec, z, model.input_scale, config.relation_temperature,
                               batch_relation=config.relation_mode == "batch")
            if statistics is not None:
                latent_mean, latent_std = (torch.from_numpy(value).to(device) for value in statistics)
                terms["relation"] = local_relation_kl(x, z, cx, cz, latent_mean, latent_std,
                                                      config.relation_temperature)
            loss = sum(terms[key] * weights[key] for key in terms)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite geometry AE training loss.")
            # One reproducible diagnostic batch per epoch avoids five extra backwards per batch.
            if number == 0:
                grad_norms = encoder_gradient_norms(terms, weights, model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            rows += len(indices)
            loss_total += float(loss.detach()) * len(indices)
            for key, term in terms.items():
                totals[key] += float(term.detach()) * len(indices)
        raw = encode_all(model, embeddings[splits["train"]], config.eval_batch_size, device)
        options = ({"candidates": valid_candidates, "statistics": latent_statistics(raw)}
                   if sampler is not None else {})
        valid = evaluate(model, embeddings, splits["valid"], config, device, **options)
        standard = (raw - raw.mean(0)) / np.maximum(raw.std(0), 1e-6)
        geometry = {"raw": geometry_metrics(raw), "standardized": geometry_metrics(standard)}
        rank = geometry["standardized"]["effective_rank"]
        low_rank = low_rank + 1 if rank < config.collapse_rank else 0
        if low_rank == config.collapse_patience:
            logger.warning("Geometry AE low effective rank %.2f for %d epochs; weights unchanged.", rank, low_rank)
        diagnostic = {"epoch": epoch, "geometry": geometry, "encoder_gradient_norms": grad_norms,
                      "weights": weights, "consecutive_low_rank_epochs": low_rank}
        if sampler is not None:
            diagnostic["local_relation"] = {"statistics_population": "compression_training_items",
                                            "statistics_refresh": "epoch_start_detached",
                                            "candidate_count": valid_candidates.shape[1]}
        if epoch % config.neighbor_interval == 0:
            valid_raw = encode_all(model, embeddings[splits["valid"]], config.eval_batch_size, device)
            valid_features = (valid_raw - raw.mean(0)) / np.maximum(raw.std(0), 1e-6)
            diagnostic["neighbors"] = neighbor_comparison(
                embeddings[splits["valid"]], valid_features, pc[splits["valid"]], config)
        diagnostics.append(diagnostic)
        row = {"epoch": epoch, "train_loss": loss_total / rows, "effective_rank": rank,
               **{f"train_{k}": v / rows for k, v in totals.items()},
               **{f"valid_{k}": v for k, v in valid.items()},
               **{f"encoder_grad_{k}": v for k, v in grad_norms.items()}}
        history.append(row)
        if valid["score"] < best:
            best, stale = valid["score"], 0
            atomic_write(directory / "best.pt", lambda stream: torch.save({
                "method": METHOD, "format_version": FORMAT_VERSION, "source": source,
                "experiment_key": experiment_key(source, settings), "ae_config": settings,
                "epoch": epoch, "valid_score": best, "model": model.state_dict(),
                "splits": {k: torch.from_numpy(v.copy()) for k, v in splits.items()},
                "history": history, "diagnostics": diagnostics,
            }, stream))
        else:
            stale += 1
        write_history(directory / "loss.csv", history)
        write_json(directory / "diagnostics.json", diagnostics)
        if epoch == 1 or epoch % 5 == 0 or epoch == config.epochs:
            logger.info("AE epoch %d/%d train %.5g valid score %.5g NMSE %.5g rank %.2f gradients %s",
                        epoch, config.epochs, row["train_loss"], valid["score"], valid["normalized_mse"],
                        rank, grad_norms)
        if stale >= config.patience:
            logger.info("Geometry AE early stopping at epoch %d", epoch)
            break
    del optimizer, model
    return export_artifact(directory, embeddings, dataset, data_config, train_mask, device=device)
