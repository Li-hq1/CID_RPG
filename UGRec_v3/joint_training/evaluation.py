"""Evaluation against a freshly encoded catalog in the current latent space."""

from dataclasses import asdict

import torch

from compression.geometry_ae.metrics import geometry_metrics
from engine.metrics import Evaluator


@torch.no_grad()
def evaluate(model, loader, config, tokenizer):
    model.eval()
    model.refresh_catalog()
    evaluator = Evaluator(asdict(config.evaluation), tokenizer)
    totals, rows = {}, 0
    for batch in loader:
        batch = {key: value.to(model.device) for key, value in batch.items()}
        predictions = model.generate(batch, n_return_sequences=evaluator.maxk)
        values = evaluator.calculate_metrics(predictions, batch["labels"])
        if model.last_generation_stats is not None:
            values.update({f"sampling_{key}": value for key, value in model.last_generation_stats.items()})
        for key, value in values.items():
            totals[key] = totals.get(key, 0.) + float(value.double().sum())
        rows += len(batch["labels"])
    if not rows:
        raise ValueError("Cannot evaluate an empty dataset.")
    return {key: value / rows for key, value in totals.items()}


@torch.no_grad()
def latent_diagnostics(model, train_ids, reference):
    model.ae.eval()
    z = model.encode_ids(train_ids.to(model.device), grad=False)
    result = geometry_metrics(z.cpu().numpy())
    result.pop("per_dim_std", None)
    result["drift_rms"] = float((z - reference.to(z.device)).square().mean().sqrt())
    return result
