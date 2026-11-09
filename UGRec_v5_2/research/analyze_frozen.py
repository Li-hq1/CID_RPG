"""Read-only frozen-CFM diagnostics, including exact CE latent redecoding.

If latent samples followed the CE item's categorical distribution exactly,
then the infinite-sample softmax readout would equal P_CE @ K.  This measures
that fixed readout operation; it is not an oracle for the unknown true target
distribution, a new trained model, or a proposed prediction ensemble.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from joint_training.config import JointConfig
from joint_training.fm import build_model
from misc.artifacts import array_hash, json_hash, write_json
from protocol import ROOT, corpus, ranking_metrics, raw_features, validation_batches
from research.analyze_initial import distribution_diagnostics, paired
from research.frozen_ce_flow import frozen_state, state_hash


@torch.no_grad()
def redecoding(model, histories, targets, ce_predictions):
    keys = F.normalize(model.catalog[1:], dim=-1)
    n = len(keys)
    kernel = torch.empty((n, n), device=model.device)
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        for start in range(0, n, 256):
            kernel[start:start + 256] = (keys[start:start + 256] @ keys.T / model.fm_config.temperature).softmax(-1)
    finally:
        torch.set_float32_matmul_precision(precision)
    diagonal = kernel.diag()
    stats = {"mean_self_probability_catalog": float(diagonal.mean()),
             "median_self_probability_catalog": float(diagonal.median()),
             "min_self_probability_catalog": float(diagonal.min()),
             "max_self_probability_catalog": float(diagonal.max()),
             "self_probability_quantiles": {str(q): float(diagonal.quantile(q)) for q in (.1, .5, .9)},
             "catalog_self_top1_fraction": float((kernel.argmax(-1) == torch.arange(n, device=model.device)).float().mean()),
             "temperature": model.fm_config.temperature,
             "validation_target_weighted_self_probability": float(diagonal[torch.tensor(targets - 1, device=model.device)].mean())}
    predictions, nll, ce_check = [], [], []
    for _, batch in validation_batches(histories, targets, 32, np.arange(len(targets))):
        batch = {key: value.to(model.device) for key, value in batch.items()}
        logp = model.mixture_log_probabilities(batch, source="ranking")
        # Keep CE evaluation numerically identical to the parent run; perform
        # only the diagnostic stochastic-matrix product at full FP32 precision.
        try:
            torch.set_float32_matmul_precision("highest")
            smooth = logp.exp() @ kernel
        finally:
            torch.set_float32_matmul_precision(precision)
        torch.testing.assert_close(smooth.sum(-1), torch.ones(len(smooth), device=model.device), atol=2e-5, rtol=2e-5)
        predictions.append((smooth.topk(10, -1).indices + 1).cpu().numpy())
        ce_check.append((logp.topk(10, -1).indices + 1).cpu().numpy())
        nll.append(-smooth.gather(1, batch["labels"] - 1).clamp_min(1e-30).log()[:, 0].cpu())
    np.testing.assert_array_equal(np.concatenate(ce_check), ce_predictions)
    predictions = np.concatenate(predictions)
    stats["redecoded_ce_metrics"] = ranking_metrics(predictions, targets)
    stats["redecoded_ce_metrics"]["nll"] = float(torch.cat(nll).mean())
    stats["redecoded_ce_vs_direct_ce"] = paired(predictions, ce_predictions, targets)
    stats["meaning"] = "Exact P_CE @ K, infinite-draw thought experiment; not a test of learned FM fit or unknown true conditional distribution"
    return stats, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="outputs/frozen_analysis_20261106")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(.2, int(args.device.split(":")[-1]))
    run = ROOT / "outputs/Beauty/beauty_frozen_ce_flow_s2024_20261106"
    experiment = json.loads((run / "experiment.json").read_text())
    for relative, digest in json.loads((run / "source_sha256.json").read_text()).items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
    checkpoint = torch.load(run / "best_flow.pt", map_location="cpu", weights_only=True)
    cfg = JointConfig.from_dict(checkpoint["parent_config"])
    cfg.training.device = args.device
    directory, mapping, train, histories, targets = corpus("Beauty")
    raw = raw_features(directory, len(mapping["id2item"]) - 1)
    assert checkpoint["source"]["raw_sha256"] == array_hash(raw)
    assert checkpoint["source"]["train_sha256"] == json_hash(train)
    model = build_model(cfg, raw).to(args.device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    assert state_hash(frozen_state(model)) == experiment["frozen_state_sha256"]
    model.refresh_catalog()
    parent = Path(experiment["parent"])
    with np.load(parent / "full_flow.npz") as ce, np.load(run / "full_flow.npz") as fm:
        np.testing.assert_array_equal(targets, ce["targets"])
        np.testing.assert_array_equal(targets, fm["targets"])
        ce_predictions, fm_predictions = ce["predictions"], fm["predictions"]
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    kernel, redecoded = redecoding(model, histories, targets, ce_predictions)
    np.savez_compressed(output / "redecoded_ce.npz", predictions=redecoded, targets=targets)
    summary = {"run": str(run), "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "frozen_result": json.loads((run / "results.json").read_text()),
               "paired_fm_vs_ce": paired(fm_predictions, ce_predictions, targets),
               "distribution": distribution_diagnostics(model, histories, targets),
               "decoder_kernel": kernel, "test_accessed": False}
    write_json(output / "summary.json", summary)
    # Scientific figure: deterministic results, no new model selection.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    initial = json.loads((ROOT / "outputs/initial_analysis_20261106/summary.json").read_text())
    names = ["CE only", "Joint: CE", "Joint: FM", "Frozen CE: FM"]
    values = [initial["ce_result"]["validation"]["ndcg@10"],
              initial["fm_result"]["ranking_selected_validation"]["ndcg@10"],
              initial["fm_result"]["validation"]["ndcg@10"],
              summary["frozen_result"]["validation"]["ndcg@10"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bars = axes[0].barh(names, values, color=["#236799", "#8aa9bf", "#bf7c45", "#bf7c45"])
    axes[0].set(xlim=(0, .071), xlabel="Full validation NDCG@10 (Beauty, seed 2024)")
    axes[0].invert_yaxis()
    for bar, value in zip(bars, values):
        axes[0].text(value + .001, bar.get_y() + bar.get_height()/2, f"{value:.5f}", va="center", fontsize=9)
    rows = json.loads((run / "metrics.json").read_text())
    selected = [r for r in rows if "validation_selection" in r]
    axes[1].plot([r["epoch"] for r in selected], [r["validation_selection"]["ndcg@10"] for r in selected],
                 marker="o", color="#bf7c45", label="Frozen CE + FM readout")
    ce_selection = json.loads((parent / "results.json").read_text())["best"]["score"]
    axes[1].axhline(ce_selection, linestyle="--", color="#236799", label="Frozen CE readout")
    axes[1].set(xlabel="Additional FM stage epoch", ylabel="Selection validation NDCG@10", ylim=(.04, .065))
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(axis="x" if axis is axes[0] else "both", alpha=.2)
        axis.set_axisbelow(True)
    fig.suptitle("Fixed seed exploration; frozen stage adds training budget", fontsize=10)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=170)
    plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
