"""Build standalone scientific figures and compact measurements from the final audit."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from misc.artifacts import write_json
from protocol import ROOT


BASE = ROOT / "outputs/item_cluster_followup"
LABELS = {
    "baseline": "Single-vector baseline", "unit_k1": "External K1, unit",
    "unit_k4": "External K4, unit", "scaled_k1": "External K1, scaled",
    "scaled_k4": "External K4, scaled", "ae_sum_multi": "AE sum, multi",
    "ae_sum_merged": "AE sum, merged", "ae_linear_multi": "AE linear, multi",
    "ae_linear_merged": "AE linear, merged",
}
COLORS = dict(zip(LABELS, ["#555555", "#9c9c9c", "#a898b9", "#007f73", "#5176bf",
                          "#cd5936", "#cc9d27", "#854bb4", "#a95686"]))


def main():
    summary = json.loads((BASE / "summary.json").read_text())
    records = {}
    for name, suffix in [("baseline", "baseline"), ("unit_k1", "k1"), ("unit_k4", "k4")]:
        directory = ROOT / "outputs/item_clusters/runs/Beauty" / f"beauty_item_vlad_{suffix}_s2024"
        pdir = BASE / "reference_probes" / suffix
        probe = json.loads((pdir / "followup_probe.json").read_text())
        assert probe["initial_state_reproduced"] and probe["reload_predictions_verified"] and not probe["test_accessed"]
        assert hashlib.sha256((pdir / "followup_probe_source.py").read_bytes()).hexdigest() == probe["probe_script_sha256"]
        assert hashlib.sha256((directory / "best_flow.pt").read_bytes()).hexdigest() == probe["checkpoint_sha256"]
        records[name] = {**json.loads((directory / "results.json").read_text()), "probe": probe,
                         "history": json.loads((directory / "metrics.json").read_text()),
                         "experiment": json.loads((directory / "experiment.json").read_text())}
    records.update(summary["runs"])
    baseline = records["baseline"]
    rows = []
    for name, r in records.items():
        rows.append({"method": name, "label": LABELS[name], "best_epoch": r["best"]["epoch"],
                     "selection_ndcg10": r["validation_selection"]["ndcg@10"],
                     "full_ndcg10": r["validation"]["ndcg@10"],
                     "full_recall10": r["validation"]["recall@10"], "full_nll": r["validation"]["nll"],
                     "full_ndcg10_gain_percent": 100 * (r["validation"]["ndcg@10"] / baseline["validation"]["ndcg@10"] - 1),
                     "parameters": r["experiment"]["trainable_parameters"],
                     "peak_memory_allocated_mib": r["peak_memory_allocated_mib"],
                     "training_seconds_nonexclusive_gpu": r["seconds"],
                     "probe": r["probe"]["trained"],
                     "train_at_best": r["history"][r["best"]["epoch"] - 1]["train"],
                     "train_at_end": r["history"][-1]["train"]})
    for name in ["report_data.json", "curves.png", "curves.pdf"]:
        if (BASE / name).exists(): raise FileExistsError(BASE / name)
    write_json(BASE / "report_data.json", {"rows": rows, "test_accessed": False, "seed": 2024,
        "note": "Single-seed exploratory evidence; full validation audits the selection-chosen checkpoint."})
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    def curves(ax, names, title, key="selection"):
        for name in names:
            if name not in records: continue
            history = records[name]["history"]
            points = [(r["epoch"], r["validation_selection"]["ndcg@10"]) for r in history if "validation_selection" in r]
            if key == "ce": points = [(r["epoch"], r["train"]["loss/ce"]) for r in history]
            ax.plot(*zip(*points), label=LABELS[name], color=COLORS[name],
                    linewidth=2, linestyle="--" if name.startswith("unit_") else "-",
                    marker="o" if key == "selection" else None, markersize=3)
        ax.set(title=title, xlabel="Epoch", ylabel="Selection NDCG@10" if key == "selection" else "Training CE")
        ax.grid(alpha=.2); ax.legend(fontsize=8, frameon=False)

    curves(axes[0, 0], ["baseline", "unit_k1", "unit_k4", "scaled_k1", "scaled_k4"],
           "A. Match HSTU token scale")
    ae_names = ["baseline", "scaled_k1", "ae_sum_multi", "ae_sum_merged", "ae_linear_multi", "ae_linear_merged"]
    curves(axes[0, 1], ae_names, "B. AE/catalog aggregation and history merge position")
    names = list(records)
    gains = [100 * (records[n]["validation"]["ndcg@10"] / baseline["validation"]["ndcg@10"] - 1) for n in names]
    ax = axes[1, 0]
    bars = ax.barh(np.arange(len(names)), gains, color=[COLORS[n] for n in names])
    ax.set_yticks(np.arange(len(names)), [LABELS[n] for n in names]); ax.invert_yaxis()
    ax.axvline(0, color="#444", linewidth=.8); ax.grid(axis="x", alpha=.2)
    for bar, gain in zip(bars, gains):
        ax.annotate(f"{gain:+.2f}%", (gain, bar.get_y() + bar.get_height()/2),
                    xytext=(4 if gain >= 0 else -4, 0), textcoords="offset points",
                    ha="left" if gain >= 0 else "right", va="center", fontsize=9)
    ax.set_xlim(min(gains) - 1.6, max(gains) + 1.6)
    ax.set(title="C. Full-validation audit of selected checkpoints", xlabel="NDCG@10 change vs baseline (%)")
    curves(axes[1, 1], ae_names, "D. Training fit (lower CE need not improve ranking)", key="ce")
    fig.suptitle("Beauty, seed 2024 | 40 epochs / 4040 updates | fixed 4096-user selection | no test", fontsize=13)
    fig.savefig(BASE / "curves.png", dpi=170)
    fig.savefig(BASE / "curves.pdf")
    print(json.dumps([{k: r[k] for k in ["method", "selection_ndcg10", "full_ndcg10", "full_ndcg10_gain_percent", "parameters"]} for r in rows]), flush=True)


if __name__ == "__main__":
    main()
