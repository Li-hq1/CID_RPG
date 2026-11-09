"""Standalone plots for the fixed-protocol VLAD simplification experiments."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from misc.artifacts import write_json
from protocol import ROOT


BASE = ROOT / "outputs/item_cluster_simplification"
LABELS = {"baseline": "Original input", "k1_g2": "NeXtVLAD K1 reference",
    "k4_g2": "NeXtVLAD K4", "norm_k1": "Normalize only K1",
    "residual_k1": "Center residual K1", "residual_k1_linear": "Residual + map K1",
    "next_k1_linear": "NeXtVLAD + map K1", "residual_k4": "Center residual K4",
    "residual_k4_linear": "Residual + map K4"}


def main():
    summary = json.loads((BASE / "summary.json").read_text())
    records = {**summary["references"], **summary["runs"]}
    records = {n: records[n] for n in LABELS if n in records}
    for f in ["report_data.json", "curves.png", "curves.pdf"]:
        if (BASE / f).exists(): raise FileExistsError(BASE / f)
    baseline, ref = records["baseline"], records["k1_g2"]
    rows = []
    for name, r in records.items():
        rows.append({"method": name, "label": LABELS[name], "best_epoch": r["best"]["epoch"],
            "selection_ndcg10": r["validation_selection"]["ndcg@10"],
            "full_ndcg10": r["validation"]["ndcg@10"], "full_recall10": r["validation"]["recall@10"],
            "full_nll": r["validation"]["nll"], "parameters": r["experiment"]["trainable_parameters"],
            "outside_selection_ndcg10": r["validation_outside_selection"]["ndcg@10"],
            "added_parameters": r["experiment"]["trainable_parameters"] - baseline["experiment"]["trainable_parameters"],
            "full_gain_percent_vs_positive_reference": 100*(r["validation"]["ndcg@10"]/ref["validation"]["ndcg@10"]-1),
            "full_gain_percent_vs_baseline": 100*(r["validation"]["ndcg@10"]/baseline["validation"]["ndcg@10"]-1),
            "peak_memory_allocated_mib": r["peak_memory_allocated_mib"], "seconds_nonexclusive_gpu": r["seconds"],
            "probe": r.get("probe", {}).get("trained"), "initial_probe": r.get("probe", {}).get("initial"),
            "train_at_best": r["history"][r["best"]["epoch"]-1]["train"],
            "train_at_end": r["history"][-1]["train"]})
    write_json(BASE / "report_data.json", {"rows": rows, "seed": 2024, "test_accessed": False,
        "note": "Single seed; fixed 4096-user selection; full validation audits the selected epoch only."})
    colors = dict(zip(records, plt.get_cmap("tab10").colors))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    def curves(ax, names, title):
        for name in names:
            if name not in records: continue
            pts = [(h["epoch"], h["validation_selection"]["ndcg@10"])
                   for h in records[name]["history"] if "validation_selection" in h]
            ax.plot(*zip(*pts), label=LABELS[name], color=colors[name], marker="o", markersize=3)
        ax.set(title=title, xlabel="Epoch", ylabel="Selection NDCG@10")
        ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=8)
    curves(axes[0, 0], ["baseline", "k1_g2", "norm_k1", "residual_k1", "residual_k1_linear", "next_k1_linear"],
           "A. Normalization, centers and shared mapping")
    curves(axes[0, 1], ["k1_g2", "k4_g2", "residual_k1", "residual_k1_linear", "residual_k4", "residual_k4_linear"],
           "B. Do multiple center directions help?")
    ax = axes[1, 0]
    for row in rows:
        ax.scatter(row["added_parameters"], row["full_ndcg10"], c=[colors[row["method"]]], s=65)
        ax.annotate(row["label"], (row["added_parameters"], row["full_ndcg10"]), xytext=(4, 4),
                    textcoords="offset points", fontsize=8)
    ax.axhline(ref["validation"]["ndcg@10"], color="#888", linestyle="--", linewidth=1)
    ax.set(title="C. Parameter cost and selected-checkpoint audit", xlabel="Parameters added to original model",
           ylabel="Full-validation NDCG@10")
    ax.grid(alpha=.2); ax.margins(x=.3, y=.2)
    ax = axes[1, 1]
    names = [n for n in records if n not in ["baseline", "k4_g2"]]
    for name in names:
        pts = [(h["epoch"], h["train"]["loss/ce"]) for h in records[name]["history"]]
        ax.plot(*zip(*pts), color=colors[name], label=LABELS[name])
    ax.set(title="D. Same training objective and update budget", xlabel="Epoch", ylabel="Training CE")
    ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Beauty / seed 2024 | 40 epochs, 4040 updates | scale matched | no test", fontsize=13)
    fig.savefig(BASE / "curves.png", dpi=170); fig.savefig(BASE / "curves.pdf")
    print(json.dumps(rows, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
