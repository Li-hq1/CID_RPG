"""Standalone figures and compact measurements from the fixed-reference audit."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from misc.artifacts import write_json
from protocol import ROOT


BASE = ROOT / "outputs/item_cluster_optimization"
LABELS = {
    "baseline": "No-VLAD baseline", "k1_g2": "K1/G2 reference", "k1_g1": "K1/G1",
    "k1_g4": "K1/G4", "k2_g2": "K2/G2 sum", "k4_g2": "K4/G2 sum",
    "k4_g2_mass": "K4/G2 mass", "k4_g2_context": "K4/G2 context",
    "k1_g2_scale_half": "K1/G2 scale x0.5", "k1_g2_scale_double": "K1/G2 scale x2",
}
COLORS = dict(zip(LABELS, ["#777777", "#007e73", "#a46f38", "#d4a128", "#7984c7",
                          "#5176bf", "#d95d44", "#8147ab", "#b45b87", "#2880a8"]))


def main():
    summary = json.loads((BASE / "summary.json").read_text())
    records = {**summary["references"], **summary["runs"]}
    records = {name: records[name] for name in LABELS if name in records}
    reference = records["k1_g2"]
    baseline = records["baseline"]
    for filename in ["report_data.json", "curves.png", "curves.pdf"]:
        if (BASE / filename).exists(): raise FileExistsError(BASE / filename)
    rows = []
    for name, r in records.items():
        rows.append({"method": name, "label": LABELS[name], "best_epoch": r["best"]["epoch"],
                     "selection_ndcg10": r["validation_selection"]["ndcg@10"],
                     "full_ndcg10": r["validation"]["ndcg@10"],
                     "full_recall10": r["validation"]["recall@10"], "full_nll": r["validation"]["nll"],
                     "full_gain_percent_vs_positive_reference": 100 * (r["validation"]["ndcg@10"] / reference["validation"]["ndcg@10"] - 1),
                     "full_gain_percent_vs_baseline": 100 * (r["validation"]["ndcg@10"] / baseline["validation"]["ndcg@10"] - 1),
                     "parameters": r["experiment"]["trainable_parameters"],
                     "peak_memory_allocated_mib": r["peak_memory_allocated_mib"],
                     "seconds_nonexclusive_gpu": r["seconds"],
                     "probe": r.get("probe", {}).get("trained"),
                     "initial_probe": r.get("probe", {}).get("initial"),
                     "train_at_best": r["history"][r["best"]["epoch"] - 1]["train"],
                     "train_at_end": r["history"][-1]["train"]})
    write_json(BASE / "report_data.json", {"rows": rows, "seed": 2024, "test_accessed": False,
        "note": "Single-seed exploration; fixed 4096-user selection chooses epochs; full validation audits only."})
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(19, 10), constrained_layout=True)

    def curves(ax, names, title):
        for name in names:
            if name not in records: continue
            points = [(r["epoch"], r["validation_selection"]["ndcg@10"])
                      for r in records[name]["history"] if "validation_selection" in r]
            ax.plot(*zip(*points), label=LABELS[name], color=COLORS[name], linewidth=2,
                    linestyle="--" if name == "baseline" else "-", marker="o", markersize=3)
        ax.set(title=title, xlabel="Epoch", ylabel="Selection NDCG@10")
        ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=8)

    curves(axes[0, 0], ["baseline", "k1_g1", "k1_g2", "k1_g4"], "A. Group/expansion capacity at K=1")
    curves(axes[0, 1], ["baseline", "k1_g2", "k2_g2", "k4_g2"], "B. Number of cluster tokens at G=2")
    curves(axes[0, 2], ["baseline", "k1_g2", "k1_g2_scale_half", "k1_g2_scale_double"], "C. Input scale at positive K1/G2 reference")
    curves(axes[1, 0], ["k1_g2", "k4_g2", "k4_g2_mass", "k4_g2_context"], "D. Pooling contextual cluster states")
    ax = axes[1, 1]
    names = list(records)
    gains = [100 * (records[n]["validation"]["ndcg@10"] / reference["validation"]["ndcg@10"] - 1) for n in names]
    bars = ax.barh(np.arange(len(names)), gains, color=[COLORS[n] for n in names])
    ax.set_yticks(np.arange(len(names)), [LABELS[n] for n in names]); ax.invert_yaxis()
    ax.axvline(0, color="#444", linewidth=.8); ax.grid(axis="x", alpha=.2)
    for bar, gain in zip(bars, gains):
        ax.annotate(f"{gain:+.2f}%", (gain, bar.get_y()+bar.get_height()/2),
                    xytext=(4 if gain >= 0 else -4, 0), textcoords="offset points",
                    ha="left" if gain >= 0 else "right", va="center", fontsize=9)
    ax.set_xlim(min(gains)-1.7, max(gains)+1.7)
    ax.set(title="E. Full-validation audit vs positive reference", xlabel="NDCG@10 change (%)")
    ax = axes[1, 2]
    for name in ["k1_g2", "k4_g2_mass", "k4_g2_context", "k1_g2_scale_half", "k1_g2_scale_double"]:
        if name not in records: continue
        points = [(r["epoch"], r["train"]["loss/ce"]) for r in records[name]["history"]]
        ax.plot(*zip(*points), label=LABELS[name], color=COLORS[name], linewidth=2)
    ax.set(title="F. Training CE (same objective and updates)", xlabel="Epoch", ylabel="Training CE")
    ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Beauty / seed 2024 | 40 epochs / 4040 updates | fixed selection users | no test", fontsize=13)
    fig.savefig(BASE / "curves.png", dpi=170)
    fig.savefig(BASE / "curves.pdf")
    print(json.dumps([{k: r[k] for k in ["method", "selection_ndcg10", "full_ndcg10",
                                        "full_gain_percent_vs_positive_reference"]} for r in rows]), flush=True)


if __name__ == "__main__":
    main()
