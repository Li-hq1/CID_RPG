"""Scientific figures from saved logs, without rerunning training or evaluation."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from protocol import ROOT


def main():
    base = ROOT/"outputs/flowrec/runs/Beauty"
    labels = {"control": "Original mixture control", "prior": "Prior CE only",
              "full": "Prior + 5 CFM + Align", "no_cfm": "Prior + Align",
              "low_cfm": "Prior + 1 CFM + Align"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for name, label in labels.items():
        path = base/f"flowrec_{name}_s2024/metrics.json"
        if not path.exists():
            continue
        log = json.loads(path.read_text())
        selected = [r for r in log if "validation_selection" in r]
        axes[0].plot([r["epoch"] for r in selected],
                     [r["validation_selection"]["ndcg@10"] for r in selected],
                     marker="o", markersize=3, label=label)
        if name in {"full", "low_cfm"}:
            for key, linestyle in [("cfm", "-"), ("align", "--")]:
                axes[1].plot([r["epoch"] for r in log], [r["train"][f"loss/{key}"] for r in log],
                             linestyle=linestyle, label=f"{name}: {key.upper()} (unweighted)")
    axes[0].set(xlabel="Epoch", ylabel="NDCG@10", title="Fixed 4096-user validation selection")
    axes[1].set(xlabel="Epoch", ylabel="Training loss (log scale)", yscale="log",
                title="Bridge fitting does not measure deployment ranking")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    for suffix in ["png", "pdf"]:
        fig.savefig(ROOT/f"docs/flowrec_selection_curves.{suffix}", dpi=180)


if __name__ == "__main__":
    main()
