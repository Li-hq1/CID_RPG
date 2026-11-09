"""Collect completed validation experiments, paired comparisons and export a static figure."""

from collections import defaultdict
import json

import numpy as np

from .common import ROOT, paired_interval, per_user, write_json


DIRECTORY = ROOT / "outputs/research_20261028"


def main():
    groups = defaultdict(list)
    runs = []
    for path in sorted((DIRECTORY / "compact").glob("*/results.json")):
        run = json.loads(path.read_text())
        run["run_name"] = path.parent.name
        runs.append(run)
        c = run["config"]
        groups[(c["category"], c["representation"], c["encoder"], c["objective"])].append(run)
    group_results = []
    for key, values in groups.items():
        result = dict(zip(("category", "representation", "encoder", "objective"), key))
        result["seeds"] = [r["config"]["seed"] for r in values]
        result["best_epochs"] = [r["best_epoch"] for r in values]
        for metric in ("recall@10", "ndcg@10"):
            scores = [r["full_validation"][metric] for r in values]
            result[metric] = {"mean": float(np.mean(scores)),
                              "sample_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else None,
                              "values": scores}
        group_results.append(result)
    comparisons = {}
    for category in ("Beauty", "Sports_and_Outdoors"):
        baseline = np.load(DIRECTORY / "nonparametric" / f"{category}_markov_last_predictions.npz")
        markov = per_user(baseline["predictions"], baseline["targets"])["ndcg@10"]
        id_runs = [r for r in runs if r["config"]["category"] == category
                   and r["config"]["representation"] == "id" and r["config"]["encoder"] == "transformer"]
        user_metrics = []
        for r in id_runs:
            data = np.load(DIRECTORY / "compact" / r["run_name"] / "full_validation_predictions.npz")
            np.testing.assert_array_equal(data["targets"], baseline["targets"])
            user_metrics.append(per_user(data["predictions"], data["targets"])["ndcg@10"])
        comparisons[f"{category}_mean_seed_id_vs_markov"] = paired_interval(np.mean(user_metrics, axis=0), markov)
    for name in ("ae_infonce", "ae_no_infonce"):
        p = np.load(DIRECTORY / "checkpoints" / name / "paired_predictions.npz")
        markov_data = np.load(DIRECTORY / "nonparametric/Beauty_markov_last_predictions.npz")
        markov = per_user(markov_data["predictions"][p["indices"]], p["targets"])["ndcg@10"]
        joint = per_user(p["support_k1"], p["targets"])["ndcg@10"]
        comparisons[f"markov_vs_{name}_same_users"] = paired_interval(markov, joint)
    info = np.load(DIRECTORY / "checkpoints/ae_infonce/paired_predictions.npz")
    noinfo = np.load(DIRECTORY / "checkpoints/ae_no_infonce/paired_predictions.npz")
    np.testing.assert_array_equal(info["indices"], noinfo["indices"])
    comparisons["ae_infonce_vs_ae_no_infonce"] = paired_interval(
        per_user(info["support_k1"], info["targets"])["ndcg@10"],
        per_user(noinfo["support_k1"], noinfo["targets"])["ndcg@10"])
    result = {"split": "validation", "test_targets_used": False, "groups": group_results,
              "paired_comparisons": comparisons, "completed_training_runs": len(runs),
              "limitations": ["Compact checkpoints selected on 4096 validation users, then audited on full validation.",
                              "All selection and comparisons remain exploratory; no held-out test claim.",
                              "Bootstrap intervals condition on fitted models; they do not include model-selection uncertainty.",
                              "Compact experiments are not FMRec or MIRAGE reproductions."]}
    write_json(DIRECTORY / "summary.json", result)
    markdown = ["# 实测验证集结果", "", "所有数字均为验证集；各组 40 epochs，best checkpoint 由固定 4096 验证用户选择，随后报告完整验证集。", "",
                "| 数据集 | 表示 | 编码器 | 目标 | seeds | NDCG@10 均值 ± 样本标准差 | Recall@10 均值 |", "|---|---|---|---|---|---|---|"]
    for r in group_results:
        mean, sd = r["ndcg@10"]["mean"], r["ndcg@10"]["sample_std"]
        formatted = f"{mean:.6f}" + (f" ± {sd:.6f}" if sd is not None else "（单次）")
        markdown.append(f"| {r['category']} | {r['representation']} | {r['encoder']} | {r['objective']} | {r['seeds']} | {formatted} | {r['recall@10']['mean']:.6f} |")
    (DIRECTORY / "results_table.md").write_text("\n".join(markdown) + "\n")
    figure(group_results)
    print(json.dumps(result, indent=2))


def figure(groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.9))
    selected = [("pca", "transformer", "Fixed content"), ("adapter", "transformer", "Shared content adapter"),
                ("hybrid", "transformer", "Content + ID residual"), ("id", "hstu", "ID + HSTU"),
                ("id", "transformer", "ID + Transformer")]
    for pos, (representation, encoder, label) in enumerate(selected):
        r = next(r for r in groups if r["category"] == "Beauty" and r["representation"] == representation
                 and r["encoder"] == encoder and r["objective"] == "ce")
        mean, sd = r["ndcg@10"]["mean"], r["ndcg@10"]["sample_std"]
        axes[0].barh(pos, mean, xerr=sd, color="#29799b" if representation != "id" else "#dd994a", capsize=3)
        axes[0].text(mean + .0015, pos, f"{mean:.4f}", va="center", fontsize=9)
    baseline = json.loads((DIRECTORY / "nonparametric/Beauty.json").read_text())["unmasked"]["markov_last"]["ndcg@10"]
    axes[0].axvline(baseline, color="gray", linestyle="--", label="Transition count")
    axes[0].set_yticks(range(len(selected)), [x[2] for x in selected])
    axes[0].set_xlim(0, .07)
    axes[0].set_xlabel("Full-validation NDCG@10")
    axes[0].set_title("A. Representation and ranking signal")
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)
    checkpoint = json.loads((DIRECTORY / "checkpoints/ae_infonce/results.json").read_text())
    names = [("support_k1", "32 draws / k=1"), ("support_k5", "32 draws / k=5"),
             ("mean_sample", "Mean of 32 draws"), ("zero_noise", "Zero initial noise"),
             ("single_sample", "One draw"), ("shuffled_history", "Shuffled histories")]
    axes[1].barh(range(len(names)), [checkpoint["decoding"]["scores"][x[0]]["ndcg@10"] for x in names], color="#348b78")
    axes[1].set_yticks(range(len(names)), [x[1] for x in names])
    axes[1].set_xlabel("NDCG@10 on the same 4096 validation users")
    axes[1].set_title("B. Existing joint model: paired decoding")
    values = checkpoint["shortcut"]["by_time"]
    times = [x["t_noise_fraction"] for x in values]
    for name, label, color in (("correct_history", "Correct history", "#29799b"),
                               ("shuffled_history", "Shuffled history", "#c46555"),
                               ("unconditional", "No history", "#999999")):
        axes[2].plot(times, [x[name]["infonce"] for x in values], "o-", label=label, color=color)
    axes[2].set_xlabel("Noise fraction (0 = clean target; 1 = pure noise)")
    axes[2].set_ylabel("InfoNCE loss on 256 fixed queries")
    axes[2].set_title("C. Low-noise alignment can ignore history")
    axes[2].legend(frameon=False)
    fig.suptitle("UGRec v4 exploratory diagnostics — validation only", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(DIRECTORY / "research_overview.png", dpi=180, bbox_inches="tight")
    fig.savefig(DIRECTORY / "research_overview.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
