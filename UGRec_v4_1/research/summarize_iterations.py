"""Summarize finished runs without selecting a winner from full validation."""

from collections import defaultdict
import json
import numpy as np
from .common import ROOT, paired_interval, per_user, summarize, write_json

OUT = ROOT / "outputs/iterative_20261028"


def main():
    rows, pending = [], []
    for directory in sorted(OUT.glob("*/*")):
        if not directory.is_dir() or not (directory / "config.json").exists():
            continue
        if not (directory / "results.json").exists():
            pending.append(str(directory.relative_to(OUT)))
            continue
        result = json.loads((directory / "results.json").read_text())
        config = result["config"]
        row = {"run": directory.name, "category": config["data"]["category"],
               "seed": config["training"]["seed"], "epochs": config["training"]["epochs"],
               "seconds": result["seconds_total"], "path": str(directory),
               "config": config, "best": result["best"]}
        for head in result["best"]:
            if f"selected_by_{head}" in result["audits"]:
                row[head] = result["audits"][f"selected_by_{head}"][head]
                data = np.load(directory / f"full_selected_{head}_decoded_{head}.npz")
                selected = np.load(directory / f"selection_{head}.npz")["indices"]
                complement = np.ones(len(data["targets"]), dtype=bool)
                complement[selected] = False
                row[f"{head}_validation_complement"] = summarize(data["predictions"][complement], data["targets"][complement])
        if "one_step" in row and "prior_control" in result["audits"]["selected_by_one_step"]:
            one = np.load(directory / "full_selected_one_step_decoded_one_step.npz")
            prior = np.load(directory / "full_selected_one_step_decoded_prior_control.npz")
            np.testing.assert_array_equal(one["targets"], prior["targets"])
            row["one_step_minus_same_noise_prior"] = paired_interval(
                per_user(one["predictions"], one["targets"])["ndcg@10"],
                per_user(prior["predictions"], prior["targets"])["ndcg@10"])
        if "flow" in row and "prior_control" in result["audits"]["selected_by_flow"]:
            row["prior_control_at_flow_checkpoint"] = result["audits"]["selected_by_flow"]["prior_control"]
            flow = np.load(directory / "full_selected_flow_decoded_flow.npz")
            prior = np.load(directory / "full_selected_flow_decoded_prior_control.npz")
            np.testing.assert_array_equal(flow["targets"], prior["targets"])
            row["flow_minus_same_noise_prior"] = paired_interval(
                per_user(flow["predictions"], flow["targets"])["ndcg@10"],
                per_user(prior["predictions"], prior["targets"])["ndcg@10"])
        baseline = ROOT.parent / "UGRec_v4/outputs/research_20261028/nonparametric" / f"{row['category']}_markov_last_predictions.npz"
        if baseline.exists():
            reference = np.load(baseline)
            for head in ("flow", "one_step", "ranking"):
                if head in row:
                    data = np.load(directory / f"full_selected_{head}_decoded_{head}.npz")
                    np.testing.assert_array_equal(data["targets"], reference["targets"])
                    row[f"{head}_minus_markov"] = paired_interval(
                        per_user(data["predictions"], data["targets"])["ndcg@10"],
                        per_user(reference["predictions"], reference["targets"])["ndcg@10"])
        ce_directory = OUT / row["category"] / f"r1_ce_only_s{row['seed']}_e{row['epochs']}"
        ce_path = ce_directory / "full_selected_ranking_decoded_ranking.npz"
        if ce_path.exists() and ce_directory != directory:
            reference = np.load(ce_path)
            for head in ("flow", "one_step", "ranking"):
                if head in row:
                    data = np.load(directory / f"full_selected_{head}_decoded_{head}.npz")
                    np.testing.assert_array_equal(data["targets"], reference["targets"])
                    row[f"{head}_minus_matched_ce"] = paired_interval(
                        per_user(data["predictions"], data["targets"])["ndcg@10"],
                        per_user(reference["predictions"], reference["targets"])["ndcg@10"])
        rows.append(row)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["category"], row["run"].rsplit("_s", 1)[0], row["epochs"])].append(row)
    grouped = []
    for (category, name, epochs), runs in groups.items():
        group = {"category": category, "name": name, "epochs": epochs, "seeds": [r["seed"] for r in runs]}
        for head in ("flow", "one_step", "ranking"):
            values = [r[head]["ndcg@10"] for r in runs if head in r]
            if values:
                group[head] = {"mean": float(np.mean(values)),
                               "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                               "audited_seeds": [r["seed"] for r in runs if head in r]}
        grouped.append(group)
    summary = {"split": "validation", "completed": len(rows), "pending": pending,
               "runs": rows, "groups": grouped,
               "limitations": ["Each head selects its own checkpoint on 4096 validation users.",
                               "Full validation includes selection users; results are exploratory.",
                               "Bootstrap conditions on fitted models and does not account for selection."]}
    write_json(OUT / "summary.json", summary)
    lines = ["# v4.1 探索性验证结果", "", "Flow 与 ranking 分别在固定验证子集选 checkpoint，随后审计完整验证集；不是测试成绩。", "",
             "| 数据 | 实验 | seed | Flow N@10 | 一步生成 N@10 | 辅助排序 N@10 | 同噪声起点、不运行 flow | best epochs(flow/one/rank) |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        score = lambda key: f"{row[key]['ndcg@10']:.6f}" if key in row else "—"
        best_epochs = "/".join(str(row["best"].get(h, {}).get("epoch", "—")) for h in ("flow", "one_step", "ranking"))
        lines.append(f"| {row['category']} | {row['run']} | {row['seed']} | {score('flow')} | {score('one_step')} | {score('ranking')} | {score('prior_control_at_flow_checkpoint')} | {best_epochs} |")
    lines += ["", "## 筛查子集 best NDCG@10", "", "以下是同一选模子集的筛查分数，不能当成完整验证或测试。", "",
              "| 数据 | 实验 | Flow | 一步生成 | 辅助排序 |", "|---|---|---:|---:|---:|"]
    for row in rows:
        value = lambda key: f"{row['best'][key]['score']:.6f}" if key in row["best"] else "—"
        lines.append(f"| {row['category']} | {row['run']} | {value('flow')} | {value('one_step')} | {value('ranking')} |")
    lines += ["", f"已完成 {len(rows)} 组；尚未完成 {len(pending)} 组。"]
    (OUT / "results_table.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"completed": len(rows), "pending": pending, "groups": grouped}, indent=2))


if __name__ == "__main__":
    main()
