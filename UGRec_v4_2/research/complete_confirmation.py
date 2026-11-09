"""Finite local completion pipeline that survives a disconnected chat client."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from .common import ROOT, write_json


BASE = ROOT / "outputs/unified_20261028"


def command(*arguments):
    subprocess.run([sys.executable, "-m", *arguments], cwd=ROOT,
                   env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}, check=True)


def render_report():
    summary = json.loads((BASE / "summary.json").read_text())
    diagnosis = json.loads((BASE / "final_diagnostics.json").read_text())
    lines = ["## 冻结确认的最终结果", "", "完成日期：2026-10-29。以下均为完整 validation，均值±样本标准差；每组 seeds2024/2025/2026。", "",
             "| NDCG@10 | Beauty | Sports |", "|---|---:|---:|"]
    comparisons = summary["three_seed_comparisons"]
    for name, label in [("frozen_flow", "统一抽样排序＋RF，一步生成"), ("frozen_direct", "同内容AE＋HSTU，直接抽样排序")]:
        cells = [f"{comparisons[c][name]['mean']:.6f} ± {comparisons[c][name]['sample_std']:.6f}"
                 for c in ["Beauty", "Sports_and_Outdoors"]]
        lines.append("| " + " | ".join([label, *cells]) + " |")
    lines.extend(["", "| Flow − direct | 平均差 | 条件于已训练模型的用户bootstrap95%区间 |", "|---|---:|---:|"])
    for category in ["Beauty", "Sports_and_Outdoors"]:
        delta = comparisons[category]["flow_minus_direct"]
        lo, hi = delta["bootstrap_95pct"]
        lines.append(f"| {category} | {delta['difference']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines.extend(["", "区间先对三种子的逐用户指标平均，再bootstrap用户；不覆盖训练、调参和选模不确定性，也不是独立测试的显著性检验。直接参考不是只删除RF MSE的逐项消融，因而上述比较评价的是整条生成路径的额外收益。", "",
                  "| 排除4096选模用户后的 NDCG@10 | Flow | Direct |", "|---|---:|---:|"])
    for category in ["Beauty", "Sports_and_Outdoors"]:
        outside = comparisons[category]["outside_selection"]
        lines.append(f"| {category} | {outside['frozen_flow']:.6f} | {outside['frozen_direct']:.6f} |")
    lines.extend(["", "逐种子结果：", "", "| 数据集 | seed | Flow | Direct |", "|---|---:|---:|---:|"])
    for category in ["Beauty", "Sports_and_Outdoors"]:
        for i, seed in enumerate([2024, 2025, 2026]):
            lines.append(f"| {category} | {seed} | {comparisons[category]['frozen_flow']['scores'][i]:.6f} | {comparisons[category]['frozen_direct']['scores'][i]:.6f} |")
    lines.extend(["", "## 同一模型：噪声和速度更新各自做了什么", "", "以下均为 seed2024，固定所选 checkpoint，完整 validation；没有据此重新选配置。single只把32次采样改为1次，保持相同距离与support汇总。", "",
                  "| 解码 | Beauty NDCG@10 | Sports NDCG@10 |", "|---|---:|---:|"])
    for name, label in [("flow", "RF更新＋32 draws"), ("zero_noise", "条件源零噪声＋RF"),
                        ("single", "RF更新＋1 draw"), ("prior", "条件源＋噪声，跳过RF"),
                        ("ranking", "条件源的线性预测，跳过RF与噪声")]:
        cells = []
        for category in ["Beauty", "Sports_and_Outdoors"]:
            if name == "flow":
                value = comparisons[category]["frozen_flow"]["scores"][0]
            else:
                value = diagnosis["categories"][category]["seed2024_controls"][name]["ndcg@10"]
            cells.append(f"{value:.6f}")
        lines.append("| " + " | ".join([label, *cells]) + " |")
    lines.extend(["", "这里的 `ranking` **没有独立历史CE训练**，只是同一生成模型中W h的输出。它不能替代另训的强direct对照。Flow改善这个共同训练的起点，只能说明速度更新承担了预测修正；不自动证明它超过直接排序，更不自动证明随机生成优于确定性预测。各控制的逐用户配对区间见 `final_diagnostics.json`。", "",
                  "## 同编码器的新旧目标比较", "", "不能把改HSTU的收益归给新的排序目标。保留两个teacher0复现，均为Beauty、seed2024、40轮完整验证：", "",
                  "| 编码器 | 旧三CE＋RF四步 | 新单一抽样排序＋RF一步 |", "|---|---:|---:|"])
    for encoder, old_name, new_name in [
        ("Transformer", "v42_legacy_no_teacher_s2024_e40", "s2_history_endpoint_uniform_corrected_s2024_e40"),
        ("HSTU", "v42_legacy_hstu_no_teacher_s2024_e40", "frozen_flow_s2024_e40")]:
        old = json.loads((ROOT / "outputs/iterative_20261028/Beauty" / old_name / "results.json").read_text())
        new = json.loads((BASE / "Beauty" / new_name / "results.json").read_text())
        lines.append(f"| {encoder} | {old['audits']['selected_by_flow']['flow']['ndcg@10']:.6f} | {new['audits']['flow']['ndcg@10']:.6f} |")
    lines.extend(["", "这是整套目标/候选计算/生成步数的比较，不是仅改变负样本数的消融；均只有一个种子。效果降低就应承认为效率与简化的代价，而不能用跨编码器数字制造提升。", "",
                  "## 表示与行为分层", "", "以下为三种子逐用户平均。已见转移指训练中出现过‘历史末项→目标’；未见转移不等于冷物品。目标频次只用于事后诊断，不能在推理时用于分流。", "",
                  "| 数据集 | 分层 | 用户数 | Flow NDCG@10 | Direct NDCG@10 |", "|---|---|---:|---:|---:|"])
    for category in ["Beauty", "Sports_and_Outdoors"]:
        groups = diagnosis["categories"][category]["groups"]
        for key, label in [("seen_transition", "已见转移"), ("unseen_transition", "未见转移"), ("target_count_0_to_5", "训练频次≤5")]:
            a, b = [groups[name]["strata"][key] for name in ["frozen_flow", "frozen_direct"]]
            lines.append(f"| {category} | {label} | {a['users']} | {a['ndcg@10']:.6f} | {b['ndcg@10']:.6f} |")
    followup = json.loads((BASE / "Beauty/followup_hstu_k4096_s2024_e40/results.json").read_text())
    followup_score = followup["audits"]["flow"]["ndcg@10"]
    lines.extend(["", "分层差值区间及其他频次分层保存在 `final_diagnostics.json`。它们是对已选模型的诊断，不是另一个选模目标。Beauty频次≤5的优势在三个种子中方向一致，用户bootstrap区间不跨零；Sports的该分层均值略高，但区间跨零且一个种子为负差。因此这是值得追踪的Beauty线索，尚不是普遍长尾或冷启动优势。", "",
                  "## 事后补充：增加负样本预算", "",
                  f"看到同HSTU旧方案更强后，只把负样本从1024提高到4096，其余保持冻结设置，Beauty seed2024、40轮完整验证为 **{followup_score:.6f}**；原K1024为 .059022，旧三CE为 .061229。该结果是明确标记的事后单种子探索，不替换上面的冻结三种子结果，也不据此重新选择Sports参数。它只能初步检查抽样竞争预算是否影响精度，不能单独定位三类CE、步数与候选预算各自的因果贡献。", "",
                  "## 对本轮的判断与后续优先级", "", "v4_2已把训练收敛到 **RF回归＋一个实际生成结果的抽样排序＋局部AE**，避免训练时逐步更新全目录latent。它是可继续使用的简化研究基线，不是已经证明优于直接排序的新生成方法。", "",
                  "两个数据集的整体flow−direct区间均跨零；Beauty同HSTU旧三CE方案则高于当前简化方案。零噪声控制没有显示随机采样的必要性：Beauty该单种子中零噪声更高，条件用户bootstrap区间也不跨零；Sports两者区间跨零。应把这些结果与实际速度更新相对弱起点的明显改善区分开，不能把后者当作RF概率运输目标的独立贡献。", "",
                  "1. **先定位简化的精度损失。** 固定终点目标、梯度规则和步数，用全目录单一排序作临时诊断参考，再比较K1024/K4096；这样才有助于区分抽样近似的损失与删除历史监督的损失。随后单独检查RF历史条件梯度进入AE的影响。诊断中保留全目录参考，不等于把它重新放回最终可扩展方案。", 
                  "2. **看RF相对强direct的稳定差异。** 两数据集、三种子的均值/波动、同模型零噪声和起点控制应一起解释。还应补与RF网络容量/计算更接近的确定性MLP查询头，区分额外非线性容量与概率运输。当前线性direct已足够接近，不能因此声称其头部参数量也匹配。没有独立收益证据时，应明确保留这个未解决问题。", 
                  "3. **把排序几何与概率运输分开验证。** 成对损失失败和方向塌缩说明，可训练AE中的相对几何比删去softmax更关键。后续替代目标必须同时检查有效秩、物品分离和生成终点排序，而非只看RF MSE。", 
                  "4. **若要证明生成价值，研究多步/分布目标。** 一步条件均值与逐噪声同标签监督天然可能收缩样本。需要与确定性映射比较的分布效用、覆盖/校准及相同推理预算，而不是仅保留采样运算。多步和分布监督也有既有工作，不能直接宣称创新。", 
                  "5. **内容与行为互补仍值得单独推进。** 可在现框架研究内容空间的行为残差/可靠性条件源，但必须比较简单行为记忆＋内容排序参考，不能用弱RF基准论证必要性。", 
                  "6. **规模主张还缺真实数据验证。** 当前仅证明抽样训练的计算结构更可扩展；还需真实更大目录、不同时间切分与干净测试。不要由Sports上的结果推断大数据必然更好。", "",
                  f"最终统一分支完成 {summary['completed_training_runs']} 组训练，另有2组旧teacher0复现；中断半成品不计入。12组冻结确认的模型、AE、训练、采样与推理源码哈希已逐一核对一致。分析脚本的后续增补不改变这些计算。", "",
                  "本地有限后台流水线不调用模型API；429不会影响已经提交给系统任务管理器的本轮训练。新聊天接续说明见 `docs/接续记录_20261029.md`。"])
    path = ROOT / "docs/v4_2_研究报告.md"
    text = path.read_text()
    marker = "## 冻结确认的最终结果"
    text = text.split(marker)[0]
    text = text.split("当前确认实验进行中；")[0].rstrip()
    text = re.sub(r"<!-- final_outcome_start -->.*?<!-- final_outcome_end -->\n*", "", text, flags=re.S)
    beauty = comparisons["Beauty"]
    sports = comparisons["Sports_and_Outdoors"]
    outcome = ("<!-- final_outcome_start -->\n"
               f"本轮已完成{summary['completed_training_runs']}组统一分支训练和2组旧方案复现。核心产出是**保留RF、去掉teacher、三类CE收敛为一个抽样排序目标，并移除训练时逐步全目录编码**。\n\n"
               f"三种子完整验证：Beauty生成 {beauty['frozen_flow']['mean']:.6f}，直接排序 {beauty['frozen_direct']['mean']:.6f}；"
               f"Sports生成 {sports['frozen_flow']['mean']:.6f}，直接排序 {sports['frozen_direct']['mean']:.6f}。"
               "效果、方差、配对区间和同模型控制见文末。最终目标仍属于抽样CE家族，尚不能把计算简化等同于已证明RF的独立收益。\n"
               "<!-- final_outcome_end -->\n\n")
    # Keep the concrete result near the top without removing the mechanism text.
    anchor = "## 本轮解决什么"
    text = text.replace(anchor, outcome + anchor, 1)
    path.write_text(text.rstrip() + "\n\n" + "\n".join(lines) + "\n")


def render_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.loads((BASE / "summary.json").read_text())
    scaling = json.loads((BASE / "profile_scaling.json").read_text())
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    colors = ["#4477AA", "#228833"]
    for ax, category, title in zip(axes[:2], ["Beauty", "Sports_and_Outdoors"], ["Beauty", "Sports"]):
        groups = summary["three_seed_comparisons"][category]
        names = ["frozen_direct", "frozen_flow"]
        ax.bar([0, 1], [groups[n]["mean"] for n in names],
               yerr=[groups[n]["sample_std"] for n in names], capsize=4, color=colors, width=.55)
        ax.set_xticks([0, 1], ["Direct ranking", "RF + ranking"])
        ax.set_title(title + ": full validation")
        ax.set_ylabel("NDCG@10 (mean and sample SD, 3 seeds)")
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
    ax = axes[2]
    for name, label, color in [("full_catalog", "Full catalog", "#CC6677"), ("sampled", "Sampled K=1024", "#228833")]:
        rows = [r for r in scaling["rows"] if r["method"] == name]
        ax.plot([r["catalog_size"] for r in rows], [r["milliseconds"] for r in rows], "o-", label=label, color=color)
    ax.set_xlabel("Catalog items (synthetic benchmark)")
    ax.set_ylabel("Candidate encode + match + backward (ms)")
    ax.set_title("Candidate training cost only")
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    figure.savefig(BASE / "v4_2_confirmation.png", dpi=180)
    figure.savefig(BASE / "v4_2_confirmation.svg")
    plt.close(figure)


def main():
    status = BASE / "completion_status.json"
    write_json(status, {"state": "waiting_for_training", "test_accessed": False})
    plan = json.loads((ROOT / "research/v4_2_confirmation.json").read_text())
    expected = [BASE / e["category"] / f"{e['name']}_s{e['seed']}_e{e['epochs']}" / "results.json" for e in plan]
    expected.append(ROOT / "outputs/iterative_20261028/Beauty/v42_legacy_hstu_no_teacher_s2024_e40/results.json")
    expected.append(BASE / "Beauty/followup_hstu_k4096_s2024_e40/results.json")
    expected.append(BASE / "sports_controls_complete.json")
    deadline = time.monotonic() + 7200
    while True:
        missing = [str(p.relative_to(ROOT)) for p in expected if not p.exists()]
        if not missing:
            break
        print(json.dumps({"waiting": missing}), flush=True)
        if time.monotonic() > deadline:
            raise TimeoutError("Finite completion pipeline timed out waiting for training.")
        time.sleep(30)
    write_json(status, {"state": "summarizing", "test_accessed": False})
    command("research.summarize_unified")
    command("research.finalize_unified")
    render_report()
    render_figure()
    write_json(status, {"state": "complete", "report": "docs/v4_2_研究报告.md", "test_accessed": False})
    print("CONFIRMATION_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(BASE / "completion_status.json", {"state": "failed", "error": repr(error), "test_accessed": False})
        raise
