# UGRec v4.1：推荐感知表示与条件残差流

保留 **内容 AE → 历史编码 → Rectified Flow → 内容候选库检索** 主框架，从 v4 独立复制；v1–v4 及其结果不修改。当前处于消融验证阶段，不宣称新颖性或测试集领先。

本轮51组训练已完成。核心结论：推荐梯度进入内容AE有明确收益，但生成头尚未稳定超过匹配CE；HSTU、teacher、RF、模型规模和多样本边缘监督均有独立控制。[完整结论与下一步](docs/本轮结论与下一步_20261028.md)包含三种子、分层、负结果和复现材料。

## 新增路径

1. 历史独立 catalog CE：同时更新内容 AE、历史编码与辅助投影，不读取带噪目标。
2. RF MSE：历史/target 两侧 AE 都断图，只训练推荐网络。
3. 可选高噪声生成端 CE：训练生成/推荐网络，候选与目标 AE 断图。
4. 可选固定 teacher：独立 live-AE 分支，对齐 denoiser 中间表征，同时更新 AE 与推荐网络。支持文本 PCA、训练图 SVD 或组合。
5. 可选历史条件先验：从历史预测语义位置加噪声出发生成，另报相同起点/噪声但不经过 flow 的对照。
6. 可选端点/轨迹监督：分别监督纯噪声端点的一步预测、或可微分4步生成的最终结果。后者训练与推理使用相同Euler路径，RF回归仍独立保留。
7. 可选多样本边缘概率监督：`ranking.rollout_samples=4`、`ranking.rollout_objective=marginal_ce`；对多条生成轨迹的目录概率先平均再取负对数。默认仍为单样本 `sample_ce`。边缘目标不保证多模态，须与相同采样数的逐样本 CE 比较。

默认生成仍使用 flow，辅助 ranking head 单独报告。新探索全部跳过 test；完整 validation 包含选模子集，不是独立测试。

## 实验入口

实验直接使用 JointModel/JointTrainer，每 GPU 一个进程队列。新 name 才能训练，已有目录拒绝覆盖。

```bash
cd /home/haoquan/CID_RPG/UGRec_v4_1
../.venv/bin/python -m research.iterate --name example_joint \
  --config research/configs/round1/joint_pred02.json \
  --category Beauty --device cuda:0 --epochs 40 --seed 2024
../.venv/bin/python -m research.summarize_iterations
../.venv/bin/python -m unittest tests.test_joint_ranking
```

`research.iterate` 默认只做固定验证子集筛查；`--audit full` 可在训练结束后审计完整验证。已有筛查结果用 `research.audit_iteration <实验目录> --device cuda:0` 补审计，不重训。两个表和JSON明确区分筛查与完整验证。

[实验表](outputs/iterative_20261028/results_table.md)、[JSON汇总](outputs/iterative_20261028/summary.json)、[逐轮研究记录](docs/迭代研究记录_20261028.md)。计划和配置位于 research/round*.json、research/configs/。前期问题分析见 [v4报告](../UGRec_v4/docs/研究分析与后续路线_20261028.md)。

冻结的跨数据候选及选定规则在 `outputs/iterative_20261028/frozen_candidate.json`。复现实验使用研究入口，它固定 matmul precision=high、4096验证用户选模和评估RNG，不使用test：

```bash
../.venv/bin/python -m research.iterate --name reproduce_frozen \
  --config research/configs/round5/frozen_candidate.json \
  --category Beauty --device cuda:0 --epochs 40 --seed 2024
```

Sports 使用实际类别名 `Sports_and_Outdoors`。每个新实验必须使用新的 name，已有目录拒绝覆盖。

## 主训练入口

ranking.enabled=true 开启新联合目标；关闭则保留原 AE-only InfoNCE 交替训练。rank、RF、本地 AE、生成端 CE、teacher 的权重与梯度路由可独立配置。

```bash
../.venv/bin/python -m research.prepare_teacher --category Beauty --kind content
../.venv/bin/python -m joint_training --device cuda:0 --run-id example_teacher \
  --set ranking.enabled=true --set ranking.prediction_weight=0.2 \
  --set ranking.teacher_weight=0.1 \
  --set ranking.teacher_path=outputs/teachers/Beauty_content.pt \
  --set training.evaluate_test=false
```

Teacher artifact 校验原始内容、item顺序和训练mask。teacher 随 checkpoint 保存，恢复不依赖原文件。历史先验要求 recommender.tiger_guidance_weight=0；标准先验支持 CFG。验证固定噪声并恢复训练 RNG；resume 严格核对模型、优化与评估协议。

`configs/research_candidates/` 提供完整主入口配置。teacher 已导出到 `outputs/teachers/`，也可用 `research.prepare_teacher` 重建。主入口使用完整validation选模，matmul precision遵循PyTorch默认；上述研究筛查入口使用子集和precision=high。因此主入口不用于逐值复现筛查checkpoint。真实缓存的 rollout+teacher CPU smoke 日志在 `outputs/rollout_teacher_cli_smoke.log`。

```bash
../.venv/bin/python -m joint_training \
  --config configs/research_candidates/Beauty_rollout.json \
  --device cuda:0 --run-id my_rollout_run
```

`history_ae_grad` / `catalog_ae_grad` / `teacher_ae_grad` 分别控制三条推荐→AE路径。全部关闭后仍保留AE本地学习和teacher对推荐网络的监督，可用于联合表示学习的控制。

紧凑筛查默认原始2560维内容、AE512→256→128、Transformer128维/2层、RF256宽/2层、batch256、40epochs，与主入口默认512维/4层/100epochs不同，不能直接将两者差异归因于新增模块。--config 支持规模和优化参数覆盖。
