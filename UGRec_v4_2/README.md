# UGRec v4_2：RF 与单一抽样排序目标

保留 **内容特征 → 可训练 AE → 历史编码 → RF 连续生成 → 物品检索**。v1–v4_1 保留原样。本版本重点解决 v4_1 三类 CE 和逐步全目录编码的问题，所有新生成候选保留 RF velocity MSE，不使用 teacher 对齐。

当前研究结果见 [v4_2 报告](docs/v4_2_研究报告.md)，逐实验结果见 [结果索引](outputs/unified_20261028/results_table.md)。研究尚在进行时，索引只包含已完成实验。

## 训练目标

`RF MSE + λ × 单一抽样排序损失 + AE 本地重建/几何目标`。

- 只对实际生成终点施加排序监督；没有独立历史 CE、带噪目标恢复 CE 或 teacher loss。
- AE 在线编码历史、正样本与有限负样本；不维护每步更新的全物品 latent 目录。
- RF 的回归样本及速度标签停止梯度；历史条件保持可微。排序分支对 AE、历史编码器、生成器和可选条件源共同反传。
- 随机负样本按 item 均匀、有放回抽取。修正后的分母是全目录配分函数的无偏估计，取对数后的损失并不是全目录 CE 的无偏估计。
- `direct` 是明确标注的非生成参考，不是候选生成方案。旧 `ranking` 与旧对比分支仅保留兼容性；新模式要求它们关闭。

## 复现

从此目录运行，使用仓库的 Python 环境。候选已冻结为 `configs/v4_2_frozen.json`；三种子与 Sports 复核结果见研究报告。该配置实际执行一次 Euler 更新，`steps=4` 只供 rollout 分支使用。

便于阅读的等效配置为 `configs/v4_2_minimal.json`：已核对它在本版本默认值下展开后与冻结配置完全一致，省略旧teacher/三CE等兼容字段。论文复现仍使用完整冻结配置，避免未来默认值变化影响结果。

```bash
# 已验证的真实缓存流程检查；跳过 test
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ../.venv/bin/python -m joint_training \
  --config research/configs/v4_2/smoke.json --smoke-test --device cpu \
  --run-id another_smoke

# 有限计划；每个条目保存配置、逐轮日志、模型、逐用户预测和代码来源
../.venv/bin/python -m research.run_unified research/v4_2_confirmation.json --workers 2

# 完整验证与同 checkpoint 的零噪声/单样本/起点对照
../.venv/bin/python -m research.audit_unified outputs/unified_20261028/Beauty/RUN \
  --device cuda:0 --controls

# 已冻结 checkpoint 的测试集评估；不重新选模
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ../.venv/bin/python -m research.evaluate_frozen_test \
  --device cuda:0

# 重新生成表格与已完成三种子的配对比较
../.venv/bin/python -m research.summarize_unified
```

训练入口拒绝覆盖已有目录。复现计划需修改实验名或 seed；不要直接覆盖现有结果。新研究脚本强制 `evaluate_test=false`；模型选择仅使用固定的4096验证用户，完整 validation 包含该子集，不是独立测试。

## 扩展与检索

`JointModel.raw_items` 支持从 CPU 内容特征中按需取样；GPU 原始特征缓存是可选优化，原始特征存储本身仍随目录增长。AE 本地样本采用有界抽样，不再每步分配全目录随机排列。

完整物品 latent 目录只在验证/模型导出时编码。固定模型部署时复用导出的目录和索引；若 AE 更新，需要同步重建或更新索引。可选 `evaluation.retrieval_backend=faiss_hnsw` 避免每次请求计算全目录相似度，需要安装 `requirements-ann.txt`。它是近似检索，必须核对召回误差；小目录上的 CPU HNSW 不保证快于 GPU 精确检索。

科学边界：这些模块各有相关先例，组合不自动证明新颖性；保留 RF 也不自动证明随机生成或速度匹配提供独立收益。所有准确率结果与效率测试的适用范围均应分别说明。
