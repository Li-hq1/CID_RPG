# v4_2 研究入口

本轮新增入口是 `unified.py`，目标为 RF + 单一抽样排序 + AE 本地目标。teacher 一律关闭。`iterate.py` 只用于复现旧无 teacher 基准；此前版本的研究脚本保留用于追溯，不代表本轮新增配置。

- `v4_2_stage1.json`：9组源分布、生成步数和排序形式筛查。
- `v4_2_stage1_uniform.json`：6组均匀负样本及配分函数修正。
- `v4_2_stage1_coupled.json`：共享 RF forward 与排序，以及成对排序失败机制。
- `v4_2_stage1_weights.json`：排序权重与 AE 几何检查。
- `v4_2_stage1_endpoint_tuning.json`：单步方案的温度、容量、优化与负样本预算检查。
- `v4_2_stage2.json`：4组40epoch完整验证对照。
- `v4_2_confirmation.json`：Beauty/Sports、3种子的12组冻结确认；实际分卡执行记录见剩余队列和 `v4_2_sports_parallel.json`。
- `audit_unified.py`：完整验证、同 checkpoint 推理控制与表示几何。
- `profile_scaling.py`：候选 AE 编码、匹配、反传的独立规模计时；合成数据不用于效果声明。
- `profile_retrieval.py`：对完全相同生成向量比较精确检索与 HNSW。
- `summarize_unified.py`：结果索引、三seed均值/标准差及逐用户配对区间。
- `finalize_unified.py`：冻结源码核对、频次/转移分层与同模型控制统计。
- `complete_confirmation.py`：本轮有限后台收尾与报告/静态图生成；现已完成，不要为阅读结果重新启动训练。
- `evaluate_frozen_test.py`：只读取已选定的 12 个冻结 checkpoint，在 leave-one-out 最后一项上评估。不训练、不改选模。

训练、筛查和冻结确认位于 `outputs/unified_20261028`，数据切分与 v4_1 相同。那些运行的 test 不参与拟合或选模。20epoch子集筛查、40epoch完整validation与后来的测试集结果不可混为同一排行榜。测试集汇总是 `outputs/unified_20261028/test_summary.json`。后续运行保存源代码快照；探索早期仅保存代码hash，最终复核使用冻结代码与配置。
