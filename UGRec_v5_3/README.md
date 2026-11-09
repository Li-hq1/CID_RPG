# UGRec_v5_3：LOO 与未来多目标覆盖

从 v5_1 物理复制 AE、HSTU、速度网络和训练/保存组件，独立实现未来窗口划分与 CE/CFM 对照。保留 LOO；旧版本和共享 cache 只读。

本轮Beauty/seed2024四组初筛已完成：[实验分析](docs/初步实验分析_20261108.md)。future CE改善覆盖，当前next/future CFM均未提供增量；暂不追加调参或多种子。

先读 [方案](docs/方案.md)、[实验计划](docs/实验计划.md) 和 [执行约定](AGENTS.md)。数据与工程结果见 [验收记录](docs/验收记录.md)。本版默认 `configs/final.json` 是新多目标协议的纯 CE 对照 A，不是已经确定的最终论文方法。

| 配置 | 下一物品 CE 以外的监督 | 推荐读出 |
|---|---|---|
| `A_ce.json` | 无 | 下一物品排序头 |
| `B_next_fm.json` | 下一物品 CFM | 多样本 FM 概率混合 |
| `C_future_ce.json` | 未来分布 CE | 确定性未来头 |
| `D_future_fm.json` | 未来物品分布 CFM | 多样本 FM 概率混合 |

默认窗口 h=3：最后3项预留test、此前3项validation、更早的前缀train；至少2个训练物品。测试目标不映射、不评价。该协议筛选较长序列用户，不能与全用户旧LOO分数直接比较。Beauty审计保留8300/22363人（37.1%），验证每人3个不同物品；训练11210窗口、60506个next位置，其中45717个有完整未来监督。

```bash
cd /home/haoquan/CID_RPG/UGRec_v5_3
bash scripts/test.sh

# 数据审计和工程检查的输出 id/路径必须是新的。
bash scripts/python.sh audit_data.py --category Beauty --output docs/new_data_audit.json
bash scripts/python.sh scripts/check.py --id new_readiness --device cpu

# 正式实验前检查 GPU 资源；此处 GPU_ID 仅为示例。
GPU_ID=3 bash scripts/train.sh --config configs/A_ce.json --run-id new_A_h3_s2024
GPU_ID=3 bash scripts/train.sh --config configs/B_next_fm.json --run-id new_B_h3_s2024

# 依据 A/B 证据决定是否开展 C/D；不自动四组/多数据集/多种子扫描。
GPU_ID=3 bash scripts/train.sh --config configs/C_future_ce.json --run-id new_C_h3_s2024
GPU_ID=3 bash scripts/train.sh --config configs/D_future_fm.json --run-id new_D_h3_s2024

# 同一个新方法切回标准 LOO；未来辅助窗口仍只来自训练前缀。
bash scripts/python.sh run.py --config configs/A_ce.json --protocol leave_one_out --device cpu --check --run-id new_loo_check

# 复现 v5_1 旧 control 的 LOO 方法（它不是上述 A/B 机制）。
bash scripts/python.sh loo_run.py --config configs/legacy_control.json --device cpu --smoke-test --run-id new_legacy_smoke
```

新run保存展开配置、源码快照/哈希、数据指纹、用户选择哈希、训练日志、完整checkpoint和验证预测。目录拒绝覆盖，失败保存failure.json。默认40epoch、seed2024，固定4096人按多正例NDCG@10选模，再审计完整validation；validation包含选模用户。FM默认32样本×4步Euler，128次速度调用，不与CE声称等推理成本。辅助项可以改变共享历史表示，不保证下一物品效果不降。

`--check`保留完整模型宽度，执行4次更新及保存/重载；`--smoke-test`还缩小模型。两者均不具备benchmark资格。GPU显存默认限额为单卡总量20%，可显式设置`--memory-fraction`。CDs走修正抽样训练，所有评估均全目录。

```bash
bash scripts/python.sh compare_runs.py \
  outputs/experiments/Beauty/new_A_h3_s2024 \
  outputs/experiments/Beauty/new_B_h3_s2024 \
  --output docs/new_AB_comparison.json
```

比较器拒绝未完成、工程检查、不同划分/用户/源码/预算或非arm配置不一致的run。A/B回答现有下一物品CFM是否在多目标评测已有作用；C/D回答相同未来监督下FM是否提供增量。单种子结果仅用于趋势判断，不能先写成稳定优势或证明benchmark有问题。
