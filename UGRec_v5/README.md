# UGRec v5：control

本目录当前默认恢复为 control，`configs/final.json` 与 `configs/control.json` 相同：内容 AE、HSTU、单历史条件的一步流和噪声边缘似然。模型、损失、数据、选模规则和结果见 [方案](docs/方案.md)。

```bash
cd /home/haoquan/CID_RPG/UGRec_v5

# 默认 control：Beauty、seed 2024、40 epoch。每 5 轮用固定 4096 个验证用户选模，最后评估全部验证用户。
GPU_ID=0 bash scripts/train.sh

# 换数据集或种子时使用新的 run id，已有目录会拒绝覆盖。
GPU_ID=0 bash scripts/train.sh --category Sports_and_Outdoors --seed 2025 --run-id ugrec_v5_control_sports_s2025

# 显式指定 control：与默认方案一致，训练 8 个查询，验证 32 个查询。
GPU_ID=0 bash scripts/train.sh --config configs/control.json --run-id ugrec_v5_control_s2024

# 只检查数据、损失和保存能否走通，不代表正式指标。
bash scripts/train.sh --smoke-test --device cpu

bash scripts/test.sh
```

## 配置

| 配置 | 用途 |
|---|---|
| `configs/final.json` | 当前默认 control：禁用 VLAD，单历史条件、8 个训练噪声查询、32 个验证查询 |
| `configs/control.json` | 与默认配置完全一致，保留显式 control 入口 |
| `configs/vlad_condition_20261106/` | 六组 VLAD 条件形式实验的复现配置，需显式指定 |
| `configs/frozen.json` | v4_2 冻结原版（速度回归 + 抽样排序，两次前向），复现用 |
| `configs/minimal.json` | `frozen.json` 的简写，展开后逐字段相同 |
| `configs/matched.json` | 一次前向的抽样排序，复现 2026-10-29 结果用 |
| `configs/direct_reference.json` | 不走整流流的直接预测对照 |

用历史配置时显式传 `--config`，例如 `bash scripts/train.sh --config configs/frozen.json --run-id ugrec_v5_frozen_s2024`。

`joint_training/config.py` 里的 dataclass 默认值用于兼容旧代码。请使用 `scripts/train.sh`，它自动读取 `configs/final.json`。

2026-11-05 改为最终方案前的代码备份在 `../archive/UGRec_v5_pre_final_20261105/`。

2026-11-06 已移除内容残差码的预计算和输入分支；前一方案源码保存在 `../archive/UGRec_v5_content_codes_20261106/`。默认 control 从内容 AE 更新到 HSTU、条件生成器和物品概率，无需离线聚类。

VLAD 代码、独立配置和输出保留用于复现。先前的 [混合验证](docs/VLAD混合验证_20261106.md) 与 [条件形式探索](docs/VLAD条件形式探索_20261106.md) 均为历史实验；简洁候选未超过 control，按用户要求恢复默认 control。本次恢复不重跑训练。
