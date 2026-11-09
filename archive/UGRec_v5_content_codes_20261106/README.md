# UGRec v5：最终方案

本目录是 UGRec 的最终方案，默认配置 `configs/final.json`。模型、损失、数据、选模规则和结果见 [方案](docs/方案.md)。

```bash
cd /home/haoquan/CID_RPG/UGRec_v5

# 最终方案：Beauty、seed 2024、40 epoch。每 5 轮用固定 4096 个验证用户选模，最后评估全部验证用户。
GPU_ID=0 bash scripts/train.sh

# 换数据集或种子时使用新的 run id，已有目录会拒绝覆盖。
GPU_ID=0 bash scripts/train.sh --category Sports_and_Outdoors --seed 2025 --run-id ugrec_v5_final_sports_s2025

# 只检查数据、损失和保存能否走通，不代表正式指标。
bash scripts/train.sh --smoke-test --device cpu

bash scripts/test.sh
```

## 配置

| 配置 | 用途 |
|---|---|
| `configs/final.json` | 最终方案，`scripts/train.sh` 默认读取 |
| `configs/frozen.json` | v4_2 冻结原版（速度回归 + 抽样排序，两次前向），复现用 |
| `configs/minimal.json` | `frozen.json` 的简写，展开后逐字段相同 |
| `configs/matched.json` | 一次前向的抽样排序，复现 2026-10-29 结果用 |
| `configs/direct_reference.json` | 不走整流流的直接预测对照 |

用历史配置时显式传 `--config`，例如 `bash scripts/train.sh --config configs/frozen.json --run-id ugrec_v5_frozen_s2024`。

`joint_training/config.py` 里的 dataclass 默认值是兼容旧代码的默认，不是任何一个方案。不带配置文件启动，跑的不是最终方案。

2026-11-05 改为最终方案前的代码备份在 `../archive/UGRec_v5_pre_final_20261105/`。
