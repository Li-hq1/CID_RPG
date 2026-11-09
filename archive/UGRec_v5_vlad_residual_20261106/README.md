# UGRec v5：VLAD 混合方案

本目录是 UGRec 的最终方案，默认配置 `configs/final.json`。模型、损失、数据、选模规则和结果见 [方案](docs/方案.md)。

```bash
cd /home/haoquan/CID_RPG/UGRec_v5

# 最终方案：Beauty、seed 2024、40 epoch。每 5 轮用固定 4096 个验证用户选模，最后评估全部验证用户。
GPU_ID=0 bash scripts/train.sh

# 换数据集或种子时使用新的 run id，已有目录会拒绝覆盖。
GPU_ID=0 bash scripts/train.sh --category Sports_and_Outdoors --seed 2025 --run-id ugrec_v5_final_sports_s2025

# 不加 VLAD 的 control：同样训练 8 个查询，验证 32 个查询。
GPU_ID=0 bash scripts/train.sh --config configs/control.json --run-id ugrec_v5_control_s2024

# 只检查数据、损失和保存能否走通，不代表正式指标。
bash scripts/train.sh --smoke-test --device cpu

bash scripts/test.sh
```

## 配置

| 配置 | 用途 |
|---|---|
| `configs/final.json` | 输出端 VLAD K8、每簇 1 个训练噪声，`scripts/train.sh` 默认读取 |
| `configs/control.json` | 去掉 VLAD 的边缘似然 control，训练总查询数仍为 8 |
| `configs/frozen.json` | v4_2 冻结原版（速度回归 + 抽样排序，两次前向），复现用 |
| `configs/minimal.json` | `frozen.json` 的简写，展开后逐字段相同 |
| `configs/matched.json` | 一次前向的抽样排序，复现 2026-10-29 结果用 |
| `configs/direct_reference.json` | 不走整流流的直接预测对照 |

用历史配置时显式传 `--config`，例如 `bash scripts/train.sh --config configs/frozen.json --run-id ugrec_v5_frozen_s2024`。

`joint_training/config.py` 里的 dataclass 默认值是兼容旧代码的默认，不是任何一个方案。不带配置文件启动，跑的不是最终方案。

2026-11-05 改为最终方案前的代码备份在 `../archive/UGRec_v5_pre_final_20261105/`。

2026-11-06 已移除内容残差码的预计算和输入分支；前一方案源码保存在 `../archive/UGRec_v5_content_codes_20261106/`，既有实验和输出保留。当前训练从内容 AE 一直更新到 VLAD、条件生成器和物品概率，无需离线聚类。验证记录见 [VLAD 混合验证](docs/VLAD混合验证_20261106.md)。
