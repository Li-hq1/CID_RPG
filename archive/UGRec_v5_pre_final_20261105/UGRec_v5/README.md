# UGRec v5：冻结的原版方案

本目录是后续实验的起点，只保留冻结方案和它直接用到的模型代码。PCA 训练、独立 Geometry AE 训练、Optuna 和旧联合入口仍在 v1–v4，没有重复放进来。

模型、损失、数据和选模规则见 [方案](docs/方案.md)。

```bash
cd /home/haoquan/CID_RPG/UGRec_v5

# Beauty、seed 2024、40 epoch。每 5 轮用固定 4096 个验证用户选模，最后评估全部验证用户。
GPU_ID=0 bash scripts/train.sh

# 换数据集或种子时使用新的 run id，已有目录会拒绝覆盖。
GPU_ID=0 bash scripts/train.sh --category Sports_and_Outdoors --seed 2025 --run-id ugrec_v5_sports_s2025

# 只检查数据、损失和保存能否走通，不代表正式指标。
bash scripts/train.sh --smoke-test --device cpu

bash scripts/test.sh
```

默认配置是 `configs/frozen.json`。`configs/minimal.json` 是同一方案的简写，展开后与冻结配置一致。`configs/direct_reference.json` 是不走整流流的直接预测对照。`configs/matched.json` 把训练收成和推理相同的一次运输，不再另算速度回归；它不会被默认训练脚本使用。
