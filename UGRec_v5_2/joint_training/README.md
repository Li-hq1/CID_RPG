# v5_2 联合训练代码

`fm.py` 中的 `build_model` 按配置建立解耦CE/CFM模型或旧control；`training.py` 组合推荐目标与AE几何约束，`checkpoint.py` 保存两侧优化器及随机状态。入口为上一级 `scripts/train.sh`，默认读取 `configs/final.json`。机制和梯度边界见 `../docs/方案.md`。
