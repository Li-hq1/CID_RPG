# 联合训练代码

模型、损失和 checkpoint 在这个包里。训练入口是上一级的 `scripts/train.sh`，默认读取 `configs/final.json`（已恢复为 control，与 `configs/control.json` 完全一致）。VLAD 实验通过 `configs/vlad_condition_20261106/` 下的独立配置复现。
