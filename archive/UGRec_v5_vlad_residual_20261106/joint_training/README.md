# 联合训练代码

模型、损失和 checkpoint 在这个包里。训练入口是上一级的 `scripts/train.sh`，默认读取 `configs/final.json`（VLAD K8/S1）；`configs/control.json` 提供不带 VLAD 的配对对照。
