# v5_4 训练栈

`flowrec.py`提供Prior CE、CFM、Align CE，`flow_ranking.py`按path分派，`training.py`在同一更新中加入原有0.1倍内容AE约束。默认配置见`../configs/final.json`；`control.json`保留本版原边缘似然对照。机制与实验定义见`../docs/FlowRec实验设计.md`。

推荐入口是`../run.py`和`../scripts/train.sh`。每次运行保存配置、源码快照、初始化及数据指纹，使用固定验证子集选模，然后检查checkpoint重载并做完整验证。Test不参与训练、调参或评价。
