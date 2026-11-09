# UGRec_v5_1：按数据集独立调参版本

从固定的 UGRec_v5 control 独立复制运行代码，默认模型、训练目标和数据协议与 v5 一致。公共 `cache/` 和 `.venv` 复用，源码、配置、结果相互独立；本版不向旧版本目录写入。

请先读 [调参计划](docs/调参计划.md)、[指标遵循](docs/指标遵循.md)、[参数接口](docs/参数接口.md) 和 [Agent 执行规范](AGENTS.md)。当前配置与历史结果见 [调参进度](docs/调参进度.md)，工程验证见 [验收记录](docs/验收记录.md)。

当前规则：每次针对一个数据集，由用户在当次对话给出大致方向，例如“Beauty 调学习率和 loss 比例”或“Sports 调模型参数”，agent 据此自动选择有限候选并迭代。取消先统一调结构/容量、再调独立参数的阶段要求；各数据集可以采用不同结构和优化设置。默认范围仍为 Beauty、Sports、Toys，CDs 不自动恢复。

探索固定 seed2024，以目标数据集完整 validation NDCG@10 为主指标，保留匹配对照、机制解释和可溯源记录；不以跨数据集均值或旧 2% 联合护栏决定独立调参结果。最终方案冻结后才显式安排多种子，test 不用于调参。

当前可继承 `structure_hstu_depth_1_s2024` 的 Beauty/Sports/Toys 对应结果：HSTU 一层、Flow 两块、Sports LR=0.0005。以后分别维护每个数据集的配置和 parent/reference；历史四数据集 decision 保留原样。

单数据集 campaign 用 `scripts/tune.sh`，仍是 flow control。Optuna 筛选是另一条入口，当前准备的是 Beauty 上的直接 CE，再搜 HSTU、epoch 和推荐器学习率，见 [Optuna搜索](docs/Optuna搜索.md)。它按选模子集排序，不代替完整 validation 决策。

所有脚本以下列版本目录为基准，`scripts/*.sh` 也会自动切换到此目录：

```bash
cd /home/haoquan/CID_RPG/UGRec_v5_1
bash scripts/test.sh
bash scripts/tune.sh preflight --output docs/preflight.json
```

以上是工程检查入口，不代表用户已要求开始调参。`--device` 编号遵循当前 `CUDA_VISIBLE_DEVICES`，先检查资源再选择实际可用设备。默认 `prepare` 模式是 `check`，正式训练需显式指定 benchmark。

每个 campaign 在 `outputs/tuning/<id>/` 保存源码快照、运行清单、数据/源码/配置 SHA256、展开配置、训练日志、checkpoint、验证预测、汇总及决策。结果目录不可覆盖，复用测量须标明来源；失败的半成品保留，不伪装为完成。工程 check 和 smoke 不能作为 benchmark。

直接训练入口仍可使用，但手工运行结果不会自动成为符合 campaign 协议的调参记录。`frozen/matched/minimal/direct_reference.json` 和旧分支代码仅为回归测试及复现保留；默认研究路线仍为 control。
