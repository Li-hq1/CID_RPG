# v5_6 原control生成分支消融

本轮只回答用户指定的问题：原control的生成分支是否比HSTU直接CE预测有正收益。先读README和docs/实验设计.md，继承全局科研规则。

- 每个版本物理独立，旧版源码/配置/结果不改，共享cache只读。
- 只做原control和direct CE两组，在Beauty/Sports/Toys各seed2024、40epoch。原control两层HSTU、原LR=.001，不混入v5_1已调参结构。
- 不加新损失/模块，不做超参搜索、种子扫描或test。只用固定4096用户selection选checkpoint，完整validation审计，不能将单种子趋势称为稳定优势。
- Direct CE的模型、优化器和checkpoint必须实际不含FM参数，不能把未调用但仍保留的模块当作已删除；共享AE/HSTU/投影/输出头初值必须匹配。
- 工程测试与真实缓存短程检查通过后正式运行；每次新目录保存源码/初态/数据/选模用户hash、结果与预测，禁止覆盖。GPU按空余资源使用，不中断其他进程。
- 按数据集分别报告原control相对direct CE的收益与代价，不用平均掩盖数据集差异。本轮只判断，不原地删除v5默认实现。
