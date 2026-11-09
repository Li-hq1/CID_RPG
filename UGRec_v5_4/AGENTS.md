# UGRec_v5_4 执行约定

先读 `README.md` 和 `docs/方案.md`。继承用户全局科研规则。

- 用户当前明确授权将本目录改为FMRec模型/训练尝试。默认入口run_fmrec.py、configs/final.json；必须保留在线joint AE encoder与推荐损失直通梯度，不用独立ID物品词表替代。旧FlowRec源码在history/flowrec_before_fmrec，旧outputs/flowrec只读保留。
- 不修改 `UGRec_v5`、`UGRec_v5_1`、`UGRec_v5_2`、`UGRec_v5_3` 的源码、配置和结果。公共 cache 只读，不跨版本导入。
- 不带回 v5_1 的调参 campaign，也不把 v5_1 的已有分数当作本版结果。
- 探索只用 seed2024。不自动开多种子、数据集扫描或参数网格。正式训练前先写变种定义和对照。
- 工程检查和 smoke 不能当 benchmark。已有 run id 不覆盖。使用 GPU 前先看占用，不中断其他任务。
- FMRec核心逐项对照本地../FMRec代码，保留Transformer融合、原点积CE、干净端点FM、官方固定z0 Euler；历史重建MSE默认0，不构造对应decoder，支持显式开启。
- 当前只跑Beauty三组seed2024初筛：FMRec、同桥CE-only、同Transformer历史CE；40epoch/固定4096选模/完整validation、无test。官方默认500epoch与本轮40epoch初筛必须区分，不把早期趋势称为充分收敛结论。
- 用户后续已授权有限优化，按docs/FMRec优化实验.md执行：三组同条件补到140轮覆盖第100轮衰减；若FMRec仍未超过历史CE，仅再尝试历史MSE=.4的同预算候选。仍只Beauty/seed2024、无test，不扩参数网格，不自动替换默认入口。
- 上述有限优化已全部结束，结论见docs/FMRec优化实验结果.md与outputs/fmrec/optimization/decision.json：未见整体净收益，停止当前FMRec适配的自动优化；不因后续读取报告再次启动MSE或延长训练。默认入口保留，后续新训练需用户新授权。
