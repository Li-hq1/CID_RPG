# FlowRec 三损失实验结果

2026-11-09，Beauty、seed2024、40 epoch、4040 updates、固定4096 validation selection，随后完整22363用户 validation。所有组都未读取 test，checkpoint 重载精确核验通过；这是单种子趋势，不是稳定性结论。

| 组 | 推荐目标 | 最佳轮次 | full NDCG@10 | full Recall@10 | full NLL |
|---|---|---:|---:|---:|---:|
| 原 control | 原边缘似然混合、8噪声查询 | 15 | 0.059470 | 0.109154 | 8.06249 |
| Prior-only | Prior CE，CFM=0，Align=0 | 15 | **0.061756** | **0.113357** | 8.02205 |
| FlowRec 三损失 | Prior CE + 5 CFM + Align CE | 35 | 0.033521 | 0.065465 | 8.40927 |
| 去 CFM | Prior CE + Align CE | 35 | 0.051564 | 0.093503 | 8.20061 |

三损失组同一最佳 checkpoint 的直接 Prior full NDCG@10 为0.035124，Flow生成仅0.033521；固定selection上分别为0.039337和0.037689。二者都低于 Prior-only。对三损失最佳checkpoint做积分诊断：1步/4步/8步 Flow NDCG@10 为0.037689/0.031443/0.031195，增加Euler步数没有改善。

去掉CFM后升至0.051564，但仍低于Prior-only和control，说明当前退化主要由CFM与一步生成的耦合造成；Align也没有带来额外收益。CFM训练项本身下降且桥上端点余弦接近1，不能据此推断部署推荐有效。

## 结论与下一步边界

本轮不采用FlowRec三损失作为v5_4默认推荐目标，也不启动低CFM权重或二维权重搜索。当前默认配置已保留用户要求的三损失实现和可复现实验底稿；研究决策上，Prior-only是本轮Beauty单种子最强简洁候选，原control是更接近当前v5路线的对照。若继续研究，应先解释“桥上端点拟合好但历史先验排序差”的机制，再设计一个能直接区分该假设的单项实验。

代码、配置、逐轮日志、checkpoint、全量预测、积分诊断和曲线在`outputs/flowrec/`；汇总为`outputs/flowrec/summary.json`，图为`flowrec_selection_curves.png/pdf`。运行前保护快照还发现三个v5_1文件有外部变化（`UGRec_v5_1/README.md`、`docs/调参进度.md`、`docs/指标遵循.md`）；本campaign未写这些文件，未回滚。
