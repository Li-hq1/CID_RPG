# 在线joint AE + FMRec：有限优化结果

本轮按用户授权先补足训练预算，再只尝试一个历史约束候选；保留joint training与推荐梯度直通AE encoder。
Beauty / seed2024；各组最大140轮、35980更新；第100轮后推荐器/AE学习率从1e-3/1e-4降为1e-4/1e-5。
固定4096名selection用户按NDCG@10选checkpoint，完整22363人validation仅审计且包含selection；未访问test、未做多种子或参数网格。

## 对照到底删除了什么

| 方法 | 历史encoder | 含目标随机桥与时间模块 | L-FM | 历史MSE | 推理 |
|---|---|---|---|---|---|
| FMRec | 在线joint AE + 共享Transformer | 保留 | 1 | 0 | 30步原生采样 |
| 同桥CE-only | 同上 | 保留 | 0 | 0 | 同样30步 |
| 历史CE | 在线joint AE + 相同Transformer块 | 删除 | 0 | 0 | 历史直接预测一次 |
| FMRec + 历史MSE | 同FMRec | 保留 | 1 | .4 | 30步原生采样 |

各组CE权重均.2、AE几何项权重均.1，均无独立可学习ID物品词表。历史CE不是只去掉L-FM。
FMRec对同桥CE-only衡量FM监督局部作用；FMRec对历史CE衡量整条生成机制的净收益。

## 正式结果

| 方法 | 最佳轮 | Selection NDCG10 | Full NDCG10 | Full Recall10 | Full NLL |
|---|---:|---:|---:|---:|---:|
| FMRec | 110 | 0.01103554 | 0.00970673 | 0.01949649 | 23.9460 |
| Bridge CE-only | 125 | 0.01278696 | 0.01048854 | 0.02271609 | 18.4876 |
| History CE | 65 | 0.06359074 | 0.06179881 | 0.11724724 | 8.5226 |
| FMRec + history MSE | 125 | 0.01117729 | 0.00916878 | 0.02016724 | 24.8591 |

FMRec完整validation从40轮初筛的0.00583907升至0.00970673（+66.24%），说明早期预算确实低估了其表现。
但与等预算历史CE相比，完整NDCG@10仍为-84.29%。
FMRec相对同桥CE-only：selection -13.70%，完整validation -7.45%。这只是单种子探索，不表述为稳定增益。
历史MSE候选相对FMRec：selection +1.28%，完整validation -5.54%；相对历史CE完整validation -85.16%。它新增了历史约束，不能用其结果直接归因L-FM。
历史MSE分支可训练但参考归约方式使其影响很弱：训练后固定batch上AE encoder的原始CE/FM/history-MSE梯度范数分别为4.90593/0.188666/0.000173314（对应权重.2/1/.4）。非零梯度不等于有效约束强度；本轮只检验参考权重.4，不据失败否定所有历史约束，也不追加重归一化/权重扫描。

## 锁定checkpoint诊断

以下为相同512名selection用户的次级诊断，未参与checkpoint选择。含真实目标的桥不是可部署推荐结果。

| 方法 | 含目标t=.5桥 NDCG10 | 纯噪声1步 | 原生推理 | 原生推理打乱历史 |
|---|---:|---:|---:|---:|
| FMRec | 0.934582 | 0.034619 | 0.014882 | 0.000616 |
| Bridge CE-only | 0.987025 | 0.042687 | 0.018337 | 0.001953 |
| History CE | — | — | 0.066587 | 0.000616 |
| FMRec + history MSE | 0.936305 | 0.039526 | 0.016280 | 0.002552 |

桥恢复、噪声起点和多步生成的能力必须区分。若1步优于30步，只能说明当前checkpoint的多步生成使排序退化，不能唯一归为Euler离散误差，也不能据此静默换推理方式。
高t桥在打乱历史后仍可高分，反映目标状态本身可提供强信息；低t及原生推理受打乱历史影响，说明模型并非完全忽略历史。
训练后checkpoint也逐项检查CE/FM及开启时的MSE对AE encoder的梯度非零，不执行优化器更新。原生预测与保存结果逐条一致。

## 复现与边界

三组原模型从原best.pt完整恢复优化器、调度器、CPU/CUDA RNG、DataLoader与AE抽样generator。FMRec从40轮、另两组从35轮恢复；后两组重放36—40轮的训练及验证结果与原记录一致。
历史MSE候选从同一seed重新训练，共同AE和Transformer参数初态匹配；decoder新增随机数消耗，不声称后续噪声逐步相同。
87项测试通过，包括续训下一轮参数精确复现；完整宽度真实数据MSE 4步检查通过但不计benchmark。本次汇总的4组运行源码、数据、selection身份、初态、预测指标重算及checkpoint重载已审计。
原40轮checkpoint/结果、FlowRec源码归档与FMRec参考仓库hash未变；本轮没有修改其他版本代码，不声称并发工作区整体无变化。
140轮覆盖一次学习率衰减，并不等于穷尽原参考最大500轮训练或证明所有FM方法无效。本轮判断服务于当前方法取舍。

## 本轮决定

延长预算与唯一的历史MSE候选仍未给出相对历史CE的整体净收益，建议结束当前FMRec适配的优化，主线优先保留在线joint AE + 历史直接CE。
不自动修改默认配置；保留全部实现和独立结果供复查。

- [机器汇总与audit](../outputs/fmrec/optimization/summary.json)
- [阶段决定](../outputs/fmrec/optimization/decision.json)
- [曲线PNG](../outputs/fmrec/optimization/comparison.png) / [PDF](../outputs/fmrec/optimization/comparison.pdf)
- [预先固定的实验设计](FMRec优化实验.md)
