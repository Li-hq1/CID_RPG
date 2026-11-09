# 在线 joint AE + FMRec：Beauty 初筛结果

## 已实现的方案

按用户要求，v5_4默认已切换为在线内容AE与FMRec。AE结构和逐物品LN保留，后续使用本地官方FMRec的token状态融合Transformer、干净端点预测、原点积CE及固定z0的30步Euler。**没有独立可学习的物品ID embedding表；历史、桥、FM目标和全目录CE都对AE encoder保持梯度。**

总目标为 `L_FM + .2*L_CE + beta*L_history_MSE + .1*L_AE`，历史MSE已实现，`fmrec.mse_weight=0`默认关闭且不构造对应decoder；设为.4可使用参考权重。本轮三组均关闭。精确定义和适配差异见[实验设计](FMRec实验设计.md)。

## 匹配比较

仅Beauty/seed2024，三组各40epoch、131413个训练前缀、batch512、10280次更新，固定4096验证用户每5轮选一个checkpoint，再完整22363用户validation审计。AE/Transformer共享初态、数据、样本和selection hash一致，无test、无多种子、无超参扫描。完整validation包含selection，非独立测试。

| 组 | 最佳epoch | selection NDCG@10 | full NDCG@10 | full Recall@10 | full NLL↓ |
|---|---:|---:|---:|---:|---:|
| 同桥 CE-only：.2CE | 35 | 0.00474435 | 0.00535987 | 0.01095560 | 27.14496 |
| FMRec：FM + .2CE | 40 | 0.00650511 | 0.00583907 | 0.01189465 | 26.60802 |
| 同骨干历史 CE：.2CE | 35 | 0.06062768 | 0.06256898 | 0.12037741 | 8.09634 |

必须区分两个问题：

1. **FM监督的局部增量**：FMRec相对同桥CE-only的full NDCG@10为 **+8.94%**（绝对约+.000479），selection为+37.11%。方向支持FM目标对这套桥模型有帮助，不能说FM损失完全无用；两组部署分数仍都很低，只是单种子趋势。
2. **生成机制的整体净收益**：FMRec仍比同Transformer历史CE低 **90.67%**，没有体现出值得采用的整体排序收益。该对照使用相同四块Transformer和相同前缀预算，不能将差距归为旧HSTU与Transformer结构不同。

FMRec最佳点在第40轮上界、后期有波动且仍缓慢改善。官方默认最大500轮、每20轮验证、100轮后LR衰减保留在configs/fmrec.json/final.json；本轮是40轮首筛，未触发LR衰减。**不能据此声称充分收敛、否定官方FMRec结果或证明更长训练必无效。** 这也不是替换AE之外完全逐项相同的官方指标复现；必要差异包括AE逐物品LN/几何项、padding不作为分类类别、本项目LOO和统一NDCG选模。

## 为什么FM/CE训练拟合好而部署弱

对各自已锁定checkpoint，在相同512名selection用户上作只读诊断，不训练、不重新选epoch，也不根据诊断把部署改成1步。含目标的桥只用于解释训练拟合，**不是可部署推荐成绩**。

| 组 | 含真实目标的t=.5桥 NDCG@10 | 桥上目标cosine | 纯噪声原生30步 NDCG@10 | 同checkpoint 1步 NDCG@10 |
|---|---:|---:|---:|---:|
| fmrec | 0.86180264 | 0.955563 | 0.00478789 | 0.02109203 |
| ce_only | 0.97490341 | 0.712318 | 0.00695626 | 0.02678879 |

同一512人上的历史CE为 0.06896125。FMRec桥上MSE仅0.09040、目标方向cosine约.956，然而纯噪声部署明显较差；1步读出虽好于30步，仍低于历史CE。结果支持训练桥与部署采样的脱节，不能把桥上高命中当作FM增益。这里尚未分离时间采样、状态分布、扰动强度与积分误差，不能把原因唯一归给其中一项。

AE梯度链路已用公式、分支梯度和实际优化器更新检查；FMRec初始真实batch上CE/AE encoder梯度范数81.4231、FM为1.95538，实际加权推荐梯度17.7048。不是只有AE重建项更新encoder。已选FMRec目录的坐标RMS约1、中心化坐标std约.763、平均不同物品方向cosine约.417，不支持“物品表示已完全坍缩为同一向量”的简单解释；这也不证明其几何质量已经足够。

## 工程与复现

86项测试通过，包括与本地官方端点网络的CPU逐位输出对照、CE/FM公式和梯度、前缀无未来信息、可选MSE、固定z0 Euler、确定性评估和纯推荐更新AE。三组完整宽度真实缓存4更新与CPU smoke均通过且不计正式指标。三组40轮正式运行均精确重载，保存预测重算NDCG/Recall一致；模型/训练入口源码快照一致且与当前运行文件相符。

参数量FMRec/同桥CE-only均3878656，历史CE为3746944；后者删除时间MLP。三组峰值训练显存分别约2722/2722/2706 MiB。正式运行约1052/945/909秒，含选模/完整验证，GPU共享负载及不同推理次数影响耗时，不当作纯训练吞吐比较。

原v5_4的FlowRec源码65文件在history/flowrec_before_fmrec/归档hash核验通过，原outputs/flowrec保留。本地FMRec参考仓库未改。其他版本保护清单发现工作区中的v5_1/v5_6文件变化，未回滚，前后hash完整记录在summary.audit；不能宣称所有其他版本字节未变。v5_4运行不跨版本导入，三组本轮运行身份独立核验通过。

- [自动汇总与完整audit](../outputs/fmrec/summary.json)
- [曲线与结果图](../outputs/fmrec/comparison.png)；[PDF](../outputs/fmrec/comparison.pdf)
- [验收记录](FMRec验收记录.md)
- 正式目录：`outputs/fmrec/runs/Beauty/beauty_{fmrec|ce_only|history_ce}_initial40_s2024/`，保存best.pt、config/experiment、metrics、gradient_audit、selection/full预测、bridge_diagnostic和源码快照。

当前结论是：实现与joint梯度要求已满足，FM监督有相对同桥CE-only的局部正向趋势，但FMRec生成路径在这次40轮初筛尚未优于匹配的历史CE。不自动开启MSE、调权重、修复官方mask或更换采样器，也不据此改掉本次FMRec实验入口。
