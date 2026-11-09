# Flow residual depth 2→1：共用容量消融

记录时间：2026-11-08 20:13 PST；原始清单、日志与审计使用 UTC。

**决策：reject。保留 Flow 两块。** 相对固定 reference `starting_point_sports_rec_lr_half_s2024`，四项平均相对 NDCG@10 为 **-0.3076%**；Sports **-2.003923%**，越过预注册 2% 护栏。等权平均也不为正，故拒绝不只依赖边界判定。这个边界判定不能解释为统计显著的劣势，也不能事后四舍五入通过。

## 问题与唯一改动

校准模型在选模峰值后仍降低训练排序损失，但验证表现下降；生成查询也高度相似。本轮检验减小速度网络容量是否能改善这一取舍。只将 `recommender.diffloss_d: 2→1`，四数据集都重新训练。AE、HSTU、过渡结构、Sports LR=0.0005 及其余专用参数均固定。

协议：seed2024、40 epoch、batch256、train/eval draws=8/32、固定4096名选模用户、完整目录与全量 validation audit。四数据集独立训练；不评 test。完整 validation 包含选模用户，属于调参数据。

## 完整验证结果

| 数据集 | NDCG@10 | 相对 reference | Recall@10 | NLL | 最佳 epoch | 更新数 | 训练+审计秒 | 峰值分配 MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Beauty | 0.06019375 | +1.2177% | 0.10870635 | 8.169909 | 20 | 4040 | 587.15 | 2416.96 |
| Sports | 0.03416251 | -2.0039% | 0.06475083 | 8.542281 | 20 | 5840 | 754.55 | 3372.65 |
| Toys | 0.05445065 | -1.4513% | 0.10946837 | 7.894495 | 10 | 3480 | 533.83 | 2393.85 |
| CDs | 0.05080188 | +1.0069% | 0.10065375 | 8.217888 | 20 | 44760 | 2302.82 | 1936.04 |

四项均为本轮新测量，无父结果复用。参数量 4,255,504 → 3,926,032，减少 329,472（7.7423%）。四任务在启动前空闲的 GPU 5 并行执行；逐项时长受同卡资源竞争影响，运行末段也观测到同卡其他用户的新任务（未干预），不作为严格速度对照。

## 曲线与机制解释

| 数据集 | reference 最佳轮→40轮选模变化 | candidate 最佳轮→40轮选模变化 | reference 最终 draw cosine | candidate 最终 draw cosine |
|---|---:|---:|---:|---:|
| Beauty | -12.221% | -6.412% | 0.92071 | 0.90536 |
| CDs | -4.923% | -6.353% | 0.95875 | 0.96608 |
| Sports | -10.946% | -10.148% | 0.88524 | 0.87738 |
| Toys | -7.744% | -5.632% | 0.92526 | 0.91378 |

Beauty/Sports/Toys 的后期选模下降有所缓解，但全量 NDCG@10 的方向不一致。减少容量、减少后期下降与提高所选模型的完整验证表现不能互相替代。CDs 的曲线与最终结果单独保留，不用于抵消 Sports 护栏。

`draw_cosine` 是训练中归一化生成查询的平均两两余弦；`mixture_gain` 是边缘似然相对逐噪声对数概率平均的差。它们可辅助描述趋同及噪声利用，不能单独定位容量瓶颈，也不证明有用多模态。对照的这两项在最佳轮后并非都单调变化，本轮不把结果解释为已经找到输出趋同的唯一原因。

![固定选模用户的学习曲线](/home/haoquan/CID_RPG/UGRec_v5_1/docs/flow_depth_1_selection_curves.png)

## 审计与执行溯源

- 运行前 84 项测试通过，源码 SHA256 `68e5474866bfbb1a833593b2aa2327bbbf6c4b616068a7ba6207a9d488533895` 与 parent/reference 一致。
- 官方 run 完成后 `complete=true`、`benchmark_eligible=true`、`missing=[]`，四项均 40 epoch，更新数与训练窗口匹配。
- 源码快照、当前源码、配置和四数据集指纹匹配。逐项比较 parent，唯一结构改动为 diffloss_d。
- 四项 checkpoint 重载预测一致；独立重算 selection/full 的 NDCG/Recall 与 results 一致；目标/选模用户索引匹配，top10 无重复及越界。NLL 使用正式全目录评估结果。
- 全量验证用户 22363/35598/19412/75258；未使用 test、未做多种子，单种子结果只作趋势。

[机器汇总](../outputs/tuning/structure_flow_depth_1_s2024/summary.json) · [决策](../outputs/tuning/structure_flow_depth_1_s2024/decision.json) · [完整审计](容量审计_structure_flow_depth_1_s2024.json) · [运行清单](../outputs/tuning/structure_flow_depth_1_s2024/manifest.json) · [并行命令与 PID](../outputs/tuning/structure_flow_depth_1_s2024/parallel_execution.json)

正式生成与完成核验命令：

```bash
bash scripts/tune.sh prepare --spec configs/tuning/flow_depth_1.json --mode benchmark --device cuda:5
# 四个 run.py 按清单配置并行执行，命令见 parallel_execution.json；持有 running.lock。
bash scripts/tune.sh run structure_flow_depth_1_s2024
bash scripts/tune.sh summarize structure_flow_depth_1_s2024
bash scripts/tune.sh decide structure_flow_depth_1_s2024 --outcome reject --evidence "已记录在 decision.json" --next-hypothesis "已记录在 decision.json"
```

`decide` 已执行且不可覆盖；以上最后一条仅记录操作，不应重复执行。原始失败目录、旧版本和公共 cache 保持原状。

## 下一项记录（写于本轮决策时；2026-11-09 已执行）

阶段3b下一轮仅检验 HSTU 层数 2→1：保持 Flow width256/depth2、AE latent128 和 norm_linear 等过渡结构，全部专用参数固定，parent/reference 均为 starting_point_sports_rec_lr_half_s2024。Flow 2→1 缓解部分数据集的后期选模下降，但未形成可通过护栏的共同收益，不能把 Flow 容量过大视为主要瓶颈。校准对照与本轮在达到选模峰值后仍出现训练排序损失下降、选模 NDCG 下降，下一项用较浅 HSTU 单独检验序列条件路径的容量与泛化取舍，不把本轮已拒绝的 Flow depth1 与 HSTU depth1 叠加。四数据集都新训练，seed2024、40 epoch 及完整 audit，仍对同一 reference 使用 mean(r_d)>0 和单项2%护栏。这是阶段3b第二个有限容量候选，完成后检查信息收益，不自动枚举 AE/宽度/层数组合。模板 configs/tuning/hstu_depth_1.json；本轮仅记录，不prepare/run下一项。

[模板](../configs/tuning/hstu_depth_1.json)。该模板已于 2026-11-09 执行并拒绝，见 [调参进度](调参进度.md) 与 [HSTU深度消融](HSTU深度消融_structure_hstu_depth_1_s2024.md)。阶段3b两次深度邻域都未通过护栏，容量方向停止，不自动扩大搜索。
