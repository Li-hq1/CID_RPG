# HSTU 层数 2→1：共用容量消融

记录时间：2026-11-09 13:05 CST；原始清单、日志与审计使用 UTC。

**决策：reject。保留 HSTU 两层。** 相对固定 reference `starting_point_sports_rec_lr_half_s2024`，四项平均相对 NDCG@10 为 **+0.2859%**，但 CDs **-4.9346%**，越过预注册 2% 护栏。平均略正不能抵消护栏。这是单种子趋势，不是统计显著性结论。

## 问题与唯一改动

Flow residual depth 2→1 没有形成可通过护栏的共用收益。本轮只把 `recommender.n_layer: 2→1`，检验较浅的序列条件容量能否改善泛化并保持四数据集推荐表现。AE latent 128、Flow width 256 / 两块、`norm_linear` 过渡、Sports LR=0.0005 及其余专用参数都固定。不与已拒绝的 Flow depth 1 叠加。

协议：seed2024、40 epoch、batch 256、train/eval draws=8/32、固定 4096 名选模用户、完整目录与全量 validation audit。四数据集独立训练；不评 test。完整 validation 包含选模用户，属于调参数据。

## 完整验证结果

| 数据集 | NDCG@10 | 相对 reference | Recall@10 | NLL | 最佳epoch | 更新数 | 训练+审计秒 | 峰值分配 MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Beauty | 0.06049500 | +1.7242% | 0.11098690 | 8.033700 | 15 | 4040 | 791.39 | 2383.21 |
| Sports | 0.03574440 | +2.5338% | 0.06831844 | 8.422580 | 15 | 5840 | 1043.31 | 3338.65 |
| Toys | 0.05625818 | +1.8201% | 0.10910777 | 8.088547 | 15 | 3480 | 715.12 | 2362.55 |
| CDs | 0.04781356 | -4.9346% | 0.09317282 | 8.458233 | 40 | 44760 | 2597.95 | 1903.68 |

四项均为本轮新测量。参数量 4,255,504 → 4,140,040，减少 115,464（2.7133%），对应去掉一层 HSTU。Beauty/Sports/Toys 的 Recall@10 更高、NLL 更低，CDs 两项都变差；这些不能替代 NDCG@10，也不能把前三项的涨幅拿来覆盖 CDs。

四任务在启动前较空的 GPU 5 上并行。训练期间同卡另有其他用户进程，进程号中途变化，未加干预。逐项时长受同卡竞争影响，不作为独占速度对照。

## 曲线与机制解释

| 数据集 | reference 最佳轮→40轮选模变化 | candidate 最佳轮→40轮选模变化 | reference 最终 draw cosine | candidate 最终 draw cosine |
|---|---:|---:|---:|---:|
| Beauty | -12.221% | -10.491% | 0.92071 | 0.91856 |
| Sports | -10.946% | -6.124% | 0.88524 | 0.88267 |
| Toys | -7.744% | -10.238% | 0.92526 | 0.92491 |
| CDs | -4.923% | 0.000% | 0.95875 | 0.96116 |

Beauty/Sports/Toys 的选模峰值仍在第 15 轮，之后选模 NDCG 下降，训练排序损失继续下降。浅 HSTU 没有去掉这个后期分化；Sports 的后期降幅收窄，Toys 的后期降幅变大。最终 `draw_cosine` 仍在 0.88–0.96，与对照同量级，不能把结果解释为已经解开生成查询趋同。

CDs 的选模分数在第 25 轮 0.052072、第 40 轮 0.052106，末轮只以很小差值成为最佳，曲线在后半段平台波动，不是明显未训完。该选模峰值高于对照的选模峰值，但全量 NDCG@10 仍低 4.93%。选模子集不能代替全量比较，也不能据此加长 40 epoch 预算或改成更深 HSTU。

![固定选模用户的学习曲线](/home/haoquan/CID_RPG/UGRec_v5_1/docs/hstu_depth_1_selection_curves.png)

## 阶段3b信息收益

两次邻域都只减深度，都未通过护栏，而且失败落在不同数据集：Flow 2→1 的 Sports 为 -2.0039%，HSTU 2→1 的 CDs 为 -4.9346%。浅 HSTU 使三个较小目录的全量 NDCG 上升，所以没有证据把“再加深 HSTU”当作共用下一步；Flow 2→1 也没有给出加深 Flow 的证据。宽度和 AE latent 这一轮没有单独的压缩或表示瓶颈诊断。停止继续枚举深度、宽度和 AE 容量，也不把两个已拒绝的浅层叠在一起。

## 审计与执行溯源

- 运行前 84 项测试通过。源码 SHA256 `68e5474866bfbb1a833593b2aa2327bbbf6c4b616068a7ba6207a9d488533895` 与 parent/reference 一致，快照匹配。
- 官方 `run` 跳过四项已完成结果后 `complete=true`、`benchmark_eligible=true`、`missing=[]`。40 epoch，更新数 4040/5840/3480/44760。
- 相对 parent 的唯一配置差是 `recommender.n_layer: 2→1`。Flow 仍为两块，Sports LR 仍为 0.0005。
- 四项 checkpoint 重载标志为真；独立重算 selection/full 的 NDCG/Recall 与 results 一致；top10 无重复。全量用户 22363/35598/19412/75258。
- 未使用 test，未做多种子。

[机器汇总](../outputs/tuning/structure_hstu_depth_1_s2024/summary.json) · [决策](../outputs/tuning/structure_hstu_depth_1_s2024/decision.json) · [完整审计](容量审计_structure_hstu_depth_1_s2024.json) · [运行清单](../outputs/tuning/structure_hstu_depth_1_s2024/manifest.json) · [并行命令与 PID](../outputs/tuning/structure_hstu_depth_1_s2024/parallel_execution.json)

```bash
bash scripts/tune.sh prepare --spec configs/tuning/hstu_depth_1.json --mode benchmark --device cuda:5
# 四个 run.py 按清单配置并行执行，命令见 parallel_execution.json；持有 running.lock。
bash scripts/tune.sh run structure_hstu_depth_1_s2024
bash scripts/tune.sh summarize structure_hstu_depth_1_s2024
bash scripts/tune.sh decide structure_hstu_depth_1_s2024 --outcome reject --evidence "已记录在 decision.json" --next-hypothesis "已记录在 decision.json"
```

`decide` 已执行且不可覆盖；以上最后一条仅记录操作，不应重复执行。

## 下一步

阶段3b容量邻域到此停止。当前采用配置仍是 `starting_point_sports_rec_lr_half_s2024`。2026-11-09 用户确认不进入阶段4，不 prepare 新 campaign。
