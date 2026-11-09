# v5_6：多 embedding 后续探索

本轮在直接 CE 底座完成尺度匹配、AE 内 raw VLAD 分簇求和、同 AE 提前合并对照、可学习 concat-linear 聚合，新增 5 组 Beauty/seed2024/40epoch 实验。详细指标、机制诊断与取舍见[后续结果](docs/item_cluster_followup/结果.md)、[设计](docs/item_cluster_followup/设计.md)、[验收](docs/item_cluster_followup/验收.md)，机器审计在 `outputs/item_cluster_followup/summary.json`。

模块支持 `recommender.item_vlad_scale=unit|sqrt_dim`；AE 内分簇通过 `ae.cluster_count`、`ae.cluster_aggregation=sum|linear` 和 `recommender.ae_cluster_history=multi|merged` 配置。decoder/catalog 共用 raw 聚合 latent，历史逐簇共享原 normalizer/input_proj；同 item 共享原 RoPE、block causal mask，HSTU 后仍求和。新旧入口互斥且兼容旧 checkpoint。完整定义见上述报告。

本轮继续保持单种子探索，只用固定 4096 人 selection 选模、完整 validation 审计，不评 test。原默认配置与下面两轮历史结果保留。新实验配置在 `configs/item_cluster_followup/`，已有 run 拒绝覆盖。

```bash
cd /home/haoquan/CID_RPG/UGRec_v5_6
bash scripts/test.sh
# 复现前配置新的run id/输出目录；已有状态和结果不会被覆盖。
bash scripts/python.sh research/item_cluster_followup.py --stage preflight --method scaled_k1 --device cuda:4
bash scripts/python.sh research/item_cluster_followup.py --stage train --method scaled_k1 --device cuda:4
```

---

# v5_6：逐 item VLAD 多 embedding 探索

当前新探索基于本版直接 CE：`AE单向量 → 原input_proj → 逐item NeXtVLAD的K个128维簇 → 原HSTU → 同item的K个输出直接sum → 原投影/输出头`。没有concat映射、簇ID或新损失。catalog仍是AE单向量。

同item共享原RoPE位置，按item做block causal mask，当前item内全部簇可互相注意，未来不可见。注意力分母随实际token长度变为T*K。实现拆在 `models/item_vlad.py`，通过 `recommender.item_vlad_clusters` 开关；0为旧基线。G=expansion=2维持HSTU宽度不变。

首轮Beauty三组baseline/K1/K4已完成，固定seed2024/40epoch。完整validation NDCG@10分别0.06056862/0.05896396/0.05823801；K4较基线−3.85%，未见多token排序收益。簇学出了方向差异，不能简单归因于完全坍缩。见[结果](docs/item_clusters/结果.md)、[设计](docs/item_clusters/设计.md)及[验收](docs/item_clusters/验收.md)。输出在 `outputs/item_clusters/`，上一轮结果保留。历史源码冻结在 `history/before_item_clusters/`。

```bash
cd /home/haoquan/CID_RPG/UGRec_v5_6
bash scripts/test.sh
bash scripts/python.sh research/item_cluster_campaign.py --stage preflight --device cuda:4
bash scripts/python.sh research/item_cluster_campaign.py --stage train --method all --device cuda:4
# 或单独运行一个配置（新run id；已有目录禁止覆盖）
bash scripts/python.sh run.py --config configs/item_clusters/k4.json --device cuda:4
```

默认 final/control 配置仍保留上一轮；新探索用显式 item_clusters 配置。K1是新增变换对照，K4是多token假设，不能将单种子结果称为稳定收益。

---

# v5_6：原control是否需要FM分支

从原v5 control物理复制，唯一新增对照是HSTU直接预测CE：保留原内容AE、两层HSTU、条件投影与输出线性头，删除生成网络和噪声/混合解码。默认final仍保留control用于对照；direct入口为configs/direct_ce.json。

原control目前已经没有显式FM速度MSE。本轮所称FM分支指其一步速度网络、历史+噪声源和概率混合整条生成路径，比较的是这条路径是否比直接CE值得保留。

精确定义与固定比较协议见[实验设计](docs/实验设计.md)。本轮只做Beauty/Sports/Toys的两组单种子消融，不调参或评test。

六组已完成。重新训练的直接 CE 相对原 control，完整 validation NDCG@10 在 Beauty/Sports/Toys 分别 +1.86%/+4.96%/+3.69%，Recall@10 也均提高，参数减少24.00%、峰值训练显存减少约64%–66%。Beauty selection 近乎持平且略低；Sports 概率NLL仍是control更好。按推荐排序与成本，目前不支持保留生成分支，建议直接 CE 作为后续基线。旧control硬绕过FM会下降，简化应重新训练。只有seed2024，未做多种子、未评test，原v5默认未改。

完整说明见[实验结果](docs/实验结果.md)、[验收记录](docs/验收记录.md)及[机器汇总](outputs/fm_ablation/summary.json)。

```bash
cd /home/haoquan/CID_RPG/UGRec_v5_6
bash scripts/test.sh
bash scripts/python.sh research/campaign.py --category Beauty --device cuda:3
```
