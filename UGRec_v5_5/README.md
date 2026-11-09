# UGRec_v5_5：干净端点 FM + CE

从 v5 control 物理复制模型与数据代码，用同一个 `f(x_s,s,h)` 直接预测物品向量。新目标为 `CE + lambda * mean_coordinate_MSE`，CE 和 MSE 共享一次前向及同一个目标物品。本版保留内容AE、两层HSTU、历史锚点加噪声、cosine/T目录打分和原训练协议。

默认 `configs/final.json` 仍为 control；`configs/clean_endpoint.json` 为新方案的明确入口。方案与实验约束见 [实验设计](docs/实验设计.md)。所有实验写入 `outputs/clean_endpoint/`。

```bash
cd /home/haoquan/CID_RPG/UGRec_v5_5
bash scripts/test.sh
bash scripts/python.sh research/campaign.py --stage initial --device cuda:4
```

这是损失与输出参数化的适配，不是完整 FMRec 复现。探索是单种子趋势，test 和多种子均不自动启动。

本轮六组已完成：control完整validation NDCG@10 .059463；桥CE-only .057038；桥CE+FM（lambda=.1/1/5）为 .056027/.056990/.055168；源端CE-only .059247。没有匹配CE增量，默认保留control。完整结果与边界见 [实验结果](docs/实验结果.md)，工程检查见 [验收记录](docs/验收记录.md)。
