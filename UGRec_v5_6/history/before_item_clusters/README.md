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
