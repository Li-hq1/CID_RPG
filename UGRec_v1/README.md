# UGRec_v1：PCA 连续生成与多次兴趣采样

本版建立连续表示的生成式推荐基线：文本 embedding 经 PCA 压缩，Transformer 或 HSTU 编码历史，Rectified Flow 在条件下生成物品表示，多次独立采样后检索并聚合候选。

研究设计见 [方案](docs/方案.md)。[本版 checkpoint 索引](docs/checkpoints.json) 列出已迁入的 PCA 实验，实际文件在 `outputs/ckpt/`；原始模型目录名为 TigerDiff。

从本目录执行：

```bash
GPU_ID=0 CATEGORY=Beauty bash scripts/train.sh
GPU_ID=0 CATEGORY=Beauty ENCODER_TYPE=transformer bash scripts/train.sh
# 小样本流程检查
GPU_ID=0 RUN_ID=ugrec_v1_smoke bash scripts/train.sh --smoke-test
# 单次采样 + PCA 对照
RUN_ID=ugrec_v1_pca_knn bash scripts/train.sh --config=config_knn_pca.py
# 已有 checkpoint：使用与权重对应的保存配置
bash scripts/eval.sh \
  outputs/ckpt/AmazonReviews2014/TigerDiff/Beauty/tigerdiff_Beauty_qwen3_hstu_new-eval/best_tigerdiff_Beauty_qwen3_hstu_new-eval.pth \
  --config=docs/run_configs/tigerdiff_Beauty_qwen3_hstu_new-eval.json
bash scripts/test.sh
```

主配置 `config.py` 默认 Beauty、Qwen3 2560→128 PCA、HSTU、RF、多次采样；`config_transformer.py` 保留 Transformer 参数组，`config_hstu.py` 保留 HSTU 参数组，`config_knn_pca.py` 为单次采样对照。历史运行使用 `docs/run_configs/<run_id>.json`，CLI 参数优先于配置。不同类别可以通过 `CATEGORY` 或 `--category` 选择。

| 目录或文件 | 作用 |
| --- | --- |
| `train.py`、`engine/` | 构建实验、训练、验证、checkpoint、最终测试 |
| `models/tigerdiff.py`、`models/hstu.py` | 两种因果序列编码器与条件投影 |
| `models/diffusion/`、`models/evaluation_sampling.py` | RF 损失、ODE、候选检索聚合 |
| `data/` | 本版数据处理、文本特征、PCA、序列样本 |
| `misc/` | 参数解析、源身份、日志、随机种子 |
| `genrec/` | 原导入路径的本版兼容转发 |
| `outputs/history/mixed_evaluations/` | 曾混用 PCA/AE 的评估日志，仅作历史记录 |

`cache/` 链接到公共缓存；本版没有 AE 或 joint_training 算法依赖。旧 checkpoint 中的 embedding 来源与模型层数可能不同，请使用对应保存配置。内部 TigerDiff 名称与权重键保留，用于兼容已有实验。
