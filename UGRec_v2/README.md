# UGRec_v2：Geometry AE 两阶段推荐

本版把 v1 的 PCA 输入扩展为独立训练的 Geometry AE。先学习压缩空间、导出并冻结特征，再训练 Transformer/HSTU + RF；PCA 作为本版内的对照保留。

研究设计见 [方案](docs/方案.md)，历史实验见 [checkpoint 索引](docs/checkpoints.json)。AE 权重、诊断、导出特征保存在 `outputs/compression/<category>/Geometry_AE/`；推荐权重保存在 `outputs/ckpt/`。

从本目录执行：

```bash
# 训练独立 AE；脚本结束时输出 artifact 路径及推荐训练命令
GPU_ID=0 CATEGORY=Beauty bash scripts/train_geometry_ae.sh --relation_mode=batch
# 当前默认 Beauty AE artifact 已随整理迁入，可以直接训练推荐模型
GPU_ID=0 CATEGORY=Beauty bash scripts/train.sh
# 换 AE artifact 时显式指定路径；换类别也须使用该类别的 artifact
RUN_ID=ugrec_v2_ae_candidate bash scripts/train.sh \
  --geometry_ae_cache=outputs/compression/Beauty/Geometry_AE/d839190f3cb4f1774caf
# 本版 PCA 对照
RUN_ID=ugrec_v2_pca_control bash scripts/train.sh --sent_emb_compression=pca
# 评估已有六层模型，使用权重中的对应保存配置
bash scripts/eval.sh \
  outputs/ckpt/AmazonReviews2014/TigerDiff/Beauty/tigerdiff_Beauty_qwen3_hstu_geo-ae-local-relation_6l-new-eval/best_tigerdiff_Beauty_qwen3_hstu_geo-ae-local-relation_6l-new-eval.pth \
  --config=docs/run_configs/tigerdiff_Beauty_qwen3_hstu_geo-ae-local-relation_6l-new-eval.json
bash scripts/test.sh
```

主配置沿用整理前当前选择的 `aca737a768df334954ad` artifact（无关系蒸馏）。其他候选的 relation 设置以 artifact manifest 为准。AE CLI 当前默认 `relation_mode=none`；研究者观察 batch 关系约束更好，因此训练示例显式选 batch。更换候选使用独立 run ID；现有结果不改写。

压缩实现位于 `compression/geometry_ae/`：`model.py` 定义网络和损失，`pipeline.py` 训练/选模/导出，`cache.py` 验证源与 artifact 身份，`metrics.py` 与 `neighborhood.py` 诊断几何。`experiments.py` 保留压缩消融和推荐比较流程。其余推荐模块均为本版独立副本。

`cache/` 共用原始数据、文本 embedding 和预训练模型。旧缓存位置的 Geometry_AE 兼容链接仍可读取，但本版新 AE 默认写 `outputs/compression/`。推荐模型加载时校验源特征、训练物品、item 顺序和 artifact ID，不能将不同 AE 的权重与物品库混用。
