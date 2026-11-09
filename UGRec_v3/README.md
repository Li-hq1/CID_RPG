# UGRec_v3：联合训练与参数搜索

本版自带 Geometry AE、Transformer/HSTU 和 Rectified Flow，在同一训练过程中交替更新表示与推荐模型，可选 InfoNCE 仅更新 AE，并用 Optuna 搜索参数。

研究设计、梯度连接和搜索边界见 [方案](docs/方案.md)，详细 CLI 见 [联合训练说明](joint_training/README.md)，已有实验归属见 [checkpoint 索引](docs/checkpoints.json)。

从本目录执行：

```bash
GPU_ID=0 CATEGORY=Beauty bash scripts/train_joint.sh
GPU_ID=0 CATEGORY=Beauty bash scripts/train_joint_contrastive.sh
# 小样本、少步数流程检查，使用独立 run ID
GPU_ID=0 RUN_ID=ugrec_v3_smoke bash scripts/train_joint.sh --smoke-test
# 本机当前实现的独立搜索，使用新 study 名避免混用远端旧 study
GPU_ID=0 STUDY=ugrec_v3_Beauty_contrastive_search bash scripts/search_joint_contrastive.sh
# 评估或恢复当前兼容格式的已有联合模型
bash scripts/python.sh -m joint_training --eval-only \
  outputs/joint_training/Beauty/joint_Beauty_contrastive_ae_only_w0.01/best.pt --device=cuda:0
bash scripts/python.sh -m joint_training --resume \
  outputs/joint_training/Beauty/joint_Beauty_contrastive_ae_only_w0.01/latest.pt --device=cuda:0
bash scripts/test.sh
```

联合训练默认配置在 `joint_training/config.py`，搜索空间在 `joint_training/search_space.json`。通过 `--set section.field=value` 覆盖，脚本保留参数转发。主输出为 `outputs/joint_training/<category>/<run_id>/`；study 目录保存数据库、搜索配置和汇总，每个 trial 有独立 checkpoint 与日志。

`train.py`、`config.py`、`scripts/train.sh` 是本版自带的 PCA 冻结表示对照；联合训练使用独立配置，不会读取该平面 CONFIG。Geometry AE 代码也保存在本版内，联合模型从本版导入，不依赖 v2。

旧 bridge、fixed/EMA normalization、双向对比更新及其他已移除结构的实验保存在 `outputs/history/joint_training/`，不宣称当前入口可以恢复。当前清单中 `current_schema` 仅表示能够解析并通过本次加载检查，正式恢复仍检查源、模型、优化和评估协议。原 checkpoint 与日志未改写。

当前搜索仍以测试指标排序已完成 trial；验证集用于选 best checkpoint 与剪枝。续搜的 sampler/pruner 恢复问题也保留，详见方案。此次结构整理没有重新进行完整搜索。远端历史研究记录见 [历史环境与搜索观察](docs/history/Memory-before-ugrec.md)；本次没有同步或移动远端 study。
