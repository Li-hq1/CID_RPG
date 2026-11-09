# 在线AE + FMRec验收记录

## 核心要求

- 默认入口run_fmrec.py（run.py无配置时转入）、configs/final.json=fmrec.json、scripts/train.sh使用新入口。
- 物品表示在线来自AE encoder+无参数LN；没有可学习ID词表；训练目录实时编码，不detach、不使用冻结latent目录。
- 推荐CE/FM分别对AE encoder有非零梯度，历史/桥/干净目标/目录均活；关闭AE几何项后Adam确实更新AE参数。
- 官方Transformer计算对照通过，保留query行mask、无位置编码、随机token融合、mode时间、点积CE和30步固定z0 Euler。
- 历史重建MSE已实现且默认0，默认不构造decoder。

## 检查

- 86项测试通过：76项既有回归和10项新FMRec测试。旧默认配置回归转向显式configs/flowrec.json，未削减旧模型行为检查。
- CPU缩小smoke、三个方法各一次真实缓存完整宽度4更新/精确重载通过；这些run benchmark_eligible=false。
- 正式Beauty三组各40epoch/10280步/seed2024，status均complete，原生选模checkpoint精确重载。
- 4096 selection/full22363预测重算NDCG和Recall通过。512人诊断的原生指标与保存selection的相同前512人重算一致；checkpoint/user hash匹配。
- 三组共享AE/Transformer初态、数据和selection身份匹配；每份保存源码快照hash逐文件核验，实际运行模型与入口当前hash保持一致。
- 无test评估、无种子/超参/数据集扩展；MSE未启动，诊断未用于重新选模或改变主表30步设置。

运行源码hash：`5c164988a6fd0dc02d00cbc20f9bb336e96b414d5cc72782d9e8e28e0d93a356`，范围：`models/, joint_training/, compression/, misc/, run_fmrec.py, protocol.py`。源快照内README/设计文档后续补充不影响实际运行，区别记录在audit.postlaunch_nonruntime_snapshot_changes。

## 保护检查

原FlowRec源码65文件归档hash核验通过，已有旧输出保留；本地FMRec源码未变。保护436文件，观察到以下其他工作区变化，未回滚或重写起始保护清单；本轮独立v5_4不导入这些版本。完整前后SHA256见机器汇总audit.observed_workspace_changes。

| 变化文件 |
|---|
| UGRec_v5_1/AGENTS.md |
| UGRec_v5_1/README.md |
| UGRec_v5_1/docs/Beauty收敛_beauty_hstu3_rec_lr_half_ep80_s2024.md |
| UGRec_v5_1/docs/参数接口.md |
| UGRec_v5_1/docs/调参计划.md |
| UGRec_v5_1/docs/调参进度.md |
| UGRec_v5_1/run.py |
| UGRec_v5_6/joint_training/config.py |
| UGRec_v5_6/joint_training/config_parse.py |
| UGRec_v5_6/joint_training/model.py |
| UGRec_v5_6/models/hstu.py |
| UGRec_v5_6/run.py |

因此all_other_protected_files_unchanged=false，不能将其他版本全局保护误报为全部通过。本轮共享初始化、独立运行源码、数据、重载与结果审计均通过。

## 结论边界

首筛只训练40轮，默认官方上限500轮；未触发100轮LR衰减，FMRec最佳值在第40轮预算边界。报告仅判断当前单种子有限预算，未宣称已充分收敛或稳定优越。完整validation包含selection，桥上有目标的诊断不作部署指标。
