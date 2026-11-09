# v5_4：在线 joint AE + FMRec

本轮按用户要求用本地 `../FMRec` 官方实现替换后续推荐模型：**保留内容AE encoder在线生成物品表示，并让CE、FM及可选历史MSE的梯度直通AE**。物品表示不是独立可学习ID词表，不使用冻结物品缓存训练。

FMRec Transformer按历史token融合随机时间桥状态，直接预测干净物品；目标为 `L_FM + .2*L_CE + beta*L_history_MSE + .1*L_AE`，beta默认0。CE是FMRec原始点积softmax，不沿用此前cosine/温度；推理使用官方30步固定源噪声Euler。

参考代码commit、逐项适配差异、梯度路径与最小对照见[实验设计](docs/FMRec实验设计.md)。原FlowRec源代码完整保存在`history/flowrec_before_fmrec/`，此前结果在`outputs/flowrec/`保留；`run.py --config configs/flowrec.json`仍可复现旧版。`run.py`无配置或指定FMRec配置时转入`run_fmrec.py`，训练脚本也使用FMRec新入口。

```bash
cd /home/haoquan/CID_RPG/UGRec_v5_4
bash scripts/test.sh
bash scripts/python.sh run_fmrec.py --device cpu --smoke-test --run-id fmrec_smoke_unique
GPU_ID=5 bash scripts/train.sh --run-id fmrec_joint_unique_s2024
# 默认配置是官方最大500轮、每20轮选模、Adam/StepLR；初筛显式改成40轮/每5轮：
bash scripts/python.sh research/fmrec_campaign.py --method fmrec --device cuda:5
```

另外两个必要对照使用`configs/fmrec_ce_only.json`与`configs/fmrec_history_ce.json`。本轮初筛只用Beauty/seed2024，不扫超参、种子或数据集，不评test。已有run id拒绝覆盖。代码只依赖本版本物理副本，不运行时导入参考仓库。

三组40轮初筛已完成：完整validation NDCG@10依次为FMRec **.00583907**、同桥CE-only **.00535987**、同骨干历史CE **.06256898**。FM监督相对同桥CE-only有+8.94%的单种子局部趋势，但整条生成机制尚未优于历史CE。FMRec最佳值在第40轮预算边界，不宣称充分收敛。桥上拟合与纯噪声部署存在明显差距，详见[实验结果](docs/FMRec实验结果.md)、[验收记录](docs/FMRec验收记录.md)及[机器汇总](outputs/fmrec/summary.json)。86项测试与三组正式重载/源码/身份/预测重算均通过。

用户授权的有限优化也已完成：三组补到140轮（第100轮后LR×.1），再仅新增历史MSE=.4候选、同样140轮。完整validation NDCG@10为FMRec **.00970673**、同桥CE-only **.01048854**、历史CE **.06179881**、FMRec+历史MSE **.00916878**。L-FM对同桥CE-only为−7.45%，整条FMRec对历史CE为−84.29%；MSE未救回整体收益。**建议结束当前FMRec适配的优化，优先保留joint AE + 历史直接CE主线**，不自动修改默认入口。历史CE同时删除桥/时间模块/多步采样，同桥CE-only才是仅删除L-FM。所有结果为Beauty/seed2024探索，无test或多种子，不能外推为全部FM方法无效。87项测试和四组源码/初态/预测/梯度审计通过。设计与结果见[优化实验](docs/FMRec优化实验.md)、[优化结果](docs/FMRec优化实验结果.md)、[机器汇总](outputs/fmrec/optimization/summary.json)；本轮训练已全部结束。
