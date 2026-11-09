# 在线 joint AE 与 FMRec：最小验证

## 问题与核心假设

此前自建条件一步流及FM辅助目标没有显示相对直接CE的排序净收益。此次改用用户下载的FMRec模型和训练机制，检验其“把随机时间状态逐token融合到历史、直接预测干净端点、同输出FM+CE”能否在**在线内容表示**上产生增量。核心保留项是 `e_i = LN(AE_encoder(content_i))`，推荐梯度必须端到端更新AE，而不是换回可学习ID embedding表。

只改变embedding来源而尽量保留FMRec下游机制；AE几何约束和joint更新同时保留。旧FlowRec、自建HSTU速度网络和cosine温度目标不参与新默认路径。

## 依据与忠实性

本地参考 `FMRec/src/fmrec.py`、`model.py`、`trainer.py`、`utils.py`、`main.py`，git commit **1c39b17e59a08980c8372e4f861167e0a259c349**。运行不跨目录导入；测试将本地原始模型类作为计算对照，只移除未使用的第三方import。

保留参考：

- 128维、4块、4头Transformer，FFN宽度4倍，pre-LN残差，原tanh-GELU与LayerNorm公式；没有实际使用position embedding，完整前缀内非因果注意力。
- embedding dropout=.3，再通过可学习LayerNorm；Transformer dropout=.1；每token每坐标系数 `lambda ~ Normal(.001,.001)`，融合 `history + lambda*(x_t+time_embedding)`。
- `z0~N(0,I)`、mode时间采样s=1、eps=.001；`x_t=t*e_y+(1-t)*z0`，时间MLP输入的正弦时间为`t*(1/eps-1)`；同一个输出f直接预测e_y。
- `L_FM=mean_coordinate_MSE(f,e_y)`，`L_CE=CE(f @ E.T,y)`；**原代码是FM权重1、CE权重.2**，不是`CE+.2FM`，不沿用cosine/T=.07。
- 原Euler30步，`t_i=eps_reverse+i/N*(1-eps_reverse)`，`x←x+(f(history,x,t)-z0)/N`，减的是初始z0。没有改成常见的`(f-x)/(1-t)`。随机融合系数在eval仍抽样，固定评估随机种子与batch，验证不消耗训练RNG。
- Adam、推荐lr=.001、weight_decay0、StepLR每100epoch乘.1、batch512。AE使用自己的lr=.0001与weight_decay=.0001参数组，且在同一次反传中更新。
- 按每个用户的所有训练前缀展开样本，长度最多50、左填充，每前缀只预测下一个物品。Beauty131413个前缀；每轮257步，40轮10280步。不能拿旧版4040步实验冒充等预算对照。

原仓库attention mask实际遮挡的是query行，而不是padding key列。本轮明确保留这一行为，不静默修复；因此padding也参与上下文的计算。每个样本预先切成完整可见前缀，未出现未来token，非因果注意力不会看到未来标签。该mask习惯是否合理属于后续独立问题，不能和本轮模型移植同时改动。

AE配置复用了原几何模块schema，其中`ae.scheduler`、`warmup_ratio`、`update_every`、`max_grad_norm`属于旧训练器字段，不决定本入口的优化：本入口每batch联合反传/Adam更新，AE与推荐两个参数组共同遵循`training.decay_step/gamma`，不另加cosine、warmup或梯度裁剪。默认update_every为1，实际行为以本段与FMRecTrainer为准。

必要适配与区别：

- 删除独立ID物品词表，用现有AE结构2560→512→256→128及逐物品无参数LN供给历史/桥目标/CE目录，保留现有`.1*L_AE`几何约束。padding0置零并从CE物品类别排除；官方原表含可学习padding行且分类含0，本轮没有为padding构造伪内容。这些差异透明记录，不宣称官方数据/指标复现。
- 物品目录训练每batch从AE实时计算，**无detach、无训练缓存**；history dropout/LN独立于作为目录与FM目标的物品表示，符合官方路径结构。AE encoder在历史上下文、桥状态、干净目标、全目录分类四个位置均保持梯度。
- GPU随机融合在目标设备抽样，避免每ODE步CPU分配/传输；分布相同，不声称与官方CPU随机流逐位一致。CPU同权重、同随机数下endpoint输出应逐位一致。
- 去掉官方未使用的线性层、position embedding等死参数；历史decoder只在beta>0时构造。它们不参与主前向；移除会改变参数初始化随机数流，所以不宣称整个官方初始状态逐位相同。
- 数据沿用本项目LOO、缓存与item映射；test在映射序列前剔除。只按固定4096人validation NDCG@10选择一个checkpoint，然后完整validation审计，不照搬官方每指标分别保存最优及test阶段。

## 梯度与可选历史MSE

`L_FM + .2*L_CE + beta*L_history_MSE + .1*L_AE`一次反传：

```text
content → AE encoder → item representation
                     ├→ history tokens → FMRec Transformer → f → CE/FM
                     ├→ clean endpoint → time bridge ────────┘
                     ├→ clean FM target ───────────────────→ FM
                     └→ full classification catalog ───────→ CE
```

所有分支均不detach。单独验证CE、FM对AE encoder有非零梯度；排除AE几何项后执行optimizer，确认AE参数实际变化。默认不冻结AE，不使用v5_2梯度路由，也不回归自建速度网络。

历史MSE对应官方中间层`h_m`经过128→512→2048→物品数+1 decoder重建history multi-hot，不是另一个物品embedding MSE。开关`fmrec.mse_weight`默认0；显式设.4可采用官方权重。平衡抽样保留原语义：全正项和均匀零项有放回抽样、在完整矩阵上平均。默认关闭时不创建decoder，也不做额外前向或抽样。本轮三组均关闭，以免把增益归因于新辅助目标。

## 预先固定的三个对照

| 组 | 输入/模型 | 目标 | 部署 |
|---|---|---|---|
| FMRec | 官方随机桥、token融合Transformer | FM + .2CE + .1AE | 官方30步Euler |
| CE-only | 完全相同随机桥与Transformer | .2CE + .1AE（FM权重0） | 相同30步Euler |
| History CE | 相同Transformer块/输出LN，仅历史，无桥时间模块 | .2CE + .1AE | 单次历史前向 |

三组相同AE与Transformer共享参数初值、数据顺序种子、LOO划分和selection用户。History CE在共同初始化后删除时间MLP，保留共享初始化与初始CPU随机状态；不能声称训练dropout逐步相同。CE权重三组都.2，避免改变AE相对权重及梯度尺度混淆FM消融。

FMRec对CE-only检验新增FM监督；FMRec对History CE检验整条生成机制是否优于同骨干直接CE。旧版v5_6 HSTU直接CE分数只作为历史背景，不替代同Transformer/同前缀训练对照。即使优于旧HSTU，也不能把结构与预算带来的差异归为FM增益。

## 执行预算与判断

先Beauty、seed2024，不扫种子/超参/数据集。官方最大500epoch/每20轮验证保存在默认配置；本轮**初筛40epoch/每5轮验证**，三组同budget。40轮不会触发100轮的LR衰减，明确区别于完整官方训练安排。此预算用于先验证可运行性及早期增量，不自动宣称充分收敛；若最佳值贴近40轮且曲线仍升，只能说当前初筛未决，不能据早期失败断言FM无效。

不依据full validation回选epoch或调权重。每run完整保存初态/共享初态/数据/用户/源码hash、展开配置、逐轮loss和selection、梯度审计、checkpoint和逐用户预测。检查官方计算一致性、完整宽度真实缓存短程、梯度链路和重载后才正式训练。运行目录拒绝覆盖，test不评。保护其他版本与参考仓库hash；旧v5_4完整源快照已在history，已有outputs/flowrec保持原样。
