# v5_4：FlowRec 三损失替换实验（预注册）

## 问题与假设

原 control 只优化一次运输后多个噪声查询的边缘物品似然，没有显式历史先验监督和路径速度监督。本轮问题是：在原有内容 AE + HSTU + 条件速度网络上，Prior CE + CFM + 随机时间终点 Align CE 是否能产生比直接先验预测更好的可部署推荐结果？训练桥上的 Align CE 变小不能单独证明推理有效。

核对文献：FlowRec，arXiv:2508.17618v2，式(7)、(8)、(11)–(14)、(16)，https://arxiv.org/html/2508.17618v2 。这是损失形式移植，不是官方模型和分数复现。没有运行作者实现。

## 明确实现

保留本版原有 AE、两层 HSTU、两块速度网络、条件投影和锚点投影。所有组模型参数形状和初始化相同。物品表示 `z=LN(AE(content))`；确定性历史先验 `x0=sqrt(d)*normalize(anchor(condition))`，不加源噪声。保留原有 cosine/T softmax（T=.07）以控制打分尺度，不改为论文未归一化内积。目录<=20000采用完整 CE，其余使用原有4096均匀修正抽样；验证始终完整目录。

代码的时间 `s=1-t` 从1降至0。设目标 `x1=z_y`：

```text
xs = (1-s)*x1 + s*x0
v = existing_conditional_velocity_net(xs, s, condition)
L_prior = CE(cos(x0, Z)/T, y)
L_cfm = mean_batch,dimension[(v-(x0-x1))^2]
xhat1 = xs-s*v
L_align = CE(cos(xhat1, Z)/T, y)
L_recommend = L_prior + alpha*L_cfm + beta*L_align
L_total = L_recommend + 0.1*L_AE
```

CFM和Align只做同一次随机时间前向，不额外使用t=0/1的分类分支。端点、速度标签、历史和目录表示均不detach，按三项共同端到端优化；源先验的归一化和物品LN使两端RMS约1。MSE按维平均，权重数值不可直接当作论文的最优权重。AE项是内容表示原有约束，不是额外推荐loss；保留它保证与control的内容表示训练条件可比。

推理只读历史，从同一确定性x0开始，默认一次Euler，按同样cosine打分。可以只对固定验证子集检查1/4/8步推理，步骤只改变积分近似、不会使用目标。论文的MLP只输入状态和时间；本版保留旧的历史条件通道，并且不移植其乘性lambda噪声、四层双向Transformer、优化器和数据预处理，避免损失实验同时改变骨干。不得称为完整FlowRec复现。

## 第一阶段与必要对照

仅Beauty/seed2024。所有正式组40 epoch，batch256、每批最多1024个监督位置、相同训练窗口、4096个固定验证用户选模，完整22363人validation审计。40轮是固定预算，不因某组结果不好自动延长。test标签在映射前排除；不评test。

1. **原control重新训练**：本版`configs/control.json`，噪声边缘似然S8、sigma1、推理32查询。它回答替换是否值得；不能将其与确定性一路推理冒充相同推理成本，也不拿v5_1历史分数充本版结果。
2. **仅Prior CE**：alpha=beta=0，用prior选择checkpoint和生成，回答已有内容表示和历史骨干可以达到什么水平。
3. **三损失**：alpha=5、beta=1，以实际flow生成NDCG@10选模，记录同checkpoint的prior分数以及三项原始/加权loss、端点RMS、速度RMS，判断桥上训练收益是否转为推理收益。

上述完成后补**Prior+Align，alpha=0、beta=1**作为必要消融，直接归因CFM的作用，而不是根据最终完整验证结果搜索模块组合。第一阶段权重alpha5、beta1参考论文调参区间的下界；后续仅在原始/加权损失及验证曲线显示CFM压制排序，或full相对无CFM对照明显退化时，尝试一个低CFM权重alpha1。若确需另调beta，应先记录证据和具体假设，不开启二维网格。

## 决策与核验

以固定4096人selection为调参依据；全量validation只做锁定checkpoint的描述性审计，不据它反复选组或选epoch。所有结果是单种子趋势，不能宣称稳定优势。若CFM没有优于无CFM或prior-only，结论应保留这个负结果，不把Align训练分数下降当作流贡献。不自动扩展到Sports/Toys/CDs，不做多种子。

每组保存resolved config、源码快照和hash、初始state hash、数据及selection指纹、逐轮日志、checkpoint、全量排序结果和checkpoint重载精确检查。smoke和4更新完整宽度检查不是benchmark。旧版源码/配置/文档运行前后hash核对；所有新写入仅在v5_4。
