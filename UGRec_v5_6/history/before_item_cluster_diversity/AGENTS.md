# 本轮完成记录：VLAD 精简与残差映射（2026-11-09）

本轮6组训练与审计已全部完成。结果docs/item_cluster_simplification/结果.md，最终机器审计outputs/item_cluster_simplification/summary.json与decision_B.json。没有得到相对正收益参考一致的selection/full提升，保留原K1/G2候选，不替换默认。128参数残差K1是成本对照；K4映射仅在同K4局部比较改善，未证明有效多簇分工。原3933份输出与114文件归档不变；其他版本并发变化如实登记。105tests与六组完整宽度4更新/40epoch/4040更新/重载和预测审计通过，无test或多种子。本轮不再新增训练/网格/种子；用户后续新指令可覆盖。下方保留原授权与历史。

# 当前授权：VLAD 结构精简与残差映射（2026-11-09）

用户新指令授权本轮，覆盖下方历史完成/停训条款。围绕扩维分组必要性与残差坐标适配设计受控实验。默认旧NeXtVLAD与旧checkpoint保持兼容，AE/catalog不变。新增无扩维、无分组、无组门控、无无效assignment的单输入NetVLAD方向等价简化，以及逐簇共享无bias线性映射；映射恒等初始化，不消耗RNG，最终统一sqrt(128)尺度。只用sum与原RoPE/item-block因果mask。

固定Beauty/seed2024/40epoch/4040更新/batch256/两层HSTU；4096人selection每5轮选模，22363人full validation只审计，无test/种子扫描/LR网格。不自动替换默认。阶段A四组区分归一化、中心与映射；阶段B在A后登记决策，至多补同轻量结构K4有/无映射的必要对照，不增加无依据组合。详见docs/item_cluster_simplification/设计.md。先工程测试、完整宽度真实cache4更新，再正式训练。旧源码114文件归档history/before_item_cluster_simplification，旧输出3933份hash保护。所有新输出限outputs/item_cluster_simplification，不覆盖其他版本，不打断共享GPU任务。

# 本轮完成记录：参数与聚合优化（2026-11-09）

本轮7组已训练、诊断并审计完成。见docs/item_cluster_optimization/结果.md、outputs/item_cluster_optimization/summary.json与阶段decision。保留原正收益K1/G2、sqrt(128)、sum候选；本轮不再启动训练/网格/多种子或替换原默认。下方为已完成实验的授权与历史规则，后续用户新指令可覆盖。

# 当前授权：尺度匹配正收益上的 VLAD 参数与聚合优化

阶段P负结果后登记阶段S，允许在正收益K1/G2上仅测试输入倍率.5/2，两组与原参考全模型初态相同。聚合阶段仍固定sqrt(128)，不与尺度同时调整；见decision_S_start.json和设计.md。

用户已明确要求在 scaled K1 的正收益上继续优化，覆盖下方上一轮停止新增训练的约束。固定 Beauty/seed2024/40epoch/4040更新，两层HSTU、AE/catalog、损失、LR、数据及selection协议不变。先做K1的G1/G4和G2下K2，依据selection登记决策后再做必要的G/K联动或聚合对照；不做无依据大矩阵、种子扫描或test。G与expansion同步改变以保128维，不能称纯G消融。

本轮可改变HSTU后逐item聚合：等权sum、输入mass加权、上下文化共享评分加权，三者互斥不叠加。保持每簇输入sqrt(128)尺度、原item RoPE与block causal mask；learned评分从零初态开始，无簇ID/新loss/AE内分簇。新计算图先测试及真实缓存完整宽度4更新再正式训练。不终止其他GPU任务。旧源码快照history/before_item_cluster_optimization；全部旧输出hash见docs/item_cluster_optimization/previous_artifacts.json；新输出仅outputs/item_cluster_optimization，拒绝覆盖。设计与阶段决策登记后执行。

# 本轮完成记录（2026-11-09）

多向量后续5组已完成并审计，见docs/item_cluster_followup/结果.md、outputs/item_cluster_followup/summary.json与decision_A/B/C.json。本轮不再新增训练，不自动替换默认或扫参/种子。以下授权与历史规则用于解释已完成实验；后续用户新指令可覆盖本条。

# 当前授权：多向量后续逐步探索

用户已授权按机制分析推进，覆盖下方上一轮仅三组/不继续的范围限制。先做输入尺度匹配，再依据结果推进AE内分簇sum和必要的learned聚合/有限LR调整。固定Beauty/seed2024/40epoch/原两层HSTU；不自动扩数据集、扫种子或做大矩阵。方案、分阶段决策见docs/item_cluster_followup/设计.md。

上一轮源码80文件保存在history/before_item_cluster_followup，旧outputs/item_clusters及outputs/fm_ablation的152份结果身份已登记保护。共享cache只读，新输出路径outputs/item_cluster_followup，拒绝覆盖已有run。保留原RoPE/item-block mask/最后sum及所有审计要求。新增计算图、梯度、优化器归属和真实缓存4更新验收后才正式训练。不打断其他GPU进程。

# v5_6 当前：逐 item VLAD 多向量探索

用户最新指令覆盖上一轮仅 FM/direct 两组的范围限制。上一轮源码保存在 history/before_item_clusters/，原 outputs/fm_ablation/ 不改。

- 当前基线为重新训练的 direct CE：AE 单向量 → 原 input_proj → 逐 item NeXtVLAD K 个簇 → 原两层 HSTU → 同 item 输出直接求和 → 原 condition_proj/输出头与 CE，AE 几何项保持 .1。
- 不做前缀 VLAD，不加簇 ID/残差/concat 映射/新损失；同 item 共享原 RoPE，item 内全连接、item 间因果，padding 全屏蔽。注意力归一化分母按展开 token 长度。
- 首轮仅 Beauty baseline/K1/K4，seed2024、40 epoch、原 LR=.001 和 batch256。K1 为新增变换对照；其与 K4 区别还包含 K 相关参数量和求和尺度，不能称为完全等容量。
- 固定4096 validation用户每5轮选模，完整validation审计。无test、无种子扫描，不预设后续网格；机制诊断先于追加实验。
- 旧版源码/结果不改，共享cache只读。先通过工程测试和真实缓存完整宽度4更新，再用空余GPU训练，不打断其他任务。每个新目录拒绝覆盖并保存源码/初态/数据/selection指纹及预测。
- 机制解释/实验定义见 docs/item_clusters/设计.md；原轮规则仅作历史，不约束本轮新增模块。

<details><summary>上一轮历史指令</summary>

# v5_6 原control生成分支消融

本轮只回答用户指定的问题：原control的生成分支是否比HSTU直接CE预测有正收益。先读README和docs/实验设计.md，继承全局科研规则。

- 每个版本物理独立，旧版源码/配置/结果不改，共享cache只读。
- 只做原control和direct CE两组，在Beauty/Sports/Toys各seed2024、40epoch。原control两层HSTU、原LR=.001，不混入v5_1已调参结构。
- 不加新损失/模块，不做超参搜索、种子扫描或test。只用固定4096用户selection选checkpoint，完整validation审计，不能将单种子趋势称为稳定优势。
- Direct CE的模型、优化器和checkpoint必须实际不含FM参数，不能把未调用但仍保留的模块当作已删除；共享AE/HSTU/投影/输出头初值必须匹配。
- 工程测试与真实缓存短程检查通过后正式运行；每次新目录保存源码/初态/数据/选模用户hash、结果与预测，禁止覆盖。GPU按空余资源使用，不中断其他进程。
- 按数据集分别报告原control相对direct CE的收益与代价，不用平均掩盖数据集差异。本轮只判断，不原地删除v5默认实现。

</details>
