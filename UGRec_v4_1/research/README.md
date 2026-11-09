# v4 探索实验：定义、边界与复现

本目录用于分析瓶颈，不替代正式基准复现。完整讨论见 [研究报告](../docs/研究分析与后续路线_20261028.md)。本次运行了 22 次紧凑模型训练，并评估了传统基线、两个现有联合 checkpoint、相同生成样本的解码规则，以及内容/转移计数混合。

## 数据与结果协议

- 使用现有 AmazonReviews2014 Beauty、Sports_and_Outdoors 缓存。每个用户 `seq[:-2]` 用于训练，倒数第二项为验证目标，最后一项不用于训练、预测、指标或选模。
- 验证历史取训练序列末尾最多 50 项；默认全物品排序、不屏蔽历史，与当前主模型的候选协议对应。统计基线同时给出屏蔽历史版本。候选库沿用已有完整 item mapping，属于现有基准的 transductive catalog 设定。
- PCA128 仅对推荐训练出现的 item 拟合；内容特征使用已有冻结 Qwen embedding。没有修改共享缓存。
- 训练 checkpoint 由固定 4096 验证用户选择（抽样 seed=20261028）；选定后评估完整验证集。完整验证集包含选模子集，因此不是新的独立测试集。混合权重也仅使用该子集选择，并另报其余验证用户。
- 配对 bootstrap 默认重采样用户 2000 次；区间以已训练模型为条件，不包括参数搜索/选模的不确定性。三种子标准差与配对 bootstrap 是不同统计量。
- 新结果全部为 validation。论文表格、历史 test、本次完整 validation、4096 用户诊断不可混成同一排行榜。

## 紧凑模型定义

统一 hidden/latent=128、2 层、4 heads、最大长度 50。Transformer 与 HSTU 使用本项目现有实现，不主张两者已分别充分调参或参数/FLOPs 完全一致。训练用现有生产 tokenizer 的 first-window-all-positions / later-window-last-position 策略，batch=256、AdamW lr=1e-3、weight_decay=.01、10% warmup 后 cosine，40 epochs，每 5 epochs 验证。

| 参数 | 含义 |
|---|---|
| `pca` | 固定 PCA128 内容表示，逐物品 LayerNorm |
| `adapter` | PCA + 共享 128→256→128 残差 MLP；末层零初始化；无可训练 item ID 表 |
| `id` | 随机初始化、可学习的 item embedding |
| `hybrid` | PCA + 零初始化的可学习 item ID 残差 |
| `ce` | 仅历史条件的全物品 cosine CE，温度 .07 |
| `rf` | 现有 velocity RF；target detach，条件来自历史 |
| `rf_ce` | RF + 历史 CE；代码支持，本次没有训练该组 |
| `x0` | 网络预测 clean latent，MSE；target detach |
| `x0_ce` | clean MSE + .2 × 作用于带噪输入预测的 catalog CE；不是历史独立 CE |

生成目标使用 32 次采样、L2 support 解码。`x0` 系列按 `v=(x_t-x_hat)/t` 转换，20 个时间点对应 19 次网络求值。评估固定随机数并保存/恢复当前训练设备随机状态，避免验证频率改变该设备的训练随机流。

这些是相同训练样本与 epoch 预算的筛查，不是相同耗时/FLOPs 预算，不是 FMRec/MIRAGE 的复现，也不是主联合模型的训练替代品。`full_validation_direct_ce` 仅在 CE head 确实得到监督时有解释意义；纯 RF/x0 的未训练 CE head 不是有效基线。

## 复现命令

以下从 `/home/haoquan/CID_RPG/UGRec_v4` 执行，GPU 编号可替换。紧凑训练目录若已存在会拒绝覆盖，复现应先保留旧输出并使用不同 seed 或单独备份输出目录。

```bash
# 两个数据集的训练集统计、ItemKNN 和原始内容检索
../.venv/bin/python -m research.nonparametric --category Beauty --device cuda:2
../.venv/bin/python -m research.nonparametric --category Sports_and_Outdoors --device cuda:2

# 示例；本次已完成的具体组合、seed 见输出 results_table.md
../.venv/bin/python -m research.compact_ablation \
  --category Beauty --representation adapter --encoder transformer \
  --objective ce --epochs 40 --seed 2024 --device cuda:2

# 已有 v3 checkpoint：加载时核对 embedding、item 顺序和训练 mask 的哈希
../.venv/bin/python -m research.checkpoint_diagnostics --name ae_infonce \
  --checkpoint ../UGRec_v3/outputs/joint_training/Beauty/joint_Beauty_contrastive_ae_only_w0.01/best.pt \
  --device cuda:4
../.venv/bin/python -m research.checkpoint_diagnostics --name ae_no_infonce \
  --checkpoint ../UGRec_v3/outputs/joint_training/Beauty/joint_Beauty_layernorm_no-infoNCE_no-init-test/best.pt \
  --device cuda:4

# 复用已经保存的生成样本，不再采样
../.venv/bin/python -m research.rerank_cached --name ae_infonce --device cuda:7
../.venv/bin/python -m research.rerank_cached --name ae_no_infonce --device cuda:7

# 以下需要相应 seed2024 CE 输出及统计基线
../.venv/bin/python -m research.data_audit --category Beauty
../.venv/bin/python -m research.data_audit --category Sports_and_Outdoors
../.venv/bin/python -m research.hybrid_rank --category Beauty --device cuda:4
../.venv/bin/python -m research.hybrid_rank --category Sports_and_Outdoors --device cuda:4
../.venv/bin/python -m research.summarize

# 使用真实缓存验证搜索 trial 跳过测试集；不是模型效果实验
../.venv/bin/python -m joint_training.search \
  --search-config research/validation_smoke_search.json --smoke-test --contrastive \
  --device cuda:7 --set training.output_dir=outputs/research_20261028/protocol_smoke
```

## 输出索引

输出根目录：`../outputs/research_20261028/`。

- `summary.json`、`results_table.md`：22 次训练的分组均值/标准差及配对比较。
- `research_overview.png` / `.pdf`：表示与监督、配对解码、噪声分层对齐诊断。
- `nonparametric/{category}.json`：统计/语义基线与数据规模；同目录保存逐用户预测。
- `compact/{category}_{representation}_{encoder}_{objective}_s{seed}_e40/`：配置、训练轨迹、best checkpoint、逐用户预测、结果。
- `checkpoints/{ae_infonce,ae_no_infonce}/`：梯度/shortcut、解码、生成样本、reranking 结果。
- `data_audit/{category}.json`：训练已见/未见转移、同时间戳、目标频次分层。
- `hybrid_rank.json` / `hybrid_rank_Sports_and_Outdoors.json`：概率混合各系数、选模子集与其余验证集。
- `logs/tests.log`：149 tests，跳过 1 项，其余通过；`logs/validation_search_smoke.log`：真实缓存跳过 test 的流程检查。
- `sources/`：三篇论文 PDF、文本及 FMRec 官方代码片段；仅作为阅读证据，不是外部方法复现。

`hybrid_rank` 使用 `p=(1-α)p_content+αp_transition`，没有可用转移行时转移分支为零；α<1 时该行排序仍由内容分支决定。α=1 的零分 ties 不同于统计基线的微小 popularity tie break，因此不能把该端点和基线数值差异解释为算法差异。

v4 基础联合训练入口仍保留 AE-only InfoNCE 行为。探索脚本没有将新的推荐架构偷偷设为生产默认，也没有修改 v1–v3 模型与历史结果。
