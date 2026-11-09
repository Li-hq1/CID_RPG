# UGRec

UGRec 是基于连续物品表示与条件 Rectified Flow 的生成式推荐研究项目。研究路线从 PCA 连续生成、多次兴趣采样，推进到几何保持 AE，再到 AE 与推荐模型的联合训练。

磁盘路径暂保留 `/home/haoquan/CID_RPG`；项目名称统一为 UGRec。

| 版本 | 本阶段研究问题 | 主要方案 | 入口 |
| --- | --- | --- | --- |
| [UGRec_v1](UGRec_v1/README.md) | 连续表示能否替代 RQVAE 离散语义 ID，并通过多次生成覆盖用户兴趣？ | 文本 embedding → PCA → Transformer/HSTU 条件 → RF → 多次采样聚合 | `UGRec_v1/scripts/train.sh` |
| [UGRec_v2](UGRec_v2/README.md) | 能否用可学习、保持几何的压缩器替代 PCA，为统一训练准备表示空间？ | Geometry AE 独立训练并冻结，再训练序列模型与 RF | `UGRec_v2/scripts/train_geometry_ae.sh`、`train.sh` |
| [UGRec_v3](UGRec_v3/README.md) | 能否在统一训练过程中优化表示与推荐，并系统探索参数？ | AE 与推荐模型交替更新、可选 AE-only InfoNCE、Optuna | `UGRec_v3/scripts/train_joint.sh`、`search_joint_contrastive.sh` |
| UGRec_v4 / v4_1 / v4_2 | 联合训练之后，哪些损失和生成步骤真正影响推荐？ | 诊断、排序消融，以及 v4_2 冻结的一步整流流加抽样排序 | 保留为实验记录，不作为新实验入口 |
| [UGRec_v5](UGRec_v5/README.md) | 当前方案 | 默认 control：内容 AE、HSTU、一步整流流、源噪声的边缘似然、混合概率解码；保留 VLAD 实验与 v4_2 原版配置用于复现 | `UGRec_v5/scripts/train.sh` |
| [UGRec_v5_1](UGRec_v5_1/README.md) | 固定 v5 control 的四数据集调参 | 通用结构/专用参数分离、过渡结构消融、可溯源 campaign 与 RSI 迭代协议 | `UGRec_v5_1/scripts/tune.sh` |
| [UGRec_v5_2](UGRec_v5_2/README.md) | CE 已具备排序能力后，CFM 是否有独立价值？ | Beauty单种子CE/联合CFM/冻结CE后CFM初筛完成，未见FM增量，暂不扩展双尾部；[分析](UGRec_v5_2/docs/初步实验分析_20261106.md) | `UGRec_v5_2/scripts/train.sh` |
| [UGRec_v5_3](UGRec_v5_3/README.md) | 保留LOO并检验未来多目标覆盖 | 从v5_1独立构建新窗口与CE/CFM四组对照，先Beauty固定单种子A/B；[方案](UGRec_v5_3/docs/方案.md) | `UGRec_v5_3/scripts/train.sh` |
| [UGRec_v5_4](UGRec_v5_4/README.md) | 官方FMRec机制在在线joint AE表示上是否有增量？ | AE encoder替代ID词表，FM+.2CE、历史MSE默认关；Beauty40轮三组完成，FMRec .00584、同桥CE .00536、历史CE .06257；有限预算尚无生成净收益；[结果](UGRec_v5_4/docs/FMRec实验结果.md) | `UGRec_v5_4/run_fmrec.py` |
| [UGRec_v5_5](UGRec_v5_5/README.md) | 同一干净端点的 FM+CE 能否改善 control？ | Beauty/seed2024 六组完成，权重0.1/1/5未见匹配CE增量，保留control；[结果](UGRec_v5_5/docs/实验结果.md) | `UGRec_v5_5/research/campaign.py` |
| [UGRec_v5_6](UGRec_v5_6/README.md) | 逐 item 多 embedding 的尺度、分簇与精简是否有效？ | 最新6组精简/映射未得到一致净收益；128参数残差是成本对照，多中心方向近重复；保留原尺度匹配K1/G2参考。[结果](UGRec_v5_6/docs/item_cluster_simplification/结果.md)，单种子，历史保留 | `UGRec_v5_6/research/item_cluster_simplification.py` |

每版的 [方案文档](UGRec_v1/docs/方案.md) 按“问题 → 假设 → 模型 → 训练 → 采样 → 验证”的顺序组织。文档是本次依据现有研究整理的设计说明，不代表当时已存在的方案记录。

```text
CID_RPG/                         # 项目名称 UGRec，保留当前磁盘路径
├── UGRec_v1/                    # PCA + Transformer/HSTU + RF + 多次采样
├── UGRec_v2/                    # 独立 Geometry AE + 冻结特征推荐
├── UGRec_v3/                    # 联合训练 + Optuna
│   └── outputs/history/         # 当前实现不支持的早期联合训练实验
├── UGRec_v4/                    # 瓶颈诊断与探索性消融，实验记录
├── UGRec_v4_1/                  # 推荐感知表示与条件残差流，实验记录
├── UGRec_v4_2/                  # 冻结原版所在的实验目录
├── UGRec_v5/                    # 当前默认 control（configs/final.json），含历史实验复现配置
├── UGRec_v5_1/                  # 独立调参版：四数据集、参数作用域、结构消融与审计
├── UGRec_v5_2/                  # CE与CFM独立价值初筛；保留双尾部/梯度路由实现
├── UGRec_v5_3/                  # 保留LOO，新增未来多目标CE/CFM对照
├── UGRec_v5_4/                  # 在线joint AE + 官方FMRec机制，旧FlowRec归档
├── UGRec_v5_5/                  # 独立干净端点FM+CE对照、权重探索及审计
├── UGRec_v5_6/                  # 逐item VLAD多向量探索；原FM/direct CE消融保留
├── cache/                       # 共用 Amazon 数据、原始 embedding、预训练模型
├── docs/                        # 迁移映射、完整性清单、版本维护约定
├── archive/pre_ugrec/            # 整理前源码、旧脚本、文档和参考实现
├── pyproject.toml / uv.lock      # 共用依赖环境
└── .venv/
```

每个版本都拥有物理独立的 `models/`、`engine/`、`misc/`、`data/`、配置、脚本和测试；v2/v3 各自拥有压缩代码，v3 自带联合训练代码。版本间没有运行时代码导入。`cache` 软链接指向仓库公共缓存；数据处理代码也按版本保留，后续更改样本逻辑不会隐式修改旧版实现。

先在根目录准备环境，再进入所需版本运行：

```bash
cd /home/haoquan/CID_RPG
uv sync

cd UGRec_v1
GPU_ID=0 CATEGORY=Beauty bash scripts/train.sh
# 同一条基线切换序列编码器
GPU_ID=0 ENCODER_TYPE=transformer bash scripts/train.sh
```

版本脚本自动使用根目录的 Python 环境，并切换到版本目录。参数可以追加，例如 `--smoke-test`；这只检查小样本流程，不代表正式指标。

已有 checkpoint 保留 TigerDiff 类名、目录名与权重键，以保证加载一致；新运行使用 `ugrec_v*` 命名。历史实验名不作批量改写。推荐输出写入各版 `outputs/ckpt/`，独立 AE 写入 v2 的 `outputs/compression/`，联合训练和搜索写入 v3 的 `outputs/joint_training/`。这些大文件不纳入 Git。`experiments/` 和上游对照仓库 `FMRec/` 也不纳入 Git；权重、数据缓存和训练日志的忽略规则见根目录 `.gitignore`。

从旧位置寻找实验，请查 [迁移说明](docs/整理说明.md) 和 [checkpoint 清单](docs/migration/checkpoints.json)。原始日志、结果文件和 checkpoint 字节保持原样，其中的旧路径是历史记录；运行时应使用各版 `docs/run_configs/` 中更新路径的配置。旧 `cache/.../Geometry_AE` 位置保留兼容链接，真实 AE 权重与导出特征已归入 v2。

本次仅整理当前本机文件；其他服务器仍保留原目录，尚未同步。本机没有 Memory 中提到的远程 53-trial Optuna study；其观察记录保存在 v3 历史文档中。

后续迭代规则见 [版本维护约定](docs/版本维护.md)。
