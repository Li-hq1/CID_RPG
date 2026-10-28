# AE 与生成式推荐一体化训练

本入口独立于原有 `train.py`、`engine` 和顶层 Python config。它复用数据处理、GeometryAE、HSTU/Transformer 与 Rectified Flow 模型组件，使用自己的配置、训练循环和 checkpoint。原有两阶段训练仍可单独运行。

AE 从随机权重起步：输入 `mean` / `scale` 用推荐训练集 item 调用 GeometryAE 的 `fit_input_statistics` 估计，网络权重在 `training.seed` 下使用 PyTorch 默认 Linear 初始化。历史由 HSTU 编码，Rectified Flow 生成连续 next-item 表示，再从当前 AE 编码的 item catalog 中检索。GeometryAE 是确定性自编码器，不是带 KL 正则的 VAE；本入口没有额外引入 KL loss。推荐监督和 contrastive 都使用每条序列的全部有效位置。

## 快速运行

从项目根目录执行：

```bash
# CPU 上检查独立入口、损失、更新和保存流程。
.venv/bin/python -m joint_training --smoke-test --device=cpu --run-id=smoke_cpu

# 检查默认 AE-only 对比学习，推荐模型参数只接受 RF loss。
.venv/bin/python -m joint_training --smoke-test --device=cpu --run-id=smoke_contrastive \
  --contrastive

# 首轮基线：不启用 contrastive。
.venv/bin/python -m joint_training --device=cuda:0 --run-id=beauty_layernorm_no_contrastive

# RF 速度场上的 AE-only 对比学习。
GPU_ID=0 bash scripts/train_joint_contrastive.sh

# 用 Optuna 搜索当前 contrastive 配置。范围在 joint_training/search_space.json。
GPU_ID=0 bash scripts/search_joint_contrastive.sh

# 两侧学习率可分别设置。
.venv/bin/python -m joint_training --device=cuda:0 --run-id=beauty_lr_ablation \
  --set ae.lr=0.00003 --set recommender.lr=0.0005

# 开启可选 InfoNCE contrastive。
.venv/bin/python -m joint_training --device=cuda:0 --run-id=beauty_layernorm_contrastive \
  --contrastive --set contrastive.weight=0.01 \
  --set contrastive.num_negatives=64
```

下游 latent 固定使用无 affine 的 LayerNorm。AE 初始化只消耗一份临时随机数状态，不会改变随后推荐模型的初始化序列。

启动脚本风格与 `scripts/train_tigerdiff.sh` 一致：固定 `CUDA_VISIBLE_DEVICES`，显式传入 `--device`（默认 `cuda:0`，不用 `auto`），并保留额外参数。默认跑 Beauty / 无 contrastive 基线；smoke、contrastive、续训、只评估等变体以注释形式写在脚本下方，取消注释即可切换。

```bash
bash scripts/train_joint.sh
bash scripts/train_joint.sh --run-id=beauty_custom
```

本入口支持 CPU 和单 GPU，不实现 DDP。

训练过程会写入 TensorBoard 事件到 `ckpt/joint_training/<category>/<run_id>/tensorboard/`（分组为 `Loss/*`、`Train/*`、`Grad/*`、`AE/*`、`Val_Metric/*`、`Latent/*`、`Test_Metric/*`）。`metrics.jsonl` 仍是完整数值源；未安装 `tensorboard` 时只跳过看板、不影响训练。查看单次或对比多次实验：

```bash
# 单个 run
tensorboard --logdir ckpt/joint_training/Beauty/joint_Beauty_layernorm_no_contrastive/tensorboard --port 6006

# 同一 category 下多 run 对比
tensorboard --logdir ckpt/joint_training/Beauty --port 6006
```

SSH 远程时在本地做端口转发即可打开 `http://localhost:6006`。

## 独立配置

可修改的默认值集中在 `joint_training/config.py`。类型检查、取值检查和命令行解析在 `joint_training/config_parse.py`。本入口不读取原有 `config.py`、`config_hstu.py` 或 `config_transformer.py`。

| 配置部分 | 作用 |
| --- | --- |
| `data` | 类别、缓存、数据划分、历史长度和 item 特征来源 |
| `ae` | 结构、AE loss 权重、item batch、学习率和 AE 优化设置 |
| `recommender` | HSTU/Transformer、RF、序列 batch、学习率和推荐优化设置 |
| `contrastive` | AE-only InfoNCE 开关、权重、温度、query 上限和负例数 |
| `training` | 设备、联合训练轮数、随机种子、run ID 和输出目录 |
| `evaluation` | 指标、检索和生成采样设置 |

`--config <path.json>` 从 JSON 加载独立配置；未知字段和无效类型会被拒绝。`--set dotted.path=value` 覆盖一个已定义字段，`value` 按 JSON 解析，普通字符串也可直接书写。多次使用 `--set` 可以修改不同字段。JSON 布尔值使用 `true/false`，数组使用 JSON 数组。

```bash
.venv/bin/python -m joint_training --config joint_experiment.json \
  --run-id=beauty_custom --set ae.lr=0.00003 \
  --set recommender.encoder_type=transformer --set training.seed=2025
```

AE 默认学习率为 `3e-5`，推荐模型为 `5e-4`。两者使用独立 optimizer 和 scheduler，各 scheduler 按相应 optimizer 的实际更新次数前进。默认每个推荐 macro-step 更新一次 AE、一次推荐模型。联合训练从随机 AE 权重和训练集输入统计量开始，不加载预训练 checkpoint，也不续接预训练 scheduler。

`ae.update_every` 和 `recommender.update_every` 可以分别降低两侧更新频率；`ae.frozen=true` 是冻结 AE 的对照实验。AE objective 不重新 warmup，与 optimizer 的学习率 warmup 是两种设置；默认 AE/推荐学习率 warmup 比例分别为 `0.05/0.1`。

`ae_loss` 是 AE 的标准训练损失，包含归一化重建 MSE、cosine reconstruction、variance 和 correlation，relation 蒸馏关闭。它在推荐训练集合中出现过的 unique items 上均匀抽样，默认 item batch 为 512；推荐 batch 默认为 256。这些项从第一步起使用配置里的目标权重。`contrastive_loss` 是可选的 InfoNCE，固定只用于更新 AE encoder。

## 超参搜索

`python -m joint_training.search` 用 [Optuna](https://github.com/optuna/optuna) 对一份 joint training 配置做试验。每个 trial 走现有训练入口：固定参数来自命令行和 `--set`，被搜索的参数由 trial 覆盖。训练仍用验证集选择 `best.pt`。Optuna 比较的是该 checkpoint 在测试集上的 `evaluation.val_metric`（默认 `ndcg@10`），数值越大越好。已经用验证集分数跑过的 study 不要续搜，换一个 `--study-name`。

搜索空间在 `joint_training/search_space.json`。删掉列表中的一项就不再搜索它；保留的项每次都会采样。默认覆盖训练 epoch（`training.epochs`，50 到 150，步长 10）、推荐 batch（`recommender.train_batch_size`，128/256/512）、学习率、weight decay、warmup、AE 几何损失权重、HSTU dropout、classifier-free guidance，以及 contrastive 的权重、温度和负例数。AE 的 item batch 仍固定为 512。模型宽度和层数不在默认范围内，避免显存差得太多；需要时按同样的字段格式加进去。当前配置里的值必须落在对应范围内，否则启动时会拒绝。

```bash
GPU_ID=0 bash scripts/search_joint_contrastive.sh

# 这一次只再跑 5 个 trial，并把每个 trial 限制到 30 个 epoch。
N_TRIALS=5 bash scripts/search_joint_contrastive.sh \
  --set training.epochs=30 --set evaluation.patience=10

# 直接调用时可以换配置文件。其余参数会原样传给 joint_training。
uv run python -m joint_training.search --search-config joint_training/search_space.json \
  --study-name joint_Beauty_contrastive_search --n-trials 20 \
  --category Beauty --device cuda:0 --contrastive
```

`--n-trials` 是这一次新跑的数量。同一个 `--study-name` 再执行会从 `ckpt/joint_training/<category>/<study_name>/optuna.db` 接着搜，已完成的 trial 不会重跑。修改 `parameters`、sampler 或 pruner 后需要换一个 study name。清空重跑时，删除 study 目录和同级的 `<study_name>_t*` 目录，或直接换一个名字。

新 study 的第一个 trial 使用当前配置，方便和现有实验比较；它计入 `n_trials`。`--no-baseline` 可以关掉。Contrastive 参数出现在搜索空间里时必须带 `--contrastive`。

`pruner` 不决定 trial 的排名。它只在训练中途查看验证集上的当前最佳分数，把明显落后的 trial 提前停掉，省掉后续 epoch 和最终测试。被停掉的 trial 没有测试分数，不参与最优比较。

`median` 的比较对象是已经完整跑完的 trial 在同一 epoch 的验证最佳分数，取中位数。当前 trial 到目前为止的最佳验证分数低于这个中位数就停止。`none` 不提前停止，每个 trial 跑到自己的 early stopping 或预定 epoch。`pruner_startup_trials` 是开始剪枝前至少要有的完整 trial 数，默认 `5`。`pruner_warmup_steps` 是 epoch：小于这个 epoch 的验证只记录、不剪枝。默认 `10`，在 `eval_interval=5` 时 epoch 5 只记录，epoch 10 才可能停掉。训练结束或触发 early stopping 的那一轮不再剪枝，以便加载 `best.pt` 并算出测试分数。

每个 trial 的目录是 `ckpt/joint_training/<category>/<study_name>_t000/`，里面仍是原来的 config、checkpoint 和 TensorBoard。Study 目录保存 `optuna.db`、实际使用的 `search_config.json`、`search.log` 和 `summary.json`。全部 trial 都没有完成时进程返回非零。

## 计算图与更新顺序

```text
item feature -> AE encoder -> raw latent z -> normalization -> latent y
                                    |
                                    +-> AE decoder -> ae_loss

history y -> HSTU / Transformer -> condition c -> RF velocity v
                                                    |
                                                    +-> x_c = x_t - t*v -> InfoNCE with target y
```

decoder 使用 raw latent。历史输入、RF target、contrastive item keys 和检索 catalog 共用同一个下游 LayerNorm；InfoNCE 计算时再对 anchor/key 做 L2 normalize，RF target 不限制在单位球面。

每个 macro-step 的顺序为：

1. 从训练 unique-item 集合均匀抽样，计算 `ae_loss`。
2. 开启 contrastive 时，冻结 HSTU+RF 参数，保留经过历史编码器和 item keys 到 AE encoder 的计算图；AE optimizer 更新 `ae_loss + weight * contrastive_loss`。
3. 使用更新后的 AE 重新编码并 detach，推荐 optimizer 只根据 RF 主损失更新 HSTU+RF。

| Loss | AE encoder | AE decoder | 历史编码器 | RF |
| --- | --- | --- | --- | --- |
| `ae_loss` | 更新 | 更新 | 不更新 | 不更新 |
| AE-side contrastive | 更新 | 不更新 | 参数冻结，输入可求导 | 参数冻结，输入可求导 |
| `rf_loss` | detach 阻断 | 不更新 | 更新 | 更新 |

推荐阶段的 RF 主损失覆盖全部有效标签。`contrastive.max_queries` 只限制 AE 更新中的辅助对比项，不限制推荐监督。

冻结模块参数不等于关闭其输入梯度。AE-side contrastive 前向不能被整个 `torch.no_grad()` 包住。推荐侧必须重新编码，不能复用 AE optimizer 更新前的旧 latent。

默认关闭 contrastive 时，AE 只接受 `ae_loss`，推荐模型持续学习正在变化的 AE 表示。这是同一循环中的协同训练基线，推荐监督尚未通过梯度进入 AE。它可以检验运行流程和表示漂移，但不能被表述为已经完成 collaborative feedback。

## Latent Normalization

下游 latent 固定为无 affine 的 LayerNorm：对每个 item 独立沿 latent 维标准化，不维护 catalog running statistics。历史输入、RF target、contrastive keys 与检索 catalog 共用这一变换。验证/测试时在当前 AE 状态下重建完整检索 catalog；不能把旧 AE 的静态 latent 文件当成新模型的检索库。

## 可选 Contrastive Loss

对比学习可关闭；启用后固定使用 AE-only InfoNCE：

\[
q_b=\operatorname{normalize}(x_{c,b}),\qquad
k_j=\operatorname{normalize}(y_j),\qquad
L_{\rm ctr}=-\frac1B\sum_b\log
\frac{\exp(q_b^\top k_{i_b}/\tau)}{\sum_j\exp(q_b^\top k_j/\tau)}.
\]

对比 anchor 直接使用 RF 当前训练状态恢复出的 clean latent `x_c`，不新增 projection head；正例和负例都使用当前 AE encoder 的 latent。

对比学习使用 RF 训练时同一次随机噪声和时间步得到的速度预测，并按 `x_c=x_t-t·v` 恢复 clean latent 后计算 InfoNCE。每个 query 只有对应 ground-truth item 为正例；负例在每个 query 内去重，并排除该用户输入及标签中出现过的所有 item。

对比损失只参与 AE 更新，冻结 HSTU+RF 参数但保留通过历史 encoder 到 AE encoder 的输入梯度。没有回传模式选择参数。`contrastive.weight` 控制损失权重，`contrastive.num_negatives` 控制每个 query 的负例数。

RF 的 MSE 监督与 AE-side InfoNCE 辅助项分别记录。

默认温度为 `0.04`，对比项权重为 `0.01`，启用后从第一步生效。关闭 contrastive 或冻结 AE 会跳过对比计算。旧 AE-only checkpoint 中的模式和调度字段在加载时自动移除，新配置不再接受这些字段。

推荐监督使用每条历史的全部有效标签位置；contrastive 从这些位置中选取 query，超过 1024 时均匀抽样。每个 query 保留自己的正例，并从完整 catalog 均匀无放回抽取默认 64 个负例，排除当前样本窗口 input/label 中的 item，不使用 batch 内共享 softmax 或历史队列。AE-side 的全部正、负 keys 保留 encoder 梯度；可用负例不足时统一降低负例数，无可用负例则本次对比项为零。

每个 query 的负例从 catalog 中采样，query 内不重复，并排除该用户完整 input/label sequence 及 ground-truth target。不同 ID 但语义相近的真实 false negatives 仍是本方案的实验限制。

## 续训、评估与验收

```bash
# 继续训练，恢复联合训练状态。
.venv/bin/python -m joint_training --resume \
  ckpt/joint_training/Beauty/beauty_layernorm_no_contrastive/latest.pt --device=cuda:0

# 只评估指定联合 checkpoint。
.venv/bin/python -m joint_training --eval-only \
  ckpt/joint_training/Beauty/beauty_layernorm_no_contrastive/best.pt --device=cuda:0

# 多次 diffusion 采样使用 cosine 候选距离，其他评估设置沿用 checkpoint。
.venv/bin/python -m joint_training --eval-only \
  ckpt/joint_training/Beauty/joint_Beauty_contrastive_ae_only_w0.01/best.pt \
  --device=cuda:0 --set evaluation.eval_candidate_metric=cosine
```

`diffusion_multi` 使用 `evaluation.eval_candidate_metric` 计算候选距离；`knn` 使用 `evaluation.tiger_knn_metric`。修改距离度量时需设置当前采样路径对应的字段。

默认输出目录为 `ckpt/joint_training/<category>/<run_id>/`。主要文件为 `config.json`、`train.log`、`metrics.jsonl`、`latest.pt`、`best.pt`、`results.json` 和最终导出的 `catalog_latents.npy`。checkpoint 同时保存 AE、推荐模型、两个 optimizer/scheduler 和训练进度，检索库与 checkpoint 的 AE 必须对应。

续训恢复原定训练预算与优化日程，不能修改 `training.epochs/max_steps/seed`、模型结构、两侧 optimizer 配置或 contrastive 语义。`epochs` 是原计划总轮数。设备可以调整，续训继续写入 checkpoint 所在目录；新的优化配置或预算应另开实验。只评估可改变 `evaluation` 设置，重建对应 checkpoint 的检索 catalog，并输出 `eval_results.json`。

`--smoke-test` 使用小规模数据和推荐网络验证程序，不用于指标比较。AE 仍随机初始化，不缩小其 latent/hidden dimensions。

smoke 的 Recall/NDCG 只用于确认评估可以运行，不能据此评价模型性能。

验收应覆盖：

- 无 contrastive 时 `rf_loss` 不产生 AE 梯度；`ae_loss` 只更新 AE。
- AE-side contrastive 对 history encoder 路径和正、负 candidate keys 可求 AE 梯度，但不更新推荐参数。
- 更新前后使用正确的 fresh latent，padding 不进入 AE、监督或检索。
- 重复 target、单 candidate 和 all-positions query cap 均正确处理。
- 保存、续训、只评估与最终 catalog 导出能恢复对应模型状态。

可比较关闭与开启 AE-only contrastive 的结果，再调整权重、温度和随机种子。比较推荐 Recall/NDCG、AE 重建、latent 方差/相关性/漂移、梯度和运行开销。

`ae_loss` 覆盖推荐训练 unique items 后，单独预训练阶段拆出的 AE validation/test items 可能已成为联合训练样本；其重建分数只能作为固定监控结果，不再是联合训练的 held-out 泛化评估。联合 checkpoint 选择使用推荐 validation，最终报告 test 结果。
