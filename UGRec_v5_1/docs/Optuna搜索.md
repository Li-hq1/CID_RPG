# Optuna 搜索

2026-11-09。工具从 v4 的 `joint_training.search` 迁到本版，训练改走 `run.py`。第一轮 study 已写好，还没有启动。

## 和 campaign 的差别

Optuna 最大化固定 4096 名验证用户上的选模 NDCG@10，也就是训练里的 `best_val_score`。每个 trial 使用 `--audit subset`，不跑全量 validation，不读 test。剪枝看的是到当前轮为止的最好选模分数；最后一轮不参与剪枝。

因此 study 里的名次不是完整 validation 结果，也不能直接采用。选出候选后，仍要用该配置做一次单数据集 campaign，跑满声明的 epoch 并完成全量审计，再和当前 reference 比较。

## 第一轮空间

数据集只有 Beauty。模型不是 flow control，而是 HSTU 输出头上的直接 CE：`configs/direct_ce.json`。构造时仍按 control 的顺序初始化共享的 AE、HSTU、条件投影和输出头，然后删掉 `diffloss` 和 `uncond_condition`，优化器和 checkpoint 里没有 FM 参数。预测是一次 `cos(a, Z) / T` 的目录 CE，没有源噪声，也没有速度网络。AE 几何项权重仍是 0.1。起点是一层 HSTU。第一个 trial 就是这个配置。

| 参数 | 取值 |
|---|---|
| `recommender.n_layer` | 1 / 2 / 3 / 4 |
| `recommender.n_embd` | 64 / 128 / 256 |
| `recommender.n_head` | 2 / 4 / 8 |
| `recommender.hstu_hidden_dim` | 16 / 32 / 64 |
| `recommender.hstu_attention_dim` | 16 / 32 / 64 |
| `training.epochs` | 40–80，步长 10 |
| `recommender.lr` | 1e-4–2e-3，对数 |

宽度和 heads 的每一种组合都能整除。attention 宽度都是偶数。epoch 不低于 40，并且是选模间隔 5 的倍数。学习率只动推荐器，`ae.lr` 保持 0.0001。

不搜 AE、Flow、投影、dropout、weight decay、warmup、batch 和 loss 权重。本轮关闭剪枝：v4 的中位数剪枝会在峰值出现前停掉收敛较慢的 trial，而这一轮要比较的正是不同收敛速度。配置里仍保留 warmup 字段，改回 `median` 时才会用到。

一次默认新跑 20 个 trial。同一个 study 名再执行会接着搜，已完成的不重跑。改参数范围、sampler 或 pruner 后必须换 study 名。

## 入口

```bash
cd /home/haoquan/CID_RPG/UGRec_v5_1
GPU_ID=4 bash scripts/search_beauty_hstu.sh

# 这一次只再跑 4 个 trial
GPU_ID=4 bash scripts/search_beauty_hstu.sh --n-trials=4
```

`GPU_ID` 会设成 `CUDA_VISIBLE_DEVICES`，进程内设备仍是 `cuda:0`。先看卡上是否有空余显存。结果在 `outputs/search/Beauty/beauty_hstu_epoch_lr/`，每个 trial 的目录是同级的 `beauty_hstu_epoch_lr_t000/`。

单种子 2024。20 个 trial 大约是数小时到十几小时，视 epoch 和卡上其他进程而定。这次没有启动。
