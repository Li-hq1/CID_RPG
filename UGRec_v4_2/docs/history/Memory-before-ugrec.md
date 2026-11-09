# Memory

## 5880 1 号

- 登录：`ssh -p 10404 haoquan@172.18.36.60`（本机已有 SSH key）
- 项目：`/home/haoquan/CID_RPG`（2026-10-21 从本机整目录同步，含 `cache/`、`ckpt/`、`.venv`）
- 环境：项目内 `.venv` 的解释器指向 `~/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu`，`uv 0.11.16` 在 `~/.local/bin/uv`。`.profile` 只给登录 shell 加这条 PATH；交互式终端走 `.bashrc`。2026-10-21 已在 `.bashrc` 末尾补上 `~/.local/bin`，已打开的终端需 `source ~/.bashrc` 或 `export PATH="$HOME/.local/bin:$PATH"`。
- 已验证：Python 3.12.13，PyTorch `2.12.0+cu130`，`torch.cuda` 可见 8 张 RTX 5880。
- 启动：`cd ~/CID_RPG && uv run train.py --category=Sports_and_Outdoors`
- Joint contrastive 超参搜索：`GPU_ID=0 bash scripts/search_joint_contrastive.sh`。范围在 `joint_training/search_space.json`。Optuna 目标是 best.pt 的测试集 `evaluation.val_metric`（默认 ndcg@10）；验证集只用于选 checkpoint 和 pruner。结果在 `ckpt/joint_training/<category>/<study_name>/`。改搜索空间或目标后要换 study name。
- Beauty `joint_Beauty_contrastive_search`（2026-10-22，约 15.6 小时）：53 个 trial。trial 2 在 epoch 140 被掐掉后脚本又按 `n_trials=50` 续跑，所以是 0–1 完成、2 失败，再加上 3–52。最好仍是 trial 0 基线，测试 ndcg@10=0.0319、recall@10=0.0568；次好 trial 30 为 0.0290。`load_study` 没有把 pruner 写回，续跑用的是 Optuna 默认 MedianPruner（warmup 0），22 个 trial 在 epoch 5 被剪掉。配置里的 `pruner_warmup_steps=30` 没生效。
