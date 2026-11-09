# UGRec_v5_4：FlowRec 三损失实验

基于 v5_1 的独立版本，将推荐损失替换为 `Prior CE + alpha*CFM + beta*Align CE`。默认 `configs/final.json` 是 alpha=5、beta=1、确定性历史先验、一次Euler生成；保留原有内容AE约束。`configs/control.json` 保留原噪声边缘似然目标用于本版重新训练的对照。

这次移植保持内容AE、HSTU、条件速度网络和cosine/T打分，采用FlowRec三项目标和随机时间桥上的单步终点对齐，不是官方FlowRec结构或指标复现。精确定义、适配差异、预注册对照和调参边界见 [实验设计](docs/FlowRec实验设计.md)。本版研究结果只写入本版outputs；共享cache和旧版只读。

```bash
cd /home/haoquan/CID_RPG/UGRec_v5_4
bash scripts/test.sh
bash scripts/python.sh run.py --config configs/final.json --device cpu --smoke-test --run-id flowrec_smoke_new
GPU_ID=5 bash scripts/train.sh --run-id flowrec_new_s2024
```

探索使用Beauty/seed2024、40轮、固定4096人validation选模、完整validation审计，不用test。运行前检查GPU占用，已有run id禁止覆盖。

```bash
bash scripts/python.sh research/flowrec_campaign.py --stage initial --device cuda:5
# initial结束并分析后，必要CFM消融
bash scripts/python.sh research/flowrec_campaign.py --stage no_cfm --device cuda:5
```

执行状态与结果在 `outputs/flowrec/`。`low_cfm`是按实验设计条件触发的单项探索，不默认启动。
