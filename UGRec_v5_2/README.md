# UGRec_v5_2：历史排序 CE 与条件 Flow Matching 解耦

本版探索：**历史 CE 已提供排序能力后，条件速度场能否提供排序之外的有效候选覆盖？** 双流重训和候选预算审计已完成，当前结果仍不支持保留FM；分析见[双流重训与候选预算实验报告](docs/双流重训与候选预算实验分析_20261106.md)。

Beauty/seed2024单种子探索：CE NDCG@10为0.06041451；串联联合FM为0.04784320；冻结CE后40epoch重训FM为0.04642463；再额外80epoch续训后为0.04666063。固定唯一候选预算下，FM端点候选不如更深CE检索或未运输先验。保留纯CE作为研究参照。结果是含选模用户的validation趋势，没有多种子或test结论。

源码从 v5 control 物理复制，数据读取采用 v5_1 已校验协议的独立副本。没有跨版本导入，不改动正在调参的 v5_1。只共享原始 cache 和 Python 环境，新结果全部写入本版 `outputs/`。

下面是保留的原双尾部设计，尚未执行正式性能对照：

```text
内容 → AE → 输入投影 → HSTU共享主干（1层）
                        ├→ 排序HSTU尾部（1层）→ h_rank → 排序投影 → CE
                        │                             └→ sg(anchor)+σε = 源点
                        └→ sg(共享输出) → FM尾部（1层）→ h_fm
                                                         ↓
                    sg(目标物品)与源点的随机时间直线桥 → 速度网络 → MSE
```

`sg` 为停止梯度。默认 CE 更新 AE、输入投影、共享主干、排序尾部和排序头；MSE 更新 FM 尾部、FM 条件投影与速度网络。生成结果不接 CE，原 AE 几何约束保留。

训练每监督位置只采一个噪声/时间点；推理默认4步 Euler、32个源噪声样本，平均全目录 softmax 概率。主结果只用 FM；排序头/未运输先验是同 checkpoint 的诊断，不加权融合。**4×32意味着每用户128个速度查询，不能与 control 的1×32当作同推理成本。**

在本目录执行：

```bash
bash scripts/test.sh
# CPU小模型端到端检查；不是benchmark，已有run-id拒绝覆盖。
bash scripts/python.sh run.py --device cpu --smoke-test --run-id fm_smoke_s2024

# 纯CE研究参照；复现使用新的run-id，先确认GPU资源。
GPU_ID=5 bash scripts/train.sh --config configs/ce_only.json --category Beauty --run-id beauty_ce_only_s2024_new
```

已完成89项测试、匹配CE/联合实验、冻结CE后的FM重训与续训，以及固定候选预算审计。原 [初步实验分析](docs/初步实验分析_20261106.md) 保留训练目标比较；新候选分析和当前取舍见[双流重训报告](docs/双流重训与候选预算实验分析_20261106.md)。`final.json`保留原双尾部入口，不表示性能已验证。新增续训与诊断代码独立于生产训练路径。

| 配置 | 结构 | MSE更新范围 | 用途 |
|---|---|---|---|
| `ce_only.json` | 同一2层HSTU输出 | 无有效MSE梯度 | 本轮研究参照，CE读出/选模 |
| `final.json` | 共享1层＋两个1层尾部 | FM尾部、条件投影、速度网络 | 保留原入口，双尾部性能待验证 |
| `serial_shared.json` | 同一2层HSTU输出 | HSTU、条件投影、速度网络；不进AE/先验 | 串联起点 |
| `parallel_shared.json` | 共享1层＋两个1层尾部 | 共享主干、FM尾部、速度网络；不进AE/排序尾部/先验 | 区分拆分与梯度隔离 |
| `serial_flow_only.json` | 同一2层HSTU输出 | 仅速度网络 | 按证据再检验小模块是否足够 |
| `control.json` | 旧v5 control | 原一步生成的排序似然 | 显式旧路线参考，不是新模型匹配消融 |

以上均在 `configs/`。通过 `--set fm.mse_scope=...` 等字段调整时会做类型与兼容性检查。机制与限制见 [方案](docs/方案.md)，对照次序见 [实验计划](docs/实验计划.md)。旧 `frozen/matched/minimal/direct_reference` 配置及测试只用于回归。

每个run保存展开配置、`experiment.json`、源码快照/指纹、`metrics.json`、`best_flow.pt`、选模预测和结果；`--audit full` 重载并核验选模预测后评价全validation，并保存排序头/先验诊断。测试标签在ID映射前剔除。
