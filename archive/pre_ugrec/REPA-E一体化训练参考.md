# REPA-E 风格单阶段 VAE + 下游模型训练参考

## 1. 核心思想

传统做法通常是两阶段：

```text
Stage 1: 训练 VAE
Stage 2: 冻结 VAE，训练下游模型
```

REPA-E 的思路是把两者放到同一个 training loop 中联合训练，但**不是让所有 loss 对所有模块直接反传**。

核心原则：

```text
VAE 自己的 loss
    ↓
更新 VAE

Alignment / Bridge loss
    ↓
可以更新 VAE Encoder

下游主任务 loss
    ↓
只更新下游模型
    ×
不允许直接回传 VAE
```

也就是：

> VAE 和下游模型同时训练，但通过明确的 gradient routing 控制各个 loss 的作用范围。

---

# 2. 基本计算图

```text
Input
  ↓
VAE Encoder
  ↓
latent z
  ↓
BatchNorm
  ↓
Downstream Model
  ↓
Task Loss
```

同时：

```text
Input
  ↓
VAE Encoder
  ↓
latent z
  ↓
VAE Decoder
  ↓
VAE Loss
```

以及一个跨阶段的 Alignment / Bridge Loss：

```text
VAE Encoder → z → BN → Downstream intermediate feature
   ↑
   └──────── Alignment Loss
```

---

# 3. Loss 的作用范围

假设：

```text
L_vae   = VAE 自己原本的训练 loss
L_task  = 下游模型主任务 loss
L_align = VAE 与下游模型之间的 representation alignment loss
```

## 更新 VAE 时

使用：

\[
L_{\text{VAE}}
=
L_{\text{vae}}
+
\lambda_v L_{\text{align}}
\]

梯度范围：

```text
L_vae
  ↓
VAE Encoder / Decoder

L_align
  ↓
Downstream Model（只作为计算路径，不更新参数）
  ↓
BN
  ↓
VAE Encoder
```

注意：

> Alignment loss 主要用于调整 Encoder 产生的 latent，使其更适合下游模型。

---

## 更新下游模型时

使用：

\[
L_{\text{downstream}}
=
L_{\text{task}}
+
\lambda_d L_{\text{align}}
\]

但输入必须：

```python
z_task = z.detach()
```

因此：

```text
L_task
  ↓
Downstream Model
  ×
VAE

L_align
  ↓
Downstream Model
  ×
VAE
```

即：

\[
\frac{\partial L_{\text{task}}}
{\partial \theta_{\text{VAE}}}
=0
\]

这是 REPA-E 最重要的设计之一。

---

# 4. 为什么主任务 Loss 不直接训练 VAE

不要直接：

```text
Task Loss
   ↓
Downstream
   ↓
VAE Encoder
```

因为 VAE 可能为了让下游任务更容易，而主动改变甚至简化 latent distribution。

因此 REPA-E 风格采用：

```text
Task Loss
   ↓
Downstream
   ↓
z.detach()
   ×
VAE
```

而通过一个更稳定的：

```text
Alignment Loss
```

去指导 VAE latent 如何适配下游模型。

可以简单理解为：

```text
Task Loss:
负责训练下游模型

Alignment Loss:
负责连接 VAE 和下游模型
```

---

# 5. BatchNorm 的作用

因为 VAE 在训练过程中不断更新：

```text
VAE Encoder_t0 → latent distribution A
VAE Encoder_t1 → latent distribution B
VAE Encoder_t2 → latent distribution C
```

latent 的均值和方差会持续变化。

如果下游模型直接使用这些不断漂移的 latent，训练会比较不稳定。

所以在两者之间加入：

```text
VAE Encoder
    ↓
latent z
    ↓
BatchNorm
    ↓
Downstream Model
```

BN 的主要作用是：

> 动态跟踪 VAE latent 的 mean / variance，使下游模型始终看到尺度相对稳定的 latent。

---

# 6. BN 推荐设置

参考 REPA-E：

```python
nn.BatchNorm*(
    ...,
    affine=False,
    track_running_stats=True,
)
```

关键：

```text
affine=False
```

即不学习额外的：

```text
gamma
beta
```

BN 主要负责：

```text
统计 + normalization
```

而不是再学习一个 latent transformation。

如果是：

```text
[B, C, H, W]
```

可以用：

```python
BatchNorm2d(C)
```

如果你的 latent 结构不同，则选择对应的 normalization 实现。

---

# 7. BN 在两个更新阶段的行为

## VAE update

```python
freeze(downstream)
downstream.eval()
```

此时：

```text
BN running mean / variance 不更新
```

但注意：

> 不能使用 `torch.no_grad()` 包住 downstream forward。

因为 Alignment Loss 仍然需要：

```text
Alignment
   ↓
Downstream operations
   ↓
BN
   ↓
z
   ↓
VAE Encoder
```

计算对 VAE 的梯度。

---

## Downstream update

```python
unfreeze(downstream)
downstream.train()
```

此时：

```text
BN 正常更新 running mean / variance
```

同时：

```python
z_task = z.detach()
```

阻止主任务梯度进入 VAE。

---

# 8. 推荐训练顺序

一个 minibatch：

```text
1. VAE forward
2. 更新 VAE
3. 更新 Downstream Model
```

即：

```text
Input
 ↓
VAE
 ↓
得到 z

 ↓

VAE step:
    L_vae + L_align
    ↓
    update VAE

 ↓

Downstream step:
    z.detach()
    ↓
    L_task + L_align
    ↓
    update downstream
```

这是 minibatch 内的交替更新。

---

# 9. 简化伪代码

```python
for batch in dataloader:

    x = batch["input"]

    # =====================================
    # Shared VAE forward
    # =====================================

    z, reconstruction = vae(x)


    # =====================================
    # 1. Update VAE
    # =====================================

    freeze(downstream)
    downstream.eval()

    optimizer_vae.zero_grad()

    # VAE 自己原本的 loss
    loss_vae = compute_vae_loss(
        x,
        reconstruction
    )

    # 注意：
    # z 这里不能 detach
    # downstream 参数冻结，但保留计算图
    loss_align_vae = compute_alignment_loss(
        z=z,
        downstream=downstream,
    )

    loss = (
        loss_vae
        + lambda_align_vae * loss_align_vae
    )

    loss.backward()
    optimizer_vae.step()


    # =====================================
    # 2. Update downstream model
    # =====================================

    unfreeze(downstream)
    downstream.train()

    optimizer_downstream.zero_grad()

    # 核心：
    # 主任务不允许梯度进入 VAE
    z_task = z.detach()

    output = downstream(z_task)

    loss_task = compute_task_loss(output)

    loss_align_downstream = compute_alignment_loss(
        z=z_task,
        downstream=downstream,
    )

    loss = (
        loss_task
        + lambda_align_downstream
          * loss_align_downstream
    )

    loss.backward()
    optimizer_downstream.step()
```

---

# 10. 最关键的梯度规则

最终只需要记住这张表：

| Loss | VAE | Downstream |
|---|---:|---:|
| VAE 自身 Loss | ✅ | ❌ |
| VAE-side Alignment | ✅ | ❌ 参数冻结 |
| Main Task Loss | ❌ `detach` | ✅ |
| Downstream-side Alignment | ❌ `detach` | ✅ |

核心关系：

```text
Main Task Loss
    ×
    VAE

Alignment Loss
    ↓
    VAE Encoder
```

---

# 11. 给 Agent 的实现要求

实现时重点检查以下几点：

- VAE 和 downstream 使用独立 optimizer。
- 两者在同一个 training loop 中交替更新。
- 主任务阶段必须使用 `z.detach()`。
- VAE Alignment 阶段不能 detach `z`。
- VAE Alignment 阶段 downstream 参数冻结，但不能使用 `torch.no_grad()`。
- VAE update 时 downstream 使用 `eval()`，避免 BN statistics 被更新。
- Downstream update 时使用 `train()`，BN 正常更新 statistics。
- BN 主要用于追踪不断变化的 latent distribution。
- BN 推荐 `affine=False`，避免引入额外可学习 latent transformation。
- 不要简单把所有 loss 相加后一次 backward。

整体框架可以概括为：

```text
VAE-local Loss
      ↓
     VAE

Alignment Loss
      ↓
VAE ↔ Downstream

Main Task Loss
      ↓
 Downstream only
```

也就是：

\[
\boxed{
\text{Joint Training}
+
\text{Controlled Gradient Routing}
+
\text{Dynamic Latent Normalization}
}
\]

这就是需要参考 REPA-E 的主要部分。