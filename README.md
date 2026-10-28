# ADRec

## Quick Start
1. Install the dependencies by running the following command:

```
uv sync
```

2. Run the following command to start training the model with a specified category:
    
```
CUDA_VISIBLE_DEVICES=0 uv run train.py --category=Sports_and_Outdoors
```

Available categories:
* `Sports_and_Outdoors`
* `Beauty`
* `Toys_and_Games`
* `CDs_and_Vinyl`

Data is downloaded and processed if the local category cache is missing.
Edit the single flat `CONFIG` dictionary in `config.py` for experiment defaults.
Common overrides are explicitly listed by `uv run train.py --help`; unknown
arguments are rejected. There is no YAML inheritance or dynamic model lookup.

The launcher is `scripts/train_tigerdiff.sh`. It sets the category, run ID,
and visible GPU; experiment parameters come from
`config.py`. Defaults match the previous launchers (`lr=0.0005`,
`eval_batch_size=64`, `diffloss_w=512`), rather than the old bare `main.py`
defaults (`0.0003`, `32`, `1024`).

## Reading and Modifying the Code

Start with `train.py`: configuration, Accelerator, data splits, tokenizer,
model, DataLoaders, training, and final evaluation are created in order with
explicit imports.

| Directory | Responsibility |
| --- | --- |
| `engine/` | Ordinary functions for the epoch/batch loop, evaluation, metrics, and checkpoints |
| `models/` | TigerDiff encoder and Gaussian diffusion / rectified flow |
| `data/` | Amazon data processing, leave-one-out splits, embedding preparation, sample windows and caches |
| `misc/` | CLI validation, logging, text processing, seeds and device setup |
| `scripts/` | Category training launchers |
| `vis_analysis/` | Embedding and attention analysis, plus training-log plots |
| `tests/` | Tests of the active continuous recommendation code |

The core training loop is directly in `engine/train.py`; there is no active
Pipeline, Trainer, abstract base class, or registry. Data and model constructors
still receive a plain configuration dictionary, so local experiments can create
and modify one directly.

`supervision_mode` defaults to `all_positions`: causal attention prevents future
items from entering each history state, and every valid shifted label is sent to
the existing diffusion / rectified flow loss. Loss is averaged over valid target
positions. Later sliding windows still supervise only their last position, so
targets are not duplicated. Validation and generation still use the final history
state. The tokenizer format and existing sample caches are unchanged.

Both `all_positions` and `last_item` use causal attention. Select
`--supervision_mode=last_item` for last-target-only training.

The default history encoder is HSTU (`models/hstu.py`). Select
`--encoder_type=transformer` for the Transformer baseline. HSTU uses only RoPE,
with valid positions counting down to 1 and padding at 0; Transformer retains
its learned absolute position embeddings. Both return `[B,T,n_embd]` to the
same condition projection and diffusion loss. HSTU keeps the source model's
SiLU attention, normalized gate and raw-attention concatenation, without user
tokens, action modulation or timestamp biases. Its adjacent-pair RoPE uses
matching adjacent-pair frequencies.

`hstu_hidden_dim=64` and `hstu_attention_dim=64` are per-head U/V and Q/K
dimensions, independent of `n_embd/n_head`. HSTU defaults to group norm,
output dropout `0.1` and attention dropout `0.0`. All HSTU options have CLI
overrides. `--hstu_attention_scaling=input_length` (default) divides SiLU
weights by the input tensor length, including padding, matching the source.
`--hstu_attention_scaling=max_seq_len` divides by `max_item_seq_len`, preserving
prefix consistency when inputs are cropped. For batches padded to that maximum,
the two modes are identical.

Configurations without `encoder_type` construct Transformer for compatibility.
Use `--encoder_type=transformer` explicitly when loading Transformer checkpoints;
their parameter keys and shapes are retained, but formerly bidirectional
`last_item` experiments now run causally. HSTU must be trained from scratch.
Use distinct `run_id` values for architectures and experiments. The default
run ID and launcher include `hstu`; the launcher also accepts
`ENCODER_TYPE=transformer` with a separate default run ID.

The complete pre-HSTU configuration is preserved in `config_transformer.py`.
Use `uv run train.py --config=config_transformer.py` to load those defaults,
or add `--resume` / `--checkpoint` to restore a Transformer checkpoint.
CLI overrides still apply; omitting `--config` retains the current HSTU defaults.

Attention analysis reads saved checkpoint configuration when available and
accepts encoder/HSTU CLI overrides. Transformer plots show softmax probabilities;
HSTU plots show signed masked SiLU weights with a diverging color scale.

The learning rate uses linear warmup followed by cosine decay to zero, updated
each optimizer step. `warmup_ratio=0.05` reserves the first 5% of the planned
training budget (about 7.5 of 150 epochs); warmup steps are rounded up from
`total_steps * warmup_ratio`. Override with `--warmup_ratio=0.02`, or select a
fixed count with `--warmup_steps=1000` (`0` disables warmup). A fixed step count
takes precedence. Smoke tests disable warmup. Early stopping does not shorten
the originally planned schedule, and resume restores the scheduler position.

RPG and the previous framework are preserved in the corresponding
`models/backup/`, `engine/backup/`, `data/backup/`, `misc/backup/`, and
`tests/backup/` directories. These are reference snapshots with their original
imports, not runnable or maintained modules. Backups have no `__init__.py` and
are excluded from unittest discovery.


> [!IMPORTANT]
> TODO: Tune the hyperparameters.

## Local Data and Checkpoints

The four Amazon categories, processed metadata, and sentence-T5 embeddings have
been copied from `~/RPG_KDD2025` into `cache/AmazonReviews2014/`. Tokenized
datasets are regenerated by this project's tokenizer to match its label policy.
The local sentence-T5 model is in
`cache/models/sentence-transformers/sentence-t5-base`; `sent_emb_model_paths`
in `config.py` maps the model ID to this directory for offline loading.

New experiments default to the locally downloaded
`cache/models/Qwen/Qwen3-Embedding-4B`. Item metadata is encoded as documents
without the search-query prompt. The encoder uses left padding, SDPA attention,
BF16 on supported GPUs (FP32 on CPU), batches of 8 and a 512-token limit.
Raw normalized vectors are saved as float32 with 2560 dimensions. The existing
training-only PCA and standardization then produce 128-dimensional TigerDiff
features; the encoder is released before recommender training.

Extract a category once without starting training:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m data.extract_embeddings --category=Beauty
```

Add `--smoke-test` to encode only four items into a temporary file. For memory
constraints, use `--sent_emb_batch_size=2` or `--sent_emb_max_length=256`.
Training automatically generates missing embeddings and reuses complete caches:

```bash
CUDA_VISIBLE_DEVICES=0 uv run train.py --category=Beauty --run_id=tigerdiff_beauty_qwen3
```

Raw cache names include the model and a hash of encoding settings; tokenized
cache keys also record those settings. Changing prompts, length, dimension or
normalization creates a separate cache. Local model files are required by
default, so extraction does not need network access. Legacy sentence-T5 files
are preserved. To reproduce a legacy experiment, use its original embedding
configuration (768 dimensions, no added prompt/normalization/length settings)
and a separate run ID. Retrain for Qwen embeddings: old checkpoints contain
the old item vectors and should not be resumed for a new embedding experiment.

Previous checkpoints are preserved in `ckpt/legacy_RPG_KDD2025/`. Use the matching
model architecture and category when loading them. For the Sports TigerDiff
checkpoint, use one explicit file with matching architecture:

```bash
bash scripts/train_tigerdiff.sh --smoke-test \
    --checkpoint ckpt/legacy_RPG_KDD2025/matching-TigerDiff-file.pth
```

`--smoke-test` trains one epoch on 64 examples per split and runs validation and
test evaluation. Remove it for a full training run. To train from scratch:

```bash
bash scripts/train_tigerdiff.sh
```

Model keys and cache names are preserved. Old checkpoints with different
architectures still require matching parameters. Old plain model state
dictionaries can still be loaded.

## Geometry-Preserving Autoencoder Compression

`geometry_ae` independently compresses item content to 128 dimensions. Its
single encoder and decoder use GELU hidden layers without BatchNorm, dropout,
masking, fixed noise, or KL. The encoder widths are 2560 -> 1024 -> 512 -> 128;
`--architecture=linear` selects the linear diagnostic model. Input centering and
one global RMS scale are fitted only on the compression training split. Exported
latents are standardized on recommendation training items and remain frozen.
PCA stays the default until paired recommendation experiments demonstrate an
improvement. Existing SigmaVae artifacts and objectives remain available.

Train one candidate:

```bash
bash scripts/train_geometry_ae.sh --device=cuda:0
# Pipeline check on CPU; this does not measure compression quality:
uv run python -m compression.geometry_ae --category=Beauty --device=cpu --smoke-test
```

The objective is normalized reconstruction MSE plus 0.1 cosine reconstruction.
After ten epochs, variance (0.1), off-diagonal correlation (1.0), and batch relation
KL (0.1, temperature 0.1) ramp up over twenty epochs. Relation teachers use only
Qwen item content. Per-term encoder gradient norms are measured on the first
batch each epoch; no automatic loss reweighting is applied.

Opt into local soft-label relation distillation:

```bash
bash scripts/train_geometry_ae.sh --category=Beauty --device=cuda:0 --relation_mode=local
```

The original `--relation_mode=batch` remains the default, with unchanged old
configuration fingerprints and checkpoint loading. Local mode creates a separate
artifact. Its Qwen cosine neighbor index contains only compression training
items. Each uniformly sampled training anchor receives 8 candidates sampled from
ranks 1-32, 8 from ranks 33-128, and 16 random training items. Candidates are
unique and exclude the anchor; small populations use fewer candidates. Adjust
these with `--local_near_pool`, `--local_middle_pool`, `--local_near_count`,
`--local_middle_count`, and `--local_random_count`.

Teacher probabilities are the softmax of original Qwen cosine similarities to
these candidates; student probabilities use standardized latent cosine. The
temperature and KL weight still default to 0.1. This replaces the batch relation
training term while keeping reconstruction, cosine, statistics, and the existing
warmup schedule. Reconstruction/statistical losses use uniform anchors only.
Both anchors and candidates use detached latent mean/std fitted on all compression
training items at each epoch's start; candidate encoder gradients remain enabled.
Candidate items are encoded once per batch, without decoding extra items.

Validation candidates are fixed and come only from the compression training
population. Local checkpoints are selected by normalized MSE + 0.1 cosine error
+ 0.1 local KL. `loss.csv` and `metrics.json` retain the original batch
`relation_kl` for comparison and add `local_relation_kl`; `train_relation` and
`encoder_grad_relation` track the selected training objective. Local evaluation
uses current training-only latent statistics. Local training adds encoder work
and memory use; reducing `--train_batch_size` or candidate counts lowers the cost.
The command prints the exported directory and a recommendation command with a
distinct run ID. Point `--geometry_ae_cache` at that new directory for downstream
recommendation; the downstream interface is unchanged.

Disable relation distillation for a reconstruction/statistics-only control:

```bash
bash scripts/train_geometry_ae.sh --category=Beauty --device=cuda:0 --relation_mode=none
```

`none` skips batch/local relation losses and local candidate sampling. Validation
selection uses normalized MSE + 0.1 cosine error, with no relation term; logged
`relation_kl`, `train_relation`, and `encoder_grad_relation` are zero. Reconstruction,
cosine, variance/correlation regularizers, and neighborhood diagnostics stay enabled.
This mode creates a separate artifact; default batch and existing local artifacts
remain compatible. Setting only `--relation_weight=0` in batch/local mode still
uses relation KL during validation selection, so use `--relation_mode=none` to
disable distillation throughout training and model selection.

Run the staged experiment protocol:

```bash
# Mean/PCA baselines and five AE ablations, 4096 train items, 100 epochs:
uv run python -m compression.geometry_ae.experiments --category=Beauty --device=cuda:0
# Also train the validation-selected winner for up to 200 epochs on the full split:
uv run python -m compression.geometry_ae.experiments --category=Beauty --device=cuda:0 --full
# Also run PCA/AE recommendation comparisons at seeds 2024, 2025, 2026:
uv run python -m compression.geometry_ae.experiments --category=Beauty --device=cuda:0 --full --recommendation
```

All ablations retain the same validation/test item splits. Checkpoints and the
winner are selected by validation normalized MSE + 0.1 cosine error + 0.1 relation
KL, regardless of the candidate's training weights. Test metrics are reported
only for assessment. Recommendation runs use matching HSTU/cosine KNN settings,
100 epochs without early stopping, and fresh run IDs. `recommendation.json`
reports means and population standard deviations; AE is accepted only when both
mean NDCG@10 and Recall@10 meet or exceed PCA. This does not change the default
configuration automatically. An interrupted recommendation job must be resumed
using its printed run ID before retrying the experiment command.

Artifacts live under `cache/AmazonReviews2014/<category>/Geometry_AE/`; partial
training ablations and smoke checks have separate subdirectories. Each complete
artifact contains `best.pt`, item splits, raw/standardized latents, input/export
statistics, `loss.csv`, `diagnostics.json`, `metrics.json`, and a checksum manifest.
Diagnostics track per-dimension standard deviations, raw/standardized effective
and participation ranks, correlations, decoder output variance, and fixed sampled
neighbor overlaps every five epochs. Five consecutive effective ranks below 64
produce a warning without changing weights. New model/loss/preprocessing settings,
source values, item order, or training population create a separate identity.

Use the directory printed by pretraining:

```bash
uv run train.py --category=Beauty --sent_emb_compression=geometry_ae \
  --geometry_ae_cache=cache/AmazonReviews2014/Beauty/Geometry_AE/<fingerprint> \
  --run_id=tigerdiff_beauty_geometry_ae
```

Recommendation checkpoints validate both compression method and artifact identity
before loading weights. Smoke features require `--smoke-test`. Export-only reloads
the selected model without retraining:

```bash
uv run python -m compression.geometry_ae --category=Beauty --export-only \
  --artifact-dir=cache/AmazonReviews2014/Beauty/Geometry_AE/<fingerprint>
```

## ContRec SigmaVae Compression

The independent `compression/contrec_vae/` package adapts
`ContRec/code/vae/vae.py`; the reference source is not imported or modified.
This is ContRec's reconstruction-only SigmaVae: two encoders, fixed noise drawn
once at initialization, and MSE loss. It does not learn a posterior variance or
use KL regularization. The two 2048-dimensional branches are averaged, not
concatenated. Its input is the existing normalized 2560-dimensional
Qwen3-Embedding-4B cache, without PCA or additional input standardization.

Stage 1 runs separately from recommendation training:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m compression.contrec_vae --category=Beauty
```

Defaults reproduce the source architecture and optimization: encoders
`2560 -> 128 -> 256 -> 2048`, decoder `2048 -> 256 -> 128 -> 2560`,
BatchNorm, ReLU and the original Dropout rates, masking probability `0.1`,
noise scale `0.1`, seed `42`, AdamW with learning rate `1e-4` and weight decay
`1e-3`, batches `512/256`, and up to 1000 epochs with patience 1000.
Use `--epochs`, `--patience`, `--lr`, `--train_batch_size`, `--eval_batch_size`
or `--device=cpu` to override training; `--help` lists all options.

Only items appearing in recommendation training sequences are eligible for VAE
fitting: 10% are held out for reconstruction testing, then 20% of the remainder
for validation. Splits are seeded and saved, and the best validation MSE selects
the exported weights. At least four eligible items are required; singleton
training batches are merged to accommodate BatchNorm. Training corrupts a copy
of the input and reconstructs the clean target. Validation/export use eval mode
without masking or the training-only position bias, matching the source.

The command prints its artifact directory. All VAE products are stored under:

```text
cache/AmazonReviews2014/Beauty/ContRec_VAE/<experiment fingerprint>/
  best.pt             # Best weights, persistent noise, configuration and splits
  manifest.json       # Source fingerprints, artifact identity and file checksums
  splits.npz          # Zero-based embedding-row indices for VAE train/valid/test
  loss.csv            # Epoch train/validation reconstruction MSE
  metrics.json        # Reconstruction MSE and cosine similarity for each split
  raw_latents.npy     # float32 [n_items - 1, 2048], row 0 is item ID 1
  latents.npy         # Standardized latents consumed by recommendation training
  statistics.npz      # Mean/std fitted only on recommendation training items
  samples/            # Recommendation sequence caches for this artifact
```

The source fingerprint covers encoding settings, raw feature contents, item
ordering and the recommendation training population; training settings also
select a separate experiment directory. Raw Qwen caches remain shared in the
category's existing `processed/` directory. Complete artifacts are reused;
`--force` retrains the same experiment. Export again from saved best weights with
`--export-only --artifact-dir=<artifact directory>`, using the same category and
encoding settings. Every artifact file is atomically replaced, and the manifest
is written last so incomplete exports cannot be loaded.

Stage 2 explicitly loads the printed directory and keeps the VAE features frozen:

```bash
CUDA_VISIBLE_DEVICES=0 uv run train.py --category=Beauty \
  --sent_emb_compression=contrec_vae \
  --contrec_vae_cache=cache/AmazonReviews2014/Beauty/ContRec_VAE/<experiment fingerprint> \
  --run_id=tigerdiff_beauty_contrec_vae
```

Replace the placeholder with the directory printed by stage 1. The tokenizer
loads `latents.npy` directly, skipping PCA and repeated standardization. The raw
Qwen cache is required to verify source compatibility. Missing, incomplete,
corrupt or mismatched artifacts fail instead of starting VAE training.
Recommendation checkpoints record the exact artifact identity and reject
weights from another artifact, including old PCA checkpoints. Use a separate
`run_id` for each experiment. The default remains `sent_emb_compression=pca`.

For a quick pipeline check, add `--smoke-test` to the VAE command: it trains two
epochs on at most 64 eligible items, then encodes the full catalog. These artifacts
live under `ContRec_VAE/smoke/` and require `--smoke-test` when loaded by the
recommender. Reconstruction metrics alone do not establish recommendation gains;
compare full recommendation runs against the PCA baseline.

## Saving and Resuming

`save_interval` defaults to 20 epochs and can be overridden with
`--save_interval=10`. Each run keeps two checkpoint files: `latest` is overwritten
at this interval, at training completion and at early stopping; `best` is
overwritten whenever validation improves `val_metric` (currently `ndcg@10`),
regardless of the save interval. Final testing loads `best`. The final epoch
also runs validation even if it falls outside `eval_interval`.

Each task has this directory structure; `run_id` is the editable task name:

```text
ckpt/AmazonReviews2014/TigerDiff/Beauty/my_task/
  latest_my_task.pth
  best_my_task.pth
  train_20261006-150000-123456_my_task.log
  resume_20261006-160000-123456_my_task.log
  eval_20261006-170000-123456_my_task.log
  tensorboard/   # when TensorBoard is installed
```

No new historical checkpoint files accumulate. A save uses an atomic replacement,
so a failed write leaves the previous latest file intact. Existing checkpoints
and logs are not moved or deleted. Use a new `run_id` to retain a separate
experiment; ordinary training starts fresh and overwrites that task's latest file
on its next save. Both files contain the weights and optimizer/scheduler state
from their respective epochs; best-score metadata in `latest` does not mean
that it contains the best weights.

```bash
# Fresh training
uv run train.py --category=Beauty --run_id=my_task --save_interval=10
# Resume latest for the same category and task, targeting 200 total epochs
uv run train.py --category=Beauty --run_id=my_task --resume --epochs=200
# Only evaluate the best checkpoint
uv run train.py --category=Beauty --run_id=my_task --eval-only
# Explicit file overrides selection; directories prefer best for evaluation and latest for resume
uv run train.py --category=Beauty --run_id=my_task --eval-only \
  --checkpoint=ckpt/AmazonReviews2014/TigerDiff/Beauty/my_task
```

### Diffusion multi-sample evaluation

Evaluation defaults to `eval_sampling_mode=knn`: one diffusion sample per history,
ranked using the existing `tiger_knn_metric`. Old launch commands and model
checkpoints continue to work. Enable fixed-budget multi-sample evaluation with:

```bash
uv run train.py --category=Beauty --run_id=my_task --eval-only \
  --eval_sampling_mode=diffusion_multi \
  --eval_num_samples=8 --eval_candidates_per_sample=20 \
  --eval_candidate_metric=l2 --eval_candidate_ranking=support \
  --eval_support_temperature=1.0 --eval_log_sampling_stats=true
```

Each history produces `eval_num_samples` independent samples in one batched
diffusion call. Each sample retrieves its nearest items; their union is deduplicated
and ranked. If the union is smaller than `max(topk)`, retrieval depth doubles
using the same samples until enough unique items exist. A catalog smaller than
K returns only its available items; missing ranks count as misses. No random
items are added. All three rankings use the same candidate generation rules:

| `eval_candidate_ranking` | Ranking |
| --- | --- |
| `support` | Sum `exp(-distance / eval_support_temperature)` across the samples whose local lists contain the item. Ties use minimum distance, then item ID. |
| `first_seen` | Preserve first discovery in sample order and neighbor order; expansion appends newly discovered items. |
| `min_distance` | Minimum distance to any sample, then item ID. |

`eval_candidate_metric=l2` uses squared Euclidean distance in the unchanged
training embedding space. `cosine` uses `1 - cosine_similarity`; `inner_product`
uses negative dot product. Support uses log-sum-exp internally to avoid overflow
or underflow, without normalizing votes. This score is a retrieval heuristic,
not a diffusion likelihood or the noise/velocity training loss.

When `eval_log_sampling_stats=true`, results include `sampling_unique_candidates`,
`sampling_deduplication_rate`, `sampling_retrieval_depth`, `sampling_num_samples`,
and `sampling_candidate_occurrences`, averaged per history after distributed
gather. Deduplication rate is `1 - unique_candidates / candidate_occurrences`
at the final retrieval depth. Statistics default to off and do not affect
`val_metric` selection. The selected evaluation mode applies to validation too.

For comparisons, evaluate the same explicit checkpoint, data, batch size and
`rand_seed` (in the Python config) with each ranking; repeat over several seeds.
Include legacy cosine KNN and legacy L2 KNN to separate the metric change from
the benefit of multiple samples. Sweep sample counts (1, 4, 8, 16) and local
candidate counts alongside recall/NDCG and evaluation time. Keep the sampling
temperature, CFG strength and sampling steps fixed. Expect higher evaluation
cost from multiple trajectories; measure improvements rather than assuming them.

New checkpoints store model weights, optimizer, scheduler, completed epoch and
best validation score. `--resume` restores these states; `epochs` is the total
target, not an additional epoch count. If the target is already reached, training
is skipped and the saved best checkpoint is evaluated. Random generator and dataloader
positions are not restored, so resuming is not a bitwise replay.

`--checkpoint` without `--resume` retains the old weight-only warm start behavior.
Resuming an old pure-weight checkpoint also starts a fresh optimizer and epoch
count, with a warning. `--resume` and `--eval-only` fail clearly if their task
has no checkpoint. Old latest-only runs have no recoverable historical best
weights: continuing such a run restarts best selection from resumed validation
epochs and logs a warning. Default evaluation requires a best file; use an
explicit `--eval-only --checkpoint=<latest-file>` to evaluate an old latest file.
`steps` retains the previous whole-epoch rounding behavior.

## Analysis and Verification

```bash
uv run python -m vis_analysis.analysis_tigerdiff --analysis embedding --category Beauty
uv run python -m vis_analysis.visualization --log_paths ckpt
uv run python -m unittest discover -s tests
uv run train.py --smoke-test --category=Sports_and_Outdoors
```

Attention analysis uses the same `config.py` architecture as training. Match
its dimensions to the checkpoint before analyzing old weights. Existing RPG
and other legacy logs remain readable by the log plotter.

TensorBoard tracking remains optional: install `tensorboard` to enable event
files. Text training logs are written regardless.

## Independent Joint AE Training

The standalone `python -m joint_training` entry point trains GeometryAE and a
continuous HSTU/Transformer + Rectified Flow recommender with separate optimizers,
per-item LayerNorm over latents, and an optional contrastive loss. It uses
its own configuration and checkpoints. See [the joint-training design and usage
guide](joint_training/README.md); `bash scripts/train_joint.sh` launches it.
`bash scripts/search_joint_contrastive.sh` searches that configuration with Optuna.
