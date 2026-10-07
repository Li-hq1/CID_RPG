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

## Saving and Resuming

`save_interval` defaults to 5 epochs and can be overridden with
`--save_interval=10`. Saving overwrites one latest checkpoint; training completion
and early stopping also save, even between intervals. Validation improvements
update the best score for early stopping but do not write extra checkpoints.
Final testing now loads the latest saved weights, rather than the best validation
weights.

Each task has this directory structure; `run_id` is the editable task name:

```text
ckpt/AmazonReviews2014/TigerDiff/Beauty/my_task/
  latest_my_task.pth
  train_20261006-150000-123456_my_task.log
  resume_20261006-160000-123456_my_task.log
  eval_20261006-170000-123456_my_task.log
  tensorboard/   # when TensorBoard is installed
```

No new historical checkpoint files accumulate. A save uses an atomic replacement,
so a failed write leaves the previous latest file intact. Existing checkpoints
and logs are not moved or deleted. Use a new `run_id` to retain a separate
experiment; ordinary training starts fresh and overwrites that task's latest file
on its next save.

```bash
# Fresh training
uv run train.py --category=Beauty --run_id=my_task --save_interval=10
# Resume latest for the same category and task, targeting 200 total epochs
uv run train.py --category=Beauty --run_id=my_task --resume --epochs=200
# Only evaluate the latest checkpoint
uv run train.py --category=Beauty --run_id=my_task --eval-only
# Explicit checkpoint file or directory; latest files take priority in a directory
uv run train.py --category=Beauty --run_id=my_task --eval-only \
  --checkpoint=ckpt/AmazonReviews2014/TigerDiff/Beauty/my_task
```

New checkpoints store model weights, optimizer, scheduler, completed epoch and
best validation score. `--resume` restores these states; `epochs` is the total
target, not an additional epoch count. If the target is already reached, training
is skipped and the checkpoint is evaluated. Random generator and dataloader
positions are not restored, so resuming is not a bitwise replay.

`--checkpoint` without `--resume` retains the old weight-only warm start behavior.
Resuming an old pure-weight checkpoint also starts a fresh optimizer and epoch
count, with a warning. `--resume` and `--eval-only` fail clearly if their task
has no checkpoint. `steps` retains the previous whole-epoch rounding behavior.

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
