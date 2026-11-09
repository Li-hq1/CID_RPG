"""Compact controlled screening: representation x encoder x CE/RF objective.

This is not an FMRec/MIRAGE reproduction. Every run uses all training next-item
positions, a fixed epoch budget, a shared validation subset, and no test targets.
PCA is fit on training items only. For RF, target embeddings are stopped; ID
embeddings can still learn via histories. The optional CE loss uses the history
alone, explicitly avoiding teacher-forced noisy-target shortcuts.
"""

import argparse
import math
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA

from models.diffusion.diffloss import DiffLoss
from models.hstu import HSTUSequenceEncoder
from models.tigerdiff import InputProj, TigerSequenceEncoder
from models.evaluation_sampling import aggregate_diffusion_candidates
from .common import ROOT, SEED, batches, corpus, raw_features, summarize, write_json


def prepare(category):
    directory, mapping, train, histories, targets = corpus(category)
    n_items = len(mapping["id2item"]) - 1
    cache = ROOT / "outputs/research_20261028/prepared" / category
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "pca128.npy"
    if not path.exists():
        raw = raw_features(directory, n_items)
        train_ids = np.array(sorted({i - 1 for seq in train for i in seq}))
        pca = PCA(n_components=128, whiten=True, svd_solver="randomized", random_state=2024)
        pca.fit(raw[train_ids])
        value = pca.transform(raw).astype(np.float32)
        value = (value - value[train_ids].mean(0)) / value[train_ids].std(0).clip(1e-6)
        np.save(path, value)
    content = torch.from_numpy(np.load(path))
    # Exactly the production tokenizer's first-window-all-positions, then last-position policy.
    inputs, labels = [], []
    for seq in train:
        windows = [seq[:51]] + [seq[start:start + 51] for start in range(1, max(len(seq) - 50, 1))]
        for window_index, window in enumerate(windows):
            if len(window) < 2:
                continue
            x = window[:-1] + [0] * (51 - len(window))
            y = window[1:] + [-100] * (51 - len(window))
            if window_index > 0:
                y[:len(window) - 2] = [-100] * (len(window) - 2)
            inputs.append(x)
            labels.append(y)
    return content, TensorDataset(torch.tensor(inputs), torch.tensor(labels)), histories, targets


class CompactModel(nn.Module):
    def __init__(self, content, representation, encoder, objective):
        super().__init__()
        self.representation, self.objective = representation, objective
        self.register_buffer("content", F.pad(content, (0, 0, 1, 0)))
        self.item = nn.Embedding(len(content) + 1, 128, padding_idx=0)
        nn.init.normal_(self.item.weight, std=.02)
        with torch.no_grad():
            self.item.weight[0].zero_()
            if representation == "hybrid":
                self.item.weight.zero_()
        if representation in {"pca", "adapter"}:
            self.item.requires_grad_(False)
        self.input_proj = InputProj(128, 128)
        self.position = nn.Embedding(50, 128)
        self.encoder_type = encoder
        if encoder == "hstu":
            self.encoder = HSTUSequenceEncoder(embedding_dim=128, num_layers=2, num_heads=4,
                                               hidden_dim=32, attention_dim=32, max_seq_len=50,
                                               use_group_norm=True, output_dropout_ratio=.1,
                                               dropout_pr=0., attention_scaling="input_length")
        else:
            self.encoder = TigerSequenceEncoder(hidden_size=128, num_layers=2, num_heads=4,
                                                inner_size=256, activation_name="gelu",
                                                attn_dropout=.1, dropout=.1,
                                                layer_norm_eps=1e-6, causal=True)
        self.condition = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 128))
        self.ce_head = nn.Linear(128, 128)
        self.flow = DiffLoss(target_channels=128, z_channels=128, depth=2, width=256,
                             num_sampling_steps=20, use_rectified_flow=True,
                             rectified_flow_steps=1000, ode_solver="euler")
        if representation == "adapter":
            self.adapter = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128))
            nn.init.zeros_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)

    def catalog(self):
        if self.representation == "pca":
            value = self.content
        elif self.representation == "adapter":
            value = self.content + self.adapter(self.content)
        elif self.representation == "id":
            value = self.item.weight
        else:
            value = self.content + self.item.weight
        return F.layer_norm(value, (128,), eps=1e-6)

    def encode(self, ids, table):
        mask = ids > 0
        value = self.input_proj(table[ids] * mask[..., None])
        if self.encoder_type == "transformer":
            value = value + self.position(torch.arange(50, device=ids.device))[None]
        return self.condition(self.encoder(value, attention_mask=mask))

    def logits(self, condition, table):
        return F.normalize(self.ce_head(condition), dim=-1) @ F.normalize(table[1:], dim=-1).T / .07

    def loss(self, ids, labels):
        table = self.catalog()
        condition = self.encode(ids, table)
        valid = labels > 0
        selected = condition[valid]
        target = labels[valid]
        ce = selected.new_zeros(())
        flow = selected.new_zeros(())
        if self.objective in {"ce", "rf_ce"}:
            ce = F.cross_entropy(self.logits(selected, table), target - 1)
        if self.objective in {"rf", "rf_ce"}:
            flow = self.flow(table[target].detach(), selected)
        if self.objective in {"x0", "x0_ce"}:
            clean = table[target].detach()
            times = torch.randint(1000, (len(clean),), device=clean.device)
            noisy = self.flow.train_diffusion.q_sample(clean, times)
            prediction = self.flow.net(noisy, times, selected)
            flow = F.mse_loss(prediction, clean)
            if self.objective == "x0_ce":
                logits = F.normalize(prediction, dim=-1) @ F.normalize(table[1:], dim=-1).T / .07
                ce = .2 * F.cross_entropy(logits, target - 1)
        return ce + flow, {"ce": float(ce.detach()), "rf": float(flow.detach())}

    def sample(self, condition):
        if self.objective not in {"x0", "x0_ce"}:
            return self.flow.sample(condition)
        value = torch.randn(len(condition), 128, device=condition.device)
        times = torch.linspace(1., 0., 20, device=condition.device)
        for current, following in zip(times[:-1], times[1:]):
            timestep = (current * 999).round().long().expand(len(condition))
            clean = self.flow.net(value, timestep, condition)
            # Exact data->noise straight-path identity: v = (x_t - predicted_clean) / t.
            value = value + (following - current) * (value - clean) / current
        return value


@torch.no_grad()
def evaluate(model, histories, targets, indices, device):
    model.eval()
    table = model.catalog()
    predictions, ce_predictions = [], []
    cfg = {"eval_candidates_per_sample": 1, "eval_candidate_ranking": "support",
           "eval_candidate_metric": "l2", "eval_support_temperature": 1.}
    for ix, batch in batches(histories, targets, 128, indices):
        ids = batch["input_ids"].to(device)
        h = model.encode(ids, table)
        condition = h[torch.arange(len(ix), device=device), batch["seq_lens"].to(device) - 1]
        ce_prediction = model.logits(condition, table).topk(10).indices + 1
        ce_predictions.append(ce_prediction.cpu().numpy())
        if model.objective == "ce":
            prediction = ce_prediction
        else:
            generated = model.sample(condition.repeat_interleave(32, 0)).reshape(len(ix), 32, 128)
            prediction, _ = aggregate_diffusion_candidates(generated, table[1:], 10, cfg)
            prediction = prediction.squeeze(-1)
        predictions.append(prediction.cpu().numpy())
    prediction = np.concatenate(predictions)
    ce_prediction = np.concatenate(ce_predictions)
    return summarize(prediction, targets[indices]), prediction, summarize(ce_prediction, targets[indices])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="Beauty")
    parser.add_argument("--representation", choices=("pca", "id", "hybrid", "adapter"), default="pca")
    parser.add_argument("--encoder", choices=("hstu", "transformer"), default="transformer")
    parser.add_argument("--objective", choices=("ce", "rf", "rf_ce", "x0", "x0_ce"), default="ce")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    content, dataset, histories, targets = prepare(args.category)
    if args.prepare_only:
        print("prepared", args.category, len(dataset), flush=True)
        return
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    name = f"{args.category}_{args.representation}_{args.encoder}_{args.objective}_s{args.seed}_e{args.epochs}"
    output = ROOT / "outputs/research_20261028/compact" / name
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", vars(args))
    model = CompactModel(content, args.representation, args.encoder, args.objective).to(args.device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3, weight_decay=.01)
    loader_rng = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, generator=loader_rng)
    total_steps = args.epochs * len(loader)
    def schedule(step):
        warmup = math.ceil(total_steps * .1)
        if step < warmup:
            return (step + 1) / warmup
        return .5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, total_steps - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    indices = np.random.default_rng(SEED).permutation(len(targets))[:4096]
    log, best_score, best_epoch = [], -1., 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for ids, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, terms = model.loss(ids.to(args.device), labels.to(args.device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            losses.append(terms)
        row = {"epoch": epoch, "seconds": time.time() - started,
               "train": {key: float(np.mean([x[key] for x in losses])) for key in losses[0]}}
        if epoch % 5 == 0 or epoch == args.epochs:
            # Common random numbers across epochs and models; preserve the training RNG stream.
            with torch.random.fork_rng(devices=[torch.device(args.device)]):
                torch.manual_seed(SEED)
                torch.cuda.manual_seed_all(SEED)
                score, predictions, ce_score = evaluate(model, histories, targets, indices, args.device)
            row["validation"], row["direct_ce_validation"] = score, ce_score
            if score["ndcg@10"] > best_score:
                best_score, best_epoch = score["ndcg@10"], epoch
                torch.save(model.state_dict(), output / "best.pt")
                np.savez_compressed(output / "best_validation_predictions.npz",
                                    indices=indices, targets=targets[indices], predictions=predictions)
        log.append(row)
        write_json(output / "metrics.json", log)
        print(row, flush=True)
    model.load_state_dict(torch.load(output / "best.pt", map_location=args.device, weights_only=True))
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # Full validation is an audit of the chosen compact checkpoint, not another selection stage.
    score, predictions, ce_score = evaluate(model, histories, targets, np.arange(len(targets)), args.device)
    result = {"config": vars(args), "training_samples": len(dataset), "epochs_completed": args.epochs,
              "best_epoch": best_epoch, "best_subset_val_ndcg10": best_score,
              "full_validation": score, "full_validation_direct_ce": ce_score,
              "seconds": time.time() - started, "test_accessed": False,
              "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    write_json(output / "results.json", result)
    np.savez_compressed(output / "full_validation_predictions.npz", targets=targets, predictions=predictions)
    print(result, flush=True)


if __name__ == "__main__":
    main()
