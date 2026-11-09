"""Train-only popularity, transitions, ItemKNN and semantic retrieval on full validation."""

import argparse
import numpy as np
from scipy import sparse
import torch
from torch.nn import functional as F

from .common import ROOT, SEED, batches, corpus, raw_features, summarize, write_json


def run(category, device):
    directory, mapping, train, histories, targets = corpus(category)
    n_items = len(mapping["id2item"]) - 1
    rows, cols = zip(*[(u, i - 1) for u, seq in enumerate(train) for i in set(seq)])
    interactions = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                                      shape=(len(train), n_items))
    popularity = np.asarray(interactions.sum(0)).ravel()
    pairs = [(a - 1, b - 1) for seq in train for a, b in zip(seq[:-1], seq[1:])]
    a, b = zip(*pairs)
    transitions = sparse.csr_matrix((np.ones(len(a), dtype=np.float32), (a, b)),
                                     shape=(n_items, n_items))
    cooccurrence = (interactions.T @ interactions).tocsr()
    inverse = sparse.diags(1 / np.sqrt(popularity.clip(1)))
    cosine = (inverse @ cooccurrence @ inverse).tocsr()
    cosine.setdiag(0)
    cosine.eliminate_zeros()
    # Fixed K=100, no validation tuning. Keep exactly the strongest nonzero neighbors per item.
    for row in range(n_items):
        start, end = cosine.indptr[row:row + 2]
        if end - start > 100:
            keep = np.argpartition(cosine.data[start:end], -100)[-100:]
            remove = np.ones(end - start, bool)
            remove[keep] = False
            cosine.data[start:end][remove] = 0
    cosine.eliminate_zeros()
    raw = F.normalize(torch.tensor(raw_features(directory, n_items), device=device), dim=-1)
    methods = {name: [] for name in ("popularity", "markov_last", "itemknn_last", "itemknn_history",
                                    "semantic_last", "semantic_mean")}
    seen_methods = {name: [] for name in methods}
    pop_tiebreak = popularity / max(1, popularity.max()) * 1e-7
    for ix, batch in batches(histories, targets, batch_size=128):
        last = np.array([histories[i][-1] - 1 for i in ix])
        scores = {
            "popularity": np.broadcast_to(popularity, (len(ix), n_items)).copy(),
            "markov_last": transitions[last].toarray(),
            "itemknn_last": cosine[last].toarray(),
            "itemknn_history": (interactions[ix] @ cosine).toarray(),
        }
        with torch.no_grad():
            ids = batch["input_ids"].to(device)
            values = raw[(ids - 1).clamp_min(0)] * (ids > 0)[..., None]
            mean = values.sum(1) / batch["seq_lens"].to(device)[:, None]
            scores["semantic_last"] = (raw[last] @ raw.T).cpu().numpy()
            scores["semantic_mean"] = (F.normalize(mean, dim=-1) @ raw.T).cpu().numpy()
        for name, values in scores.items():
            values = torch.tensor(values + pop_tiebreak, device=device)
            methods[name].append((values.topk(10).indices + 1).cpu().numpy())
            for row, user in enumerate(ix):
                values[row, torch.tensor(histories[user], device=device) - 1] = -torch.inf
            seen_methods[name].append((values.topk(10).indices + 1).cpu().numpy())
    train_lengths = np.array([len(seq) for seq in train])
    seen_targets = np.array([t in seq for t, seq in zip(targets, histories)])
    result = {"category": category, "seed": SEED, "split": "validation",
              "protocol": "leave-last-two-out fitting; full catalog; fixed ItemKNN K=100; no tuning",
              "users": len(train), "items": n_items, "train_interactions": int(train_lengths.sum()),
              "train_length_quantiles": np.quantile(train_lengths, [0, .25, .5, .75, 1]).tolist(),
              "validation_target_seen_fraction": float(seen_targets.mean()),
              "train_unseen_validation_target_fraction": float((popularity[targets - 1] == 0).mean()),
              "unmasked": {}, "mask_history": {}}
    output = ROOT / "outputs/research_20261028/nonparametric" / category
    for name in methods:
        prediction = np.concatenate(methods[name])
        filtered = np.concatenate(seen_methods[name])
        result["unmasked"][name] = summarize(prediction, targets)
        result["mask_history"][name] = summarize(filtered, targets)
        np.savez_compressed(output.parent / f"{category}_{name}_predictions.npz",
                            predictions=prediction, filtered=filtered, targets=targets)
    write_json(output.with_suffix(".json"), result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    (ROOT / "outputs/research_20261028/nonparametric").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    run(args.category, args.device)
