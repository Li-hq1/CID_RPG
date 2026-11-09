"""Export a fixed teacher for the production joint_training CLI, train-only fit."""

import argparse

import numpy as np
import torch

from misc.artifacts import array_hash, json_hash
from .common import ROOT, corpus, raw_features, write_json
from .compact_ablation import prepare
from .iterate import fixed_teacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="Beauty")
    parser.add_argument("--kind", choices=("content", "behavior", "mixed"), default="content")
    args = parser.parse_args()
    content, _, _, _ = prepare(args.category)
    directory, mapping, train, _, _ = corpus(args.category)
    raw = raw_features(directory, len(content))
    mask = np.zeros(len(raw), dtype=bool)
    mask[np.array(sorted({i for seq in train for i in seq})) - 1] = True
    values = fixed_teacher(args.category, content, args.kind)
    source = {"embeddings_sha256": array_hash(raw), "item_order_sha256": json_hash(mapping["id2item"]),
              "train_mask_sha256": array_hash(mask)}
    path = ROOT / "outputs/teachers" / f"{args.category}_{args.kind}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"values": values, "source": source, "fit_split": "train", "kind": args.kind}, path)
    write_json(path.with_suffix(".json"), {"source": source, "fit_split": "train", "kind": args.kind,
                                          "shape": list(values.shape), "path": str(path)})
    print(path)


if __name__ == "__main__":
    main()
