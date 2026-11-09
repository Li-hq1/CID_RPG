"""Inference-only center collapse: does the selected model use cluster differences?"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from joint_training.checkpoint import read_checkpoint
from joint_training.config import JointConfig
from joint_training.data import initialize_random_ae
from joint_training.direct_ce import direct_log_probabilities
from joint_training.model import JointModel
from misc.artifacts import array_hash, write_json
from protocol import corpus, raw_features, ranking_metrics, validation_batches
from run import seed_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cfg = JointConfig.from_dict(json.loads((args.run / "config.json").read_text()))
    cfg.training.device = args.device
    assert cfg.recommender.item_vlad_variant == "residual" and cfg.recommender.item_vlad_clusters == 4
    torch.set_num_threads(cfg.training.cpu_threads); torch.set_float32_matmul_precision("high")
    directory, mapping, train, histories, targets = corpus(cfg.data.category)
    raw = raw_features(directory, len(mapping["id2item"])-1)
    train_ids = torch.tensor(sorted({item for seq in train for item in seq}))
    seed_run(cfg.training.seed, cfg.training.deterministic)
    model = JointModel(cfg, raw, initialize_random_ae(cfg, raw, train_ids)).to(args.device)
    checkpoint = args.run / "best_flow.pt"
    model.load_state_dict(read_checkpoint(checkpoint)["model"], strict=True)
    model.cache_features(); model.eval(); model.refresh_catalog()
    with np.load(args.run / "selection_flow.npz") as saved:
        indices = saved["indices"][:512].copy(); expected = saved["predictions"][:512].copy()
    @torch.no_grad()
    def predict():
        outputs = []
        for _, data in validation_batches(histories, targets, 32, indices):
            data = {k: v.to(args.device) for k, v in data.items() if k != "labels"}
            outputs.append(direct_log_probabilities(model, data).cpu())
        lp = torch.cat(outputs)
        return lp, (lp.topk(10, -1).indices+1).numpy()
    with torch.no_grad():
        lp, original = predict()
        assert np.array_equal(original, expected), "Original checkpoint predictions differ"
        centers = model.item_cluster_history.tokenizer.centers
        old = centers.clone(); mean = old.mean(0, keepdim=True)
        try:
            centers.copy_(mean.expand_as(centers))
            changed_lp, changed = predict()
        finally:
            centers.copy_(old)
    valid_targets = targets[indices]
    result = {"intervention": "Replace every trained center by the trained mean center, in memory only",
        "scope": "First 512 fixed selection users, diagnostic only; no checkpoint/epoch selection",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "indices_sha256": array_hash(indices), "original_predictions_verified": True,
        "parameters_restored_exactly": bool(torch.equal(centers, old)), "test_accessed": False,
        "original_metrics": ranking_metrics(original, valid_targets),
        "collapsed_metrics": ranking_metrics(changed, valid_targets),
        "top10_order_changed_user_fraction": float((original != changed).any(-1).mean()),
        "top10_set_changed_user_fraction": float((np.sort(original, axis=1) != np.sort(changed, axis=1)).any(-1).mean()),
        "log_probability_difference_relative_l2": float((changed_lp-lp).norm()/lp.norm()),
        "log_probability_difference_max_abs": float((changed_lp-lp).abs().max()),
        "center_mean_norm": float(mean.norm()),
        "center_deviation_rms": float((old-mean).square().sum(-1).mean().sqrt()),
        "note": "Inference sensitivity cannot establish whether center diversity or repeated-token dropout helped training."}
    source = Path(__file__).read_bytes()
    (args.output_dir / "probe_source.py").write_bytes(source)
    result["script_sha256"] = hashlib.sha256(source).hexdigest()
    np.savez_compressed(args.output_dir / "predictions.npz", indices=indices, targets=valid_targets,
                        original=original, collapsed=changed)
    write_json(args.output_dir / "center_diversity_probe.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__": main()
