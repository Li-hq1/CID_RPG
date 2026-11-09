"""Training-history cluster diagnostics and descriptive validation inference cost.

No new checkpoint selection and no test access. Output is a fresh sidecar.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from joint_training.checkpoint import read_checkpoint, evaluation_randomness
from joint_training.config import JointConfig
from joint_training.direct_ce import direct_log_probabilities
from joint_training.model import JointModel
from misc.artifacts import array_hash, json_hash, write_json
from protocol import ROOT, SELECTION_SEED, corpus, raw_features, validation_batches
from run import seed_run


def pair_cosine(states, mask):
    vectors = F.normalize(states[mask], dim=-1)
    k = vectors.shape[1]
    if k == 1:
        return None
    off_diagonal = ~torch.eye(k, device=vectors.device, dtype=torch.bool)
    cos = (vectors @ vectors.transpose(-1, -2))[:, off_diagonal]
    return {"mean": float(cos.mean()), "p10": float(cos.quantile(.1)),
            "median": float(cos.median()), "p90": float(cos.quantile(.9)),
            "fraction_above_0.95": float((cos > .95).float().mean())}


@torch.no_grad()
def cluster_statistics(model, batch):
    inputs = model.input_proj(model.encode_ids(batch["input_ids"], grad=False))
    mask = batch["attention_mask"].bool()
    adapter = model.item_cluster_history
    if adapter is None:
        return {"clusters": 0, "valid_training_positions": int(mask.sum())}
    tokens, assignments, mass = adapter.tokenizer(inputs, mask, return_assignments=True)
    output, states = adapter(inputs, mask, model.encoder, return_cluster_states=True)
    pi = mass[mask] / mass[mask].sum(-1, keepdim=True)
    assignment = assignments[mask]
    entropy = -(assignment * assignment.clamp_min(1e-12).log()).sum(-1).mean()
    # Intervention preserves each item's token sum before HSTU. It measures
    # sensitivity to token differences, without introducing any future data.
    collapsed = tokens.mean(2, keepdim=True).expand_as(tokens).contiguous()
    b, t, k, d = collapsed.shape
    collapsed_output = model.encoder(collapsed.reshape(b,t*k,d), mask.repeat_interleave(k,1),
                                    item_group_size=k).reshape(b,t,k,d).sum(2)
    result = {"clusters": k, "valid_training_positions": int(mask.sum()),
        "input_pair_cosine": pair_cosine(tokens, mask),
        "contextual_pair_cosine": pair_cosine(states, mask),
        "assignment_entropy": float(entropy),
        "mean_cluster_mass_fraction": pi.mean(0).cpu().tolist(),
        "std_cluster_mass_fraction": pi.std(0).cpu().tolist(),
        "mean_pre_hstu_token_norm": float(tokens[mask].norm(dim=-1).mean()),
        "mean_post_hstu_token_norm": float(states[mask].norm(dim=-1).mean()),
        "mean_sum_norm": float(output[mask].norm(dim=-1).mean()),
        "mean_collapsed_vs_native_cosine": float(F.cosine_similarity(output[mask],collapsed_output[mask]).mean()),
        "collapsed_vs_native_relative_l2": float((output[mask]-collapsed_output[mask]).norm()/output[mask].norm()),
    }
    result["normalized_assignment_entropy"] = float(entropy / np.log(k)) if k > 1 else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    output = args.run / "item_cluster_diagnostics.json"
    if output.exists(): raise FileExistsError(output)
    config = JointConfig.from_dict(json.loads((args.run / "config.json").read_text()))
    config.training.device = args.device
    torch.set_num_threads(config.training.cpu_threads)
    torch.set_float32_matmul_precision("high")
    directory, mapping, train, histories, targets = corpus(config.data.category)
    raw = raw_features(directory, len(mapping["id2item"]) - 1)
    checkpoint_path = args.run / "best_flow.pt"
    checkpoint = read_checkpoint(checkpoint_path)
    identity = {"category": config.data.category, "raw_sha256": array_hash(raw),
        "train_sha256": json_hash(train), "item_order_sha256": json_hash(mapping["id2item"]),
        "validation_histories_sha256": json_hash(histories), "validation_targets_sha256": array_hash(targets),
        "input": "raw", "teacher_kind": None}
    if checkpoint["source"] != identity: raise ValueError("Checkpoint source mismatch")
    # Fixed training-only subset; no validation targets used in mechanism diagnostics.
    chosen = np.random.default_rng(SELECTION_SEED).permutation(len(train))[:256]
    training_histories = [seq[-50:] for seq in train]
    batches = [batch for _, batch in validation_batches(training_histories, np.ones(len(train),dtype=np.int64),
                                                       256, chosen)]
    batch = {key: value.to(args.device) for key,value in batches[0].items() if key != "labels"}
    seed_run(config.training.seed, config.training.deterministic)
    from joint_training.data import initialize_random_ae
    train_ids = torch.tensor(sorted({item for seq in train for item in seq}))
    model = JointModel(config,raw,initialize_random_ae(config,raw,train_ids)).to(args.device)
    initial_hash = json_hash({name:array_hash(value.detach().cpu().numpy())
                             for name,value in model.state_dict().items()})
    experiment = json.loads((args.run / "experiment.json").read_text())
    if initial_hash != experiment["initial_state_sha256"]:
        raise ValueError("Initial diagnostic state does not reproduce training initialization")
    model.cache_features(); model.eval()
    initial = cluster_statistics(model,batch)
    model.load_state_dict(checkpoint["model"],strict=True)
    trained = cluster_statistics(model,batch)
    with np.load(args.run / "selection_flow.npz") as saved:
        selected = saved["indices"][:512].copy()
        expected = saved["predictions"][:512].copy()
    infer_batches = [{key:value.to(args.device) for key,value in batch.items() if key != "labels"}
        for _,batch in validation_batches(histories,targets,32,selected)]
    model.refresh_catalog()
    def predict():
        predictions=[]
        with torch.no_grad(), evaluation_randomness(model.device,SELECTION_SEED):
            for batch in infer_batches:
                predictions.append((direct_log_probabilities(model,batch).topk(10,-1).indices+1).cpu().numpy())
        return np.concatenate(predictions)
    first = predict()
    if not np.array_equal(first, expected): raise ValueError("Diagnostic reload changed predictions")
    elapsed=[]
    for _ in range(3):
        if model.device.type == "cuda": torch.cuda.synchronize(model.device)
        start=time.perf_counter(); predict()
        if model.device.type == "cuda": torch.cuda.synchronize(model.device)
        elapsed.append(time.perf_counter()-start)
    result={"initial":initial,"trained":trained,"training_history_indices_sha256":array_hash(chosen),
        "training_histories_sha256":json_hash(training_histories),"test_accessed":False,
        "checkpoint_sha256":hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "checkpoint_reload_predictions_verified":True,
        "initial_state_reproduced": True,
        "diagnostic_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cost":{"device":args.device,"users":len(selected),"batch_size":32,"seconds":elapsed,
            "mean_ms_per_user":float(np.mean(elapsed)*1000/len(selected)),"device_is_exclusive":False,
            "scope":"cached catalog, history AE/HSTU, probabilities, top10 and CPU prediction copy",
            "selection_indices_sha256":array_hash(selected)}}
    with (args.run / "item_cluster_diagnostics_source.py").open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    write_json(output,result);print(json.dumps(result),flush=True)


if __name__ == "__main__": main()
