"""Scale-matched VLAD and pooling training-item diagnostics and exact reload audit, without checkpoint selection."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from compression.geometry_ae.model import loss_terms
from joint_training.checkpoint import read_checkpoint
from joint_training.config import JointConfig
from joint_training.data import initialize_random_ae
from joint_training.direct_ce import direct_log_probabilities
from joint_training.model import JointModel
from misc.artifacts import array_hash, json_hash, write_json
from protocol import SELECTION_SEED, corpus, raw_features, validation_batches
from research.item_cluster_diagnostics import pair_cosine
from run import seed_run


@torch.no_grad()
def statistics(model, batch, item_ids):
    mask = batch["attention_mask"].bool()
    inputs = model.input_proj(model.encode_ids(batch["input_ids"], grad=False))
    cfg = model.config
    adapter = model.item_cluster_history
    ae_tokenizer = getattr(model.ae, "cluster_tokenizer", None)
    result = {"valid_training_positions": int(mask.sum()),
              "catalog_input_proj_norm_mean": float(inputs[mask].norm(dim=-1).mean())}
    if adapter is not None:
        tokens, assignment, mass = adapter.prepare_tokens(inputs, mask, return_details=True)
        unmapped = adapter.tokenizer(inputs, mask) * adapter.token_scale
        result["token_variant"] = adapter.variant
        result["assignment_learned"] = adapter.variant == "nextvlad" and adapter.tokenizer.K > 1
        result["token_mapping"] = {"mode": adapter.mapping,
            "mapped_vs_unmapped_cosine": float(F.cosine_similarity(tokens[mask], unmapped[mask], dim=-1).mean()),
            "mapped_vs_unmapped_relative_l2": float((tokens[mask]-unmapped[mask]).norm()/unmapped[mask].norm())}
        if adapter.mapping == "linear":
            w = adapter.token_mapping.weight
            result["token_mapping"]["weight_relative_distance_from_identity"] = float((w-torch.eye(w.shape[0], device=w.device)).norm()/w.shape[0]**.5)
            result["token_mapping"]["singular_values"] = torch.linalg.svdvals(w).cpu().tolist()
        if getattr(adapter.tokenizer, "centers", None) is not None:
            result["center_norms"] = adapter.tokenizer.centers.norm(dim=-1).cpu().tolist()
    elif ae_tokenizer is not None:
        tokens, raw_tokens, assignment, mass = model.history_cluster_tokens(batch["input_ids"],ae_grad=False,
                                                                          return_details=True)
        result["raw_cluster_norm_mean"] = float(raw_tokens[mask].norm(dim=-1).mean())
        result["raw_cluster_pair_cosine"] = pair_cosine(raw_tokens,mask)
    else:
        tokens = inputs[:, :, None]
        assignment = mass = None
    b,t,k,d = tokens.shape
    merged_history = getattr(cfg.recommender,"ae_cluster_history","multi") == "merged" and ae_tokenizer is not None
    branch_updates = []
    valid = mask if merged_history else mask.repeat_interleave(k,1)
    def capture_update(layer, arguments, output):
        source = arguments[0][valid]
        update = output[valid] - source
        branch_updates.append({"input_norm_mean": float(source.norm(dim=-1).mean()),
                               "update_norm_mean": float(update.norm(dim=-1).mean()),
                               "update_to_input_relative_l2": float(update.norm()/source.norm())})
    hooks = [layer.register_forward_hook(capture_update) for layer in model.encoder.layers]
    try:
        if merged_history:
            output = model.encoder(inputs, mask)
            states = output[:, :, None]
        else:
            states = model.encoder(tokens.reshape(b,t*k,d),mask.repeat_interleave(k,1),
                                   item_group_size=k).reshape(b,t,k,d)
            output = states.sum(2)
    finally:
        for hook in hooks: hook.remove()
    result["hstu_layer_updates"] = branch_updates
    if adapter is not None:
        output, weights = adapter.pool(states, mass, mask, return_weights=True)
        pi = weights[mask]
        pooled_entropy = -(pi * pi.clamp_min(1e-12).log()).sum(-1).mean()
        plain = states.sum(2)
        result["pooling"] = {
            "mode": adapter.pool.mode,
            "normalized_weight_entropy": float(pooled_entropy / np.log(k)) if k > 1 else None,
            "mean_weight": pi.mean(0).cpu().tolist(),
            "std_weight": pi.std(0, unbiased=False).cpu().tolist(),
            "mean_effective_clusters": float((1. / pi.square().sum(-1)).mean()),
            "pooled_vs_sum_relative_l2": float((output[mask]-plain[mask]).norm()/plain[mask].norm()),
            "condition_vs_sum_cosine_mean": float(F.cosine_similarity(
                model.condition_proj(output[mask]), model.condition_proj(plain[mask])).mean()),
            "query_norm": float(adapter.pool.query.norm()) if adapter.pool.query is not None else None,
        }
    result.update(clusters=k,history_mode="merged" if merged_history else "multi",
        hstu_input_token_norm_mean=float((inputs[mask] if merged_history else tokens[mask]).norm(dim=-1).mean()),
        input_pair_cosine=pair_cosine(tokens,mask), contextual_pair_cosine=pair_cosine(states,mask),
        mean_hstu_output_token_norm=float(states[mask].norm(dim=-1).mean()),
        mean_sum_norm=float(output[mask].norm(dim=-1).mean()))
    if assignment is not None:
        assignment=assignment[mask];pi=mass[mask]/mass[mask].sum(-1,keepdim=True)
        entropy=-(assignment*assignment.clamp_min(1e-12).log()).sum(-1).mean()
        result.update(assignment_entropy=float(entropy),normalized_assignment_entropy=float(entropy/np.log(k)) if k>1 else None,
                      mean_mass_fraction=pi.mean(0).cpu().tolist())
    raw = model.raw_items(item_ids)
    reconstructed, z = model.ae(raw)
    terms=loss_terms(raw,reconstructed,z,model.ae.input_scale,cfg.ae.relation_temperature,batch_relation=False)
    std=z.std(0,unbiased=False)
    result["ae"]={"latent_norm_mean":float(z.norm(dim=-1).mean()),"latent_std_mean":float(std.mean()),
                  "latent_std_min":float(std.min()),"fraction_std_below_1":float((std<1).float().mean()),
                  "terms":{key:float(value) for key,value in terms.items()}}
    agg=getattr(model.ae,"cluster_aggregation",None)
    if agg is not None and getattr(agg,"weight",None) is not None:
        identity=torch.eye(z.shape[-1],device=z.device).repeat(1,k)
        result["aggregation"]={"weight_relative_distance_from_sum":float((agg.weight-identity).norm()/identity.norm()),
                               "bias_norm":float(agg.bias.norm())}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True,type=Path)
    parser.add_argument("--device",required=True)
    parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args()
    output_dir=args.output_dir or args.run
    output_dir.mkdir(parents=True,exist_ok=True)
    output=output_dir / "followup_probe.json"
    if output.exists():raise FileExistsError(output)
    config=JointConfig.from_dict(json.loads((args.run/"config.json").read_text()))
    config.training.device=args.device
    torch.set_num_threads(config.training.cpu_threads);torch.set_float32_matmul_precision("high")
    directory,mapping,train,histories,targets=corpus(config.data.category)
    raw=raw_features(directory,len(mapping["id2item"])-1)
    train_ids=torch.tensor(sorted({item for seq in train for item in seq}))
    chosen=np.random.default_rng(SELECTION_SEED).permutation(len(train))[:256]
    training_histories=[seq[-50:]for seq in train]
    batch=next(validation_batches(training_histories,np.ones(len(train),dtype=np.int64),256,chosen))[1]
    batch={key:value.to(args.device)for key,value in batch.items() if key!="labels"}
    item_ids=train_ids[torch.randperm(len(train_ids),generator=torch.Generator().manual_seed(SELECTION_SEED))[:256]].to(args.device)
    seed_run(config.training.seed,config.training.deterministic)
    model=JointModel(config,raw,initialize_random_ae(config,raw,train_ids)).to(args.device)
    state={name:array_hash(value.detach().cpu().numpy())for name,value in model.state_dict().items()}
    experiment=json.loads((args.run/"experiment.json").read_text())
    assert json_hash(state)==experiment["initial_state_sha256"],"Initialization mismatch"
    model.cache_features();model.eval()
    initial=statistics(model,batch,item_ids)
    checkpoint_path=args.run/"best_flow.pt";checkpoint=read_checkpoint(checkpoint_path)
    identity={"category":config.data.category,"raw_sha256":array_hash(raw),"train_sha256":json_hash(train),
        "item_order_sha256":json_hash(mapping["id2item"]),"validation_histories_sha256":json_hash(histories),
        "validation_targets_sha256":array_hash(targets),"input":"raw","teacher_kind":None}
    assert checkpoint["source"]==identity
    model.load_state_dict(checkpoint["model"],strict=True)
    trained=statistics(model,batch,item_ids)
    model.refresh_catalog()
    with np.load(args.run/"selection_flow.npz") as saved:
        indices=saved["indices"][:512].copy();expected=saved["predictions"][:512].copy()
    predictions=[]
    with torch.no_grad():
        for _,data in validation_batches(histories,targets,32,indices):
            data={key:value.to(args.device)for key,value in data.items()if key!="labels"}
            predictions.append((direct_log_probabilities(model,data).topk(10,-1).indices+1).cpu().numpy())
    assert np.array_equal(np.concatenate(predictions),expected),"Reload predictions differ"
    script=Path(__file__).read_bytes()
    with (output_dir/"followup_probe_source.py").open("xb")as stream:stream.write(script)
    result={"initial":initial,"trained":trained,"initial_parameter_hashes":state,
        "initial_state_reproduced":True,"reload_predictions_verified":True,
        "training_history_indices_sha256":array_hash(chosen),"training_item_ids_sha256":array_hash(item_ids.cpu().numpy()),
        "probe_script_sha256":hashlib.sha256(script).hexdigest(),
        "checkpoint_sha256":hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "test_accessed":False,"scope":"256 training histories/items; 512 selection users only for reload check"}
    write_json(output,result);print(json.dumps({"initial":initial,"trained":trained}),flush=True)


if __name__=="__main__":main()
