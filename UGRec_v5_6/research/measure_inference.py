"""Fixed validation-only inference cost, no training or checkpoint selection."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from joint_training.checkpoint import evaluation_randomness, read_checkpoint
from joint_training.config import JointConfig
from joint_training.direct_ce import direct_log_probabilities
from joint_training.model import JointModel
from misc.artifacts import array_hash, json_hash, write_json
from protocol import ROOT, SELECTION_SEED, corpus, raw_features, validation_batches


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--device",required=True)
    args=parser.parse_args()
    output=args.run/"inference_cost.json"
    if output.exists():raise FileExistsError(output)
    config=JointConfig.from_dict(json.loads((args.run/"config.json").read_text()))
    config.training.device=args.device
    torch.set_num_threads(config.training.cpu_threads)
    torch.set_float32_matmul_precision("high")
    directory,mapping,train,histories,targets=corpus(config.data.category)
    raw=raw_features(directory,len(mapping["id2item"])-1)
    path=args.run/"best_flow.pt"
    checkpoint=read_checkpoint(path)
    actual={"category":config.data.category,"raw_sha256":array_hash(raw),"train_sha256":json_hash(train),
        "item_order_sha256":json_hash(mapping["id2item"]),"validation_histories_sha256":json_hash(histories),
        "validation_targets_sha256":array_hash(targets),"input":"raw","teacher_kind":None}
    if checkpoint["source"]!=actual:raise ValueError("Inference source differs from checkpoint")
    model=JointModel(config,raw).to(args.device)
    model.load_state_dict(checkpoint["model"],strict=True)
    model.cache_features();model.eval();model.refresh_catalog()
    with np.load(args.run/"selection_flow.npz") as saved:
        selected=saved["indices"][:512].copy()
    batches=[{key:value.to(args.device) for key,value in batch.items() if key!="labels"}
        for _,batch in validation_batches(histories,targets,32,selected)]
    network_queries=[0]
    handle=None
    if hasattr(model,"diffloss"):
        def count(module,inputs):network_queries[0]+=len(inputs[0])
        handle=model.diffloss.net.register_forward_pre_hook(count)
    def predict():
        with torch.no_grad(),evaluation_randomness(model.device,SELECTION_SEED):
            for batch in batches:
                logp=(direct_log_probabilities(model,batch) if config.flow_ranking.path=="direct_ce"
                      else model.mixture_log_probabilities(batch))
                logp.topk(10,dim=-1)
    predict()
    if model.device.type=="cuda":
        torch.cuda.synchronize(model.device)
        torch.cuda.reset_peak_memory_stats(model.device)
    network_queries[0]=0
    elapsed=[]
    for _ in range(3):
        started=time.perf_counter();predict()
        if model.device.type=="cuda":torch.cuda.synchronize(model.device)
        elapsed.append(time.perf_counter()-started)
    if handle:handle.remove()
    result={"category":config.data.category,"device":args.device,"batch_size":32,"users":len(selected),
        "warmup_passes":1,"repetitions":3,"seconds":elapsed,
        "mean_ms_per_user":float(np.mean(elapsed)*1000/len(selected)),
        "fm_queries_per_user":network_queries[0]/3/len(selected),
        "has_fm_module":hasattr(model,"diffloss"),
        "checkpoint_has_fm_parameters":any(name.startswith("diffloss.") for name in checkpoint["model"]),
        "parameters":sum(p.numel() for p in model.parameters()),
        "checkpoint_bytes":path.stat().st_size,"checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "selection_indices_sha256":array_hash(selected),"test_accessed":False,
        "scope":"cached catalog, inputs on device; history AE/HSTU, output probabilities and top10 included",
        "device_is_exclusive":False}
    if model.device.type=="cuda":
        result["peak_memory_allocated_mib"]=torch.cuda.max_memory_allocated(model.device)/2**20
    write_json(output,result);print(json.dumps(result),flush=True)


if __name__=="__main__":main()
