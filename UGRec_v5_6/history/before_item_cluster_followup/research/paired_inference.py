"""Interleaved latency check on one GPU; retains original unpaired measurements."""
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
from protocol import ROOT, SELECTION_SEED, canonical_category, corpus, raw_features, validation_batches


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--category',required=True)
    parser.add_argument('--device',required=True)
    args=parser.parse_args()
    category=canonical_category(args.category)
    base=ROOT/'outputs/fm_ablation'/category
    output=base/'paired_inference_cost.json'
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    directory,mapping,train,histories,targets=corpus(category)
    raw=raw_features(directory,len(mapping['id2item'])-1)
    expected={'category':category,'raw_sha256':array_hash(raw),'train_sha256':json_hash(train),
        'item_order_sha256':json_hash(mapping['id2item']),'validation_histories_sha256':json_hash(histories),
        'validation_targets_sha256':array_hash(targets),'input':'raw','teacher_kind':None}
    models={};digests={};selected=None
    for name in ['control','direct_ce']:
        path=ROOT/'outputs/fm_ablation/runs'/category/f'{category.lower()}_{name}_s2024'
        config=JointConfig.from_dict(json.loads((path/'config.json').read_text()));config.training.device=args.device
        checkpoint_path=path/'best_flow.pt';checkpoint=read_checkpoint(checkpoint_path)
        assert checkpoint['source']==expected
        model=JointModel(config,raw).to(args.device);model.load_state_dict(checkpoint['model'],strict=True)
        model.cache_features();model.eval();model.refresh_catalog();models[name]=model
        digests[name]=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        with np.load(path/'selection_flow.npz') as saved:
            indices=saved['indices'][:512].copy()
        if selected is None:selected=indices
        else:assert np.array_equal(selected,indices)
    batches=[{key:value.to(args.device) for key,value in batch.items() if key!='labels'}
        for _,batch in validation_batches(histories,targets,32,selected)]
    def predict(name,batch):
        model=models[name]
        logp=direct_log_probabilities(model,batch) if name=='direct_ce' else model.mixture_log_probabilities(batch)
        return logp.topk(10,dim=-1).indices
    values={name:{'wall_seconds':[],'gpu_ms':[]} for name in models}
    with torch.no_grad(),evaluation_randomness(torch.device(args.device),SELECTION_SEED):
        for batch in batches:
            for name in models:predict(name,batch)
        torch.cuda.synchronize(args.device)
        for repetition in range(3):
            totals={name:{'wall':0.,'gpu':0.} for name in models}
            for batch_index,batch in enumerate(batches):
                order=['control','direct_ce'] if (repetition+batch_index)%2==0 else ['direct_ce','control']
                for name in order:
                    with torch.cuda.device(args.device):
                        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                        started=time.perf_counter();begin.record()
                        predict(name,batch);end.record();end.synchronize()
                    totals[name]['wall']+=time.perf_counter()-started
                    totals[name]['gpu']+=begin.elapsed_time(end)
            for name in models:
                values[name]['wall_seconds'].append(totals[name]['wall'])
                values[name]['gpu_ms'].append(totals[name]['gpu'])
    for name,row in values.items():
        row['mean_wall_ms_per_user']=float(np.mean(row['wall_seconds'])*1000/len(selected))
        row['mean_gpu_ms_per_user']=float(np.mean(row['gpu_ms'])/len(selected))
    result={'category':category,'device':args.device,'users':len(selected),'batch_size':32,'warmup_passes':1,
        'repetitions':3,'order':'alternates methods for every batch and repetition','measurements':values,
        'checkpoint_sha256':digests,'selection_indices_sha256':array_hash(selected),'test_accessed':False,
        'device_is_exclusive':False,'scope':'cached catalog and device inputs; includes history encoder, probabilities and top10',
        'wall_speedup':values['control']['mean_wall_ms_per_user']/values['direct_ce']['mean_wall_ms_per_user']}
    write_json(output,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
