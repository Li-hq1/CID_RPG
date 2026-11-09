"""Training-only clock probe and local output-gradient diagnostic; no model updates."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from joint_training.checkpoint import read_checkpoint
from joint_training.config import JointConfig
from joint_training.contrastive import select_positions
from joint_training.model import JointModel
from misc.artifacts import array_hash, json_hash, write_json
from protocol import ROOT, corpus, raw_features, training_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--device', default='cuda:3')
    args = parser.parse_args()
    output = args.run/'training_clock_probe.json'
    if output.exists():
        raise FileExistsError(output)
    config = JointConfig.from_dict(json.loads((args.run/'config.json').read_text()))
    config.training.device = args.device
    if config.flow_ranking.path != 'clean_endpoint':
        raise ValueError('Expected clean-endpoint model')
    torch.set_num_threads(config.training.cpu_threads)
    torch.set_float32_matmul_precision('high')
    directory,mapping,train,histories,targets=corpus(config.data.category)
    raw=raw_features(directory,len(mapping['id2item'])-1)
    checkpoint_path=args.run/'best_flow.pt'
    checkpoint=read_checkpoint(checkpoint_path)
    actual={'category':config.data.category,'raw_sha256':array_hash(raw),'train_sha256':json_hash(train),
        'item_order_sha256':json_hash(mapping['id2item']),'validation_histories_sha256':json_hash(histories),
        'validation_targets_sha256':array_hash(targets),'input':'raw','teacher_kind':None}
    if actual != checkpoint['source']:
        raise ValueError('Probe data do not match checkpoint')
    model=JointModel(config,raw).to(args.device)
    model.load_state_dict(checkpoint['model'],strict=True)
    model.cache_features();model.eval();model.refresh_catalog()
    dataset=training_dataset(train)
    indices=torch.randperm(len(dataset),generator=torch.Generator().manual_seed(2024))[:256]
    ids,labels=dataset[indices]
    batch={'input_ids':ids.to(args.device),'labels':labels.to(args.device),
        'attention_mask':(ids>0).to(args.device),'seq_lens':(ids>0).sum(-1).to(args.device)}
    torch.manual_seed(2024)
    selected=select_positions(batch,1024)
    with torch.no_grad():
        encoded,_=model.encode_history(batch,ae_grad=False)
        condition=model.condition_proj(encoded[selected.rows,selected.positions])
        keys=model.catalog[1:]
        labels=selected.target_ids.to(args.device)-1
        clean=keys[labels]
        base=model.flow_prior(condition)
        unit_keys=F.normalize(keys,dim=-1)
    result={'kind':'training-only bridge vs source and local output gradients',
        'checkpoint_sha256':hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        'training_windows_sha256':array_hash(indices.numpy()),'queries':len(labels),
        'test_accessed':False,'model_updated':False,'clock':{},
        'gradient_scope':'derivatives with respect to the same predicted output only; not shared-parameter gradients'}
    for s in [0.,.5,.9,1.]:
        with torch.no_grad():
            state=(1-s)*clean+s*base
            times=torch.full((len(clean),),round(s*(model.diffloss.train_diffusion.num_timesteps-1)),device=args.device,dtype=torch.long)
            prediction=model.diffloss.net(state,times,condition)
        prediction=prediction.detach().requires_grad_(True)
        logits=F.normalize(prediction,dim=-1) @ unit_keys.T/config.flow_ranking.temperature
        ce=F.cross_entropy(logits,labels)
        fm=F.mse_loss(prediction,clean)
        grad_ce=torch.autograd.grad(ce,prediction)[0]
        grad_fm=torch.autograd.grad(fm,prediction)[0]
        cosine=F.cosine_similarity(grad_ce.flatten(),grad_fm.flatten(),dim=0)
        result['clock'][str(s)]={'ce':float(ce.detach()),'fm':float(fm.detach()),
            'endpoint_target_cosine':float(F.cosine_similarity(prediction.detach(),clean).mean()),
            'prediction_rms':float(prediction.detach().square().mean().sqrt()),
            'hit10':float((logits.detach().topk(10,dim=-1).indices==labels[:,None]).any(-1).float().mean()),
            'ce_output_grad_norm':float(grad_ce.norm()),'fm_output_grad_norm':float(grad_fm.norm()),
            'output_gradient_cosine':float(cosine),
            'weighted_fm_ce_grad_ratio':float(config.flow_ranking.fm_weight*grad_fm.norm()/(config.flow_ranking.ce_weight*grad_ce.norm()).clamp_min(1e-12))}
    write_json(output,result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    main()
