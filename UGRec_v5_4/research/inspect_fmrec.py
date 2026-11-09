"""Read-only explanation of bridge fitting vs deployment; never reselect epochs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from joint_training.checkpoint import evaluation_randomness
from joint_training.fmrec_config import FMRecConfig
from joint_training.fmrec_data import left_padded_batch
from joint_training.fmrec_model import FMRecJointModel
from misc.artifacts import array_hash,json_hash,write_json
from protocol import ROOT,SELECTION_SEED,corpus,raw_features,ranking_metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--device',required=True)
    args=parser.parse_args()
    output=args.run/'bridge_diagnostic.json'
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    config=FMRecConfig.read(args.run/'config.json');config.training.device=args.device
    path,mapping,train,histories,targets=corpus(config.data.category)
    raw=raw_features(path,len(mapping['id2item'])-1)
    checkpoint=torch.load(args.run/'best.pt',map_location='cpu',weights_only=True)
    expected={'category':config.data.category,'raw_sha256':array_hash(raw),'train_sha256':json_hash(train),
        'item_order_sha256':json_hash(mapping['id2item']),'validation_histories_sha256':json_hash(histories),
        'validation_targets_sha256':array_hash(targets),'input':'raw','teacher_kind':None}
    assert expected==checkpoint['source']
    model=FMRecJointModel(config,raw).to(args.device)
    model.load_state_dict(checkpoint['model'],strict=True);model.cache_features();model.eval();model.refresh_catalog()
    with np.load(args.run/'selection.npz') as saved:indices=saved['indices'][:512].copy()
    count=len(indices);t=.5;predictions=[];losses=[];cosines=[]
    result={'kind':'locked checkpoint secondary validation diagnostic; NOT candidate selection',
        'epoch':checkpoint['epoch'],'users':count,'selection_indices_sha256':array_hash(indices),
        'checkpoint_sha256':hashlib.sha256((args.run/'best.pt').read_bytes()).hexdigest(),
        'test_accessed':False,'target_conditioned_bridge_is_not_inference':True}
    with torch.no_grad():
        catalog=model.catalog
        unit=F.normalize(catalog,dim=-1)
        result['catalog']={'coordinate_rms':float(catalog.square().mean().sqrt()),
            'centered_coordinate_std':float((catalog-catalog.mean(0)).square().mean().sqrt()),
            'average_pair_cosine':float((unit.sum(0).square().sum()-len(unit))/(len(unit)*(len(unit)-1)))}
        if config.fmrec.method!='history_ce':
            with evaluation_randomness(model.device,SELECTION_SEED):
                for start in range(0,count,config.training.eval_batch_size):
                    chosen=indices[start:start+config.training.eval_batch_size]
                    ids=left_padded_batch(histories,chosen,config.data.max_item_seq_len).to(args.device)
                    history,mask=model.history(ids)
                    labels=torch.as_tensor(targets[chosen],device=args.device)
                    clean=catalog[labels-1];z0=torch.randn_like(clean)
                    bridge=t*clean+(1-t)*z0
                    endpoint,_=model.endpoint(history,bridge,clean.new_full((len(clean),),t*(1/config.fmrec.eps-1)),mask)
                    logits=endpoint@catalog.T
                    losses.append(F.mse_loss(endpoint,clean,reduction='none').mean(-1).cpu())
                    cosines.append(F.cosine_similarity(endpoint,clean,dim=-1).cpu())
                    predictions.append((logits.topk(10,-1).indices+1).cpu().numpy())
            result['target_conditioned_bridge_t05']={**ranking_metrics(np.concatenate(predictions),targets[indices]),
                'fm_mse':float(torch.cat(losses).mean()),'target_cosine':float(torch.cat(cosines).mean())}
        for name,steps in [('native',None)]+([('one_step',1)] if config.fmrec.method!='history_ce' else []):
            predicted=[];nll=[]
            with evaluation_randomness(model.device,SELECTION_SEED):
                for start in range(0,count,config.training.eval_batch_size):
                    chosen=indices[start:start+config.training.eval_batch_size]
                    ids=left_padded_batch(histories,chosen,config.data.max_item_seq_len).to(args.device)
                    logp=model.log_probabilities(ids,steps=steps)
                    predicted.append((logp.topk(10,-1).indices+1).cpu().numpy())
                    labels=torch.as_tensor(targets[chosen],device=args.device)
                    nll.append(-logp.gather(1,labels[:,None]-1)[:,0].cpu())
            result[name]={**ranking_metrics(np.concatenate(predicted),targets[indices]),'nll':float(torch.cat(nll).mean())}
    write_json(output,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
