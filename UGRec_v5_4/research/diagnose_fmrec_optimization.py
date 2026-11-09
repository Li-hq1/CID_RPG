"""Locked-checkpoint diagnostics of bridge time and history dependence; no selection."""

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
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    output=args.output or args.run/'optimization_diagnostic.json'
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
    with np.load(args.run/'selection.npz') as saved:
        indices=saved['indices'][:512].copy();expected_native=saved['predictions'][:512].copy()
    result={'epoch':checkpoint['epoch'],'users':len(indices),'selection_indices_sha256':array_hash(indices),
        'checkpoint_sha256':hashlib.sha256((args.run/'best.pt').read_bytes()).hexdigest(),
        'test_accessed':False,'used_for_checkpoint_selection':False,'target_conditioned_bridge_is_not_inference':True}
    modes=[('native',None,None),('native_shuffled_history',None,None)]
    if config.fmrec.method!='history_ce':
        modes += [('one_step',None,1),('one_step_shuffled_history',None,1)]
        for t in [.001,.05,.2,.5,.8,.999]:
            modes += [(f'bridge_t{t}',t,None),(f'bridge_t{t}_shuffled_history',t,None)]
    with torch.no_grad():
        catalog=model.catalog;unit=F.normalize(catalog,dim=-1)
        result['catalog']={'coordinate_rms':float(catalog.square().mean().sqrt()),
            'centered_coordinate_std':float((catalog-catalog.mean(0)).square().mean().sqrt()),
            'average_pair_cosine':float((unit.sum(0).square().sum()-len(unit))/(len(unit)*(len(unit)-1)))}
        for name,t,steps in modes:
            predictions=[];nll=[];mse=[];cosine=[];norms=[]
            with evaluation_randomness(model.device,SELECTION_SEED):
                for start in range(0,len(indices),config.training.eval_batch_size):
                    chosen=indices[start:start+config.training.eval_batch_size]
                    ids=left_padded_batch(histories,chosen,config.data.max_item_seq_len).to(args.device)
                    if name.endswith('shuffled_history'):ids=ids.roll(1,0)
                    labels=torch.as_tensor(targets[chosen],device=args.device);clean=catalog[labels-1]
                    if t is None:prediction=model.predict_query(ids,steps=steps)
                    else:
                        history,mask=model.history(ids);noise=torch.randn_like(clean)
                        bridge=t*clean+(1-t)*noise
                        prediction,_=model.endpoint(history,bridge,clean.new_full((len(clean),),t*(1/config.fmrec.eps-1)),mask)
                    logp=(prediction@catalog.T).log_softmax(-1)
                    predictions.append((logp.topk(10,-1).indices+1).cpu().numpy())
                    nll.append(-logp.gather(1,labels[:,None]-1)[:,0].cpu())
                    mse.append((prediction-clean).square().mean(-1).cpu())
                    cosine.append(F.cosine_similarity(prediction,clean,dim=-1).cpu())
                    norms.append(prediction.square().mean(-1).sqrt().cpu())
            predicted=np.concatenate(predictions)
            result[name]={**ranking_metrics(predicted,targets[indices]),'nll':float(torch.cat(nll).mean()),
                'target_mse':float(torch.cat(mse).mean()),'target_cosine':float(torch.cat(cosine).mean()),
                'query_coordinate_rms':float(torch.cat(norms).mean())}
            if name=='native':
                assert np.array_equal(predicted,expected_native),'Native predictions differ from saved checkpoint.'
                result['native_predictions_verified']=True
    # Read-only gradient audit at this trained checkpoint, without optimizer steps.
    model.train()
    chosen=indices[:config.training.eval_batch_size]
    ids=left_padded_batch(histories,chosen,config.data.max_item_seq_len).to(args.device)
    labels=torch.as_tensor(targets[chosen],device=args.device)
    with evaluation_randomness(model.device,SELECTION_SEED):
        objective,terms=model.objectives(ids,labels)
        norms={}
        for name,term in {**terms,'weighted_recommendation':objective}.items():
            gradients=torch.autograd.grad(term,tuple(model.ae.encoder.parameters()),retain_graph=True)
            norm=torch.stack([gradient.square().sum() for gradient in gradients]).sum().sqrt()
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite trained-checkpoint gradient.')
            norms[name]=float(norm)
        result['trained_checkpoint_ae_encoder_grad_norm']=norms
        result['gradient_audit_optimizer_steps']=0
    model.eval()
    write_json(output,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
