"""Train FMRec with an online, jointly optimized content AE instead of an item table."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from joint_training.checkpoint import evaluation_randomness, random_state, restore_random_state
from joint_training.data import initialize_random_ae
from joint_training.fmrec_config import FMRecConfig
from joint_training.fmrec_data import PrefixDataset,left_padded_batch
from joint_training.fmrec_model import FMRecJointModel
from joint_training.fmrec_training import FMRecTrainer
from misc.artifacts import array_hash,atomic_write,json_hash,write_json
from protocol import ROOT,SELECTION_SEED,canonical_category,corpus,ranking_metrics,raw_features


def state_hash(model,*,shared=False):
    return json_hash({name:array_hash(value.detach().cpu().numpy()) for name,value in model.state_dict().items()
                      if not shared or not name.startswith('endpoint.time_embed.')})


def snapshot(directory):
    sources = ['run_fmrec.py','protocol.py','AGENTS.md','README.md']
    for folder in ['models','joint_training','compression','misc','configs','scripts','tests','research','docs']:
        sources += [str(path.relative_to(ROOT)) for path in sorted((ROOT/folder).rglob('*'))
                    if path.is_file() and path.suffix in {'.py','.sh','.json','.md'} and '__pycache__' not in path.parts]
    hashes = {}
    for name in sources:
        source = ROOT/name;target = directory/'source_snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    write_json(directory/'source_hashes.json',hashes)
    return json_hash(hashes)


@torch.no_grad()
def evaluate(model,histories,targets,indices,*,steps=None):
    model.eval();model.refresh_catalog()
    predictions,nll = [],[]
    with evaluation_randomness(model.device,SELECTION_SEED):
        for start in range(0,len(indices),model.config.training.eval_batch_size):
            chosen = indices[start:start+model.config.training.eval_batch_size]
            ids = left_padded_batch(histories,chosen,model.config.data.max_item_seq_len).to(model.device)
            # Labels enter scoring only after target-free sampling has completed.
            logp = model.log_probabilities(ids,steps=steps)
            predictions.append((logp.topk(min(10,logp.shape[-1]),-1).indices+1).cpu().numpy())
            labels = torch.as_tensor(targets[chosen],device=model.device)
            nll.append(-logp.gather(1,labels[:,None]-1)[:,0].cpu())
    predictions = np.concatenate(predictions)
    metrics = ranking_metrics(predictions,targets[indices]);metrics['nll'] = float(torch.cat(nll).mean())
    return metrics,predictions


def save(path,trainer,source,epoch,score):
    atomic_write(path,lambda stream:torch.save({'format':'fmrec_joint_v1','model':trainer.model.state_dict(),
        'config':trainer.config.to_dict(),'trainer':trainer.state_dict(),'source':source,
        'epoch':epoch,'score':score,'rng':random_state(trainer.model.device)},stream))


def load_continuation(checkpoint,trainer,source):
    """Extend a saved trajectory; model, objective, data and optimizer stay fixed."""
    if checkpoint.get('format')!='fmrec_joint_v1' or checkpoint['source']!=source:
        raise ValueError('FMRec continuation checkpoint/data identity mismatch.')
    previous=checkpoint['config'];requested=trainer.config.to_dict()
    for section in ['data','ae','fmrec','training']:
        old=dict(previous[section]);new=dict(requested[section])
        if section=='training':
            for key in ['epochs','device','run_id','output_dir','patience']:
                old.pop(key,None);new.pop(key,None)
        if old!=new:raise ValueError(f'Continuation cannot change {section} semantics.')
    if requested['training']['epochs']<=checkpoint['epoch']:
        raise ValueError('Continuation budget must exceed the checkpoint epoch.')
    trainer.model.load_state_dict(checkpoint['model'],strict=True)
    trainer.model.invalidate_catalog()
    trainer.load_state_dict(checkpoint['trainer'])
    restore_random_state(checkpoint['rng'],trainer.model.device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/fmrec.json')
    parser.add_argument('--category');parser.add_argument('--device');parser.add_argument('--run-id')
    parser.add_argument('--epochs',type=int);parser.add_argument('--max-steps',type=int)
    parser.add_argument('--eval-users',type=int);parser.add_argument('--eval-interval',type=int)
    parser.add_argument('--smoke-test',action='store_true')
    parser.add_argument('--audit',choices=['full','subset'],default='full')
    parser.add_argument('--resume-from',type=Path,help='Extend a best.pt into a new run, restoring optimizer and RNGs.')
    args = parser.parse_args();config = FMRecConfig.read(args.config)
    for key,value in [('device',args.device),('run_id',args.run_id),('epochs',args.epochs),('max_steps',args.max_steps),
                      ('selection_users',args.eval_users),('eval_interval',args.eval_interval)]:
        if value is not None:setattr(config.training,key,value)
    if args.category:config.data.category = canonical_category(args.category)
    if args.smoke_test:
        config.training.smoke_test = True;config.training.max_steps = config.training.max_steps or 4
        config.training.selection_users = min(config.training.selection_users,8)
        config.ae.hidden_dims = (32,);config.ae.latent_dim = 16
        config.fmrec.num_blocks = 2;config.fmrec.last = 1;config.fmrec.sample_N = 2
        config.training.batch_size = 4;config.ae.batch_size = 8;args.audit = 'subset'
    config.validate()
    started = time.time()
    output = Path(config.training.output_dir)
    if not output.is_absolute():output = ROOT/output
    directory = output/config.data.category/config.training.run_id
    directory.mkdir(parents=True,exist_ok=False)
    source_hash = snapshot(directory);write_json(directory/'config.json',config.to_dict())
    torch.set_num_threads(config.training.cpu_threads);torch.set_float32_matmul_precision('high')
    random.seed(config.training.seed);np.random.seed(config.training.seed);torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(config.training.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    cache = Path(config.data.cache_dir)
    processed,mapping,train,histories,targets = corpus(config.data.category,cache if cache.is_absolute() else ROOT/cache)
    raw = raw_features(processed,len(mapping['id2item'])-1)
    train_ids = torch.tensor(sorted({item for seq in train for item in seq}))
    source = {'category':config.data.category,'raw_sha256':array_hash(raw),'train_sha256':json_hash(train),
              'item_order_sha256':json_hash(mapping['id2item']),'validation_histories_sha256':json_hash(histories),
              'validation_targets_sha256':array_hash(targets),'input':'raw','teacher_kind':None}
    dataset = PrefixDataset(train,config.data.max_item_seq_len)
    ae = initialize_random_ae(config,raw,train_ids)
    model = FMRecJointModel(config,raw,ae).to(config.training.device);model.cache_features()
    generator = torch.Generator().manual_seed(config.training.seed)
    loader = DataLoader(dataset,batch_size=config.training.batch_size,shuffle=True,generator=generator)
    trainer = FMRecTrainer(model,train_ids,generator)
    indices = np.random.default_rng(SELECTION_SEED).permutation(len(targets))[:min(len(targets),config.training.selection_users)]
    eligible = not config.training.smoke_test and config.training.max_steps is None and args.audit=='full'
    experiment={'scheme':'fmrec_joint','method':config.fmrec.method,
        'source_snapshot_sha256':source_hash,'initial_state_sha256':state_hash(model),
        'shared_initial_state_sha256':state_hash(model,shared=True),'source':source,
        'selection_indices_sha256':array_hash(indices),'selection_users':len(indices),'evaluation_seed':SELECTION_SEED,
        'trainable_parameters':sum(p.numel() for p in model.parameters()),'test_accessed':False,'benchmark_eligible':eligible,
        'data':{'users':len(targets),'items':len(raw),'training_prefixes':len(dataset)},
        'runtime':{'torch':str(torch.__version__),'cuda':torch.version.cuda,'device':str(model.device),
                   'device_name':torch.cuda.get_device_name(model.device) if model.device.type=='cuda' else 'cpu'},
        'reference_commit':'1c39b17e59a08980c8372e4f861167e0a259c349',
        'padding':'zero, excluded from catalog CE','attention_mask':'reference query rows',
        'has_id_item_embedding_table':False,'history_mse_enabled':config.fmrec.mse_weight>0}
    history=[];best={'epoch':0,'score':-1.};stale=0;stop_reason='maximum_epochs'
    start_epoch=0
    if args.resume_from is not None:
        parent=args.resume_from.resolve()
        if parent.name!='best.pt':raise ValueError('Continue a selected best.pt; other checkpoints are not supported.')
        checkpoint=torch.load(parent,map_location='cpu',weights_only=True)
        parent_experiment=json.loads((parent.parent/'experiment.json').read_text())
        if parent_experiment['selection_indices_sha256']!=array_hash(indices):
            raise ValueError('Continuation selection users changed.')
        if parent_experiment['initial_state_sha256']!=experiment['initial_state_sha256']:
            raise ValueError('Continuation initialization changed.')
        parent_hashes=json.loads((parent.parent/'source_hashes.json').read_text())
        for name,digest in parent_hashes.items():
            if name.startswith(('models/','joint_training/','compression/','misc/')) or name=='protocol.py':
                if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
                    raise ValueError(f'Continuation model/training source changed: {name}')
        history=[row for row in json.loads((parent.parent/'metrics.json').read_text()) if row['epoch']<=checkpoint['epoch']]
        if not history or history[-1]['steps']!=checkpoint['trainer']['steps']:
            raise ValueError('Continuation epoch/step history mismatch.')
        load_continuation(checkpoint,trainer,source)
        start_epoch=checkpoint['epoch'];best={'epoch':start_epoch,'score':checkpoint['score']}
        trainer.gradient_audit=json.loads((parent.parent/'gradient_audit.json').read_text())
        restored,predictions=evaluate(model,histories,targets,indices)
        if restored['ndcg@10']!=best['score']:raise RuntimeError('Continuation changed checkpoint predictions.')
        with np.load(parent.parent/'selection.npz') as saved:
            if not np.array_equal(predictions,saved['predictions']):raise RuntimeError('Continuation prediction identity mismatch.')
        save(directory/'best.pt',trainer,source,start_epoch,best['score'])
        np.savez_compressed(directory/'selection.npz',indices=indices,targets=targets[indices],predictions=predictions)
        write_json(directory/'metrics.json',history)
        experiment['continuation']={'checkpoint':str(parent),'epoch':start_epoch,
            'checkpoint_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),
            'parent_source_snapshot_sha256':parent_experiment['source_snapshot_sha256'],
            'optimizer_scheduler_rng_restored':True,'parent_predictions_verified':True,
            'inherited_history_through_epoch':start_epoch,'core_source_unchanged':True}
    write_json(directory/'experiment.json',experiment)
    if model.device.type=='cuda':torch.cuda.reset_peak_memory_stats(model.device)
    for epoch in range(start_epoch+1,config.training.epochs+1):
        totals={};batches=0
        for ids,labels in loader:
            if config.training.max_steps is not None and trainer.steps>=config.training.max_steps:break
            values=trainer.step(ids.to(model.device),labels.to(model.device));batches+=1
            for key,value in values.items():totals[key]=totals.get(key,0.)+value
        if not batches:break
        trainer.scheduler.step()
        row={'epoch':epoch,'steps':trainer.steps,'train':{key:value/batches for key,value in totals.items()},
             'lr':[group['lr'] for group in trainer.optimizer.param_groups]}
        limited=config.training.max_steps is not None and trainer.steps>=config.training.max_steps
        if epoch%config.training.eval_interval==0 or epoch==config.training.epochs or limited:
            metrics,predictions=evaluate(model,histories,targets,indices);row['validation_selection']=metrics
            if metrics['ndcg@10']>best['score']:
                best={'epoch':epoch,'score':metrics['ndcg@10']};stale=0
                save(directory/'best.pt',trainer,source,epoch,best['score'])
                np.savez_compressed(directory/'selection.npz',indices=indices,targets=targets[indices],predictions=predictions)
            else:stale+=1
        history.append(row);write_json(directory/'metrics.json',history)
        if 'validation_selection' in row:
            save(directory/'last.pt',trainer,source,epoch,row['validation_selection']['ndcg@10'])
        write_json(directory/'gradient_audit.json',trainer.gradient_audit)
        print(json.dumps(row),flush=True)
        if limited:stop_reason='engineering_step_limit';break
        if stale>=config.training.patience:stop_reason='selection_patience';break
    if best['epoch']==0:raise RuntimeError('No validation checkpoint.')
    checkpoint=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
    assert checkpoint['format']=='fmrec_joint_v1' and checkpoint['source']==source
    model.load_state_dict(checkpoint['model'],strict=True);model.invalidate_catalog()
    restored,predictions=evaluate(model,histories,targets,indices)
    with np.load(directory/'selection.npz') as saved:
        if not np.array_equal(predictions,saved['predictions']):raise RuntimeError('Checkpoint reload changed predictions.')
    if restored['ndcg@10']!=best['score']:raise RuntimeError('Checkpoint reload changed selection score.')
    result={'best':best,'validation_selection':restored,'steps':trainer.steps,'completed_epochs':history[-1]['epoch'],
            'stop_reason':stop_reason,'benchmark_eligible':eligible,'test_accessed':False,'checkpoint_reload_verified':True}
    if args.resume_from is not None:result['continuation']=experiment['continuation']
    if args.audit=='full':
        metrics,predictions=evaluate(model,histories,targets,np.arange(len(targets)))
        result['validation']=metrics;np.savez_compressed(directory/'full.npz',targets=targets,predictions=predictions)
    if model.device.type=='cuda':
        result['peak_memory_allocated_mib']=torch.cuda.max_memory_allocated(model.device)/2**20
        result['peak_memory_reserved_mib']=torch.cuda.max_memory_reserved(model.device)/2**20
    result['seconds']=time.time()-started
    write_json(directory/'results.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
