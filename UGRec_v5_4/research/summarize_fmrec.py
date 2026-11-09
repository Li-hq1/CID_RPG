"""Audit three matched online-AE FMRec runs and plot the bounded Beauty screen."""

import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from misc.artifacts import json_hash,write_json
from protocol import ROOT,ranking_metrics


METHODS=['fmrec','ce_only','history_ce']


def main():
    base=ROOT/'outputs/fmrec'
    result={'seed':2024,'category':'Beauty','test_accessed':False,'budget':'initial40',
            'methods':{},'audit':{}}
    identities=[];runtime_hashes=[];nonruntime_changes={}
    for method in METHODS:
        directory=base/'runs/Beauty'/f'beauty_{method}_initial40_s2024'
        metrics=json.loads((directory/'results.json').read_text())
        experiment=json.loads((directory/'experiment.json').read_text())
        config=json.loads((directory/'config.json').read_text())
        history=json.loads((directory/'metrics.json').read_text())
        gradients=json.loads((directory/'gradient_audit.json').read_text())
        status=json.loads((base/f'status_{method}.json').read_text())
        assert status['state']=='complete' and status['returncode']==0
        assert metrics['benchmark_eligible'] and metrics['checkpoint_reload_verified'] and not metrics['test_accessed']
        assert not experiment['test_accessed'] and not experiment['has_id_item_embedding_table']
        assert not experiment['history_mse_enabled'] and experiment['selection_users']==4096
        assert experiment['method']==method and config['fmrec']['mse_weight']==0
        assert config['training']['seed']==2024 and config['training']['epochs']==40
        assert config['fmrec']['ce_weight']==.2 and config['fmrec']['sample_N']==30
        assert not config['ae']['frozen'] and gradients['ce_ae_encoder_grad_norm']>0
        if method=='fmrec':assert config['fmrec']['fm_weight']==1 and gradients['fm_ae_encoder_grad_norm']>0
        else:assert config['fmrec']['fm_weight']==0
        expected_steps=40*math.ceil(experiment['data']['training_prefixes']/config['training']['batch_size'])
        assert metrics['steps']==expected_steps and metrics['completed_epochs']==40 and len(history)==40
        for filename,key,count in [('selection.npz','validation_selection',4096),('full.npz','validation',experiment['data']['users'])]:
            with np.load(directory/filename) as saved:
                assert len(saved['targets'])==count
                recalculated=ranking_metrics(saved['predictions'],saved['targets'])
                for name,value in recalculated.items():assert abs(value-metrics[key][name])<1e-12
        hashes=json.loads((directory/'source_hashes.json').read_text())
        assert json_hash(hashes)==experiment['source_snapshot_sha256']
        for name,digest in hashes.items():
            assert hashlib.sha256((directory/'source_snapshot'/name).read_bytes()).hexdigest()==digest
        runtime={name:digest for name,digest in hashes.items() if name.startswith(('models/','joint_training/','misc/','compression/'))
                 or name in {'run_fmrec.py','protocol.py'}}
        runtime_hashes.append(runtime)
        for name,digest in runtime.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
        nonruntime_changes[method]=[name for name,digest in hashes.items() if name not in runtime and
            (not (ROOT/name).exists() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest)]
        identities.append({key:experiment[key] for key in ['source','shared_initial_state_sha256','data','selection_indices_sha256']})
        checkpoint_path=directory/'best.pt';checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        assert checkpoint['format']=='fmrec_joint_v1' and checkpoint['source']==experiment['source']
        assert checkpoint['config']==config
        assert checkpoint['epoch']==metrics['best']['epoch']
        assert not any('item_embeddings' in name or 'decoder' in name and not name.startswith('ae.')
                       for name in checkpoint['model'])
        at_boundary=metrics['best']['epoch']==40
        result['methods'][method]={**metrics,'parameters':experiment['trainable_parameters'],
            'gradient_audit':gradients,'history':history,'best_at_budget_boundary':at_boundary,
            'checkpoint_sha256':hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),'directory':str(directory)}
        diagnostic=json.loads((directory/'bridge_diagnostic.json').read_text())
        assert diagnostic['checkpoint_sha256']==result['methods'][method]['checkpoint_sha256']
        assert diagnostic['epoch']==metrics['best']['epoch'] and not diagnostic['test_accessed']
        with np.load(directory/'selection.npz') as saved:
            from misc.artifacts import array_hash
            assert diagnostic['selection_indices_sha256']==array_hash(saved['indices'][:512])
            recalculated=ranking_metrics(saved['predictions'][:512],saved['targets'][:512])
            for name,value in recalculated.items():assert abs(diagnostic['native'][name]-value)<1e-12
        result['methods'][method]['diagnostic']=diagnostic
    assert all(value==identities[0] for value in identities)
    assert all(value==runtime_hashes[0] for value in runtime_hashes)
    result['identity']=identities[0]
    for other in ['ce_only','history_ce']:
        full=result['methods']['fmrec']['validation']['ndcg@10']/result['methods'][other]['validation']['ndcg@10']-1
        selection=result['methods']['fmrec']['validation_selection']['ndcg@10']/result['methods'][other]['validation_selection']['ndcg@10']-1
        result['fmrec_vs_'+other]={'full_ndcg10_change_percent':full*100,'selection_ndcg10_change_percent':selection*100}
    protected=json.loads((ROOT/'docs/fmrec_protected_versions.json').read_text())
    changes={}
    for name,digest in protected.items():
        path=ROOT.parent/name;actual=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if actual!=digest:changes[name]={'before_sha256':digest,'after_sha256':actual}
    assert not any(name.startswith('FMRec/') for name in changes),'Reference repository changed.'
    archived=json.loads((ROOT/'history/flowrec_before_fmrec/archive_hashes.json').read_text())
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in archived.items())
    result['audit']={'shared_initial_data_selection_matched':True,'runtime_source_identical_across_three_runs':True,
        'runtime_source_sha256':json_hash(runtime_hashes[0]),'saved_source_snapshots_verified':True,
        'runtime_scope':'models/, joint_training/, compression/, misc/, run_fmrec.py, protocol.py',
        'postlaunch_nonruntime_snapshot_changes':nonruntime_changes,
        'ranking_metrics_recomputed':True,'checkpoint_reloads_verified':True,'ce_and_fm_gradients_reach_ae':True,
        'diagnostic_checkpoint_users_native_metrics_verified':True,
        'reference_repository_unchanged':True,'legacy_source_archive_verified':True,
        'protected_files':len(protected),'all_other_protected_files_unchanged':not changes,
        'observed_workspace_changes':changes}
    write_json(base/'summary.json',result)
    labels={'fmrec':'FMRec (FM + 0.2 CE)','ce_only':'Same bridge (0.2 CE)','history_ce':'History Transformer (0.2 CE)'}
    colors={'fmrec':'#4c78a8','ce_only':'#e45756','history_ce':'#f2a541'}
    lines=['# Online-AE FMRec | Beauty | seed 2024 | initial 40 epochs','',
        'All methods: 131413 training prefixes, batch 512, 10280 updates; selection 4096 users; full validation 22363 users. No test.',
        'Reference uses up to 500 epochs; this screen uses 40 epochs and does not establish full convergence.','',
        '| Method | Best epoch | Selection NDCG10 | Full NDCG10 | Full Recall10 | Full NLL | Parameters |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,m in result['methods'].items():
        lines.append(f"| {name} | {m['best']['epoch']} | {m['validation_selection']['ndcg@10']:.8f} | {m['validation']['ndcg@10']:.8f} | {m['validation']['recall@10']:.8f} | {m['validation']['nll']:.5f} | {m['parameters']} |")
    for name in ['ce_only','history_ce']:
        lines+=['',f"FMRec vs {name}: "+json.dumps(result['fmrec_vs_'+name])]
    lines+=['','Audit: '+json.dumps(result['audit'],ensure_ascii=False)]
    (base/'summary.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    for name,m in result['methods'].items():
        selected=[row for row in m['history'] if 'validation_selection' in row]
        axes[0].plot([row['epoch'] for row in selected],[row['validation_selection']['ndcg@10'] for row in selected],
                     label=labels[name],color=colors[name],marker='o',markersize=3)
        axes[1].plot([row['epoch'] for row in m['history']],[row['train']['ce'] for row in m['history']],
                     label=labels[name],color=colors[name])
    axes[0].set(xlabel='Epoch',ylabel='Selection NDCG@10',title='Locked-checkpoint selection')
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Epoch',ylabel='Raw CE (unweighted)',title='Training classification loss')
    values=[result['methods'][name]['validation']['ndcg@10'] for name in METHODS]
    axes[2].bar(range(3),values,color=[colors[name] for name in METHODS])
    axes[2].set_xticks(range(3),['FMRec','Bridge CE','History CE'])
    axes[2].set(ylabel='Full validation NDCG@10',title='Each selected checkpoint',ylim=(0,max(values)*1.2))
    for index,value in enumerate(values):axes[2].text(index,value+max(values)*.025,f'{value:.5f}',ha='center')
    for ax in axes:ax.grid(alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('Online AE + reference FMRec | matched Transformer/prefix training | seed 2024 | 40-epoch screen, no test')
    fig.savefig(base/'comparison.png',dpi=180);fig.savefig(base/'comparison.pdf');plt.close(fig)
    print('\n'.join(lines))


if __name__=='__main__':main()
