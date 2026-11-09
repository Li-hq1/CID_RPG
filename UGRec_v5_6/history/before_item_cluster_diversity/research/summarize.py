"""Audit both methods per dataset; report gains without averaging away differences."""
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from misc.artifacts import json_hash, write_json
from protocol import ROOT, ranking_metrics

CATEGORIES=['Beauty','Sports_and_Outdoors','Toys_and_Games']


def main():
    base=ROOT/'outputs/fm_ablation'
    result={'seed':2024,'test_accessed':False,'datasets':{},'audit':{}}
    runtime_sources=[]
    for category in CATEGORIES:
        pair={}
        metadata=[]
        for name in ['control','direct_ce']:
            directory=base/'runs'/category/f'{category.lower()}_{name}_s2024'
            metrics=json.loads((directory/'results.json').read_text())
            experiment=json.loads((directory/'experiment.json').read_text())
            config=json.loads((directory/'config.json').read_text())
            history=json.loads((directory/'metrics.json').read_text())
            cost=json.loads((directory/'inference_cost.json').read_text())
            assert metrics['benchmark_eligible'] and metrics['checkpoint_reload_verified']
            assert metrics['completed_epochs']==40 and len(history)==40
            expected_steps=40*math.ceil(experiment['data']['training_windows']/256)
            assert metrics['steps']==expected_steps
            assert not metrics['test_accessed'] and not cost['test_accessed']
            assert experiment['selection_users']==4096
            assert experiment['has_fm_module']==(name=='control')
            assert cost['has_fm_module']==(name=='control')
            assert cost['checkpoint_has_fm_parameters']==(name=='control')
            assert cost['fm_queries_per_user']==(32 if name=='control' else 0)
            assert config['recommender']['n_layer']==2 and config['recommender']['lr']==.001
            assert config['training']['seed']==2024
            for filename,key in [('selection_flow.npz','validation_selection'),('full_flow.npz','validation')]+([('full_direct_readout.npz','validation_direct_readout')] if name=='control' else []):
                with np.load(directory/filename) as saved:
                    recalculated=ranking_metrics(saved['predictions'],saved['targets'])
                    assert len(saved['targets'])==(4096 if key=='validation_selection' else experiment['data']['users'])
                    for metric,value in recalculated.items():
                        assert abs(value-metrics[key][metric])<1e-12,(category,name,key,metric)
            hashes=json.loads((directory/'source_hashes.json').read_text())
            assert json_hash(hashes)==experiment['source_snapshot_sha256']
            for relative,digest in hashes.items():
                assert hashlib.sha256((directory/'source_snapshot'/relative).read_bytes()).hexdigest()==digest
            runtime={p:h for p,h in hashes.items() if p.startswith(('models/','joint_training/','compression/','misc/')) or p in {'run.py','protocol.py','research/measure_inference.py'}}
            runtime_sources.append(runtime)
            metadata.append({key:experiment[key] for key in ['shared_initial_state_sha256','source','selection_indices_sha256','data']})
            pair[name]={**metrics,'inference':cost,'parameters':experiment['trainable_parameters'],
                'shared_initial_state_sha256':experiment['shared_initial_state_sha256'],
                'history':history,'directory':str(directory)}
        assert metadata[0]==metadata[1],category
        control,direct=pair['control'],pair['direct_ce']
        pair['direct_ce_gain_percent']=100*(direct['validation']['ndcg@10']/control['validation']['ndcg@10']-1)
        pair['direct_ce_selection_gain_percent']=100*(direct['validation_selection']['ndcg@10']/control['validation_selection']['ndcg@10']-1)
        pair['control_flow_over_own_head_gain_percent']=100*(control['validation']['ndcg@10']/control['validation_direct_readout']['ndcg@10']-1)
        pair['direct_inference_speedup']=control['inference']['mean_ms_per_user']/direct['inference']['mean_ms_per_user']
        paired=json.loads((base/category/'paired_inference_cost.json').read_text())
        assert paired['category']==category and not paired['test_accessed']
        assert paired['users']==512 and paired['batch_size']==32 and paired['repetitions']==3
        for name in ['control','direct_ce']:
            assert paired['checkpoint_sha256'][name]==pair[name]['inference']['checkpoint_sha256']
            assert paired['selection_indices_sha256']==pair[name]['inference']['selection_indices_sha256']
            pair[name]['peak_memory_reduction_percent_vs_control']=100*(1-pair[name]['peak_memory_allocated_mib']/control['peak_memory_allocated_mib'])
        pair['paired_inference']=paired
        pair['identity']=metadata[0]
        result['datasets'][category]=pair
    assert all(source==runtime_sources[0] for source in runtime_sources)
    for relative,digest in runtime_sources[0].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest
    protected=json.loads((ROOT/'docs/protected_versions.json').read_text())
    changes={name:{'before_sha256':digest,'after_sha256':hashlib.sha256((ROOT.parent/name).read_bytes()).hexdigest() if (ROOT.parent/name).exists() else None}
        for name,digest in protected.items() if not (ROOT.parent/name).exists() or hashlib.sha256((ROOT.parent/name).read_bytes()).hexdigest()!=digest}
    # Other work in the shared workspace can update v5_1 independently. Preserve
    # and report its exact changes; never silently rewrite the protection baseline.
    # All six runs above use the independent, hash-verified v5_6 runtime only.
    assert all(name.startswith('UGRec_v5_1/') for name in changes),changes
    result['audit']={'shared_initial_data_selection_matched_per_pair':True,'same_runtime_source_across_six_runs':True,
        'runtime_source_sha256':json_hash(runtime_sources[0]),'all_saved_source_hashes_verified':True,
        'all_predictions_metrics_recomputed':True,'all_checkpoint_reloads_verified':True,
        'paired_inference_checkpoint_and_users_matched':True,
        'old_code_config_unchanged':not any(not name.endswith('.md') for name in changes),
        'protected_unchanged_outside_v5_1':True,'protected_files':len(protected),
        'observed_protected_changes':changes,
        'protection_note':'v5_1 changed in the shared workspace during this task; changes retained and listed. No v5_1 runtime imports; all six v5_6 runtime snapshots match.'}
    result['parameter_reduction_percent']=100*(1-result['datasets']['Beauty']['direct_ce']['parameters']/result['datasets']['Beauty']['control']['parameters'])
    write_json(base/'summary.json',result)
    lines=['# Original control vs HSTU direct CE','',
        'Single seed 2024; 40 epochs; 4096-user validation selection; full validation audit; no test.','',
        '| Dataset | Control NDCG10 | Direct CE NDCG10 | Direct gain | Control own-head readout | Control Recall10 | Direct Recall10 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for category,pair in result['datasets'].items():
        c,d=pair['control'],pair['direct_ce']
        lines.append(f"| {category} | {c['validation']['ndcg@10']:.8f} | {d['validation']['ndcg@10']:.8f} | {pair['direct_ce_gain_percent']:+.2f}% | {c['validation_direct_readout']['ndcg@10']:.8f} | {c['validation']['recall@10']:.8f} | {d['validation']['recall@10']:.8f} |")
    lines+=['','Interleaved timing on cuda:3 (same 512 users; alternating methods, batch 32; shared GPU):','',
            '| Dataset | Control ms/user | Direct ms/user | Speedup | Control peak training MiB | Direct peak training MiB |',
            '|---|---:|---:|---:|---:|---:|']
    for category,pair in result['datasets'].items():
        c,d=pair['control'],pair['direct_ce']
        timing=pair['paired_inference']
        lines.append(f"| {category} | {timing['measurements']['control']['mean_wall_ms_per_user']:.4f} | {timing['measurements']['direct_ce']['mean_wall_ms_per_user']:.4f} | {timing['wall_speedup']:.2f}x | {c['peak_memory_allocated_mib']:.1f} | {d['peak_memory_allocated_mib']:.1f} |")
    lines+=['','Original separate timing (retained; shared-load drift can reverse rankings):','',
            '| Dataset | Control ms/user | Direct ms/user | Speedup |','|---|---:|---:|---:|']
    for category,pair in result['datasets'].items():
        c,d=pair['control'],pair['direct_ce']
        lines.append(f"| {category} | {c['inference']['mean_ms_per_user']:.4f} | {d['inference']['mean_ms_per_user']:.4f} | {pair['direct_inference_speedup']:.2f}x |")
    lines+=['',f"Parameters reduced by {result['parameter_reduction_percent']:.2f}%. Shared-device timings are descriptive, not exclusive throughput benchmarks.",
            '', 'Audit: '+json.dumps(result['audit'],ensure_ascii=False)]
    (base/'summary.md').write_text('\n'.join(lines)+'\n')
    fig,axs=plt.subplots(2,3,figsize=(13.8,7.6),layout='constrained')
    for column,(category,pair) in enumerate(result['datasets'].items()):
        for name,label,color in [('control','Control','#4c78a8'),('direct_ce','Direct CE','#f58518')]:
            selected=[row for row in pair[name]['history'] if 'validation_selection'in row]
            axs[0,column].plot([row['epoch']for row in selected],[row['validation_selection']['ndcg@10'] for row in selected],label=label,color=color,marker='o',markersize=3)
        axs[0,column].set(title=category.replace('_and_',' / '),xlabel='Epoch',ylabel='Selection NDCG@10')
        axs[0,column].legend()
        values=[pair['control']['validation']['ndcg@10'],pair['direct_ce']['validation']['ndcg@10'],pair['control']['validation_direct_readout']['ndcg@10']]
        axs[1,column].bar(range(3),values,color=['#4c78a8','#f58518','#999999'],width=.65)
        axs[1,column].set_xticks(range(3),['Control','Direct CE','Control head\n(same checkpoint)'],fontsize=9)
        axs[1,column].set(ylabel='Full validation NDCG@10',ylim=(0,max(values)*1.2))
        for index,value in enumerate(values):axs[1,column].text(index,value+max(values)*.025,f'{value:.5f}',ha='center',fontsize=9)
    for ax in axs.flat:ax.grid(alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('Original control vs removal of the generation branch | matched backbone, seed 2024 | no test',fontsize=12)
    fig.savefig(base/'comparison.png',dpi=180);fig.savefig(base/'comparison.pdf');plt.close(fig)
    print('\n'.join(lines))


if __name__=='__main__':main()
