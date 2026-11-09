"""Audit completed fixed-budget runs and render a descriptive single-seed report."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from misc.artifacts import json_hash, write_json
from protocol import ROOT, ranking_metrics

NAMES = ['control','ce_only','joint','fm01','fm5','source_ce']
LABELS = {'control':'Control','ce_only':'Bridge CE','joint':'Bridge CE + FM (1)',
          'fm01':'Bridge CE + FM (0.1)','fm5':'Bridge CE + FM (5)','source_ce':'Source CE'}


def main():
    base=ROOT/'outputs/clean_endpoint'
    summary={'seed':2024,'category':'Beauty','test_accessed':False,'runs':{},'audit':{}}
    identities=[]; runtime_sources=[]
    for name in NAMES:
        directory=base/'runs/Beauty'/f'endpoint_{name}_s2024'
        result=json.loads((directory/'results.json').read_text())
        experiment=json.loads((directory/'experiment.json').read_text())
        config=json.loads((directory/'config.json').read_text())
        history=json.loads((directory/'metrics.json').read_text())
        assert result['checkpoint_reload_verified'] and result['benchmark_eligible']
        assert result['steps']==4040 and result['completed_epochs']==40
        assert not result['test_accessed'] and experiment['selection_users']==4096
        assert len(history)==40
        for filename,key in [('selection_flow.npz','validation_selection'),('full_flow.npz','validation')]:
            with np.load(directory/filename) as saved:
                assert saved['predictions'].shape==(4096 if key=='validation_selection' else 22363,10)
                recalculated=ranking_metrics(saved['predictions'],saved['targets'])
                for metric,value in recalculated.items():
                    assert abs(value-result[key][metric])<1e-12,(name,filename,metric)
        hashes=json.loads((directory/'source_hashes.json').read_text())
        assert json_hash(hashes)==experiment['source_snapshot_sha256']
        for relative,digest in hashes.items():
            assert hashlib.sha256((directory/'source_snapshot'/relative).read_bytes()).hexdigest()==digest
        runtime={p:h for p,h in hashes.items() if p.startswith(('models/','joint_training/','compression/','misc/')) or p in {'run.py','protocol.py'}}
        runtime_sources.append(runtime)
        identities.append({key:experiment[key] for key in ['initial_state_sha256','source','selection_indices_sha256','trainable_parameters','data']})
        row={'label':LABELS[name],**result,'initial_state_sha256':experiment['initial_state_sha256'],
             'runtime_source_sha256':json_hash(runtime),'device':experiment['runtime']['device'],
             'fm_weight':config['flow_ranking']['fm_weight'] if name!='control' else None,
             'ce_weight':config['flow_ranking']['ce_weight'] if name!='control' else None,
             'endpoint_training':config['flow_ranking']['endpoint_training'] if name!='control' else None,
             'history':history,'directory':str(directory)}
        probe=directory/'training_clock_probe.json'
        if probe.exists(): row['training_clock_probe']=json.loads(probe.read_text())
        summary['runs'][name]=row
    assert all(v==identities[0] for v in identities)
    assert all(v==runtime_sources[0] for v in runtime_sources)
    for path,digest in runtime_sources[0].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,('runtime changed',path)
    protected=json.loads((ROOT/'docs/protected_versions.json').read_text())
    changes=[name for name,digest in protected.items() if not (ROOT.parent/name).is_file() or hashlib.sha256((ROOT.parent/name).read_bytes()).hexdigest()!=digest]
    summary['audit']={'identity_equal_across_runs':True,'runtime_source_equal_across_runs':True,
        'saved_source_snapshots_verified':True,'saved_predictions_metrics_recomputed':True,
        'protected_files':len(protected),'changed_protected_files':changes,
        'runtime_source_sha256':json_hash(runtime_sources[0]),'trainable_parameters':identities[0]['trainable_parameters']}
    # Shared workspace: report independently changed documents without reverting them.
    # Any protected model, code, config or script change still fails this audit.
    assert all(name.endswith('.md') for name in changes),changes
    summary['audit']['protected_code_config_unchanged']=True
    summary['audit']['observed_document_changes']={name:{'before_sha256':protected[name],
        'after_sha256':hashlib.sha256((ROOT.parent/name).read_bytes()).hexdigest()} for name in changes}
    variants=['ce_only','joint','fm01','fm5']
    chosen=max(variants,key=lambda name:summary['runs'][name]['validation_selection']['ndcg@10'])
    summary['best_bridge_variant_by_selection']=chosen
    summary['adopt_bridge_variant']=summary['runs'][chosen]['validation_selection']['ndcg@10']>summary['runs']['control']['validation_selection']['ndcg@10']
    write_json(base/'summary.json',summary)
    lines=['# Beauty / seed2024 clean-endpoint comparison','',
           'Each run: 40 epochs, 4040 updates, selection 4096, full validation 22363. Test unused. Single-seed trends only.','',
           '| Run | Epoch | Selection NDCG@10 | Full NDCG@10 | Full Recall@10 | Full NLL | Seconds |','|---|---:|---:|---:|---:|---:|---:|']
    for name,row in summary['runs'].items():
        s,v=row['validation_selection'],row['validation']
        lines.append(f"| {LABELS[name]} | {row['best']['epoch']} | {s['ndcg@10']:.8f} | {v['ndcg@10']:.8f} | {v['recall@10']:.8f} | {v['nll']:.5f} | {row['seconds']:.1f} |")
    lines+=['',f'Best bridge variant by selection: {chosen}. Adopt over control: {summary["adopt_bridge_variant"]}.',
            '', 'Audit: '+json.dumps(summary['audit'],ensure_ascii=False),
            '', 'Wall times were measured with unrelated workloads on the same devices and are not exclusive throughput benchmarks.']
    (base/'summary.md').write_text('\n'.join(lines)+'\n')
    colors={'control':'#222222','ce_only':'#4c78a8','joint':'#f58518','fm01':'#54a24b','fm5':'#e45756','source_ce':'#b279a2'}
    fig,axs=plt.subplots(1,3,figsize=(16,4.6),layout='constrained')
    for name,row in summary['runs'].items():
        selected=[r for r in row['history'] if 'validation_selection'in r]
        axs[0].plot([r['epoch'] for r in selected],[r['validation_selection']['ndcg@10'] for r in selected],label=LABELS[name],color=colors[name],linestyle='--' if name=='source_ce' else '-',linewidth=1.8)
        if name in ['ce_only','joint','fm01','fm5']:
            axs[1].plot([r['epoch'] for r in row['history']],[r['train']['loss/fm'] for r in row['history']],label=LABELS[name],color=colors[name])
    axs[0].set(xlabel='Epoch',ylabel='NDCG@10',title='Fixed validation selection (4096 users)')
    axs[0].legend(fontsize=8,loc='lower right')
    axs[1].set(xlabel='Epoch',ylabel='Mean coordinate MSE',title='Training endpoint error (unweighted)')
    for i,name in enumerate(NAMES):
        value=summary['runs'][name]['validation']['ndcg@10']
        axs[2].barh(i,value,color=colors[name],height=.65)
        axs[2].text(value+.0002,i,f'{value:.5f}',va='center',fontsize=9)
    axs[2].set_yticks(range(len(NAMES)),[LABELS[n] for n in NAMES],fontsize=9)
    axs[2].invert_yaxis();axs[2].set_xlim(0,.068)
    axs[2].set(xlabel='NDCG@10',title='Locked-checkpoint full validation audit')
    for ax in axs:ax.grid(alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('Beauty, seed 2024 | same backbone and 32-query decoding | no test evaluation',fontsize=12)
    fig.savefig(base/'comparison.png',dpi=180)
    fig.savefig(base/'comparison.pdf')
    plt.close(fig)
    print('\n'.join(lines))

if __name__=='__main__':
    main()
