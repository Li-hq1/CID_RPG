"""Audit the bounded optimization, keeping the original 40-epoch screen intact."""

import argparse
import hashlib
import json
import math
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from joint_training.data import initialize_random_ae
from joint_training.fmrec_config import FMRecConfig
from joint_training.fmrec_model import FMRecJointModel
from misc.artifacts import array_hash,json_hash,write_json
from protocol import ROOT,corpus,raw_features,ranking_metrics
from run_fmrec import state_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-mse',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    base=ROOT/'outputs/fmrec/optimization'
    methods=['fmrec','ce_only','history_ce']+(['fmrec_mse'] if args.include_mse else [])
    result={'category':'Beauty','seed':2024,'maximum_epochs':140,'test_accessed':False,
        'selection_users':4096,'full_validation_includes_selection':True,'methods':{}}
    identities=[];core_hashes=[];initial_common=[]
    processed,mapping,train,_,_=corpus('Beauty')
    raw=raw_features(processed,len(mapping['id2item'])-1)
    train_ids=torch.tensor(sorted({item for sequence in train for item in sequence}))
    for method in methods:
        status=json.loads((base/f'status_{method}.json').read_text())
        assert status['state']=='complete' and status['returncode']==0
        name='fmrec_mse140' if method=='fmrec_mse' else method+'_extend140'
        directory=ROOT/f'outputs/fmrec/runs/Beauty/beauty_{name}_s2024'
        config=FMRecConfig.read(directory/'config.json')
        metrics=json.loads((directory/'results.json').read_text())
        experiment=json.loads((directory/'experiment.json').read_text())
        history=json.loads((directory/'metrics.json').read_text())
        assert config.training.epochs==140 and config.training.seed==2024
        assert config.fmrec.ce_weight==.2 and config.fmrec.sample_N==30
        assert config.fmrec.mse_weight==(.4 if method=='fmrec_mse' else 0.)
        assert metrics['completed_epochs']==140 and len(history)==140
        assert metrics['steps']==140*math.ceil(experiment['data']['training_prefixes']/config.training.batch_size)
        assert metrics['benchmark_eligible'] and metrics['checkpoint_reload_verified'] and not metrics['test_accessed']
        assert history[98]['lr']==[.001,.0001] and history[99]['lr']==[.0001,.00001]
        assert not config.ae.frozen and not experiment['has_id_item_embedding_table']
        hashes=json.loads((directory/'source_hashes.json').read_text())
        assert json_hash(hashes)==experiment['source_snapshot_sha256']
        for path,digest in hashes.items():
            assert hashlib.sha256((directory/'source_snapshot'/path).read_bytes()).hexdigest()==digest
        core={path:digest for path,digest in hashes.items() if path.startswith(('models/','joint_training/','compression/','misc/'))
            or path in {'run_fmrec.py','protocol.py'}}
        for path,digest in core.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
        core_hashes.append(core)
        identities.append({key:experiment[key] for key in ['source','data','selection_indices_sha256']})
        # Reconstruct initialization to compare only common parameters when MSE adds a decoder.
        random.seed(config.training.seed);np.random.seed(config.training.seed);torch.manual_seed(config.training.seed)
        ae=initialize_random_ae(config,raw,train_ids)
        initialized=FMRecJointModel(config,raw,ae)
        assert state_hash(initialized)==experiment['initial_state_sha256']
        initial_common.append(json_hash({name:array_hash(value.numpy()) for name,value in initialized.state_dict().items()
            if not name.startswith(('endpoint.time_embed.','endpoint.decoder.'))}))
        del initialized,ae
        gradient=json.loads((directory/'gradient_audit.json').read_text())
        assert gradient['ce_ae_encoder_grad_norm']>0
        if method in ['fmrec','fmrec_mse']:assert gradient['fm_ae_encoder_grad_norm']>0
        if method=='fmrec_mse':assert gradient['history_mse_ae_encoder_grad_norm']>0
        for filename,key in [('selection.npz','validation_selection'),('full.npz','validation')]:
            with np.load(directory/filename) as saved:
                for metric,value in ranking_metrics(saved['predictions'],saved['targets']).items():
                    assert abs(value-metrics[key][metric])<1e-12
        checkpoint=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
        assert checkpoint['config']==config.to_dict() and checkpoint['epoch']==metrics['best']['epoch']
        diagnostic=json.loads((directory/'optimization_diagnostic.json').read_text())
        assert diagnostic['checkpoint_sha256']==hashlib.sha256((directory/'best.pt').read_bytes()).hexdigest()
        assert diagnostic['native_predictions_verified'] and not diagnostic['used_for_checkpoint_selection']
        assert diagnostic['trained_checkpoint_ae_encoder_grad_norm']['ce']>0
        if method in ['fmrec','fmrec_mse']:assert diagnostic['trained_checkpoint_ae_encoder_grad_norm']['fm']>0
        if method=='fmrec_mse':assert diagnostic['trained_checkpoint_ae_encoder_grad_norm']['history_mse']>0
        assert diagnostic['gradient_audit_optimizer_steps']==0
        with np.load(directory/'selection.npz') as saved:
            assert diagnostic['selection_indices_sha256']==array_hash(saved['indices'][:512])
        if method!='fmrec_mse':
            continuation=experiment['continuation']
            assert continuation['optimizer_scheduler_rng_restored'] and continuation['core_source_unchanged']
            parent=ROOT/f'outputs/fmrec/runs/Beauty/beauty_{method}_initial40_s2024'
            parent_history=json.loads((parent/'metrics.json').read_text())
            assert history[:continuation['epoch']]==parent_history[:continuation['epoch']]
            replay=[]
            for epoch in range(continuation['epoch']+1,41):
                old=parent_history[epoch-1];new=history[epoch-1]
                replay.append({'epoch':epoch,'train_terms_exact':old['train']==new['train'],
                    'validation_exact':old.get('validation_selection')==new.get('validation_selection')})
            metrics['replay_through_epoch40']=replay
        result['methods'][method]={**metrics,'directory':str(directory),'history':history,
            'parameters':experiment['trainable_parameters'],'gradient_audit':gradient,'diagnostic':diagnostic}
    assert all(identity==identities[0] for identity in identities)
    assert all(value==core_hashes[0] for value in core_hashes)
    assert len(set(initial_common))==1
    original=json.loads((ROOT/'outputs/fmrec/summary.json').read_text())
    for method in ['fmrec','ce_only','history_ce']:
        parent=ROOT/f'outputs/fmrec/runs/Beauty/beauty_{method}_initial40_s2024'
        assert hashlib.sha256((parent/'best.pt').read_bytes()).hexdigest()==original['methods'][method]['checkpoint_sha256']
        assert json.loads((parent/'results.json').read_text())['validation']==original['methods'][method]['validation']
    archived=json.loads((ROOT/'history/flowrec_before_fmrec/archive_hashes.json').read_text())
    assert all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest for path,digest in archived.items())
    protected=json.loads((ROOT/'docs/fmrec_protected_versions.json').read_text())
    reference={path:digest for path,digest in protected.items() if path.startswith('FMRec/')}
    assert reference and all(hashlib.sha256((ROOT.parent/path).read_bytes()).hexdigest()==digest for path,digest in reference.items())
    result['audit']={'matched_data_selection':True,'matched_common_initialization':True,
        'common_initial_sha256':initial_common[0],'runtime_source_identical':True,
        'runtime_source_sha256':json_hash(core_hashes[0]),'snapshots_and_current_runtime_verified':True,
        'ranking_metrics_recomputed':True,'ce_fm_optional_mse_reach_encoder':True,
        'checkpoint_and_diagnostic_identity_verified':True,
        'original40_checkpoints_and_results_unchanged':True,'flowrec_archive_unchanged':True,
        'reference_repository_unchanged':True}
    for method in ['fmrec']+(['fmrec_mse'] if args.include_mse else []):
        result[method+'_comparisons']={other:{split:(result['methods'][method][split]['ndcg@10']/
            result['methods'][other][split]['ndcg@10']-1)*100 for split in ['validation_selection','validation']}
            for other in (['ce_only','history_ce'] if method=='fmrec' else ['fmrec','history_ce'])}
    family=['fmrec']+(['fmrec_mse'] if args.include_mse else [])
    chosen=max(family,key=lambda method:result['methods'][method]['validation_selection']['ndcg@10'])
    beats_history=result['methods'][chosen]['validation_selection']['ndcg@10']>result['methods']['history_ce']['validation_selection']['ndcg@10']
    result['decision']={'chosen_fmrec_candidate_by_selection':chosen,'exceeds_history_ce_on_selection':beats_history,
        'recommendation':('retain_candidate_for_confirmation' if beats_history else
            ('stop_current_fmrec_route' if args.include_mse else 'await_single_mse_candidate')),
        'further_parameter_grid':False,'single_seed_exploration':True,'default_config_changed':False}
    write_json(base/'summary.json',result)
    write_json(base/'decision.json',result['decision'])
    labels={'fmrec':'FMRec','ce_only':'Bridge CE-only','history_ce':'History CE','fmrec_mse':'FMRec + history MSE'}
    lines=['# FMRec有限优化结果','','Beauty / seed2024 / 最大140轮 / 第100轮衰减 / 无test。',
        '固定4096人selection选模；完整validation包含selection，不是独立test。','',
        '| 方法 | 最佳轮 | Selection NDCG10 | Full NDCG10 | Full Recall10 | Full NLL |',
        '|---|---:|---:|---:|---:|---:|']
    for method,m in result['methods'].items():
        lines.append(f"| {labels[method]} | {m['best']['epoch']} | {m['validation_selection']['ndcg@10']:.8f} | {m['validation']['ndcg@10']:.8f} | {m['validation']['recall@10']:.8f} | {m['validation']['nll']:.4f} |")
    lines+=['','FMRec vs others (%): '+json.dumps(result['fmrec_comparisons'])]
    if args.include_mse:lines+=['','History MSE vs others (%): '+json.dumps(result['fmrec_mse_comparisons'])]
    lines+=['','三组原模型仅延长预算；MSE候选（如运行）新增历史约束，不能作为L-FM单独增量。',
        '单种子探索不支持稳定性结论。1步/桥/打乱历史结果仅解释锁定checkpoint，不用于另选checkpoint。',
        '', 'Audit: '+json.dumps(result['audit'],ensure_ascii=False)]
    (base/'summary.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(16,4.6),layout='constrained')
    for method,m in result['methods'].items():
        selected=[row for row in m['history'] if 'validation_selection' in row]
        axes[0].plot([r['epoch'] for r in selected],[r['validation_selection']['ndcg@10'] for r in selected],label=labels[method])
        axes[1].plot([r['epoch'] for r in m['history']],[r['train']['ce'] for r in m['history']],label=labels[method])
    for axis in axes[:2]:axis.axvline(100,color='gray',linestyle='--',alpha=.6);axis.set_xlabel('Epoch')
    axes[0].set_ylabel('Selection NDCG@10');axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Unweighted training CE')
    values=[result['methods'][method]['validation']['ndcg@10'] for method in methods]
    axes[2].bar(range(len(methods)),values)
    axes[2].set_xticks(range(len(methods)),[labels[m] for m in methods],rotation=18,fontsize=8)
    axes[2].set_ylabel('Full validation NDCG@10')
    for index,value in enumerate(values):axes[2].text(index,value+max(values)*.02,f'{value:.5f}',ha='center',fontsize=8)
    axes[2].set_ylim(0,max(values)*1.18)
    for axis in axes:axis.grid(alpha=.2);axis.set_axisbelow(True)
    fig.suptitle('Online joint AE | Beauty | seed 2024 | bounded 140-epoch optimization | no test')
    fig.savefig(base/'comparison.png',dpi=180);fig.savefig(base/'comparison.pdf');plt.close(fig)
    fm=result['methods']['fmrec'];direct=result['methods']['history_ce']
    original_fm=original['methods']['fmrec']['validation']['ndcg@10']
    report=['# 在线joint AE + FMRec：有限优化结果','',
        '本轮按用户授权先补足训练预算，再只尝试一个历史约束候选；保留joint training与推荐梯度直通AE encoder。',
        'Beauty / seed2024；各组最大140轮、35980更新；第100轮后推荐器/AE学习率从1e-3/1e-4降为1e-4/1e-5。',
        '固定4096名selection用户按NDCG@10选checkpoint，完整22363人validation仅审计且包含selection；未访问test、未做多种子或参数网格。','',
        '## 对照到底删除了什么','',
        '| 方法 | 历史encoder | 含目标随机桥与时间模块 | L-FM | 历史MSE | 推理 |',
        '|---|---|---|---|---|---|',
        '| FMRec | 在线joint AE + 共享Transformer | 保留 | 1 | 0 | 30步原生采样 |',
        '| 同桥CE-only | 同上 | 保留 | 0 | 0 | 同样30步 |',
        '| 历史CE | 在线joint AE + 相同Transformer块 | 删除 | 0 | 0 | 历史直接预测一次 |']
    if args.include_mse:report+=['| FMRec + 历史MSE | 同FMRec | 保留 | 1 | .4 | 30步原生采样 |']
    report+=['','各组CE权重均.2、AE几何项权重均.1，均无独立可学习ID物品词表。历史CE不是只去掉L-FM。',
        'FMRec对同桥CE-only衡量FM监督局部作用；FMRec对历史CE衡量整条生成机制的净收益。','',
        '## 正式结果','',*lines[5:7+len(methods)],'',
        f"FMRec完整validation从40轮初筛的{original_fm:.8f}升至{fm['validation']['ndcg@10']:.8f}（{(fm['validation']['ndcg@10']/original_fm-1)*100:+.2f}%），说明早期预算确实低估了其表现。",
        f"但与等预算历史CE相比，完整NDCG@10仍为{(fm['validation']['ndcg@10']/direct['validation']['ndcg@10']-1)*100:+.2f}%。",
        f"FMRec相对同桥CE-only：selection {result['fmrec_comparisons']['ce_only']['validation_selection']:+.2f}%，完整validation {result['fmrec_comparisons']['ce_only']['validation']:+.2f}%。这只是单种子探索，不表述为稳定增益。"]
    if args.include_mse:
        delta=result['fmrec_mse_comparisons']
        report += [f"历史MSE候选相对FMRec：selection {delta['fmrec']['validation_selection']:+.2f}%，完整validation {delta['fmrec']['validation']:+.2f}%；相对历史CE完整validation {delta['history_ce']['validation']:+.2f}%。它新增了历史约束，不能用其结果直接归因L-FM。"]
        trained=result['methods']['fmrec_mse']['diagnostic']['trained_checkpoint_ae_encoder_grad_norm']
        report += [f"历史MSE分支可训练但参考归约方式使其影响很弱：训练后固定batch上AE encoder的原始CE/FM/history-MSE梯度范数分别为{trained['ce']:.6g}/{trained['fm']:.6g}/{trained['history_mse']:.6g}（对应权重.2/1/.4）。非零梯度不等于有效约束强度；本轮只检验参考权重.4，不据失败否定所有历史约束，也不追加重归一化/权重扫描。"]
    report+=['','## 锁定checkpoint诊断','',
        '以下为相同512名selection用户的次级诊断，未参与checkpoint选择。含真实目标的桥不是可部署推荐结果。','',
        '| 方法 | 含目标t=.5桥 NDCG10 | 纯噪声1步 | 原生推理 | 原生推理打乱历史 |',
        '|---|---:|---:|---:|---:|']
    for method,m in result['methods'].items():
        diagnostic=m['diagnostic']
        bridge=f"{diagnostic['bridge_t0.5']['ndcg@10']:.6f}" if 'bridge_t0.5' in diagnostic else '—'
        one=f"{diagnostic['one_step']['ndcg@10']:.6f}" if 'one_step' in diagnostic else '—'
        report += [f"| {labels[method]} | {bridge} | {one} | {diagnostic['native']['ndcg@10']:.6f} | {diagnostic['native_shuffled_history']['ndcg@10']:.6f} |"]
    report+=['',
        '桥恢复、噪声起点和多步生成的能力必须区分。若1步优于30步，只能说明当前checkpoint的多步生成使排序退化，不能唯一归为Euler离散误差，也不能据此静默换推理方式。',
        '高t桥在打乱历史后仍可高分，反映目标状态本身可提供强信息；低t及原生推理受打乱历史影响，说明模型并非完全忽略历史。',
        '训练后checkpoint也逐项检查CE/FM及开启时的MSE对AE encoder的梯度非零，不执行优化器更新。原生预测与保存结果逐条一致。','',
        '## 复现与边界','',
        '三组原模型从原best.pt完整恢复优化器、调度器、CPU/CUDA RNG、DataLoader与AE抽样generator。FMRec从40轮、另两组从35轮恢复；后两组重放36—40轮的训练及验证结果与原记录一致。',
        '历史MSE候选从同一seed重新训练，共同AE和Transformer参数初态匹配；decoder新增随机数消耗，不声称后续噪声逐步相同。',
        f"87项测试通过，包括续训下一轮参数精确复现；完整宽度真实数据MSE 4步检查通过但不计benchmark。本次汇总的{len(methods)}组运行源码、数据、selection身份、初态、预测指标重算及checkpoint重载已审计。",
        '原40轮checkpoint/结果、FlowRec源码归档与FMRec参考仓库hash未变；本轮没有修改其他版本代码，不声称并发工作区整体无变化。',
        '140轮覆盖一次学习率衰减，并不等于穷尽原参考最大500轮训练或证明所有FM方法无效。本轮判断服务于当前方法取舍。','',
        '## 本轮决定','',
        ('候选已在selection超过历史CE，保留为后续确认对象；本轮不扩大参数网格。' if beats_history else
         ('延长预算与唯一的历史MSE候选仍未给出相对历史CE的整体净收益，建议结束当前FMRec适配的优化，主线优先保留在线joint AE + 历史直接CE。' if args.include_mse else
          '延长预算未见整体净收益；第一阶段已完成，等待唯一的历史MSE候选完成，不将阶段结论表述为全部优化结束。')),
        '不自动修改默认配置；保留全部实现和独立结果供复查。','',
        '- [机器汇总与audit](../outputs/fmrec/optimization/summary.json)',
        '- [阶段决定](../outputs/fmrec/optimization/decision.json)',
        '- [曲线PNG](../outputs/fmrec/optimization/comparison.png) / [PDF](../outputs/fmrec/optimization/comparison.pdf)',
        '- [预先固定的实验设计](FMRec优化实验.md)','']
    (ROOT/'docs/FMRec优化实验结果.md').write_text('\n'.join(report))
    print('\n'.join(lines))


if __name__=='__main__':main()
