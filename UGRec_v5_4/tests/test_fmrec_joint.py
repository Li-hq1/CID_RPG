import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from io import BytesIO

import torch
from torch import nn
from torch.nn import functional as F

from joint_training.fmrec_config import FMRecConfig
from joint_training.fmrec_data import PrefixDataset
from joint_training.fmrec_model import FMRecJointModel,balanced_history_mse
from joint_training.fmrec_training import FMRecTrainer
from joint_training.checkpoint import evaluation_randomness
from models.fmrec import EndpointNetwork,euler_sample,sample_time
from run_fmrec import load_continuation
from joint_training.checkpoint import random_state


ROOT=Path(__file__).resolve().parents[1]


def fixture(method='fmrec',**settings):
    torch.manual_seed(37)
    c=FMRecConfig.from_dict({'data':{'max_item_seq_len':4},
        'ae':{'architecture':'linear','latent_dim':8,'batch_size':5,'lr':.001},
        'fmrec':{'method':method,'fm_weight':1. if method=='fmrec' else 0.,
                 'num_blocks':2,'heads':2,'last':1,'dropout':0.,'emb_dropout':0.,'sample_N':3,**settings},
        'training':{'local_weight':0.}})
    return FMRecJointModel(c,torch.randn(5,10))


def batch():return torch.tensor([[0,0,1,2],[0,2,3,4]]),torch.tensor([3,5])


class JointFMRecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_endpoint_matches_local_official_implementation(self):
        source=ROOT.parent/'FMRec/src/fmrec.py'
        if not source.exists():self.skipTest('Local reference repository unavailable.')
        tree=ast.parse(source.read_text())
        # Only unused third-party imports are removed; all original model classes
        # are executed without rewriting their computations.
        tree.body=[node for node in tree.body if not (
            isinstance(node,ast.ImportFrom) and node.module.startswith(('timm','scipy')))]
        tree.body=[node for node in tree.body if not (
            isinstance(node,ast.Import) and any(alias.name.startswith('scipy') for alias in node.names))]
        namespace={};exec(compile(tree,str(source),'exec'),namespace)
        args=SimpleNamespace(hidden_size=8,dropout=0.,num_blocks=2,last=1,lambda_uncertainty=.001,item_num=6)
        reference=namespace['FM_xstart'](8,args).eval()
        settings=fixture().config.fmrec
        settings.heads=4
        actual=EndpointNetwork(8,settings,5).eval()
        actual.load_state_dict({key:value for key,value in reference.state_dict().items()
                                if key in actual.state_dict()},strict=True)
        history=torch.randn(2,4,8);xt=torch.randn(2,8);times=torch.tensor([1.,517.])
        mask=torch.tensor([[0,0,1,1],[0,1,1,1]])
        torch.manual_seed(72);expected,_=reference(history,xt,times,mask)
        torch.manual_seed(72);result,_=actual(history,xt,times,mask)
        torch.testing.assert_close(result,expected,rtol=0,atol=0)

    def test_full_ce_and_fm_match_reference_formula_and_both_train_ae(self):
        model=fixture();ids,targets=batch()
        loss,terms=model.objectives(ids,targets)
        torch.testing.assert_close(loss,terms['fm']+.2*terms['ce'])
        for name,term in terms.items():
            grads=torch.autograd.grad(term,list(model.ae.encoder.parameters()),retain_graph=True)
            self.assertGreater(sum(float(g.square().sum()) for g in grads),0.,name)
        loss.backward()
        self.assertGreater(float(model.endpoint.att.transformer_blocks[0].attention.w_layer.weight.grad.norm()),0.)
        self.assertFalse(any(isinstance(module,nn.Embedding) for module in model.modules()))

    def test_history_bridge_target_and_catalog_remain_live(self):
        model=fixture();ids,targets=batch()
        original=model.endpoint.forward
        seen={}
        def record(history,xt,times,mask,**kwargs):
            seen['history']=history;seen['xt']=xt
            history.retain_grad();xt.retain_grad()
            return original(history,xt,times,mask,**kwargs)
        with patch.object(model.endpoint,'forward',side_effect=record):
            objective,_=model.objectives(ids,targets);objective.backward()
        self.assertGreater(float(seen['history'].grad.norm()),0.)
        self.assertGreater(float(seen['xt'].grad.norm()),0.)
        # Remove all prediction dependence on AE to isolate live clean-target and
        # classification-catalog gradients; each loss must still update encoder.
        model.zero_grad(set_to_none=True)
        with patch.object(model.endpoint,'forward',return_value=(torch.randn(2,8,requires_grad=True),None)):
            _,terms=model.objectives(ids,targets)
        for term in terms.values():
            grads=torch.autograd.grad(term,list(model.ae.encoder.parameters()),retain_graph=True)
            self.assertGreater(sum(float(g.square().sum()) for g in grads),0.)

    def test_prefix_expansion_has_no_future_or_target_leakage(self):
        dataset=PrefixDataset([[1,2,3,4],[5,6]],3)
        self.assertEqual(len(dataset),4)
        expected=[([0,0,1],2),([0,1,2],3),([1,2,3],4),([0,0,5],6)]
        for index,(prefix,target) in enumerate(expected):
            ids,label=dataset[index];self.assertEqual(ids.tolist(),prefix);self.assertEqual(int(label),target)
        long=PrefixDataset([list(range(1,8))],3)
        self.assertEqual(long[5][0].tolist(),[4,5,6])

    def test_default_mse_is_absent_and_optional_mse_updates_ae(self):
        model=fixture()
        self.assertIsNone(model.endpoint.decoder)
        self.assertFalse(any('decoder' in name and not name.startswith('ae.') for name in model.state_dict()))
        model=fixture(mse_weight=.4)
        objective,terms=model.objectives(*batch())
        self.assertIn('history_mse',terms)
        grads=torch.autograd.grad(terms['history_mse'],list(model.ae.encoder.parameters()),retain_graph=True)
        self.assertGreater(sum(float(g.square().sum()) for g in grads),0.)
        torch.testing.assert_close(objective,terms['fm']+.2*terms['ce']+.4*terms['history_mse'])
        output=torch.zeros(2,6,requires_grad=True)
        torch.testing.assert_close(balanced_history_mse(batch()[0],output,1.),torch.tensor(5/12))

    def test_ablation_shares_transformer_initialization_and_removes_time_module(self):
        flow=fixture();no_fm=fixture('ce_only');direct=fixture('history_ce')
        for name,value in direct.state_dict().items():
            torch.testing.assert_close(value,flow.state_dict()[name],rtol=0,atol=0)
        self.assertEqual(set(flow.state_dict()),set(no_fm.state_dict()))
        self.assertFalse(hasattr(direct.endpoint,'time_embed'))
        objective,terms=direct.objectives(*batch())
        self.assertEqual(set(terms),{'ce'});torch.testing.assert_close(objective,.2*terms['ce'])

    def test_reference_euler_uses_fixed_original_noise(self):
        model=fixture();settings=model.config.fmrec
        z0=torch.tensor([[1.,2.]])
        class Network:
            def __init__(self):self.times=[]
            def __call__(self,history,state,t,mask):self.times.append(t);return state*2,None
        network=Network();state=z0.clone()
        for _ in range(3):state=state+(2*state-z0)/3
        actual=euler_sample(network,None,None,z0,settings)
        torch.testing.assert_close(actual,state)
        self.assertAlmostEqual(float(network.times[0][0]),.999,places=5)

    def test_time_distributions_and_inference_rng_are_reproducible(self):
        model=fixture()
        for mode in ['mode','uniform','logit_normal','cosmap']:
            model.config.fmrec.sampling_method=mode
            times=sample_time(model.config.fmrec,1000,'cpu')
            self.assertTrue(bool(((times>=.001)&(times<=1)).all()))
        model.eval();model.refresh_catalog();ids,_=batch()
        rng=torch.get_rng_state()
        with evaluation_randomness('cpu',83):first=model.log_probabilities(ids)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        with evaluation_randomness('cpu',83):second=model.log_probabilities(ids)
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        direct=fixture('history_ce').eval();direct.refresh_catalog()
        first=direct.log_probabilities(ids);torch.manual_seed(883)
        torch.testing.assert_close(first,direct.log_probabilities(ids),rtol=0,atol=0)

    def test_joint_optimizer_updates_ae_from_recommendation_without_geometry_loss(self):
        model=fixture();trainer=FMRecTrainer(model,torch.arange(1,6))
        before=model.ae.encoder[0].weight.detach().clone()
        trainer.step(*batch())
        self.assertFalse(torch.equal(before,model.ae.encoder[0].weight))
        self.assertGreater(trainer.gradient_audit['ce_ae_encoder_grad_norm'],0)
        self.assertGreater(trainer.gradient_audit['fm_ae_encoder_grad_norm'],0)
        self.assertEqual(trainer.optimizer.param_groups[0]['lr'],.001)
        self.assertEqual(trainer.config.training.local_weight,0.)

    def test_default_entry_is_fmrec_and_mse_disabled(self):
        final=FMRecConfig.read(ROOT/'configs/final.json')
        config=FMRecConfig.read(ROOT/'configs/fmrec.json')
        self.assertEqual(final.to_dict(),config.to_dict())
        self.assertEqual((config.fmrec.fm_weight,config.fmrec.ce_weight,config.fmrec.mse_weight),(1.,.2,0.))
        self.assertIn('run_fmrec.py',(ROOT/'scripts/train.sh').read_text())
        value=config.to_dict();value['ae']['frozen']=True
        with self.assertRaises(ValueError):FMRecConfig.from_dict(value)

    def test_continuation_restores_shuffling_dropout_adam_and_scheduler(self):
        from torch.utils.data import DataLoader
        model=fixture(dropout=.1,emb_dropout=.3)
        model.config.training.decay_step=1
        model.config.training.epochs=3
        generator=torch.Generator().manual_seed(71)
        dataset=PrefixDataset([[1,2,3],[2,4,5]],4)
        loader=DataLoader(dataset,batch_size=2,shuffle=True,generator=generator)
        trainer=FMRecTrainer(model,torch.arange(1,6),generator)
        def epoch(current,current_loader):
            losses=[current.step(ids,labels) for ids,labels in current_loader]
            current.scheduler.step()
            return losses
        epoch(trainer,loader)
        buffer=BytesIO()
        torch.save({'format':'fmrec_joint_v1','source':{},'config':model.config.to_dict(),
            'model':model.state_dict(),'trainer':trainer.state_dict(),'epoch':1,'rng':random_state('cpu')},buffer)
        buffer.seek(0);saved=torch.load(buffer,weights_only=True)
        expected=epoch(trainer,loader)
        restored=fixture(dropout=.1,emb_dropout=.3)
        restored.config.training.decay_step=1
        restored.config.training.epochs=4
        restored.config.training.run_id='continuation_test'
        restored_generator=torch.Generator().manual_seed(99)
        restored_loader=DataLoader(dataset,batch_size=2,shuffle=True,generator=restored_generator)
        restored_trainer=FMRecTrainer(restored,torch.arange(1,6),restored_generator)
        load_continuation(saved,restored_trainer,{})
        self.assertEqual(expected,epoch(restored_trainer,restored_loader))
        for name,value in model.state_dict().items():
            torch.testing.assert_close(value,restored.state_dict()[name],rtol=0,atol=0)
        self.assertEqual(trainer.scheduler.state_dict(),restored_trainer.scheduler.state_dict())
        self.assertEqual(trainer.steps,restored_trainer.steps)
        restored.config.fmrec.fm_weight=.5
        with self.assertRaisesRegex(ValueError,'fmrec semantics'):
            load_continuation(saved,restored_trainer,{})


if __name__=='__main__':unittest.main()
