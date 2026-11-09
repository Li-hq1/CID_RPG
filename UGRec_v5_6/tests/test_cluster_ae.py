import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from compression.geometry_ae.cluster_model import ClusterAggregation
from joint_training.checkpoint import load_checkpoint,read_checkpoint,save_checkpoint
from joint_training.config import JointConfig
from joint_training.direct_ce import direct_log_probabilities
from joint_training.model import JointModel
from joint_training.training import JointTrainer
from tests.test_direct_ce import fixture
from tests.test_joint_model import make_batch


def cluster_model(aggregation="sum",history="multi"):
    base=fixture()
    config=base.config.to_dict()
    config["ae"].update(cluster_count=4,cluster_aggregation=aggregation)
    config["recommender"]["ae_cluster_history"]=history
    torch.manual_seed(29)
    return JointModel(JointConfig.from_dict(config),torch.randn(5,6))


class ClusterAETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_common_initial_weights_and_rng_unchanged(self):
        baseline=fixture();rng=torch.get_rng_state()
        model=cluster_model()
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        for key,value in baseline.state_dict().items():
            torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
        merged=cluster_model(history="merged")
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value,merged.state_dict()[key],rtol=0,atol=0)

    def test_raw_ae_latent_is_same_aggregate_for_decoder_and_catalog(self):
        model=cluster_model();raw=model.raw_items(torch.arange(1,6))
        components=model.ae.components(raw)
        reconstructed,z=model.ae(raw)
        torch.testing.assert_close(z,components.sum(1),rtol=0,atol=0)
        torch.testing.assert_close(reconstructed,model.ae.decode(z),rtol=0,atol=0)
        torch.testing.assert_close(model.encode_ids(torch.arange(1,6)),model.normalizer(z),rtol=0,atol=0)
        # Raw residuals retain a trainable amplitude; no unit-norm upper bound.
        with torch.no_grad():model.ae.cluster_tokenizer.fc0.bias.mul_(50)
        self.assertGreater(float(model.ae.components(raw).detach().norm(dim=-1).max()),1.)

    def test_reconstruction_and_ce_both_reach_cluster_parameters(self):
        model=cluster_model();raw=model.raw_items(torch.arange(1,6))
        reconstructed,_=model.ae(raw)
        F.mse_loss(reconstructed,raw).backward()
        for name in ["fc0.weight","fc_gk.weight","fc_g.weight","centers"]:
            grad=dict(model.ae.cluster_tokenizer.named_parameters())[name].grad
            self.assertIsNotNone(grad);self.assertGreater(float(grad.norm()),0.)
        model.zero_grad(set_to_none=True)
        loss,_=model.joint_objectives(make_batch());loss.backward()
        for name in ["fc0.weight","fc_gk.weight","fc_g.weight","centers"]:
            self.assertGreater(float(dict(model.ae.cluster_tokenizer.named_parameters())[name].grad.norm()),0.)

    def test_history_components_and_padding_and_future_causality(self):
        model=cluster_model();model.eval();batch=make_batch()
        tokens,raw,_,_=model.history_cluster_tokens(batch["input_ids"],return_details=True)
        expected=model.input_proj(model.normalizer(raw)).masked_fill(~batch["attention_mask"].bool()[...,None,None],0.)
        torch.testing.assert_close(tokens,expected,rtol=0,atol=0)
        self.assertEqual(float(tokens[~batch["attention_mask"].bool()].detach().abs().sum()),0.)
        encoded,_=model.encode_history(batch)
        expected=model.encoder(tokens.reshape(2,16,8),batch["attention_mask"].repeat_interleave(4,1),
                               item_group_size=4).reshape(2,4,4,8).sum(2)
        torch.testing.assert_close(encoded,expected,rtol=0,atol=0)
        changed=copy.deepcopy(batch);changed["input_ids"][0,1]=5
        actual,_=model.encode_history(changed)
        torch.testing.assert_close(actual[0,0],encoded[0,0],rtol=0,atol=0)
        changed=copy.deepcopy(batch);changed["input_ids"][~batch["attention_mask"].bool()]=5
        actual,_=model.encode_history(changed)
        torch.testing.assert_close(actual,encoded,rtol=0,atol=0)

    def test_optimizer_ownership_and_exact_resume(self):
        model=cluster_model();trainer=JointTrainer(model,model.config,torch.arange(1,6),{"fixture":True},4)
        ae_params={id(p)for g in trainer.ae_optimizer.param_groups for p in g["params"]}
        rec_params={id(p)for g in trainer.rec_optimizer.param_groups for p in g["params"]}
        self.assertFalse(ae_params&rec_params)
        self.assertTrue(all(id(p)in ae_params for p in model.ae.cluster_tokenizer.parameters()))
        metrics=trainer.step(make_batch(),1,diagnostics=True)
        self.assertGreater(metrics["cluster_ce_grad_norm"],0.)
        self.assertGreater(metrics["cluster_geometry_grad_norm"],0.)
        with tempfile.TemporaryDirectory()as folder:
            path=Path(folder)/"checkpoint.pt"
            save_checkpoint(path,trainer,epoch=1,best_epoch=1,best_score=0.,stale_epochs=0)
            trainer.step(make_batch(),2)
            expected={key:value.clone()for key,value in model.state_dict().items()}
            restored=cluster_model();second=JointTrainer(restored,restored.config,torch.arange(1,6),{"fixture":True},4)
            load_checkpoint(read_checkpoint(path),second,resume=True)
            second.step(make_batch(),2)
            for key,value in restored.state_dict().items():torch.testing.assert_close(value,expected[key],rtol=0,atol=0)

    def test_learned_aggregation_starts_as_sum_and_is_not_bypassed(self):
        model=cluster_model("linear");summed=cluster_model("sum")
        for key,value in summed.state_dict().items():
            torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
        raw=model.raw_items(torch.arange(1,6));components=model.ae.components(raw)
        torch.testing.assert_close(model.ae.latent(raw),components.sum(1),atol=2e-7,rtol=2e-6)
        model.ae(raw)[0].square().mean().backward()
        self.assertGreater(float(model.ae.cluster_aggregation.weight.grad.norm()),0.)
        model.zero_grad(set_to_none=True)
        loss,_=model.joint_objectives(make_batch());loss.backward()
        self.assertGreater(float(model.ae.cluster_aggregation.weight.grad.norm()),0.)
        with torch.no_grad():model.ae.cluster_aggregation.weight.zero_()
        self.assertEqual(float(model.ae.latent(raw).detach().abs().sum()),0.)

    def test_merged_history_uses_same_catalog_and_unexpanded_hstu(self):
        multi,merged=cluster_model(),cluster_model(history="merged")
        ids=torch.arange(1,6)
        torch.testing.assert_close(multi.encode_ids(ids),merged.encode_ids(ids),rtol=0,atol=0)
        batch=make_batch();expected=merged.encoder(merged.input_proj(merged.encode_ids(batch["input_ids"])),batch["attention_mask"])
        actual,_=merged.encode_history(batch)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        merged.eval();merged.refresh_catalog();del batch["labels"]
        self.assertTrue(torch.isfinite(direct_log_probabilities(merged,batch)).all())

    def test_invalid_compositions_are_rejected(self):
        values=cluster_model().config.to_dict()
        for section,patch in [("ae",{"cluster_count":-1}),("ae",{"cluster_groups":8}),
                              ("ae",{"frozen":True}),("recommender",{"item_vlad_clusters":4}),
                              ("recommender",{"encoder_type":"transformer"})]:
            data=copy.deepcopy(values);data[section].update(patch)
            with self.assertRaises(ValueError):JointConfig.from_dict(data)


if __name__=="__main__":unittest.main()
