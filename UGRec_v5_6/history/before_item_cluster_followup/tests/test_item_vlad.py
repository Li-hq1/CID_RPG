import copy
import unittest

import torch
from torch.nn import functional as F

from joint_training.config import JointConfig
from joint_training.direct_ce import direct_log_probabilities
from joint_training.model import JointModel
from models.hstu import HSTUSequenceEncoder, reverse_position_ids
from models.item_vlad import ItemNeXtVLAD, ItemClusterHistory
from tests.test_direct_ce import fixture
from tests.test_joint_model import make_batch


class ItemVLADTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2718)
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def encoder(self, scaling="input_length"):
        return HSTUSequenceEncoder(8, 2, 4, 4, 2, 4, output_dropout_ratio=0., attention_scaling=scaling)

    def clustered_model(self, clusters=4):
        baseline = fixture()
        values = baseline.config.to_dict()
        values["recommender"]["item_vlad_clusters"] = clusters
        torch.manual_seed(29)
        return JointModel(JointConfig.from_dict(values), torch.randn(5, 6))

    def test_vlad_formula_is_per_item_and_padding_is_zero(self):
        vlad = ItemNeXtVLAD(8, 4)
        x, mask = torch.randn(2, 4, 8), make_batch()["attention_mask"].bool()
        tokens, assignments, mass = vlad(x, mask, return_assignments=True)
        expanded = vlad.fc0(x)
        activation = assignments * vlad.fc_g(expanded).sigmoid()[..., None]
        expected = (activation[..., None] * (expanded.reshape(2,4,2,1,8) - vlad.centers)).sum(2)
        expected = F.normalize(expected, dim=-1).masked_fill(~mask[...,None,None], 0.)
        torch.testing.assert_close(tokens, expected)
        torch.testing.assert_close(mass, activation.sum(2))
        torch.testing.assert_close(assignments.sum(-1), torch.ones_like(assignments[...,0]))
        changed = x.clone(); changed[:, 1:] += 37
        torch.testing.assert_close(vlad(changed, mask)[:,0], tokens[:,0])
        self.assertFalse(any("proj" in name or "bn" in name for name, _ in vlad.named_parameters()))
        self.assertEqual(tokens.shape, (2,4,4,8))
        self.assertEqual(float(tokens[~mask].detach().abs().sum()), 0.)

    def test_repeated_item_rope_and_block_causality(self):
        encoder, mask = self.encoder(), torch.tensor([[1,1,1,0]], dtype=torch.bool)
        x = torch.randn(1, 16, 8)
        captured = []
        hook = encoder.layers[0].register_forward_pre_hook(lambda _, args: captured.append(args[2].clone()))
        _, weights = encoder(x, mask.repeat_interleave(4, 1), True, item_group_size=4)
        hook.remove()
        torch.testing.assert_close(captured[0], torch.tensor([[3]*4+[2]*4+[1]*4+[0]*4]))
        torch.testing.assert_close(captured[0], reverse_position_ids(mask).repeat_interleave(4, 1))
        for layer in weights:
            self.assertEqual(float(layer[:,:,0:4,4:].detach().abs().sum()), 0.)
            self.assertGreater(float(layer[:,:,0,1:4].detach().abs().sum()), 0.)
            self.assertEqual(float(layer[:,:,:,12:].detach().abs().sum()), 0.)
            self.assertEqual(float(layer[:,:,12:,:].detach().abs().sum()), 0.)

    def test_cluster_permutation_equivariance_and_exact_sum(self):
        encoder, adapter = self.encoder(), ItemClusterHistory(8,4)
        x, mask = torch.randn(2,4,8), make_batch()["attention_mask"].bool()
        result, states = adapter(x, mask, encoder, return_cluster_states=True)
        torch.testing.assert_close(result, states.sum(2), rtol=0, atol=0)
        permutation = torch.tensor([2,0,3,1])
        tokens = adapter.tokenizer(x,mask)[:,:,permutation]
        changed = encoder(tokens.reshape(2,16,8), mask.repeat_interleave(4,1), item_group_size=4).reshape(2,4,4,8)
        torch.testing.assert_close(changed, states[:,:,permutation], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(changed.sum(2), result, atol=5e-6, rtol=2e-5)

    def test_future_and_padding_invariance_through_all_layers(self):
        model, batch = self.clustered_model(), make_batch()
        model.eval()
        expected,_ = model.encode_history(batch)
        changed = copy.deepcopy(batch); changed["input_ids"][0,2:] = 5
        actual,_ = model.encode_history(changed)
        torch.testing.assert_close(actual[0,:2],expected[0,:2], rtol=0, atol=0)
        changed = copy.deepcopy(batch)
        changed["input_ids"][~batch["attention_mask"].bool()] = 5
        actual,_ = model.encode_history(changed)
        torch.testing.assert_close(actual,expected, rtol=0, atol=0)
        self.assertEqual(float(expected[~batch["attention_mask"].bool()].detach().abs().sum()), 0.)

    def test_k1_matches_original_hstu_and_attention_denominator_scales_with_k(self):
        for scaling in ["input_length", "max_seq_len"]:
            encoder = self.encoder(scaling)
            x, mask = torch.randn(2,4,8), make_batch()["attention_mask"].bool()
            expected = encoder(x,mask)
            torch.testing.assert_close(encoder(x,mask,item_group_size=1), expected, rtol=0, atol=0)
            duplicated = x.repeat_interleave(4,1)
            expanded = encoder(duplicated,mask.repeat_interleave(4,1),item_group_size=4).reshape(2,4,4,8)
            torch.testing.assert_close(expanded, expected[:,:,None].expand_as(expanded), atol=2e-6,rtol=2e-5)

    def test_initial_shared_weights_gradients_and_target_free_reload(self):
        baseline, model = fixture(), self.clustered_model()
        for name,value in baseline.state_dict().items():
            torch.testing.assert_close(value,model.state_dict()[name],rtol=0,atol=0)
        loss,_ = model.joint_objectives(make_batch())
        loss.backward()
        for name in ["ae.encoder.0.weight","item_cluster_history.tokenizer.fc0.weight",
                     "item_cluster_history.tokenizer.fc_gk.weight","item_cluster_history.tokenizer.fc_g.weight",
                     "item_cluster_history.tokenizer.centers","encoder.layers.0.uvqk_weight",
                     "ranking_head.weight"]:
            grad = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(grad,name)
            self.assertGreater(float(grad.norm()),0.,name)
        model.eval();model.refresh_catalog()
        batch=make_batch();del batch["labels"]
        expected=direct_log_probabilities(model,batch)
        restored=self.clustered_model();restored.load_state_dict(model.state_dict())
        restored.eval();restored.refresh_catalog()
        torch.testing.assert_close(direct_log_probabilities(restored,batch),expected,rtol=0,atol=0)
        self.assertFalse(hasattr(model,"diffloss"))

    def test_reject_incompatible_config_and_partial_item_mask(self):
        values=fixture().config.to_dict()
        for patch in [{"item_vlad_clusters":-1},{"item_vlad_clusters":4,"item_vlad_groups":8},
                      {"item_vlad_clusters":4,"encoder_type":"transformer"}]:
            data=copy.deepcopy(values);data["recommender"].update(patch)
            with self.assertRaises(ValueError):JointConfig.from_dict(data)
        data=copy.deepcopy(values);data["recommender"]["item_vlad_clusters"]=4
        data["flow_ranking"].update(path="mixture",noise_scale=1.,draws=8)
        data["evaluation"].update(eval_sampling_mode="mixture",eval_num_samples=4)
        with self.assertRaises(ValueError):JointConfig.from_dict(data)
        with self.assertRaises(ValueError):
            self.encoder()(torch.randn(1,4,8),torch.tensor([[1,0,1,1]]),item_group_size=2)


if __name__ == "__main__":unittest.main()
