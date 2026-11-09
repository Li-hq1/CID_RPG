import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from joint_training.checkpoint import load_checkpoint, migrate_checkpoint, read_checkpoint, save_checkpoint
from joint_training.config import JointConfig
from joint_training.model import JointModel
from joint_training.training import JointTrainer
from models.item_vlad import ItemClusterHistory, ItemResidualTokens
from tests.test_direct_ce import fixture
from tests import test_item_vlad
from tests.test_joint_model import make_batch


def model_for(variant, mapping="none", k=1):
    config = fixture().config.to_dict()
    config["recommender"].update(item_vlad_clusters=k, item_vlad_variant=variant,
        item_vlad_mapping=mapping, item_vlad_groups=2 if variant == "nextvlad" else 1,
        item_vlad_expansion=2 if variant == "nextvlad" else 1, item_vlad_scale="sqrt_dim")
    torch.manual_seed(29)
    return JointModel(JointConfig.from_dict(config), torch.randn(5, 6))


class ItemSimplificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_single_descriptor_assignment_cancellation_and_zero_routing_gradient(self):
        torch.manual_seed(7)
        x, centers = torch.randn(2, 3, 8, dtype=torch.float64), torch.randn(4, 8, dtype=torch.float64)
        scores = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
        residual = x[..., None, :] - centers
        weighted = F.normalize(scores.softmax(-1)[..., None] * residual, dim=-1)
        torch.testing.assert_close(weighted, F.normalize(residual, dim=-1), atol=1e-15, rtol=1e-14)
        (weighted * torch.randn_like(weighted)).sum().backward()
        self.assertLess(float(scores.grad.abs().max()), 2e-15)

    def test_exact_residual_formula_padding_and_parameters(self):
        x, mask = torch.randn(2, 4, 128), make_batch()["attention_mask"].bool()
        for k in [1, 4]:
            adapter = ItemClusterHistory(128, k, 1, 1, scale="sqrt_dim", variant="residual")
            expected = F.normalize(x[..., None, :] - adapter.tokenizer.centers, dim=-1)*128**.5
            expected = expected.masked_fill(~mask[..., None, None], 0.)
            torch.testing.assert_close(adapter.prepare_tokens(x, mask), expected, rtol=0, atol=0)
            self.assertEqual(sum(p.numel() for p in adapter.parameters()), k*128)
            self.assertEqual(list(dict(adapter.named_parameters())), ["tokenizer.centers"])
            mapped = ItemClusterHistory(128, k, 1, 1, variant="residual", mapping="linear")
            self.assertEqual(sum(p.numel() for p in mapped.parameters()), k*128 + 128**2)
        control = ItemResidualTokens(128, 1, centered=False)
        self.assertEqual(sum(p.numel() for p in control.parameters()), 0)
        torch.testing.assert_close(control(x, mask).squeeze(2), F.normalize(x, dim=-1).masked_fill(~mask[..., None], 0.))

    def test_mapping_identity_common_state_and_rng(self):
        for variant in ["residual", "nextvlad"]:
            a = model_for(variant); rng = torch.get_rng_state()
            b = model_for(variant, "linear")
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for name, value in a.state_dict().items():
                torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)
            a.eval(); b.eval()
            torch.testing.assert_close(a.encode_history(make_batch())[0], b.encode_history(make_batch())[0], rtol=0, atol=0)
            for name, value in fixture().state_dict().items():
                torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)

    def test_mapping_is_used_before_hstu_with_fixed_scale(self):
        adapter = ItemClusterHistory(8, 4, 1, 1, variant="residual", mapping="linear", scale="sqrt_dim")
        with torch.no_grad(): adapter.token_mapping.weight.copy_(torch.randn(8, 8))
        x, mask = torch.randn(2, 4, 8), make_batch()["attention_mask"].bool()
        expected = F.normalize(F.linear(x[..., None, :] - adapter.tokenizer.centers, adapter.token_mapping.weight), dim=-1)*8**.5
        expected = expected.masked_fill(~mask[..., None, None], 0.)
        actual = adapter.prepare_tokens(x, mask)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual[mask].norm(dim=-1), torch.full_like(actual[mask][..., 0], 8**.5))
        captured = []
        encoder = test_item_vlad.ItemVLADTest().encoder()
        hook = encoder.register_forward_pre_hook(lambda _, args: captured.append(args[0].clone()))
        adapter(x, mask, encoder); hook.remove()
        torch.testing.assert_close(captured[0], actual.reshape(2, 16, 8), rtol=0, atol=0)

    def test_mapping_respects_cluster_permutation_rope_and_sum(self):
        adapter = ItemClusterHistory(8, 4, 1, 1, variant="residual", mapping="linear", scale="sqrt_dim")
        encoder = test_item_vlad.ItemVLADTest().encoder()
        x, mask = torch.randn(2, 4, 8), make_batch()["attention_mask"].bool()
        a, states = adapter(x, mask, encoder, return_cluster_states=True)
        permutation = torch.tensor([2, 0, 3, 1])
        with torch.no_grad(): adapter.tokenizer.centers.copy_(adapter.tokenizer.centers[permutation].clone())
        b, permuted = adapter(x, mask, encoder, return_cluster_states=True)
        torch.testing.assert_close(permuted, states[:, :, permutation], rtol=2e-5, atol=3e-6)
        torch.testing.assert_close(a, b, rtol=2e-5, atol=5e-6)
        torch.testing.assert_close(a, states.sum(2), rtol=0, atol=0)

    def test_future_padding_and_item_locality(self):
        for variant, mapping, k in [("normalized_input", "none", 1), ("residual", "none", 4),
                                    ("residual", "linear", 4), ("nextvlad", "linear", 1)]:
            model = model_for(variant, mapping, k); model.eval(); batch = make_batch()
            expected = model.encode_history(batch)[0]
            changed = copy.deepcopy(batch); changed["input_ids"][0, 2:] = 5
            torch.testing.assert_close(model.encode_history(changed)[0][0, :2], expected[0, :2], rtol=0, atol=0)
            changed = copy.deepcopy(batch); changed["input_ids"][~batch["attention_mask"].bool()] = 5
            torch.testing.assert_close(model.encode_history(changed)[0], expected, rtol=0, atol=0)
            self.assertEqual(float(expected[~batch["attention_mask"].bool()].detach().abs().sum()), 0.)

    def test_gradients_optimizer_and_exact_next_update_resume(self):
        for variant, mapping in [("normalized_input", "none"), ("residual", "none"),
                                 ("residual", "linear"), ("nextvlad", "linear")]:
            model = model_for(variant, mapping)
            trainer = JointTrainer(model, model.config, torch.arange(1, 6), {"fixture": True}, 4)
            metrics = trainer.step(make_batch(), 1, diagnostics=True)
            if variant != "normalized_input": self.assertGreater(metrics["item_vlad_ce_grad_norm"], 0.)
            if mapping == "linear":
                self.assertGreater(metrics["item_mapping_ce_grad_norm"], 0.)
                ids = {id(p) for g in trainer.rec_optimizer.param_groups for p in g["params"]}
                self.assertIn(id(model.item_cluster_history.token_mapping.weight), ids)
                self.assertFalse(torch.equal(model.item_cluster_history.token_mapping.weight.detach(), torch.eye(8)))
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/"checkpoint.pt"
                save_checkpoint(path, trainer, epoch=1, best_epoch=1, best_score=0., stale_epochs=0)
                trainer.step(make_batch(), 2)
                expected = {n: v.clone() for n, v in model.state_dict().items()}
                restored = model_for(variant, mapping)
                second = JointTrainer(restored, restored.config, torch.arange(1, 6), {"fixture": True}, 4)
                load_checkpoint(read_checkpoint(path), second, resume=True); second.step(make_batch(), 2)
                for n, v in restored.state_dict().items():
                    torch.testing.assert_close(v, expected[n], rtol=0, atol=0)

    def test_legacy_migration_and_invalid_combinations(self):
        model = model_for("nextvlad"); values = model.config.to_dict()
        values["recommender"].pop("item_vlad_variant"); values["recommender"].pop("item_vlad_mapping")
        migrated = migrate_checkpoint({"config": values})
        restored = JointModel(JointConfig.from_dict(migrated["config"]), model.raw_features)
        restored.load_state_dict(model.state_dict(), strict=True)
        for patch in [{"item_vlad_variant": "bad"}, {"item_vlad_mapping": "bad"},
                      {"item_vlad_variant": "residual"},
                      {"item_vlad_clusters": 0, "item_vlad_mapping": "linear"},
                      {"item_vlad_variant": "normalized_input", "item_vlad_groups": 1,
                       "item_vlad_expansion": 1, "item_vlad_clusters": 4}]:
            values = model.config.to_dict(); values["recommender"].update(patch)
            with self.assertRaises(ValueError): JointConfig.from_dict(values)


if __name__ == "__main__": unittest.main()
