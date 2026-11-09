import copy
import tempfile
import unittest
from pathlib import Path

import torch

from joint_training.checkpoint import load_checkpoint, migrate_checkpoint, read_checkpoint, save_checkpoint
from joint_training.config import JointConfig
from joint_training.model import JointModel
from joint_training.training import JointTrainer
from models.item_vlad import ItemClusterAggregation, ItemClusterHistory
from tests.test_direct_ce import fixture
from tests.test_joint_model import make_batch


def pooled_model(mode="sum", k=2):
    config = fixture().config.to_dict()
    config["recommender"].update(item_vlad_clusters=k, item_vlad_pooling=mode,
                                 item_vlad_scale="sqrt_dim")
    torch.manual_seed(29)
    return JointModel(JointConfig.from_dict(config), torch.randn(5, 6))


class ItemPoolingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_mass_formula_and_gradients(self):
        states = torch.randn(2, 3, 4, 8, requires_grad=True)
        mass = torch.rand(2, 3, 4, requires_grad=True)
        pool = ItemClusterAggregation(8, "mass")
        result, weights = pool(states, mass, return_weights=True)
        expected = mass / mass.sum(-1, keepdim=True)
        torch.testing.assert_close(weights, expected, rtol=0, atol=0)
        torch.testing.assert_close(result, (states * (4 * expected)[..., None]).sum(-2), rtol=0, atol=0)
        result.square().mean().backward()
        self.assertGreater(float(mass.grad.norm()), 0.)
        self.assertEqual(sum(p.numel() for p in pool.parameters()), 0)
        self.assertFalse(torch.allclose(result, states.sum(-2)))

    def test_context_starts_as_sum_and_can_learn_unequal_weights(self):
        states = torch.randn(2, 3, 4, 8)
        pool = ItemClusterAggregation(8, "context")
        result, weights = pool(states, return_weights=True)
        torch.testing.assert_close(result, states.sum(-2), rtol=0, atol=0)
        torch.testing.assert_close(weights, torch.full_like(weights, .25), rtol=0, atol=0)
        result.square().mean().backward()
        self.assertGreater(float(pool.query.grad.norm()), 0.)
        with torch.no_grad():
            pool.query.add_(pool.query.grad)
        changed, learned = pool(states, return_weights=True)
        self.assertFalse(torch.allclose(changed, result))
        self.assertGreater(float(learned.detach().std()), 0.)

    def test_shared_state_and_rng_match_sum(self):
        summed = pooled_model(); rng = torch.get_rng_state()
        context = pooled_model("context")
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in summed.state_dict().items():
            torch.testing.assert_close(value, context.state_dict()[key], rtol=0, atol=0)
        a, b = summed.encode_history(make_batch())[0], context.encode_history(make_batch())[0]
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_permutation_invariance_and_padding(self):
        states, mass = torch.randn(2, 3, 4, 8), torch.rand(2, 3, 4)
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
        order = torch.tensor([2, 0, 3, 1])
        for mode in ["sum", "mass", "context"]:
            pool = ItemClusterAggregation(8, mode)
            if pool.query is not None:
                with torch.no_grad(): pool.query.normal_()
            a, weights = pool(states, mass, mask, return_weights=True)
            b, swapped = pool(states[:, :, order], mass[:, :, order], mask, return_weights=True)
            torch.testing.assert_close(a, b, atol=3e-6, rtol=2e-5)
            torch.testing.assert_close(weights[:, :, order], swapped, atol=1e-7, rtol=1e-6)
            self.assertEqual(float(a[~mask].detach().abs().sum()), 0.)
            self.assertEqual(float(weights[~mask].detach().abs().sum()), 0.)

    def test_k1_equivalence(self):
        states, mass = torch.randn(2, 3, 1, 8), torch.rand(2, 3, 1)
        for mode in ["sum", "mass", "context"]:
            pool = ItemClusterAggregation(8, mode)
            torch.testing.assert_close(pool(states, mass), states.squeeze(-2), rtol=0, atol=0)

    def test_future_and_padding_invariance(self):
        for mode in ["mass", "context"]:
            model = pooled_model(mode); model.eval(); batch = make_batch()
            if model.item_cluster_history.pool.query is not None:
                with torch.no_grad(): model.item_cluster_history.pool.query.normal_()
            expected = model.encode_history(batch)[0]
            changed = copy.deepcopy(batch); changed["input_ids"][0, 2:] = 5
            actual = model.encode_history(changed)[0]
            torch.testing.assert_close(actual[0, :2], expected[0, :2], rtol=0, atol=0)
            changed = copy.deepcopy(batch); changed["input_ids"][~batch["attention_mask"].bool()] = 5
            torch.testing.assert_close(model.encode_history(changed)[0], expected, rtol=0, atol=0)

    def test_optimizer_gradients_and_exact_resume(self):
        for mode in ["mass", "context"]:
            model = pooled_model(mode)
            trainer = JointTrainer(model, model.config, torch.arange(1, 6), {"fixture": True}, 4)
            metrics = trainer.step(make_batch(), 1, diagnostics=True)
            self.assertGreater(metrics["item_vlad_ce_grad_norm"], 0.)
            if mode == "context":
                self.assertGreater(metrics["item_pool_ce_grad_norm"], 0.)
                rec = {id(p) for group in trainer.rec_optimizer.param_groups for p in group["params"]}
                self.assertIn(id(model.item_cluster_history.pool.query), rec)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "checkpoint.pt"
                save_checkpoint(path, trainer, epoch=1, best_epoch=1, best_score=0., stale_epochs=0)
                trainer.step(make_batch(), 2)
                expected = {key: value.clone() for key, value in model.state_dict().items()}
                restored = pooled_model(mode)
                second = JointTrainer(restored, restored.config, torch.arange(1, 6), {"fixture": True}, 4)
                load_checkpoint(read_checkpoint(path), second, resume=True)
                second.step(make_batch(), 2)
                for key, value in restored.state_dict().items():
                    torch.testing.assert_close(value, expected[key], rtol=0, atol=0)

    def test_legacy_sum_migration_and_reject_unused_pooling(self):
        model = pooled_model(); values = model.config.to_dict()
        values["recommender"].pop("item_vlad_pooling")
        checkpoint = migrate_checkpoint({"config": values})
        self.assertEqual(checkpoint["config"]["recommender"]["item_vlad_pooling"], "sum")
        restored = JointModel(JointConfig.from_dict(checkpoint["config"]), model.raw_features)
        restored.load_state_dict(model.state_dict(), strict=True)
        for patch in [{"item_vlad_pooling": "unknown"}, {"item_vlad_pooling": "context", "item_vlad_clusters": 0}]:
            values = model.config.to_dict(); values["recommender"].update(patch)
            with self.assertRaises(ValueError): JointConfig.from_dict(values)

    def test_fixed_scale_multiplier_preserves_weights_and_matches_formula(self):
        reference = ItemClusterHistory(8, 2, scale="sqrt_dim")
        x, mask = torch.randn(2, 4, 8), make_batch()["attention_mask"].bool()
        for multiplier in [.5, 1., 2.]:
            changed = ItemClusterHistory(8, 2, scale="sqrt_dim", scale_multiplier=multiplier)
            changed.load_state_dict(reference.state_dict(), strict=True)
            for key, value in reference.state_dict().items():
                torch.testing.assert_close(value, changed.state_dict()[key], rtol=0, atol=0)
            tokens = changed.tokenizer(x, mask) * changed.token_scale
            torch.testing.assert_close(tokens[mask].norm(dim=-1),
                                       torch.full((int(mask.sum()), 2), multiplier * 8 ** .5))
        for multiplier in [0., -1., float("nan")]:
            with self.assertRaises(ValueError):
                ItemClusterHistory(8, 2, scale_multiplier=multiplier)


if __name__ == "__main__":
    unittest.main()
