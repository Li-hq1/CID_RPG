import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from joint_training.checkpoint import load_checkpoint, read_checkpoint, save_checkpoint
from joint_training.config import JointConfig
from joint_training.model import JointModel
from joint_training.training import JointTrainer
from models.item_vlad import ItemProjectionTokens, token_orthogonality
from tests.test_direct_ce import fixture
from tests.test_joint_model import make_batch


def model_for(variant="residual", weight=0.):
    c = fixture().config.to_dict()
    c["recommender"].update(item_vlad_clusters=4, item_vlad_variant=variant, item_vlad_mapping="linear",
        item_vlad_groups=1, item_vlad_expansion=1, item_vlad_scale="sqrt_dim",
        item_vlad_projection_rank=2, item_vlad_orthogonality_weight=weight)
    torch.manual_seed(29)
    return JointModel(JointConfig.from_dict(c), torch.randn(5, 6))


class ItemDiversityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_recursive_projection_formula_raw_reconstruction_and_energy(self):
        x, mask = torch.randn(2, 4, 8), make_batch()["attention_mask"].bool()
        m = ItemProjectionTokens(8, 4, 2, sequential=True)
        with torch.no_grad(): m.factors[1].add_(.2*m.factors[0])
        actual, remaining = m.components(x, mask, return_residuals=True)
        r = x
        expected = []
        for q in m.bases():
            part = (r@q)@q.T
            expected.append(part); r = r-part
        expected.append(r)
        expected = torch.stack(expected, -2).masked_fill(~mask[..., None, None], 0.)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual.sum(-2)[mask], x[mask], rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(actual[mask].square().sum((-1, -2)), x[mask].square().sum(-1), rtol=2e-6, atol=3e-6)
        for a, b in zip(remaining, remaining[1:]):
            self.assertTrue(bool((b[mask].square().sum(-1) <= a[mask].square().sum(-1)+1e-5).all()))

    def test_parallel_control_same_capacity_init_but_different_after_subspace_overlap(self):
        rng = torch.get_rng_state()
        a = ItemProjectionTokens(8, 4, 2, sequential=True)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        b = ItemProjectionTokens(8, 4, 2, sequential=False)
        x, mask = torch.randn(2, 3, 8), torch.ones(2, 3, dtype=torch.bool)
        torch.testing.assert_close(a(x, mask), b(x, mask), rtol=1e-5, atol=1e-6)
        with torch.no_grad(): a.factors[1].add_(.5*a.factors[0])
        b.load_state_dict(a.state_dict(), strict=True)
        self.assertFalse(torch.allclose(a(x, mask), b(x, mask)))
        torch.testing.assert_close(b.components(x, mask).sum(-2), x, rtol=1e-6, atol=1e-6)
        large = ItemProjectionTokens(128, 4, 32)
        self.assertEqual(sum(p.numel() for p in large.parameters()), 12288)

    def test_token_orthogonality_formula_sign_scale_and_padding(self):
        x = torch.tensor([[[[1., 0.], [1., 0.]], [[1., 0.], [0., 1.]]]], requires_grad=True)
        mask = torch.tensor([[1, 0]], dtype=torch.bool)
        torch.testing.assert_close(token_orthogonality(x, mask), torch.tensor(1.))
        torch.testing.assert_close(token_orthogonality(x, ~mask), torch.tensor(0.))
        flipped = x.detach().clone(); flipped[:, :, 1] *= -1
        torch.testing.assert_close(token_orthogonality(flipped, mask), torch.tensor(1.))
        torch.testing.assert_close(token_orthogonality(17*x, mask), token_orthogonality(x, mask))
        torch.testing.assert_close(token_orthogonality(x[:, :, :1], mask), torch.tensor(0.))
        zero = token_orthogonality(x, torch.zeros_like(mask)); zero.backward()
        self.assertEqual(float(zero.detach()), 0.)
        self.assertEqual(float(x.grad.norm()), 0.)

    def test_shared_backbone_penalty_initial_state_and_cpu_initial_functions(self):
        a, b = model_for(), model_for(weight=.3)
        for k, v in a.state_dict().items(): torch.testing.assert_close(v, b.state_dict()[k], rtol=0, atol=0)
        a.eval(); b.eval()
        torch.testing.assert_close(a.encode_history(make_batch())[0], b.encode_history(make_batch())[0], rtol=0, atol=0)
        s, p = model_for("sequential_projection"), model_for("parallel_projection")
        for k, v in s.state_dict().items(): torch.testing.assert_close(v, p.state_dict()[k], rtol=0, atol=0)
        for k, v in fixture().state_dict().items(): torch.testing.assert_close(v, s.state_dict()[k], rtol=0, atol=0)

    def test_penalty_uses_actual_mapped_tokens_and_routes_gradients(self):
        m = model_for(weight=.3); batch = make_batch()
        loss, terms = m.joint_objectives(batch)
        _, _, tokens = m.encode_history(batch, return_item_tokens=True)
        expected = token_orthogonality(tokens, batch["attention_mask"].bool())
        torch.testing.assert_close(terms["item_orthogonality"], expected)
        torch.testing.assert_close(loss, m.config.flow_ranking.weight*terms["ce"]+.3*expected)
        centers = m.item_cluster_history.tokenizer.centers
        ce_grad = torch.autograd.grad(terms["ce"], centers, retain_graph=True)[0]
        orth_grad = torch.autograd.grad(terms["item_orthogonality"], centers, retain_graph=True)[0]
        self.assertGreater(float(ce_grad.norm()), 0.)
        self.assertGreater(float(orth_grad.norm()), 0.)
        params = [m.ae.encoder[0].weight, m.input_proj.linear.weight, m.encoder.layers[0].uvqk_weight]
        grads = torch.autograd.grad(terms["item_orthogonality"], params, allow_unused=True)
        self.assertGreater(float(grads[0].norm()), 0.)
        self.assertGreater(float(grads[1].norm()), 0.)
        self.assertIsNone(grads[2])

    def test_projection_scale_mask_and_future_isolation(self):
        for variant in ["sequential_projection", "parallel_projection"]:
            m = model_for(variant); m.eval(); batch = make_batch()
            result, _, tokens = m.encode_history(batch, return_item_tokens=True)
            mask = batch["attention_mask"].bool()
            torch.testing.assert_close(tokens[mask].norm(dim=-1), torch.full_like(tokens[mask][..., 0], 8**.5))
            self.assertEqual(float(tokens[~mask].detach().abs().sum()), 0.)
            changed = copy.deepcopy(batch); changed["input_ids"][0, 2:] = 5
            torch.testing.assert_close(m.encode_history(changed)[0][0, :2], result[0, :2], rtol=0, atol=0)

    def test_projection_and_penalty_optimizer_exact_resume(self):
        for variant, weight in [("sequential_projection", 0.), ("parallel_projection", 0.), ("residual", .3)]:
            m = model_for(variant, weight)
            t = JointTrainer(m, m.config, torch.arange(1, 6), {"fixture": True}, 4)
            metrics = t.step(make_batch(), 1, diagnostics=True)
            self.assertGreater(metrics["item_vlad_ce_grad_norm"], 0.)
            if weight:
                self.assertGreater(metrics["item_orthogonality_center_grad_norm"], 0.)
            else:
                for factor in m.item_cluster_history.tokenizer.factors:
                    self.assertGreater(float(factor.grad.norm()), 0.)
                    self.assertTrue(any(id(factor) == id(p) for g in t.rec_optimizer.param_groups for p in g["params"]))
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/"checkpoint.pt"
                save_checkpoint(path, t, epoch=1, best_epoch=1, best_score=0., stale_epochs=0)
                t.step(make_batch(), 2)
                expected = {k: v.clone() for k, v in m.state_dict().items()}
                restored = model_for(variant, weight)
                second = JointTrainer(restored, restored.config, torch.arange(1, 6), {"fixture": True}, 4)
                load_checkpoint(read_checkpoint(path), second, resume=True); second.step(make_batch(), 2)
                for k, v in restored.state_dict().items(): torch.testing.assert_close(v, expected[k], rtol=0, atol=0)

    def test_reject_degenerate_or_unused_options(self):
        c = model_for().config.to_dict()
        for patch in [{"item_vlad_orthogonality_weight": -1.}, {"item_vlad_orthogonality_weight": .1, "item_vlad_clusters": 1},
                      {"item_vlad_variant": "sequential_projection", "item_vlad_projection_rank": 3},
                      {"item_vlad_projection_rank": 0}]:
            values = copy.deepcopy(c); values["recommender"].update(patch)
            with self.assertRaises(ValueError): JointConfig.from_dict(values)


if __name__ == "__main__": unittest.main()
