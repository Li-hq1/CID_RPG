import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from joint_training.checkpoint import evaluation_randomness, load_checkpoint, read_checkpoint, save_checkpoint
from joint_training.config import JointConfig
from joint_training.multi_target_model import MultiTargetModel, sampled_logp, straight_bridge
from joint_training.training import JointTrainer
from multi_target import FutureDataset, ranking_metrics, split_sequences
from protocol import training_windows


ROOT = Path(__file__).resolve().parents[1]


def fixture(arm="future_fm"):
    values = json.loads((ROOT / "configs/final.json").read_text())
    values["experiment"]["arm"] = arm
    values["data"]["max_item_seq_len"] = 50
    values["training"].update(device="cpu", epochs=2)
    values["ae"].update(architecture="linear", hidden_dims=[8], latent_dim=4, batch_size=4)
    values["recommender"].update(n_embd=8, n_layer=1, n_head=2, n_inner=16,
        hstu_hidden_dim=4, hstu_attention_dim=4, diffloss_w=16, diffloss_d=1,
        hstu_output_dropout=0., hstu_attention_dropout=0.)
    values["evaluation"].update(eval_num_samples=2, eval_batch_size=2, mixture_query_chunk_size=1)
    config = JointConfig.from_dict(values)
    torch.manual_seed(71)
    model = MultiTargetModel(config, torch.randn(12, 6))
    samples = FutureDataset([list(range(1, 11)), [3, 4, 5, 6, 7, 8]], 3)
    batch = {key: torch.stack([samples[0][key], samples[1][key]]) for key in samples[0]}
    return config, model, batch


class ProtocolTest(unittest.TestCase):
    def mapping(self):
        return {"item2id": {str(i): i for i in range(1, 31)},
                "id2item": ["PAD"] + [str(i) for i in range(1, 31)]}

    def test_complete_test_suffix_never_looked_up(self):
        seq = {"a": [str(i) for i in range(1, 8)] + ["UNMAPPED_TEST"] * 3}
        train, histories, targets, users, audit = split_sequences(seq, self.mapping(), horizon=3)
        self.assertEqual(train, [[1, 2, 3, 4]])
        self.assertEqual(histories, train)
        self.assertEqual(targets.tolist(), [[5, 6, 7]])
        self.assertEqual(users, ["a"])
        self.assertFalse(audit["test_targets_accessed"])

    def test_loo_keeps_original_boundary_and_short_users_are_reported(self):
        seq = {"short": ["1", "2", "3"], "long": ["1", "2", "3", "4", "5"]}
        train, _, targets, users, audit = split_sequences(seq, self.mapping(), protocol="leave_one_out")
        self.assertEqual(users, ["long"])
        self.assertEqual(train, [[1, 2, 3]])
        self.assertEqual(targets.tolist(), [[4]])
        self.assertEqual(audit["original_users"], 2)

    def test_next_training_windows_match_base_and_future_never_crosses_train(self):
        sequences = [list(range(1, 61)), [2, 4, 6]]
        dataset = FutureDataset(sequences, 3)
        inputs, labels = training_windows(sequences)
        self.assertEqual(len(dataset), len(inputs))
        for i in range(len(dataset)):
            sample = dataset[i]
            self.assertEqual(sample["input_ids"].tolist(), inputs[i])
            self.assertEqual(sample["labels"].tolist(), labels[i])
            for p in (sample["future_labels"] > 0).all(-1).nonzero().flatten():
                future = sample["future_labels"][p]
                self.assertEqual(future[0], sample["labels"][p])
                self.assertLessEqual(int(future.max()), 60)
        self.assertEqual(dataset[-1]["future_labels"].sum(), 0)

    def test_future_distribution_respects_order_and_full_window(self):
        dataset = FutureDataset([[1, 2, 3, 4, 5]], 3)
        sample = dataset[0]
        self.assertEqual(sample["future_labels"][:4].tolist(), [[2, 3, 4], [3, 4, 5], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(dataset.next_queries, 4)
        self.assertEqual(dataset.auxiliary_queries, 2)

    def test_multi_positive_metrics_deduplicate_targets(self):
        prediction = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        metrics = ranking_metrics(prediction, np.array([[2, 2, 12]]))
        self.assertEqual(metrics["recall@10"], .5)
        self.assertAlmostEqual(metrics["ndcg@10"], (1 / np.log2(3)) / (1 + 1 / np.log2(3)))
        self.assertEqual(metrics["step1/recall@10"], 1.)

    def test_metrics_refuse_duplicate_lists_and_padding(self):
        with self.assertRaises(ValueError):
            ranking_metrics([[1] * 10], [[1, 2]])
        with self.assertRaises(ValueError):
            ranking_metrics([list(range(10))], [[1, 2]])


class ObjectiveTest(unittest.TestCase):
    def test_cfm_future_endpoints_are_individual_observed_items(self):
        _, model, batch = fixture("future_fm")
        with patch.object(model, "encode_ids", wraps=model.encode_ids) as encode:
            model.joint_objectives(batch)
        call = encode.call_args_list[-1]
        endpoints = call.args[0]
        self.assertFalse(call.kwargs["grad"])
        complete = (batch["future_labels"] > 0).all(-1)
        allowed = batch["future_labels"][complete]
        self.assertEqual(endpoints.shape, (len(allowed),))
        self.assertTrue((endpoints[:, None] == allowed).any(-1).all())

    def test_invalid_protocol_or_silent_extra_loss_is_rejected(self):
        config, _, _ = fixture()
        config.data.split = "leave_one_out"
        with self.assertRaisesRegex(ValueError, "data.split"):
            config.validate()
        config.data.split = config.experiment.protocol
        config.ranking.endpoint_weight = 1.
        with self.assertRaisesRegex(ValueError, "declared auxiliary"):
            config.validate()

    def test_bridge_is_detached_and_time_direction_is_consistent(self):
        source = torch.randn(2, 4, requires_grad=True)
        target = torch.randn(2, 4, requires_grad=True)
        for fraction, endpoint in [(0., target), (1., source)]:
            state, velocity = straight_bridge(source, target, torch.full((2, 1), fraction))
            torch.testing.assert_close(state, endpoint)
            torch.testing.assert_close(velocity, source - target)
            self.assertFalse(state.requires_grad)
            self.assertFalse(velocity.requires_grad)

    def test_euler_oracle_recovers_target_with_fractional_times(self):
        _, model, _ = fixture()
        source, target = torch.randn(2, 4), torch.randn(2, 4)
        times = []
        def oracle(state, time, condition):
            times.append(time)
            return source - target
        with patch.object(model.diffloss.net, "forward", side_effect=oracle):
            actual = model.transport(source, torch.randn(2, 8), 4)
        torch.testing.assert_close(actual, target)
        self.assertTrue(times[1].is_floating_point())
        self.assertNotEqual(float(times[1][0]), round(float(times[1][0])))

    def test_auxiliary_cannot_update_ae_or_source_head_but_can_update_history(self):
        for arm in ("future_ce", "next_fm", "future_fm"):
            _, model, batch = fixture(arm)
            with torch.no_grad():
                for parameter in model.diffloss.net.parameters():
                    parameter.add_(torch.randn_like(parameter) * .01)
            _, terms = model.joint_objectives(batch)
            terms["auxiliary"].backward()
            self.assertTrue(all(p.grad is None for p in model.ae.parameters()), arm)
            self.assertTrue(all(p.grad is None for p in model.ranking_head.parameters()), arm)
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()), arm)

    def test_next_ce_updates_ae_and_ranking_head(self):
        _, model, batch = fixture("ce")
        _, terms = model.joint_objectives(batch)
        terms["ranking"].backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.ae.parameters()))
        self.assertIsNotNone(model.ranking_head.weight.grad)
        self.assertTrue(all(p.grad is None for p in model.diffloss.parameters()))

    def test_all_arms_have_same_initialized_model_and_next_ce(self):
        losses, states = [], []
        for arm in ("ce", "next_fm", "future_ce", "future_fm"):
            _, model, batch = fixture(arm)
            states.append(copy.deepcopy(model.state_dict()))
            _, terms = model.joint_objectives(batch)
            losses.append(terms["ranking"].detach())
        for state, loss in zip(states[1:], losses[1:]):
            torch.testing.assert_close(loss, losses[0], rtol=0, atol=0)
            for key in state:
                torch.testing.assert_close(state[key], states[0][key], rtol=0, atol=0)

    def test_sampled_partition_with_complete_uniform_draw_matches_full_catalog(self):
        _, model, _ = fixture("future_ce")
        query = torch.randn(2, 4, requires_grad=True)
        target = torch.tensor([2, 4])
        positives = torch.tensor([[2, 2, 3], [4, 5, 6]])
        negatives = torch.arange(1, model.n_items)
        actual = sampled_logp(query, target, positives, negatives, model, keys_grad=False)
        keys = model.encode_ids(negatives, grad=False)
        expected = (F.normalize(query, dim=-1) @ F.normalize(keys, dim=-1).T /
                    model.config.ranking.temperature).log_softmax(-1).gather(1, target[:, None] - 1)[:, 0]
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(query.grad).all())

    def test_probability_mixture_is_average_softmax_not_average_embedding(self):
        _, model, batch = fixture()
        model.eval()
        model.refresh_catalog()
        samples = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]]).repeat(2, 1)
        with patch.object(model, "flow_prior", return_value=samples), patch.object(model, "transport", return_value=samples):
            actual = model.log_probabilities(batch).exp()
        expected = (F.normalize(samples, dim=-1) @ model.unit_catalog.T /
                    model.config.ranking.temperature).softmax(-1).reshape(2, 2, -1).mean(1)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.sum(1), torch.ones(2))

    def test_evaluation_restores_training_rng_and_generation_ignores_labels(self):
        _, model, batch = fixture()
        model.eval()
        model.refresh_catalog()
        before = torch.get_rng_state().clone()
        with evaluation_randomness("cpu", 42):
            first = model.log_probabilities(batch)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        batch["labels"][:] = 12
        batch["future_labels"][:] = 12
        with evaluation_randomness("cpu", 42):
            second = model.log_probabilities(batch)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_checkpoint_binds_experiment_and_restores_predictions(self):
        config, model, batch = fixture()
        trainer = JointTrainer(model, config, torch.arange(1, 11), {"fixture": True}, 2)
        trainer.step(batch, 1)
        model.eval()
        model.refresh_catalog()
        with evaluation_randomness("cpu", 17):
            original = model.log_probabilities(batch)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            save_checkpoint(path, trainer, epoch=1, best_epoch=1, best_score=0., stale_epochs=0)
            saved = read_checkpoint(path)
            with torch.no_grad():
                model.ranking_head.weight.add_(1.)
            load_checkpoint(saved, trainer)
            model.eval()
            model.refresh_catalog()
            with evaluation_randomness("cpu", 17):
                restored = model.log_probabilities(batch)
            torch.testing.assert_close(restored, original, rtol=0, atol=0)
            config.experiment.horizon = 5
            with self.assertRaisesRegex(ValueError, "experiment"):
                load_checkpoint(saved, trainer)


if __name__ == "__main__":
    unittest.main()
