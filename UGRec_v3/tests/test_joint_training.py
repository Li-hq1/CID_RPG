"""Integration tests for alternating optimization and exact epoch-boundary resume."""

import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock

import torch

from joint_training.checkpoint import load_checkpoint, read_checkpoint, save_checkpoint
from joint_training.config import JointConfig
from joint_training.model import JointModel
from joint_training.training import JointTrainer
from tests.test_joint_model import make_batch


def config_and_model(*, contrastive=False, frozen=False, encoder_type="transformer",
                     grad_checkpointing=False):
    config = JointConfig.from_dict({
        "data": {"max_item_seq_len": 4},
        "ae": {"architecture": "linear", "latent_dim": 3, "hidden_dims": [8],
               "batch_size": 5, "frozen": frozen, "warmup_ratio": 0.},
        "recommender": {"encoder_type": encoder_type, "n_embd": 8, "n_layer": 1,
                        "n_head": 2, "n_inner": 16, "diffloss_w": 16, "diffloss_d": 1,
                        "num_sampling_steps": 2, "rectified_flow_steps": 10,
                        "embd_pdrop": 0., "attn_pdrop": 0., "tiger_uncond_prob": 0.,
                        "warmup_ratio": 0., "tiger_guidance_weight": 0.,
                        "grad_checkpointing": grad_checkpointing},
        "contrastive": {"enabled": contrastive},
        "evaluation": {"eval_sampling_mode": "knn"},
        "training": {"epochs": 2, "max_steps": 4},
    })
    torch.manual_seed(7)
    model = JointModel(config, torch.randn(5, 6))
    trainer = JointTrainer(model, config, torch.arange(1, 6), {"fixture": True}, 4)
    return config, model, trainer


class JointTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_contrastive_disabled_skips_computation_and_rec_uses_updated_ae(self):
        _, model, trainer = config_and_model()
        model.select_contrastive_positions = Mock(side_effect=AssertionError("contrastive is disabled"))
        model.contrastive_loss = Mock(side_effect=AssertionError("contrastive is disabled"))
        before = model.ae.encoder[0].weight.detach().clone()
        captured = {}
        forward = model.rf_loss

        def observed(batch):
            captured["ae"] = model.ae.encoder[0].weight.detach().clone()
            return forward(batch)

        model.rf_loss = observed
        metrics = trainer.step(make_batch(), 1, diagnostics=True)
        self.assertFalse(torch.equal(before, captured["ae"]))
        self.assertTrue(torch.equal(captured["ae"], model.ae.encoder[0].weight))
        self.assertTrue(all(value.grad is None for value in model.ae.parameters()))
        self.assertEqual(trainer.ae_steps, 1)
        self.assertEqual(trainer.rec_steps, 1)

    def test_default_contrastive_only_updates_ae(self):
        for encoder_type in ("hstu", "transformer"):
            with self.subTest(encoder_type=encoder_type):
                _, model, trainer = config_and_model(contrastive=True, encoder_type=encoder_type)
                model.contrastive_loss = Mock(wraps=model.contrastive_loss)
                metrics = trainer.step(make_batch(), 1, diagnostics=True)
                model.contrastive_loss.assert_called_once()
                self.assertGreater(metrics["ae_contrastive_grad_norm"], 0.)
                self.assertTrue(all(parameter.grad is None for parameter in model.ae.parameters()))
                self.assertTrue(all(parameter.requires_grad for parameter in model.rec_parameters()))

    def test_ae_contrastive_query_cap_does_not_limit_rf_supervision(self):
        for checkpointing in (False, True):
            with self.subTest(checkpointing=checkpointing):
                config, model, trainer = config_and_model(
                    contrastive=True, grad_checkpointing=checkpointing
                )
                config.contrastive.max_queries = 2
                targets = []
                hook = model.diffloss.register_forward_pre_hook(
                    lambda module, args, kwargs: targets.append(len(kwargs["target"])), with_kwargs=True
                )
                try:
                    metrics = trainer.step(make_batch(), 1, diagnostics=True)
                finally:
                    hook.remove()
                self.assertEqual(targets, [2, 4])
                self.assertEqual(metrics["contrastive_ae/queries"], 2)
                self.assertGreater(metrics["ae_contrastive_grad_norm"], 0.)

    def test_frozen_ae_baseline_preserves_weights(self):
        _, model, trainer = config_and_model(frozen=True, contrastive=True)
        model.select_contrastive_positions = Mock(side_effect=AssertionError("AE is frozen"))
        model.contrastive_loss = Mock(side_effect=AssertionError("AE is frozen"))
        before = {key: value.clone() for key, value in model.ae.state_dict().items()}
        metrics = trainer.step(make_batch(), 1)
        self.assertEqual(metrics["ae_updated"], 0.)
        for key, value in model.ae.state_dict().items():
            torch.testing.assert_close(value, before[key])

    def test_schedules_follow_independent_update_counts(self):
        config, model, _ = config_and_model()
        config.ae.update_every = 2
        trainer = JointTrainer(model, config, torch.arange(1, 6), {}, 4)
        for _ in range(4):
            trainer.step(make_batch(), 1)
        self.assertEqual((trainer.ae_steps, trainer.rec_steps), (2, 4))
        self.assertEqual(trainer.ae_scheduler.last_epoch, 2)
        self.assertEqual(trainer.rec_scheduler.last_epoch, 4)

    def test_checkpoint_restores_both_optimizers_and_rng_for_identical_next_step(self):
        _, model, trainer = config_and_model(contrastive=True)
        trainer.step(make_batch(), 1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "latest.pt"
            save_checkpoint(path, trainer, epoch=1, best_epoch=1, best_score=.1, stale_epochs=0)
            trainer.step(make_batch(), 2)
            expected = {key: value.clone() for key, value in model.state_dict().items()}
            _, restored, second = config_and_model(contrastive=True)
            restored.refresh_catalog()
            load_checkpoint(read_checkpoint(path), second, resume=True)
            self.assertEqual(len(restored.catalog), 0)
            second.step(make_batch(), 2)
            self.assertEqual(second.global_step, 2)
            for key, value in restored.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
            self.assertEqual(second.ae_scheduler.state_dict(), trainer.ae_scheduler.state_dict())
            self.assertEqual(second.rec_scheduler.state_dict(), trainer.rec_scheduler.state_dict())

    def test_read_checkpoint_migrates_retired_ae_only_settings(self):
        config, _, trainer = config_and_model(contrastive=True)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "latest.pt"
            save_checkpoint(path, trainer, epoch=1, best_epoch=1, best_score=.1, stale_epochs=0)
            state = read_checkpoint(path)
            state["config"]["contrastive"].update(
                mode="ae_only", loss_type="infonce", lambda_ae=.01, lambda_rec=.01,
                start_epoch_ae=6, start_epoch_rec=1, ramp_epochs_ae=5, ramp_epochs_rec=5,
            )
            torch.save(state, path)
            migrated = read_checkpoint(path)
            self.assertEqual(migrated["config"]["contrastive"], config.to_dict()["contrastive"])
            load_checkpoint(migrated, trainer, resume=True)

    def test_read_checkpoint_rejects_retired_contrastive_modes(self):
        _, _, trainer = config_and_model(contrastive=True)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "latest.pt"
            save_checkpoint(path, trainer, epoch=1, best_epoch=1, best_score=.1, stale_epochs=0)
            state = read_checkpoint(path)
            for mode in ("both", "ae_hstu"):
                with self.subTest(mode=mode):
                    state["config"]["contrastive"]["mode"] = mode
                    torch.save(state, path)
                    with self.assertRaisesRegex(ValueError, "AE-only"):
                        read_checkpoint(path)
            state["config"]["contrastive"].pop("mode")
            state["config"]["contrastive"]["lambda_ae"] = .01
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "AE-only"):
                read_checkpoint(path)

    def test_resume_rejects_changed_evaluation_or_sample_source(self):
        config, _, trainer = config_and_model()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "latest.pt"
            save_checkpoint(path, trainer, epoch=1, best_epoch=1, best_score=.1, stale_epochs=0,
                            last_validation_epoch=0)
            state = read_checkpoint(path)
            self.assertEqual(state["last_validation_epoch"], 0)
            config.evaluation.val_metric = "recall@5"
            with self.assertRaisesRegex(ValueError, "evaluation"):
                load_checkpoint(state, trainer, resume=True)
            load_checkpoint(state, trainer, resume=False)
            trainer.source = {"fixture": True, "sample_source_sha256": "changed"}
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                load_checkpoint(state, trainer, resume=True)

    def test_resume_accepts_retired_initialization_and_position_fields(self):
        _, model, trainer = config_and_model(contrastive=True)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "latest.pt"
            save_checkpoint(path, trainer, epoch=1, best_epoch=1, best_score=.1, stale_epochs=0)
            state = read_checkpoint(path)
            state["config"]["ae"]["init_mode"] = "pretrained"
            state["config"]["ae"]["init_artifact"] = "cache/other"
            state["config"]["recommender"]["supervision_mode"] = "all_positions"
            state["config"]["bridge"] = state["config"].pop("contrastive")
            state["config"]["bridge"]["positions"] = "all_positions"
            state["model"] = {
                key.replace("contrastive_head.", "bridge_head.", 1) if key.startswith("contrastive_head.") else key: value
                for key, value in state["model"].items()
            }
            before = {key: value.clone() for key, value in model.state_dict().items()}
            load_checkpoint(state, trainer, resume=True)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, before[key])
            state["config"]["recommender"]["supervision_mode"] = "last_item"
            with self.assertRaisesRegex(ValueError, "every valid position"):
                load_checkpoint(state, trainer, resume=True)


if __name__ == "__main__":
    unittest.main()
