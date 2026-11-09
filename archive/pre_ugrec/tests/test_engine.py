import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers.optimization import get_scheduler

from config import CONFIG
from engine.checkpoint import load_checkpoint, prepare_test_model
from engine.evaluate import evaluate
from engine.train import train


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        return SimpleNamespace(loss=(self.weight * batch["x"]).square().mean())

    def generate(self, batch, n_return_sequences=1):
        return torch.ones(len(batch["x"]), n_return_sequences, 1, dtype=torch.long,
                          device=batch["x"].device)


class EngineTest(unittest.TestCase):
    def test_ratio_warmup_scales_with_training_budget_and_cosine_decays(self):
        accelerator = Accelerator(cpu=True)
        config = deepcopy(CONFIG)
        config.update(epochs=5, steps=None, warmup_ratio=0.2, warmup_steps=None,
                      patience=None, use_ddp=False, run_id="test", accelerator=accelerator)
        loader = DataLoader([{"x": torch.tensor(1.), "labels": torch.tensor([1])}] * 4,
                            batch_size=2)
        learning_rates = []
        schedulers = []

        def scheduler(*args, **kwargs):
            self.assertEqual(args[0], "cosine")
            self.assertEqual(kwargs["num_warmup_steps"], 2)
            self.assertEqual(kwargs["num_training_steps"], 10)
            result = get_scheduler(*args, **kwargs)
            schedulers.append(result)
            return result

        def validation(*args, **kwargs):
            learning_rates.append(schedulers[0].get_last_lr()[0])
            return {"ndcg@10": 0.5}

        try:
            with tempfile.TemporaryDirectory() as directory:
                with patch("engine.train.evaluate", side_effect=validation), \
                     patch("engine.train.get_scheduler", side_effect=scheduler):
                    train(TinyModel(), loader, loader, config, None, accelerator,
                          str(Path(directory) / "latest.pth"))
            self.assertEqual(schedulers[0].last_epoch, 10)
            self.assertAlmostEqual(schedulers[0].lr_lambdas[0](1), 0.5)
            self.assertEqual(len(learning_rates), 5)
            self.assertAlmostEqual(learning_rates[0], config["lr"])
            self.assertTrue(all(a > b for a, b in zip(learning_rates, learning_rates[1:])))
            self.assertAlmostEqual(learning_rates[-1], 0.0)
        finally:
            accelerator.end_training()

    def test_whole_epochs_scheduler_and_final_checkpoint_semantics(self):
        accelerator = Accelerator(cpu=True)
        config = deepcopy(CONFIG)
        config.update(lr=0.05, warmup_steps=0, steps=3, epochs=1, use_ddp=False,
                      run_local_time="test", run_id="test", accelerator=accelerator)
        loader = DataLoader([{"x": torch.tensor(1.), "labels": torch.tensor([1])}] * 4, batch_size=2)
        model = TinyModel()
        snapshots = []
        schedulers = []

        def validation(current_model, *args, **kwargs):
            snapshots.append(accelerator.unwrap_model(current_model).weight.detach().clone())
            return {"ndcg@10": 1.0 if len(snapshots) == 1 else 0.5}

        def scheduler(*args, **kwargs):
            result = get_scheduler(*args, **kwargs)
            schedulers.append(result)
            return result

        try:
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = str(Path(directory) / "latest.pth")
                with patch("engine.train.evaluate", side_effect=validation), patch("engine.train.get_scheduler", side_effect=scheduler):
                    model, best_epoch, score = train(model, loader, loader, config, None, accelerator, checkpoint)
                self.assertEqual((best_epoch, score), (1, 1.0))
                self.assertEqual(len(snapshots), 2)
                self.assertEqual(schedulers[0].last_epoch, 4)
                self.assertLess(snapshots[1].item(), snapshots[0].item())
                loaded = TinyModel()
                load_checkpoint(loaded, checkpoint, torch.device("cpu:0"))
                torch.testing.assert_close(loaded.weight, snapshots[1])
                best = Path(directory) / "best.pth"
                state = load_checkpoint(TinyModel(), best, "cpu")
                self.assertEqual((state["epoch"], state["best_epoch"]), (1, 1))
                final_model, _ = prepare_test_model(model, loader, accelerator, best)
                torch.testing.assert_close(accelerator.unwrap_model(final_model).weight, snapshots[0])
        finally:
            accelerator.end_training()

    def test_real_metric_evaluation_and_zero_training_steps(self):
        accelerator = Accelerator(cpu=True)
        config = deepcopy(CONFIG)
        config.update(use_ddp=False, steps=0, accelerator=accelerator)
        loader = DataLoader([{"x": torch.tensor(1.), "labels": torch.tensor([1])}] * 3, batch_size=2)
        tokenizer = SimpleNamespace(eos_token=None, item2tokens=[None] * 10)
        model = TinyModel()
        try:
            results = evaluate(model, loader, config, tokenizer, accelerator)
            self.assertEqual(results["recall@10"], 1.0)
            self.assertEqual(results["ndcg@10"], 1.0)
            self.assertEqual(results["n_visited_items"], 10.0)
            same_model, epoch, score = train(model, loader, loader, config, tokenizer, accelerator, "unused.pth")
            self.assertIs(same_model, model)
            self.assertEqual((epoch, score), (None, None))
        finally:
            accelerator.end_training()

    def test_missing_latest_checkpoint_has_actionable_error(self):
        accelerator = Accelerator(cpu=True)
        try:
            with self.assertRaisesRegex(FileNotFoundError, "Checkpoint not found"):
                prepare_test_model(TinyModel(), [], accelerator, "/missing/latest.pth")
        finally:
            accelerator.end_training()

    def test_patience_stops_after_validation_without_improvement(self):
        accelerator = Accelerator(cpu=True)
        config = deepcopy(CONFIG)
        config.update(epochs=5, steps=None, patience=1, warmup_steps=0, use_ddp=False,
                      run_local_time="test", run_id="test", accelerator=accelerator)
        loader = DataLoader([{"x": torch.tensor(1.), "labels": torch.tensor([1])}], batch_size=1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                with patch("engine.train.evaluate", return_value={"ndcg@10": 0.0}) as validation:
                    _, epoch, score = train(TinyModel(), loader, loader, config, None,
                                            accelerator, str(Path(directory) / "latest.pth"))
                self.assertEqual(validation.call_count, 2)
                self.assertEqual((epoch, score), (1, 0.0))
        finally:
            accelerator.end_training()
