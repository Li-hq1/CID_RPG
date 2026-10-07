import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from config import CONFIG
from engine.checkpoint import load_checkpoint, resolve_checkpoint, save_checkpoint
from engine.train import train
from misc.config import parse_config
from misc.logging import checkpoint_paths, get_file_name
from tests.test_engine import TinyModel


class CheckpointTest(unittest.TestCase):
    def test_config_modes_and_save_interval(self):
        config, args = parse_config(["--resume", "--save_interval=7", "--run_id=ablation_a"])
        self.assertTrue(args.resume)
        self.assertEqual(config["save_interval"], 7)
        _, args = parse_config(["--eval-only"])
        self.assertTrue(args.eval_only)
        with self.assertRaisesRegex(ValueError, "save_interval"):
            parse_config(["--save_interval=0"])
        with self.assertRaisesRegex(ValueError, "run_id"):
            parse_config(["--run_id=../outside"])

    def test_short_checkpoint_and_matching_log_directory(self):
        config = deepcopy(CONFIG)
        config.update(dataset="AmazonReviews2014", model="TigerDiff", category="Beauty",
                      run_id="task_a", run_timestamp="20261006-150000")
        paths = checkpoint_paths(config)
        root = Path("ckpt/AmazonReviews2014/TigerDiff/Beauty/task_a")
        self.assertEqual(paths["root"], root)
        self.assertEqual(paths["latest"], root / "latest_task_a.pth")
        self.assertEqual(paths["log"], root / "train_20261006-150000_task_a.log")
        self.assertTrue(get_file_name(config, ".log").endswith("_task_a.log"))

    def test_directory_selection_prefers_latest_and_loads_legacy_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "epoch_1.pth"
            torch.save(TinyModel().state_dict(), legacy)
            self.assertEqual(resolve_checkpoint(root), legacy)
            loaded = TinyModel()
            self.assertEqual(load_checkpoint(loaded, root, "cpu:0"), {})
            latest = root / "latest_task.pth"
            latest.write_bytes(legacy.read_bytes())
            another = root / "epoch_999.pth"
            another.write_bytes(legacy.read_bytes())
            self.assertEqual(resolve_checkpoint(root), latest)

    def test_atomic_write_failure_preserves_previous_checkpoint(self):
        accelerator = SimpleNamespace(is_main_process=True, unwrap_model=lambda model: model)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pth"
            save_checkpoint(TinyModel(), path, accelerator, epoch=1)
            previous = path.read_bytes()
            with patch("engine.checkpoint.torch.save", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    save_checkpoint(TinyModel(), path, accelerator, epoch=2)
            self.assertEqual(path.read_bytes(), previous)
            self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_saving_occurs_only_at_interval_and_final_epoch(self):
        accelerator = Accelerator(cpu=True)
        config = deepcopy(CONFIG)
        config.update(epochs=7, steps=None, patience=None, warmup_steps=0, save_interval=3,
                      use_ddp=False, run_local_time="test", run_id="test", accelerator=accelerator)
        loader = DataLoader([{"x": torch.tensor(1.), "labels": torch.tensor([1])}], batch_size=1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "latest.pth"
                with patch("engine.train.evaluate", side_effect=[{"ndcg@10": i} for i in range(7)]):
                    with patch("engine.train.save_checkpoint", wraps=save_checkpoint) as save:
                        train(TinyModel(), loader, loader, config, None, accelerator, path)
                self.assertEqual([call.kwargs["epoch"] for call in save.call_args_list], [3, 6, 7])
                self.assertEqual(list(Path(directory).iterdir()), [path])
                state = load_checkpoint(TinyModel(), path, "cpu")
                self.assertEqual(state["epoch"], 7)
                self.assertEqual(state["best_epoch"], 7)
        finally:
            accelerator.end_training()

    def test_resume_restores_optimizer_scheduler_and_epoch(self):
        accelerator = Accelerator(cpu=True)
        config = deepcopy(CONFIG)
        config.update(epochs=2, steps=None, patience=None, warmup_steps=0, save_interval=5,
                      use_ddp=False, run_local_time="test", run_id="test", accelerator=accelerator)
        loader = DataLoader([{"x": torch.tensor(1.), "labels": torch.tensor([1])}], batch_size=1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "latest.pth"
                with patch("engine.train.evaluate", return_value={"ndcg@10": 0.5}):
                    train(TinyModel(), loader, loader, config, None, accelerator, path)
                config["epochs"] = 4
                resumed = TinyModel()
                with patch("engine.train.evaluate", return_value={"ndcg@10": 0.5}) as validation:
                    train(resumed, loader, loader, config, None, accelerator, path, resume_checkpoint=path)
                self.assertEqual(validation.call_count, 2)
                state = load_checkpoint(TinyModel(), path, "cpu")
                self.assertEqual(state["epoch"], 4)
                self.assertEqual(state["best_epoch"], 1)
                self.assertEqual(state["scheduler"]["last_epoch"], 4)
                self.assertEqual(next(iter(state["optimizer"]["state"].values()))["step"].item(), 4)
                unchanged = path.read_bytes()
                with patch("engine.train.evaluate") as validation:
                    train(TinyModel(), loader, loader, config, None, accelerator, path, resume_checkpoint=path)
                validation.assert_not_called()
                self.assertEqual(path.read_bytes(), unchanged)
        finally:
            accelerator.end_training()
