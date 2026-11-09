import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from contextlib import ExitStack, nullcontext

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
        config.update(sent_emb_compression="pca", geometry_ae_cache=None, eval_interval=1)
        config.update(dataset="AmazonReviews2014", model="TigerDiff", category="Beauty",
                      run_id="task_a", run_timestamp="20261006-150000")
        paths = checkpoint_paths(config)
        root = Path("outputs/ckpt/AmazonReviews2014/TigerDiff/Beauty/task_a")
        self.assertEqual(paths["root"], root)
        self.assertEqual(paths["latest"], root / "latest_task_a.pth")
        self.assertEqual(paths["best"], root / "best_task_a.pth")
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
            best = root / "best_task.pth"
            best.write_bytes(legacy.read_bytes())
            self.assertEqual(resolve_checkpoint(root, prefer="best"), best)
            self.assertEqual(resolve_checkpoint(root), latest)
            best.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "No checkpoint found"):
                resolve_checkpoint(root, prefer="best")

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
        config.update(sent_emb_compression="pca", geometry_ae_cache=None, eval_interval=1)
        config.update(epochs=7, steps=None, patience=None, warmup_steps=0, save_interval=3,
                      use_ddp=False, run_local_time="test", run_id="test", accelerator=accelerator)
        loader = DataLoader([{"x": torch.tensor(1.), "labels": torch.tensor([1])}], batch_size=1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "latest.pth"
                with patch("engine.train.evaluate", side_effect=[{"ndcg@10": i} for i in range(7)]):
                    with patch("engine.train.save_checkpoint", wraps=save_checkpoint) as save:
                        train(TinyModel(), loader, loader, config, None, accelerator, path)
                latest_saves = [call for call in save.call_args_list if call.args[1] == path]
                best_path = Path(directory) / "best.pth"
                best_saves = [call for call in save.call_args_list if call.args[1] == best_path]
                self.assertEqual([call.kwargs["epoch"] for call in latest_saves], [3, 6, 7])
                self.assertEqual([call.kwargs["epoch"] for call in best_saves], list(range(1, 8)))
                self.assertEqual(set(Path(directory).iterdir()), {path, best_path})
                state = load_checkpoint(TinyModel(), path, "cpu")
                self.assertEqual(state["epoch"], 7)
                self.assertEqual(state["best_epoch"], 7)
                self.assertEqual(load_checkpoint(TinyModel(), best_path, "cpu")["epoch"], 7)
        finally:
            accelerator.end_training()

    def test_resume_restores_optimizer_scheduler_and_epoch(self):
        accelerator = Accelerator(cpu=True)
        config = deepcopy(CONFIG)
        config.update(sent_emb_compression="pca", geometry_ae_cache=None, eval_interval=1)
        config.update(epochs=2, steps=None, patience=None, warmup_steps=0, save_interval=5,
                      use_ddp=False, run_local_time="test", run_id="test", accelerator=accelerator)
        loader = DataLoader([{"x": torch.tensor(1.), "labels": torch.tensor([1])}], batch_size=1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "latest.pth"
                with patch("engine.train.evaluate", return_value={"ndcg@10": 0.5}):
                    train(TinyModel(), loader, loader, config, None, accelerator, path)
                best_path = Path(directory) / "best.pth"
                best_bytes = best_path.read_bytes()
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
                self.assertEqual(best_path.read_bytes(), best_bytes)
                unchanged = path.read_bytes()
                with patch("engine.train.evaluate") as validation:
                    train(TinyModel(), loader, loader, config, None, accelerator, path, resume_checkpoint=path)
                validation.assert_not_called()
                self.assertEqual(path.read_bytes(), unchanged)
        finally:
            accelerator.end_training()

    def test_resume_without_best_restarts_selection_from_resumed_validation(self):
        accelerator = Accelerator(cpu=True)
        config = deepcopy(CONFIG)
        config.update(sent_emb_compression="pca", geometry_ae_cache=None, eval_interval=1)
        config.update(epochs=2, steps=None, patience=None, warmup_steps=0, save_interval=5,
                      use_ddp=False, run_id="test", accelerator=accelerator)
        loader = DataLoader([{"x": torch.tensor(1.), "labels": torch.tensor([1])}], batch_size=1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                latest = Path(directory) / "latest.pth"
                save_checkpoint(TinyModel(), latest, accelerator, epoch=1, best_epoch=1, best_val_score=0.9)
                with patch("engine.train.evaluate", return_value={"ndcg@10": 0.5}):
                    train(TinyModel(), loader, loader, config, None, accelerator, latest, resume_checkpoint=latest)
                state = load_checkpoint(TinyModel(), Path(directory) / "best.pth", "cpu")
                self.assertEqual((state["epoch"], state["best_val_score"]), (2, 0.5))
        finally:
            accelerator.end_training()

    def test_final_epoch_validates_when_outside_eval_interval(self):
        accelerator = Accelerator(cpu=True)
        config = deepcopy(CONFIG)
        config.update(sent_emb_compression="pca", geometry_ae_cache=None, eval_interval=1)
        config.update(epochs=1, steps=None, eval_interval=5, patience=None, warmup_steps=0,
                      use_ddp=False, run_id="test", accelerator=accelerator)
        loader = DataLoader([{"x": torch.tensor(1.), "labels": torch.tensor([1])}], batch_size=1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                latest = Path(directory) / "latest.pth"
                with patch("engine.train.evaluate", return_value={"ndcg@10": 0.5}) as validation:
                    train(TinyModel(), loader, loader, config, None, accelerator, latest)
                validation.assert_called_once()
                self.assertTrue((Path(directory) / "best.pth").is_file())
                self.assertTrue(latest.is_file())
        finally:
            accelerator.end_training()

    def test_resume_does_not_overwrite_newer_best_with_worse_score(self):
        accelerator = Accelerator(cpu=True)
        config = deepcopy(CONFIG)
        config.update(sent_emb_compression="pca", geometry_ae_cache=None, eval_interval=1)
        config.update(epochs=2, steps=None, patience=None, warmup_steps=0,
                      use_ddp=False, run_id="test", accelerator=accelerator)
        loader = DataLoader([{"x": torch.tensor(1.), "labels": torch.tensor([1])}], batch_size=1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                latest, best = Path(directory) / "latest.pth", Path(directory) / "best.pth"
                save_checkpoint(TinyModel(), latest, accelerator, epoch=1, best_epoch=1, best_val_score=0.5)
                save_checkpoint(TinyModel(), best, accelerator, epoch=2, best_epoch=2, best_val_score=0.9)
                previous = best.read_bytes()
                with patch("engine.train.evaluate", return_value={"ndcg@10": 0.7}):
                    _, best_epoch, score = train(TinyModel(), loader, loader, config, None,
                                                accelerator, latest, resume_checkpoint=latest)
                self.assertEqual((best_epoch, score), (2, 0.9))
                self.assertEqual(best.read_bytes(), previous)
        finally:
            accelerator.end_training()

    def test_entry_uses_best_for_test_and_latest_for_resume(self):
        import train as entry
        cases = [
            (False, False, False, True), (True, False, False, True),
            (True, False, True, True), (False, True, False, True),
            (True, False, True, False),
        ]
        for resume, eval_only, completed, has_best in cases:
            with self.subTest(resume=resume, eval_only=eval_only, completed=completed, has_best=has_best):
                with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                    config = deepcopy(CONFIG)
                    config.update(sent_emb_compression="pca", geometry_ae_cache=None, eval_interval=1)
                    config.update(dataset="AmazonReviews2014", model="TigerDiff", ckpt_dir=directory)
                    paths = checkpoint_paths(config)
                    paths["root"].mkdir(parents=True)
                    paths["latest"].touch()
                    if has_best:
                        paths["best"].touch()
                    args = SimpleNamespace(checkpoint=None, resume=resume, eval_only=eval_only, smoke_test=False)
                    accelerator = Mock(is_main_process=True)
                    accelerator.main_process_first.side_effect = lambda: nullcontext()
                    model = Mock()
                    dataset = Mock()
                    dataset.split.return_value = {}
                    tokenizer = Mock()
                    tokenizer.tokenize.return_value = dict.fromkeys(("train", "val", "test"), [{}])
                    replacements = {
                        "parse_config": (config, args), "Accelerator": accelerator,
                        "init_device": (torch.device("cpu"), False), "init_seed": None,
                        "init_logger": None, "AmazonReviews2014": dataset,
                        "TigerDiffTokenizer": tokenizer, "TigerDiff": model,
                        "load_checkpoint": {}, "prepare_test_model": (model, []),
                        "evaluate": {"recall@5": 0.1, "ndcg@5": 0.1, "recall@10": 0.2, "ndcg@10": 0.2},
                        "train": (model, None, None) if completed else (model, 1, 0.9),
                    }
                    mocks = {name: stack.enter_context(patch.object(entry, name, return_value=value))
                             for name, value in replacements.items()}
                    if not has_best:
                        with self.assertRaisesRegex(RuntimeError, "No best checkpoint"):
                            entry.main()
                        mocks["prepare_test_model"].assert_not_called()
                        continue
                    entry.main()
                    self.assertEqual(mocks["prepare_test_model"].call_args.args[3], paths["best"])
                    if resume:
                        self.assertEqual(mocks["train"].call_args.kwargs["resume_checkpoint"], paths["latest"])
                    if eval_only:
                        mocks["train"].assert_not_called()
                        self.assertEqual(mocks["load_checkpoint"].call_args.args[1], paths["best"])
