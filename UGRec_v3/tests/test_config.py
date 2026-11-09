import contextlib
import io
import unittest

from config import CONFIG
from misc.config import parse_config


class ConfigTest(unittest.TestCase):
    def test_defaults_match_training_launchers_and_do_not_share_mutable_values(self):
        config, _ = parse_config([])
        self.assertEqual((config["lr"], config["eval_batch_size"], config["diffloss_w"]),
                         (0.0005, 64, 512))
        config["topk"].append(20)
        self.assertEqual(CONFIG["topk"], [5, 10])
        self.assertEqual(config["supervision_mode"], "all_positions")

    def test_legacy_supervision_can_be_selected(self):
        config, _ = parse_config(["--supervision_mode=last_item"])
        self.assertEqual(config["supervision_mode"], "last_item")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_config(["--supervision_mode=unknown"])

    def test_explicit_overrides_and_boolean_conversion(self):
        config, args = parse_config(["--category=Beauty", "--lr", "0.001",
                                    "--use_rectified_flow=False", "--checkpoint=weights.pth"])
        self.assertEqual(config["category"], "Beauty")
        self.assertEqual(config["lr"], 0.001)
        self.assertFalse(config["use_rectified_flow"])
        self.assertEqual(args.checkpoint, "weights.pth")

    def test_smoke_options_win_over_training_overrides(self):
        config, args = parse_config(["--smoke-test", "--epochs=20", "--steps=100", "--warmup_steps=10"])
        self.assertTrue(args.smoke_test)
        self.assertEqual((config["epochs"], config["steps"], config["warmup_steps"]), (1, None, 0))
        self.assertEqual((config["train_batch_size"], config["eval_batch_size"]), (32, 32))

    def test_ratio_warmup_and_fixed_step_override(self):
        config, _ = parse_config([])
        self.assertEqual(config["warmup_ratio"], 0.1)
        self.assertIsNone(config["warmup_steps"])
        config, _ = parse_config(["--warmup_ratio=0.02"])
        self.assertEqual(config["warmup_ratio"], 0.02)
        self.assertIsNone(config["warmup_steps"])
        config, _ = parse_config(["--warmup_ratio=0.02", "--warmup_steps=0"])
        self.assertEqual(config["warmup_steps"], 0)
        for argv in (["--warmup_ratio=-0.1"], ["--warmup_ratio=1"],
                     ["--warmup_ratio=nan"], ["--warmup_steps=-1"]):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                parse_config(argv)

    def test_unknown_arguments_and_rpg_are_rejected(self):
        for argv in [["--unknown=1"], ["--lr_extra=1"], ["--model=RPG"]]:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_config(argv)

    def test_invalid_training_config_fails_before_loading_data(self):
        with self.assertRaisesRegex(ValueError, "train_batch_size"):
            parse_config(["--train_batch_size=0"])
