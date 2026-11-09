import json
import unittest
from pathlib import Path

from joint_training.config import parse_config
from protocol import training_windows


ROOT = Path(__file__).resolve().parents[1]


class SchemeConfigTest(unittest.TestCase):
    def test_readable_config_matches_frozen_scheme(self):
        frozen, _ = parse_config(["--config", str(ROOT / "configs/frozen.json")])
        minimal, _ = parse_config(["--config", str(ROOT / "configs/minimal.json")])
        self.assertEqual(frozen.to_dict(), minimal.to_dict())
        flow = frozen.flow_ranking
        self.assertTrue(flow.enabled)
        self.assertEqual(flow.path, "endpoint")
        self.assertEqual(flow.source, "history")
        self.assertEqual(flow.noise_scale, 0.3)
        self.assertEqual(flow.sampling, "uniform_corrected")
        self.assertEqual(flow.negatives, 1024)
        self.assertEqual(frozen.recommender.encoder_type, "hstu")
        self.assertEqual(frozen.recommender.num_sampling_steps, 2)
        self.assertFalse(frozen.training.evaluate_test)
        self.assertFalse(frozen.ranking.enabled)
        self.assertFalse(frozen.contrastive.enabled)

    def test_direct_reference_is_not_the_scheme(self):
        direct, _ = parse_config(["--config", str(ROOT / "configs/direct_reference.json")])
        self.assertEqual(direct.flow_ranking.path, "direct")
        self.assertNotEqual(direct.training.run_id, "ugrec_v5")

    def test_later_windows_supervise_only_the_last_position(self):
        short, short_labels = training_windows([[1, 2, 3, 4]])
        self.assertEqual(len(short), 1)
        self.assertEqual(len(short[0]), 50)
        self.assertEqual(short[0][:3], [1, 2, 3])
        self.assertEqual(short_labels[0][:4], [2, 3, 4, -100])

        sequence = list(range(1, 61))
        inputs, labels = training_windows([sequence])
        self.assertGreater(len(inputs), 1)
        self.assertNotIn(-100, labels[0][:50])
        self.assertEqual(labels[1][:49], [-100] * 49)
        self.assertEqual(labels[1][49], 52)

    def test_frozen_file_records_the_scheme_run_id(self):
        values = json.loads((ROOT / "configs/frozen.json").read_text())
        self.assertEqual(values["training"]["run_id"], "ugrec_v5")
