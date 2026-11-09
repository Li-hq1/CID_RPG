import unittest
from unittest.mock import patch

from misc.logging import format_hyper_parameters, get_file_name


class UtilsTest(unittest.TestCase):
    def test_file_name_uses_timestamp_and_task_name_only(self):
        config = {
            "run_id": "tigerdiff_toy",
            "run_timestamp": "20261006-150000",
            "lr": 0.0005,
            "accelerator": object(),
        }
        argv = ["main.py"] + [f"--param_{idx}=value_{idx}" for idx in range(100)]

        with patch("sys.argv", argv):
            name = get_file_name(config, suffix=".pth")

        self.assertEqual(name, "20261006-150000_tigerdiff_toy.pth")

    def test_format_hyper_parameters_dumps_loggable_config(self):
        config = {
            "lr": 0.003,
            "topk": [5, 10],
            "device": object(),
            "accelerator": object(),
        }

        dumped = format_hyper_parameters(config)

        self.assertIn("lr: 0.003", dumped)
        self.assertIn("topk:", dumped)
        self.assertIn("- 5", dumped)
        self.assertIn("device: ", dumped)
        self.assertNotIn("accelerator", dumped)


if __name__ == "__main__":
    unittest.main()
