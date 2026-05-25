import unittest
from unittest.mock import patch

from genrec.utils import format_hyper_parameters, get_file_name


class UtilsTest(unittest.TestCase):
    def test_get_file_name_caps_long_command_line_component(self):
        config = {
            "run_id": "tigerdiff_toy",
            "run_local_time": "May-07-2026_01-26",
            "file_name_max_len": 180,
            "lr": 0.0005,
            "accelerator": object(),
        }
        argv = ["main.py"] + [f"--param_{idx}=value_{idx}" for idx in range(100)]

        with patch("sys.argv", argv):
            name = get_file_name(config, suffix=".pth")

        self.assertLessEqual(len(name), 180)
        self.assertTrue(name.endswith(".pth"))
        self.assertIn("May-07-2026_01-26", name)

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
