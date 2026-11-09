"""Unit tests for joint TensorBoard metric tagging and event writing."""

import tempfile
import unittest
from pathlib import Path

from joint_training.tensorboard import JointTensorBoard, metric_tag, scalar_items


class JointTensorBoardTest(unittest.TestCase):
    def test_metric_tag_groups(self):
        self.assertEqual(metric_tag("ae_loss"), "Loss/ae_loss")
        self.assertEqual(metric_tag("rf_loss"), "Loss/rf_loss")
        self.assertEqual(metric_tag("ae_lr"), "Train/ae_lr")
        self.assertEqual(metric_tag("ae_grad_norm"), "Grad/ae_grad_norm")
        self.assertEqual(metric_tag("ae_contrastive"), "Loss/ae_contrastive")
        self.assertEqual(metric_tag("ae/reconstruction"), "AE/reconstruction")

    def test_scalar_items_skip_non_finite_and_non_numeric(self):
        values = {"ndcg@10": 0.1, "note": "x", "bad": float("nan"), "ok": 2}
        self.assertEqual(list(scalar_items(values, prefix="Val_Metric")),
                         [("Val_Metric/ndcg@10", 0.1), ("Val_Metric/ok", 2.0)])

    def test_writer_creates_event_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tb = JointTensorBoard(Path(tmp) / "tensorboard")
            self.assertTrue(tb.enabled)
            tb.log_training({"ae_loss": 1.25, "ae/reconstruction": 0.5}, 1)
            tb.log_validation({"ndcg@10": 0.04}, 1)
            tb.close()
            events = list((Path(tmp) / "tensorboard").glob("events.out.tfevents.*"))
            self.assertTrue(events)


if __name__ == "__main__":
    unittest.main()
