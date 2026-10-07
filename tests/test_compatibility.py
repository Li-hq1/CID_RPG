import sys
import unittest


class CompatibilityTest(unittest.TestCase):
    def test_old_model_imports_forward_to_active_classes_without_rpg(self):
        from models.tigerdiff import TigerDiff, TigerSequenceEncoder
        from data.tigerdiff_data import TigerDiffTokenizer
        from genrec.models import TigerDiff as OldTigerDiff
        from genrec.models.TigerDiff.model import TigerSequenceEncoder as OldEncoder
        from genrec.models.TigerDiff.tokenizer import TigerDiffTokenizer as OldTokenizer
        from models.diffusion.diffloss import DiffLoss
        from genrec.models.diffusion.diffloss import DiffLoss as OldDiffLoss
        self.assertIs(OldTigerDiff, TigerDiff)
        self.assertIs(OldEncoder, TigerSequenceEncoder)
        self.assertIs(OldTokenizer, TigerDiffTokenizer)
        self.assertIs(OldDiffLoss, DiffLoss)
        self.assertNotIn("genrec.models.RPG.model", sys.modules)

    def test_main_entry_forwards_to_explicit_training_entry(self):
        import main
        import train
        self.assertIs(main.main, train.main)
