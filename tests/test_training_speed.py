"""Small CPU equivalence checks for training and decoding optimizations."""
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from model import Config, GPT
from sample import load_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_inference_keeps_metadata_without_training_tensors(self):
        model = GPT(Config(16, block_size=16, n_layer=1, n_head=2, n_embd=8)).eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            torch.save(dict(model=model.state_dict(), optimizer={"unused": torch.ones(3)},
                            model_config=model.cfg.__dict__, vocab=[str(i) for i in range(16)],
                            step=7, args={"task": "goblin"}), path)
            loaded, vocab, metadata = load_checkpoint(path, "cpu")
        self.assertEqual(metadata["step"], 7)
        self.assertEqual(metadata["args"], {"task": "goblin"})
        self.assertNotIn("model", metadata)
        self.assertNotIn("optimizer", metadata)
        self.assertEqual(len(vocab), 16)
        tokens = torch.tensor([[1, 2, 3]])
        torch.testing.assert_close(loaded(tokens)[0], model(tokens)[0])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
