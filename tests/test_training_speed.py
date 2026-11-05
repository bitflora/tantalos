"""Small CPU equivalence checks for training and decoding optimizations."""
from pathlib import Path
import sys
import random
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from model import Config, GPT
from sample import load_checkpoint
from data import Levels, GoblinPlans, PAD


def datasets():
    levels = Levels(torch.tensor([5, 6, 7, 8], dtype=torch.int16), torch.tensor([0, 1, 4]),
                    [[9, 10], [10, 9]], [11, 12])
    goblin = GoblinPlans(torch.tensor([5, 6, 7], dtype=torch.int16), torch.tensor([0, 1, 3]),
                        torch.tensor([8, 9, 10, 11, 12, 13], dtype=torch.int16),
                        torch.tensor([0, 1, 4, 6]), torch.tensor([0, 2, 3]), [1, 2], [1, 1])
    return levels, goblin


class BatchTests(unittest.TestCase):
    def test_direct_copies_match_sequences_and_rng(self):
        for levels in datasets():
            for seed in range(20):
                random.seed(seed)
                seqs = [levels.sequence(i, .7) for i in [0, 1, 0]]
                after = random.getstate()
                width = max(len(ids) - 1 for ids, _ in seqs)
                x = torch.full((3, width), PAD, dtype=torch.long)
                y = torch.full((3, width), -100, dtype=torch.long)
                for row, (ids, learn) in enumerate(seqs):
                    x[row, :len(ids) - 1] = torch.tensor(ids[:-1])
                    y[row, :len(ids) - 1] = torch.tensor([t if keep else -100
                                                       for t, keep in zip(ids[1:], learn[1:])])
                random.seed(seed)
                actual_x, actual_y, count = levels.batch([0, 1, 0], .7, "cpu")
                self.assertEqual(random.getstate(), after)
                torch.testing.assert_close(actual_x, x)
                torch.testing.assert_close(actual_y, y)
                self.assertEqual(count, int((y != -100).sum()))


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


class GenerationTests(unittest.TestCase):
    def test_last_only_preserves_logits_and_full_kv_cache(self):
        for norm in [False, True]:
            model = GPT(Config(16, 16, 2, 2, 8, dropout=0, qk_norm=norm)).eval()
            tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
            full = [{"size": 5} for _ in model.blocks]
            last = [{"size": 5} for _ in model.blocks]
            with torch.no_grad():
                expected, _ = model(tokens, caches=full)
                actual, _ = model(tokens, caches=last, last_only=True)
                torch.testing.assert_close(actual, expected[:, -1:])
                for a, b in zip(full, last):
                    for key in ["k", "v"]:
                        torch.testing.assert_close(a[key][:, :, :3], b[key][:, :, :3])
                torch.testing.assert_close(model(tokens[:, :1], caches=full, start=3)[0],
                                           model(tokens[:, :1], caches=last, start=3)[0])
            self.assertEqual(model(tokens)[0].shape, (2, 3, 16))
            with self.assertRaises(ValueError):
                model(tokens, tokens, last_only=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
