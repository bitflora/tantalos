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
    def test_batched_prefill_matches_independent_greedy_and_scores(self):
        torch.manual_seed(17)
        for norm in [False, True]:
            model = GPT(Config(16, 16, 2, 2, 8, dropout=0, qk_norm=norm)).eval()
            prompts = [[1, 2], [3, 4, 5], [3, 4, 5], [6], [7] * 14, [8, 9], [10, 11, 12]]
            expected = [model.generate([p], 5, 0, top_k=1)[0] for p in prompts]
            calls = []
            hook = model.register_forward_pre_hook(lambda m, args: calls.append(tuple(args[0].shape)))
            actual, scores = model.generate_many(prompts, 5, 0, top_k=1, batch=3, logprob=True)
            hook.remove()
            self.assertEqual(actual, expected)
            self.assertTrue(any(b > 1 and t > 1 for b, t in calls))
            with torch.no_grad():
                for p, seq, score in zip(prompts, actual, scores):
                    reference = 0
                    for j, token in enumerate(seq):
                        logits, _ = model(torch.tensor([p + seq[:j]]))
                        reference += logits[0, -1].log_softmax(-1)[token].item()
                    self.assertAlmostEqual(score, reference, places=5)

    def test_prefill_preserves_constrained_refills_and_sample_order(self):
        class Constraint:
            def __init__(self):
                self.finished = []

            def __call__(self, idx, seqs):
                mask = torch.zeros(len(idx), 16, dtype=torch.bool)
                for r, (i, seq) in enumerate(zip(idx, seqs)):
                    if len(seq) >= i % 3 + 1:
                        mask[r, 0] = True
                    else:
                        mask[r, 1:4] = True
                return mask

            def finish(self, i, seq):
                self.finished.append(i)
                return seq

        model = GPT(Config(16, 16, 2, 2, 8, dropout=0, qk_norm=True)).eval()
        prompts = [[1, 2], [3, 4, 5], [1, 2], [6, 7, 8], [9, 10], [9, 10]]
        results = []
        for size in [1, 4]:
            constraint = Constraint()
            torch.manual_seed(7)
            seqs, scores = model.generate_many(prompts, 5, 0, batch=3, constrain=constraint,
                                               logprob=True, prefill_batch=size)
            self.assertEqual(sorted(constraint.finished), list(range(len(prompts))))
            results.append((seqs, scores))
        self.assertEqual(results[0][0], results[1][0])
        torch.testing.assert_close(torch.tensor(results[0][1]), torch.tensor(results[1][1]))

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
