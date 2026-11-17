"""train/value.py, the value heads of bake-off run 3 (docs/architecture-plan-2026-11-15.md):
record boundaries, labels, features matching a single full pass, and guided sampling writing
whole records that join up into the levels returned.

    .venv/Scripts/python -m unittest tests/test_value.py -v
"""
import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
import rated  # noqa: E402
import sample  # noqa: E402
import value  # noqa: E402
from data import EOS, Vocab  # noqa: E402
from model import Config, build_model  # noqa: E402

LEVEL = ("size 8 8 4 open 1 1 1 6 6 2 white +z 1 1 0 6 6 0 entry 1 2 1 +x exit 6 6 1 -x inv "
         "button 2 2 1 cube > exit cube 3 3 1 standard")


def vocab():
    return Vocab.build(ROOT / "data" / "v5b" / "vocab.txt")


class Boundaries(unittest.TestCase):
    def test_before_each_record_and_the_end(self):
        v = vocab()
        p0 = sample.prompt(v, rated.prompt(v), ())
        ids = p0 + v.encode(LEVEL.split()) + [EOS]
        ends = value.boundaries(ids, len(p0), v.itos)
        toks = LEVEL.split()
        # The last token of each record: `4` (size), `2` (open), `0` (white), `+x`, `inv`, `exit`
        # (the wire's), `standard`.
        self.assertEqual([toks[e - len(p0)] for e in ends], ["4", "2", "0", "+x", "inv", "exit", "standard"])
        self.assertEqual(ends[-1], len(ids) - 2)


class Labels(unittest.TestCase):
    def test_data_and_samples(self):
        line, v, ln, st, fam = value.labels({"tokens": LEVEL, "verdict": "solvable", "min_len": 6,
                                             "metrics": {"mechanic_bits": 0}})
        self.assertEqual((v, ln, st), (0, math.log1p(6), 0.0))
        self.assertEqual(fam[value.FAMILIES.index("cubes+buttons")], 1.0)
        self.assertIsNone(value.labels({"tokens": LEVEL, "verdict": "invalid"}))
        lab = value.labels({"tokens": LEVEL, "verdict": "published", "solver_verdict": "uncertain"})
        self.assertEqual(lab[1], 1)
        self.assertTrue(math.isnan(lab[2]) and math.isnan(lab[3]))
        lab = value.labels({"sample": LEVEL, "outcome": "solvable", "min_len": 3, "state": 2, "tricks": 0})
        self.assertEqual(lab[3], 1.0)


class Guided(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.v = vocab()
        self.model = build_model(Config(len(self.v), 160, 2, 2, 32, 0.0, qk_norm=True)).eval()

    def test_features_match_one_pass(self):
        p0 = sample.prompt(self.v, rated.prompt(self.v), ())
        ids = p0 + self.v.encode(LEVEL.split()) + [EOS]
        ends = value.boundaries(ids, len(p0), self.v.itos)
        both = value.features(self.model, [ids, ids[:20]], [ends, [19]], len(p0))
        alone = value.features(self.model, [ids[: ends[2] + 1]], [[ends[2]]], len(p0))
        self.assertEqual(both.shape, (len(ends) + 1, 64))
        torch.testing.assert_close(both[2], alone[0], atol=1e-4, rtol=1e-4)

    def test_generates_whole_levels(self):
        heads = value.Heads(64).eval()
        lines = value.generate(self.model, self.v, heads, 3, rated.prompt(self.v), temperature=1.0, k=3)
        self.assertEqual(len(lines), 3)
        for line in lines:
            self.assertTrue(all(t in self.v.stoi for t in line.split()))
            self.assertLessEqual(len(line.split()), 160)


if __name__ == "__main__":
    unittest.main()
