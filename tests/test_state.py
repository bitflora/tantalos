"""train/state.py and model.State, the state-conditioned arm (docs/architecture-plan-2026-11-15.md,
run 6):

- the grid of a level's geometry is `tantalos-score grid`'s (solid, goo, white sides);
- items paint their cells with their word and facing (glass boxes, plate targets, rail ends as
  parts); a wire target (`> panel 0`) is no item;
- snapshots sit at item words, each holding the items before it; training's `features` give
  each token the latest snapshot at or before it, at most SNAPSHOTS of them;
- a sampled row's grid (`Rows`) is the snapshot at its last item word;
- the model: a token sees only its own snapshot (a later grid changes nothing before it), and
  the grid does change what comes after it; encoding a grid alone or padded in a batch gives the
  same memory.

    .venv/Scripts/python -m unittest tests/test_state.py -v
"""
import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "tests"))
import sample  # noqa: E402
import state  # noqa: E402
from data import EOS, Vocab  # noqa: E402
from test_arch import DEVICE, tiny  # noqa: E402
from test_sample_grammar import TANTALOS  # noqa: E402

LEVEL = ("size 6 5 4 open 0 0 1 5 4 3 goo 1 1 0 2 1 0 white +x 0 0 1 4 4 3 black -z 0 0 1 1 1 1 "
         "entry 0 2 1 +x exit 5 2 1 -x glass 2 3 1 3 3 2 +z +y "
         "button 3 0 1 weighted > panel 0 > exit panel 4 4 1 +z +y 3 off "
         "plate 1 3 1 +z 4 1 1 +z rail 2 0 2 2 2 2 osc "
         "gel 3 3 3 -z bounce stream on")


def vocab():
    return Vocab.build(ROOT / "data" / "v5b" / "vocab.txt")


class Raster(unittest.TestCase):
    def test_geometry(self):
        lv = state.Level(LEVEL.split())
        self.assertTrue(lv.ok)
        self.assertEqual(lv.cell.shape, (6, 5, 4))
        self.assertEqual(lv.cell[0, 0, 0], state.SOLID)
        self.assertEqual(lv.cell[0, 0, 1], state.OPEN)
        self.assertEqual(lv.cell[1, 1, 0], state.GOO)
        # (4, 0, 1): its +x neighbour is open, so no +x surface; -y is the wall, not painted white.
        self.assertEqual(lv.white[4, 0, 1] & 1, 0)
        self.assertEqual(lv.white[5, 2, 1] & 1, 0)  # +x box stops at x 4
        self.assertEqual(lv.white[4, 4, 3] & 1, 0)  # +x of x 4 is open (x 5)
        # The floor: black -z means every -z surface white but the box's.
        self.assertEqual(lv.white[3, 3, 1] >> 5 & 1, 1)
        self.assertEqual(lv.white[0, 0, 1] >> 5 & 1, 0)

    @unittest.skipUnless(TANTALOS.exists(), "needs target/release/tantalos-score")
    def test_geometry_matches_rust(self):
        toks = LEVEL.split()
        geo = " ".join(toks[: toks.index("entry")])
        (cell, white), = state.rust_grids([geo], TANTALOS)
        lv = state.Level(geo.split())
        np.testing.assert_array_equal(lv.cell, cell)
        np.testing.assert_array_equal(lv.white, white)

    def test_items_paint_their_cells(self):
        toks = LEVEL.split()
        lv = state.Level(toks)
        words = [toks[s] for s in lv.starts]
        self.assertEqual(words, ["entry", "exit", "glass", "button", "panel", "plate", "rail", "gel"])
        _, _, item = lv.grids(len(toks))
        self.assertEqual(item[0, 2, 1], state.code("entry", "+x"))
        self.assertEqual(item[2, 3, 1], state.code("glass", "+z"))
        self.assertEqual(item[3, 3, 2], state.part("glass.box"))
        self.assertEqual(item[4, 1, 1], state.part("plate.target"))
        self.assertEqual(item[2, 2, 2], state.part("rail.end"))
        self.assertEqual(item[4, 4, 1], state.code("panel", "+z"))
        self.assertEqual(item[3, 3, 3], state.code("gel", "-z"))
        # Before the panel's word, only the items before it.
        _, _, before = lv.grids(lv.starts[4])
        self.assertEqual(before[4, 4, 1], 0)
        self.assertEqual(before[3, 0, 1], state.code("button"))


class Features(unittest.TestCase):
    def test_snapshots_and_which(self):
        v = vocab()
        p = sample.prompt(v, [], [])
        ids = p + v.encode(LEVEL.split()) + [EOS]
        g = state.features(torch.tensor([ids]), v.itos, limit=None)
        lv = state.Level(LEVEL.split())
        starts = [len(p) + s for s in lv.starts]
        self.assertEqual(g.grids.shape[0], len(starts))
        which = g.which[0].tolist()
        self.assertEqual(which[: starts[0]], [0] * starts[0])
        for k, s in enumerate(starts):
            self.assertEqual(which[s], k + 1)
            self.assertEqual(which[s - 1], k)
        g = state.features(torch.tensor([ids] * 3), v.itos, limit=3, rng=random.Random(1))
        self.assertEqual(g.grids.shape[0], 9)
        self.assertEqual(g.slots, 3)
        self.assertEqual(tuple(g.grids.shape[2:]), (8, 8, 4))

    def test_rows_follow_snapshots(self):
        v = vocab()
        p = sample.prompt(v, [], [])
        ids = p + v.encode(LEVEL.split())
        rows = state.Rows(v.itos, 1)
        rows.start(0, p)
        self.assertEqual(rows.take(), {0: None})
        snaps = dict(state.snapshots(ids, v.itos, limit=None))
        for k in range(len(p), len(ids)):
            rows.feed([ids[k]])
            fresh = rows.take()
            if k in snaps:
                self.assertEqual(list(fresh), [0])
                for a, b in zip(fresh[0], snaps[k]):
                    np.testing.assert_array_equal(a, b)
            else:
                self.assertEqual(fresh, {}, v.itos[ids[k]])


class Model(unittest.TestCase):
    def test_tokens_see_only_their_snapshot(self):
        v = vocab()
        m = tiny("state", v, block=256)
        with torch.no_grad():  # an untrained net barely reads its memory: make it count
            for c in m.cross.values():
                c.proj.weight.mul_(100)
        p = sample.prompt(v, [], [])
        ids = torch.tensor([p + v.encode(LEVEL.split()) + [EOS]])
        g = m.featurize(ids, None)
        lv = state.Level(LEVEL.split())
        last = len(p) + lv.starts[-1]
        changed = state.Grids(g.grids.clone(), g.sizes, g.owner, g.slot, g.which)
        changed.grids[-1, 2] = 5  # the last snapshot's items, all different
        with torch.no_grad():
            a, _ = m((ids.to(DEVICE), g.to(DEVICE)))
            b, _ = m((ids.to(DEVICE), changed.to(DEVICE)))
        self.assertTrue(torch.equal(a[0, :last], b[0, :last]))
        self.assertFalse(torch.equal(a[0, last:], b[0, last:]))

    def test_padding_changes_nothing(self):
        v = vocab()
        m = tiny("state", v)
        lv = state.Level(LEVEL.split())
        g = lv.grids(len(LEVEL.split()))
        big = state.Level("size 13 13 9 open 0 0 1 12 12 8".split()).grids(99)
        with torch.no_grad():
            alone, ok1 = m.encoder(*(t.to(DEVICE) for t in state.stack([g])))
            both, ok2 = m.encoder(*(t.to(DEVICE) for t in state.stack([g, big])))
        # Alone: 6 x 5 x 4 padded to 8 x 8 x 4, tokens 2 x 2 x 1; with a 13 x 13 x 9 one: 4 x 4 x 3.
        self.assertEqual(alone.shape[1], 4)
        both = both[0].view(4, 4, 3, -1)[:2, :2, :1].reshape(-1, both.shape[2])
        keep = ok1[0]
        torch.testing.assert_close(alone[0][keep], both[keep], atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
