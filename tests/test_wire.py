"""train/wire.py and model.Wire, the device-and-wire arm (docs/architecture-plan-2026-11-15.md,
run 4):

- flat levels to wire form and back, exactly, with every target kind's counting (fields across
  fizzlers and laser fields, gels splat or poured, funnels and their polarity); `compile` refuses
  wires it can't write;
- a training sequence's flags follow its tokens into wire form;
- the tracker gives the same features token by token as for the whole sequence;
- the cache and the samplers' per-row pointer keys give one full pass's logits, the prompt
  holding devices or not; pointers only go to devices of the right class written so far;
- a tiny model learns wired levels, and greedy samples compile and match a full pass.

    .venv/Scripts/python -m unittest tests/test_wire.py -v

The readiness check on the training data: `.venv/Scripts/python train/wire.py check`.
"""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "tests"))
import sample  # noqa: E402
import wire  # noqa: E402
from data import EOS, Vocab  # noqa: E402
from test_arch import DEVICE, greedy_matches_full, tiny  # noqa: E402
from test_sample_grammar import TANTALOS, editor_rules  # noqa: E402

KINDS = ("size 8 8 4 open 1 1 1 6 6 2 entry 1 2 1 +x exit 6 6 1 -x fizzler 3 2 1 +x +y 2 on "
         "laserfield 4 2 1 +x +y 2 on gel 2 3 1 +z bounce splat gel 3 3 1 +z speed stream off "
         "button 2 2 1 cube > exit > field 1 > gel 1 > funnel 0 > polarity 0 pedestal 5 5 1 inf > field 0 "
         "funnel 4 4 1 +x on")
WIRED = [
    "size 1 4 8 open 0 2 7 0 3 7 black -x black -z entry 0 2 7 +y exit 0 3 7 -y laserfield 0 3 7 +x +z 1 on "
    "cube 0 2 7 reflection catcher 0 2 7 +x center > field 0",
    "size 2 3 1 open 0 0 0 1 2 0 black +x white -x 0 1 0 1 2 0 black +y entry 1 2 0 -x exit 0 2 0 -y cube 1 1 0 "
    "reflection laser 0 1 0 +x center on relay 0 2 0 +z center > exit relay 1 2 0 +z center > exit",
    "size 7 7 4 open 1 1 1 5 5 2 entry 3 1 1 +y exit 4 5 1 -y button 2 2 1 weighted > exit",
]


def vocab():
    return Vocab.build(ROOT / "data" / "v5b" / "vocab.txt")


class Form(unittest.TestCase):
    def test_round_trip(self):
        for level in [KINDS, *WIRED]:
            toks = level.split()
            self.assertEqual(wire.compile(wire.to_wire(toks)), toks)

    def test_targets_by_device(self):
        w = " ".join(wire.to_wire(KINDS.split()))
        # Devices: entry 0, exit 1, fizzler 2, laserfield 3, gel 4 (splat), gel 5, button 6,
        # pedestal 7, funnel 8.
        self.assertTrue(w.endswith("> @6 @1 > @6 @3 > @6 @5 > @6 @8 funnel > @6 @8 polarity > @7 @2"), w)
        self.assertNotIn(">", w[: w.index("> @6")])

    def test_compile_sorts_and_refuses(self):
        toks = wire.to_wire(KINDS.split())
        cut = toks.index(">")
        items = toks[:cut]
        # Any order, repeats: written sorted, once.
        shuffled = items + "> @7 @2 > @6 @8 polarity > @6 @1 > @6 @1 > @6 @3 > @6 @5 > @6 @8 funnel".split()
        self.assertEqual(wire.compile(shuffled), KINDS.split())
        for bad in ["> @1 @2", "> @6 @4", "> @6 @0", "> @6 @8", "> @6 @9", "> @6", "> @6 @1 @2"]:
            with self.subTest(bad=bad), self.assertRaises(wire.WireError):
                wire.compile(items + bad.split())

    def test_recode_flags(self):
        v = vocab()
        p = sample.prompt(v, [], [])
        ids = p + v.encode(WIRED[0].split()) + [EOS]
        flags = [False] * len(p) + list(range(1, len(ids) - len(p) + 1))
        new, nf = wire.recode(ids, flags, v.itos, len(v), 1000)
        self.assertEqual(new[: len(p)], p)
        self.assertEqual(new[-1], EOS)
        self.assertEqual(nf[-1], flags[-1])
        cut = new.index(v.stoi[">"])
        # Item tokens keep theirs; the wire takes its first flat token's.
        for j in range(len(p), cut):
            self.assertEqual(v.itos[new[j]], WIRED[0].split()[nf[j] - 1])
        first = WIRED[0].split().index(">") + 1
        self.assertEqual(set(nf[cut:-1]), {first})


class Tracking(unittest.TestCase):
    def test_incremental_matches_whole(self):
        v = vocab()
        p = sample.prompt(v, [], [])
        for level in [KINDS, *WIRED]:
            ids = wire.recode(p + v.encode(level.split()) + [EOS], [True] * (len(p) + len(level.split()) + 1),
                              v.itos, len(v), 1000)[0]
            whole = wire.features([ids], v.itos, len(v))[0].tolist()
            rows = wire.Rows(v.itos, len(v), 1)
            rows.start(0, ids[:5])
            got = whole[:5] + [list(rows.feed([i])[0][:4]) for i in ids[5:]]
            self.assertEqual(got, whole)
            # Pointers follow `>` (a sender) and the sender (a receiver).
            for k, i in enumerate(ids[:-1]):
                if v.itos[i] == ">" if i < len(v) else False:
                    self.assertEqual(whole[k][1], wire.SENDER)
                    self.assertEqual(whole[k + 1][1], wire.RECEIVER)


def wired_seqs(v):
    p = sample.prompt(v, [], [])
    out = []
    for level in WIRED:
        s = p + v.encode(level.split()) + [EOS]
        out.append(wire.recode(s, [True] * len(s), v.itos, len(v), 1000)[0])
    return p, out


class Model(unittest.TestCase):
    def test_cache_and_rows_match_full(self):
        v = vocab()
        m = tiny("wire", v, block=128)
        _, seqs = wired_seqs(v)
        for seq in seqs:
            ids = torch.tensor([seq], device=DEVICE)
            with torch.no_grad():
                full, _ = m(ids)
                for n in (7, len(seq) - 8):  # a prompt without devices, and one with all of them
                    with self.subTest(n=n):
                        caches = [{"size": len(seq)} for _ in m.blocks]
                        parts = [m(ids[:, :n], caches=caches)[0]]
                        rows = m._rows(1)
                        rows.start(0, seq[:n])
                        for i in range(n, len(seq)):
                            parts.append(m(m._step(ids[:, i : i + 1], rows), caches=caches, start=i)[0])
                        torch.testing.assert_close(torch.cat(parts, dim=1), full, atol=1e-4, rtol=1e-4)

    def test_pointers_only_to_written_devices_of_the_class(self):
        v = vocab()
        m = tiny("wire", v, block=128)
        _, seqs = wired_seqs(v)
        seq = seqs[1]  # cube 2, laser 3, relay 4, relay 5 (entry 0, exit 1)
        with torch.no_grad():
            logits = m(torch.tensor([seq], device=DEVICE))[0][0]
        gt = v.stoi[">"]
        at = seq.index(gt)
        ok = torch.isfinite(logits[at, len(v):]).nonzero().flatten().tolist()
        self.assertEqual(ok, [4, 5])  # senders: the relays
        ok = torch.isfinite(logits[at + 1, len(v):]).nonzero().flatten().tolist()
        self.assertEqual(ok, [1, 3])  # receivers: the exit, the laser
        self.assertFalse(torch.isfinite(logits[at - 1, len(v):]).any())
        self.assertEqual(int(torch.isfinite(logits[at, : len(v)]).sum()), 1)  # EOS only

    def test_learns_wired_levels(self):
        v = vocab()
        p, seqs = wired_seqs(v)
        width = max(map(len, seqs))
        x = torch.zeros((len(seqs), width - 1), dtype=torch.long)
        y = torch.full((len(seqs), width - 1), -100, dtype=torch.long)
        for r, s in enumerate(seqs):
            x[r, : len(s) - 1] = torch.tensor(s[:-1])
            y[r, len(p) - 1 : len(s) - 1] = torch.tensor(s[len(p):])
        x, y = x.to(DEVICE), y.to(DEVICE)
        m = tiny("wire", v, block=128)
        m.train()
        opt = m.optimizer(3e-3, 0.0)
        for _ in range(300):
            _, loss = m(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        self.assertLess(loss.item(), 0.3)
        gt = v.stoi[">"]
        prompts = [s[: s.index(gt) + 1] for s in seqs]  # up to the first `>`: pointers next
        greedy_matches_full(self, m, prompts, 40)
        for s, pr in zip(seqs, prompts):
            out = m.generate([pr], 40, EOS, top_k=1)[0]
            toks, err = m.compile_ids(pr[len(p):] + out)
            self.assertIsNone(err)
            self.assertEqual(" ".join(toks), WIRED[seqs.index(s)])
        if editor_rules(TANTALOS):
            # Constrained (wire.Legal): every level compiles and parses.
            lines = sample.generate(m, v, 9, temperature=1.0, batch=3, constrained=True, tantalos=TANTALOS)
            self.assertFalse([x for x in lines if "<wire-error>" in x], lines)
            results = sample.score(lines, TANTALOS)
            self.assertTrue(all(r["verdict"] != "unparsed" for r in results), lines)
            self.assertTrue(any(">" in x for x in lines), lines)


if __name__ == "__main__":
    unittest.main()
