"""Flat-token architectures in train/model.py ARCHS (architecture plan, step 0):

- incremental decoding with the key/value cache gives the logits of one full forward pass, in
  every way the sampling code feeds it (one prompt then a token at a time; padded rows;
  generate_many's per-row positions);
- generate, generate_many and the refill sampler agree when greedy;
- a checkpoint saves and reloads through sample.load_checkpoint unchanged, and an old one
  (no `arch` in its config) loads as a GPT;
- a tiny model overfits a few levels, and its samples, written through `tantalos-score grammar`
  (constrained, as evaluation does), parse in `tantalos-score score`. Skipped without a
  tantalos-score that has the editor's rules.

Wire's structured representation has its own equivalent tests in test_wire.py and
the 32-level readiness smoke in train/wire_readiness.py.

    .venv/Scripts/python -m unittest tests/test_arch.py -v
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "tests"))
import sample  # noqa: E402
from data import EOS, Vocab  # noqa: E402
from model import ARCHS as ALL_ARCHS, Config, build_model  # noqa: E402

# These fixtures use flat level tokens. The wire representation has equivalent cache,
# checkpoint, refill, overfit and scorer checks in test_wire.py and wire_readiness.py.
ARCHS = {k: v for k, v in ALL_ARCHS.items() if k != 'wire'}
from test_sample_grammar import LEVELS, TANTALOS, editor_rules  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def vocab():
    return Vocab.build(ROOT / "data" / "v5b" / "vocab.txt")


def tiny(arch, v, block=96, layers=2):
    torch.manual_seed(0)
    itos = tuple(v.itos) if arch == "spatial" else ()
    return build_model(Config(len(v), block, layers, 2, 32, 0.0, qk_norm=True, arch=arch, itos=itos)).to(DEVICE).eval()


def part(m, ids, a, b):
    """Input positions a..b of `ids` as a model takes them after a cached prefix: with their
    features (from the whole sequence) for a model that has them."""
    if m.featurize is None:
        return ids[:, a:b]
    return ids[:, a:b], m.featurize(ids.cpu()).to(ids.device)[:, a:b]


def sequences(v):
    """The levels as training sequences: the unconditioned prompt, the level, EOS."""
    p = sample.prompt(v, [], [])
    return p, [p + v.encode(level.split()) + [EOS] for level in LEVELS]


def greedy_matches_full(test, m, prompts, n):
    """Greedy samples of `m` from `prompts` (each sampler) are each token's argmax in one full
    forward pass over the finished sequence."""
    for name, outs in [("generate", m.generate(prompts, n, EOS, top_k=1)),
                       ("many", m.generate_many(prompts, n, EOS, top_k=1, batch=2)),
                       ("refill", m.generate_refill(prompts[0], 2, n, EOS, top_k=1, batch=2))]:
        for p, out in zip(prompts if name != "refill" else [prompts[0]] * 2, outs):
            ids = torch.tensor([p + out], device=DEVICE)
            # In bfloat16 as the samplers run (an untrained net's logits nearly tie).
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                full = m(ids)[0][0, len(p) - 1 : -1].float().argmax(-1).tolist()
            test.assertEqual(full, out, name)


class Consistency(unittest.TestCase):
    def test_cache_matches_full_forward(self):
        v = vocab()
        for arch in ARCHS:
            with self.subTest(arch=arch):
                m = tiny(arch, v)
                _, seqs = sequences(v)
                ids = torch.tensor([seqs[0]], device=DEVICE)
                with torch.no_grad():
                    full, _ = m(ids)
                    n = 7
                    caches = [{"size": ids.shape[1]} for _ in m.blocks]
                    parts = [m(ids[:, :n], caches=caches)[0]]
                    for i in range(n, ids.shape[1]):
                        parts.append(m(part(m, ids, i, i + 1), caches=caches, start=i)[0])
                torch.testing.assert_close(torch.cat(parts, dim=1), full, atol=1e-4, rtol=1e-4)

    def test_padded_rows_match_alone(self):
        """generate's padded path (prompts of different lengths) writes what each would alone."""
        v = vocab()
        _, seqs = sequences(v)
        prompts = [seqs[0][:9], seqs[1][:14], seqs[2][:11]]
        for arch in ARCHS:
            with self.subTest(arch=arch):
                m = tiny(arch, v)
                together = m.generate(prompts, 20, EOS, top_k=1)
                alone = [m.generate([p], 20, EOS, top_k=1)[0] for p in prompts]
                self.assertEqual(together, alone)

    def test_samplers_agree_when_greedy(self):
        v = vocab()
        p, _ = sequences(v)
        for arch in ARCHS:
            with self.subTest(arch=arch):
                m = tiny(arch, v)
                one = m.generate([p], 40, EOS, top_k=1)[0]
                many = m.generate_many([p] * 3, 40, EOS, top_k=1, batch=2)
                refill = m.generate_refill(p, 3, 40, EOS, top_k=1, batch=2)
                self.assertEqual(many, [one] * 3)
                self.assertEqual(refill, [one] * 3)

    def test_greedy_samples_match_full_forward(self):
        """Each token a greedy sampler writes is the argmax of one full forward pass over the
        finished sequence: whatever a step fed the net (a spatial model's features from its
        row's parser) is what the whole sequence gives."""
        v = vocab()
        _, seqs = sequences(v)
        prompts = [seqs[0][:9], seqs[1][:14], seqs[2][:9]]
        for arch in ARCHS:
            with self.subTest(arch=arch):
                greedy_matches_full(self, tiny(arch, v), prompts, 30)


class Checkpoints(unittest.TestCase):
    def test_save_and_reload(self):
        v = vocab()
        _, seqs = sequences(v)
        ids = torch.tensor([seqs[1]], device=DEVICE)
        for arch in ARCHS:
            with self.subTest(arch=arch), tempfile.TemporaryDirectory() as d:
                m = tiny(arch, v)
                path = Path(d) / "best.pt"
                torch.save({"model": m.state_dict(), "model_config": m.cfg.__dict__, "vocab": v.itos}, path)
                again, v2, _ = sample.load_checkpoint(path, DEVICE)
                self.assertIs(type(again), ARCHS[arch])
                self.assertEqual(v2.itos, v.itos)
                with torch.no_grad():
                    torch.testing.assert_close(again(ids)[0], m(ids)[0])

    def test_old_config_is_gpt(self):
        v = vocab()
        cfg = {k: val for k, val in tiny("gpt", v).cfg.__dict__.items() if k != "arch"}
        self.assertIs(type(build_model(Config(**cfg))), ARCHS["gpt"])

    def test_unknown_arch(self):
        with self.assertRaises(ValueError):
            build_model(Config(10, arch="nope"))


@unittest.skipUnless(editor_rules(TANTALOS), f"no tantalos-score with editor rules at {TANTALOS}")
class Overfit(unittest.TestCase):
    def test_learns_levels_and_samples_parse(self):
        v = vocab()
        p, seqs = sequences(v)
        width = max(map(len, seqs))
        x = torch.zeros((len(seqs), width - 1), dtype=torch.long)
        y = torch.full((len(seqs), width - 1), -100, dtype=torch.long)
        for r, s in enumerate(seqs):
            x[r, : len(s) - 1] = torch.tensor(s[:-1])
            y[r, len(p) - 1 : len(s) - 1] = torch.tensor(s[len(p) :])
        x, y = x.to(DEVICE), y.to(DEVICE)
        for arch in ARCHS:
            with self.subTest(arch=arch):
                m = tiny(arch, v)
                m.train()
                opt = m.optimizer(3e-3, 0.0)
                for _ in range(400):
                    _, loss = m(x, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                m.eval()
                # Real levels now: every role and cell the parser gives comes up.
                greedy_matches_full(self, m, [seqs[0][: len(p) + 40], seqs[1][: len(p) + 3], p], 200)
                # The levels share one prompt, so which one comes is a guess (log 3 over ~20
                # tokens: ~0.06 at best); untrained, the loss is ~5.
                self.assertLess(loss.item(), 0.3, f"{arch} didn't learn {len(seqs)} levels")
                lines = sample.generate(m, v, 6, temperature=0.5, batch=3, constrained=True, tantalos=TANTALOS)
                results = sample.score(lines, TANTALOS)
                self.assertTrue(all(r["verdict"] != "unparsed" for r in results), lines)


if __name__ == "__main__":
    unittest.main()
