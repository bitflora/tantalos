"""train/spatial.py, the spatial arm's per-token roles and cells
(docs/architecture-plan-2026-11-15.md, run 2):

- at every prefix of real levels, the tokens `tantalos-score grammar` allows next are among
  those the parser expects, and the token that comes is one the parser places (not `junk`);
- roles and cells of a hand-checked level;
- incremental parsing (copies, `Rows`) gives the features of parsing the whole sequence.

    .venv/Scripts/python -m unittest tests/test_spatial.py -v

The readiness check runs it on 10,000 levels (a few minutes per thousand):

    .venv/Scripts/python tests/test_spatial.py --n 10000
"""
import argparse
import json
import random
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
import spatial  # noqa: E402
from data import BOS, EOS, SEP, Vocab  # noqa: E402
from spatial import NONE, ROLE, ROLES, Parser, Rows  # noqa: E402

TANTALOS = ROOT / "target" / "release" / "tantalos-score.exe"
DATA = [ROOT / "data" / "ws5" / "test.jsonl", ROOT / "data" / "v5b" / "levels-1.jsonl"]


def levels(n, seed=0):
    """n levels' tokens, drawn from the Workshop test set and generated data alike."""
    pool = []
    for path in DATA:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 20_000:
                    break
                pool.append(json.loads(line)["tokens"])
    random.Random(seed).shuffle(pool)
    return pool[:n]


def disagreements(lines, tantalos=TANTALOS):
    """Per level, the prefixes where the grammar allows a token the parser doesn't expect, or
    where the parser can't place the token that comes: (level index, position, what)."""
    proc = subprocess.Popen([str(tantalos), "grammar", "--editor-rules", "off"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)
    out = []
    try:
        for li, line in enumerate(lines):
            toks = line.split()
            prefixes = [" ".join(toks[:k]) for k in range(len(toks) + 1)]
            proc.stdin.write(json.dumps({"levels": prefixes}) + "\n")
            proc.stdin.flush()
            results = json.loads(proc.stdout.readline())["results"]
            p = Parser()
            p.phase = "level"
            for k, r in enumerate(results):
                if "error" in r:
                    out.append((li, k, f"grammar: {r['error']}"))
                    break
                extra = set(r["allowed"]) - p.expect()
                if extra:
                    out.append((li, k, f"grammar allows {sorted(extra)[:6]} after ...{' '.join(toks[max(0, k - 8):k])}"))
                if k < len(toks):
                    role = p.feed(toks[k])[0]
                    if role == ROLE["junk"]:
                        out.append((li, k, f"junk: {toks[k]!r} after ...{' '.join(toks[max(0, k - 8):k])}"))
    finally:
        proc.stdin.close()
        proc.wait()
        proc.stdout.close()
    return out


def parse(line):
    p = Parser()
    p.phase = "level"
    return [(t, ROLES[f[0]], f[1:]) for t, f in zip(line.split(), map(p.feed, line.split()))]


class Grammar(unittest.TestCase):
    @unittest.skipUnless(TANTALOS.exists(), f"no {TANTALOS}")
    def test_agrees_with_grammar(self):
        bad = disagreements(levels(40))
        self.assertEqual(bad[:10], [])


class Roles(unittest.TestCase):
    def test_hand_checked(self):
        line = ("size 8 8 4 open 1 1 1 6 6 2 white +z 1 1 0 6 6 0 entry 1 2 1 +x exit 6 6 1 -x inv "
                "fizzler 3 2 1 +x +y 2 on button 2 2 1 cube > exit > field 0 > panel 0 "
                "panel 4 4 1 +z +y 3 off")
        got = parse(line)
        roles = {i: r for i, (_, r, _) in enumerate(got)}
        toks = line.split()
        at = lambda word, k=0: [i for i, t in enumerate(toks) if t == word][k]  # noqa: E731
        self.assertEqual([roles[i] for i in range(4)], ["size", "size.x", "size.y", "size.z"])
        self.assertEqual(got[at("open") + 5], ("6", "open.y1", (6, 6, NONE)))
        self.assertEqual(got[at("white") + 1][1], "white.dir")
        self.assertEqual(got[at("entry") + 3], ("1", "entry.z", (1, 2, 1)))
        self.assertEqual(got[at("entry") + 4], ("+x", "entry.D", (1, 2, 1)))
        self.assertEqual(got[at("inv")][1], "exit.inv")
        self.assertEqual(got[at("fizzler")][1], "fizzler")
        self.assertEqual(got[at("fizzler") + 6][1:], ("fizzler.L", (3, 2, 1)))
        b = at("button")
        self.assertEqual(got[b + 4][1], "button.K")
        self.assertEqual(got[b + 5][1], "wire")
        # `> exit`: the exit's cell; `> field 0`: the fizzler's; `> panel 0`: written later, none.
        self.assertEqual(got[b + 6], ("exit", "wire.kind", (6, 6, 1)))
        self.assertEqual(got[b + 9], ("0", "wire.index", (3, 2, 1)))
        self.assertEqual(got[b + 12], ("0", "wire.index", (NONE, NONE, NONE)))
        self.assertEqual(got[at("panel", 1) + 7][1], "panel.state")
        self.assertNotIn("junk", [r for _, r, _ in got])

    def test_junk_recovers(self):
        got = parse("size 8 8 4 open 1 1 1 6 6 2 entry 1 +x 2 exit 6 6 1 -x")
        self.assertIn("junk", [r for _, r, _ in got])
        self.assertEqual(got[-5][1], "exit")
        self.assertEqual(got[-1][1:], ("exit.D", (6, 6, 1)))


class Incremental(unittest.TestCase):
    def test_rows_match_whole_sequences(self):
        v = Vocab.build(ROOT / "data" / "v5b" / "vocab.txt")
        lines = levels(20, seed=1)
        seqs = [[BOS, SEP] + v.encode(line.split()) + [EOS] for line in lines]
        whole = spatial.features(seqs, v.itos)
        rows = Rows(v.itos, len(seqs))
        cut = 7
        got = []
        for r, s in enumerate(seqs):
            p, feats = rows.prompt(s[:cut])
            rows.rows[r] = p
            got.append(list(feats))
        for k in range(cut, max(map(len, seqs))):
            live = [r for r, s in enumerate(seqs) if k < len(s)]
            toks = [s[k] if k < len(s) else 0 for s in seqs]
            for r, f in enumerate(rows.feed(toks, set(live))):
                if r in live:
                    got[r].append(f)
        self.assertEqual(got, whole)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000)
    a = ap.parse_args()
    lines = levels(a.n)
    bad = disagreements(lines)
    print(f"{len(lines)} levels, {sum(len(x.split()) for x in lines)} tokens: "
          f"{len({b[0] for b in bad})} levels disagree ({len(bad)} prefixes)")
    for b in bad[:30]:
        print("  ", b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].startswith("--n"):
        main()
    else:
        unittest.main()
