"""train/milestones.py and model.Plan, the solution-first arm (docs/architecture-plan-2026-11-15.md,
run 5):

- a plan's milestones: the steps that do something, cube kinds, targets and what a press drives,
  repeats written once, no coordinates (so a mirrored level has the same ones);
- the solver labels a level it solves (a checked plan), and none it can't;
- training sequences hold the milestones between two <sep>s, learned with the level; levels
  without milestones are left out; the plain loader is unchanged;
- the constraint: milestone words (a wire target after `>`) until <sep>, then the level grammar;
- a tiny plan model learns levels with their milestones, and its constrained samples parse.

    .venv/Scripts/python -m unittest tests/test_milestones.py -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "tests"))
import data  # noqa: E402
import milestones as M  # noqa: E402
import rated  # noqa: E402
import sample  # noqa: E402
from data import EOS, SEP, Vocab  # noqa: E402
from test_arch import DEVICE, greedy_matches_full, tiny  # noqa: E402
from test_sample_grammar import TANTALOS, editor_rules  # noqa: E402

CUBES = ("size 7 7 4 open 1 1 1 5 5 2 entry 3 1 1 +y exit 4 5 1 -y dropper 2 2 2 sphere "
         "button 2 4 1 sphere > exit pedestal 5 2 1 inf > exit > field 0 fizzler 3 3 1 +x +z 1 on")
CUBES_PLAN = ("walk 2 2 1 pickup 2 2 1 walk 2 3 1 putdown 2 4 1 fire blue 1 1 2 -x enter blue "
              "enter orange press 5 2 1 walk 4 5 1 exit")
SOLVABLE = ("size 7 7 4 open 1 1 1 5 5 2 entry 3 1 1 +y exit 4 5 1 -y dropper 2 2 2 standard "
            "button 2 4 1 weighted > exit")


def plan_vocab():
    return Vocab.build(ROOT / "data" / "v5b" / "vocab.txt", milestones=True)


class Abstract(unittest.TestCase):
    def test_cubes_presses_and_repeats(self):
        got = M.milestones(CUBES.split(), CUBES_PLAN.split())
        self.assertEqual(" ".join(got), "pickup sphere putdown button > exit enter press > exit > field")
        self.assertTrue(all(w in M.WORDS for w in got))

    def test_laser_toss_and_unwired(self):
        level = CUBES.replace("dropper 2 2 2 sphere", "cube 2 2 1 reflection")
        plan = "pickup 2 2 1 putdown 3 3 1 +x pickup 3 3 1 toss 1 1 1 knock 4 4 1 knock 4 4 1"
        self.assertEqual(" ".join(M.milestones(level.split(), plan.split())),
                         "pickup reflection putdown laser pickup reflection toss knock")

    def test_words_match_the_vocabulary(self):
        self.assertEqual(M.WORDS, data.MILESTONE_WORDS)
        v = plan_vocab()
        self.assertTrue(all(w in v.stoi for w in M.WORDS))
        plain = Vocab.build(ROOT / "data" / "v5b" / "vocab.txt")
        self.assertEqual(v.itos[: len(plain)], plain.itos)  # the extra words go last

    @unittest.skipUnless(TANTALOS.exists(), "needs target/release/tantalos-score")
    def test_solver_labels(self):
        got = M.label_levels([SOLVABLE, "size 2 2 2", ""], editor_rules=True, tantalos=TANTALOS)
        self.assertEqual(got[0], "pickup standard putdown button > exit".split())
        self.assertIsNone(got[1])
        self.assertIsNone(got[2])


class Data(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.levels = d / "levels.jsonl"
        rows = [{"tokens": CUBES, "verdict": "solvable", "min_len": 5},
                {"tokens": SOLVABLE, "verdict": "solvable", "min_len": 1},
                {"tokens": SOLVABLE.replace("weighted", "cube"), "verdict": "unsolvable"}]
        self.levels.write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.labels = d / "labels.jsonl"
        labels = [{"key": M.key(rows[0]), "milestones": "pickup sphere putdown button > exit", "min_len": 5, "plan": ""},
                  {"key": M.key(rows[1]), "milestones": "", "min_len": 1, "plan": ""}]
        self.labels.write_text("".join(json.dumps(r) + "\n" for r in labels))
        self.cache = data.LEVEL_CACHE
        data.LEVEL_CACHE = d / "cache"

    def tearDown(self):
        data.LEVEL_CACHE = self.cache
        self.tmp.cleanup()

    def test_sequences_hold_milestones(self):
        v = plan_vocab()
        for _ in range(2):  # the second time from the cache
            train, val, skipped = data.load([self.levels], v, 512, val_share=0.0, milestones=self.labels)
            self.assertEqual((len(train), len(val), skipped), (2, 0, 1))
            ids, learn = train.sequence(0, 0.0)
            seps = [i for i, t in enumerate(ids) if t == SEP]
            self.assertEqual(len(seps), 2)
            self.assertEqual(v.decode(ids[seps[0] + 1 : seps[1]]), "pickup sphere putdown button > exit".split())
            self.assertEqual(v.decode(ids[seps[1] + 1 : -1]), CUBES.split())
            self.assertEqual(learn, [False] * (seps[0] + 1) + [True] * (len(ids) - seps[0] - 1))
            ids, _ = train.sequence(1, 0.0)
            seps = [i for i, t in enumerate(ids) if t == SEP]
            self.assertEqual(seps[1], seps[0] + 1)  # no milestones: <sep> <sep>
        plain, _, skipped = data.load([self.levels], v, 512, val_share=0.0)
        self.assertEqual((len(plain), skipped), (3, 0))
        self.assertEqual(sum(t == SEP for t in plain.sequence(0, 0.0)[0]), 1)

    def test_rated_prompt(self):
        self.assertEqual(rated.prompt(plan_vocab())[0], "<verdict:solvable>")
        self.assertEqual(rated.prompt(Vocab.build(ROOT / "data" / "v5b" / "vocab.txt"))[0], "<verdict:published>")


class FakeGrammar:
    """Every level word allowed, ending anywhere."""

    def __init__(self, v):
        self.v = v

    def next(self, levels):
        return [{"allowed": ["size", "1"], "end": True} for _ in levels]


class Constraint(unittest.TestCase):
    def test_phases(self):
        v = plan_vocab()
        legal = M.Legal(FakeGrammar(v), v)
        gt, exit_ = v.stoi[">"], v.stoi["exit"]
        mask = legal([0, 1, 2, 3], [[], [v.stoi["press"], gt], [v.stoi["enter"]] * M.MAX_WORDS,
                                    [v.stoi["enter"], SEP, v.stoi["size"]]])
        self.assertTrue(mask[0, v.stoi["pickup"]] and mask[0, SEP] and not mask[0, v.stoi["size"]])
        self.assertEqual(set(mask[1].nonzero().flatten().tolist()), {v.stoi[w] for w in M.wire.ORDER})
        self.assertEqual(mask[2].nonzero().flatten().tolist(), [SEP])
        self.assertEqual(set(mask[3].nonzero().flatten().tolist()), {v.stoi["size"], v.stoi["1"], EOS})
        self.assertTrue(mask[1, exit_])
        layout = M.Legal(FakeGrammar(v), v, layout=True)
        self.assertEqual(set(layout([0], [[]])[0].nonzero().flatten().tolist()), {v.stoi["size"], v.stoi["1"], EOS})


class Model(unittest.TestCase):
    def test_learns_levels_with_milestones(self):
        v = plan_vocab()
        p = sample.prompt(v, [], [])
        pairs = [("pickup sphere putdown button > exit enter press > exit > field", CUBES),
                 ("", SOLVABLE)]
        seqs = [p + v.encode(ms.split()) + [SEP] + v.encode(lv.split()) + [EOS] for ms, lv in pairs]
        width = max(map(len, seqs))
        x = torch.zeros((len(seqs), width - 1), dtype=torch.long)
        y = torch.full((len(seqs), width - 1), -100, dtype=torch.long)
        for r, s in enumerate(seqs):
            x[r, : len(s) - 1] = torch.tensor(s[:-1])
            y[r, len(p) - 1 : len(s) - 1] = torch.tensor(s[len(p):])
        x, y = x.to(DEVICE), y.to(DEVICE)
        m = tiny("plan", v, block=256)  # room for 64 milestone words and the level
        m.train()
        opt = m.optimizer(3e-3, 0.0)
        for _ in range(300):
            _, loss = m(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        self.assertLess(loss.item(), 0.3)
        # Prompted with the first milestone, it plans and lays out the first level.
        first = p + [v.stoi["pickup"]]
        greedy_matches_full(self, m, [first], 80)
        out = m.generate([first], 80, EOS, top_k=1)[0]
        toks, err = m.compile_ids([v.stoi["pickup"]] + out)
        self.assertIsNone(err)
        self.assertEqual(" ".join(toks), CUBES)
        self.assertEqual(" ".join(m.last_milestones), pairs[0][0])
        self.assertEqual(m.compile_ids([v.stoi["enter"]])[1], "no layout after the milestones")
        if editor_rules(TANTALOS):
            # CUBES wires its pedestal to `field 0` before the fizzler, which the grammar doesn't
            # offer there; it offers `> dropper 0 >` instead, and then nothing (a grammar dead
            # end), so a sample of the first level can end unparsed. The rest must parse.
            lines = sample.generate(m, v, 6, temperature=0.5, batch=3, constrained=True, tantalos=TANTALOS)
            results = sample.score(lines, TANTALOS)
            bad = [l for l, r in zip(lines, results) if r["verdict"] == "unparsed"]
            self.assertTrue(all(l.endswith(">") for l in bad), bad)
            self.assertLessEqual(len(bad), 3, lines)


if __name__ == "__main__":
    unittest.main()
