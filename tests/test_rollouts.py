"""Rollout-label regressions: .venv/Scripts/python -m unittest discover -s tests -p test_rollouts.py

The value-label collector and probe (train/goblin_rollouts.py), with stand-ins for the net, the
env and `glados-score check` where they are needed: which levels it may draw, how prefixes are
labelled, the AUCs, the beam-ranking comparison, and that a prefix's hidden state is the one the
net has after reading it. Nothing here runs the solver.
"""

import json
from argparse import Namespace
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import goblin_rollouts  # noqa: E402
from model import GPT, Config  # noqa: E402

WORDS = ["size", "1", "2", "3", "walk", "press", "fire", "blue", "exit"]
FIRST = 10


class Vocab:
    def __init__(self, words=WORDS):
        self.words = list(words)
        self.stoi = {w: i + FIRST for i, w in enumerate(self.words)}

    def __len__(self):
        return len(self.words) + FIRST

    def encode(self, words):
        return [self.stoi[w] for w in words]

    def decode(self, ids):
        return [self.words[i - FIRST] for i in ids]


class Checked:
    """Stands in for goblin_eval.check: `solves` are the plans it calls solved."""

    def __init__(self, solves=()):
        self.solves, self.seen = set(solves), []

    def __call__(self, items, glados, mode):
        plans = [p for it in items for p in it["plans"]]
        self.seen += plans
        return [{"verdict": "solved" if p in self.solves else "unsolved"} for p in plans]


class PrefixTests(unittest.TestCase):
    def test_a_plans_prefixes_end_at_its_steps_and_leave_out_the_whole_plan(self):
        self.assertEqual(goblin_rollouts.prefixes_of("fire blue 1 2 3 walk 2 exit"),
                         ["", "fire blue 1 2 3", "fire blue 1 2 3 walk 2"])
        self.assertEqual(goblin_rollouts.prefixes_of(""), [])

    def test_a_prefix_counts_every_sample_through_it(self):
        level = {"identity": "ws:1", "split": "probe-train"}
        plans = ["walk 1 press 2 exit", "walk 1 press 3", "walk 2 exit", "walk 1 press 2 exit"]
        solved = {"walk 1 press 2 exit": True, "walk 1 press 3": False, "walk 2 exit": True}
        rows = {r["prefix"]: r for r in goblin_rollouts.prefix_rows(level, plans, solved)}
        self.assertEqual((rows[""]["n"], rows[""]["solved"]), (4, 3))
        self.assertEqual((rows["walk 1"]["n"], rows["walk 1"]["solved"]), (3, 2))
        self.assertEqual((rows["walk 1 press 2"]["n"], rows["walk 1 press 2"]["solved"]), (2, 2))
        self.assertEqual(rows["walk 1 press 2"]["steps"], 2)
        self.assertEqual(rows["walk 2"]["solved"], 1)
        self.assertNotIn("walk 1 press 3", rows)  # a whole plan is its own verdict, not a prefix
        self.assertEqual(len(rows), 4)

    def test_a_check_runs_each_distinct_plan_once_and_an_empty_one_fails_unchecked(self):
        real = goblin_rollouts.goblin_eval.check
        goblin_rollouts.goblin_eval.check = c = Checked(["walk 1 exit"])
        try:
            got = goblin_rollouts.checked({"tokens": "size 1 2 3"}, ["walk 1 exit", "", "walk 1 exit", "walk 2"],
                                          "none")
        finally:
            goblin_rollouts.goblin_eval.check = real
        self.assertEqual(got, {"": False, "walk 1 exit": True, "walk 2": False})
        self.assertEqual(sorted(c.seen), ["walk 1 exit", "walk 2"])


class PoolTests(unittest.TestCase):
    """Only Goblin train-side identities may be drawn."""

    def setUp(self):
        self.real = goblin_rollouts.school.test_identities
        self.dir = tempfile.TemporaryDirectory()
        root = Path(self.dir.name)
        (root / "runs/school/e1").mkdir(parents=True)
        (root / "data/goblin").mkdir(parents=True)
        school = [{"tokens": "size 1 1 1", "identity": "ws:1", "tries": 1, "solved_rounds": [1]},
                  {"tokens": "size 2 2 2", "identity": "ws:2", "tries": 1, "solved_rounds": []},
                  {"tokens": "size 3 3 3", "identity": "ws:9", "tries": 1, "solved_rounds": []}]
        train = [{"tokens": "size 1 2 3", "identity": "ws:3", "src": "ws"},
                 {"tokens": "size 3 2 1", "identity": "ws:3", "src": "ws"},  # another version
                 {"tokens": "size 2 1 3", "identity": "tok:4", "src": "v6"},
                 {"tokens": "size 2 3 1", "identity": "tok:5", "src": "gen"},
                 {"tokens": "size 1 3 2", "identity": "ws:8", "src": "ws"}]
        for path, rows in ((root / goblin_rollouts.SCHOOL, school), (root / goblin_rollouts.TRAIN, train)):
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.root = root

    def tearDown(self):
        goblin_rollouts.school.test_identities = self.real
        self.dir.cleanup()

    def test_held_out_identities_are_never_in_the_pool(self):
        goblin_rollouts.school.test_identities = lambda: {"ws:8", "ws:9"}
        got = goblin_rollouts.pool(self.root)
        self.assertEqual(got["school-solved"], {"ws:1": "size 1 1 1"})
        self.assertEqual(got["school-unsolved"], {"ws:2": "size 2 2 2"})
        self.assertEqual(got["train-ws"], {"ws:3": "size 1 2 3"})  # one row an identity
        self.assertEqual(got["train-v6"], {"tok:4": "size 2 1 3"})

    def test_no_guard_no_draw(self):
        goblin_rollouts.school.test_identities = lambda: set()
        with self.assertRaises(SystemExit):
            goblin_rollouts.pool(self.root)

    def test_a_draw_takes_a_quarter_from_each_stratum_and_keeps_the_split(self):
        goblin_rollouts.school.test_identities = lambda: {"ws:8", "ws:9"}
        levels = goblin_rollouts.pool(self.root)
        import random
        got = goblin_rollouts.draw(levels, 4, random.Random(0), 512, Vocab())
        self.assertEqual([lv["stratum"] for lv in got], list(goblin_rollouts.STRATA))
        for lv in got:
            self.assertEqual(lv["split"], goblin_rollouts.split_of(lv["identity"]))

    def test_the_split_is_fixed_and_about_a_quarter(self):
        ids = [f"tok:{i}" for i in range(4000)]
        test = sum(goblin_rollouts.split_of(i) == "probe-test" for i in ids)
        self.assertTrue(800 < test < 1200, test)
        self.assertEqual([goblin_rollouts.split_of(i) for i in ids[:50]],
                         [goblin_rollouts.split_of(i) for i in ids[:50]])


class AucTests(unittest.TestCase):
    def test_auc(self):
        auc = goblin_rollouts.auc
        self.assertEqual(auc([1, 2, 3], [0, 0, 1], [1, 1, 0]), 1.0)
        self.assertEqual(auc([3, 2, 1], [0, 0, 1], [1, 1, 0]), 0.0)
        self.assertEqual(auc([1, 1], [1, 0], [0, 1]), 0.5)  # a tie counts half
        # One item with 3 of 4 rollouts solved above one with 1 of 4: 3*3 of 4*4 pairs right,
        # plus the ties within each item at half.
        self.assertAlmostEqual(auc([2, 1], [3, 1], [1, 3]), (9 + 0.5 * (3 * 1 + 1 * 3)) / 16)
        self.assertIsNone(auc([1, 2], [1, 1], [0, 0]))

    def test_within_compares_only_inside_a_group(self):
        rows = [{"identity": "a", "steps": 0, "n": 1, "solved": 1, "s": 10},
                {"identity": "a", "steps": 0, "n": 1, "solved": 0, "s": 9},
                {"identity": "b", "steps": 0, "n": 1, "solved": 1, "s": 1},
                {"identity": "b", "steps": 0, "n": 1, "solved": 0, "s": 0},
                {"identity": "c", "steps": 0, "n": 2, "solved": 2, "s": 5}]  # one outcome: left out
        # Across levels the scores are useless (a's failure beats b's success); within, perfect.
        self.assertLess(goblin_rollouts.auc([r["s"] for r in rows], [r["solved"] for r in rows],
                                            [r["n"] - r["solved"] for r in rows]), 1.0)
        self.assertEqual(goblin_rollouts.within(rows, lambda r: r["s"]), (1.0, 2))

    def test_within_weights_a_group_by_its_pairs(self):
        rows = [{"identity": "a", "n": 4, "solved": 2, "s": 1},  # a: 4 pairs, all ties
                {"identity": "b", "n": 1, "solved": 1, "s": 1},
                {"identity": "b", "n": 1, "solved": 0, "s": 0}]  # b: 1 pair, right
        self.assertAlmostEqual(goblin_rollouts.within(rows, lambda r: r["s"])[0], (0.5 * 4 + 1) / 5)


class FitTests(unittest.TestCase):
    def test_a_head_learns_a_separable_label(self):
        torch.manual_seed(0)
        x = torch.randn(200, 8)
        won = (x[:, 0] > 0).float() * 3
        n = torch.full((200,), 3.0)
        for hidden in (0, 16):
            head = goblin_rollouts.fit(x, n, won, hidden, epochs=200)
            with torch.no_grad():
                s = head(x).tolist()
            self.assertGreater(goblin_rollouts.auc(s, won.tolist(), (n - won).tolist()), 0.98)


class RankTests(unittest.TestCase):
    def kid(self, parent, child, logprob, value, identity="ws:1"):
        return {"identity": identity, "parent": parent, "child": child, "logprob": logprob, "value": value}

    def test_the_beams_pick_against_the_heads(self):
        kids = [self.kid("", "walk 1", -0.1, 0.0), self.kid("", "walk 2", -2.0, 0.5),
                self.kid("", "walk 3", -3.0, 0.25),
                self.kid("walk 1", "walk 1 press 1", -0.5, 0.0),  # values all equal: not counted
                self.kid("walk 1", "walk 1 press 2", -0.7, 0.0),
                self.kid("walk 2", "walk 2 press 1", -0.5, 1.0)]  # one child: not counted
        head = {"walk 1": 0.0, "walk 2": 0.9, "walk 3": 0.5}
        got = goblin_rollouts.rank_check(kids, lambda k: head.get(k["child"], 0.0))
        self.assertEqual(got["parents"], 1)
        self.assertEqual((got["head better"], got["beam better"]), (1, 0))
        self.assertEqual(got["mean value, beam pick"], 0.0)
        self.assertEqual(got["mean value, head pick"], 0.5)
        self.assertEqual(got["mean value, best pick"], 0.5)
        self.assertEqual(got["mean value, random pick"], 0.25)
        # Pairs with different values: (1,2), (1,3), (2,3); the head orders all three right,
        # log probability only (2,3).
        self.assertEqual((got["pairs"], got["head right"], got["beam right"]), (3, 3, 1))
        self.assertEqual(got["parent groups"], 3)


class ChildrenTests(unittest.TestCase):
    """The beam's expansion of a parent: distinct children, their scores and their rollouts."""

    def test_children_are_distinct_and_rolled_out(self):
        vocab = Vocab()
        calls = []

        class Net:
            def generate_many(self, prompts, max_new, stop, temperature, top_k, batch, constrain,
                              logprob=False):
                calls.append(len(prompts))
                if logprob:  # the candidate steps
                    steps = ["walk 1", "walk 1", "walk 2", ""]
                    return [vocab.encode(s.split()) for s in steps], [-0.2, -0.2, -1.5, 0.0]
                start = " ".join(constrain.starts[0])
                return [vocab.encode(("exit" if start.endswith("2") else "press 3").split())] * len(prompts)

        class Env:
            let_stop = False

            def open(self, key, tokens):
                return True

            def next(self, keys, plans):
                return [{"allowed": ["walk", "press", "exit"]} for _ in plans]

            def close(self, keys):
                pass

        real = goblin_rollouts.goblin_eval.check
        goblin_rollouts.goblin_eval.check = Checked(["walk 2 exit"])
        try:
            a = Namespace(candidates=4, m=3, rows=8, temperature=1.0, max_plan=256, glados="none")
            kids = goblin_rollouts.children(Net(), vocab, Env(), {"identity": "ws:1", "tokens": "size 1 2 3"}, "", a)
        finally:
            goblin_rollouts.goblin_eval.check = real
        got = {k["child"]: (k["logprob"], k["drawn"], k["m"], k["value"]) for k in kids}
        self.assertEqual(got, {"walk 1": (-0.2, 2, 3, 0.0), "walk 2": (-1.5, 1, 3, 1.0)})
        self.assertEqual(calls, [4, 3, 3])


class FeatureTests(unittest.TestCase):
    """A prefix's features come from one pass over the longest plan, and must equal what the net
    has after reading only that prefix."""

    def test_states_at_a_prefix_are_the_prefix_alone(self):
        torch.manual_seed(0)
        vocab = Vocab()
        model = GPT(Config(vocab_size=len(vocab), block_size=64, n_layer=2, n_head=2, n_embd=16, dropout=0.0))
        model.eval()
        feats = goblin_rollouts.featurize(model, vocab, "size 1 2", ["walk 1 press 2 exit", "walk 1"])
        self.assertEqual(set(feats), {"", "walk 1", "walk 1 press 2", "walk 1 press 2 exit"})
        for prefix, (h, lp) in feats.items():
            ids = goblin_rollouts.goblin_search.prompt_of(vocab, "size 1 2", prefix)
            alone, token_lp = goblin_rollouts.states(model, ids, [len(ids) - 1])
            self.assertTrue(torch.allclose(h, alone[0], atol=1e-5), prefix)
            sep = ids.index(goblin_rollouts.SEP)
            self.assertAlmostEqual(lp, float(token_lp[sep + 1:].sum()), places=4)
        self.assertEqual(feats[""][1], 0.0)


if __name__ == "__main__":
    unittest.main()
