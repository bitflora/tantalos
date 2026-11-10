"""Search-baseline regressions: .venv/Scripts/python -m unittest discover -s tests -p test_search.py

The beam search's own bookkeeping (train/goblin_search.py), with a stand-in net, env and
`tantalos-score check`: which children it keeps, when it stops, and what it records. Nothing here
runs the net or the solver.
"""

from argparse import Namespace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import goblin_search  # noqa: E402
from data import BOS, EOS, SEP  # noqa: E402

WORDS = ["size", "1", "2", "3", "walk", "press", "fire", "blue", "exit"]
FIRST = 10  # the first word's id, out of BOS/SEP/EOS's way


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


class Env:
    """A stand-in `tantalos-score env`: `answers` maps a plan to what it says about it."""

    let_stop = False

    def __init__(self, answers, opens=True):
        self.answers, self.opens = answers, opens
        self.asked = []

    def open(self, key, tokens):
        return self.opens

    def next(self, keys, plans):
        self.asked += list(plans)
        return [self.answers.get(p, {"allowed": ["walk"]}) for p in plans]

    def close(self, keys):
        pass


class Net:
    """A stand-in net: one round per `generate_many` call, each the step (as words) and log
    probability its rows write, in row order. A round says nothing about the rows after it, and
    there are no rounds left: those write nothing, which the search reads as a dead end."""

    def __init__(self, vocab, rounds):
        self.vocab, self.rounds = vocab, list(rounds)
        self.prompts = []  # the prefixes it was asked to continue, per round

    def generate_many(self, prompts, max_new, stop, temperature, top_k, batch, constrain, logprob=False):
        self.prompts.append([" ".join(constrain.starts[i]) for i in range(len(prompts))])
        this = list(self.rounds.pop(0)) if self.rounds else []
        assert len(this) <= len(prompts), f"{len(this)} steps for {len(prompts)} rows"
        this += [("", 0.0)] * (len(prompts) - len(this))
        return [self.vocab.encode(w.split()) for w, _ in this], [lp for _, lp in this]


class Ticks:
    """A budget that runs out after `n` looks, so a test doesn't wait on a clock."""

    def __init__(self, n=8):
        self.n = n

    def left(self):
        self.n -= 1
        return self.n

    def spent(self):
        return 0.0


def settings(**kw):
    a = Namespace(candidates=2, width=2, score="total", rows=8, temperature=1.0,
                  step_temperature=1.0, max_plan=256, tantalos="none", seconds=10.0)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


class Checked:
    """Stands in for goblin_eval.check: `solves` are the plans it calls solved."""

    def __init__(self, solves=()):
        self.solves, self.seen = set(solves), []

    def __call__(self, items, tantalos, mode):
        plans = [p for it in items for p in it["plans"]]
        self.seen += plans
        return [{"verdict": "solved", "len": len(p.split())} if p in self.solves else {"verdict": "unsolved"}
                for p in plans]


class BeamTests(unittest.TestCase):
    def setUp(self):
        self.vocab = Vocab()
        self.real_check = goblin_search.goblin_eval.check
        self.checked = Checked()
        goblin_search.goblin_eval.check = self.checked

    def tearDown(self):
        goblin_search.goblin_eval.check = self.real_check

    def beam(self, rounds, answers, solves=(), opens=True, **kw):
        self.checked.solves = set(solves)
        self.net = Net(self.vocab, rounds)
        self.env = Env(answers, opens)
        cost = goblin_search.Cost()
        out = goblin_search.beam_level(self.net, self.vocab, {"tokens": "size 1 2 3"}, self.env,
                                      settings(**kw), Ticks(), cost)
        return out, cost

    def test_the_goal_ends_the_level_and_the_plan_is_checked(self):
        out, cost = self.beam(
            [[("walk 1", -1.0), ("walk 2", -2.0)], [("press 3", -1.0)]],
            {"walk 1": {"allowed": ["press"]}, "walk 2": {"allowed": ["press"]},
             "walk 1 press 3": {"end": True}},
            solves=["walk 1 press 3"])
        self.assertTrue(out["solved"])
        self.assertEqual(out["plan"], "walk 1 press 3")
        self.assertEqual(out["rounds"], 2)
        self.assertEqual(self.checked.seen, ["walk 1 press 3"])  # only the child at the goal
        self.assertEqual(cost.checks, 1)
        self.assertEqual(cost.tokens, 2 + 2 + 2)

    def test_the_best_children_are_the_ones_continued(self):
        out, _ = self.beam(
            [[("walk 1", -0.5), ("walk 2", -3.0), ("walk 3", -1.0), ("press 1", -9.0)]],
            {f"walk {n}": {"allowed": ["press"]} for n in (1, 2, 3)} | {"press 1": {"allowed": ["press"]}},
            width=2, candidates=4)
        self.assertEqual(self.net.prompts[1], ["walk 1"] * 4 + ["walk 3"] * 4)
        self.assertFalse(out["solved"])

    def test_mean_scoring_ranks_by_the_step_instead_of_the_plan(self):
        self.beam(
            [[("walk 1", -2.0), ("press 1 2 3", -2.5)]],
            {"walk 1": {"allowed": ["press"]}, "press 1 2 3": {"allowed": ["press"]}},
            width=1, candidates=2, score="mean")
        # -2.5 over 4 words beats -2.0 over 2; by total log probability "walk 1" would win.
        self.assertEqual(self.net.prompts[1], ["press 1 2 3"] * 2)

    def test_the_same_step_twice_counts_once(self):
        out, _ = self.beam(
            [[("walk 1", -1.0), ("walk 1", -1.0)]],
            {"walk 1": {"allowed": ["press"]}},
            width=2, candidates=2)
        self.assertEqual(self.net.prompts[1], ["walk 1", "walk 1"])
        self.assertFalse(out["solved"])

    def test_a_beam_with_nothing_alive_starts_again_and_says_why(self):
        out, _ = self.beam([[("walk 1", -1.0), ("walk 2", -1.0)]],
                           {"walk 1": {"dead": True}, "walk 2": {"dead": True}})
        self.assertFalse(out["solved"])
        self.assertGreater(out["passes"], 1)  # the budget wasn't out, so it tried again
        self.assertIn("dead 2", out["note"])

    def test_a_step_that_never_ended_is_dropped(self):
        out, _ = self.beam([[("", 0.0), ("", 0.0)]], {})
        self.assertFalse(out["solved"])
        self.assertEqual(out["children"], 0)

    def test_a_level_the_env_cannot_open_is_said_so(self):
        out, _ = self.beam([], {}, opens=False)
        self.assertFalse(out["solved"])
        self.assertIn("could not open", out["note"])

    def test_a_plan_past_the_limit_is_dropped(self):
        out, _ = self.beam([[("walk 1 2 3", -1.0), ("walk 1 2 3", -1.0)]], {}, max_plan=3)
        self.assertEqual(out["children"], 0)


class BoundedTests(unittest.TestCase):
    def constraint(self, answer, seq, budget=None, one_step=True):
        vocab = Vocab()
        env = Env({}, opens=True)
        env.next = lambda keys, plans: [answer]
        prompts = [[BOS] + vocab.encode("size 1 2 3".split()) + [SEP]]
        constrain = goblin_search.Bounded(env, vocab, prompts, 256, budget, one_step)
        return constrain([0], [seq]), vocab

    def test_a_whole_step_may_only_end(self):
        mask, _ = self.constraint({"allowed": ["press", "walk"], "whole": True},
                                  [FIRST + WORDS.index("walk")])
        self.assertEqual(mask[0].nonzero().flatten().tolist(), [EOS])

    def test_a_step_part_way_through_writes_on(self):
        mask, vocab = self.constraint({"allowed": ["press", "walk"]},
                                      [FIRST + WORDS.index("walk")])
        allowed = {vocab.words[i - FIRST] for i in mask[0].nonzero().flatten().tolist() if i >= FIRST}
        self.assertEqual(allowed, {"press", "walk"})

    def test_an_empty_plan_is_not_a_finished_step(self):
        mask, _ = self.constraint({"allowed": ["walk"], "whole": True}, [])
        self.assertNotEqual(mask[0].nonzero().flatten().tolist(), [EOS])

    def test_a_row_may_only_end_once_the_budget_is_out(self):
        mask, _ = self.constraint({"allowed": ["press", "walk"]}, [FIRST + WORDS.index("walk")],
                                  budget=goblin_search.Budget(0.0), one_step=False)
        self.assertEqual(mask[0].nonzero().flatten().tolist(), [EOS])


class SampleTests(unittest.TestCase):
    """Best-of-N: the rounds it writes and what it keeps of them."""

    def setUp(self):
        self.real_check = goblin_search.goblin_eval.check
        self.checked = Checked(["press 1"])
        goblin_search.goblin_eval.check = self.checked

    def tearDown(self):
        goblin_search.goblin_eval.check = self.real_check

    def sample(self, plans, budget):
        vocab = Vocab()

        class Net:
            def generate_many(self, prompts, max_new, stop, temperature, top_k, batch, constrain):
                return [vocab.encode(p.split()) for p in plans]

        cost = goblin_search.Cost()
        out = goblin_search.sample_level(Net(), vocab, {"tokens": "size 1 2 3"}, Env({}),
                                         settings(rows=len(plans)), budget, cost)
        return out, cost

    def test_a_round_that_solves_ends_the_level(self):
        out, cost = self.sample(["walk 1", "press 1", "press 1"], goblin_search.Budget(10.0))
        self.assertTrue(out["solved"])
        self.assertEqual(out["plan"], "press 1")
        self.assertEqual((out["rounds"], out["plans"]), (1, 2))  # the same plan twice is checked once
        self.assertEqual(cost.checks, 2)
        self.assertEqual(cost.tokens, 6)

    def test_no_round_is_started_once_the_budget_is_out(self):
        out, cost = self.sample(["press 1"], goblin_search.Budget(0.0))
        self.assertFalse(out["solved"])
        self.assertEqual((out["rounds"], cost.checks), (0, 0))


class SummaryTests(unittest.TestCase):
    def records(self):
        return [{"by": "greedy", "solved": True, "seconds": 0.5, "plan_steps": 3, "peak_mb": 100.0,
                 "cost": {"forwards": 10, "tokens": 12, "checks": 1, "envs": 14}},
                {"by": "beam", "solved": True, "seconds": 30.0, "plan_steps": 9, "peak_mb": 700.0,
                 "cost": {"forwards": 100, "tokens": 900, "checks": 2, "envs": 120}},
                {"by": "beam", "solved": False, "seconds": 60.0, "peak_mb": 650.0,
                 "cost": {"forwards": 200, "tokens": 1800, "checks": 0, "envs": 240}}]

    def test_what_a_method_is_credited_with(self):
        s = goblin_search.method_summary(self.records())
        self.assertAlmostEqual(s["solved"], 2 / 3)
        self.assertAlmostEqual(s["greedy_solved"], 1 / 3)
        self.assertEqual((s["beyond_greedy"], s["beyond_greedy_of"]), (1, 2))
        self.assertEqual(s["seconds_mean_unsolved_by_greedy"], 45.0)
        self.assertEqual(s["plan_steps_mean"], 6.0)
        self.assertEqual(s["peak_mb"], 700.0)
        self.assertEqual(s["cost"], {"forwards": 310, "tokens": 2712, "checks": 3, "envs": 374})


class StepsTests(unittest.TestCase):
    def test_steps_are_counted_by_their_verb(self):
        self.assertEqual(goblin_search.steps_in("fire blue 7 0 4 -z enter blue"), 2)
        self.assertEqual(goblin_search.steps_in("press 17 1 1 walk 16 2 1 exit"), 3)
        self.assertEqual(goblin_search.steps_in(""), 0)


if __name__ == "__main__":
    unittest.main()
