"""School regressions: .venv/Scripts/python -m unittest discover -s tests -p test_school.py"""

import contextlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import school


class SchoolTests(unittest.TestCase):
    def test_polish_does_not_reuse_another_plans_verdicts(self):
        # Both ways of shortening the first plan solve; neither for the second does.
        good = {"walk 0 0 0", "press 0 0 0", "pickup 0 0 0 putdown 1 0 0"}

        def check(items, *args):
            return [{"verdict": "solved" if p in good else "unfinished"}
                    for item in items for p in item["plans"]]

        with patch.object(school.goblin_eval, "check", side_effect=check):
            plans = school.polish("level", ["walk 0 0 0 press 0 0 0",
                                             "pickup 0 0 0 putdown 1 0 0"], "mock")
        self.assertEqual(plans, ["press 0 0 0", "pickup 0 0 0 putdown 1 0 0"])
        self.assertTrue(all(p in good for p in plans))

    def test_checks_remain_grouped_across_batches_and_empty_items(self):
        items = [{"tokens": str(i), "plans": list(range(i % 3))} for i in range(130)]

        def check(part, *args):
            return [(item["tokens"], p) for item in part for p in item["plans"]]

        with patch.object(school.goblin_eval, "check", side_effect=check) as mocked:
            results = school.check_items(items, "mock", "conservative")
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(results, [[(item["tokens"], p) for p in item["plans"]] for item in items])

    def test_attempt_keeps_alignment_when_a_prompt_is_skipped(self):
        levels = [{"tokens": t} for t in ("first", "unknown", "last")]
        args = SimpleNamespace(chunk=3, k=2, env=None, tantalos="mock")

        def check(items, *args):
            return [{"level": item["tokens"], "plan": p} for item in items for p in item["plans"]]

        with (patch.object(school, "level_prompts", return_value=[[1], "unknown_tokens", [2]]),
              patch.object(school.goblin_eval, "generate_plans", side_effect=[
                  ["g1", "g2"], ["s1", "s2", "s3", "s4"]]),
              patch.object(school.goblin_eval, "check", side_effect=check),
              contextlib.redirect_stdout(io.StringIO())):
            school.attempt(None, None, levels, args, lambda: None)
        self.assertEqual(levels[1]["results"], [])
        self.assertEqual(levels[0]["results"], [{"level": "first", "plan": p} for p in ("g1", "s1", "s2")])
        self.assertEqual(levels[2]["results"], [{"level": "last", "plan": p} for p in ("g2", "s3", "s4")])

    def test_pool_excludes_probe_and_test_identities(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(school, "ROOT", Path(tmp)):
            root = Path(tmp)
            school.write_jsonl(root / "data/goblin/test-ws.jsonl", [{"identity": "ws:1"}])
            school.write_jsonl(root / "data/goblin/probe.jsonl", [{"identity": "ws:2"}])
            rows = [{"workshop_id": i, "tokens": str(i), "solver_verdict": "uncertain"}
                    for i in (1, 2, 3)]
            school.write_jsonl(root / school.WS_TRAIN, rows)
            school.write_jsonl(root / "data/goblin/pool/v6.jsonl", rows)
            self.assertEqual(set(school.candidates()), {"ws:3"})

    def test_buffer_rejects_held_out_and_invalid_alternatives(self):
        row = {"identity": "id", "tokens": "level", "plan": "exit", "plans": ["exit", "bad"]}
        with patch.object(school.goblin_eval, "check") as check:
            with self.assertRaisesRegex(ValueError, "held-out"):
                school.validate_buffer([row], {"id"}, "mock")
            check.assert_not_called()
            check.return_value = [{"verdict": "solved"}, {"verdict": "unfinished"}]
            with self.assertRaisesRegex(ValueError, "invalid plan"):
                school.validate_buffer([row], set(), "mock")
            check.return_value = [{"verdict": "solved"}] * 2
            school.validate_buffer([row], set(), "mock")

    def test_buffer_rejects_missing_primary_plan(self):
        with self.assertRaisesRegex(ValueError, "primary plan"):
            school.validate_buffer([{"identity": "id", "plan": "exit", "plans": []}], set(), "mock")

    def test_failed_atomic_replace_preserves_previous_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "levels.jsonl"
            school.save_state(path, {"old": {"identity": "old"}})
            before = path.read_bytes()
            with patch.object(school.os, "replace", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    school.save_state(path, {"new": {"identity": "new"}})
            self.assertEqual(path.read_bytes(), before)
            school.save_state(path, {"new": {"identity": "new"}})
            self.assertEqual(school.load_state(path), {"new": {"identity": "new"}})

    def test_draw_fills_short_side_without_duplicates(self):
        ids, counts = school.draw(dict.fromkeys("abcde"), {}, 8, 0, random.Random(1))
        self.assertEqual(set(ids), set("abcde"))
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(sum(counts.values()), len(ids))

    def test_completion_marker_is_last_and_interruption_can_resume(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            out = Path(tmp)
            args = SimpleNamespace(name="test", round=0, model="mock", seed=1, k=1, free=True, no_prefix=False,
                                   levels=1, tantalos="mock", chunk=1)
            pool = {"id": {"identity": "id", "tokens": "level", "frontier": True, "source": "test"}}

            def attempt(model, vocab, levels, args, save):
                for lv in levels:
                    lv.update(plans=["exit"], kind=["greedy"], results=[{"verdict": "solved"}])
                save()

            original = school.write_jsonl

            def fail_attempts(path, rows):
                if path.name == "attempts.jsonl":
                    raise OSError("interrupted")
                original(path, rows)

            with (patch.object(school, "OUT", out),
                  patch.object(school, "candidates", return_value=pool),
                  patch.object(school, "test_identities", return_value=set()),
                  patch.object(school.sample, "load_checkpoint", return_value=(SimpleNamespace(
                      cfg=SimpleNamespace(block_size=1024)), None, None)),
                  patch.object(school, "attempt", side_effect=attempt),
                  patch.object(school.goblin_eval, "check", return_value=[{"verdict": "solved"}])):
                with patch.object(school, "write_jsonl", side_effect=fail_attempts):
                    with self.assertRaisesRegex(OSError, "interrupted"):
                        school.round_(args)
                round_dir = out / "test/r00"
                self.assertFalse((round_dir / "summary.json").exists())
                with patch.object(school, "test_identities", return_value={"id"}):
                    with self.assertRaisesRegex(SystemExit, "saved round contains held-out"):
                        school.round_(args)
                school.round_(args)
                self.assertTrue((round_dir / "attempts.jsonl").exists())
                self.assertEqual(json.loads((round_dir / "summary.json").read_text())["solved_levels"], 1)
                self.assertEqual(school.load_state(out / "test/levels.jsonl")["id"]["tries"], 1)


if __name__ == "__main__":
    unittest.main()
