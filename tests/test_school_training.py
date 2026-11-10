import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import run
import school
from goblin_eval import reference_records


class ReferenceTests(unittest.TestCase):
    def test_saved_subset_and_order_are_preserved(self):
        records = [{"tokens": t} for t in ("a", "b", "c")]
        reference = [{"tokens": t, "set": "test-frontier"} for t in ("c", "a")]
        self.assertEqual(reference_records(records, reference, "test-frontier"), [records[2], records[0]])
        with self.assertRaises(ValueError):
            reference_records(records[:2], reference, "test-frontier")
        with self.assertRaises(ValueError):
            reference_records(records, reference, "test-v6")


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.parent = self.root / "parent" / "best.pt"
        self.child = self.root / "child"
        self.state = {"init": str(self.parent), "task": "goblin", "constrained_eval": True}

    def report(self, directory, score, token="level", **settings):
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint = self.parent if directory.parent == self.parent.parent else self.child / "best.pt"
        report = dict(checkpoint=str(checkpoint), constrained=True, let_stop=False, max_plan=256, k=8,
                      headline={"set": "test-ws+v6", "solved": score}, sets={"test-ws+v6": {"n": 2}})
        report.update(settings)
        (directory / "eval.json").write_text(json.dumps(report))
        (directory / "solutions.jsonl").write_text("\n".join(json.dumps({"set": s, "tokens": token})
                                                             for s in ("test-ws", "test-v6")))

    def test_free_parent_does_not_approve_constrained_child(self):
        self.report(self.parent.parent / "eval", .20, constrained=False)
        self.report(self.child / "eval", .40)
        self.assertEqual(run.quick_guard(self.child, self.state)[0], self.parent)

    def test_regression_and_pass(self):
        self.report(self.parent.parent / "eval-constrained", .573)
        for score, expected in ((.50, self.parent), (.56, self.child / "best.pt")):
            self.report(self.child / "eval", score)
            self.assertEqual(run.quick_guard(self.child, self.state)[0], expected)

    def test_mismatched_levels_or_settings_fail_closed(self):
        self.report(self.parent.parent / "eval-constrained", .573)
        for settings in ({"token": "different"}, {"max_plan": 512}, {"let_stop": True}, {"k": 4}):
            self.report(self.child / "eval", .8, **settings)
            self.assertEqual(run.quick_guard(self.child, self.state)[0], self.parent)

    def test_missing_parent_fails_closed(self):
        self.report(self.child / "eval", .8)
        self.assertEqual(run.quick_guard(self.child, self.state)[0], self.parent)


class ExportTests(unittest.TestCase):
    def test_bad_augmented_plan_is_audited_and_excluded(self):
        row = {"identity": "puzzle", "tokens": "level", "plan": "a", "plans": ["a", "b"]}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.jsonl"
            with patch.object(school, "read_jsonl", return_value=[row]), \
                 patch.object(school, "test_identities", return_value=set()), \
                 patch.object(school, "validate_buffer"), \
                 patch.object(school, "check_items", return_value=[[{"verdict": "solved"}], [{"verdict": "illegal"}]]), \
                 patch.object(school.goblin_data, "augment", side_effect=lambda rows, syms, label: rows):
                school.export(SimpleNamespace(name="test", out=out, tantalos="checker"))
            self.assertEqual(json.loads(out.read_text())["plan"], "a")
            self.assertEqual(json.loads(out.with_suffix(".rejected.jsonl").read_text())["plan"], "b")

    def test_each_plan_keeps_identity_and_is_rechecked(self):
        row = {"identity": "puzzle", "tokens": "level", "plan": "a", "plans": ["a", "b"], "min_len": 1}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.jsonl"
            with patch.object(school, "read_jsonl", return_value=[row]), \
                 patch.object(school, "test_identities", return_value=set()), \
                 patch.object(school, "validate_buffer") as validate, \
                 patch.object(school, "check_items", return_value=[[{"verdict": "solved"}], [{"verdict": "solved"}]]), \
                 patch.object(school.goblin_data, "augment", side_effect=lambda rows, syms, label: rows):
                school.export(SimpleNamespace(name="test", out=out, tantalos="checker"))
                rows = [json.loads(line) for line in out.read_text().splitlines()]
                self.assertEqual([r["plan"] for r in rows], ["a", "b"])
                self.assertTrue(all(r["identity"] == "puzzle" and r["min_len"] is None for r in rows))
                self.assertEqual(validate.call_count, 1)

    def test_failed_check_does_not_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.jsonl"
            with patch.object(school, "read_jsonl", return_value=[]), \
                 patch.object(school, "test_identities", return_value=set()), \
                 patch.object(school, "validate_buffer", side_effect=ValueError("invalid")):
                with self.assertRaises(ValueError):
                    school.export(SimpleNamespace(name="test", out=out, tantalos="checker"))
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
