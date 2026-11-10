"""train/sample.py's Grammar: several `tantalos-score grammar` servers answer as one does.

Needs a tantalos-score with the editor's rules in its grammar (2026-11-01); skipped without one.
Point TANTALOS_SCORE at another build to test that one.

    .venv/Scripts/python -m unittest tests/test_sample_grammar.py -v
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
import sample  # noqa: E402

TANTALOS = Path(os.environ.get("TANTALOS_SCORE", sample.DEFAULT_TANTALOS))
LEVELS = [
    "size 8 8 4 open 1 1 1 6 6 2 entry 1 2 1 +x exit 6 6 1 -x plate 1 2 1 +z 3 3 1 +z",
    "size 7 7 4 open 1 1 1 5 5 2 entry 3 1 1 +y exit 4 5 1 -y button 2 2 1 weighted > exit",
    "size 6 9 5 open 1 1 1 4 7 3 entry 1 4 1 +x exit 4 4 2 -x turret 2 2 1 4 cube 3 6 1 companion",
]


def editor_rules(tantalos):
    """Whether this build's grammar knows the editor's rules (it takes --editor-rules)."""
    if not tantalos.exists():
        return False
    p = subprocess.run([str(tantalos), "grammar", "--editor-rules", "off"], input="", capture_output=True, text=True)
    return p.returncode == 0


@unittest.skipUnless(editor_rules(TANTALOS), f"no tantalos-score with editor rules at {TANTALOS}")
class SplitServers(unittest.TestCase):
    def test_same_answers_in_order(self):
        prefixes = [" ".join(level.split()[:k]) for level in LEVELS for k in range(len(level.split()) + 1)]
        one, three = sample.Grammar(TANTALOS, procs=1), sample.Grammar(TANTALOS, procs=3)
        try:
            self.assertEqual(one.next(prefixes), three.next(prefixes))
            # Asked again (warm caches, the levels pinned to the same servers): the same.
            self.assertEqual(one.next(prefixes[::-1]), three.next(prefixes[::-1]))
        finally:
            one.stop()
            three.stop()

    def test_editor_rules_in_the_answers(self):
        g = sample.Grammar(TANTALOS, procs=2)
        try:
            # A faith plate may not share the entry door's cell (src/grammar.rs tests).
            [a] = g.next(["size 8 8 4 open 1 1 1 6 6 2 entry 1 2 1 +x exit 6 6 1 -x plate 1 2"])
            self.assertEqual(a["allowed"], ["2"])
        finally:
            g.stop()


if __name__ == "__main__":
    unittest.main()
