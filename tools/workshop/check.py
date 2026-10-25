"""Check convert.py against compiled maps of known puzzles: fixtures/workshop/<name>.bsp.gz is the
Puzzle Maker's compile of <name>.p2c (slimmed to the lumps convert.py reads).

    python tools/workshop/check.py

For each pair, the converted tokens and the .p2c's own tokens must get the same solver label
(verdict, shortest plan, metrics). Token differences are listed; known harmless ones are faces the
map can't show (white under a dropper, glass against a wall, sealed pockets nobody can reach).
"""
import difflib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bsp import Bsp  # noqa: E402
from convert import convert  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXE = ".exe" if sys.platform == "win32" else ""
SCORE = ROOT / "target" / "release" / f"glados-score{EXE}"
P2C = ROOT / "target" / "release" / f"glados-p2c{EXE}"


def score(tokens):
    out = subprocess.run([str(SCORE), "score"], input=tokens + "\n", capture_output=True, text=True, check=True)
    r = json.loads(out.stdout)
    return r, {k: r.get(k) for k in ("verdict", "min_len", "metrics", "problem")}


def main():
    failed = 0
    for bsp in sorted((ROOT / "fixtures" / "workshop").glob("*.bsp.gz")):
        name = bsp.name[:-len(".bsp.gz")]
        got = convert(Bsp.open(bsp))
        want = subprocess.run([str(P2C), "tokens", str(bsp.with_name(name + ".p2c"))],
                              capture_output=True, text=True, check=True).stdout.splitlines()[1]
        (got_r, got_label), (_, want_label) = score(got), score(want)
        ok = got_label == want_label
        failed += not ok
        print(f"{name}: {'same label' if ok else 'DIFFERENT LABEL'} ({want_label['verdict']}, "
              f"plan {want_label['min_len']})")
        if not ok:
            print(f"  want {want_label}\n  got  {got_label}")
        a, b = want.split(), got_r.get("tokens", got).split()
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
            if op != "equal":
                print(f"  token diff: .p2c {' '.join(a[i1:i2]) or '-'} | map {' '.join(b[j1:j2]) or '-'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
