"""Reverse-generated levels for data/v5, deduplicated, plus mutants of them for more.

    .venv/Scripts/python tools/reverse_mutants.py --count N  RAW.jsonl...

src/bin/reverse.rs makes only a few thousand distinct chambers (about 5 layouts per room, at
most 6 rooms), so RAW files sampled until they stop giving new levels are deduplicated by
tokens into data/v5/levels-reverse.jsonl. Then `glados mutate` edits them in rounds, as
data/v5/mutate.py does for Workshop levels: round 1 mutates the reverse levels, each later round
the previous round's new mutants, until there are N distinct mutants new to data/v5 and the
Workshop levels. `glados mutate` keeps only solvable mutants that need portals, labelled by the
solver; a mutant is no longer solvable by construction, so it carries no "src" and gets no
`<maxlen>` (train/data.py). The reverse levels break no editor rule, so a mutant must break none
either (glados score with editor rules on). "mutated_from" is "reverse:<line>" of
levels-reverse.jsonl.

Round files stay in data/v5/reverse-mutants/ and are reused when the script runs again. Outputs
are written there first and moved in as data/v5/levels-reverse.jsonl /
levels-reverse-mutants.jsonl only when complete, because training reads data/v5/levels-*.jsonl.
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

PER_LEVEL = {1: 20}  # round 1 has few parents; later rounds have plenty
TRIES = 60


def lines(path):
    with open(path, encoding="utf-8") as f:
        return [l for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", nargs="+", help="src/bin/reverse.rs output")
    ap.add_argument("--count", type=int, required=True, help="mutants wanted")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]), help="checkout with data/")
    a = ap.parse_args()
    root = Path(a.root)
    v5, work = root / "data" / "v5", root / "data" / "v5" / "reverse-mutants"
    glados = root / "target" / "release" / "glados.exe"
    work.mkdir(exist_ok=True)

    def clean(records):
        """The records our editor rules don't flag (glados score, budget 1, rules on)."""
        out = subprocess.run([str(glados), "score", "--budget", "1"],
                             input="".join(r["tokens"] + "\n" for r in records),
                             capture_output=True, text=True, encoding="utf-8", check=True).stdout.splitlines()
        assert len(out) == len(records)
        return [r for r, s in zip(records, out) if json.loads(s)["verdict"] != "invalid"]

    def write(records, name):
        tmp = work / name
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)
        shutil.move(tmp, v5 / name)

    reverse, seen = [], set()
    for p in a.raw:
        for l in lines(p):
            r = json.loads(l)
            if r["tokens"] not in seen:
                seen.add(r["tokens"])
                reverse.append(r)
    kept = clean(reverse)
    print(f"{len(reverse)} distinct reverse levels, {len(kept)} pass the editor rules", flush=True)
    reverse = kept
    write(reverse, "levels-reverse.jsonl")

    for p in [*v5.glob("levels-*.jsonl"), root / "data/ws3/train.jsonl", root / "data/ws3/test.jsonl"]:
        seen |= {json.loads(l)["tokens"] for l in lines(p)}
    parents = work / "parents-1.jsonl"
    with open(parents, "w", encoding="utf-8", newline="\n") as f:
        for i, r in enumerate(reverse):
            f.write(json.dumps({"tokens": r["tokens"], "workshop_id": f"reverse:{i}"}) + "\n")
    pool = []
    for rnd in range(1, 20):
        out = work / f"round-{rnd}.jsonl"
        per = PER_LEVEL.get(rnd, 5)
        if out.exists():
            print(f"round {rnd}: kept from an earlier run", flush=True)
        else:
            print(f"round {rnd}: mutating {len(lines(parents))} levels, {per} each", flush=True)
            tmp = out.with_suffix(".part")
            subprocess.run([str(glados), "mutate", "--seed", str(500 + rnd), "--per-level", str(per),
                            "--tries", str(TRIES), "--max-tokens", "969", "--out", str(tmp), str(parents)],
                           check=True)
            tmp.replace(out)
        fresh = [r for r in map(json.loads, lines(out)) if r["tokens"] not in seen]
        seen |= {r["tokens"] for r in fresh}
        fresh = clean(fresh)
        pool += fresh
        print(f"round {rnd}: {len(fresh)} new valid mutants, {len(pool)} in all", flush=True)
        if len(pool) >= a.count or not fresh:
            break
        # Next round's parents: enough for the shortfall (about 4 new valid mutants of 5).
        want = min(len(fresh), int((a.count - len(pool)) / 4) + 1000)
        parents = work / f"parents-{rnd + 1}.jsonl"
        with open(parents, "w", encoding="utf-8", newline="\n") as f:
            for r in random.Random(rnd).sample(fresh, want):
                f.write(json.dumps({"tokens": r["tokens"], "workshop_id": r["mutated_from"]}) + "\n")
    if len(pool) < a.count:
        print(f"only {len(pool)} mutants; writing them all", flush=True)
    picked = random.Random(0).sample(pool, min(a.count, len(pool)))
    write(picked, "levels-reverse-mutants.jsonl")
    print(f"wrote {len(reverse)} reverse levels and {len(picked)} mutants of them", flush=True)


if __name__ == "__main__":
    sys.exit(main())
