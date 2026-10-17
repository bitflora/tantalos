"""Mutants of the Workshop training levels for data/v5, made again from the rebuilt data/ws3.

    .venv/Scripts/python tools/workshop_mutants.py --count N [--root CHECKOUT]

data/v5/levels-mutants.jsonl came from Workshop levels decoded before the rail-height and lift
fixes in tools/workshop/convert.py, so its mutants carry the old decoding's mistakes. This makes
them again, as data/v5/mutate.py and refill.py did: round 1 mutates data/ws3/train.jsonl (never
test.jsonl), each later round the previous round's new mutants, until there are N distinct
mutants new to data/v5 and the Workshop levels. `glados mutate` keeps only solvable mutants that
need portals. A mutant counts as invalid only for editor rules its original Workshop level
doesn't break (our rules flag about 19% of published levels, which the game accepted).

Round files go to data/v5/mutants-<date>/ and are reused when the script runs again. The old
levels-mutants.jsonl moves to data/v5/old/ and the new one is moved in only when complete,
because training reads data/v5/levels-*.jsonl.
"""

import argparse
import datetime
import json
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

PER_LEVEL = {1: 10}  # round 1 has few parents; later rounds have plenty
# Round 1's parents are mostly levels our solver can't solve, whose mutants rarely are either:
# each of those spends all its tries, which made the first round take hours (mutate.py's was cut).
TRIES = {1: 25}
RULE = re.compile(r"[(, ]([a-z]+(?:-[a-z]+)+)[,)]")


def lines(path):
    with open(path, encoding="utf-8") as f:
        return [l for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, required=True, help="mutants wanted")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]), help="checkout with data/")
    a = ap.parse_args()
    root = Path(a.root)
    v5 = root / "data" / "v5"
    work = v5 / f"mutants-{datetime.date.today().isoformat()}"
    glados = root / "target" / "release" / "glados.exe"
    work.mkdir(exist_ok=True)

    def problems(records):
        """Per record, the editor rules our checker says it breaks (glados score, budget 1)."""
        out = subprocess.run([str(glados), "score", "--budget", "1"],
                             input="".join(r["tokens"] + "\n" for r in records),
                             capture_output=True, text=True, encoding="utf-8", check=True).stdout.splitlines()
        assert len(out) == len(records)
        return [set(RULE.findall(json.loads(s).get("problem", ""))) for s in out]

    ws = [json.loads(l) for l in lines(root / "data/ws3/train.jsonl")]
    original = {str(r["workshop_id"]): p for r, p in zip(ws, problems(ws))}

    def made_valid(records):
        return [r for r, p in zip(records, problems(records)) if not p - original.get(r["mutated_from"], set())]

    seen = {r["tokens"] for r in ws} | {json.loads(l)["tokens"] for l in lines(root / "data/ws3/test.jsonl")}
    for p in v5.glob("levels-*.jsonl"):
        if p.name != "levels-mutants.jsonl":
            seen |= {json.loads(l)["tokens"] for l in lines(p)}
    parents = work / "parents-1.jsonl"
    with open(parents, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps({"tokens": r["tokens"], "workshop_id": r["workshop_id"]}) + "\n" for r in ws)
    pool = []
    for rnd in range(1, 20):
        out = work / f"round-{rnd}.jsonl"
        per = PER_LEVEL.get(rnd, 5)
        if out.exists():
            print(f"round {rnd}: kept from an earlier run", flush=True)
        else:
            print(f"round {rnd}: mutating {len(lines(parents))} levels, {per} each", flush=True)
            tmp = out.with_suffix(".part")
            subprocess.run([str(glados), "mutate", "--seed", str(700 + rnd), "--per-level", str(per),
                            "--tries", str(TRIES.get(rnd, 60)), "--max-tokens", "969", "--out", str(tmp), str(parents)],
                           check=True)
            tmp.replace(out)
        fresh = [r for r in map(json.loads, lines(out)) if r["tokens"] not in seen]
        seen |= {r["tokens"] for r in fresh}
        fresh = made_valid(fresh)
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
    tmp = work / "levels-mutants.jsonl"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r) + "\n" for r in picked)
    old = v5 / "levels-mutants.jsonl"
    if old.exists():
        (v5 / "old").mkdir(exist_ok=True)
        shutil.move(old, v5 / "old" / f"levels-mutants-before-{work.name.split('-', 1)[1]}.jsonl")
    shutil.move(tmp, old)
    print(f"wrote {len(picked)} mutants of {len({r['mutated_from'] for r in picked})} Workshop levels "
          f"to {old}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
