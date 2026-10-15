"""Make data/ws3 from out/workshop/levels.jsonl (fetch.py build), trusting published puzzles.

    python tools/workshop/ws3.py

Workshop puzzles passed the game's editor and were published, so they are taken as solvable
even where our solver says `unsolvable` (it doesn't model every trick) or `uncertain` (its
budget ran out). Those get an assumed label: verdict solvable, plan length and difficulty
(chain depth) 20% below the largest among the levels the solver did solve, no metrics beyond
that, and "assumed": true with the solver's own verdict in "solver_verdict". `invalid` and
`unsupported` levels are left out (the converter got them wrong).

Levels are deduplicated by tokens. data/ws2's test levels stay held out and its train levels
stay in train; new puzzles go to test when md5(workshop_id) % 9 == 0 (about 11%, like ws2).
Writes data/ws3/{train,test}.jsonl and vocab.txt.
"""

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "out" / "workshop" / "levels.jsonl"
OUT = ROOT / "data" / "ws3"
TRUSTED = ("unsolvable", "uncertain")
MAX_TOKENS = 969  # 1024 block minus <bos>, condition slots, <sep> and <eos> (train/data.py)


def ids(path):
    with open(path, encoding="utf-8") as f:
        return {json.loads(l)["workshop_id"] for l in f if l.strip()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    levels = [json.loads(l) for l in open(a.src, encoding="utf-8") if l.strip()]
    solved = [r for r in levels if r["verdict"] == "solvable"]
    max_len = max(r["min_len"] for r in solved)
    max_depth = max(r["metrics"]["chain_depth"] for r in solved if r.get("metrics"))
    assumed_len, assumed_depth = round(0.8 * max_len), round(0.8 * max_depth, 2)
    print(f"{len(solved)} solved: longest plan {max_len}, deepest chain {max_depth}; "
          f"assumed plan length {assumed_len}, chain depth {assumed_depth}")

    old_test, old_train = ids(ROOT / "data/ws2/test.jsonl"), ids(ROOT / "data/ws2/train.jsonl")
    seen, counts = set(), Counter()
    split = {"train": [], "test": []}
    for r in levels:
        counts[r["verdict"]] += 1
        if r["verdict"] != "solvable" and r["verdict"] not in TRUSTED:
            continue
        if "tokens" not in r or r["tokens"] in seen:
            counts["dropped: duplicate or no tokens"] += 1
            continue
        n = len(r["tokens"].split())
        if n > MAX_TOKENS:
            counts["dropped: too long"] += 1
            continue
        seen.add(r["tokens"])
        r["n_tokens"] = n
        if r["verdict"] in TRUSTED:
            r = {k: v for k, v in r.items() if k not in ("liberal_len", "metrics", "min_len")}
            r.update(verdict="solvable", min_len=assumed_len, needs_portals=True,
                     metrics={"chain_depth": assumed_depth},
                     assumed=True, solver_verdict=r["verdict"])
        wid = r["workshop_id"]
        if wid in old_test:
            name = "test"
        elif wid in old_train:
            name = "train"
        else:
            name = "test" if int(hashlib.md5(wid.encode()).hexdigest(), 16) % 9 == 0 else "train"
        split[name].append(r)

    out.mkdir(parents=True, exist_ok=True)
    for name, rs in split.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            f.writelines(json.dumps(r) + "\n" for r in rs)
        kinds = Counter(r.get("solver_verdict", "solvable") for r in rs)
        print(f"{name}: {len(rs)} levels ({dict(kinds)} by the solver's verdict)")
    shutil.copy(ROOT / "data/ws2/vocab.txt", out / "vocab.txt")
    print(f"from {len(levels)} converted: {dict(counts)}")


if __name__ == "__main__":
    main()
