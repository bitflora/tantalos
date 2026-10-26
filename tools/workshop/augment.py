"""Workshop training levels as their 8 mirrored/turned versions, instead of 20 copies of each.

    python tools/workshop/augment.py                 # data/ws5/train.jsonl -> data/v5b/levels-workshop-aug-x2.jsonl
    python tools/workshop/augment.py --copies 3

data/v5b/workshop-x20.jsonl (levels-workshop-x20.jsonl until 2026-10-26) repeats every ws5 train
level 20 times, and models learn those levels by heart (2026-10-25: 1.59 nats a token on held-out Workshop levels against 0.24 on
trained ones). This writes each level's versions from `glados-data augment` (src/augment.rs:
mirror in x, then 0-3 quarter turns about z; a version that repeats an earlier one's tokens, or
whose mirrored field would lose its wall, is left out) `--copies` times each, so about the same
number of Workshop sequences with 8 times as many distinct ones. Each version is the original
record with its `tokens` and `n_tokens` replaced and `variant` added (r0 = the level itself, as
re-encoded); the labels carry over, since a version plays the same (tests/augment.rs, and
`glados-data augment --check` on Workshop levels). Versions over 969 tokens are left out (the
greedy box cover can come out longer in another orientation).

Since 2026-10-26 the default output, data/v5b/levels-workshop-aug-x2.jsonl, is what run.py
trains on (its default `levels-*.jsonl` pattern); the x20 file was renamed
data/v5b/workshop-x20.jsonl, out of that pattern. To train on the x20 copies again, name the
files with run.py's `--data`, e.g.

    --data "data/v5b/levels-[1-5].jsonl" "data/v5b/levels-mutants*.jsonl" data/v5b/workshop-x20.jsonl

The held-out data/ws5/test.jsonl is never augmented: it stays the Workshop test loss's set.

Levels listed in tools/workshop/excluded-tricks.jsonl (their Workshop text says they need bunny
hopping or a glitch) are left out.
"""
import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "target" / "release" / ("glados-data.exe" if sys.platform == "win32" else "glados-data")
MAX_TOKENS = 969  # 1024 block minus <bos>, condition slots, <sep> and <eos> (train/data.py)
# Workshop levels whose text says they need bunny hopping or a glitch: not rated-style examples.
EXCLUDED = ROOT / "tools" / "workshop" / "excluded-tricks.jsonl"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default=str(ROOT / "data" / "ws5" / "train.jsonl"))
    ap.add_argument("--copies", type=int, default=2, help="times each version is written")
    ap.add_argument("--out", default=None, help="default data/v5b/levels-workshop-aug-xN.jsonl")
    a = ap.parse_args()
    out = Path(a.out) if a.out else ROOT / "data" / "v5b" / f"levels-workshop-aug-x{a.copies}.jsonl"
    src = Path(a.src)
    if src.resolve() == (ROOT / "data" / "ws5" / "test.jsonl").resolve():
        sys.exit("data/ws5/test.jsonl is the held-out set: don't train on its versions")
    records = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    excluded = set()
    if EXCLUDED.exists():
        excluded = {json.loads(l)["id"] for l in open(EXCLUDED, encoding="utf-8") if l.strip()}
    skipped = sum(1 for r in records if r.get("workshop_id") in excluded)
    print(f"{skipped} of {len(records)} levels skipped: bunny-hop/glitch ({EXCLUDED.relative_to(ROOT)})")
    res = subprocess.run([str(DATA), "augment", str(src)], capture_output=True, text=True,
                         encoding="utf-8", check=True)
    counts, per_level = Counter(), Counter()
    tmp = out.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for line in res.stdout.splitlines():
            v = json.loads(line)
            if "error" in v:
                counts["left out: " + v["error"].split(":")[0]] += 1
                continue
            if v["n_tokens"] > MAX_TOKENS:
                counts["left out: too long"] += 1
                continue
            if records[v["line"] - 1].get("workshop_id") in excluded:
                continue
            r = {**records[v["line"] - 1], "tokens": v["tokens"], "n_tokens": v["n_tokens"], "variant": v["variant"]}
            f.write((json.dumps(r) + "\n") * a.copies)
            counts["versions"] += 1
            per_level[v["line"]] += 1
    tmp.replace(out)
    spread = Counter(per_level.values())
    print(f"{len(records)} levels -> {counts['versions']} versions x {a.copies} = "
          f"{counts['versions'] * a.copies} lines in {out}")
    print(f"versions per level: {dict(sorted(spread.items()))}; {dict(c for c in counts.items() if c[0] != 'versions')}")
    print(res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "")


if __name__ == "__main__":
    main()
