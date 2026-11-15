"""The rest of the 2026-11-14 rated-style fine-tune set (beside data/v5b/ft-ws45-aug-x2.jsonl,
which is `tools/workshop/augment.py --min-stars 4`):

    .venv/Scripts/python train/ft_data.py

- data/v5b/ft-self.jsonl: the level net's own good levels, re-scored into dataset records: the
  strong rated-prompt samples (solvable, novel, 2+ state changes, at most 1 trick; rated.strong)
  of the runs' eval/rated-samples.jsonl, and generated levels the user rated 2+ and found
  solvable in the workbench (train/generated-annotations.jsonl). Levels in
  data/ingame/{human-unsolvable,invalid,dead}.jsonl are left out. They condition as `<src:gen>`
  (no workshop_id), like train/harvest.py's levels. Written --copies times.
- data/v5b/ft-base-sample.jsonl: a uniform random slice of the default training data
  (data/v5b/levels-*.jsonl), --base lines, so the fine-tune keeps its validity and solvability.
- data/v5b/ft-ws45-weighted.jsonl (--weighted): the versions in ft-ws45-aug-x2.jsonl, written
  by how much they teach hard puzzles instead of twice each: 2 copies, 4 for a 5-star level,
  4 more for a long one (solution 10+ actions), and 1 (2 at 5 stars) for a short solvable one
  (5 actions or fewer). Levels our solver can't solve (`published`, 86% of the set) get no
  extra weight: that verdict mixes hard puzzles with solver gaps. With --weighted, the default
  --base keeps the base slice at 15% of the whole set, and ft-self.jsonl keeps only levels with
  solutions of 8+ actions (--self-min-len): the 2026-11-14 A/B found the short "strong" ones trivial.

Train on them with run.py's --data.
"""
import argparse
import glob
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rated  # noqa: E402
import sample  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
V5B = ROOT / "data" / "v5b"
RECORD_KEYS = ("tokens", "n_tokens", "verdict", "min_len", "needs_portals", "metrics")


def bad_tokens():
    out = set()
    for name in ("human-unsolvable", "invalid", "dead"):
        p = ROOT / "data" / "ingame" / f"{name}.jsonl"
        if p.exists():
            out |= {json.loads(l)["tokens"] for l in open(p, encoding="utf-8") if l.strip()}
    return out


def self_levels():
    found = {}
    for f in sorted(glob.glob(str(ROOT / "runs" / "**" / "rated-samples.jsonl"), recursive=True)):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if "outcome" in r and rated.strong(r):
                found.setdefault(r["tokens"], f"strong sample, {Path(f).relative_to(ROOT)}")
    ann = ROOT / "train" / "generated-annotations.jsonl"
    for l in open(ann, encoding="utf-8"):
        r = json.loads(l)
        if (r.get("rating") or 0) >= 2 and r.get("human_verdict") == "solvable" and r.get("tokens"):
            found.setdefault(r["tokens"], "rated 2+ in the workbench")
    bad = bad_tokens()
    return {t: why for t, why in found.items() if t not in bad}


def copies(r):
    """Copies of a 4-5 star Workshop version in ft-ws45-weighted.jsonl."""
    five, n = r.get("stars") == 5, r.get("min_len") or 0
    if r.get("verdict") == "solvable" and 0 < n <= 5:
        return 2 if five else 1
    return (4 if five else 2) + (4 if n >= 10 else 0)


def weighted():
    """Write ft-ws45-weighted.jsonl from ft-ws45-aug-x2.jsonl; the number of lines written."""
    seen, total = set(), 0
    with open(V5B / "ft-ws45-weighted.jsonl", "w", encoding="utf-8", newline="\n") as out:
        for line in open(V5B / "ft-ws45-aug-x2.jsonl", encoding="utf-8"):
            if line in seen:
                continue
            seen.add(line)
            k = copies(json.loads(line))
            out.write(line * k)
            total += k
    print(f"ft-ws45-weighted.jsonl: {len(seen)} versions -> {total} lines")
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--base", type=int, default=None,
                    help="lines of default data to keep (default 23,000, or 15%% of the set with --weighted)")
    ap.add_argument("--weighted", action="store_true", help="also write ft-ws45-weighted.jsonl")
    ap.add_argument("--self-min-len", type=int, default=None,
                    help="own levels need a solution this long (default 8 with --weighted, else any)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.weighted:
        ws = weighted()
        a.base = a.base or round(ws * 0.15 / 0.85)
    a.base = a.base or 23_000
    if a.self_min_len is None:
        a.self_min_len = 8 if a.weighted else 0

    levels = self_levels()
    results = sample.score(list(levels), budget=50_000)
    kept = 0
    with open(V5B / "ft-self.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for t, r in zip(levels, results):
            if r.get("verdict") != "solvable" or (r.get("min_len") or 0) < a.self_min_len:
                continue
            rec = {k: r[k] for k in RECORD_KEYS if k in r}
            rec.update(source="model", why=levels[t])
            f.write((json.dumps(rec) + "\n") * a.copies)
            kept += 1
    print(f"ft-self.jsonl: {kept} of {len(levels)} levels solvable on re-scoring, x{a.copies}")

    files = sorted(glob.glob(str(V5B / "levels-*.jsonl")))
    total = sum(sum(1 for _ in open(p, "rb")) for p in files)
    rng = random.Random(a.seed)
    pick = set(rng.sample(range(total), min(a.base, total)))
    i = 0
    with open(V5B / "ft-base-sample.jsonl", "w", encoding="utf-8", newline="\n") as out:
        for p in files:
            for line in open(p, encoding="utf-8"):
                if i in pick:
                    out.write(line if line.endswith("\n") else line + "\n")
                i += 1
    print(f"ft-base-sample.jsonl: {len(pick)} of {total} lines of {len(files)} files")


if __name__ == "__main__":
    main()
