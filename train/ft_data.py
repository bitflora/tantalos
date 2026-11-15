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

Train on all three with run.py's --data.
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--base", type=int, default=23_000, help="lines of default data to keep")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    levels = self_levels()
    results = sample.score(list(levels), budget=50_000)
    kept = 0
    with open(V5B / "ft-self.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for t, r in zip(levels, results):
            if r.get("verdict") != "solvable":
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
