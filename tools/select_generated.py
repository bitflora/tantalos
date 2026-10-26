"""The generated levels worth keeping: solvable, state-changing, without near-copies.

    python tools/select_generated.py

Reads data/v5b/levels-[1-5].jsonl and data/v5b/levels-mutants*.jsonl (about 850 MB) and writes
data/v6/generated.jsonl, in this order:

1. Keep a record only if `verdict == "solvable"` and its shortest solution uses at least 2
   state-changing mechanics and at most 1 trick (`rated.mechanics`): levels that teach a trick
   rather than change the room's state are what the Workshop corpus is meant to replace.
2. Drop exact duplicates (same `tokens`).
3. Drop near-copies: group records by their first four tokens (`size X Y Z`); within a group,
   keep a record only if its 4-gram Jaccard similarity to every record already kept in that
   group is below 0.9.

(a copy of this lives in data/v6/ too, but data/ is ignored by git: this is the tracked one)
"""
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# This file is named after the stdlib `select` module: don't let its directory shadow that.
HERE = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0] or ".").resolve() == HERE:
    sys.path.pop(0)

ROOT = HERE
while not (ROOT / "train" / "rated.py").exists():  # works from data/v6/ and tools/
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "train"))
import rated  # noqa: E402
from evaluate import jaccard, shingles  # noqa: E402

OUT = ROOT / "data" / "v6" / "generated.jsonl"
NEAR = 0.9


def sources():
    d = ROOT / "data" / "v5b"
    return sorted(d.glob("levels-[1-5].jsonl")) + sorted(d.glob("levels-mutants*.jsonl"))


def step1(files):
    """(src, tokens, line) of the records worth keeping, and the counts by source file."""
    rows, read, solvable, kept = [], Counter(), Counter(), Counter()
    for p in files:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                read[p.name] += 1
                r = json.loads(line)
                if r.get("verdict") != "solvable":
                    continue
                solvable[p.name] += 1
                state, tricks = rated.mechanics(r.get("metrics"))
                if state is None or state < 2 or tricks > 1:
                    continue
                kept[p.name] += 1
                rows.append((p.name, r["tokens"], line if line.endswith("\n") else line + "\n"))
        print(f"  {p.name:28} {read[p.name]:7} read {solvable[p.name]:7} solvable {kept[p.name]:6} kept")
    print(f"step 1: {len(rows)} of {sum(solvable.values())} solvable records kept "
          f"({100 * len(rows) / max(sum(solvable.values()), 1):.1f}%)")
    return rows, kept


def step2(rows):
    """Without exact duplicate token lines, and how many each source lost."""
    seen, out, dropped = set(), [], Counter()
    for src, tokens, line in rows:
        if tokens in seen:
            dropped[src] += 1
            continue
        seen.add(tokens)
        out.append((src, tokens, line))
    print(f"step 2: {len(out)} left; exact duplicates dropped: "
          f"{dict(sorted(dropped.items())) or 'none'}")
    return out


def step3(rows):
    """Without near-copies within a size group, and how many each source lost."""
    groups = defaultdict(list)
    for i, (_, tokens, _) in enumerate(rows):
        groups[" ".join(tokens.split(maxsplit=4)[:4])].append(i)
    keep = [False] * len(rows)
    copied = Counter()
    t0 = time.time()
    for n, (key, idxs) in enumerate(groups.items(), 1):
        kept_sh = []
        index = defaultdict(list)  # a shingle -> the kept records of this group that have it
        for i in idxs:
            src, tokens, _ = rows[i]
            s = shingles(tokens)
            hits = Counter(j for h in s for j in index.get(h, ()))
            if any(c >= NEAR * (len(s) + len(kept_sh[j])) / (1 + NEAR)
                   and jaccard(s, kept_sh[j]) >= NEAR for j, c in hits.items()):
                copied[src] += 1
                continue
            keep[i] = True
            for h in s:
                index[h].append(len(kept_sh))
            kept_sh.append(s)
        if len(idxs) > 500:
            print(f"  group {key!r}: {len(idxs)} -> {sum(keep[i] for i in idxs)} "
                  f"in {time.time() - t0:.1f}s")
    out = [(src, line) for (src, _, line), k in zip(rows, keep) if k]
    print(f"step 3: {len(out)} left; near-copies dropped: "
          f"{dict(sorted(copied.items())) or 'none'} ({time.time() - t0:.0f}s)")
    return out


def main():
    files = sources()
    if not files:
        sys.exit("no data/v5b/levels-*.jsonl to read")
    rows, _ = step1(files)
    rows = step2(rows)
    rows = step3(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for _, line in rows:
            f.write(line)
    tmp.replace(OUT)
    by_src = Counter(src for src, _ in rows)
    print(f"wrote {len(rows)} records to {OUT}: {dict(sorted(by_src.items()))}")


if __name__ == "__main__":
    main()
