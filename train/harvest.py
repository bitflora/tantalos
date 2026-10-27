"""Keep a checkpoint's best levels as training data (one round of expert iteration).

    python train/harvest.py runs/v4-warm-3h/best.pt --n 4096
    python train/harvest.py runs/v4-warm-3h/best.pt --n 4096 --cond "<verdict:solvable> <diff:4>"

Samples `--n` levels with the given conditions, scores them with `glados-score score`, and keeps
the ones that are solvable, need portals (the solver can't finish them without firing one)
and are novel: not an exact copy of a level in `--data`, and no same-size level there with
token 4-gram Jaccard similarity of `--max-similarity` or more (the measure evaluate.py's novelty
section uses). Duplicates within the harvest are dropped too.

Kept levels are written in the dataset format (tokens, verdict, min_len, needs_portals, metrics),
plus "source": "model" and the checkpoint, to data/harvest/<run>.jsonl (or --out). train/data.py
reads them like generated levels: they condition as `<src:gen>`, since they're machine-made
and a new source token would change the vocabulary and break warm starts.

To train on them, pass them with the usual data:

    .venv/Scripts/python train/run.py start NAME --hours H --from RUN \\
        --data "data/v5b/levels-*.jsonl" data/harvest/RUN.jsonl --note "..."

Put earlier harvests in `--data` here so later rounds only keep levels new to all of them.

`--keep-unsolved FILE` also writes the sampled levels the solver runs out of budget on
(`uncertain`: editor-valid, not proven unsolvable), once each, to FILE: Goblin's self-play pool
(docs/goblin-plan-2026-10-27.md), levels it can only get by beating the search.

Pause and resume: each chunk's results are saved to `OUT.partial.jsonl` as soon as the chunk is
scored (`OUT` is --out), and nothing else is written until the end. To pause, create the empty
file `OUT.stop`: the harvest finishes its chunk, prints PAUSED and exits with code 3. (Ctrl-C or
a crash loses only the chunk under way.) To resume, run the same command again: it carries on
from the saved chunks (and the seed it saved, unless you pass --seed) if checkpoint, --cond, --chunk,
--temperature, --top-k and --budget are the same. --n may be raised to extend the harvest.
"""

import argparse
import glob
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import sample  # noqa: E402
from evaluate import nearest  # noqa: E402
from seeding import resolve  # noqa: E402

SOLVABLE = "<verdict:solvable>"
# Fields of a `glados-score score` line that make up a dataset record.
RECORD_KEYS = ("tokens", "n_tokens", "verdict", "min_len", "needs_portals", "metrics")


def expand(patterns):
    paths = []
    for p in patterns:
        found = sorted(glob.glob(p))
        if not found and not Path(p).exists():
            sys.exit(f"no data files match {p}")
        paths += found or [p]
    return paths


def known_hashes(paths):
    hashes = set()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    hashes.add(hash(json.loads(line)["tokens"]))
    return hashes


def candidate(r):
    """Solvable and needs a portal (a `glados-score score` result)."""
    if r["verdict"] != "solvable":
        return False
    if "needs_portals" not in r:
        sys.exit("glados-score score didn't report needs_portals: rebuild it (cargo build --release)")
    return bool(r["needs_portals"])


def novel(candidates, data, max_similarity=0.9, counts=None):
    """The candidates that are distinct, not copies of a level in the `data` files, and have no
    same-size level there with Jaccard similarity `max_similarity` or more. Returns (kept, the
    nearest similarity of each non-copy)."""
    counts = Counter() if counts is None else counts
    distinct = list({r["tokens"]: r for r in candidates}.values())
    counts["distinct"] = len(distinct)
    hashes = known_hashes(data)
    fresh = [r for r in distinct if hash(r["tokens"]) not in hashes]
    counts["not copies"] = len(fresh)
    near = nearest(data, [(-1, r["tokens"]) for r in fresh])
    kept = [r for r, s in zip(fresh, near) if s is None or s < max_similarity]
    counts["novel (kept)"] = len(kept)
    return kept, near


def keep(results, data, max_similarity=0.9, counts=None):
    """harvest's rules on `glados-score score` results: solvable, needs portals, novel (`novel`)."""
    return novel([r for r in results if candidate(r)], data, max_similarity, counts)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--n", type=int, default=1024, help="levels to sample")
    ap.add_argument("--cond", default=SOLVABLE, help="condition tokens, space separated")
    ap.add_argument("--data", nargs="+", default=["data/v5b/levels-*.jsonl"],
                    help="levels the harvest must be new to (globs allowed)")
    ap.add_argument("--max-similarity", type=float, default=0.9,
                    help="drop levels this close (Jaccard) to a same-size --data level")
    ap.add_argument("--out", default=None, help="default data/harvest/<run>.jsonl")
    ap.add_argument("--keep-unsolved", default=None, metavar="FILE",
                    help="also write the levels the solver runs out of budget on here (appended)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=1024, help="levels sampled and scored at a time")
    ap.add_argument("--budget", type=int, default=50_000)
    ap.add_argument("--glados", default=str(sample.DEFAULT_GLADOS))
    ap.add_argument("--seed", type=int, default=None,
                    help="default: a fresh, unpredictable seed (printed; pass it back to repeat)")
    args = ap.parse_args()
    args.seed_given = args.seed is not None
    args.seed = resolve(args.seed)

    run = Path(args.checkpoint).parent
    out = Path(args.out) if args.out else Path("data/harvest") / f"{run.name}.jsonl"
    if out.exists():
        sys.exit(f"{out} exists: pick another --out")
    data = expand(args.data)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model, vocab, _ = sample.load_checkpoint(args.checkpoint, device)
    conds = args.cond.split()
    unknown = [t for t in conds if t not in vocab.stoi]
    if unknown:
        sys.exit(f"unknown condition tokens: {unknown}")

    counts = Counter()
    candidates = []
    unsolved = {}
    # Chunks already scored by an earlier run of this command (see the docstring).
    partial = out.with_suffix(".partial.jsonl")
    stop = out.with_suffix(".stop")
    header = {"checkpoint": str(args.checkpoint), "cond": args.cond, "seed": args.seed,
              "chunk": args.chunk, "temperature": args.temperature, "top_k": args.top_k,
              "budget": args.budget}
    saved = {}
    if partial.exists():
        with open(partial, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip().endswith("}")]
        if rows and rows[0].get("header") == header:
            saved = {r["start"]: r for r in rows[1:]}
            print(f"resuming: {len(saved)} chunks of {args.chunk} already scored in {partial}", flush=True)
        elif rows and args.seed_given:
            sys.exit(f"{partial} was made with {rows[0].get('header')}, not {header}: "
                     "delete it to start over, or pass the same options")
        elif rows:
            # No --seed given: take the saved one, and check the rest.
            header["seed"] = rows[0]["header"]["seed"]
            if rows[0]["header"] != header:
                sys.exit(f"{partial} was made with {rows[0]['header']}, not {header}: "
                         "delete it to start over, or pass the same options")
            args.seed = header["seed"]
            saved = {r["start"]: r for r in rows[1:]}
            print(f"resuming with seed {args.seed}: {len(saved)} chunks already scored", flush=True)
    stop.unlink(missing_ok=True)
    if not saved:
        with open(partial, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"header": header}) + "\n")
    t0 = time.time()
    for start in range(0, args.n, args.chunk):
        k = min(args.chunk, args.n - start)
        if start in saved and saved[start]["k"] == k:
            results = saved[start]["results"]
        else:
            torch.manual_seed(args.seed + start)  # each chunk reproducible by itself
            lines = sample.generate(model, vocab, k, conds, (), args.temperature, args.top_k, args.batch)
            results = sample.score(lines, args.glados, args.budget)
            with open(partial, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps({"start": start, "k": k, "results": results}) + "\n")
        for r in results:
            counts["sampled"] += 1
            if r["verdict"] == "unparsed":
                continue
            counts["parsed"] += 1
            if r["verdict"] == "uncertain" and args.keep_unsolved:
                unsolved.setdefault(r["tokens"], r)
            if r["verdict"] != "solvable":
                continue
            counts["solvable"] += 1
            if candidate(r):
                counts["needs portals"] += 1
                candidates.append(r)
        print(f"  {counts['sampled']:>6} sampled in {time.time() - t0:.0f}s: "
              f"{counts['solvable']} solvable, {counts['needs portals']} need portals", flush=True)
        if stop.exists() and start + k < args.n:
            print(f"PAUSED after {counts['sampled']} of {args.n} levels: run the same command to resume",
                  flush=True)
            sys.exit(3)

    # Novelty: distinct, not in --data, and no near neighbour there.
    print(f"checking {len(candidates)} levels against {len(data)} data files", flush=True)
    kept, near = novel(candidates, data, args.max_similarity, counts)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for r in kept:
            rec = {k: r[k] for k in RECORD_KEYS if k in r}
            rec["n_tokens"] = len(r["tokens"].split())
            rec.update(source="model", checkpoint=str(args.checkpoint), cond=args.cond)
            f.write(json.dumps(rec) + "\n")
    if args.keep_unsolved:
        Path(args.keep_unsolved).parent.mkdir(parents=True, exist_ok=True)
        with open(args.keep_unsolved, "a", encoding="utf-8", newline="\n") as f:
            for r in unsolved.values():
                rec = {k: r[k] for k in ("tokens", "verdict") if k in r}
                rec.update(n_tokens=len(r["tokens"].split()), source="model", checkpoint=str(args.checkpoint),
                           cond=args.cond)
                f.write(json.dumps(rec) + "\n")
        print(f"wrote {len(unsolved)} levels the solver ran out of budget on to {args.keep_unsolved}")
    sims = [s for s in near if s is not None]
    print(f"conditions: {args.cond or '(none)'}")
    for name, c in counts.items():
        print(f"  {name:<14} {c:>6}  ({c / max(counts['sampled'], 1):.0%})")
    if sims:
        print(f"nearest-neighbour similarity of the non-copies: mean {sum(sims) / len(sims):.2f}")
    print(f"wrote {len(kept)} levels to {out}")
    partial.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
