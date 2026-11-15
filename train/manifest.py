"""The frozen manifest of the generator architecture bake-off (docs/architecture-plan-2026-11-15.md):
what every arm trains and is scored on, written once before any arm's data is derived, so a
later run can be checked against it.

    python train/manifest.py write [--out runs/arch-manifest.json] [--audit]
    python train/manifest.py check RUN [--manifest runs/arch-manifest.json]

`write` records the run.py default data files (data/v5b/levels-*.jsonl) with their sizes and
SHA-256, the vocab, the Workshop holdout, the split seed and share, the training and evaluation
settings run.py uses, the git commit and the tantalos-score binary's hash, plus the known
deviations (v7-fresh-16h, the provisional control, trained on the data before the 2026-11-13
patch). It refuses to overwrite an existing manifest. `--audit` also counts training records
that share exact tokens or a Workshop puzzle (`data.identity`) with the holdout (a few minutes:
it reads every data file).

`check RUN` compares runs/RUN's checkpoint (best.pt, else last.pt) with the manifest: its data
files, vocab, holdout, seed and block, and whether those files still hash the same. Exit 1 on a
difference, listing each.
"""
import argparse
import datetime
import glob
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from data import identity  # noqa: E402

DEFAULT = ROOT / "runs" / "arch-manifest.json"
DATA_PATTERN = "data/v5b/levels-*.jsonl"
VOCAB = "data/v5b/vocab.txt"
HOLDOUT = "data/ws5/test.jsonl"
TANTALOS = "target/release/tantalos-score.exe"
# What run.py start passes train.py for a fresh level run (train/run.py, `train = [...]`), and
# what its evaluation uses (evaluate.py --constrained, rated.py at 0.7).
TRAIN = {"block": 1024, "batch": 16, "eval_every": 2000, "score_n": 64, "lr": 1e-3, "min_lr": 1e-4,
         "warmup": 300, "seed": 0, "val_share": 0.05, "hours": 12}
EVAL = {"decoding": "constrained", "solver_budget": 50_000, "editor_rules": True, "rated_n": 512,
        "rated_temperature": 0.7, "bakeoff_rated_n": 2048, "bakeoff_seeds": [1, 2]}
DEVIATIONS = [
    {"run": "v7-fresh-16h", "role": "provisional control",
     "what": "trained 2026-11-11 on data before the 2026-11-13 Workshop patch (plate throw height, "
             "lasers re-converted low), for 16 hours, not 12; re-evaluated with the current scorer"},
]


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def describe(path):
    p = ROOT / path
    return {"path": path, "bytes": p.stat().st_size, "sha256": sha256(p)}


def data_files():
    return sorted(Path(p).relative_to(ROOT).as_posix() for p in glob.glob(str(ROOT / DATA_PATTERN)))


def git_commit():
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    return out.stdout.strip() + (" (uncommitted changes)" if dirty else "")


def audit(files):
    """Training records sharing exact tokens or a Workshop puzzle with the holdout, per file."""
    tokens, puzzles = set(), set()
    with open(ROOT / HOLDOUT, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            tokens.add(hashlib.blake2b(r["tokens"].encode(), digest_size=8).digest())
            puzzles.add(identity(r))
    out = {"holdout_levels": len(tokens), "files": {}}
    for path in files:
        n = exact = family = 0
        with open(ROOT / path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                n += 1
                exact += hashlib.blake2b(r["tokens"].encode(), digest_size=8).digest() in tokens
                family += identity(r) in puzzles
        out["files"][path] = {"records": n, "exact_overlap": exact, "puzzle_overlap": family}
        print(f"  {path}: {n} records, {exact} exact and {family} same-puzzle matches with the holdout")
    return out


def write(a):
    out = Path(a.out)
    if out.exists():
        sys.exit(f"{out} exists: the manifest is frozen (delete it by hand only to restart the series)")
    files = data_files()
    if not files:
        sys.exit(f"no files match {DATA_PATTERN}")
    print(f"hashing {len(files)} data files, the vocab, the holdout and tantalos-score...")
    manifest = {
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
        "plan": "docs/architecture-plan-2026-11-15.md",
        "commit": git_commit(),
        "data_pattern": DATA_PATTERN,
        "data": [describe(p) for p in files],
        "vocab": describe(VOCAB),
        "holdout": describe(HOLDOUT),
        "tantalos_score": describe(TANTALOS) if (ROOT / TANTALOS).exists() else None,
        "split": "by puzzle (train/data.py identity), seeded by --seed",
        "train": TRAIN,
        "eval": EVAL,
        "deviations": DEVIATIONS,
    }
    if a.audit:
        print("auditing overlap with the holdout...")
        manifest["holdout_audit"] = audit(files)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def check(a):
    import torch

    manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    run = ROOT / "runs" / a.run
    ck_path = next((run / n for n in ("best.pt", "last.pt") if (run / n).exists()), None)
    if ck_path is None:
        sys.exit(f"no checkpoint in {run}")
    args = torch.load(ck_path, map_location="cpu", weights_only=False).get("args", {})
    problems = []

    def rel(p):
        p = Path(p)
        try:
            return (p if p.is_absolute() else (ROOT / p)).resolve().relative_to(ROOT).as_posix()
        except ValueError:
            return p.as_posix()

    want = {d["path"]: d for d in manifest["data"]}
    have = sorted(rel(p) for p in args.get("data", []))
    if have != sorted(want):
        problems.append(f"data: {have} instead of {sorted(want)}")
    if args.get("vocab") and rel(args["vocab"]) != manifest["vocab"]["path"]:
        problems.append(f"vocab: {args['vocab']}")
    if rel(args.get("holdout") or "") != manifest["holdout"]["path"]:
        problems.append(f"holdout: {args.get('holdout')}")
    for key in ("seed", "block"):
        if args.get(key) != manifest["train"][key]:
            problems.append(f"{key}: {args.get(key)} instead of {manifest['train'][key]}")
    for d in [*manifest["data"], manifest["vocab"], manifest["holdout"]]:
        p = ROOT / d["path"]
        if not p.exists() or p.stat().st_size != d["bytes"] or sha256(p) != d["sha256"]:
            problems.append(f"{d['path']} changed since the manifest")
    if problems:
        print(f"{a.run} differs from {a.manifest}:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print(f"{a.run} matches {a.manifest}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--out", default=str(DEFAULT))
    w.add_argument("--audit", action="store_true", help="also count overlap with the holdout (reads all data)")
    c = sub.add_parser("check")
    c.add_argument("run")
    c.add_argument("--manifest", default=str(DEFAULT))
    a = ap.parse_args()
    {"write": write, "check": check}[a.cmd](a)


if __name__ == "__main__":
    main()
