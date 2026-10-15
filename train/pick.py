"""Pick 5 generated levels from a run's best model for play-testing.

Samples solvable levels (sample.py), keeps the canonical ones that aren't trivial (random play
rarely solves them, plan of 5+ moves), spreads the choice from short to long plans, and copies
them into the Puzzle Maker's folder as glados-<run>-<number>.p2c (playtest.py). The pick is
also kept in runs/NAME/playtest/ with picks.json. run.py calls this after every training run.

    python train/pick.py runs/NAME [--n 5] [--samples 80] [--seed 0]
"""
import argparse, json, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def choose(rows, n):
    good = [r for r in rows if r["verdict"] == "solvable" and r.get("canonical")
            and r.get("min_len", 0) >= 5 and r.get("metrics", {}).get("random_success", 1) < 0.1]
    if len(good) < n:  # relax: any solvable level that isn't trivial
        seen = {r["line"] for r in good}
        good += sorted((r for r in rows if r["verdict"] == "solvable" and r["line"] not in seen
                        and r.get("min_len", 0) >= 3), key=lambda r: -r.get("min_len", 0))
    good.sort(key=lambda r: r["min_len"])
    if len(good) <= n:
        return good
    return [good[round(i * (len(good) - 1) / (n - 1))] for i in range(n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run directory, e.g. runs/v4-fresh-2h")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--samples", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run = Path(args.run)
    out = run / "playtest"
    shutil.rmtree(out, ignore_errors=True)
    subprocess.check_call([sys.executable, "-u", str(ROOT / "train" / "sample.py"), str(run / "best.pt"),
                           "--n", str(args.samples), "--verdict", "solvable", "--p2c", "--out", str(out),
                           "--seed", str(args.seed)], cwd=ROOT)
    rows = [json.loads(l) for l in (out / "scores.jsonl").read_text(encoding="utf-8").splitlines()]
    picks = choose(rows, args.n)
    if not picks:
        sys.exit("no solvable levels to pick")
    files = [out / "p2c" / f"{r['line']:05d}.p2c" for r in picks]
    (out / "picks.json").write_text(json.dumps(
        [{"file": f.name, "min_len": r["min_len"], "portal_trips": r.get("metrics", {}).get("portal_trips")}
         for f, r in zip(files, picks)], indent=1), encoding="utf-8")
    subprocess.check_call([sys.executable, str(ROOT / "train" / "playtest.py"), *map(str, files),
                           "--prefix", f"{run.name}-"], cwd=ROOT)
    print(f"picked {len(files)} levels for play-testing; kept in {out}")


if __name__ == "__main__":
    main()
