"""Pick 5 generated levels from a run's best model for play-testing.

Samples solvable levels (sample.py), keeps the canonical ones that aren't trivial (random play
rarely solves them, plan of 5+ moves), spreads the choice from short to long plans, checks the
candidates in Portal 2's editor (portal2.py) so only levels the game accepts are handed out,
and copies them into the Puzzle Maker's folder as glados-<run>-<number>.p2c (playtest.py). The
pick is also kept in runs/NAME/playtest/ with picks.json (each pick's "game": valid, or
unchecked when the game couldn't be used). run.py calls this after every training run.

    python train/pick.py runs/NAME [--n 5] [--samples 80] [--seed 0] [--no-game]
"""
import argparse, json, shutil, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import portal2  # noqa: E402

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


def game_valid(files, out):
    """The subset of files Portal 2's editor accepts, or None if the game couldn't be used."""
    results = out / "game.jsonl"
    results.unlink(missing_ok=True)
    try:
        portal2.check(files, results, launch_game=True, progress=lambda s: print(s, flush=True))
    except portal2.GameUnavailable as e:
        print(f"in-game check of the picks skipped: Portal 2 unavailable ({e})", flush=True)
        return None
    valid = {Path(j["file"]).name for j in map(json.loads, open(results, encoding="utf-8"))
             if j["verdict"] == "valid"}
    return [f for f in files if f.name in valid]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run directory, e.g. runs/v4-fresh-2h")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--samples", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-game", action="store_true", help="don't check the picks in Portal 2")
    args = ap.parse_args()

    run = Path(args.run)
    out = run / "playtest"
    shutil.rmtree(out, ignore_errors=True)
    subprocess.check_call([sys.executable, "-u", str(ROOT / "train" / "sample.py"), str(run / "best.pt"),
                           "--n", str(args.samples), "--verdict", "solvable", "--p2c", "--out", str(out),
                           "--seed", str(args.seed)], cwd=ROOT)
    rows = [json.loads(l) for l in (out / "scores.jsonl").read_text(encoding="utf-8").splitlines()]
    file_of = lambda r: out / "p2c" / f"{r['line']:05d}.p2c"
    candidates = choose(rows, args.n * 3)
    if not candidates:
        sys.exit("no solvable levels to pick")
    valid = None if args.no_game else game_valid([file_of(r) for r in candidates], out)
    if valid is None:
        picks, game = choose(rows, args.n), "unchecked"
    else:
        names = {f.name for f in valid}
        picks, game = choose([r for r in candidates if file_of(r).name in names], args.n), "valid"
        print(f"{len(valid)} of {len(candidates)} candidates are valid in Portal 2's editor", flush=True)
        if not picks:
            sys.exit("Portal 2's editor rejected every candidate: nothing to play-test")
    files = [file_of(r) for r in picks]
    (out / "picks.json").write_text(json.dumps(
        [{"file": f.name, "min_len": r["min_len"], "portal_trips": r.get("metrics", {}).get("portal_trips"),
          "game": game} for f, r in zip(files, picks)], indent=1), encoding="utf-8")
    subprocess.check_call([sys.executable, str(ROOT / "train" / "playtest.py"), *map(str, files),
                           "--prefix", f"{run.name}-"], cwd=ROOT)
    print(f"picked {len(files)} levels for play-testing ({game} in Portal 2); kept in {out}")


if __name__ == "__main__":
    main()
