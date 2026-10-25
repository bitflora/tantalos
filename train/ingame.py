"""Check scored levels in Portal 2's own editor, after an evaluation.

    python train/ingame.py runs/v2/eval                    # samples.jsonl from evaluate.py
    python train/ingame.py --samples out/x/scores.jsonl    # any `glados-score score` output

Takes every parsed level our checker passes (solvable, uncertain, unsolvable) and a share of
the ones it calls invalid (so rules that are too strict show up too), writes them as .p2c,
checks them in the game (train/portal2.py) and compares. Writes
- <dir>/ingame/results.jsonl: per level, our verdict and problem, the game's verdict and items
- <dir>/ingame/report.md (also appended to <dir>/report.md if there is one)
- data/ingame/disagreements.jsonl: levels where we and the game disagree, for fixing rules
- data/ingame/invalid.jsonl: game-invalid levels as training records (verdict "invalid"). The
  game's word is final, so retrain.sh trains on this file without relabelling it.

If it stops early (the game crashed too often, ...), run it again with --resume to keep the
levels already checked.

If the game can't be used (already running, a different build, ...) it says so in the
report and exits 0, so an unattended run carries on.
"""
import argparse, collections, datetime, json, random, re, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import portal2  # noqa: E402
from seeding import resolve  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PASSED = ("solvable", "uncertain", "unsolvable")


def shape(problem):
    """A problem message with positions and indices dropped, for grouping."""
    return re.sub(r"\[[^\]]*\]|\b\d+\b", "#", problem)


def append_new(path, records, key="tokens"):
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if path.exists():
        seen = {json.loads(l)[key] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    new = [r for r in records if r[key] not in seen]
    with open(path, "a", encoding="utf-8") as f:
        for r in new:
            f.write(json.dumps(r) + "\n")
    return len(new)


def summarize(rows, source):
    ours = lambda r: "passed" if r["verdict"] in PASSED else "invalid"
    table = collections.Counter((ours(r), r["game"]["verdict"]) for r in rows)
    missed = collections.Counter()
    strict = collections.Counter()
    for r in rows:
        g = r["game"]["verdict"]
        if g == "invalid" and ours(r) == "passed":
            missed[" ".join(sorted({i["name"] for i in r["game"]["invalid_items"]}))] += 1
        if g == "valid" and ours(r) == "invalid":
            for p in r.get("problem", "").split("; "):
                strict[shape(p)] += 1
    passed = [r for r in rows if ours(r) == "passed" and r["game"]["verdict"] in ("valid", "invalid")]
    solvable = [r for r in passed if r["verdict"] == "solvable"]
    rate = lambda rs: sum(r["game"]["verdict"] == "valid" for r in rs) / max(len(rs), 1)
    lines = [
        "## In-game editor check",
        "",
        f"Source: `{source}`. {len(rows)} levels checked in Portal 2's Puzzle Maker.",
        "",
        f"- **Game-valid, of levels our checker passes: {rate(passed):.1%}** ({len(passed)} levels)",
        f"- Game-valid, of those we call solvable: {rate(solvable):.1%} ({len(solvable)} levels)",
        "",
        "| ours \\ game | valid | invalid | other |",
        "|---|---|---|---|",
    ]
    for o in ("passed", "invalid"):
        other = sum(n for (oo, g), n in table.items() if oo == o and g not in ("valid", "invalid"))
        lines.append(f"| {o} | {table[(o, 'valid')]} | {table[(o, 'invalid')]} | {other} |")
    lines += ["", "Missed by our checker (game's invalid items):", ""]
    lines += [f"- {n} × {k}" for k, n in missed.most_common(12)] or ["- none"]
    lines += ["", "Too strict (our problem, game says valid):", ""]
    lines += [f"- {n} × {k}" for k, n in strict.most_common(12)] or ["- none"]
    return "\n".join(lines) + "\n"


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # the report has "×"; Windows consoles default to cp1252
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", nargs="?", help="evaluation directory holding samples.jsonl")
    ap.add_argument("--samples", help="scored JSONL (default <dir>/samples.jsonl)")
    ap.add_argument("--out", help="output directory (default <dir>/ingame)")
    ap.add_argument("--reject-share", type=float, default=0.2,
                    help="share of the levels our checker calls invalid to check too")
    ap.add_argument("--limit", type=int, default=None, help="at most this many levels")
    ap.add_argument("--glados", default=str(ROOT / "target" / "release" / "glados-score.exe"))
    ap.add_argument("--store", default=str(ROOT / "data" / "ingame"))
    ap.add_argument("--seed", type=int, default=None,
                    help="default: a fresh, unpredictable seed (printed; pass it back to repeat)")
    ap.add_argument("--resume", action="store_true",
                    help="keep the levels already checked in <out>/game.jsonl (same samples and options)")
    ap.add_argument("--max-crashes", type=int, default=None,
                    help="give up after this many game crashes (default 10, or 5%% of the levels if more)")
    ap.add_argument("--no-launch", action="store_true",
                    help="use a game you started with -insecure -netconport PORT -netconpassword PW")
    ap.add_argument("--port", type=int, help="with --no-launch: its -netconport")
    ap.add_argument("--password", default="", help="with --no-launch: its -netconpassword")
    a = ap.parse_args()
    a.seed = resolve(a.seed)
    if not a.dir and not a.samples:
        ap.error("give an evaluation directory or --samples")
    samples = Path(a.samples or Path(a.dir) / "samples.jsonl")
    out = Path(a.out or (Path(a.dir) if a.dir else samples.parent) / "ingame")
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(a.seed)
    records, seen = [], set()
    for l in open(samples, encoding="utf-8"):
        r = json.loads(l)
        if "tokens" in r and r["tokens"] not in seen:
            seen.add(r["tokens"])
            records.append(r)
    picked = [r for r in records if r["verdict"] in PASSED or
              (r["verdict"] == "invalid" and rng.random() < a.reject_share)]
    if a.limit:
        rng.shuffle(picked)
        picked = picked[: a.limit]
    print(f"{len(records)} distinct parsed levels; checking {len(picked)}", flush=True)
    if not picked:
        return

    p2c = out / "p2c"
    files = [p2c / f"{i:05}.p2c" for i in range(1, len(picked) + 1)]
    results_path = out / "game.jsonl"
    if not (a.resume and all(f.exists() for f in files)):
        if a.resume:
            print("can't resume: the .p2c files don't match these levels; starting over", flush=True)
        if p2c.exists():
            for f in p2c.glob("*.p2c"):
                f.unlink()
        subprocess.run([str(Path(a.glados).resolve()), "score", "--budget", "1", "--p2c-all", str(p2c)],
                       input="\n".join(r["tokens"] for r in picked) + "\n",
                       capture_output=True, text=True, check=True)
        results_path.unlink(missing_ok=True)
    report_path = out / "report.md"
    try:
        portal2.check(files, results_path, launch_game=not a.no_launch, max_crashes=a.max_crashes,
                      progress=lambda s: print(s, flush=True), port=a.port, password=a.password)
    except portal2.GameUnavailable as e:
        text = f"## In-game editor check\n\nSkipped: Portal 2 unavailable ({e}).\n"
        report_path.write_text(text, encoding="utf-8")
        print(text)
        if a.dir and (Path(a.dir) / "report.md").exists():
            with open(Path(a.dir) / "report.md", "a", encoding="utf-8") as f:
                f.write("\n" + text)
        return

    game = {Path(j["file"]).name: j for j in map(json.loads, open(results_path, encoding="utf-8"))}
    rows = []
    for f, r in zip(files, picked):
        g = game.get(f.name)
        if g is None:
            continue
        keep = {k: r[k] for k in ("tokens", "verdict", "problem", "min_len", "metrics", "setting") if k in r}
        rows.append({**keep, "game": {k: g[k] for k in ("verdict", "invalid_items") if k in g}})
    with open(out / "results.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    when = datetime.datetime.now().isoformat(timespec="seconds")
    store = Path(a.store)
    disagree = [{**r, "source": str(samples), "checked": when} for r in rows
                if (r["verdict"] in PASSED) == (r["game"]["verdict"] == "invalid")
                and r["game"]["verdict"] in ("valid", "invalid")]
    invalid = [{"tokens": r["tokens"], "verdict": "invalid", "game_items": r["game"]["invalid_items"],
                "source": str(samples), "checked": when}
               for r in rows if r["game"]["verdict"] == "invalid"]
    n_dis = append_new(store / "disagreements.jsonl", disagree)
    n_inv = append_new(store / "invalid.jsonl", invalid)

    text = summarize(rows, samples)
    text += f"\n{n_dis} new disagreements and {n_inv} new game-invalid levels added to `{store}`.\n"
    report_path.write_text(text, encoding="utf-8")
    print(text)
    if a.dir and (Path(a.dir) / "report.md").exists():
        with open(Path(a.dir) / "report.md", "a", encoding="utf-8") as f:
            f.write("\n" + text)
    if a.dir and not a.out and Path(a.dir).name == "eval":
        import record

        record.safely(Path(a.dir).parent)


if __name__ == "__main__":
    main()
