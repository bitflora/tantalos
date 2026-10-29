"""Keep train/runs.csv (and train/goblin-runs.csv): one row per training run, with its
settings, results and notes.

    python train/record.py runs/v3c                       # add or refresh a run's row
    python train/record.py runs/v3c --note "lr too high"  # and set its notes

train.py, evaluate.py and ingame.py call this when they finish, so a run's row fills in as it
goes: settings and losses after training, validity after evaluating, game-valid share after
the in-game check. Refreshing a row keeps its notes. A run started with `train.py --note` gets
that note if its row has none yet.

A Goblin run (train.py --task goblin; Goblin is the net that reads a level and writes its
plan, called the solver net and task "solve" before 2026-10-27) gets its row in
train/goblin-runs.csv instead, with its own result columns (GOBLIN_FIELDS): train_* from the
eval during training, eval_* from goblin_eval.py's headline set (test-gen; eval_solved is the greedy solved share, the main
score), and the greedy solved shares of its other sets.

Everything else comes from the run directory: <run>/run.json (written by train.py at the
start: start time, commit, note), the args saved in last.pt, log.jsonl, train.log,
eval/eval.json and eval/ingame/report.md. Runs from before run.json existed take their start
time from when train.log (or the directory) was created, and have no commit.
"""

import argparse
import csv
import datetime
import json
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "train" / "runs.csv"
FIELDS = [
    "run",
    "started",
    "ended",
    "minutes",
    "commit",
    "data",
    "init",
    "model",
    "block",
    "batch",
    "lr",
    "min_lr",
    "warmup",
    "steps",
    "best_val",
    "final_val",
    "tok_s",
    "train_parsed",
    "train_solvable",
    "eval_parsed",
    "eval_solvable",
    "eval_invalid",
    "holdout_best",
    "holdout_final",
    "rated_yield",
    "rated_strong",
    "rated_gap",
    "ingame_checked",
    "game_valid_of_passed",
    "game_valid_of_solvable",
    "notes",
]
GOBLIN_CSV = ROOT / "train" / "goblin-runs.csv"
SETTINGS = FIELDS[: FIELDS.index("train_parsed")]
GOBLIN_FIELDS = SETTINGS + [
    "train_parsed",
    "train_solved",
    "eval_n",
    "eval_parsed",
    "eval_solved",
    "eval_pass_at_k",
    "eval_optimal",
    "ws_solved",
    "published_solved",
    "published_liberal_solved",
    "v6_solved",
    "gen_solved",
    "mut_solved",
    "hard_solved",
    "probe_solved",
    "frontier_solved",
    "frontier_liberal_solved",
    "decoding",
    "notes",
]
# goblin_eval.py's other sets, and the column of each one's greedy solved share. (The headline
# set's goes in eval_solved too: test-ws+v6 on data/goblin, test-gen on data/plans.)
GOBLIN_SETS = {"test-ws": "ws_solved", "published": "published_solved",
               "published-liberal": "published_liberal_solved", "test-v6": "v6_solved",
               "test-gen": "gen_solved", "test-mut": "mut_solved", "test-hard": "hard_solved",
               "probe": "probe_solved", "test-frontier": "frontier_solved",
               "test-frontier-liberal": "frontier_liberal_solved"}


GOBLIN = "goblin"


def task_name(task):
    """A run's task as it's called now: "goblin" for the old name "solve" (runs, checkpoints and
    eval files from before 2026-10-27 keep it), "levels" when there is none."""
    return GOBLIN if task in (GOBLIN, "solve") else (task or "levels")


def table(task):
    """(CSV file, its fields) for a run of `task`."""
    return (GOBLIN_CSV, GOBLIN_FIELDS) if task_name(task) == GOBLIN else (CSV, FIELDS)


def key(run):
    run = Path(run).resolve()
    try:
        return run.relative_to(ROOT).as_posix()
    except ValueError:
        return run.as_posix()


def created(path):
    # On Windows st_ctime is when the file was created.
    return datetime.datetime.fromtimestamp(path.stat().st_ctime)


def stamp(t):
    return t.strftime("%Y-%m-%d %H:%M")


def pct(x):
    return f"{100 * x:.1f}" if x is not None else ""


def checkout(run):
    """The checkout a run was trained from (the directory holding its runs/), which its
    relative paths start from."""
    run = Path(run).resolve()
    return next((p.parent for p in [run, *run.parents] if p.name == "runs"), ROOT)


def where(path, base):
    """`path` as seen from this checkout, if it still exists; else as it was given (a run in a
    worktree means its own data/v3, not this checkout's)."""
    p = base / path
    return key(p) if p.exists() else Path(path).as_posix()


def short_paths(paths, base):
    """Data files, with repeats counted: a, b, c×5."""
    out, counts = [], {}
    for p in paths:
        p = where(p, base)
        if p not in counts:
            out.append(p)
        counts[p] = counts.get(p, 0) + 1
    return " ".join(p + (f"×{counts[p]}" if counts[p] > 1 else "") for p in out)


def checkpoint_info(run):
    """Saved args and model shape, from last.pt (or best.pt)."""
    for name in ("last.pt", "best.pt"):
        if (run / name).exists():
            import torch

            ck = torch.load(run / name, map_location="cpu", weights_only=False)
            return ck.get("args", {}), ck.get("model_config", {}), ck.get("step")
    return {}, {}, None


def ingame_report(path):
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    row = {}
    if m := re.search(r"(\d+) levels checked", text):
        row["ingame_checked"] = m.group(1)
    if m := re.search(r"Game-valid, of levels our checker passes: ([\d.]+)%", text):
        row["game_valid_of_passed"] = m.group(1)
    if m := re.search(r"Game-valid, of those we call solvable: ([\d.]+)%", text):
        row["game_valid_of_solvable"] = m.group(1)
    return row


def measure(run):
    """Everything this run's directory says about it, as CSV fields."""
    run = Path(run)
    row = {"run": key(run)}
    meta = json.loads((run / "run.json").read_text()) if (run / "run.json").exists() else {}

    log_path = run / "train.log"
    if "started" in meta:
        start = datetime.datetime.fromisoformat(meta["started"])
    elif log_path.exists():
        start = created(log_path)
    else:
        start = created(run)
    row["started"] = stamp(start)
    last = next((run / n for n in ("last.pt", "best.pt") if (run / n).exists()), None)
    if last:
        end = datetime.datetime.fromtimestamp(last.stat().st_mtime)
        row["ended"] = stamp(end)
        row["minutes"] = round((end - start).total_seconds() / 60)
    row["commit"] = meta.get("commit", "")

    args, cfg, step = checkpoint_info(run)
    # Which CSV the row goes in: the checkpoint says, or run.json before there is one.
    row["task"] = task_name(args.get("task") or meta.get("task"))
    if args:
        base = checkout(run)
        row["data"] = short_paths(args.get("data", []), base)
        row["init"] = where(args["init"], base) if args.get("init") else ""
        row["batch"] = args.get("batch", "")
        row["lr"] = args.get("lr", "")
        row["min_lr"] = args.get("min_lr", "")
        row["warmup"] = args.get("warmup", "")
        if args.get("away"):
            row["data"] += f" (away loss, weight {args.get('away_weight')})"
    if cfg:
        row["model"] = f"{cfg['n_layer']}x{cfg['n_head']}x{cfg['n_embd']}"
        row["block"] = cfg["block_size"]

    entries = []
    if (run / "log.jsonl").exists():
        entries = [json.loads(l) for l in (run / "log.jsonl").read_text().splitlines() if l.strip()]
    if entries:
        # A resumed run appends to the same log; the last entry is the newest.
        row["steps"] = entries[-1]["step"]
        row["best_val"] = f"{min(e['val_loss'] for e in entries):.4f}"
        row["final_val"] = f"{entries[-1]['val_loss']:.4f}"
        held = [e["holdout_loss"] for e in entries if "holdout_loss" in e]
        if held:  # loss on held-out Workshop levels (train.py --holdout)
            row["holdout_best"] = f"{min(held):.4f}"
            row["holdout_final"] = f"{held[-1]:.4f}"
        s = entries[-1].get("samples")
        if s:
            row["train_parsed"] = pct(s["parsed"])
            row["train_solvable"] = pct(s["solvable"])
        s = entries[-1].get("goblin") or entries[-1].get("solve")
        if s:
            row["train_parsed"] = pct(s["parsed"])
            row["train_solved"] = pct(s["solved"])
    elif step is not None:
        row["steps"] = step

    if log_path.exists():
        speeds = [
            float(m.group(1).replace(",", ""))
            for m in re.finditer(r"^step .* ([\d,]+) tok/s$", log_path.read_text(errors="replace"), re.M)
        ]
        if speeds:
            row["tok_s"] = round(statistics.median(speeds))

    ev = run / "eval" / "eval.json"
    if ev.exists():
        e = json.loads(ev.read_text())
        if task_name(e.get("task")) == GOBLIN:
            h = e["headline"]
            row["eval_n"] = e["sets"][h["set"]]["n"]
            row["eval_parsed"] = pct(h["parsed"])
            row["eval_solved"] = pct(h["solved"])
            row["eval_pass_at_k"] = pct(h["pass_at_k"])
            row["eval_optimal"] = pct(h["optimal"])
            row["decoding"] = "constrained" if e.get("constrained") else "free"
            for name, field in GOBLIN_SETS.items():
                if name in e["sets"]:
                    row[field] = pct(e["sets"][name]["solved"])
        else:
            v = e["validity"]
            row["eval_parsed"] = pct(v["parsed"])
            row["eval_solvable"] = pct(v["solvable"])
            row["eval_invalid"] = pct(v["verdicts"].get("invalid", 0) / v["n"])
            if e.get("rated"):  # train/rated.py: per 1,000 samples, and points
                row["rated_yield"] = f"{e['rated']['yield_per_1000']:.0f}"
                row["rated_strong"] = f"{e['rated']['strong_per_1000']:.0f}"
                if e["rated"]["profile_gap"] is not None:
                    row["rated_gap"] = f"{e['rated']['profile_gap']:.1f}"
    row.update(ingame_report(run / "eval" / "ingame" / "report.md"))
    return row, meta.get("note", "")


def load_rows(path=CSV):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_rows(rows, path=CSV, fields=FIELDS):
    rows.sort(key=lambda r: (r["started"], r["run"]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def update(run, note=None):
    """Add or refresh `run`'s row (in runs.csv, or goblin-runs.csv for a Goblin run). `note`
    replaces its notes; without one they're kept. Returns the row, with `task` added."""
    row, run_note = measure(run)
    path, fields = table(row["task"])
    rows = load_rows(path)
    old = next((r for r in rows if r["run"] == row["run"]), None)
    if old is None:
        old = {f: "" for f in fields}
        rows.append(old)
    notes = old.get("notes", "")
    old.update({f: row.get(f, "") for f in fields})
    old["notes"] = note if note is not None else (notes or run_note)
    save_rows(rows, path, fields)
    return {**old, "task": row["task"]}


def write_meta(out, note, task="levels"):
    """<out>/run.json at the start of training: when, which commit, the task, and the note if any."""
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""

    commit = git("rev-parse", "--short", "HEAD")
    if commit and git("status", "--porcelain", "--untracked-files=no"):
        commit += "+dirty"
    meta = {
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "commit": commit,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "task": task,
    }
    if note:
        meta["note"] = note
    (Path(out) / "run.json").write_text(json.dumps(meta, indent=2) + "\n")


def safely(run, **kw):
    """update() for the end of a script: a bad record is reported, never fatal."""
    try:
        r = update(run, **kw)
        print(f"recorded {key(run)} in {table(r['task'])[0].relative_to(ROOT).as_posix()}")
    except Exception as e:  # noqa: BLE001
        print(f"couldn't record {run}: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories (holding last.pt / log.jsonl)")
    ap.add_argument("--note", default=None, help="set the notes (for one run)")
    a = ap.parse_args()
    if a.note is not None and len(a.runs) > 1:
        ap.error("--note goes with one run")
    for run in a.runs:
        r = update(run, a.note)
        score = (f"eval solved {r['eval_solved'] or '-'}" if r["task"] == GOBLIN
                 else f"eval solvable {r['eval_solvable'] or '-'}")
        print(f"{r['run']}: {r['steps']} steps, val {r['best_val']}, {score}")


if __name__ == "__main__":
    main()
