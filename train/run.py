"""Start, watch and stop training runs with one command each. Meant to need no thought.

    .venv/Scripts/python train/run.py start v4-a --hours 2 --note "first v4 run, fresh"
    .venv/Scripts/python train/run.py start v4-b --hours 1 --from v4-a --note "v4-a continued"
    .venv/Scripts/python train/run.py status          # the newest run (or: status v4-a)
    .venv/Scripts/python train/run.py wait v4-a       # blocks until it ends, then prints status
    .venv/Scripts/python train/run.py stop v4-a
    .venv/Scripts/python train/run.py pause v4-a      # stop training for now, keep its place
    .venv/Scripts/python train/run.py resume v4-a     # carry on where it paused
    .venv/Scripts/python train/run.py start v4-c --hours 10 --pause-after 5 --note "..."  # pauses itself at 5 h
    .venv/Scripts/python train/run.py list            # recent rows of train/runs.csv
    .venv/Scripts/python train/run.py note v4-a "what it showed"
    .venv/Scripts/python train/run.py ingame v4-a     # redo the Portal 2 steps if they were skipped

`start` builds the target/release/glados-*.exe binaries, then returns: the run carries on in the background,
detached, and survives the terminal or session that started it. Every run does every step, in
order, with nothing to remember:
  1. train (train.py) for --hours
  2. evaluate best.pt (evaluate.py --constrained: levels written from the tokens `glados-score
     grammar` allows, clear of the editor's rules; --quick runs decode freely)
  3. check the evaluation's levels in Portal 2's own editor (ingame.py, about 15 minutes). It
     launches the game itself, unless the game is already running (someone playing): then it
     is skipped, and `status` says INCOMPLETE until `run.py ingame NAME` redoes it.
  4. train best.pt away from every level Portal 2 has rejected (data/ingame/invalid.jsonl, x5,
     labelled invalid) and every level a play-tester found unsolvable though our solver passed
     it (data/ingame/human-unsolvable.jsonl, x5, labelled unsolvable), and every level of an
     evaluation whose exit can never be reached (glados-score's "dead": walls part entry and
     exit, or the exit needs a laser with no emitter or a button with no cube it takes;
     collected into data/ingame/dead.jsonl, x5, labelled unsolvable) for 15 minutes into
     runs/NAME/away, evaluate that and check it in Portal 2 too. It is the final model unless
     its EVAL solvable share is over 3 points lower (then away/WORSE is written and best.pt
     stays final).
  5. stage a blind A/B play-test (abtest.py): 4 levels from the final model and 4 from its
     parent (--from), or from rated Workshop levels for a fresh run, checked in Portal 2 too,
     shuffled into the Puzzle Maker folder as glados-ab-NAME-NN; the verdicts go in
     runs/NAME/playtest/results.csv and `abtest.py score NAME --out runs/NAME/playtest` unblinds them
  6. record the run (and NAME/away) in train/runs.csv
Pausing a Portal 2 check (steps 3 and 4): `pause NAME` while it checks has ingame.py finish the
level in hand, quit the game (its editor autosaves put back) and exit; `resume NAME` carries the
check on (ingame.py --resume keeps the levels already checked), then does the remaining steps.
Pausing: `pause NAME` (while it trains) has train.py save last.pt and exit; the run is then
PAUSED and holds neither the GPU nor the machine. `resume NAME` rebuilds the binaries and trains
on for the time still owed (--hours less the training already done), continuing the weights,
optimizer and learning-rate schedule, then does the remaining steps. So a run paused halfway comes out
like one run of the same length, not like two chained runs (which each restart the schedule).
Only the batches drawn differ.
Everything lands in runs/NAME: status.json (what it's doing now), run.log (each phase),
train.log, evaluate.log, ingame.log, pick.log, eval/report.md.

Settings: data/v5b (all levels-*.jsonl: generated, mutants, and the Workshop levels as their
8 mirrored/turned versions x2 since 2026-10-26, levels-workshop-aug-x2.jsonl, from data/ws6 since
2026-11-06 (the ws5 one is workshop-ws5-aug-x2.jsonl, outside the pattern); the old x20 copies
are workshop-x20.jsonl, outside the pattern) unless --data; block 1024, batch 16. A fresh run uses
lr 1e-3 down to 1e-4; one continuing another (--from) uses 2e-4 down to 2e-5, as the warm runs
in runs.csv did. Anything after `--` goes to train.py as is.

Goblin runs (`start NAME --task goblin`) train the second net, Goblin, which reads a level and
writes its plan (called the solver net, task "solve", before 2026-10-27; `--task solve` and old
runs still work). They train on data/goblin (train/goblin_data.py makes it; data/plans until it
exists). Their steps: train, evaluate best.pt (goblin_eval.py --constrained, about 75 minutes:
plans for held-out levels written from legal steps only, replayed by `glados-score check`;
`--quick` runs decode freely, about 15 minutes, unless `--constrained-eval` is given), record in train/goblin-runs.csv (`list --task
goblin` shows them). No Portal 2 steps, no training away, no play-test picks. The main score is
EVAL solved %: the share of held-out levels whose greedy plan solves. --from a level-writing run
starts from its weights with the fresh learning rates (a new task); --from a Goblin run continues it.

Quick runs (`start NAME --quick --from OLD --hours H`, for train/broadcast.py's rounds) fine-tune
OLD at a low rate (1e-4 down to 1e-5), evaluate on a smaller sample (level runs: no generator
timing, fewer per-condition levels; Goblin runs: no published set, pass@4), and skip the Portal 2,
training-away and picking steps. Then a guard: if the EVAL score is more than 3 points (level
runs; 5 for Goblin runs) under OLD's, WORSE is written and OLD stays the final model.
With `--constrained-eval`, Goblin uses the full evaluation on the parent's saved levels;
the guard requires matching settings, headline levels and parent checkpoint, retaining OLD
if no compatible baseline exists. This includes the probe and frontier sets.

Only one run at a time: they share the GPU. While a broadcast (train/broadcast.py) is going,
only its own runs (--broadcast NAME) start.
"""

import argparse
import csv
import ctypes
import datetime
import glob
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
sys.path.insert(0, str(ROOT / "train"))
from record import GOBLIN, task_name  # noqa: E402

PY = sys.executable
GLADOS = ROOT / "target" / "release" / "glados-score.exe"
ACTIVE = ("starting", "training", "evaluating", "ingame", "away", "picking")
PAUSED_EXIT = 3  # train.py's exit code when a PAUSE file stopped it (train.py PAUSED_EXIT)
NO_WINDOW = 0x08000000  # CREATE_NO_WINDOW: no console pops up, none is needed
DETACH = NO_WINDOW | 0x00000200  # | CREATE_NEW_PROCESS_GROUP
BREAKAWAY = 0x01000000  # CREATE_BREAKAWAY_FROM_JOB: outlive the session that started it


def now():
    return datetime.datetime.now()


def hhmm(iso):
    return datetime.datetime.fromisoformat(iso).strftime("%H:%M")


def run_dir(name):
    p = Path(name)
    if p.parts and p.parts[0] == "runs":
        p = Path(*p.parts[1:])
    return RUNS / p


def read_status(d):
    try:
        return json.loads((d / "status.json").read_text())
    except (OSError, ValueError):
        return None


def write_status(d, **changes):
    s = read_status(d) or {}
    s.update(changes)
    (d / "status.json").write_text(json.dumps(s, indent=2) + "\n")
    return s


def alive(pid):
    if sys.platform != "win32":
        try:
            import os

            os.kill(pid, 0)
            return True
        except OSError:
            return False
    k = ctypes.windll.kernel32
    h = k.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    ok = k.GetExitCodeProcess(h, ctypes.byref(code))
    k.CloseHandle(h)
    return bool(ok) and code.value == 259  # STILL_ACTIVE


def all_runs():
    """(dir, status) for every run started by this script, newest first."""
    found = []
    for f in RUNS.rglob("status.json"):
        s = read_status(f.parent)
        if s and "started" in s:
            found.append((f.parent, s))
    return sorted(found, key=lambda x: x[1]["started"], reverse=True)


def running():
    return [(d, s) for d, s in all_runs() if s.get("state") in ACTIVE and alive(s.get("pid", 0))]


def name_of(d):
    return d.relative_to(RUNS).as_posix()


def checkpoint_of(src):
    """--from: a run name (its final model: away/best.pt unless that was worse), a run
    directory or a .pt file."""
    final = (read_status(run_dir(src)) or {}).get("final")
    if final and Path(final).is_file():
        return Path(final).resolve()
    for p in (Path(src), ROOT / src, run_dir(src) / "best.pt", Path(src) / "best.pt"):
        if p.is_file():
            return p.resolve()
    sys.exit(f"--from {src}: no such checkpoint (looked for {run_dir(src) / 'best.pt'})")


def tail(path, n=15):
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return "(no log)"


# ---------------------------------------------------------------------------------------- start


def task_of(checkpoint):
    """The task of the run a checkpoint belongs to (its own dir, or the run above an away/)."""
    for p in (checkpoint.parent, checkpoint.parent.parent):
        s = read_status(p)
        if s:
            return task_name(s.get("task"))
    return "levels"


def start(a):
    d = run_dir(a.name)
    if d.exists():
        sys.exit(f"{d.relative_to(ROOT).as_posix()} already exists: pick another name")
    busy = running()
    if busy:
        sys.exit(f"{name_of(busy[0][0])} is still running (one run at a time). "
                 f"Wait for it, or: train/run.py stop {name_of(busy[0][0])}")
    live = [b for b in broadcasts() if b != a.broadcast]
    if live:
        sys.exit(f"the broadcast {live[0]} is going (it uses the GPU and Portal 2): "
                 f"train/broadcast.py stop {live[0]}, or wait for it")
    if a.quick and not a.from_:
        sys.exit("--quick fine-tunes a model: give --from")
    if a.pause_after is not None and not 0 < a.pause_after < a.hours:
        sys.exit(f"--pause-after {a.pause_after}: give a number of hours between 0 and --hours ({a.hours})")
    if a.pause_after is not None and (a.quick or a.broadcast):
        sys.exit("--pause-after isn't for --quick or broadcast runs")
    goblin = a.task == GOBLIN
    if a.constrained_eval and not goblin:
        sys.exit("--constrained-eval is only for Goblin")
    plans = goblin_plans()
    if goblin and not a.data and not plans:
        sys.exit(f"{GOBLIN_DATA.relative_to(ROOT).as_posix()} is missing: a Goblin run needs plan data. Make it with "
                 "train/goblin_data.py first (CPU-heavy, hours; not while a run is going). No run started")
    data = []
    for pattern in a.data or ([str(plans)] if goblin else ["data/v5b/levels-*.jsonl"]):
        hits = sorted(glob.glob(str(ROOT / pattern))) or sorted(glob.glob(pattern))
        if not hits:
            sys.exit(f"--data {pattern}: no such file")
        data += [Path(h).resolve() for h in hits]
    vocab = Path(a.vocab).resolve() if a.vocab else data[0].parent / "vocab.txt"
    if goblin and not a.vocab and not vocab.exists():
        vocab = ROOT / "data" / "v5" / "vocab.txt"  # the level models' words, so --from works
    if not vocab.exists():
        sys.exit(f"{vocab} is missing: pass --vocab")
    init = checkpoint_of(a.from_) if a.from_ else None
    # Continuing a run of the same task fine-tunes it; a level model starting Goblin
    # learns a new task, at the fresh rates.
    warm = init is not None and task_of(init) == a.task
    lr, min_lr, warmup = ("2e-4", "2e-5", "200") if warm else ("1e-3", "1e-4", "300")
    if a.quick:
        lr, min_lr, warmup = QUICK_LR
    build = subprocess.run(["cargo", "build", "--release", "--quiet", "--bins"], cwd=ROOT) \
        if not a.dry_run else subprocess.CompletedProcess([], 0)
    if build.returncode or not GLADOS.exists():
        sys.exit("`cargo build --release --bins` failed (output above): fix that first, no run started")

    train = [
        "train/train.py", "--data", *map(str, data), "--vocab", str(vocab), "--out", str(d),
        "--block", "1024", "--batch", "16", "--eval-every", "2000", "--score-n", "64",
        "--lr", lr, "--min-lr", min_lr, "--warmup", warmup, "--note", a.note,
    ]
    if init:
        train += ["--init", str(init)]
    if goblin:
        train += ["--task", GOBLIN]
    else:
        train += holdout_args(data)
    train += a.extra
    if goblin:
        tests = data[0].parent if (data[0].parent / "test-gen.jsonl").exists() else (plans or OLD_PLANS).parent
        evaluate = ["train/goblin_eval.py", str(d / "best.pt"), "--plans", str(tests)]
        if a.quick and not a.constrained_eval:  # broadcast rounds: free decoding keeps the round short
            evaluate += ["--n-published", "0", "--k", "4"]
        else:
            evaluate += ["--constrained", "--out", str(d / "eval")]
        if a.constrained_eval and init:
            for baseline in (Path(init).parent / "eval-constrained", Path(init).parent / "eval"):
                try:
                    report = json.loads((baseline / "eval.json").read_text())
                    if report.get("constrained") and (baseline / "solutions.jsonl").exists():
                        evaluate += ["--reference-solutions", str(baseline / "solutions.jsonl")]
                        break
                except (OSError, ValueError):
                    continue
    else:
        evaluate = ["train/evaluate.py", str(d / "best.pt"), "--data", *map(str, data)]
        if a.quick:  # broadcast rounds: free decoding keeps the round short
            evaluate += ["--n-cond", "8", "--generator-count", "0", "--neighbours", "64", "--rated-n", "0"]
        else:  # since 2026-11-01: levels written from legal tokens only (glados-score grammar)
            evaluate += ["--constrained"]
    if a.dry_run:
        print("train:   ", " ".join(train))
        print("evaluate:", " ".join(evaluate))
        if a.quick:
            baseline = 'matched constrained parent evaluation' if a.constrained_eval else str(Path(init).parent / 'eval')
            print(f"guard:    WORSE if EVAL is more than {QUICK_WORSE[a.task]:.0%} under {baseline}")
        return

    d.mkdir(parents=True)
    write_status(
        d, state="starting", started=now().isoformat(timespec="seconds"), hours=a.hours, task=a.task,
        ingame=not (a.no_ingame or goblin or a.quick), init=str(init or ""), train=train, evaluate=evaluate,
        data=list(map(str, data)), vocab=str(vocab), extra=a.extra, quick=a.quick,
        broadcast=a.broadcast or "", pause_at=a.pause_after, constrained_eval=a.constrained_eval,
    )
    p = detach(d)
    ends = now() + datetime.timedelta(hours=a.hours)
    print(f"started runs/{a.name} (pid {p.pid}) from {init.relative_to(ROOT).as_posix() if init else 'scratch'}")
    if a.pause_after is not None:
        print(f"it pauses itself after {a.pause_after:g} of its {a.hours:g} hours of training, about "
              f"{now() + datetime.timedelta(hours=a.pause_after):%H:%M}; then: train/run.py resume {a.name}")
        print(f"wait:   .venv/Scripts/python train/run.py wait {a.name}   (exits 3 when it has paused)")
        return
    after = (75 if a.constrained_eval or not a.quick else 20) if goblin else 75
    print(f"training ends about {ends:%H:%M}, results about {ends + datetime.timedelta(minutes=after):%H:%M}")
    print(f"check:  .venv/Scripts/python train/run.py status {a.name}")
    print(f"wait:   .venv/Scripts/python train/run.py wait {a.name}")


def detach(d, *steps):
    """Run pipeline(d, steps) in a process of its own that outlives this one."""
    kw = dict(cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cmd = [PY, "-u", str(Path(__file__).resolve()), "_pipeline", str(d), *steps]
    if sys.platform == "win32":
        try:
            p = subprocess.Popen(cmd, creationflags=DETACH | BREAKAWAY, **kw)
        except OSError:  # the job the session runs in doesn't allow breaking away
            p = subprocess.Popen(cmd, creationflags=DETACH, **kw)
    else:
        p = subprocess.Popen(cmd, start_new_session=True, **kw)
    write_status(d, pid=p.pid)
    return p


STEPS = ("training", "evaluating", "ingame", "away", "away-ingame", "picking")
GOBLIN_STEPS = ("training", "evaluating")  # a Goblin run's
GOBLIN_DATA = ROOT / "data" / "goblin" / "train.jsonl"  # its training data (train/goblin_data.py)
OLD_PLANS = ROOT / "data" / "plans" / "train.jsonl"  # what it was until data/goblin was built
INVALID = ROOT / "data" / "ingame" / "invalid.jsonl"  # levels Portal 2's editor rejected, so far
HUMAN = ROOT / "data" / "ingame" / "human-unsolvable.jsonl"  # passed our solver, unsolvable in play
DEAD = ROOT / "data" / "ingame" / "dead.jsonl"  # evaluated levels whose exit can never be reached
# Workshop levels no level run trains on: their loss (train.py --holdout) shows overfitting to
# the Workshop part, which the val loss shows poorly.
WS_TEST = ROOT / "data" / "ws5" / "test.jsonl"


def holdout_args(data):
    """train.py's --holdout for a level run on `data` (none if the file is missing or trained on)."""
    if not WS_TEST.exists() or WS_TEST.resolve() in {Path(p).resolve() for p in data}:
        return []
    return ["--holdout", str(WS_TEST)]
AWAY_MINUTES, AWAY_REPEAT, AWAY_WORSE = 15, 5, 0.03  # as runs/v5-away-invalid
QUICK_LR = ("1e-4", "1e-5", "100")  # --quick: 30-minute warm starts at 1e-4 matched; 2e-4 diverged
QUICK_WORSE = {"levels": 0.03, GOBLIN: 0.05}  # --quick's guard against its parent


def goblin_plans():
    """A Goblin run's default training data: data/goblin, else the older data/plans (or None)."""
    return next((p for p in (GOBLIN_DATA, OLD_PLANS) if p.exists()), None)


def broadcasts():
    """Names of the broadcasts going now (train/broadcast.py)."""
    out = []
    for f in sorted((ROOT / "out" / "broadcast").glob("*/status.json")):
        try:
            s = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if s.get("state") in ("starting", "prep", "checking", "playing", "training") and alive(s.get("pid", 0)):
            out.append(f.parent.name)
    return out


def eval_score(eval_dir, task):
    """The EVAL score in eval_dir: solvable share (level runs) or greedy solved share (Goblin runs)."""
    try:
        e = json.loads((Path(eval_dir) / "eval.json").read_text())
        return e["headline"]["solved"] if task_name(task) == GOBLIN else e["validity"]["solvable"]
    except (OSError, ValueError, KeyError):
        return None


def decoding(eval_dir):
    """Whether eval_dir's samples were written from legal tokens only (None: no evaluation)."""
    try:
        return bool(json.loads((Path(eval_dir) / "eval.json").read_text()).get("constrained"))
    except (OSError, ValueError):
        return None


def quick_guard(d, s):
    """--quick: WORSE if this run's EVAL is too far under its parent's; returns (final model, line)."""
    task = task_name(s.get("task"))
    if task == GOBLIN and s.get("constrained_eval"):
        return constrained_guard(d, s)
    new = eval_score(d / "eval", task)
    old = eval_score(Path(s["init"]).parent / "eval", task)
    if old is not None and decoding(d / "eval") != decoding(Path(s["init"]).parent / "eval"):
        old = None  # constrained and free EVALs don't compare
    if new is None:
        (d / "WORSE").touch()
        return Path(s["init"]), "no EVAL score: the parent stays the final model"
    if old is not None and new < old - QUICK_WORSE[task]:
        (d / "WORSE").touch()
        return Path(s["init"]), f"EVAL {new:.0%} vs the parent's {old:.0%}: worse, the parent stays the final model"
    vs = f" vs the parent's {old:.0%}" if old is not None else " (the parent has no EVAL to compare)"
    return d / "best.pt", f"EVAL {new:.0%}{vs}: this run is the final model"


def constrained_guard(d, s):
    """Fail closed unless the parent and child have comparable greedy evaluations."""
    parent = Path(s["init"])
    def reject(reason):
        (d / "WORSE").touch()
        return parent, reason + ": the parent stays the final model"

    def read(directory):
        report = json.loads((directory / "eval.json").read_text())
        if not 0 <= report["headline"]["solved"] <= 1:
            raise ValueError("invalid score")
        names = report["headline"]["set"].split("+")
        if names == ["test-ws", "v6"]:
            names = ["test-ws", "test-v6"]
        rows = [json.loads(line) for line in (directory / "solutions.jsonl").read_text().splitlines() if line.strip()]
        identities = sorted((r["set"], r["tokens"]) for r in rows if r["set"] in names)
        if not identities or len(identities) != report["sets"][report["headline"]["set"]]["n"]:
            raise ValueError("missing headline solutions")
        return report, identities

    try:
        new, identities = read(d / "eval")
    except (OSError, ValueError, KeyError, TypeError):
        return reject("missing or invalid constrained evaluation")
    for directory in (parent.parent / "eval-constrained", parent.parent / "eval"):
        try:
            old, old_ids = read(directory)
            keys = ("constrained", "let_stop", "max_plan", "k")
            if (not new.get("constrained") or not old.get("constrained") or
                    any(new.get(k) != old.get(k) for k in keys) or
                    new["headline"]["set"] != old["headline"]["set"] or identities != old_ids):
                continue
            if Path(old["checkpoint"]).resolve() != parent.resolve():
                continue
            before, after = old["headline"]["solved"], new["headline"]["solved"]
            if after < before - QUICK_WORSE[GOBLIN]:
                return reject(f"constrained EVAL {after:.1%} vs {before:.1%}: worse")
            return d / "best.pt", f"constrained EVAL {after:.1%} vs {before:.1%}: guard passed"
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return reject("no comparable constrained parent baseline")


def collect_dead(d):
    """Add the levels of d's evaluation that glados-score calls dead (the exit can never be
    reached, whatever the solver's search would say) to DEAD, labelled unsolvable. Returns how
    many were new."""
    seen = set()
    if DEAD.exists():
        seen = {json.loads(l)["tokens"] for l in DEAD.read_text(encoding="utf-8").splitlines() if l.strip()}
    new = []
    for f in (d / "eval" / "samples.jsonl", d / "eval" / "rated-samples.jsonl"):
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line) if line.strip() else {}
            if r.get("dead") and r.get("tokens") and r["tokens"] not in seen:
                seen.add(r["tokens"])
                new.append({"tokens": r["tokens"], "verdict": "unsolvable", "dead": r["dead"],
                            "source": str(f.relative_to(ROOT).as_posix())})
    if new:
        DEAD.parent.mkdir(parents=True, exist_ok=True)
        with open(DEAD, "a", encoding="utf-8") as out:
            out.writelines(json.dumps(r) + "\n" for r in new)
    return len(new)


def away_train(d, s):
    """train.py arguments for training d's best.pt away from the game-rejected levels."""
    name = name_of(d)
    return [
        "train/train.py", "--data", *s["data"], *[str(f) for f in (INVALID, HUMAN, DEAD) if f.exists()] * AWAY_REPEAT,
        "--vocab", s["vocab"],
        "--out", str(d / "away"), "--init", str(d / "best.pt"), "--block", "1024", "--batch", "16",
        "--eval-every", "1000", "--score-n", "64", "--lr", "2e-4", "--min-lr", "2e-5", "--warmup", "200",
        "--note", f"{name} trained away from the levels Portal 2 rejected or play-testers couldn't solve ({AWAY_MINUTES} min, x{AWAY_REPEAT})",
        *holdout_args(s["data"]),
        *s.get("extra", []),
    ]


def solvable(eval_dir):
    try:
        return json.loads((eval_dir / "eval.json").read_text())["validity"]["solvable"]
    except (OSError, ValueError, KeyError):
        return None


def final_dir(d):
    """The directory holding the run's final model: d/away unless training away made it worse."""
    a = d / "away"
    return a if (a / "best.pt").exists() and solvable(a / "eval") is not None and not (a / "WORSE").exists() else d


def complete(d):
    """Did every Portal 2 step happen? (The in-game checks, and play-test picks checked in game.)"""
    evals = [d] + ([d / "away"] if (d / "away" / "eval" / "eval.json").exists() else [])
    return all(ingame_outcome(e)[0] for e in evals) and picks_outcome(d)[0]


def ingame_outcome(d):
    """What the in-game check of d's evaluation found: (done, one-line summary)."""
    try:
        text = (d / "eval" / "ingame" / "report.md").read_text(encoding="utf-8")
    except OSError:
        return False, "not run"
    for l in text.splitlines():
        if l.startswith("Skipped"):
            return False, l
        if "Game-valid, of levels our checker passes" in l:
            return True, l.strip("-* ").replace("**", "")
    return False, "no result in eval/ingame/report.md"


def picks_outcome(d):
    """(done, summary) for the play-test levels abtest.py (pick.py before 2026-10-25) copied in."""
    try:
        picks = json.loads((d / "playtest" / "picks.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, "none picked (see pick.log)"
    names = ", ".join(p.get("name") or f"glados-{d.name}-{p['file']}".removesuffix(".p2c") for p in picks)
    checked = all(p.get("game") == "valid" for p in picks)
    blind = (" (blind A/B: this run's model against "
             + ("its parent" if (read_status(d) or {}).get("init") else "rated Workshop levels")
             + f"; verdicts go in runs/{name_of(d)}/playtest/results.csv, then `abtest.py score`)"
             if (d / "playtest" / "key.json").exists() else "")
    return checked, f"{names} ({'valid in Portal 2' if checked else 'NOT checked in Portal 2'}){blind}"


def has_verdicts(d):
    """Has the user already written play-test verdicts for d's A/B set?"""
    try:
        with open(d / "playtest" / "results.csv", newline="", encoding="utf-8") as f:
            return any(r.get("verdict", "").strip() for r in csv.DictReader(f))
    except OSError:
        return False


def pipeline(d, steps=STEPS):
    """The detached part of `start` (and `ingame`): the steps, then record."""
    d = Path(d).resolve()
    s = write_status(d, pid=__import__("os").getpid())
    log = open(d / "run.log", "a", encoding="utf-8")
    if task_name(s.get("task")) == GOBLIN or s.get("quick"):
        steps = [x for x in steps if x in GOBLIN_STEPS]

    def say(msg):
        log.write(f"[{now():%H:%M:%S}] {msg}\n")
        log.flush()

    if sys.platform == "win32":  # don't let Windows sleep between phases either
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)

    def phase(state, args, logname, may_pause=False):
        write_status(d, state=state, phase_started=now().isoformat(timespec="seconds"), log=logname)
        say(f"{state}: {' '.join(args[:2])}")
        with open(d / logname, "a", encoding="utf-8") as out:
            code = subprocess.call([PY, "-u", *args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                                   creationflags=NO_WINDOW if sys.platform == "win32" else 0)
        if code and not (may_pause and code == PAUSED_EXIT):
            raise RuntimeError(f"{state} failed (exit {code}); see {d.name}/{logname}")
        return code

    game = s.get("ingame", True)
    paused = False
    resume_check = s.get("resume_check")  # the Portal 2 check a resume carries on (ingame.py --resume)
    write_status(d, resume_check=None, paused_steps=None)

    def check_game(step, eval_dir, logname):
        """A Portal 2 check that `pause` can stop between levels; True if it paused."""
        nonlocal paused
        args = ["train/ingame.py", str(eval_dir), "--stop-file", str(d / "PAUSE")]
        if resume_check == step:
            args.append("--resume")
        try:  # the game being busy (or crashing) shouldn't sink the run: status flags it
            if phase("ingame", args, logname, may_pause=True) == PAUSED_EXIT:
                paused = True
                left = list(steps[list(steps).index(step):])
                write_status(d, state="paused", paused=now().isoformat(timespec="seconds"), paused_steps=left)
                say(f"paused in the Portal 2 check ({step}): run.py resume {name_of(d)} carries it on")
                return True
        except RuntimeError as e:
            say(f"{e} (carrying on)")
        return False

    try:
        if "training" in steps:
            # A resumed run (resume) trains on from last.pt for the time still owed.
            left = s.get("train_left_s")
            until = now() + datetime.timedelta(seconds=left if left is not None else s["hours"] * 3600)
            until = until.strftime("%Y-%m-%d %H:%M:%S")
            write_status(d, train_until=until)
            args = s["train"] + (["--resume"] if left is not None else []) + ["--until", until]
            if s.get("pause_at") is not None:
                args += ["--pause-at", str(s["pause_at"])]
            if phase("training", args, "train.log", may_pause=True) == PAUSED_EXIT:
                paused = True
                write_status(d, state="paused", paused=now().isoformat(timespec="seconds"), train_left_s=None,
                             pause_at=None)
                say(f"paused after {trained_hours(d):.2f} of {s['hours']} h of training: run.py resume {name_of(d)}")
                return
            (d / "PAUSE").unlink(missing_ok=True)  # asked for as training ended: too late, carry on
        if not (d / "best.pt").exists():
            raise RuntimeError("training wrote no best.pt; see train.log")
        if "evaluating" in steps:
            phase("evaluating", s["evaluate"], "evaluate.log")
        if "ingame" in steps and game:
            if check_game("ingame", d / "eval", "ingame.log"):
                return
            say(f"in-game check: {ingame_outcome(d)[1]}")
        if "away" in steps and "data" in s and INVALID.exists():
            # Train best.pt away from every level the game has rejected (this run's check adds
            # to them), then evaluate that. A failure leaves best.pt as the final model.
            try:
                (d / "away").mkdir(exist_ok=True)
                say(f"{collect_dead(d)} new dead levels (exit never reachable) added to {DEAD.relative_to(ROOT).as_posix()}")
                until = (now() + datetime.timedelta(minutes=AWAY_MINUTES)).strftime("%Y-%m-%d %H:%M:%S")
                phase("away", away_train(d, s) + ["--until", until], "away/train.log")
                if not (d / "away" / "best.pt").exists():
                    raise RuntimeError("it saved no model (too short to reach its first eval); see away/train.log")
                constrained = ["--constrained"] if "--constrained" in s.get("evaluate", []) else []
                phase("evaluating", ["train/evaluate.py", str(d / "away" / "best.pt"), "--data", *s["data"],
                                     *constrained], "away/evaluate.log")
                main, away = solvable(d / "eval"), solvable(d / "away" / "eval")
                if main is not None and away is not None and away < main - AWAY_WORSE:
                    (d / "away" / "WORSE").touch()
                say(f"trained away: EVAL {away:.0%} solvable vs {main:.0%} before; "
                    f"final model {final_dir(d).relative_to(ROOT).as_posix()}/best.pt")
            except Exception as e:  # noqa: BLE001
                say(f"training away failed: {e} (carrying on with best.pt)")
        if "away-ingame" in steps and game and (d / "away" / "eval" / "samples.jsonl").exists():
            if check_game("away-ingame", d / "away" / "eval", "away/ingame.log"):
                return
            say(f"in-game check after training away: {ingame_outcome(d / 'away')[1]}")
        if s.get("quick"):
            final, why = quick_guard(d, s)
            write_status(d, final=str(final), guard=why)
            say(why)
        else:
            write_status(d, final=str(final_dir(d) / "best.pt"))
        if "picking" in steps:
            try:  # always: a blind A/B set to play-test; a failure here never sinks the run
                if has_verdicts(d):
                    say("play-test set kept: it already has verdicts in playtest/results.csv")
                else:
                    shutil.rmtree(d / "playtest", ignore_errors=True)
                    phase("picking", ["train/abtest.py", "pick", d.name, str(final_dir(d) / "best.pt"),
                                      s.get("init") or "workshop", "--out", str(d / "playtest"), "--constrained"]
                          + (["--game"] if game else []), "pick.log")
            except RuntimeError as e:
                say(f"{e} (carrying on)")
            say(f"play-test levels: {picks_outcome(d)[1]}")
        (d / "PAUSE").unlink(missing_ok=True)  # asked for after the last check: nothing left to pause
        write_status(d, state="done", ended=now().isoformat(timespec="seconds"))
        say("done")
    except BaseException as e:  # noqa: BLE001
        write_status(d, state="failed", failed_in=(read_status(d) or {}).get("state"),
                     ended=now().isoformat(timespec="seconds"), error=str(e))
        say(f"failed: {e}\n{traceback.format_exc()}")
    finally:
        if not paused:  # a paused run is recorded once it ends
            sys.path.insert(0, str(ROOT / "train"))
            import record

            record.safely(d)
            if (d / "away" / "best.pt").exists():
                record.safely(d / "away")


def trained_hours(d):
    """Hours of training d's last.pt holds (train.py's last.json); 0 if none."""
    try:
        return json.loads((d / "last.json").read_text())["trained_s"] / 3600
    except (OSError, ValueError, KeyError):
        return 0.0


# --------------------------------------------------------------------------------------- status


def describe(d):
    s = read_status(d)
    if not s:
        return f"{d}: not a run started by train/run.py (no status.json)"
    name = f"runs/{name_of(d)}"
    state = s.get("state", "?")
    if state in ACTIVE and not alive(s.get("pid", 0)):
        state = "died"
    lines = []
    entries = []
    if (d / "log.jsonl").exists():
        entries = [json.loads(l) for l in (d / "log.jsonl").read_text().splitlines() if l.strip()]

    if state in ("starting", "training"):
        head = f"{name}: TRAINING"
        if s.get("train_until"):
            left = (datetime.datetime.fromisoformat(s["train_until"]) - now()).total_seconds() / 60
            head += f", {max(left, 0):.0f} min left (training ends {hhmm(s['train_until'])})"
        if s.get("resumed"):
            head += f" (resumed {hhmm(s['resumed'])})"
        if s.get("pause_at") is not None:
            head += f"; it pauses itself after {s['pause_at']:g} of {s['hours']:g} hours of training"
        lines.append(head)
        steps = [l for l in tail(d / "train.log", 40).splitlines() if l.startswith("step ")]
        if steps:
            lines.append("  latest: " + " ".join(steps[-1].split()))
    elif state == "away":
        lines.append(f"{name}: TRAINING AWAY from the levels Portal 2 rejected (since {hhmm(s['phase_started'])},"
                     f" {AWAY_MINUTES} min; the main training is over)")
    elif state in ("evaluating", "ingame", "picking"):
        lines.append(f"{name}: {state.upper()} ({s.get('log', '')}, since {hhmm(s['phase_started'])};"
                     " training is over)")
    elif state == "done":
        lines.append(f"{name}: DONE at {hhmm(s['ended'])}")
    elif state == "stopped":
        lines.append(f"{name}: STOPPED at {hhmm(s['ended'])}")
    elif state == "paused" and s.get("paused_steps"):
        which = "after training away" if s["paused_steps"][0] == "away-ingame" else "of best.pt"
        lines.append(f"{name}: PAUSED at {hhmm(s['paused'])} in the Portal 2 check {which} (training is over;"
                     f" resume carries the check on, then: {', '.join(s['paused_steps'][1:]) or 'nothing else'})")
    elif state == "paused":
        lines.append(f"{name}: PAUSED at {hhmm(s['paused'])} after {trained_hours(d):.2f} of {s['hours']} hours"
                     " of training (evaluation and the other steps come after it resumes)")
    else:
        lines.append(f"{name}: {'DIED (process gone)' if state == 'died' else 'FAILED'} "
                     f"{s.get('error', '')}".rstrip())

    if entries:
        e = entries[-1]
        best = min(x["val_loss"] for x in entries)
        line = f"  step {e['step']}: val loss {e['val_loss']:.4f} (best {best:.4f}, lower is better)"
        if e.get("val_by_source"):
            line += " [by source: " + ", ".join(f"{k} {v:.4f}" for k, v in e["val_by_source"].items()) + "]"
        held = [x["holdout_loss"] for x in entries if "holdout_loss" in x]
        if held:
            line += (f"; Workshop test loss {held[-1]:.4f} (best {min(held):.4f}; held-out Workshop "
                     f"levels, lower is better; rising while val falls = memorizing)")
        if e.get("samples"):
            line += f"; samples {e['samples']['parsed']:.0%} parse, {e['samples']['solvable']:.0%} solvable"
        g = e.get("goblin") or e.get("solve")
        if g:
            line += f"; val levels {g['solved']:.0%} solved (greedy), {g['parsed']:.0%} parse"
        lines.append(line)
    ev = d / "eval" / "eval.json"
    goblin = task_name(s.get("task")) == GOBLIN
    if goblin:
        lines += goblin_lines(d)
    elif ev.exists():
        e = json.loads(ev.read_text())
        v = e["validity"]
        how = "constrained" if e.get("constrained") else "free decoding"
        lines.append(f"  EVAL: {v['solvable']:.0%} solvable, {v['parsed']:.0%} parse, {how} "
                     f"(a guardrail since 2026-10-25, RATED STYLE is the headline; differences under ~10 points are noise;"
                     f" compare only runs with the same decoding)")
        if e.get("rated"):
            r = e["rated"]
            gap = f"{r['profile_gap']:.1f}" if r["profile_gap"] is not None else "-"
            lines.append(f"  RATED STYLE: yield {r['yield_per_1000']:.0f} / 1,000 (valid, novel, not unsolvable), "
                         f"strong {r['strong_per_1000']:.0f} / 1,000, profile gap {gap} points vs 5-star "
                         f"levels (higher yield and lower gap are better; yield differences under ~50 are noise)")
        lines.append(f"  report: runs/{name_of(d)}/eval/report.md")
    game = s.get("ingame", True)
    ig_done, ig_text = ingame_outcome(d)
    pk_done, pk_text = picks_outcome(d)
    if ev.exists() and not goblin:
        lines.append(f"  in Portal 2: {ig_text if game or ig_done else 'not checked (Portal 2 checks were off for this run)'}")
    away = d / "away"
    if (away / "eval" / "eval.json").exists():
        v = json.loads((away / "eval" / "eval.json").read_text())["validity"]
        aw_done, aw_text = ingame_outcome(away)
        lines.append(f"  AFTER TRAINING AWAY from the levels Portal 2 rejected (runs/{name_of(away)}):"
                     f" EVAL {v['solvable']:.0%} solvable, {v['parsed']:.0%} parse")
        lines.append(f"    in Portal 2: {aw_text if game or aw_done else 'not checked'}")
    elif state == "done" and "data" in s and not goblin and not s.get("quick"):
        lines.append("  training away: didn't happen (see run.log)")
    if state == "done" and s.get("quick"):
        lines.append(f"  quick run: {s.get('guard', 'no guard result')}")
        lines.append(f"  final model: {s.get('final', d / 'best.pt')}")
    elif state == "done":
        f = final_dir(d)
        why = " (training away made it worse)" if (away / "WORSE").exists() else ""
        lines.append(f"  final model: runs/{name_of(f)}/best.pt{why}")
    if (state == "done" and not goblin and not s.get("quick")) or (d / "playtest" / "picks.json").exists():
        lines.append(f"  play-test levels in the Puzzle Maker: {pk_text}")
    if state in ("failed", "died"):
        log = s.get("log", "train.log")
        lines.append(f"  last lines of {log}:")
        lines += ["    " + l for l in tail(d / log).splitlines()]
    if state in ACTIVE:
        return "\n".join(lines)
    n = name_of(d)
    todo = []
    if state == "done" and game and not complete(d):
        lines[0] += "  -- INCOMPLETE: a Portal 2 check didn't happen"
        todo.append("Tell the user a Portal 2 check was skipped (the 'in Portal 2' lines above say why)."
                    " Once Portal 2 is closed, run"
                    f"  .venv/Scripts/python train/run.py ingame {n}  and then wait for it the same way.")
    if state in ("failed", "died"):
        todo.append("Report the log lines above to the user. Don't start the run again until the cause is fixed.")
    if state == "paused":
        lines.append("  NEXT STEPS (do all of them):")
        what = ("in its Portal 2 check (above); the game is closed" if s.get("paused_steps")
                else "and how many hours of training are done (above)")
        lines.append(f"    1. Tell the user runs/{n} is paused {what}."
                     " It holds neither the GPU nor the machine now.")
        lines.append(f"    2. Only when the user says to carry on:  .venv/Scripts/python train/run.py resume {n}"
                     f"  then  .venv/Scripts/python train/run.py wait {n}  (run_in_background), as after start.")
        return "\n".join(lines)
    if s.get("broadcast"):
        todo = [f"Nothing: train/broadcast.py ({s['broadcast']}) started this run and carries on by itself."]
    elif goblin:
        todo.append("Report everything above to the user.")
        todo.append(f'Record the conclusion:  .venv/Scripts/python train/run.py note {n} "WHAT IT SHOWED"'
                    " (give the EVAL solved %, pass@8, the test-ws and published-liberal solved %, and how"
                    " they compare with earlier Goblin runs in  train/run.py list --task goblin)")
    else:
        todo.append("Report everything above to the user, including the play-test level names: a blind A/B"
                    " set (don't say which level is which side). Ask for a verdict per level: good, ok,"
                    " trivial, boring, unsolvable or broken.")
        todo.append(f'Record the conclusion:  .venv/Scripts/python train/run.py note {n} "WHAT IT SHOWED"'
                    " (give the RATED STYLE yield and profile gap, the Workshop test loss, the EVAL solvable %,"
                    " the Portal 2 game-valid %, before and after training away, which is the final model,"
                    " and how they compare with earlier runs in  train/run.py list)")
    lines.append("  NEXT STEPS (do all of them):")
    lines += [f"    {i}. {t}" for i, t in enumerate(todo, 1)]
    return "\n".join(lines)


def goblin_lines(d):
    """A Goblin run's evaluation (goblin_eval.py's eval/eval.json), for describe()."""
    try:
        e = json.loads((d / "eval" / "eval.json").read_text())
    except (OSError, ValueError):
        return []
    h, k = e["headline"], e.get("k", 8)
    how = "constrained" if e.get("constrained") else "free decoding"
    out = [f"  EVAL: {h['solved']:.0%} solved (greedy, {how}, {h['set']} levels; the main score, differences under"
           f" ~5 points are noise), pass@{k} {h['pass_at_k']:.0%}, optimal {h['optimal']:.0%}, {h['parsed']:.0%} parse"]
    for name, v in e["sets"].items():
        if name != h["set"]:
            out.append(f"    {name}: {v['solved']:.0%} solved (greedy), pass@{k} {v['pass_at_k']:.0%}, {v['n']} levels")
    out.append(f"  report: runs/{name_of(d)}/eval/report.md")
    return out


def newest():
    runs = all_runs()
    if not runs:
        sys.exit("no runs started by train/run.py yet")
    return runs[0][0]


def status(a):
    print(describe(run_dir(a.name) if a.name else newest()))


def wait(a):
    d = run_dir(a.name) if a.name else newest()
    while True:
        s = read_status(d) or {}
        if s.get("state") not in ACTIVE or not alive(s.get("pid", 0)):
            break
        time.sleep(30)
    print(describe(d))
    s = read_status(d) or {}
    if s.get("state") == "paused":
        sys.exit(3)
    if s.get("state") != "done":
        sys.exit(1)
    if s.get("ingame", True) and not complete(d):
        sys.exit(2)


def ingame(a):
    """Redo a finished run's Portal 2 steps: check its evaluation, pick checked play-test levels."""
    d = run_dir(a.name)
    s = read_status(d)
    if s and task_name(s.get("task")) == GOBLIN:
        sys.exit(f"runs/{a.name} is a Goblin run: it has no Portal 2 steps")
    if not s or not (d / "eval" / "samples.jsonl").exists():
        sys.exit(f"runs/{a.name} has no evaluation to check: did it finish?")
    busy = running()
    if busy:
        sys.exit(f"{name_of(busy[0][0])} is still running (one run at a time). Wait for it first")
    write_status(d, state="ingame", phase_started=now().isoformat(timespec="seconds"), ingame=True, error="")
    p = detach(d, "ingame", "away-ingame", "picking")
    print(f"checking runs/{a.name} in Portal 2 in the background (pid {p.pid}), about 40 minutes")
    print(f"wait:   .venv/Scripts/python train/run.py wait {a.name}")


def stop(a):
    d = run_dir(a.name)
    s = read_status(d)
    if not s or s.get("state") not in ACTIVE + ("paused",):
        sys.exit(f"runs/{a.name} isn't running")
    if s.get("state") != "paused" and alive(s["pid"]):
        subprocess.call(["taskkill", "/T", "/F", "/PID", str(s["pid"])],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    write_status(d, state="stopped", ended=now().isoformat(timespec="seconds"))
    sys.path.insert(0, str(ROOT / "train"))
    import record

    record.safely(d)
    print(f"stopped runs/{a.name}; its last checkpoint is runs/{a.name}/best.pt (if it got to step 2000)")


def pause(a):
    """Ask a training run to pause: train.py saves last.pt and exits, and the run waits for resume."""
    d = run_dir(a.name)
    s = read_status(d) or {}
    state = s.get("state")
    if state == "paused":
        sys.exit(f"runs/{a.name} is already paused. To carry on: train/run.py resume {a.name}")
    if state not in ACTIVE or not alive(s.get("pid", 0)):
        sys.exit(f"runs/{a.name} isn't running")
    if state not in ("starting", "training", "ingame"):
        sys.exit(f"runs/{a.name} is {state}: only training and the Portal 2 checks can pause. "
                 f"Let it finish this step (train/run.py wait {a.name}), or stop it")
    if s.get("broadcast"):
        sys.exit(f"runs/{a.name} belongs to the broadcast {s['broadcast']}: don't pause it")
    (d / "PAUSE").touch()
    if state == "ingame":
        print(f"asked runs/{a.name} to pause; waiting for its Portal 2 check to finish the level in hand and "
              "close the game (a minute or so)")
    else:
        print(f"asked runs/{a.name} to pause; waiting for it to save last.pt (an eval under way finishes first)")
    give_up = time.time() + 20 * 60
    while time.time() < give_up:
        s = read_status(d) or {}
        if s.get("state") == "paused":
            print(describe(d))
            return
        if state == "ingame" and s.get("state") in ACTIVE and s.get("state") != "ingame" and alive(s.get("pid", 0)):
            print(f"runs/{a.name} finished that check before it could pause (now: {s.get('state')}); "
                  "the request stands: it pauses at its next Portal 2 check, if it has one")
            return
        if s.get("state") not in ("starting", "training", "ingame") or not alive(s.get("pid", 0)):
            (d / "PAUSE").unlink(missing_ok=True)
            sys.exit(f"runs/{a.name} left training before it could pause (now: {s.get('state')}); "
                     f"it carries on with its other steps: train/run.py status {a.name}")
        time.sleep(5)
    sys.exit(f"runs/{a.name} hasn't paused after 20 minutes. The request stands (runs/{a.name}/PAUSE): "
             f"check with  train/run.py status {a.name}")


def resume(a):
    """Carry on a paused run: train from last.pt for the time still owed, then the other steps."""
    d = run_dir(a.name)
    s = read_status(d)
    if not s or s.get("state") != "paused":
        sys.exit(f"runs/{a.name} isn't paused (state: {(s or {}).get('state', 'no such run')})")
    busy = running()
    if busy:
        sys.exit(f"{name_of(busy[0][0])} is running (one run at a time). Wait for it, or pause or stop it first")
    live = broadcasts()
    if live:
        sys.exit(f"the broadcast {live[0]} is going: wait for it, or train/broadcast.py stop {live[0]}")
    if s.get("paused_steps"):  # paused in a Portal 2 check: carry the check on, then the rest
        if a.pause_after is not None:
            sys.exit("--pause-after is for training, which is over for this run. Nothing resumed")
        (d / "PAUSE").unlink(missing_ok=True)
        steps = s["paused_steps"]
        write_status(d, state="ingame", phase_started=now().isoformat(timespec="seconds"),
                     resumed=now().isoformat(timespec="seconds"), resume_check=steps[0], error="")
        p = detach(d, *steps)
        print(f"resumed runs/{a.name} (pid {p.pid}): the Portal 2 check carries on, then {', '.join(steps[1:])}")
        print(f"wait:   .venv/Scripts/python train/run.py wait {a.name}")
        return
    if not (d / "last.pt").exists():
        sys.exit(f"runs/{a.name}/last.pt is missing: nothing to resume from")
    done = trained_hours(d)
    if a.pause_after is not None and not done < a.pause_after < s["hours"]:
        sys.exit(f"--pause-after {a.pause_after}: give a number of hours between the {done:.2f} done and "
                 f"the run's {s['hours']:g}. Nothing resumed")
    left = max((s["hours"] - done) * 3600, 5 * 60)  # some minutes at least: training ends on an eval
    build = subprocess.run(["cargo", "build", "--release", "--quiet", "--bins"], cwd=ROOT)
    if build.returncode or not GLADOS.exists():
        sys.exit("`cargo build --release --bins` failed (output above): fix that first, nothing resumed")
    (d / "PAUSE").unlink(missing_ok=True)
    write_status(d, state="starting", resumed=now().isoformat(timespec="seconds"), train_left_s=left, error="",
                 pause_at=a.pause_after)
    p = detach(d)
    ends = now() + datetime.timedelta(seconds=left)
    after = 20 if task_name(s.get("task")) == GOBLIN else 75
    print(f"resumed runs/{a.name} (pid {p.pid}) after {done:.2f} of {s['hours']} hours of training")
    if a.pause_after is not None:
        print(f"it pauses itself again after {a.pause_after:g} hours of training in all, about "
              f"{now() + datetime.timedelta(hours=a.pause_after - done):%H:%M}")
        print(f"wait:   .venv/Scripts/python train/run.py wait {a.name}   (exits 3 when it has paused)")
        return
    print(f"training ends about {ends:%H:%M}, results about {ends + datetime.timedelta(minutes=after):%H:%M}")
    print(f"wait:   .venv/Scripts/python train/run.py wait {a.name}")


def list_runs(a):
    goblin = a.task == GOBLIN
    path = ROOT / "train" / ("goblin-runs.csv" if goblin else "runs.csv")
    if not path.exists():
        sys.exit(f"{path.relative_to(ROOT).as_posix()} doesn't exist yet: no runs recorded there")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[-a.n:]
    if goblin:  # solved%: EVAL solved % (greedy, the headline set), the main score; pass: pass@8
        print(f"{'run':32} {'started':16} {'min':>4} {'solved%':>7} {'pass':>5} {'ws%':>5} {'val':>6}  from / notes")
    else:
        print(f"{'run':32} {'started':16} {'min':>4} {'solv%':>5} {'val':>6} {'ws':>6} {'yield':>5}  from / notes")
    for r in rows:
        notes = (r["init"] + " | " if r["init"] else "") + r["notes"]
        if goblin:
            print(f"{r['run'][:32]:32} {r['started']:16} {r['minutes']:>4} {r['eval_solved'] or '-':>7} "
                  f"{r['eval_pass_at_k'] or '-':>5} {r['ws_solved'] or '-':>5} {r['best_val'] or '-':>6}  {notes[:80]}")
        else:
            print(f"{r['run'][:32]:32} {r['started']:16} {r['minutes']:>4} {r['eval_solvable'] or '-':>5} "
                  f"{r['best_val'] or '-':>6} {r.get('holdout_final') or '-':>6} {r.get('rated_yield') or '-':>5}  "
                  f"{notes[:76]}")
    live = running()
    if live:
        print(f"\nrunning now: runs/{name_of(live[0][0])}")


def note(a):
    sys.path.insert(0, str(ROOT / "train"))
    import record

    r = record.update(run_dir(a.name), a.text)
    print(f"noted on runs/{name_of(run_dir(a.name))} in {record.table(r['task'])[0].relative_to(ROOT).as_posix()}")


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1 :]
    if argv[:1] == ["_pipeline"]:
        return pipeline(argv[1], argv[2:] or STEPS)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("start", help="start a run in the background")
    p.add_argument("name", help="the run's name: it goes to runs/NAME")
    p.add_argument("--hours", type=float, required=True, help="training time (evaluation comes after)")
    p.add_argument("--note", required=True, help="what the run is for (goes in train/runs.csv)")
    p.add_argument("--from", dest="from_", default=None, help="continue from this run (or .pt)")
    p.add_argument("--task", type=task_name, choices=("levels", GOBLIN), default="levels",
                   help="levels: the level-writing net; goblin: Goblin, the net that writes plans (needs data/goblin)")
    p.add_argument("--data", nargs="+", default=None,
                   help="default: data/v5b/levels-*.jsonl, or data/goblin/train.jsonl for --task goblin")
    p.add_argument("--vocab", default=None, help="default: vocab.txt beside the first data file")
    p.add_argument("--no-ingame", action="store_true",
                   help="skip the Portal 2 checks. ONLY if the user says the game mustn't be launched")
    p.add_argument("--ingame", action="store_true", help=argparse.SUPPRESS)  # old flag: always on now
    p.add_argument("--quick", action="store_true",
                   help="a short fine-tune of --from with a small eval and a guard (train/broadcast.py's rounds)")
    p.add_argument("--constrained-eval", action="store_true",
                   help="Goblin: full constrained evaluation and matched parent guard for --quick")
    p.add_argument("--broadcast", default=None, help=argparse.SUPPRESS)  # set by train/broadcast.py
    p.add_argument("--pause-after", type=float, default=None,
                   help="pause by itself after this many hours of training (resume carries on)")
    p.add_argument("--dry-run", action="store_true", help="print the train and evaluate commands, start nothing")
    p.set_defaults(fn=start)
    for cmd, fn, h in [("status", status, "what a run is doing"), ("wait", wait, "block until it ends")]:
        p = sub.add_parser(cmd, help=h)
        p.add_argument("name", nargs="?", help="default: the newest run")
        p.set_defaults(fn=fn)
    p = sub.add_parser("ingame", help="redo a finished run's Portal 2 steps (if they were skipped)")
    p.add_argument("name")
    p.set_defaults(fn=ingame)
    p = sub.add_parser("stop", help="stop a run")
    p.add_argument("name")
    p.set_defaults(fn=stop)
    p = sub.add_parser("pause", help="pause a run's training or Portal 2 check (it keeps its place; resume carries on)")
    p.add_argument("name")
    p.set_defaults(fn=pause)
    p = sub.add_parser("resume", help="carry on a paused run")
    p.add_argument("name")
    p.add_argument("--pause-after", type=float, default=None,
                   help="pause again once this many hours of training are done in all")
    p.set_defaults(fn=resume)
    p = sub.add_parser("list", help="recent runs from train/runs.csv (--task goblin: train/goblin-runs.csv)")
    p.add_argument("-n", type=int, default=12)
    p.add_argument("--task", type=task_name, choices=("levels", GOBLIN), default="levels")
    p.set_defaults(fn=list_runs)
    p = sub.add_parser("note", help="write a run's conclusion into train/runs.csv (or goblin-runs.csv)")
    p.add_argument("name")
    p.add_argument("text")
    p.set_defaults(fn=note)
    a = ap.parse_args(argv)
    a.extra = extra
    a.fn(a)


if __name__ == "__main__":
    main()
