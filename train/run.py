"""Start, watch and stop training runs with one command each. Meant to need no thought.

    .venv/Scripts/python train/run.py start v4-a --hours 2 --note "first v4 run, fresh"
    .venv/Scripts/python train/run.py start v4-b --hours 1 --from v4-a --note "v4-a continued"
    .venv/Scripts/python train/run.py status          # the newest run (or: status v4-a)
    .venv/Scripts/python train/run.py wait v4-a       # blocks until it ends, then prints status
    .venv/Scripts/python train/run.py stop v4-a
    .venv/Scripts/python train/run.py list            # recent rows of train/runs.csv
    .venv/Scripts/python train/run.py note v4-a "what it showed"
    .venv/Scripts/python train/run.py ingame v4-a     # redo the Portal 2 steps if they were skipped

`start` builds target/release/glados.exe, then returns: the run carries on in the background,
detached, and survives the terminal or session that started it. Every run does every step, in
order, with nothing to remember:
  1. train (train.py) for --hours
  2. evaluate best.pt (evaluate.py, about 5 minutes)
  3. check the evaluation's levels in Portal 2's own editor (ingame.py, about 15 minutes). It
     launches the game itself, unless the game is already running (someone playing): then it
     is skipped, and `status` says INCOMPLETE until `run.py ingame NAME` redoes it.
  4. train best.pt away from every level Portal 2 has rejected (data/ingame/invalid.jsonl, x5,
     labelled invalid) for 15 minutes into runs/NAME/away, evaluate that and check it in Portal
     2 too. It is the final model unless its EVAL solvable share is over 3 points lower (then
     away/WORSE is written and best.pt stays final).
  5. pick 5 levels from the final model for play-testing (pick.py), checked in Portal 2 too,
     and copy them into the Puzzle Maker folder as glados-NAME-NNNNN
  6. record the run (and NAME/away) in train/runs.csv
Everything lands in runs/NAME: status.json (what it's doing now), run.log (each phase),
train.log, evaluate.log, ingame.log, pick.log, eval/report.md.

Settings: data/v5 (all levels-*.jsonl: generated, Workshop, mutants) unless --data; block 1024, batch 16. A fresh run uses
lr 1e-3 down to 1e-4; one continuing another (--from) uses 2e-4 down to 2e-5, as the warm runs
in runs.csv did. Anything after `--` goes to train.py as is.

Solver-net runs (`start NAME --task solve`) train the second net, which reads a level and
writes its plan, on data/plans (train/plans.py makes it). Their steps: train, evaluate best.pt
(solve_eval.py, about 15 minutes: plans for held-out levels, replayed by `glados check`), record
in train/solver-runs.csv (`list --task solve` shows them).
No Portal 2 steps, no training away, no play-test picks. The main score is EVAL solved %: the
share of test-gen levels whose greedy plan solves. --from a level-writing run warm-starts from
its weights with the fresh learning rates (a new task); --from a solve run continues it.

Only one run at a time: they share the GPU.
"""

import argparse
import csv
import ctypes
import datetime
import glob
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
PY = sys.executable
GLADOS = ROOT / "target" / "release" / "glados.exe"
ACTIVE = ("starting", "training", "evaluating", "ingame", "away", "picking")
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
            return s.get("task", "levels")
    return "levels"


def start(a):
    d = run_dir(a.name)
    if d.exists():
        sys.exit(f"{d.relative_to(ROOT).as_posix()} already exists: pick another name")
    busy = running()
    if busy:
        sys.exit(f"{name_of(busy[0][0])} is still running (one run at a time). "
                 f"Wait for it, or: train/run.py stop {name_of(busy[0][0])}")
    solve = a.task == "solve"
    if solve and not a.data and not PLANS.exists():
        sys.exit(f"{PLANS.relative_to(ROOT).as_posix()} is missing: a solve run needs plan data. Make it with "
                 "train/plans.py first (CPU-heavy, hours; not while a run is going). No run started")
    data = []
    for pattern in a.data or ([str(PLANS)] if solve else ["data/v5/levels-*.jsonl"]):
        hits = sorted(glob.glob(str(ROOT / pattern))) or sorted(glob.glob(pattern))
        if not hits:
            sys.exit(f"--data {pattern}: no such file")
        data += [Path(h).resolve() for h in hits]
    vocab = Path(a.vocab).resolve() if a.vocab else data[0].parent / "vocab.txt"
    if solve and not a.vocab and not vocab.exists():
        vocab = ROOT / "data" / "v5" / "vocab.txt"  # the level models' words, so --from works
    if not vocab.exists():
        sys.exit(f"{vocab} is missing: pass --vocab")
    init = checkpoint_of(a.from_) if a.from_ else None
    # Continuing a run of the same task fine-tunes it; a level model starting the solver net
    # learns a new task, at the fresh rates.
    warm = init is not None and task_of(init) == a.task
    lr, min_lr, warmup = ("2e-4", "2e-5", "200") if warm else ("1e-3", "1e-4", "300")
    build = subprocess.run(["cargo", "build", "--release", "--quiet", "--bin", "glados"], cwd=ROOT)
    if build.returncode or not GLADOS.exists():
        sys.exit("`cargo build --release --bin glados` failed (output above): fix that first, no run started")

    train = [
        "train/train.py", "--data", *map(str, data), "--vocab", str(vocab), "--out", str(d),
        "--block", "1024", "--batch", "16", "--eval-every", "2000", "--score-n", "64",
        "--lr", lr, "--min-lr", min_lr, "--warmup", warmup, "--note", a.note,
    ]
    if init:
        train += ["--init", str(init)]
    if solve:
        train += ["--task", "solve"]
    train += a.extra
    if solve:
        tests = data[0].parent if (data[0].parent / "test-gen.jsonl").exists() else PLANS.parent
        evaluate = ["train/solve_eval.py", str(d / "best.pt"), "--plans", str(tests)]
    else:
        evaluate = ["train/evaluate.py", str(d / "best.pt"), "--data", *map(str, data)]

    d.mkdir(parents=True)
    write_status(
        d, state="starting", started=now().isoformat(timespec="seconds"), hours=a.hours, task=a.task,
        ingame=not (a.no_ingame or solve), init=str(init or ""), train=train, evaluate=evaluate,
        data=list(map(str, data)), vocab=str(vocab), extra=a.extra,
    )
    p = detach(d)
    ends = now() + datetime.timedelta(hours=a.hours)
    print(f"started runs/{a.name} (pid {p.pid}) from {init.relative_to(ROOT).as_posix() if init else 'scratch'}")
    after = 20 if solve else 75
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
SOLVE_STEPS = ("training", "evaluating")  # a solver-net run's
PLANS = ROOT / "data" / "plans" / "train.jsonl"  # its training data (train/plans.py)
INVALID = ROOT / "data" / "ingame" / "invalid.jsonl"  # levels Portal 2's editor rejected, so far
AWAY_MINUTES, AWAY_REPEAT, AWAY_WORSE = 15, 5, 0.03  # as runs/v5-away-invalid


def away_train(d, s):
    """train.py arguments for training d's best.pt away from the game-rejected levels."""
    name = name_of(d)
    return [
        "train/train.py", "--data", *s["data"], *[str(INVALID)] * AWAY_REPEAT, "--vocab", s["vocab"],
        "--out", str(d / "away"), "--init", str(d / "best.pt"), "--block", "1024", "--batch", "16",
        "--eval-every", "1000", "--score-n", "64", "--lr", "2e-4", "--min-lr", "2e-5", "--warmup", "200",
        "--note", f"{name} trained away from the levels Portal 2 rejected ({AWAY_MINUTES} min, x{AWAY_REPEAT})",
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
    """(done, summary) for the play-test levels pick.py copied in."""
    try:
        picks = json.loads((d / "playtest" / "picks.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, "none picked (see pick.log)"
    names = ", ".join(f"glados-{d.name}-{p['file']}".removesuffix(".p2c") for p in picks)
    checked = all(p.get("game") == "valid" for p in picks)
    return checked, f"{names} ({'valid in Portal 2' if checked else 'NOT checked in Portal 2'})"


def pipeline(d, steps=STEPS):
    """The detached part of `start` (and `ingame`): the steps, then record."""
    d = Path(d).resolve()
    s = write_status(d, pid=__import__("os").getpid())
    log = open(d / "run.log", "a", encoding="utf-8")
    if s.get("task") == "solve":
        steps = [x for x in steps if x in SOLVE_STEPS]

    def say(msg):
        log.write(f"[{now():%H:%M:%S}] {msg}\n")
        log.flush()

    if sys.platform == "win32":  # don't let Windows sleep between phases either
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)

    def phase(state, args, logname):
        write_status(d, state=state, phase_started=now().isoformat(timespec="seconds"), log=logname)
        say(f"{state}: {' '.join(args[:2])}")
        with open(d / logname, "a", encoding="utf-8") as out:
            code = subprocess.call([PY, "-u", *args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                                   creationflags=NO_WINDOW if sys.platform == "win32" else 0)
        if code:
            raise RuntimeError(f"{state} failed (exit {code}); see {d.name}/{logname}")

    game = s.get("ingame", True)
    try:
        if "training" in steps:
            until = (now() + datetime.timedelta(hours=s["hours"])).strftime("%Y-%m-%d %H:%M:%S")
            write_status(d, train_until=until)
            phase("training", s["train"] + ["--until", until], "train.log")
        if not (d / "best.pt").exists():
            raise RuntimeError("training wrote no best.pt; see train.log")
        if "evaluating" in steps:
            phase("evaluating", s["evaluate"], "evaluate.log")
        if "ingame" in steps and game:
            try:  # the game being busy (or crashing) shouldn't sink the run: status flags it
                phase("ingame", ["train/ingame.py", str(d / "eval")], "ingame.log")
            except RuntimeError as e:
                say(f"{e} (carrying on)")
            say(f"in-game check: {ingame_outcome(d)[1]}")
        if "away" in steps and "data" in s and INVALID.exists():
            # Train best.pt away from every level the game has rejected (this run's check adds
            # to them), then evaluate that. A failure leaves best.pt as the final model.
            try:
                (d / "away").mkdir(exist_ok=True)
                until = (now() + datetime.timedelta(minutes=AWAY_MINUTES)).strftime("%Y-%m-%d %H:%M:%S")
                phase("away", away_train(d, s) + ["--until", until], "away/train.log")
                if not (d / "away" / "best.pt").exists():
                    raise RuntimeError("it saved no model (too short to reach its first eval); see away/train.log")
                phase("evaluating", ["train/evaluate.py", str(d / "away" / "best.pt"), "--data", *s["data"]],
                      "away/evaluate.log")
                main, away = solvable(d / "eval"), solvable(d / "away" / "eval")
                if main is not None and away is not None and away < main - AWAY_WORSE:
                    (d / "away" / "WORSE").touch()
                say(f"trained away: EVAL {away:.0%} solvable vs {main:.0%} before; "
                    f"final model {final_dir(d).relative_to(ROOT).as_posix()}/best.pt")
            except Exception as e:  # noqa: BLE001
                say(f"training away failed: {e} (carrying on with best.pt)")
        if "away-ingame" in steps and game and (d / "away" / "eval" / "samples.jsonl").exists():
            try:
                phase("ingame", ["train/ingame.py", str(d / "away" / "eval")], "away/ingame.log")
            except RuntimeError as e:
                say(f"{e} (carrying on)")
            say(f"in-game check after training away: {ingame_outcome(d / 'away')[1]}")
        write_status(d, final=str(final_dir(d) / "best.pt"))
        if "picking" in steps:
            try:  # always: 5 levels to play-test; a failure here never sinks the run
                phase("picking", ["train/pick.py", str(d), "--checkpoint", str(final_dir(d) / "best.pt")]
                      + ([] if game else ["--no-game"]), "pick.log")
            except RuntimeError as e:
                say(f"{e} (carrying on)")
            say(f"play-test levels: {picks_outcome(d)[1]}")
        write_status(d, state="done", ended=now().isoformat(timespec="seconds"))
        say("done")
    except BaseException as e:  # noqa: BLE001
        write_status(d, state="failed", failed_in=(read_status(d) or {}).get("state"),
                     ended=now().isoformat(timespec="seconds"), error=str(e))
        say(f"failed: {e}\n{traceback.format_exc()}")
    finally:
        sys.path.insert(0, str(ROOT / "train"))
        import record

        record.safely(d)
        if (d / "away" / "best.pt").exists():
            record.safely(d / "away")


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
    else:
        lines.append(f"{name}: {'DIED (process gone)' if state == 'died' else 'FAILED'} "
                     f"{s.get('error', '')}".rstrip())

    if entries:
        e = entries[-1]
        best = min(x["val_loss"] for x in entries)
        line = f"  step {e['step']}: val loss {e['val_loss']:.4f} (best {best:.4f}, lower is better)"
        if e.get("samples"):
            line += f"; samples {e['samples']['parsed']:.0%} parse, {e['samples']['solvable']:.0%} solvable"
        if e.get("solve"):
            line += f"; val levels {e['solve']['solved']:.0%} solved (greedy), {e['solve']['parsed']:.0%} parse"
        lines.append(line)
    ev = d / "eval" / "eval.json"
    solve = s.get("task") == "solve"
    if solve:
        lines += solve_lines(d)
    elif ev.exists():
        v = json.loads(ev.read_text())["validity"]
        lines.append(f"  EVAL: {v['solvable']:.0%} solvable, {v['parsed']:.0%} parse "
                     f"(the main score; differences under ~10 points are noise)")
        lines.append(f"  report: runs/{name_of(d)}/eval/report.md")
    game = s.get("ingame", True)
    ig_done, ig_text = ingame_outcome(d)
    pk_done, pk_text = picks_outcome(d)
    if ev.exists() and not solve:
        lines.append(f"  in Portal 2: {ig_text if game or ig_done else 'not checked (Portal 2 checks were off for this run)'}")
    away = d / "away"
    if (away / "eval" / "eval.json").exists():
        v = json.loads((away / "eval" / "eval.json").read_text())["validity"]
        aw_done, aw_text = ingame_outcome(away)
        lines.append(f"  AFTER TRAINING AWAY from the levels Portal 2 rejected (runs/{name_of(away)}):"
                     f" EVAL {v['solvable']:.0%} solvable, {v['parsed']:.0%} parse")
        lines.append(f"    in Portal 2: {aw_text if game or aw_done else 'not checked'}")
    elif state == "done" and "data" in s and not solve:
        lines.append("  training away: didn't happen (see run.log)")
    if state == "done":
        f = final_dir(d)
        why = " (training away made it worse)" if (away / "WORSE").exists() else ""
        lines.append(f"  final model: runs/{name_of(f)}/best.pt{why}")
    if (state == "done" and not solve) or (d / "playtest" / "picks.json").exists():
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
    if solve:
        todo.append("Report everything above to the user.")
        todo.append(f'Record the conclusion:  .venv/Scripts/python train/run.py note {n} "WHAT IT SHOWED"'
                    " (give the EVAL solved %, pass@8, the test-ws and published-liberal solved %, and how"
                    " they compare with earlier solve runs in  train/run.py list --task solve)")
    else:
        todo.append("Report everything above to the user, including the play-test level names.")
        todo.append(f'Record the conclusion:  .venv/Scripts/python train/run.py note {n} "WHAT IT SHOWED"'
                    " (give the EVAL solvable %, the Portal 2 game-valid %, before and after training away,"
                    " which is the final model, and how they compare with earlier runs in  train/run.py list)")
    lines.append("  NEXT STEPS (do all of them):")
    lines += [f"    {i}. {t}" for i, t in enumerate(todo, 1)]
    return "\n".join(lines)


def solve_lines(d):
    """A solver-net run's evaluation (solve_eval.py's eval/eval.json), for describe()."""
    try:
        e = json.loads((d / "eval" / "eval.json").read_text())
    except (OSError, ValueError):
        return []
    h, k = e["headline"], e.get("k", 8)
    out = [f"  EVAL: {h['solved']:.0%} solved (greedy, {h['set']} levels; the main score, differences under"
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
    if s.get("state") != "done":
        sys.exit(1)
    if s.get("ingame", True) and not complete(d):
        sys.exit(2)


def ingame(a):
    """Redo a finished run's Portal 2 steps: check its evaluation, pick checked play-test levels."""
    d = run_dir(a.name)
    s = read_status(d)
    if s and s.get("task") == "solve":
        sys.exit(f"runs/{a.name} is a solver-net run: it has no Portal 2 steps")
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
    if not s or s.get("state") not in ACTIVE:
        sys.exit(f"runs/{a.name} isn't running")
    if alive(s["pid"]):
        subprocess.call(["taskkill", "/T", "/F", "/PID", str(s["pid"])],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    write_status(d, state="stopped", ended=now().isoformat(timespec="seconds"))
    sys.path.insert(0, str(ROOT / "train"))
    import record

    record.safely(d)
    print(f"stopped runs/{a.name}; its last checkpoint is runs/{a.name}/best.pt (if it got to step 2000)")


def list_runs(a):
    solve = a.task == "solve"
    path = ROOT / "train" / ("solver-runs.csv" if solve else "runs.csv")
    if not path.exists():
        sys.exit(f"{path.relative_to(ROOT).as_posix()} doesn't exist yet: no runs recorded there")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[-a.n:]
    if solve:  # solved%: EVAL solved % (greedy, test-gen), the main score; pass: pass@8
        print(f"{'run':32} {'started':16} {'min':>4} {'solved%':>7} {'pass':>5} {'ws%':>5} {'val':>6}  from / notes")
    else:
        print(f"{'run':32} {'started':16} {'min':>4} {'solv%':>5} {'val':>6}  from / notes")
    for r in rows:
        notes = (r["init"] + " | " if r["init"] else "") + r["notes"]
        if solve:
            print(f"{r['run'][:32]:32} {r['started']:16} {r['minutes']:>4} {r['eval_solved'] or '-':>7} "
                  f"{r['eval_pass_at_k'] or '-':>5} {r['ws_solved'] or '-':>5} {r['best_val'] or '-':>6}  {notes[:80]}")
        else:
            print(f"{r['run'][:32]:32} {r['started']:16} {r['minutes']:>4} {r['eval_solvable'] or '-':>5} "
                  f"{r['best_val'] or '-':>6}  {notes[:90]}")
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
    p.add_argument("--task", choices=("levels", "solve"), default="levels",
                   help="levels: the level-writing net; solve: the solver net (needs data/plans)")
    p.add_argument("--data", nargs="+", default=None,
                   help="default: data/v5/levels-*.jsonl, or data/plans/train.jsonl for --task solve")
    p.add_argument("--vocab", default=None, help="default: vocab.txt beside the first data file")
    p.add_argument("--no-ingame", action="store_true",
                   help="skip the Portal 2 checks. ONLY if the user says the game mustn't be launched")
    p.add_argument("--ingame", action="store_true", help=argparse.SUPPRESS)  # old flag: always on now
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
    p = sub.add_parser("list", help="recent runs from train/runs.csv (--task solve: train/solver-runs.csv)")
    p.add_argument("-n", type=int, default=12)
    p.add_argument("--task", choices=("levels", "solve"), default="levels")
    p.set_defaults(fn=list_runs)
    p = sub.add_parser("note", help="write a run's conclusion into train/runs.csv (or solver-runs.csv)")
    p.add_argument("name")
    p.add_argument("text")
    p.set_defaults(fn=note)
    a = ap.parse_args(argv)
    a.extra = extra
    a.fn(a)


if __name__ == "__main__":
    main()
