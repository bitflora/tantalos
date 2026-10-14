"""Start, watch and stop training runs with one command each. Meant to need no thought.

    .venv/Scripts/python train/run.py start v4-a --hours 2 --note "first v4 run, fresh"
    .venv/Scripts/python train/run.py start v4-b --hours 1 --from v4-a --note "v4-a continued"
    .venv/Scripts/python train/run.py status          # the newest run (or: status v4-a)
    .venv/Scripts/python train/run.py wait v4-a       # blocks until it ends, then prints status
    .venv/Scripts/python train/run.py stop v4-a
    .venv/Scripts/python train/run.py list            # recent rows of train/runs.csv
    .venv/Scripts/python train/run.py note v4-a "what it showed"

`start` returns at once: the run carries on in the background, detached, and survives the
terminal or session that started it. It trains (train.py) for --hours, evaluates best.pt
(evaluate.py, about 5 minutes), with --ingame checks the samples in Portal 2 (ingame.py; the game
must be free), and records the run in train/runs.csv. Everything lands in runs/NAME:
status.json (what it's doing now), run.log (each phase), train.log, evaluate.log,
eval/report.md.

Settings: data/v4 (all levels-*.jsonl) unless --data; block 1024, batch 16. A fresh run uses
lr 1e-3 down to 1e-4; one continuing another (--from) uses 2e-4 down to 2e-5, as the warm runs
in runs.csv did. Anything after `--` goes to train.py as is.

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
ACTIVE = ("starting", "training", "evaluating", "ingame")
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
    """--from: a run name, a run directory or a .pt file."""
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


def start(a):
    d = run_dir(a.name)
    if d.exists():
        sys.exit(f"{d.relative_to(ROOT).as_posix()} already exists: pick another name")
    busy = running()
    if busy:
        sys.exit(f"{name_of(busy[0][0])} is still running (one run at a time). "
                 f"Wait for it, or: train/run.py stop {name_of(busy[0][0])}")
    if not GLADOS.exists():
        sys.exit(f"{GLADOS} is missing: run `cargo build --release` first")
    data = []
    for pattern in a.data:
        hits = sorted(glob.glob(str(ROOT / pattern))) or sorted(glob.glob(pattern))
        if not hits:
            sys.exit(f"--data {pattern}: no such file")
        data += [Path(h).resolve() for h in hits]
    vocab = Path(a.vocab).resolve() if a.vocab else data[0].parent / "vocab.txt"
    if not vocab.exists():
        sys.exit(f"{vocab} is missing: pass --vocab")
    init = checkpoint_of(a.from_) if a.from_ else None
    lr, min_lr, warmup = ("2e-4", "2e-5", "200") if init else ("1e-3", "1e-4", "300")

    train = [
        "train/train.py", "--data", *map(str, data), "--vocab", str(vocab), "--out", str(d),
        "--block", "1024", "--batch", "16", "--eval-every", "2000", "--score-n", "64",
        "--lr", lr, "--min-lr", min_lr, "--warmup", warmup, "--note", a.note,
    ]
    if init:
        train += ["--init", str(init)]
    train += a.extra
    evaluate = ["train/evaluate.py", str(d / "best.pt"), "--data", *map(str, data)]

    d.mkdir(parents=True)
    write_status(
        d, state="starting", started=now().isoformat(timespec="seconds"), hours=a.hours,
        ingame=a.ingame, init=str(init or ""), train=train, evaluate=evaluate,
    )
    kw = dict(cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cmd = [PY, "-u", str(Path(__file__).resolve()), "_pipeline", str(d)]
    if sys.platform == "win32":
        try:
            p = subprocess.Popen(cmd, creationflags=DETACH | BREAKAWAY, **kw)
        except OSError:  # the job the session runs in doesn't allow breaking away
            p = subprocess.Popen(cmd, creationflags=DETACH, **kw)
    else:
        p = subprocess.Popen(cmd, start_new_session=True, **kw)
    write_status(d, pid=p.pid)
    ends = now() + datetime.timedelta(hours=a.hours)
    print(f"started runs/{a.name} (pid {p.pid}) from {init.relative_to(ROOT).as_posix() if init else 'scratch'}")
    print(f"training ends about {ends:%H:%M}, results about {ends + datetime.timedelta(minutes=6 + 30 * a.ingame):%H:%M}")
    print(f"check:  .venv/Scripts/python train/run.py status {a.name}")
    print(f"wait:   .venv/Scripts/python train/run.py wait {a.name}")


def pipeline(d):
    """The detached part of `start`: train, evaluate, (check in game), record."""
    d = Path(d)
    s = write_status(d, pid=__import__("os").getpid())
    log = open(d / "run.log", "a", encoding="utf-8")

    def say(msg):
        log.write(f"[{now():%H:%M:%S}] {msg}\n")
        log.flush()

    if sys.platform == "win32":  # don't let Windows sleep between phases either
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)

    def phase(state, args, logname):
        write_status(d, state=state, phase_started=now().isoformat(timespec="seconds"))
        say(f"{state}: {' '.join(args[:2])}")
        with open(d / logname, "a", encoding="utf-8") as out:
            code = subprocess.call([PY, "-u", *args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                                   creationflags=NO_WINDOW if sys.platform == "win32" else 0)
        if code:
            raise RuntimeError(f"{state} failed (exit {code}); see {d.name}/{logname}")

    try:
        until = (now() + datetime.timedelta(hours=s["hours"])).strftime("%Y-%m-%d %H:%M:%S")
        write_status(d, train_until=until)
        phase("training", s["train"] + ["--until", until], "train.log")
        if not (d / "best.pt").exists():
            raise RuntimeError("training wrote no best.pt; see train.log")
        phase("evaluating", s["evaluate"], "evaluate.log")
        if s.get("ingame"):
            try:
                phase("ingame", ["train/ingame.py", str(d / "eval")], "ingame.log")
            except RuntimeError as e:  # the game being busy shouldn't sink the run
                say(f"{e} (carrying on)")
        write_status(d, state="done", ended=now().isoformat(timespec="seconds"))
        say("done")
    except BaseException as e:  # noqa: BLE001
        write_status(d, state="failed", ended=now().isoformat(timespec="seconds"), error=str(e))
        say(f"failed: {e}\n{traceback.format_exc()}")
    finally:
        sys.path.insert(0, str(ROOT / "train"))
        import record

        record.safely(d)


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
    elif state in ("evaluating", "ingame"):
        lines.append(f"{name}: {state.upper()} (since {hhmm(s['phase_started'])}; training is over)")
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
        lines.append(line)
    ev = d / "eval" / "eval.json"
    if ev.exists():
        v = json.loads(ev.read_text())["validity"]
        lines.append(f"  EVAL: {v['solvable']:.0%} solvable, {v['parsed']:.0%} parse "
                     f"(the main score; differences under ~10 points are noise)")
        lines.append(f"  report: runs/{name_of(d)}/eval/report.md")
    ig = d / "eval" / "ingame" / "report.md"
    if ig.exists():
        for l in ig.read_text(encoding="utf-8").splitlines():
            if "Game-valid, of levels our checker passes" in l:
                lines.append("  in game: " + l.strip("-* ").replace("**", ""))
    if state in ("failed", "died"):
        log = {"evaluating": "evaluate.log", "ingame": "ingame.log"}.get(s.get("state"), "train.log")
        lines.append(f"  last lines of {log}:")
        lines += ["    " + l for l in tail(d / log).splitlines()]
    if state in ("done", "failed", "died", "stopped"):
        lines.append(f"  next: write what it showed with  train/run.py note {name_of(d)} \"...\"")
    return "\n".join(lines)


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
    sys.exit(0 if (read_status(d) or {}).get("state") == "done" else 1)


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
    with open(ROOT / "train" / "runs.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[-a.n:]
    print(f"{'run':32} {'started':16} {'min':>4} {'solv%':>5} {'val':>6}  from / notes")
    for r in rows:
        notes = (r["init"] + " | " if r["init"] else "") + r["notes"]
        print(f"{r['run'][:32]:32} {r['started']:16} {r['minutes']:>4} {r['eval_solvable'] or '-':>5} "
              f"{r['best_val'] or '-':>6}  {notes[:90]}")
    live = running()
    if live:
        print(f"\nrunning now: runs/{name_of(live[0][0])}")


def note(a):
    sys.path.insert(0, str(ROOT / "train"))
    import record

    record.update(run_dir(a.name), a.text)
    print(f"noted on runs/{name_of(run_dir(a.name))} in train/runs.csv")


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1 :]
    if argv[:1] == ["_pipeline"]:
        return pipeline(argv[1])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("start", help="start a run in the background")
    p.add_argument("name", help="the run's name: it goes to runs/NAME")
    p.add_argument("--hours", type=float, required=True, help="training time (evaluation comes after)")
    p.add_argument("--note", required=True, help="what the run is for (goes in train/runs.csv)")
    p.add_argument("--from", dest="from_", default=None, help="continue from this run (or .pt)")
    p.add_argument("--data", nargs="+", default=["data/v4/levels-*.jsonl"])
    p.add_argument("--vocab", default=None, help="default: vocab.txt beside the first data file")
    p.add_argument("--ingame", action="store_true", help="check samples in Portal 2 afterwards")
    p.set_defaults(fn=start)
    for cmd, fn, h in [("status", status, "what a run is doing"), ("wait", wait, "block until it ends")]:
        p = sub.add_parser(cmd, help=h)
        p.add_argument("name", nargs="?", help="default: the newest run")
        p.set_defaults(fn=fn)
    p = sub.add_parser("stop", help="stop a run")
    p.add_argument("name")
    p.set_defaults(fn=stop)
    p = sub.add_parser("list", help="recent runs from train/runs.csv")
    p.add_argument("-n", type=int, default=12)
    p.set_defaults(fn=list_runs)
    p = sub.add_parser("note", help="write a run's conclusion into train/runs.csv")
    p.add_argument("name")
    p.add_argument("text")
    p.set_defaults(fn=note)
    a = ap.parse_args(argv)
    a.extra = extra
    a.fn(a)


if __name__ == "__main__":
    main()
