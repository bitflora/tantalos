"""The broadcast loop: the level net writes levels, the solver net writes their solutions, and
Portal 2 plays them on stream; then both nets train on what the round found, and the next round
asks for harder levels once the solver net keeps up (train/tier.py).

    .venv/Scripts/python train/broadcast.py start NAME [--rounds N] [--play-hours 2] [--no-train]
    .venv/Scripts/python train/broadcast.py status NAME
    .venv/Scripts/python train/broadcast.py wait NAME
    .venv/Scripts/python train/broadcast.py stop NAME

`start` runs the loop in a process of its own (like train/run.py) and serves the stream overlay
at http://localhost:8765/ (an OBS browser source, 1920x1080; docs/streaming.md). `stop` leaves a
STOP file: the loop ends after the level being played (or stops the training run going), and
play.py puts the editor autosaves back. Never kill the process.

A round, in out/broadcast/NAME/round-RR/:

1. prep: sample levels from the level net at the round's tiers, keep the solvable, portal-
   needing, novel ones (harvest.keep), write the solver net's greedy plan and pass@k samples for
   each and check them (solve_eval.solve_levels), and check the kept levels in Portal 2's editor
   (portal2.check; rejects go to data/ingame/invalid.jsonl as in every training run).
2. play: about half levels the net solved and half it didn't, easiest first. A solved plan is
   played as is (`glados route --plan`). A failed one is played as far as its steps fit, then
   the step that fits no move is tried; then the level is played again with the Rust solver's
   reference route.
3. record: episodes.jsonl (a row per level played), rounds.jsonl (a row per round), and
   model-gaps.jsonl (reference legs that failed even played carefully: for the user to judge).
4. feed back: data/harvest/broadcast-NAME-RR.jsonl (the editor-valid kept levels; ones played
   cleanly in the game x3) and data/plans/broadcast-NAME-RR.jsonl (the Rust solver's plans for
   them; ones the net failed x3).
5. train (unless --no-train): `run.py start --quick` for the level net, then the solver net, on
   their usual data plus every broadcast file so far; each is kept only if its quick eval isn't
   worse than its parent (run.py's guard), and the next round uses the final models.
6. the tier for the next round (tier.next_tier).
"""

import argparse
import datetime
import gc
import http.server
import json
import math
import random
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run  # noqa: E402  (run dirs, detaching, liveness)
import tier as tiers  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "broadcast"
OVERLAY = Path(__file__).resolve().parent / "broadcast" / "overlay.html"
GLADOS = ROOT / "target" / "release" / "glados.exe"
HARVEST = ROOT / "data" / "harvest"
PLANS = ROOT / "data" / "plans"
INGAME = ROOT / "data" / "ingame"
LEVEL_DATA = "data/v5/levels-*.jsonl"
ACTIVE = ("starting", "prep", "checking", "playing", "training")
PORT = 8765
CLEAN_REPEAT, FAILED_REPEAT = 3, 3
TICKER = 12  # lines of the overlay's ticker


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def out_dir(name):
    return OUT / name


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    """Atomically, so the overlay never reads half a file."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    for _ in range(20):  # the overlay's server may have it open for a moment
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.05)
    tmp.replace(path)


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def running():
    """(dir, status) of broadcasts going now."""
    out = []
    for d in sorted(OUT.glob("*")) if OUT.exists() else []:
        s = read_json(d / "status.json")
        if s and s.get("state") in ACTIVE and run.alive(s.get("pid", 0)):
            out.append((d, s))
    return out


# --- what the overlay shows ---------------------------------------------------------------

class State:
    """out/broadcast/NAME/state.json: everything the overlay draws, rewritten on each change."""

    def __init__(self, d):
        self.d = d
        self.path = d / "state.json"
        self.s = read_json(self.path) or {
            "name": d.name, "phase": "starting", "phase_text": "", "phase_ends": None,
            "round": 0, "rounds": None, "tier": None, "models": {}, "level": None,
            "score": {"played": 0, "net_solved": 0, "clean": 0}, "history": [],
            "training": None, "ticker": [], "previews": [],
        }

    def set(self, **kw):
        self.s.update(kw)
        self.s["updated"] = now()
        write_json(self.path, self.s)

    def say(self, text):
        print(f"[{now()}] {text}", flush=True)
        self.s["ticker"] = (self.s["ticker"] + [f"{datetime.datetime.now():%H:%M} {text}"])[-TICKER:]
        self.set()

    def phase(self, phase, text, minutes=None):
        ends = None
        if minutes:
            ends = (datetime.datetime.now() + datetime.timedelta(minutes=minutes)).isoformat(timespec="seconds")
        if phase != "playing":
            self.s["level"] = None  # the chamber card is for the level being played
        self.set(phase=phase, phase_text=text, phase_ends=ends)
        write_status(self.d, state=phase)


def serve(d, port=PORT):
    """The overlay at /, and the broadcast folder's files (state.json, previews) beside it."""

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(d), **kw)

        def do_GET(self):
            if self.path in ("/", "/overlay.html") or self.path.startswith("/?"):
                body = OVERLAY.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def log_message(self, *a):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


# --- status -------------------------------------------------------------------------------

def write_status(d, **changes):
    s = read_json(d / "status.json", {})
    s.update(changes)
    write_json(d / "status.json", s)


def stopping(d):
    return (d / "STOP").exists()


# --- prep: levels, plans, the editor check ---------------------------------------------------

def free_gpu():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sample_round(st, a, round_dir, gen_ckpt, tier):
    """Samples the round's levels at its tiers and keeps the solvable, portal-needing, novel
    ones. Returns (kept records with "tier", {tier: sampled})."""
    import torch
    import harvest
    import sample

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, _ = sample.load_checkpoint(gen_ckpt, device)
    gone = tiers.missing(vocab)
    if gone:
        raise SystemExit(f"{gen_ckpt}: the level net has no {gone} tokens")
    sampled, scored = Counter(), []
    for t, n in tiers.mix(tier, a.sample):
        st.say(f"designing {n} chambers at {tiers.NAMES[t]}")
        for start in range(0, n, a.chunk):
            k = min(a.chunk, n - start)
            lines = sample.generate(model, vocab, k, tiers.conds(t), (), 1.0, None, a.batch)
            for r in sample.score(lines, GLADOS):
                r["tier"] = t
                scored.append(r)
            sampled[t] += k
    del model
    free_gpu()
    known = [str(p) for p in sorted(ROOT.glob(LEVEL_DATA))] + [str(p) for p in sorted(HARVEST.glob("*.jsonl"))]
    kept = harvest.keep(scored, known)
    st.say(f"{len(kept)} of {sum(sampled.values())} chambers are solvable, need portals and are new")
    with open(round_dir / "sampled.jsonl", "w", encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps({k: r[k] for k in ("tokens", "verdict", "tier", "min_len") if k in r}) + "\n")
    return kept, dict(sampled)


def solve_round(st, a, kept, solve_ckpt):
    """The solver net's greedy plan and `a.k` samples for each kept level, checked."""
    import torch
    import sample
    import solve_eval

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, _ = sample.load_checkpoint(solve_ckpt, device)
    st.say(f"the solver net is solving {len(kept)} chambers")
    solved = solve_eval.solve_levels(model, vocab, kept, a.k, 1.0, GLADOS, ["conservative"], 64)
    del model
    free_gpu()
    for r, s in zip(kept, solved):
        c = s["conservative"]
        r["greedy_plan"] = s["greedy_plan"]
        r["net"] = c["greedy"]
        r["net_pass"] = any(x.get("verdict") == "solved" for x in [c["greedy"]] + c["samples"])
    n = max(len(kept), 1)
    st.say(f"the solver net solved {sum(r['net'].get('verdict') == 'solved' for r in kept) / n:.0%} greedily, "
           f"{sum(r['net_pass'] for r in kept) / n:.0%} in {a.k + 1} tries")


def write_p2c(records, folder):
    """The levels as folder/NNNNN.p2c (in order); returns the paths."""
    folder.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(GLADOS), "score", "--budget", "1", "--p2c-all", str(folder)],
                   input="\n".join(r["tokens"] for r in records) + "\n",
                   capture_output=True, text=True, check=True)
    return [folder / f"{i:05}.p2c" for i in range(1, len(records) + 1)]


def editor_check(st, round_dir, kept, files):
    """Portal 2's editor on the kept levels; sets r["game"] (valid / invalid / ...) and stores
    the rejects as invalid training data. Without the game, every level counts as unchecked."""
    import ingame
    import portal2

    results = round_dir / "game.jsonl"
    st.say(f"Portal 2's editor is checking {len(files)} chambers")
    try:
        portal2.check(files, results, launch_game=True, progress=lambda s: print(s, flush=True))
    except portal2.GameUnavailable as e:
        st.say(f"editor check skipped: Portal 2 unavailable ({e})")
        for r in kept:
            r["game"] = "unchecked"
        return
    if not results.exists():  # nothing to check: portal2.check writes no file for an empty list
        for r in kept:
            r["game"] = "unchecked"
        return
    with open(results, encoding="utf-8") as f:
        game = {Path(j["file"]).name: j for j in map(json.loads, f)}
    invalid = []
    for r, f in zip(kept, files):
        g = game.get(f.name, {})
        r["game"] = g.get("verdict", "unchecked")
        if r["game"] == "invalid":
            invalid.append({"tokens": r["tokens"], "verdict": "invalid", "game_items": g.get("invalid_items"),
                            "source": str(round_dir), "checked": now()})
    n = ingame.append_new(INGAME / "invalid.jsonl", invalid)
    st.say(f"the editor rejected {len(invalid)} of {len(files)} ({n} new invalid levels to train away from)")


def queue(kept, n, seed):
    """About half levels the net solved and half it didn't, preferring ones with a plan worth
    watching (pick.choose's bar: 5+ steps, rarely solved at random), easiest first."""
    ok = [r for r in kept if r.get("game") in ("valid", "unchecked")]
    good = [r for r in ok if r.get("min_len", 0) >= 5 and r.get("metrics", {}).get("random_success", 1) < 0.1]
    pool = good if len(good) >= n else ok
    rng = random.Random(seed)
    solved = [r for r in pool if r["net"].get("verdict") == "solved"]
    failed = [r for r in pool if r["net"].get("verdict") != "solved"]
    rng.shuffle(solved)
    rng.shuffle(failed)
    half = min(len(solved), n // 2)
    picked = solved[:half] + failed[: n - half]
    if len(picked) < n:
        picked += solved[half : half + n - len(picked)]
    return sorted(picked, key=lambda r: r.get("min_len", 0))


# --- play ---------------------------------------------------------------------------------

def route(p2c, plan=None):
    """(head, legs) of `glados route` (with `plan`: that plan, as far as it fits)."""
    import play

    return play.route(GLADOS, p2c, plan)


def mechanics(bits):
    import solve_eval

    return solve_eval.mechanics_of(bits or 0)


def card(r, p2c, preview, i, n):
    """The overlay's level card."""
    m = r.get("metrics", {})
    words = r["tokens"].split()
    size = [int(x) for x in words[1:4]] if words[:1] == ["size"] else None
    return {"n": i, "of": n, "file": p2c.name, "tier": tiers.NAMES[r["tier"]], "size": size,
            "min_len": r.get("min_len"), "chain_depth": m.get("chain_depth"),
            "portal_trips": m.get("portal_trips"), "cubes": m.get("cubes"),
            "mechanics": mechanics(m.get("mechanic_bits")), "summary": r.get("summary", ""),
            "preview": preview, "part": None, "net": None, "reference": None}


def steps_of(legs):
    return [{"text": l["text"], "action": l["action"], "status": "pending"} for l in legs]


def clean(row):
    """Played through with no leg put right by teleporting."""
    c = (row or {}).get("counts", {})
    return bool(row) and "error" not in row and not c.get("recovered") and not c.get("failed") \
        and not c.get("skipped")


def play_round(st, a, d, round_dir, picked, files, session):
    """Plays the queue until the round's time is up; returns the episodes."""
    import play

    episodes = []
    ends = time.time() + a.play_hours * 3600
    for i, (r, p2c) in enumerate(zip(picked, files), 1):
        if stopping(d) or time.time() > ends:
            break
        level = card(r, p2c, f"round-{st.s['round']:02}/queue/{p2c.with_suffix('.jpg').name}", i, len(picked))
        ep = {"round": st.s["round"], "file": str(p2c), "tokens": r["tokens"], "tier": r["tier"],
              "min_len": r.get("min_len"), "metrics": r.get("metrics"), "models": st.s["models"],
              "plan": r.get("greedy_plan"), "net": r["net"], "net_pass": r["net_pass"], "started": now()}
        t0 = time.time()
        try:
            head, legs = route(p2c, r.get("greedy_plan") or "")
        except play.NoRoute as e:
            ep["error"] = f"no route: {e}"
            st.say(f"chamber {i}: can't route the net's plan ({e})")
            episodes.append(ep)
            append_jsonl(d / "episodes.jsonl", ep)
            continue
        net = head.get("net", {})
        solved = net.get("verdict") == "solved"
        level["net"] = {"verdict": net.get("verdict"), "problem": net.get("problem"),
                        "steps_ok": net.get("steps_ok"), "steps": steps_of(legs)}
        st.set(level=level)
        st.say(f"chamber {i}/{len(picked)} ({level['tier']}, {r.get('min_len')} steps): "
               + ("the net has a solution" if solved else "the net's plan breaks at step "
                  f"{net.get('steps_ok', 0) + 1}"))

        def follow(part, steps):
            def on_leg(event, rec):
                k = rec["leg"] - 1
                if 0 <= k < len(steps):
                    steps[k]["status"] = "playing" if event == "start" else rec.get("status", "?")
                    st.set(level=level)
            return on_leg

        folder = round_dir / "play" / p2c.stem
        level["part"] = "net"
        st.set(level=level)
        if legs:
            row, _ = session.play_level(p2c, head, legs, folder / "net", follow("net", level["net"]["steps"]))
        else:
            row = {"error": "the net's plan has no step that fits"}
        ep["net_play"] = row
        if not solved and not stopping(d):
            try:
                rhead, rlegs = route(p2c)
                level["reference"] = {"steps": steps_of(rlegs)}
                level["part"] = "reference"
                st.set(level=level)
                st.say("now GLaDOS's reference solution")
                rrow, rresults = session.play_level(p2c, rhead, rlegs, folder / "reference",
                                                    follow("reference", level["reference"]["steps"]))
                ep["reference_play"] = rrow
                for rec in rresults:
                    if rec.get("cls") == "plan?":
                        append_jsonl(d / "model-gaps.jsonl", {"file": str(p2c), "tokens": r["tokens"],
                                                              "folder": str(folder / "reference"), "leg": rec})
            except play.NoRoute as e:
                ep["reference_play"] = {"error": f"no route: {e}"}
        played = ep.get("reference_play") if not solved else ep.get("net_play")
        ep["clean"] = clean(played)
        ep["seconds"] = round(time.time() - t0)
        episodes.append(ep)
        append_jsonl(d / "episodes.jsonl", ep)
        sc = st.s["score"]
        sc["played"] += 1
        sc["net_solved"] += solved
        sc["clean"] += ep["clean"]
        st.set(score=sc)
        st.say(f"chamber {i}: " + ("solved by the net" if solved and ep["clean"] else
                                   "the net's solution worked, the harness stumbled" if solved else
                                   "reference solution played" + ("" if ep["clean"] else " (with help)")))
    return episodes


# --- feeding back ----------------------------------------------------------------------------

def plans_for(records):
    """The Rust solver's plan for each record (`glados score --plan-only`), as plans data rows."""
    out = []
    if not records:
        return out
    p = subprocess.run([str(GLADOS), "score", "--plan-only", "--budget", "50000"],
                       input="\n".join(r["tokens"] for r in records) + "\n",
                       capture_output=True, text=True, check=True)
    for r, line in zip(records, p.stdout.splitlines()):
        s = json.loads(line)
        if "plan" in s:
            out.append({"tokens": s["tokens"], "plan": s["plan"], "min_len": s.get("min_len", r.get("min_len")),
                        "needs_portals": r.get("needs_portals"), "metrics": r.get("metrics"), "src": "gen",
                        "net_failed": r["net"].get("verdict") != "solved"})
    return out


def feed_back(st, name, rnd, kept, episodes, gen_ckpt):
    """Writes this round's harvest and plans files; returns their paths."""
    clean_tokens = {e["tokens"] for e in episodes if e.get("clean")}
    valid = [r for r in kept if r.get("game") in ("valid", "unchecked")]
    harvest_file = HARVEST / f"broadcast-{name}-{rnd:02}.jsonl"
    plans_file = PLANS / f"broadcast-{name}-{rnd:02}.jsonl"
    HARVEST.mkdir(parents=True, exist_ok=True)
    keys = ("tokens", "n_tokens", "verdict", "min_len", "needs_portals", "metrics")
    with open(harvest_file, "w", encoding="utf-8") as f:
        for r in valid:
            row = {k: r[k] for k in keys if k in r}
            row["n_tokens"] = len(r["tokens"].split())
            row.update(source="model", checkpoint=str(gen_ckpt), cond=" ".join(tiers.conds(r["tier"])),
                       broadcast=f"{name}/{rnd:02}")
            for _ in range(CLEAN_REPEAT if r["tokens"] in clean_tokens else 1):
                f.write(json.dumps(row) + "\n")
    plans = plans_for(valid)
    with open(plans_file, "w", encoding="utf-8") as f:
        for p in plans:
            for _ in range(FAILED_REPEAT if p.pop("net_failed") else 1):
                f.write(json.dumps(p) + "\n")
    st.say(f"{len(valid)} new chambers for the level net, {len(plans)} solutions for the solver net")
    return harvest_file, plans_file


# --- training --------------------------------------------------------------------------------

def run_name(name, rnd, task):
    return f"bc-{name}-r{rnd:02}-{task}"


def train_one(st, d, name, rnd, task, parent, hours, data):
    """A `run.py start --quick` run from `parent`; returns the model the next round uses (the
    run's final one: itself, or its parent if the run's eval was worse)."""
    rn = run_name(name, rnd, task)
    cmd = [run.PY, str(ROOT / "train" / "run.py"), "start", rn, "--quick", "--from", str(parent),
           "--hours", f"{hours:g}", "--broadcast", name, "--note",
           f"broadcast {name} round {rnd}: {task} net on the round's harvest", "--data", *data]
    if task == "solve":
        cmd += ["--task", "solve"]
    st.say(f"training the {'solver' if task == 'solve' else 'level'} net ({hours * 60:.0f} min)")
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if p.returncode:
        st.say(f"training didn't start: {(p.stderr or p.stdout).strip()[-200:]}")
        return parent
    rd = run.run_dir(rn)
    training = {"task": task, "run": rn, "started": now(), "hours": hours, "losses": [], "state": "training",
                "result": None}
    st.set(training=training)
    while True:
        s = run.read_status(rd) or {}
        training["state"] = s.get("state")
        training["losses"] = losses(rd / "log.jsonl")
        st.set(training=training)
        if s.get("state") not in run.ACTIVE or not run.alive(s.get("pid", 0)):
            break
        if stopping(d):
            subprocess.run([run.PY, str(ROOT / "train" / "run.py"), "stop", rn], cwd=ROOT)
            break
        time.sleep(10)
    s = run.read_status(rd) or {}
    final = s.get("final")
    worse = (rd / "WORSE").exists()
    if s.get("state") != "done" or not final or not Path(final).is_file():
        training["result"] = f"failed ({s.get('state')}): keeping the old model"
        st.set(training=training)
        st.say(training["result"])
        return parent
    training["result"] = "kept the old model (the new one scored worse)" if worse else "promoted"
    st.set(training=training)
    st.say(f"{'solver' if task == 'solve' else 'level'} net: {training['result']}")
    return Path(final)


def losses(log):
    """[step, train loss, val loss] points from a train.py log, thinned to ~200."""
    pts = []
    try:
        for line in open(log, encoding="utf-8"):
            e = json.loads(line)
            if "step" in e and ("loss" in e or "val_loss" in e):
                pts.append([e["step"], e.get("loss"), e.get("val_loss")])
    except (OSError, ValueError):
        pass
    every = max(1, math.ceil(len(pts) / 200))
    return pts[::every] + (pts[-1:] if pts and (len(pts) - 1) % every else [])


# --- the loop --------------------------------------------------------------------------------

def tier_stats(kept, sampled, tier):
    at = [r for r in kept if r["tier"] == tier]
    solved = sum(r["net"].get("verdict") == "solved" for r in at) / len(at) if at else None
    return solved, len(at) / max(sampled.get(tier, 0), 1)


def loop(d):
    import play

    a = argparse.Namespace(**read_json(d / "status.json")["args"])
    name = d.name
    st = State(d)
    if sys.platform == "win32":  # keep the PC (and the display, for the capture) awake
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
    httpd = serve(d, a.port)
    write_status(d, state="starting", pid=__import__("os").getpid())
    models = read_json(d / "status.json").get("models") or {"gen": a.gen, "solve": a.solve}
    tier = read_json(d / "status.json").get("tier", a.tier)
    rnd = read_json(d / "status.json").get("round", 0)
    st.set(rounds=a.rounds or None, models={k: str(v) for k, v in models.items()},
           tier={"n": tier, "name": tiers.NAMES[tier], "conds": " ".join(tiers.TIERS[tier])})
    state = "done"
    try:
        while not stopping(d) and (not a.rounds or rnd < a.rounds):
            rnd += 1
            round_dir = d / f"round-{rnd:02}"
            round_dir.mkdir(exist_ok=True)
            st.set(round=rnd, level=None, training=None,
                   tier={"n": tier, "name": tiers.NAMES[tier], "conds": " ".join(tiers.TIERS[tier])})
            t_round = time.time()

            st.phase("prep", "GLaDOS is designing test chambers", 15)
            kept, sampled = sample_round(st, a, round_dir, models["gen"], tier)
            solve_round(st, a, kept, models["solve"])
            st.phase("checking", "Checking the chambers in the editor", 10)
            files = write_p2c(kept, round_dir / "kept")
            if not a.no_check:
                editor_check(st, round_dir, kept, files)
            else:
                for r in kept:
                    r["game"] = "unchecked"
            with open(round_dir / "levels.jsonl", "w", encoding="utf-8") as f:
                for r in kept:
                    f.write(json.dumps(r) + "\n")
            picked = queue(kept, a.queue, rnd)
            qfiles = write_p2c(picked, round_dir / "queue")
            import preview

            for r, f in zip(picked, qfiles):
                try:
                    preview.write(f)
                    r["summary"] = preview.summary(f)
                except Exception as e:  # a preview is nice to have
                    print(f"preview of {f}: {e}", flush=True)
            st.set(previews=[f"round-{rnd:02}/queue/{f.with_suffix('.jpg').name}" for f in qfiles])

            episodes = []
            if picked and not stopping(d):
                st.phase("playing", "Testing", a.play_hours * 60)
                with play.Session(launch=True, stream=True, restart_every=a.restart_every) as session:
                    episodes = play_round(st, a, d, round_dir, picked, qfiles, session)

            harvest_file, plans_file = feed_back(st, name, rnd, kept, episodes, models["gen"])
            solved, yld = tier_stats(kept, sampled, tier)
            new_tier, why = tiers.next_tier(tier, solved, yld)
            n = max(len(kept), 1)
            row = {"round": rnd, "tier": tiers.NAMES[tier], "sampled": sum(sampled.values()), "kept": len(kept),
                   "game_valid": sum(r.get("game") == "valid" for r in kept) / n,
                   "net_solved": sum(r["net"].get("verdict") == "solved" for r in kept) / n,
                   "net_pass": sum(r["net_pass"] for r in kept) / n,
                   "tier_solved": solved, "tier_yield": yld, "played": len(episodes),
                   "played_net_solved": sum(e["net"].get("verdict") == "solved" for e in episodes),
                   "clean": sum(bool(e.get("clean")) for e in episodes),
                   "models": {k: str(v) for k, v in models.items()},
                   "harvest": str(harvest_file), "plans": str(plans_file), "next_tier": tiers.NAMES[new_tier],
                   "why": why}

            if not a.no_train and not stopping(d):
                st.phase("training", "GLaDOS is learning", a.gen_hours * 60 + a.solve_hours * 60 + 25)
                level_data = [LEVEL_DATA] + [str(p.relative_to(ROOT)) for p in sorted(HARVEST.glob(f"broadcast-{name}-*.jsonl"))]
                models["gen"] = train_one(st, d, name, rnd, "gen", models["gen"], a.gen_hours, level_data)
                if not stopping(d):
                    plan_data = ["data/plans/train.jsonl"] + [str(p.relative_to(ROOT)) for p in sorted(PLANS.glob(f"broadcast-{name}-*.jsonl"))]
                    models["solve"] = train_one(st, d, name, rnd, "solve", models["solve"], a.solve_hours, plan_data)
                row["models_after"] = {k: str(v) for k, v in models.items()}
            row["minutes"] = round((time.time() - t_round) / 60)
            append_jsonl(d / "rounds.jsonl", row)
            st.s["history"].append({k: row[k] for k in ("round", "tier", "net_solved", "net_pass", "played",
                                                        "played_net_solved", "clean", "game_valid")})
            st.say(f"round {rnd} done: {why}")
            tier = new_tier
            st.set(models={k: str(v) for k, v in models.items()},
                   tier={"n": tier, "name": tiers.NAMES[tier], "conds": " ".join(tiers.TIERS[tier])})
            write_status(d, round=rnd, tier=tier, models={k: str(v) for k, v in models.items()})
        if stopping(d):
            state = "stopped"
    except BaseException as e:
        state = "failed"
        (d / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        st.say(f"the loop failed: {e!r}")
        raise
    finally:
        st.phase(state, {"done": "That's all for now", "stopped": "Stopped", "failed": "Technical difficulties"}[state])
        write_status(d, state=state, ended=now())
        httpd.shutdown()


# --- commands ------------------------------------------------------------------------------

def start(a):
    d = out_dir(a.name)
    if d.exists():
        sys.exit(f"{d} exists: pick another name")
    if running():
        sys.exit(f"a broadcast is going: {running()[0][0].name}")
    if run.running():
        sys.exit(f"a training run is going: {run.name_of(run.running()[0][0])} (one GPU)")
    if not OVERLAY.exists():
        sys.exit(f"{OVERLAY} is missing")
    gen = run.checkpoint_of(a.gen)
    solve = run.checkpoint_of(a.solve)
    subprocess.run(["cargo", "build", "--release", "--bin", "glados"], cwd=ROOT, check=True)
    d.mkdir(parents=True)
    args = {k: v for k, v in vars(a).items() if k not in ("cmd", "func")}
    args.update(gen=str(gen), solve=str(solve))
    write_status(d, state="starting", started=now(), args=args, models={"gen": str(gen), "solve": str(solve)},
                 tier=a.tier, round=0)
    kw = dict(cwd=ROOT, stdin=subprocess.DEVNULL)
    log = open(d / "broadcast.log", "a", encoding="utf-8")
    cmd = [run.PY, "-u", str(Path(__file__).resolve()), "_loop", str(d)]
    if sys.platform == "win32":
        try:
            p = subprocess.Popen(cmd, creationflags=run.DETACH | run.BREAKAWAY, stdout=log, stderr=log, **kw)
        except OSError:
            p = subprocess.Popen(cmd, creationflags=run.DETACH, stdout=log, stderr=log, **kw)
    else:
        p = subprocess.Popen(cmd, start_new_session=True, stdout=log, stderr=log, **kw)
    write_status(d, pid=p.pid)
    print(f"broadcast {a.name} started (pid {p.pid}); overlay at http://localhost:{a.port}/")
    print(f"  level net {gen}\n  solver net {solve}\n  tier {tiers.NAMES[a.tier]}")


def describe(d):
    s = read_json(d / "status.json", {})
    st = read_json(d / "state.json", {})
    live = s.get("state") in ACTIVE and run.alive(s.get("pid", 0))
    state = s.get("state", "?") if live or s.get("state") not in ACTIVE else "DIED"
    lines = [f"broadcast {d.name}: {state} (started {s.get('started')}, round {st.get('round', 0)})"]
    if st.get("phase_text"):
        lines.append(f"  {st['phase_text']}")
    sc = st.get("score", {})
    lines.append(f"  played {sc.get('played', 0)}, net solved {sc.get('net_solved', 0)}, clean {sc.get('clean', 0)}")
    for row in [json.loads(l) for l in open(d / "rounds.jsonl", encoding="utf-8")] if (d / "rounds.jsonl").exists() else []:
        lines.append(f"  round {row['round']} {row['tier']}: kept {row['kept']}, net solved {row['net_solved']:.0%} "
                     f"(pass {row['net_pass']:.0%}), played {row['played']}, clean {row['clean']}; {row['why']}")
    lines += [f"  {t}" for t in st.get("ticker", [])[-5:]]
    if state in ("failed", "DIED"):
        lines.append("  last lines of broadcast.log:")
        lines += ["    " + l for l in run.tail(d / "broadcast.log", 15).splitlines()]
    steps = []
    if state == "DIED":
        steps.append("The loop process died without saying why: report the log lines above to the user. "
                     "Check that portal2.exe isn't still running before anything else.")
    elif state == "failed":
        steps.append(f"Report the failure (log above, {d / 'error.txt'}) to the user; don't restart it blindly.")
    elif state in ("done", "stopped"):
        steps.append("Tell the user the rounds' results above.")
        if (d / "model-gaps.jsonl").exists():
            steps.append(f"Tell the user {d / 'model-gaps.jsonl'} has reference legs that failed played carefully "
                         "(possible solver-rule gaps, with screenshots); they decide on rule changes.")
    if steps:
        lines.append("NEXT STEPS:")
        lines += [f"  {i}. {t}" for i, t in enumerate(steps, 1)]
    return "\n".join(lines), state


def status(a):
    print(describe(out_dir(a.name))[0])


def wait(a):
    d = out_dir(a.name)
    while True:
        s = read_json(d / "status.json", {})
        if s.get("state") not in ACTIVE or not run.alive(s.get("pid", 0)):
            break
        time.sleep(30)
    text, state = describe(d)
    print(text)
    sys.exit(0 if state in ("done", "stopped") else 1)


def stop(a):
    d = out_dir(a.name)
    if not d.exists():
        sys.exit(f"no broadcast {a.name}")
    (d / "STOP").write_text(now() + "\n")
    print(f"asked {a.name} to stop after the current level (or training run); `wait {a.name}` to see it end")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start", help="start a broadcast")
    s.add_argument("name")
    s.add_argument("--rounds", type=int, default=0, help="rounds to run (default: until stopped)")
    s.add_argument("--gen", default="v5-fresh-10h", help="level net: run name or checkpoint")
    s.add_argument("--solve", default="solve-fresh-3h", help="solver net: run name or checkpoint")
    s.add_argument("--tier", type=int, default=1, help="starting tier, 0-based (tier.py; default T2)")
    s.add_argument("--sample", type=int, default=2000, help="levels sampled per round")
    s.add_argument("--chunk", type=int, default=512, help="levels sampled and scored at a time")
    s.add_argument("--batch", type=int, default=64)
    s.add_argument("--k", type=int, default=8, help="sampled solver-net plans per level (pass@k)")
    s.add_argument("--queue", type=int, default=40, help="levels lined up to play per round")
    s.add_argument("--play-hours", type=float, default=2.0)
    s.add_argument("--gen-hours", type=float, default=0.5, help="level net training per round")
    s.add_argument("--solve-hours", type=float, default=0.33, help="solver net training per round")
    s.add_argument("--restart-every", type=int, default=20, help="restart Portal 2 after this many levels")
    s.add_argument("--no-train", action="store_true", help="play only; still writes the feedback files")
    s.add_argument("--no-check", action="store_true", help="skip the editor check of the kept levels")
    s.add_argument("--port", type=int, default=PORT, help="the overlay's port")
    s.set_defaults(func=start)
    for cmd, f in (("status", status), ("wait", wait), ("stop", stop)):
        p = sub.add_parser(cmd)
        p.add_argument("name")
        p.set_defaults(func=f)
    if len(sys.argv) == 3 and sys.argv[1] == "_loop":
        return loop(Path(sys.argv[2]))
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
