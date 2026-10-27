"""Training data for Goblin (the net that writes plans): levels with the solver's plans, as plan tokens.

    .venv/Scripts/python train/goblin_data.py              # every source, into data/plans/
    .venv/Scripts/python train/goblin_data.py --limit 2000 --out data/plans-try

Solves the solvable levels of the usual data files again with `glados-score score --plan-only` (the
solver throws its plans away when it labels data, so they are found again) and keeps the ones
whose plan grounds (`glados-score route` can make it concrete). Each level gets up to --plans
different shortest plans (`Model::shortest_plans`): the solver's own first, then others drawn
at random from all its shortest plans, so the net isn't taught that one of several equally
good orders or portal choices is wrong. Only the plan is found again: min_len,
needs_portals and metrics come from the data files. Duplicate levels are scored once. The
Workshop file repeated 20 times in data/v5b is left out; the Workshop levels come from
data/ws5, scored without our editor rules as the Workshop pipeline does. Each source's levels
stream through one `glados-score score` process, so no thread waits on the slowest level of a batch.

Writes, under --out (default data/plans/):
    parts/SOURCE.jsonl   every planned level of one source (a finished part is reused, but only
                         if the --limit, --budget and --plans stored beside it match this run's - parts
                         from before settings were recorded are trusted - so a run that stopped
                         picks up where it was without a smaller run passing for a whole one)
    parts/SOURCE.partial the part being made; a run that stopped carries it on, by the same
                         rule (settings beside it in SOURCE.partial.params.json)
    train.jsonl          the rest, generated and Workshop
    test-gen.jsonl       generated levels held out by a hash of their tokens (1 in 50)
    test-ws.jsonl        the ws5 test split
Each record: tokens, plan (the solver's own), plans (every plan found, plan first), min_len,
needs_portals, metrics, src ("gen" or "ws"), and workshop_id for Workshop levels. Plan tokens are documented in src/plan.rs.

Uses the CPU hard (the solver, on every thread): not while a training run or data generation
is going. Run it with the release binary (`cargo build --release` first).
"""

import argparse
import glob
import hashlib
import json
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sample import DEFAULT_GLADOS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "workshop"))
from annotations import excluded_ids  # noqa: E402

# (name, files, editor rules, src, budget): the levels a source holds. Workshop levels were
# labelled with a budget of 500k (tools/workshop); at 50k some fall to uncertain.
SOURCES = [
    ("v5b-gen", ["data/v5b/levels-[0-9].jsonl"], True, "gen", None),
    ("v5b-mutants", ["data/v5b/levels-mutants*.jsonl"], False, "gen", None),
    ("ws5-train", ["data/ws5/train.jsonl"], False, "ws", 500_000),
    ("ws5-test", ["data/ws5/test.jsonl"], False, "ws", 500_000),
]
TEST_WS = "ws5-test"
FROM_DATA = ("min_len", "needs_portals", "metrics")
TEST_EVERY = 50


def held_out(tokens):
    return int(hashlib.md5(tokens.encode()).hexdigest(), 16) % TEST_EVERY == 0


def solvable_records(patterns, seen, limit, skip_ids=()):
    """Solvable records of the files, one per level not in `seen` (which grows). Levels in
    `skip_ids` (the workbench's "exclude from test data" flag) are left out."""
    n = 0
    for pattern in patterns:
        paths = sorted(glob.glob(str(ROOT / pattern)))
        if not paths:
            sys.exit(f"no data files match {pattern}")
        for path in paths:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r.get("verdict") != "solvable" or r["tokens"] in seen:
                        continue
                    if str(r.get("workshop_id")) in skip_ids:
                        continue
                    seen.add(r["tokens"])
                    yield r
                    n += 1
                    if limit and n >= limit:
                        return


def plan_part(name, patterns, editor_rules, src, budget, out, seen, settings, args, skip_ids=()):
    """Plans one source's solvable levels into `out`; returns counts. Carries on from the
    `.partial` file a stopped run left, if it was made with the same settings."""
    cmd = [str(Path(args.glados).resolve()), "score", "--plan-only", "--budget", str(budget or args.budget),
           "--plans", str(args.plans)]
    if not editor_rules:
        cmd += ["--editor-rules", "off"]
    counts = Counter()
    tmp = out.with_suffix(".partial")
    tmp_params = out.with_suffix(".partial.params.json")
    done = set()
    if tmp.exists():
        old = json.loads(tmp_params.read_text()) if tmp_params.exists() else None
        # A partial from before partials recorded their settings is trusted, as finished parts are.
        if old is None or old == settings:
            with open(tmp, encoding="utf-8") as f:
                lines = [l for l in f if l.strip()]
            if lines and not lines[-1].endswith("\n"):
                # Stopped halfway through writing a record: drop it, it is planned again.
                lines.pop()
                with open(tmp, "w", encoding="utf-8") as f:
                    f.writelines(lines)
            done = {json.loads(l)["tokens"] for l in lines}
            print(f"  {name}: carrying on from {len(done)} planned levels in {tmp.name}", flush=True)
        else:
            print(f"  {name}: {tmp.name} was built with {old}, not {settings}: starting over", flush=True)
    tmp_params.write_text(json.dumps(settings) + "\n")
    start = time.time()
    # One glados process for the whole source: a thread feeds it the levels while this one
    # reads its answers, which come back in input order. `sent` holds the records sent and not
    # yet answered (appended before each level goes out, so its answer always finds it).
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")
    sent = deque()
    feed_error = []

    def feed():
        try:
            for r in solvable_records(patterns, seen, args.limit, skip_ids):
                if r["tokens"] in done:
                    counts["kept"] += 1
                    continue
                sent.append(r)
                proc.stdin.write(r["tokens"] + "\n")
        except BrokenPipeError:
            pass  # glados stopped: its exit code says why
        except BaseException as e:
            feed_error.append(e)
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass

    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    with open(tmp, "a" if done else "w", encoding="utf-8") as w:
        for line in proc.stdout:
            s = json.loads(line)
            r = sent.popleft()
            counts["levels"] += 1
            if "plan" in s:
                rec = {"tokens": s["tokens"], "plan": s["plan"], "plans": s.get("plans", [s["plan"]])}
                rec.update((k, r[k]) for k in FROM_DATA if k in r)
                if s["min_len"] != r.get("min_len"):
                    # The data's label is from an earlier solver, which can differ.
                    counts["min_len_differs"] += 1
                    rec["min_len"] = s["min_len"]
                rec["src"] = src
                if "workshop_id" in r:
                    rec["workshop_id"] = r["workshop_id"]
                w.write(json.dumps(rec) + "\n")
                counts["planned"] += 1
                counts["plans"] += len(rec["plans"])
            elif "plan_error" in s:
                counts["plan_error"] += 1
                if counts["plan_error"] <= 5:
                    print(f"  {name}: no plan: {s['plan_error']}", flush=True)
            else:
                counts[f"now {s['verdict']}"] += 1
            if counts["levels"] % args.every == 0:
                w.flush()
                rate = counts["levels"] / max(time.time() - start, 1e-9)
                print(f"  {name}: {counts['levels']} levels, {counts['planned']} planned, "
                      f"{rate:.0f}/s", flush=True)
    feeder.join()
    if proc.wait() != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    if feed_error:
        raise feed_error[0]
    if sent:
        sys.exit(f"{name}: glados answered {counts['levels']} levels and left {len(sent)} unanswered")
    tmp.replace(out)
    tmp_params.unlink(missing_ok=True)
    counts["seconds"] = round(time.time() - start)
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/plans")
    ap.add_argument("--only", nargs="*", help="source names to score (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="most levels per source (0: all)")
    ap.add_argument("--budget", type=int, default=50_000, help="for generated levels")
    ap.add_argument("--plans", type=int, default=8, help="most different shortest plans per level")
    ap.add_argument("--every", type=int, default=2048, help="a progress line every N levels")
    ap.add_argument("--glados", default=str(DEFAULT_GLADOS))
    args = ap.parse_args()
    if sys.platform == "win32":
        import ctypes

        # Keep Windows awake for the whole job (ES_CONTINUOUS | ES_SYSTEM_REQUIRED).
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)

    out = ROOT / args.out
    (out / "parts").mkdir(parents=True, exist_ok=True)
    excluded = excluded_ids()
    seen = set()
    for name, patterns, editor_rules, src, budget in SOURCES:
        part = out / "parts" / f"{name}.jsonl"
        params = part.with_suffix(".params.json")
        # What this run asks of the part. A part from an earlier --limit or --budget would
        # otherwise pass for finished, and later sources would skip its levels.
        settings = {"limit": args.limit, "budget": budget or args.budget, "plans": args.plans}
        if part.exists():
            old = json.loads(params.read_text()) if params.exists() else None
            # A part made before parts recorded their settings has none: trust it rather than
            # rescore it (its levels still count as seen). Parts made now record them, so one
            # built with a different --limit, --budget or --plans is rescored, not taken for
            # finished (a part from before --plans has one plan per level, and is rescored).
            if old is None or old == settings:
                # Done earlier: its levels still count as seen, so later sources skip them.
                with open(part, encoding="utf-8") as f:
                    seen.update(json.loads(l)["tokens"] for l in f if l.strip())
                print(f"{name}: kept from an earlier run ({part})")
                continue
        if args.only and name not in args.only:
            continue
        if part.exists():
            print(f"{name}: {part.name} was built with {old}, not {settings}: scoring again",
                  flush=True)
        print(f"{name}: scoring", flush=True)
        counts = plan_part(name, patterns, editor_rules, src, budget, part, seen, settings, args,
                           excluded if name == TEST_WS else ())
        params.write_text(json.dumps(settings) + "\n")
        print(f"{name}: {dict(counts)}", flush=True)

    # Split.
    files = {k: open(out / f"{k}.jsonl", "w", encoding="utf-8") for k in ("train", "test-gen", "test-ws")}
    sizes = Counter()
    plan_lens = []
    for name, *_ in SOURCES:
        part = out / "parts" / f"{name}.jsonl"
        if not part.exists():
            continue
        with open(part, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if name == TEST_WS and str(r.get("workshop_id")) in excluded:
                    continue
                key = "test-ws" if name == TEST_WS else "test-gen" if r["src"] == "gen" and held_out(r["tokens"]) else "train"
                files[key].write(line)
                sizes[key] += 1
                plan_lens.append(len(r["plan"].split()))
    for f in files.values():
        f.close()
    plan_lens.sort()
    if plan_lens:
        q = lambda p: plan_lens[min(len(plan_lens) - 1, int(p * len(plan_lens)))]
        print(f"plan tokens: median {q(0.5)}, 99% {q(0.99)}, max {plan_lens[-1]}")
    print(f"written to {out}: {dict(sizes)}")


if __name__ == "__main__":
    main()
