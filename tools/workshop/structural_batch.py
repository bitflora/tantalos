"""Generate a resumable, source-capped pool of structural Workshop remixes."""
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import msvcrt
from pathlib import Path
import random
import re
import sqlite3
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
STRUCTURAL = ROOT / "target/release/tantalos-structural.exe"
SCORE = ROOT / "target/release/tantalos-score.exe"
SEED = 20261114


def rows(path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def workshop_id(row):
    identity = row.get("identity", "")
    return str(row.get("workshop_id") or (identity[3:] if identity.startswith("ws:") else ""))


def sources(min_parent, min_wires):
    data = ROOT / "data"
    train_ids = {workshop_id(r): r.get("title") for r in rows(data / "ws6/train.jsonl")}
    holdout = {workshop_id(r) for r in rows(data / "ws6/test.jsonl")}
    for path in (data / "goblin").glob("test*.jsonl"):
        holdout.update(workshop_id(r) for r in rows(path))
    annotations = ROOT / "train/workshop-annotations.jsonl"
    holdout.update(workshop_id(r) for r in rows(annotations) if r.get("excluded"))
    for name in ("structural-scale", "structural-followup"):
        path = ROOT / "out" / name / "parents.jsonl"
        if path.exists():
            holdout.update(workshop_id(r) for r in rows(path))
    previous = ROOT / "out/structural-2026-11-10"
    holdout.update(path.name.split("-")[1] for path in previous.glob("search-*") if path.is_dir())

    pool = {}
    for row in rows(data / "goblin/train.jsonl"):
        if row.get("src") != "ws" or not row.get("plan") or row.get("min_len", 0) < min_parent:
            continue
        wid = workshop_id(row)
        if not wid or wid not in train_ids or wid in holdout:
            continue
        if wid in pool and row.get("variant") != "r0":
            continue
        tokens = row["tokens"]
        targets = re.findall(r"(?:^| )> (\S+)", tokens)
        if sum(target != "exit" for target in targets) < min_wires:
            continue
        plans = list(dict.fromkeys([row["plan"], *(row.get("plans") or []), *(row.get("alt_plans") or [])]))
        pool[wid] = dict(workshop_id=wid, title=train_ids[wid], tokens=tokens,
                         plans=[plan for plan in plans if plan], source_variant=row.get("variant"),
                         archived_actions=row["min_len"])
    ordered = [pool[wid] for wid in sorted(pool)]
    random.Random(SEED).shuffle(ordered)
    ordered.sort(key=lambda r: min(r["archived_actions"] // 4, 3), reverse=True)
    return ordered


def invoke(command, timeout, **options):
    return subprocess.run(command, encoding="utf-8", errors="replace",
                          capture_output=True, timeout=timeout, **options)


def old_plans_fail(tokens, plans, deadline):
    request = json.dumps(dict(tokens=tokens, plans=plans)) + "\n"
    for mode in ("conservative", "liberal"):
        result = invoke([SCORE, "check", "--mode", mode], max(1, min(20, deadline-time.time())), input=request)
        if result.returncode:
            return False
        checks = json.loads(result.stdout)["results"]
        if len(checks) != len(plans) or any(c["verdict"] not in ("illegal", "unfinished") for c in checks):
            return False
    return True


def generate(parent, scratch, deadline):
    summary = Counter()
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        work = Path(directory)
        tokens = work / "parent.tokens.txt"
        archive = work / "archive.txt"
        plan = work / "parent.plan.txt"
        output = work / "results"
        tokens.write_text(parent["tokens"], encoding="utf-8")
        archive.write_text("\n".join(parent["plans"]), encoding="utf-8")
        try:
            recovered = invoke([STRUCTURAL, "recover", tokens, archive], max(1, min(40, deadline-time.time())))
            if recovered.returncode:
                return "parent rejected", summary, []
            plan.write_text(recovered.stdout.strip(), encoding="utf-8")
            search = invoke([STRUCTURAL, tokens, plan, output], max(1, min(150, deadline-time.time())))
            if search.returncode:
                return "search failed", summary, []
        except subprocess.TimeoutExpired:
            return "timeout", summary, []
        summary.update({name: int(count) for name, count in re.findall(
            r'"([^"]+)": (\d+),', (output / "report.txt").read_text())})
        initial = sorted(rows(output / "candidates.jsonl"), key=lambda r: (
            r["shortest_proven"], r["chain"], r["operation_distance"], r["actions"]), reverse=True)
        accepted = []
        seen = set()
        for row in initial:
            if time.time() >= deadline or len(accepted) >= 2:
                break
            path = output / f"{row['id']}.tokens.txt"
            try:
                if not old_plans_fail(row["tokens"], parent["plans"], deadline):
                    summary["archive plan rejection"] += 1
                    continue
                audit = invoke([STRUCTURAL, "audit", path], max(1, min(50, deadline-time.time())))
                if audit.returncode:
                    summary["audit failed"] += 1
                    continue
            except subprocess.TimeoutExpired:
                summary["audit timeout"] += 1
                continue
            checks = [json.loads(line) for line in audit.stdout.splitlines()]
            lengths = [r["actions"] for r in checks if r.get("verdict") == "solved"]
            if len(lengths) != 4 or min(lengths) < 16 or min(lengths)*5 < row["parent_actions"]*4:
                summary["audit solution rejection"] += 1
                continue
            route = rows(output / f"{row['id']}.route.jsonl")
            signature = tuple(r["action"] for r in route if r.get("action") not in (None,"walk","fire","enter"))
            if signature in seen:
                summary["same source action sequence"] += 1
                continue
            seen.add(signature)
            wid = parent["workshop_id"]
            row.update(workshop_id=wid, identity=f"ws:{wid}", source_title=parent["title"],
                       source_variant=parent["source_variant"], src="workshop-structural",
                       source_url=f"https://steamcommunity.com/sharedfiles/filedetails/?id={wid}",
                       split="train", audit=checks, min_actions=min(lengths), operations=signature,
                       editor_verified=False, playtest_verified=False)
            accepted.append(row)
        return "complete", summary, accepted


def snapshot(db, out, pool_size, started, deadline, running):
    destination = out / "qualified.jsonl"
    temporary = destination.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        for (record,) in db.execute("SELECT record FROM variants ORDER BY wid, rowid"):
            file.write(record + "\n")
    temporary.replace(destination)
    status = dict(running=running, pool=pool_size, elapsed_seconds=round(time.time()-started),
                  deadline_epoch=deadline, jobs=dict(db.execute("SELECT status,count(*) FROM jobs GROUP BY status")),
                  candidates=db.execute("SELECT count(*) FROM variants").fetchone()[0],
                  yielding_sources=db.execute("SELECT count(DISTINCT wid) FROM variants").fetchone()[0])
    temporary = out / "status.tmp"
    temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
    temporary.replace(out / "status.json")
    print(json.dumps(status), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--min-parent", type=int, default=12, help="Minimum archived solution length")
    parser.add_argument("--min-wires", type=int, default=2, help="Minimum non-exit control wires")
    parser.add_argument("--exclude-primary", action="store_true", help="Skip the default source queue")
    parser.add_argument("--limit", type=int, help="Limit the queue for a smoke test")
    args = parser.parse_args()
    if min(args.hours, args.workers, args.min_parent, args.min_wires) <= 0 or (
        args.limit is not None and args.limit <= 0
    ):
        parser.error("hours, workers, and source limits must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "runner.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise SystemExit(f"Another batch driver owns {args.out}") from error
        run(args)


def run(args):
    scratch = args.out / "scratch"
    scratch.mkdir(exist_ok=True)
    started = time.time()
    deadline = started + args.hours * 3600
    pool = sources(args.min_parent, args.min_wires)
    if args.exclude_primary:
        primary = {row["workshop_id"] for row in sources(12, 2)}
        pool = [row for row in pool if row["workshop_id"] not in primary]
    if args.limit:
        pool = pool[:args.limit]
    print(f"Queued {len(pool)} sources", flush=True)
    with sqlite3.connect(args.out / "run.sqlite") as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS jobs(wid TEXT PRIMARY KEY, status TEXT, summary TEXT);
            CREATE TABLE IF NOT EXISTS variants(tokens TEXT PRIMARY KEY, wid TEXT, record TEXT);
        """)
        done = {wid for (wid,) in db.execute("SELECT wid FROM jobs")}
        queue = iter(r for r in pool if r["workshop_id"] not in done)
        snapshot(db, args.out, len(pool), started, deadline, True)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            active = {}
            def refill():
                while len(active) < args.workers and time.time() < deadline:
                    parent = next(queue, None)
                    if parent is None:
                        break
                    active[executor.submit(generate, parent, scratch, deadline)] = parent["workshop_id"]
            refill()
            last_snapshot = time.time()
            while active:
                completed, _ = wait(active, timeout=20, return_when=FIRST_COMPLETED)
                for future in completed:
                    wid = active.pop(future)
                    try:
                        status, summary, variants = future.result()
                    except (OSError, ValueError, KeyError) as error:
                        print(f"{wid}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
                        status, summary, variants = "worker error", {}, []
                    with db:
                        db.execute("INSERT INTO jobs VALUES(?,?,?)", (wid, status, json.dumps(summary)))
                        for row in variants:
                            db.execute("INSERT OR IGNORE INTO variants VALUES(?,?,?)",
                                       (row["tokens"], wid, json.dumps(row)))
                refill()
                if time.time() - last_snapshot >= 30:
                    snapshot(db, args.out, len(pool), started, deadline, True)
                    last_snapshot = time.time()
        snapshot(db, args.out, len(pool), started, deadline, False)


if __name__ == "__main__":
    main()
