"""Goblin's data: levels with the Rust solver's plans as plan tokens, in data/goblin/.

    .venv/Scripts/python train/goblin_data.py                    # select, plan, assemble
    .venv/Scripts/python train/goblin_data.py --limit 2000 --out data/goblin-try
    .venv/Scripts/python train/goblin_data.py --only ws ws-test  # plan only these parts now
    .venv/Scripts/python train/goblin_data.py --deep             # and ws-deep (overnight)
    .venv/Scripts/python train/goblin_data.py --assemble         # just the last step again

The lessons of the level net's data (v6, 2026-10-26) carried over to Goblin
(docs/goblin-plan-2026-10-27.md):
- Workshop levels first, as their 8 mirrored/turned versions (`tantalos-data augment --plans`
  moves the plans with them) rather than copies;
- generated levels that change the room's state over trivial ones, without near-copies
  (tools/select_generated.py);
- held-out sets split by puzzle (a Workshop level, its mutants and versions on one side);
- long plans not starved: rows are copied or thinned by plan length.

Steps:
1. select (cheap): each part's levels to plan, in parts/NAME.todo.jsonl.
2. plan (the CPU on every thread): `tantalos-score score --plan-only --plans 8` (and
   `--alternatives`, plans that solve it another way) into parts/NAME.jsonl. Resumable: a
   stopped run carries on from parts/NAME.partial; a finished part is kept while its settings
   (in NAME.params.json) match.
3. assemble: splits by identity, mechanism primers (small levels showing one or two mechanics,
   up to PRIMERS each; a tenth held out as the probe set), augmentation, balancing, then
   train.jsonl, the test sets and BUILD.md.

The parts:
    ws        ws5 train levels our solver solves (bunny-hop/glitch levels left out), 500k
    ws-test   ws5 test levels it solves (annotation-excluded ones left out), 500k
    ws-deep   ws5 train levels out of the 500k budget, tried at 1M (--deep; the solved
              ones join ws)
    ws6       data/ws6 train levels (the 2026-11-02 download) that ws5 lacks and our solver solves,
              500k; they join ws. data/ws6/test stays out of training; the held-out sets still
              come from ws5
    mut      Workshop mutants (data/v5b/levels-mutants*.jsonl): at most MAX_PER_PARENT per
              parent, near-copies dropped; mutants of ws5 test levels make test-mut
    gen       generated levels (data/v5b/levels-[1-5].jsonl): every one that changes the room's
              state twice or more, SIMPLE of the others, near-copies dropped
    v6        levels the level net wrote (data/goblin/harvest/*.jsonl, harvest.py), if any

The sets written (each row: tokens, plan, plans, alt_plans, min_len, states, needs_portals,
metrics, src, identity, and workshop_id / variant / primer where they apply):
    train.jsonl        everything not held out, augmented and balanced
    test-ws.jsonl      ws-test, never augmented
    test-mut.jsonl     mutants of ws5 test levels (up to 1,024)
    test-gen.jsonl     generated levels held out by identity (1 in 50)
    test-v6.jsonl      level-net levels held out by identity (1 in 20)
    test-hard.jsonl    held-out levels whose search took more than HARD states
    test-frontier.jsonl ws5 test levels our solver runs out of budget on (no plans: what
                       Goblin can only get by beating the search)
    probe.jsonl        held-out primers, `primer` naming the mechanic each shows

Uses the CPU hard: not while a training run or data generation is going. Run it with the
release binary (`cargo build --release` first).
"""

import argparse
import glob
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data import MECHANICS  # noqa: E402
import rated  # noqa: E402
from sample import DEFAULT_TANTALOS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "workshop"))
from annotations import excluded_ids  # noqa: E402

AUGMENT = ROOT / "target" / "release" / "tantalos-data.exe"
TRICKS = ROOT / "tools" / "workshop" / "excluded-tricks.jsonl"
WS_TRAIN, WS_TEST = "data/ws5/train.jsonl", "data/ws5/test.jsonl"
WS6_TRAIN = "data/ws6/train.jsonl"
MUTANTS = ["data/v5b/levels-mutants*.jsonl"]
GENERATED = ["data/v5b/levels-[1-5].jsonl"]
HARVEST = ["data/goblin/harvest/*.jsonl"]

MAX_PER_PARENT = 20  # mutants kept per Workshop level
SIMPLE = 50_000  # generated levels kept that change the room's state less than twice
PRIMERS = 2_000  # primers per mechanic
HARD = 50_000  # states: a held-out level harder than this goes to test-hard too
TEST_MUT = 1024
FRONTIER = 512
COPIES_WS = 2  # each Workshop version written twice, as in the level net's data
LONG, SHORT = 0.20, 0.15  # 12+ step plans at least, 3-4 step plans at most, of train rows
NEAR = 0.9


class Part:
    def __init__(self, name, files, verdicts, editor_rules, src, budget, alternatives, test=False):
        self.name, self.files, self.verdicts = name, files, verdicts
        self.editor_rules, self.src, self.budget = editor_rules, src, budget
        self.alternatives, self.test = alternatives, test


PARTS = [
    Part("ws", [WS_TRAIN], ("solvable",), False, "ws", 500_000, 4),
    Part("ws-test", [WS_TEST], ("solvable",), False, "ws", 500_000, 0, test=True),
    Part("ws-deep", [WS_TRAIN], ("uncertain",), False, "ws", 1_000_000, 4),
    Part("ws6", [WS6_TRAIN], ("solvable",), False, "ws", 500_000, 4),
    Part("mut", MUTANTS, ("solvable",), False, "mut", 50_000, 2),
    Part("gen", GENERATED, ("solvable",), True, "gen", 50_000, 2),
    Part("v6", HARVEST, ("solvable",), True, "v6", 50_000, 2),
]


def digest(text):
    return int(hashlib.md5(text.encode()).hexdigest(), 16)


def identity(r):
    """The puzzle a record belongs to: its Workshop level (a mutant's parent's), or its tokens.
    Every held-out split goes by it, so versions and mutants of one puzzle stay together."""
    if r.get("workshop_id"):
        return f"ws:{r['workshop_id']}"
    parent = r.get("mutated_from")
    if parent is not None and str(parent).isdigit() and len(str(parent)) > 6:
        return f"ws:{parent}"
    return f"tok:{digest(r['tokens']):032x}"


def mechanics_of(r):
    bits = (r.get("metrics") or {}).get("mechanic_bits") or 0
    return {m for i, m in enumerate(MECHANICS) if (bits >> i) & 1} - {"portal_trip"}


def state_changes(r):
    return rated.mechanics(r.get("metrics"))[0]


def near_copies():
    """tools/select_generated.py's steps 2 and 3 (exact and near-copy removal)."""
    spec = importlib.util.spec_from_file_location("select_generated", ROOT / "tools" / "select_generated.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.step2, mod.step3


def read(patterns):
    for pattern in patterns:
        for path in sorted(glob.glob(str(ROOT / pattern))):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)


def ws_test_ids():
    return {str(r["workshop_id"]) for r in read([WS_TEST]) if r.get("workshop_id")}


def tricks_ids():
    if not TRICKS.exists():
        return set()
    return {json.loads(l)["id"] for l in open(TRICKS, encoding="utf-8") if l.strip()}


def select(part, args, rng):
    """The records of `part` to plan, as a list (step 1)."""
    if part.name in ("ws", "ws-test", "ws-deep", "ws6"):
        skip = tricks_ids() if part.name != "ws-test" else excluded_ids()
        if part.name == "ws6":
            skip = skip | {str(r.get("workshop_id")) for r in read([WS_TRAIN, WS_TEST])}
        out = []
        for r in read(part.files):
            verdict = r.get("solver_verdict") if part.name == "ws-deep" else r.get("verdict")
            if verdict in part.verdicts and str(r.get("workshop_id")) not in skip:
                out.append(r)
        return out[: args.limit] if args.limit else out
    rows = []
    for r in read(part.files):
        if r.get("verdict") in part.verdicts:
            rows.append(r)
            if args.limit and len(rows) >= args.limit * 4:
                break
    if part.name == "mut":
        by_parent = defaultdict(list)
        for r in rows:
            by_parent[identity(r)].append(r)
        rows = []
        for group in by_parent.values():
            rng.shuffle(group)
            rows += group[:MAX_PER_PARENT]
    if part.name == "gen":
        changing = [r for r in rows if (state_changes(r) or 0) >= 2]
        rest = [r for r in rows if (state_changes(r) or 0) < 2]
        rng.shuffle(rest)
        rows = changing + rest[:SIMPLE]
    if part.name in ("mut", "gen"):
        step2, step3 = near_copies()
        lines = [(part.name, r["tokens"], json.dumps(r) + "\n") for r in rows]
        rows = [json.loads(line) for _, line in step3(step2(lines))]
    rng.shuffle(rows)
    return rows[: args.limit] if args.limit else rows


def plan_part(part, todo, out, settings, args):
    """Plans `todo` (records) into `out`; returns counts. Carries on from the `.partial` file a
    stopped run left, if it was made with the same settings."""
    cmd = [str(Path(args.tantalos).resolve()), "score", "--plan-only", "--budget", str(part.budget),
           "--plans", str(args.plans)]
    if part.alternatives:
        cmd += ["--alternatives", str(part.alternatives)]
    if not part.editor_rules:
        cmd += ["--editor-rules", "off"]
    counts = Counter()
    tmp = out.with_suffix(".partial")
    tmp_params = out.with_suffix(".partial.params.json")
    done = set()
    if tmp.exists():
        old = json.loads(tmp_params.read_text()) if tmp_params.exists() else None
        if old == settings:
            with open(tmp, encoding="utf-8") as f:
                lines = [l for l in f if l.strip()]
            if lines and not lines[-1].endswith("\n"):
                # Stopped halfway through writing a record: drop it, it is planned again.
                lines.pop()
                with open(tmp, "w", encoding="utf-8") as f:
                    f.writelines(lines)
            done = {json.loads(l)["source_tokens"] for l in lines}
            print(f"  {part.name}: carrying on from {len(done)} planned levels in {tmp.name}", flush=True)
        else:
            print(f"  {part.name}: {tmp.name} was built with {old}, not {settings}: starting over", flush=True)
    tmp_params.write_text(json.dumps(settings) + "\n")
    start = time.time()
    # One tantalos process for the whole part: a thread feeds it the levels while this one reads
    # its answers, which come back in input order.
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")
    sent = deque()
    feed_error = []

    def feed():
        try:
            for r in todo:
                if r["tokens"] in done:
                    counts["kept"] += 1
                    continue
                sent.append(r)
                proc.stdin.write(r["tokens"] + "\n")
        except BrokenPipeError:
            pass  # tantalos stopped: its exit code says why
        except BaseException as e:  # noqa: BLE001
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
                rec = {"tokens": s["tokens"], "source_tokens": r["tokens"], "plan": s["plan"],
                       "plans": s.get("plans", [s["plan"]]), "alt_plans": s.get("alt_plans", []),
                       "min_len": s["min_len"], "states": s.get("states")}
                for k in ("needs_portals", "metrics", "workshop_id", "mutated_from", "stars", "votes"):
                    if k in r:
                        rec[k] = r[k]
                rec["src"] = part.src
                rec["identity"] = identity(r)
                if s["min_len"] != r.get("min_len") and r.get("min_len") is not None:
                    counts["min_len_differs"] += 1
                w.write(json.dumps(rec) + "\n")
                counts["planned"] += 1
                counts["plans"] += len(rec["plans"])
                counts["alt_plans"] += len(rec["alt_plans"])
            else:
                counts[f"now {s['verdict']}"] += 1
            if counts["levels"] % args.every == 0:
                w.flush()
                rate = counts["levels"] / max(time.time() - start, 1e-9)
                print(f"  {part.name}: {counts['levels']} levels, {counts['planned']} planned, "
                      f"{rate:.1f}/s", flush=True)
    feeder.join()
    if proc.wait() != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    if feed_error:
        raise feed_error[0]
    if sent:
        sys.exit(f"{part.name}: tantalos answered {counts['levels']} levels and left {len(sent)} unanswered")
    tmp.replace(out)
    tmp_params.unlink(missing_ok=True)
    counts["seconds"] = round(time.time() - start)
    return counts


def load_part(out, name):
    p = out / "parts" / f"{name}.jsonl"
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def primers(rows, rng):
    """Small levels whose solution shows one or two mechanics, up to PRIMERS per mechanic, the
    rarest mechanic first. Returns {row index: mechanic}. Where a mechanic has too few of
    those, larger levels with up to three mechanics make up the rest."""
    picked = {}
    for strict in (True, False):
        pool = defaultdict(list)
        for i, r in enumerate(rows):
            ms = mechanics_of(r)
            n = len(r["tokens"].split())
            if not ms or i in picked:
                continue
            if (strict and n <= 220 and len(ms) <= 2) or (not strict and n <= 320 and len(ms) <= 3):
                for m in ms:
                    pool[m].append(i)
        have = Counter(picked.values())
        for m in sorted(pool, key=lambda m: len(pool[m])):
            idx = [i for i in pool[m] if i not in picked]
            rng.shuffle(idx)
            for i in idx[: max(PRIMERS - have[m], 0)]:
                picked[i] = m
    return picked


def augment(rows, syms_of, label):
    """`rows` with their plans moved by each symmetry `syms_of(row)` lists (`tantalos-data augment
    --plans`): one row per version that keeps a plan, its other fields as the original's."""
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        for s in syms_of(r):
            groups[s].append(i)
    out = []
    dropped = Counter()
    for s, idx in sorted(groups.items()):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "in.jsonl"
            with open(src, "w", encoding="utf-8") as f:
                for i in idx:
                    r = rows[i]
                    f.write(json.dumps({"tokens": r["tokens"], "plan": r["plan"], "plans": r["plans"],
                                        "alt_plans": r.get("alt_plans", [])}) + "\n")
            p = subprocess.run([str(AUGMENT), "augment", "--plans", "--syms", s, str(src)],
                               capture_output=True, text=True, encoding="utf-8", check=True)
        for line in p.stdout.splitlines():
            v = json.loads(line)
            if "error" in v:
                dropped[v["error"].split(" in ")[0][:30]] += 1
                continue
            r = dict(rows[idx[v["line"] - 1]])
            r.update(tokens=v["tokens"], plan=v["plan"], plans=v.get("plans", [v["plan"]]),
                     alt_plans=v.get("alt_plans", []), variant=v["variant"])
            r.pop("source_tokens", None)
            out.append(r)
    print(f"  augment {label}: {len(rows)} rows -> {len(out)} versions; left out: {dict(dropped) or 'none'}",
          flush=True)
    return out


def bucket(n):
    return "3-4" if n <= 4 else "5-7" if n <= 7 else "8-11" if n <= 11 else "12+"


def balance(rows, rng):
    """Copies of 12+ step rows until they make up LONG of the rows (at most 3 copies each), and
    3-4 step rows thinned to SHORT."""
    by = defaultdict(list)
    for r in rows:
        by[bucket(r["min_len"])].append(r)
    n = len(rows)
    short = by["3-4"]
    if len(short) > SHORT * n:
        rng.shuffle(short)
        keep = int(SHORT * (n - len(short)) / (1 - SHORT))
        by["3-4"] = short[:keep]
    n = sum(len(v) for v in by.values())
    long = by["12+"]
    if long and len(long) < LONG * n:
        want = int(LONG * (n - len(long)) / (1 - LONG))
        copies = min(3, -(-want // len(long)))
        extra = (long * copies)[: want]
        by["12+"] = extra
    out = [r for v in by.values() for r in v]
    rng.shuffle(out)
    return out


def assemble(out, args, rng):
    """Step 3: the sets, from the planned parts."""
    t0 = time.time()
    test_ids = {f"ws:{i}" for i in ws_test_ids()}
    sets = defaultdict(list)
    train = defaultdict(list)  # by src: rows still to augment
    counts = Counter()
    for r in load_part(out, "ws") + load_part(out, "ws-deep") + load_part(out, "ws6"):
        train["ws"].append(r)
    for r in load_part(out, "ws-test"):
        r.pop("source_tokens", None)
        sets["test-ws"].append(r)
    for r in load_part(out, "mut"):
        if r["identity"] in test_ids:
            sets["test-mut"].append(r)
        else:
            train["mut"].append(r)
    for r in load_part(out, "gen"):
        (sets["test-gen"] if digest(r["identity"]) % 50 == 0 else train["gen"]).append(r)
    for r in load_part(out, "v6"):
        (sets["test-v6"] if digest(r["identity"]) % 20 == 0 else train["v6"]).append(r)
    rng.shuffle(sets["test-mut"])
    sets["test-mut"] = sets["test-mut"][:TEST_MUT]
    # Primers come out of the generated levels and mutants; a tenth (by identity) are the probe.
    small = train["gen"] + train["mut"]
    picked = primers(small, rng)
    for i, m in picked.items():
        r = small[i]
        r["primer"] = m
        r["src"] = "primer"
    train["gen"] = [r for r in train["gen"] if "primer" not in r]
    train["mut"] = [r for r in train["mut"] if "primer" not in r]
    prim = [small[i] for i in picked]
    sets["probe"] = [r for r in prim if digest(r["identity"]) % 10 == 0]
    train["primer"] = [r for r in prim if digest(r["identity"]) % 10 != 0]
    for name in ("test-ws", "test-mut", "test-gen", "test-v6"):
        sets["test-hard"] += [r for r in sets[name] if (r.get("states") or 0) > HARD]
    frontier = [r for r in read([WS_TEST]) if r.get("solver_verdict") == "uncertain"
                and str(r.get("workshop_id")) not in excluded_ids()]
    rng.shuffle(frontier)
    sets["test-frontier"] = [{"tokens": r["tokens"], "workshop_id": r.get("workshop_id"), "src": "ws",
                              "identity": identity(r), "stars": r.get("stars")} for r in frontier[:FRONTIER]]
    for rows in sets.values():
        for r in rows:
            r.pop("source_tokens", None)

    syms = ["r0", "r90", "r180", "r270", "m0", "m90", "m180", "m270"]
    rows = []
    rows += augment(train["ws"], lambda r: syms, "ws") * COPIES_WS
    rows += augment(train["v6"], lambda r: rng.sample(syms, 2), "v6")
    for src in ("mut", "gen", "primer"):
        rows += augment(train[src], lambda r: [rng.choice(syms)], src)
    for r in rows:
        counts[f"before balance {r['src']}"] += 1
    before = Counter(bucket(r["min_len"]) for r in rows)
    rows = balance(rows, rng)
    after = Counter(bucket(r["min_len"]) for r in rows)

    def write(name, items):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in items:
                f.write(json.dumps(r) + "\n")

    write("train", rows)
    for name, items in sets.items():
        write(name, items)
    too_long = sum(len(r["tokens"].split()) + len(r["plan"].split()) + 3 > 1024 for r in rows if r["src"] == "ws")
    n_ws = sum(r["src"] == "ws" for r in rows)
    by_src = Counter(r["src"] for r in rows)
    prim_by = Counter(r["primer"] for r in train["primer"])
    lines = [
        f"# data/goblin, built {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "Made by train/goblin_data.py (its docstring says how). Rows by source in train.jsonl:",
        "",
        "| src | rows |",
        "|---|---|",
        *[f"| {k} | {v:,} |" for k, v in sorted(by_src.items())],
        f"| all | {len(rows):,} |",
        "",
        "Plan length buckets (min_len) before and after balancing:",
        "",
        "| bucket | before | after |",
        "|---|---|---|",
        *[f"| {b} | {before[b]:,} ({before[b] / max(sum(before.values()), 1):.0%}) | "
          f"{after[b]:,} ({after[b] / max(len(rows), 1):.0%}) |" for b in ("3-4", "5-7", "8-11", "12+")],
        "",
        f"Workshop rows longer than a 1024 block (level + plan + 3): {too_long:,} of {n_ws:,} "
        f"({too_long / max(n_ws, 1):.1%}).",
        "",
        "Held-out sets:",
        "",
        "| set | levels |",
        "|---|---|",
        *[f"| {k} | {len(v):,} |" for k, v in sorted(sets.items())],
        "",
        "Primers per mechanic (train side; a tenth more in probe.jsonl):",
        "",
        ", ".join(f"{m} {n}" for m, n in prim_by.most_common()),
        "",
        f"Assembled in {time.time() - t0:.0f}s.",
    ]
    (out / "BUILD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/goblin")
    ap.add_argument("--only", nargs="*", help="parts to plan now (default: all but ws-deep)")
    ap.add_argument("--deep", action="store_true", help="plan ws-deep too (1M states a level: hours)")
    ap.add_argument("--assemble", action="store_true", help="skip planning: assemble the parts there are")
    ap.add_argument("--limit", type=int, default=0, help="most levels per part (0: all), for a try")
    ap.add_argument("--plans", type=int, default=8, help="most different shortest plans per level")
    ap.add_argument("--every", type=int, default=2048, help="a progress line every N levels")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tantalos", default=str(DEFAULT_TANTALOS))
    args = ap.parse_args()
    if sys.platform == "win32":
        import ctypes

        # Keep Windows awake for the whole job (ES_CONTINUOUS | ES_SYSTEM_REQUIRED).
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    out = ROOT / args.out
    (out / "parts").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    for part in PARTS if not args.assemble else []:
        if part.name == "ws-deep" and not args.deep and not (args.only and "ws-deep" in args.only):
            continue
        if args.only and part.name not in args.only:
            continue
        if part.name == "v6" and not any(glob.glob(str(ROOT / p)) for p in part.files):
            print("v6: no harvested levels yet (data/goblin/harvest/), skipped")
            continue
        dest = out / "parts" / f"{part.name}.jsonl"
        params = dest.with_suffix(".params.json")
        settings = {"limit": args.limit, "budget": part.budget, "plans": args.plans,
                    "alternatives": part.alternatives, "seed": args.seed}
        if dest.exists() and params.exists() and json.loads(params.read_text()) == settings:
            print(f"{part.name}: kept from an earlier run ({dest.relative_to(ROOT).as_posix()})")
            continue
        todo_path = dest.with_suffix(".todo.jsonl")
        todo_params = dest.with_suffix(".todo.params.json")
        if todo_path.exists() and todo_params.exists() and json.loads(todo_params.read_text()) == settings:
            todo = [json.loads(l) for l in open(todo_path, encoding="utf-8")]
        else:
            print(f"{part.name}: selecting", flush=True)
            todo = select(part, args, random.Random(f"{args.seed} {part.name}"))
            with open(todo_path, "w", encoding="utf-8", newline="\n") as f:
                for r in todo:
                    f.write(json.dumps(r) + "\n")
            todo_params.write_text(json.dumps(settings) + "\n")
        print(f"{part.name}: planning {len(todo)} levels", flush=True)
        counts = plan_part(part, todo, dest, settings, args)
        params.write_text(json.dumps(settings) + "\n")
        print(f"{part.name}: {dict(counts)}", flush=True)
    assemble(out, args, rng)


if __name__ == "__main__":
    main()
