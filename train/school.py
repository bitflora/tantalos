"""Goblin's school: one round of self-play (expert iteration) by hand.

    .venv/Scripts/python train/school.py round NAME --model runs/goblin-v6init-12h-b/best.pt --round 0
    .venv/Scripts/python train/school.py round try --model runs/goblin-v6init-12h-b/best.pt --round 0 \
        --levels 256 --k 8

A round draws levels from Goblin's frontier (the levels the Rust solver runs out of budget on),
asks the net to solve them, and keeps the plans `glados-score check` accepts in a replay buffer
for the next training run (docs/goblin-next-steps-2026-10-27.md, Task 5; the plan behind it is
docs/goblin-plan-2026-10-27.md "Next" 3, E1). It writes runs/school/NAME/rRR/ (RR = two digits):

    pool.jsonl             the levels drawn this round
    attempts.jsonl         every plan tried, with its verdict
    liberal-queue.jsonl    levels only the liberal model solves: for Portal 2, not training data
    summary.json           the round's counts and timings
    work.jsonl, progress.json   the round so far, saved as it goes (kept after it ends)

and, at the run's root, `levels.jsonl` (the state: one line per level tried) and `buffer.jsonl`
(one row per solved level, in Goblin's data format; a level is replaced only by a shorter plan).

Steps, in order (each one's time goes into summary.json):

1. Pool: data/goblin/pool/*.jsonl (levels the level net wrote that the solver ran out of budget
   on; harvest.py --keep-unsolved) and the ws5 *train* levels the solver calls `uncertain`,
   never a test or probe identity. Half the draw is levels tried before but not solved on every
   try (a level never solved comes up only every 2^tries rounds); half is new.
2. Attempt: one greedy plan and --k samples per level, checked conservatively. Plans are written
   from legal steps only (`glados-score env`, as `goblin_eval.py --constrained`; --free turns it
   off), so a plan fails only by running out of tokens before the exit.
3. Prefix resampling, twice: from the longest checked prefix of the best failing plan, cut to
   128 words (64 in the second pass; PREFIX_WORDS), 16 continuations, the whole plan within
   256 words.
4. Liberal: the plans of levels still unsolved, checked with mode="liberal".
5. Polish: drop each unnecessary step from the solved plans, until no step can go; at most 4
   plans per level, different sets of verbs first.
6. Buffer: the solved levels, with their shortest plan; recheck every retained plan and reject
   held-out test/probe identities before publishing training data.
7. Summary.

An interrupted round (Ctrl-C, a crash, the machine needed) goes on where it stopped when the same
command is run again: the levels drawn, every plan tried and checked, and each step's progress
are saved after every chunk, and the seed is the saved one. A finished round (summary.json there)
isn't run again.

No Portal 2 and no training here; a round uses the GPU to sample and the CPU to check, so don't
start one while a training run, a data build or a harvest is going.
"""

import argparse
import glob
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import goblin_eval  # noqa: E402
import goblin_data  # noqa: E402
import sample  # noqa: E402
from data import BOS, PLAN_WORDS, SEP  # noqa: E402
from seeding import resolve  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "school"
POOL_FILES = ["data/goblin/pool/*.jsonl"]
WS_TRAIN = "data/ws5/train.jsonl"
TESTS = ["data/goblin/test-*.jsonl", "data/goblin/probe.jsonl"]
ROWS = 64  # plans generated at once (goblin_eval.generate_plans)
PREFIX_TRIES = 16  # continuations sampled per prefix resampling pass
# Longest prefix (words) continued in each prefix resampling pass. Constrained plans that fail
# mostly run to the 256-word limit, every step legal: continuing all of one wrote 256 more words
# after 254 (510 in pass 2) and took 2 of the try round's 2.2 hours.
PREFIX_WORDS = (128, 64)
MAX_PLANS = 4  # plans kept per solved level
# A step starts at one of these; colours, numbers and directions are its arguments (the verbs of
# src/plan.rs `decode`, which is also what the plan words list has apart from `blue`/`orange`).
VERB = set(PLAN_WORDS) - {"blue", "orange"} | {"exit"}


def read_jsonl(path, keep=lambda r: True):
    with open(path, encoding="utf-8") as f:
        return [r for r in (json.loads(l) for l in f if l.strip()) if keep(r)]


def expand(patterns):
    paths = []
    for p in patterns:
        found = sorted(glob.glob(str(ROOT / p)))
        if not found and not (ROOT / p).exists():
            sys.exit(f"no files match {p}")
        paths += found or [str(ROOT / p)]
    return paths


def split_steps(plan):
    """A plan's steps, as lists of tokens: a step starts at a verb, and the tokens up to the next
    verb are its arguments (src/plan.rs `decode` reads `putdown`'s and `loop`'s optional ones the
    same way: they are directions, not verbs)."""
    steps = []
    for t in plan.split():
        if t in VERB or not steps:
            steps.append([t])
        else:
            steps[-1].append(t)
    return steps


def join_steps(steps):
    return " ".join(t for step in steps for t in step)


def solved(result):
    return result.get("verdict") == "solved"


def verbs(plan):
    return frozenset(s[0] for s in split_steps(plan))


# --- the pool ----------------------------------------------------------------------------------

def test_identities():
    """The identities held out by data/goblin's tests and probe: never drawn or buffered."""
    out = set()
    for path in expand(TESTS):
        for r in read_jsonl(path):
            out.add(r.get("identity") or goblin_data.identity(r))
    return out


def candidates():
    """{identity: record} of the levels a round may draw: the self-play pool, plus the ws5 train
    levels the solver calls uncertain. A record keeps `tokens`, `identity` and `frontier`."""
    held = test_identities()
    out = {}
    for path in expand([WS_TRAIN]):
        for r in read_jsonl(path, lambda r: r.get("solver_verdict") == "uncertain"):
            ident = goblin_data.identity(r)
            if ident not in held:
                out.setdefault(ident, {"tokens": r["tokens"], "identity": ident, "frontier": True,
                                       "source": f"ws5 {r.get('workshop_id')}"})
    for path in expand(POOL_FILES):
        for r in read_jsonl(path):
            ident = goblin_data.identity(r)
            if ident not in held:
                out.setdefault(ident, {"tokens": r["tokens"], "identity": ident, "frontier": True,
                                       "source": r.get("checkpoint") or path})
    return out


def load_state(path):
    """{identity: row} from runs/school/NAME/levels.jsonl."""
    if not path.exists():
        return {}
    return {r["identity"]: r for r in read_jsonl(path)}


def save_state(path, state):
    write_jsonl(path, state.values())


def draw(pool, state, n, rnd, rng):
    """Half levels tried before that weren't solved on every try (preferring ones solved at
    least once; a never-solved one only every 2^tries rounds), half levels never tried. Fills a
    short side from the other. Returns (identities, the counts)."""
    tried = [i for i, s in state.items() if s["tries"] and len(s["solved_rounds"]) < s["tries"]
             and i in pool]
    some = [i for i in tried if state[i]["solved_rounds"]]
    never_solved = [i for i in tried if not state[i]["solved_rounds"]
                    and rnd % (2 ** state[i]["tries"]) == 0]
    fresh = [i for i in pool if i not in state or not state[i]["tries"]]
    rng.shuffle(some)
    rng.shuffle(never_solved)
    rng.shuffle(fresh)
    # Prefer levels solved sometimes, but keep the never-solved ones in the rotation.
    half = min(n // 2, len(some) + len(never_solved))
    first = (some + never_solved)[:half]
    second = fresh[: n - len(first)]
    if len(first) + len(second) < n:  # one side was short
        selected = set(first) | set(second)
        rest = [i for i in (some + never_solved + fresh) if i not in selected]
        rng.shuffle(rest)
        first += rest[: n - len(first) - len(second)]
    counts = {"tried": len(first), "new": len(second)}
    return first + second, counts


# --- the steps ---------------------------------------------------------------------------------

def level_prompts(model, vocab, levels):
    """Each level's `<bos> level <sep>` prompt ids, or the reason it can't be asked."""
    return [goblin_eval.plan_prompt(vocab, lv["tokens"], model.cfg.block_size) for lv in levels]


def check_items(items, glados, mode):
    """One verdict list per item, in order; retain batching and build each level once."""
    out = []
    step = 128
    for start in range(0, len(items), step):
        part = items[start : start + step]
        results = goblin_eval.check(part, glados, mode)
        at = 0
        for item in part:
            n = len(item["plans"])
            out.append(results[at : at + n])
            at += n
    return out


def attempt(model, vocab, levels, args, save):
    """One greedy plan and `args.k` samples per level, conservatively checked. Sets lv["plans"],
    lv["kind"] and lv["results"]; levels that have them already (a resumed round) are skipped."""
    for start in range(0, len(levels), args.chunk):
        part = levels[start : start + args.chunk]
        if all("results" in lv for lv in part):
            continue
        prompts = level_prompts(model, vocab, part)
        asked = [i for i, p in enumerate(prompts) if not isinstance(p, str)]
        greedy = goblin_eval.generate_plans(model, vocab, [prompts[i] for i in asked], 1.0, 1, ROWS, args.env)
        sampled = goblin_eval.generate_plans(
            model, vocab, [prompts[i] for i in asked for _ in range(args.k)], 1.0, None, ROWS, args.env)
        for j, i in enumerate(asked):
            part[i]["plans"] = [greedy[j]] + sampled[j * args.k : (j + 1) * args.k]
            part[i]["kind"] = ["greedy"] + ["sample"] * args.k
        for lv in part:
            if lv.get("plans") is None:
                lv["plans"], lv["kind"], lv["results"] = [], [], []
        checked = [lv for lv in part if lv["plans"]]
        items = [{"tokens": lv["tokens"], "plans": lv["plans"]} for lv in checked]
        for lv, verdicts in zip(checked, check_items(items, args.glados, "conservative")):
            lv["results"] = verdicts
        save()
        print(f"  attempt: {start + len(part)} of {len(levels)} levels", flush=True)


def best_prefix(lv, most):
    """The longest checked prefix of the level's best failing plan, cut to the whole steps that
    fit in `most` words, or None."""
    best = None
    for plan, r in zip(lv["plans"], lv["results"]):
        steps_ok = r.get("steps_ok") or 0
        if steps_ok and (best is None or steps_ok > best[0]):
            best = (steps_ok, plan)
    if best is None:
        return None
    steps = split_steps(best[1])
    if best[0] > len(steps):
        return None  # the plan is shorter than its prefix: nothing to cut
    kept, words = [], 0
    for step in steps[: best[0]]:
        if words + len(step) > most:
            break
        kept.append(step)
        words += len(step)
    return join_steps(kept) if kept else None


def prefix_pass(model, vocab, levels, args, i, save, n=PREFIX_TRIES):
    """Prefix resampling pass `i` (from 0): the levels with no solved plan continue the longest
    prefix that checked, cut to PREFIX_WORDS[i] words; prefix and continuation together stay
    within goblin_eval's plan limit. A level's `passes` counts the passes done for it (a resumed
    round skips them). Returns how many levels were asked."""
    block = model.cfg.block_size
    todo = []
    for lv in levels:
        if lv.get("passes", 0) > i:
            continue
        if lv.get("asking") == i:  # interrupted while asked: ask again
            todo.append(lv)
            continue
        prefix = best_prefix(lv, PREFIX_WORDS[i])
        words = (prefix or "").split()
        if (any(solved(r) for r in lv["results"]) or prefix is None or prefix in lv.get("prefixes", [])
                or len(lv["tokens"].split()) + len(words) + 3 >= block):
            lv["passes"] = i + 1
            continue
        lv.setdefault("prefixes", []).append(prefix)
        lv["asking"] = i
        todo.append(lv)
    for start in range(0, len(todo), args.chunk):
        part = todo[start : start + args.chunk]
        prompts = [[BOS] + vocab.encode(lv["tokens"].split()) + [SEP] + vocab.encode(lv["prefixes"][-1].split())
                   for lv in part]
        outs = goblin_eval.generate_plans(model, vocab, [p for p in prompts for _ in range(n)], 1.0, None, ROWS,
                                          args.env)
        for j, lv in enumerate(part):
            prefix = lv["prefixes"][-1]
            lv["plans"] += [f"{prefix} {o}".strip() for o in outs[j * n : (j + 1) * n]]
            lv["kind"] += ["prefix"] * n
        items = [{"tokens": lv["tokens"], "plans": lv["plans"][-n:]} for lv in part]
        results = check_items(items, args.glados, "conservative")
        for lv, verdicts in zip(part, results):
            lv["results"] += verdicts
            lv["passes"] = i + 1
            del lv["asking"]
        save()
    return len(todo)


def liberal_pass(levels, args, save):
    """The still-unsolved levels' plans, checked with the liberal model (sets lv["liberal"]).
    Returns the queue rows."""
    todo = [lv for lv in levels if not any(solved(r) for r in lv["results"]) and lv["plans"]
            and "liberal" not in lv]
    for start in range(0, len(todo), args.chunk):
        part = todo[start : start + args.chunk]
        results = check_items([{"tokens": lv["tokens"], "plans": lv["plans"]} for lv in part],
                              args.glados, "liberal")
        for lv, verdicts in zip(part, results):
            lv["liberal"] = [p for p, r in zip(lv["plans"], verdicts) if solved(r)]
        save()
    return [{"tokens": lv["tokens"], "identity": lv["identity"], "plan": lv["liberal"][0]}
            for lv in levels if lv.get("liberal")]


def polish(tokens, plans, glados):
    """`plans` with every step dropped that isn't needed: for each plan, remove one step at a
    time and keep the shorter one while `check` still solves it."""
    steps = [split_steps(p) for p in plans]
    active = [i for i, s in enumerate(steps) if len(s) > 1]
    while active:
        cands = {}
        for i in active:
            seen, c = set(), []
            for j in range(len(steps[i])):
                plan = join_steps(steps[i][:j] + steps[i][j + 1 :])
                if plan not in seen and plan:
                    seen.add(plan)
                    c.append(plan)
            cands[i] = c
        items = [{"tokens": tokens, "plans": cands[i]} for i in active]
        results = check_items(items, glados, "conservative")
        following = []
        for i, verdicts in zip(active, results):
            hit = next((c for c, r in zip(cands[i], verdicts) if solved(r)), None)
            if hit is not None:
                steps[i] = split_steps(hit)
                if len(steps[i]) > 1:
                    following.append(i)
        active = following
    return [join_steps(s) for s in steps]


def keep_plans(plans, limit=MAX_PLANS):
    """At most `limit` distinct plans, preferring ones whose sets of verbs differ."""
    out, seen, seen_verbs = [], set(), set()
    for p in plans:
        if not p or p in seen:
            continue
        v = verbs(p)
        if v in seen_verbs and len(out) < limit:
            continue
        seen.add(p)
        seen_verbs.add(v)
        out.append(p)
        if len(out) >= limit:
            break
    for p in plans:  # fewer than `limit` different verb sets: fill with the rest
        if len(out) >= limit:
            break
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out[:limit]


def buffer_row(lv, plans, rnd):
    shortest = min(plans, key=lambda p: len(split_steps(p)))
    plans = [shortest] + [p for p in plans if p != shortest]
    return {"tokens": lv["tokens"], "plan": shortest, "plans": plans, "alt_plans": [],
            "min_len": len(split_steps(shortest)), "src": "school", "identity": lv["identity"],
            "found_by": "goblin", "round": rnd, "trusted": "check"}


def validate_buffer(rows, held, glados):
    """Check every retained plan, including old rows, before publishing training data."""
    rows = list(rows)
    if any(r["identity"] in held for r in rows):
        raise ValueError("buffer contains held-out test/probe identities; audit it before continuing")
    for row in rows:
        if not row.get("plans") or row.get("plan") not in row["plans"]:
            raise ValueError(f"buffer row {row['identity']} has no retained primary plan")
    results = check_items(rows, glados, "conservative")
    for row, verdicts in zip(rows, results):
        if not all(solved(r) for r in verdicts):
            raise ValueError(f"buffer row {row['identity']} has an invalid plan; audit it before continuing")


def write_jsonl(path, rows):
    write_atomic(path, "".join(json.dumps(row) + "\n" for row in rows))


def write_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def round_(a):
    base = OUT / a.name
    out = base / f"r{a.round:02d}"
    if (out / "summary.json").exists():
        sys.exit(f"{out} is finished: pick another --round")
    progress_path, work_path = out / "progress.json", out / "work.jsonl"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else None
    if out.exists() and progress is None:
        sys.exit(f"{out} exists but has no progress.json: remove it or pick another --round")
    if progress:
        if a.seed is not None and a.seed != progress["seed"]:
            sys.exit(f"{out} was started with --seed {progress['seed']}: resume with that one, or no --seed")
        if progress["model"] != str(a.model) or progress["k"] != a.k or progress["free"] != a.free:
            sys.exit(f"{out} was started with --model {progress['model']} --k {progress['k']}"
                     f"{' --free' if progress['free'] else ''}: resume with the same")
        progress.setdefault("resumed", []).append(time.strftime("%Y-%m-%d %H:%M:%S"))
    else:
        progress = {"seed": resolve(a.seed), "model": str(a.model), "k": a.k, "free": a.free, "timings": {}}
    seed = progress["seed"]
    rng = random.Random(seed)
    torch.manual_seed(seed + len(progress.get("resumed", [])))  # a resume samples afresh
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timings = progress["timings"]
    levels = []
    step_t = [time.time()]

    def save():
        """The round so far: every level's plans and verdicts, and the time each step took."""
        write_jsonl(work_path, levels)
        write_atomic(progress_path, json.dumps(progress, indent=2) + "\n")

    def timed(step):
        timings[step] = round(timings.get(step, 0) + time.time() - step_t[0], 1)
        step_t[0] = time.time()

    # 1. Pool (a resumed round keeps the levels it drew, with what was done on them).
    pool = candidates()
    state = load_state(base / "levels.jsonl")
    if work_path.exists():
        levels = read_jsonl(work_path)
        held = test_identities()
        if any(lv["identity"] in held for lv in levels):
            sys.exit("saved round contains held-out test/probe identities: audit it before resuming")
        drawn_counts = progress["drawn"]
        print(f"resuming {out} (seed {seed}): {len(levels)} levels", flush=True)
    else:
        ids, drawn_counts = draw(pool, state, a.levels, a.round, rng)
        levels = [dict(pool[i]) for i in ids]
        progress["drawn"] = drawn_counts
        write_jsonl(out / "pool.jsonl", ({k: lv[k] for k in ("tokens", "identity", "frontier", "source")}
                                         for lv in levels))
        timed("pool")
        save()
        print(f"drew {len(levels)} levels ({drawn_counts['tried']} tried before, {drawn_counts['new']} new) "
              f"of {len(pool):,} candidates", flush=True)
    step_t[0] = time.time()
    model, vocab, _ = sample.load_checkpoint(a.model, device)
    a.env = None if a.free else goblin_eval.Env(a.glados)

    # 2. Attempt.
    attempt(model, vocab, levels, a, save)
    timed("attempt")

    # 3. Prefix resampling, twice.
    for i in range(len(PREFIX_WORDS)):
        n = prefix_pass(model, vocab, levels, a, i, save)
        timed("prefix")
        save()
        if n:
            print(f"  prefix pass {i + 1}: {n} levels", flush=True)
    if a.env:
        a.env.stop()

    # 4. Liberal.
    queue = liberal_pass(levels, a, save)
    write_jsonl(out / "liberal-queue.jsonl", queue)
    timed("liberal")
    save()

    # 5. Polish.
    for j, lv in enumerate(levels):
        if "solved_plans" in lv:
            continue
        done = [p for p, r in zip(lv["plans"], lv["results"]) if solved(r)]
        lv["solved_plans"] = keep_plans(polish(lv["tokens"], done, a.glados)) if done else []
        if j % 16 == 15:
            save()
    timed("polish")
    save()

    # 6. Buffer (done again on a resume: a level's row comes out the same, and the state counts
    # this round once per level).
    buffer_path = base / "buffer.jsonl"
    buffer = {r["identity"]: r for r in read_jsonl(buffer_path)} if buffer_path.exists() else {}
    newly, shorter = [], []
    for lv in levels:
        if not lv["solved_plans"]:
            continue
        row = buffer_row(lv, lv["solved_plans"], a.round)
        old = buffer.get(lv["identity"])
        if old is None:
            buffer[lv["identity"]] = row
            newly.append(lv["identity"])
        elif row["min_len"] < old.get("min_len", 10**9):
            buffer[lv["identity"]] = row
            shorter.append(lv["identity"])
    validate_buffer(buffer.values(), test_identities(), a.glados)
    write_jsonl(buffer_path, buffer.values())
    timed("buffer")

    # State: tries, the rounds it was solved in, the last round drawn.
    for lv in levels:
        row = state.setdefault(lv["identity"], {"tokens": lv["tokens"], "identity": lv["identity"],
                                                "tries": 0, "solved_rounds": [], "last_round": None})
        row["tokens"] = lv["tokens"]
        if row["last_round"] == a.round:
            continue
        row["tries"] += 1
        row["last_round"] = a.round
        if lv["solved_plans"]:
            row["solved_rounds"].append(a.round)
    save_state(base / "levels.jsonl", state)

    # 7. Summary.
    greedy = sum(solved(lv["results"][0]) for lv in levels if lv["results"])
    by_sample = sum(not solved(lv["results"][0])
                    and any(solved(r) for r, k in zip(lv["results"], lv["kind"]) if k == "sample")
                    for lv in levels if lv["results"])
    by_prefix = sum(any(solved(r) for r, k in zip(lv["results"], lv["kind"]) if k == "prefix")
                    and not any(solved(r) for r, k in zip(lv["results"], lv["kind"]) if k != "prefix")
                    for lv in levels)
    liberal = sum(bool(lv.get("liberal")) for lv in levels if not lv["solved_plans"])
    solved_levels = [lv for lv in levels if lv["solved_plans"]]
    beyond = sum(lv["frontier"] for lv in solved_levels)
    kinds = Counter(k for lv in levels for k, r in zip(lv["kind"], lv["results"]) if solved(r))
    summary = {
        "name": a.name, "round": a.round, "model": str(a.model), "seed": seed, "drawn": len(levels),
        "drawn_tried": drawn_counts["tried"], "drawn_new": drawn_counts["new"],
        "candidates": len(pool), "k": a.k, "constrained": not a.free,
        "solved_greedy": greedy, "solved_by_sample": by_sample, "solved_by_prefix": by_prefix,
        "solved_by_kind": dict(kinds), "liberal_only": liberal,
        "solved_levels": len(solved_levels), "newly_solved": len(newly),
        "shorter_than_before": len(shorter), "beyond_bfs": beyond,
        "buffer": len(buffer), "plans_kept": sum(len(lv["solved_plans"]) for lv in solved_levels),
        "seconds": round(sum(timings.values()), 1), "timings": timings,
    }
    write_jsonl(out / "attempts.jsonl", ({"tokens": lv["tokens"], "identity": lv["identity"],
                                "frontier": lv["frontier"],
                                "plans": [{"plan": p, "kind": k, "verdict": r["verdict"],
                                           "steps_ok": r.get("steps_ok")}
                                          for p, k, r in zip(lv["plans"], lv["kind"], lv["results"])],
                                "liberal": lv.get("liberal", []),
                                "kept": lv["solved_plans"]} for lv in levels))
    # The completion marker is published only after all other round outputs.
    write_atomic(out / "summary.json", json.dumps(summary, indent=2) + "\n")

    pct = lambda n: f"{n}/{len(levels)} ({n / max(len(levels), 1):.0%})"  # noqa: E731
    print(f"\nschool {a.name} round {a.round}: {len(levels)} levels drawn "
          f"({drawn_counts['tried']} tried before, {drawn_counts['new']} new), k={a.k}")
    print(f"  solved greedy            {pct(greedy)}")
    print(f"  solved by a sample       {pct(by_sample)}")
    print(f"  solved by prefix         {pct(by_prefix)}")
    print(f"  liberal only             {pct(liberal)}")
    print(f"  solved levels            {pct(len(solved_levels))} ({beyond} beyond the Rust search)")
    print(f"  newly solved             {len(newly)} (buffer: {len(buffer)} levels, "
          f"{summary['plans_kept']} plans kept)")
    print(f"  time                     {summary['seconds']:.0f}s " +
          ", ".join(f"{k} {v:.0f}s" for k, v in timings.items()))
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("round", help="one round of self-play")
    r.add_argument("name")
    r.add_argument("--model", required=True, help="Goblin checkpoint")
    r.add_argument("--round", type=int, required=True, help="round number (writes rNN)")
    r.add_argument("--levels", type=int, default=4096, help="levels to draw")
    r.add_argument("--k", type=int, default=32, help="sampled plans per level")
    r.add_argument("--chunk", type=int, default=128, help="levels sampled and checked at a time")
    r.add_argument("--glados", default=str(sample.DEFAULT_GLADOS))
    r.add_argument("--free", action="store_true", help="let the net write any token (no glados-score env)")
    r.add_argument("--seed", type=int, default=None,
                   help="default: a fresh, unpredictable seed (printed; pass it back to repeat)")
    r.set_defaults(func=round_)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
