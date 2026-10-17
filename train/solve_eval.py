"""Evaluate a solver-net checkpoint (train.py --task solve): how often its plans solve levels.

    .venv/Scripts/python train/solve_eval.py runs/solve-fresh-2h/best.pt

For each level it writes one greedy plan and --k sampled ones, and `glados check` replays
them on our solver's model. The levels:
  test-gen   generated levels held out of training (data/plans/test-gen.jsonl, --n-gen of them)
  test-ws    the Workshop test split our solver solves (data/plans/test-ws.jsonl)
  published  Workshop levels our solver can't solve (`verdict: published` in data/ws3/test.jsonl,
             --n-published of them). The conservative check uses the very rules the solver
             fails on, so it mostly says no; `published-liberal` (check --mode liberal) is a
             weak signal. The real test for these is playing them in Portal 2.

Writes <checkpoint dir>/eval/ (or --out): report.md, eval.json and solutions.jsonl (every
level with its plans and verdicts). The headline number is the share of test-gen levels the
greedy plan solves.
"""

import argparse
import json
import random
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from data import BOS, EOS, LEN_BUCKETS, MECHANICS, SEP, len_bucket  # noqa: E402
from sample import DEFAULT_GLADOS, load_checkpoint  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HEADLINE = "test-gen"
# Verdicts for levels the net was never asked about: tokens its vocabulary lacks, or a level
# that leaves no room in the block for a plan.
SKIPPED = ("unknown_tokens", "too_long")
# Most plan tokens written per level: 80-odd steps, far beyond the solver's plans. (An untrained
# net that never writes <eos> would otherwise fill the block for every level.)
MAX_PLAN = 512


def plan_prompt(vocab, tokens, block_size):
    """`<bos> level <sep>` as ids, or the reason the level can't be asked (a SKIPPED verdict)."""
    words = tokens.split()
    if any(w not in vocab.stoi for w in words):
        return "unknown_tokens"
    if len(words) + 3 >= block_size:
        return "too_long"
    return [BOS] + vocab.encode(words) + [SEP]


def generate_plans(model, vocab, prompts, temperature=1.0, top_k=None, rows=64):
    """One plan (a token string) per prompt. Prompts are sorted by length and generated `rows`
    at a time, so equal lengths share a batch. top_k=1 is greedy."""
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    plans = [None] * len(prompts)
    for start in range(0, len(order), rows):
        chunk = order[start : start + rows]
        outs = model.generate([prompts[i] for i in chunk], MAX_PLAN, EOS, temperature, top_k)
        for i, o in zip(chunk, outs):
            plans[i] = " ".join(vocab.decode(o))
    return plans


def check(items, glados=DEFAULT_GLADOS, mode="conservative"):
    """One `glados check` verdict per plan, in order. An item is `{tokens, plans: [...]}` (or
    `{tokens, plan}` for one), optionally with `min_len`. An item's plans share one input line,
    so `glados check` builds the level's model once for all of them. Returns the per-plan result
    dicts, flattened in item order."""
    lines = []
    for it in items:
        line = {k: it[k] for k in ("tokens", "min_len") if it.get(k) is not None}
        line["plans"] = it["plans"] if "plans" in it else [it["plan"]]
        lines.append(line)
    args = [str(Path(glados).resolve()), "check", "--mode", mode]
    text = "".join(json.dumps(line) + "\n" for line in lines)
    out = subprocess.run(args, input=text, capture_output=True, text=True, encoding="utf-8", check=True)
    results = []
    for line in out.stdout.splitlines():
        results.extend(json.loads(line)["results"])
    want = sum(len(it["plans"]) if "plans" in it else 1 for it in items)
    if len(results) != want:
        raise ValueError(f"glados check answered {len(results)} of {want} plans")
    return results


def summarize(results):
    """Shares of verdicts over `results` (glados check output)."""
    n = max(len(results), 1)
    verdicts = Counter(r["verdict"] for r in results)
    return {
        "n": len(results),
        "parsed": 1 - (verdicts["unparsed"] + sum(verdicts[v] for v in SKIPPED)) / n,
        "solved": verdicts["solved"] / n,
        "optimal": sum(bool(r.get("optimal")) for r in results) / n,
        "verdicts": dict(verdicts),
    }


def solve_levels(model, vocab, records, k, temperature, glados, modes, rows):
    """Greedy and `k` sampled plans for each record, checked in each mode. Returns one dict
    per record: the record's fields, and per mode `greedy` (verdict) and `samples` (verdicts)."""
    block = model.cfg.block_size
    prompts = [plan_prompt(vocab, r["tokens"], block) for r in records]
    asked = [i for i, p in enumerate(prompts) if not isinstance(p, str)]
    greedy = dict(zip(asked, generate_plans(model, vocab, [prompts[i] for i in asked], 1.0, 1, rows)))
    sampled = generate_plans(model, vocab, [prompts[i] for i in asked for _ in range(k)], temperature, None, rows)
    samples = {i: sampled[j * k : (j + 1) * k] for j, i in enumerate(asked)}
    out = [{"tokens": r["tokens"], "min_len": r.get("min_len"), "metrics": r.get("metrics"),
            "workshop_id": r.get("workshop_id"), "greedy_plan": greedy.get(i), "sample_plans": samples.get(i, [])}
           for i, r in enumerate(records)]
    items = [{"tokens": o["tokens"], "min_len": o["min_len"],
              "plans": [o["greedy_plan"]] + o["sample_plans"]}
             for o in out if o["greedy_plan"] is not None]
    for mode in modes:
        results = iter(check(items, glados, mode) if items else [])
        for o, p in zip(out, prompts):
            if isinstance(p, str):
                o[mode] = {"greedy": {"verdict": p}, "samples": []}
            else:
                o[mode] = {"greedy": next(results), "samples": [next(results) for _ in range(k)]}
    return out


def mechanics_of(bits):
    return [m for i, m in enumerate(MECHANICS) if (bits >> i) & 1]


def problem_kind(text):
    """A check problem with its numbers taken out, so like failures count together."""
    return re.sub(r"\d+", "N", text or "")


def set_summary(levels, mode, k):
    """Everything report.md says about one set of levels in one check mode."""
    greedy = [lv[mode]["greedy"] for lv in levels]
    s = summarize(greedy)
    n = max(len(levels), 1)
    s["pass_at_k"] = sum(any(r["verdict"] == "solved" for r in lv[mode]["samples"]) for lv in levels) / n
    s["sample_solved"] = (sum(r["verdict"] == "solved" for lv in levels for r in lv[mode]["samples"])
                          / max(k * len(levels), 1))
    s["k"] = k
    # Length against the solver's (BFS) plan length, for the greedy plans that solve.
    extra = [r["len"] - lv["min_len"] for lv, r in zip(levels, greedy)
             if r["verdict"] == "solved" and lv.get("min_len") is not None and "len" in r]
    if extra:
        s["length"] = {
            "n": len(extra),
            "mean_extra_steps": sum(extra) / len(extra),
            "extra_steps": dict(sorted(Counter(min(max(e, -1), 3) for e in extra).items())),
        }
    # Solved share by mechanic and by plan length, where the data says.
    by_mech, by_len = {}, {}
    for lv, r in zip(levels, greedy):
        ok = r["verdict"] == "solved"
        bits = (lv.get("metrics") or {}).get("mechanic_bits")
        for m in mechanics_of(bits) if bits is not None else []:
            by_mech.setdefault(m, [0, 0])
            by_mech[m][0] += 1
            by_mech[m][1] += ok
        if lv.get("min_len") is not None:
            lo, hi = LEN_BUCKETS[len_bucket(lv["min_len"])]
            key = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
            by_len.setdefault(key, [0, 0])
            by_len[key][0] += 1
            by_len[key][1] += ok
    s["by_mechanic"] = {m: {"n": a, "solved": b / a} for m, (a, b) in by_mech.items()}
    s["by_min_len"] = {key: {"n": a, "solved": b / a} for key, (a, b) in by_len.items()}
    # Where the greedy plans that fail go wrong.
    fails = [r for r in greedy if r["verdict"] != "solved"]
    s["failing_steps"] = dict(Counter(r.get("fail_verb", f"({r['verdict']})") for r in fails).most_common())
    s["problems"] = dict(Counter(problem_kind(r.get("problem")) for r in fails if r.get("problem")).most_common(8))
    return s


def pct(x):
    return f"{100 * x:.1f}%"


def report(ck_path, sets, k, seconds):
    lines = [f"# Solver-net evaluation: {ck_path}", ""]
    head = sets.get(HEADLINE)
    if head:
        lines += [f"**Solved (greedy, {HEADLINE}): {pct(head['solved'])}** of {head['n']} levels "
                  f"(the headline number); pass@{k} {pct(head['pass_at_k'])}, optimal {pct(head['optimal'])}.", ""]
    lines += [f"| set | levels | solved (greedy) | pass@{k} | sampled plans solved | optimal | parsed | verdicts |",
              "|---|---|---|---|---|---|---|---|"]
    for name, s in sets.items():
        verdicts = ", ".join(f"{v} {c}" for v, c in sorted(s["verdicts"].items(), key=lambda x: -x[1]))
        lines.append(f"| {name} | {s['n']} | {pct(s['solved'])} | {pct(s['pass_at_k'])} | {pct(s['sample_solved'])} "
                     f"| {pct(s['optimal'])} | {pct(s['parsed'])} | {verdicts} |")
    lines += ["", "`published` levels have no solver plan: optimal doesn't apply, and the conservative check "
              "mostly says no by construction. `published-liberal` is a weak signal; play them to know.", ""]
    for name, s in sets.items():
        lines += [f"## {name}", ""]
        if "length" in s:
            L = s["length"]
            hist = ", ".join(f"{'shorter' if e < 0 else f'+{e}' + ('+' if e == 3 else '')} {c}"
                             for e, c in L["extra_steps"].items())
            lines += [f"Length against the solver's shortest plan (greedy plans that solve, {L['n']}): "
                      f"mean +{L['mean_extra_steps']:.2f} steps; {hist}.", ""]
        if s["by_min_len"]:
            lines += ["| solver plan length | levels | solved |", "|---|---|---|"]
            lines += [f"| {key} | {v['n']} | {pct(v['solved'])} |" for key, v in
                      sorted(s["by_min_len"].items(), key=lambda x: int(x[0].split("-")[0].rstrip("+")))]
            lines.append("")
        if s["by_mechanic"]:
            lines += ["| mechanic | levels | solved |", "|---|---|---|"]
            lines += [f"| {m} | {v['n']} | {pct(v['solved'])} |" for m, v in
                      sorted(s["by_mechanic"].items(), key=lambda x: -x[1]["n"])]
            lines.append("")
        if s["failing_steps"]:
            lines.append("Failing greedy plans, by the step that fails (or why none does): "
                         + ", ".join(f"{v} {c}" for v, c in s["failing_steps"].items()) + ".")
            lines.append("")
        if s["problems"]:
            lines += ["Commonest problems:", ""] + [f"- {c}x {p}" for p, c in s["problems"].items()] + [""]
    lines.append(f"({seconds / 60:.1f} minutes)")
    return "\n".join(lines) + "\n"


def read_jsonl(path, keep=lambda r: True):
    with open(path, encoding="utf-8") as f:
        return [r for r in (json.loads(l) for l in f if l.strip()) if keep(r)]


def pick(records, n, seed):
    if n and len(records) > n:
        records = random.Random(seed).sample(records, n)
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--plans", default="data/plans", help="directory with test-gen.jsonl and test-ws.jsonl")
    ap.add_argument("--published", default="data/ws3/test.jsonl", help="Workshop levels; those with verdict published are used")
    ap.add_argument("--n-gen", type=int, default=1024, help="test-gen levels (0: all)")
    ap.add_argument("--n-ws", type=int, default=0, help="test-ws levels (0: all)")
    ap.add_argument("--n-published", type=int, default=256)
    ap.add_argument("--k", type=int, default=8, help="sampled plans per level, for pass@k")
    ap.add_argument("--temperature", type=float, default=1.0, help="for the sampled plans")
    ap.add_argument("--rows", type=int, default=64, help="plans generated at once")
    ap.add_argument("--out", default=None, help="default: <checkpoint dir>/eval")
    ap.add_argument("--glados", default=str(DEFAULT_GLADOS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    torch.manual_seed(args.seed)
    model, vocab, ck = load_checkpoint(args.checkpoint, device)
    if ck.get("args", {}).get("task") != "solve":
        sys.exit(f"{args.checkpoint} isn't a solver net (train.py --task solve)")
    plans = ROOT / args.plans if not Path(args.plans).is_absolute() else Path(args.plans)
    published = ROOT / args.published if not Path(args.published).is_absolute() else Path(args.published)

    todo = []  # (set name, records, check modes)
    for name, n in ((HEADLINE, args.n_gen), ("test-ws", args.n_ws)):
        path = plans / f"{name}.jsonl"
        if path.exists():
            todo.append((name, pick(read_jsonl(path, lambda r: bool(r.get("plan"))), n, args.seed), ["conservative"]))
        else:
            print(f"{path} is missing: no {name} levels")
    if published.exists() and args.n_published:
        recs = read_jsonl(published, lambda r: r.get("verdict") == "published")
        todo.append(("published", pick(recs, args.n_published, args.seed), ["conservative", "liberal"]))
    if not todo:
        sys.exit("no levels to evaluate on")

    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "eval"
    out.mkdir(parents=True, exist_ok=True)
    sets, solutions = {}, []
    for name, records, modes in todo:
        t = time.time()
        levels = solve_levels(model, vocab, records, args.k, args.temperature, args.glados, modes, args.rows)
        for mode in modes:
            key = name if mode == "conservative" else f"{name}-{mode}"
            sets[key] = set_summary(levels, mode, args.k)
            print(f"{key}: {pct(sets[key]['solved'])} solved (greedy), pass@{args.k} "
                  f"{pct(sets[key]['pass_at_k'])}, {len(levels)} levels, {time.time() - t:.0f}s", flush=True)
        solutions += [{"set": name, **lv} for lv in levels]

    seconds = time.time() - t0
    head = HEADLINE if HEADLINE in sets else next(iter(sets))
    result = {
        "task": "solve",
        "checkpoint": str(args.checkpoint),
        "k": args.k,
        "headline": {"set": head, "solved": sets[head]["solved"], "parsed": sets[head]["parsed"],
                     "pass_at_k": sets[head]["pass_at_k"], "optimal": sets[head]["optimal"]},
        "sets": sets,
    }
    (out / "eval.json").write_text(json.dumps(result, indent=2) + "\n")
    with open(out / "solutions.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(s) + "\n" for s in solutions)
    (out / "report.md").write_text(report(args.checkpoint, sets, args.k, seconds), encoding="utf-8")
    print(f"wrote {out / 'report.md'}")

    run = Path(args.checkpoint).parent
    if (run / "log.jsonl").exists():
        import record

        record.safely(run)


if __name__ == "__main__":
    main()
