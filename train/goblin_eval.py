"""Evaluate a Goblin checkpoint (train.py --task goblin; the solver net, task "solve", before
2026-10-27): how often its plans solve levels.

    .venv/Scripts/python train/goblin_eval.py runs/solve-fresh-3h/best.pt

For each level it writes one greedy plan and --k sampled ones, and `glados-score check` replays
them on our solver's model. The levels, from --plans (data/goblin, train/goblin_data.py; the
older data/plans has only the first two and `published`):
  test-ws    the Workshop test split our solver solves (all of them)
  test-v6    levels the level net wrote, held out by puzzle (--n-sets of them)
  test-gen   generated levels held out (--n-gen)
  test-mut   mutants of Workshop test levels (--n-sets)
  test-hard  held-out levels whose search took over 50k states (--n-sets)
  probe      held-out primers, small levels that show one mechanic each; reported per mechanic
  test-frontier  Workshop test levels our solver runs out of budget on (--n-sets): solving one
             means beating the search. Checked in conservative and liberal mode
             (`test-frontier-liberal`, a weak signal: play them to know)
  published  (data/plans only) Workshop levels our solver can't solve, from data/ws3/test.jsonl

--constrained writes every plan with `glados-score env` (src/env.rs) choosing which tokens may
come next, so each step fits a move of the conservative model, and a plan can't stop (write
`exit`) before the goal: it solves the level or runs out of tokens (--max-plan; 256 is enough
for every held-out solver plan but one). --let-stop lets it stop after any whole step instead,
where it would by itself. The frontier's liberal check sees the same plans. Its report goes to
<checkpoint dir>/eval-constrained/ (or --out). A run's own eval (train/run.py, since 2026-10-29)
is constrained, with --out <run>/eval; only an eval written to <run>/eval is recorded in the
run's row.

Writes <checkpoint dir>/eval/ (or --out): report.md, eval.json and solutions.jsonl (every
level with its plans and verdicts). The headline number is the share of test-ws and test-v6
levels together (the Workshop and the level net's: what Goblin is for) the greedy plan solves;
on data/plans, the test-gen share.
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
from seeding import resolve  # noqa: E402
from record import GOBLIN, task_name  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HEADLINE = "test-ws+v6"  # test-ws and test-v6 together
OLD_HEADLINE = "test-gen"  # data/plans has no test-v6
# data/goblin's sets checked in conservative mode, in report order, and how many of each
# (`--n-sets`, else all) are asked; the frontier is checked in both modes.
SETS = ("test-ws", "test-v6", "test-gen", "test-mut", "test-hard", "probe")
# Verdicts for levels the net was never asked about: tokens its vocabulary lacks, or a level
# that leaves no room in the block for a plan.
SKIPPED = ("unknown_tokens", "too_long")
# Most plan tokens written per level: 80-odd steps, far beyond the solver's plans. (An untrained
# net that never writes <eos> would otherwise fill the block for every level.)
MAX_PLAN = 512
# The same for a constrained plan, which can't stop before the goal: only 1 of ~6,700 held-out
# solver plans is longer, and a row that wanders runs to the limit.
MAX_CONSTRAINED = 256


def plan_prompt(vocab, tokens, block_size):
    """`<bos> level <sep>` as ids, or the reason the level can't be asked (a SKIPPED verdict)."""
    words = tokens.split()
    if any(w not in vocab.stoi for w in words):
        return "unknown_tokens"
    if len(words) + 3 >= block_size:
        return "too_long"
    return [BOS] + vocab.encode(words) + [SEP]


def generate_plans(model, vocab, prompts, temperature=1.0, top_k=None, rows=64, env=None):
    """One plan (a token string) per prompt. Prompts are sorted by length and generated `rows`
    at a time (GPT.generate_many: a row that ends starts the next prompt; 64 rows fit the 4 GB
    card, 128 don't). top_k=1 is greedy. With `env` (an Env), only legal steps are written."""
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    asked = [prompts[i] for i in order]
    constrain = LegalSteps(env, vocab, asked) if env else None
    outs = model.generate_many(asked, MAX_CONSTRAINED if env else MAX_PLAN, EOS, temperature, top_k, rows, constrain)
    plans = [None] * len(prompts)
    for i, o in zip(order, outs):
        plans[i] = " ".join(vocab.decode(o))
    return plans


class Env:
    """`glados-score env` (src/env.rs), running: the plan tokens that may come next on a level."""

    def __init__(self, glados=DEFAULT_GLADOS, mode="conservative", let_stop=False):
        self.mode = mode
        self.let_stop = let_stop  # let plans stop (`exit`) after any whole step, not only at the goal
        self.proc = subprocess.Popen([str(Path(glados).resolve()), "env"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)

    def ask(self, request):
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"glados-score env stopped (exit code {self.proc.poll()})")
        return json.loads(line)

    def open(self, key, tokens):
        """Whether the level could be opened (not an invalid or unsupported one)."""
        return bool(self.ask({"op": "open", "id": key, "tokens": tokens, "mode": self.mode}).get("ok"))

    def next(self, keys, plans):
        """Per (level key, plan so far): {allowed: [tokens], end, dead, whole}."""
        return self.ask({"op": "next", "ids": keys, "plans": plans})["results"]

    def close(self, keys):
        self.ask({"op": "close", "ids": [k for k in keys if k is not None]})

    def stop(self):
        self.proc.stdin.close()
        self.proc.wait()


class LegalSteps:
    """A `GPT.generate_many` constraint for these prompts (`<bos> level <sep>`, maybe followed by
    the start of a plan, made of whole steps, to continue): each row may write only the tokens
    `env` allows next on its level: `exit` only at the goal (with env.let_stop, also after any
    whole step, where the model would stop by itself: an unfinished plan). A level is opened
    when its first prompt starts and closed when its last one ends; one env can't open is left
    unconstrained. A plan cut off by the length limit mid-step goes back to its last whole step."""

    def __init__(self, env, vocab, prompts):
        self.env, self.vocab = env, vocab
        self.exit_id = vocab.stoi["exit"]
        self.levels, self.starts = [], []
        self.left = Counter()  # prompts of each level not yet ended
        for p in prompts:
            sep = p.index(SEP)
            self.levels.append(" ".join(vocab.decode(p[1:sep])))
            self.starts.append(vocab.decode(p[sep + 1 :]))
            self.left[self.levels[-1]] += 1
        self.keys = {}  # level -> its env key (None: it couldn't be opened)
        self.whole = {}  # prompt -> its length at its last whole step
        self.opened = 0

    def key(self, i):
        level = self.levels[i]
        if level not in self.keys:
            key = str(self.opened)
            self.opened += 1
            self.keys[level] = key if self.env.open(key, level) else None
        return self.keys[level]

    def plan(self, i, seq):
        return " ".join(self.starts[i] + self.vocab.decode(seq))

    def __call__(self, idx, seqs):
        mask = torch.ones((len(idx), len(self.vocab)), dtype=torch.bool)
        ask = [r for r, i in enumerate(idx) if self.key(i) is not None]
        if ask:
            answers = self.env.next([self.key(idx[r]) for r in ask], [self.plan(idx[r], seqs[r]) for r in ask])
            for r, a in zip(ask, answers):
                ids = [self.vocab.stoi[t] for t in a.get("allowed", []) if t in self.vocab.stoi]
                if a.get("whole"):
                    self.whole[idx[r]] = len(seqs[r])
                    if self.env.let_stop:
                        ids.append(self.exit_id)
                if a.get("end") or not ids:
                    ids.append(EOS)
                mask[r] = False
                mask[r, ids] = True
        return mask

    def finish(self, i, seq):
        key = self.key(i)
        # A plan that ended where env let it ends at a whole step; only one cut off by the length
        # limit may be mid-step.
        if key is not None and len(seq) > self.whole.get(i, 0):
            a = self.env.next([key], [self.plan(i, seq)])[0]
            if not a.get("whole") and not a.get("end"):
                seq = seq[: self.whole.get(i, 0)]
        self.whole.pop(i, None)
        level = self.levels[i]
        self.left[level] -= 1
        if not self.left[level]:
            if key is not None:
                self.env.close([key])
            del self.keys[level]
        return seq


def check(items, glados=DEFAULT_GLADOS, mode="conservative"):
    """One `glados-score check` verdict per plan, in order. An item is `{tokens, plans: [...]}` (or
    `{tokens, plan}` for one), optionally with `min_len`. An item's plans share one input line,
    so `glados-score check` builds the level's model once for all of them. Returns the per-plan result
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
        raise ValueError(f"glados-score check answered {len(results)} of {want} plans")
    return results


def summarize(results):
    """Shares of verdicts over `results` (glados-score check output)."""
    n = max(len(results), 1)
    verdicts = Counter(r["verdict"] for r in results)
    return {
        "n": len(results),
        "parsed": 1 - (verdicts["unparsed"] + sum(verdicts[v] for v in SKIPPED)) / n,
        "solved": verdicts["solved"] / n,
        "optimal": sum(bool(r.get("optimal")) for r in results) / n,
        "verdicts": dict(verdicts),
    }


def solve_levels(model, vocab, records, k, temperature, glados, modes, rows, env=None):
    """Greedy and `k` sampled plans for each record, checked in each mode. Returns one dict
    per record: the record's fields, and per mode `greedy` (verdict) and `samples` (verdicts).
    With `env` (an Env), plans are written from legal steps only."""
    block = model.cfg.block_size
    prompts = [plan_prompt(vocab, r["tokens"], block) for r in records]
    asked = [i for i, p in enumerate(prompts) if not isinstance(p, str)]
    greedy = dict(zip(asked, generate_plans(model, vocab, [prompts[i] for i in asked], 1.0, 1, rows, env)))
    sampled = generate_plans(model, vocab, [prompts[i] for i in asked for _ in range(k)], temperature, None, rows,
                             env)
    samples = {i: sampled[j * k : (j + 1) * k] for j, i in enumerate(asked)}
    out = [{"tokens": r["tokens"], "min_len": r.get("min_len"), "metrics": r.get("metrics"),
            "workshop_id": r.get("workshop_id"), "primer": r.get("primer"), "src": r.get("src"),
            "greedy_plan": greedy.get(i), "sample_plans": samples.get(i, [])}
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
    # Primers: the mechanic each shows (goblin_data.py's probe set).
    by_primer = {}
    for lv, r in zip(levels, greedy):
        if lv.get("primer"):
            by_primer.setdefault(lv["primer"], [0, 0])
            by_primer[lv["primer"]][0] += 1
            by_primer[lv["primer"]][1] += r["verdict"] == "solved"
    if by_primer:
        s["by_primer"] = {m: {"n": a, "solved": b / a} for m, (a, b) in sorted(by_primer.items())}
    s["by_min_len"] = {key: {"n": a, "solved": b / a} for key, (a, b) in by_len.items()}
    # Where the greedy plans that fail go wrong.
    fails = [r for r in greedy if r["verdict"] != "solved"]
    s["failing_steps"] = dict(Counter(r.get("fail_verb", f"({r['verdict']})") for r in fails).most_common())
    s["problems"] = dict(Counter(problem_kind(r.get("problem")) for r in fails if r.get("problem")).most_common(8))
    return s


def pct(x):
    return f"{100 * x:.1f}%"


def report(ck_path, sets, k, seconds, constrained=False):
    lines = [f"# Goblin evaluation: {ck_path}", ""]
    if constrained:
        lines += [f"Constrained decoding ({constrained}): every plan was written from the steps "
                  "`glados-score env` allows (conservative model), so none is illegal there.", ""]
    name = HEADLINE if HEADLINE in sets else OLD_HEADLINE
    head = sets.get(name)
    if head:
        lines += [f"**Solved (greedy, {name}): {pct(head['solved'])}** of {head['n']} levels "
                  f"(the headline number); pass@{k} {pct(head['pass_at_k'])}, optimal {pct(head['optimal'])}.", ""]
    lines += [f"| set | levels | solved (greedy) | pass@{k} | sampled plans solved | optimal | parsed | verdicts |",
              "|---|---|---|---|---|---|---|---|"]
    for name, s in sets.items():
        verdicts = ", ".join(f"{v} {c}" for v, c in sorted(s["verdicts"].items(), key=lambda x: -x[1]))
        lines.append(f"| {name} | {s['n']} | {pct(s['solved'])} | {pct(s['pass_at_k'])} | {pct(s['sample_solved'])} "
                     f"| {pct(s['optimal'])} | {pct(s['parsed'])} | {verdicts} |")
    lines += ["", "`test-frontier` and `published` levels have no solver plan: optimal doesn't apply, and a "
              "conservative solve beats the solver's own search. Their `-liberal` rows are a weak signal; "
              "play them to know.", ""]
    for name, s in sets.items():
        lines += [f"## {name}", ""]
        if s.get("by_primer"):
            lines += ["| primer's mechanic | levels | solved |", "|---|---|---|"]
            lines += [f"| {m} | {v['n']} | {pct(v['solved'])} |" for m, v in s["by_primer"].items()]
            lines.append("")
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
    global MAX_PLAN, MAX_CONSTRAINED
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--plans", default=None,
                    help="directory with test-gen.jsonl and test-ws.jsonl (default: data/goblin, else data/plans)")
    ap.add_argument("--published", default="data/ws3/test.jsonl", help="Workshop levels; those with verdict published are used")
    ap.add_argument("--n-gen", type=int, default=1024, help="test-gen levels (0: all)")
    ap.add_argument("--n-ws", type=int, default=0, help="test-ws levels (0: all)")
    ap.add_argument("--n-published", type=int, default=256, help="published levels (data/plans) or test-frontier ones")
    ap.add_argument("--n-sets", type=int, default=1024, help="levels of each other set (0: all)")
    ap.add_argument("--k", type=int, default=8, help="sampled plans per level, for pass@k")
    ap.add_argument("--temperature", type=float, default=1.0, help="for the sampled plans")
    ap.add_argument("--rows", type=int, default=64, help="plans generated at once")
    ap.add_argument("--out", default=None, help="default: <checkpoint dir>/eval")
    ap.add_argument("--glados", default=str(DEFAULT_GLADOS))
    ap.add_argument("--seed", type=int, default=None,
                    help="default: a fresh, unpredictable seed (printed; pass it back to repeat)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--constrained", action="store_true",
                    help="write only legal steps (glados-score env); default --out: <checkpoint dir>/eval-constrained")
    ap.add_argument("--let-stop", action="store_true", help="with --constrained: plans may stop before the goal")
    ap.add_argument("--max-plan", type=int, default=None,
                    help=f"most tokens per plan (default {MAX_PLAN}, constrained {MAX_CONSTRAINED})")
    args = ap.parse_args()
    args.seed = resolve(args.seed)

    t0 = time.time()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    torch.manual_seed(args.seed)
    model, vocab, ck = load_checkpoint(args.checkpoint, device)
    if task_name(ck.get("args", {}).get("task")) != GOBLIN:
        sys.exit(f"{args.checkpoint} isn't a Goblin checkpoint (train.py --task goblin)")
    if args.plans is None:
        args.plans = "data/goblin" if (ROOT / "data" / "goblin" / "test-gen.jsonl").exists() else "data/plans"
    plans = ROOT / args.plans if not Path(args.plans).is_absolute() else Path(args.plans)
    published = ROOT / args.published if not Path(args.published).is_absolute() else Path(args.published)

    todo = []  # (set name, records, check modes)
    sizes = {"test-gen": args.n_gen, "test-ws": args.n_ws, "probe": 0}
    for name in SETS:
        path = plans / f"{name}.jsonl"
        recs = read_jsonl(path, lambda r: bool(r.get("plan"))) if path.exists() else []
        if recs:
            todo.append((name, pick(recs, sizes.get(name, args.n_sets), args.seed), ["conservative"]))
        elif name in ("test-gen", "test-ws"):
            print(f"{path} is missing: no {name} levels")
    frontier = plans / "test-frontier.jsonl"
    if frontier.exists() and args.n_published:
        todo.append(("test-frontier", pick(read_jsonl(frontier), args.n_published, args.seed), ["conservative", "liberal"]))
    elif published.exists() and args.n_published:
        recs = read_jsonl(published, lambda r: r.get("verdict") == "published")
        todo.append(("published", pick(recs, args.n_published, args.seed), ["conservative", "liberal"]))
    if not todo:
        sys.exit("no levels to evaluate on")

    out = Path(args.out) if args.out else Path(args.checkpoint).parent / ("eval-constrained" if args.constrained else "eval")
    env = Env(args.glados, let_stop=args.let_stop) if args.constrained else None
    if args.max_plan:
        MAX_PLAN = MAX_CONSTRAINED = args.max_plan
    args.max_plan = MAX_CONSTRAINED if args.constrained else MAX_PLAN
    out.mkdir(parents=True, exist_ok=True)
    sets, solutions = {}, []
    for name, records, modes in todo:
        t = time.time()
        levels = solve_levels(model, vocab, records, args.k, args.temperature, args.glados, modes, args.rows, env)
        for mode in modes:
            key = name if mode == "conservative" else f"{name}-{mode}"
            sets[key] = set_summary(levels, mode, args.k)
            print(f"{key}: {pct(sets[key]['solved'])} solved (greedy), pass@{args.k} "
                  f"{pct(sets[key]['pass_at_k'])}, {len(levels)} levels, {time.time() - t:.0f}s", flush=True)
        solutions += [{"set": name, **lv} for lv in levels]
        if name in ("test-ws", "test-v6"):
            both = [lv for lv in solutions if lv["set"] in ("test-ws", "test-v6")]
            if {lv["set"] for lv in both} == {"test-ws", "test-v6"}:
                sets = {HEADLINE: set_summary(both, "conservative", args.k), **sets}

    if env:
        env.stop()
    seconds = time.time() - t0
    head = next((h for h in (HEADLINE, OLD_HEADLINE) if h in sets), next(iter(sets)))
    result = {
        "task": GOBLIN,
        "checkpoint": str(args.checkpoint),
        "k": args.k,
        "constrained": args.constrained,
        "let_stop": args.let_stop,
        "max_plan": args.max_plan,
        "headline": {"set": head, "solved": sets[head]["solved"], "parsed": sets[head]["parsed"],
                     "pass_at_k": sets[head]["pass_at_k"], "optimal": sets[head]["optimal"]},
        "sets": sets,
    }
    (out / "eval.json").write_text(json.dumps(result, indent=2) + "\n")
    with open(out / "solutions.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(s) + "\n" for s in solutions)
    how = args.constrained and (f"may stop after any step, at most {args.max_plan} tokens" if args.let_stop
                                else f"no stopping before the goal, at most {args.max_plan} tokens")
    (out / "report.md").write_text(report(args.checkpoint, sets, args.k, seconds, how), encoding="utf-8")
    print(f"wrote {out / 'report.md'}")

    run = Path(args.checkpoint).parent
    if (run / "log.jsonl").exists() and out.resolve() == (run / "eval").resolve():
        import record

        record.safely(run)


if __name__ == "__main__":
    main()
