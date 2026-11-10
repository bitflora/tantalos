"""Search against sampling for Goblin, at equal wall-clock time (plan.md step 3).

    .venv/Scripts/python train/goblin_search.py runs/goblin-school-r01-2h/best.pt --name first \
        --seconds 60 --n 32

One checkpoint, one set of levels, one time budget a level, and two ways of spending it:

  sample  constrained best-of-N (`goblin_eval.py --constrained` decoding, temperature
          --temperature), `--rows` plans at a time until the level is solved or the budget is out.
  beam    bounded step-level beam search: the beam's `--width` plan prefixes are each continued
          by `--candidates` sampled single steps (one `generate_many` call for all of them), the
          children are scored by their total log probability and the best --width kept. A child
          the env says has reached the goal is checked and ends the level; dead ends and prefixes
          past --max-plan words are dropped.

Both are given a free greedy plan first, so each is at least as good as the run's own evaluation
and the comparison is about what the extra budget buys. Both write only legal steps
(`tantalos-score env`, src/env.rs), so a plan fails only by never reaching the exit, and both
confirm a solve with `tantalos-score check` in conservative mode.

Per level and method it records: whether it solved, the solved plan's length in steps and in
tokens, wall-clock seconds, net forward passes, plan tokens written, `tantalos-score check` plans,
`tantalos-score env` requests, and peak GPU memory. Writes <out>/results.jsonl, summary.json and
report.md (default out: out/search/NAME).

This answers one question only: whether bounded search with the policy alone buys solves that
best-of-N sampling at the same cost does not. It needs the GPU and the solver, so don't start it
while a training run, a school round or a data build is going.
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import goblin_eval  # noqa: E402
from data import BOS, EOS, SEP  # noqa: E402
from record import GOBLIN, task_name  # noqa: E402
from sample import DEFAULT_TANTALOS, load_checkpoint  # noqa: E402
from seeding import resolve  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SETS = ("test-ws", "test-v6", "test-frontier")
MAX_STEP = 24  # most tokens in one step of a beam expansion (the longest plan step is 7)
SEARCH_KEY = "search"  # our own env key; LegalSteps numbers its own


class Bounded(goblin_eval.LegalSteps):
    """LegalSteps that keeps to the clock, and with `one_step` stops each row at its first whole
    step (the base class notes in `whole` where a row's plan last stood at a step boundary, so a
    row that is there now has written its step and may only end).

    Once `budget` is out every row may only end, so a method stops inside a round instead of
    running the clock over: without that, one round of long plans on a big Workshop level can
    take several times the budget, and the methods are no longer compared at equal cost."""

    def __init__(self, env, vocab, prompts, limit=goblin_eval.MAX_CONSTRAINED, budget=None, one_step=False):
        super().__init__(env, vocab, prompts, limit=limit)
        self.budget, self.one_step = budget, one_step

    def __call__(self, idx, seqs):
        mask = super().__call__(idx, seqs)
        if self.budget is not None and self.budget.left() <= 0:
            mask[:] = False
            mask[:, EOS] = True
            return mask
        if self.one_step:
            for r, i in enumerate(idx):
                if seqs[r] and self.whole.get(i) == len(seqs[r]):
                    mask[r] = False
                    mask[r, EOS] = True
        return mask


class Cost:
    """What a method spent on one level."""

    def __init__(self):
        self.forwards = self.tokens = self.checks = self.envs = 0

    def __iadd__(self, other):
        for k in ("forwards", "tokens", "checks", "envs"):
            setattr(self, k, getattr(self, k) + getattr(other, k))
        return self

    def dict(self):
        return {"forwards": self.forwards, "tokens": self.tokens, "checks": self.checks, "envs": self.envs}


class Meter:
    """Counts the net's forward passes and the env's requests until `stop`."""

    def __init__(self, model, env):
        self.cost = Cost()
        self.model, self.env = model, env
        self.forward, self.ask = model.forward, env.ask
        model.forward = self._forward
        env.ask = self._ask

    def _forward(self, *a, **kw):
        self.cost.forwards += 1
        return self.forward(*a, **kw)

    def _ask(self, request):
        self.cost.envs += 1
        return self.ask(request)

    def stop(self):
        self.model.forward, self.env.ask = self.forward, self.ask
        return self.cost


class Budget:
    def __init__(self, seconds):
        self.seconds, self.t0 = seconds, time.time()

    def left(self):
        return self.seconds - (time.time() - self.t0)

    def spent(self):
        return time.time() - self.t0


def prompt_of(vocab, tokens, prefix=""):
    """`<bos> level <sep> prefix` as ids."""
    ids = [BOS] + vocab.encode(tokens.split()) + [SEP]
    return ids + vocab.encode(prefix.split()) if prefix else ids


ARGUMENTS = {"blue", "orange", "+x", "-x", "+y", "-y", "+z", "-z"}


def steps_in(plan):
    """A plan's length in steps: `tantalos-score check` counts them, this is for plans it hasn't
    seen. A step starts at a verb; a verb is a word that isn't an argument."""
    return sum(1 for w in plan.split() if not w.isdigit() and w not in ARGUMENTS)


def solve_check(level, plans, tantalos, cost):
    """The first of `plans` that `tantalos-score check` calls solved, as (plan, result), or None."""
    if not plans:
        return None
    cost.checks += len(plans)
    results = goblin_eval.check([{"tokens": level["tokens"], "min_len": level.get("min_len"), "plans": plans}],
                                tantalos, "conservative")
    for plan, r in zip(plans, results):
        if r.get("verdict") == "solved":
            return plan, r
    return None


def greedy_first(model, vocab, level, env, args, budget, cost):
    """The greedy constrained plan, and its check result if it solves."""
    prompts = [prompt_of(vocab, level["tokens"])]
    constrain = Bounded(env, vocab, prompts, args.max_plan, budget)
    outs = model.generate_many(prompts, args.max_plan, EOS, 1.0, 1, args.rows, constrain)
    plan = " ".join(vocab.decode(outs[0]))
    cost.tokens += len(outs[0])
    return plan, solve_check(level, [plan] if plan else [], args.tantalos, cost)


def sample_level(model, vocab, level, env, args, budget, cost):
    """Constrained best-of-N: rounds of `args.rows` plans until one solves or the budget is out."""
    rounds = plans_tried = 0
    while budget.left() > 0:
        prompts = [prompt_of(vocab, level["tokens"]) for _ in range(args.rows)]
        constrain = Bounded(env, vocab, prompts, args.max_plan, budget)
        outs = model.generate_many(prompts, args.max_plan, EOS, args.temperature, None, args.rows, constrain)
        rounds += 1
        cost.tokens += sum(len(o) for o in outs)
        plans = sorted({" ".join(vocab.decode(o)) for o in outs if o})
        plans_tried += len(plans)
        hit = solve_check(level, plans, args.tantalos, cost)
        if hit:
            return {"solved": True, "plan": hit[0], "result": hit[1], "rounds": rounds, "plans": plans_tried}
    return {"solved": False, "rounds": rounds, "plans": plans_tried}


def beam_level(model, vocab, level, env, args, budget, cost):
    """Bounded step-level beam search with the policy alone (no value head).

    The beam holds (prefix, total log probability). Each round continues every prefix by
    `args.candidates` sampled single steps, in one batched call, and keeps the best `args.width`
    children; a child the env says is at the goal is checked, and one that is dead or past
    `args.max_plan` words is dropped. A beam with nothing left to continue starts again from
    nothing, so the level's whole budget goes on it, as sampling spends its own; the candidate
    steps are sampled, so a fresh beam goes elsewhere. Returns the rounds and children of all
    its passes, and why the last one stopped."""
    if not env.open(SEARCH_KEY, level["tokens"]):
        return {"solved": False, "rounds": 0, "children": 0, "note": "env could not open the level"}
    rounds = seen = passes = depth = 0
    why = Counter()

    def spend():
        """One pass of the beam: the solved plan and its check result, or None."""
        nonlocal rounds, seen, depth
        beam = [("", 0.0)]
        while beam and budget.left() > 0:
            prompts, parents = [], []
            for prefix, score in beam:
                for _ in range(args.candidates):
                    prompts.append(prompt_of(vocab, level["tokens"], prefix))
                    parents.append((prefix, score))
            constrain = Bounded(env, vocab, prompts, args.max_plan, budget, one_step=True)
            outs, logps = model.generate_many(prompts, MAX_STEP, EOS, args.step_temperature, None,
                                              args.rows, constrain, logprob=True)
            rounds += 1
            cost.tokens += sum(len(o) for o in outs)
            late = budget.left() <= 0  # the clock ended these rows, not the level
            children = {}  # plan -> score (the same step sampled twice costs nothing more)
            for (prefix, score), out, lp in zip(parents, outs, logps):
                if not out:
                    why["the clock" if late else "no step"] += 1
                    continue
                plan = f"{prefix} {' '.join(vocab.decode(out))}".strip()
                words = len(plan.split())
                if words > args.max_plan:
                    why["too long"] += 1
                    continue
                depth = max(depth, words)
                children.setdefault(plan, score + lp)
            if not children:
                return None
            seen += len(children)
            plans = list(children)
            answers = env.next([SEARCH_KEY] * len(plans), plans)
            at_goal = [p for p, a in zip(plans, answers) if a.get("end")]
            hit = solve_check(level, at_goal, args.tantalos, cost)
            if hit:
                return hit
            why["dead"] += sum(bool(a.get("dead")) for a in answers)
            alive = [(p, children[p]) for p, a in zip(plans, answers)
                     if not a.get("end") and not a.get("dead") and a.get("allowed")]
            key = (lambda c: c[1] / max(len(c[0].split()), 1)) if args.score == "mean" else (lambda c: c[1])
            beam = sorted(alive, key=key, reverse=True)[: args.width]
        return None

    try:
        while budget.left() > 0:
            hit = spend()
            passes += 1
            if hit:
                return {"solved": True, "plan": hit[0], "result": hit[1], "rounds": rounds,
                        "children": seen, "passes": passes, "depth": depth}
        return {"solved": False, "rounds": rounds, "children": seen, "passes": passes,
                "depth": depth, "note": ", ".join(f"{k} {v}" for k, v in why.most_common())}
    finally:
        env.close([SEARCH_KEY])


def run_method(name, model, vocab, level, env, args):
    """One method on one level, from the free greedy plan on. Returns its record."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    cost = Cost()
    meter = Meter(model, env)
    budget = Budget(args.seconds)
    try:
        plan, hit = greedy_first(model, vocab, level, env, args, budget, cost)
        if hit:
            out = {"solved": True, "by": "greedy", "plan": hit[0], "result": hit[1]}
        else:
            search = sample_level if name == "sample" else beam_level
            out = {"by": name, "greedy_plan": plan, **search(model, vocab, level, env, args, budget, cost)}
    finally:
        cost += meter.stop()
    out["method"] = name
    out["seconds"] = round(budget.spent(), 2)
    out["cost"] = cost.dict()
    out["peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else None
    if out.get("solved"):
        r = out.get("result") or {}
        out["plan_steps"] = r.get("len") if r.get("len") is not None else steps_in(out["plan"])
        out["plan_tokens"] = len(out["plan"].split())
    out.pop("result", None)
    return out


def method_summary(records):
    """What report.md says about one method over one set of levels."""
    n = max(len(records), 1)
    greedy = [r for r in records if r.get("by") == "greedy"]
    extra = [r for r in records if r.get("by") != "greedy"]
    won = [r for r in extra if r.get("solved")]
    steps = [r["plan_steps"] for r in records if r.get("solved") and r.get("plan_steps") is not None]
    cost = Counter()
    for r in records:
        cost.update(r["cost"])
    return {
        "levels": len(records),
        "solved": sum(bool(r.get("solved")) for r in records) / n,
        "greedy_solved": len(greedy) / n,
        "beyond_greedy": len(won),
        "beyond_greedy_of": len(extra),
        "seconds_total": round(sum(r["seconds"] for r in records), 1),
        "seconds_mean": round(sum(r["seconds"] for r in records) / n, 2),
        "seconds_mean_unsolved_by_greedy": round(sum(r["seconds"] for r in extra) / max(len(extra), 1), 2),
        "plan_steps_mean": round(sum(steps) / len(steps), 2) if steps else None,
        "peak_mb": max([r["peak_mb"] for r in records if r.get("peak_mb")] or [0]),
        "cost": dict(cost),
    }


def pct(x):
    return f"{100 * x:.1f}%"


def report(args, sets, seconds):
    lines = [f"# Goblin: search against sampling ({args.checkpoint})", "",
             f"{args.seconds:.0f} s a level a method, {args.width}-wide beam with {args.candidates} candidate "
             f"steps a prefix against best-of-N sampling in rounds of {args.rows}, both constrained, both "
             f"after a free greedy plan. Temperature {args.temperature}, plans up to {args.max_plan} words, "
             f"beam scored by {args.score} log probability, its steps sampled at "
             f"{args.step_temperature}. Seed {args.seed}."
             + (f" Only levels the greedy plan of {args.missed} failed." if args.missed else ""), ""]
    for name, methods in sets.items():
        lines += [f"## {name}", "",
                  "| method | levels | solved | greedy already | solved beyond greedy | mean steps "
                  "| mean s (levels greedy missed) | forward passes | plan tokens | check plans | env asks "
                  "| peak MB |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for method, s in methods.items():
            c = s["cost"]
            lines.append(
                f"| {method} | {s['levels']} | {pct(s['solved'])} | {pct(s['greedy_solved'])} "
                f"| {s['beyond_greedy']} of {s['beyond_greedy_of']} | {s['plan_steps_mean']} "
                f"| {s['seconds_mean_unsolved_by_greedy']} | {c['forwards']} | {c['tokens']} | {c['checks']} "
                f"| {c['envs']} | {s['peak_mb']} |")
        lines.append("")
    lines.append(f"({seconds / 60:.1f} minutes)")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--name", required=True, help="out/search/NAME (unless --out)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--plans", default="data/goblin", help="the test sets' directory")
    ap.add_argument("--sets", default=",".join(SETS), help="comma-separated test sets to use")
    ap.add_argument("--n", type=int, default=32, help="levels from each set (0: all)")
    ap.add_argument("--missed", default=None,
                    help="a goblin_eval solutions.jsonl: draw only levels whose greedy plan failed "
                         "there, so the budget goes on the levels the answer turns on")
    ap.add_argument("--seconds", type=float, default=60.0, help="wall-clock budget a level a method")
    ap.add_argument("--methods", default="sample,beam")
    ap.add_argument("--width", type=int, default=4, help="beam width")
    ap.add_argument("--candidates", type=int, default=8, help="steps sampled per beam prefix")
    ap.add_argument("--score", choices=("total", "mean"), default="total", help="how beam children are ranked")
    ap.add_argument("--rows", type=int, default=32, help="plans or steps generated at once")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--step-temperature", type=float, default=None,
                    help="for the beam's candidate steps (default: --temperature). The policy is "
                         "confident enough that 8 sampled steps are often only 2 different ones, "
                         "so a warmer one widens the search")
    ap.add_argument("--max-plan", type=int, default=goblin_eval.MAX_CONSTRAINED, help="most words in a plan")
    ap.add_argument("--tantalos", default=str(DEFAULT_TANTALOS))
    ap.add_argument("--seed", type=int, default=None, help="default: a fresh one (printed)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    args.seed = resolve(args.seed)
    if args.step_temperature is None:
        args.step_temperature = args.temperature
    print(f"seed {args.seed}", flush=True)

    t0 = time.time()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    torch.manual_seed(args.seed)
    model, vocab, ck = load_checkpoint(args.checkpoint, device)
    if task_name(ck.get("args", {}).get("task")) != GOBLIN:
        sys.exit(f"{args.checkpoint} isn't a Goblin checkpoint (train.py --task goblin)")

    plans = Path(args.plans) if Path(args.plans).is_absolute() else ROOT / args.plans
    missed = None
    if args.missed:
        missed = {r["tokens"] for r in goblin_eval.read_jsonl(args.missed)
                  if r.get("conservative", {}).get("greedy", {}).get("verdict") != "solved"}
        print(f"{len(missed)} levels {args.missed} missed greedily", flush=True)
    todo = []
    for name in args.sets.split(","):
        path = plans / f"{name}.jsonl"
        if not path.exists():
            sys.exit(f"{path} is missing")
        recs = goblin_eval.read_jsonl(path, lambda r: bool(r.get("tokens")))
        if missed is not None:
            recs = [r for r in recs if r["tokens"] in missed]
            if not recs:
                sys.exit(f"no {name} level was missed greedily in {args.missed}")
        if args.n and len(recs) > args.n:
            recs = random.Random(args.seed).sample(recs, args.n)
        todo.append((name, recs))
    methods = args.methods.split(",")
    out = Path(args.out) if args.out else ROOT / "out" / "search" / args.name
    out.mkdir(parents=True, exist_ok=True)

    env = goblin_eval.Env(args.tantalos)
    records, sets = [], {}
    with open(out / "results.jsonl", "w", encoding="utf-8") as f:
        for name, recs in todo:
            for i, level in enumerate(recs):
                for method in methods:
                    r = run_method(method, model, vocab, level, env, args)
                    r = {"set": name, "identity": level.get("identity"), "tokens": level["tokens"],
                         "min_len": level.get("min_len"), **r}
                    records.append(r)
                    f.write(json.dumps(r) + "\n")
                    f.flush()
                done = {m: next(r for r in records[-len(methods):] if r["method"] == m) for m in methods}
                print(f"{name} {i + 1}/{len(recs)}: "
                      + ", ".join(f"{m} {'solved' if d.get('solved') else '-'} ({d['by']}, {d['seconds']:.0f}s)"
                                  for m, d in done.items()), flush=True)
            sets[name] = {m: method_summary([r for r in records if r["set"] == name and r["method"] == m])
                          for m in methods}
    env.stop()
    everything = {m: method_summary([r for r in records if r["method"] == m]) for m in methods}
    sets = {**sets, "all sets": everything} if len(sets) > 1 else sets
    seconds = time.time() - t0
    (out / "summary.json").write_text(json.dumps(
        {"checkpoint": str(args.checkpoint), "seed": args.seed, "seconds": args.seconds,
         "missed": args.missed, "sets_asked": args.sets, "n": args.n,
         "width": args.width, "candidates": args.candidates, "rows": args.rows, "score": args.score,
         "temperature": args.temperature, "step_temperature": args.step_temperature,
         "max_plan": args.max_plan, "sets": sets}, indent=2) + "\n")
    (out / "report.md").write_text(report(args, sets, seconds), encoding="utf-8")
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
