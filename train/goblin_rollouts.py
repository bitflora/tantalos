"""Rollout labels for a Goblin value head, and a probe of whether they carry signal (plan.md step 3).

    .venv/Scripts/python train/goblin_rollouts.py collect runs/goblin-school-r01-2h/best.pt --name r01 \
        --levels 96 --k 48
    .venv/Scripts/python train/goblin_rollouts.py probe out/rollouts/r01

The search experiment (docs/goblin-school-next-2026-10-29.md, 2026-11-01) found that a beam over
the policy alone can't tell which of its near-identical children is closer to the exit, while
sampling gets a solver verdict for every plan it writes. Those verdicts are value labels: every
step prefix of a sampled plan is labelled with whether that plan went on to solve the level.

collect   draws Goblin *train-side* levels only (school.test_identities() is never drawn), a
          quarter each from the school's solved and unsolved levels (runs/school/e1) and from
          data/goblin/train.jsonl's Workshop and v6 levels, one row per identity. Each level gets
          a greedy plan and --k constrained samples at --temperature (as goblin_search's
          sampling), all checked conservatively. Writes:

            levels.jsonl    a level's stratum, split, greedy verdict and solved samples
            rollouts.jsonl  every plan: (identity, plan, by, solved, steps)
            prefixes.jsonl  every distinct step prefix of the sampled plans: (identity, prefix,
                            steps, n rollouts through it, solved of them), with what
                            `glados-score env` says there (allowed next tokens, dead, end) --
                            the env keeps no state it can hand out, so that is the cheap part
            children.jsonl  the beam's situation, on held-out levels whose samples both solved
                            and failed: at the root and at --parents random prefixes, --candidates
                            single steps sampled as goblin_search's beam does, each distinct
                            child with its log probability (the beam's score) and --m rollouts
                            from it, checked: its empirical value

          A level's split ("probe-train" or "probe-test", a quarter) is a hash of its identity,
          so the probe is measured on levels it never saw.

probe     reads a collect directory: Goblin's final hidden state (after ln_f) at the end of each
          prefix, frozen, and a logistic and a one-hidden-layer MLP head on it, trained on the
          probe-train levels to predict eventual success. Reports AUC on probe-test levels
          (all prefixes, past the root, and within a level, against prefix log probability and
          depth), and on children.jsonl whether the head's top child has a higher empirical
          value than the beam's (top log probability). Writes probe.json and report.md.

Both use the GPU (collect also the solver), so neither runs while a training run, a school
round or a data build is going.
"""

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import goblin_eval  # noqa: E402
import goblin_search  # noqa: E402
import school  # noqa: E402
from data import EOS, SEP  # noqa: E402
from record import GOBLIN, task_name  # noqa: E402
from sample import DEFAULT_GLADOS, load_checkpoint  # noqa: E402
from seeding import resolve  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCHOOL = "runs/school/e1/levels.jsonl"
TRAIN = "data/goblin/train.jsonl"
STRATA = ("school-solved", "school-unsolved", "train-ws", "train-v6")
ENV_KEY = "rollouts"


# --- levels ------------------------------------------------------------------------------------

def split_of(identity):
    """`probe-test` for a quarter of identities, fixed by a hash, else `probe-train`."""
    return "probe-test" if hashlib.sha1(identity.encode()).digest()[0] % 4 == 0 else "probe-train"


def pool(root=ROOT):
    """{stratum: {identity: tokens}}: Goblin train-side levels, never a test or probe identity."""
    held = school.test_identities()
    if not held:
        sys.exit("no held-out identities found: refusing to draw levels without the guard")
    out = {s: {} for s in STRATA}
    for r in school.read_jsonl(root / SCHOOL):
        out["school-solved" if r["solved_rounds"] else "school-unsolved"].setdefault(r["identity"], r["tokens"])
    with open(root / TRAIN, encoding="utf-8") as f:
        for line in f:
            # Cheap filter before parsing: only the Workshop and v6 rows are wanted.
            if '"src": "ws"' not in line and '"src": "v6"' not in line:
                continue
            r = json.loads(line)
            if r.get("src") in ("ws", "v6"):
                ident = r.get("identity") or school.goblin_data.identity(r)
                out[f"train-{r['src']}"].setdefault(ident, r["tokens"])
    for s in out:
        out[s] = {i: t for i, t in out[s].items() if i not in held}
    return out


def draw(levels, n, rng, block_size, vocab):
    """n levels, a quarter from each stratum, askable ones only (goblin_eval.plan_prompt)."""
    picked = []
    for s in STRATA:
        ids = sorted(levels[s])
        rng.shuffle(ids)
        got = 0
        for i in ids:
            if got == n // len(STRATA):
                break
            if isinstance(goblin_eval.plan_prompt(vocab, levels[s][i], block_size), list):
                picked.append({"identity": i, "tokens": levels[s][i], "stratum": s, "split": split_of(i)})
                got += 1
    return picked


# --- collecting --------------------------------------------------------------------------------

def prefixes_of(plan):
    """A plan's step prefixes, from the empty one to all but its last step."""
    steps = school.split_steps(plan)
    return [school.join_steps(steps[:i]) for i in range(len(steps))]


def checked(level, plans, glados):
    """{plan: solved} for these plans (each distinct one checked once; an empty plan fails)."""
    distinct = sorted({p for p in plans if p})
    out = {"": False}
    if distinct:
        results = goblin_eval.check([{"tokens": level["tokens"], "plans": distinct}], glados, "conservative")
        out.update({p: r.get("verdict") == "solved" for p, r in zip(distinct, results)})
    return out


def rollouts(model, vocab, env, level, prefix, n, args, top_k=None):
    """n constrained continuations of `prefix` on the level, as whole plans."""
    prompts = [goblin_search.prompt_of(vocab, level["tokens"], prefix) for _ in range(n)]
    constrain = goblin_eval.LegalSteps(env, vocab, prompts, args.max_plan)
    outs = model.generate_many(prompts, args.max_plan, EOS, args.temperature, top_k, args.rows, constrain)
    return [f"{prefix} {' '.join(vocab.decode(o))}".strip() for o in outs]


def prefix_rows(level, plans, solved):
    """The distinct step prefixes of the sampled plans, each with its rollouts and successes."""
    n, won = Counter(), Counter()
    for p in plans:
        for q in prefixes_of(p):
            n[q] += 1
            won[q] += solved[p]
    return [{"identity": level["identity"], "split": level["split"], "prefix": q,
             "steps": len(school.split_steps(q)), "n": n[q], "solved": won[q]} for q in n]


def env_facts(env, level, rows):
    """What `glados-score env` says at each prefix: how many tokens may come next, dead, end."""
    if not rows or not env.open(ENV_KEY, level["tokens"]):
        return
    try:
        answers = env.next([ENV_KEY] * len(rows), [r["prefix"] for r in rows])
        for r, a in zip(rows, answers):
            r["allowed"], r["dead"], r["end"] = len(a.get("allowed", [])), bool(a.get("dead")), bool(a.get("end"))
    finally:
        env.close([ENV_KEY])


def children(model, vocab, env, level, parent, args):
    """The beam's expansion of `parent`: --candidates sampled single steps, each distinct child
    with its log probability, how often it was drawn, and --m rollouts from it."""
    prompts = [goblin_search.prompt_of(vocab, level["tokens"], parent) for _ in range(args.candidates)]
    constrain = goblin_search.Bounded(env, vocab, prompts, args.max_plan, one_step=True)
    outs, lps = model.generate_many(prompts, goblin_search.MAX_STEP, EOS, args.temperature, None,
                                    args.rows, constrain, logprob=True)
    kids = {}
    for out, lp in zip(outs, lps):
        if out:
            child = f"{parent} {' '.join(vocab.decode(out))}".strip()
            kids.setdefault(child, {"logprob": lp, "drawn": 0})["drawn"] += 1
    rows = []
    for child, k in kids.items():
        plans = rollouts(model, vocab, env, level, child, args.m, args)
        solved = checked(level, plans, args.glados)
        rows.append({"identity": level["identity"], "parent": parent, "child": child, **k,
                     "m": len(plans), "value": sum(solved[p] for p in plans) / max(len(plans), 1)})
    return rows


def parents_of(plans, n, rng):
    """The root and n random interior step prefixes of the sampled plans."""
    inner = sorted({q for p in plans for q in prefixes_of(p)[1:]})
    return [""] + (rng.sample(inner, n) if len(inner) > n else inner)


def collect(a):
    t0 = time.time()
    device = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    model, vocab, ck = load_checkpoint(a.checkpoint, device)
    if task_name(ck.get("args", {}).get("task")) != GOBLIN:
        sys.exit(f"{a.checkpoint} isn't a Goblin checkpoint (train.py --task goblin)")
    levels = draw(pool(), a.levels, rng, model.cfg.block_size, vocab)
    out = Path(a.out) if a.out else ROOT / "out" / "rollouts" / a.name
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(levels)} levels ({Counter(lv['split'] for lv in levels)}) -> {out}", flush=True)

    env = goblin_eval.Env(a.glados)
    ranked = 0
    files = {k: open(out / f"{k}.jsonl", "w", encoding="utf-8")
             for k in ("levels", "rollouts", "prefixes", "children")}
    try:
        for i, lv in enumerate(levels):
            if (time.time() - t0) / 60 > a.minutes:
                print(f"--minutes {a.minutes} spent: stopping after {i} levels", flush=True)
                break
            greedy = rollouts(model, vocab, env, lv, "", 1, a, top_k=1)
            sampled = rollouts(model, vocab, env, lv, "", a.k, a)
            solved = checked(lv, greedy + sampled, a.glados)
            for by, plans in (("greedy", greedy), ("sample", sampled)):
                for p in plans:
                    files["rollouts"].write(json.dumps({"identity": lv["identity"], "by": by, "plan": p,
                                                        "solved": solved[p], "steps": len(school.split_steps(p))}) + "\n")
            rows = prefix_rows(lv, sampled, solved)
            env_facts(env, lv, rows)
            for r in rows:
                files["prefixes"].write(json.dumps(r) + "\n")
            won = sum(solved[p] for p in sampled)
            kids = []
            if lv["split"] == "probe-test" and 0 < won < len(sampled) and ranked < a.rank_levels:
                ranked += 1
                for parent in parents_of(sampled, a.parents, rng):
                    kids += children(model, vocab, env, lv, parent, a)
                for r in kids:
                    files["children"].write(json.dumps(r) + "\n")
            files["levels"].write(json.dumps({**{k: lv[k] for k in ("identity", "stratum", "split", "tokens")},
                                              "greedy": solved[greedy[0]], "k": len(sampled), "solved": won,
                                              "prefixes": len(rows), "children": len(kids)}) + "\n")
            for f in files.values():
                f.flush()
            print(f"{i + 1}/{len(levels)} {lv['stratum']} {lv['split']}: greedy {'+' if solved[greedy[0]] else '-'}, "
                  f"{won}/{len(sampled)} sampled, {len(rows)} prefixes, {len(kids)} children "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    finally:
        for f in files.values():
            f.close()
        env.stop()
    (out / "collect.json").write_text(json.dumps(
        {"checkpoint": str(a.checkpoint), "seed": a.seed, "k": a.k, "m": a.m, "temperature": a.temperature,
         "candidates": a.candidates, "parents": a.parents, "max_plan": a.max_plan,
         "minutes": round((time.time() - t0) / 60, 1)}, indent=2) + "\n")


# --- the probe ---------------------------------------------------------------------------------

@torch.no_grad()
def states(model, ids, at):
    """Goblin's final hidden states (after ln_f) at positions `at` of one sequence, and the log
    probability of each token given the ones before it (the first token's is 0)."""
    device = next(model.parameters()).device
    x = torch.tensor([ids], device=device)
    h = model.tok(x) + model.pos(torch.arange(len(ids), device=device))
    for block in model.blocks:
        h = block(h)
    h = model.ln_f(h)
    lp = F.log_softmax(model.head(h).float(), dim=-1)[0]
    token_lp = torch.zeros(len(ids), device=device)
    token_lp[1:] = lp[:-1].gather(1, x[0, 1:, None]).squeeze(1)
    return h[0, at].float().cpu(), token_lp.cpu()


def featurize(model, vocab, tokens, plans):
    """{prefix: (hidden state, total log probability of the prefix's tokens)} for every step
    prefix of `plans` (and the plans themselves), from one pass per plan that adds a new one."""
    out = {}
    for plan in sorted(plans, key=lambda p: -len(p)):  # long plans first cover the most prefixes
        steps = school.split_steps(plan)
        todo = [i for i in range(len(steps) + 1) if school.join_steps(steps[:i]) not in out]
        if not todo:
            continue
        ids = goblin_search.prompt_of(vocab, tokens, plan)
        sep = ids.index(SEP)
        if len(ids) > model.cfg.block_size:
            continue
        ends = [sep + sum(len(s) for s in steps[:i]) for i in todo]  # the prefix's last token
        h, token_lp = states(model, ids, ends)
        cum = token_lp.cumsum(0)
        for row, i, e in zip(h, todo, ends):
            out[school.join_steps(steps[:i])] = (row, float(cum[e] - cum[sep]))
    return out


def auc(scores, pos, neg):
    """Area under the ROC curve of `scores` for items with `pos` positive and `neg` negative
    instances each (ties count half). None if either class is empty."""
    P, N = sum(pos), sum(neg)
    if not P or not N:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    total, neg_below, i = 0.0, 0, 0
    while i < len(order):
        j = i
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        p = sum(pos[order[k]] for k in range(i, j))
        n = sum(neg[order[k]] for k in range(i, j))
        total += p * (neg_below + n / 2)
        neg_below += n
        i = j
    return total / (P * N)


def within(rows, score, key=lambda r: r["identity"]):
    """AUC of `score` over pairs inside the same group only (a level, or a level and depth):
    each group's AUC weighted by its positive-negative pairs. And how many groups have both."""
    by = defaultdict(list)
    for r in rows:
        by[key(r)].append(r)
    total = pairs = groups = 0
    for rs in by.values():
        pos, neg = [r["solved"] for r in rs], [r["n"] - r["solved"] for r in rs]
        a = auc([score(r) for r in rs], pos, neg)
        if a is not None:
            w = sum(pos) * sum(neg)
            total, pairs, groups = total + a * w, pairs + w, groups + 1
    return (total / pairs if pairs else None), groups


class Head(nn.Module):
    def __init__(self, d, hidden=0):
        super().__init__()
        self.net = nn.Linear(d, 1) if not hidden else nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit(x, n, won, hidden=0, decay=1e-2, epochs=400, val=None, seed=0):
    """A head on features x, by binomial log loss (won of n rollouts per row). With `val` (x, n,
    won), keeps the epoch with the lowest validation loss."""
    torch.manual_seed(seed)
    head = Head(x.shape[1], hidden)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2 if not hidden else 3e-3, weight_decay=decay)

    def loss(m, x, n, won):
        z = m(x)
        return -(won * F.logsigmoid(z) + (n - won) * F.logsigmoid(-z)).sum() / n.sum()

    best, best_state = float("inf"), None
    for _ in range(epochs):
        opt.zero_grad()
        loss(head, x, n, won).backward()
        opt.step()
        if val is not None:
            with torch.no_grad():
                v = loss(head, *val).item()
            if v < best:
                best, best_state = v, {k: t.clone() for k, t in head.state_dict().items()}
    if best_state is not None:
        head.load_state_dict(best_state)
    return head


def rank_check(kids, score):
    """The beam's choice among each parent's distinct children (top log probability) against
    `score`'s, by the children's empirical values, over parents whose children's values differ."""
    by = defaultdict(list)
    for k in kids:
        by[(k["identity"], k["parent"])].append(k)
    out = Counter()
    picked = defaultdict(float)
    pairs = Counter()
    for ks in by.values():
        if len(ks) < 2 or len({k["value"] for k in ks}) < 2:
            continue
        out["parents"] += 1
        beam = max(ks, key=lambda k: k["logprob"])
        head = max(ks, key=score)
        picked["beam"] += beam["value"]
        picked["head"] += head["value"]
        picked["random"] += sum(k["value"] for k in ks) / len(ks)
        picked["best"] += max(k["value"] for k in ks)
        out["head better"] += head["value"] > beam["value"]
        out["beam better"] += beam["value"] > head["value"]
        for i, a in enumerate(ks):
            for b in ks[i + 1:]:
                if a["value"] == b["value"]:
                    continue
                hi, lo = (a, b) if a["value"] > b["value"] else (b, a)
                pairs["pairs"] += 1
                pairs["head right"] += score(hi) > score(lo)
                pairs["beam right"] += hi["logprob"] > lo["logprob"]
    m = max(out["parents"], 1)
    return {**out, **{f"mean value, {k} pick": round(v / m, 3) for k, v in picked.items()},
            **pairs, "parent groups": len(by)}


def probe(a):
    t0 = time.time()
    src = Path(a.dir)
    meta = json.loads((src / "collect.json").read_text())
    device = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    model, vocab, _ = load_checkpoint(a.checkpoint or meta["checkpoint"], device)
    levels = {r["identity"]: r for r in school.read_jsonl(src / "levels.jsonl")}
    rows = [r for r in school.read_jsonl(src / "prefixes.jsonl") if r["identity"] in levels]
    plans = defaultdict(list)
    for r in school.read_jsonl(src / "rollouts.jsonl"):
        if r["by"] == "sample":
            plans[r["identity"]].append(r["plan"])
    kids = school.read_jsonl(src / "children.jsonl") if (src / "children.jsonl").exists() else []

    feats = {}
    for ident, lv in levels.items():
        extra = [k["child"] for k in kids if k["identity"] == ident]
        for q, f in featurize(model, vocab, lv["tokens"], plans[ident] + extra).items():
            feats[(ident, q)] = f
    rows = [r for r in rows if (r["identity"], r["prefix"]) in feats]
    for r in rows:
        r["logprob"] = feats[(r["identity"], r["prefix"])][1]
    kids = [k for k in kids if (k["identity"], k["child"]) in feats]
    print(f"features for {len(feats)} prefixes ({time.time() - t0:.0f}s)", flush=True)

    train = [r for r in rows if r["split"] == "probe-train"]
    test = [r for r in rows if r["split"] == "probe-test"]
    # A fifth of the training levels picks the epoch.
    fit_rows = [r for r in train if hashlib.sha1(r["identity"].encode()).digest()[1] % 5]
    val_rows = [r for r in train if not hashlib.sha1(r["identity"].encode()).digest()[1] % 5]

    def tensors(rs):
        x = torch.stack([feats[(r["identity"], r["prefix"])][0] for r in rs])
        return x, torch.tensor([float(r["n"]) for r in rs]), torch.tensor([float(r["solved"]) for r in rs])

    xf, nf, wf = tensors(fit_rows)
    mu, sd = xf.mean(0), xf.std(0) + 1e-5
    norm = lambda x: (x - mu) / sd  # noqa: E731
    val = tensors(val_rows)
    val = (norm(val[0]), val[1], val[2])
    heads = {"linear": fit(norm(xf), nf, wf, 0, val=val, seed=a.seed),
             "mlp": fit(norm(xf), nf, wf, 128, val=val, seed=a.seed)}

    def scorer(head, key):
        @torch.no_grad()
        def score(r):
            return float(head(norm(feats[(r["identity"], r[key])][0][None]))[0])
        return score

    def aucs(rs, score):
        return auc([score(r) for r in rs], [r["solved"] for r in rs], [r["n"] - r["solved"] for r in rs])

    baselines = {"prefix log probability": lambda r: r["logprob"], "depth (steps)": lambda r: r["steps"],
                 "allowed next tokens": lambda r: r.get("allowed", 0)}
    scores = {**{f"{k} head": scorer(h, "prefix") for k, h in heads.items()}, **baselines}
    deep = [r for r in test if r["steps"] > 0]
    roots = [r for r in test if r["steps"] == 0]
    table = {}
    for name, s in scores.items():
        w, nw = within(test, s)
        wd, nd = within(test, s, lambda r: (r["identity"], r["steps"]))
        table[name] = {"all": aucs(test, s), "past the root": aucs(deep, s), "roots (levels)": aucs(roots, s),
                       "within a level": w, "levels within": nw, "within a level and depth": wd,
                       "groups within": nd, "train all": aucs(train, s)}
    ranking = {name: rank_check(kids, scorer(h, "child")) for name, h in heads.items()}

    def inst(rs):
        return sum(r["n"] for r in rs), sum(r["solved"] for r in rs)

    lv_list = list(levels.values())
    summary = {"dir": str(src), "checkpoint": meta["checkpoint"], "levels": len(levels),
               "levels by split": dict(Counter(lv["split"] for lv in lv_list)),
               "levels with both outcomes": sum(0 < lv["solved"] < lv["k"] for lv in lv_list),
               "greedy solved": sum(lv["greedy"] for lv in lv_list),
               "prefixes": {"probe-train": len(train), "probe-test": len(test)},
               "instances (n, solved)": {"probe-train": inst(train), "probe-test": inst(test)},
               "auc": table, "ranking": ranking, "children": len(kids), "seconds": round(time.time() - t0)}
    (src / "probe.json").write_text(json.dumps(summary, indent=2) + "\n")
    (src / "report.md").write_text(report(summary, meta, lv_list), encoding="utf-8")
    print((src / "report.md").read_text(encoding="utf-8"))


def fmt(x):
    return "-" if x is None else f"{x:.3f}"


def report(s, meta, levels):
    by = defaultdict(list)
    for lv in levels:
        by[lv["stratum"]].append(lv)
    lines = [f"# Goblin rollout labels: does a value head have signal? ({s['checkpoint']})", "",
             f"{s['levels']} train-side levels, greedy plus {meta['k']} constrained samples each at temperature "
             f"{meta['temperature']}, checked conservatively; every step prefix of a sample labelled by "
             f"whether that sample solved. Collection took {meta['minutes']} min, the probe "
             f"{s['seconds']} s. Seed {meta['seed']}.", "",
             "| stratum | levels | greedy solved | sampled solved | levels with both outcomes |",
             "|---|---|---|---|---|"]
    for st in STRATA:
        ls = by.get(st, [])
        if ls:
            k = sum(lv["k"] for lv in ls)
            lines.append(f"| {st} | {len(ls)} | {sum(lv['greedy'] for lv in ls)} | "
                         f"{100 * sum(lv['solved'] for lv in ls) / max(k, 1):.1f}% | "
                         f"{sum(0 < lv['solved'] < lv['k'] for lv in ls)} |")
    lines += ["", f"Prefixes: {s['prefixes']}; rollout instances (n, solved): {s['instances (n, solved)']}.", "",
              "## AUC on held-out (probe-test) levels", "",
              "| predictor | all prefixes | past the root | roots only (levels) | within a level "
              "| within a level and depth | train-side all |", "|---|---|---|---|---|---|---|"]
    for name, t in s["auc"].items():
        lines.append(f"| {name} | {fmt(t['all'])} | {fmt(t['past the root'])} | {fmt(t['roots (levels)'])} "
                     f"| {fmt(t['within a level'])} ({t['levels within']}) "
                     f"| {fmt(t['within a level and depth'])} ({t['groups within']}) | {fmt(t['train all'])} |")
    lines += ["", "## The beam's children", "",
              f"{s['children']} children of {meta['candidates']} sampled steps a parent (the root and "
              f"{meta['parents']} random prefixes of each held-out level with both outcomes), {meta['m']} "
              "rollouts each. Only parents whose children's values differ count.", "",
              "| head | parents | head's pick better | beam's better | mean value: beam / head / random / best "
              "| pairs ordered right: head / beam |", "|---|---|---|---|---|---|"]
    for name, r in s["ranking"].items():
        lines.append(f"| {name} | {r.get('parents', 0)} | {r.get('head better', 0)} | {r.get('beam better', 0)} "
                     f"| {r.get('mean value, beam pick', 0)} / {r.get('mean value, head pick', 0)} / "
                     f"{r.get('mean value, random pick', 0)} / {r.get('mean value, best pick', 0)} "
                     f"| {r.get('head right', 0)} / {r.get('beam right', 0)} of {r.get('pairs', 0)} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("checkpoint")
    c.add_argument("--name", required=True, help="out/rollouts/NAME (unless --out)")
    c.add_argument("--out", default=None)
    c.add_argument("--levels", type=int, default=96, help="levels drawn, a quarter from each stratum")
    c.add_argument("--k", type=int, default=48, help="sampled plans a level")
    c.add_argument("--m", type=int, default=12, help="rollouts from each beam child")
    c.add_argument("--candidates", type=int, default=8, help="steps sampled a parent, as the beam")
    c.add_argument("--parents", type=int, default=2, help="random prefixes a ranked level, besides the root")
    c.add_argument("--rank-levels", type=int, default=24, help="most held-out levels whose children are rolled out")
    c.add_argument("--rows", type=int, default=48, help="plans generated at once")
    c.add_argument("--temperature", type=float, default=1.0)
    c.add_argument("--max-plan", type=int, default=goblin_eval.MAX_CONSTRAINED)
    c.add_argument("--minutes", type=float, default=40.0, help="stop drawing new levels after this long")
    c.add_argument("--glados", default=str(DEFAULT_GLADOS))
    c.add_argument("--seed", type=int, default=None)
    c.add_argument("--cpu", action="store_true")
    p = sub.add_parser("probe")
    p.add_argument("dir")
    p.add_argument("--checkpoint", default=None, help="default: the one collect used")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    if a.cmd == "collect":
        a.seed = resolve(a.seed)
        collect(a)
    else:
        probe(a)


if __name__ == "__main__":
    main()
