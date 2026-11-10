"""Evaluate a checkpoint against the generator that made its training data.

    python train/evaluate.py runs/first/best.pt --data data/levels.jsonl

Five questions:
- Validity: of levels asked to be solvable, how many parse, are canonical, are solvable?
- Conditioning: ask for each condition value in turn (alongside <verdict:solvable>, except for
  the verdict itself). How often does the level have it, against how often it has it unasked?
- Novelty: how many levels are exact copies of dataset levels, and how close is the nearest
  dataset level of the same size (Jaccard on token 4-grams)? Dataset levels measured against
  each other are the baseline, since the generator breeds by mutation and has near neighbours too.
- Model vs generator: solvable rate, solvable levels per second, and the plan length, portal,
  cube, item and score distributions of their solvable levels.
- Rated style: asked for Workshop levels like the best-rated ones, how many come out valid,
  novel and close to 5-star levels in mechanisms (train/rated.py)?

Writes report.md, eval.json, samples.jsonl and rated-samples.jsonl to <checkpoint dir>/eval (or --out).
"""

import argparse
import json
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import rated  # noqa: E402
import sample  # noqa: E402
from data import GROUPS, PROVENANCE, conditions, groups_in  # noqa: E402
from seeding import resolve  # noqa: E402

SOLVABLE = "<verdict:solvable>"
SHINGLE = 4


def realized(record):
    """The condition token a scored level actually has, per group (None: doesn't apply).
    None overall if the level didn't parse."""
    if "tokens" not in record:
        return None
    have = conditions(record)
    # A sampled level never has a Workshop id: provenance isn't something it can show.
    return {g: None if g in PROVENANCE else next((t for t in have if t in tokens), None)
            for g, tokens in GROUPS.items()}


def size_key(tokens):
    return " ".join(tokens.split(maxsplit=4)[:4])


def shingles(tokens):
    t = tokens.split()
    return {hash(" ".join(t[i : i + SHINGLE])) for i in range(max(len(t) - SHINGLE + 1, 1))}


def jaccard(a, b):
    return len(a & b) / len(a | b)


# Metrics compared between model and generator. Older datasets lack the later ones, and a row
# is only shown when both sides have it.
METRIC_KEYS = ("score", "portal_trips", "necessary_classes", "random_success",
               "necessary_items", "mechanics", "chain_depth", "travel")


class Stats:
    """Distributions over solvable levels, fed one scored or dataset record at a time."""

    def __init__(self):
        self.n = 0
        self.verdicts = Counter()
        self.cond = Counter()
        self.values = defaultdict(list)

    def add(self, r):
        self.n += 1
        self.verdicts[r["verdict"]] += 1
        if r["verdict"] != "solvable":
            return
        self.cond.update(conditions(r))
        m = r.get("metrics") or {}
        self.values["plan length"].append(r["min_len"])
        self.values["tokens"].append(len(r["tokens"].split()))
        for key in METRIC_KEYS:
            if key in m:
                self.values[key].append(m[key])

    def summary(self):
        solvable = self.verdicts["solvable"]
        out = {
            "n": self.n,
            "solvable": solvable / max(self.n, 1),
            "verdicts": dict(self.verdicts),
            "conditions": {t: c / max(solvable, 1) for t, c in sorted(self.cond.items())},
        }
        for key, v in self.values.items():
            if v:
                q = statistics.quantiles(v, n=10) if len(v) > 1 else [v[0]] * 9
                out[key] = {"mean": statistics.fmean(v), "median": statistics.median(v), "p90": q[8]}
        return out


def scan_dataset(paths, picks, rng):
    """Pass 1: hashes of every level, generator stats, and `picks` random solvable levels."""
    hashes, stats, chosen, seen = set(), Stats(), [], 0
    line_no = 0
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                hashes.add(hash(r["tokens"]))
                stats.add(r)
                if r["verdict"] == "solvable":
                    seen += 1
                    if len(chosen) < picks:
                        chosen.append((line_no, r["tokens"]))
                    elif (j := rng.randrange(seen)) < picks:
                        chosen[j] = (line_no, r["tokens"])
                line_no += 1
    return hashes, stats, chosen


def nearest(paths, probes):
    """Pass 2: for each (exclude_line, tokens) probe, the highest Jaccard similarity to a
    dataset level of the same size, other than the line it came from (None: no such level)."""
    by_size = defaultdict(list)
    for i, (skip, tokens) in enumerate(probes):
        by_size[size_key(tokens)].append((i, skip, shingles(tokens)))
    best = [None] * len(probes)
    line_no = 0
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                # The size is the first thing in the line; skip parsing the rest when no probe
                # shares it.
                start = line.find('"size ') + 1
                key = " ".join(line[start : start + 40].split(" ", 4)[:4])
                if key in by_size:
                    s = shingles(json.loads(line)["tokens"])
                    for i, skip, p in by_size[key]:
                        if skip != line_no:
                            best[i] = max(best[i] or 0.0, jaccard(p, s))
                line_no += 1
    return best


def time_generator(tantalos, count, seed):
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "levels.jsonl"
        data = Path(tantalos).resolve().with_name("tantalos-data" + Path(tantalos).suffix)
        args = [str(data), "dataset", "--seed", str(seed), "--count", str(count), "--out", str(out)]
        t = time.time()
        subprocess.run(args, check=True, capture_output=True)
        dt = time.time() - t
        with open(out, encoding="utf-8") as f:
            verdicts = Counter(json.loads(line)["verdict"] for line in f)
    return {"levels": count, "seconds": dt, "solvable": verdicts["solvable"]}


def similarity(values):
    alone = sum(v is None for v in values)
    values = [v for v in values if v is not None]
    if not values:
        return {"no_same_size": alone} if alone else {}
    return {
        "n": len(values),
        "no_same_size": alone,
        "mean": statistics.fmean(values),
        "share_0.9+": sum(v >= 0.9 for v in values) / len(values),
        "share_1.0": sum(v >= 1.0 for v in values) / len(values),
    }


def pct(x):
    return f"{x:.0%}"


def report(ev):
    v, gen, mod = ev["validity"], ev["generator"], ev["model"]
    lines = [
        f"# Evaluation of `{ev['checkpoint']}`",
        "",
        f"Step {ev['step']}, val loss {ev['val_loss']:.4f}. Temperature {ev['temperature']}, "
        f"top-k {ev['top_k']}, solver budget {ev['budget']}, seed {ev['seed']}"
        + (", constrained decoding (tantalos-score grammar)." if ev.get("constrained") else ", free decoding."),
        "",
        f"## Validity ({v['n']} levels asked for `{SOLVABLE}`)",
        "",
        "| parsed | canonical | solvable | uncertain | unsolvable | invalid/unsupported |",
        "|---|---|---|---|---|---|",
        "| {} | {} | {} | {} | {} | {} |".format(
            pct(v["parsed"]),
            pct(v["canonical"]),
            pct(v["solvable"]),
            pct(v["verdicts"].get("uncertain", 0) / v["n"]),
            pct(v["verdicts"].get("unsolvable", 0) / v["n"]),
            pct((v["verdicts"].get("invalid", 0) + v["verdicts"].get("unsupported", 0)) / v["n"]),
        ),
        "",
        f"Distinct: {pct(v['distinct'])} of parsed levels. Why the rest failed to parse:",
        "",
    ]
    lines += [f"- {n} × {why}" for why, n in v["parse_failures"]]
    lines += [
        "",
        "## Conditioning",
        "",
        "Each value asked for on its own (with `<verdict:solvable>` except for verdicts). "
        "*Accuracy*: share of the levels the group applies to (parsed; solvable for `len` and "
        "`trips`) that have the value. *Hit*: share of all levels asked. *Unasked*: accuracy "
        "with only `<verdict:solvable>` (no condition at all, for verdicts). *Solvable*: share "
        "of the levels asked that are solvable (the only measure for `src` and `tier`, which a "
        "level can't show).",
        "",
        "| condition | asked | accuracy | hit | unasked | solvable |",
        "|---|---|---|---|---|---|",
    ]
    for c in ev["conditioning"]:
        acc = pct(c["accuracy"]) if c["measurable"] else "–"
        base = pct(c["unasked"]) if c["unasked"] is not None else "–"
        solv = pct(c["solvable"]) if "solvable" in c else "–"
        lines.append(f"| `{c['token']}` | {c['asked']} | {acc} | {pct(c['hit'])} | {base} | {solv} |")
    nov = ev["novelty"]
    m, g = nov["model_nearest"], nov["generator_nearest"]
    lines += [
        "",
        "## Novelty",
        "",
        f"Exact copies of dataset levels: {nov['copies']} of {nov['parsed']} parsed levels "
        f"({pct(nov['copies'] / max(nov['parsed'], 1))}).",
        "",
        "Nearest dataset level of the same size, Jaccard similarity on token 4-grams:",
        "",
        "| | levels | no same-size level | mean | ≥ 0.9 | identical |",
        "|---|---|---|---|---|---|",
    ]
    for name, s in (("model (solvable)", m), ("generator (solvable, vs the rest)", g)):
        if "n" in s:
            lines.append(
                f"| {name} | {s['n']} | {s['no_same_size']} | {s['mean']:.2f} | {pct(s['share_0.9+'])} "
                f"| {pct(s['share_1.0'])} |"
            )
    lines += [
        "",
        "## Model vs generator",
        "",
        "| | model | generator |",
        "|---|---|---|",
        f"| solvable share | {pct(mod['solvable'])} | {pct(gen['solvable'])} |",
    ]
    speed = ev["speed"]
    gs = speed.get("generator")
    lines.append(
        "| solvable levels / s | {:.1f} | {} |".format(
            speed["model_solvable_per_s"], f"{gs['solvable'] / gs['seconds']:.1f}" if gs else "–"
        )
    )
    for key in ("plan length", *METRIC_KEYS, "tokens"):
        if key in mod and key in gen:
            a, b = mod[key], gen[key]
            lines.append(
                f"| {key} (mean / median / p90) | {a['mean']:.2f} / {a['median']:.2f} / {a['p90']:.2f} "
                f"| {b['mean']:.2f} / {b['median']:.2f} / {b['p90']:.2f} |"
            )
    graded = ("len", "trips", "cubes", "size", "diff")
    for group in (*graded, *[g for g in GROUPS if GROUPS[g][-1].endswith(":yes>")]):
        tokens = GROUPS[group] if group in graded else [GROUPS[group][1]]
        for t in tokens:
            lines.append(
                f"| `{t}` | {pct(mod['conditions'].get(t, 0))} | {pct(gen['conditions'].get(t, 0))} |"
            )
    lines += [
        "",
        f"Model: the {v['n']} levels asked for `{SOLVABLE}`. Generator: all {gen['n']} dataset "
        "levels. Speeds were measured on this machine"
        + (f" ({gs['levels']} generator levels)" if gs else "")
        + "; model speed includes scoring.",
    ]
    return "\n".join(lines) + "\n"


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # the report has ≥ and ×; Windows defaults to cp1252
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--data", nargs="+", default=["data/levels.jsonl"])
    ap.add_argument("--n", type=int, default=256, help="levels asked for <verdict:solvable> alone")
    ap.add_argument("--n-cond", type=int, default=32, help="levels per condition value")
    ap.add_argument("--neighbours", type=int, default=256, help="dataset levels for the novelty baseline")
    ap.add_argument("--generator-count", type=int, default=1000, help="levels to time the generator on (0: skip)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--budget", type=int, default=50_000)
    ap.add_argument("--tantalos", default=str(sample.DEFAULT_TANTALOS))
    ap.add_argument("--out", default=None, help="directory (default <checkpoint dir>/eval)")
    ap.add_argument("--p2c", action="store_true", help="also write every solvable level as .p2c")
    ap.add_argument("--seed", type=int, default=None,
                    help="default: a fresh, unpredictable seed (printed; pass it back to repeat)")
    ap.add_argument("--cpu", action="store_true", help="sample on the CPU (e.g. while training)")
    ap.add_argument("--constrained", action="store_true",
                    help="sample only tokens `tantalos-score grammar` allows (src/grammar.rs), here and in "
                    "the rated-style section; EVAL and yield then compare only with constrained runs")
    ap.add_argument("--rated-n", type=int, default=512,
                    help="levels for the rated-style section (train/rated.py; 0: skip)")
    ap.add_argument("--rated-temperature", type=float, default=0.7,
                    help="its sampling temperature, fixed so runs compare (0.7 cut broken levels "
                    "from 58%% to 50%% against 1.0 on v5-cont-10h-d, 2026-10-25)")
    args = ap.parse_args()
    args.seed = resolve(args.seed)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model, vocab, ck = sample.load_checkpoint(args.checkpoint, device)
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "eval"
    out.mkdir(parents=True, exist_ok=True)

    # What to sample: the solvable baseline, nothing at all (the verdicts' base rate), and each
    # condition value in turn.
    settings = [("baseline", [SOLVABLE], args.n), ("unconditioned", [], 2 * args.n_cond)]
    known = groups_in(vocab)
    for group in known:
        for t in GROUPS[group]:
            if t != SOLVABLE and t in vocab.stoi:  # older checkpoints lack later values
                settings.append((t, [t] if group == "verdict" else [SOLVABLE, t], args.n_cond))

    lines, owner = [], []
    t0 = time.time()
    for name, conds, n in settings:
        print(f"sampling {n:>4} x {' '.join(conds) or '(no conditions)'}", flush=True)
        t = time.time()
        got = sample.generate(model, vocab, n, conds, (), args.temperature, args.top_k, args.batch,
                              constrained=args.constrained, tantalos=args.tantalos)
        if name == "baseline":
            baseline_gen_s = time.time() - t
        lines += got
        owner += [name] * n
    print(f"sampled {len(lines)} levels in {time.time() - t0:.0f}s; scoring", flush=True)
    t = time.time()
    results = sample.score(lines, args.tantalos, args.budget, out / "p2c" if args.p2c else None)
    score_s = time.time() - t
    # A parsed level whose tokens can't be re-encoded has no "tokens" (the later code skips it);
    # only a binary that never reports them is stale.
    parsed_rows = [r for r in results if r["verdict"] != "unparsed"]
    if parsed_rows and not any("tokens" in r for r in parsed_rows):
        sys.exit("tantalos-score score didn't report canonical tokens: rebuild it (cargo build --release)")
    with open(out / "samples.jsonl", "w", encoding="utf-8") as f:
        for name, line, r in zip(owner, lines, results):
            # `sample` is the model's raw line; `line` is tantalos-score score's line number.
            f.write(json.dumps({**r, "setting": name, "sample": line}) + "\n")

    by_setting = defaultdict(list)
    for name, r in zip(owner, results):
        by_setting[name].append(r)
    have = {name: [realized(r) for r in rs] for name, rs in by_setting.items()}

    # Validity.
    base = by_setting["baseline"]
    parsed = [r for r in base if "tokens" in r]
    validity = sample.summarize(base)
    validity["distinct"] = len({r["tokens"] for r in parsed}) / max(len(parsed), 1)
    # "token 12: row 3 has more than 8 columns" -> "row N has more than N columns"
    failures = Counter(
        re.sub(r'"[^"]*"', "…", re.sub(r"\d+", "N", r["problem"].split(": ", 1)[-1]))
        for r in base
        if r["verdict"] == "unparsed"
    )
    validity["parse_failures"] = failures.most_common(8)

    # Conditioning.
    def accuracy(setting, group, token):
        hs = [h[group] for h in have[setting] if h is not None and h[group] is not None]
        hits = sum(x == token for x in hs)
        return len(hs), hits, hits / max(len(hs), 1)

    conditioning = []
    for group in known:
        for t in GROUPS[group]:
            if t not in vocab.stoi:
                continue
            setting = "baseline" if t == SOLVABLE else t
            measurable, hits, acc = accuracy(setting, group, t)
            ref = "unconditioned" if group == "verdict" else "baseline"
            m0, _, acc0 = accuracy(ref, group, t)
            conditioning.append(
                {
                    "token": t,
                    "asked": len(by_setting[setting]),
                    "measurable": measurable,
                    "hits": hits,
                    "accuracy": acc,
                    "hit": hits / len(by_setting[setting]),
                    "solvable": sum(r["verdict"] == "solvable" for r in by_setting[setting])
                    / len(by_setting[setting]),
                    "unasked": acc0 if m0 else None,
                }
            )

    # Novelty and the generator's side.
    print("reading the dataset", flush=True)
    hashes, gen_stats, picks = scan_dataset(args.data, args.neighbours, rng)
    all_parsed = [r for r in results if "tokens" in r]
    copies = sum(hash(r["tokens"]) in hashes for r in all_parsed)
    model_solvable = [r["tokens"] for r in base if r["verdict"] == "solvable"]
    print("finding nearest neighbours", flush=True)
    near = nearest(args.data, [(-1, t) for t in model_solvable] + picks)
    novelty = {
        "parsed": len(all_parsed),
        "copies": copies,
        "model_nearest": similarity(near[: len(model_solvable)]),
        "generator_nearest": similarity(near[len(model_solvable) :]),
    }

    model_stats = Stats()
    for r in base:
        model_stats.add(r)
    speed = {
        "model_solvable_per_s": validity["solvable"] * len(base) / (baseline_gen_s + score_s * len(base) / len(results)),
    }
    if args.generator_count:
        print(f"timing the generator on {args.generator_count} levels", flush=True)
        speed["generator"] = time_generator(args.tantalos, args.generator_count, args.seed + 1)

    rated_summary = None
    if args.rated_n:
        rated_summary, rated_rows, rated_prompt = rated.run(
            model, vocab, args.rated_n, args.tantalos, args.budget, args.rated_temperature,
            constrained=args.constrained)
        rated_summary.update(prompt=rated_prompt, temperature=args.rated_temperature)
        with open(out / "rated-samples.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps({k: v for k, v in r.items() if k != "desc"}) + "\n" for r in rated_rows)

    ev = {
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "step": ck.get("step"),
        "val_loss": ck.get("val_loss", float("nan")),
        "temperature": args.temperature,
        "top_k": args.top_k,
        "budget": args.budget,
        "constrained": args.constrained,
        "validity": validity,
        "conditioning": conditioning,
        "novelty": novelty,
        "model": model_stats.summary(),
        "generator": gen_stats.summary(),
        "speed": speed,
    }
    if rated_summary:
        ev["rated"] = rated_summary
    (out / "eval.json").write_text(json.dumps(ev, indent=2))
    text = report(ev)
    if rated_summary:
        text += "\n" + "\n".join(rated.section(rated_summary, rated_summary["prompt"], rated_summary["temperature"]))
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out / 'report.md'}, eval.json and samples.jsonl")
    if not args.out:
        import record

        record.safely(Path(args.checkpoint).parent)


if __name__ == "__main__":
    main()
