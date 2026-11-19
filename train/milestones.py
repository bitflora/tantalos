"""Milestones: a level's solution as the ordered mechanic steps it takes, without coordinates
(arm 5 of docs/architecture-plan-2026-11-15.md, the solution-first planner and layout decoder).

    plan:       walk 7 4 2 fire blue 12 5 5 +y knock 8 3 2 knock 9 4 2 plate 0 5 2 fire orange 4 2 2 -z
                enter orange walk 12 4 5 pickup 12 4 5 walk 12 5 5 enter blue toss 4 1 1
                dash 0 3 2 4 0 1
    milestones: knock knock plate enter pickup sphere enter toss button > exit dash > exit

A milestone is one plan step (src/plan.rs) that does something, as words the level and plan
vocabularies already have:
- `walk`, `fire` and `exit` are left out (every solution walks, aims and leaves);
- a portal trip is `enter`; the mechanic steps keep their word (`knock`, `plate`, `bounce`,
  `leap`, `climb`, `funnel`, `ride`, `dive`, `loop`, `sweep`);
- a cube step that takes a cube names its kind (`pickup KIND`, `grab KIND`, `fetch KIND`); one
  that lets it go names where it went (`putdown|toss|throw [TARGET]`): the target is `button` when the cube lands on a floor button, `laser`
  when it is put down facing a way (in a beam);
- a step that presses something names it and what it drives: `putdown|toss|throw button`,
  `reach`, `press`, `dash`, `fetch`, `double`, each followed by `> W` for every kind of device
  its button or pedestal is wired to (`> exit`, `> field`, `> bridge`, ...);
- a step the same as the one before it is written once.

Labels come from plans the solver found and `tantalos-score check` replayed (`label`), for the
run's data: every level labelled solvable, generated levels and mutants, each Workshop puzzle once
(its mirrored/turned versions share its milestones: they hold no coordinates), and the Workshop
test levels, for the test loss. Levels without a checked plan get no milestones and arm 5 leaves
them out.

    .venv/Scripts/python train/milestones.py label                 # CPU on every thread, a few hours
    .venv/Scripts/python train/milestones.py label --limit 2000 --out data/milestones-try
    .venv/Scripts/python train/milestones.py show FILE.jsonl       # the milestones of each line's plan
    .venv/Scripts/python train/milestones.py eval runs/arch-plan-12h/best.pt   # arm 5, both ways

`eval` samples a plan model (model.Plan) two ways: the layout decoder alone, prompted with the
milestones (and conditions) of held-out levels, and end to end, writing its own milestones under
the rated-style prompt. For each: the solver's verdicts, and of the solvable levels, how many have
a shortest plan with exactly the declared milestones, how many use every step they declare, and
how many the solver solved around them (a bypass). Every sample counts in the denominator.

The labels: data/milestones/labels.jsonl, one line per level labelled
({key, milestones, min_len, plan}; the key is `key(record)`), and BUILD.md: what went in, how many were
checked, how many distinct puzzles have labels.
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
import wire  # noqa: E402
from data import MILESTONE_WORDS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EXE = ".exe" if sys.platform == "win32" else ""
TANTALOS = ROOT / "target" / "release" / f"tantalos-score{EXE}"
OUT = ROOT / "data" / "milestones"
DATA = "data/v5b/levels-*.jsonl"  # run.py's default data
WS_TEST = ROOT / "data" / "ws5" / "test.jsonl"  # the Workshop test loss's levels (train.py --holdout)

# Plan steps (src/plan.rs) and the plan's other words.
STEPS = {"walk", "fire", "enter", "pickup", "putdown", "toss", "throw", "press", "dash", "knock", "bounce",
         "leap", "plate", "climb", "funnel", "ride", "dive", "exit", "fetch", "reach", "loop", "grab",
         "double", "sweep"}
SILENT = {"walk", "fire", "exit"}
COLORS = {"blue", "orange"}
DIRS = {"+x", "-x", "+y", "-y", "+z", "-z"}
CUBE_KINDS = ("standard", "companion", "reflection", "sphere", "franken")
SAME = {"knock", "bounce", "leap", "plate", "climb", "funnel", "ride", "dive", "loop", "sweep", "enter"}
# Every word a milestone sequence can hold: the steps', cube kinds, targets, `>` and wire targets
# (data.MILESTONE_WORDS, the vocabulary's copy, is checked against it in tests/test_milestones.py).
WORDS = sorted((STEPS - SILENT) | set(CUBE_KINDS) | {"button", "laser", ">"} | set(wire.ORDER))
assert WORDS == MILESTONE_WORDS, "data.MILESTONE_WORDS is out of date"
MAX_WORDS = 64  # longer milestone sequences are cut here (BUILD.md counts them)


def steps(plan):
    """Plan tokens as (step word, cells, colors, dirs)."""
    out = []
    for t in plan:
        if t in STEPS:
            out.append([t, [], [], []])
        elif not out:
            raise ValueError(f"plan starts with {t!r}")
        elif t.isdigit():
            cells = out[-1][1]
            if not cells or len(cells[-1]) == 3:
                cells.append([])
            cells[-1].append(int(t))
        elif t in COLORS:
            out[-1][2].append(t)
        elif t in DIRS:
            out[-1][3].append(t)
        else:
            raise ValueError(f"not a plan word: {t!r}")
    return [(w, [tuple(c) for c in cells], colors, dirs) for w, cells, colors, dirs in out]


class Level:
    """What milestones need of a level: its buttons, pedestals and cubes by cell, and each
    sender's wire targets."""

    def __init__(self, tokens):
        _, devices = wire.records(tokens)
        self.buttons, self.pedestals, self.loose, self.droppers = {}, {}, {}, []
        for word, toks, ws in devices:
            nums = toks[1:4]
            if len(nums) < 3 or not all(v.isdigit() for v in nums):
                continue
            cell = tuple(int(v) for v in nums)
            targets = sorted({t for t, _ in ws}, key=lambda t: wire.RANK.get(t, 99))
            kind = next((k for k in toks[4:] if k in CUBE_KINDS), "standard")
            if word == "button":
                self.buttons[cell] = targets
            elif word == "pedestal":
                self.pedestals[cell] = targets
            elif word == "cube":
                self.loose[cell] = kind
            elif word == "dropper":
                self.droppers.append((cell, kind))

    def cube(self, cell):
        """The kind of the cube at `cell`: a loose cube there, else the dropper above it."""
        if cell in self.loose:
            return self.loose[cell]
        above = [(c[2], k) for c, k in self.droppers if c[:2] == cell[:2] and c[2] >= cell[2]]
        return min(above)[1] if above else "standard"

    @staticmethod
    def drives(targets):
        return [w for t in targets for w in (">", t)]


def milestones(tokens, plan):
    """The milestone words of a level (tokens) solved by a plan (plan tokens)."""
    lv = Level(tokens)
    out, last, held = [], None, "standard"
    for word, cells, _, dirs in steps(plan):
        if word in SILENT:
            continue
        group = [word]
        if word in SAME:
            pass
        elif word in ("pickup", "grab"):
            held = lv.cube(cells[-1]) if cells else "standard"
            group.append(held)
        elif word in ("putdown", "toss", "throw"):
            cell = cells[0] if cells else None
            if cell is not None:
                lv.loose[cell] = held  # where it lies now, for the next pickup there
            if word == "putdown" and dirs:
                group.append("laser")
            elif cell in lv.buttons:
                group += ["button"] + lv.drives(lv.buttons[cell])
        elif word == "reach":
            group += lv.drives(lv.buttons.get(cells[0], [])) if cells else []
        elif word in ("press", "dash"):
            group += lv.drives(lv.pedestals.get(cells[0], [])) if cells else []
        elif word == "fetch":
            held = lv.cube(cells[1]) if len(cells) > 1 else "standard"
            group += [held] + (lv.drives(lv.pedestals.get(cells[0], [])) if cells else [])
        elif word == "double":
            targets = [t for c in cells[:2] for t in lv.pedestals.get(c, [])]
            group += lv.drives(sorted(set(targets), key=lambda t: wire.RANK.get(t, 99)))
        if group != last:
            out += group
        last = group
    return out


def key(record):
    """What a label is looked up by: the Workshop puzzle (all its versions share one), else the
    level's tokens."""
    wid = record.get("workshop_id")
    if wid is not None:
        return f"ws {wid}"
    return hashlib.sha1(record["tokens"].encode()).hexdigest()


def identity(record):
    wid = record.get("workshop_id") or record.get("mutated_from")
    return f"ws {wid}" if wid is not None else record["tokens"]


def load(path=OUT / "labels.jsonl"):
    """{key: milestone words} of a labels file."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["key"]] = r["milestones"].split()
    return out


# --- labelling ----------------------------------------------------------------------------------

def todo(files, limit=None):
    """(name, editor_rules, budget, records) parts to plan: the levels labelled solvable, generated
    levels and mutants each once, then one version of each Workshop puzzle (its `r0` if there is
    one). Workshop puzzles the solver didn't solve when the data was built (`published`) are left
    out: at a budget of 500k they would take most of a day, for few plans."""
    gen, ws, seen = [], {}, set()
    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("verdict") != "solvable":
                    continue
                k = key(r)
                if r.get("workshop_id") is not None:
                    if k not in ws or r.get("variant") == "r0" and ws[k].get("variant") != "r0":
                        ws[k] = r
                elif k not in seen:
                    seen.add(k)
                    gen.append(r)
    ws = list(ws.values())
    if limit:
        gen, ws = gen[:limit], ws[: max(limit // 10, 1)]
    return [("gen", True, 50_000, gen), ("ws", False, 500_000, ws)]


def plan_and_check(records, editor_rules, budget, counts, log, tantalos=TANTALOS, plans=1):
    """Yields (record, solver line) for records the solver plans and `check` replays: one
    `score --plan-only` for them all, the plans checked in chunks. `plans` > 1 also asks for up
    to that many different shortest plans (the line's "plans"; the first is the one checked)."""
    cmd = [str(Path(tantalos).resolve()), "score", "--plan-only", "--budget", str(budget)]
    if plans > 1:
        cmd += ["--plans", str(plans)]
    if not editor_rules:
        cmd += ["--editor-rules", "off"]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")
    sent = deque()

    def feed():
        try:
            for r in records:
                sent.append(r)
                proc.stdin.write((r["tokens"] if r["tokens"].strip() else "<empty>") + "\n")
        except BrokenPipeError:
            pass
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass

    threading.Thread(target=feed, daemon=True).start()
    chunk = []
    start = time.time()
    for line in proc.stdout:
        s = json.loads(line)
        r = sent.popleft()
        counts["levels"] += 1
        if "plan" in s:
            chunk.append((r, s))
        else:
            counts[f"no plan: {s.get('verdict')}"] += 1
        if len(chunk) >= 2000:
            yield from check(chunk, editor_rules, counts, tantalos)
            chunk = []
        if counts["levels"] % 20000 == 0:
            log(f"  {counts['levels']} levels, {counts['checked']} checked, "
                f"{counts['levels'] / max(time.time() - start, 1e-9):.1f}/s")
    yield from check(chunk, editor_rules, counts, tantalos)
    if proc.wait() != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def check(chunk, editor_rules, counts, tantalos):
    """The chunk's plans replayed by `tantalos-score check` (it doesn't apply editor rules: a plan
    is checked on the solver's model of the level)."""
    if not chunk:
        return
    text = "".join(json.dumps({"tokens": s["tokens"], "plan": s["plan"]}) + "\n" for _, s in chunk)
    out = subprocess.run([str(Path(tantalos).resolve()), "check"], input=text, capture_output=True,
                         text=True, encoding="utf-8", check=True)
    for (r, s), line in zip(chunk, out.stdout.splitlines()):
        verdict = json.loads(line)["results"][0]
        verdict = verdict.get("verdict") if isinstance(verdict, dict) else verdict
        if verdict == "solved":
            counts["checked"] += 1
            yield r, s
        else:
            counts[f"check: {verdict}"] += 1


def label(args):
    files = sorted(glob.glob(str(ROOT / args.data))) + [str(WS_TEST)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "labels.jsonl"
    done = set()
    if dest.exists():
        with open(dest, encoding="utf-8") as f:
            done = {json.loads(l)["key"] for l in f if l.strip() and l.endswith("\n")}
    log_lines = []

    def log(m):
        print(m, flush=True)
        log_lines.append(m)

    start = time.time()
    parts = todo(files, args.limit)
    lengths, cut, families, totals = Counter(), 0, set(), Counter()
    with open(dest, "a", encoding="utf-8") as w:
        for name, editor_rules, budget, records in parts:
            records = [r for r in records if key(r) not in done]
            log(f"{name}: {len(records)} levels to plan (budget {budget}, editor rules "
                f"{'on' if editor_rules else 'off'})")
            counts, t0 = Counter(), time.time()
            for r, s in plan_and_check(records, editor_rules, budget, counts, log, args.tantalos):
                words = milestones(s["tokens"].split(), s["plan"].split())
                if len(words) > MAX_WORDS:
                    words, cut = words[:MAX_WORDS], cut + 1
                lengths[min(len(words) // 8 * 8, MAX_WORDS)] += 1
                families.add(identity(r))
                w.write(json.dumps({"key": key(r), "milestones": " ".join(words), "min_len": s["min_len"],
                                    "plan": s["plan"]}) + "\n")
            w.flush()
            counts["seconds"] = round(time.time() - t0)
            log(f"{name}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
            totals.update({f"{name} {k}": v for k, v in counts.items()})
    report = [f"# data/milestones, labelled {time.strftime('%Y-%m-%d %H:%M')}", "",
              "Made by `train/milestones.py label` (its docstring says how).", "",
              f"Inputs: {', '.join(Path(f).name for f in files)}", "",
              "| count | levels |", "|---|---|"]
    report += [f"| {k} | {v:,} |" for k, v in sorted(totals.items())]
    report += ["", f"Distinct puzzles with labels (this pass): {len(families):,}",
               f"Milestone lengths (words, by 8): {dict(sorted(lengths.items()))}; cut at {MAX_WORDS}: {cut}",
               "", f"Took {time.time() - start:.0f}s."]
    with open(out / "BUILD.md", "a", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n\n")
    log(f"wrote {dest}")


def show(args):
    with open(args.file, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            for plan in [r["plan"]] if "plan" in r else []:
                print(" ".join(milestones(r["tokens"].split(), plan.split())))


# --- sampling and evaluation (model.Plan) -------------------------------------------------------

def label_levels(levels, editor_rules=False, budget=50_000, tantalos=TANTALOS, plans=1):
    """For each level (a token string): the milestone words of its checked plan, or None if the
    solver has none for it. With `plans` > 1, a list of each shortest plan's milestones instead."""
    records = [{"tokens": t} for t in levels]
    found = {}
    for r, s in plan_and_check(records, editor_rules, budget, Counter(), lambda m: None, tantalos, plans):
        toks = s["tokens"].split()
        if plans > 1:
            found[id(r)] = [milestones(toks, p.split()) for p in s.get("plans", [s["plan"]])]
        else:
            found[id(r)] = milestones(toks, s["plan"].split())
    return [found.get(id(r)) for r in records]


class Legal:
    """A `generate_many` constraint for a plan model: milestone words (a wire target after `>`),
    at most MAX_WORDS of them, until <sep>; after it, the level tokens `tantalos-score grammar`
    allows (sample.LegalLevels). `layout`: the prompts hold their milestones and <sep> already
    (the layout decoder alone)."""

    def __init__(self, grammar, vocab, layout=False):
        from data import EOS, SEP

        self.grammar, self.vocab, self.layout = grammar, vocab, layout
        self.eos, self.sep = EOS, SEP
        self.words = [vocab.stoi[w] for w in WORDS]
        self.targets = [vocab.stoi[w] for w in wire.ORDER]
        self.gt = vocab.stoi[">"]

    def __call__(self, idx, seqs):
        import torch

        mask = torch.zeros((len(seqs), len(self.vocab)), dtype=torch.bool)
        rows, levels = [], []
        for r, s in enumerate(seqs):
            if self.layout or self.sep in s:
                cut = s.index(self.sep) + 1 if not self.layout else 0
                rows.append(r)
                levels.append(" ".join(self.vocab.decode(s[cut:])))
            elif len(s) >= MAX_WORDS:
                mask[r, self.sep] = True
            elif s and s[-1] == self.gt:
                mask[r, self.targets] = True
            else:
                mask[r, self.words] = True
                mask[r, self.sep] = True
        if levels:
            for r, a in zip(rows, self.grammar.next(levels)):
                ids = [self.vocab.stoi[t] for t in a.get("allowed", []) if t in self.vocab.stoi]
                if a.get("end") or not ids:
                    ids.append(self.eos)
                mask[r, ids] = True
        return mask

    def finish(self, i, seq):
        return seq


def held_out(n, seed, labels, data=DATA, val_share=0.05, split_seed=0):
    """`n` labelled records from the validation split train.py makes (data.load: the seeded hash
    of each puzzle's identity), a Workshop puzzle once (its r0), spread over the input files."""
    import random
    import zlib

    import data as D

    cut = int(val_share * 2**32)
    per_file = []
    for path in sorted(glob.glob(str(ROOT / data))):
        got, seen = [], set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if zlib.crc32(f"{split_seed} {D.identity(r)}".encode()) >= cut:
                    continue
                k = key(r)
                if k in labels and k not in seen and r.get("variant", "r0") == "r0":
                    seen.add(k)
                    got.append((r, labels[k]))
        per_file.append(got)
    rng = random.Random(seed)
    pool = [x for got in per_file for x in rng.sample(got, min(len(got), n))]
    return rng.sample(pool, min(n, len(pool)))


def follow(levels, declared, tantalos=TANTALOS, budget=50_000):
    """Each sampled level against the milestones it was written for: the solver's verdict
    (editor rules on, as EVAL), and for solvable ones whether one of its shortest plans (up to
    8) has exactly those milestones (`exact`), and whether its plans use every step word they
    name (`uses`; one that doesn't solved the level around them: a bypass)."""
    import sample

    results = sample.score(levels, tantalos, budget)
    solvable = [i for i, r in enumerate(results) if r["verdict"] == "solvable"]
    found = label_levels([levels[i] for i in solvable], True, budget, tantalos, plans=8)
    rows = []
    for i, (lv, want, r) in enumerate(zip(levels, declared, results)):
        row = {"tokens": lv, "declared": " ".join(want), "verdict": r["verdict"], "min_len": r.get("min_len")}
        rows.append(row)
    for i, got in zip(solvable, found):
        want = declared[i]
        row = rows[i]
        row["solver"] = [" ".join(g) for g in got or []]
        steps_want = Counter(w for w in want if w in STEPS)
        row["exact"] = any(g == list(want) for g in got or [])
        row["uses"] = any(not steps_want - Counter(w for w in g if w in STEPS) for g in got or [])
    return rows


def tally(rows):
    n = max(len(rows), 1)
    v = Counter(r["verdict"] for r in rows)
    solv = [r for r in rows if r["verdict"] == "solvable"]
    return {"n": len(rows), "verdicts": dict(v), "solvable": len(solv) / n,
            "exact": sum(r.get("exact", False) for r in rows) / n,
            "uses": sum(r.get("uses", False) for r in rows) / n,
            "bypass": sum(not r.get("uses", False) for r in solv) / n,
            "no_plan": sum("solver" in r and not r["solver"] for r in solv) / n,
            "declared_words": sum(len(r["declared"].split()) for r in rows) / n}


def evaluate(args):
    """Arm 5 two ways: the layout decoder alone, given held-out milestones (and their levels'
    conditions), and end to end, writing its own milestones under the rated-style prompt."""
    import torch

    import data as D
    import rated
    import sample
    from seeding import resolve

    seed = resolve(args.seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, _ = sample.load_checkpoint(args.checkpoint, device)
    if model.cfg.arch != "plan":
        sys.exit(f"{args.checkpoint} is a {model.cfg.arch!r} model, not a plan model")
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "eval-milestones"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    held = held_out(args.n, seed, load(args.labels))
    grammar = sample.Grammar(args.tantalos)
    try:
        prompts = [[D.BOS] + vocab.encode(D.slots(D._conditions(r, r["tokens"].split()), D.groups_in(vocab)))
                   + [D.SEP] + vocab.encode(w) + [D.SEP] for r, w in held]
        outs = model.generate_many(prompts, model.cfg.block_size, D.EOS, args.temperature, None, args.batch,
                                   Legal(grammar, vocab, layout=True))
        alone = follow([" ".join(vocab.decode(o)) for o in outs], [w for _, w in held], args.tantalos)
        p = sample.prompt(vocab, rated.prompt(vocab), [])
        outs = model.generate_many([p] * args.n, model.cfg.block_size, D.EOS, args.temperature, None, args.batch,
                                   Legal(grammar, vocab))
        levels, declared = [], []
        for o in outs:
            toks, _ = model.compile_ids(o)
            levels.append(" ".join(toks))
            declared.append(list(model.last_milestones))
        whole = follow(levels, declared, args.tantalos)
    finally:
        grammar.stop()
    report = {"checkpoint": str(args.checkpoint), "seed": seed, "temperature": args.temperature,
              "decoder_alone": tally(alone), "end_to_end": tally(whole),
              "end_to_end_prompt": " ".join(rated.prompt(vocab)), "seconds": round(time.time() - t0)}
    for name, rows in (("alone", alone), ("whole", whole)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for name in ("decoder_alone", "end_to_end"):
        t = report[name]
        print(f"{name}: {t['n']} levels, solvable {t['solvable']:.0%}, exact milestones {t['exact']:.0%}, "
              f"uses every declared step {t['uses']:.0%}, bypass {t['bypass']:.0%}; verdicts {t['verdicts']}; "
              f"{t['declared_words']:.1f} milestone words")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lb = sub.add_parser("label", help="plan, check and label the run's data")
    lb.add_argument("--data", default=DATA)
    lb.add_argument("--out", default=str(OUT))
    lb.add_argument("--limit", type=int, default=None, help="at most this many generated levels (a tenth of it Workshop)")
    lb.add_argument("--tantalos", default=str(TANTALOS))
    sh = sub.add_parser("show", help="milestones of each line's plan (JSONL with tokens and plan)")
    sh.add_argument("file")
    ev = sub.add_parser("eval", help="a plan model: the decoder alone on held-out milestones, and end to end")
    ev.add_argument("checkpoint")
    ev.add_argument("--n", type=int, default=128, help="levels each way")
    ev.add_argument("--temperature", type=float, default=0.7)
    ev.add_argument("--batch", type=int, default=32)
    ev.add_argument("--labels", default=str(OUT / "labels.jsonl"))
    ev.add_argument("--out", default=None, help="default: <checkpoint dir>/eval-milestones")
    ev.add_argument("--seed", type=int, default=None)
    ev.add_argument("--tantalos", default=str(TANTALOS))
    args = ap.parse_args()
    {"label": label, "show": show, "eval": evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
