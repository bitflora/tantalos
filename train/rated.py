"""How much a level net's output looks like the Workshop's best-rated levels (the rated-style track).

The aim (2026-10-25) is varied levels that use the mechanisms of highly rated Workshop puzzles,
whose solutions change the room's state over time (devices powered, lasers redirected, panels
deployed) rather than lean on tricks (turret knocks, dashes, latches, gel). Most such levels
are too big for our solver, so solvable % alone pulls the other way. This track samples with
the Workshop prompt (`<verdict:published> <src:ws> <tier:best>`, or `<tier:top>` for models
without it) and reports, per sample:

- outcome: unparsed; broken (an item inside a solid or against nothing: `invalid` with editor
  rules off); editor (only our model of the editor's rules objects); or the solver's verdict
  with editor rules off: solvable, uncertain (out of budget, the usual verdict for big rated
  levels), unsolvable;
- novelty: not a copy of, nor 4-gram Jaccard >= 0.9 like, a same-size level of data/ws5/train;
- mechanism mix (item families, wires, size) against 5-star Workshop levels, as the mean gap in
  points between the share of levels with each family (the "profile gap", lower is closer);
- for solvable ones: how many state-changing mechanics and tricks the shortest solution uses.

Headlines: **yield**, valid (parsed, not broken, editor-clean), novel and not unsolvable, per
1,000 samples; **strong**, solvable, novel, at least 2 state changes and at most 1 trick, per
1,000. evaluate.py adds this as its "Rated style" section; abtest.py picks play-test levels
with the same checks.

    python train/rated.py CHECKPOINT [--n 512] [--temperature 1.0]   # just this track
"""
import argparse, json, statistics as st, sys, time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from data import MECHANICS  # noqa: E402

WS_TRAIN = ROOT / "data" / "ws5" / "train.jsonl"
# Levels whose Workshop text says they need bunny hopping or a glitch: not "rated" levels.
EXCLUDED = ROOT / "out" / "workshop" / "excluded-tricks-2026-10-25.jsonl"

FAMILIES = {
    "cubes+buttons": ["dropper", "button", "cube"],
    "pedestal": ["pedestal"],
    "fields": ["fizzler", "laserfield"],
    "lasers": ["laser", "catcher", "relay"],
    "bridge": ["bridge"],
    "funnel": ["funnel"],
    "plate": ["plate"],
    "gel": ["gel"],
    "angled panel": ["panel", "glasspanel"],
    "flip": ["flip"],
    "moving": ["piston", "rail", "stairs"],
    "turret": ["turret"],
    "glass/grating": ["glass", "grating"],
    "goo": ["goo"],
}
ITEM_WORDS = {w for ws in FAMILIES.values() for w in ws} - {"goo"} | {"entry", "exit"}
# Solution mechanics (glados metrics, data.MECHANICS) that change the room's state, and tricks.
STATE = {"cube_button", "stand_button", "powered", "redirect", "panel_use", "stairs_use",
         "platform_use", "fizzler_use", "bridge_use", "funnel_use", "plate_use"}
TRICKS = {"turret_knock", "dash", "latch", "toss", "bounce", "leap", "conversion", "gel_portal"}
NEAR = 0.9


def prompt(vocab):
    tier = "<tier:best>" if "<tier:best>" in vocab.stoi else "<tier:top>"
    return ["<verdict:published>", "<src:ws>", tier]


def describe(tokens):
    """Item families, items, wires and floor area of a token line (None if it has no size)."""
    level = tokens.split()
    if len(level) < 4 or level[0] != "size" or not all(t.isdigit() for t in level[1:4]):
        return None
    words = [t for i, t in enumerate(level) if t in ITEM_WORDS and i + 1 < len(level)
             and level[i + 1].isdigit() and (i == 0 or level[i - 1] != ">")]
    has = set(words) | ({"goo"} if "goo" in level else set())
    return {"families": sorted(f for f, ws in FAMILIES.items() if has & set(ws)),
            "items": max(len(words) - 2, 0), "wires": level.count(">"),
            "area": int(level[1]) * int(level[2])}


def mechanics(metrics):
    """(state-changing mechanics, tricks) a solution uses, or (None, None) without metrics."""
    bits = (metrics or {}).get("mechanic_bits")
    if bits is None:
        return None, None
    used = {m for i, m in enumerate(MECHANICS) if (bits >> i) & 1}
    return len(used & STATE), len(used & TRICKS)


def outcome(on, off):
    """One sample's outcome from its `glados-score score` results with editor rules on and off."""
    if on["verdict"] == "unparsed":
        return "unparsed"
    if off["verdict"] in ("invalid", "unsupported"):
        return "broken"
    if on["verdict"] == "invalid":
        return "editor"
    return off["verdict"]


def excluded_ids():
    if not EXCLUDED.exists():
        return set()
    return {json.loads(l)["id"] for l in EXCLUDED.read_text(encoding="utf-8").splitlines() if l.strip()}


def profile(descs):
    """Share of levels (in points) with each family, and mean items, wires, area."""
    descs = [d for d in descs if d]
    n = max(len(descs), 1)
    fam = Counter(f for d in descs for f in d["families"])
    return {"n": len(descs), "families": {f: round(100 * fam[f] / n, 1) for f in FAMILIES},
            "items": st.mean(d["items"] for d in descs) if descs else None,
            "wires": st.mean(d["wires"] for d in descs) if descs else None,
            "area": st.median(d["area"] for d in descs) if descs else None}


def reference():
    """The 5-star Workshop training levels' profile, and their solved ones' mechanics."""
    skip = excluded_ids()
    rows = [json.loads(l) for l in WS_TRAIN.read_text(encoding="utf-8").splitlines() if l.strip()]
    best = [r for r in rows if r.get("stars") == 5 and r["workshop_id"] not in skip]
    ref = profile(describe(r["tokens"]) for r in best)
    solved = [mechanics(r.get("metrics")) for r in rows if r.get("stars", 0) >= 4
              and r["workshop_id"] not in skip and r.get("verdict") == "solvable"]
    solved = [m for m in solved if m[0] is not None]
    ref["solved"] = {"n": len(solved), "state": st.mean(s for s, _ in solved),
                     "strong": sum(s >= 2 and t <= 1 for s, t in solved) / len(solved),
                     "tricks": st.mean(t for _, t in solved)}
    return ref


def gap(p, ref):
    """Mean distance in points between two profiles' family shares (lower is closer)."""
    return st.mean(abs(p["families"][f] - ref["families"][f]) for f in FAMILIES)


def assess(lines, glados, budget=50_000, p2c_dir=None):
    """Score token lines both ways and describe each: one row per line."""
    import sample
    from evaluate import nearest

    on = sample.score(lines, glados, budget)
    off = sample.score(lines, glados, budget, p2c_dir, editor_rules=False, p2c_all=True)
    known = {hash(json.loads(l)["tokens"]) for l in WS_TRAIN.read_text(encoding="utf-8").splitlines() if l.strip()}
    rows = []
    for i, (line, a, b) in enumerate(zip(lines, on, off)):
        s, t = mechanics(b.get("metrics"))
        rows.append({"i": i, "line": b.get("line", i + 1), "sample": line, "outcome": outcome(a, b),
                     "tokens": b.get("tokens"), "min_len": b.get("min_len"), "metrics": b.get("metrics"),
                     "problem": a.get("problem") or b.get("problem"), "state": s, "tricks": t,
                     "desc": describe(line)})
    probes = [r for r in rows if r["outcome"] not in ("unparsed",) and r["tokens"]]
    for r, near in zip(probes, nearest([str(WS_TRAIN)], [(-1, r["tokens"]) for r in probes])):
        r["copy"] = hash(r["tokens"]) in known
        r["nearest_ws"] = near
        r["novel"] = not r["copy"] and (near or 0) < NEAR
    return rows


def valid(r):
    return r["outcome"] in ("solvable", "uncertain", "unsolvable")


def strong(r):
    return r["outcome"] == "solvable" and r.get("novel") and r["state"] is not None \
        and r["state"] >= 2 and r["tricks"] <= 1


def summarize(rows, ref):
    n = max(len(rows), 1)
    out = Counter(r["outcome"] for r in rows)
    ok = [r for r in rows if valid(r)]
    solved = [r for r in ok if r["state"] is not None]
    p = profile(r["desc"] for r in ok)
    return {
        "n": len(rows),
        "outcomes": {k: out[k] / n for k in ("unparsed", "broken", "editor", "solvable", "uncertain", "unsolvable")},
        "valid": len(ok) / n,
        "novel_of_valid": sum(bool(r.get("novel")) for r in ok) / max(len(ok), 1),
        "copies": sum(bool(r.get("copy")) for r in rows),
        "yield_per_1000": 1000 * sum(bool(r.get("novel")) and r["outcome"] != "unsolvable" for r in ok) / n,
        "strong_per_1000": 1000 * sum(strong(r) for r in rows) / n,
        "profile": p,
        "profile_gap": gap(p, ref) if p["n"] else None,
        "solved": {"n": len(solved),
                   "state": st.mean(r["state"] for r in solved) if solved else None,
                   "tricks": st.mean(r["tricks"] for r in solved) if solved else None,
                   "strong": sum(r["state"] >= 2 and r["tricks"] <= 1 for r in solved) / max(len(solved), 1)},
        "reference": ref,
    }


def section(s, prompt_tokens, temperature):
    """The report.md section."""
    pct = lambda x: f"{100 * x:.0f}%" if x is not None else "-"
    num = lambda x, f=".1f": format(x, f) if x is not None else "-"
    o, ref, p = s["outcomes"], s["reference"], s["profile"]
    lines = [
        f"## Rated style ({s['n']} levels asked for `{' '.join(prompt_tokens)}`, temperature {temperature})",
        "",
        "Like the Workshop's 5-star levels? (train/rated.py) **Yield**: valid, novel, not unsolvable, per 1,000 "
        "samples. **Strong**: solvable, novel, 2+ state-changing mechanics and at most 1 trick, per 1,000.",
        "",
        f"- **yield {s['yield_per_1000']:.0f} / 1,000**, strong {s['strong_per_1000']:.0f} / 1,000; "
        f"profile gap {num(s['profile_gap'])} points (lower is closer to 5-star levels)",
        f"- outcomes: {pct(o['unparsed'])} unparsed, {pct(o['broken'])} broken (items in or against nothing), "
        f"{pct(o['editor'])} editor rules only, {pct(o['solvable'])} solvable, {pct(o['uncertain'])} uncertain, "
        f"{pct(o['unsolvable'])} unsolvable",
        f"- novel: {pct(s['novel_of_valid'])} of valid levels; {s['copies']} copies of Workshop levels",
        "",
        "| | valid samples | 5-star Workshop |",
        "|---|---:|---:|",
        f"| levels | {p['n']} | {ref['n']} |",
        f"| items | {num(p['items'])} | {num(ref['items'])} |",
        f"| wires | {num(p['wires'])} | {num(ref['wires'])} |",
        f"| floor area (median) | {num(p['area'], '.0f')} | {num(ref['area'], '.0f')} |",
    ]
    lines += [f"| {f} | {p['families'][f]:.0f}% | {ref['families'][f]:.0f}% |" for f in FAMILIES]
    so, rs = s["solved"], ref["solved"]
    lines += [f"| solved: state changes | {num(so['state'], '.2f')} | {num(rs['state'], '.2f')} (4-5 star) |",
              f"| solved: tricks | {num(so['tricks'], '.2f')} | {num(rs['tricks'], '.2f')} |",
              f"| solved: 2+ state, <= 1 trick | {pct(so['strong']) if so['n'] else '-'} | {pct(rs['strong'])} |",
              ""]
    return lines


def run(model, vocab, n, glados, budget=50_000, temperature=1.0, top_k=None, batch=32, p2c_dir=None):
    """Sample n levels with the rated prompt and assess them: (summary, rows, prompt)."""
    import sample

    toks = prompt(vocab)
    t = time.time()
    print(f"rated: sampling {n} x {' '.join(toks)} at temperature {temperature}", flush=True)
    lines = sample.generate(model, vocab, n, toks, (), temperature, top_k, batch)
    print(f"rated: sampled in {time.time() - t:.0f}s; scoring", flush=True)
    t = time.time()
    rows = assess(lines, glados, budget, p2c_dir)
    print(f"rated: scored and compared in {time.time() - t:.0f}s", flush=True)
    return summarize(rows, reference()), rows, toks


def main():
    import torch
    import sample
    from seeding import resolve

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--budget", type=int, default=50_000)
    ap.add_argument("--glados", default=str(sample.DEFAULT_GLADOS))
    ap.add_argument("--out", default=None, help="directory (default <checkpoint dir>/rated)")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    seed = resolve(a.seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, _ = sample.load_checkpoint(a.checkpoint, device)
    out = Path(a.out) if a.out else Path(a.checkpoint).parent / "rated"
    out.mkdir(parents=True, exist_ok=True)
    s, rows, toks = run(model, vocab, a.n, a.glados, a.budget, a.temperature, a.top_k)
    s.update(prompt=toks, temperature=a.temperature, top_k=a.top_k, seed=seed)
    (out / "rated.json").write_text(json.dumps(s, indent=1))
    with open(out / "samples.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps({k: v for k, v in r.items() if k != "desc"}) + "\n" for r in rows)
    text = "\n".join(section(s, toks, a.temperature))
    (out / "rated.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
