"""Blind A/B play-tests: levels from two models (or a model and real rated Workshop levels),
mixed and unlabelled in the Puzzle Maker; the play-tester's verdicts are unblinded afterwards.

The user plays 5-10 levels per training run. Ten levels can only show big differences (2 of 10
good against 6 of 10 is real, 4 against 6 is noise), so they are spent comparing: half from
the new model, half from its parent, or from rated Workshop levels when there is no parent.
Verdicts pile up in train/playtests.jsonl, so later they can check the automatic scores
(train/rated.py) against what a person thought.

    python train/abtest.py pick NAME A B [--n 4] [--game]   # stage 2n levels as glados-ab-NAME-NN
    python train/abtest.py score NAME                       # after playing: unblind results.csv
    python train/abtest.py list

A and B are checkpoints, or `workshop`: rated (4-5 star) levels of data/ws5/test.jsonl, never
trained on, without the bunny-hop/glitch ones. A model's levels come from the rated-style prompt
(rated.py) at --temperature; each side keeps valid (parsed, nothing in or against a solid,
editor-clean), novel levels the solver doesn't call unsolvable, and not trivial ones (a solvable
level needs 5+ moves and random play rarely solving it). They are ranked: solvable with 2+
state changes and at most 1 trick first, then uncertain (too big for the solver, like most
rated levels), then the other solvable ones; within that, more item families and wires first,
and one level per item set. --game checks the picks in Portal 2's editor first (it launches the
game, and never touches one that is running). --constrained samples the models' levels with
`glados-score grammar` (run.py passes it since 2026-11-01).

Files, in <out> (default out/abtest/NAME; run.py uses runs/RUN/playtest):
- key.json: which side each file is. Don't open it before playing.
- results.csv: one row per staged file; fill `verdict` (good, ok, trivial, boring, unsolvable,
  broken), and optionally `mechanics` (what the solution used) and `notes`.
- picks.json: the staged files and their Portal 2 check, for run.py's status.
"""
import argparse, csv, datetime, json, random, re, sys, time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
import playtest  # noqa: E402
import preview  # noqa: E402
import rated  # noqa: E402

WS_TEST = ROOT / "data" / "ws5" / "test.jsonl"
REGISTRY = ROOT / "train" / "playtests.jsonl"
VERDICTS = ("good", "ok", "trivial", "boring", "unsolvable", "broken")


def signature(p2c):
    """The item set, for one level per item set."""
    return re.sub(r"^\d+x\d+x\d+: ", "", preview.summary(p2c))


def rank(r):
    desc = r["desc"] or {"families": [], "wires": 0}
    tier = 0 if rated.strong(r) else 1 if r["outcome"] == "uncertain" else 2
    return tier, -(len(desc["families"]) + desc["wires"] / 2)


def playable(r):
    if not rated.valid(r) or r["outcome"] == "unsolvable" or not r.get("novel"):
        return False
    if r["outcome"] == "solvable":
        m = r.get("metrics") or {}
        return (r.get("min_len") or 0) >= 5 and m.get("random_success", 1) < 0.1
    return True


def model_side(checkpoint, n, pool, temperature, glados, out, seed, constrained=False):
    import torch
    import sample

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, _ = sample.load_checkpoint(checkpoint, device)
    toks = rated.prompt(vocab)
    how = ", constrained" if constrained else ""
    print(f"{checkpoint}: sampling {pool} x {' '.join(toks)} at temperature {temperature}{how}", flush=True)
    lines = sample.generate(model, vocab, pool, toks, (), temperature, None, 32, constrained=constrained,
                            glados=glados)
    del model
    torch.cuda.empty_cache()
    rows = rated.assess(lines, glados, p2c_dir=out / "p2c")
    good = sorted((r for r in rows if playable(r)), key=rank)
    print(f"  {len(good)} of {pool} playable; outcomes {dict(Counter(r['outcome'] for r in rows))}", flush=True)
    return good, {"source": str(checkpoint), "prompt": toks, "temperature": temperature, "pool": pool,
                  "constrained": constrained, "playable": len(good)}


def workshop_side(n, glados, out, seed):
    skip = rated.excluded_ids()
    rows = [json.loads(l) for l in WS_TEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("stars", 0) >= 4 and r["workshop_id"] not in skip]
    random.Random(seed).shuffle(rows)
    rows = rows[: n * 6]
    assessed = rated.assess([r["tokens"] for r in rows], glados, p2c_dir=out / "p2c")
    good = []
    for r, a in zip(rows, assessed):
        # Published levels count as solvable; only a broken conversion is left out.
        if a["outcome"] in ("unparsed", "broken", "editor"):
            continue
        a.update(workshop_id=r["workshop_id"], title=r.get("title"), stars=r.get("stars"))
        good.append(a)
    good.sort(key=lambda a: -(a.get("stars") or 0))
    return good, {"source": "workshop", "pool": len(rows), "playable": len(good)}


def diverse(rows, n, out):
    picked, seen = [], set()
    for r in rows:
        f = out / "p2c" / f"{r['line']:05d}.p2c"
        if not f.exists():
            continue
        sig = signature(f)
        if sig in seen:
            continue
        seen.add(sig)
        picked.append((r, f))
        if len(picked) == n:
            break
    return picked


def pick(a):
    out = Path(a.out) if a.out else ROOT / "out" / "abtest" / a.name
    if (out / "key.json").exists():
        sys.exit(f"{out} already has a key.json: pick another NAME")
    out.mkdir(parents=True, exist_ok=True)
    seed = a.seed if a.seed is not None else random.SystemRandom().randrange(2**31)
    sides = {}
    for label, src in (("A", a.a), ("B", a.b)):
        d = out / f"side-{label}"
        if src == "workshop":
            rows, info = workshop_side(a.n, a.glados, d, seed)
        else:
            rows, info = model_side(src, a.n, a.pool, a.temperature, a.glados, d, seed, a.constrained)
        chosen = diverse(rows, a.n * 3 if a.game else a.n, d)
        sides[label] = (chosen, info)
    game = "unchecked"
    if a.game:
        import pick as pick_mod

        # Both sides number their files from 00001: check copies named by side.
        cand = out / "game-candidates"
        cand.mkdir(exist_ok=True)
        copies = {}
        for label, (chosen, _) in sides.items():
            for _, f in chosen:
                c = cand / f"{label}-{f.name}"
                c.write_bytes(f.read_bytes())
                copies[c.name] = f
        ok = pick_mod.game_valid(sorted(cand.glob("*.p2c")), out)
        if ok is not None:
            game, names = "valid", {str(copies[c.name]) for c in ok}
            sides = {k: ([(r, f) for r, f in chosen if str(f) in names], info) for k, (chosen, info) in sides.items()}
    n = min(a.n, *(len(chosen) for chosen, _ in sides.values()))
    if n == 0:
        sys.exit("a side has no playable level: nothing to compare")
    if n < a.n:
        print(f"only {n} playable levels on one side: staging {n} per side")
    staged = [(label, r, f) for label, (chosen, _) in sides.items() for r, f in chosen[:n]]
    random.Random(seed).shuffle(staged)
    dest = Path(a.dest) if a.dest else playtest.PUZZLES
    dest.mkdir(parents=True, exist_ok=True)
    now, key, picks = int(time.time()), [], []
    for i, (label, r, f) in enumerate(staged, 1):
        name = f"glados-ab-{a.name}-{i:02d}"
        dst = dest / f"{name}.p2c"
        text = playtest.restamp(f.read_bytes().decode("utf-8"), now + i)
        text = playtest.retitle(text, name)
        text = playtest.DESCRIPTION.sub(lambda m: f'{m[1]}"{preview.summary(f)}"', text, count=1)
        playtest.write(dst, text)
        preview.write(dst, name=name.removeprefix("glados-"))
        desc = r.get("desc") or {}
        key.append({"file": name, "side": label, "source": sides[label][1]["source"], "copy_of": str(f),
                    "outcome": r["outcome"], "min_len": r.get("min_len"), "state": r.get("state"),
                    "tricks": r.get("tricks"), "families": desc.get("families"), "wires": desc.get("wires"),
                    "items": desc.get("items"), "nearest_ws": r.get("nearest_ws"),
                    "workshop_id": r.get("workshop_id"), "stars": r.get("stars"), "tokens": r.get("tokens")})
        picks.append({"file": f"{name}.p2c", "name": name, "game": game})
        print(dst)
    info = {k: info for k, (_, info) in sides.items()}
    (out / "key.json").write_text(json.dumps({"name": a.name, "date": datetime.date.today().isoformat(),
                                              "seed": seed, "sides": info, "levels": key}, indent=1))
    (out / "picks.json").write_text(json.dumps(picks, indent=1))
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "verdict", "mechanics", "notes"])
        w.writerows([[p["name"], "", "", ""] for p in picks])
    print(f"staged {len(picks)} levels ({n} per side, {game} in Portal 2); "
          f"verdicts go in {out / 'results.csv'} ({', '.join(VERDICTS)})")


def score(a):
    out = Path(a.out) if a.out else ROOT / "out" / "abtest" / a.name
    key = json.loads((out / "key.json").read_text())
    side_of = {k["file"]: k for k in key["levels"]}
    with open(out / "results.csv", newline="", encoding="utf-8") as fh:
        results = [r for r in csv.DictReader(fh) if r.get("verdict", "").strip()]
    bad = [r["verdict"] for r in results if r["verdict"].strip().lower() not in VERDICTS]
    if bad:
        sys.exit(f"unknown verdicts {bad}: use {', '.join(VERDICTS)}")
    if not results:
        sys.exit(f"no verdicts in {out / 'results.csv'} yet")
    counts = {s: Counter() for s in ("A", "B")}
    rows = []
    for r in results:
        k = side_of[r["file"]]
        v = r["verdict"].strip().lower()
        counts[k["side"]][v] += 1
        rows.append({"test": key["name"], "date": key["date"], "scored": datetime.date.today().isoformat(),
                     **{x: k.get(x) for x in k if x != "copy_of"}, "verdict": v,
                     "mechanics": r.get("mechanics", ""), "notes": r.get("notes", "")})
    for s in ("A", "B"):
        c = counts[s]
        n = sum(c.values())
        print(f"{s} ({key['sides'][s]['source']}): {c['good'] + c['ok']} of {n} good or ok; "
              + ", ".join(f"{v} {c[v]}" for v in VERDICTS if c[v]))
    registry = Path(a.registry)
    done = set()
    if registry.exists():
        done = {(j["test"], j["file"]) for j in map(json.loads, filter(str.strip, registry.read_text(encoding="utf-8").splitlines()))}
    new = [r for r in rows if (r["test"], r["file"]) not in done]
    with registry.open("a", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in new)
    print(f"{len(new)} verdicts added to {registry} (commit it)")


def list_tests(a):
    for k in sorted((ROOT / "out" / "abtest").glob("*/key.json")) + sorted(ROOT.glob("runs/*/playtest/key.json")):
        key = json.loads(k.read_text())
        res = k.parent / "results.csv"
        filled = 0
        if res.exists():
            with open(res, newline="", encoding="utf-8") as fh:
                filled = sum(1 for r in csv.DictReader(fh) if r.get("verdict", "").strip())
        print(f"{key['name']:24} {key['date']}  {len(key['levels'])} levels, {filled} verdicts  "
              f"A {key['sides']['A']['source']}  B {key['sides']['B']['source']}")


def main():
    import sample

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick", help="stage a blind A/B set")
    p.add_argument("name")
    p.add_argument("a", help="checkpoint, or `workshop`")
    p.add_argument("b", help="checkpoint, or `workshop`")
    p.add_argument("--n", type=int, default=4, help="levels per side")
    p.add_argument("--pool", type=int, default=192, help="levels sampled per model")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--game", action="store_true", help="check the picks in Portal 2's editor first")
    p.add_argument("--constrained", action="store_true",
                   help="models sample only tokens `glados-score grammar` allows (src/grammar.rs)")
    p.add_argument("--glados", default=str(sample.DEFAULT_GLADOS))
    p.add_argument("--out", default=None)
    p.add_argument("--dest", default=None, help="stage here instead of the Puzzle Maker folder (tests)")
    p.add_argument("--seed", type=int, default=None)
    p = sub.add_parser("score", help="unblind the verdicts in results.csv")
    p.add_argument("name")
    p.add_argument("--out", default=None)
    p.add_argument("--registry", default=str(REGISTRY), help="where verdicts pile up (default %(default)s)")
    sub.add_parser("list", help="the A/B sets and how many verdicts each has")
    a = ap.parse_args()
    {"pick": pick, "score": score, "list": list_tests}[a.cmd](a)


if __name__ == "__main__":
    main()
