"""Stage Workshop levels our solver calls unsolvable in the Puzzle Maker folder, for play-testing.

A published puzzle was solved by its author, so "unsolvable" is a solver gap. The user plays
these and reports how they solved them (see the memory/CLAUDE.md notes). One command does it all:

    python train/stage_unsolv.py --n 5            # pick, re-score, write out/unsolv-DATE/, stage
    python train/stage_unsolv.py --n 5 --dry-run  # same picks, nothing written or copied
    python train/stage_unsolv.py --order rating   # best rated first instead of smallest

Steps: take levels of data/ws5/train.jsonl labelled unsolvable, drop those already staged
(train/unsolv-staged.jsonl, plus any staged Title), whose compiled map paints gel with
`@pre_paint_sprayer` entities (the converter doesn't model it, so the copy wouldn't play like
the original) or that have no map in out/workshop/bsp. Smallest first (or with `--order rating`
most Workshop stars first, then most ratings: `stars`/`votes` from `fetch.py ratings`), one per
item set. Re-score
each pick with the current release `glados.exe` (50k and 250k budgets, editor rules off) and
`glados editor`; keep only those still unsolvable and without editor flags. Write p2c/, tokens,
scores, MANIFEST.md in out/unsolv-DATE[-N]/, copy to the Puzzle Maker folder as
glados-ws5b-unsolv-NNNNN (the series continues the highest number there; Title `unsolv <title>`,
items in Description, a .jpg preview, timestamps a second apart), and append to the registry.

It never starts or touches Portal 2. The real-editor check (`train/portal2.py check`) needs the
game closed; run it afterwards if the user asks (see CLAUDE.md "Staging unsolvable levels").
Needs target/release/glados.exe (don't rebuild it during a training run).
"""
import argparse, datetime, json, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "tools" / "workshop"))
import playtest  # noqa: E402
import preview  # noqa: E402
import bsp  # noqa: E402

GLADOS = ROOT / "target" / "release" / "glados.exe"
DATA = ROOT / "data" / "ws5" / "train.jsonl"
BSPS = ROOT / "out" / "workshop" / "bsp"
REGISTRY = ROOT / "train" / "unsolv-staged.jsonl"
SERIES = "ws5b"


def registry():
    return [json.loads(l) for l in REGISTRY.read_text(encoding="utf-8").splitlines() if l.strip()] \
        if REGISTRY.exists() else []


def staged_titles():
    out = set()
    for p in playtest.PUZZLES.glob("glados-*-unsolv-*.p2c"):
        m = re.search(r'"Title"\s+"unsolv ([^"]*)"', p.read_bytes().decode("utf-8", "replace"))
        if m:
            out.add(m[1])
    return out


def pre_painted(wid):
    """True if the compiled map paints gel with @pre_paint_sprayer entities (or has no map)."""
    path = BSPS / f"{wid}.bsp"
    if not path.exists():
        return True
    for ent in bsp.Bsp.open(path).entities():
        e = dict(ent)
        if e.get("classname") == "info_paint_sprayer" and "pre_paint_sprayer" in e.get("targetname", ""):
            return True
    return False


def run(args, stdin=None):
    r = subprocess.run([str(GLADOS), *args], input=stdin, capture_output=True, text=True,
                       encoding="utf-8", cwd=ROOT)
    if r.returncode:
        sys.exit(f"glados {' '.join(args)} failed:\n{r.stderr[-2000:]}")
    return [json.loads(l) for l in r.stdout.splitlines() if l.strip()]


def score(tokens, budget, p2c_dir=None):
    args = ["score", "--editor-rules", "off", "--budget", str(budget)]
    if p2c_dir:
        args += ["--p2c-all", str(p2c_dir)]
    return run(args, "\n".join(tokens) + "\n")


def rating(r):
    if "stars" not in r:
        return "?"
    return f"{r['stars']}/5 ({r['votes']:,})" if r["stars"] else f"not yet ({r['votes']:,})"


def candidates(n, skip_ids, skip_titles, pool, order="size"):
    """Smallest-first (or best-rated-first) unsolvable, unstaged levels with a usable map, one
    per item set."""
    rows = []
    for line in DATA.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("solver_verdict") == "unsolvable" and r.get("workshop_id") not in skip_ids \
                and r.get("title", "") not in skip_titles:
            rows.append(r)
    rows.sort(key=lambda r: r["n_tokens"])
    if order == "rating":
        rows.sort(key=lambda r: (-r.get("stars", 0), -r.get("votes", 0)))
    picked, seen, tmp = [], set(), ROOT / "out" / ".stage-probe"
    for i in range(0, len(rows), 40):
        if len(picked) >= pool:
            break
        chunk = [r for r in rows[i:i + 40] if not pre_painted(r["workshop_id"])]
        if not chunk:
            continue
        s50 = score([r["tokens"] for r in chunk], 50_000, tmp)
        for r, s in zip(chunk, s50):
            if s.get("verdict") != "unsolvable" or len(picked) >= pool:
                continue
            p2c = tmp / f"{s['line']:05d}.p2c"
            sig = re.sub(r"^\d+x\d+x\d+: ", "", preview.summary(p2c))
            if sig in seen:
                continue
            seen.add(sig)
            picked.append(r)
    return picked[:pool]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5, help="how many levels to stage")
    ap.add_argument("--dry-run", action="store_true", help="show the picks, write and copy nothing")
    ap.add_argument("--order", choices=["size", "rating"], default="size",
                    help="smallest first, or most Workshop stars (then ratings) first")
    a = ap.parse_args()
    if not GLADOS.exists():
        sys.exit("target/release/glados.exe is missing: cargo build --release (not during a training run)")

    reg = registry()
    skip_ids = {r["workshop_id"] for r in reg}
    skip_titles = staged_titles()
    tmp = ROOT / "out" / ".stage-probe"
    tmp.mkdir(parents=True, exist_ok=True)
    # a pool larger than n: some picks fail the 250k re-score or the editor check
    pool = candidates(a.n, skip_ids, skip_titles, a.n * 3, a.order)
    tokens = [r["tokens"] for r in pool]
    s50 = score(tokens, 50_000)
    s250 = score(tokens, 250_000)
    day = datetime.date.today().isoformat()
    n = 1
    while (out := ROOT / "out" / (f"unsolv-{day}" + (f"-{n}" if n > 1 else ""))).exists():
        n += 1
    keep = []
    for r, x, y in zip(pool, s50, s250):
        if x["verdict"] == "unsolvable" and y["verdict"] == "unsolvable":
            keep.append((r, x, y))
    keep = keep[:a.n]
    if len(keep) < a.n:
        print(f"only {len(keep)} of {a.n} still unsolvable at 250k; run again after this batch")
    if a.dry_run or not keep:
        for r, _, _ in keep:
            print(r["workshop_id"], r["n_tokens"], rating(r), r["title"])
        return

    (out / "p2c").mkdir(parents=True)
    ktok = [r["tokens"] for r, _, _ in keep]
    (out / "tokens.txt").write_text("\n".join(ktok) + "\n", encoding="utf-8")
    sc50 = score(ktok, 50_000, out / "p2c")
    sc250 = score(ktok, 250_000)
    for name, rows in (("scores-50k.jsonl", sc50), ("scores-250k.jsonl", sc250)):
        (out / name).write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
    files = sorted((out / "p2c").glob("*.p2c"))
    flags = run(["editor", *map(str, files)])
    (out / "editor.jsonl").write_text("".join(json.dumps(x) + "\n" for x in flags), encoding="utf-8")
    if len(files) != len(keep):
        sys.exit(f"{len(files)} p2c files for {len(keep)} levels: check {out}")

    last = max([int(m[1]) for p in playtest.PUZZLES.glob(f"glados-{SERIES}-unsolv-*.p2c")
                if (m := re.search(r"-(\d{5})\.p2c$", p.name))] or [0])
    now = int(time.time())
    rows = []
    for i, ((r, x, y), src, fl) in enumerate(zip(keep, files, flags)):
        name = f"glados-{SERIES}-unsolv-{last + 1 + i:05d}.p2c"
        dst = playtest.PUZZLES / name
        text = playtest.restamp(src.read_bytes().decode("utf-8"), now + i)
        text = playtest.retitle(text, f"unsolv {r['title']}")
        text = playtest.DESCRIPTION.sub(lambda m: f'{m[1]}"{preview.summary(src)}"', text, count=1)
        playtest.write(dst, text)
        preview.write(dst, name=f"unsolv {r['title']}")
        bad = {k: v for k, v in fl.items() if k not in ("file", "path") and v}
        rows.append((name, r, src, bad))
        with REGISTRY.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"workshop_id": r["workshop_id"], "title": r["title"], "file": name,
                                "date": day, "stars": r.get("stars"), "votes": r.get("votes")}) + "\n")
        print(f"{dst}  <- workshop {r['workshop_id']} {r['title']!r} ({rating(r)})" + (f"  EDITOR FLAGS {bad}" if bad else ""))

    lines = [f"# {len(rows)} more unsolvable Workshop levels, staged {day}", "",
             f"Source: `data/ws5/train.jsonl`, picked by `train/stage_unsolv.py`. Still unsolvable with the "
             f"current release build at 50k and 250k (`glados score --editor-rules off`). A published "
             f"puzzle was solved by its author, so the verdict is a solver gap.", "",
             "| staged file | workshop id | title | rating | tokens | items | editor flags | in game |",
             "|---|---|---|---|---:|---|---|---|"]
    for name, r, src, bad in rows:
        lines.append(f"| `{name}` | {r['workshop_id']} | {r['title']} | {rating(r)} | {r['n_tokens']} | "
                     f"{preview.summary(src)} | {bad or 'none'} | not run |")
    lines += ["", "Files: `tokens.txt`, `p2c/NNNNN.p2c` (same order), `scores-50k/250k.jsonl`, `editor.jsonl`."]
    (out / "MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"manifest: {out / 'MANIFEST.md'}")


if __name__ == "__main__":
    main()
