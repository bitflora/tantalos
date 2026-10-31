"""Stage Goblin's self-play solutions (train/school.py) for replay in Portal 2 by train/play.py.

    .venv/Scripts/python train/stage_school_play.py e1 --round 0
    .venv/Scripts/python train/play.py --plans out/school-play/e1-r00/plans.jsonl --launch --fast --out out/school-play/e1-r00/play
    .venv/Scripts/python train/playreport.py out/school-play/e1-r00/play --md out/school-play/e1-r00/REPORT.md

The buffer's plans pass only our conservative model; the round's liberal-queue plans only the
liberal one. Before more training on them, the game should agree. This picks, in play order:
every liberal-only plan of the round (routed with --mode liberal), every buffer level from a
Workshop level, and a sample of --gen buffer levels the level net wrote; and writes them to
out/school-play/NAME-rRR/: p2c/NNNNN.p2c, plans.jsonl (p2c, plan, mode, kind, identity, source:
what play.py --plans reads) and MANIFEST.md. A plan `glados-score route` can't place step by step
in full is left out and listed in the manifest with why: those need route.rs work first, and
are often the riskiest plans (portal loops, flings), so the batch is biased toward easy ones. Nothing touches the game; needs target/release/glados-score.exe.
"""
import argparse, json, random, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GLADOS = ROOT / "target" / "release" / "glados-score.exe"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="the school run (runs/school/NAME)")
    ap.add_argument("--round", type=int, default=0, help="its round whose levels to stage")
    ap.add_argument("--gen", type=int, default=60, help="buffer levels the level net wrote, sampled")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: out/school-play/NAME-rRR")
    a = ap.parse_args()

    school = ROOT / "runs" / "school" / a.name
    rnd = school / f"r{a.round:02}"
    source = {r["identity"]: r.get("source") or "" for r in read_jsonl(rnd / "pool.jsonl")}
    buffer = [r for r in read_jsonl(school / "buffer.jsonl") if r.get("round") == a.round]
    ws = [r for r in buffer if source.get(r["identity"], "").startswith("ws5")]
    gen = [r for r in buffer if not source.get(r["identity"], "").startswith("ws5")]
    gen = random.Random(a.seed).sample(gen, min(a.gen, len(gen)))
    liberal = read_jsonl(rnd / "liberal-queue.jsonl") if (rnd / "liberal-queue.jsonl").exists() else []
    picks = ([(r, "liberal", "liberal") for r in liberal] + [(r, "ws", "conservative") for r in ws]
             + [(r, "gen", "conservative") for r in gen])
    if not picks:
        sys.exit(f"nothing to stage in {rnd}")

    out = Path(a.out) if a.out else ROOT / "out" / "school-play" / f"{a.name}-r{a.round:02}"
    if (out / "plans.jsonl").exists():
        sys.exit(f"{out} is already staged")
    p2c = out / "p2c"
    p2c.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(GLADOS), "score", "--budget", "1", "--p2c-all", str(p2c)],
                   input="\n".join(r["tokens"] for r, _, _ in picks) + "\n",
                   capture_output=True, text=True, check=True)

    rows, skipped = [], []
    for i, (r, kind, mode) in enumerate(picks, 1):
        f = p2c / f"{i:05}.p2c"
        row = {"p2c": f"p2c/{f.name}", "plan": r["plan"], "mode": mode, "kind": kind,
               "identity": r["identity"], "source": source.get(r["identity"], "")}
        why = None
        if not f.exists():
            why = "no p2c written"
        else:
            run = subprocess.run([str(GLADOS), "route", "--plan", r["plan"], "--mode", mode, str(f)],
                                 capture_output=True, text=True, encoding="utf-8")
            if run.returncode != 0:
                why = (run.stderr or run.stdout).strip().splitlines()[-1:] or ["route failed"]
                why = why[0]
            else:
                net = json.loads(run.stdout.splitlines()[0]).get("net") or {}
                if net.get("verdict") not in (None, "solved"):
                    why = f"route's check: {net.get('verdict')}"
                elif net.get("steps_ok", 0) < net.get("steps", 0) or not net.get("steps"):
                    # e.g. --mode liberal grounds only the steps the conservative model allows
                    why = f"route places {net.get('steps_ok', 0)} of {net.get('steps', 0)} steps"
        if why:
            skipped.append((row, why))
        else:
            rows.append(row)
    with open(out / "plans.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    count = lambda k: sum(r["kind"] == k for r in rows)
    lines = [f"# Self-play replay: {a.name} round {a.round}", "",
             f"{len(rows)} levels to play: {count('liberal')} liberal-only plans (of {len(liberal)}), "
             f"{count('ws')} Workshop buffer levels (of {len(ws)}), {count('gen')} level-net buffer levels "
             f"(sampled {len(gen)}, seed {a.seed}). Order: liberal, ws, gen.", "",
             "```bash",
             f".venv/Scripts/python train/play.py --plans {(out / 'plans.jsonl').relative_to(ROOT).as_posix()} "
             f"--launch --fast --out {(out / 'play').relative_to(ROOT).as_posix()}",
             f".venv/Scripts/python train/playreport.py {(out / 'play').relative_to(ROOT).as_posix()} "
             f"--md {(out / 'REPORT.md').relative_to(ROOT).as_posix()}",
             "```", ""]
    if skipped:
        lines += [f"Left out ({len(skipped)}):", ""] + [f"- {r['p2c']} {r['kind']} {r['identity']}: {w}"
                                                        for r, w in skipped] + [""]
    (out / "MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
