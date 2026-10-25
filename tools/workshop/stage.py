"""Stage one Workshop level in the Puzzle Maker folder, for manual play-testing.

    python tools/workshop/stage.py --id 100091397
    python tools/workshop/stage.py --id 100091397 --dry-run
    python tools/workshop/stage.py --id 100091397 --force   # make another copy

The workbench's **Stage** button calls this. It compiles the level's token sequence with the
release `glados.exe` (`score --editor-rules off --p2c-all`), copies the resulting .p2c into the
Puzzle Maker folder as `glados-wb-NNNNN.p2c`, adds a preview .jpg, retitles it `wb <title>`, and
marks the level `staged` in `train/workshop-annotations.jsonl`. Each new copy is also logged in
`train/workbench-staged.jsonl` (workshop id, title, file, date, rating) so agents and the next
run can find it. A level already staged there (or in `train/unsolv-staged.jsonl`) is reused
unless `--force` is given.

It never starts or touches Portal 2. It needs `target/release/glados.exe`; build it first
(`cargo build --release`, not during a training run). The Puzzle Maker folder is the same one
`train/playtest.py` uses; `--puzzles DIR` or `GLADOS_PUZZLES` overrides it. `--registry FILE`,
`--annotations FILE`, `--all FILE`, `--train FILE` and `--test FILE` override the other paths.
"""

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "tools" / "workshop"))

import annotations  # noqa: E402
import playtest  # noqa: E402
import preview  # noqa: E402

DEFAULT_ALL = ROOT / "out" / "workshop" / "levels.jsonl"
DEFAULT_TRAIN = ROOT / "data" / "ws5" / "train.jsonl"
DEFAULT_TEST = ROOT / "data" / "ws5" / "test.jsonl"
DEFAULT_REGISTRY = ROOT / "train" / "workbench-staged.jsonl"
LEGACY_REGISTRY = ROOT / "train" / "unsolv-staged.jsonl"
SERIES = "wb"
MAX_TITLE = 120


class StageError(Exception):
    pass


def glados_bin():
    override = os.environ.get("GLADOS_BIN")
    if override:
        return Path(override)
    for name in ("glados.exe", "glados"):
        path = ROOT / "target" / "release" / name
        if path.is_file():
            return path
    raise StageError("target/release/glados.exe is missing: cargo build --release (not during a training run)")


def find_level(wid, paths):
    """The corpus/split record for `wid` that has tokens (the corpus usually comes first)."""
    fallback = None
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if str(r.get("workshop_id")) != wid:
                    continue
                if str(r.get("tokens") or "").strip():
                    return r
                fallback = fallback or r
    if fallback is not None:
        raise StageError("this level has no token sequence to stage")
    raise StageError(f"no level {wid} in {', '.join(str(p) for p in paths)}")


def registry_rows(path):
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def existing_file(wid, puzzles, registries):
    """The newest staged file for `wid` that is still in the Puzzle Maker folder, if any."""
    for path in registries:
        for row in reversed(registry_rows(path)):
            if str(row.get("workshop_id")) != wid:
                continue
            name = row.get("file")
            if name and (Path(puzzles) / name).is_file():
                return row
    return None


def next_name(puzzles):
    numbers = []
    for path in Path(puzzles).glob(f"glados-{SERIES}-*.p2c"):
        m = re.search(r"-(\d{5})\.p2c$", path.name)
        if m:
            numbers.append(int(m.group(1)))
    n = max(numbers, default=0) + 1
    while (Path(puzzles) / f"glados-{SERIES}-{n:05d}.p2c").exists() or \
            (Path(puzzles) / f"glados-{SERIES}-{n:05d}.jpg").exists():
        n += 1
    return f"glados-{SERIES}-{n:05d}.p2c"


def safe_title(title, wid):
    title = re.sub(r"\s+", " ", str(title or "")).strip() or f"workshop {wid}"
    title = title.replace('"', "'")
    return f"wb {title}"[:MAX_TITLE]


def compile_p2c(tokens, budget):
    """Run the release solver and return its first .p2c for this one token line."""
    (ROOT / "out").mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="glados-stage-", dir=ROOT / "out"))
    try:
        cmd = [str(glados_bin()), "score", "--editor-rules", "off", "--budget", str(budget),
               "--p2c-all", str(work)]
        try:
            r = subprocess.run(cmd, input=tokens.strip() + "\n", text=True, encoding="utf-8",
                               errors="replace", capture_output=True, cwd=ROOT, timeout=180)
        except subprocess.TimeoutExpired:
            raise StageError("glados score timed out after 180s")
        if r.returncode:
            raise StageError("glados score failed:\n" + r.stderr.strip()[-2000:])
        files = sorted(work.glob("*.p2c"))
        if not files:
            detail = (r.stderr or r.stdout).strip()[-2000:]
            raise StageError("glados score wrote no .p2c for this level" + (":\n" + detail if detail else ""))
        text = files[0].read_bytes().decode("utf-8")
        try:
            summary = preview.summary(files[0])
        except Exception as e:
            raise StageError(f"could not read the compiled level: {e}")
        return text, summary
    finally:
        shutil.rmtree(work, ignore_errors=True)


def stage(args):
    wid = str(args.id)
    all_paths = [args.all, args.train, args.test]
    record = find_level(wid, all_paths)
    tokens = record["tokens"]
    title = safe_title(record.get("title"), wid)
    puzzles = Path(args.puzzles)
    if not puzzles.is_dir():
        raise StageError(f"Puzzle Maker folder not found: {puzzles} (pass --puzzles DIR)")
    registries = [args.registry, LEGACY_REGISTRY]
    existing = existing_file(wid, puzzles, registries)
    if existing and not args.force:
        ann = annotations.update(wid, path=args.annotations, staged=True)
        return {
            "ok": True,
            "already": True,
            "workshop_id": wid,
            "file": existing["file"],
            "path": str(puzzles / existing["file"]),
            "title": existing.get("title") or title,
            "annotation": {"workshop_id": wid, **ann},
        }

    name = next_name(puzzles)
    if args.dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "workshop_id": wid,
            "file": name,
            "path": str(puzzles / name),
            "title": title,
            "tokens": len(tokens.split()),
        }

    text, summary = compile_p2c(tokens, args.budget)
    dst = puzzles / name
    try:
        text = playtest.restamp(text, int(datetime.datetime.now().timestamp()))
        text = playtest.retitle(text, title)
        text = playtest.DESCRIPTION.sub(lambda m: f'{m[1]}"{summary}"', text, count=1)
        playtest.write(dst, text)
        preview.write(dst, name=title)
    except Exception as e:
        for p in (dst, dst.with_suffix(".jpg")):
            try:
                p.unlink()
            except OSError:
                pass
        raise StageError(f"could not write the staged level: {e}")

    row = {
        "workshop_id": wid,
        "title": record.get("title") or title,
        "file": name,
        "date": datetime.date.today().isoformat(),
        "stars": record.get("stars"),
        "votes": record.get("votes"),
        "tokens": len(tokens.split()),
    }
    registry = Path(args.registry)
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    ann = annotations.update(wid, path=args.annotations, staged=True)
    return {
        "ok": True,
        "already": False,
        "workshop_id": wid,
        "file": name,
        "path": str(dst),
        "preview": str(dst.with_suffix(".jpg")),
        "title": title,
        "annotation": {"workshop_id": wid, **ann},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True, help="Workshop id to stage")
    ap.add_argument("--all", default=str(DEFAULT_ALL), help=f"corpus (default {DEFAULT_ALL})")
    ap.add_argument("--train", default=str(DEFAULT_TRAIN), help=f"ws5 train split (default {DEFAULT_TRAIN})")
    ap.add_argument("--test", default=str(DEFAULT_TEST), help=f"ws5 test split (default {DEFAULT_TEST})")
    ap.add_argument("--annotations", default=str(annotations.DEFAULT),
                    help=f"annotation store (default {annotations.DEFAULT})")
    ap.add_argument("--registry",
                    default=os.environ.get("GLADOS_STAGE_REGISTRY", str(DEFAULT_REGISTRY)),
                    help=f"staged-file registry (default {DEFAULT_REGISTRY})")
    ap.add_argument("--puzzles", default=os.environ.get("GLADOS_PUZZLES", str(playtest.PUZZLES)),
                    help="Puzzle Maker folder (default %(default)s)")
    ap.add_argument("--budget", type=int, default=50_000, help="solver budget for the p2c compile (default 50000)")
    ap.add_argument("--force", action="store_true", help="stage another copy even if one already exists")
    ap.add_argument("--dry-run", action="store_true", help="show the target file, write nothing")
    a = ap.parse_args()
    if not Path(a.registry).is_absolute():
        a.registry = str(ROOT / a.registry)
    try:
        print(json.dumps(stage(a), ensure_ascii=False))
    except StageError as e:
        print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False))
        return 1
    except Exception as e:  # a bad corpus line, a missing file, etc.
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
