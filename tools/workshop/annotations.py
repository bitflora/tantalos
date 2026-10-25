"""Read and write the workbench's per-level annotations.

The `glados-workbench` browser app saves one JSON object per annotated Workshop level in
`train/workshop-annotations.jsonl`. Agents can use this module or its command line instead of
the browser:

    python tools/workshop/annotations.py --get 100091397
    python tools/workshop/annotations.py --excluded
    python tools/workshop/annotations.py --json
    python tools/workshop/annotations.py --set 100091397 --rating 4 --staged \
        --notes "solver gap: laser timing"
    python tools/workshop/annotations.py --set 100091397 --no-staged --clear-notes

The file is the canonical store: `train/workshop-annotations.jsonl`, one object per line:

    {"workshop_id":"100091397","staged":true,"rating":4,"complexity":3,"human_solved":true,"notes":"...","excluded":true,
     "updated":1792888260}

`rating` is 0 (unrated) or 1-5; `complexity` is the same scale for how complex the level is; `human_solved` means the user solved it by hand. `excluded` means "leave this level out of the ws5 test split";
`tools/workshop/ws3.py` moves it to train when rebuilding the split, and `train/plans.py`
leaves it out of `test-ws.jsonl`.
"""

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "train" / "workshop-annotations.jsonl"


def empty():
    return {"staged": False, "rating": 0, "complexity": 0, "human_solved": False, "notes": "", "excluded": False, "updated": 0}


def normalize(a):
    out = empty()
    out["staged"] = bool(a.get("staged", False))
    out["rating"] = max(0, min(5, int(a.get("rating", 0) or 0)))
    out["complexity"] = max(0, min(5, int(a.get("complexity", 0) or 0)))
    out["human_solved"] = bool(a.get("human_solved", False))
    out["notes"] = str(a.get("notes", "") or "")
    out["excluded"] = bool(a.get("excluded", False))
    out["updated"] = int(a.get("updated", 0) or 0)
    return out


def load(path=DEFAULT):
    """{workshop_id: annotation}; missing files are an empty store."""
    path = Path(path)
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[str(row["workshop_id"])] = normalize(row)
    return out


def save(records, path=DEFAULT):
    """Write the non-empty annotations atomically, sorted by id."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for wid in sorted(records, key=str):
        a = normalize(records[wid])
        if a == empty():
            continue
        lines.append(json.dumps({"workshop_id": str(wid), **a}, ensure_ascii=False) + "\n")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(lines), encoding="utf-8", newline="\n")
    tmp.replace(path)


def excluded_ids(path=DEFAULT):
    """Workshop ids marked "exclude from test data"."""
    return {wid for wid, a in load(path).items() if a["excluded"]}


def get(wid, path=DEFAULT):
    return load(path).get(str(wid), empty())


def update(wid, path=DEFAULT, **changes):
    """Set fields on one level and save; returns its annotation."""
    records = load(path)
    a = normalize(records.get(str(wid), empty()))
    for key in ("staged", "rating", "complexity", "human_solved", "notes", "excluded"):
        if key in changes and changes[key] is not None:
            a[key] = changes[key]
    a = normalize(a)
    a["updated"] = int(time.time())
    records[str(wid)] = a
    save(records, path)
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--annotations", default=str(DEFAULT), help=f"annotation file (default {DEFAULT})")
    ap.add_argument("--get", metavar="ID", help="print one level's annotation")
    ap.add_argument("--set", metavar="ID", help="update one level's annotation")
    ap.add_argument("--excluded", action="store_true", help="print the ids excluded from test data")
    ap.add_argument("--json", action="store_true", help="print all annotations as a JSON object")
    ap.add_argument("--staged", action="store_true", help="with --set: staged for manual playtesting")
    ap.add_argument("--no-staged", action="store_true", help="with --set: not staged")
    ap.add_argument("--rating", type=int, choices=range(0, 6), metavar="0-5", help="with --set: my rating")
    ap.add_argument("--complexity", type=int, choices=range(0, 6), metavar="0-5", help="with --set: my complexity rating")
    ap.add_argument("--human-solved", action="store_true", help="with --set: I solved it by hand in Portal 2")
    ap.add_argument("--no-human-solved", action="store_true", help="with --set: not human-solved")
    ap.add_argument("--notes", help="with --set: replace the notes")
    ap.add_argument("--clear-notes", action="store_true", help="with --set: clear the notes")
    ap.add_argument("--exclude", action="store_true", help="with --set: exclude from test data")
    ap.add_argument("--no-exclude", action="store_true", help="with --set: do not exclude from test data")
    a = ap.parse_args()
    path = Path(a.annotations)

    if a.set:
        if a.staged and a.no_staged:
            ap.error("--staged and --no-staged conflict")
        if a.exclude and a.no_exclude:
            ap.error("--exclude and --no-exclude conflict")
        changes = {
            "staged": True if a.staged else False if a.no_staged else None,
            "rating": a.rating,
            "complexity": a.complexity,
            "human_solved": True if a.human_solved else False if a.no_human_solved else None,
            "notes": "" if a.clear_notes else a.notes,
            "excluded": True if a.exclude else False if a.no_exclude else None,
        }
        print(json.dumps(update(a.set, path, **changes), ensure_ascii=False))
        return

    if a.get:
        print(json.dumps(get(a.get, path), ensure_ascii=False))
        return
    if a.excluded:
        for wid in sorted(excluded_ids(path)):
            print(wid)
        return
    print(json.dumps(load(path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
