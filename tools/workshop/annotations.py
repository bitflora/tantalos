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

The generated-level store (`--generated`, `train/generated-annotations.jsonl`, written by
`glados-workbench --generated`) has the same fields keyed by `"id"` (a hash of the level's
tokens), plus `human_verdict` (solvable, unsolvable or broken, from a play-test), `tokens` and
`origin` (run, file and line). An unsolvable or broken verdict also adds the level to
`data/ingame/human-unsolvable.jsonl` (source `glados-workbench ...`), which training runs
train away from; clearing it removes that row again:

    python tools/workshop/annotations.py --generated --set 3f0c... --verdict unsolvable \
        --notes "exit needs both buttons; the second is unreachable"

`rating` is 0 (unrated) or 1-5; `complexity` is the same scale for how complex the level is; `human_solved` means the user solved it by hand. `excluded` means "leave this level out of the ws5 test split";
`tools/workshop/ws3.py` moves it to train when rebuilding the split, and `train/goblin_data.py`
leaves it out of `test-ws.jsonl`.
"""

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "train" / "workshop-annotations.jsonl"
GENERATED = ROOT / "train" / "generated-annotations.jsonl"
HUMAN_UNSOLVABLE = ROOT / "data" / "ingame" / "human-unsolvable.jsonl"
HUMAN_SOURCE = "glados-workbench"
VERDICTS = ("solvable", "unsolvable", "broken")
# Kept with a generated level's annotation, but not what makes one non-empty.
LEVEL_FIELDS = ("tokens", "origin")


def empty():
    return {"staged": False, "rating": 0, "complexity": 0, "human_solved": False, "notes": "", "excluded": False,
            "human_verdict": "", "tokens": "", "origin": "", "updated": 0}


def is_empty(a):
    return all(a[k] == v for k, v in empty().items() if k not in LEVEL_FIELDS + ("updated",))


def key_of(path):
    """The id field of a store: `id` for the generated-level store, else `workshop_id`."""
    return "id" if Path(path).resolve() == GENERATED.resolve() else "workshop_id"


def normalize(a):
    out = empty()
    out["staged"] = bool(a.get("staged", False))
    out["rating"] = max(0, min(5, int(a.get("rating", 0) or 0)))
    out["complexity"] = max(0, min(5, int(a.get("complexity", 0) or 0)))
    out["human_solved"] = bool(a.get("human_solved", False))
    out["notes"] = str(a.get("notes", "") or "")
    out["excluded"] = bool(a.get("excluded", False))
    verdict = str(a.get("human_verdict", "") or "")
    out["human_verdict"] = verdict if verdict in VERDICTS else ""
    out["tokens"] = " ".join(str(a.get("tokens", "") or "").split())
    out["origin"] = str(a.get("origin", "") or "")
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
        out[str(row["workshop_id"] if "workshop_id" in row else row["id"])] = normalize(row)
    return out


def save(records, path=DEFAULT):
    """Write the non-empty annotations atomically, sorted by id. The generated-level fields are
    written only when set, so the Workshop store keeps its shape (as glados-workbench does)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    key = key_of(path)
    lines = []
    for wid in sorted(records, key=str):
        a = normalize(records[wid])
        if is_empty(a):
            continue
        row = {key: str(wid)}
        for k, v in a.items():
            if k in ("human_verdict",) + LEVEL_FIELDS and not v:
                continue
            row[k] = v
        lines.append(json.dumps(row, ensure_ascii=False) + "\n")
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
    old = normalize(records.get(str(wid), empty()))
    a = dict(old)
    for key in ("staged", "rating", "complexity", "human_solved", "notes", "excluded", "human_verdict") + LEVEL_FIELDS:
        if key in changes and changes[key] is not None:
            a[key] = changes[key]
    if a["human_verdict"] and a["human_verdict"] not in VERDICTS:
        raise ValueError(f"unknown human verdict {a['human_verdict']!r} (one of {', '.join(VERDICTS)})")
    a = normalize(a)
    a["updated"] = int(time.time())
    bad = lambda x: x["human_verdict"] in ("unsolvable", "broken")
    if (bad(a) or bad(old)) and a["tokens"]:
        note = None
        if bad(a):
            note = f"Human play-test (workbench): {a['human_verdict']}."
            if a["notes"].strip():
                note += " " + a["notes"].strip()
        sync_human_unsolvable(a["tokens"], a["origin"], note)
    records[str(wid)] = a
    save(records, path)
    return a


def sync_human_unsolvable(tokens, origin, note, path=HUMAN_UNSOLVABLE):
    """Make human-unsolvable.jsonl agree with a generated level's verdict, like
    glados-workbench: drop the rows this tool wrote for these tokens, then add one back if
    `note` is given and nobody else's row covers the level."""
    path = Path(path)
    tokens = " ".join(tokens.split())
    kept, covered, changed = [], False, False
    # Other rows are kept byte for byte, line endings included; a new row matches them.
    old = path.read_bytes().decode("utf-8") if path.exists() else ""
    eol = "\r\n" if "\r\n" in old else "\n"
    for raw in old.splitlines(keepends=True):
        line = raw.rstrip("\r\n")
        if not line.strip():
            continue
        row = json.loads(line)
        same = " ".join(str(row.get("tokens", "")).split()) == tokens
        if same and str(row.get("source", "")).startswith(HUMAN_SOURCE):
            changed = True
            continue
        covered |= same
        kept.append(raw if raw.endswith("\n") else raw + eol)
    if note is not None and not covered:
        kept.append(json.dumps({"tokens": tokens, "verdict": "unsolvable",
                                "source": f"{HUMAN_SOURCE} {origin}",
                                "checked": time.strftime("%Y-%m-%d", time.gmtime()),
                                "note": note}, ensure_ascii=False) + eol)
        changed = True
    if changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes("".join(kept).encode("utf-8"))
        tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--annotations", default=None, help=f"annotation file (default {DEFAULT})")
    ap.add_argument("--generated", action="store_true",
                    help=f"use the generated-level store ({GENERATED})")
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
    ap.add_argument("--verdict", choices=VERDICTS + ("none",),
                    help="with --set: human play-test verdict (generated levels; unsolvable/broken "
                         "also go to data/ingame/human-unsolvable.jsonl)")
    ap.add_argument("--exclude", action="store_true", help="with --set: exclude from test data")
    ap.add_argument("--no-exclude", action="store_true", help="with --set: do not exclude from test data")
    a = ap.parse_args()
    path = Path(a.annotations or (GENERATED if a.generated else DEFAULT))

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
            "human_verdict": "" if a.verdict == "none" else a.verdict,
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
