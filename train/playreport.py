"""Sums up a batch of train/play.py runs (its --out folder) as Markdown.

    python train/playreport.py out/play/base                # to the terminal
    python train/playreport.py out/play/base --md REPORT.md

Per leg status (ok, unconfirmed, retried, recovered, failed, skipped) and, for the legs that
didn't work first time, why (play.py `classify`: exec, route, blocked, plan?), by action; then
every leg that needs a look (blocked, plan?, route), with its problems, the careful retry's
outcome, events and screenshots. Legs after one put right by teleporting are marked tainted:
their trouble may come from the recovery rather than from themselves.
"""
import argparse, collections, json
from pathlib import Path


def load(folder):
    levels = []
    for l in (folder / "summary.jsonl").read_text().splitlines():
        if not l.strip():
            continue
        row = json.loads(l)
        legs_file = folder / row["key"] / "legs.jsonl"
        row["leg_records"] = [json.loads(x) for x in legs_file.read_text().splitlines() if x.strip()] \
            if legs_file.exists() else []
        levels.append(row)
    return levels


def why(r):
    return "; ".join(([r["why"]] if r.get("why") else []) + r.get("problems", []))


def report(folder, levels):
    out = [f"# Play report: {folder}", ""]
    played = [l for l in levels if "error" not in l]
    errors = [l for l in levels if "error" in l]
    legs = [r for l in played for r in l["leg_records"] if r["leg"] > 0]
    status = collections.Counter(r["status"] for r in legs)
    out.append(f"{len(played)} levels played, {len(errors)} not ({len(legs)} legs). "
               + ", ".join(f"{v} {k}" for k, v in status.most_common()) + ".")
    clean = sum(all(r["status"] in ("ok", "unconfirmed") for r in l["leg_records"] if r["leg"] > 0)
                for l in played)
    out.append(f"Levels played through without a retry or teleport: {clean}/{len(played)}.")
    first = sum(r["status"] in ("ok", "unconfirmed") for r in legs)
    out.append(f"Legs right first time: {first}/{len(legs)} ({100 * first / max(len(legs), 1):.0f}%).")
    out += ["", "## By action", "", "| action | legs | first time | retried (exec) | recovered/failed | classes |",
            "|---|---|---|---|---|---|"]
    by = collections.defaultdict(list)
    for r in legs:
        by[r["action"]].append(r)
    for a, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        cls = collections.Counter(r.get("cls") for r in rs if r.get("cls"))
        out.append(f"| {a} | {len(rs)} | {sum(r['status'] in ('ok', 'unconfirmed') for r in rs)} | "
                   f"{sum(r['status'] == 'retried' for r in rs)} | "
                   f"{sum(r['status'] in ('recovered', 'failed') for r in rs)} | "
                   + ", ".join(f"{v} {k}" for k, v in cls.most_common()) + " |")
    starts = [(l, r) for l in played for r in l["leg_records"] if r["leg"] == 0]
    if starts:
        out += ["", "## Levels that didn't start as the route has them", ""]
        for l, r in starts:
            out.append(f"- `{l['key']}`: " + "; ".join(r["problems"])
                       + (f" (still after recovery: {'; '.join(r['after_recovery'])})" if r.get("after_recovery") else ""))
    if errors:
        out += ["", "## Not played", ""]
        out += [f"- `{l['key']}`: {l['error'][:300]}" for l in errors]
    look = [(l, r) for l in played for r in l["leg_records"] if r.get("cls") in ("blocked", "plan?", "route")]
    out += ["", f"## To look at ({len(look)})", ""]
    for l, r in look:
        t = " (tainted: after a recovery)" if r.get("tainted") else ""
        out.append(f"### `{l['key']}` leg {r['leg']}: {r['text']} — **{r['cls']}**, {r['status']}{t}")
        out.append("")
        out.append(f"- first try: {why(r['first']) or 'no problem seen'}")
        if r.get("retry"):
            out.append(f"- careful retry: {why(r['retry']) or 'no problem seen'}")
        ev = r["first"].get("events", []) + (r.get("retry") or {}).get("events", [])
        if ev:
            out.append("- events: " + "; ".join(f"{e['kind']}: {e['text']}" for e in ev))
        if r.get("probe"):
            p = {k: v for k, v in r["probe"].items() if k != "look_shot"}
            if p:
                out.append(f"- probe: {json.dumps(p)}")
        shots = [r.get("fail_shot"), r.get("retry_shot"), (r.get("probe") or {}).get("look_shot")]
        shots = [s for s in shots if s]
        if shots:
            out.append("- screenshots: " + ", ".join(f"[{s}]({l['key']}/{s}.jpg)" for s in shots))
        out.append("")
    execs = [(l, r) for l in played for r in l["leg_records"] if r.get("cls") == "exec"]
    out += [f"## Harness slips (exec, {len(execs)})", ""]
    for l, r in execs:
        out.append(f"- `{l['key']}` leg {r['leg']} {r['action']}: {why(r['first'])}")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--md", help="write the report here (paths in it are relative to the batch folder)")
    a = ap.parse_args()
    folder = Path(a.folder)
    text = report(folder, load(folder))
    if a.md:
        Path(a.md).write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
