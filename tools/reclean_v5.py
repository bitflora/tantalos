"""Apply the current editor rules to data/v5's generated levels again.

    .venv/Scripts/python tools/reclean_v5.py [--root CHECKOUT]

The 2026-10-16 cleanup (data/v5/cleanup.log) dropped generated levels our editor rules called
invalid into data/v5/dropped-2026-10-16.jsonl.gz. The rules have changed since: new ones
(panel-in-rail-track, field-in-rail-track, rail-on-door-frame) and narrower old ones (shared-cell,
in-field, fields-overlap). So, per data/v5/levels-[1-9].jsonl:

- a level the rules now flag is dropped, unless it was already labelled invalid (those stay, as
  in the first cleanup);
- a level dropped from that file before that the rules no longer flag comes back.

Rules are checked with `glados score --budget 1` (editor rules on); labels are not recomputed.
Newly dropped levels go to data/v5/dropped-<date>b.jsonl.gz with "file" and "problem". Each
file is rewritten next to itself and replaced only when complete. A summary is appended to
data/v5/cleanup.log.
"""

import argparse
import datetime
import gzip
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

RULE = re.compile(r"[(, ]([a-z]+(?:-[a-z]+)+)[,)]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]), help="checkout with data/")
    a = ap.parse_args()
    root = Path(a.root)
    v5 = root / "data" / "v5"
    glados = root / "target" / "release" / "glados.exe"

    def problems(records):
        out = subprocess.run([str(glados), "score", "--budget", "1"],
                             input="".join(r["tokens"] + "\n" for r in records),
                             capture_output=True, text=True, encoding="utf-8", check=True).stdout.splitlines()
        assert len(out) == len(records)
        return [json.loads(s).get("problem", "") if json.loads(s)["verdict"] == "invalid" else "" for s in out]

    before = defaultdict(list)
    with gzip.open(v5 / "dropped-2026-10-16.jsonl.gz", "rt", encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            before[r.pop("file")].append(r)

    today = datetime.date.today().isoformat()
    log = [f"\n{today} (tools/reclean_v5.py): current editor rules applied again; levels labelled "
           "invalid kept; levels dropped on 2026-10-16 that now pass restored."]
    totals = Counter()
    with gzip.open(v5 / f"dropped-{today}b.jsonl.gz", "wt", encoding="utf-8") as dropped:
        for path in sorted(v5.glob("levels-[1-9].jsonl")):
            rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
            check = [r for r in rows if r["verdict"] != "invalid"]
            bad = {id(r): p for r, p in zip(check, problems(check)) if p}
            back = [r for r, p in zip(before[path.name], problems(before[path.name])) if not p]
            for r in back:
                r.pop("problem_2026_10_16", None)
            rules = Counter()
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                for r in rows:
                    if id(r) in bad:
                        rules.update(set(RULE.findall(bad[id(r)])))
                        dropped.write(json.dumps({**r, "file": path.name, "problem": bad[id(r)]}) + "\n")
                    else:
                        f.write(json.dumps(r) + "\n")
                f.writelines(json.dumps(r) + "\n" for r in back)
            tmp.replace(path)
            kept = len(rows) - len(bad) + len(back)
            line = (f"{path.name}: {len(rows)} -> {kept} (dropped {len(bad)}, restored {len(back)} of "
                    f"{len(before[path.name])}); rules {dict(rules.most_common())}")
            print(line, flush=True)
            log.append(line)
            totals.update(before=len(rows), after=kept, dropped=len(bad), restored=len(back))
    log.append(f"total: {totals['before']} -> {totals['after']} (dropped {totals['dropped']}, "
               f"restored {totals['restored']})")
    print(log[-1])
    with open(v5 / "cleanup.log", "a", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")


if __name__ == "__main__":
    sys.exit(main())
