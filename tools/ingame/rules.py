"""Score `glados editor` predictions against the game's INVALID ITEM lists, item by item.

    python tools/ingame/rules.py out/ingame/sample/p2c out/ingame/sample/ingame.jsonl

The second file is `train/portal2.py check` output for the same .p2c files.

Prints level-level agreement, then per rule how many of its flags the game also flagged
(precision), and the game's flags nothing predicted (misses) by item name, with examples.
"""
import argparse, collections, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("p2c_dir")
    ap.add_argument("game")
    ap.add_argument("--glados", default=str(ROOT / "target" / "release" / "glados.exe"))
    ap.add_argument("--examples", type=int, default=6)
    a = ap.parse_args()
    game = {}
    for l in open(a.game, encoding="utf-8"):
        j = json.loads(l)
        if j["verdict"] in ("valid", "invalid"):
            game[Path(j["file"]).stem] = {(i["name"], tuple(i["pos"])) for i in j["invalid_items"]}
    out = subprocess.run([a.glados, "editor", a.p2c_dir], capture_output=True, text=True, check=True)
    pred = {}
    for l in out.stdout.splitlines():
        j = json.loads(l)
        pred[Path(j["file"]).stem] = {(f["name"], tuple(f["pos"])): f["rules"] for f in j["flags"]}

    levels = collections.Counter()
    rule_hit = collections.Counter()
    rule_n = collections.Counter()
    rule_fp_ex = collections.defaultdict(list)
    rule_lv = collections.Counter()
    rule_lv_bad = collections.Counter()
    miss = collections.Counter()
    miss_ex = collections.defaultdict(list)
    tp = fp = fn = 0
    for stem, g in sorted(game.items()):
        p = pred.get(stem, {})
        levels[(bool(p), bool(g))] += 1
        for r in {r.rstrip("^") for rules in p.values() for r in rules}:
            rule_lv[r] += 1
            rule_lv_bad[r] += bool(g)
        for key, rules in p.items():
            hit = key in g
            tp += hit
            fp += not hit
            for r in rules:
                rule_n[r] += 1
                rule_hit[r] += hit
                if not hit:
                    rule_fp_ex[r].append(f"{stem}:{key[0]}@{key[1]}")
        for key in g - p.keys():
            fn += 1
            miss[key[0]] += 1
            miss_ex[key[0]].append(f"{stem}@{key[1]}")
    n = sum(levels.values())
    agree = levels[(True, True)] + levels[(False, False)]
    print(f"{n} levels; level verdicts agree on {agree} ({agree / n:.1%}):")
    print(f"  both invalid {levels[(True, True)]}, both valid {levels[(False, False)]}, "
          f"we flag / game valid {levels[(True, False)]}, we pass / game invalid {levels[(False, True)]}")
    print(f"items: {tp} agreed, {fp} only ours, {fn} only the game's")
    print("\nrules (precision = share of its flags the game also flags):")
    for r, k in rule_n.most_common():
        print(f"  {r:20} {rule_hit[r]:5}/{k:<5} {rule_hit[r] / k:6.1%}   wrong e.g. {' '.join(rule_fp_ex[r][:a.examples])}")
    print("\nlevel precision (share of levels where the rule fires that the game rejects):")
    for r, k in sorted(rule_lv.items(), key=lambda kv: rule_lv_bad[kv[0]] / kv[1]):
        print(f"  {r:40} {rule_lv_bad[r]:5}/{k:<5} {rule_lv_bad[r] / k:6.1%}")
    print("\nmissed (game flags, nothing predicted):")
    for name, k in miss.most_common():
        print(f"  {name:14} {k:5}   e.g. {' '.join(miss_ex[name][:a.examples])}")


main()
