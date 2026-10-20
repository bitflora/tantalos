"""Sample levels from a checkpoint and score them with `glados score`.

    python train/sample.py runs/first/best.pt --n 64 --verdict solvable --len 5-7
    python train/sample.py runs/first/best.pt --cond "<trips:2+> <fizzler:yes>" --prefix "size 8 8 6"

Writes the token lines and the scores next to each other (default: <run>/samples/), and the
solvable levels as .p2c with --p2c.
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from data import ANY, BOS, EOS, SEP, GROUPS, Vocab, groups_in, slots  # noqa: E402
from model import GPT, Config  # noqa: E402
from seeding import resolve  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GLADOS = ROOT / "target" / "release" / ("glados.exe" if sys.platform == "win32" else "glados")


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = GPT(Config(**ck["model_config"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, Vocab(ck["vocab"]), ck


def prompt(vocab, conds, prefix):
    # Checkpoints from before condition slots have no `?` tokens and take the conditions as is;
    # later ones get a slot for each group they were trained with.
    cond = slots(conds, groups_in(vocab)) if ANY["verdict"] in vocab.stoi else list(conds)
    return [BOS] + vocab.encode(cond) + [SEP] + vocab.encode(prefix)


def generate(model, vocab, n, conds=(), prefix=(), temperature=1.0, top_k=None, batch=32):
    """`n` token lines (prefix included)."""
    p = prompt(vocab, list(conds), list(prefix))
    lines = []
    for start in range(0, n, batch):
        k = min(batch, n - start)
        outs = model.generate([p] * k, model.cfg.block_size, EOS, temperature, top_k)
        lines.extend(" ".join(list(prefix) + vocab.decode(o)) for o in outs)
    return lines


def score(lines, glados=DEFAULT_GLADOS, budget=50_000, p2c_dir=None):
    """One result dict per line, from `glados score`."""
    args = [str(Path(glados).resolve()), "score", "--budget", str(budget)]
    if p2c_dir:
        args += ["--p2c", str(p2c_dir)]
    # Empty lines would be skipped by glados and break the alignment.
    text = "\n".join(line if line.strip() else "<empty>" for line in lines) + "\n"
    out = subprocess.run(args, input=text, capture_output=True, text=True, check=True)
    return [json.loads(line) for line in out.stdout.splitlines()]


def summarize(results):
    n = max(len(results), 1)
    verdicts = Counter(r["verdict"] for r in results)
    solvable = [r for r in results if r["verdict"] == "solvable"]
    return {
        "n": len(results),
        "parsed": 1 - verdicts["unparsed"] / n,
        "canonical": sum(r.get("canonical", False) for r in results) / n,
        "solvable": len(solvable) / n,
        "verdicts": dict(verdicts),
        "mean_min_len": sum(r["min_len"] for r in solvable) / max(len(solvable), 1),
        "portal_trips": sum(r.get("metrics", {}).get("portal_trips", 0) for r in solvable)
        / max(len(solvable), 1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--cond", default="", help="condition tokens, space separated")
    for group, tokens in GROUPS.items():
        values = [t[len(group) + 2 : -1] for t in tokens]
        ap.add_argument(f"--{group}", choices=values, help=f"shorthand for <{group}:...>")
    ap.add_argument("--prefix", default="", help="level tokens to start from, e.g. 'size 8 8 6'")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--out", default=None, help="directory (default <checkpoint dir>/samples)")
    ap.add_argument("--p2c", action="store_true", help="also write solvable levels as .p2c")
    ap.add_argument("--glados", default=str(DEFAULT_GLADOS))
    ap.add_argument("--budget", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=None,
                    help="default: a fresh, unpredictable seed (printed; pass it back to repeat)")
    args = ap.parse_args()
    args.seed = resolve(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model, vocab, _ = load_checkpoint(args.checkpoint, device)
    conds = args.cond.split()
    for group in GROUPS:
        value = getattr(args, group)
        if value is not None:
            conds.append(f"<{group}:{value}>")
    unknown = [t for t in conds + args.prefix.split() if t not in vocab.stoi]
    if unknown:
        sys.exit(f"unknown tokens: {unknown}")

    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "samples"
    out.mkdir(parents=True, exist_ok=True)
    lines = generate(model, vocab, args.n, conds, args.prefix.split(), args.temperature, args.top_k)
    (out / "levels.txt").write_text("\n".join(lines) + "\n")
    results = score(lines, args.glados, args.budget, out / "p2c" if args.p2c else None)
    with open(out / "scores.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in results)
    print(f"conditions: {' '.join(conds) or '(none)'}")
    print(json.dumps(summarize(results), indent=2))
    print(f"wrote {out / 'levels.txt'} and {out / 'scores.jsonl'}")


if __name__ == "__main__":
    main()
