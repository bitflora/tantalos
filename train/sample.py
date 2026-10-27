"""Sample levels from a checkpoint and score them with `glados-score score`.

    python train/sample.py runs/first/best.pt --n 64 --verdict solvable --len 5-7
    python train/sample.py runs/first/best.pt --cond "<trips:2+> <fizzler:yes>" --prefix "size 8 8 6"

Writes the token lines and the scores next to each other (default: <run>/samples/), and the
solvable levels as .p2c with --p2c.
"""

import argparse
import json
import queue
import subprocess
import sys
import threading
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from data import ANY, BOS, EOS, SEP, GROUPS, Vocab, groups_in, slots  # noqa: E402
from model import GPT, Config  # noqa: E402
from seeding import resolve  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EXE = ".exe" if sys.platform == "win32" else ""
# score, check and route; the other glados-* binaries sit beside it (src/bin/).
DEFAULT_GLADOS = ROOT / "target" / "release" / f"glados-score{EXE}"


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


def generate(model, vocab, n, conds=(), prefix=(), temperature=1.0, top_k=None, batch=32, refill=False):
    """`n` token lines (prefix included). `refill` starts a new level in a batch row as soon as
    its last one ends (GPT.generate_refill): levels from the same distribution (greedy ones
    identical), 1.6 times as fast for v6-fresh-12h at batch 64 (2026-10-27), but other samples
    for a given seed than without it."""
    p = prompt(vocab, list(conds), list(prefix))
    if refill:
        outs = model.generate_refill(p, n, model.cfg.block_size, EOS, temperature, top_k, batch)
        return [" ".join(list(prefix) + vocab.decode(o)) for o in outs]
    lines = []
    for start in range(0, n, batch):
        k = min(batch, n - start)
        outs = model.generate([p] * k, model.cfg.block_size, EOS, temperature, top_k)
        lines.extend(" ".join(list(prefix) + vocab.decode(o)) for o in outs)
    return lines


def score(lines, glados=DEFAULT_GLADOS, budget=50_000, p2c_dir=None, editor_rules=True, p2c_all=False):
    """One result dict per line, from `glados-score score`. p2c_dir gets the solvable levels as
    .p2c (every parsed one with p2c_all); editor_rules=False is how Workshop levels are scored."""
    args = [str(Path(glados).resolve()), "score", "--budget", str(budget)]
    if p2c_dir:
        args += ["--p2c-all" if p2c_all else "--p2c", str(p2c_dir)]
    if not editor_rules:
        args += ["--editor-rules", "off"]
    # Empty lines would be skipped by glados and break the alignment.
    text = "\n".join(line if line.strip() else "<empty>" for line in lines) + "\n"
    out = subprocess.run(args, input=text, capture_output=True, text=True, check=True)
    return [json.loads(line) for line in out.stdout.splitlines()]


class Scorer:
    """`score` as one long-lived `glados-score score` process, so the CPU scores levels while the
    GPU samples the next ones: `send` a list of lines (returns at once), `take` that many results
    later, in order. It runs at below-normal priority: it still gets every idle core, but the
    sampling thread (which keeps the GPU fed) wins when they compete."""

    def __init__(self, glados=DEFAULT_GLADOS, budget=50_000):
        args = [str(Path(glados).resolve()), "score", "--budget", str(budget)]
        flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS if sys.platform == "win32" else 0
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                     encoding="utf-8", creationflags=flags)
        self.inbox, self.results = queue.Queue(), queue.Queue()
        # A thread each way, so neither pipe fills up while this one is busy sampling.
        self.writer = threading.Thread(target=self._write, daemon=True)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.writer.start()
        self.reader.start()

    def _write(self):
        try:
            for lines in iter(self.inbox.get, None):
                # Empty lines would be skipped by glados and break the alignment.
                self.proc.stdin.write("".join((l if l.strip() else "<empty>") + "\n" for l in lines))
                self.proc.stdin.flush()
        except OSError:
            pass  # glados stopped: take() and close() say so
        finally:
            try:
                self.proc.stdin.close()
            except OSError:
                pass

    def _read(self):
        for line in self.proc.stdout:
            self.results.put(json.loads(line))
        self.results.put(None)

    def send(self, lines):
        self.inbox.put(list(lines))

    def take(self, k):
        out = []
        for _ in range(k):
            r = self.results.get()
            if r is None:
                raise RuntimeError(f"glados-score score stopped (exit code {self.proc.wait()}) "
                                   f"after {len(out)} of {k} results")
            out.append(r)
        return out

    def close(self):
        self.inbox.put(None)
        self.writer.join()
        if self.proc.wait() != 0:
            raise subprocess.CalledProcessError(self.proc.returncode, self.proc.args)


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
