"""Train a small GPT on `glados dataset` output.

    cargo run --release -- dataset --count 200000 --out data/levels.jsonl
    python train/train.py --data data/levels.jsonl --out runs/first

Every --eval-every steps: validation loss, then --score-n sampled levels run through
`glados score` (parse rate, solvable rate). Checkpoints: <out>/last.pt and <out>/best.pt
(lowest validation loss). Log: <out>/log.jsonl.
"""

import argparse
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import sample  # noqa: E402
from data import Vocab, load  # noqa: E402
from model import GPT, Config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="JSONL files from `glados dataset`")
    ap.add_argument("--vocab", default=None, help="level vocab (default: vocab.txt beside the data)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--embd", type=int, default=384)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--iters", type=int, default=20_000)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument(
        "--cond-drop",
        type=float,
        default=1.0,
        help="each sequence hides each condition with a chance drawn uniformly from [0, this]",
    )
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--score-n", type=int, default=32, help="levels sampled and scored per eval (0: off)")
    ap.add_argument("--glados", default=str(sample.DEFAULT_GLADOS))
    ap.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        amp = torch.autocast("cuda", dtype=dtype)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        dtype, amp = torch.float32, nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)

    vocab_file = args.vocab or Path(args.data[0]).parent / "vocab.txt"
    vocab = Vocab.build(vocab_file)
    t0 = time.time()
    train, val, skipped = load(args.data, vocab, args.block, seed=args.seed)
    print(
        f"data: {len(train)} train, {len(val)} val, {skipped} too long for block {args.block}; "
        f"vocab {len(vocab)}; loaded in {time.time() - t0:.0f}s"
    )

    cfg = Config(len(vocab), args.block, args.layers, args.heads, args.embd, args.dropout)
    model = GPT(cfg).to(device)
    opt = model.optimizer(args.lr, args.weight_decay)
    step, best = 0, float("inf")
    if args.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step, best = ck["step"], ck["best_val"]
        print(f"resumed at step {step}")
    print(f"model: {model.n_params() / 1e6:.1f}M parameters on {device} ({dtype})")

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        frac = min(1.0, (s - args.warmup) / max(1, args.iters - args.warmup))
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * frac))

    def batch(levels, drop):
        idx = [random.randrange(len(levels)) for _ in range(args.batch)]
        return levels.batch(idx, drop, device)

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for _ in range(args.eval_batches):
            x, y = batch(val, args.cond_drop)
            with amp:
                losses.append(model(x, y)[1].item())
        model.train()
        return sum(losses) / len(losses)

    def checkpoint(path, val_loss):
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "model_config": cfg.__dict__,
                "vocab": vocab.itos,
                "step": step,
                "best_val": best,
                "val_loss": val_loss,
                "args": vars(args),
            },
            path,
        )

    log = open(out / "log.jsonl", "a")
    model.train()
    t0, tokens = time.time(), 0
    while step < args.iters:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        for _ in range(args.accum):
            x, y = batch(train, args.cond_drop)
            with amp:
                _, loss = model(x, y)
            scaler.scale(loss / args.accum).backward()
            tokens += int((y != -100).sum())
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1

        if step % 50 == 0:
            dt = time.time() - t0
            print(f"step {step:>6}  loss {loss.item():.4f}  lr {lr_at(step):.2e}  {tokens / dt:,.0f} tok/s")
            t0, tokens = time.time(), 0

        if step % args.eval_every == 0 or step == args.iters:
            val_loss = evaluate()
            entry = {"step": step, "train_loss": loss.item(), "val_loss": val_loss}
            if args.score_n:
                model.eval()
                lines = sample.generate(model, vocab, args.score_n, ["<verdict:solvable>"])
                model.train()
                entry["samples"] = sample.summarize(sample.score(lines, args.glados))
            improved = val_loss < best
            best = min(best, val_loss)
            checkpoint(out / "last.pt", val_loss)
            if improved:
                checkpoint(out / "best.pt", val_loss)
            log.write(json.dumps(entry) + "\n")
            log.flush()
            s = entry.get("samples")
            extra = (
                f"  samples: {s['parsed']:.0%} parse, {s['solvable']:.0%} solvable, "
                f"mean plan {s['mean_min_len']:.1f}"
                if s
                else ""
            )
            print(f"== step {step}: val {val_loss:.4f}{' (best)' if improved else ''}{extra}")
            t0, tokens = time.time(), 0


if __name__ == "__main__":
    main()
