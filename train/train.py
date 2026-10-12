"""Train a small GPT on `glados dataset` output.

    cargo run --release -- dataset --count 200000 --out data/levels.jsonl
    python train/train.py --data data/levels.jsonl --out runs/first

--until 07:45 trains until a clock time instead of for --iters steps; --init starts from another
run's weights. Every --eval-every steps: validation loss, then --score-n sampled levels run through
`glados score` (parse rate, solvable rate). Checkpoints: <out>/last.pt and <out>/best.pt
(lowest validation loss). Log: <out>/log.jsonl.

Training batches hold levels of similar length (--bucket), so they pad little. On the GPU the
process keeps to --gpu-mem: on Windows, outgrowing the free GPU memory doesn't fail, the driver
pages to system memory instead and everything slows down.
"""

import argparse
import datetime
import json
import math
import os
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

# Hand back cached GPU memory nobody is using once 80% of the --gpu-mem cap is taken, rather
# than all at once when it runs out. (Read when CUDA starts.)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import sample  # noqa: E402
from data import Away, Vocab, load  # noqa: E402
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
    ap.add_argument(
        "--bucket",
        type=int,
        default=32,
        help="draw this many batches' worth of random levels at a time and batch them by length, "
        "which cuts padding (0: each batch random)",
    )
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--iters", type=int, default=20_000)
    ap.add_argument(
        "--until",
        default=None,
        help="stop at this local time ('07:45' or '2026-10-10 07:45') instead of after --iters; "
        "the learning-rate schedule then runs over the time left",
    )
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
    ap.add_argument(
        "--gpu-mem",
        type=float,
        default=None,
        help="GB of GPU memory to keep to (default: what is free at the start less 0.5, "
        "but at least 2.5; 0: no limit)",
    )
    ap.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    ap.add_argument(
        "--away",
        nargs="*",
        default=[],
        help="game-rejected levels (data/ingame/invalid.jsonl): push down the tokens that placed "
        "the rejected items when asked for a solvable level (unlikelihood)",
    )
    ap.add_argument("--away-weight", type=float, default=0.5, help="weight of the --away loss")
    ap.add_argument("--away-batch", type=int, default=8, help="rejected levels per step")
    ap.add_argument("--init", default=None, help="start from this checkpoint's weights (new optimizer and schedule)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    start = time.time()
    deadline = until(args.until) if args.until else None
    if deadline:
        print(f"training until {datetime.datetime.fromtimestamp(deadline):%Y-%m-%d %H:%M}")
    keep_awake()
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
        free, total = torch.cuda.mem_get_info()
        cap = args.gpu_mem * 2**30 if args.gpu_mem is not None else max(free - 2**29, 2.5 * 2**30)
        if cap > 0:
            torch.cuda.set_per_process_memory_fraction(min(cap / total, 1.0))
            print(f"GPU memory: keeping to {cap / 2**30:.1f} of {total / 2**30:.1f} GB ({free / 2**30:.1f} free)")
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

    away = Away(args.away, vocab, args.block) if args.away else None
    if away is not None:
        print(f"away: {len(away)} game-rejected levels with placements to push away from")
        if not len(away):
            away = None

    cfg = Config(len(vocab), args.block, args.layers, args.heads, args.embd, args.dropout)
    init = torch.load(args.init, map_location=device, weights_only=False) if args.init else None
    if init:
        # The model's shape comes from the checkpoint; --block/--layers/--heads/--embd are ignored.
        if init["vocab"] != vocab.itos:
            sys.exit(f"{args.init} was trained with a different vocabulary")
        if init["model_config"]["block_size"] != args.block:
            sys.exit(f"{args.init} has block {init['model_config']['block_size']}: pass --block to match")
        cfg = Config(**init["model_config"])
    model = GPT(cfg).to(device)
    opt = model.optimizer(args.lr, args.weight_decay)
    step, best = 0, float("inf")
    if init:
        model.load_state_dict(init["model"])
        print(f"weights from {args.init}")
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
        if deadline:
            frac = min(1.0, (time.time() - start) / (deadline - start))
        else:
            frac = min(1.0, (s - args.warmup) / max(1, args.iters - args.warmup))
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * frac))

    def batch(levels, drop):
        idx = [random.randrange(len(levels)) for _ in range(args.batch)]
        return levels.batch(idx, drop, device)

    # A batch's loss is the mean over its targets. Sorted batches of long levels hold far more
    # targets than batches of short ones, so scale each by its share of the average count:
    # every target then weighs the same, as in random batches (whose counts vary little).
    per_batch = args.batch * (len(train.level_ids) / max(len(train), 1) + 1)

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for _ in range(args.eval_batches):
            x, y, _ = batch(val, args.cond_drop)
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
    train_batches = batches(train, args.batch, args.bucket, args.cond_drop, device)
    t0, tokens = time.time(), 0
    def finished():
        return time.time() >= deadline if deadline else step >= args.iters

    while not finished():
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        for _ in range(args.accum):
            x, y, targets = next(train_batches)
            with amp:
                _, loss = model(x, y)
            weight = targets / per_batch if args.bucket > 1 else 1.0
            scaler.scale(loss * weight / args.accum).backward()
            tokens += targets
        if away is not None:
            ax, ay = away.batch(args.away_batch, device)
            with amp:
                logits, _ = model(ax)
            mask = ay != -100
            p = logits.float().softmax(-1).gather(-1, ay.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            away_loss = -torch.log((1 - p[mask]).clamp(min=1e-6)).mean()
            scaler.scale(args.away_weight * away_loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1

        if step % 50 == 0:
            dt = time.time() - t0
            extra = f"  away {away_loss.item():.4f}" if away is not None else ""
            print(f"step {step:>6}  loss {loss.item():.4f}{extra}  lr {lr_at(step):.2e}  {tokens / dt:,.0f} tok/s")
            t0, tokens = time.time(), 0

        done = finished()
        if step % args.eval_every == 0 or done:
            val_loss = evaluate()
            entry = {"step": step, "train_loss": loss.item(), "val_loss": val_loss}
            if args.score_n:
                model.eval()
                lines = sample.generate(model, vocab, args.score_n, ["<verdict:solvable>"])
                model.train()
                try:
                    entry["samples"] = sample.summarize(sample.score(lines, args.glados))
                except (OSError, ValueError, subprocess.CalledProcessError) as e:
                    # Not worth the run: one died here when Windows couldn't start glados.
                    print(f"scoring samples failed, carrying on: {e}")
            if device == "cuda":
                # Hand back what sampling took, and note the most held since the last eval.
                torch.cuda.empty_cache()
                entry["gpu_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
                torch.cuda.reset_peak_memory_stats()
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
            if "gpu_gb" in entry:
                extra += f"  gpu {entry['gpu_gb']:.1f} GB"
            print(f"== step {step}: val {val_loss:.4f}{' (best)' if improved else ''}{extra}")
            t0, tokens = time.time(), 0
        if done:
            break


def batches(levels, size, bucket, drop, device):
    """Endless training batches: `bucket` batches' worth of random levels at a time, sorted by
    length and dealt out as batches in random order, so a batch's levels are of similar length
    and pad little. (In v3 data, targets fill 85% of a batch's positions this way, and 52% in
    random batches.)"""
    lengths = levels.lengths()
    n = size * max(bucket, 1)
    while True:
        idx = [random.randrange(len(levels)) for _ in range(n)]
        if bucket > 1:
            idx.sort(key=lengths.__getitem__)
        chunks = [idx[i : i + size] for i in range(0, n, size)]
        random.shuffle(chunks)
        for chunk in chunks:
            yield levels.batch(chunk, drop, device)


def until(text):
    """Unix time of a local 'HH:MM' (the next one) or 'YYYY-MM-DD HH:MM'."""
    now = datetime.datetime.now()
    if len(text) <= 5:
        t = datetime.datetime.combine(now.date(), datetime.time.fromisoformat(text))
        if t <= now:
            t += datetime.timedelta(days=1)
    else:
        t = datetime.datetime.fromisoformat(text)
    if t <= now:
        sys.exit(f"--until {text} is in the past")
    return t.timestamp()


def keep_awake():
    """Ask Windows not to sleep while this process runs (released when it exits)."""
    if sys.platform == "win32":
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


if __name__ == "__main__":
    main()
