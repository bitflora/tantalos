"""Train a small GPT on `glados-data dataset` output.

    cargo run --release --bin glados-data -- dataset --count 200000 --out data/levels.jsonl
    python train/train.py --data data/levels.jsonl --out runs/first

--until 07:45 trains until a clock time instead of for --iters steps; --init starts from another
run's weights. Every --eval-every steps: validation loss, then --score-n sampled levels run through
`glados-score score` (parse rate, solvable rate). Checkpoints: <out>/last.pt and <out>/best.pt
(lowest validation loss). Log: <out>/log.jsonl. The run gets a row in train/runs.csv
(train/record.py), with --note as its notes.

--task solve trains the solver net instead: it reads a level and writes its plan (data/plans from
train/plans.py; sequences in train/data.py). Its eval replays greedy plans for --score-n
validation levels with `glados-score check` and logs the solved share. --init from a level model keeps
the embeddings of the tokens both know (level words), and drops its conditions.

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
import record  # noqa: E402
import sample  # noqa: E402
import solve_eval  # noqa: E402
from data import COND_TOKENS, Away, Vocab, load, load_solve  # noqa: E402
from model import GPT, Config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="JSONL files from `glados-data dataset`")
    ap.add_argument(
        "--task",
        choices=("levels", "solve"),
        default="levels",
        help="levels: write levels from conditions; solve: write a level's plan (data/plans, "
        "train/plans.py), the solver net",
    )
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
    ap.add_argument(
        "--score-n",
        type=int,
        default=32,
        help="levels sampled and scored per eval; with --task solve, validation levels whose "
        "greedy plans `glados-score check` replays (0: off)",
    )
    ap.add_argument(
        "--holdout",
        default=None,
        help="levels never trained on (run.py: data/ws5/test.jsonl, the held-out Workshop levels): "
        "each eval logs their loss as holdout_loss. The val split can't show overfitting to the "
        "Workshop part, whose x20 copies fall in both train and val",
    )
    ap.add_argument("--holdout-n", type=int, default=256,
                    help="how many of them, the same ones at every eval, under their full conditions")
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
        "the rejected items when asked for a solvable level (unlikelihood). Experimental: at "
        "--away-weight 5 it made the model stop writing real coordinates within 500 steps",
    )
    ap.add_argument(
        "--away-weight",
        type=float,
        default=5.0,
        help="how many ordinary training targets each pushed-away token counts as",
    )
    ap.add_argument("--away-batch", type=int, default=8, help="rejected levels per step")
    ap.add_argument("--init", default=None, help="start from this checkpoint's weights (new optimizer and schedule)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--note", default=None, help="what this run is for, for train/runs.csv")
    args = ap.parse_args()

    start = time.time()
    deadline = until(args.until) if args.until else None
    if deadline:
        print(f"training until {datetime.datetime.fromtimestamp(deadline):%Y-%m-%d %H:%M}")
    keep_awake()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not (args.resume and (out / "run.json").exists()):
        record.write_meta(out, args.note, args.task)
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
    vocab = Vocab.build(vocab_file, args.task)
    t0 = time.time()
    solve = args.task == "solve"
    if solve:
        if args.away:
            sys.exit("--away is for --task levels")
        train, val, lengths = load_solve(args.data, vocab, args.block, seed=args.seed)
        print(f"data: {len(train)} train ({len(train.offsets) - 1} plans), {len(val)} val; vocab {len(vocab)}; "
              f"loaded in {time.time() - t0:.0f}s")
        print(f"  {lengths.report()}; {lengths.unknown} with tokens not in {vocab_file}, dropped")
        # The same validation levels at every eval, so their solved shares compare.
        solve_picks = random.Random(args.seed).sample(range(len(val)), min(args.score_n, len(val)))
    else:
        train, val, skipped = load(args.data, vocab, args.block, seed=args.seed)
        print(
            f"data: {len(train)} train, {len(val)} val, {skipped} too long for block {args.block}; "
            f"vocab {len(vocab)}; loaded in {time.time() - t0:.0f}s"
        )
    holdout = None
    if args.holdout and not solve:
        if Path(args.holdout).resolve() in {Path(p).resolve() for p in args.data}:
            sys.exit(f"--holdout {args.holdout} is also in --data: it wouldn't be held out")
        _, holdout, _ = load([args.holdout], vocab, args.block, val_share=1.0)
        holdout_picks = random.Random(args.seed).sample(range(len(holdout)), min(args.holdout_n, len(holdout)))
        print(f"holdout: {len(holdout_picks)} of the {len(holdout)} levels in {args.holdout}")

    away = Away(args.away, vocab, args.block) if args.away else None
    if away is not None:
        print(f"away: {len(away)} game-rejected levels with placements to push away from")
        if not len(away):
            away = None
    # Coordinates are numbers: the away loss compares a rejected one only with other numbers.
    numbers = [vocab.stoi[t] for t in vocab.itos if t.isdigit()]
    number_of = torch.full((len(vocab),), -1, dtype=torch.long)
    number_of[numbers] = torch.arange(len(numbers))

    cfg = Config(len(vocab), args.block, args.layers, args.heads, args.embd, args.dropout, qk_norm=True)
    init = torch.load(args.init, map_location=device, weights_only=False) if args.init else None
    if init:
        # The model's shape comes from the checkpoint; --layers/--heads/--embd are ignored. Its
        # vocabulary and block may grow: tokens it knows keep their embeddings (by name, so new
        # words and conditions can go anywhere), new ones start fresh, and so do new positions.
        lost = [t for t in init["vocab"] if t not in vocab.stoi]
        if solve:
            # A level model warm-starting the solver net: its conditions have no use here.
            lost = [t for t in lost if t not in COND_TOKENS]
        if lost:
            sys.exit(f"{args.init} knows tokens this data doesn't have: {lost[:8]}")
        if init["model_config"]["block_size"] > args.block:
            sys.exit(f"{args.init} has block {init['model_config']['block_size']}: pass --block at least that")
        cfg = Config(**{**init["model_config"], "vocab_size": len(vocab), "block_size": args.block})
    model = GPT(cfg).to(device)
    opt = model.optimizer(args.lr, args.weight_decay)
    step, best = 0, float("inf")
    if init:
        state, own = init["model"], model.state_dict()
        for k, v in state.items():
            if k not in ("tok.weight", "head.weight", "pos.weight"):
                own[k] = v
        row = {t: i for i, t in enumerate(init["vocab"])}
        kept = [(i, row[t]) for i, t in enumerate(vocab.itos) if t in row]
        own["tok.weight"][[i for i, _ in kept]] = state["tok.weight"][[j for _, j in kept]]
        own["head.weight"] = own["tok.weight"]
        n = state["pos.weight"].shape[0]
        own["pos.weight"][:n] = state["pos.weight"]
        model.load_state_dict(own)
        new = len(vocab) - len(kept)
        print(f"weights from {args.init}" + (f" ({new} new tokens, block {n} -> {args.block})"
                                             if new or n != args.block else ""))
    if args.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step, best = ck["step"], ck["best_val"]
        print(f"resumed at step {step}")
    if init or step:
        # Otherwise a run continuing another draws the very batches that run drew first. (The
        # validation split stays: it is seeded separately, by --seed alone.)
        random.seed(f"{args.seed} {args.init} {step}")
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
    # (Records, not len(train): a solve level can hold several plans.)
    per_batch = args.batch * (len(train.level_ids) / max(len(train.offsets) - 1, 1) + 1)

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

    @torch.no_grad()
    def holdout_loss():
        """Mean loss per target over the held-out picks, every condition given (drop 0)."""
        model.eval()
        total = count = 0
        for i in range(0, len(holdout_picks), args.batch):
            x, y, n = holdout.batch(holdout_picks[i : i + args.batch], 0.0, device)
            with amp:
                total += model(x, y)[1].item() * n
            count += n
        model.train()
        return total / max(count, 1)

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
            ax, ay = away.batch(args.away_batch, args.cond_drop, device)
            with amp:
                logits, _ = model(ax)
            mask = ay != -100
            nums = torch.tensor(numbers, device=logits.device)
            sub = logits[mask].float()[:, nums].softmax(-1)
            which = number_of.to(logits.device)[ay[mask]]
            keep = which >= 0
            p = sub[keep].gather(-1, which[keep].unsqueeze(-1)).squeeze(-1)
            pushed = -torch.log((1 - p).clamp(min=1e-6))
            away_loss = pushed.mean()
            # Each pushed token counts as --away-weight ordinary targets. (Averaged over a few
            # tokens instead, each weighed ~100 ordinary ones and the model stopped writing real
            # coordinates altogether.)
            scale = args.away_weight / max(targets * args.accum, 1)
            scaler.scale(scale * pushed.sum()).backward()
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
            if holdout is not None:
                entry["holdout_loss"] = holdout_loss()
            if args.score_n and solve:
                model.eval()
                levels = [" ".join(vocab.decode(val.level(i))) for i in solve_picks]
                prompts = [solve_eval.plan_prompt(vocab, lv, args.block) for lv in levels]
                plans = solve_eval.generate_plans(model, vocab, prompts, top_k=1)
                model.train()
                try:
                    items = [{"tokens": lv, "plan": p, "min_len": val.min_lens[i] if val.min_lens[i] >= 0 else None}
                             for lv, p, i in zip(levels, plans, solve_picks)]
                    entry["solve"] = solve_eval.summarize(solve_eval.check(items, args.glados))
                except (OSError, ValueError, subprocess.CalledProcessError) as e:
                    print(f"checking plans failed, carrying on: {e}")
            elif args.score_n:
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
            if entry.get("solve"):
                v = entry["solve"]
                extra += (f"  val levels: {v['solved']:.0%} solved (greedy), {v['optimal']:.0%} optimal, "
                          f"{v['parsed']:.0%} parse")
            if "gpu_gb" in entry:
                extra += f"  gpu {entry['gpu_gb']:.1f} GB"
            if "holdout_loss" in entry:
                extra = f"  holdout {entry['holdout_loss']:.4f}" + extra
            print(f"== step {step}: val {val_loss:.4f}{' (best)' if improved else ''}{extra}")
            t0, tokens = time.time(), 0
        if done:
            break
    log.close()
    record.safely(out)


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
