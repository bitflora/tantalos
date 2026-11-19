"""A small decoder-only transformer (GPT-2 style, pre-norm), trained from scratch.

Level tokens are a few hundred words of a ~170-word language, so a ~10M parameter model is
plenty to start with and fits a 4 GB GPU with room to spare.

Other architectures (docs/architecture-plan-2026-11-15.md) sit behind `Config.arch` and
`build_model`; checkpoints made before it have no `arch` and load as "gpt". Every one is a
`Decoder`: it shares the sampling code (generate, generate_refill, generate_many), which only
needs `forward(idx, targets, caches, start, positions, mask)`, `self.blocks` and `self.cfg`.
"""

import math
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    # LayerNorm on each head's queries and keys before attention, against the late loss spikes
    # of long runs (2026-10-20). Off for checkpoints made before it, so they load unchanged.
    qk_norm: bool = False
    # Which architecture (ARCHS below): "gpt" is the original; "modern" is RMSNorm, SwiGLU,
    # RoPE, no biases, no dropout (arm 1 of the 2026-11-15 bake-off).
    arch: str = "gpt"
    # The vocabulary (id -> token), for architectures that read the tokens themselves ("spatial").
    itos: tuple = ()


def attend(q, k, v, cache, mask, dropout_p):
    """Attention for one layer (q, k, v: batch x heads x tokens x head size), keeping keys and
    values in `cache` for generation (see Block.forward). Returns batch x heads x tokens x size."""
    b, h, t, d = q.shape
    if cache is not None and "at" in cache:
        # generate_refill: one new token per row, each written at its own row's position
        # into a cache allocated beforehand; attention only up to cache["len"].
        rows = torch.arange(b, device=q.device)
        cache["k"][rows, :, cache["at"]] = k[:, :, 0]
        cache["v"][rows, :, cache["at"]] = v[:, :, 0]
        k, v = cache["k"][:, :, : cache["len"]], cache["v"][:, :, : cache["len"]]
    elif cache is not None:
        n = cache.get("n", 0)
        if n == 0:
            shape = (b, h, cache["size"], d)
            cache["k"], cache["v"] = k.new_empty(shape), v.new_empty(shape)
        cache["k"][:, :, n : n + t] = k
        cache["v"][:, :, n : n + t] = v
        cache["n"] = n = n + t
        k, v = cache["k"][:, :, :n], cache["v"][:, :, :n]
    # New tokens after a cached prefix come one at a time and may see everything before.
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        is_causal=mask is None and q.size(2) == k.size(2),
        dropout_p=dropout_p,
    )


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )
        self.n_head = cfg.n_head
        if cfg.qk_norm:
            self.q_norm = nn.LayerNorm(cfg.n_embd // cfg.n_head)
            self.k_norm = nn.LayerNorm(cfg.n_embd // cfg.n_head)
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cache=None, mask=None):
        """`cache` (a dict, for generation) keeps this layer's keys and values between calls:
        room for `cache["size"]` positions, allocated on the first call and filled in order.
        (Growing them with torch.cat reallocated both every token; during training on the
        4 GB laptop GPU that ran into system memory, and 64 samples took two minutes.)
        `mask` (bool, batch x 1 x new x all, True = may look): for padded batches; without
        one, attention is causal."""
        b, t, c = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(c, dim=2)
        q, k, v = (z.view(b, t, self.n_head, c // self.n_head).transpose(1, 2) for z in (q, k, v))
        if hasattr(self, "q_norm"):
            q, k = self.q_norm(q), self.k_norm(k)
        y = attend(q, k, v, cache, mask, self.dropout if self.training else 0.0)
        x = x + self.resid_drop(self.proj(y.transpose(1, 2).reshape(b, t, c)))
        return x + self.mlp(self.ln2(x))


class Decoder(nn.Module):
    """What every architecture shares: initialisation, the optimizer and the sampling code.
    A subclass builds `self.cfg`, `self.blocks` (each taking `cache`; see `attend`) and
    `forward(idx, targets=None, caches=None, start=0, positions=None, mask=None)`."""

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def _init_weights(self, residual=("proj.weight", "mlp.2.weight")):
        """Normal(0.02) everywhere, the projections into the residual stream scaled down by
        depth (GPT-2)."""
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith(residual):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * self.cfg.n_layer))

    def n_params(self):
        pos = getattr(self, "pos", None)
        return sum(p.numel() for p in self.parameters()) - (pos.weight.numel() if pos is not None else 0)

    def optimizer(self, lr, weight_decay, betas=(0.9, 0.95)):
        """AdamW; no decay on biases, norms and embeddings' 1-D params."""
        decay = [p for p in self.parameters() if p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = next(self.parameters()).is_cuda
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)

    # Architectures that read the level's structure (Spatial) keep a parser per sampled row:
    # `_rows(b)` gives them (None for the others), `_step` pairs each step's new tokens with
    # their features. `featurize` (CPU ids -> features) is None for the others.
    featurize = None

    def _rows(self, b):
        return None

    def _generation_state(self, caches):
        return None

    def _restore_generation_state(self, caches, row, state):
        pass

    def _step(self, nxt, tracked, rows=None):
        """The input for one new token per row (nxt: rows x 1): the ids, with their features
        if `tracked` (only `rows` move on)."""
        if not tracked:
            return nxt
        feats = tracked.feed(nxt[:, 0].tolist(), rows)
        return nxt, torch.tensor(feats, dtype=torch.long, device=nxt.device)[:, None, :]

    @torch.no_grad()
    def generate(self, prompts, max_new, stop, temperature=1.0, top_k=None):
        """Continue each prompt (list of id lists) until `stop` or `max_new` tokens.
        Returns the new ids per prompt, without `stop`. On a GPU it runs in bfloat16, as
        training does."""
        device = next(self.parameters()).device
        amp = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and torch.cuda.is_bf16_supported()
            else nullcontext()
        )
        if len({len(p) for p in prompts}) > 1:
            with amp:
                return self._generate_padded(prompts, max_new, stop, temperature, top_k, device)
        # Prompts of equal length run as one batch, so positions line up without padding.
        outs = [None] * len(prompts)
        by_len = {}
        for i, p in enumerate(prompts):
            by_len.setdefault(len(p), []).append(i)
        for n, rows in by_len.items():
            steps = min(max_new, self.cfg.block_size - n)
            caches = [{"size": n + steps} for _ in self.blocks]
            tracked = self._rows(len(rows))
            for r, i in enumerate(rows):
                if tracked:
                    tracked.start(r, prompts[i])
            with amp:
                logits, _ = self(torch.tensor([prompts[i] for i in rows], device=device), caches=caches)
            done = torch.zeros(len(rows), dtype=torch.bool, device=device)
            new = []
            for step in range(steps):
                logits = logits[:, -1, :].float()
                if top_k == 1:  # greedy: argmax, so tied logits don't get a random pick
                    nxt = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / max(temperature, 1e-5)
                    if top_k:
                        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                        logits[logits < kth] = -float("inf")
                    nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
                nxt[done] = stop
                done |= nxt.squeeze(1) == stop
                new.append(nxt)
                if done.all() or step == steps - 1:
                    break
                with amp:
                    logits, _ = self(self._step(nxt, tracked), caches=caches, start=n + step)
            gen = torch.cat(new, dim=1).tolist() if new else [[] for _ in rows]
            for r, i in enumerate(rows):
                seq = gen[r]
                outs[i] = seq[: seq.index(stop)] if stop in seq else seq
        return outs

    @torch.no_grad()
    def generate_refill(self, prompt, n, max_new, stop, temperature=1.0, top_k=None, batch=64):
        """`n` continuations of one prompt, like `generate([prompt] * n, ...)` run `batch` rows
        at a time, but a row that reaches `stop` starts the next continuation at once instead of
        waiting for the longest one in its batch (sampling is latency-bound: a batch of 64
        levels spent most of its steps on a few long ones). The prompt runs once; its keys and
        values stay at the start of every row's cache. Returns the new ids in start order."""
        device = next(self.parameters()).device
        amp = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and torch.cuda.is_bf16_supported()
            else nullcontext()
        )
        p = len(prompt)
        size = min(self.cfg.block_size, p + max_new)
        b = min(batch, n)
        with amp:
            first = [{"size": p} for _ in self.blocks]
            prompt_logits, _ = self(torch.tensor([prompt], device=device), caches=first)
        prompt_logits = prompt_logits[0, -1].float()
        tracked = self._rows(b)
        for r in range(b):
            if tracked:
                tracked.start(r, prompt)
        caches = []
        for c in first:
            shape = (b, c["k"].shape[1], size, c["k"].shape[3])
            # Zeros, not new_empty: a masked-out NaN still poisons attention (0 * NaN). The
            # dtypes differ under QK-norm.
            k, v = c["k"].new_zeros(shape), c["v"].new_zeros(shape)
            k[:, :, :p], v[:, :, :p] = c["k"], c["v"]
            caches.append({"k": k, "v": v})
        logits = prompt_logits.expand(b, -1).clone()
        slots = list(range(b))  # the continuation each row is writing (None: no more to start)
        outs = [[] for _ in range(n)]
        started = b
        lengths = [p] * b  # on the CPU too, to cut the attention at the longest row
        while any(s is not None for s in slots):
            if top_k == 1:
                nxt = logits.argmax(dim=-1)
            else:
                logits = logits / max(temperature, 1e-5)
                if top_k:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                    logits[logits < kth] = -float("inf")
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(1)
            fresh = set()
            for r, t in enumerate(nxt.tolist()):
                s = slots[r]
                if s is None:
                    continue
                if t != stop:
                    outs[s].append(t)
                    lengths[r] += 1
                    if len(outs[s]) < max_new and lengths[r] < size:
                        continue
                # Finished: the row takes the next continuation, from the prompt's logits.
                slots[r] = started if started < n else None
                started += 1
                lengths[r] = p
                fresh.add(r)
            if not any(s is not None for s in slots):
                break
            # Where each row's token goes. A fresh row feeds one at its prompt's end; it's
            # overwritten by its first real one, and its logits are replaced by the prompt's.
            at =torch.tensor([lengths[r] - 1 if r not in fresh else p for r in range(b)], device=device)
            longest = max(lengths) + 1
            for c in caches:
                c["at"], c["len"] = at, longest
            mask = (torch.arange(longest, device=device)[None, :] <= at[:, None])[:, None, None, :]
            inp = self._step(nxt[:, None], tracked)
            for r in fresh:  # its next token is its first, after the prompt
                if tracked:
                    tracked.start(r, prompt)
            with amp:
                logits, _ = self(inp, caches=caches, positions=at[:, None], mask=mask)
            logits = logits[:, -1, :].float()
            if fresh:
                logits[sorted(fresh)] = prompt_logits
        return outs

    @torch.no_grad()
    def generate_many(self, prompts, max_new, stop, temperature=1.0, top_k=None, batch=64, constrain=None,
                      logprob=False):
        """`generate` for many prompts, `batch` rows at a time, started in the order given: a row
        that finishes starts the next prompt at once, as in `generate_refill`, instead of waiting
        for the longest row of its batch (Goblin's constrained plans: half run to `max_new`, the
        rest end within a few dozen tokens). Each prompt runs alone through the net once and its
        keys and values are copied into its row; consecutive copies of one prompt (k samples of
        a level) share that run. Returns the new ids per prompt, without `stop`.

        `constrain`, if given, is asked before every token: `constrain(idx, seqs)` with the
        prompt index of each row still writing and its new ids so far returns a bool mask
        (rows x vocab, on the CPU) of the tokens each may write next; `constrain.finish(i, seq)`
        is told when prompt i ends and returns its ids to keep (goblin_eval.LegalSteps).

        With `logprob`, returns (ids, scores): each prompt's total log probability of the tokens
        it wrote, under the distribution they were drawn from (constrained if `constrain`, before
        `temperature` and `top_k`), not counting `stop`. Tokens a `constrain.finish` drops stay
        in the score."""
        device = next(self.parameters()).device
        amp = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and torch.cuda.is_bf16_supported()
            else nullcontext()
        )
        n = len(prompts)
        if n == 0:
            return ([], []) if logprob else []
        block = self.cfg.block_size
        size = min(block, max(len(p) for p in prompts) + max_new)
        b = min(batch, n)
        caches = None
        memo = {}  # the last prompt run: its ids, keys and values per layer, last logits

        def run(p):
            if memo.get("ids") != p:
                first = [{"size": len(p)} for _ in self.blocks]
                with amp:
                    lg, _ = self(torch.tensor([p], device=device), caches=first)
                memo.update(ids=p, kv=[(c["k"], c["v"]) for c in first], logits=lg[0, -1].float(),
                            state=self._generation_state(first))
            return memo["kv"], memo["logits"], memo["state"]

        slots = [None] * b  # the prompt each row is writing (None: no more to start)
        outs = [None] * n
        scores = [0.0] * n
        plen = [0] * b
        logits = torch.zeros((b, self.cfg.vocab_size), device=device)
        started = 0
        tracked = self._rows(b)

        def start(r):
            """Row r takes the next prompt, if any: its keys, values and logits."""
            nonlocal started, caches
            if started == n:
                slots[r] = None
                return
            i, started = started, started + 1
            kv, lg, state = run(prompts[i])
            if caches is None:
                # Zeros, not new_empty: a masked-out NaN still poisons attention (0 * NaN). The
                # dtypes differ under QK-norm.
                caches = [{"k": k.new_zeros((b, k.shape[1], size, k.shape[3])),
                           "v": v.new_zeros((b, v.shape[1], size, v.shape[3]))} for k, v in kv]
            p = len(prompts[i])
            for c, (k, v) in zip(caches, kv):
                c["k"][r, :, :p], c["v"][r, :, :p] = k[0], v[0]
            self._restore_generation_state(caches, r, state)
            logits[r] = lg
            slots[r], outs[i], plen[r] = i, [], p
            if tracked:
                tracked.start(r, prompts[i])

        def ask():
            """The tokens each row may write next (CPU), or None if unconstrained."""
            rows = [r for r in range(b) if slots[r] is not None]
            if constrain is None or not rows:
                return None
            allowed = torch.ones((b, self.cfg.vocab_size), dtype=torch.bool)
            allowed[rows] = constrain([slots[r] for r in rows], [outs[slots[r]] for r in rows])
            return allowed

        for r in range(b):
            start(r)
        allowed = ask()
        pool = ThreadPoolExecutor(1)
        while True:
            rows = [r for r in range(b) if slots[r] is not None]
            if not rows:
                break
            lg = logits
            if allowed is not None:
                lg = lg.masked_fill(~allowed.to(device), -float("inf"))
            lp = F.log_softmax(lg.float(), dim=-1) if logprob else None
            if top_k == 1:
                nxt = lg.argmax(dim=-1)
            else:
                lg = lg / max(temperature, 1e-5)
                if top_k:
                    kth = torch.topk(lg, min(top_k, lg.size(-1))).values[:, -1:]
                    lg[lg < kth] = -float("inf")
                nxt = torch.multinomial(F.softmax(lg, dim=-1), 1).squeeze(1)
            # Where each row's token goes: its own position, for a row going on. A row that
            # starts a prompt feeds one at that prompt's end, overwritten by its first real one;
            # its logits stay the prompt's. An idle row feeds one at 0.
            at = [0] * b
            going = [False] * b
            toks = nxt.tolist()
            chosen = lp.gather(1, nxt[:, None]).squeeze(1).tolist() if lp is not None else None
            for r in rows:
                i, t = slots[r], toks[r]
                if t != stop:
                    if chosen is not None:
                        scores[i] += chosen[r]
                    outs[i].append(t)
                    if len(outs[i]) < max_new and plen[r] + len(outs[i]) < block:
                        at[r], going[r] = plen[r] + len(outs[i]) - 1, True
                        continue
                if constrain is not None:
                    outs[i] = constrain.finish(i, outs[i])
                start(r)
                if slots[r] is not None:
                    at[r] = plen[r]
            # The next step's constraint, worked out (mostly waiting on env) while this thread
            # runs the net.
            asked = pool.submit(ask) if constrain is not None else None
            if any(going):
                longest = max(at) + 1
                at = torch.tensor(at, device=device)
                for c in caches:
                    c["at"], c["len"] = at, longest
                mask = (torch.arange(longest, device=device)[None, :] <= at[:, None])[:, None, None, :]
                inp = self._step(nxt[:, None], tracked, {r for r in range(b) if going[r]})
                with amp:
                    new, _ = self(inp, caches=caches, positions=at[:, None], mask=mask)
                keep = torch.tensor(going, device=device)[:, None]
                logits = torch.where(keep, new[:, -1, :].float(), logits)
            allowed = asked.result() if asked is not None else None
        pool.shutdown()
        return (outs, scores) if logprob else outs

    def _generate_padded(self, prompts, max_new, stop, temperature, top_k, device):
        """`generate` for prompts of different lengths, as one batch: each is padded on the left
        to the longest, its pads hidden from attention, and its positions start at 0 at its
        first real token, so each row comes out as it would alone. A row stops where it would
        alone too: after `max_new` tokens or at the end of the block."""
        b = len(prompts)
        lens = torch.tensor([len(p) for p in prompts], device=device)
        n = int(lens.max())
        pads = n - lens
        steps = min(max_new, self.cfg.block_size - int(lens.min()))
        ids = torch.zeros((b, n), dtype=torch.long, device=device)
        for r, p in enumerate(prompts):
            ids[r, n - len(p):] = torch.tensor(p, device=device)
        size = n + steps
        caches = [{"size": size} for _ in self.blocks]
        # Which cache slots hold real tokens: all but each row's pads.
        real = torch.arange(size, device=device)[None, :] >= pads[:, None]
        pos = (torch.arange(n, device=device)[None, :] - pads[:, None]).clamp(min=0)
        causal = torch.ones((n, n), dtype=torch.bool, device=device).tril()
        # A pad looks only at itself, so no row of the softmax is empty.
        mask = (causal[None] & real[:, None, :n]) | torch.eye(n, dtype=torch.bool, device=device)[None]
        tracked = self._rows(b)
        inp = ids
        if tracked:
            for r, p in enumerate(prompts):
                tracked.start(r, p)
            inp = (ids, self.featurize(ids.cpu()).to(device))
        logits, _ = self(inp, caches=caches, positions=pos, mask=mask[:, None])
        done = torch.zeros(b, dtype=torch.bool, device=device)
        # A row's last step: where it reaches max_new, or the end of the block.
        last = torch.clamp(self.cfg.block_size - lens, max=max_new) - 1
        new = []
        for step in range(steps):
            logits = logits[:, -1, :].float()
            if top_k == 1:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-5)
                if top_k:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                    logits[logits < kth] = -float("inf")
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            nxt[done] = stop
            done |= nxt.squeeze(1) == stop
            new.append(nxt)
            done |= last <= step
            if done.all() or step == steps - 1:
                break
            at = n + step
            # A row past its last step is done; its position is only kept inside the block.
            where = (lens + step).clamp(max=self.cfg.block_size - 1)[:, None]
            logits, _ = self(self._step(nxt, tracked), caches=caches, positions=where,
                             mask=real[:, None, None, : at + 1])
        gen = torch.cat(new, dim=1).tolist() if new else [[] for _ in prompts]
        outs = []
        for r, seq in enumerate(gen):
            seq = seq[: int(last[r]) + 1]
            outs.append(seq[: seq.index(stop)] if stop in seq else seq)
        return outs



class GPT(Decoder):
    """The original: GPT-2 style, pre-norm LayerNorm, GELU MLP, learned positions."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tying
        self._init_weights()

    def forward(self, idx, targets=None, caches=None, start=0, positions=None, mask=None):
        """`caches` (one dict per layer) and `start` (position of idx's first token) are for
        generation: feed the prompt once, then one new token per call. A padded batch gives
        each row's `positions` (batch x tokens) and the attention `mask` instead (Block)."""
        t = idx.shape[1]
        if positions is None:
            assert start + t <= self.cfg.block_size, f"sequence {start + t} longer than block {self.cfg.block_size}"
            positions = torch.arange(start, start + t, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(positions))
        for i, block in enumerate(self.blocks):
            x = block(x, caches[i] if caches is not None else None, mask)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
        )
        return logits, loss


def rope(x, cos, sin):
    """Rotary position embedding of x (batch x heads x tokens x head size) by the angles
    whose cos and sin are given (broadcasting to batch x 1 x tokens x half the head size)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).to(x.dtype)


class ModernBlock(nn.Module):
    """Pre-norm block with RMSNorm, QK-norm, RoPE and a SwiGLU MLP; no biases, no dropout."""

    def __init__(self, cfg):
        super().__init__()
        d, hd = cfg.n_embd, cfg.n_embd // cfg.n_head
        self.n_head = cfg.n_head
        self.norm1 = nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.q_norm = nn.RMSNorm(hd)
        self.k_norm = nn.RMSNorm(hd)
        self.proj = nn.Linear(d, d, bias=False)
        self.norm2 = nn.RMSNorm(d)
        hidden = mlp_hidden(d)
        self.gate_up = nn.Linear(d, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x, cache=None, mask=None, angles=None):
        b, t, c = x.shape
        q, k, v = self.qkv(self.norm1(x)).split(c, dim=2)
        q, k, v = (z.view(b, t, self.n_head, c // self.n_head).transpose(1, 2) for z in (q, k, v))
        cos, sin = angles
        q, k = rope(self.q_norm(q), cos, sin), rope(self.k_norm(k), cos, sin)
        y = attend(q, k, v, cache, mask, 0.0)
        x = x + self.proj(y.transpose(1, 2).reshape(b, t, c))
        gate, up = self.gate_up(self.norm2(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(gate) * up)


def mlp_hidden(d):
    """SwiGLU width with the parameters of a 4d GELU MLP (3 matrices of d x h = 2 of d x 4d),
    rounded up to a multiple of 64."""
    return -(-8 * d // 3 // 64) * 64


class Modern(Decoder):
    """Arm 1 of the 2026-11-15 bake-off: today's GPT with a current backbone. RMSNorm, SwiGLU,
    RoPE (no learned positions, so no block-size limit in the weights), no biases, no
    dropout (`cfg.dropout` is ignored), QK-norm always. Tied embeddings as GPT."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0 and (cfg.n_embd // cfg.n_head) % 2 == 0
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(ModernBlock(cfg) for _ in range(cfg.n_layer))
        self.norm = nn.RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight
        hd = cfg.n_embd // cfg.n_head
        self.register_buffer("inv_freq", 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd)), persistent=False)
        self._init_weights(residual=("proj.weight", "down.weight"))

    def forward(self, idx, targets=None, caches=None, start=0, positions=None, mask=None):
        t = idx.shape[1]
        if positions is None:
            assert start + t <= self.cfg.block_size, f"sequence {start + t} longer than block {self.cfg.block_size}"
            positions = torch.arange(start, start + t, device=idx.device)
        ang = positions.float()[..., None] * self.inv_freq  # (tokens or batch x tokens) x half
        ang = ang[None, None] if ang.dim() == 2 else ang[:, None]
        angles = (ang.cos(), ang.sin())
        x = self.tok(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, caches[i] if caches is not None else None, mask, angles)
        logits = self.head(self.norm(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
        )
        return logits, loss


class Spatial(GPT):
    """Arm 2 of the 2026-11-15 bake-off: today's GPT, each token's input plus two embeddings of
    the level's structure, from train/spatial.py: the field it fills (its role: `panel.x`,
    `wire.index`, ...) and the cell it is about (summed x, y and z embeddings; NONE where a
    coordinate isn't known). Both come from the tokens written so far only.

    Input: ids, or (ids, features) with features batch x tokens x 4 (role, x, y, z). Plain ids
    are parsed here (on the CPU, so it waits on the GPU) when they're whole sequences from
    their start; a step after a cached prefix needs its features (the samplers keep a parser
    per row: `_rows`). Training passes `featurize` to the batching so the parse happens before
    the batch goes to the GPU."""

    def __init__(self, cfg):
        import spatial

        if not cfg.itos:
            raise ValueError("a spatial model needs the vocabulary in its config (Config.itos)")
        super().__init__(cfg)
        self.role = nn.Embedding(len(spatial.ROLES), cfg.n_embd)
        self.cell = nn.ModuleList(nn.Embedding(spatial.COORDS, cfg.n_embd) for _ in range(3))
        for e in [self.role, *self.cell]:
            nn.init.normal_(e.weight, std=0.02)

    def featurize(self, ids):
        import spatial

        return torch.tensor(spatial.features(ids.tolist(), self.cfg.itos), dtype=torch.long)

    def _rows(self, b):
        import spatial

        return spatial.Rows(self.cfg.itos, b)

    def forward(self, idx, targets=None, caches=None, start=0, positions=None, mask=None):
        if isinstance(idx, tuple):
            idx, feats = idx
        elif start == 0 and positions is None:
            feats = self.featurize(idx.cpu()).to(idx.device)
        else:
            raise ValueError("a spatial model's step after a cached prefix needs its features")
        t = idx.shape[1]
        if positions is None:
            assert start + t <= self.cfg.block_size, f"sequence {start + t} longer than block {self.cfg.block_size}"
            positions = torch.arange(start, start + t, device=idx.device)
        x = self.tok(idx) + self.pos(positions) + self.role(feats[..., 0])
        for a in range(3):
            x = x + self.cell[a](feats[..., a + 1])
        x = self.drop(x)
        for i, block in enumerate(self.blocks):
            x = block(x, caches[i] if caches is not None else None, mask)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
        )
        return logits, loss


from wire import State, Rows, MAX_DEVICES
from wire_pointer import DevicePointer
from wire_model import WireConstraint


class WireGPT(GPT):
    def __init__(self, cfg):
        if not cfg.itos or '<wires>' not in cfg.itos:
            raise ValueError('wire architecture requires its extended vocabulary in Config.itos')
        super().__init__(cfg)
        self.pointer = DevicePointer(cfg.n_embd)
        self.pointer.apply(self._init)
        self.stoi = {w: i for i, w in enumerate(cfg.itos)}
        self.register_buffer('ptr_ids', torch.tensor([self.stoi[f'<ptr:{i}>']
                                                     for i in range(MAX_DEVICES)]), persistent=False)
        self.last_generation_stats = {}

    def featurize(self, ids):
        out = []
        for row in ids.tolist():
            state = State()
            out.append([state.feed(self.cfg.itos[i]) for i in row])
        return torch.tensor(out, dtype=torch.long)

    def _rows(self, b):
        return Rows(self.cfg.itos, b)

    def _generation_state(self, caches):
        return caches[0]['wire_hidden']

    def _restore_generation_state(self, caches, row, state):
        c = caches[0]
        if 'wire_hidden' not in c:
            c['wire_hidden'] = state.new_zeros((c['k'].shape[0], self.cfg.block_size, self.cfg.n_embd))
        c['wire_hidden'][row].zero_()
        c['wire_hidden'][row, :state.shape[1]] = state[0]

    def forward(self, idx, targets=None, caches=None, start=0, positions=None, mask=None):
        if isinstance(idx, tuple):
            idx, feats = idx
        else:
            if start or (caches and 'at' in caches[0]):
                raise ValueError('incremental wire decoding needs parser features')
            feats = self.featurize(idx.cpu()).to(idx.device)
        b, t = idx.shape
        if positions is None:
            if start + t > self.cfg.block_size:
                raise ValueError('sequence exceeds block size')
            positions = torch.arange(start, start + t, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(positions))
        for i, block in enumerate(self.blocks):
            x = block(x, caches[i] if caches is not None else None, mask)
        hidden = self.ln_f(x)
        logits = self.head(hidden)
        # Logical positions allow the same pointer features with left padding, KV caches,
        # and refilled rows. Padding never overwrites a real token at position zero.
        pos = positions.expand(b, t)
        if caches is None:
            history = hidden.new_zeros((b, self.cfg.block_size, self.cfg.n_embd))
        else:
            history = caches[0].get('wire_hidden')
            if history is None:
                history = hidden.new_zeros((b, self.cfg.block_size, self.cfg.n_embd))
                caches[0]['wire_hidden'] = history
        real = idx != self.stoi['<pad>']
        rows, cols = real.nonzero(as_tuple=True)
        history[rows, pos[rows, cols]] = hidden[rows, cols]
        rows, cols = feats.any(dim=-1).nonzero(as_tuple=True)
        if rows.numel():
            locations = feats[rows, cols]
            count = int((locations != 0).nonzero()[:, 1].max()) + 1
            locations = locations[:, :count]
            allowed = locations > 0
            keys = history[rows[:, None], (locations - 1).clamp(min=0)]
            pointer, _ = self.pointer(hidden[rows, cols, None, :], keys, allowed[:, None, :])
            # A pointer decision is normalized only over compatible device endpoints.
            logits = logits.clone()
            logits[rows, cols] = -torch.inf
            logits[rows[:, None], cols[:, None], self.ptr_ids[None, :count]] = pointer[:, 0]
        loss = None if targets is None else F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate_many(self, prompts, max_new, stop, temperature=1.0, top_k=None,
                      batch=32, constrain=None, logprob=False):
        adapter = WireConstraint(self, prompts, constrain)
        result = super().generate_many(prompts, max_new, stop, temperature, top_k,
                                       batch, adapter, logprob)
        self.last_generation_stats = adapter.stats
        return result

    def generate(self, prompts, max_new, stop, temperature=1.0, top_k=None):
        return self.generate_many(prompts, max_new, stop, temperature, top_k, batch=len(prompts) or 1)

    def generate_refill(self, prompt, n, max_new, stop, temperature=1.0, top_k=None, batch=64):
        return self.generate_many([prompt] * n, max_new, stop, temperature, top_k, batch)

ARCHS = {"gpt": GPT, "modern": Modern, "spatial": Spatial, "wire": WireGPT}


def build_model(cfg):
    """The model `cfg.arch` names (a checkpoint's config from before `arch` is a GPT)."""
    if cfg.arch not in ARCHS:
        raise ValueError(f"unknown architecture {cfg.arch!r} (known: {', '.join(ARCHS)})")
    return ARCHS[cfg.arch](cfg)
