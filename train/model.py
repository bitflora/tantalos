"""A small decoder-only transformer (GPT-2 style, pre-norm), trained from scratch.

Level tokens are a few hundred words of a ~170-word language, so a ~10M parameter model is
plenty to start with and fits a 4 GB GPU with room to spare.
"""

import math
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
        if cache is not None and "at" in cache:
            # generate_refill: one new token per row, each written at its own row's position
            # into a cache allocated beforehand; attention only up to cache["len"].
            rows = torch.arange(b, device=x.device)
            cache["k"][rows, :, cache["at"]] = k[:, :, 0]
            cache["v"][rows, :, cache["at"]] = v[:, :, 0]
            k, v = cache["k"][:, :, : cache["len"]], cache["v"][:, :, : cache["len"]]
        elif cache is not None:
            n = cache.get("n", 0)
            if n == 0:
                shape = (b, self.n_head, cache["size"], c // self.n_head)
                cache["k"], cache["v"] = k.new_empty(shape), v.new_empty(shape)
            cache["k"][:, :, n : n + t] = k
            cache["v"][:, :, n : n + t] = v
            cache["n"] = n = n + t
            k, v = cache["k"][:, :, :n], cache["v"][:, :, :n]
        # New tokens after a cached prefix come one at a time and may see everything before.
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=mask is None and q.size(2) == k.size(2),
            dropout_p=self.dropout if self.training else 0.0,
        )
        x = x + self.resid_drop(self.proj(y.transpose(1, 2).reshape(b, t, c)))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
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
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self):
        return sum(p.numel() for p in self.parameters()) - self.pos.weight.numel()

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
                    logits, _ = self(nxt, caches=caches, start=n + step)
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
            with amp:
                logits, _ = self(nxt[:, None], caches=caches, positions=at[:, None], mask=mask)
            logits = logits[:, -1, :].float()
            if fresh:
                logits[sorted(fresh)] = prompt_logits
        return outs

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
        logits, _ = self(ids, caches=caches, positions=pos, mask=mask[:, None])
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
            logits, _ = self(nxt, caches=caches, positions=where, mask=real[:, None, None, : at + 1])
        gen = torch.cat(new, dim=1).tolist() if new else [[] for _ in prompts]
        outs = []
        for r, seq in enumerate(gen):
            seq = seq[: int(last[r]) + 1]
            outs.append(seq[: seq.index(stop)] if stop in seq else seq)
        return outs
