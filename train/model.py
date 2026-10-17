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
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cache=None):
        """`cache` (a dict, for generation) keeps this layer's keys and values between calls:
        room for `cache["size"]` positions, allocated on the first call and filled in order.
        (Growing them with torch.cat reallocated both every token; during training on the
        4 GB laptop GPU that ran into system memory, and 64 samples took two minutes.)"""
        b, t, c = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(c, dim=2)
        q, k, v = (z.view(b, t, self.n_head, c // self.n_head).transpose(1, 2) for z in (q, k, v))
        if cache is not None:
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
            is_causal=q.size(2) == k.size(2),
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

    def forward(self, idx, targets=None, caches=None, start=0):
        """`caches` (one dict per layer) and `start` (position of idx's first token) are for
        generation: feed the prompt once, then one new token per call."""
        t = idx.shape[1]
        assert start + t <= self.cfg.block_size, f"sequence {start + t} longer than block {self.cfg.block_size}"
        x = self.drop(self.tok(idx) + self.pos(torch.arange(start, start + t, device=idx.device)))
        for i, block in enumerate(self.blocks):
            x = block(x, caches[i] if caches is not None else None)
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
