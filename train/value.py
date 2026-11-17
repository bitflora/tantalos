"""Run 3 of the generator bake-off (docs/architecture-plan-2026-11-15.md): value heads.

Small MLP heads read a frozen level net's hidden state where a level prefix ends at a record
boundary (a geometry group or an item with its wires is complete), and predict what the
finished level will be:
- verdict: solvable / uncertain / unsolvable (the solver's, editor rules as in the data);
- log(1 + solution length), for solvable levels;
- strong given solvable: the solution uses 2+ state-changing mechanics and at most 1 trick
  (rated.py's definition, from the data's metrics);
- item families used (rated.FAMILIES), for reporting.

The input is the net's final hidden state (after ln_f) at the prefix's last token, next to its
mean over the level's tokens so far. Every prefix is read under the rated prompt
(`<verdict:published> <src:ws> <tier:best>`, other groups `?`), the prompt guided sampling
uses: under a level's own condition tokens the heads could read its verdict off the prompt.

Guided sampling (`generate`): at every record boundary, each level draws k continuations of its
next record (until the following record's word, or the end). If the first draft ends the level,
the level ends (so levels end as often as the net's own); otherwise the heads score the drafts
that go on and the one with the highest P(solvable) * P(strong | solvable) - lam * P(unsolvable)
is kept. (Letting the heads pick ending drafts too, 2026-11-16, kept levels going to the end of
the block: some draft that goes on nearly always rates higher.)

    python train/value.py train runs/v7-fresh-16h/best.pt --out runs/arch-value
    python train/rated.py runs/v7-fresh-16h/best.pt --constrained --temperature 0.7 \\
        --guide runs/arch-value/heads.pt --k 4 --out runs/arch-value/rated-k4

Training data: a sample of the default level files (data/v5b, labels already in them) and the
rated-style samples of earlier runs' evals (their outcomes and metrics); v7-fresh-16h's own
rated samples are held out as the test that matches guided sampling. Held out from training
also: a tenth of puzzles (by data.identity, so mirrored copies stay together).
"""
import argparse
import glob
import hashlib
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
import rated  # noqa: E402
import sample  # noqa: E402
import spatial  # noqa: E402
from data import EOS, identity, source  # noqa: E402

VERDICTS = ("solvable", "uncertain", "unsolvable")
FAMILIES = list(rated.FAMILIES)
RECORD = frozenset(spatial.ROLE[w] for w in spatial.GEOM_WORDS | set(spatial.ITEMS))
# Level files and how many levels to take from each (the default data pattern, data/v5b).
SOURCES = {
    "data/v5b/levels-1.jsonl": 8000, "data/v5b/levels-2.jsonl": 8000, "data/v5b/levels-3.jsonl": 8000,
    "data/v5b/levels-4.jsonl": 8000, "data/v5b/levels-5.jsonl": 8000,
    "data/v5b/levels-mutants.jsonl": 6000, "data/v5b/levels-mutants-loops.jsonl": 6000,
    "data/v5b/levels-mutants-newrules.jsonl": 6000, "data/v5b/levels-workshop-aug-x2.jsonl": 40000,
}
SAMPLES = ["runs/*/eval/rated-samples.jsonl", "runs/*/away/eval/rated-samples.jsonl"]
CUTS = 6  # prefixes per level, besides the whole level


def labels(record):
    """(verdict index, log length or nan, strong-given-solvable or nan, families) of a data record
    or rated sample row, or None if it has no usable verdict."""
    if "outcome" in record:  # a rated sample
        v, line = record["outcome"], record["sample"]
        s, t = record.get("state"), record.get("tricks")
    else:
        v = record["verdict"]
        if v == "published":
            v = record.get("solver_verdict")
        line = record["tokens"]
        s, t = rated.mechanics(record.get("metrics"))
    if v not in VERDICTS:
        return None
    n = record.get("min_len")
    length = math.log1p(n) if v == "solvable" and n is not None else math.nan
    strong = float(s >= 2 and t <= 1) if v == "solvable" and s is not None else math.nan
    d = rated.describe(line)
    if d is None:
        return None
    fam = [float(f in d["families"]) for f in FAMILIES]
    return line, VERDICTS.index(v), length, strong, fam


def held_out(record):
    return int(hashlib.md5(identity(record).encode()).hexdigest(), 16) % 10 == 0


def read_sources(seed=0, scale=1.0):
    """[(line, labels..., split, source)]: split is train, val or test."""
    rng = random.Random(seed)
    rows = []
    for path, want in SOURCES.items():
        want = int(want * scale)
        with open(ROOT / path, encoding="utf-8") as f:
            total = sum(1 for _ in f)
        p = min(1.0, 1.3 * want / total)
        got = 0
        with open(ROOT / path, encoding="utf-8") as f:
            for line in f:
                if got >= want or rng.random() > p:
                    continue
                r = json.loads(line)
                lab = labels(r)
                if lab is None:
                    continue
                rows.append((*lab, "val" if held_out(r) else "train", source(r)))
                got += 1
        print(f"  {path}: {got}", flush=True)
    for pattern in SAMPLES:
        for path in sorted(glob.glob(str(ROOT / pattern))):
            test = "v7-fresh-16h" in Path(path).parts
            got = 0
            for line in open(path, encoding="utf-8"):
                lab = labels(json.loads(line))
                if lab is not None:
                    rows.append((*lab, "test" if test else "train", "samples"))
                    got += 1
            print(f"  {Path(path).relative_to(ROOT)}: {got}{' (test)' if test else ''}", flush=True)
    return rows


def amp_for(device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def final_norm(model):
    return model.ln_f if hasattr(model, "ln_f") else model.norm


@torch.no_grad()
def features(model, seqs, ends, start):
    """Head inputs for each (sequence, prefix end): the hidden state at the end and its mean over
    the level's tokens (from `start`) up to the end. seqs: id lists; ends: per sequence, a list
    of positions. Returns (sum of len(ends)) x 2d, float32 on the model's device."""
    device = next(model.parameters()).device
    got = {}
    hook = final_norm(model).register_forward_hook(lambda m, i, o: got.__setitem__("h", o))
    try:
        b, t = len(seqs), max(map(len, seqs))
        ids = torch.zeros((b, t), dtype=torch.long)
        for r, s in enumerate(seqs):
            ids[r, : len(s)] = torch.tensor(s)
        with amp_for(device):
            inp = ids.to(device)
            if model.featurize is not None:
                inp = (inp, model.featurize(ids).to(device))
            model(inp)
    finally:
        hook.remove()
    h = got["h"].float()
    level = h.clone()
    level[:, :start] = 0
    mean = level.cumsum(1) / (torch.arange(t, device=device) - start + 1).clamp(min=1)[None, :, None]
    rows = torch.tensor([r for r, e in enumerate(ends) for _ in e], device=device)
    cols = torch.tensor([c for e in ends for c in e], device=device)
    hc = h[rows, cols]
    mc = torch.where((cols >= start)[:, None], mean[rows, cols], hc)
    return torch.cat([hc, mc], dim=1)


def boundaries(ids, start, itos):
    """Positions in ids (prompt + level + EOS) after which a record ends: the token before a
    record word, before EOS, or the last token."""
    p = spatial.Parser()
    p.phase = "level"
    out = []
    for j in range(start, len(ids)):
        if ids[j] == EOS:
            out.append(j - 1)
            break
        role = p.feed_id(ids[j], itos)[0]
        if role in RECORD and j > start:
            out.append(j - 1)
    else:
        out.append(len(ids) - 1)
    return [e for e in out if e >= start + 3]  # from `size x y z` on


def extract(model, vocab, rows, cuts=CUTS, seed=0, tokens=24_000):
    """Features and labels of the given rows' prefixes (up to `cuts` random boundaries per level,
    and always the whole level)."""
    rng = random.Random(seed)
    p0 = sample.prompt(vocab, rated.prompt(vocab), ())
    block = model.cfg.block_size
    items = []
    for i, row in enumerate(rows):
        toks = row[0].split()
        if any(t not in vocab.stoi for t in toks):
            continue
        ids = p0 + vocab.encode(toks) + [EOS]
        if len(ids) > block:
            continue
        ends = boundaries(ids, len(p0), vocab.itos)
        if not ends:
            continue
        last = ends[-1]
        ends = sorted(rng.sample(ends[:-1], min(cuts, len(ends) - 1))) + [last]
        items.append((i, ids, ends, [(e - len(p0) + 1) / (last - len(p0) + 1) for e in ends]))
    items.sort(key=lambda x: len(x[1]))
    feats, meta = [], []
    t0, done = time.time(), 0
    j = 0
    while j < len(items):
        k = j
        while k < len(items) and (k - j + 1) * len(items[k][1]) <= tokens:
            k += 1
        k = max(k, j + 1)
        part = items[j:k]
        feats.append(features(model, [x[1] for x in part], [x[2] for x in part], len(p0)).half().cpu())
        for i, _, ends, fracs in part:
            for e, fr in zip(ends, fracs):
                meta.append((i, fr, e == ends[-1]))
        if (done + k - j) // 10_000 > done // 10_000:
            print(f"  {done + k - j}/{len(items)} levels, {time.time() - t0:.0f}s", flush=True)
        done += k - j
        j = k
    print(f"  features of {len(items)} levels, {len(meta)} prefixes in {time.time() - t0:.0f}s", flush=True)
    x = torch.cat(feats)
    lv = torch.tensor([rows[i][1] for i, _, _ in meta])
    ln = torch.tensor([rows[i][2] for i, _, _ in meta], dtype=torch.float32)
    st = torch.tensor([rows[i][3] for i, _, _ in meta], dtype=torch.float32)
    fam = torch.tensor([rows[i][4] for i, _, _ in meta], dtype=torch.float32)
    split = [rows[i][5] for i, _, _ in meta]
    src = [rows[i][6] for i, _, _ in meta]
    level = torch.tensor([i for i, _, _ in meta])
    frac = torch.tensor([f for _, f, _ in meta])
    final = torch.tensor([f for _, _, f in meta])
    return {"x": x, "verdict": lv, "length": ln, "strong": st, "families": fam, "split": split,
            "source": src, "level": level, "frac": frac, "final": final}


class Heads(nn.Module):
    def __init__(self, d_in, hidden=512, dropout=0.1):
        super().__init__()
        self.body = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))
        self.out = nn.Linear(hidden, len(VERDICTS) + 2 + len(FAMILIES))

    def forward(self, x):
        y = self.out(self.body(x.float()))
        return {"verdict": y[:, :3], "length": y[:, 3], "strong": y[:, 4], "families": y[:, 5:]}


def losses(out, d, idx):
    """Each head's mean loss on rows idx (strong and length only where known)."""
    v = F.cross_entropy(out["verdict"], d["verdict"][idx])
    ln, st = d["length"][idx], d["strong"][idx]
    has_len, has_st = ~torch.isnan(ln), ~torch.isnan(st)
    length = F.mse_loss(out["length"][has_len], ln[has_len]) if has_len.any() else out["length"].sum() * 0
    strong = (F.binary_cross_entropy_with_logits(out["strong"][has_st], st[has_st]) if has_st.any()
              else out["strong"].sum() * 0)
    fam = F.binary_cross_entropy_with_logits(out["families"], d["families"][idx])
    return {"verdict": v, "length": length, "strong": strong, "families": fam}


def constants(d, idx):
    """The constant predictions a head must beat: the training rows' class shares and means."""
    v = torch.bincount(d["verdict"][idx], minlength=3).float()
    ln, st = d["length"][idx], d["strong"][idx]
    return {"verdict": (v / v.sum()).clamp(min=1e-6).log(), "length": ln[~torch.isnan(ln)].mean(),
            "strong": torch.logit(st[~torch.isnan(st)].mean().clamp(1e-4, 1 - 1e-4)),
            "families": torch.logit(d["families"][idx].mean(0).clamp(1e-4, 1 - 1e-4))}


def constant_out(c, n):
    return {"verdict": c["verdict"].expand(n, 3), "length": c["length"].expand(n),
            "strong": c["strong"].expand(n), "families": c["families"].expand(n, -1)}


def auc(score, label):
    """Area under the ROC curve (ties counted half), or None with one class only."""
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = torch.cat([pos, neg]).argsort()
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(order) + 1, dtype=torch.float32)
    allv = torch.cat([pos, neg])
    for v in allv.unique():  # average ranks of ties
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def report(heads, d, c, idx, device):
    heads.eval()
    out = {k: [] for k in ("verdict", "length", "strong", "families")}
    for j in range(0, len(idx), 8192):
        o = heads(d["x"][idx[j: j + 8192]].to(device))
        for k in out:
            out[k].append(o[k].cpu())
    out = {k: torch.cat(v) for k, v in out.items()}
    got = {k: float(v) for k, v in losses(out, d, idx).items()}
    base = {k: float(v) for k, v in losses(constant_out(c, len(idx)), d, idx).items()}
    st, has = d["strong"][idx], ~torch.isnan(d["strong"][idx])
    unsolv = (d["verdict"][idx] == 2).float()
    return {"n": len(idx), "loss": got, "constant": base,
            "auc_strong": auc(out["strong"][has], st[has]),
            "auc_unsolvable": auc(out["verdict"].softmax(-1)[:, 2], unsolv),
            "auc_solvable": auc(out["verdict"].softmax(-1)[:, 0], (d["verdict"][idx] == 0).float())}


def fit(d, device, epochs=40, lr=1e-3, batch=2048, seed=0):
    torch.manual_seed(seed)
    tr = torch.tensor([i for i, s in enumerate(d["split"]) if s == "train"])
    va = torch.tensor([i for i, s in enumerate(d["split"]) if s == "val"])
    c = constants(d, tr)
    heads = Heads(d["x"].shape[1]).to(device)
    # Start the output at the constant predictions, so the heads begin at the baseline.
    with torch.no_grad():
        heads.out.weight.mul_(0.1)
        heads.out.bias.copy_(torch.cat([c["verdict"], c["length"][None], c["strong"][None], c["families"]]))
    opt = torch.optim.AdamW(heads.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * math.ceil(len(tr) / batch))
    x = d["x"].to(device)
    dev = {k: v.to(device) for k, v in d.items() if isinstance(v, torch.Tensor) and k != "x"}
    best, best_state = float("inf"), None
    for ep in range(epochs):
        heads.train()
        perm = tr[torch.randperm(len(tr))]
        tot = 0.0
        for j in range(0, len(perm), batch):
            idx = perm[j: j + batch].to(device)
            ls = losses(heads(x[idx]), dev, idx)
            loss = sum(ls.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(idx)
        heads.eval()
        with torch.no_grad():
            vl = {k: 0.0 for k in ("verdict", "length", "strong", "families")}
            for j in range(0, len(va), 8192):
                idx = va[j: j + 8192].to(device)
                for k, v in losses(heads(x[idx]), dev, idx).items():
                    vl[k] += float(v) * len(idx) / len(va)
        val = sum(vl.values())
        print(f"  epoch {ep + 1}: train {tot / len(tr):.4f}, val {val:.4f} "
              f"({', '.join(f'{k} {v:.4f}' for k, v in vl.items())})", flush=True)
        if val < best:
            best, best_state = val, {k: v.detach().clone() for k, v in heads.state_dict().items()}
    heads.load_state_dict(best_state)
    return heads, c


def summary_lines(rep):
    lines = ["| set | prefixes | verdict CE (const) | log length MSE (const) | strong BCE (const) | families BCE (const) "
             "| AUC strong | AUC unsolvable | AUC solvable |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    f = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
    for name, r in rep.items():
        L, C = r["loss"], r["constant"]
        lines.append(f"| {name} | {r['n']} | {L['verdict']:.3f} ({C['verdict']:.3f}) | {L['length']:.3f} "
                     f"({C['length']:.3f}) | {L['strong']:.3f} ({C['strong']:.3f}) | {L['families']:.3f} "
                     f"({C['families']:.3f}) | {f(r['auc_strong'])} | {f(r['auc_unsolvable'])} | {f(r['auc_solvable'])} |")
    return lines


def train_main(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, vocab, _ = sample.load_checkpoint(a.checkpoint, device)
    cache = out / "features.pt"
    if cache.exists() and not a.refresh:
        d = torch.load(cache, weights_only=False)
        print(f"features from {cache}", flush=True)
    else:
        print("reading levels", flush=True)
        rows = read_sources(a.seed, a.scale)
        d = extract(model, vocab, rows, seed=a.seed)
        torch.save(d, cache)
    del model
    torch.cuda.empty_cache()
    print(f"{len(d['split'])} prefixes: " + ", ".join(f"{s} {d['split'].count(s)}" for s in ("train", "val", "test")),
          flush=True)
    t = time.time()
    heads, c = fit(d, device, a.epochs, a.lr)
    print(f"trained in {time.time() - t:.0f}s", flush=True)
    idx = lambda f: torch.tensor([i for i in range(len(d["split"])) if f(i)], dtype=torch.long)  # noqa: E731
    sets = {
        "val (all)": idx(lambda i: d["split"][i] == "val"),
        **{f"val {s}": idx(lambda i, s=s: d["split"][i] == "val" and d["source"][i] == s) for s in ("gen", "mutant", "ws")},
        "val, whole level": idx(lambda i: d["split"][i] == "val" and bool(d["final"][i])),
        "val, first third": idx(lambda i: d["split"][i] == "val" and float(d["frac"][i]) <= 1 / 3),
        "v7 samples (test)": idx(lambda i: d["split"][i] == "test"),
        "v7 samples, first third": idx(lambda i: d["split"][i] == "test" and float(d["frac"][i]) <= 1 / 3),
    }
    rep = {k: report(heads, d, c, v, device) for k, v in sets.items() if len(v)}
    beats = lambda name: [h for h in ("verdict", "length", "strong", "families")  # noqa: E731
                          if name in rep and rep[name]["loss"][h] < rep[name]["constant"][h]]
    ready = len(beats("val (all)")) == 4  # the plan's check: held-out prefixes
    v7 = beats("v7 samples (test)")
    torch.save({"heads": heads.state_dict(), "d_in": d["x"].shape[1], "checkpoint": str(a.checkpoint),
                "families": FAMILIES, "verdicts": VERDICTS, "report": rep, "ready": ready}, out / "heads.pt")
    text = ["# Value heads (bake-off run 3)", "",
            f"Backbone: `{a.checkpoint}` (frozen). Lower loss is better; (const) is the constant "
            "prediction from the training rows' shares and means. Ready when every head beats it on "
            "held-out data.", "", *summary_lines(rep), "",
            f"**{'READY' if ready else 'NOT READY'}**: {len(beats('val (all)'))} of 4 heads beat the constant "
            "baseline on held-out puzzles (the plan's check). On v7-fresh-16h's own rated samples, the "
            f"distribution guided sampling sees: {len(v7)} of 4 ({', '.join(v7) or 'none'}); AUC above "
            "0.5 still means the head ranks drafts better than chance.", ""]
    (out / "heads.md").write_text("\n".join(text), encoding="utf-8")
    (out / "heads.json").write_text(json.dumps(rep, indent=1))
    print("\n".join(text))


def load_heads(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    heads = Heads(ck["d_in"]).to(device)
    heads.load_state_dict(ck["heads"])
    heads.eval()
    return heads


class Boundary:
    """A `generate_many` constraint for one round of guided sampling: each row writes one record
    and stops (EOS forced) once the next record's word comes; with a grammar, only tokens it
    allows. Rows are candidates; `base[i]` is the parser after candidate i's level so far."""

    def __init__(self, vocab, levels, base, grammar=None):
        self.vocab, self.levels, self.grammar = vocab, levels, grammar
        self.state = {}
        self.base = base

    def last_is_record(self, i):
        return self.state.get(i, (None, 0, False))[2]

    def __call__(self, idx, seqs):
        V = len(self.vocab)
        mask = torch.zeros((len(seqs), V), dtype=torch.bool)
        ask = []
        for r, (i, s) in enumerate(zip(idx, seqs)):
            p, fed, last = self.state.get(i) or (self.base[i].copy(), 0, False)
            for t in s[fed:]:
                last = p.feed_id(t, self.vocab.itos)[0] in RECORD
            self.state[i] = (p, len(s), last)
            if len(s) >= 2 and last:
                mask[r, EOS] = True
            elif self.grammar is None:
                mask[r] = True
            else:
                ask.append(r)
        if ask:
            answers = self.grammar.next([" ".join(self.vocab.decode(self.levels[idx[r]] + seqs[r])) for r in ask])
            for r, a in zip(ask, answers):
                ids = [self.vocab.stoi[t] for t in a.get("allowed", []) if t in self.vocab.stoi]
                if a.get("end") or not ids:
                    ids.append(EOS)
                mask[r, ids] = True
        return mask

    def finish(self, i, seq):
        return seq


def value(out, lam):
    pv = out["verdict"].float().softmax(-1)
    return pv[:, 0] * torch.sigmoid(out["strong"].float()) - lam * pv[:, 2]


@torch.no_grad()
def generate(model, vocab, heads, n, conds=(), temperature=1.0, top_k=None, batch=32, k=4, lam=0.25,
             constrained=False, tantalos=sample.DEFAULT_TANTALOS, log=None):
    """n levels (token lines), each record the best of k by the heads (see the module doc)."""
    p0 = sample.prompt(vocab, list(conds), [])
    block = model.cfg.block_size
    levels = [[] for _ in range(n)]
    parsers = []
    for _ in range(n):
        p = spatial.Parser()
        p.phase = "level"
        parsers.append(p)
    done = [False] * n
    grammar = sample.Grammar(tantalos) if constrained else None
    rounds, t0 = 0, time.time()
    try:
        while not all(done):
            live = [i for i in range(n) if not done[i]]
            cand = [i for i in live for _ in range(k)]
            cons = Boundary(vocab, [levels[i] for i in cand], [parsers[i] for i in cand], grammar)
            prompts = [p0 + levels[i] for i in cand]
            outs = model.generate_many(prompts, block, EOS, temperature, top_k, batch * k, cons)
            recs, ended = [], []
            for j, s in enumerate(outs):
                if len(s) >= 2 and cons.last_is_record(j):
                    recs.append(s[:-1])
                    ended.append(False)
                else:
                    recs.append(s)
                    ended.append(True)
            seqs = [p0 + levels[i] + r for i, r in zip(cand, recs)]
            scores = []
            for j in range(0, len(seqs), 64):
                part = seqs[j: j + 64]
                x = features(model, part, [[len(s) - 1] for s in part], len(p0))
                scores.append(value(heads(x), lam).cpu())
            scores = torch.cat(scores).view(len(live), k)
            # Whether the level ends is the net's own call (its first draft), so the heads can't
            # keep levels going: they choose only among drafts that go on.
            stops = torch.tensor(ended).view(len(live), k)
            scores[stops] = -float("inf")
            best = [0 if stops[li, 0] else int(scores[li].argmax()) for li in range(len(live))]
            for li, i in enumerate(live):
                j = li * k + best[li]
                for t in recs[j]:
                    parsers[i].feed_id(t, vocab.itos)
                levels[i] += recs[j]
                if ended[j] or len(p0) + len(levels[i]) >= block - 1:
                    done[i] = True
            rounds += 1
            if log and rounds % 10 == 0:
                log(f"guided: round {rounds}, {sum(done)}/{n} levels done, {time.time() - t0:.0f}s")
    finally:
        if grammar is not None:
            grammar.stop()
    return [" ".join(vocab.decode(lv)) for lv in levels]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="extract prefix features from a frozen checkpoint and train the heads")
    t.add_argument("checkpoint")
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--scale", type=float, default=1.0, help="share of the usual level counts to read")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--refresh", action="store_true", help="extract the features again")
    a = ap.parse_args()
    if a.cmd == "train":
        train_main(a)


if __name__ == "__main__":
    main()
