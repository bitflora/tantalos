"""Load `glados dataset` JSONL into tensors, and turn labels into conditioning tokens.

A training sequence is

    <bos> <cond>... <sep> <level tokens>... <eos>

Conditions describe the level (verdict, plan length, portal trips, cubes, which items it has).
Each is dropped at random during training, so at sampling time any subset can be given and the
rest are left to the model. Loss is only taken on the level tokens and <eos>.
"""

import json
import random
from array import array
from pathlib import Path

import torch

SPECIAL = ["<pad>", "<bos>", "<sep>", "<eos>"]
PAD, BOS, SEP, EOS = range(4)

LEN_BUCKETS = [(0, 0), (1, 2), (3, 4), (5, 7), (8, 11), (12, 10**9)]
FEATURES = ["fizzler", "laserfield", "glass", "grating", "pedestal", "goo"]

# One group per condition; a sequence carries at most one token from each.
GROUPS = {
    "verdict": [f"<verdict:{v}>" for v in ("solvable", "uncertain", "unsolvable", "invalid")],
    "len": [f"<len:{lo}-{hi}>" if hi < 10**9 else f"<len:{lo}+>" for lo, hi in LEN_BUCKETS],
    "trips": ["<trips:0>", "<trips:1>", "<trips:2+>"],
    "cubes": ["<cubes:0>", "<cubes:1>", "<cubes:2+>"],
    **{f: [f"<{f}:no>", f"<{f}:yes>"] for f in FEATURES},
}
COND_TOKENS = [t for group in GROUPS.values() for t in group]


def conditions(record):
    """Condition tokens for one dataset record, one per group that applies."""
    level = record["tokens"].split()
    verdict = record["verdict"]
    if verdict not in ("solvable", "uncertain", "unsolvable"):
        verdict = "invalid"
    out = [f"<verdict:{verdict}>"]
    if verdict == "solvable":
        n = record["min_len"]
        lo, hi = next(b for b in LEN_BUCKETS if b[0] <= n <= b[1])
        out.append(GROUPS["len"][LEN_BUCKETS.index((lo, hi))])
        trips = record.get("metrics", {}).get("portal_trips")
        if trips is not None:
            out.append(GROUPS["trips"][min(trips, 2)])
    cubes = level.count("dropper")
    out.append(GROUPS["cubes"][min(cubes, 2)])
    has = set(level)
    for f in FEATURES:
        present = any(t.startswith("g") and t[1:].isdigit() for t in level) if f == "goo" else f in has
        out.append(GROUPS[f][int(present)])
    return out


class Vocab:
    def __init__(self, tokens):
        self.itos = list(tokens)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        assert len(self.stoi) == len(self.itos), "duplicate tokens"

    @classmethod
    def build(cls, level_vocab_file):
        level = Path(level_vocab_file).read_text().split()
        return cls(SPECIAL + COND_TOKENS + level)

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi[t] for t in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids]


class Levels:
    """All records of one split, as flat tensors."""

    def __init__(self, level_ids, offsets, conds):
        self.level_ids = level_ids  # int16, all levels end to end
        self.offsets = offsets  # int64, len N + 1
        self.conds = conds  # list of lists of ids

    def __len__(self):
        return len(self.conds)

    def sequence(self, i, drop):
        """Token ids and loss mask for record i, each condition dropped with prob `drop`."""
        conds = [c for c in self.conds[i] if random.random() >= drop]
        level = self.level_ids[self.offsets[i] : self.offsets[i + 1]].tolist()
        ids = [BOS] + conds + [SEP] + level + [EOS]
        learn = [False] * (len(conds) + 2) + [True] * (len(level) + 1)
        return ids, learn

    def batch(self, idx, drop, device):
        """Inputs, targets (-100 = ignored), padded to the longest sequence in the batch."""
        seqs = [self.sequence(i, drop) for i in idx]
        width = max(len(ids) for ids, _ in seqs) - 1
        x = torch.full((len(seqs), width), PAD, dtype=torch.long)
        y = torch.full((len(seqs), width), -100, dtype=torch.long)
        for row, (ids, learn) in enumerate(seqs):
            n = len(ids) - 1
            x[row, :n] = torch.tensor(ids[:-1])
            target = torch.tensor(ids[1:])
            target[~torch.tensor(learn[1:])] = -100
            y[row, :n] = target
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def load(paths, vocab, block_size, val_share=0.05, seed=0):
    """Train and validation `Levels` from JSONL files, streamed. Records whose sequence
    wouldn't fit in `block_size` are skipped. Each record goes to validation with probability
    `val_share` (seeded)."""
    max_prefix = 2 + len(GROUPS)
    rng = random.Random(seed)
    skipped = 0
    splits = {
        "train": (array("h"), array("q", [0]), []),
        "val": (array("h"), array("q", [0]), []),
    }
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                split = "val" if rng.random() < val_share else "train"
                level = vocab.encode(r["tokens"].split())
                if max_prefix + len(level) + 1 > block_size:
                    skipped += 1
                    continue
                ids, offsets, conds = splits[split]
                ids.extend(level)
                offsets.append(len(ids))
                conds.append(vocab.encode(conditions(r)))
    out = {}
    for name, (ids, offsets, conds) in splits.items():
        out[name] = Levels(
            torch.frombuffer(ids, dtype=torch.int16) if ids else torch.zeros(0, dtype=torch.int16),
            torch.frombuffer(offsets, dtype=torch.int64),
            conds,
        )
    return out["train"], out["val"], skipped
