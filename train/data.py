"""Load `glados dataset` JSONL into tensors, and turn labels into conditioning tokens.

A training sequence is

    <bos> <cond>... <sep> <level tokens>... <eos>

Conditions describe the level: verdict, plan length, portal trips, cubes, size, difficulty,
which items and shapes it has, and which mechanics its solution uses.
There is one slot per group, always in the same order, holding either a value or the group's
`?` token, so the level starts at the same position however much is specified. (With slots
simply left out, prompts with few conditions put the level where training rarely had it, and
the model lost count of the size numbers.) Each sequence hides each condition with its own
random probability, so at sampling time any subset can be given and the rest are left to the
model. Loss is only taken on the level tokens and <eos>.

Checkpoints trained before a group existed don't have its tokens; `groups_in(vocab)` lists the
groups a checkpoint knows, and prompts for it only fill those slots.
"""

import json
import random
from array import array
from pathlib import Path

import numpy as np
import torch

SPECIAL = ["<pad>", "<bos>", "<sep>", "<eos>"]
PAD, BOS, SEP, EOS = range(4)

LEN_BUCKETS = [(0, 0), (1, 2), (3, 4), (5, 7), (8, 11), (12, 10**9)]
FEATURES = ["fizzler", "laserfield", "glass", "grating", "pedestal", "goo", "turret", "gel", "wall", "balcony"]
# Items added later, as `<name:yes|no>` groups after all the others: an emitter (`laser`),
# a catcher or relay (`receiver`), faith plates, bridges, funnels, angled panels (any kind),
# flip panels, stairs, pistons, rails, and cubes placed without a dropper.
LATER_FEATURES = ["laser", "receiver", "plate", "bridge", "funnel", "panel", "flip", "stairs",
                  "piston", "rail", "loosecube"]
# Floor area (width x depth): small, medium, large, extra large.
SIZE_BUCKETS = [("s", 49), ("m", 100), ("l", 144), ("xl", 10**9)]
# Difficulty: the solution's chain depth (steps that each need an earlier one; glados metrics).
DIFF_BUCKETS = [0.0, 1.5, 3.0, 6.0, float("inf")]
# glados `metrics::MECHANICS`, in bit order. Portal trips have their own group (`trips`).
MECHANICS = [
    "portal_trip",
    "cube_portal",
    "cube_button",
    "stand_button",
    "toss",
    "latch",
    "dash",
    "fizzler_use",
    "laser_use",
    "turret_knock",
    "bounce",
    "leap",
    "conversion",
    "gel_portal",
    "redirect",
    "powered",
    "plate_use",
    "fling",
    "bridge_use",
    "funnel_use",
    "stairs_use",
    "platform_use",
    "panel_use",
    "projection_portal",
]
MECH_GROUPS = MECHANICS[1:14]
# Mechanics added with the later items: their groups come after all the others.
LATER_MECH_GROUPS = MECHANICS[14:]

# One group per condition; a sequence carries at most one token from each.
GROUPS = {
    "verdict": [f"<verdict:{v}>" for v in ("solvable", "uncertain", "unsolvable", "invalid")],
    "len": [f"<len:{lo}-{hi}>" if hi < 10**9 else f"<len:{lo}+>" for lo, hi in LEN_BUCKETS],
    "trips": ["<trips:0>", "<trips:1>", "<trips:2+>"],
    "cubes": ["<cubes:0>", "<cubes:1>", "<cubes:2+>"],
    **{f: [f"<{f}:no>", f"<{f}:yes>"] for f in FEATURES},
    "size": [f"<size:{name}>" for name, _ in SIZE_BUCKETS],
    "diff": [f"<diff:{i}>" for i in range(len(DIFF_BUCKETS))],
    **{m: [f"<{m}:no>", f"<{m}:yes>"] for m in MECH_GROUPS},
    **{f: [f"<{f}:no>", f"<{f}:yes>"] for f in LATER_FEATURES},
    **{m: [f"<{m}:no>", f"<{m}:yes>"] for m in LATER_MECH_GROUPS},
}
# "Not specified", and the value of a group that doesn't apply (plan length of an unsolvable level).
ANY = {group: f"<{group}:?>" for group in GROUPS}
COND_TOKENS = [t for group, tokens in GROUPS.items() for t in tokens + [ANY[group]]]


def columns(level):
    """The level's columns as lists of runs (e.g. ["s2", "o3"]), in the order written; `=` and
    left-out columns aren't repeated, which doesn't matter for spotting shapes."""
    out, cur = [], []
    for t in level[4:]:
        if t == "|":
            out.append(cur)
            cur = []
        elif t in ("/", "="):
            continue
        elif len(t) > 1 and t[0] in "sog" and t[1:].isdigit():
            cur.append(t)
        else:
            break
    return out


def shapes(level):
    """(wall, balcony, goo): a column solid to the ceiling; open space under something solid;
    any goo. Reads both geometry formats (`open`/`goo` boxes, or column runs)."""
    if len(level) > 4 and level[4] in ("open", "goo"):
        return box_shapes(level)
    height = level[3] if len(level) > 3 else None
    cols = columns(level)
    wall = any(c == [f"s{height}"] for c in cols)
    balcony = any(
        any(a[0] in "og" and b[0] == "s" for a, b in zip(c, c[1:])) for c in cols
    )
    goo = any(t[0] == "g" for c in cols for t in c)
    return wall, balcony, goo


def box_shapes(level):
    try:
        size = [int(v) for v in level[1:4]]
    except ValueError:
        return False, False, False
    space = np.zeros(size, dtype=bool)  # x, y, z: open or goo
    goo, word, i = False, None, 4
    while i < len(level):
        if level[i] in ("open", "goo"):
            word = level[i]
            i += 1
            continue
        box = level[i:i + 6]
        if word is None or len(box) < 6 or not all(v.isdigit() for v in box):
            break
        lo, hi = [int(v) for v in box[:3]], [int(v) for v in box[3:]]
        space[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] = True
        goo |= word == "goo"
        i += 6
    wall = bool((~space.any(axis=2)).any())
    below = np.maximum.accumulate(space, axis=2)  # open space at or under each cell
    balcony = bool((below[:, :, :-1] & ~space[:, :, 1:]).any())
    return wall, balcony, goo


def conditions(record):
    """Condition tokens for one dataset record, one per group that applies."""
    level = record["tokens"].split()
    verdict = record["verdict"]
    if verdict not in ("solvable", "uncertain", "unsolvable"):
        verdict = "invalid"
    out = [f"<verdict:{verdict}>"]
    metrics = record.get("metrics") or {}
    if verdict == "solvable":
        n = record["min_len"]
        lo, hi = next(b for b in LEN_BUCKETS if b[0] <= n <= b[1])
        out.append(GROUPS["len"][LEN_BUCKETS.index((lo, hi))])
        trips = metrics.get("portal_trips")
        if trips is not None:
            out.append(GROUPS["trips"][min(trips, 2)])
        depth = metrics.get("chain_depth")
        if depth is not None:
            out.append(GROUPS["diff"][next(i for i, hi in enumerate(DIFF_BUCKETS) if depth <= hi)])
        bits = metrics.get("mechanic_bits")
        if bits is not None:
            for i, m in enumerate(MECHANICS):
                if m in GROUPS:
                    out.append(GROUPS[m][(bits >> i) & 1])
    loose = loose_cubes(level)
    cubes = level.count("dropper") + loose
    out.append(GROUPS["cubes"][min(cubes, 2)])
    has = set(level)
    wall, balcony, goo = shapes(level)
    for f in FEATURES:
        if f == "goo":
            present = goo
        elif f == "wall":
            present = wall
        elif f == "balcony":
            present = balcony
        else:
            present = f in has
        out.append(GROUPS[f][int(present)])
    if len(level) > 3 and level[0] == "size" and level[1].isdigit() and level[2].isdigit():
        area = int(level[1]) * int(level[2])
        out.append(GROUPS["size"][next(i for i, (_, hi) in enumerate(SIZE_BUCKETS) if area <= hi)])
    words = {
        "laser": ["laser"], "receiver": ["catcher", "relay"], "plate": ["plate"],
        "bridge": ["bridge"], "funnel": ["funnel"], "panel": ["panel", "glasspanel"],
        "flip": ["flip"], "stairs": ["stairs"], "piston": ["piston"], "rail": ["rail"],
    }
    for f in LATER_FEATURES:
        # Items, not wire targets (`> laser 0`).
        present = loose > 0 if f == "loosecube" else any(
            t in words[f] and (i == 0 or level[i - 1] != ">") for i, t in enumerate(level))
        out.append(GROUPS[f][int(present)])
    return out


def loose_cubes(level):
    """Cubes placed without a dropper: `cube x y z K` (not a `cube` button)."""
    return sum(1 for i, t in enumerate(level)
               if t == "cube" and i + 3 < len(level) and all(v.isdigit() for v in level[i + 1:i + 4]))


# Item keywords in the level tokens, by the name Portal 2 logs for an invalid item
# (`INVALID ITEM: <TargetName> (x, y, z)`, see src/editor.rs). A cube is written as its dropper.
GAME_NAMES = {
    "doorentry": ["entry"],
    "doorexit": ["exit"],
    "button": ["button", "pedestal"],
    "cubedropper": ["dropper"],
    "cube": ["dropper", "cube"],
    "target": ["glass", "grating"],
    "barrierhazard": ["fizzler", "laserfield"],
    "turret": ["turret"],
    "paintsplat": ["gel"],
    "paintdropper": ["gel"],
    "laseremit": ["laser"],
    "lasercatch": ["catcher"],
    "laserrelay": ["relay"],
    "catapult": ["plate"],
    "bridge": ["bridge"],
    "tbeam": ["funnel"],
    "angledPanel": ["panel"],
    "angledClearPanel": ["glasspanel"],
    "flipPanel": ["flip"],
    "stairs": ["stairs"],
    "pistonlift": ["piston"],
    "raillift": ["rail"],
}
# Tokens after the keyword that give each item's cells: one cell, or a box (two cells).
BOX_ITEMS = ("glass", "grating")


def item_spans(level, game_items):
    """Positions (in `level`, a token list) of the coordinates of the items the game marked
    invalid: the tokens that placed them. Items not found (goo is written as geometry; cube and
    dropper positions differ) are skipped."""
    out = []
    for item in game_items:
        words = GAME_NAMES.get(item.get("name"), [])
        pos = list(item.get("pos", []))
        for i, t in enumerate(level):
            if t not in words or i + 4 > len(level):
                continue
            try:
                cell = [int(v) for v in level[i + 1 : i + 4]]
            except ValueError:
                continue
            if t in BOX_ITEMS and i + 7 <= len(level):
                try:
                    hi = [int(v) for v in level[i + 4 : i + 7]]
                except ValueError:
                    continue
                inside = all(min(a, b) <= p <= max(a, b) for a, b, p in zip(cell, hi, pos))
                span = range(i + 1, i + 7)
            else:
                # A dropper's cube sits on the floor below it: same column.
                inside = cell == pos or (item.get("name") == "cube" and cell[:2] == pos[:2])
                span = range(i + 1, i + 4)
            if inside:
                out.extend(span)
                break
    return sorted(set(out))


class Away:
    """Levels the game rejected, prompted with their own conditions but `<verdict:solvable>`
    (hidden at random like training prompts), with the positions whose tokens placed a
    rejected item. Training pushes those coordinates down among the numbers that could go
    there (unlikelihood), so the model puts the item somewhere else rather than nowhere.

    A first version pushed against all tokens under one fixed `<verdict:solvable>` prompt;
    the model learned to stop writing coordinates under that prompt (0% of its levels parsed)."""

    def __init__(self, paths, vocab, block_size):
        self.seqs = []
        self.groups = groups_in(vocab)
        self.any_ids = vocab.encode([ANY[g] for g in self.groups])
        for p in paths:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    level = r["tokens"].split()
                    spans = item_spans(level, r.get("game_items", []))
                    if not spans or any(t not in vocab.stoi for t in level):
                        continue
                    if 2 + len(self.groups) + len(level) > block_size:
                        continue
                    conds = [c for c in conditions(r) if not c.startswith("<verdict:")]
                    conds = slots(["<verdict:solvable>"] + conds, self.groups)
                    self.seqs.append((vocab.encode(conds), vocab.encode(level), spans))

    def __len__(self):
        return len(self.seqs)

    def batch(self, size, drop, device):
        """Inputs, the token to push away at each position (-100 = none), padded. The verdict
        always stays; other conditions are hidden with a probability drawn from [0, `drop`]."""
        picks = [self.seqs[random.randrange(len(self.seqs))] for _ in range(size)]
        seqs = []
        for conds, level, spans in picks:
            p = random.uniform(0, drop)
            shown = [c if i == 0 or random.random() >= p else a
                     for i, (c, a) in enumerate(zip(conds, self.any_ids))]
            prompt = [BOS] + shown + [SEP]
            # Predicting level[s] happens at input position len(prompt) + s - 1.
            seqs.append((prompt + level, [len(prompt) + s - 1 for s in spans]))
        width = max(len(ids) for ids, _ in seqs) - 1
        x = torch.full((size, width), PAD, dtype=torch.long)
        y = torch.full((size, width), -100, dtype=torch.long)
        for row, (ids, at) in enumerate(seqs):
            x[row, : len(ids) - 1] = torch.tensor(ids[:-1])
            for k in at:
                y[row, k] = ids[k + 1]
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def groups_in(vocab):
    """The condition groups a vocabulary (a checkpoint's) has tokens for, in slot order."""
    return [g for g in GROUPS if ANY[g] in vocab.stoi]


def slots(conds, groups=None):
    """One token per group (of `groups`, default all), in group order: the given condition,
    or the group's `?`."""
    out = {g: ANY[g] for g in (groups or GROUPS)}
    for t in conds:
        group = next((g for g, tokens in GROUPS.items() if t in tokens or t == ANY[g]), None)
        if group is None:
            raise ValueError(f"{t} isn't a condition")
        if group not in out:
            raise ValueError(f"{t}: this model wasn't trained with {group} conditions")
        if out[group] != ANY[group]:
            raise ValueError(f"two conditions for {group}: {out[group]} and {t}")
        out[group] = t
    return list(out.values())


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

    def __init__(self, level_ids, offsets, conds, any_ids):
        self.level_ids = level_ids  # int16, all levels end to end
        self.offsets = offsets  # int64, len N + 1
        self.conds = conds  # list of lists of ids, one per group
        self.any_ids = any_ids  # each group's `?`

    def __len__(self):
        return len(self.conds)

    def lengths(self):
        """Level tokens in each record."""
        return (self.offsets[1:] - self.offsets[:-1]).tolist()

    def sequence(self, i, drop):
        """Token ids and loss mask for record i. Each condition is hidden with a probability
        drawn uniformly from [0, `drop`] for this sequence, so both full and sparse prompts
        are common."""
        p = random.uniform(0, drop)
        conds = [c if random.random() >= p else a for c, a in zip(self.conds[i], self.any_ids)]
        level = self.level_ids[self.offsets[i] : self.offsets[i + 1]].tolist()
        ids = [BOS] + conds + [SEP] + level + [EOS]
        learn = [False] * (len(conds) + 2) + [True] * (len(level) + 1)
        return ids, learn

    def batch(self, idx, drop, device):
        """Inputs, targets (-100 = ignored), padded to the longest sequence in the batch, and
        the number of targets (counted here, so the caller needn't wait on the GPU for it)."""
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
        targets = int((y != -100).sum())
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True), targets


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
                conds.append(vocab.encode(slots(conditions(r))))
    out = {}
    for name, (ids, offsets, conds) in splits.items():
        out[name] = Levels(
            torch.frombuffer(ids, dtype=torch.int16) if ids else torch.zeros(0, dtype=torch.int16),
            torch.frombuffer(offsets, dtype=torch.int64),
            conds,
            vocab.encode(ANY.values()),
        )
    return out["train"], out["val"], skipped
