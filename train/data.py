"""Load `glados dataset` JSONL into tensors, and turn labels into conditioning tokens.

A training sequence is

    <bos> <cond>... <sep> <level tokens>... <eos>

Conditions describe the level: verdict, plan length (or an upper bound on it), portal trips,
cubes, size, difficulty, which items and shapes it has, which mechanics its solution uses, and where it came from
(generator or Workshop, and how popular the Workshop puzzle was).
There is one slot per group, always in the same order, holding either a value or the group's
`?` token, so the level starts at the same position however much is specified. (With slots
simply left out, prompts with few conditions put the level where training rarely had it, and
the model lost count of the size numbers.) Each sequence hides each condition with its own
random probability, so at sampling time any subset can be given and the rest are left to the
model. Loss is only taken on the level tokens and <eos>.

Checkpoints trained before a group existed don't have its tokens; `groups_in(vocab)` lists the
groups a checkpoint knows, and prompts for it only fill those slots.

The solver net (task `solve`, `load_solve`) reads a level and writes its plan (src/plan.rs):

    <bos> <level tokens>... <sep> <plan tokens>... <eos>

with the loss on the plan and <eos> only, and no conditions.
"""

import json
import random
import zlib
from array import array
from pathlib import Path

import numpy as np
import torch

SPECIAL = ["<pad>", "<bos>", "<sep>", "<eos>"]
PAD, BOS, SEP, EOS = range(4)
# Plan words (src/plan.rs `plan::vocab`) beyond the level format's numbers and directions. Some
# (`exit`, `plate`, `funnel`) are level words too and are the same token. New words go last, so
# an older vocabulary is a prefix of a newer one.
PLAN_WORDS = ["walk", "fire", "enter", "pickup", "putdown", "toss", "throw", "press", "dash", "knock",
              "bounce", "leap", "plate", "climb", "funnel", "ride", "dive", "blue", "orange", "exit",
              "fetch", "reach"]

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
# Where a level came from, not what's in it, so it can't be read off a sampled level: `src`
# is the generator or a published Workshop puzzle (records with a `workshop_id`); `tier` is how
# many players subscribed to that puzzle (`subscribers`, lifetime), so prompting with
# `<src:ws> <tier:top>` asks for levels like the ones people played most.
PROVENANCE = {"src": ["<src:gen>", "<src:ws>"], "tier": ["<tier:mid>", "<tier:top>"]}
TOP_SUBSCRIBERS = 100
# Verdicts. `published`: a Workshop puzzle our solver couldn't confirm (unsolvable or out of
# budget). It passed the game's editor and was published, so it is taken as solvable, but its
# plan length and metrics are unknown, so it gets none.
VERDICTS = ("solvable", "uncertain", "unsolvable", "invalid", "published")

# One group per condition; a sequence carries at most one token from each.
GROUPS = {
    "verdict": [f"<verdict:{v}>" for v in VERDICTS],
    "len": [f"<len:{lo}-{hi}>" if hi < 10**9 else f"<len:{lo}+>" for lo, hi in LEN_BUCKETS],
    "trips": ["<trips:0>", "<trips:1>", "<trips:2+>"],
    "cubes": ["<cubes:0>", "<cubes:1>", "<cubes:2+>"],
    **{f: [f"<{f}:no>", f"<{f}:yes>"] for f in FEATURES},
    "size": [f"<size:{name}>" for name, _ in SIZE_BUCKETS],
    "diff": [f"<diff:{i}>" for i in range(len(DIFF_BUCKETS))],
    **{m: [f"<{m}:no>", f"<{m}:yes>"] for m in MECH_GROUPS},
    **{f: [f"<{f}:no>", f"<{f}:yes>"] for f in LATER_FEATURES},
    **{m: [f"<{m}:no>", f"<{m}:yes>"] for m in LATER_MECH_GROUPS},
    **PROVENANCE,
    # An upper bound on the plan length, same buckets as `len`: the actions a reverse-generated
    # level's construction spends (`constructed_len`). The level is solvable by construction,
    # so this holds even where the solver ran out of budget and `len` is unknown.
    "maxlen": [f"<maxlen:{lo}-{hi}>" if hi < 10**9 else f"<maxlen:{lo}+>" for lo, hi in LEN_BUCKETS],
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
    if record.get("assumed"):  # ws3 before `published`: made-up plan length and chain depth
        verdict = "published"
    if verdict not in VERDICTS:
        verdict = "invalid"
    # A reverse-generated level is solvable by construction, whatever the solver managed.
    reverse = record.get("src") == "reverse"
    if reverse and verdict in ("uncertain", "unsolvable"):
        verdict = "solvable"
    out = [f"<verdict:{verdict}>"]
    metrics = record.get("metrics") or {}
    if reverse and verdict == "solvable" and record.get("constructed_len") is not None:
        out.append(GROUPS["maxlen"][len_bucket(record["constructed_len"])])
    if verdict == "solvable" and record.get("min_len") is not None:
        out.append(GROUPS["len"][len_bucket(record["min_len"])])
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
    workshop = "workshop_id" in record
    out.append(GROUPS["src"][int(workshop)])
    if workshop and record.get("subscribers") is not None:
        out.append(GROUPS["tier"][int(record["subscribers"] >= TOP_SUBSCRIBERS)])
    return out


def len_bucket(n):
    return next(i for i, (lo, hi) in enumerate(LEN_BUCKETS) if lo <= n <= hi)


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
    def build(cls, level_vocab_file, task="levels"):
        """`levels`: special tokens, conditions, level words. `solve`: special tokens, level
        words, then the plan words the level format doesn't have; no conditions."""
        level = Path(level_vocab_file).read_text().split()
        if task == "solve":
            return cls(SPECIAL + level + [w for w in PLAN_WORDS if w not in level])
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


class Solves(Levels):
    """Levels with their plans (task `solve`), one split, as flat tensors. The level is the
    prompt; the plan and <eos> are what is learned."""

    def __init__(self, level_ids, level_offsets, plan_ids, plan_offsets, min_lens):
        # As in Levels, `level_ids`/`offsets` hold what is learned: here the plans.
        super().__init__(plan_ids, plan_offsets, None, [])
        self.prompt_ids = level_ids  # int16, all levels end to end
        self.prompt_offsets = level_offsets  # int64, len N + 1
        self.min_lens = min_lens  # the solver's plan length per record (-1: unknown)

    def __len__(self):
        return len(self.offsets) - 1

    def lengths(self):
        """Whole sequence (level and plan) of each record, for batching by length."""
        levels = self.prompt_offsets[1:] - self.prompt_offsets[:-1]
        return (levels + self.offsets[1:] - self.offsets[:-1]).tolist()

    def level(self, i):
        return self.prompt_ids[self.prompt_offsets[i] : self.prompt_offsets[i + 1]].tolist()

    def sequence(self, i, drop=0.0):
        level = self.level(i)
        plan = self.level_ids[self.offsets[i] : self.offsets[i + 1]].tolist()
        ids = [BOS] + level + [SEP] + plan + [EOS]
        learn = [False] * (len(level) + 2) + [True] * (len(plan) + 1)
        return ids, learn


class LengthStats:
    """Plan and whole-sequence lengths of solver-net records, and how many don't fit a block."""

    def __init__(self, block_size):
        self.block_size = block_size
        self.plans, self.seqs = array("i"), array("i")
        self.too_long = 0

    def add(self, n_level, n_plan):
        """Counts one record; False if its sequence (<bos> level <sep> plan <eos>) doesn't fit."""
        n = n_level + n_plan + 3
        self.plans.append(n_plan)
        self.seqs.append(n)
        if n > self.block_size:
            self.too_long += 1
            return False
        return True

    def report(self):
        if not self.plans:
            return "no plans"

        def dist(a):
            q = np.percentile(np.frombuffer(a, dtype=np.int32), [50, 90, 99, 100]).astype(int)
            return f"median {q[0]}, 90% {q[1]}, 99% {q[2]}, max {q[3]}"

        n = len(self.plans)
        return (f"{n} plans; plan tokens {dist(self.plans)}; whole sequence {dist(self.seqs)}; "
                f"{self.too_long} ({self.too_long / n:.2%}) longer than block {self.block_size}, dropped")


def load_solve(paths, vocab, block_size, val_share=0.05, seed=0):
    """Train and validation `Solves` from data/plans JSONL ({tokens, plan, min_len, ...}),
    streamed, and the `LengthStats` of every record with a plan. Records too long for
    `block_size`, or with tokens the vocabulary lacks, are dropped. A record goes to validation
    by a hash of its level, so a level repeated in the data stays on one side."""
    stats = LengthStats(block_size)
    stats.unknown = 0
    splits = {name: (array("h"), array("q", [0]), array("h"), array("q", [0]), array("h"))
              for name in ("train", "val")}
    cut = int(val_share * 2**32)
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if not r.get("plan"):
                    continue
                level, plan = r["tokens"].split(), r["plan"].split()
                if not stats.add(len(level), len(plan)):
                    continue
                try:
                    level_ids, plan_ids = vocab.encode(level), vocab.encode(plan)
                except KeyError:
                    stats.unknown += 1
                    continue
                split = "val" if zlib.crc32(f"{seed} {r['tokens']}".encode()) < cut else "train"
                lv, lo, pl, po, ml = splits[split]
                lv.extend(level_ids)
                lo.append(len(lv))
                pl.extend(plan_ids)
                po.append(len(pl))
                ml.append(r["min_len"] if r.get("min_len") is not None else -1)

    def tensor(a, dtype):
        return torch.frombuffer(a, dtype=dtype) if a else torch.zeros(0, dtype=dtype)

    out = {
        name: Solves(tensor(lv, torch.int16), tensor(lo, torch.int64), tensor(pl, torch.int16),
                     tensor(po, torch.int64), ml.tolist())
        for name, (lv, lo, pl, po, ml) in splits.items()
    }
    return out["train"], out["val"], stats


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Plan lengths in solver-net data (data/plans/*.jsonl), and how many records a block drops.")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--block", type=int, default=1024)
    a = ap.parse_args()
    lengths = LengthStats(a.block)
    for p in a.files:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip() and (r := json.loads(line)).get("plan"):
                    lengths.add(len(r["tokens"].split()), len(r["plan"].split()))
    print(lengths.report())
