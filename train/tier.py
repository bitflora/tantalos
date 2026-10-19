"""Difficulty tiers for the broadcast loop (train/broadcast.py): which condition tokens the
level net is asked for, and when to move up or down.

A tier is a set of the level net's own condition tokens (train/data.py GROUPS): the shortest
plan's length bucket, the chain-depth bucket and portal trips. Each round samples mostly at the
current tier, some at the next (so the harvest already holds harder levels when the solver net
gets there) and some at the one before.

After a round, `next_tier` looks at how the solver net did on that round's levels at the
current tier: it moves up once the net solves at least UP of them greedily and the level net
still writes enough of them (YIELD: kept levels per sampled level at the tier), and down when
the net solves fewer than DOWN.
"""

TIERS = [
    ["<len:3-4>"],
    ["<len:5-7>", "<diff:2>"],
    ["<len:8-11>", "<diff:3>"],
    ["<len:12+>", "<diff:4>", "<trips:2+>"],
]
NAMES = ["T1", "T2", "T3", "T4"]
MIX = ((0, 0.7), (1, 0.2), (-1, 0.1))  # (tier offset, share of the round's samples)
UP, DOWN, YIELD = 0.5, 0.15, 0.15
BASE = ["<verdict:solvable>"]


def conds(tier):
    """Condition tokens for sampling at `tier` (0-based)."""
    return BASE + TIERS[tier]


def mix(tier, n):
    """[(tier, count)] for a round of `n` samples at `tier`: offsets past either end fold
    back into the current tier."""
    counts = {}
    for off, share in MIX:
        t = tier + off if 0 <= tier + off < len(TIERS) else tier
        counts[t] = counts.get(t, 0) + round(n * share)
    return sorted(counts.items())


def missing(vocab):
    """Tier tokens the level net's vocabulary doesn't have."""
    return sorted({t for tier in TIERS for t in BASE + tier if t not in vocab.stoi})


def next_tier(tier, solved_share, yield_share):
    """The tier for the next round, and why."""
    if solved_share is None:
        return tier, "no levels at this tier"
    if solved_share >= UP and yield_share >= YIELD and tier + 1 < len(TIERS):
        return tier + 1, f"net solved {solved_share:.0%} (>= {UP:.0%}): up"
    if solved_share < DOWN and tier > 0:
        return tier - 1, f"net solved {solved_share:.0%} (< {DOWN:.0%}): down"
    return tier, f"net solved {solved_share:.0%}, yield {yield_share:.0%}: stay"
