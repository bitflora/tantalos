"""How `--seed` behaves for sampling and evaluation.

A number repeats a batch exactly. Leaving it out asks for a fresh, unpredictable one: a fixed
default made warm-started checkpoints draw from the same random stream, so consecutive runs
sampled near-neighbours (the same "glados sample 19" slot again and again). Whichever seed is
used is printed, so a batch worth keeping can be reproduced with --seed.
"""

import secrets


def resolve(seed):
    """`seed`, or a fresh unpredictable one if it is None. Prints the result."""
    if seed is None:
        seed = secrets.randbits(32)
    print(f"seed: {seed}", flush=True)
    return seed
