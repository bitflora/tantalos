"""The chamber built so far, as a voxel grid, for the state-conditioned arm (arm 6 of
docs/architecture-plan-2026-11-15.md, model.State): a small 3D-conv encoder reads the grid and
the decoder cross-attends to it.

A level's tokens (src/tokens.rs: `size`, `open`/`goo` boxes, `white`/`black` boxes per direction,
then the items) become three uint8 grids, x-major, at most 32 a side:

- `cell`: 0 outside the chamber, 1 solid, 2 open, 3 open goo;
- `white`: per open cell, a bit per side (+x -x +y -y +z -z) with a white (portal) surface there;
- `item`: what stands in the cell, an item word and the way it faces (`code`), or a part of one
  (a glass box's cells, a faith plate's target, a rail's far end). A later item overwrites.

The geometry is read the way `tokens::decode` reads it (`tantalos-score grid` prints Rust's own,
and `check` compares them: `python train/state.py check`).

Snapshots are taken at item boundaries: the grid of the tokens before an item's word. In training
(`features`) a sequence gets up to `SNAPSHOTS` of them at random boundaries, and each token after a
boundary attends to the latest snapshot at or before it (`which`); in sampling (`Rows`) a row's
grid is rebuilt every time it starts an item, so it always sees every item written before.

    .venv/Scripts/python train/state.py check [--n 1000]     # against tantalos-score grid
"""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import spatial  # noqa: E402
from data import SEP  # noqa: E402

GRID = 32
SNAPSHOTS = 4  # per training sequence: 8 ran at 20k tok/s on the 3050, 4 at 36k (the GPT: 56k)
DIRS = ["+x", "-x", "+y", "-y", "+z", "-z"]
DIR = {d: k for k, d in enumerate(DIRS)}
STEP = {"+x": (0, 1), "-x": (0, -1), "+y": (1, 1), "-y": (1, -1), "+z": (2, 1), "-z": (2, -1)}
OUTSIDE, SOLID, OPEN, GOO = range(4)
CELLS = 4
WHITES = 64

ITEM_WORDS = sorted(spatial.ITEM_WORDS)
PARTS = ["glass.box", "plate.target", "rail.end"]
# Item code: 1 + word * 7 + facing (0-5, 6: none); then the parts.
CODES = 1 + 7 * len(ITEM_WORDS) + len(PARTS)


def code(word, facing=None):
    return 1 + 7 * ITEM_WORDS.index(word) + (DIR[facing] if facing in DIR else 6)


def part(name):
    return 1 + 7 * len(ITEM_WORDS) + PARTS.index(name)


def _int(t):
    return int(t) if t.isdigit() else None


class Level:
    """A level's tokens read once: its geometry grids and its items in order, each with where its
    word stands (`starts`, token index) and the cells it paints. `ok` is False if reading stopped
    early (tokens no level has); what was read before stays."""

    def __init__(self, toks):
        self.ok = True
        self.starts, self.paints = [], []
        self.size = None
        try:
            i = self._geometry(toks)
            self._items(toks, i)
        except (IndexError, ValueError, KeyError):
            self.ok = False
        if not hasattr(self, "white"):  # no geometry read: an empty grid
            self.size = (1, 1, 1)
            self.cell = np.zeros((1, 1, 1), np.uint8)
            self.white = np.zeros((1, 1, 1), np.uint8)

    def _geometry(self, toks):
        if toks[0] != "size":
            raise ValueError("no size")
        size = tuple(int(t) for t in toks[1:4])
        if not all(1 <= s <= GRID for s in size):
            raise ValueError("size")
        self.size = size
        opened = np.zeros(size, bool)
        goo = np.zeros(size, bool)
        # faces[a][i, j, k]: the face on axis a below cell index i (between i-1 and i).
        faces = [np.zeros(tuple(s + (a == b) for b, s in enumerate(size)), bool) for a in range(3)]
        i = 4
        while i < len(toks) and toks[i] in ("open", "goo", "white", "black"):
            word = toks[i]
            i += 1
            d = None
            if word in ("white", "black"):
                d = toks[i]
                i += 1
                if word == "black":
                    self._paint(faces, (0, 0, 0), tuple(s - 1 for s in size), d, True)
            while i + 6 <= len(toks) and all(t.isdigit() for t in toks[i : i + 6]):
                lo, hi = tuple(int(t) for t in toks[i : i + 3]), tuple(int(t) for t in toks[i + 3 : i + 6])
                i += 6
                if any(lo[a] > hi[a] or hi[a] >= size[a] for a in range(3)):
                    raise ValueError("box")
                box = tuple(slice(lo[a], hi[a] + 1) for a in range(3))
                if word == "open":
                    opened[box] = True
                elif word == "goo":
                    opened[box] = True
                    goo[box] = True
                else:
                    self._paint(faces, lo, hi, d, word == "white")
        self.cell = np.where(opened, np.where(goo, GOO, OPEN), SOLID).astype(np.uint8)
        white = np.zeros(size, np.uint8)
        solid = np.pad(~opened, 1, constant_values=True)  # outside is solid
        for k, d in enumerate(DIRS):
            a, s = STEP[d]
            # The neighbour on side d of every cell, and the face between them.
            nb = solid[tuple(slice(1 + (s if b == a else 0), 1 + (s if b == a else 0) + size[b]) for b in range(3))]
            f = faces[a][tuple(slice(1, size[b] + 1) if (b == a and s > 0) else slice(0, size[b]) for b in range(3))]
            white |= ((opened & nb & f).astype(np.uint8) << k)
        self.white = white
        return i

    @staticmethod
    def _paint(faces, lo, hi, d, v):
        a, s = STEP[d]
        sl = tuple(slice(lo[b] + (s > 0 and b == a), hi[b] + 1 + (s > 0 and b == a)) for b in range(3))
        faces[a][sl] = v

    def _items(self, toks, i):
        size = self.size
        n = len(toks)

        def cell(j):
            c = tuple(int(t) for t in toks[j : j + 3])
            if len(c) != 3 or any(not (0 <= c[a] < size[a]) for a in range(3)):
                raise ValueError("cell")
            return c

        while i < n:
            word = toks[i]
            if word not in spatial.ITEM_WORDS:
                raise ValueError(f"not an item: {word}")
            fields = spatial.ITEMS[spatial.SAME.get(word, word)]
            start, i = i, i + 1
            vals = toks[i : i + len(fields)]
            if len(vals) < len(fields):
                raise ValueError("short item")
            named = {}
            for f, v in zip(fields, vals):
                key = f if isinstance(f, str) else f[0]
                named[key] = v
            i += len(fields)
            if word == "gel" and named["mode"] == "stream":  # a poured gel's dropper: on or off
                i += 1
            facing = next((named[f[0]] for f in fields if not isinstance(f, str) and named[f[0]] in DIR), None)
            at = cell(start + 1)
            paints = []
            base = spatial.SAME.get(word, word)
            if base == "glass":
                far = cell(start + 4)
                lo, hi = tuple(min(at[a], far[a]) for a in range(3)), tuple(max(at[a], far[a]) for a in range(3))
                for x in range(lo[0], hi[0] + 1):
                    for y in range(lo[1], hi[1] + 1):
                        for z in range(lo[2], hi[2] + 1):
                            paints.append(((x, y, z), part("glass.box")))
            elif base == "plate":
                paints.append((cell(start + 5), part("plate.target")))
            elif base == "rail":
                paints.append((cell(start + 4), part("rail.end")))
            paints.append((at, code(word, facing)))
            if i < n and toks[i] == spatial.OPTIONAL.get(word):
                i += 1
            while i < n and toks[i] == ">":
                target = toks[i + 1]
                if target not in spatial.TARGETS:
                    raise ValueError("wire target")
                i += 2 if target == "exit" else 3
            self.starts.append(start)
            self.paints.append(paints)

    def grids(self, before):
        """(cell, white, item) of the level with the items whose word stands before token index
        `before` (shape: the chamber's size)."""
        item = np.zeros(self.size, np.uint8)
        for s, paints in zip(self.starts, self.paints):
            if s >= before:
                break
            for c, v in paints:
                item[c] = v
        return self.cell, self.white, item


def level_tokens(ids, itos):
    """The tokens of a sequence's level part (after the first <sep>; ids from BOS or left-padded)
    and the index of its first token in `ids` (None: no <sep> yet)."""
    if SEP not in ids:
        return [], None
    at = ids.index(SEP) + 1
    return [itos[i] for i in ids[at:]], at


def snapshots(ids, itos, limit=SNAPSHOTS, rng=random):
    """A sequence's snapshots: [(position, (cell, white, item))] at item boundaries, at most
    `limit` of them (chosen at random, None: all), in order. The token at `position` (the item's
    word) and every one after it see that grid, until the next snapshot."""
    toks, at = level_tokens(ids, itos)
    if at is None or not toks:
        return []
    lv = Level(toks)
    starts = lv.starts
    if limit is not None and len(starts) > limit:
        starts = sorted(rng.sample(starts, limit))
    return [(at + s, lv.grids(s)) for s in starts]


class Grids:
    """A batch's snapshots for model.State: `grids` (N x 3 x X x Y x Z, uint8; X, Y, Z the batch's
    largest, rounded up to 4), `sizes` (N x 3), `owner` (N: the row), `slot` (N: 1 for a row's
    first snapshot, ...), `which` (B x T: the slot each token sees, 0 none); `slots` (an int,
    the most a row has, known on the CPU so the model needn't wait on the GPU for it)."""

    def __init__(self, grids, sizes, owner, slot, which, slots=None):
        self.grids, self.sizes, self.owner, self.slot, self.which = grids, sizes, owner, slot, which
        self.slots = int(slot.max()) if slots is None and slot.numel() else (slots or 0)

    def to(self, device, non_blocking=False):
        return Grids(*(t.to(device, non_blocking=non_blocking) for t in
                       (self.grids, self.sizes, self.owner, self.slot, self.which)), slots=self.slots)

    def __getitem__(self, key):
        """The same snapshots for some of the tokens (`which[key]`)."""
        return Grids(self.grids, self.sizes, self.owner, self.slot, self.which[key], slots=self.slots)


def stack(snaps):
    """Snapshots [(cell, white, item)] to (N x 3 x X x Y x Z uint8, N x 3 sizes) tensors, padded
    with zeros (outside) to the largest, rounded up to 4."""
    import torch

    if not snaps:
        return torch.zeros((0, 3, 4, 4, 4), dtype=torch.uint8), torch.zeros((0, 3), dtype=torch.long)
    dims = [max(g[0].shape[a] for g in snaps) for a in range(3)]
    dims = [(d + 3) // 4 * 4 for d in dims]
    out = np.zeros((len(snaps), 3, *dims), np.uint8)
    sizes = np.zeros((len(snaps), 3), np.int64)
    for n, g in enumerate(snaps):
        x, y, z = g[0].shape
        for c in range(3):
            out[n, c, :x, :y, :z] = g[c]
        sizes[n] = (x, y, z)
    return torch.from_numpy(out), torch.from_numpy(sizes)


def features(ids, itos, limit=SNAPSHOTS, rng=random):
    """Grids for a batch of ids (B x T tensor, CPU)."""
    import torch

    rows = ids.tolist()
    which = torch.zeros(ids.shape, dtype=torch.long)
    snaps, owner, slot = [], [], []
    for b, row in enumerate(rows):
        for k, (pos, g) in enumerate(snapshots(row, itos, limit, rng)):
            snaps.append(g)
            owner.append(b)
            slot.append(k + 1)
            which[b, pos:] = k + 1
    grids, sizes = stack(snaps)
    return Grids(grids, sizes, torch.tensor(owner, dtype=torch.long), torch.tensor(slot, dtype=torch.long), which)


class Rows:
    """A sampled batch's rows (model.Decoder's samplers): each keeps its ids and, when it starts an
    item, the grid of what came before (`fresh`: the rows whose grid changed since the model last
    took them, with their grids)."""

    def __init__(self, itos, b):
        self.itos = itos
        self.ids = [None] * b
        self.grid = [None] * b
        self.fresh = {}
        self.prev = [None] * b  # each row's last token, to tell an item from a wire target
        self.started = [False] * b  # past its prompt's <sep>

    def start(self, r, ids):
        self.ids[r] = list(ids)
        snaps = snapshots(self.ids[r], self.itos, limit=None)
        self.grid[r] = snaps[-1][1] if snaps else None
        self.fresh[r] = self.grid[r]
        self.prev[r] = self.itos[ids[-1]] if ids else None
        self.started[r] = SEP in ids

    def feed(self, toks, rows=None):
        for r, t in enumerate(toks):
            if (rows is not None and r not in rows) or self.ids[r] is None:
                continue
            word = self.itos[t]
            if self.started[r] and word in spatial.ITEM_WORDS and self.prev[r] != ">":
                toks_r, _ = level_tokens(self.ids[r], self.itos)
                lv = Level(toks_r)
                self.grid[r] = lv.grids(len(toks_r))
                self.fresh[r] = self.grid[r]
            self.ids[r].append(t)
            self.prev[r] = word
            if t == SEP:
                self.started[r] = True

    def take(self):
        """The rows whose grid changed, and their grids (None: no grid yet), and forget them."""
        out, self.fresh = self.fresh, {}
        return out


def rust_grids(levels, tantalos):
    """`tantalos-score grid` for each level's tokens: (cell, white) arrays or an error string."""
    text = "".join(json.dumps({"tokens": t}) + "\n" for t in levels)
    out = subprocess.run([str(tantalos), "grid"], input=text, capture_output=True, text=True, check=True).stdout
    res = []
    for line in out.splitlines():
        r = json.loads(line)
        if "error" in r:
            res.append(r["error"])
            continue
        size = tuple(r["size"])
        v = np.frombuffer(bytes.fromhex(r["cells"]), np.uint8).reshape(size)
        cell = np.where(v & 1, SOLID, np.where(v & 2, GOO, OPEN)).astype(np.uint8)
        res.append((cell, (v >> 2).astype(np.uint8)))
    return res


def check(args):
    """Our grid of each level's geometry (the tokens before its first item: the grammar's state
    there) and of the whole level against `tantalos-score grid`, on `n` levels spread over the
    data files."""
    import glob

    files = sorted(glob.glob(str(ROOT / args.data)))
    rng = random.Random(args.seed)
    picks = []
    per = 2 * -(-args.n // len(files))  # then n of them: a small file gives what it has
    for f in files:
        lines = Path(f).read_text(encoding="utf-8").splitlines() if Path(f).stat().st_size < 2e8 else None
        if lines is None:
            lines = []
            with open(f, encoding="utf-8") as fh:
                for k, line in enumerate(fh):
                    if k >= 200_000:
                        break
                    lines.append(line)
        for line in rng.sample(lines, min(per, len(lines))):
            picks.append(json.loads(line)["tokens"])
    picks = rng.sample(picks, min(args.n, len(picks)))
    geo, whole = [], []
    for t in picks:
        toks = t.split()
        lv = Level(toks)
        first = lv.starts[0] if lv.starts else len(toks)
        geo.append(" ".join(toks[:first]))
        whole.append(t)
    bad = {"geometry": 0, "whole": 0}
    shown = 0
    for name, levels in (("geometry", geo), ("whole", whole)):
        for t, r in zip(levels, rust_grids(levels, args.tantalos)):
            lv = Level(t.split())
            if isinstance(r, str):
                bad[name] += 1
                if shown < 5:
                    print(f"{name}: rust error {r}: {t[:120]}")
                    shown += 1
                continue
            same = lv.ok and lv.cell.shape == r[0].shape and (lv.cell == r[0]).all() and (lv.white == r[1]).all()
            if not same:
                bad[name] += 1
                if shown < 5:
                    shown += 1
                    if lv.cell.shape != r[0].shape:
                        print(f"{name}: shape {lv.cell.shape} vs {r[0].shape}")
                    else:
                        dc = np.argwhere(lv.cell != r[0])
                        dw = np.argwhere(lv.white != r[1])
                        print(f"{name}: {len(dc)} cells, {len(dw)} white differ; first "
                              f"{dc[:3].tolist()} {dw[:3].tolist()}: ours {[int(lv.white[tuple(c)]) for c in dw[:3]]} "
                              f"rust {[int(r[1][tuple(c)]) for c in dw[:3]]}: {t[:160]}")
    n = len(picks)
    print(f"{n} levels: geometry grids differ on {bad['geometry']}, whole levels on {bad['whole']}")
    return 0 if bad["geometry"] == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--n", type=int, default=1000)
    c.add_argument("--data", default="data/v5b/levels-*.jsonl")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--tantalos", default=str(ROOT / "target" / "release" / "tantalos-score.exe"))
    args = ap.parse_args()
    return check(args)


if __name__ == "__main__":
    sys.exit(main())
