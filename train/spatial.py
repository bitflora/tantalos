"""Per-token level structure for the spatial-embedding arm (arm 2 of
docs/architecture-plan-2026-11-15.md): for every token of a training or sampled sequence, the
field it fills (its *role*: `size.x`, `open.y1`, `panel.F`, `wire.index`, ...) and the cell it is
about (x, y, z, each 0-32 or NONE).

`Parser` reads a sequence token by token, from what has been written only, so teacher forcing and
incremental decoding see the same features at every prefix. It follows the token format of
src/tokens.rs in the order `tokens::encode` writes and `src/grammar.rs` constrains: `size`,
`open`/`goo` boxes, `white`/`black` per direction, then the items with their fields, optional
words (`inv`, `wait`, `rev`) and wires (`> panel 2`). It never fails: a token it can't place
(unconstrained model output) gets the role `junk`, and the next item word starts over.

The cell of a token: the coordinates written so far of the cell (or box corner) its record is
writing, the earlier ones of a partly written cell filled in, the rest NONE; for a wire's target
(`> panel 2`: the `2`; `> exit`: `exit`) the cell of the item it names, if written already.
Size, conditions and specials have no cell.

`expect()` gives the tokens the parser allows next (a superset of the grammar's), so
tests/test_spatial.py can check it against `tantalos-score grammar` level by level.
"""
from data import BOS, EOS, PAD, SEP

NONE = 33  # "no coordinate", the last row of each coordinate embedding
COORDS = 34

NUMS = frozenset(str(i) for i in range(33))
DIRS = frozenset(["+x", "-x", "+y", "-y", "+z", "-z"])
CUBE_KINDS = frozenset(["standard", "companion", "reflection", "sphere", "franken"])
BUTTON_KINDS = frozenset(["weighted", "cube", "sphere"])
PAINT_KINDS = frozenset(["bounce", "speed", "portal", "water"])
ON_OFF = frozenset(["on", "off"])

# Field kinds: a coordinate of the record's first (x y z) or second (X Y Z) cell, or the words
# the field takes.
X, Y, Z, X2, Y2, Z2 = "x", "y", "z", "x2", "y2", "z2"
COORD = {X: (0, 0), Y: (0, 1), Z: (0, 2), X2: (1, 0), Y2: (1, 1), Z2: (1, 2)}
CELL = [X, Y, Z]
ITEMS = {
    "entry": CELL + [("D", DIRS)],
    "exit": CELL + [("D", DIRS)],
    "glass": CELL + [X2, Y2, Z2, ("N", DIRS), ("G", DIRS)],
    "fizzler": CELL + [("N", DIRS), ("A", DIRS), ("L", NUMS), ("state", ON_OFF)],
    "gel": CELL + [("N", DIRS), ("K", PAINT_KINDS), ("mode", frozenset(["stream", "splat"]))],
    "dropper": CELL + [("K", CUBE_KINDS)],
    "button": CELL + [("K", BUTTON_KINDS)],
    "pedestal": CELL + [("S", NUMS | {"inf"})],
    "turret": CELL + [("F", NUMS)],
    "cube": CELL + [("K", CUBE_KINDS)],
    "laser": CELL + [("N", DIRS), ("O", DIRS | {"center"}), ("state", ON_OFF)],
    "catcher": CELL + [("N", DIRS), ("O", DIRS | {"center"})],
    "plate": CELL + [("N", DIRS), X2, Y2, Z2, ("N2", DIRS)],
    "bridge": CELL + [("N", DIRS), ("S", DIRS), ("state", ON_OFF)],
    "funnel": CELL + [("N", DIRS), ("state", ON_OFF)],
    "panel": CELL + [("N", DIRS), ("F", DIRS), ("A", NUMS), ("state", ON_OFF)],
    "flip": CELL + [("N", DIRS)],
    "stairs": CELL + [("F", DIRS), ("state", ON_OFF)],
    "piston": CELL + [("N", DIRS), ("B", NUMS), ("T", NUMS), ("state", ON_OFF)],
    "rail": CELL + [X2, Y2, Z2, ("mode", frozenset(["osc", "on", "off"]))],
}
# Words that share another item's fields (and roles).
SAME = {"grating": "glass", "laserfield": "fizzler", "relay": "catcher", "glasspanel": "panel"}
ITEM_WORDS = frozenset(ITEMS) | frozenset(SAME)
# An optional last word, and which items send wires.
OPTIONAL = {"exit": "inv", "dropper": "wait", "funnel": "rev"}
SENDERS = frozenset(["button", "pedestal", "catcher", "relay"])
# Wire targets and the item words whose items they count (`field 1`: the second fizzler or laser
# field; `polarity 0`: the first funnel).
TARGETS = {
    "exit": ("exit",), "field": ("fizzler", "laserfield"), "gel": ("gel",), "laser": ("laser",),
    "bridge": ("bridge",), "funnel": ("funnel",), "polarity": ("funnel",), "panel": ("panel",),
    "glasspanel": ("glasspanel",), "flip": ("flip",), "stairs": ("stairs",), "piston": ("piston",),
    "rail": ("rail",), "dropper": ("dropper",),
}
BOXES = ("x0", "y0", "z0", "x1", "y1", "z1")
GEOM_WORDS = frozenset(["open", "goo", "white", "black"])
NEXT_RECORD = GEOM_WORDS | ITEM_WORDS


def _roles():
    roles = ["pad", "bos", "cond", "sep", "eos", "junk", "size", "size.x", "size.y", "size.z"]
    for sec in ("open", "goo", "white", "black"):
        roles.append(sec)
        if sec in ("white", "black"):
            roles.append(f"{sec}.dir")
        roles += [f"{sec}.{b}" for b in BOXES]
    for word, fields in ITEMS.items():
        roles.append(word)
        roles += [f"{word}.{f if isinstance(f, str) else f[0]}" for f in fields]
    roles += [f"{w}.{o}" for w, o in OPTIONAL.items()]
    roles += ["gel.state", "wire", "wire.kind", "wire.index"]
    return roles


ROLES = _roles()
ROLE = {r: i for i, r in enumerate(ROLES)}


class Parser:
    """The roles and cells of a sequence's tokens, one `feed` at a time (see the module doc)."""

    __slots__ = ("phase", "word", "fields", "fi", "cells", "corner", "boxi", "wire", "items", "sec")

    def __init__(self):
        self.phase = "start"  # start, cond (after BOS), level (after SEP), end (after EOS)
        self.word = None  # the record being written: an item word (as in ITEMS), or a geometry one
        self.sec = None  # the item word as written (`grating` for a glass-like record)
        self.fields = ()  # the record's fields
        self.fi = 0  # fields written so far
        self.cells = [[NONE] * 3, [NONE] * 3]  # the record's first and second cell
        self.corner = 0  # which of them is being written
        self.boxi = 0  # numbers written in the current box group
        self.wire = None  # in a wire: ">" (target word next) or the target word (index next)
        self.items = {}  # item word as written -> the first cells of its items so far

    def copy(self):
        p = Parser.__new__(Parser)
        p.phase, p.word, p.sec, p.fields, p.fi = self.phase, self.word, self.sec, self.fields, self.fi
        p.cells = [self.cells[0][:], self.cells[1][:]]
        p.corner, p.boxi, p.wire = self.corner, self.boxi, self.wire
        p.items = {k: v[:] for k, v in self.items.items()}
        return p

    def _cell(self):
        return tuple(self.cells[self.corner])

    def _start(self, word):
        """A new record: an item word or a geometry word."""
        self.cells = [[NONE] * 3, [NONE] * 3]
        self.corner, self.fi, self.boxi, self.wire = 0, 0, 0, None
        self.sec = word
        if word in GEOM_WORDS or word == "size":
            self.word, self.fields = word, ()
            return ROLE[word]
        self.word = SAME.get(word, word)
        self.fields = ITEMS[self.word]
        if self.word == "gel":
            self.fields = list(self.fields)
        return ROLE[self.word]

    def feed(self, tok):
        """The token's (role, x, y, z), and the parser moved past it. `tok` is a string; the
        specials come as ids (`feed_id`)."""
        if self.phase != "level":
            return ROLE["junk"], NONE, NONE, NONE
        word = self.word
        if word is None:
            if tok == "size":
                self._start("size")
                return ROLE["size"], NONE, NONE, NONE
            return self._other(tok)
        if word == "size":
            if self.fi < 3 and tok in NUMS:
                self.fi += 1
                return ROLE[("size.x", "size.y", "size.z")[self.fi - 1]], NONE, NONE, NONE
            return self._other(tok)
        if word in GEOM_WORDS:
            if word in ("white", "black") and self.fi == 0:
                if tok in DIRS:
                    self.fi = 1
                    return ROLE[f"{word}.dir"], NONE, NONE, NONE
                return self._other(tok)
            if tok in NUMS:
                i = self.boxi % 6
                self.boxi += 1
                if i == 0:
                    self.cells = [[NONE] * 3, [NONE] * 3]
                self.corner = i // 3
                self.cells[self.corner][i % 3] = int(tok)
                return ROLE[f"{word}.{BOXES[i]}"], *self._cell()
            return self._other(tok)
        if word == "junk":
            return self._other(tok)
        # An item.
        if self.wire is not None:
            return self._wire(tok)
        if self.fi < len(self.fields):
            f = self.fields[self.fi]
            if isinstance(f, str):
                if tok not in NUMS:
                    return self._other(tok)
                self.fi += 1
                c, a = COORD[f]
                self.corner = c
                self.cells[c][a] = int(tok)
                if f == Z:
                    self.items.setdefault(self.sec, []).append(tuple(self.cells[0]))
                return ROLE[f"{word}.{f}"], *self._cell()
            name, allowed = f
            if tok in ITEM_WORDS and tok not in allowed:
                return self._other(tok)
            self.fi += 1
            if word == "gel" and name == "mode" and tok == "stream":
                self.fields.append(("state", ON_OFF))
            role = f"{word}.{name}" if tok in allowed else "junk"
            return ROLE[role], *self._cell()
        # The record's fields are done: an optional word, a wire, or the next record.
        if tok == OPTIONAL.get(word) and self.fi == len(self.fields):
            self.fi += 1  # once
            return ROLE[f"{word}.{tok}"], *self._cell()
        if tok == ">" and self.sec in SENDERS:
            self.wire = ">"
            return ROLE["wire"], *self._cell()
        return self._other(tok)

    def _wire(self, tok):
        if self.wire == ">":
            if tok in TARGETS:
                if tok == "exit":
                    self.wire = None
                    cells = self.items.get("exit")
                    return ROLE["wire.kind"], *(cells[0] if cells else (NONE, NONE, NONE))
                self.wire = tok
                return ROLE["wire.kind"], *self._cell()
            self.wire = None
            return self._other(tok)
        if tok in NUMS:
            cells = [c for w in TARGETS[self.wire] for c in self.items.get(w, ())]
            self.wire = None
            i = int(tok)
            return ROLE["wire.index"], *(cells[i] if i < len(cells) else (NONE, NONE, NONE))
        self.wire = None
        return self._other(tok)

    def _other(self, tok):
        """A token that doesn't continue the record: the next record, or junk."""
        if tok in NEXT_RECORD:
            return self._start(tok), NONE, NONE, NONE
        self.word, self.fields, self.fi = "junk", (), 0
        return ROLE["junk"], NONE, NONE, NONE

    def feed_id(self, i, itos):
        if i == PAD:
            return ROLE["pad"], NONE, NONE, NONE
        if i == BOS:
            self.phase = "cond"
            return ROLE["bos"], NONE, NONE, NONE
        if i == SEP:
            self.phase = "level"
            return ROLE["sep"], NONE, NONE, NONE
        if i == EOS:
            self.phase = "end"
            return ROLE["eos"], NONE, NONE, NONE
        if self.phase == "cond":
            return ROLE["cond"], NONE, NONE, NONE
        return self.feed(itos[i])

    def expect(self):
        """The level tokens this parser could place next (in the level phase): what the
        grammar allows must be among them."""
        word = self.word
        if word is None:
            return frozenset(["size"])
        if word == "size":
            return NUMS if self.fi < 3 else frozenset(["open"])
        if word in GEOM_WORDS:
            if word in ("white", "black") and self.fi == 0:
                return DIRS
            if self.boxi % 6 or (self.boxi == 0 and word != "black"):
                return NUMS
            return NUMS | NEXT_RECORD
        if word == "junk":
            return NEXT_RECORD
        if self.wire == ">":
            return frozenset(TARGETS)
        if self.wire is not None:
            return NUMS
        if self.fi < len(self.fields):
            f = self.fields[self.fi]
            return NUMS if isinstance(f, str) else f[1]
        out = ITEM_WORDS
        if word in OPTIONAL and self.fi == len(self.fields):
            out = out | {OPTIONAL[word]}
        if self.sec in SENDERS:
            out = out | {">"}
        return out


def features(rows, itos):
    """Features of whole sequences (lists of ids from BOS, or left-padded with PAD): a list per
    row of (role, x, y, z) per token."""
    out = []
    for ids in rows:
        p = Parser()
        out.append([p.feed_id(i, itos) for i in ids])
    return out


class Rows:
    """Parsers for the rows of a batch being sampled (model.Decoder's samplers): `start` a row on
    a prompt, `feed` each row the token it wrote."""

    def __init__(self, itos, b):
        self.itos, self.rows = itos, [None] * b
        self.memo = {}

    def prompt(self, ids):
        """A parser after `ids`, and their features (memoized: samplers start many rows on one
        prompt)."""
        key = tuple(ids)
        if key not in self.memo:
            p = Parser()
            feats = [p.feed_id(i, self.itos) for i in ids]
            self.memo = {key: (p, feats)}
        p, feats = self.memo[key]
        return p.copy(), feats

    def start(self, r, ids):
        self.rows[r] = self.prompt(ids)[0]

    def feed(self, toks, rows=None):
        """Each row's (role, x, y, z) for the token it writes (`rows`: only those; the rest get
        `pad` and keep their state)."""
        out = []
        for r, t in enumerate(toks):
            if (rows is None or r in rows) and self.rows[r] is not None:
                out.append(self.rows[r].feed_id(t, self.itos))
            else:
                out.append((ROLE["pad"], NONE, NONE, NONE))
        return out
