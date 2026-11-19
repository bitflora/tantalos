"""The device-and-wire form of levels (arm 4 of docs/architecture-plan-2026-11-15.md, from Codex's
plan): the same chamber, but every wire is written after all the items, as a pair of **pointers**
to the devices it joins, instead of inside its sender's record as `> panel 2`.

    flat:  size .. geometry .. button 2 2 1 cube > exit > field 0  fizzler 3 2 1 ..  exit ..
    wire:  size .. geometry .. button 2 2 1 cube  fizzler 3 2 1 ..  exit ..  > @0 @2 > @0 @1

`@d` is device d: the d-th item record of the level (every item word counts, wirable or not), so
a pointer names a device directly rather than by its kind and count. A wire to a funnel names
which of its inputs with one more word, `funnel` (on/off) or `polarity`. Wires come grouped by
sender in device order, each sender's in the order `tokens::encode` writes them, so `to_wire` and
`compile` are inverses on every level `encode` wrote (`check` tests that on the training data).

The model (model.Wire) has one output per word and one per device slot (`@d` is id V + d): a
pointer is scored by attention from the current position to the last token of each device's
record, and only devices that can take that end of a wire count (a button can't drive a button):
- **senders** (the first pointer): button, pedestal, catcher, relay;
- **receivers** (the second): exit, fizzler/laserfield, a poured gel, laser, bridge, funnel,
  panel, glasspanel, flip, stairs, piston, rail, dropper.
`Tracker` follows a sequence token by token and gives each token's features (the device it
belongs to, which kind of pointer may come next, the device whose record it closes), the same
for teacher forcing and sampling. `compile` turns the model's output back into the flat tokens
everything else reads; what can't compile is counted, never dropped.

    .venv/Scripts/python train/wire.py check            # readiness: round trip of the training data
"""
import argparse
import glob
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import spatial  # noqa: E402
from data import BOS, EOS, PAD, SEP  # noqa: E402
from spatial import GEOM_WORDS, ITEMS, ROLE, ROLES, SENDERS, Parser  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
POINTERS = 160  # device slots: more than any training level has (`check` reports the most)

# Wire targets in `tokens::encode`'s order: a sender's wires are written sorted by (this, index).
ORDER = ["exit", "field", "gel", "laser", "bridge", "funnel", "polarity", "panel", "glasspanel",
         "flip", "stairs", "piston", "rail", "dropper"]
RANK = {w: i for i, w in enumerate(ORDER)}
# The target word of a receiving device (by its item word as written), and the devices it counts.
TARGET_OF = {"exit": "exit", "fizzler": "field", "laserfield": "field", "gel": "gel", "laser": "laser",
             "bridge": "bridge", "funnel": "funnel", "panel": "panel", "glasspanel": "glasspanel",
             "flip": "flip", "stairs": "stairs", "piston": "piston", "rail": "rail", "dropper": "dropper"}
FUNNEL_KINDS = ("funnel", "polarity")
NONE, SENDER, RECEIVER = 0, 1, 2
ITEM_ROLES = frozenset(ROLE[w] for w in ITEMS)
OTHER_STARTS = frozenset(ROLE[w] for w in GEOM_WORDS | {"size"})
WIRE_ROLES = frozenset(ROLE[w] for w in ("wire", "wire.kind", "wire.index"))


class WireError(ValueError):
    pass


def ptr(d):
    return f"@{d}"


def is_ptr(tok):
    return isinstance(tok, str) and tok.startswith("@") and tok[1:].isdigit()


def device_class(word, toks):
    """Which end of a wire a device (its item word as written, its record's tokens) can be."""
    if word in SENDERS:
        return SENDER
    if word == "gel":
        return RECEIVER if "stream" in toks else NONE  # only a poured gel takes input
    return RECEIVER if word in TARGET_OF else NONE


def records(tokens):
    """A flat level's (or a wire level's device part's) records: the head (size, geometry) and one
    (word, tokens, wires) per item, wires as (target word, index or None). Never fails: tokens the
    parser can't place stay in their record, for `tantalos-score` to reject."""
    p = Parser()
    p.phase = "level"
    head, devices = [], []
    in_wire = None
    for t in tokens:
        role = p.feed(t)[0]
        if role in ITEM_ROLES:
            devices.append((t, [t], []))
            continue
        if role in WIRE_ROLES and devices:
            if t == ">":
                in_wire = devices[-1][2]
            elif role == ROLE["wire.kind"]:
                in_wire.append([t, None])
            else:
                in_wire[-1][1] = int(t)
            continue
        (devices[-1][1] if devices and role not in OTHER_STARTS else head).append(t)
        if role in OTHER_STARTS and devices:
            raise WireError(f"geometry word {t!r} after the items")
    return head, devices


def to_wire(tokens):
    """A flat level's tokens in wire form (WireError for a wire to a device that doesn't exist)."""
    head, devices = records(tokens)
    lists = {}  # target word -> its devices in order
    for d, (word, toks, _) in enumerate(devices):
        if word in TARGET_OF:
            lists.setdefault(TARGET_OF[word], []).append(d)
    lists["polarity"] = lists.get("funnel", [])
    out = list(head)
    wires = []
    for d, (word, toks, ws) in enumerate(devices):
        out += toks
        for target, i in ws:
            cands = lists.get(target, [])
            j = 0 if i is None else i
            if j >= len(cands):
                raise WireError(f"wire to {target} {i}, which doesn't exist")
            w = [">", ptr(d), ptr(cands[j])]
            if target in FUNNEL_KINDS:
                w.append(target)
            wires += w
    return out + wires


def compile(tokens):
    """Wire-form tokens (pointers as `@d`) back to the flat level: each wire under its sender,
    sorted and each once, as `encode` writes them. WireError for a wire that can't be written:
    malformed, from a device that sends nothing, to one that takes no input, past the devices."""
    tokens = list(tokens)
    cut = tokens.index(">") if ">" in tokens else len(tokens)
    if any(is_ptr(t) for t in tokens[:cut]):
        raise WireError("a pointer among the items")
    head, devices = records(tokens[:cut])
    if any(ws for _, _, ws in devices):
        raise WireError("a flat wire among the items")
    count = {}
    index = []  # each device's (target word, index among its kind), or None
    for word, toks, _ in devices:
        t = TARGET_OF.get(word)
        if t is None:
            index.append(None)
            continue
        index.append((t, count.get(t, 0)))
        count[t] = count.get(t, 0) + 1
    wires = {}
    rest = tokens[cut:]
    i = 0
    while i < len(rest):
        if rest[i] != ">" or i + 2 >= len(rest) or not (is_ptr(rest[i + 1]) and is_ptr(rest[i + 2])):
            raise WireError(f"malformed wire at {' '.join(rest[i:i + 4])!r}")
        s, d = int(rest[i + 1][1:]), int(rest[i + 2][1:])
        i += 3
        if s >= len(devices) or d >= len(devices):
            raise WireError(f"wire @{s} -> @{d}: only {len(devices)} devices")
        if device_class(devices[s][0], devices[s][1]) != SENDER:
            raise WireError(f"wire from a {devices[s][0]}")
        if device_class(devices[d][0], devices[d][1]) != RECEIVER:
            raise WireError(f"wire to a {devices[d][0]}")
        target, k = index[d]
        if target == "funnel":
            if i >= len(rest) or rest[i] not in FUNNEL_KINDS:
                raise WireError("wire to a funnel without `funnel` or `polarity`")
            target = rest[i]
            i += 1
        wires.setdefault(s, set()).add((RANK[target], k))
    out = list(head)
    for d, (word, toks, _) in enumerate(devices):
        out += toks
        for r, k in sorted(wires.get(d, ())):
            out += [">", ORDER[r]] if ORDER[r] == "exit" else [">", ORDER[r], str(k)]
    return out


class Tracker:
    """One sequence's structure, a token at a time (`feed` an id; ids >= `v` are pointers): per
    token (dev, nxt, ended, cls)
    - dev: 1 + the device whose record the token is in (0: none);
    - nxt: which pointer must come next: SENDER, RECEIVER, or NONE (a word);
    - ended: the device whose record ended just before this token (its key is the previous
      token's), -1 if none; cls: that device's class (`device_class`).
    The prompt (BOS ... SEP) and anything after EOS are plain words."""

    __slots__ = ("itos", "v", "phase", "parser", "cur", "word", "toks", "n", "words", "classes", "expect")

    def __init__(self, itos, v):
        self.itos, self.v = itos, v
        self.phase = "start"
        self.parser = Parser()
        self.parser.phase = "level"
        self.cur = None  # the device being written
        self.word, self.toks = None, []
        self.n = 0
        self.words = []  # each device's item word
        self.classes = []  # each finished device's class
        self.expect = None  # in the wires: "src", "dst", "kind" or None (`>` or the end next)

    def copy(self):
        t = Tracker.__new__(Tracker)
        t.itos, t.v, t.phase, t.parser = self.itos, self.v, self.phase, self.parser.copy()
        t.cur, t.word, t.toks, t.n = self.cur, self.word, self.toks[:], self.n
        t.words, t.classes, t.expect = self.words[:], self.classes[:], self.expect
        return t

    def _end(self):
        """Close the device being written: (ended, cls)."""
        if self.cur is None:
            return -1, NONE
        d, cls = self.cur, device_class(self.word, self.toks)
        self.classes[d] = cls
        self.cur = None
        return d, cls

    def _nxt(self):
        return {"src": SENDER, "dst": RECEIVER}.get(self.expect, NONE) if self.phase == "wires" else NONE

    def feed(self, i):
        if i == PAD:
            return 0, NONE, -1, NONE
        if i == BOS:
            self.phase = "cond"
            return 0, NONE, -1, NONE
        if i == SEP:
            self.phase = "level"
            return 0, NONE, -1, NONE
        if self.phase in ("start", "cond", "end"):
            if i == EOS:
                self.phase = "end"
            return 0, NONE, -1, NONE
        if i == EOS:
            ended, cls = self._end()
            self.phase = "end"
            return 0, NONE, ended, cls
        if self.phase == "level":
            if i >= self.v:  # a pointer among the items: junk, the wires begin
                ended, cls = self._end()
                self.phase, self.expect = "wires", None
                return 0, NONE, ended, cls
            tok = self.itos[i]
            if tok == ">":
                ended, cls = self._end()
                self.phase, self.expect = "wires", "src"
                return 0, self._nxt(), ended, cls
            role = self.parser.feed(tok)[0]
            if role in ITEM_ROLES:
                ended, cls = self._end()
                self.cur, self.word, self.toks = self.n, tok, [tok]
                self.words.append(tok)
                self.classes.append(NONE)
                self.n += 1
                return (1 + self.cur if self.cur < POINTERS else 0), NONE, ended, cls
            if role in OTHER_STARTS or role == ROLE["junk"] and self.cur is not None and tok in spatial.NEXT_RECORD:
                ended, cls = self._end()
                return 0, NONE, ended, cls
            if self.cur is None:
                return 0, NONE, -1, NONE
            self.toks.append(tok)
            return (1 + self.cur if self.cur < POINTERS else 0), NONE, -1, NONE
        # The wires.
        if i >= self.v:
            d = i - self.v
            if self.expect == "src":
                self.expect = "dst"
            elif self.expect == "dst":
                self.expect = "kind" if d < len(self.words) and self.words[d] == "funnel" else None
            else:
                self.expect = None
            return 0, self._nxt(), -1, NONE
        tok = self.itos[i]
        self.expect = "src" if tok == ">" else None
        return 0, self._nxt(), -1, NONE


FEATS = 4


def features(rows, itos, v):
    """Features of whole id sequences (from BOS, or left-padded): rows x tokens x FEATS."""
    out = []
    for ids in rows:
        t = Tracker(itos, v)
        out.append([t.feed(i) for i in ids])
    return torch.tensor(out, dtype=torch.long).view(len(rows), -1, FEATS)


def recode(ids, flags, itos, v, block):
    """A flat training sequence (ids from BOS, level, maybe EOS) and a flag per token (learned,
    or pushed away) in wire form. Item tokens keep their flags; the wires take the first flat
    wire token's, EOS its own. Cut to `block` + 1 ids. WireError if it can't be written."""
    try:
        s = ids.index(SEP) + 1
    except ValueError:
        return ids, flags
    end = ids.index(EOS) if EOS in ids else len(ids)
    level = [itos[i] for i in ids[s:end]]
    head, devices = records(level)
    wired = None
    pos = s
    keep = list(range(s))  # flat positions of the tokens kept, in the new order
    p = Parser()
    p.phase = "level"
    for k, t in enumerate(level):
        role = p.feed(t)[0]
        if role in WIRE_ROLES:
            if wired is None:
                wired = flags[s + k]
        else:
            keep.append(s + k)
    wire_toks = to_wire(level)[len(keep) - s:]
    stoi = {t: j for j, t in enumerate(itos)}
    new_ids = [ids[j] for j in keep] + [v + int(t[1:]) if is_ptr(t) else stoi[t] for t in wire_toks]
    new_flags = [flags[j] for j in keep] + [wired] * len(wire_toks)
    if end < len(ids):
        new_ids.append(EOS)
        new_flags.append(flags[end])
    if any(i >= v + POINTERS for i in new_ids):
        raise WireError("more devices than pointer slots")
    return new_ids[: block + 1], new_flags[: block + 1]


class Rows:
    """Trackers for the rows of a batch being sampled (model.Decoder's samplers). A row started
    since the last step is `fresh`: model.Wire sets its pointer keys from its prompt. A row
    started after it was fed a token for the coming step (generate_refill feeds every row, then
    restarts the finished ones) is `void`: that token is from its last level and is ignored."""

    def __init__(self, itos, v, b):
        self.itos, self.v = itos, v
        self.rows = [None] * b
        self.prompts = [None] * b
        self.fresh, self.void, self.pending = set(), set(), set()
        self.memo = {}

    def start(self, r, ids):
        key = tuple(ids)
        if key not in self.memo:
            t = Tracker(self.itos, self.v)
            for i in ids:
                t.feed(i)
            self.memo = {key: t}
        self.rows[r] = self.memo[key].copy()
        self.prompts[r] = key
        self.fresh.add(r)
        if r in self.pending:
            self.void.add(r)

    def stepped(self):
        """The step these rows were fed for has run."""
        self.fresh.clear()
        self.void.clear()
        self.pending.clear()

    def feed(self, toks, rows=None):
        """Per row: (dev, nxt, ended, cls, fed) for the token it writes (`rows`: only those)."""
        out = []
        for r, t in enumerate(toks):
            if (rows is None or r in rows) and self.rows[r] is not None:
                self.pending.add(r)
                out.append((*self.rows[r].feed(t), 1))
            else:
                out.append((0, NONE, -1, NONE, 0))
        return out


class Legal:
    """A `Decoder.generate_many` constraint for a wire model (sample.LegalLevels' counterpart):
    the items as `tantalos-score grammar` allows them, without their wires; `>` (the wires) or
    the end where the grammar lets the level end; then wires from a sender at or after the last
    one, to a receiver it doesn't drive yet."""

    def __init__(self, grammar, vocab, n_out):
        self.grammar, self.vocab, self.n_out = grammar, vocab, n_out
        self.v = len(vocab)
        self.state = {}  # prompt index -> (ids seen, Tracker, wires so far, last sender)
        self.gt = vocab.stoi[">"]

    def _state(self, i, seq):
        st = self.state.get(i)
        if st is None or len(st["seen"]) > len(seq) or seq[: len(st["seen"])] != st["seen"]:
            t = Tracker(self.vocab.itos, self.v)
            for x in (BOS, SEP):
                t.feed(x)
            st = self.state[i] = {"seen": [], "t": t, "wires": set(), "src": -1, "cur": []}
        for x in seq[len(st["seen"]):]:
            t = st["t"]
            if t.phase == "wires" and x >= self.v:
                if t.expect == "src":
                    st["src"], st["cur"] = x - self.v, [x - self.v]
                elif t.expect == "dst":
                    st["cur"].append(x - self.v)
                    if t.words[x - self.v] != "funnel":
                        st["wires"].add((st["cur"][0], x - self.v, None))
            elif t.phase == "wires" and t.expect == "kind":
                st["wires"].add((st["cur"][0], st["cur"][1], self.vocab.itos[x]))
            t.feed(x)
            st["seen"].append(x)
        return st

    def __call__(self, idx, seqs):
        mask = torch.zeros((len(seqs), self.n_out), dtype=torch.bool)
        ask, at = [], []
        for r, (i, seq) in enumerate(zip(idx, seqs)):
            st = self._state(i, seq)
            t = st["t"]
            if t.phase == "level":
                ask.append(" ".join(self.vocab.decode(seq)))
                at.append((r, st))
                continue
            classes = self._classes(t)
            if t.expect == "src":
                ok = [d for d, c in enumerate(classes) if c == SENDER and d >= st["src"]]
            elif t.expect == "dst":
                s = st["cur"][0]
                ok = [d for d, c in enumerate(classes) if c == RECEIVER and self._free(st, s, d, t)]
            elif t.expect == "kind":
                s, d = st["cur"]
                ok = None
                kinds = [k for k in FUNNEL_KINDS if (s, d, k) not in st["wires"]]
                mask[r, [self.vocab.stoi[k] for k in kinds] or [EOS]] = True
                continue
            else:
                ok = None
                mask[r, EOS] = True
                if any(c == SENDER and d >= st["src"] for d, c in enumerate(classes)):
                    mask[r, self.gt] = True
                continue
            if ok:
                mask[r, [self.v + d for d in ok if d < POINTERS]] = True
            else:
                mask[r, EOS] = True
        if ask:
            for (r, st), a in zip(at, self.grammar.next(ask)):
                ids = [self.vocab.stoi[x] for x in a.get("allowed", []) if x in self.vocab.stoi and x != ">"]
                if a.get("end") or not ids:
                    ids.append(EOS)
                    classes = self._classes(st["t"])
                    if SENDER in classes and RECEIVER in classes:
                        ids.append(self.gt)
                mask[r, ids] = True
        return mask

    @staticmethod
    def _classes(t):
        """Each device's class, the one being written as if it ended now."""
        out = t.classes[:]
        if t.cur is not None:
            out[t.cur] = device_class(t.word, t.toks)
        return out

    def _free(self, st, s, d, t):
        if t.words[d] == "funnel":
            return any((s, d, k) not in st["wires"] for k in FUNNEL_KINDS)
        return (s, d, None) not in st["wires"]

    def finish(self, i, seq):
        self.state.pop(i, None)
        return seq


# --- readiness check ----------------------------------------------------------------------------

def _check_file(path):
    stats = Counter()
    most, longer = 0, Counter()
    bad = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            toks = json.loads(line)["tokens"].split()
            stats["levels"] += 1
            try:
                w = to_wire(toks)
                back = compile(w)
            except WireError as e:
                stats["error"] += 1
                if len(bad) < 5:
                    bad.append((str(e), " ".join(toks)[:300]))
                continue
            if back != toks:
                stats["mismatch"] += 1
                if len(bad) < 5:
                    bad.append(("mismatch", " ".join(toks)[:300]))
                continue
            _, devices = records(toks)
            most = max(most, len(devices))
            stats["wires"] += sum(len(ws) for _, _, ws in devices)
            longer[len(w) - len(toks)] += 1
            stats["tokens"] += len(toks)
            stats["wire_tokens"] += len(w)
    return path, stats, most, longer, bad


def check(paths, procs=8):
    tot, longer, most = Counter(), Counter(), 0
    with Pool(procs) as pool:
        for path, stats, m, lg, bad in pool.imap_unordered(_check_file, paths):
            tot += stats
            longer += lg
            most = max(most, m)
            print(f"{path}: {stats['levels']:,} levels, {stats['error']} errors, {stats['mismatch']} mismatches, "
                  f"most devices {m}", flush=True)
            for b in bad:
                print("   ", b)
    print(f"total: {tot['levels']:,} levels, {tot['error']} errors, {tot['mismatch']} mismatches; "
          f"{tot['wires']:,} wires; most devices {most} (slots {POINTERS}); tokens {tot['tokens']:,} flat, "
          f"{tot['wire_tokens']:,} wire form; length change: "
          + ", ".join(f"{k:+d}: {n:,}" for k, n in sorted(longer.items())[:12]))
    return tot["error"] == 0 and tot["mismatch"] == 0 and most <= POINTERS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="round trip of every training level")
    c.add_argument("--data", nargs="*", default=sorted(glob.glob(str(ROOT / "data" / "v5b" / "levels-*.jsonl"))))
    c.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    sys.exit(0 if check(a.data, a.procs) else 1)


if __name__ == "__main__":
    main()
