"""Lossless device records and explicit device-index wires for architecture arm 4.

Records preserve device order, so compiling does not change family-local indices.
The model writes the unwired body followed by explicit source/kind/target triples.
"""
from dataclasses import dataclass

from spatial import ITEMS, SAME, ITEM_WORDS, OPTIONAL, SENDERS, TARGETS

NUMBERS = frozenset(str(i) for i in range(33))
MAX_DEVICES = 256  # More than can fit in a 1024-token training block.
EXTRA = ['<wires>', '<wire>'] + [f'<ptr:{i}>' for i in range(MAX_DEVICES)]


def pointer_index(word):
    if not word.startswith('<ptr:') or not word.endswith('>'):
        raise ValueError('invalid pointer token')
    number = word[5:-1]
    if not number.isdigit() or str(int(number)) != number or not 0 <= int(number) < MAX_DEVICES:
        raise ValueError('invalid pointer index')
    return int(number)


@dataclass(frozen=True)
class Wire:
    source: int
    target: int
    kind: str


@dataclass(frozen=True)
class Records:
    geometry: tuple
    devices: tuple
    wires: tuple


def endpoints(devices, kind):
    if kind not in TARGETS:
        raise ValueError(f"unknown wire kind {kind}")
    return [i for i, d in enumerate(devices) if d[0] in TARGETS[kind]]


def legal_targets(devices, source, kind):
    """Device-index mask for a wire, after all devices have been emitted."""
    if not 0 <= source < len(devices) or devices[source][0] not in SENDERS:
        return [False] * len(devices)
    # Standard target indices are vocabulary numbers 0..32; exit has no index.
    candidates = set(endpoints(devices, kind)[:1 if kind == 'exit' else 33])
    return [i in candidates and not (kind == "gel" and d[-1] == "splat")
            for i, d in enumerate(devices)]


def parse(tokens):
    words = tokens.split() if isinstance(tokens, str) else list(tokens)
    if len(words) < 4 or words[0] != "size":
        raise ValueError("expected level size")
    at = next((i for i in range(4, len(words)) if words[i] in ITEM_WORDS), len(words))
    geometry = tuple(words[:at])
    devices, pending = [], []
    while at < len(words):
        start = at
        kind = words[at]
        if kind not in ITEM_WORDS:
            raise ValueError(f"expected device, got {kind}")
        fields = ITEMS[SAME.get(kind, kind)]
        at += 1
        for field in fields:
            if at >= len(words):
                raise ValueError("truncated device")
            allowed = NUMBERS if isinstance(field, str) else field[1]
            if words[at] not in allowed:
                raise ValueError(f"invalid {kind} field: {words[at]}")
            at += 1
        if kind == "gel" and words[at - 1] == "stream":
            if at >= len(words) or words[at] not in {"on", "off"}:
                raise ValueError("missing gel state")
            at += 1
        if at < len(words) and words[at] == OPTIONAL.get(kind):
            at += 1
        devices.append(tuple(words[start:at]))
        while at < len(words) and words[at] == ">":
            if kind not in SENDERS or at + 1 >= len(words):
                raise ValueError("invalid wire source or truncated wire")
            target_kind = words[at + 1]
            at += 2
            index = 0
            if target_kind != "exit":
                if at >= len(words) or not words[at].isdigit():
                    raise ValueError("missing target index")
                index = int(words[at])
                at += 1
            pending.append((len(devices) - 1, target_kind, index))
    wires = []
    for source, kind, index in pending:
        candidates = endpoints(devices, kind)
        if index >= len(candidates):
            raise ValueError("wire target out of range")
        target = candidates[index]
        if kind == "gel" and devices[target][-1] == "splat":
            raise ValueError("paint splats cannot receive wires")
        wires.append(Wire(source, target, kind))
    return Records(geometry, tuple(devices), tuple(wires))


def compile(records):
    grouped = [[] for _ in records.devices]
    for wire in records.wires:
        if not 0 <= wire.source < len(grouped) or records.devices[wire.source][0] not in SENDERS:
            raise ValueError("invalid wire source")
        candidates = endpoints(records.devices, wire.kind)
        if wire.target not in candidates:
            raise ValueError("incompatible wire target")
        grouped[wire.source].extend([">", wire.kind])
        if wire.kind != "exit":
            grouped[wire.source].append(str(candidates.index(wire.target)))
    words = list(records.geometry)
    for device, links in zip(records.devices, grouped):
        words.extend(device)
        words.extend(links)
    result = " ".join(words)
    parse(result)  # Reject malformed records as well as invalid pointers.
    return result


def serialize(tokens):
    records = parse(tokens)
    if len(records.devices) > MAX_DEVICES:
        raise ValueError('too many devices for pointer vocabulary')
    out = list(records.geometry)
    for device in records.devices:
        out.extend(device)
    out.append('<wires>')
    for w in records.wires:
        out.extend(['<wire>', f'<ptr:{w.source}>', w.kind, f'<ptr:{w.target}>'])
    return out


def deserialize(words):
    words = words.split() if isinstance(words, str) else list(words)
    boundary = words.index('<wires>')
    records = parse(words[:boundary])
    tail = words[boundary + 1:]
    if len(tail) % 4:
        raise ValueError('incomplete wire record')
    wires = []
    for at in range(0, len(tail), 4):
        marker, source, kind, target = tail[at:at + 4]
        if marker != '<wire>' or not source.startswith('<ptr:') or not target.startswith('<ptr:'):
            raise ValueError('invalid wire record')
        wires.append(Wire(pointer_index(source), pointer_index(target), kind))
    return compile(Records(records.geometry, records.devices, tuple(wires)))


class State:
    """Incremental structured-prefix parser; features name legal device-end positions."""
    def __init__(self):
        self.pos = -1
        self.body_start = 0
        self.body = []
        self.records = None
        self.ends = []
        self.phase = 'prompt'
        self.source = None
        self.kind = None
        self.wires = []

    def feed(self, word):
        features = [0] * MAX_DEVICES
        if word == '<pad>':
            return features
        self.pos += 1
        if word == '<sep>':
            self.phase, self.body_start = 'body', self.pos + 1
        elif self.phase == 'body':
            if word == '<wires>':
                try:
                    self.records = parse(self.body)
                    if len(self.records.devices) > MAX_DEVICES:
                        raise ValueError('too many devices')
                    end = self.body_start + len(self.records.geometry)
                    for device in self.records.devices:
                        end += len(device)
                        self.ends.append(end)  # last token position + 1
                    self.phase = 'wire'
                except ValueError:
                    self.phase = 'invalid'
            else:
                self.body.append(word)
        elif self.phase == 'wire':
            self.phase = 'source' if word == '<wire>' else 'done' if word == '<eos>' else 'invalid'
        elif self.phase == 'source':
            try:
                self.source = pointer_index(word)
                if not 0 <= self.source < len(self.ends) or self.records.devices[self.source][0] not in SENDERS:
                    raise ValueError('invalid source')
                self.phase = 'kind'
            except ValueError:
                self.phase = 'invalid'
        elif self.phase == 'kind':
            self.kind = word
            self.phase = 'target' if word in TARGETS else 'invalid'
        elif self.phase == 'target':
            try:
                target = pointer_index(word)
                legal = legal_targets(self.records.devices, self.source, self.kind)
                if not 0 <= target < len(legal) or not legal[target]:
                    raise ValueError('invalid target')
                self.wires.append(Wire(self.source, target, self.kind))
                self.phase = 'wire'
            except ValueError:
                self.phase = 'invalid'
        if self.phase == 'source':
            legal = [d[0] in SENDERS for d in self.records.devices]
        elif self.phase == 'target':
            legal = legal_targets(self.records.devices, self.source, self.kind)
        else:
            return features
        for i, ok in enumerate(legal):
            if ok:
                features[i] = self.ends[i]
        return features

    def finish(self):
        if self.records is None or self.phase not in ('wire', 'done'):
            raise ValueError('incomplete structured level')
        return compile(Records(self.records.geometry, self.records.devices, tuple(self.wires)))


class Rows:
    def __init__(self, itos, b):
        self.itos, self.rows = itos, [None] * b

    def start(self, r, ids):
        state = State()
        for i in ids:
            state.feed(self.itos[i])
        self.rows[r] = state

    def feed(self, toks, rows=None):
        return [self.rows[r].feed(self.itos[t]) if (rows is None or r in rows)
                else [0] * MAX_DEVICES for r, t in enumerate(toks)]
