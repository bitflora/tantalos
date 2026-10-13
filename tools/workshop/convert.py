"""Recover a Puzzle Maker chamber from its compiled map (.bsp), as glados tokens (src/tokens.rs).

    python tools/workshop/convert.py maps/*.bsp --report report.jsonl --tokens levels.txt

Workshop puzzles are published compiled; the .p2c never leaves the author's machine. But the
Puzzle Maker's compiler leaves the puzzle readable in the map:
- Cell v spans world [128v, 128v + 128) on each axis. Open cells are found by flooding from
  the entry door through empty BSP leaves, without leaving through the door corridors.
- White (portalable) faces are the ones whose wall material is a white tile.
- Every item instance is named <kind><item index>, e.g. `button23-proxy`, `barrierhazard8_brush`.
  Logic entities in an instance keep the instance's angles, so they give the item's rotation;
  the item's cell is floor((anchor origin + its up) / 128) for a suitable anchor entity.
- Buttons drive items through their instance's func_instance_io_proxy outputs.
- Goo, glass and grating aren't instances: they're water, window and grate brushes.

Only what the token format can say is recovered. Anything else (angled panels, lasers, light
bridges, faith plates, loose cubes, BEEmod items, ...) makes the map unsupported, with the reason.
Checked against compiled maps of known .p2c files: see tools/workshop/README.md.
"""
import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bsp import Bsp, BspError, aabb  # noqa: E402

CELL = 128
MAX_NUM = 32
CONTENTS_SOLID, CONTENTS_WINDOW, CONTENTS_GRATE = 0x1, 0x2, 0x8
CONTENTS_SLIME, CONTENTS_WATER = 0x10, 0x20
DIRS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
DIR_TOKEN = {(1, 0, 0): "+x", (-1, 0, 0): "-x", (0, 1, 0): "+y", (0, -1, 0): "-y", (0, 0, 1): "+z", (0, 0, -1): "-z"}
UP = (0, 0, 1)
DOWN = (0, 0, -1)

# Instances that are puzzle items we can express.
ITEMS = {"doorentry", "doorexit", "button", "cubedropper", "barrierhazard", "paintdropper", "turret"}
# Instances the compiler adds around every chamber, or that only decorate (antlines, signs).
DECOR = {
    "InstanceAuto", "arrival_sign_", "departure_sign_", "elevator_", "environment_act", "relay_spawn_",
    "observationroom", "indicator_panel", "indicator_toggle", "entrance_airlock_door",
    "entrance_airlock_door_", "exit_airlock_door", "exit_airlock_door_", "placement_helper",
}
INSTANCE = re.compile(r"@?([A-Za-z_]+?)(\d+)(?=[-_]|$)")
LOGIC = {"logic_relay", "logic_branch", "logic_auto", "logic_case", "math_counter", "func_instance_io_proxy",
         "func_areaportal", "point_template", "env_entity_maker", "logic_branch_listener"}
CUBE_KINDS = ["standard", "companion", "reflection", "sphere", "franken"]
PAINT_KINDS = {0: "bounce", 2: "speed", 3: "portal", 4: "water"}
FLOOR_BUTTONS = {"prop_floor_button": "weighted", "prop_under_floor_button": "weighted",
                 "prop_floor_cube_button": "cube", "prop_floor_ball_button": "sphere"}
PEDESTALS = {"prop_button", "prop_under_button"}


class Unsupported(Exception):
    """The map isn't a puzzle the token format can describe (or isn't a Puzzle Maker map)."""


def vec(s):
    return tuple(float(x) for x in s.split())


def rotate(angles, v):
    """World axis that local axis v points along after Source angles (src/geom.rs `rotate`)."""
    p, y, r = (math.radians(a) for a in angles)
    sp, cp, sy, cy, sr, cr = math.sin(p), math.cos(p), math.sin(y), math.cos(y), math.sin(r), math.cos(r)
    forward = (cp * cy, cp * sy, -sp)
    left = (sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, sr * cp)
    up = (cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp)
    w = [v[0] * forward[i] + v[1] * left[i] + v[2] * up[i] for i in range(3)]
    axis = [a for a in range(3) if abs(w[a]) > 0.99]
    if len(axis) != 1 or any(abs(w[a]) > 0.01 for a in range(3) if a != axis[0]):
        return None
    d = [0, 0, 0]
    d[axis[0]] = 1 if w[axis[0]] > 0 else -1
    return tuple(d)


def axis_of(d):
    return next(a for a in range(3) if d[a])


def add(c, d, k=1):
    return tuple(c[i] + k * d[i] for i in range(3))


def cell_of(p):
    return tuple(math.floor(x / CELL) for x in p)


class Entity:
    def __init__(self, kv):
        self.kv = kv
        self.d = dict(kv)
        self.cls = self.d.get("classname", "")
        self.name = self.d.get("targetname", "")

    def origin(self):
        return vec(self.d.get("origin", "0 0 0"))

    def angles(self):
        return vec(self.d.get("angles", "0 0 0"))

    def outputs(self):
        """(output, target entity name) for each I/O connection."""
        out = []
        for k, v in self.kv:
            sep = "\x1b" if "\x1b" in v else ","
            parts = v.split(sep)
            if k.startswith("On") and len(parts) >= 2:
                out.append((k, parts[0]))
        return out


class Instance:
    def __init__(self, kind, index):
        self.kind, self.index = kind, index
        self.ents = []

    @property
    def name(self):
        return f"{self.kind}{self.index}"

    def direct(self, cls=None, suffix=None):
        """Entities of the instance itself, not of an instance nested in it."""
        out = []
        for e in self.ents:
            rest = e.name.lstrip("@")[len(self.name):]
            if rest.count("-") > 1 or (rest.startswith("_") and "-" in rest):
                continue
            if cls and e.cls not in (cls if isinstance(cls, (set, tuple)) else {cls}):
                continue
            if suffix and not e.name.endswith(suffix):
                continue
            out.append(e)
        return out

    def angles(self):
        votes = collections.Counter(tuple(round(a) % 360 for a in e.angles()) for e in self.direct(LOGIC))
        if not votes:
            raise Unsupported(f"{self.name}: no entity to read its rotation from")
        return votes.most_common(1)[0][0]

    def up(self):
        u = rotate(self.angles(), (0, 0, 1))
        if u is None:
            raise Unsupported(f"{self.name}: rotation isn't axis-aligned")
        return u

    def anchor_cell(self, ent, up=None):
        up = up or self.up()
        return cell_of(add(ent.origin(), up))

    def one(self, cls=None, suffix=None):
        es = self.direct(cls, suffix)
        if not es:
            raise Unsupported(f"{self.name}: no {cls or ''}{suffix or ''} entity")
        return es[0]


class World:
    """Solid world brushes, bucketed by cell for point tests."""

    def __init__(self, bsp):
        all_brushes = bsp.brushes()
        self.brushes = []
        self.grid = collections.defaultdict(list)
        self.water = []
        for i in bsp.model_brushes(0):
            contents, sides = all_brushes[i]
            lo, hi = aabb(sides)
            if contents & (CONTENTS_WATER | CONTENTS_SLIME):
                self.water.append((lo, hi))
            if not contents & CONTENTS_SOLID:
                continue
            if max(h - l for l, h in zip(lo, hi)) > 1e6:
                continue
            k = len(self.brushes)
            self.brushes.append(sides)
            for x in range(math.floor(lo[0] / CELL), math.floor(hi[0] / CELL) + 1):
                for y in range(math.floor(lo[1] / CELL), math.floor(hi[1] / CELL) + 1):
                    for z in range(math.floor(lo[2] / CELL), math.floor(hi[2] / CELL) + 1):
                        self.grid[(x, y, z)].append(k)

    def _inside(self, sides, p, eps=0.01):
        return all(n[0] * p[0] + n[1] * p[1] + n[2] * p[2] <= d + eps for n, d, _ in sides)

    def solid_at(self, p):
        return any(self._inside(self.brushes[k], p) for k in self.grid.get(cell_of(p), ()))

    def passable(self, c, d):
        """Whether the segment between the centres of cell c and its neighbour misses all solid."""
        centre = [CELL * v + CELL / 2 for v in c]
        return not any(self.solid_at([centre[i] + d[i] * t for i in range(3)]) for t in range(0, CELL + 1, 2))

    def surface(self, c, d):
        """Material of the wall face of cell c towards d, by majority over 5 sample points."""
        centre = [CELL * v + CELL / 2 for v in c]
        ax = axis_of(d)
        o1, o2 = [a for a in range(3) if a != ax]
        names = collections.Counter()
        for u, v in [(0, 0), (-40, -40), (40, 40), (-40, 40), (40, -40)]:
            p = list(centre)
            p[ax] += d[ax] * CELL / 2
            p[o1] += u
            p[o2] += v
            q = list(p)
            q[ax] += d[ax]  # just inside the wall
            for k in self.grid.get(cell_of(q), ()):
                sides = self.brushes[k]
                if self._inside(sides, q):
                    for n, dist, mat in sides:
                        if abs(n[ax] + d[ax]) < 0.001 and abs(n[0] * p[0] + n[1] * p[1] + n[2] * p[2] - dist) < 0.5:
                            names[mat.lower()] += 1
        return names.most_common(1)[0][0] if names else None

    def in_water(self, p):
        return any(all(lo[a] <= p[a] <= hi[a] for a in range(3)) for lo, hi in self.water)


def entity_models(bsp, ents):
    """(entity or None for the world, model index, origin offset) for every brush model."""
    out = [(None, 0, (0.0, 0.0, 0.0))]
    for e in ents:
        m = e.d.get("model", "")
        if m.startswith("*"):
            out.append((e, int(m[1:]), e.origin()))
    return out


def model_box(bsp, e):
    mins, maxs, _, _ = bsp.models()[int(e.d["model"][1:])]
    o = e.origin()
    return [mins[a] + o[a] for a in range(3)], [maxs[a] + o[a] for a in range(3)]


def cells_in(lo, hi, a):
    """Cells a box covers along axis a (ignoring slivers within 1 unit of a boundary)."""
    return range(math.floor((lo[a] + 1) / CELL), math.floor((hi[a] - 1) / CELL) + 1)


def merge_panels(panels):
    """Greedy rectangles of same-kind panel faces in one plane: (kind, lo cell, hi cell, normal)."""
    groups = collections.defaultdict(set)
    for (c, d), kind in panels.items():
        groups[(kind, d, c[axis_of(d)])].add(c)
    out = []
    for (kind, d, _), cells in sorted(groups.items()):
        ax = axis_of(d)
        o1, o2 = [a for a in range(3) if a != ax]
        left = set(cells)
        for c in sorted(cells):
            if c not in left:
                continue
            hi = list(c)
            while tuple(hi[a] + (a == o1) for a in range(3)) in left:
                hi[o1] += 1

            def row(k):
                return [tuple(k if a == o2 else x if a == o1 else c[a] for a in range(3))
                        for x in range(c[o1], hi[o1] + 1)]

            while all(x in left for x in row(hi[o2] + 1)):
                hi[o2] += 1
            for k in range(c[o2], hi[o2] + 1):
                left -= set(row(k))
            out.append((kind, c, tuple(hi), d))
    return out


def screen(entities):
    """The map's entities and item instances by kind, from its entity lump alone; raises
    Unsupported for items the tokens can't express, so callers can skip the rest of the map."""
    ents = [Entity(kv) for kv in entities]
    instances = {}
    for e in ents:
        m = INSTANCE.match(e.name)
        if m:
            kind, idx = m.group(1), int(m.group(2))
            instances.setdefault((kind, idx), Instance(kind, idx)).ents.append(e)
    # 2012-era maps wrap glass in `target<N>` instances of paint-bomb filters; the glass itself
    # is found from its brushes like any other.
    for key, inst in list(instances.items()):
        if key[0] == "target" and all(e.cls == "filter_activator_class" for e in inst.ents):
            del instances[key]
    other = sorted({k for k, _ in instances} - ITEMS - DECOR)
    if other:
        raise Unsupported("items: " + ", ".join(other))
    by_kind = collections.defaultdict(list)
    for (kind, _), inst in sorted(instances.items()):
        by_kind[kind].append(inst)
    if len(by_kind["doorentry"]) != 1 or len(by_kind["doorexit"]) != 1:
        raise Unsupported("not one entry and one exit door (not a Puzzle Maker map?)")
    return ents, by_kind


def convert(bsp):
    """Tokens for the puzzle in `bsp`, or raise Unsupported."""
    ents, by_kind = screen(bsp.entities())

    # Doors: the entry's area portal and the exit's logic sit in the door's cell.
    entry_i, exit_i = by_kind["doorentry"][0], by_kind["doorexit"][0]
    entry_n = entry_i.up()
    entry = entry_i.anchor_cell(entry_i.one("func_areaportal"), entry_n)
    exit_n = exit_i.up()
    exit_c = exit_i.anchor_cell(exit_i.one("math_counter", "-counter"), exit_n)

    # Open cells: flood from the doors and every item, through empty leaves, not out the doors.
    world = World(bsp)
    leaf = bsp.leaf_contents()
    blocked = set()
    for c, n in ((entry, entry_n), (exit_c, exit_n)):
        for cc in (c, add(c, UP)):
            blocked.add((cc, tuple(-x for x in n)))

    def is_open(c):
        return not leaf([CELL * v + CELL / 2 for v in c]) & CONTENTS_SOLID

    open_cells = set()

    def flood(start):
        if start in open_cells:
            return
        if not is_open(start):
            raise Unsupported(f"item cell {start} isn't open")
        todo = [start]
        open_cells.add(start)
        while todo:
            a = todo.pop()
            for d in DIRS:
                b = add(a, d)
                if b in open_cells or (a, d) in blocked or not is_open(b) or not world.passable(a, d):
                    continue
                open_cells.add(b)
                todo.append(b)
                if len(open_cells) > MAX_NUM ** 3:
                    raise Unsupported("open space leaks out of the chamber")

    flood(entry)
    flood(exit_c)

    # Items. Each returns its cell(s) so their rooms get flooded too.
    item_cells = []
    toks_fields, field_index = [], {}
    for inst in by_kind["barrierhazard"]:
        brushes = inst.direct(("trigger_portal_cleanser", "trigger_hurt"), "_brush")
        kinds = {e.cls for e in brushes}
        if "trigger_portal_cleanser" in kinds:
            kind, cls = "fizzler", "trigger_portal_cleanser"
        elif "trigger_hurt" in kinds:
            kind, cls = "laserfield", "trigger_hurt"
        else:
            raise Unsupported(f"{inst.name}: field with no trigger")
        brushes = [e for e in brushes if e.cls == cls]
        ang = inst.angles()
        n, along = rotate(ang, (0, 0, 1)), rotate(ang, (0, 1, 0))
        if n is None or along is None:
            raise Unsupported(f"{inst.name}: rotation isn't axis-aligned")
        f = tuple(n[(i + 1) % 3] * along[(i + 2) % 3] - n[(i + 2) % 3] * along[(i + 1) % 3] for i in range(3))
        boxes = [model_box(bsp, e) for e in brushes]
        lo = [min(b[0][a] for b in boxes) for a in range(3)]
        hi = [max(b[1][a] for b in boxes) for a in range(3)]
        na, aa, fa = axis_of(n), axis_of(along), axis_of(f)
        span = cells_in(lo, hi, aa)
        if not span:
            raise Unsupported(f"{inst.name}: empty field")
        anchor = [0, 0, 0]
        ncells = cells_in(lo, hi, na)
        anchor[na] = ncells[0] if n[na] > 0 else ncells[-1]
        anchor[aa] = span[0] if along[aa] > 0 else span[-1]
        anchor[fa] = math.floor((lo[fa] + hi[fa]) / 2 / CELL)
        on = all(e.d.get("StartDisabled", "0") == "0" for e in brushes)
        field_index[inst.index] = len(toks_fields)
        toks_fields.append((kind, tuple(anchor), n, along, len(span), on))
        item_cells.append(tuple(anchor))

    toks_gels, gel_index = [], {}
    for inst in by_kind["paintdropper"]:
        sprayers = inst.direct("info_paint_sprayer", "-sprayer")
        if not sprayers:
            raise Unsupported(f"{inst.name}: no sprayer")
        ptype = int(sprayers[0].d.get("PaintType", "-1"))
        if ptype not in PAINT_KINDS:
            raise Unsupported(f"{inst.name}: paint type {ptype}")
        if inst.up() != DOWN:
            raise Unsupported(f"{inst.name}: gel dropper not on a ceiling")
        dropper = inst.anchor_cell(sprayers[0], DOWN)
        on = sprayers[0].d.get("start_active", "1") == "1"
        gel_index[inst.index] = len(toks_gels)
        toks_gels.append((dropper, PAINT_KINDS[ptype], on))
        item_cells.append(dropper)

    toks_droppers = []
    for inst in by_kind["cubedropper"]:
        if inst.up() != DOWN:
            raise Unsupported(f"{inst.name}: cube dropper not on a ceiling")
        cell = inst.anchor_cell(inst.one("env_entity_maker", "-cube_dropper_box_spawner"), DOWN)
        kind = int(inst.one("math_counter", "-counter_cube_type").d.get("startvalue", "0"))
        if not 0 <= kind < len(CUBE_KINDS):
            raise Unsupported(f"{inst.name}: cube type {kind}")
        toks_droppers.append((cell, CUBE_KINDS[kind]))
        item_cells.append(cell)

    def targets(inst):
        out = []
        for proxy in inst.direct("func_instance_io_proxy", "-proxy"):
            for _, target in proxy.outputs():
                m = INSTANCE.match(target)
                if not m:
                    raise Unsupported(f"{inst.name}: wired to {target}")
                kind, idx = m.group(1), int(m.group(2))
                if kind in ("indicator_panel", "indicator_toggle"):
                    continue
                if kind == "doorexit":
                    t = ("exit",)
                elif kind == "barrierhazard" and idx in field_index:
                    t = ("field", field_index[idx])
                elif kind == "paintdropper" and idx in gel_index:
                    t = ("gel", gel_index[idx])
                else:
                    raise Unsupported(f"{inst.name}: wired to {kind}{idx}")
                if t not in out:
                    out.append(t)
        return out

    toks_buttons, toks_pedestals = [], []
    for inst in by_kind["button"]:
        floor = [e for e in inst.direct(set(FLOOR_BUTTONS)) if e.name.endswith("-button")]
        peds = [e for e in inst.direct(PEDESTALS) if e.name.endswith("-button")]
        if inst.up() != UP and not peds:
            raise Unsupported(f"{inst.name}: button not on a floor")
        if floor:
            cell = inst.anchor_cell(floor[0], UP)
            toks_buttons.append((cell, FLOOR_BUTTONS[floor[0].cls], targets(inst)))
        elif peds:
            p = peds[0]
            if rotate(p.angles(), (0, 0, 1)) != UP:
                raise Unsupported(f"{inst.name}: pedestal not on a floor")
            cell = cell_of(add(p.origin(), UP))
            if p.d.get("istimer", "0") == "1":
                t = round(float(p.d.get("Delay", "3")))
                if not 3 <= t <= 30:
                    raise Unsupported(f"{inst.name}: timer {t}")
                timer = str(t)
            else:
                timer = "inf"
            toks_pedestals.append((cell, timer, targets(inst)))
        else:
            raise Unsupported(f"{inst.name}: unknown button")
        item_cells.append(cell)

    toks_turrets = []
    for inst in by_kind["turret"]:
        t = inst.one("npc_portal_turret_floor", "-turret")
        if rotate(t.angles(), (0, 0, 1)) != UP:
            raise Unsupported(f"{inst.name}: turret not upright")
        # The .p2c's Facing (what Level reads) is the npc's yaw + 180.
        yaw = (t.angles()[1] + 180) % 360
        f = round(yaw / 45)
        if abs(yaw - 45 * f) > 0.5:
            raise Unsupported(f"{inst.name}: turret yaw {yaw}")
        cell = cell_of(add(t.origin(), UP))
        toks_turrets.append((cell, f % 8))
        item_cells.append(cell)

    for c in item_cells:
        flood(c)

    # Gel lands on the floor under its dropper.
    gels = []
    for dropper, kind, on in toks_gels:
        c = dropper
        while add(c, DOWN) in open_cells:
            c = add(c, DOWN)
        gels.append((c, kind, on))

    goo = {c for c in open_cells if world.in_water([CELL * v + CELL / 2 for v in c])}

    # Glass and grating: thin window/grate brushes just inside the face they're on.
    panels = {}
    for ent, model, off in entity_models(bsp, ents):
        if model >= len(bsp.models()):
            continue
        all_brushes = bsp.brushes()
        for i in bsp.model_brushes(model):
            contents, sides = all_brushes[i]
            mats = {m.lower() for _, _, m in sides}
            if any("metalgrate" in m for m in mats) and contents & CONTENTS_GRATE:
                kind = "grating"
            elif any(("/glass/" in m or "glasswindow" in m) and not m.startswith("tools/") for m in mats) \
                    and contents & CONTENTS_WINDOW:
                kind = "glass"
            else:
                continue
            lo, hi = aabb(sides)
            lo = [lo[a] + off[a] for a in range(3)]
            hi = [hi[a] + off[a] for a in range(3)]
            ext = [hi[a] - lo[a] for a in range(3)]
            ax = min(range(3), key=lambda a: ext[a])
            if ext[ax] > 16:
                continue
            mid = (lo[ax] + hi[ax]) / 2
            plane = round(mid / CELL) * CELL
            d = [0, 0, 0]
            d[ax] = 1 if plane > mid else -1
            d = tuple(d)
            o1, o2 = [a for a in range(3) if a != ax]
            for u in cells_in(lo, hi, o1):
                for v in cells_in(lo, hi, o2):
                    c = [0, 0, 0]
                    c[ax], c[o1], c[o2] = math.floor(mid / CELL), u, v
                    c = tuple(c)
                    if c in open_cells and add(c, d) in open_cells:
                        panels[(c, d)] = kind

    # Coordinates: as compiled, which is the editor's own frame (as in the .p2c, and like
    # generated levels: floors usually at z = 2, goo pits below). An axis that doesn't fit in
    # 0..31 (2012-era maps put their floor at world z = 0, so pits go negative) is shifted to
    # start at 0, or z at 2 when there's room.
    lo = [min(c[a] for c in open_cells) for a in range(3)]
    hi = [max(c[a] for c in open_cells) for a in range(3)]
    base = []
    for a in range(3):
        if 0 <= lo[a] and hi[a] < MAX_NUM:
            base.append(0)
        else:
            pad = 2 if a == 2 and hi[a] - lo[a] + 3 <= MAX_NUM else 0
            base.append(lo[a] - pad)
    size = [hi[a] - base[a] + 1 for a in range(3)]
    if max(size) > MAX_NUM:
        raise Unsupported(f"chamber {size} is bigger than {MAX_NUM}")

    def loc(c):
        return [str(c[a] - base[a]) for a in range(3)]

    out = ["size"] + [str(s) for s in size]
    for y in range(size[1]):
        for x in range(size[0]):
            runs = []
            for z in range(size[2]):
                c = (x + base[0], y + base[1], z + base[2])
                k = "g" if c in goo else "o" if c in open_cells else "s"
                if runs and runs[-1][0] == k:
                    runs[-1][1] += 1
                else:
                    runs.append([k, 1])
            out += [f"{k}{n}" for k, n in runs] + ["|"]
        out.append("/")

    for c in sorted(open_cells):
        for d in DIRS:
            if add(c, d) in open_cells:
                continue
            mat = world.surface(c, d)
            if mat and "white" in mat:
                out += ["white", DIR_TOKEN[d]] + loc(c) + loc(c)

    out += ["entry"] + loc(entry) + [DIR_TOKEN[entry_n]]
    out += ["exit"] + loc(exit_c) + [DIR_TOKEN[exit_n]]

    for kind, lo_c, hi_c, d in merge_panels(panels):
        ax = axis_of(d)
        # Hang it from a surface along one of its edges: prefer the floor.
        grow = None
        for g in [x for x in [UP, DOWN] + DIRS[:4] if axis_of(x) != ax]:
            ga = axis_of(g)
            edge = lo_c[ga] if g[ga] > 0 else hi_c[ga]
            o = next(a for a in range(3) if a not in (ax, ga))
            if any(add(tuple(edge if a == ga else lo_c[a] if a != o else k for a in range(3)),
                       tuple(-v for v in g)) not in open_cells
                   for k in range(lo_c[o], hi_c[o] + 1)):
                grow = g
                break
        grow = grow or next(x for x in DIRS if axis_of(x) != ax)
        out += [kind] + loc(lo_c) + loc(hi_c) + [DIR_TOKEN[d], DIR_TOKEN[grow]]

    for kind, anchor, n, along, length, on in toks_fields:
        out += [kind] + loc(anchor) + [DIR_TOKEN[n], DIR_TOKEN[along], str(length), "on" if on else "off"]
    for c, kind, on in gels:
        out += ["gel"] + loc(c) + ["+z", kind, "stream", "on" if on else "off"]
    for c, kind in toks_droppers:
        out += ["dropper"] + loc(c) + [kind]

    def wires(ts):
        w = []
        for t in ts:
            w += [">"] + [str(x) for x in t]
        return w

    for c, kind, ts in toks_buttons:
        out += ["button"] + loc(c) + [kind] + wires(ts)
    for c, timer, ts in toks_pedestals:
        out += ["pedestal"] + loc(c) + [timer] + wires(ts)
    for c, f in toks_turrets:
        out += ["turret"] + loc(c) + [str(f)]
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bsp", nargs="+")
    ap.add_argument("--report", help="JSONL: one line per map with its tokens or why it's unsupported")
    ap.add_argument("--tokens", help="write the converted maps' tokens here, one per line (for glados score)")
    a = ap.parse_args()
    report = open(a.report, "w", encoding="utf-8") if a.report else None
    toks = open(a.tokens, "w", encoding="utf-8") if a.tokens else None
    counts = collections.Counter()
    for path in a.bsp:
        rec = {"file": str(path)}
        try:
            rec["tokens"] = convert(Bsp.open(path))
            counts["ok"] += 1
            if toks:
                toks.write(rec["tokens"] + "\n")
        except (Unsupported, BspError) as e:
            rec["error"] = str(e)
            counts["unsupported" if isinstance(e, Unsupported) else "bad file"] += 1
        line = json.dumps(rec)
        if report:
            report.write(line + "\n")
        else:
            print(line)
    print(dict(counts), file=sys.stderr)


if __name__ == "__main__":
    main()
