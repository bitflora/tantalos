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
- Buttons, laser catchers and relays drive items through their instance's
  func_instance_io_proxy outputs (a funnel's inputs 3 and 4 reverse it).
- Goo, glass and grating aren't instances: they're water, window and grate brushes.
- Lasers, bridges and funnels: the prop's origin is 16 units into the wall, its forward the way
  it fires or projects (a bridge's walkable side is its up); catchers likewise face out of the
  wall. Centred and `_OFFSET` items share the `_center` models: an offset one's origin is 32
  units off the face's middle, toward the side it sits. Faith plates: the plate's arm and its trigger_catapult's launch target (an
  info_target, or a helper trigger launching straight up). Panels and stairs: their arms and
  steps, with the instance's angles (local +x: tilt / rising direction) and their branches'
  InitialValue. Pistons: the platform's func_movelinear and its level counters. Rails: the
  track's func_movelinear (movedir, movedistance) from its own cell. Loose cubes: `cube<N>`
  instances' prop_weighted_cube (CubeType).

Only what the token format can say is recovered: BEEmod items and the like make the map
unsupported, with the reason. Checked against compiled maps of known .p2c files: see
tools/workshop/README.md.
"""
import argparse
import collections
import itertools
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
ITEMS = {"doorentry", "doorexit", "button", "cubedropper", "barrierhazard", "paintdropper", "turret",
         "cube", "laseremit", "lasercatch", "laserrelay", "catapult", "bridge", "tbeam",
         "angledPanel", "angledClearPanel", "flipPanel", "stairs", "pistonlift", "raillift"}
# Instances the compiler adds around every chamber, or that only decorate (antlines, signs).
DECOR = {
    "InstanceAuto", "arrival_sign_", "departure_sign_", "elevator_", "environment_act", "relay_spawn_",
    "observationroom", "indicator_panel", "indicator_toggle", "entrance_airlock_door",
    "entrance_airlock_door_", "exit_airlock_door", "exit_airlock_door_", "placement_helper",
    "target",  # a faith plate target's marker; the plate's trigger names the real one
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
        return [(k, t) for k, t, _ in self.connections()]

    def connections(self):
        """(output, target entity name, input) for each I/O connection."""
        out = []
        for k, v in self.kv:
            sep = "\x1b" if "\x1b" in v else ","
            parts = v.split(sep)
            if k.startswith("On") and len(parts) >= 2:
                out.append((k, parts[0], parts[1]))
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

    # The later items now: their backs block the flood.
    later = later_items(bsp, ents, by_kind, [], blocked, is_open)

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
    toks_fields, field_index, field_planes = [], {}, []
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
        # The editor puts a field 0.375 voxel toward its N x A side, or in the middle of its
        # voxel (LocalPos 0), which tokens can't say: a middle one goes on a side.
        mid = abs((lo[fa] + hi[fa]) / 2 / CELL % 1 - 0.5) < 0.01
        planes = {(c, f[fa]) for c in itertools.product(*[
            cells_in(lo, hi, a) if a != fa else [anchor[fa]] for a in range(3)])}
        field_index[inst.index] = len(toks_fields)
        toks_fields.append((kind, tuple(anchor), n, along, len(span), on))
        field_planes.append((planes, mid))
        item_cells.append(tuple(anchor))
    # A middle field shares its voxel with one on a side (a common Workshop trick). Two on one
    # side of a voxel the editor rejects (in-game checks, 2026-10-24), so a middle one that
    # would land on a side field's faces goes on the other side: the far end, along reversed.
    for i, (planes, mid) in enumerate(field_planes):
        sides = [p for j, (p, m) in enumerate(field_planes) if j != i and not m]
        if mid and any(planes & p for p in sides):
            if any({(c, -s) for c, s in planes} & p for p in sides):
                raise Unsupported("a field in the middle of a voxel with fields on both sides")
            kind, anchor, n, along, ln, on = toks_fields[i]
            far = add(anchor, along, ln - 1)
            toks_fields[i] = (kind, far, n, tuple(-v for v in along), ln, on)

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

    toks_droppers, dropper_insts = [], []
    for inst in by_kind["cubedropper"]:
        if inst.up() != DOWN:
            raise Unsupported(f"{inst.name}: cube dropper not on a ceiling")
        cell = inst.anchor_cell(inst.one("env_entity_maker", "-cube_dropper_box_spawner"), DOWN)
        kind = int(inst.one("math_counter", "-counter_cube_type").d.get("startvalue", "0"))
        if not 0 <= kind < len(CUBE_KINDS):
            raise Unsupported(f"{inst.name}: cube type {kind}")
        # Auto drop off: the dropper only drops when its inputs first come on.
        auto = inst.direct("logic_relay", "-@relay_spawn_on_entrance") or inst.direct(
            "logic_relay", "relay_spawn_on_entrance")
        wait = any(e.d.get("StartDisabled", "0") == "1" for e in auto)
        toks_droppers.append((cell, CUBE_KINDS[kind], wait))
        dropper_insts.append(inst)
        item_cells.append(cell)

    dropper_index = {inst.index: i for i, inst in enumerate(dropper_insts)}

    def targets(inst):
        out = []
        for proxy in inst.direct("func_instance_io_proxy", "-proxy"):
            for _, target, input_ in proxy.connections():
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
                elif kind == "cubedropper" and idx in dropper_index:
                    t = ("dropper", dropper_index[idx])
                elif kind == "tbeam" and input_ in ("ProxyRelay3", "ProxyRelay4") \
                        and (kind, idx) in later["targets"]:
                    t = ("polarity", later["targets"][(kind, idx)][1])
                elif (kind, idx) in later["targets"]:
                    t = later["targets"][(kind, idx)]
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
        # The .p2c's Facing (what Level reads) is the npc's yaw + 180, to the nearest 45° the
        # tokens can say (well inside the solver's sight cones).
        yaw = (t.angles()[1] + 180) % 360
        f = round(yaw / 45)
        cell = cell_of(add(t.origin(), UP))
        toks_turrets.append((cell, f % 8))
        item_cells.append(cell)

    for c in item_cells:
        flood(c)
    later_cells = [c for c in later["cells"]]
    for c in later_cells:
        if c not in open_cells and is_open(c) and c not in later["embedded"]:
            flood(c)
    open_cells -= later["embedded"]
    # Faith plate targets: in an open cell. A plate throwing straight up throws no higher than
    # the open column over it; one with no room over it is no use, and left out.
    keep = []
    for word, fields, sender in later["items"]:
        if word == "plate":
            c, t = fields[0], fields[2]
            if (t, c) in later["vertical"]:
                top = c
                while add(top, UP) in open_cells and top[2] < t[2]:
                    top = add(top, UP)
                if top == c:
                    continue
                fields = [c, fields[1], top, fields[3]]
            elif t not in open_cells:
                near = [add(t, d) for d in DIRS if add(t, d) in open_cells]
                if not near:
                    raise Unsupported(f"faith plate target {t} isn't in the chamber")
                fields = [c, fields[1], near[0], fields[3]]
        keep.append((word, fields, sender))
    later["items"] = keep

    goo = {c for c in open_cells if world.in_water([CELL * v + CELL / 2 for v in c])}
    filled = settle_goo(goo, open_cells)
    later_item_cells = [f for _, fields, _ in later["items"] for f in fields
                        if isinstance(f, tuple) and len(f) == 3 and not isinstance(f[0], str)]
    for c in [entry, exit_c] + item_cells + later_item_cells:
        if c in filled:
            raise Unsupported(f"item at {c} under a goo surface the editor can't make")
    # Items the Puzzle Maker won't place in goo (src/editor.rs, in-goo; checked in the game).
    in_goo = [entry, exit_c] + [f[1] for f in toks_fields]
    in_goo += [b[0] for b in toks_buttons] + [p[0] for p in toks_pedestals]
    in_goo += [t[0] for t in toks_turrets]
    in_goo += [fields[0] for word, fields, _ in later["items"] if word in GOO_SHY]
    for c in in_goo:
        if c in goo:
            raise Unsupported(f"item at {c} in goo")

    # Gel lands on the floor under its dropper.
    gels = []
    for dropper, kind, on in toks_gels:
        c = dropper
        while add(c, DOWN) in open_cells:
            c = add(c, DOWN)
        gels.append((c, kind, on))

    # Glass and grating: thin window/grate brushes just inside the face they're on.
    panels = {}
    for ent, model, off in entity_models(bsp, ents):
        if model >= len(bsp.models()):
            continue
        all_brushes = bsp.brushes()
        for i in bsp.model_brushes(model):
            contents, sides = all_brushes[i]
            mats = {m.lower() for _, _, m in sides}
            # Under a lift's shaft the compiler covers its pit with grate strips (4 x 44 x 128,
            # black-metal edges, slotted for the arm) 16 units below the floor: 84 of 84 such
            # brushes in 1,507 Workshop maps sat at a lift. Read as grating, they put a panel
            # in the lift's track and the editor rejected the lift.
            if any("metalgrate" in m for m in mats) and contents & CONTENTS_GRATE:
                if any("black_wall_metal" in m for m in mats):
                    continue
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

    # Glass or grating in goo, unless it lies on the goo's surface (src/editor.rs, in-goo).
    for (c, d), kind in panels.items():
        if c in goo and not (d == UP and add(c, UP) not in goo):
            raise Unsupported(f"{kind} at {c} in goo")

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

    panel_white = later["white"]
    for c in sorted(open_cells):
        for d in DIRS:
            if add(c, d) in open_cells:
                continue
            if (c, d) in panel_white:
                white = panel_white[(c, d)]
            else:
                mat = world.surface(c, d)
                white = bool(mat and "white" in mat)
            if white:
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
    for c, kind, wait in toks_droppers:
        out += ["dropper"] + loc(c) + [kind] + (["wait"] if wait else [])

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
    for word, fields, sender in later["items"]:
        toks = [word]
        for f in fields:
            toks += loc(f) if isinstance(f, tuple) and len(f) == 3 and not isinstance(f[0], str) \
                else [f] if isinstance(f, str) else [str(f)]
        out += toks
        if sender is not None:
            out += wires(targets(sender))
    return " ".join(out)


# Later items the Puzzle Maker won't place in goo; panels, pistons, rails and bridges it
# mostly accepts there (src/editor.rs, in-goo).
GOO_SHY = {"laser", "catcher", "relay", "plate", "funnel", "stairs"}


def settle_goo(goo, open_cells):
    """Make the goo a shape the Puzzle Maker can build, and return the cells it fills in.

    The editor's goo item sits on a floor and floods sideways and down (checked in the game,
    2026-10-20: one on a ledge fills the shaft beside it to the bottom; one at the bottom of a
    pit fills only its own voxel; one over open air, over goo or over an item is invalid). So
    a pool's surface layer needs a cell with a floor under it. Workshop maps have pools two or
    more deep with none (in 1,085 of 7,302 levels; not buildable in the editor, however their
    authors made them): their cells under the surface become solid, which plays the same, since
    goo kills at any depth. Changes `goo` and `open_cells` in place.
    """
    filled = set()
    changed = True
    while changed:
        changed = False
        seen = set()
        for start in sorted(goo):
            if start in seen:
                continue
            seen.add(start)
            group = [start]
            for c in group:
                for d in DIRS[:4]:
                    n = add(c, d)
                    if n in goo and n not in seen:
                        seen.add(n)
                        group.append(n)
            # Fed from above, or with a floor for the goo item: fine as it is.
            if any(add(c, UP) in goo for c in group) or any(add(c, DOWN) not in open_cells for c in group):
                continue
            for c in group:
                below = add(c, DOWN)
                while below in goo:
                    goo.discard(below)
                    open_cells.discard(below)
                    filled.add(below)
                    below = add(below, DOWN)
            changed = True
    return filled


def dir_of(angles, v=(1, 0, 0), what="item"):
    d = rotate(angles, v)
    if d is None:
        raise Unsupported(f"{what}: rotation isn't axis-aligned")
    return d


def model_face_material(bsp, ent, normal):
    """Material of the outermost side facing `normal` of a brush entity's model."""
    m = ent.d.get("model", "")
    if not m.startswith("*"):
        return None
    all_brushes = bsp.brushes()
    o = ent.origin()
    best = None
    for i in bsp.model_brushes(int(m[1:])):
        _, sides = all_brushes[i]
        for n, dist, mat in sides:
            if all(abs(n[a] - normal[a]) < 0.01 for a in range(3)):
                along = dist + sum(normal[a] * o[a] for a in range(3))
                if not mat.lower().startswith("tools/") and (best is None or along > best[0]):
                    best = (along, mat.lower())
    return best[1] if best else None


def later_items(bsp, ents, by_kind, item_cells, blocked, is_open):
    """The items added to the tokens after turrets: their tokens (word, fields: cells as 3-tuples,
    words, numbers; and the instance whose wires follow), the wire targets they are (kind, index)
    -> (target word, position), their cells to flood from, and white faces they decide."""
    by_name = {e.name: e for e in ents}
    items, targets, cells, white, embedded = [], {}, [], {}, set()

    def branch(inst, suffix="-branch_toggle", default="1"):
        es = inst.direct("logic_branch", suffix)
        return (es[0].d.get("InitialValue", default) if es else default) == "1"

    def embed(c, d, depth=1):
        """The item's back: don't flood into the hole it sits in, which is solid in the
        Puzzle Maker's voxels."""
        blocked.add((c, d))
        for k in range(1, depth + 1):
            embedded.add(add(c, d, k))

    def section(word, rows, target=None):
        rows.sort(key=lambda r: [str(x) for x in r[0]])
        for i, (fields, inst, sends) in enumerate(rows):
            items.append((word, fields, inst if sends else None))
            if target:
                targets[(inst.kind, inst.index)] = (target, i)

    def off_centre(origin, cell, n):
        """The side of the face (against n) the origin sits toward, or "center"."""
        centre = [CELL * cell[a] + CELL / 2 - n[a] * CELL / 2 for a in range(3)]
        dev = [origin[a] - centre[a] for a in range(3)]
        a = max((a for a in range(3) if not n[a]), key=lambda a: abs(dev[a]))
        if abs(dev[a]) < 16:
            return "center"
        d = [0, 0, 0]
        d[a] = 1 if dev[a] > 0 else -1
        return DIR_TOKEN[tuple(d)]

    rows = []
    for inst in by_kind["cube"]:
        e = inst.one("prop_weighted_cube", "-cube")
        kind = int(e.d.get("CubeType", "0"))
        if not 0 <= kind < len(CUBE_KINDS):
            raise Unsupported(f"{inst.name}: cube type {kind}")
        c = cell_of(e.origin())
        rows.append(([c, CUBE_KINDS[kind]], inst, False))
        cells.append(c)
    section("cube", rows)

    for word, kind, cls, suffix in [("laser", "laseremit", "env_portal_laser", "-laser"),
                                    ("catcher", "lasercatch", "prop_laser_catcher", "-laser_catcher"),
                                    ("relay", "laserrelay", "prop_laser_relay", "-relay")]:
        rows = []
        for inst in by_kind[kind]:
            e = inst.one(cls, suffix)
            if word == "relay":
                n = dir_of(e.angles(), (0, 0, 1), inst.name)
                c = cell_of(add(e.origin(), n, 8))
            else:
                n = dir_of(e.angles(), (1, 0, 0), inst.name)
                c = cell_of(add(e.origin(), n, 32))
            # Offset items use the `_center` model too, 32 units off the face's middle.
            offset = off_centre(e.origin(), c, n)
            fields = [c, DIR_TOKEN[n], offset]
            if word == "laser":
                fields.append("on" if branch(inst) else "off")
            rows.append((fields, inst, word != "laser"))
            cells.append(c)
            embed(c, tuple(-v for v in n))
        section(word, rows, "laser" if word == "laser" else None)

    rows, vertical = [], []
    # Each plate's arm origin: a target there lies on that plate (a wall plate's voxel is hollow,
    # so the surface test below can't find it).
    plates_at = {}
    for inst in by_kind["catapult"]:
        arm = inst.one("prop_dynamic", "-catapult_arm")
        plates_at[tuple(round(v) for v in arm.origin())] = (inst.anchor_cell(arm, inst.up()), inst.up())
    for inst in by_kind["catapult"]:
        arm = inst.one("prop_dynamic", "-catapult_arm")
        n = inst.up()
        c = inst.anchor_cell(arm, n)
        trig = inst.one("trigger_catapult", "-trigger")
        tgt = by_name.get(trig.d.get("launchTarget", ""))
        if tgt is None:
            raise Unsupported(f"{inst.name}: no launch target")
        tn = UP
        if tgt.cls == "trigger_catapult":
            # Straight up: as high as the helper's speed throws (`convert` keeps it inside).
            v = float(tgt.d.get("playerSpeed", "0"))
            apex = tgt.origin()[2] + v * v / (2 * 600)
            t = (c[0], c[1], math.floor(apex / CELL))
            vertical.append((t, c))
        else:
            # The target lies on a surface: the cell on its open side, against the surface.
            o = tgt.origin()
            t, tn = cell_of(add(o, UP)), UP
            on_plate = plates_at.get(tuple(round(v) for v in o))
            for a in range(3) if on_plate is None else []:
                if abs(o[a] / CELL - round(o[a] / CELL)) < 0.01:
                    for sgn in (1, -1):
                        d = tuple(sgn if i == a else 0 for i in range(3))
                        c2 = cell_of(add(o, d))
                        if is_open(c2) and not is_open(cell_of(add(o, d, -1))):
                            t, tn = c2, d
            if on_plate is not None:
                t, tn = on_plate
        rows.append(([c, DIR_TOKEN[n], t, DIR_TOKEN[tn]], inst, False))
        cells += [c, t]
        embed(c, tuple(-v for v in n))
    section("plate", rows)

    for word, kind, cls, suffix in [("bridge", "bridge", "prop_wall_projector", "-bridge"),
                                    ("funnel", "tbeam", "prop_tractor_beam", "-tbeam")]:
        rows = []
        for inst in by_kind[kind]:
            e = inst.one(cls, suffix)
            n = dir_of(e.angles(), (1, 0, 0), inst.name)
            c = cell_of(add(e.origin(), n, 32))
            if word == "bridge":
                side = dir_of(e.angles(), (0, 0, 1), inst.name)
                fields = [c, DIR_TOKEN[n], DIR_TOKEN[side], "on" if branch(inst) else "off"]
            else:
                fields = [c, DIR_TOKEN[n], "on" if branch(inst) else "off"]
                if branch(inst, "-branch_toggle_polarity", "0"):
                    fields.append("rev")
            rows.append((fields, inst, False))
            cells.append(c)
            embed(c, tuple(-v for v in n))
        section(word, rows, word)

    for word, kind in [("panel", "angledPanel"), ("glasspanel", "angledClearPanel")]:
        rows = []
        for inst in by_kind[kind]:
            arms = inst.one("prop_dynamic", "-model_arms")
            n, tilt = inst.up(), dir_of(inst.angles(), (1, 0, 0), inst.name)
            c = cell_of(add(arms.origin(), n, 8))
            m = re.match(r"ramp_(\d+)_deg_open", arms.d.get("DefaultAnim", "ramp_45_deg_open"))
            angle = int(m.group(1)) if m else 45
            if angle not in (30, 45, 60, 90):
                raise Unsupported(f"{inst.name}: angle {angle}")
            face = (c, tuple(-v for v in n))
            if word == "panel":
                tops = [e for e in inst.ents if e.cls == "func_brush" and "panel_top" in e.name]
                mat = model_face_material(bsp, tops[0], n) if tops else None
                white[face] = bool(mat and "white" in mat)
            else:
                white[face] = False
            rows.append(([c, DIR_TOKEN[n], DIR_TOKEN[tilt], str(angle // 15),
                          "on" if branch(inst) else "off"], inst, False))
            cells.append(c)
            embed(c, tuple(-v for v in n))
        section(word, rows, word)

    rows = []
    for inst in by_kind["flipPanel"]:
        door = inst.one("func_door_rotating", "-flipping_panel")
        n = inst.up()
        c = cell_of(add(door.origin(), n, 20))
        mat = model_face_material(bsp, door, n)
        flipped = branch(inst, default="0")
        # The face's own whiteness is the side out at the start.
        white[(c, tuple(-v for v in n))] = bool(mat and "white" in mat) != flipped
        rows.append(([c, DIR_TOKEN[n]], inst, False))
        cells.append(c)
        embed(c, tuple(-v for v in n))
    section("flip", rows, "flip")

    rows = []
    for inst in by_kind["stairs"]:
        step1 = inst.one("func_brush", "-brush_step_1")
        if inst.up() != UP:
            raise Unsupported(f"{inst.name}: stairs on a wall or ceiling")
        d = dir_of(inst.angles(), (1, 0, 0), inst.name)
        c = cell_of(add(step1.origin(), UP, 8))
        rows.append(([c, DIR_TOKEN[d], "on" if branch(inst) else "off"], inst, False))
        cells += [c, add(c, d)]
        embed(c, DOWN)
        embed(add(c, d), DOWN)
    section("stairs", rows, "stairs")

    rows = []
    for inst in by_kind["pistonlift"]:
        lift = inst.one("func_movelinear", "-lift_platform")
        n = inst.up()
        c = cell_of(add(lift.origin(), n, 4))
        level = lambda suffix, dflt: int(float(next(
            (e.d.get("startvalue", dflt) for e in inst.direct("math_counter", suffix)), dflt)))
        lo, hi = level("-counter_bottom_level", "0"), level("-counter_top_level", "1")
        if not (0 <= lo < hi <= 4):
            raise Unsupported(f"{inst.name}: levels {lo}..{hi}")
        up = branch(inst, "-branch_starting_state", "0")
        rows.append(([c, DIR_TOKEN[n], str(lo), str(hi), "on" if up else "off"], inst, False))
        cells += [add(c, n, k) for k in range(lo, hi + (n == UP))]
        embed(c, tuple(-v for v in n), 2)
    section("piston", rows, "piston")

    rows = []
    for inst in by_kind["raillift"]:
        track = inst.one("func_movelinear", "-track_lift_movelinear")
        model = inst.one("prop_dynamic", "-track_lift_model")
        d = dir_of(vec(track.d.get("movedir", "0 0 0")), (1, 0, 0), inst.name)
        n = round(float(track.d.get("movedistance", "0")) / CELL)
        # The model's origin is on a face of the item's voxel (on a wall rail, its top face);
        # the platform's movelinear sits 68 units from it, inside the voxel. Behind the start
        # cell the compiler always leaves a hollow (553/553 rails in data/ws3). The platform
        # starts `startposition` of the way along its track (tokens can't say; ours start at 0).
        at = round(float(track.d.get("startposition", "0")) * n)
        start = add(cell_of(track.origin()), d, -at)
        end = add(start, d, n)
        back = tuple(-v for v in inst.up())
        for k in range(n + 1):
            embed(add(start, d, k), back)
        oscillate = bool(inst.direct("logic_branch", "-branch_active"))
        if oscillate:
            mode = "osc"
        else:
            mode = "on" if branch(inst, "-branch", "0") else "off"
        rows.append(([start, end, mode], inst, False))
        cells += [add(start, d, k) for k in range(n + 1)]
    section("rail", rows, "rail")

    return {"items": items, "targets": targets, "cells": cells, "white": white,
            "embedded": embedded, "vertical": vertical}


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
