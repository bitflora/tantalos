"""Draw the Puzzle Maker's "Open" screen preview for a .p2c: an isometric view and a list of items.

The editor shows <name>.jpg (555x312) next to <name>.p2c as a level's thumbnail and only renders
one itself when the level is saved in the editor, so generated levels have none. This draws a
stand-in: the chamber in oblique 2.5D on the left (Z up, the near walls cut down to a low lip so
you can see in; light = portalable (white) surface, dark = black surface, lit from above with the
corners shaded; teal = goo; fields, glass and gel as tinted planes on their faces; buttons and
faith plates as discs on the floor, other items as balls over their shadows, laser beams in red,
entry and exit as squares), the level name, size and item counts on the right. The exact faces of fields,
glass and gel and the beams come from `tantalos-p2c geometry` (without the release binary, barriers are lines). The projection is the one
`tantalos-view` (src/viewer.html) draws with, turned to the corner the editor's own preview looks
from, so a level is oriented the way the game shows it. playtest.py calls it for every level it
copies in.

    python train/preview.py level.p2c [more.p2c ...]       # writes level.jpg next to each
"""
import argparse, json, math, re, subprocess
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 555, 312
MAP = 300  # square map area on the left
BG, TEXT, DIM = (38, 40, 44), (235, 235, 235), (150, 155, 160)
# Surfaces: portalable (white) panels light and warm, black panels dark and blue-grey. Floors get
# brighter with height; the far walls are shaded by which way they face.
WHITE_FLOOR, BLACK_FLOOR = ((150, 150, 144), (238, 236, 226)), ((44, 48, 56), (96, 102, 114))
WALL_WHITE, WALL_WHITE_X = (196, 196, 188), (172, 172, 165)
WALL, WALL_X = (52, 57, 64), (46, 51, 58)
EDGE_WHITE, EDGE_BLACK = (110, 110, 104), (24, 26, 30)
GOO, GOO_EDGE = (36, 118, 92), (90, 190, 150)
LIP, LIP_TOP = 0.3, (210, 214, 220, 150)  # the near walls are cut this high above the floor
SHADOW = (0, 0, 0, 80)
PLANES = {"fizzler": (120, 170, 255, 70), "laserfield": (255, 110, 220, 70),
          "glass": (150, 215, 230, 60), "grating": (182, 196, 206, 90)}
FLAT = {"ITEM_BUTTON_FLOOR", "ITEM_CATAPULT"}  # items that lie on the floor, drawn as discs
P2C_EXE = Path(__file__).resolve().parents[1] / "target/release/tantalos-p2c.exe"
TURNS = 3  # quarter-turns of the level about z, to the corner the editor's own preview looks from

TOKEN = re.compile(r'"((?:[^"\\]|\\.)*)"|([{}])')


def parse_kv(text):
    """KeyValues -> nested list of (key, str | list)."""
    stack, cur, key = [], [], None
    for m in TOKEN.finditer(text):
        if m[2] == "{":
            node = []
            cur.append((key, node))
            stack.append(cur)
            cur, key = node, None
        elif m[2] == "}":
            cur = stack.pop()
        elif key is None:
            key = m[1]
        else:
            cur.append((key, m[1]))
            key = None
    return cur


def block(node, name):
    return next((v for k, v in node if k == name and isinstance(v, list)), [])


def ints(s):
    return [int(float(v)) for v in s.split()]


def load(path):
    root = block(parse_kv(Path(path).read_text(encoding="utf-8")), "portal2_puzzle")
    props = {k: v for k, v in root if isinstance(v, str)}
    size = ints(props["ChamberSize"])
    # Solid, and Portal0..2: the bit at lattice voxel v says the face between v - e_axis and v is
    # portalable (src/p2c.rs).
    layers = {name: set() for name in ("Solid", "Portal0", "Portal1", "Portal2")}
    for name, zs in block(root, "Voxels"):
        if name in layers and isinstance(zs, list):
            for zk, rows in zs:
                for yk, bits in rows:
                    z, y = int(zk[1:]), int(yk[1:])
                    layers[name].update((x, y, z) for x, b in enumerate(bits[1:]) if b == "1")
    solid = layers["Solid"]
    portal = [layers[f"Portal{a}"] for a in range(3)]
    items = [dict(kv) for k, kv in block(root, "Items") if k == "Item"]
    conns = [dict(kv) for k, kv in block(root, "Connections") if k == "Connection"]
    return props, size, solid, portal, items, conns


def num(item, key):
    try:
        return int(float(item.get(key, "0")))
    except ValueError:
        return 0


GELS = ["Bounce gel", "Speed gel", "Portal gel", "Water"]
GEL_COLOURS = [(40, 110, 255), (255, 140, 20), (225, 225, 225), (110, 200, 255)]


def describe(item):
    """(label, colour) for an item worth showing, or None."""
    t = item.get("Type", "")
    if item.get("Enabled") == "0" or t.endswith("_EXTENT") or "COOP" in t:
        return None
    if t == "ITEM_ENTRY_DOOR":
        return "Entry", (60, 220, 90)
    if t == "ITEM_EXIT_DOOR":
        return "Exit", (255, 70, 60)
    if t == "ITEM_CUBE":
        kind = ["Cube", "Companion cube", "Reflection cube", "Sphere", "Franken cube"]
        return kind[min(num(item, "ITEM_PROPERTY_CUBE_TYPE"), 4)], (200, 200, 200)
    if t == "ITEM_BUTTON_FLOOR":
        kind = ["Floor button", "Cube button", "Sphere button"]
        return kind[min(num(item, "ITEM_PROPERTY_BUTTON_TYPE"), 2)], (230, 50, 90)
    if t == "ITEM_BUTTON_PEDESTAL":
        return "Pedestal button", (230, 50, 90)
    if t in ("ITEM_PAINT_SPLAT", "ITEM_DROPPER_PAINT"):
        k = min(num(item, "ITEM_PROPERTY_PAINT_TYPE"), 3)
        return GELS[k], GEL_COLOURS[k]
    if t == "ITEM_BARRIER_HAZARD":
        if num(item, "ITEM_PROPERTY_BARRIER_HAZARD_TYPE"):
            return "Laser field", (255, 110, 220)
        return "Fizzler", (120, 170, 255)
    if t == "ITEM_BARRIER":
        return ("Grating" if num(item, "ITEM_PROPERTY_BARRIER_TYPE") else "Glass"), (170, 230, 230)
    if t == "ITEM_TURRET":
        return "Turret", (255, 225, 40)
    if t.startswith("ITEM_LASER_EMITTER"):
        return "Laser", (255, 40, 40)
    if t.startswith("ITEM_LASER_CATCHER"):
        return "Laser catcher", (200, 40, 40)
    if t.startswith("ITEM_LASER_RELAY"):
        return "Laser relay", (200, 40, 40)
    if t == "ITEM_CATAPULT":
        return "Faith plate", (160, 230, 60)
    if t == "ITEM_LIGHT_BRIDGE":
        return "Light bridge", (140, 210, 255)
    if t == "ITEM_TBEAM":
        return "Funnel", (170, 100, 255)
    panels = {"ITEM_PANEL_ANGLED": "Angled panel", "ITEM_PANEL_FLIP": "Flip panel",
              "ITEM_PANEL_STAIRS": "Stairs", "ITEM_PANEL_CLEAR": "Glass panel",
              "ITEM_PISTON_PLATFORM": "Piston", "ITEM_RAIL_PLATFORM": "Rail platform"}
    if t in panels:
        return panels[t], (210, 170, 110)
    return None  # lights, antlines, observation rooms, BEEmod items


def font(px, bold=False):
    for name in (["arialbd.ttf"] if bold else []) + ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            pass
    return ImageFont.load_default(px)


def geometry(p2c):
    """The level's geometry from `tantalos-p2c geometry` (src/view.rs: the exact faces of fields,
    glass and gel, and each item's facing), or None when the binary isn't built."""
    if not P2C_EXE.exists():
        return None
    try:
        out = subprocess.run([str(P2C_EXE), "geometry", str(p2c)], capture_output=True, text=True,
                             timeout=60, check=True).stdout
        return json.loads(out.splitlines()[0])
    except (subprocess.SubprocessError, OSError, ValueError, IndexError):
        return None


def shade(c, f):
    return tuple(max(0, min(255, int(v * f))) for v in c[:3]) + tuple(c[3:])


def mix(a, b, t):
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


def background():
    """BG with a soft light behind the room, darker towards the corners."""
    glow = Image.radial_gradient("L").resize((int(MAP * 1.6), int(MAP * 1.6)))
    img = Image.new("RGB", (W, H), BG)
    light = Image.new("RGB", glow.size, (62, 66, 74))
    mask = glow.point(lambda v: max(0, 255 - int(v * 1.45)))  # 0 by the box's edges
    img.paste(light, (MAP // 2 - glow.width // 2, H // 2 - glow.height // 2), mask)
    return img


def render(p2c, name=None):
    props, (sx, sy, sz), solid, portal, items, conns = load(p2c)
    geom = geometry(p2c)
    img = background()
    d = ImageDraw.Draw(img, "RGBA")

    goo = {tuple(ints(i["VoxelPos"])[:3]) for i in items if i.get("Type") == "ITEM_GOO"}

    def is_solid(x, y, z):
        return (x, y, z) in solid if 0 <= x < sx and 0 <= y < sy and 0 <= z < sz else True

    def rotxy(x, y):  # the level turned TURNS quarter turns about z
        if TURNS % 4 == 1:
            return -y, x
        if TURNS % 4 == 2:
            return -x, -y
        if TURNS % 4 == 3:
            return y, -x
        return x, y

    # Oblique 2.5D, Z up: the projection tantalos-view draws with, the level turned TURNS quarter
    # turns first. `rx + ry` is depth, so cells with a larger one are nearer (drawn last).
    def project(x, y, z):
        rx, ry = rotxy(x, y)
        return (rx - ry) * 0.5, (rx + ry) * 0.25 - z, rx + ry

    box = MAP - 12
    us = [project(x, y, z)[0] for x in (0, sx) for y in (0, sy) for z in (0, sz)]
    vs = [project(x, y, z)[1] for x in (0, sx) for y in (0, sy) for z in (0, sz)]
    scale = min((box - 8) / max(1e-6, max(us) - min(us)), (box - 8) / max(1e-6, max(vs) - min(vs)))
    cu, cv = (min(us) + max(us)) / 2, (min(vs) + max(vs)) / 2
    ox, oy = 6 + box / 2, 6 + box / 2

    def screen(x, y, z):  # a voxel point -> an image point
        u, v, _ = project(x, y, z)
        return ox + (u - cu) * scale, oy + (v - cv) * scale

    def depth(x, y):
        return project(x, y, 0)[2]

    def face(x, y, z, dr, h=1.0):  # the corners of cell (x, y, z)'s face toward dr (walls h high)
        if dr == "-z":
            return [(x, y, z), (x + 1, y, z), (x + 1, y + 1, z), (x, y + 1, z)]
        if dr == "+z":
            return [(x, y, z + 1), (x + 1, y, z + 1), (x + 1, y + 1, z + 1), (x, y + 1, z + 1)]
        (ax, ay), (bx, by) = {"-x": ((0, 0), (0, 1)), "+x": ((1, 0), (1, 1)),
                              "-y": ((0, 0), (1, 0)), "+y": ((0, 1), (1, 1))}[dr]
        return [(x + ax, y + ay, z), (x + bx, y + by, z), (x + bx, y + by, z + h), (x + ax, y + ay, z + h)]

    def white(x, y, z, dr):  # whether cell (x, y, z)'s face toward dr is portalable
        axis, sign = {"-x": (0, 0), "+x": (0, 1), "-y": (1, 0), "+y": (1, 1), "-z": (2, 0), "+z": (2, 1)}[dr]
        v = [x, y, z]
        v[axis] += sign
        return tuple(v) in portal[axis]

    def poly(points, fill, edge=None):
        d.polygon([screen(*p) for p in points], fill=fill, outline=edge)

    def disc(x, y, z, r, fill, edge=None):  # a flat circle lying at height z, centred on (x, y)
        pts = [(x + r * math.cos(a), y + r * math.sin(a), z) for a in (i * math.pi / 10 for i in range(20))]
        poly(pts, fill, edge)

    def ground(x, y, z):  # the height of the floor under cell (x, y, z)
        while z > 0 and not is_solid(x, y, z - 1):
            z -= 1
        return z

    steps = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}
    near = {dr for dr, (dx, dy) in steps.items() if sum(rotxy(dx, dy)) > 0.01}
    top = max(1, sz - 1)

    ops = []  # (depth, rank, draw): painter's order, far to near
    for z in range(sz):
        lift = 0.88 + 0.24 * z / top  # light from above: higher surfaces brighter
        for y in range(sy):
            for x in range(sx):
                if is_solid(x, y, z):
                    continue
                dep = depth(x + .5, y + .5)
                floored = is_solid(x, y, z - 1) or (x, y, z) in goo
                # Corners and edges of the room collect less light (ambient occlusion).
                walls_here = sum(is_solid(x + dx, y + dy, z) for dx, dy in steps.values())
                if (x, y, z) in goo:
                    ops.append((dep, 0, lambda x=x, y=y, z=z: poly(face(x, y, z, "-z"), GOO, GOO_EDGE)))
                elif floored:
                    lo, hi = WHITE_FLOOR if white(x, y, z, "-z") else BLACK_FLOOR
                    fill = shade(mix(lo, hi, z / top), 1 - 0.07 * walls_here)
                    ops.append((dep, 0, lambda x=x, y=y, z=z, f=fill: poly(face(x, y, z, "-z"), f, shade(f, 0.82))))
                for dr, (dx, dy) in steps.items():
                    if not is_solid(x + dx, y + dy, z):
                        continue
                    w, side = white(x, y, z, dr), dx != 0
                    base = (WALL_WHITE if side else WALL_WHITE_X) if w else (WALL if side else WALL_X)
                    if dr in near:  # cut away, leaving a low lip at floor level
                        if floored:
                            f = shade(base, 1.08)
                            ops.append((dep + .5, 5, lambda x=x, y=y, z=z, dr=dr, f=f: (
                                poly(face(x, y, z, dr, LIP), f, shade(f, 0.7)),
                                d.line([screen(*p) for p in face(x, y, z, dr, LIP)[2:]], fill=LIP_TOP, width=1))))
                        continue
                    f = shade(base, lift * (0.9 if floored else 1.0))
                    ops.append((dep, 1, lambda x=x, y=y, z=z, dr=dr, f=f: poly(face(x, y, z, dr), f, shade(f, 0.8))))

    if geom:  # fields, glass and gel on their real faces
        for b in geom.get("barriers", []) + geom.get("hazards", []):
            fill = PLANES.get(b["kind"], (200, 200, 200, 60))
            for c, dr in b["faces"]:
                ops.append((depth(c[0] + .5, c[1] + .5), 2, lambda c=c, dr=dr, fill=fill: (
                    poly(face(c[0], c[1], c[2], dr), fill, fill[:3] + (150,)))))
        for p in geom.get("paint", []):
            c, dr = p["face"]
            kinds = ["bounce", "speed", "portal", "water"]
            fill = GEL_COLOURS[kinds.index(p["kind"]) if p["kind"] in kinds else 0] + (170,)
            ops.append((depth(c[0] + .5, c[1] + .5), 2, lambda c=c, dr=dr, fill=fill: (
                poly(face(c[0], c[1], c[2], dr), fill))))

    by_index = {i.get("Index"): i for i in items}
    shown = [(i, describe(i)) for i in items]
    colour_of = {i.get("Index"): desc[1] for i, desc in shown if desc}
    lines = ["CONNECTION_PISTON_ANCHOR_TO_EXTENT", "CONNECTION_RAIL_ANCHOR_TO_EXTENT"]
    if not geom:
        lines.append("CONNECTION_BARRIER_ANCHOR_TO_EXTENT")
    for c in conns:  # tracks (and, without the geometry, barrier extents), drawn where they run
        if c.get("Type") in lines:
            a, b = by_index.get(c.get("Sender")), by_index.get(c.get("Receiver"))
            if a and b and a.get("Index") in colour_of and "VoxelPos" in a and "VoxelPos" in b:
                pa, pb = ints(a["VoxelPos"]), ints(b["VoxelPos"])
                ops.append(((depth(pa[0] + .5, pa[1] + .5) + depth(pb[0] + .5, pb[1] + .5)) / 2, 3,
                            lambda pa=pa, pb=pb, col=colour_of[a["Index"]]: d.line(
                                [screen(pa[0] + .5, pa[1] + .5, pa[2] + .5),
                                 screen(pb[0] + .5, pb[1] + .5, pb[2] + .5)],
                                fill=col, width=max(2, int(scale * 0.14)))))

    for beam in (geom or {}).get("beams") or []:  # laser beams, as the solver traces them
        if len(beam) >= 2:
            dep = sum(depth(p[0], p[1]) for p in beam) / len(beam)
            ops.append((dep, 4, lambda beam=beam: (
                d.line([screen(*p) for p in beam], fill=(255, 60, 60, 200), width=max(2, int(scale * 0.1))),
                d.line([screen(*p) for p in beam], fill=(255, 210, 210, 230), width=1))))

    facing = {str(g.get("i")): g.get("d") for g in (geom or {}).get("items", [])}
    r = max(3, min(12, int(scale * 0.2)))
    doors = []
    for i, desc in shown:
        if not desc or "VoxelPos" not in i:
            continue
        x, y, z = ints(i["VoxelPos"])[:3]
        if desc[0] in ("Entry", "Exit"):
            doors.append((desc, (x, y, z)))
            continue
        dep = depth(x + .5, y + .5)
        if i.get("Type") in FLAT and facing.get(i.get("Index"), "+z") == "+z" and is_solid(x, y, z - 1):
            ops.append((dep, 3, lambda x=x, y=y, z=z, col=desc[1]: (  # lies on the floor: a disc
                disc(x + .5, y + .5, z + .03, 0.36, col, shade(col, 0.55)),
                disc(x + .5, y + .5, z + .05, 0.2, shade(col, 0.8)))))
            continue

        def ball(x=x, y=y, z=z, g=ground(x, y, z), col=desc[1]):
            disc(x + .5, y + .5, g + .02, 0.3, SHADOW)
            cx, cy = screen(x + .5, y + .5, z + .45)
            if z - g >= 1:  # high up: a thin stalk down to its shadow shows how high
                d.line([(cx, cy), screen(x + .5, y + .5, g + .02)], fill=(0, 0, 0, 90), width=1)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col, outline=shade(col, 0.45))
            h = max(1, r // 3)
            d.ellipse([cx - r * .45 - h, cy - r * .45 - h, cx - r * .45 + h, cy - r * .45 + h],
                      fill=(255, 255, 255, 110))
        ops.append((dep, 4, ball))

    for _, _, draw in sorted(ops, key=lambda o: (o[0], o[1])):
        draw()

    s = r + 1
    for (label, col), (x, y, z) in doors:  # on top, so they're never hidden
        cx, cy = screen(x + .5, y + .5, z + .5)
        d.rectangle([cx - s + 2, cy - s + 3, cx + s + 2, cy + s + 3], fill=(0, 0, 0, 90))
        d.rectangle([cx - s, cy - s, cx + s, cy + s], fill=col, outline=shade(col, 0.45))

    counts = Counter(desc for _, desc in shown if desc and desc[0] not in ("Entry", "Exit"))
    x0, y = MAP + 10, 10
    title = name or props.get("Title", Path(p2c).stem)
    d.text((x0, y), title[:28], font=font(17, True), fill=TEXT)
    y += 24
    d.text((x0, y), f"{sx} x {sy} x {sz}", font=font(13), fill=DIM)
    y += 24
    f = font(14)
    rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0][0]))
    fit = (H - 24 - y) // 19
    if len(rows) > fit:
        more, rows = len(rows) - fit + 1, rows[:fit - 1]
        d.text((x0, y + 19 * len(rows)), f"+ {more} more", font=f, fill=DIM)
    for (label, colour), count in rows:
        d.ellipse([x0, y + 3, x0 + 10, y + 13], fill=colour)
        d.text((x0 + 16, y), f"{label}" + (f" x{count}" if count > 1 else ""), font=f, fill=TEXT)
        y += 19
    if not rows:
        d.text((x0, y), "no items", font=f, fill=DIM)
    # Key, bottom right.
    k, ky = font(11), H - 17
    d.rectangle([x0, ky + 2, x0 + 9, ky + 11], fill=(60, 220, 90))
    d.text((x0 + 13, ky), "entry", font=k, fill=DIM)
    d.rectangle([x0 + 50, ky + 2, x0 + 59, ky + 11], fill=(255, 70, 60))
    d.text((x0 + 63, ky), "exit", font=k, fill=DIM)
    d.rectangle([x0 + 92, ky + 2, x0 + 101, ky + 11], fill=WALL_WHITE, outline=EDGE_WHITE)
    d.text((x0 + 105, ky), "portalable", font=k, fill=DIM)
    d.rectangle([x0 + 167, ky + 2, x0 + 176, ky + 11], fill=BLACK_FLOOR[1], outline=(0, 0, 0))
    d.text((x0 + 180, ky), "not", font=k, fill=DIM)
    return img


def summary(p2c):
    """One-line item list, for the level's Description."""
    _, (sx, sy, sz), _, _, items, _ = load(p2c)
    counts = Counter(desc[0] for desc in map(describe, items) if desc and desc[0] not in ("Entry", "Exit"))
    parts = [f"{n} {label}" if n > 1 else label for label, n in counts.most_common()]
    return f"{sx}x{sy}x{sz}: " + (", ".join(parts) or "no items")


def write(p2c, jpg=None, name=None):
    jpg = Path(jpg or Path(p2c).with_suffix(".jpg"))
    render(p2c, name).save(jpg, quality=90)
    return jpg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("levels", nargs="+")
    for p in ap.parse_args().levels:
        print(write(p))


if __name__ == "__main__":
    main()
