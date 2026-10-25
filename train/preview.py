"""Draw the Puzzle Maker's "Open" screen preview for a .p2c: an isometric view and a list of items.

The editor shows <name>.jpg (555x312) next to <name>.p2c as a level's thumbnail and only renders
one itself when the level is saved in the editor, so generated levels have none. This draws a
stand-in: the chamber in oblique 2.5D on the left (Z up, the near walls cut away so you can see
in; darker = lower floor, teal = goo, grey = wall, and items as coloured dots, barriers and
tracks as lines), the level name, size and item counts on the right. The projection is the one
`glados-view` (src/viewer.html) draws with, turned to the corner the editor's own preview looks
from, so a level is oriented the way the game shows it. playtest.py calls it for every level it
copies in.

    python train/preview.py level.p2c [more.p2c ...]       # writes level.jpg next to each
"""
import argparse, re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 555, 312
MAP = 300  # square map area on the left
BG, TEXT, DIM = (38, 40, 44), (235, 235, 235), (150, 155, 160)
WALL, WALL_X = (52, 57, 64), (46, 51, 58)  # the far walls, shaded by which way they face
GOO = (40, 120, 90)
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
    solid = set()
    for zk, rows in block(block(root, "Voxels"), "Solid"):
        for yk, bits in rows:
            z, y = int(zk[1:]), int(yk[1:])
            solid.update((x, y, z) for x, b in enumerate(bits[1:]) if b == "1")
    items = [dict(kv) for k, kv in block(root, "Items") if k == "Item"]
    conns = [dict(kv) for k, kv in block(root, "Connections") if k == "Connection"]
    return props, size, solid, items, conns


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


def render(p2c, name=None):
    props, (sx, sy, sz), solid, items, conns = load(p2c)
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

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

    # Oblique 2.5D, Z up: the projection glados-view draws with, the level turned TURNS quarter
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

    def quad(points, fill):
        d.polygon([screen(*p) for p in points], fill=fill, outline=(0, 0, 0))

    def floor(x, y, z):  # the four corners of cell (x, y, z)'s floor
        return [(x, y, z), (x + 1, y, z), (x + 1, y + 1, z), (x, y + 1, z)]

    def wall(x, y, z, dr):  # the four corners of a vertical face
        if dr == "-x":
            return [(x, y, z), (x, y + 1, z), (x, y + 1, z + 1), (x, y, z + 1)]
        if dr == "+y":
            return [(x, y + 1, z), (x + 1, y + 1, z), (x + 1, y + 1, z + 1), (x, y + 1, z + 1)]
        if dr == "+x":
            return [(x + 1, y, z), (x + 1, y + 1, z), (x + 1, y + 1, z + 1), (x + 1, y, z + 1)]
        return [(x, y, z), (x + 1, y, z), (x + 1, y, z + 1), (x, y, z + 1)]  # -y

    # Floors and walls. Only the walls facing away from the camera are drawn: the near ones are
    # cut away so the room reads from the front (glados-view's `nearFace`).
    walls = [s for s in (("+x", 1, 0), ("-x", -1, 0), ("+y", 0, 1), ("-y", 0, -1))
             if sum(rotxy(s[1], s[2])) <= 0.01]

    ops = []
    for z in range(sz):
        for y in range(sy):
            for x in range(sx):
                if is_solid(x, y, z):
                    continue
                if (x, y, z) in goo:
                    fill = GOO
                elif is_solid(x, y, z - 1):
                    v = 70 + int(110 * z / max(1, sz - 1))
                    fill = (v, v + 4, v + 10)
                else:
                    fill = None
                dep = project(x + .5, y + .5, z)[2]
                if fill is not None:
                    ops.append((dep, 0, floor(x, y, z), fill))
                for dr, dx, dy in walls:
                    if is_solid(x + dx, y + dy, z):
                        ops.append((dep, 1, wall(x, y, z, dr), WALL if dx else WALL_X))
    for _, _, points, fill in sorted(ops, key=lambda o: (o[0], o[1])):
        quad(points, fill)

    by_index = {i.get("Index"): i for i in items}
    shown = [(i, describe(i)) for i in items]
    colour_of = {i.get("Index"): desc[1] for i, desc in shown if desc}
    for c in conns:  # barrier and track extents, drawn where they run
        if c.get("Type") in ("CONNECTION_BARRIER_ANCHOR_TO_EXTENT", "CONNECTION_PISTON_ANCHOR_TO_EXTENT",
                             "CONNECTION_RAIL_ANCHOR_TO_EXTENT"):
            a, b = by_index.get(c.get("Sender")), by_index.get(c.get("Receiver"))
            if a and b and a.get("Index") in colour_of and "VoxelPos" in a and "VoxelPos" in b:
                pa, pb = ints(a["VoxelPos"]), ints(b["VoxelPos"])
                d.line([screen(pa[0] + .5, pa[1] + .5, pa[2] + .5), screen(pb[0] + .5, pb[1] + .5, pb[2] + .5)],
                       fill=colour_of[a["Index"]], width=max(2, int(scale * 0.14)))

    def depth(item):  # how near an item's cell is, for painter's order
        x, y, z = ints(item["VoxelPos"])
        return project(x + .5, y + .5, z)[2]

    r = max(3, min(14, int(scale * 0.22)))
    for i, desc in sorted((s for s in shown if s[1] and "VoxelPos" in s[0]),
                          key=lambda s: (s[1][0] in ("Entry", "Exit"), depth(s[0]))):
        x, y, z = ints(i["VoxelPos"])
        cx, cy = screen(x + .5, y + .5, z + .5)
        shape = d.rectangle if desc[0] in ("Entry", "Exit") else d.ellipse
        shape([cx - r, cy - r, cx + r, cy + r], fill=desc[1], outline=(0, 0, 0))

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
    d.text((x0 + 63, ky), "exit   lighter floor = higher", font=k, fill=DIM)
    return img


def summary(p2c):
    """One-line item list, for the level's Description."""
    _, (sx, sy, sz), _, items, _ = load(p2c)
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
