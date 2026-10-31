"""Builds the Goblin bot (Portal 2 player model) in Blender, rigidly skinned to Chell's skeleton.

    blender -b --factory-startup -P tools/goblin_model/goblin_blender.py -- SKELETON.json OUTDIR [--no-bake]

SKELETON.json comes from mdl_info.py --json on models/player/chell/player.mdl. Writes to OUTDIR:
goblin.smd (reference mesh + Chell's skeleton), goblin_color.png, goblin_emit.png, goblin_normal.png
and preview-*.png renders. --no-bake renders the procedural materials only (fast look).

Coordinates: built in Blender's frame (Z up, facing -Y, inches). Chell's mesh space ("M") is Y up,
facing +Z, and the SMD is written in it (Portal 2's studiomdl doesn't turn SMDs about Z).
"""
import bmesh
import bpy
import json
import math
import os
import sys
from mathutils import Matrix, Vector

ARGV = sys.argv[sys.argv.index("--") + 1:]
SKEL_PATH, OUT = ARGV[0], ARGV[1]
NO_BAKE = "--no-bake" in ARGV
os.makedirs(OUT, exist_ok=True)

SKEL = json.load(open(SKEL_PATH))
B_FROM_M = Matrix(((1, 0, 0), (0, 0, -1), (0, 1, 0)))
M_FROM_B = B_FROM_M.transposed()

J = {}
for _b in SKEL["bones"]:
    _w = Matrix([_b["pose_to_bone"][0], _b["pose_to_bone"][1], _b["pose_to_bone"][2], [0, 0, 0, 1]]).inverted()
    J[_b["name"]] = B_FROM_M @ _w.translation

bpy.ops.wm.read_factory_settings(use_empty=True)
SCENE = bpy.context.scene
PARTS = []
MATS = {}


# ---------------------------------------------------------------- materials

def node_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs[0], out.inputs[0])
    MATS[name] = m
    return m, nt, bsdf


def coords(nt, obj=None):
    tc = nt.nodes.new("ShaderNodeTexCoord")
    if obj is not None:
        tc.object = obj
    return tc.outputs["Object"]


def noise(nt, vec, scale, detail=6.0, rough=0.55, distort=0.0):
    n = nt.nodes.new("ShaderNodeTexNoise")
    n.inputs["Scale"].default_value = scale
    n.inputs["Detail"].default_value = detail
    n.inputs["Roughness"].default_value = rough
    n.inputs["Distortion"].default_value = distort
    nt.links.new(vec, n.inputs["Vector"])
    return n.outputs["Fac"]


def ramp(nt, fac, stops):
    r = nt.nodes.new("ShaderNodeValToRGB")
    els = r.color_ramp.elements
    els[0].position, els[0].color = stops[0][0], (*[stops[0][1]] * 3, 1)
    els[1].position, els[1].color = stops[1][0], (*[stops[1][1]] * 3, 1)
    for pos, val in stops[2:]:
        e = els.new(pos)
        e.color = (val, val, val, 1)
    nt.links.new(fac, r.inputs["Fac"])
    return r.outputs["Color"]


def mix(nt, fac, a, b, blend="MIX"):
    m = nt.nodes.new("ShaderNodeMix")
    m.data_type = "RGBA"
    m.blend_type = blend
    for k, val in ((0, fac), (6, a), (7, b)):
        sock = m.inputs[k]
        if isinstance(val, (int, float)):
            sock.default_value = val if k == 0 else (val, val, val, 1)
        elif isinstance(val, tuple):
            sock.default_value = (*val, 1) if len(val) == 3 else val
        else:
            nt.links.new(val, sock)
    return m.outputs[2]


def math_node(nt, op, a, b=None):
    m = nt.nodes.new("ShaderNodeMath")
    m.operation = op
    for sock, val in ((m.inputs[0], a), (m.inputs[1], b)):
        if val is None:
            continue
        if isinstance(val, (int, float)):
            sock.default_value = val
        else:
            nt.links.new(val, sock)
    return m.outputs[0]


def ao(nt, dist):
    a = nt.nodes.new("ShaderNodeAmbientOcclusion")
    a.inputs["Distance"].default_value = dist
    a.samples = 16
    return ramp(nt, a.outputs["AO"], [(0.0, 0.35), (1.0, 1.0)])


def bump(nt, height, strength, dist=0.2):
    b = nt.nodes.new("ShaderNodeBump")
    b.inputs["Strength"].default_value = strength
    b.inputs["Distance"].default_value = dist
    nt.links.new(height, b.inputs["Height"])
    return b.outputs["Normal"]


def worn(nt, bsdf, base, chip_color, chip_amount, green, cracks, rough, metallic=0.0):
    """Painted shell: chipped to metal, sage-green patches, grime, hairline cracks, cavity dirt."""
    v = coords(nt)
    big = noise(nt, v, 0.09, 4.0, 0.5, 0.3)
    grime = ramp(nt, big, [(0.3, 0.62), (0.72, 1.0)])
    chips_n = noise(nt, v, 0.75, 14.0, 0.72)
    chips = ramp(nt, chips_n, [(0.62 - chip_amount, 0.0), (0.64 - chip_amount, 1.0)])
    col = base
    if green:
        gn = noise(nt, v, 0.11, 5.0, 0.6, 0.6)
        gmask = ramp(nt, gn, [(0.585, 0.0), (0.61, 1.0)])
        speck = ramp(nt, noise(nt, v, 3.0, 4.0), [(0.4, 0.75), (0.7, 1.0)])
        col = mix(nt, gmask, col, mix(nt, 1.0, (0.33, 0.50, 0.33), speck, "MULTIPLY"))
    col = mix(nt, chips, col, chip_color)
    col = mix(nt, 1.0, col, grime, "MULTIPLY")
    height = math_node(nt, "MULTIPLY", ramp(nt, chips_n, [(0.6 - chip_amount, 1.0), (0.66 - chip_amount, 0.0)]), 1.0)
    if cracks:
        vor = nt.nodes.new("ShaderNodeTexVoronoi")
        vor.feature = "DISTANCE_TO_EDGE"
        vor.inputs["Scale"].default_value = 0.11
        warp = nt.nodes.new("ShaderNodeVectorMath")
        warp.operation = "ADD"
        nt.links.new(v, warp.inputs[0])
        wn = nt.nodes.new("ShaderNodeTexNoise")
        wn.inputs["Scale"].default_value = 0.4
        nt.links.new(v, wn.inputs["Vector"])
        nt.links.new(wn.outputs["Color"], warp.inputs[1])
        nt.links.new(warp.outputs[0], vor.inputs["Vector"])
        line = ramp(nt, vor.outputs["Distance"], [(0.0, 0.0), (0.016, 1.0)])
        fade = ramp(nt, noise(nt, v, 0.25, 3.0), [(0.33, 1.0), (0.37, 0.0)])
        crack = mix(nt, fade, line, 1.0, "LIGHTEN")
        col = mix(nt, 1.0, col, mix(nt, 1.0, crack, (0.5, 0.5, 0.5), "LIGHTEN"), "MULTIPLY")
        height = math_node(nt, "MULTIPLY", height, ramp(nt, crack, [(0.0, 0.6), (1.0, 1.0)]))
    col = mix(nt, 1.0, col, ao(nt, 1.2), "MULTIPLY")
    nt.links.new(col, bsdf.inputs["Base Color"])
    r = ramp(nt, chips, [(0.0, rough), (1.0, 0.35)])
    nt.links.new(r, bsdf.inputs["Roughness"])
    met = ramp(nt, chips, [(0.0, metallic), (1.0, 0.85)])
    nt.links.new(met, bsdf.inputs["Metallic"])
    nt.links.new(bump(nt, height, 0.45, 0.15), bsdf.inputs["Normal"])


def make_materials(eye_spaces):
    _m, nt, b = node_mat("paint")
    worn(nt, b, (0.78, 0.76, 0.70), (0.30, 0.30, 0.31), 0.02, True, True, 0.55)

    _m, nt, b = node_mat("panel")  # limb plates: same paint, more worn, no green
    worn(nt, b, (0.76, 0.75, 0.70), (0.25, 0.25, 0.26), 0.05, False, False, 0.5)

    _m, nt, b = node_mat("metal")
    v = coords(nt)
    g = ramp(nt, noise(nt, v, 0.6, 8.0, 0.6), [(0.3, 0.55), (0.75, 1.0)])
    col = mix(nt, 1.0, (0.46, 0.46, 0.48), g, "MULTIPLY")
    col = mix(nt, 1.0, col, ao(nt, 0.8), "MULTIPLY")
    nt.links.new(col, b.inputs["Base Color"])
    b.inputs["Metallic"].default_value = 0.9
    nt.links.new(ramp(nt, noise(nt, v, 1.5, 6.0), [(0.3, 0.28), (0.7, 0.5)]), b.inputs["Roughness"])
    nt.links.new(bump(nt, noise(nt, v, 4.0, 8.0), 0.12, 0.05), b.inputs["Normal"])

    _m, nt, b = node_mat("darkmetal")
    v = coords(nt)
    g = ramp(nt, noise(nt, v, 0.8, 6.0), [(0.3, 0.6), (0.75, 1.0)])
    col = mix(nt, 1.0, (0.19, 0.19, 0.2), g, "MULTIPLY")
    col = mix(nt, 1.0, col, ao(nt, 1.5), "MULTIPLY")
    nt.links.new(col, b.inputs["Base Color"])
    b.inputs["Metallic"].default_value = 0.7
    b.inputs["Roughness"].default_value = 0.45

    _m, nt, b = node_mat("rubber")
    v = coords(nt)
    scuff = ramp(nt, noise(nt, v, 0.9, 10.0, 0.7), [(0.6, 0.0), (0.66, 1.0)])
    col = mix(nt, scuff, (0.045, 0.045, 0.05), (0.18, 0.18, 0.19))
    col = mix(nt, 1.0, col, ao(nt, 1.0), "MULTIPLY")
    nt.links.new(col, b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.55
    b.inputs["Metallic"].default_value = 0.3

    _m, nt, b = node_mat("hazard")
    v = coords(nt)
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(v, sep.inputs[0])
    ang = math_node(nt, "ARCTAN2", sep.outputs[1], sep.outputs[0])
    s = math_node(nt, "ADD", math_node(nt, "MULTIPLY", ang, 11.6), sep.outputs[2])
    stripe = math_node(nt, "GREATER_THAN", math_node(nt, "FRACT", math_node(nt, "DIVIDE", s, 3.2)), 0.5)
    col = mix(nt, stripe, (0.05, 0.05, 0.05), (0.80, 0.60, 0.06))
    chips = ramp(nt, noise(nt, v, 0.9, 12.0, 0.7), [(0.6, 0.0), (0.63, 1.0)])
    col = mix(nt, chips, col, (0.3, 0.3, 0.31))
    col = mix(nt, 1.0, col, ao(nt, 1.0), "MULTIPLY")
    nt.links.new(col, b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.5

    for name, iris, emit, space in (("eye_yellow", (0.95, 0.78, 0.05), 0.6, eye_spaces[0]),
                                    ("eye_purple", (0.48, 0.22, 0.72), 0.45, eye_spaces[1])):
        _m, nt, b = node_mat(name)
        v = coords(nt, space)
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(v, sep.inputs[0])
        rad = math_node(nt, "SQRT", math_node(nt, "ADD", math_node(nt, "MULTIPLY", sep.outputs[0], sep.outputs[0]),
                                              math_node(nt, "MULTIPLY", sep.outputs[2], sep.outputs[2])))
        if name == "eye_yellow":
            pupil = math_node(nt, "LESS_THAN", rad, 0.55)
            ring = ramp(nt, rad, [(0.5, 0.45), (1.3, 1.0), (1.9, 0.75)])
            col = mix(nt, 1.0, iris, ring, "MULTIPLY")
            col = mix(nt, pupil, col, (0.02, 0.02, 0.01))
            glow = math_node(nt, "SUBTRACT", 1.0, pupil)
        else:
            vor = nt.nodes.new("ShaderNodeTexVoronoi")
            vor.feature = "DISTANCE_TO_EDGE"
            vor.inputs["Scale"].default_value = 1.1
            nt.links.new(v, vor.inputs["Vector"])
            crack = ramp(nt, vor.outputs["Distance"], [(0.0, 0.05), (0.06, 1.0)])
            facet = nt.nodes.new("ShaderNodeTexVoronoi")
            facet.inputs["Scale"].default_value = 1.1
            nt.links.new(v, facet.inputs["Vector"])
            shade = ramp(nt, facet.outputs["Distance"], [(0.0, 1.0), (1.0, 0.55)])
            col = mix(nt, 1.0, iris, shade, "MULTIPLY")
            col = mix(nt, 1.0, col, crack, "MULTIPLY")
            glow = 1.0
            nt.links.new(bump(nt, crack, 0.6, 0.1), b.inputs["Normal"])
        nt.links.new(col, b.inputs["Base Color"])
        b.inputs["Roughness"].default_value = 0.12
        b.inputs["Coat Weight"].default_value = 0.6
        nt.links.new(col, b.inputs["Emission Color"])
        if isinstance(glow, float):
            b.inputs["Emission Strength"].default_value = emit
        else:
            nt.links.new(math_node(nt, "MULTIPLY", glow, emit), b.inputs["Emission Strength"])

    for name, col in (("wire_red", (0.55, 0.06, 0.04)), ("wire_blue", (0.05, 0.18, 0.55)),
                      ("wire_yellow", (0.75, 0.55, 0.05)), ("wire_copper", (0.6, 0.3, 0.12))):
        _m, nt, b = node_mat(name)
        b.inputs["Base Color"].default_value = (*col, 1)
        b.inputs["Roughness"].default_value = 0.4
        if name == "wire_copper":
            b.inputs["Metallic"].default_value = 1.0


# ---------------------------------------------------------------- mesh helpers

def obj_from_bm(name, bm, mat, bone=None):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    SCENE.collection.objects.link(ob)
    ob.data.materials.append(MATS[mat])
    me.shade_smooth()
    me.set_sharp_from_angle(angle=math.radians(38))
    if bone:
        vg = ob.vertex_groups.new(name=bone)
        vg.add(range(len(me.vertices)), 1.0, "REPLACE")
    PARTS.append(ob)
    return ob


def frame(z_dir, x_hint=None):
    """Rotation whose Z axis is z_dir."""
    z = Vector(z_dir).normalized()
    x = Vector(x_hint) if x_hint is not None else (Vector((1, 0, 0)) if abs(z.x) < 0.9 else Vector((0, 1, 0)))
    x = (x - z * x.dot(z)).normalized()
    y = z.cross(x)
    return Matrix((x, y, z)).transposed().to_4x4()


def sphere(name, c, r, mat, bone, seg=14, rings=9, scale=(1, 1, 1)):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=rings, radius=1.0)
    bmesh.ops.transform(bm, matrix=Matrix.Translation(c) @ Matrix.Diagonal((r * scale[0], r * scale[1], r * scale[2], 1)),
                        verts=bm.verts)
    return obj_from_bm(name, bm, mat, bone)


def cyl(name, p1, p2, r1, mat, bone, r2=None, seg=12, x_hint=None):
    p1, p2 = Vector(p1), Vector(p2)
    d = p2 - p1
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=seg, radius1=r1,
                          radius2=r1 if r2 is None else r2, depth=d.length)
    m = Matrix.Translation((p1 + p2) / 2) @ frame(d, x_hint)
    bmesh.ops.transform(bm, matrix=m, verts=bm.verts)
    if min(r1, r2 if r2 else r1) >= 0.7:
        bevel_all(bm, min(r1, r2 if r2 else r1) * 0.18, 2, sharp_only=True)
    return obj_from_bm(name, bm, mat, bone)


def box(name, center, size, mat, bone, z_dir=(0, 0, 1), x_hint=None, bevel=0.25, taper=1.0):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        if v.co.z > 0:
            v.co.x *= taper
            v.co.y *= taper
    bmesh.ops.scale(bm, vec=size, verts=bm.verts)
    if bevel:
        bevel_all(bm, bevel, 2)
    m = Matrix.Translation(center) @ frame(z_dir, x_hint)
    bmesh.ops.transform(bm, matrix=m, verts=bm.verts)
    return obj_from_bm(name, bm, mat, bone)


def box_between(name, p1, p2, w, d, mat, bone, x_hint=None, bevel=0.25, taper=1.0, extend=0.0):
    p1, p2 = Vector(p1), Vector(p2)
    axis = (p2 - p1).normalized()
    p1, p2 = p1 - axis * extend, p2 + axis * extend
    return box(name, (p1 + p2) / 2, (w, d, (p2 - p1).length), mat, bone, p2 - p1, x_hint, bevel, taper)


def bevel_all(bm, offset, segs, sharp_only=False):
    edges = [e for e in bm.edges if not sharp_only or (len(e.link_faces) == 2 and
                                                        e.link_faces[0].normal.dot(e.link_faces[1].normal) < 0.5)]
    if edges:
        bmesh.ops.bevel(bm, geom=edges, offset=offset, segments=segs, profile=0.5, affect="EDGES",
                        clamp_overlap=True)


def torus(name, c, axis, R, r, mat, bone, seg=24, rseg=10, x_hint=None):
    bm = bmesh.new()
    rings = []
    for i in range(seg):
        a = 2 * math.pi * i / seg
        ring = []
        for j in range(rseg):
            b = 2 * math.pi * j / rseg
            rr = R + r * math.cos(b)
            ring.append(bm.verts.new((rr * math.cos(a), rr * math.sin(a), r * math.sin(b))))
        rings.append(ring)
    for i in range(seg):
        for j in range(rseg):
            a, b = rings[i], rings[(i + 1) % seg]
            bm.faces.new((a[j], b[j], b[(j + 1) % rseg], a[(j + 1) % rseg]))
    bmesh.ops.transform(bm, matrix=Matrix.Translation(c) @ frame(axis, x_hint), verts=bm.verts)
    return obj_from_bm(name, bm, mat, bone)


def lathe(name, c, profile, mat, bone, seg=64, sy=1.0):
    """Closed (r, z) profile swept around Z through c; y squashed by sy."""
    bm = bmesh.new()
    rings = []
    for i in range(seg):
        a = 2 * math.pi * i / seg
        rings.append([bm.verts.new((c[0] + r * math.cos(a), c[1] + r * math.sin(a) * sy, c[2] + z)) for r, z in profile])
    n = len(profile)
    for i in range(seg):
        a, b = rings[i], rings[(i + 1) % seg]
        for j in range(n):
            bm.faces.new((a[j], a[(j + 1) % n], b[(j + 1) % n], b[j]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return obj_from_bm(name, bm, mat, bone)


def gear(name, c, axis, r, depth, mat, bone, teeth=12, x_hint=None):
    bm = bmesh.new()
    pts = []
    for i in range(teeth * 4):
        a = 2 * math.pi * i / (teeth * 4)
        rr = r if (i % 4) in (0, 1) else r * 0.82
        pts.append((rr * math.cos(a), rr * math.sin(a)))
    bot = [bm.verts.new((x, y, -depth / 2)) for x, y in pts]
    top = [bm.verts.new((x, y, depth / 2)) for x, y in pts]
    bm.faces.new(bot[::-1])
    bm.faces.new(top)
    for i in range(len(pts)):
        j = (i + 1) % len(pts)
        bm.faces.new((bot[i], bot[j], top[j], top[i]))
    bmesh.ops.transform(bm, matrix=Matrix.Translation(c) @ frame(axis, x_hint), verts=bm.verts)
    ob = obj_from_bm(name, bm, mat, bone)
    cyl(name + "_hub", Vector(c) - Vector(axis).normalized() * (depth * 0.8), Vector(c) + Vector(axis).normalized() * (depth * 0.8),
        r * 0.3, "metal", bone, seg=12)
    return ob


def apply_modifiers(ob):
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.convert(target="MESH")


def weight_by_height(ob, stops):
    """stops: [(bone, z), ...] ascending; each vertex blends the two bones around its height."""
    vgs = {b: ob.vertex_groups.get(b) or ob.vertex_groups.new(name=b) for b, _ in stops}
    for v in ob.data.vertices:
        z = (ob.matrix_world @ v.co).z
        if z <= stops[0][1]:
            vgs[stops[0][0]].add([v.index], 1.0, "REPLACE")
            continue
        if z >= stops[-1][1]:
            vgs[stops[-1][0]].add([v.index], 1.0, "REPLACE")
            continue
        for (b0, z0), (b1, z1) in zip(stops, stops[1:]):
            if z0 <= z <= z1:
                t = (z - z0) / (z1 - z0)
                t = t * t * (3 - 2 * t)
                vgs[b0].add([v.index], 1 - t, "REPLACE")
                vgs[b1].add([v.index], t, "REPLACE")
                break


# ---------------------------------------------------------------- the goblin

HC = Vector((0, -1.0, 74.5))
HR = Vector((11.5, 10.5, 12.0))


def g2(du, dv, su, sv):
    return math.exp(-(du * du) / (su * su) - (dv * dv) / (sv * sv))


def sstep(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


def face_disp(u, v):
    au = abs(u)
    d = 0.0
    vb = 0.25 + 0.16 * au                                   # grumpy brow: inner ends low
    d += 2.4 * math.exp(-((v - vb) / 0.08) ** 2) * sstep(0.74, 0.5, au)
    d -= 0.45 * math.exp(-((v - (vb + 0.12)) / 0.028) ** 2) * sstep(0.6, 0.35, au)
    d -= 0.3 * math.exp(-((v - (vb + 0.25)) / 0.025) ** 2) * sstep(0.5, 0.25, au)
    d += 0.6 * g2(u, v - 0.5, 0.5, 0.2)
    d -= 0.4 * g2(au - 0.05, v - 0.2, 0.018, 0.06)          # frown furrows
    d -= 2.6 * g2(au - 0.34, v - 0.09, 0.19, 0.13)         # eye sockets
    d += 0.55 * g2(au - 0.34, v + 0.06, 0.14, 0.035)        # bags
    d += 0.35 * g2(au - 0.36, v + 0.12, 0.14, 0.03)
    d += 2.3 * g2(u, v - 0.03, 0.07, 0.16)                 # nose bridge
    d += 4.6 * g2(u, v + 0.15, 0.17, 0.12)                  # nose bulb
    d += 1.9 * g2(au - 0.15, v + 0.19, 0.08, 0.065)          # nose wings
    d -= 0.9 * g2(au - 0.075, v + 0.235, 0.035, 0.022)      # nostrils
    d += 2.0 * g2(au - 0.5, v + 0.24, 0.2, 0.2)             # cheeks
    fold = seg_dist(au, v, 0.2, -0.17, 0.37, -0.6)
    d -= 0.6 * math.exp(-(fold / 0.03) ** 2)
    vm = -0.5 - 0.55 * u * u                                # frown
    wm = sstep(0.34, 0.24, au)
    d -= 1.5 * math.exp(-((v - vm) / 0.028) ** 2) * wm
    d += 0.55 * math.exp(-((v - (vm + 0.06)) / 0.05) ** 2) * wm
    d += 1.3 * math.exp(-((v - (vm - 0.075)) / 0.05) ** 2) * sstep(0.3, 0.18, au)
    d += 1.5 * g2(u, v + 0.8, 0.22, 0.12)                   # chin
    d -= 0.35 * g2(u, v + 0.68, 0.15, 0.02)
    d += 1.4 * g2(au - 0.4, v + 0.72, 0.13, 0.16)           # jowls
    return d


def build_head():
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=200, v_segments=150, radius=1.0)
    for vert in bm.verts:
        n = vert.co.copy()
        t = n.z
        wx = 1 - 0.15 * t + 0.10 * math.exp(-((t + 0.5) / 0.35) ** 2)
        wy = 1 - 0.08 * t
        x, y, z = n.x * HR.x * wx, n.y * HR.y * wy, n.z * HR.z * (1 + 0.07 * max(t, 0))
        if n.y < 0:
            y *= 1 - 0.12 * (-n.y) ** 2                     # flatter face
        nrm = Vector((x / HR.x ** 2, y / HR.y ** 2, z / HR.z ** 2)).normalized()
        front = max(0.0, min(1.0, -n.y * 1.25)) ** 0.8
        d = face_disp(x / HR.x, z / HR.z) * front if front > 0 else 0.0
        vert.co = HC + Vector((x, y, z)) + nrm * d
    ob = obj_from_bm("head", bm, "paint", "head")
    mod = ob.modifiers.new("dec", "DECIMATE")
    mod.ratio = 0.22
    apply_modifiers(ob)
    return ob


def head_surface(x, z, ob):
    """Front surface point of the head at (x, z)."""
    from mathutils.bvhtree import BVHTree
    deps = bpy.context.evaluated_depsgraph_get()
    tree = BVHTree.FromObject(ob, deps)
    hit = tree.ray_cast(Vector((x, -40, z)), Vector((0, 1, 0)))
    return hit[0], hit[1]


def place_eyes(eyes):
    for c, side, mat in eyes:
        if mat == "eye_yellow":
            sphere("eye_y", c, 2.5, mat, "head", 32, 24)
            # droopy upper lid: shell over the top, tilted forward
            bm = bmesh.new()
            bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=24, radius=2.75)
            bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z < 1.0], context="VERTS")
            bmesh.ops.solidify(bm, geom=bm.faces[:], thickness=0.25)
            rot = Matrix.Rotation(math.radians(12), 4, "X") @ Matrix.Rotation(math.radians(-10 * side), 4, "Y")
            bmesh.ops.transform(bm, matrix=Matrix.Translation(c) @ rot, verts=bm.verts)
            obj_from_bm("eyelid", bm, "paint", "head")
            torus("eye_rim_y", c + Vector((0, -0.6, 0)), (0, 1, 0), 2.7, 0.32, "metal", "head", 32, 8)
        else:
            sphere("eye_p", c + Vector((0, -0.2, 0)), 2.9, mat, "head", 32, 24, (1.15, 0.8, 1.0))
            torus("eye_rim_p", c + Vector((0, -0.6, 0)), (0, 1, 0), 3.35, 0.38, "darkmetal", "head", 32, 8)
            for k in range(6):
                a = 2 * math.pi * k / 6 + 0.3
                sphere("rim_bolt", c + Vector((3.85 * math.cos(a), -0.7, 3.6 * math.sin(a))), 0.28, "metal", "head", 8, 5)


def build_ear(side):
    s = side
    p0 = HC + Vector((s * 9.6, 1.0, 4.0))
    n0 = Vector((0.38 * s, -0.92, 0.0)).normalized()
    e1 = Vector((0.52 * s, 0.12, 0.85))
    e1 = (e1 - n0 * e1.dot(n0)).normalized()
    e2 = n0.cross(e1) * s
    outline = [(-1.0, -3.6), (2.5, -4.7), (7.0, -4.0), (11.5, -2.3), (15.5, -0.6), (17.0, 0.0), (15.0, 0.9),
               (11.0, 1.9), (6.5, 3.0), (2.5, 4.2), (-1.0, 4.0)]

    def leaf(name, scale, shift, thick, offset, mat):
        bm = bmesh.new()
        front, back = [], []
        for a, b in outline:
            p = p0 + e1 * (a * scale + shift) + e2 * (b * scale)
            front.append(bm.verts.new(p + n0 * (offset + thick / 2)))
            back.append(bm.verts.new(p + n0 * (offset - thick / 2)))
        bm.faces.new(front)
        bm.faces.new(back[::-1])
        for i in range(len(outline)):
            j = (i + 1) % len(outline)
            bm.faces.new((front[i], back[i], back[j], front[j]))
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bevel_all(bm, min(thick * 0.4, 0.45), 3)
        return obj_from_bm(name, bm, mat, "head")

    leaf("ear", 1.0, 0.0, 1.5, 0.0, "paint")
    leaf("ear_inner", 0.62, 2.6, 0.4, 0.75, "darkmetal")
    discs = []
    for a, b, r in ((3.0, -0.6, 1.25), (6.2, 0.4, 0.95), (9.0, -0.2, 0.75)):
        c = p0 + e1 * a + e2 * b + n0 * 1.05
        cyl("ear_disc", c - n0 * 0.35, c + n0 * 0.35, r, "metal", "head", seg=20)
        cyl("ear_pin", c, c + n0 * 0.6, r * 0.35, "darkmetal", "head", seg=12)
        discs.append(c)
    cyl("ear_rod", discs[0] + n0 * 0.5, discs[2] + n0 * 0.5, 0.28, "metal", "head", seg=10)
    cyl("ear_rod2", discs[0] + n0 * 0.2 - e2 * 1.0, discs[1] + n0 * 0.2 - e2 * 0.8, 0.22, "metal", "head", seg=10)
    tip = p0 + e1 * 15.6
    cyl("ear_spike", tip, tip + e1 * 4.2, 0.85, "metal", "head", r2=0.04, seg=16)
    cyl("ear_hinge", p0 - e2 * 3.2 + e1 * 0.5, p0 + e2 * 3.2 + e1 * 0.5, 1.1, "metal", "head", seg=16)


def build_collar():
    c = Vector((0, 0.5, 60.6))
    prof = [(8.0, -2.8), (11.0, -2.8), (11.6, -2.2), (11.6, 2.4), (11.0, 3.0), (8.0, 3.0)]
    lathe("collar", c, prof, "rubber", "neck1", 72, 0.94)
    lathe("hazard", c, [(11.45, -2.45), (11.82, -2.25), (11.82, -0.25), (11.45, -0.05)], "hazard", "neck1", 96, 0.94)
    for k in range(14):
        a = 2 * math.pi * k / 14
        dirv = Vector((math.cos(a), math.sin(a) * 0.94, 0))
        p = c + Vector((11.55 * math.cos(a), 11.55 * math.sin(a) * 0.94, 1.5))
        cyl("collar_bolt", p - dirv * 0.1, p + dirv * 0.45, 0.42, "metal", "neck1", seg=8)
    # latch and chain on the left (+X)
    box("latch", c + Vector((12.4, 0.0, 0.2)), (1.8, 3.6, 4.6), "darkmetal", "neck1", bevel=0.3)
    gear("collar_gear", c + Vector((10.1, -6.2, 0.4)), (0.6, -0.8, 0), 2.0, 0.9, "darkmetal", "neck1", 10)
    bar0, bar1 = c + Vector((13.0, 0.0, 1.0)), c + Vector((16.5, 0.0, 1.0))
    cyl("latch_bar", bar0, bar1, 0.55, "metal", "neck1", seg=12)
    box("latch_tab", bar1 + Vector((0.4, 0, 0)), (0.9, 1.6, 2.4), "metal", "neck1", bevel=0.2)
    cyl("hook", bar1 + Vector((0.4, 0, -1.0)), bar1 + Vector((0.4, 0, -2.6)), 0.3, "metal", "neck1", seg=10)
    p = bar1 + Vector((0.4, 0, -3.5))
    for k in range(3):
        axis = (1, 0, 0) if k % 2 == 0 else (0, 1, 0)
        torus("chain", p, axis, 0.85, 0.24, "metal", "neck1", 16, 8, x_hint=(0, 0, 1))
        p = p + Vector((0, 0, -1.25))


def build_torso():
    c = Vector((0, 0.7, 45.5))
    r = Vector((10.0, 8.6, 13.0))
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=64, v_segments=40, radius=1.0)
    for vert in bm.verts:
        n = vert.co.copy()
        t = n.z
        wx = 1 - 0.07 * t
        vert.co = c + Vector((n.x * r.x * wx, n.y * r.y * wx, n.z * r.z))
    torso = obj_from_bm("torso", bm, "paint")
    cut = box("cavity_cut", c + Vector((0, -8.6, 2.0)), (13.0, 10.0, 12.5), "darkmetal", None, bevel=1.6)
    vent = cyl("vent_cut", c + Vector((-1.8, -11.5, -8.6)), c + Vector((-1.8, -4.5, -8.6)), 2.1, "darkmetal", None, seg=24)
    PARTS.remove(cut)
    PARTS.remove(vent)
    for cutter in (cut, vent):
        mod = torso.modifiers.new("cut", "BOOLEAN")
        mod.operation = "DIFFERENCE"
        mod.solver = "EXACT"
        mod.material_mode = "TRANSFER"
        mod.object = cutter
    apply_modifiers(torso)
    for cutter in (cut, vent):
        bpy.data.objects.remove(cutter)
    stops = [("root", 43.0), ("spine1", 50.0), ("spine2", 54.5), ("spine3", 58.5)]
    weight_by_height(torso, stops)

    # works inside the chest
    back = c.y - 1.6
    for x, rr in ((-4.6, 0.85), (-1.6, 1.0), (1.6, 1.0), (4.6, 0.85)):
        bot, top = Vector((x, back - 1.2, 41.5)), Vector((x, back - 1.2, 53.5))
        cyl("piston", bot, top, rr, "metal", "spine1", seg=16)
        cyl("piston_sleeve", bot + Vector((0, 0, 1.5)), bot + Vector((0, 0, 6.0)), rr * 1.35, "darkmetal", "spine1", seg=16)
        cyl("piston_cap", top - Vector((0, 0, 1.4)), top, rr * 1.5, "metal", "spine1", seg=16)
    box("chest_block", Vector((0, back - 2.4, 46.0)), (3.4, 2.4, 3.2), "metal", "spine1", bevel=0.3)
    cyl("chest_drum", Vector((0, back - 3.8, 46.0)), Vector((0, back - 2.0, 46.0)), 1.1, "darkmetal", "spine1", seg=16)
    box("chest_valve", Vector((2.9, back - 2.6, 49.5)), (1.6, 1.6, 2.2), "metal", "spine1", bevel=0.2)
    for i, (mat, pts) in enumerate((
            ("wire_red", [(-5.5, -0.5, 53), (-3.0, -3.5, 49), (-6.0, -3.0, 44), (-3.5, -1.0, 41.5)]),
            ("wire_blue", [(5.5, -0.5, 53), (3.2, -3.8, 48), (5.8, -3.2, 44.5), (3.0, -1.0, 41.5)]),
            ("wire_yellow", [(3.0, -1.0, 53.5), (4.8, -4.2, 51), (2.5, -4.0, 47), (4.0, -1.0, 42)]),
            ("wire_copper", [(-2.8, -1.0, 53.5), (-4.6, -4.0, 50), (-2.6, -4.2, 46.5), (-5.0, -1.0, 42.5)]))):
        cu = bpy.data.curves.new(f"wire{i}", "CURVE")
        cu.dimensions = "3D"
        cu.bevel_depth = 0.28
        cu.bevel_resolution = 3
        sp = cu.splines.new("BEZIER")
        sp.bezier_points.add(len(pts) - 1)
        for bp, p in zip(sp.bezier_points, pts):
            bp.co = Vector((p[0], back + p[1], p[2]))
            bp.handle_left_type = bp.handle_right_type = "AUTO"
        ob = bpy.data.objects.new(f"wire{i}", cu)
        SCENE.collection.objects.link(ob)
        ob.data.materials.append(MATS[mat])
        apply_modifiers(ob)
        vg = ob.vertex_groups.new(name="spine1")
        vg.add(range(len(ob.data.vertices)), 1.0, "REPLACE")
        PARTS.append(ob)
    # lip over the chest opening and side gear
    gear("side_gear", Vector((9.7, 1.4, 46.0)), (1, 0, 0), 2.6, 1.0, "darkmetal", "spine1", 12)
    gear("side_gear_r", Vector((-9.7, 1.4, 41.5)), (-1, 0, 0), 1.9, 0.9, "darkmetal", "root", 10)
    for k, z in enumerate((40.0, 40.0, 54.8, 54.8)):
        x = (-7.2, 7.2, -7.2, 7.2)[k]
        sphere("rivet", Vector((x, -7.0 + abs(z - 47) * 0.12, z)), 0.38, "metal", "spine1" if z > 47 else "root", 8, 5)


def build_arm(s):
    S = "L" if s > 0 else "R"
    sh, el, wr = J[f"bicep_{S}"], J[f"elbow_{S}"], J[f"wrist_{S}"]
    up, fo = (el - sh).normalized(), (wr - el).normalized()
    lateral = up.cross(fo).normalized()
    sphere("shoulder", sh, 2.2, "metal", f"bicep_{S}", 20, 14)
    cyl("upper_arm", sh, el, 1.0, "metal", f"bicep_{S}", seg=16)
    off = lateral * 1.25
    cyl("arm_piston", sh + up * 2.0 + off, el - up * 1.5 + off, 0.5, "metal", f"bicep_{S}", seg=12)
    cyl("arm_sleeve", sh + up * 1.5 + off, sh + up * 5.5 + off, 0.72, "darkmetal", f"bicep_{S}", seg=12)
    sphere("elbow", el, 1.6, "metal", f"elbow_{S}", 20, 14)
    gear("elbow_gear", el + lateral * 1.2 * s, lateral, 1.8, 0.6, "darkmetal", f"elbow_{S}", 10)
    box_between("forearm", el + fo * 1.6, wr - fo * 0.6, 2.4, 2.3, "panel", f"elbow_{S}", x_hint=lateral, bevel=0.45,
                taper=0.8)
    cyl("forearm_piston", el + fo * 1.0 - lateral * 1.5, wr - fo * 1.2 - lateral * 1.2, 0.38, "metal", f"elbow_{S}", seg=10)
    sphere("wrist", wr, 0.95, "metal", f"wrist_{S}", 16, 10)
    knuck = (J[f"index_0_{S}"] + J[f"mid_0_{S}"] + J[f"ring_0_{S}"]) / 3
    box_between("palm", wr + (knuck - wr) * 0.25, knuck, 2.2, 1.0, "darkmetal", f"wrist_{S}",
                x_hint=J[f"index_0_{S}"] - J[f"ring_0_{S}"], bevel=0.25)
    for f, r in (("index", 0.36), ("mid", 0.38), ("ring", 0.34), ("thumb", 0.36)):
        j0, j1, j2 = J[f"{f}_0_{S}"], J[f"{f}_1_{S}"], J[f"{f}_2_{S}"]
        tip = j2 + (j2 - j1).normalized() * 1.3
        for a, b, bone in ((j0, j1, f"{f}_0_{S}"), (j1, j2, f"{f}_1_{S}")):
            sphere("knuckle", a, r * 1.3, "metal", bone, 8, 6)
            cyl("finger", a, b, r, "metal", bone, seg=8)
        sphere("knuckle", j2, r * 1.2, "metal", f"{f}_2_{S}", 8, 6)
        cyl("claw", j2, tip, r, "darkmetal", f"{f}_2_{S}", r2=0.05, seg=8)


def build_leg(s):
    S = "L" if s > 0 else "R"
    hip, kn, an, toe = J[f"thigh_{S}"], J[f"knee_{S}"], J[f"ankle_{S}"], J[f"toeBase_{S}"]
    th, sh = (kn - hip).normalized(), (an - kn).normalized()
    side = Vector((1, 0, 0))
    sphere("hip", hip, 2.4, "metal", f"thigh_{S}", 20, 14)
    cyl("thigh", hip, kn, 1.35, "metal", f"thigh_{S}", seg=16)
    cyl("thigh_piston", hip + th * 3 + Vector((0, -1.6, 0)), kn - th * 2 + Vector((0, -1.5, 0)), 0.65, "metal",
        f"thigh_{S}", seg=12)
    cyl("thigh_sleeve", kn - th * 7 + Vector((0, -1.5, 0)), kn - th * 2.5 + Vector((0, -1.5, 0)), 0.9, "darkmetal",
        f"thigh_{S}", seg=12)
    sphere("knee", kn, 2.0, "metal", f"knee_{S}", 20, 14)
    gear("knee_gear", kn + side * s * 1.4, side * s, 2.1, 0.7, "darkmetal", f"knee_{S}", 10)
    box_between("shin_plate", kn + sh * 2.0 + Vector((0, -1.0, 0)), an - sh * 1.2 + Vector((0, -0.8, 0)), 2.7, 1.5,
                "panel", f"knee_{S}", x_hint=side, bevel=0.45, taper=0.75)
    cyl("shin", kn, an, 0.85, "metal", f"knee_{S}", seg=14)
    cyl("shin_piston", kn + sh * 2 + Vector((0, 1.4, 0)), an - sh * 1.8 + Vector((0, 1.3, 0)), 0.5, "metal",
        f"knee_{S}", seg=10)
    sphere("ankle", an, 1.45, "metal", f"ankle_{S}", 18, 12)
    box_between("foot_strut", an, toe, 2.3, 1.4, "darkmetal", f"ankle_{S}", x_hint=side, bevel=0.3)
    sphere("toe_joint", toe, 1.1, "metal", f"toeBase_{S}", 16, 10)
    for ang in (-30, 0, 30):
        d = Matrix.Rotation(math.radians(ang), 3, "Z") @ Vector((0, -1, 0))
        k = toe + d * 2.6 + Vector((0, 0, -0.9))
        cyl("toe", toe, k, 0.55, "metal", f"toeBase_{S}", seg=10)
        sphere("toe_knuckle", k, 0.6, "metal", f"toeBase_{S}", 10, 8)
        cyl("toe_claw", k, k + d * 1.9 + Vector((0, 0, -0.55)), 0.55, "darkmetal", f"toeBase_{S}", r2=0.06, seg=10)
    heel = Vector((an.x, an.y + 3.4, 0.7))
    cyl("heel", an, heel, 0.7, "metal", f"ankle_{S}", seg=10)
    cyl("heel_claw", heel, heel + Vector((0, 1.4, -0.5)), 0.6, "darkmetal", f"ankle_{S}", r2=0.06, seg=10)


def build():
    # eye spaces must exist before the eye materials are made
    head_tmp_spaces = []
    for side in (-1, 1):
        e = bpy.data.objects.new(f"eye_space_{side}", None)
        SCENE.collection.objects.link(e)
        head_tmp_spaces.append(e)
    make_materials(head_tmp_spaces)
    head = build_head()
    eyes = []
    for e, side, mat in zip(head_tmp_spaces, (-1, 1), ("eye_yellow", "eye_purple")):
        p, _n = head_surface(0.34 * side * HR.x, HC.z + 0.09 * HR.z, head)
        c = p + Vector((0, 0.9, 0))
        e.location = c
        eyes.append((c, side, mat))
    place_eyes(eyes)
    for s in (-1, 1):
        build_ear(s)
    build_collar()
    build_torso()
    for s in (-1, 1):
        build_arm(s)
        build_leg(s)


# ---------------------------------------------------------------- render, bake, export

def setup_render():
    SCENE.render.engine = "CYCLES"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for kind in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = kind
                prefs.get_devices()
                if any(d.type == kind for d in prefs.devices):
                    for d in prefs.devices:
                        d.use = d.type == kind
                    SCENE.cycles.device = "GPU"
                    print("render device", kind)
                    break
            except TypeError:
                continue
    except Exception as e:  # noqa: BLE001
        print("GPU setup failed, CPU:", e)
    SCENE.cycles.samples = 64
    SCENE.cycles.use_denoising = True
    SCENE.render.resolution_x, SCENE.render.resolution_y = 960, 1280
    SCENE.view_settings.view_transform = "AgX"
    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.95, 0.95, 0.95, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.9
    SCENE.world = world
    for name, loc, energy, size in (("key", (-60, -90, 140), 90000, 60), ("fill", (90, -60, 60), 30000, 80),
                                    ("rim", (20, 90, 110), 40000, 50)):
        ld = bpy.data.lights.new(name, "AREA")
        ld.energy = energy
        ld.size = size
        lo = bpy.data.objects.new(name, ld)
        SCENE.collection.objects.link(lo)
        lo.location = loc
        lo.rotation_euler = (Vector((0, 0, 45)) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    floor = bpy.data.meshes.new("floor")
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=400)
    bm.to_mesh(floor)
    fo = bpy.data.objects.new("floor", floor)
    SCENE.collection.objects.link(fo)
    fm, nt, b = node_mat("floor")
    b.inputs["Base Color"].default_value = (0.9, 0.9, 0.9, 1)
    b.inputs["Roughness"].default_value = 0.8
    floor.materials.append(fm)
    fo.is_shadow_catcher = True


def render(name, cam_loc, target, lens=70, res=(960, 1280)):
    cd = bpy.data.cameras.get("cam") or bpy.data.cameras.new("cam")
    cd.lens = lens
    co = bpy.data.objects.get("cam") or bpy.data.objects.new("cam", cd)
    if co.name not in SCENE.collection.objects:
        SCENE.collection.objects.link(co)
    co.location = cam_loc
    co.rotation_euler = (Vector(target) - Vector(cam_loc)).to_track_quat("-Z", "Y").to_euler()
    SCENE.camera = co
    SCENE.render.resolution_x, SCENE.render.resolution_y = res
    SCENE.render.film_transparent = False
    SCENE.render.filepath = os.path.join(OUT, f"preview-{name}.png")
    bpy.ops.render.render(write_still=True)


def render_views(tag):
    render(f"{tag}-front34", (55, -150, 62), (0, 0, 45), 55)
    render(f"{tag}-face", (14, -60, 76), (0, 0, 73), 70, (960, 960))
    render(f"{tag}-back34", (-80, 120, 70), (0, 0, 45), 55)


def join_parts():
    bpy.ops.object.select_all(action="DESELECT")
    for ob in PARTS:
        ob.select_set(True)
    bpy.context.view_layer.objects.active = PARTS[0]
    bpy.ops.object.join()
    ob = bpy.context.view_layer.objects.active
    ob.name = "goblin"
    ob.data.shade_smooth()
    ob.data.set_sharp_from_angle(angle=math.radians(38))
    return ob


def bake(ob, size=2048):
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(60), island_margin=0.003, area_weight=0.0)
    bpy.ops.object.mode_set(mode="OBJECT")
    images = {}
    for kind, cs in (("color", "sRGB"), ("emit", "sRGB"), ("normal", "Non-Color")):
        img = bpy.data.images.new(f"goblin_{kind}", size, size, alpha=False)
        img.colorspace_settings.name = cs
        images[kind] = img
    SCENE.cycles.samples = 24
    SCENE.render.bake.margin = 12

    def target(img):
        for slot in ob.material_slots:
            nt = slot.material.node_tree
            node = nt.nodes.get("bake_target") or nt.nodes.new("ShaderNodeTexImage")
            node.name = "bake_target"
            node.image = img
            nt.nodes.active = node

    # Cycles leaves metals' base colour out of the diffuse pass: bake everything as dielectric
    for slot in ob.material_slots:
        for node in slot.material.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                for link in list(node.inputs["Metallic"].links):
                    slot.material.node_tree.links.remove(link)
                node.inputs["Metallic"].default_value = 0.0
    target(images["color"])
    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"}, use_clear=True, margin=12)
    target(images["emit"])
    bpy.ops.object.bake(type="EMIT", use_clear=True, margin=12)
    target(images["normal"])
    SCENE.render.bake.normal_g = "NEG_Y"
    bpy.ops.object.bake(type="NORMAL", normal_space="TANGENT", use_clear=True, margin=12)
    for kind, img in images.items():
        img.filepath_raw = os.path.join(OUT, f"goblin_{kind}.png")
        img.file_format = "PNG"
        img.save()
    # the baked look: what the game will show (roughly)
    m, nt, b = node_mat("goblin")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = images["color"]
    nt.links.new(tex.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.45
    b.inputs["Metallic"].default_value = 0.2
    et = nt.nodes.new("ShaderNodeTexImage")
    et.image = images["emit"]
    nt.links.new(et.outputs["Color"], b.inputs["Emission Color"])
    b.inputs["Emission Strength"].default_value = 1.0
    ob.data.materials.clear()
    ob.data.materials.append(m)


def write_smd(ob, path):
    me = ob.data
    me.calc_loop_triangles()
    uv = me.uv_layers.active.data
    normals = me.corner_normals
    groups = {g.index: g.name for g in ob.vertex_groups}
    index = {b["name"]: i for i, b in enumerate(SKEL["bones"])}
    lines = ["version 1", "nodes"]
    for i, b in enumerate(SKEL["bones"]):
        lines.append(f'{i} "{b["name"]}" {b["parent"]}')
    lines += ["end", "skeleton", "time 0"]
    for i, b in enumerate(SKEL["bones"]):
        pos, rot = Vector(b["pos"]), b["rot"]
        lines.append(f"{i} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f} {rot[0]:.6f} {rot[1]:.6f} {rot[2]:.6f}")
    lines += ["end", "triangles"]
    mats = [s.material.name for s in ob.material_slots]
    for tri in me.loop_triangles:
        lines.append(mats[tri.material_index] if mats else "goblin")
        for li in tri.loops:
            v = me.vertices[me.loops[li].vertex_index]
            ws = sorted(((g.weight, groups[g.group]) for g in v.groups if g.weight > 0.01), reverse=True)[:3]
            tot = sum(w for w, _ in ws) or 1.0
            p = M_FROM_B @ v.co
            n = (M_FROM_B @ normals[li].vector).normalized()
            u = uv[li].uv
            links = " ".join(f"{index[name]} {w / tot:.6f}" for w, name in ws)
            lines.append(f"{index[ws[0][1]]} {p.x:.6f} {p.y:.6f} {p.z:.6f} {n.x:.6f} {n.y:.6f} {n.z:.6f} "
                         f"{u.x:.6f} {u.y:.6f} {len(ws)} {links}")
    lines.append("end")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}: {len(me.loop_triangles)} triangles, {len(me.vertices)} vertices")


def main():
    build()
    counts = sorted(((len(o.data.vertices), o.name) for o in PARTS), reverse=True)
    print("vertices", sum(c for c, _ in counts), "biggest parts", counts[:8])
    for ob in PARTS:
        if not ob.vertex_groups:
            raise SystemExit(f"part {ob.name} has no bone")
    setup_render()
    render_views("proc")
    if NO_BAKE:
        return
    gob = join_parts()
    bake(gob)
    render_views("baked")
    write_smd(gob, os.path.join(OUT, "goblin.smd"))
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "goblin.blend"))


main()
