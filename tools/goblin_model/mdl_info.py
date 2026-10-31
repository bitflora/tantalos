"""Dump what a replacement for a Portal 2 (MDL v49) model has to match: bones, attachments,
hitboxes, included models, pose parameters, IK chains, materials, sequences.

    python tools/goblin_model/mdl_info.py MODEL.mdl            (human-readable)
    python tools/goblin_model/mdl_info.py MODEL.mdl --json OUT  (bones with world matrices etc.)
"""
import json
import math
import struct
import sys

BONE_SIZE = 216


def cstr(data, pos):
    return data[pos:data.index(b"\0", pos)].decode("latin-1")


def euler_matrix(x, y, z):
    """RadianEuler -> 3x3 rotation, Source's AngleMatrix(RadianEuler) convention."""
    sx, cx = math.sin(x), math.cos(x)
    sy, cy = math.sin(y), math.cos(y)
    sz, cz = math.sin(z), math.cos(z)
    return [
        [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
        [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
        [-sy, sx * cy, cx * cy],
    ]


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def matvec(a, v):
    return [sum(a[i][k] * v[k] for k in range(3)) for i in range(3)]


def parse(path):
    with open(path, "rb") as f:
        d = f.read()
    i = lambda off: struct.unpack_from("<i", d, off)[0]
    v3 = lambda off: list(struct.unpack_from("<3f", d, off))
    info = {
        "name": cstr(d, 12), "version": i(4), "checksum": i(8),
        "eyeposition": v3(80), "illumposition": v3(92),
        "hull_min": v3(104), "hull_max": v3(116), "view_bbmin": v3(128), "view_bbmax": v3(140),
        "flags": i(152), "mass": struct.unpack_from("<f", d, 328)[0], "contents": i(332),
        "surfaceprop": cstr(d, i(308)),
    }
    bones = []
    for b in range(i(156)):
        o = i(160) + b * BONE_SIZE
        pos = v3(o + 32)
        quat = list(struct.unpack_from("<4f", d, o + 44))
        rot = v3(o + 60)
        p2b = struct.unpack_from("<12f", d, o + 96)
        flags, proctype, procindex = struct.unpack_from("<iii", d, o + 160)
        contents = i(o + 180)
        surf = cstr(d, o + i(o + 176)) if i(o + 176) else ""
        bones.append({"name": cstr(d, o + i(o)), "parent": i(o + 4), "pos": pos, "quat": quat,
                      "rot": rot, "pose_to_bone": [list(p2b[0:4]), list(p2b[4:8]), list(p2b[8:12])],
                      "flags": flags, "proctype": proctype, "surfaceprop": surf,
                      "contents": contents})
    for b in bones:
        local = euler_matrix(*b["rot"])
        if b["parent"] < 0:
            b["world_rot"], b["world_pos"] = local, b["pos"]
        else:
            p = bones[b["parent"]]
            b["world_rot"] = matmul(p["world_rot"], local)
            b["world_pos"] = [a + c for a, c in zip(matvec(p["world_rot"], b["pos"]), p["world_pos"])]
    info["bones"] = bones

    atts = []
    for a in range(i(240)):
        o = i(244) + a * 92
        m = struct.unpack_from("<12f", d, o + 12)
        atts.append({"name": cstr(d, o + i(o)), "flags": i(o + 4), "bone": i(o + 8),
                     "matrix": [list(m[0:4]), list(m[4:8]), list(m[8:12])]})
    info["attachments"] = atts

    sets = []
    for s in range(i(172)):
        o = i(176) + s * 12
        boxes = []
        for h in range(i(o + 4)):
            ho = o + i(o + 8) + h * 68
            nm = i(ho + 32)
            boxes.append({"bone": i(ho), "group": i(ho + 4), "min": v3(ho + 8), "max": v3(ho + 20),
                          "name": cstr(d, ho + nm) if nm else ""})
        sets.append({"name": cstr(d, o + i(o)), "boxes": boxes})
    info["hitboxsets"] = sets

    info["includemodels"] = []
    for m in range(i(336)):
        o = i(340) + m * 8
        info["includemodels"].append({"label": cstr(d, o + i(o)), "name": cstr(d, o + i(o + 4))})

    info["textures"] = [cstr(d, i(208) + t * 64 + i(i(208) + t * 64)) for t in range(i(204))]
    info["cdtextures"] = [cstr(d, i(i(216) + t * 4)) for t in range(i(212))]

    pps = []
    for p in range(i(300)):
        o = i(304) + p * 20
        flags, start, end, loop = struct.unpack_from("<i3f", d, o + 4)
        pps.append({"name": cstr(d, o + i(o)), "flags": flags, "start": start, "end": end, "loop": loop})
    info["poseparams"] = pps

    chains = []
    for c in range(i(284)):
        o = i(288) + c * 16
        nlinks, linkidx = i(o + 8), i(o + 12)
        links = [i(o + linkidx + k * 28) for k in range(nlinks)]
        chains.append({"name": cstr(d, o + i(o)), "type": i(o + 4), "links": links})
    info["ikchains"] = chains

    seqs = []
    for s in range(i(188)):
        o = i(192) + s * 212
        seqs.append(cstr(d, o + i(o + 4)))
    info["sequences"] = seqs
    info["numlocalanim"] = i(180)

    parts = []
    for p in range(i(232)):
        o = i(236) + p * 16
        models = []
        for m in range(i(o + 4)):
            mo = o + i(o + 12) + m * 148
            models.append({"name": cstr(d, mo), "numvertices": i(mo + 80)})
        parts.append({"name": cstr(d, o + i(o)), "models": models})
    info["bodyparts"] = parts
    info["flexdesc"] = i(260)
    kvi, kvs = i(312), i(316)
    info["keyvalues"] = d[kvi:kvi + kvs].decode("latin-1").rstrip("\0") if kvs else ""
    return info


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    info = parse(argv[1])
    if "--json" in argv:
        with open(argv[argv.index("--json") + 1], "w") as f:
            json.dump(info, f, indent=1)
        return 0
    for k in ("name", "version", "eyeposition", "illumposition", "hull_min", "hull_max", "view_bbmin",
              "view_bbmax", "flags", "mass", "contents", "surfaceprop", "includemodels", "textures",
              "cdtextures", "poseparams", "ikchains", "numlocalanim", "flexdesc", "keyvalues"):
        print(f"{k}: {info[k]}")
    print("bodyparts:", json.dumps(info["bodyparts"]))
    print(f"sequences ({len(info['sequences'])}):", info["sequences"][:40])
    print(f"bones ({len(info['bones'])}):")
    for n, b in enumerate(info["bones"]):
        wp = ", ".join(f"{x:7.2f}" for x in b["world_pos"])
        print(f"  {n:3d} {b['name']:40s} parent={b['parent']:3d} world=({wp}) flags={b['flags']:#x} "
              f"proc={b['proctype']}")
    print("attachments:")
    for a in info["attachments"]:
        print(f"  {a['name']} bone={info['bones'][a['bone']]['name']} flags={a['flags']} "
              f"t=({a['matrix'][0][3]:.2f},{a['matrix'][1][3]:.2f},{a['matrix'][2][3]:.2f})")
    for s in info["hitboxsets"]:
        print(f"hitboxset {s['name']}:")
        for h in s["boxes"]:
            print(f"  {info['bones'][h['bone']]['name']} group={h['group']} {h['min']} {h['max']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
