"""Minimal reader for Portal 2 compiled maps (VBSP version 21): just what puzzle recovery needs.

Lumps used: entities 0, planes 1, texdata 2, nodes 5, texinfo 6, leafs 10, models 14,
leafbrushes 17, brushes 18, brushsides 19, texdata strings 43/44.
"""
import gzip
import re
import struct


HEADER_SIZE = 8 + 16 * 64 + 4
# Lumps the puzzle recovery reads; the rest (lighting, pakfile, ...) is most of a map's size.
USED_LUMPS = (0, 1, 2, 5, 6, 10, 14, 17, 18, 19, 43, 44)


class BspError(Exception):
    pass


def lump_table(header: bytes):
    """[(offset, length, version, fourCC)] for the 64 lumps, from the first HEADER_SIZE bytes."""
    if len(header) < HEADER_SIZE or header[:4] != b"VBSP":
        raise BspError("not a VBSP file")
    version = struct.unpack_from("<i", header, 4)[0]
    if version != 21:
        raise BspError(f"BSP version {version}, expected 21")
    return [struct.unpack_from("<iiii", header, 8 + 16 * i) for i in range(64)]


def slim(header: bytes, lumps: dict) -> bytes:
    """A valid BSP holding only `lumps` ({index: bytes}); every other lump is empty."""
    table = lump_table(header)
    out = bytearray(header[:HEADER_SIZE])
    body = bytearray()
    for i in range(64):
        data = lumps.get(i, b"")
        ofs = HEADER_SIZE + len(body) if data else 0
        struct.pack_into("<iiii", out, 8 + 16 * i, ofs, len(data), table[i][2], table[i][3] if data else 0)
        body += data
        body += b"\0" * (-len(body) % 4)
    return bytes(out + body)


def entities_ok(lump: bytes):
    """Whether an entity lump looks like one: text, starting with the worldspawn."""
    head = lump[:4096].lstrip()
    return head.startswith(b"{") and b'"worldspawn"' in head and b"\0" not in lump.rstrip(b"\0")


def parse_entities(lump: bytes):
    """Each entity as an ordered list of (key, value); keys repeat for outputs."""
    txt = lump.decode("latin-1")
    return [re.findall(r'"([^"]*)"\s+"([^"]*)"', blk) for blk in re.findall(r"\{(.*?)\}", txt, re.S)]


class Bsp:
    def __init__(self, data: bytes):
        lump_table(data)
        self.b = data
        self._planes = None
        self._brushes = None

    @classmethod
    def open(cls, path):
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rb") as f:
            return cls(f.read())

    def lump(self, i):
        ofs, ln, _ver, _cc = struct.unpack_from("<iiii", self.b, 8 + 16 * i)
        if ofs < 0 or ofs + ln > len(self.b):
            raise BspError(f"lump {i} runs past the end of the file (truncated download?)")
        d = self.b[ofs:ofs + ln]
        if d[:4] == b"LZMA":
            raise BspError(f"lump {i} is compressed")
        return d

    def entities(self):
        return parse_entities(self.lump(0))

    def check(self):
        """Raise BspError unless the lumps recovery reads are internally consistent (catches
        corrupt downloads: every index must land inside the lump it points into)."""
        def n(i, size):
            return len(self.lump(i)) // size
        planes, nodes, leafs, brushes, sides = n(1, 20), n(5, 32), n(10, 32), n(18, 12), n(19, 8)
        lb, texinfo = n(17, 2), n(6, 72)
        if not entities_ok(self.lump(0)):
            raise BspError("entity lump isn't a map's entities")
        for nx, ny, nz, _ in self.planes():
            if abs(nx * nx + ny * ny + nz * nz - 1) > 0.01:
                raise BspError("plane normal isn't unit length")
        nl, bs, br = self.lump(5), self.lump(19), self.lump(18)
        for i in range(nodes):
            pl, c0, c1 = struct.unpack_from("<iii", nl, i * 32)
            for c in (c0, c1):
                if not (pl < planes and (0 <= c < nodes or 0 <= -1 - c < leafs)):
                    raise BspError("node points outside the tree")
        for i in range(sides):
            pn, t = struct.unpack_from("<Hh", bs, i * 8)
            if pn >= planes or t >= texinfo:
                raise BspError("brush side points outside the planes or texinfo")
        for i in range(brushes):
            first, cnt, _ = struct.unpack_from("<iii", br, i * 12)
            if first < 0 or first + cnt > sides:
                raise BspError("brush points outside the brush sides")
        lv = self.lump(10)
        for i in range(leafs):
            fb, nb = struct.unpack_from("<HH", lv, i * 32 + 24)
            if fb + nb > lb:
                raise BspError("leaf points outside the leaf brushes")
        if any(b >= brushes for b in struct.unpack(f"<{lb}H", self.lump(17)[:lb * 2])):
            raise BspError("leaf brush points outside the brushes")
        if any(m[3] >= nodes for m in self.models()):
            raise BspError("model head node outside the tree")
        self.texinfo_names()

    def planes(self):
        if self._planes is None:
            p = self.lump(1)
            self._planes = [struct.unpack_from("<ffff", p, i) for i in range(0, len(p), 20)]
        return self._planes

    def texinfo_names(self):
        """Material name of each texinfo."""
        sd, st = self.lump(43), self.lump(44)
        offs = struct.unpack(f"<{len(st) // 4}i", st)
        names = [sd[o:sd.index(b"\0", o)].decode("latin-1") for o in offs]
        td = self.lump(2)
        tdname = [names[struct.unpack_from("<i", td, i + 12)[0]] for i in range(0, len(td), 32)]
        ti = self.lump(6)
        out = []
        for i in range(0, len(ti), 72):
            t = struct.unpack_from("<i", ti, i + 68)[0]
            out.append(tdname[t] if 0 <= t < len(tdname) else "")
        return out

    def brushes(self):
        """[(contents, [(normal, dist, material)])] for every brush in the map."""
        if self._brushes is None:
            self._brushes = self._read_brushes()
        return self._brushes

    def _read_brushes(self):
        planes, tex = self.planes(), self.texinfo_names()
        br, bs = self.lump(18), self.lump(19)
        out = []
        for i in range(0, len(br), 12):
            first, n, contents = struct.unpack_from("<iii", br, i)
            sides = []
            for j in range(first, first + n):
                pn, t, _disp, _bevel = struct.unpack_from("<Hhhh", bs, j * 8)
                nx, ny, nz, d = planes[pn]
                sides.append(((nx, ny, nz), d, tex[t] if 0 <= t < len(tex) else ""))
            out.append((contents, sides))
        return out

    def models(self):
        """(mins, maxs, origin, headnode) per brush model; 0 is the world."""
        m = self.lump(14)
        out = []
        for i in range(0, len(m), 48):
            v = struct.unpack_from("<9f3i", m, i)
            out.append((v[0:3], v[3:6], v[6:9], v[9]))
        return out

    def model_brushes(self, model):
        """Indices of the brushes in a brush model, found by walking its BSP tree."""
        nodes, leafs, lb = self.lump(5), self.lump(10), self.lump(17)
        leafbrush = struct.unpack(f"<{len(lb) // 2}H", lb)
        out, stack = set(), [self.models()[model][3]]
        while stack:
            n = stack.pop()
            if n >= 0:
                stack += struct.unpack_from("<ii", nodes, n * 32 + 4)
            else:
                fb, nb = struct.unpack_from("<HH", leafs, (-1 - n) * 32 + 24)
                out.update(leafbrush[fb:fb + nb])
        return out

    def leaf_contents(self):
        """Function: world point -> contents of the world leaf it's in (outside the map is solid)."""
        nodes, leafs, planes = self.lump(5), self.lump(10), self.planes()

        def contents(p):
            n = 0
            while n >= 0:
                pl, c0, c1 = struct.unpack_from("<iii", nodes, n * 32)
                nx, ny, nz, d = planes[pl]
                n = c0 if nx * p[0] + ny * p[1] + nz * p[2] - d >= 0 else c1
            return struct.unpack_from("<i", leafs, (-1 - n) * 32)[0]

        return contents


def aabb(sides):
    """Box of a brush from its axial sides (every brush has all six)."""
    lo, hi = [-1e9] * 3, [1e9] * 3
    for n, d, _ in sides:
        for a in range(3):
            if abs(n[a]) > 0.999:
                if n[a] > 0:
                    hi[a] = min(hi[a], d)
                else:
                    lo[a] = max(lo[a], -d)
    return lo, hi
