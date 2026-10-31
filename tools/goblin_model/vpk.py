"""Read and write Valve VPK archives (v1 and v2 directories), enough for Portal 2 mods.

    python tools/goblin_model/vpk.py list DIR.vpk [SUBSTRING]
    python tools/goblin_model/vpk.py extract DIR.vpk OUTDIR PATH [PATH ...]   (a PATH ending in / takes a folder)
    python tools/goblin_model/vpk.py pack SRCDIR OUT_dir.vpk                   (one self-contained v1 file)
"""
import os
import struct
import sys
import zlib

SIG = 0x55AA1234


def _cstr(buf, pos):
    end = buf.index(b"\0", pos)
    return buf[pos:end].decode("latin-1"), end + 1


def read_dir(path):
    """{path: (archive_index, offset, length, preload_bytes)} for a *_dir.vpk."""
    with open(path, "rb") as f:
        data = f.read()
    sig, ver, tree = struct.unpack_from("<III", data, 0)
    if sig != SIG:
        raise ValueError(f"{path}: not a VPK")
    head = 12 if ver == 1 else 28
    pos, entries = head, {}
    while True:
        ext, pos = _cstr(data, pos)
        if not ext:
            break
        while True:
            folder, pos = _cstr(data, pos)
            if not folder:
                break
            while True:
                name, pos = _cstr(data, pos)
                if not name:
                    break
                _crc, npre, arch, off, length, _term = struct.unpack_from("<IHHIIH", data, pos)
                pos += 18
                pre = data[pos:pos + npre]
                pos += npre
                full = (f"{folder}/" if folder != " " else "") + name + (f".{ext}" if ext != " " else "")
                if arch == 0x7FFF:
                    off += head + tree
                entries[full] = (arch, off, length, pre)
    return entries


def read_file(dir_path, entries, name):
    arch, off, length, pre = entries[name]
    if length == 0:
        return pre
    if arch == 0x7FFF:
        src = dir_path
    else:
        src = dir_path.replace("_dir.vpk", f"_{arch:03d}.vpk")
    with open(src, "rb") as f:
        f.seek(off)
        return pre + f.read(length)


def pack(src_dir, out_path):
    """Writes every file under src_dir into one v1 VPK, data after the tree."""
    files = []
    for root, _dirs, names in os.walk(src_dir):
        for n in names:
            full = os.path.join(root, n)
            rel = os.path.relpath(full, src_dir).replace("\\", "/").lower()
            files.append((rel, full))
    tree_parts = {}
    for rel, full in files:
        folder, name = os.path.split(rel)
        base, ext = os.path.splitext(name)
        tree_parts.setdefault(ext[1:] or " ", {}).setdefault(folder or " ", []).append((base, full))
    tree, blob = bytearray(), bytearray()
    for ext in sorted(tree_parts):
        tree += ext.encode() + b"\0"
        for folder in sorted(tree_parts[ext]):
            tree += folder.encode() + b"\0"
            for base, full in sorted(tree_parts[ext][folder]):
                with open(full, "rb") as f:
                    content = f.read()
                tree += base.encode() + b"\0"
                tree += struct.pack("<IHHIIH", zlib.crc32(content) & 0xFFFFFFFF, 0, 0x7FFF,
                                    len(blob), len(content), 0xFFFF)
                blob += content
            tree += b"\0"
        tree += b"\0"
    tree += b"\0"
    with open(out_path, "wb") as f:
        f.write(struct.pack("<III", SIG, 1, len(tree)))
        f.write(tree)
        f.write(blob)
    return len(files)


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    cmd = argv[1]
    if cmd == "list":
        entries = read_dir(argv[2])
        sub = argv[3].lower() if len(argv) > 3 else ""
        for name in sorted(entries):
            if sub in name:
                print(name, entries[name][2] + len(entries[name][3]))
    elif cmd == "extract":
        entries = read_dir(argv[2])
        out = argv[3]
        for want in argv[4:]:
            names = [n for n in entries if n.startswith(want)] if want.endswith("/") else [want]
            for name in names:
                dest = os.path.join(out, name)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest, "wb") as f:
                    f.write(read_file(argv[2], entries, name))
                print(dest)
    elif cmd == "pack":
        print(pack(argv[2], argv[3]), "files")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
