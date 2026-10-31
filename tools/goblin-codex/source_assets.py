"""Read installed Source VPK assets without modifying the game installation."""
import argparse
import json
import struct
from pathlib import Path

GAME = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Portal 2")


def cstring(data, at):
    end = data.index(b"\0", at)
    return data[at:end].decode("utf-8"), end + 1


class VPK:
    def __init__(self, path):
        self.path = Path(path)
        self.data = self.path.read_bytes()
        magic, version, tree_size = struct.unpack_from("<3I", self.data)
        assert magic == 0x55AA1234 and version in (1, 2)
        self.start = 12 if version == 1 else 28
        self.blob_start = self.start + tree_size
        self.entries = {}
        at = self.start
        while True:
            ext, at = cstring(self.data, at)
            if not ext:
                break
            while True:
                folder, at = cstring(self.data, at)
                if not folder:
                    break
                while True:
                    name, at = cstring(self.data, at)
                    if not name:
                        break
                    crc, preload, archive, offset, size, term = struct.unpack_from("<IHHIIH", self.data, at)
                    assert term == 0xFFFF
                    at += 18
                    prefix = self.data[at:at + preload]
                    at += preload
                    key = ((folder + "/") if folder != " " else "") + name + "." + ext
                    self.entries[key] = (archive, offset, size, prefix)

    def read(self, name):
        archive, offset, size, prefix = self.entries[name]
        if archive == 0x7FFF:
            return prefix + self.data[self.blob_start + offset:self.blob_start + offset + size]
        archive_path = self.path.with_name(self.path.name.replace("_dir.vpk", f"_{archive:03}.vpk"))
        with archive_path.open("rb") as f:
            f.seek(offset)
            return prefix + f.read(size)


def skeleton(data):
    assert data[:4] == b"IDST"
    version = struct.unpack_from("<I", data, 4)[0]
    count, offset = struct.unpack_from("<2I", data, 156)
    bones = []
    for i in range(count):
        at = offset + i * 216
        name_offset, parent = struct.unpack_from("<2i", data, at)
        name, _ = cstring(data, at + name_offset)
        pos = struct.unpack_from("<3f", data, at + 32)
        quat = struct.unpack_from("<4f", data, at + 44)
        rot = struct.unpack_from("<3f", data, at + 60)
        bones.append(dict(name=name, parent=parent, pos=pos, quat=quat, rot=rot))
    return dict(version=version, bones=bones)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", type=Path, default=GAME)
    p.add_argument("--out", type=Path, default=Path("out/goblin-codex/source"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    vpk = VPK(args.game / "portal2/pak01_dir.vpk")
    models = [n for n in vpk.entries if n in (
        "models/player.mdl", "models/player_animations.mdl",
        "models/player/chell/player.mdl", "models/player/chell/headless_player.mdl",
        "models/player/ballbot/ballbot.mdl", "models/player/eggbot/eggbot.mdl",
    )]
    print("Player models:", *models, sep="\n")
    for name in models:
        if True:
            raw = vpk.read(name)
            dest = args.out / name.replace("/", "_")
            dest.write_bytes(raw)
            info = skeleton(raw)
            dest.with_suffix(".json").write_text(json.dumps(info, indent=2))
            print(name, info["version"], len(info["bones"]), [b["name"] for b in info["bones"][:8]])
    dlc = VPK(args.game / "portal2_dlc2/pak01_dir.vpk")
    name = "models/info_character/info_character_player.mdl"
    raw = dlc.read(name)
    dest = args.out / "bendy.mdl"
    dest.write_bytes(raw)
    dest.with_suffix('.json').write_text(json.dumps(skeleton(raw), indent=2))
    print('BENDY', len(skeleton(raw)['bones']), [b['name'] for b in skeleton(raw)['bones']])


if __name__ == "__main__":
    main()
