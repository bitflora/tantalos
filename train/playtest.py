"""Copy generated .p2c files into the Puzzle Maker's folder for play-testing.

Generated levels carry a fixed creation time (src/builder.rs keeps fixtures byte-stable), so
the editor lists them all as made on Oct 8 2025. Copies get Timestamp_Created/_Modified set
to now, a second apart in argument order so the editor's date sort keeps that order. Files
are named glados-<source name>.p2c unless the name already starts with glados-.

    python train/playtest.py out/sample/p2c/00003.p2c favorites/laser.p2c
    python train/playtest.py out/sample/p2c --prefix v3-          # glados-v3-00001.p2c, ...
    python train/playtest.py --restamp                            # fix glados-* already copied in
"""
import argparse, re, time
from pathlib import Path

PUZZLES = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Portal 2\portal2\puzzles\76561197975512835")
STAMP = re.compile(r'("Timestamp_(?:Created|Modified)"\s+)"0x[0-9A-Fa-f]{16}"')


def restamp(text, when):
    text, n = STAMP.subn(lambda m: f'{m[1]}"0x{int(when):016X}"', text)
    if n != 2:
        raise ValueError(f"expected 2 timestamps, found {n}")
    return text


def write(path, text):
    path.write_bytes(text.encode("utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("levels", nargs="*", help=".p2c files or directories of them")
    ap.add_argument("--prefix", default="", help="goes between glados- and the source name")
    ap.add_argument("--restamp", action="store_true",
                    help="instead, restamp the glados-*.p2c already in the folder with their copy time")
    args = ap.parse_args()

    if args.restamp:
        for f in sorted(PUZZLES.glob("glados-*.p2c")):
            write(f, restamp(f.read_bytes().decode("utf-8"), f.stat().st_mtime))
            print(f.name)
        return
    files = [g for p in map(Path, args.levels) for g in (sorted(p.glob("*.p2c")) if p.is_dir() else [p])]
    if not files:
        ap.error("no levels given")
    now = int(time.time())
    for i, f in enumerate(files):
        name = f.name if f.name.startswith("glados-") else f"glados-{args.prefix}{f.name}"
        write(PUZZLES / name, restamp(f.read_bytes().decode("utf-8"), now + i))
        print(PUZZLES / name)


if __name__ == "__main__":
    main()
