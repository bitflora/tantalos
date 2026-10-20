"""Copy generated .p2c files into the Puzzle Maker's folder for play-testing.

Generated levels carry a fixed creation time (src/builder.rs keeps fixtures byte-stable), so
the editor lists them all as made on Oct 8 2025. Copies get Timestamp_Created/_Modified set
to now, a second apart in argument order so the editor's date sort keeps that order. Files
are named glados-<source name>.p2c unless the name already starts with glados-. Each copy gets
a preview image (<name>.jpg, train/preview.py: isometric view and item list) for the editor's Open
screen, its Description lists the items, and its Title is set to the file name: `glados score
--p2c` titles levels by their line number, which stops being unique once a run name is added.

    python train/playtest.py out/sample/p2c/00003.p2c favorites/laser.p2c
    python train/playtest.py out/sample/p2c --prefix v3-          # glados-v3-00001.p2c, ...
    python train/playtest.py --restamp                            # fix glados-* already copied in
    python train/playtest.py --titles                             # set their Titles to their names
    python train/playtest.py --previews                           # add previews to glados-* there
"""
import argparse, re, time
from pathlib import Path

import preview

PUZZLES = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Portal 2\portal2\puzzles\76561197975512835")
STAMP = re.compile(r'("Timestamp_(?:Created|Modified)"\s+)"0x[0-9A-Fa-f]{16}"')
DESCRIPTION = re.compile(r'("Description"\s+)"[^"]*"')
TITLE = re.compile(r'("Title"\s+)"[^"]*"')


def restamp(text, when):
    text, n = STAMP.subn(lambda m: f'{m[1]}"0x{int(when):016X}"', text)
    if n != 2:
        raise ValueError(f"expected 2 timestamps, found {n}")
    return text


def write(path, text):
    path.write_bytes(text.encode("utf-8"))


def retitle(text, title):
    text, n = TITLE.subn(lambda m: f'{m[1]}"{title}"', text, count=1)
    if n != 1:
        raise ValueError(f"expected 1 title, found {n}")
    return text


def add_preview(path):
    """Set path's Title to its file name, write its .jpg preview and list items in Description."""
    text = path.read_bytes().decode("utf-8")
    text = retitle(text, path.stem)
    write(path, DESCRIPTION.sub(lambda m: f'{m[1]}"{preview.summary(path)}"', text, count=1))
    preview.write(path, name=path.stem.removeprefix("glados-"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("levels", nargs="*", help=".p2c files or directories of them")
    ap.add_argument("--prefix", default="", help="goes between glados- and the source name")
    ap.add_argument("--restamp", action="store_true",
                    help="instead, restamp the glados-*.p2c already in the folder with their copy time")
    ap.add_argument("--titles", action="store_true",
                    help="instead, set the glados-*.p2c Titles to their file names")
    ap.add_argument("--previews", action="store_true",
                    help="instead, add previews to the glados-*.p2c already in the folder")
    args = ap.parse_args()

    if args.previews:
        for f in sorted(PUZZLES.glob("glados-*.p2c")):
            add_preview(f)
            print(f.with_suffix(".jpg").name)
        return

    if args.titles:
        for f in sorted(PUZZLES.glob("glados-*.p2c")):
            write(f, retitle(f.read_bytes().decode("utf-8"), f.stem))
            print(f.name)
        return

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
        add_preview(PUZZLES / name)
        print(PUZZLES / name)


if __name__ == "__main__":
    main()
