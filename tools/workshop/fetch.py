"""Collect Puzzle Maker chambers from the Portal 2 Workshop as glados training levels.

    python tools/workshop/fetch.py list --pages 100       # find puzzles: out/workshop/items.jsonl
    python tools/workshop/fetch.py download               # fetch them:   out/workshop/bsp/<id>.bsp
    python tools/workshop/fetch.py build                  # convert+solve: out/workshop/levels.jsonl

`list` reads the Workshop's public browse pages (top rated by default), singleplayer only and
without the item tags the tokens can't express, then keeps the items published from the Puzzle
Maker (file `mymaps/<number>.bsp`). Metadata comes from the keyless GetPublishedFileDetails API.

`download` doesn't take whole maps (5-40 MB, mostly lighting and packed files). It reads the
header, then the entity lump, and stops there if the instance names already rule the map out
(BEEmod, unsupported items); otherwise it fetches only the lumps convert.py reads (~0.6 MB) and
saves them as a slim BSP. Every map's outcome goes to out/workshop/fetched.jsonl, so reruns skip
what's done. Steam's CDN drops connections now and then; requests retry with backoff.

`build` converts every slim map (tools/workshop/convert.py), scores the tokens with
`glados score`, and writes the dataset records (the same fields as `glados dataset`, plus
`workshop_id` and `title`). Use --p2c DIR to also get the solvable ones as .p2c files.
"""
import argparse
import concurrent.futures
import http.client
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bsp import HEADER_SIZE, USED_LUMPS, Bsp, BspError, entities_ok, lump_table, parse_entities, slim  # noqa: E402
from convert import Unsupported, convert, screen  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
APP = 620
BROWSE = "https://steamcommunity.com/workshop/browse/"
DETAILS = "https://api.steampowered.com/ISteamRemoteStorage/GetPublishedFileDetails/v1/"
# Workshop tags of Puzzle Maker items the tokens can't express, and of non-Puzzle-Maker maps.
EXCLUDED_TAGS = [
    "Cooperative", "Custom Visuals", "Custom Story", "Angled Panel", "Flip Panel", "Glass Panel",
    "Faith Plate", "Light Bridge", "Tractor Beam", "Laser Emitter", "Laser Catcher", "Laser Relay",
    "Piston Platform", "Rail Platform", "Stairs",
]
PUZZLE_MAKER_FILE = re.compile(r"mymaps/\d+\.bsp")
UA = {"User-Agent": "glados-workshop-fetch/1 (research; low rate)"}


def fetch_url(url, data=None, headers=None, timeout=60):
    req = urllib.request.Request(url, data=data, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def retrying(fn, tries=4, what="request"):
    delay = 1.0
    for attempt in range(tries):
        try:
            return fn()
        except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as e:
            if attempt == tries - 1:
                raise IOError(f"{what}: {e}") from e
            time.sleep(delay)
            delay = min(delay * 2, 30)


def get_range(url, start, length):
    """Bytes [start, start + length) of url. Steam's CDN sometimes answers a range request with
    the right status and length but the wrong bytes (zeros, or another part of the file), so a
    range only counts once two responses agree."""
    def once():
        status, body = fetch_url(url, headers={"Range": f"bytes={start}-{start + length - 1}"})
        if status == 200:  # range ignored: the whole file
            body = body[start:start + length]
        if len(body) != length:
            raise ValueError(f"got {len(body)} of {length} bytes")
        return body
    seen = []
    for _ in range(6):
        body = retrying(once, what=f"bytes {start}+{length}")
        if body in seen:
            return body
        seen.append(body)
    raise IOError(f"bytes {start}+{length}: the CDN keeps returning different data")


# ---- list ----

def browse_page(sort, page, excluded):
    q = [("appid", APP), ("browsesort", sort), ("actualsort", sort), ("section", "readytouseitems"),
         ("days", -1), ("numperpage", 30), ("p", page), ("requiredtags[]", "Singleplayer")]
    q += [("excludedtags[]", t) for t in excluded]
    _, body = retrying(lambda: fetch_url(BROWSE + "?" + urllib.parse.urlencode(q)), what=f"browse page {page}")
    seen, ids = set(), []
    for m in re.finditer(r"sharedfiles/filedetails/\?id=(\d+)", body.decode("utf-8", "replace")):
        if m.group(1) not in seen:
            seen.add(m.group(1))
            ids.append(m.group(1))
    return ids


def details(ids):
    out = []
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        form = {"itemcount": len(chunk), **{f"publishedfileids[{j}]": x for j, x in enumerate(chunk)}}
        _, body = retrying(lambda: fetch_url(DETAILS, data=urllib.parse.urlencode(form).encode()), what="details")
        out += json.loads(body)["response"]["publishedfiledetails"]
    return out


def keep(d, excluded):
    tags = {t["tag"] for t in d.get("tags", [])}
    return (d.get("result") == 1 and d.get("consumer_app_id") == APP and d.get("file_url")
            and PUZZLE_MAKER_FILE.fullmatch(d.get("filename") or "") and not tags & set(excluded))


def cmd_list(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items_path = out / "items.jsonl"
    known = set()
    if items_path.exists():
        known = {json.loads(l)["id"] for l in open(items_path, encoding="utf-8")}
    excluded = [] if a.all_items else EXCLUDED_TAGS
    added = 0
    with open(items_path, "a", encoding="utf-8") as f:
        for sort in a.sort:
            prev = None
            for page in range(a.start, a.start + a.pages):
                ids = browse_page(sort, page, excluded)
                if not ids or ids == prev:  # past the last page (Steam repeats it)
                    break
                prev = ids
                new = [i for i in ids if i not in known]
                for d in details(new) if new else []:
                    known.add(d["publishedfileid"])
                    if not keep(d, excluded):
                        continue
                    rec = {k: d.get(k) for k in ("title", "file_url", "file_size", "filename", "time_created",
                                                 "time_updated", "subscriptions", "lifetime_subscriptions",
                                                 "favorited", "views")}
                    rec = {"id": d["publishedfileid"], "sort": sort, "page": page, **rec,
                           "tags": [t["tag"] for t in d.get("tags", [])]}
                    f.write(json.dumps(rec) + "\n")
                    added += 1
                f.flush()
                print(f"{sort} page {page}: {len(new)} new ids, {added} puzzles added", file=sys.stderr)
                time.sleep(a.delay)
    print(f"{added} added; {items_path}")


# ---- download ----

def fetch_one(item, bsp_dir, tries=4):
    """Fetch, retrying the whole map when its data doesn't check out. While a CDN edge is still
    pulling a big file from Steam's origin it can serve the same wrong bytes (often zeros) to
    every request, so agreement between responses isn't enough; waiting usually fixes it."""
    for attempt in range(tries):
        try:
            return fetch_once(item, bsp_dir)
        except BspError:
            if attempt == tries - 1:
                raise
            time.sleep(15 * (attempt + 1))


def fetch_once(item, bsp_dir):
    url = item["file_url"]
    header = get_range(url, 0, HEADER_SIZE)
    table = lump_table(header)
    total = int(item["file_size"])
    if any(table[i][0] + table[i][1] > total for i in USED_LUMPS):
        raise BspError("lump table runs past the end of the file")
    ofs, ln = table[0][:2]
    ents = get_range(url, ofs, ln)
    if not entities_ok(ents):
        raise BspError("entity lump isn't a map's entities")
    try:
        screen(parse_entities(ents))
    except Unsupported as e:
        return {"status": "unsupported", "reason": str(e), "bytes": HEADER_SIZE + ln}
    lumps, fetched = {0: ents}, HEADER_SIZE + ln
    # Fetch the rest in as few requests as possible: join lumps less than 64 KB apart.
    want = sorted((table[i][0], table[i][1], i) for i in USED_LUMPS if i != 0 and table[i][1] > 0)
    groups = []
    for o, l, i in want:
        if groups and o - (groups[-1][0] + groups[-1][1]) < 65536:
            g = groups[-1]
            g[1] = max(g[1], o + l - g[0])
            g[2].append(i)
        else:
            groups.append([o, l, [i]])
    for o, l, idx in groups:
        data = get_range(url, o, l)
        fetched += l
        for i in idx:
            lumps[i] = data[table[i][0] - o:table[i][0] - o + table[i][1]]
    blob = slim(header, lumps)
    Bsp(blob).check()
    (bsp_dir / f"{item['id']}.bsp").write_bytes(blob)
    return {"status": "ok", "bytes": fetched}


def cmd_download(a):
    out = Path(a.out)
    bsp_dir = out / "bsp"
    bsp_dir.mkdir(parents=True, exist_ok=True)
    items = [json.loads(l) for l in open(out / "items.jsonl", encoding="utf-8")]
    log_path = out / "fetched.jsonl"
    done = {}
    if log_path.exists():
        for l in open(log_path, encoding="utf-8"):
            r = json.loads(l)
            done[r["id"]] = r["status"]
    todo = [it for it in items if it["id"] not in done or (a.retry_errors and done[it["id"]] == "error")]
    todo = list({it["id"]: it for it in todo}.values())[:a.max or None]
    print(f"{len(todo)} to fetch ({len(done)} already done)", file=sys.stderr)
    counts, total_bytes = {}, 0
    with open(log_path, "a", encoding="utf-8") as log, \
            concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
        futures = {pool.submit(fetch_one, it, bsp_dir): it for it in todo}
        for n, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            it = futures[fut]
            try:
                r = fut.result()
            except (IOError, BspError, Unsupported) as e:
                r = {"status": "error", "reason": str(e)}
            r = {"id": it["id"], **r}
            log.write(json.dumps(r) + "\n")
            log.flush()
            counts[r["status"]] = counts.get(r["status"], 0) + 1
            total_bytes += r.get("bytes", 0)
            if n % 25 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} {counts} {total_bytes / 1e6:.0f} MB", file=sys.stderr)


# ---- build ----

def cmd_build(a):
    out = Path(a.out)
    titles = {}
    for l in open(out / "items.jsonl", encoding="utf-8"):
        r = json.loads(l)
        titles[r["id"]] = r["title"]
    ids, lines, problems = [], [], {}
    with open(out / "report.jsonl", "w", encoding="utf-8") as report:
        for path in sorted((out / "bsp").glob("*.bsp")):
            rec = {"id": path.stem}
            try:
                rec["tokens"] = convert(Bsp.open(path))
                ids.append(path.stem)
                lines.append(rec["tokens"])
            except (Unsupported, BspError) as e:
                rec["error"] = str(e)
                problems[path.stem] = str(e)
            report.write(json.dumps(rec) + "\n")
    print(f"converted {len(lines)}; {len(problems)} unsupported after download (see report.jsonl)",
          file=sys.stderr)
    # Published puzzles already passed the game's own editor; our model of its rules would only
    # add false positives (e.g. fizzlers centred in a doorway, fields the game let cross).
    cmd = [a.glados, "score", "--budget", str(a.budget), "--editor-rules", "off"]
    if a.p2c:
        cmd += ["--p2c", a.p2c]
    res = subprocess.run(cmd, input="\n".join(lines) + "\n", capture_output=True, text=True, check=True)
    verdicts = {}
    with open(out / "levels.jsonl", "w", encoding="utf-8") as f:
        for wid, line in zip(ids, res.stdout.splitlines()):
            r = json.loads(line)
            r.pop("line", None)
            r.pop("canonical", None)
            r = {**r, "workshop_id": wid, "title": titles.get(wid, "")}
            verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
            f.write(json.dumps(r) + "\n")
    print(f"{verdicts}; {out / 'levels.jsonl'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "out" / "workshop"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list", help="find Puzzle Maker puzzles on the Workshop")
    p.add_argument("--sort", nargs="+", default=["toprated"],
                   help="browse orders: toprated, mostrecent, trend, totaluniquesubscribers, ...")
    p.add_argument("--pages", type=int, default=20, help="pages of 30 per sort (Steam stops near 1667)")
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--delay", type=float, default=1.0, help="seconds between browse pages")
    p.add_argument("--all-items", action="store_true", help="don't exclude tags of unsupported items")
    p = sub.add_parser("download", help="fetch the listed puzzles' slim maps")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max", type=int, default=0)
    p.add_argument("--retry-errors", action="store_true")
    p = sub.add_parser("build", help="convert and score the downloaded maps")
    p.add_argument("--glados", default=str(ROOT / "target" / "release" / ("glados.exe" if sys.platform == "win32" else "glados")))
    p.add_argument("--budget", type=int, default=50_000)
    p.add_argument("--p2c", help="also write the solvable levels as .p2c files here")
    a = ap.parse_args()
    {"list": cmd_list, "download": cmd_download, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
