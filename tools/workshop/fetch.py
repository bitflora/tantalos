"""Collect Puzzle Maker chambers from the Portal 2 Workshop as glados training levels.

    python tools/workshop/fetch.py list --pages 100       # find puzzles: out/workshop/items.jsonl
    python tools/workshop/fetch.py download               # fetch them:   out/workshop/bsp/<id>.bsp
    python tools/workshop/fetch.py build                  # convert+solve: out/workshop/levels.jsonl
    python tools/workshop/fetch.py pipeline               # download while converting+scoring

`pipeline` runs `download` (and optionally a `--retry-errors` pass) as a child process and,
every `--interval` seconds, builds only the maps that download has finished (`build
--incremental` appends to report.jsonl/levels.jsonl, using fetched.jsonl as the readiness
signal). It finalizes levels.jsonl at the end. This overlaps the network-bound download with
the CPU-bound converter and solver instead of doing the two phases one after the other.

`list` reads the Workshop's public browse pages (top rated by default), singleplayer only and
without the tags of co-op and custom maps, then keeps the items published from the Puzzle
Maker (file `mymaps/<number>.bsp`). Metadata comes from the keyless GetPublishedFileDetails API.

`download` doesn't take whole maps (5-40 MB, mostly lighting and packed files). It reads the
header, then the entity lump, and stops there if the instance names already rule the map out
(BEEmod, unsupported items); otherwise it fetches only the lumps convert.py reads (~0.6 MB) and
saves them as a slim BSP. Every map's outcome goes to out/workshop/fetched.jsonl, so reruns skip
what's done. Steam's CDN drops connections now and then; requests retry with backoff.

`build` converts every slim map (tools/workshop/convert.py), scores the tokens with
`glados score`, and writes the dataset records (the same fields as `glados dataset`, plus
`workshop_id`, `title`, and the popularity that train/data.py's `tier` condition reads:
`subscribers` (lifetime unique) and `favorited`). Use --p2c DIR to also get the solvable ones as .p2c files.
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
# Workshop tags of non-Puzzle-Maker maps (every Puzzle Maker item has tokens now).
EXCLUDED_TAGS = ["Cooperative", "Custom Visuals", "Custom Story"]
PUZZLE_MAKER_FILE = re.compile(r"mymaps/\d+\.bsp")
# Steam's CDN throttles custom/script user agents by answering a range request with 206 and an
# empty body (IncompleteRead), which used to leave thousands of maps as "bytes 0+1036" errors.
# The Steam client UA gets the same public bytes reliably; keep the request rate low instead.
UA = {"User-Agent": "Valve/Steam HTTP Client 1.0", "Accept-Encoding": "identity", "Connection": "close"}
RATING = re.compile(r'"star_rating":(-?\d+),"total_votes":(\d+)')
# Level files that carry `stars`/`votes` (`ratings` updates them in place; ws3.py copies them on).
RATED_FILES = ["out/workshop/levels.jsonl", "data/ws5/train.jsonl", "data/ws5/test.jsonl"]


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
            if getattr(e, "code", None) == 429:  # steamcommunity.com rate limit: back off for a while
                delay = max(delay, 60)
            time.sleep(delay)
            delay = min(delay * 2, 30)


def get_range(url, start, length):
    """Bytes [start, start + length) of url. Steam's CDN sometimes answers a range request with
    the right status and length but the wrong bytes (zeros, or another part of the file), so a
    range only counts once two responses agree. Empty 206s are common and cheap to retry, so
    retry them immediately; a range that keeps failing is left for the next `download` pass
    rather than spending minutes in backoff on the whole map."""
    def once():
        status, body = fetch_url(url, headers={"Range": f"bytes={start}-{start + length - 1}"})
        if status == 200:  # range ignored: the whole file
            body = body[start:start + length]
        if len(body) != length:
            raise ValueError(f"got {len(body)} of {length} bytes")
        return body
    seen = []
    for attempt in range(12):
        try:
            body = once()
        except (ValueError, http.client.HTTPException):
            time.sleep(0.05)  # empty/short body: transient, try again straight away
            continue
        except (urllib.error.URLError, OSError):
            time.sleep(min(0.25 * (attempt + 1), 2.0))
            continue
        if body in seen:
            return body
        seen.append(body)
    raise IOError(f"bytes {start}+{length}: the CDN kept returning bad data")


# ---- list ----

def browse_page(sort, page, excluded):
    """The page's item ids, in order, and the star ratings it shows ({id: {stars, votes}})."""
    q = [("appid", APP), ("browsesort", sort), ("actualsort", sort), ("section", "readytouseitems"),
         ("days", -1), ("numperpage", 30), ("p", page), ("requiredtags[]", "Singleplayer")]
    q += [("excludedtags[]", t) for t in excluded]
    _, body = retrying(lambda: fetch_url(BROWSE + "?" + urllib.parse.urlencode(q)), what=f"browse page {page}")
    text = body.decode("utf-8", "replace")
    seen, ids = set(), []
    for m in re.finditer(r"sharedfiles/filedetails/\?id=(\d+)", text):
        if m.group(1) not in seen:
            seen.add(m.group(1))
            ids.append(m.group(1))
    return ids, page_ratings(text)


def page_ratings(text):
    """The browse page embeds its items as JSON inside a JSON string: each one starts with
    {"publishedfileid":...} and carries "star_rating" (1-5, the page's star image; -1 for "not
    enough ratings yet", stored as 0) and "total_votes" (the item page's "N ratings")."""
    text = text.replace('\\"', '"')
    starts = [(m.start(), m[1]) for m in re.finditer(r'\{"publishedfileid":"(\d+)"', text)]
    out = {}
    for (s, wid), (e, _) in zip(starts, starts[1:] + [(len(text), None)]):
        m = RATING.search(text, s, e)
        if m:
            out[wid] = {"stars": max(int(m[1]), 0), "votes": int(m[2])}
    return out


def detail_rating(wid):
    """The star rating on an item's own page, for items no browse page showed."""
    url = f"https://steamcommunity.com/sharedfiles/filedetails/?id={wid}"
    # Item pages answer 429 to every request without an Accept header (browse pages don't).
    _, body = retrying(lambda: fetch_url(url, headers={"Accept": "*/*"}), what=f"item {wid}")
    text = body.decode("utf-8", "replace")
    stars = re.search(r"/(\d)-star_large\.png|/(not-yet)_large\.png", text)
    if not stars:
        return None  # removed or hidden item
    votes = re.search(r'numRatings">([\d,]+) ratings', text)
    return {"stars": int(stars[1] or 0), "votes": int(votes[1].replace(",", "")) if votes else 0}


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
                ids, rated = browse_page(sort, page, excluded)
                if not ids or ids == prev:  # past the last page (Steam repeats it)
                    break
                prev = ids
                save_ratings(out, rated, sort=sort, page=page)
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


# ---- ratings ----

def rating_lines(out):
    path = out / "ratings.jsonl"
    if not path.exists():
        return []
    lines = []
    for line in open(path, encoding="utf-8"):
        try:
            lines.append(json.loads(line))
        except ValueError:  # a partial last line
            pass
    return lines


def save_ratings(out, rated, sort, page):
    """Append one browse page's (or item page's) ratings to out/ratings.jsonl."""
    with open(out / "ratings.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": int(time.time()), "sort": sort, "page": page, "ratings": rated}) + "\n")


def load_ratings(out):
    """{workshop id: {"stars", "votes"}}, the latest seen for each item."""
    rated = {}
    for r in rating_lines(out):
        rated.update({k: v for k, v in r["ratings"].items() if v})
    return rated


def annotate(rated, paths):
    """Set `stars`/`votes` in place on every record of these level files that has a rating."""
    for path in paths:
        if not path.exists():
            continue
        n = total = 0
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(path, encoding="utf-8") as src, open(tmp, "w", encoding="utf-8", newline="\n") as dst:
            for line in src:
                if not line.strip():
                    continue
                r = json.loads(line)
                total += 1
                if str(r.get("workshop_id")) in rated:
                    r.update(rated[str(r["workshop_id"])])
                    n += 1
                dst.write(json.dumps(r) + "\n")
        tmp.replace(path)
        print(f"{path}: {n} of {total} levels rated")


def cmd_ratings(a):
    """Re-read every browse page items.jsonl was listed from, for the star ratings they show; then
    the item pages of converted levels no browse page showed (they moved to another page)."""
    out = Path(a.out)
    if not a.annotate_only:
        fresh = time.time() - a.max_age * 86400
        done = {(r["sort"], str(r["page"])) for r in rating_lines(out) if r["time"] >= fresh}
        pages = sorted({(r["sort"], r["page"]) for r in map(json.loads, open(out / "items.jsonl", encoding="utf-8"))})
        todo = [p for p in pages if (p[0], str(p[1])) not in done]
        print(f"{len(todo)} browse pages to read ({len(pages) - len(todo)} read in the last {a.max_age} days)",
              file=sys.stderr)
        for n, (sort, page) in enumerate(todo, 1):
            try:
                _, rated = browse_page(sort, page, EXCLUDED_TAGS)
                save_ratings(out, rated, sort=sort, page=page)
            except IOError as e:
                print(f"{sort} page {page}: {e} (a rerun retries it)", file=sys.stderr)
            if n % 50 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} pages, {len(load_ratings(out))} items rated", file=sys.stderr)
            time.sleep(a.delay)
        converted = read_ids(out / "levels.jsonl", key="workshop_id")
        missing = sorted(converted - set(load_ratings(out)) - {p for s, p in done if s == "item"})
        print(f"{len(missing)} converted levels not on any page read: their item pages", file=sys.stderr)
        for n, wid in enumerate(missing, 1):
            try:
                r = detail_rating(wid)
                save_ratings(out, {wid: r}, sort="item", page=wid)
            except IOError as e:
                print(f"{e} (a rerun retries it)", file=sys.stderr)
            if n % 50 == 0 or n == len(missing):
                print(f"{n}/{len(missing)} item pages", file=sys.stderr)
            time.sleep(a.delay * 4)  # item pages are rate-limited sooner
    rated = load_ratings(out)
    print(f"{len(rated)} items rated; {out / 'ratings.jsonl'}")
    annotate(rated, [ROOT / f for f in (a.files or RATED_FILES)])


# ---- download ----

def fetch_one(item, bsp_dir, tries=2):
    """Fetch, retrying the whole map when its data doesn't check out. While a CDN edge is still
    pulling a big file from Steam's origin it can serve the same wrong bytes (often zeros) to
    every request, so agreement between responses isn't enough; waiting usually fixes it. Keep
    this short: a map that still fails is logged and picked up by a later --retry-errors pass."""
    for attempt in range(tries):
        try:
            return fetch_once(item, bsp_dir)
        except BspError:
            if attempt == tries - 1:
                raise
            time.sleep(5)


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

def read_ids(path, status=None, key="id"):
    """Ids in a JSONL file, skipping a partial last line and unparsable records."""
    ids = set()
    if not path.exists():
        return ids
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
            if status is None or r.get("status") == status:
                ids.add(str(r[key]))
        except (ValueError, KeyError, TypeError):
            pass
    return ids


def score_levels(a, ids, lines, meta, out, mode):
    """Score converted token lines in one `glados score` pass and write their records."""
    if not lines:
        return {}
    # Published puzzles already passed the game's own editor; our model of its rules would only
    # add false positives (e.g. fizzlers centred in a doorway, fields the game let cross).
    cmd = [a.glados, "score", "--budget", str(a.budget), "--editor-rules", "off"]
    if a.p2c:
        cmd += ["--p2c", a.p2c]
    res = subprocess.run(cmd, input="\n".join(lines) + "\n", capture_output=True, text=True, check=True)
    verdicts = {}
    with open(out / "levels.jsonl", mode, encoding="utf-8") as f:
        for wid, line in zip(ids, res.stdout.splitlines()):
            r = json.loads(line)
            r.pop("line", None)
            r.pop("canonical", None)
            r = {**r, "workshop_id": wid, **meta.get(wid, {"title": ""})}
            verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
            f.write(json.dumps(r) + "\n")
    return verdicts


def cmd_build(a):
    out = Path(a.out)
    meta, rated = {}, load_ratings(out)
    for l in open(out / "items.jsonl", encoding="utf-8"):
        r = json.loads(l)
        meta[r["id"]] = {"title": r["title"], "subscribers": r.get("lifetime_subscriptions"),
                         "favorited": r.get("favorited"), **rated.get(r["id"], {})}
    incremental = getattr(a, "incremental", False)
    if incremental:
        # Only what a completed download has produced and no build has seen yet. fetched.jsonl
        # is flushed after each map's .bsp is written, so it is the readiness signal; old bsp
        # files with no fetched record (previous runs, fixtures) are taken once they are a
        # minute old, which also keeps us away from a file being written right now.
        done = read_ids(out / "report.jsonl")
        ok = read_ids(out / "fetched.jsonl", status="ok")
        now = time.time()
        paths = []
        for path in sorted((out / "bsp").glob("*.bsp")):
            if path.stem in done:
                continue
            try:
                old = now - path.stat().st_mtime > 60
            except OSError:
                continue
            if path.stem in ok or old:
                paths.append(path)
        if not paths:
            return 0
    else:
        paths = sorted((out / "bsp").glob("*.bsp"))
    ids, lines, problems = [], [], {}
    with open(out / "report.jsonl", "a" if incremental else "w", encoding="utf-8") as report:
        for path in paths:
            rec = {"id": path.stem}
            try:
                rec["tokens"] = convert(Bsp.open(path))
                ids.append(path.stem)
                lines.append(rec["tokens"])
            except (Unsupported, BspError) as e:
                rec["error"] = str(e)
                problems[path.stem] = str(e)
            report.write(json.dumps(rec) + "\n")
    print(f"converted {len(lines)} of {len(paths)}; {len(problems)} unsupported "
          f"(see report.jsonl)", file=sys.stderr)
    verdicts = score_levels(a, ids, lines, meta, out, "a" if incremental else "w")
    print(f"{verdicts}; {out / 'levels.jsonl'}")
    return len(paths)


def cmd_finalize(a):
    """Merge an incrementally built levels.jsonl, keeping the last record for each id."""
    out = Path(a.out)
    path = out / "levels.jsonl"
    records, order = {}, []
    if path.exists():
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            wid = r.get("workshop_id")
            if wid not in records:
                order.append(wid)
            records[wid] = r
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for wid in sorted(order, key=str):
            f.write(json.dumps(records[wid]) + "\n")
    tmp.replace(path)
    print(f"finalized {len(records)} levels; {path}")


def cmd_pipeline(a):
    """Download while converting and scoring completed maps, then finalize levels.jsonl."""
    out = Path(a.out)
    ns = argparse.Namespace(out=a.out, glados=a.glados, budget=a.budget, p2c=a.p2c,
                            incremental=True)
    passes = [["download", "--workers", str(a.workers)]]
    if a.max:
        passes[0] += ["--max", str(a.max)]
    if a.retry_errors:
        passes.append(["download", "--retry-errors", "--workers", str(a.workers)])
        if a.max:
            passes[-1] += ["--max", str(a.max)]
    for args in passes:
        print(f"[{time.strftime('%H:%M:%S')}] {' '.join(args)}", file=sys.stderr)
        proc = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), *args],
                                cwd=ROOT)
        while proc.poll() is None:
            try:
                n = cmd_build(ns)
            except Exception as e:  # a bad batch must not stop the download
                print(f"build failed: {e}", file=sys.stderr)
                n = 0
            print(f"[{time.strftime('%H:%M:%S')}] built {n} new maps", file=sys.stderr)
            time.sleep(a.interval)
        try:
            cmd_build(ns)  # catch up on whatever finished while the last build ran
        except Exception as e:
            print(f"final build failed: {e}", file=sys.stderr)
        if proc.returncode:
            print(f"download exited {proc.returncode}", file=sys.stderr)
    cmd_finalize(argparse.Namespace(out=a.out))


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
    p = sub.add_parser("ratings", help="star ratings of the listed puzzles, onto the level files")
    p.add_argument("--delay", type=float, default=0.5, help="seconds between pages")
    p.add_argument("--max-age", type=float, default=7, help="re-read pages read more than this many days ago")
    p.add_argument("--annotate-only", action="store_true", help="fetch nothing, only update the level files")
    p.add_argument("--files", nargs="+", help=f"level files to annotate (default {' '.join(RATED_FILES)})")
    p = sub.add_parser("download", help="fetch the listed puzzles' slim maps")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max", type=int, default=0)
    p.add_argument("--retry-errors", action="store_true")
    p = sub.add_parser("build", help="convert and score the downloaded maps")
    p.add_argument("--glados", default=str(ROOT / "target" / "release" / ("glados.exe" if sys.platform == "win32" else "glados")))
    p.add_argument("--budget", type=int, default=50_000)
    p.add_argument("--p2c", help="also write the solvable levels as .p2c files here")
    p.add_argument("--incremental", action="store_true",
                   help="only maps downloaded since the last build; append to report/levels")
    p = sub.add_parser("finalize", help="merge an incrementally built levels.jsonl")
    p = sub.add_parser("pipeline", help="download while converting and scoring completed maps")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max", type=int, default=0, help="stop each download pass after this many")
    p.add_argument("--interval", type=float, default=120.0,
                   help="seconds between incremental builds while download runs")
    p.add_argument("--retry-errors", action="store_true", help="run a retry pass after the first")
    p.add_argument("--glados", default=str(ROOT / "target" / "release" / ("glados.exe" if sys.platform == "win32" else "glados")))
    p.add_argument("--budget", type=int, default=50_000)
    p.add_argument("--p2c", help="also write the solvable levels as .p2c files here")
    a = ap.parse_args()
    {"list": cmd_list, "ratings": cmd_ratings, "download": cmd_download, "build": cmd_build,
     "finalize": cmd_finalize, "pipeline": cmd_pipeline}[a.cmd](a)


if __name__ == "__main__":
    main()
