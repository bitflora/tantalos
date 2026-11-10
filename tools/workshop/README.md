# Workshop puzzles as training levels

Portal 2's Workshop holds tens of thousands of Puzzle Maker chambers, but they're published
compiled (`.bsp`); the `.p2c` stays on the author's machine. These tools recover the puzzle from
the compiled map, as tantalos tokens, and label it with the solver like any generated level.

```bash
python tools/workshop/fetch.py list --pages 100   # out/workshop/items.jsonl (30 puzzles a page)
python tools/workshop/fetch.py download           # out/workshop/bsp/<id>.bsp.zst (zstd), fetched.jsonl
python tools/workshop/fetch.py download --retry-errors
python tools/workshop/fetch.py build              # out/workshop/levels.jsonl, report.jsonl
python tools/workshop/check.py                    # converter vs. known puzzles (needs release tantalos)
```

`build` converts every slim map first and scores them at the end. To overlap that CPU work
with the network, use `pipeline`: it starts `download` (plus a `--retry-errors` pass with
`--retry-errors`) as a child process and, every `--interval` seconds (default 120), converts and
scores only the maps whose `fetched.jsonl` record says `ok` (`build --incremental`). It appends
to `report.jsonl` / `levels.jsonl` and dedupes `levels.jsonl` at the end. A build interrupted
mid-download resumes where it left off, because `report.jsonl` is the set of maps already seen.

```bash
python tools/workshop/fetch.py pipeline --workers 8 --retry-errors
```

`levels.jsonl` has the fields `tantalos-data dataset` writes (`tokens`, `verdict`, `min_len`, `metrics`),
plus `workshop_id`, `title`, `subscribers` (lifetime unique), `favorited` and, once `ratings` has
run, the Workshop page's star rating: `stars` (1-5; 0 = "not enough ratings yet") and `votes` (its
"N ratings"). Training reads the
id as `<src:ws>` and the subscribers as `<tier:top>` (≥ 100) or `<tier:mid>` (train/data.py).
`build --p2c DIR` also writes the solvable ones as `.p2c`.

```bash
python tools/workshop/fetch.py ratings                 # out/workshop/ratings.jsonl, then annotate
python tools/workshop/fetch.py ratings --annotate-only # only copy ratings onto the level files
```

`ratings` reads the star ratings without an API key: each browse page embeds its 30 items'
`star_rating` and `total_votes` (the same numbers as the item page). It re-reads every
(sort, page) `items.jsonl` was listed from (about 2,500 pages, an hour or so; pages read in the
last `--max-age` days are skipped, so an interrupted run resumes). Converted levels that have
moved off those pages are then looked up on their own item page. Those pages get rate-limited (429)
sooner, and a 429 makes the request wait a minute. Last, it sets `stars`/`votes` in place in
`out/workshop/levels.jsonl` and `data/ws5/{train,test}.jsonl` (`--files` for others); `build` and
`ws3.py` carry them on from there. `list` also saves the ratings of every page it reads.

## Triaging the downloaded levels

`tantalos-workbench` (repo root, needs the release `tantalos-score.exe` for staging) serves a local browser app
for `out/workshop/levels.jsonl` joined with `data/ws5/{train,test}.jsonl`, the ratings and item
metadata. It shows the Workshop stars/votes, subscribers, tags, the solver and dataset verdicts,
all metrics, conversion problems, the level tokens and a 2.5D drawing of the chamber. You can mark
a level as staged for manual playtesting, give it your own 1–5 stars, write notes, and exclude it
from the test data.

The annotations are in `train/workshop-annotations.jsonl` (server-side, one JSON object per
`workshop_id`). Agents can read or write them without the browser:

```bash
python tools/workshop/annotations.py --get 100091397
python tools/workshop/annotations.py --excluded
python tools/workshop/annotations.py --set 100091397 --rating 4 --staged --notes "worth playing"
```

`excluded` is honoured when the ws5 split is rebuilt (`ws3.py` moves the level to train) and when
`data/plans/test-ws.jsonl` is rebuilt (`train/goblin_data.py` skips it). It does not remove the level
from training.

## Mirrored and turned copies for training

```bash
python tools/workshop/augment.py               # data/ws5/train.jsonl -> data/v5b/workshop-aug-x2.jsonl
target/release/tantalos-data.exe augment --check FILE   # also label each version against its level
```

`data/v5b/workshop-x20.jsonl` repeats each ws5 train level 20 times and models memorize
them (2026-10-25: Workshop test loss 1.59 against 0.24 on trained levels). `augment.py` writes each
level's 8 versions instead (mirror in x, then 0-3 quarter turns about z; `src/augment.rs`), twice
each: 258k lines, 129k distinct sequences. Records keep their labels and get `variant` (`r0` is
the level itself). A version is left out if it isn't exactly the level moved (solid, goo, white
faces, glass, fields, items: `augment::moved_level`, never failed on ws5), adds a mounting problem
(180: a mirrored field anchored at a wall end that isn't there), changes our editor flags (167:
the decoder's own pedestal-side and goo-cell choices), doesn't read back (14: wires renumbered
past 32 in levels with over 32 panels) or is over 969 tokens (143). Since 2026-10-26 it is
`data/v5b/levels-workshop-aug-x2.jsonl`, in run.py's default `levels-*.jsonl`, and the x20 file is
`data/v5b/workshop-x20.jsonl`, outside it (name it with `--data` to train on it again). ws5 test is
never augmented. Since 2026-11-06 it is built from `data/ws6/train.jsonl` (42,819 levels: 339k
versions, 679k lines; 161 bunny-hop/glitch levels skipped); the ws5-based file is
`data/v5b/workshop-ws5-aug-x2.jsonl`, and ws6 test is never augmented either.

`--check` on 600 ws5 levels: 99.1% of versions got the original's verdict, plan length and
editor flags at first; the rest showed the solver depending on orientation (bridge routes,
first-wins device claims, turret sight through voxel corners, an entry door onto a faith plate).
Fixed 2026-10-25 (docs/solver-gaps-2026-10.md, "Orientation"): now 2 of 4,180 versions differ,
one level at the edge of the 50k budget.

## How it works

- **Listing**: the public browse pages (`toprated` by default; `--sort mostrecent trend ...`), Singleplayer, minus
  co-op and custom-visuals/story maps (every Puzzle Maker item has tokens now). Steam stops
  paging near page 1667 per sort. Metadata from the
  keyless `ISteamRemoteStorage/GetPublishedFileDetails`; Puzzle Maker uploads are `mymaps/<n>.bsp`.
- **Downloading** uses HTTP ranges: header, then the entity lump (~0.2 MB). If instance names rule
  the map out, stop; else fetch the 11 other lumps `convert.py` reads (~0.6 MB of a 5–40 MB map)
  and save a slim BSP. Steam's CDN sometimes returns wrong bytes with a 206 (zeros or junk, even
  consistently while an edge is filling its cache), so each range must come back identical twice
  and the slim map must pass `Bsp.check()`. Empty 206s are retried immediately, and a map that
  still fails is logged rather than retried for minutes; the next `download --retry-errors` pass
  picks it up. The CDN throttles custom user agents with empty 206s, so requests use the Steam
  client's user agent (`fetch.py` `UA`) while still keeping the request rate low.
- **Converting** (`convert.py`), from what the Puzzle Maker compiler leaves in the map:
  - cell v spans world [128v, 128v+128); open cells are flooded from the doors and items through
    empty BSP leaves, never out through the door corridors;
  - white faces are wall materials named `white`; goo is water brushes; glass/grating are thin
    window/grate brushes inside the face's cell, merged into rectangles;
  - goo is kept to shapes the editor can build: its goo item sits on a floor and floods
    sideways and down (checked in the game), so a pool two or more deep whose surface has no
    cell over a floor (15% of Workshop maps) gets solid cells under its surface, which plays
    the same. A map with an item under such a surface, or an item the editor won't place in
    goo standing in it (turrets, buttons, doors, fields, lasers, plates...), is unsupported;
  - item instances are named `<kind><p2c index>` (`button23-proxy`, `barrierhazard8_brush`);
    logic entities keep the instance's angles; an item's cell is floor((anchor + up) / 128);
  - wires are the outputs of each button's `func_instance_io_proxy`;
  - cube type: `counter_cube_type` startvalue, where 6 is Franken (its `case_cube_type` keeps
    only the monster-box spawner; 4 never occurs), and a loose `cube<N>-cube` that is a
    `prop_monster_box` is Franken too (until 2026-10-25 these ~3,500 maps were rejected); pedestal: `prop_button` `Delay` (99999999999 =
    infinite; `istimer` is only the ticking sound, off for buttons that drive only droppers);
    two pedestal buttons in one cell (a map can place them inside the editor's 128-unit voxel)
    can't both be placed, so the later one moves to the nearest free neighbouring cell with a
    floor under it;
    gel: `info_paint_sprayer` `PaintType` (0 bounce, 2 speed, 3 portal, 4 water) and `start_active`;
    a dropper in bomb mode has a `prop_paint_bomb` (its `PaintType`) instead, dropping one bomb
    after another while its `-branch_toggle` is true (`InitialValue` = on at the start): it is
    written as a stream, so the solver paints a stream's spot where a bomb splats wider;
    fields: their trigger brushes' extent, `trigger_hurt` = laser field;
  - turrets: the `.p2c` Facing is the npc's yaw + 180 (kept, so imports match generated levels);
  - lasers, bridges, funnels: the prop's forward (into the room) and its origin 16 units into the
    wall; a bridge's walkable side is its up; `_center` models are centred; catchers and relays
    wire like buttons, and a funnel's proxy inputs 3/4 are its reversing wires;
  - faith plates: the arm's cell and the trigger_catapult's launch target (an info_target, or a
    helper trigger throwing straight up, whose height comes from its speed);
  - panels, stairs: the arms' cell, the instance's angles (local +x) and the branch's
    InitialValue; an angled panel's whiteness from its `panel_top` brush, a flip panel's from
    its door's outward face; pistons from their func_movelinear and level counters; rails from
    the track's movedir/movedistance; loose cubes from `cube<N>` instances' CubeType. The voxel
    behind each such item is forced solid (it is in the editor; compiled, it's a hollow).
- **Labelling**: `tantalos-score score --editor-rules off`. A published map already passed the real
  editor, and our model of its rules flags some of them (a fizzler centred in a doorway, fields
  crossing), so those flags are skipped.

## Checked

`check.py` converts compiles of two known `.p2c` files (`fixtures/workshop/`, slimmed and gzipped)
and requires the solver to give both the same verdict, shortest plan and every metric. It does.
The token differences left are faces the map can't show and play can't use: white under a cube
dropper, glass over a wall, an empty sealed pocket.

## What's left out

Rejected with a reason in `fetched.jsonl` / `report.jsonl`:
- **BEEmod maps with BEE's own items** (most recent uploads). Since 2026-10-25 a BEE map of
  vanilla items converts: its short instance names (`bt`, `cd`, `le`, `fp`, ...) map to
  vanilla kinds by the entities each holds (`convert.py` `BEE_ITEMS`, `bee_kind`; `ap` can be
  BEE's autoportal, `fp` a plate or a flip panel), its decoration is dropped (`BEE_DECOR`), and
  where newer BEE builds leave out an item's logic its rotation comes from its brushes or
  props, a field's from its triggers and emitters, a door's from the entity behind it, a
  dropper's cell from the ceiling over its model. Still rejected: BEE's own items (`tr`
  trigger, `dr` door, `lg` logic gate, `chdr`, `pgun`, autoportals, custom fizzlers, its
  pistons), 1-second pedestal timers, BEE gel droppers (gel picked at run time), custom
  entrances. Of the ~20k BEE maps `download` ruled out, only ~830 have nothing but vanilla
  item names; `download --rescreen` fetches those. A sample of 150 of them: 34 converted, 13%
  of those `invalid` with editor rules off (vanilla Workshop: 16%).
- Items the tokens can't express: pedestal and floor buttons off the floor (8,640 maps of
  2026-10-21's build, the largest gap: with those buttons left out, 8,355 of them convert),
  wall-mounted gel droppers and stairs, cube types past franken. (Turrets facing between multiples of 45° are
  rounded to the nearest.)
- Some converted puzzles come out `unsolvable`/`uncertain` because the solver doesn't model a
  trick they need (e.g. speed-gel jumps over goo), not because conversion failed.

The puzzles belong to their authors (Steam Subscriber Agreement): fine to train on privately; don't
redistribute the downloaded maps or the derived dataset.
