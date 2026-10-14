# Workshop puzzles as training levels

Portal 2's Workshop holds tens of thousands of Puzzle Maker chambers, but they're published
compiled (`.bsp`); the `.p2c` stays on the author's machine. These tools recover the puzzle from
the compiled map, as glados tokens, and label it with the solver like any generated level.

```bash
python tools/workshop/fetch.py list --pages 100   # out/workshop/items.jsonl (30 puzzles a page)
python tools/workshop/fetch.py download           # out/workshop/bsp/<id>.bsp, fetched.jsonl
python tools/workshop/fetch.py download --retry-errors
python tools/workshop/fetch.py build              # out/workshop/levels.jsonl, report.jsonl
python tools/workshop/check.py                    # converter vs. known puzzles (needs release glados)
```

`levels.jsonl` has the fields `glados dataset` writes (`tokens`, `verdict`, `min_len`, `metrics`),
plus `workshop_id`, `title`, `subscribers` (lifetime unique) and `favorited`. Training reads the
id as `<src:ws>` and the subscribers as `<tier:top>` (≥ 100) or `<tier:mid>` (train/data.py).
`build --p2c DIR` also writes the solvable ones as `.p2c`.

## How it works

- **Listing**: the public browse pages (`toprated` by default; `--sort mostrecent trend ...`), Singleplayer, minus
  co-op and custom-visuals/story maps (every Puzzle Maker item has tokens now). Steam stops
  paging near page 1667 per sort. Metadata from the
  keyless `ISteamRemoteStorage/GetPublishedFileDetails`; Puzzle Maker uploads are `mymaps/<n>.bsp`.
- **Downloading** uses HTTP ranges: header, then the entity lump (~0.2 MB). If instance names rule
  the map out, stop; else fetch the 11 other lumps `convert.py` reads (~0.6 MB of a 5–40 MB map)
  and save a slim BSP. Steam's CDN sometimes returns wrong bytes with a 206 (zeros or junk, even
  consistently while an edge is filling its cache), so each range must come back identical twice,
  the slim map must pass `Bsp.check()`, and failing maps are retried after a pause.
- **Converting** (`convert.py`), from what the Puzzle Maker compiler leaves in the map:
  - cell v spans world [128v, 128v+128); open cells are flooded from the doors and items through
    empty BSP leaves, never out through the door corridors;
  - white faces are wall materials named `white`; goo is water brushes; glass/grating are thin
    window/grate brushes inside the face's cell, merged into rectangles;
  - item instances are named `<kind><p2c index>` (`button23-proxy`, `barrierhazard8_brush`);
    logic entities keep the instance's angles; an item's cell is floor((anchor + up) / 128);
  - wires are the outputs of each button's `func_instance_io_proxy`;
  - cube type: `counter_cube_type` startvalue; pedestal: `prop_button` `istimer`/`Delay`;
    gel: `info_paint_sprayer` `PaintType` (0 bounce, 2 speed, 3 portal, 4 water) and `start_active`;
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
- **Labelling**: `glados score --editor-rules off`. A published map already passed the real
  editor, and our model of its rules flags some of them (a fizzler centred in a doorway, fields
  crossing), so those flags are skipped.

## Checked

`check.py` converts compiles of two known `.p2c` files (`fixtures/workshop/`, slimmed and gzipped)
and requires the solver to give both the same verdict, shortest plan and every metric. It does.
The token differences left are faces the map can't show and play can't use: white under a cube
dropper, glass over a wall, an empty sealed pocket.

## What's left out

Rejected with a reason in `fetched.jsonl` / `report.jsonl`:
- **BEEmod maps** (instance names like `bt`, `cd`, `ip`, `orl`): most recent uploads. BEE2's
  compiler names things differently and adds its own items; not attempted.
- Items the tokens can't express: pedestal and floor buttons off the floor, wall-mounted gel
  droppers and stairs, cube types past franken. (Turrets facing between multiples of 45° are
  rounded to the nearest.)
- Some converted puzzles come out `unsolvable`/`uncertain` because the solver doesn't model a
  trick they need (e.g. speed-gel jumps over goo), not because conversion failed.

The puzzles belong to their authors (Steam Subscriber Agreement): fine to train on privately; don't
redistribute the downloaded maps or the derived dataset.
