# Level tooling reference

Implementation details and recorded game checks. Dated measurements describe their stated
fixtures and samples, rather than guarantees for all levels. See the module headers and
[solver gaps](solver-gaps-2026-10.md) for modelling limits.

## Format facts (confirmed on the real files)
- Interior = cells `0..ChamberSize` per axis. Voxel layers cover `size + 1` per axis. The extra plane
  holds face bits for the outer walls.
- Rows are `"f" + one 0/1 char per x`, written Y-descending within each `Z` block.
- `PortalN` bit at voxel v means the face between `v - e_N` and v is white.
- `VoxelPos` is the open cell an item occupies. The Source "up" vector of `Angles` is its surface
  normal. Every door, button, dropper and light panel in both real files passes this check.
- Glass/grating and fizzlers/laser fields are an anchor item plus `*_EXTENT` items joined by
  `CONNECTION_BARRIER_ANCHOR_TO_EXTENT`. Glass: local -x = panel normal, local +z = away from
  the surface it hangs on; the panel is the normal-side face of each cell in the rectangle the
  extents span. Fields: local +z = normal (away from the mounting wall), local +y = the axis
  the two extents lie along, `LocalPos` = 0.375 × (normal × that axis). In the style map a
  fizzler runs wall to wall right under a glass panel in the same plane, and the builder's
  output for both matches the editor's item for item (`tests/tier1.rs`).

## Solver
`solve` runs two rule sets and reports **solvable** (conservative rules find a plan, printed),
**unsolvable** (liberal rules find none), or **uncertain** (only liberal rules solve it, plan
printed; or a budget ran out). It rejects levels that are invalid or use items it doesn't model
(unsupported non-Puzzle-Maker items).

Search state = (region, portal face class per color, cubes as (kind, home, location), latched
pedestals, turrets knocked over, painted faces). Actions: drop, cube-step, walk through a field,
fire, enter portal, pick up, put down (loose or on an accepting button), toss off a ledge, throw
through a portal, press a latching pedestal, press a timed pedestal and dash on foot before it
runs out, knock a turret over, bounce off repulsion gel, leap off propulsion gel. Reaching the
exit region with the door open ends the plan (the plan's worked example is 5, not 6, because
walking out isn't counted).

The search packs each state into a few 32-bit words (region, portals, pedestals, turrets, cube
locations, 2 bits of paint per face), stores it once in an open-addressing table, and builds
successors in reused buffers, from per-region lists of edges, targets and buttons. On the v3
data that labels levels about 15 times faster than hashing and cloning `State`s did, with
identical results (checked state by state against the old code).

- **Turrets** shoot within a cone around the way they look, which is their `Facing` + 180°
  (the Puzzle Maker's turret instance turns the npc round; checked in the game). Regions split where the set of turrets that can shoot
  changes, so walking out of a turret's sight is a move. Conservative rules never go where a
  live turret can shoot; liberal ones run through but do nothing else there. A turret is knocked
  over from its own cell or one beside it, by a cube landing on it, or by a portal opening in
  the floor under it while the other is open (it falls through).
- **Gel**: a pouring dropper paints where its stream lands after every action. With a portal
  on the floor under the stream and the other on a ceiling, the gel lands under the ceiling
  portal instead. Liberal rules also spray gel out of a wall portal into the room; that is
  modelled only where the spray lands on a turret's floor, so a portal there can drop it.
  Splats without a dropper are painted from the start; water washes paint off.
  Droppers can be wired to buttons (they pour while the buttons are down). Conversion gel makes
  faces portalable; repulsion floors give bounces; a run of propulsion floor gives a leap.

| | Conservative | Liberal |
|---|---|---|
| climb a 1-voxel ledge on a cube | no (unmeasured) | yes |
| fire while carrying | no | yes |
| shot grazing a voxel edge | blocked | allowed if one way round is clear |
| shot targets | face centre | centre + 4 inset points |
| fizzler that can ever be on | blocks shots | only one that is always on |
| walking speed (timed dashes) | 140 u/s | 190 u/s |
| turret sight | 60° either side, 16 voxels, grating doesn't stop bullets | 45°, 8 voxels, grating does |
| in a live turret's sight | never | run through, nothing else |
| paint spreads (Manhattan) | 1 voxel | 2 voxels |
| repulsion bounce | up 1–2 voxels, 1 sideways | up 1–4, 2 sideways |
| propulsion leap | 2 painted cells of run-up, gap up to 3 | 1 cell, gap up to 5 |

| Chamber | Verdict |
|---|---|
| flat_room, glass_bridge | 0 |
| goo_crossing, grating_window | 3: portal over the goo / through the grating |
| pit_button | 2: toss the cube into the pit (jumping in is a trap) |
| pillar_button | 4: throw the cube through a portal onto the button |
| worked_example | 5: carry the cube through a portal to the button |
| fizzler_cube | 4: press the pedestal that turns the fizzler off, then carry the cube through |
| timed_field | 1: press the 6 s pedestal and run past the laser field (4 s: uncertain, 3 s: unsolvable) |
| sphere_button | 2: the sphere, not the cube, on the sphere button |
| step_ledge_cube | uncertain: only a cube-step solves it |
| goo_pit, step_ledge, worked_example_black, glass_window | unsolvable |
| turret_corridor | 2: knock the turret over, walk out of its old sight |
| bounce_gel, speed_gel | 1: bounce up the ledge / leap the goo |
| conversion_gel | 3: portals on the converted floor and the ceiling over the ledge, walk in |
| redirect_gel | 3: portals under the stream and over the ledge, then bounce |
| washed_gel | unsolvable: water washes the repulsion gel off |

## Metrics and score
`solve` also prints, and `evolve` optimises, metrics from the full state space and ablations:
plan length and portal trips, number of shortest plans, dead-end (softlock) states, cubes that
aren't needed, face classes that are (painting one black makes the level unsolvable), and the
share of random players that stumble out. Also:
- **Components:** floor buttons and pedestals the level can't be solved without (switch one
  off and re-solve), and fizzlers/laser fields that shape the plan (switched off, the plan
  gets shorter or easier, or a necessary button or pedestal drives them). Only items the plan
  uses are tested; the rest can't be needed. Switching a face class or a pedestal off only
  takes actions away, so those are read off the explored graph rather than solved again; a
  button the exit can't open without needs no search, and fields and turrets only need plans
  as short as the original.
- **Mechanics:** which of portal trip, cube through a portal, cube on a button, firing while
  standing on a button, tossing off a ledge, latching pedestal, timed dash, fizzler and laser
  field the plan uses.
- **Chain depth:** replay the plan leaving out one step; later steps that can no longer be done
  need it. The longest weighted chain of such needs (walking and cube chores 0.5, the rest 1):
  serial puzzles score higher than piles of independent chores.
- **Distance:** player travel between region centres (portal jumps included), how far cubes end
  up from home, and backtracking (moving away from the exit), in voxels.

`Metrics::score` folds them into one number: log of weighted effort (portal shots and timed
dashes 1, trips 2, latching presses 0, other actions 0.5) + random players failing + necessary
classes + per kind log(1 + necessary items) + mechanics used + log chain depth + log distances
− unnecessary cubes − softlock share. The first version rewarded plan length linearly and bred
30-action staircase slogs; the second (effort, surfaces) bred small portal mazes. Weights are
still a guess pending playtests.

## Generator and search
`generator::Genome`: room size (6–14 wide at random, mutations 4–16; 3–8 voxels tall over the
floor), two doors on the outer walls, and a list of features: platforms and pits, goo, walls
from floor to ceiling with an optional opening, balconies (a one-voxel slab with space above and
below), white patches, glass/grating panels, fizzlers/laser fields, ceiling cube droppers of any
kind, turrets facing any of 8 ways, gel droppers of any kind, floor buttons of any kind and
pedestals, wired to the exit, a field or a gel dropper (which then pours only while they're
down). Feature counts
scale with floor area. Items stand on the highest floor of their column. The search scores
levels with the conservative rules, so it can't win by leaning on unmeasured moves. `evolve`
keeps an elitist population of solved levels, breeds children by mutation, drops behavioural
duplicates (same size and metrics) so it doesn't fill with clones, and gives each set of
mechanics at most an eighth of the places. Big random rooms are rarely solvable (about 70%
of those over 100 cells of floor are unsolvable), so the search drifts back to small ones unless
`--min-area` holds it to big rooms. Results depend only on the seed.

What it finds today: portal-hopping mazes with many necessary surfaces, 10–25 actions. Solvable
under the model and not trivial, but mazy rather than elegant. That's the plan's "hard ≠ fun"
risk; next steps are a style prior from real levels and in-game checks of the winners.

## Assumptions to verify in-game
- Turrets: the cone, range, what stops bullets, and that knocking one over from beside it,
  dropping a cube on it or opening a portal under it puts it out for good. A player in its
  sight is assumed dead (conservative) or able to run past (liberal).
- Gel: how far paint spreads, bounce heights and reach, leap run-up and distance; gel through
  a portal lands from a ceiling portal, and (liberal only) out of a wall portal up to two cells
  into the room (one turret-floor case confirmed in the game, 2026-10-22). Generated gel copies
  the style map's one gel (splat on the floor, dropper on the ceiling above,
  `CONNECTION_PAINT_DROPPER`); the export type is a guess (set to the paint type), and splats
  without a dropper are unchecked.
- Goo floods every open cell reachable sideways or downward from a goo item. (The editor itself
  only takes a goo item on a floor; whether the game floods beyond the items is unchecked.)
- A 1-voxel ledge needs a cube to climb; 2+ voxels can't be climbed; any drop is survivable.
- Generated files open in the editor and compile. **Confirmed for one level (2026-10-09):** a
  model sample (fizzler, cube, button, 2 portal trips, 8-action plan) opened in the Puzzle
  Maker, played, and was solved. The editor's puzzles live in
  `steamapps/common/Portal 2/portal2/puzzles/<SteamID64>/`, not in Steam's userdata.
- Line of sight: from the centre of a standable cell (64u eye) to the centre of the face.
  Grazing an edge counts as blocked.
- Wall and floor portals can be walked into from the standable cell in front; ceiling portals
  can't. Portal momentum and faith plates are modelled in `src/flings.rs`; see its assumptions.
- Floor buttons are held only by cubes they accept (weighted: any; cube: not the sphere;
  sphere: only the sphere). Every receiver is the AND of its inputs; the exit is inverted by
  `Start Open`, a field by `Start Enabled`.
- Pedestal timer: 3–30 s, infinity stored as a value below 3 (writes 0). Timed pedestals allow
  only walking and dropping while the clock runs.
- Fizzlers clear portals and send a held or falling cube back to where it started; laser fields
  stop the player only. Glass stops shots; grating, laser fields don't.
- Barrier and field layouts (see format facts) rest on one real file. **Panels confirmed
  (2026-10-10):** glass and grating 1×1, tall, wide, 3×2, a full wall, and a floor bridge all
  built in the editor. Which placements the editor refuses is now measured in the game (next
  section). The generator slides a field back to the wall behind it and leaves out a field,
  button, pedestal or dropper that would sit in its sweep.

## Editor placement rules (checked in the game)
The Puzzle Maker marks misplaced items red and won't build. `src/editor.rs` predicts which items
it flags, by the names and positions the game logs, and `mount_problems` (so `score`,
`dataset` and relabelling) calls such levels invalid. `tantalos-p2c editor FILE|DIR` prints them.

Fitted on 2,403 generated levels checked in Portal 2 (2026-10-10): the new rules agree with the
game on 98.0% of levels (6 flagged that the game accepts, 43 missed), against 79.3% for the
sweep rules they replace. Each rule matched the game on every item it flagged, except where noted:
- Doors are sunk 3×3 cells wide and 4 deep into their wall (`EmbeddedVoxels` in
  `portal2_dlc2/scripts/editoritems.txt`), and a dropper into its ceiling cell; two footprints
  may not overlap. Two doors on one wall need 3 cells between them both ways.
- Goo must have solid under it, and only cubes may share a cell with it (or anchor on a panel).
- Glass/grating on a door's wall face, on any face of a dropper's cell, or on the face a
  pedestal's column stands against (its local −x side) clashes with both.
- A field collides with what is in the cells it sweeps (not the cells just across its plane):
  floor buttons and droppers always, doors unless the field runs across their front, pedestals
  when it is on their column side. Cubes never.
- Glass across a field's path clashes, whatever the field. Parallel fields through the same cells,
  and a fizzler crossing a laser field, clash (76% of items flagged).
- Left to the in-game check (the game decides by finer geometry; flagged 20–55% of the time):
  grating across a field, a panel on the field's own face, other crossing fields, fields side by
  side, a door beside a field's emitters, a field beside a pedestal.

The token decoder places one goo item per group of goo cells joined sideways at one height,
on a floor and not in another item's cell; it used to put goo at the top of every goo run,
which the editor rejects (all goo flags in the sample, and v2-4's).

**In-game check** (`train/portal2.py`, `train/ingame.py`): starts Portal 2 with `-insecure`
(VAC off for that session) and drives it over its TCP console (`-netconport`, random
`-netconpassword`; a `-hijack` process per command was slower, and popped up "did not accept
the request to load a savegame" when the game was busy). It opens the editor from the console,
and patches one byte in
`client.dll` in memory so `puzzlemaker_load_dev <path>` calls the loader the My Chambers list
uses (the retail command is a stub), plus the editor's active flag that
`puzzlemaker_export` checks. Export then logs `INVALID ITEM:  <name> ( x , y , z )` for every
misplaced item and refuses, or writes a VMF. The editor only marks misplaced items as it
draws, so each export waits 0.1 s after its load (sooner, and every level passes as valid), and
a level it must reject is checked first and every 50 levels, restarting the game and
rechecking if it passes. Exports slow down over a session, so the game restarts every 100
levels (and after a crash; the level is recorded as `crash`): about 0.4 s per level.
It refuses to touch a game that is already running or wasn't started with `-insecure`, refuses
a `client.dll` other than the Jan 2025 build, and puts back the editor autosaves it overwrites.
`retrain.sh` runs `train/ingame.py` after evaluating: it checks every sample our checker passes
and a fifth of the ones it rejects, appends `ingame/report.md` to the report, adds
disagreements to `data/ingame/disagreements.jsonl` (for fitting rules with
`tools/ingame/rules.py`) and game-invalid levels to `data/ingame/invalid.jsonl`, which later
runs train on as invalid without relabelling. If the game can't be used it logs that and the
run carries on.

