# tantalos

Tooling for generating and solving Portal 2 Puzzle Maker levels. See [plan.md](plan.md).

**Slice 1:** `.p2c` read/write, level model, walkable regions, ASCII maps.
**Slice 2:** Tier-0 solver: portal line of sight, face classes, cubes/floor buttons/exit logic,
BFS for the shortest plan.
**Slice 3:** level metrics, random chamber genomes, and an evolutionary search that writes the
best levels it finds.
**Slice 4:** Tier 1 (glass, grating, fizzlers, laser fields, pedestal buttons with timers, cube
and button types) and two solver modes that bracket the truth.
**Slice 5:** levels as token sequences, a labelled dataset, scoring of model output, and a
small GPT trained from scratch on them (`train/`).
**Slice 6:** turrets and gel (repulsion, propulsion, conversion, water), in the solver,
tokens, metrics and generator.
**Slice 7:** every other Puzzle Maker item: laser emitters, catchers and relays with
reflection cubes (`lasers`), faith plates and flings through portals (`flings`), light
bridges, excursion funnels, angled/glass/flip panels, stairs, piston and rail platforms
(`dynamics`: device state bits, conditional regions and moves), loose cubes and droppers that
wait for their inputs; in the solver, tokens, metrics, generator, routes, `train/play.py`
and the Workshop converter. The modelling assumptions are listed in each module's header;
the ones most worth checking in the game are in "Unverified" below.

The command-line tools are separate binaries (`src/bin/`; each prints its usage when run
without arguments, and they share `src/cli.rs`):

| binary | commands |
|---|---|
| `tantalos-score` | `score`, `check`, `route`: solving and checking, what training, evaluation and play call |
| `tantalos-data` | `dataset`, `mutate`: making training data |
| `tantalos-p2c` | `info`, `solve`, `tokens`, `vocab`, `editor`, `roundtrip`, `fixtures`: one-off tools for .p2c files |
| `tantalos-view` | the browser app for saved solutions |
| `tantalos-workbench` | the browser app for triaging Workshop levels |
| `tantalos-evolve` | evolutionary search for interesting levels |
| `reverse` | the reverse generator (solvable by construction) |

```bash
cargo test
cargo run --bin tantalos-p2c -- info fixtures/synthetic/worked_example.p2c
cargo run --bin tantalos-p2c -- solve fixtures/synthetic/*.p2c
cargo run --release --bin tantalos-evolve -- --seed 1 --gens 40   # best levels + report.md in out/evolve-1
cargo run --bin tantalos-p2c -- roundtrip fixtures/real/*.p2c
cargo run --bin tantalos-p2c -- fixtures            # regenerate fixtures/synthetic from src/fixtures.rs
cargo run --bin tantalos-p2c -- tokens fixtures/synthetic/worked_example.p2c
```

## Training a level model
```bash
cargo build --release
./target/release/tantalos-data dataset --seed 1 --count 200000 --out data/levels.jsonl
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv/Scripts/pip install -r train/requirements.txt
.venv/Scripts/python train/train.py --data data/levels.jsonl --out runs/first
.venv/Scripts/python train/sample.py runs/first/best.pt --n 64 --verdict solvable --len 5-7 --p2c
.venv/Scripts/python train/evaluate.py runs/first/best.pt --data data/levels.jsonl   # runs/first/eval/report.md
bash train/night.sh 22:45 07:40   # generate until 22:45, train until 07:40, then evaluate
RELABEL=1 bash train/retrain.sh runs/v2 "+3 hours"   # re-score data/v2 with the current solver, train, evaluate
AWAY_MINUTES=15 bash train/retrain.sh runs/v4 "+1 hour"   # train, evaluate, check in the game, train away from rejects, repeat checks
.venv/Scripts/python train/ingame.py runs/v2/eval      # check samples in Portal 2's editor (also run by retrain.sh)
```
Every `retrain.sh` run checks the samples our solver passes in Portal 2's editor
(`train/ingame.py`; the game must not already be running) and adds the levels the game rejects
to `data/ingame/invalid.jsonl`. It then trains away from all of them for `AWAY_MINUTES`: they
are trained on as invalid, repeated `AWAY_REPEAT` times. That model is evaluated and checked
in the game too, and is the run's final model (`<out>/away/best.pt`) unless its solvable share
fell more than 3 points. An unlikelihood loss on the tokens that placed each rejected item
(`AWAY_LOSS=1`, `train.py --away`) is off: on the first overnight run it taught the model to
stop writing real coordinates (0% of its levels parsed), because most rejections are two
items colliding in normal places, so it pushed normal placements down.
Every training run has a row in `train/runs.csv`: its settings, losses, evaluation and in-game
results, and notes. `train.py`, `evaluate.py` and `ingame.py` fill it in as they finish;
`train.py --note "..."` or `train/record.py RUN --note "..."` sets the notes.
`train.py --until 07:40` trains to a clock time, with the learning-rate schedule spread over the
time left, and `--init CKPT` starts from another run's weights. While training it asks Windows
not to sleep; a closed lid still sleeps. On this laptop the GPU throttles at about 87 °C, and
running the generator at the same time slows both, so `night.sh` runs them in turn.
- **Tokens** (`src/tokens.rs`): size, then rows of columns as solid/open/goo runs, then white
  surfaces as boxes, doors, panels, fields, droppers, buttons and pedestals with their wires.
  Repeats are free: `=` repeats the previous column. Columns left out of a row, and rows left
  out at the end, repeat the last one, so the model never has to count to the room's width or
  depth. That was the main way early samples failed to parse. The worked example is 56 tokens;
  generated rooms average about 105 and stay under about 310. Round trip: `encode(decode(encode(p))) == encode(p)` and the
  solver's verdict is unchanged (tests on every fixture and hundreds of genomes). It keeps what
  play depends on, not the file's bytes. BEEmod and other non-Puzzle-Maker items are refused.
  Turrets and gel came later, then the other items (lasers, plates, bridges, funnels, panels,
  stairs, pistons, rails, loose cubes); their words are at the end of the vocabulary so earlier
  token ids are unchanged. Their condition groups (`train/data.py` `LATER_FEATURES`,
  `LATER_MECH_GROUPS`) come after the old ones, which shifts level-token ids, so
  `train.py --init` matches embeddings by token name: a checkpoint from before them warm-starts
  with its 274 tokens kept and the new ones fresh (and `--block` may grow).
- **Dataset** (`tantalos-data dataset`): JSONL, one level per line with its tokens, verdict, plan length
  and metrics, plus `vocab.txt` beside it. Jobs are 30% fresh random genomes, 70% mutations of a
  pool of solvable ones that drifts toward higher `Metrics::score`. Over a 20k run the share of
  solvable levels needing 0 actions falls from 34% to 10%, and portal use rises from 5% to 33%.
  About 250 levels/s on 12 threads. Seed-determined, independent of thread count.
  With turrets, gel and bigger rooms: about 13 levels/s, falling to 3 as the pool drifts to big
  state spaces; the faster search (below) and pipelined jobs brought the start of a run to about
  100 levels/s (12 threads, sharing the laptop with that old generator and a training run). Jobs
  used to run in batches of 64 that waited for their slowest level, which kept the threads less
  than half busy; now job `j` takes its parent from the pool as it stood up to 512 jobs earlier
  (64 at the start), and results are still taken in order. 40% of mutations now take their parent
  from a reserve of 100 levels per mechanic (picked evenly over the mechanics seen), so rare ones
  keep breeding: in 6k levels, cube-through-portal rose from 2% to 9% of solvable levels,
  bounces from 2% to 13%, gel through a portal from 0 to 2%, and 0-action plans fell from 43% to
  12%. Leaps stay near zero (orange gel rarely lands at the edge of a gap). Records carry
  `mechanic_bits`, chain depth and the other metrics for conditioning.
- **Scoring** (`tantalos-score score`): one token line in, one JSON line out (`unparsed`, `invalid`,
  `unsupported`, `unsolvable`, `uncertain`, `solvable` + metrics; `tokens`, the canonical form
  of anything that parsed, and `canonical` if the line was already exactly that). `--p2c DIR`
  writes the solvable ones for playtesting.
- **Model** (`train/`): GPT-2-style decoder, 6 layers × 384 wide (~11M parameters), block 512,
  trained from scratch. A sequence is `<bos> conditions <sep> level <eos>`. Conditions: verdict,
  plan-length bucket, portal trips, cubes, and yes/no for fizzler, laser field, glass, grating,
  pedestal and goo. Each has a fixed slot holding its value or `?` (not given, or doesn't
  apply), so the level always starts at the same position. Each sequence hides each condition
  with a chance drawn from [0, 1], so sampling can fix any subset. The first run instead left
  hidden conditions out, so the level's position varied. Prompts with only `<verdict:solvable>`
  were then unlike anything in training, and the model dropped a size number: at step 5,000,
  5% parsed with that prompt and 64% with full condition sets. Every eval
  samples levels conditioned on `<verdict:solvable>` and scores them with `tantalos-score score`.
  Generation keeps a key/value cache, so each new token costs one position, not the whole level.
- **Evaluation** (`train/evaluate.py`): samples 256 levels asked to be solvable, 64 with no
  conditions, and 32 for every other condition value, scores them all, and writes `report.md`.
  It covers: *validity* (parse, canonical, verdicts, the commonest parse errors);
  *conditioning* (how often a level has what was asked for, against how often it has it
  unasked); *novelty* (exact copies of dataset levels, and Jaccard similarity on token
  4-grams to the nearest dataset level of the same size, with dataset levels against each other
  as the baseline, because the generator's mutations have close neighbours too); and *model vs
  generator* (solvable share, solvable levels per second, and the plan length, portal, cube,
  item and score distributions of their solvable levels). `--cpu` samples on the CPU, for use
  while the GPU is training: two jobs on a 4 GB card run out of memory and both crawl.
- **First result** (RTX 3050 Laptop, 4 GB): about 50k tokens/s, so roughly 10 steps/s at batch
  32. On a 20k-level set, 3,000 steps (about 8 minutes) brought validation loss to 0.56 (0.89 at step 500).
  Of 256 samples asked for a solvable level, 46% parsed and 19% were solvable (mean plan 2
  actions). Most remaining failures are miscounted positions (too many columns, numbers out of
  range). Next: the 200k set and longer runs, then step 5 (evaluation: novelty, conditioning
  accuracy).

## Fixtures
- `fixtures/real/`: two BEE2 style/skybox preview maps from
  [BEEmod/BEE2-items](https://github.com/BEEmod/BEE2-items/tree/master/icons/previews/p2c).
  These were the only public `.p2c` files findable on GitHub/Sourcegraph (2026-10-08).
  They are used to check round-trips and coordinate conventions. Keep them local; don't redistribute.
- `fixtures/synthetic/`: chambers built in code (`src/fixtures.rs`), including the plan's worked
  example. Item properties copy the real files, so they *should* open in the Puzzle Maker.
  **Not yet verified in-game.**

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

## Solver (Tiers 0 and 1, turrets and gel)
`solve` runs two rule sets and reports **solvable** (conservative rules find a plan, printed),
**unsolvable** (liberal rules find none), or **uncertain** (only liberal rules solve it, plan
printed; or a budget ran out). It rejects levels that are invalid or use items it doesn't model
(lasers, faith plates, bridges, funnels, ...).

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
  can't. Coming out, players and cubes drop straight down with no momentum (no flings yet).
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

## Playing a solution in the game
`tantalos-score route FILE` makes the conservative plan concrete (`src/route.rs`): for each step, the
cells to walk through, where to stand, the face each portal goes on and the point to aim at,
the cube to pick up and where it ends up, then a last walk out of the exit door. Choices the
plan leaves open go to the nearest cell on foot, avoiding cubes, buttons, pedestals, turrets,
gel and open floor portals when there is another way. It replays the plan with a concrete
state (player cell, portal faces, cube cells) and checks after every step that it abstracts to
the solver's state; `tests/route.rs` grounds every solvable fixture and about 130 generated
levels, and checks the walks are legal steps or drops and the shots clear.

`train/play.py LEVEL.p2c --launch` plays the route in Portal 2 with the player's own controls,
for watching: it loads the level into the editor (as the in-game check does), builds it with
`puzzlemaker_compile_and_preview`, walks out of the elevator, and plays each step with `setang`
(a perfect mouse) and +forward, +jump, +attack, +attack2 and +use, steering by positions a small
VScript prints each tick. Commands go over the game's TCP console, as for the in-game check. Each step is checked (player
cell, portals, cubes); one that didn't work is loaded back and retried carefully, and only then
put right by teleporting (`setpos`, a cube's `SetOrigin`, `portal_place`) and reported as
recovered. `--out DIR` keeps the evidence per level and runs batches; `train/playreport.py`
sums one up. `--spike` checks the console, VScript, coordinates, movement and these commands
instead of playing. It has played batches of generated levels in the game (see
`out/play/REPORT.md`); `play.Session` lets other scripts drive the game a level at a time.

`tantalos-score route --plan "TOKENS" FILE` routes a given plan (Goblin's) instead of the
solver's own. A plan that doesn't solve the level is routed as far as its steps fit, then an
`attempt` leg tries the step that fits no move (a shot at the face it names, or a walk toward
the cell); the head line's `net` says what `plan::check` found. play.py plays an attempt leg once
and marks it `net-failed`.

## Broadcasting
`train/broadcast.py` runs the loop for a Twitch stream: the level net writes levels, the solver
net solves them, Portal 2 plays them, and both nets train on each round's levels before the next
asks for harder ones (`train/tier.py`). Its overlay is an OBS browser source; see
`docs/streaming.md`.

## Looking at solutions in a browser
`tantalos-view` is a local browser app for the solutions you already have on disk. It scans folders
for `<name>.route.jsonl` files with their `<name>.p2c` beside them — what `tantalos-score route FILE >
NAME.route.jsonl` and `tantalos-score check --route DIR` write — lists them in a sidebar (grouped by
folder) and draws the chamber in oblique 2.5D (Z up, open-top cutaway) with its portalable faces,
gel, barriers, hazards, laser beams and devices, then plays the route back: the walks, the
portals and the cube, driven by each leg's `after` state. Picking a solution parses its `.p2c`
and reads its route — it never solves — so switching is instant, and `Save .html` writes a
self-contained copy with the data inlined.

`tantalos-view [--roots DIR]... [--port N] [--no-open] [--list]`: `--roots`/`--dir` add scan roots
(default `fixtures`, `out`, `runs`), `--port` fixes the port (default: any free one), `--no-open`
doesn't launch the browser, and `--list` just prints what it found. It works the same for a plan
the Rust solver made (`tantalos-score route`) and for one Goblin wrote (`tantalos-score check --route`).

## Triaging the downloaded Workshop levels
`tantalos-workbench` is a local browser app for the levels `tools/workshop/fetch.py` downloaded. It
merges `out/workshop/levels.jsonl`, `data/ws5/{train,test}.jsonl`, the Workshop ratings and item
metadata, and shows every field we have: title and id, subscribers/favourites/tags, the Workshop
star rating and vote count, the solver's verdict and the dataset verdict, plan length, all
metrics, conversion problems, and the chamber drawn in 2.5D from its tokens. The sidebar filters
by verdict, ws5 split, Workshop rating, your own rating, staged/excluded/notes, and sorts by any
of the useful numbers.

The app also stores your triage of each level in `train/workshop-annotations.jsonl` (one JSON
object per line, keyed by `workshop_id`):

```json
{"workshop_id":"100091397","staged":true,"rating":4,"notes":"solver gap: laser timing","excluded":true,"updated":1792888260}
```

- **staged** — copied into the Puzzle Maker for manual playtesting.
- **rating** — your 0–5 stars (0 = not rated).
- **notes** — free text.
- **excluded** — keep this level out of the ws5 **test** split. `tools/workshop/ws3.py` moves it
  to train when it rebuilds `data/ws5`, and `train/goblin_data.py` leaves it out of `test-ws.jsonl`.
  It can still be used for training.

The annotations are server-side, so agents and other tools read the same store as the browser:

```bash
python tools/workshop/annotations.py --get 100091397
python tools/workshop/annotations.py --excluded
python tools/workshop/annotations.py --json
python tools/workshop/annotations.py --set 100091397 --rating 4 --staged --notes "worth playing"
python tools/workshop/annotations.py --set 100091397 --exclude
```

`tantalos-workbench [--all FILE] [--train FILE] [--test FILE] [--ratings FILE] [--items FILE]
[--annotations FILE] [--port N] [--no-open] [--list]` defaults to the files above. `--list` prints
the corpus and annotation summary without starting a server. The level files are never rewritten;
only the annotation store is.

## Plans as tokens (for Goblin, the net that solves levels)
`src/plan.rs` writes a route as plan tokens, one group per step in the level's own coordinates
and without the walks: `fire blue 0 5 4 -x fire orange 7 0 1 +x pickup 2 2 1 enter orange
putdown 2 6 3 exit` is the worked example. `tantalos-score score --plan` adds each solvable level's
plan (`"plan"`, or `"plan_error"` when the route can't ground it), and
`train/goblin_data.py` builds `data/goblin/` from the existing data that way. `tantalos-score check` replays
plans a model writes: each step has to fit an action the solver could take next (any face of
the right face class, any cell of the region it leads to), and the plan has to reach the exit.
It reports `solved` (with `optimal` when given the level's `min_len`), `illegal` (the step
that fits nothing, `fail_verb`), `unfinished` or `unparsed`. `--route DIR` writes a solved
plan as a route for `train/play.py`, grounded by the router, so its faces and cells can
differ from the plan's. On a first sample, every generated level's plan grounded, but 2-10%
of Workshop-derived ones didn't (the router can't place some shots and cube drops).

## Unverified (check in the game)

The later items' rules were fitted to `editoritems.txt`, the editor's instances and compiled
Workshop maps, not yet watched in the game. The ones that most change verdicts:
- Angled panels: local +x points at the edge that swings out, the hinge is opposite, and the
  face leans toward the hinge (so a wall panel with +x down leans up and out). At 90° the
  panel lies on its hinge edge (a wall, or a floor for the cell above a wall panel's top).
- Light bridges sit 0.375 voxel against their walkable side: side up, the floor of the cells
  they cross; side down, the floor of the cells above.
- Flings: g = 600 u/s², speed sqrt(2gh) after dropping h voxels (capped at 2000 u/s), the
  player a point at its feet; conservative rules want ±10% speed to land alike. Dives reach
  two cells out from a ledge (three, liberal). Eyes may stand 0.4 voxel toward a drop.
- Faith plates land on their target; `Builder::faith_plate` writes a speed for a lob two voxels
  over the higher end (the compiled trigger aims with it).
- Lasers pass glass and grating; a reflection cube redirects a beam crossing its cell toward
  the way the player faced setting it down; beams don't hurt the player.
- Funnels carry from any cell beside them and let you step out sideways anywhere further on;
  oscillating rails act as a walkway along their track.
- Flip panels start with the tile's own side out; flipping clears a portal on them.
