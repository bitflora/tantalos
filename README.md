# glados

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

```bash
cargo test
cargo run -- info fixtures/synthetic/worked_example.p2c
cargo run -- solve fixtures/synthetic/*.p2c
cargo run --release -- evolve --seed 1 --gens 40   # best levels + report.md in out/evolve-1
cargo run -- roundtrip fixtures/real/*.p2c
cargo run -- fixtures            # regenerate fixtures/synthetic from src/fixtures.rs
cargo run -- tokens fixtures/synthetic/worked_example.p2c
```

## Training a level model
```bash
cargo build --release
./target/release/glados dataset --seed 1 --count 200000 --out data/levels.jsonl
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv/Scripts/pip install -r train/requirements.txt
.venv/Scripts/python train/train.py --data data/levels.jsonl --out runs/first
.venv/Scripts/python train/sample.py runs/first/best.pt --n 64 --verdict solvable --len 5-7 --p2c
```
- **Tokens** (`src/tokens.rs`): size, then rows of columns as solid/open/goo runs, then white
  surfaces as boxes, doors, panels, fields, droppers, buttons and pedestals with their wires.
  Repeats are free: `=` repeats the previous column. Columns left out of a row, and rows left
  out at the end, repeat the last one, so the model never has to count to the room's width or
  depth. That was the main way early samples failed to parse. The worked example is 56 tokens;
  generated rooms average about 105 and stay under about 310. Round trip: `encode(decode(encode(p))) == encode(p)` and the
  solver's verdict is unchanged (tests on every fixture and hundreds of genomes). It keeps what
  play depends on, not the file's bytes. Lasers, bridges, funnels, gel and the like are refused.
- **Dataset** (`glados dataset`): JSONL, one level per line with its tokens, verdict, plan length
  and metrics, plus `vocab.txt` beside it. Jobs are 30% fresh random genomes, 70% mutations of a
  pool of solvable ones that drifts toward higher `Metrics::score`. Over a 20k run the share of
  solvable levels needing 0 actions falls from 34% to 10%, and portal use rises from 5% to 33%.
  About 250 levels/s on 12 threads. Seed-determined, independent of thread count.
- **Scoring** (`glados score`): one token line in, one JSON line out (`unparsed`, `invalid`,
  `unsupported`, `unsolvable`, `uncertain`, `solvable` + metrics; `canonical` if the line is
  exactly what `encode` would write). `--p2c DIR` writes the solvable ones for playtesting.
- **Model** (`train/`): GPT-2-style decoder, 6 layers × 384 wide (~11M parameters), block 512,
  trained from scratch. A sequence is `<bos> conditions <sep> level <eos>`. Conditions: verdict,
  plan-length bucket, portal trips, cubes, and yes/no for fizzler, laser field, glass, grating,
  pedestal and goo. Each is hidden 25% of the time, so sampling can fix any subset. Every eval
  samples levels conditioned on `<verdict:solvable>` and scores them with `glados score`.
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

## Solver (Tiers 0 and 1)
`solve` runs two rule sets and reports **solvable** (conservative rules find a plan, printed),
**unsolvable** (liberal rules find none), or **uncertain** (only liberal rules solve it, plan
printed; or a budget ran out). It rejects levels that are invalid or use items it doesn't model
(lasers, faith plates, bridges, funnels, gel, turrets, ...).

Search state = (region, portal face class per color, cubes as (kind, home, location), latched
pedestals). Actions: drop, cube-step, walk through a field, fire, enter portal, pick up, put
down (loose or on an accepting button), toss off a ledge, throw through a portal, press a
latching pedestal, press a timed pedestal and dash on foot before it runs out. Reaching the exit
region with the door open ends the plan (the plan's worked example is 5, not 6, because walking
out isn't counted).

| | Conservative | Liberal |
|---|---|---|
| climb a 1-voxel ledge on a cube | no (unmeasured) | yes |
| fire while carrying | no | yes |
| shot grazing a voxel edge | blocked | allowed if one way round is clear |
| shot targets | face centre | centre + 4 inset points |
| fizzler that can ever be on | blocks shots | only one that is always on |
| walking speed (timed dashes) | 140 u/s | 190 u/s |

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

## Metrics and score
`solve` also prints, and `evolve` optimises, metrics from the full state space and ablations:
plan length and portal trips, number of shortest plans, dead-end (softlock) states, cubes that
aren't needed, face classes that are (painting one black makes the level unsolvable), and the
share of random players that stumble out. `Metrics::score` folds them into one number:
log of weighted effort (portal shots and timed dashes 1, trips 2, latching presses 0, other
actions 0.5) + random players failing +
necessary classes − unnecessary cubes − softlock share. The first version rewarded plan length
linearly and bred 30-action staircase slogs; weights are still a guess pending playtests.

## Generator and search
`generator::Genome`: room size, two doors on the outer walls, and a list of features (platforms
and pits on a heightmap, goo, white patches, glass/grating panels, fizzlers/laser fields, ceiling
cube droppers of any kind, floor buttons of any kind and pedestals, wired to the exit or a
field). The search scores levels with the conservative rules, so it can't win by leaning on
unmeasured moves. `evolve` keeps an elitist population of solved levels, breeds children by mutation, and
drops behavioural duplicates (same size and metrics) so it doesn't fill with clones. Results
depend only on the seed. 40 generations of 48 took 25 s–3 min on 12 threads: later generations,
and levels with timed pedestals (a dash is re-solved in every state), are slower.

What it finds today: portal-hopping mazes with many necessary surfaces, 10–25 actions. Solvable
under the model and not trivial, but mazy rather than elegant. That's the plan's "hard ≠ fun"
risk; next steps are a style prior from real levels and in-game checks of the winners.

## Assumptions to verify in-game
- Goo floods every open cell reachable sideways or downward from a goo item.
- A 1-voxel ledge needs a cube to climb; 2+ voxels can't be climbed; any drop is survivable.
- Generated files open in the editor and compile.
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
- Barrier and field layouts (see format facts) rest on one real file; confirm in the editor.
