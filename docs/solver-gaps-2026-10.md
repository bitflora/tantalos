# Why the solver fails on Workshop levels (triage, 2026-10-22)

Of the 16,214 Workshop training levels (`data/ws5/train.jsonl`) our solver solves 2,617. The
other 13,597 are trained as `published` (assumed solvable, no plan, no metrics) and make poor
mutation parents (about 1.2 mutants each against 7 for solvable ones, at ~50× the cost per
mutant). Their `solver_verdict` is **unsolvable** (6,370: both models searched everything and
found no way out) or **uncertain** (7,227: the 50k-state search ran out; 1,169 of all such
Workshop maps were solved by the liberal model only).

Published levels are almost always solvable, so an "unsolvable" is nearly always the model
being wrong, and an "uncertain" is the model being wrong or the search being too small.

## Method

Seeded samples (seed 42) of 100 unsolvable and 100 uncertain levels, scored with
`glados score --editor-rules off` (as `tools/workshop/fetch.py` does), at budget 50k unless
noted. Each level was also scored with one thing *relaxed* in its tokens (`relax.py`,
`relax2.py`):

- **exit**: the exit door takes no inputs (open from the start);
- **white**: every surface portalable;
- **turrets**: turrets removed; **fields**: fizzlers and laser fields off and unwired;
  **goo**: goo turned into open floor-less space;
- **exit-lasers / exit-buttons / exit-pedestals**: only that kind of input to the exit removed;
- **latch**: timed pedestals made latching.

A relaxation that makes a level solvable says *where* the model gets stuck, not that fixing
that mechanic would solve it: forcing the exit open skips the whole puzzle. Read the counts
as "the failure involves this", an upper bound on what a fix there could unlock.

Everything is in `out/solver-gaps-2026-10/` (samples, scripts, raw verdicts).

## Results

Levels out of 100 that become **solvable**:

| relaxed            | unsolvable sample | uncertain sample |
|--------------------|------------------:|-----------------:|
| nothing            | 0                 | 0                |
| exit open          | 31                | 32               |
| all surfaces white | 31                | 16               |
| turrets removed    | 1                 | 19               |
| fields off         | 7                 | 6                |
| goo removed        | 0                 | 0                |
| exit open + white  | 72                | 50               |
| all of the above   | 71                | 68               |
| exit: no laser inputs    | 9           | –                |
| exit: no button inputs   | 10          | –                |
| exit: no pedestal inputs | 3           | –                |
| timed pedestals latch    | 1           | –                |

Budget, uncertain sample: 50k → 0 solvable; **250k → 4 solvable, 9 proven unsolvable, 87
still uncertain**. (A 1M run was stopped after making no progress on the first level for an
hour while sharing the CPU.) So budget alone recovers ~4% (95% CI 2–10%): the uncertain
levels are mostly the same model gaps, in bigger levels whose search doesn't end.

Size: unsolved levels aren't much bigger than solved ones (median 282 tokens unsolvable,
367 uncertain, 218 solvable; median 4–5 devices against 2). No single item kind explains the
failures: every kind has a 7–16% solve rate (`feat.py`).

## Causes found

Estimates scale the sample share to all 13,597 unsolved training levels (95% intervals for
n = 100 are roughly ±8–9 points at 30%, ±5 at 10%).

1. **Getting the exit open (31–32% of both samples, ~4,300 levels).** The player can reach
   the exit but the model can't satisfy its inputs. Split by input kind (unsolvable sample):
   - **Laser inputs (9%, ~600 of the unsolvable).** Hand-checked on "EZ" (3660480385): one
     reflection cube must send the beam through two floor relays and into a catcher, which
     only works aimed at an angle that is neither straight nor 45°. `src/lasers.rs` lets a
     cube face only the 4 horizontal directions from a cell, and powers a catcher only head
     on. In game, cubes aim anywhere and catchers take angled beams. **Fix: angled beams.**
     Rays from a cube spot toward each receiver/portal face (a finite set of aims), traced
     through the grid with a voxel walk; catchers accept beams within their cone.
   - **Floor-button inputs (10%).** Cube delivery. Hand-checked "Smallest Puzzle in the
     Multiverse" (68464713): solvable with fields off but not with the timed pedestal made
     latching, so the blocker is getting the cube past a field, not the timer alone. Not
     narrowed further; candidates are the conservative/liberal cube rules (no throwing, no
     carrying through portals in flight, cube steps) — needs an in-game look.
   - **Pedestals (3%)**, timed-pedestal limits (1%): minor. (`dash` lets the player only walk
     during a timer: no cube pickup, no portals. Making timers latch unlocks 1 of 100.)
2. **Movement and portal surfaces (31% unsolvable, 16% uncertain; ~3,100).** Making every
   surface portalable solves these, so the model can't find the way between regions with the
   real portal surfaces. Titles cluster around jumps and flings ("One Jump", "bunny hop",
   "Cross The Chasm", "Jump For Fun", "Terminal Velocity", "jump jump jump …"). Our movement
   has no jumping at all (`src/regions.rs`: a gap of one cell can't be crossed) and flings
   only by dropping off a ledge into a floor portal with little or no air control
   (`src/flings.rs`). **Gap jumps are not it:** a prototype that lets the player jump a
   one-cell gap (`gap-jump-proto.patch`) unlocked 1 of the 100. What remains likely is
   momentum: running/jumping into portals, chained flings, air control, angled-panel
   launches. Needs in-game observation of a few of these levels before modelling.
3. **Turrets (19% of uncertain by relaxation; ~1–4% real).** Conservative rules never enter
   a region a live turret can see; the liberal model may run through but the search then
   explodes and runs out. An exposure model was tried and doesn't pay: see "Turret exposure"
   below.
4. **Budget (4% of uncertain, ~300).** Cheap to take: score Workshop levels at 250k.
5. Fields (6–7%), goo (0%): not worth a dedicated fix.

### Also: 3,618 Workshop maps never reach training

`out/workshop/levels.jsonl` marks 3,618 converted maps `invalid`. About 2,000 are "exit/entry
door has no floor to stand on" (1,471 exit, 519 entry, 118 both); the rest are mostly turrets
or panels "with nothing behind them" (converter offsets) and "more than 32 turrets". Since
these maps were published, elevated doors are presumably legal in the editor (the game may
put a ledge in front of them). That is unverified: check one in game (e.g. 101868816 "walls
that open") before changing the converter or the door rule. If it holds, it adds ~2,000
levels, of which the usual ~16% would be solvable.

## Angled laser beams: done on branch `angled-lasers` (2026-10-22)

What changed (`src/lasers.rs`, `src/solver.rs`, `src/dataset.rs`; patch in
`out/solver-gaps-2026-10/angled-lasers.patch`):

- **Angled spots.** A reflection cube can sit at 1/4, 1/2 or 3/4 of the way along a beam
  through its cell, aimed at any floor relay, wall catcher, turret or white wall face on its
  level. Its beam is traced at that angle through the grid. A relay counts within 0.3 voxel
  of its centre, a turret within 0.35, a catcher if the beam lands on its face within 0.4 of
  centre and 70° of head on. An aim is kept only if it reaches what it aimed at. An angled
  spot redirects only the beam it sits on; any other beam stops at the cube.
- **Cubes on laser items.** With angled spots, a cube may also sit in front of a wall emitter
  or catcher, and on a floor emitter (its beam going up into the cube and out sideways).
- **Second pass only.** `label`, `plan_only` and `judge` first run the plain model. Only when
  it doesn't solve a level that has a reflection cube and an emitter do they build
  `Model::new_angled`. Levels the plain model solves keep their labels.
- **Bug fix (both passes):** an emitter on a white wall shares its ray with a portal's beam
  out of that wall, and `Beams::ends` dropped every portal-out ray's end. So the emitter's own
  beam was never seen going into a portal. Now `ends` comes from the emitters' and spots'
  rays only.

Checks: all tests pass, plus three new ones in `tests/tier1.rs` built from "EZ", "Easy test"
and "laser field jumping". Before/after on 300 generated laser levels (data/v5b) and 200
solvable Workshop laser levels:

- **Generated:** verdicts are unchanged. 8 have different metrics, from the ends fix.
- **Workshop:** 199 stay solvable and 1 becomes uncertain (the ends fix adds portal routes).
- **Angled pass on its own:** byte-identical output on all 500.

Results, budget 50k:

| sample (100 each) | before | after |
|---|---|---|
| unsolvable | 100 unsolvable | 4 solvable, 11 uncertain, 85 unsolvable |
| uncertain  | 100 uncertain  | 100 uncertain |

At 250k, the 11 new uncertain levels come out as 1 more solvable, 3 unsolvable and 7 still
uncertain. So this fix is worth about 5% of the unsolvable levels (~300 of 6,370; 95% CI
2–11%) and nothing measurable on the uncertain ones.

Solved: "EZ", "Easy test", "laser field jumping" and "Fizzle Lizzle". Still failing in the
laser group: "Simicult 777", "2 Simple Chambers", "Simple lasers", "Daily Chamber 3" and
"lazers", mostly two-cube levels where the angled search runs out of budget.

Limits:

- Angled aims are sampled, not every angle.
- Beams stay horizontal.
- A plan that uses an angled spot still encodes the cube's facing as its nearest axis
  (`Spot::dir`; the real yaw is in `Spot::aim`). Plan tokens, `glados check` and the play
  harness can't replay those plans yet. That only matters for the solver net's plan data
  (`train/plans.py`), and only for levels solved through the angled pass.

## Turret exposure: tried, not worth it (2026-10-22)

The 19 uncertain-sample levels that turn solvable with turrets removed fail for two
reasons. Neither one is that the player can't cross a turret's sight for a moment:

- **The start is in sight (8 of 19).** Under conservative sight (60° either side, 16 voxels,
  grating doesn't stop bullets) a turret sees the entry, often from 9–14 voxels away or from a
  balcony two voxels up. Liberal sight (45°, 8 voxels) sees none of these. Conservative rules
  die on the first state.
- **The search runs out (11 of 19).** The liberal model runs past turrets but hits the 50k
  budget, even in small levels (one has 9 regions and 3 turrets). Turrets split regions by who
  can see them and add knock-over actions, and removing them shrinks the search. At 250k, 1 of
  the 19 solves.

What was tried (scratch worktree, not committed; `out/solver-gaps-2026-10/turret-*.patch`,
`tdiag*.rs`). Samples: 100 uncertain Workshop levels (U), 200 solvable Workshop levels with
turrets (W), 300 generated turret levels from data/v5b (G):

| variant | U solved | W still solvable | G solvable / uncertain / unsolvable |
|---|---|---|---|
| current solver | 0 | 198 | 229 / 44 / 27 |
| graze: walk ≤ K cells in sight, moving only (K = 1, 2) | 1, 2 | – | – |
| + exposure counter in the state, any action ≤ 2 in a row | 3 | 182 | 229 / 46 / 25 |
| same, moves only on graze cells | 2 | 190 | 230 / 45 / 25 |
| no graze; starting in sight, act (not move) for 2 actions | 1 | 198 | 235 / 38 / 27 |
| that + liberal turret sight in conservative rules | 4 | 194 | 231 / 42 / 27 |

- Graze cells split regions, and the searches that were just inside the budget run out.
- The "act at the start" gains aren't real. Actions belong to a region, so "knock it over"
  from the start region reached turrets 3–8 cells from the entry. All 6 generated gains did
  that, and `a_turret_facing_the_entry_blocks_it` caught it. Limiting the knock to turrets next
  to the entry would leave almost nothing.
- Even liberal sight, an unmeasured guess, gets only 4 of 100.

So turrets are ~1–4% of the uncertain levels, not 19%. The rest of what the turret
relaxation showed is search size: the same "bigger search than the budget" problem as the
rest of the uncertain group.

## A smaller search: tried, not the bottleneck (2026-10-22)

These experiments asked whether making the search smaller would turn uncertain levels
solvable. Mostly it wouldn't. Scratch work, nothing committed: `search-size.patch`,
`sdiag.rs`, `cdiag.rs`, `coarse.rs` and the outputs in `out/solver-gaps-2026-10/`.

**What the conservative search does on the 100 uncertain levels:** 30 solve or are proven
unsolvable by it (they're uncertain through the liberal model). 28 stop early (1–7,000 states)
on an incomplete model, which can't be called unsolvable; that is a model gap, not search
size. **42 run out of the 50k budget.**

**What fills the budget** (states the search reached, split by state part): the player's
region has only a handful of values. The rest of the world (cubes, buttons, devices, paint)
mostly has tens to hundreds of combinations. The multiplier is:

- **Portal pairs (most levels).** There are 20–300 portal face classes, so hundreds to
  thousands of blue × orange pairs. Classes already merge faces that behave the same; they
  split on landing details, flings, and, in funnel/bridge levels, on `routes`, since every face
  has its own projection out of a portal. In laser levels `beam_out` does the same.
- **Reflection cubes in laser spots (laser levels).** With 48 spots (the cap) and 2–3 cubes
  there are thousands of placements (12,002 in one level with no portals at all). The relevance
  filter in `lasers.rs` keeps every spot whose beam ends on a white face once any portal
  beam could hit something, which in white rooms is all of them.

**Three ways to cut it, measured:**

| idea | sound? | effect |
|---|---|---|
| drop a portal class another one dominates (does at least as much, seen from at least everywhere) | yes | removes 10–30% of classes |
| second pass with 1–2 representative classes per (entry, landing regions, role) group | plans found are valid | 2 of the 70 search-limited conservative runs solve |
| second pass putting reflection cubes down only where a beam reaches | plans found are valid | 0 of 38 laser levels |

**The decisive check:** the conservative search alone, at 1M states (20× the budget), on 10 of
the levels that run out (`big-1m.txt`):

- 1 solves (5 actions, at 84k states);
- 5 are proven unsolvable (60k–703k states);
- 1 stops on an incomplete model;
- 3 still run out at 1M.

So most of these levels are model gaps too, only bigger; a search that is larger, or smarter
by a constant factor, mostly proves that sooner. The gain from search is about what a bigger
budget gives: 250k solved 4 of 100, 1M about 1 in 10 of the search-limited levels (~4% of
the uncertain). Take it by re-scoring at 250k. Search work after that pays only for turning
uncertain into unsolvable (cleaner labels, cheaper mutation), not for the hit rate.

## Re-score at 250k (2026-10-22)

All 15,368 unsolved Workshop levels in data/ws5 (13,597 train, 1,771 test) were scored with
the current solver (angled lasers included), at 50k and at 250k. Raw results are in
`out/rescore-250k/` (`verdicts.jsonl`: old label, 50k, 250k per level). The data files are
unchanged.

| | solvable | uncertain | unsolvable |
|---|---|---|---|
| stored labels | 0 | 8,137 | 7,231 |
| current solver, 50k | 304 | 8,618 | 6,446 |
| current solver, 250k | **679** | 7,738 | 6,951 |

- At 50k, the current solver solves **304** more: 225 stored as unsolvable and 79 as
  uncertain. That is the angled-laser pass and the `ends` fix.
- 250k adds **375** more, all from 50k-uncertain, and turns 505 uncertain into unsolvable. No
  verdict at 50k gets worse.
- Total: **679 newly solvable** (4.4% of the unsolved; 599 train, 80 test). Solvable Workshop
  training levels go from 2,617 to 3,216 (+23%).
- Time: 24 min at 50k and 101 min at 250k on 12 threads, about 4× the cost.

## Play-tested small levels (2026-10-23)

The five levels of item 4 below were staged as `glados-ws3-unsolv-00011..15`
(`out/ws3-unsolv-more/`). What the play-tester found:

- **3818309212 "???"**: easy. The first pedestal toggles the angled panel that blocks the hall;
  the second pedestal is wired to nothing (its compiled proxy has no outputs; the converter is
  right). Drop down the shaft onto the stairs, which are always up and face a wall, and step
  off their top step **sideways** onto the exit ledge. The solver only let stairs lead
  straight ahead. **Fixed:** `Stairs::exits` (src/level.rs) adds the cells beside the top step,
  one voxel up, when standable and not walled off; the solver and `glados route` use it. On
  800 unsolved Workshop levels with stairs, 7 become solvable (none get worse); on 1,000
  generated stairs levels from data/v5b, 1 becomes solvable and 4 get a shorter plan. The
  play harness hasn't walked a sideways exit yet.
- **68464713 "Smallest Puzzle in the Multiverse"**: solvable with two tricks the solver lacks.
  The two fizzlers cross in the middle of the room, making four quadrants. (1) Press the 3 s
  pedestal (it turns off the fizzler between the cube's quadrant and the next one), run to the
  cube, and carry it across that fizzler before the timer ends; it takes a fast, straight run.
  Our dash allows only walking, and its time counts a whole voxel per cell (5 steps = 3.7 s at
  conservative speed, 3.4 s at liberal speed, against about 2 s along the real path).
  (2) Put the cube down right beside the other fizzler. Then stand on the edge of that
  fizzler's floor button (the corner nearest the cube, diagonal across the fizzler) and pick
  the cube up across the fizzler while the button holds it off. Then press the pedestal again
  and carry the cube through the first fizzler to the exit button. **Fixed (2026-10-24)** with
  two new actions and plan words:
  - `Action::Fetch` / `fetch P C E`: press a timed pedestal, pick up a loose cube on the way
    (0.25 s), and carry it on, through no fizzler that is on, to region E. Conservative rules
    take the worst cell of the cube's region (at most 8 cells), liberal ones the best.
  - `Action::Reach` / `reach B C`: stand on a weighted floor button and pick up a cube in a
    cell beside the button's (or across a corner) in another region, across fizzlers that are
    on but off while the button is held. Conservative rules want every cell of that region in
    reach; liberal ones any.
  - Dash timing (`DashTimes`, shared by the solver and `glados route`): steps across a corner
    cost 1.4 voxels, a cell counts as reached on entering it (half a voxel before its middle),
    and a held cube half a voxel sooner still. Escaping through the exit still needs the exit
    cell's middle in time.
  - The level now solves conservatively in 8 steps, the play-tester's route. Old vs. new
    release binary (equal speed): 800 random unsolved Workshop levels, 1 uncertain → solvable;
    800 unsolved Workshop levels with a timed pedestal, 4 uncertain → solvable and 6
    unsolvable → uncertain; 1,000 random v5b levels, 2 uncertain → solvable and 22 shorter
    plans; 1,000 v5b levels with a timed pedestal, 3 → solvable, 3 unsolvable → uncertain and
    39 shorter plans. No verdict got worse. Mechanics: a fetch counts as "timed dash"; a reach
    has no bit of its own (a new one would add a training-condition group).
  - `train/play.py` has `act_fetch` and `act_reach`, not yet tried in game. `glados route`
    can now put a cube down in the player's own cell when its region has no other.
- 00011..00013 (One Jump, FLAAD TEST, You can't simply jump around it): not recorded here.

## Recommended order

1. ~~**Angled laser beams**~~: done (above), ~5% of unsolvable levels rather than the 9% the
   relaxation suggested. The rest of the laser group is two-cube searches running out of
   budget.
2. ~~**Turret exposure**~~: tried (above), ~1% and it costs search. If anything, measure turret
   range, cone and pitch in game (a ruler level). The conservative 16-voxel range decides 8 of
   the 19.
3. **Re-score Workshop at 250k** (cheap, ~300 more on its own). ~~Search efficiency~~: tried
   (above). Most search-limited levels turn out unsolvable by the model at 1M states, so the
   hit rate needs model fixes, not a smaller search.
4. **In-game look at 5–10 small levels** from the movement group and the button group
   (smallest: 3719326070 "One Jump" 53 tokens, 3818309212 "???" 99, 3779764375 "FLAAD
   TEST" 109, 3803335172 "You can't simply jump around it" 132, 68464713 "Smallest Puzzle in
   the Multiverse" 99) to learn the actual tricks before modelling momentum or cube rules.
   A human play-through of each takes minutes and settles what hand-reading the tokens
   couldn't.
5. **Elevated doors**: verify in game, then the converter/door rule (~2,000 maps).

Every fix changes labels: re-run `fetch.py build` scoring, rebuild `data/ws*`, and treat EVAL
as not comparable across it, as with the editor-rule rounds.
