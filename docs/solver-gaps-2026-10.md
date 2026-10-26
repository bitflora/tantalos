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
`glados-score score --editor-rules off` (as `tools/workshop/fetch.py` does), at budget 50k unless
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
  (`Spot::dir`; the real yaw is in `Spot::aim`). Plan tokens, `glados-score check` and the play
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
  one voxel up, when standable and not walled off; the solver and `glados-score route` use it. On
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
  - Dash timing (`DashTimes`, shared by the solver and `glados-score route`): steps across a corner
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
  - `train/play.py` has `act_fetch` and `act_reach`, not yet tried in game. `glados-score route`
    can now put a cube down in the player's own cell when its region has no other.
- 00011..00013 (One Jump, FLAAD TEST, You can't simply jump around it): not recorded here.
- **3094057883 "Simple Test 01"** (staged as `glados-ws5b-unsolv-00002`, 2026-10-24; the user's
  favorite, `favorites/ws5-simple-test-01.p2c`): momentum. Exit portal high on a wall, entrance
  in the floor under it; walk in, and just as you come out, move the exit to the tilted panel:
  the fall back through the floor portal flings you onto the fence platform. Press the
  pedestal, put the exit on the panel it tilts, a floor portal where you land coming out of it
  slowly, and walk in holding the cube: the second fall through lands you by the exit's button.
  Cube on the button, then dive off the raised platform into a floor portal to the exit. Two
  gaps, **fixed (2026-10-24)**:
  - `Action::Loop`: with a floor portal under the way out of the other portal (straight down,
    or a cell further out, `loop_falls`), walk in and fall back through at the speed of that
    fall, flying out of the other portal or (`refire`) of a face the player shoots on the way
    down from where they walked in. Not grounded by `glados-score route` yet (no plan word, no play
    harness leg), so such plans are left out of solver-net data.
  - The dropper's cube falls into goo in an alcove; the player catches it from the floor
    beside its fall. A dropper's cube that would fall into goo now lies in such a floor's
    region.
  - Liberal rules solve it in 11 actions (the play-tester's route, with the cube carried
    through the first loop); conservative rules don't trust the 4-voxel fling out of the tilted
    panel (its trials land in different places), so it is uncertain.
- **2026-10-24 batch** (`glados-ws5-unsolv-00004`, `glados-ws5b-unsolv-00004..05`). Two the
  editor rejects, so they are converter problems, not solver gaps. `train/portal2.py check`
  names the items:
  - **3093140791 "Easy chamber"** (ws5b-00004), invalid: `barrierhazard (13,9,2)` twice. The
    converter puts a fizzler and a laser field on the same mounting face (`fizzler 13 9 2 +x +z
    1` and `laserfield 13 9 2 +x +z 1`). In the map they are 48 units apart in that voxel
    (laser at y=1168, fizzler at y=1216, both instances facing `0 90 90`), with a third field,
    a laser at y=1136 facing the other way, in the voxel before. The fizzler is in the
    middle of its voxel (LocalPos 0), which tokens can't say; our builder always puts fields
    0.375 voxel toward their N × A side.
  - **3784503220 "Replica. (vertical fix)"** (ws5-00004), invalid: `target (8,12,5)`. The
    plate at 8 12 3 throws straight up: its trigger launches at a helper `trigger_catapult`
    (`launchDirection -90 90 0`, playerSpeed 493, apex about 2 voxels above the plate's floor),
    so the map has no target entity. The converter puts the target in mid-air at the apex. 38c8db4
    wrote that target with `VERTICAL_ALIGNMENT 1` on plate and target, assuming the editor then
    accepts it, and `editor.rs` exempts such targets from `target-not-on-surface`. The game
    still rejects it, so that assumption is wrong.
  - **3722716739 "2026 9"** (ws5b-00005): the editor accepts it, but the play-tester couldn't
    solve it, and the solver calls it unsolvable even under liberal rules. No lead.
  - "Flying a Timing" (ws5-00001, reported earlier): `button (25,0,1)` twice and `target
    (14,4,2)`. Both fixed: the vertical faith plate no longer writes a target in mid-air, and
    one of the two pedestal buttons that shared a cell moves to the free neighbouring cell.
    Re-converted and staged again 2026-10-24; the editor accepts it.
  - **In-game tests** (33 test chambers, `train/portal2.py check --keep-vmf` keeps each valid
    level's export):
    - Fields: a fizzler and a laser field on the same side of one voxel are rejected, all of
      it or one overlapping cell (3 of 3). Accepted: one in the middle of the voxel and one on
      a side, of any kinds; fields on the two sides of a voxel; a fizzler and a laser field on
      the two sides of one face. (Same-kind cases kept hanging the editor's check: untested.)
    - Faith plates: `VERTICAL_ALIGNMENT 1` on the plate makes the export throw straight up,
      through a helper `trigger_catapult`, wherever the target is: on the plate, on the
      ceiling, or on a floor 3 voxels away. Only a target in the air is rejected. A plate
      with no target exports no launcher at all. The throw speed is √(76800 h) for
      `CatapultHeight` h (h = 1..8, any room height): it rises 64 h units over a trigger
      56 units above the floor.
  - **Fixed:**
    - `editor.rs`: parallel fields on one side of a cell collide whatever their kinds; the
      vertical exemption from `target-not-on-surface` is gone. On 2,000 sampled v5b levels,
      3 become invalid (a fizzler and a laser field on one face), the rest are unchanged.
    - `tools/workshop/convert.py`: a field in the middle of its voxel moves to the voxel's
      other side when a side field already holds its own side. With side fields on both
      sides (a laser-fizzler-laser sandwich) the map is unsupported. Of the 377 converted
      Workshop levels with two fields on one face, 302 re-convert, 74 become unsupported and
      1 keeps a pair (a floor-mounted fizzler). Under the new rule, `fields-overlap` flags all
      302 old conversions and 2 new ones. In game, 20 sampled old conversions were all
      flagged for fields, the new ones once (a field across a bridge), and 16 of 20 pass.
    - `builder.rs`: a floor target straight over a floor plate writes the target on the plate,
      `VERTICAL_ALIGNMENT` 1 and `CatapultHeight` = twice the voxels up. `Level` reads such a
      plate's height back from `CatapultHeight` (`level::vertical_reach`), so the tokens don't
      change: 9,557 Workshop and v5b levels with one re-encode the same as before. In game,
      21 Workshop levels with one (Replica among them) have no target flagged; 17 pass.
    - `tools/workshop/convert.py`: two pedestal buttons in one 128-unit cell (a map can place
      them inside the editor's voxel) no longer collide; the later one moves to the nearest
      free neighbouring cell with a floor under it. The only staged level that hit it, "Flying
      a Timing", now passes.
    - "C2 - Dumb Kinda Duplex" (ws3-00002) is staged again too: the goo surface now has a
      solid floor under it, and the editor accepts it.
    - "Easy chamber" and "Replica" are staged again under the same names; both pass the
      editor now, and the solver calls both unsolvable.
    - Not done: `out/workshop/levels.jsonl` and data/ws5 still hold the old conversions of
      the 377 levels (fields stacked; 74 now unsupported).
- **3543260079 "My First Test Chamber"** (`glados-ws5b-unsolv-00003`, 2026-10-24): a turret in
  an alcove behind a laser field (with a button) sees the entrance; a piston, up until its
  button is pressed, stands below a ledge with the second cube. Play-tester: portal behind the
  laser field and knock the turret over; put a cube on the alcove's button (the piston goes
  down), portals near the cube and near the piston, stand on the piston and reach through the
  portal to take the cube off the button: the piston lifts you to the second cube (or get
  there with momentum). Both cubes on the two exit buttons. **Fixed (2026-10-24):**
  - Starting where a live turret can shoot was fatal under conservative rules (no action at
    all); now they may step out of its sight, into a region beside the entry's cell
    (`start_exits`), and nowhere else it can shoot.
  - `Action::PortalReach`: within reach of a portal (its face's cell or one beside it), take
    a cube off the floor button the other portal comes out onto, staying put (and riding the
    platform the button released). Not grounded by `glados-score route` yet.
  - Conservative rules solve it in 18 actions, the play-tester's second route (they never
    need to knock the turret over); liberal ones also by a portal loop.

- **2026-10-24 batch 2** (`glados-ws5b-unsolv-00006`, `-00009`, `-00010`), all solved by the
  play-tester and **fixed (2026-10-24)**; each is in `tests/playtested.rs`.
  - **1498933548 "Laser Management"** (00010; `favorites/ws5-laser-management.p2c`): the prism
    turns the laser onto a white wall near the start. Stand on the button and portal the beam
    into the receptacle on your side (the fizzler goes off only while the button is held and
    the receptacle lit). Portal across the fizzler and walk through. Put the prism on the
    button, pointing at the ground-level receptacle, and portal both sides so the laser
    reaches it. Gap: a reflection cube lying on a floor button did not press it. **Fixed:**
    a reflection cube put down in a button's cell (`button_spot`, `Model::pressed`) holds the
    button down and redirects the beam. Conservative-solvable, routable.
  - **68655995 "The Grate Escape"** (00009): put a portal at the top of the funnel and the other
    on the wall across; the funnel beam out of it crosses the player's column, and moving
    that portal about half a panel higher draws them up; at the top, step out sideways.
    Gaps: funnel beams stopped at grating (they pass it; glass still stops them), and no
    rise. **Fixed:** `dynamics::funnel_path` (grating passes; a ride along a beam stops at a
    grating), and `climb_moves`: from a cell of the beam out of a portal, up the column
    through cells some other portal's beam crosses, stepping out sideways anywhere. Conservative-
    solvable; `glados-score route` can't ground the lift yet.
  - **68651531 "Mari0 - Test chamber 04 (Update 1)"** (00006): gel dropped through the floor
    portal under the dispenser and ceiling portals paints the floor; bounces off it, fed
    back in through ceiling portals, carry the player up to the exit ledge (running jump
    from the start ledge at the end). **Fixed, liberal rules only:** with two or more white
    ceiling faces and bounce gel, a bounce off painted floor reaches `size_z - 1` voxels up
    and 4 cells sideways (a guess; the ratchet's real limits are unmeasured). Conservative
    rules stay uncertain. Not routed.
  - Old vs. new binary on unsolved Workshop levels (`score --editor-rules off`): 1,200 with a
    funnel, 3 unsolvable → solvable, 1 uncertain → solvable, 6 unsolvable → uncertain; 1,200
    with lasers, 5 + 2 → solvable; 1,200 with bounce gel, 1 → solvable, 8 unsolvable →
    uncertain; 1,500 random, 10 unsolvable → solvable. No verdict got worse.

- **3718840820 "Flying a Timing"** (`glados-ws5-unsolv-00001`, 2026-10-24). Play-tester: portal
  the laser (floor emitter) from a ceiling portal to a floor portal under the catcher, which
  turns off the laser field; take the cube behind it to the button that raises the stairs;
  walk up them onto the first of four faith plates (the third on a wall), which throw you
  across the goo. There, with the lift near the bottom, press the 15 s pedestal that swings a
  walkway out under a cube dropper, ride the lift up, take the cube off the walkway and get
  back before the timer runs out; the cube on the exit's button. Three gaps, **fixed
  (2026-10-24), liberal rules only** (`tests/playtested.rs`):
  - Stairs whose top step is a faith plate went nowhere (the plate's cell isn't standable).
    Now a `Move::Plate` to where the plate chain lands; `glados-score route` writes it as `plate`
    (up the stairs onto the plate). Conservative-solvable and routed with the exit unwired.
  - **Rail platforms stood a voxel too low.** The platform is the top face of its item's
    voxel, so whoever rides it stands in the cell above (`Rail::stops`): in all 9,264 rails
    of the compiled Workshop maps (any orientation) the platform's top is the top of its
    movelinear's voxel, the editor places every item's instance at its voxel's middle
    (editoritems `Offset 64 64 64`), and this level's lift only meets the walkway that way
    (the play-tester rode it in the staged file). The generator now builds rails in the
    voxels under the cells they carry the player through (a pit's, or a one-voxel trench it
    digs along the wall); `rail_ferry` moved down into its trench.
  - A timed dash couldn't ride, and the dropper's cube (lost in goo but for the walkway) didn't
    exist. Now an auto-respawning dropper's cube lies on a device floor under it whenever that
    floor is there (`AutoDrop`); liberal dashes ride oscillating rails at their speed with no
    wait and may fetch such a cube. Conservative rules don't (unknown platform phase and
    respawn delay), so the level is uncertain. The dash isn't routed.
  - Old vs. new binary (`score --budget 50000 --editor-rules off`): 1,491 Workshop levels
    with rails: 82 unsolvable and 52 uncertain → solvable, 69 unsolvable → uncertain, 7 worse
    (3 solvable → uncertain, 4 uncertain → unsolvable); 4,000 unsolved Workshop levels: 34 →
    solvable, 28 → uncertain, 1 worse; 3,000 generated v5b levels with rails: 2 better, 62
    worse (55 of them solvable before). Every worse one has a rail: solutions that needed the
    platform a voxel lower than the game puts it. **data/v5b's rail levels (~120k) are not
    re-scored.**

- **3372907955 "SUPER SIMPLE #2"** (`glados-ws5b-unsolv-00012`, 2026-10-24). A corridor with
  two 3 s pedestals side by side, both wired to the exit. Play-tester: press both quickly one
  after the other, then walk out before the timers run out. Gap: a timed dash held only its
  own pedestal on, so the exit (AND of both) never opened; a timed pedestal pressed during
  another's dash was not modelled. **Fixed (2026-10-24), conservative-solvable in 1 action**
  (`tests/playtested.rs`): `Action::DoubleDash` presses a second timed pedestal reached in
  time (0.25 s to press once its cell is entered, `PRESS_TIME`) and walks on with both on
  until the first of the two runs out, out of the exit or to regions the single dash
  doesn't reach. No cube fetch during it, no third pedestal. Not routed (no plan word).
  - Old vs. new binary (`score --budget 50000 --editor-rules off`) on all 1,366 Workshop
    levels with two or more timed pedestals: 2 unsolvable and 1 uncertain → solvable, 3
    unsolvable → uncertain (liberal-solvable), none worse; scoring took ~7% longer.

- **3817082781 "sp_timed_lazer_change"** (`glados-ws5b-unsolv-00014`, 2026-10-24). The laser
  crosses the funnel's column in mid-air; nothing else takes portals. Play-tester: put the
  reflection cube into the funnel facing the catcher and go to the exit while it rises; it
  turns the beam onto the catcher for a moment, and you must leave in that window. Gap: cubes
  never rode funnels. **Fixed (2026-10-24), liberal rules only** (`tests/playtested.rs`):
  `Action::Sweep` (`lasers::Sweep`, `Event::Funnel`): a reflection cube put into a funnel from
  the cell behind it redirects each beam it crosses (both heights in a vertical funnel, high
  in a level one); if that opens the exit, the player escapes on reaching the exit cell's
  middle before the cube reaches the beam, at an assumed 250 u/s (`FUNNEL_SPEED`, the
  instance's `linearForce`). The compiled map's exit door doesn't close while the player is in
  its `door_clear` trigger (64 units each side of the door), so being there in time is enough.
  The cube needs 1.0 s here, the walk 0.94 s at liberal speed, 1.28 s at conservative speed, so
  the level is uncertain. Not routed (no plan word).
- **3495602854 "Mini Puzzle"** (`glados-ws5b-unsolv-00015`): play-tester: aim the prism so
  the beam passes the relay and hits the catcher, then press the 3 s pedestal and run out. The
  laser part already solved (an angled spot at the emitter); the dash doesn't: the pedestal is
  at the far wall (its prop at x=16 in a cell 0..128), and the exit cell's middle is 5.4 voxels
  of walk (about 640 units in a straight line) away, 3.4 s even at 190 u/s. **Still
  unsolvable**. Found: portal2.fgd says `OnButtonReset` fires once the button is back out,
  and switch001.mdl's `up` and `down` take 0.13 s each, so timers now run 0.13 s longer
  (liberal 0.26 s, `RESET_LAG`). That isn't enough. A likely explanation, unmeasured:
  `cl_forwardspeed` and `cl_sidespeed` are both 175, so running diagonally (W+A) may reach
  about 247 u/s if the engine doesn't clamp the sum. Measure it in game before changing
  `walk_speed`.
  - Old vs. new binary (`score --budget 50000 --editor-rules off`): 1,200 Workshop levels with
    a timed pedestal, 1 uncertain → solvable; 1,500 random v5b levels, no verdict changed, 2
    shorter plans; 1,216 of the 1,848 Workshop levels with a funnel, a reflection cube and a
    laser (stopped early), none changed. Nothing got worse. `tests/tier1.rs`
    `timed_field(4)` is now conservative-solvable (it was uncertain).
- **252742140 "Parkour, Parkour PARKOUR PARKOUR"** (`glados-ws5b-unsolv-00021`, 2026-10-24).
  Strips of floor between one-voxel channels of goo; play-tester: just jump across. Gap: the
  solver had no plain jump, only leaps off propulsion gel. **Fixed:** `Move::Jump`
  (`Geo::jumps`, src/regions.rs): a running jump across one voxel that can't be stood in, onto
  a floor at the same height, both rule sets. About 127 units of flight at walking speed
  (scaled from the measured gel leap, 290 units at 400 u/s) against 96 needed. Plans write it
  as `walk` (no new plan word); `glados-score route` gives it a `jump` leg and train/play.py
  `act_jump` jumps at the edge (untested in game). One-voxel pits in generated levels are now
  jumpable too, so v5b labels go stale where those were the obstacle (fixture
  `piston_bridge` widened to two).
- **3666439290 "First chamber"** (`glados-ws5b-unsolv-00018`, 2026-10-24). A laser field cuts
  the player off; the only white face is high on their side's wall, and a conversion gel
  stream pours on the far side. Play-tester: spread the gel around until you can portal past
  the field. Gap: gel out of a wall portal painted nothing (liberal: only a turret's floor).
  **Fixed:** `gel::arc`: out of a wall portal the stream flies on at the speed it fell into the
  floor portal (`gel::stream_speed`, sv_gravity, no drag: unmeasured) until it hits a surface,
  which it paints; faces painted that way can send it on in turn. Conservative rules use one
  speed (and a turret in the way takes the gel); liberal ones 60–100% of it. Here the stream
  out of the high wall portal hits the back wall at z=1 (only about 13 units above that
  voxel's floor at nominal speed, so close), and from there it lands on the floor in front of
  the player: a floor portal drops them out under the stream. Conservative-solvable.
  **Cost (2026-10-25):** made a class of every wall face (each sends the stream somewhere
  else) and tracked paint nobody reads; on 1,500 generated levels solvable fell 908 → 674 (the
  search ran out of budget). Now wall faces are grouped as before and a class's gel lands where
  every face of it sends it (liberal: any), ceiling classes still split by where gel falls, and
  paint is tracked only on faces conversion gel can whiten and (with bounce or speed gel)
  floors: 819 solvable. The rest is real growth (more faces to whiten) under the 50k budget.

## Orientation: the same level turned or mirrored (2026-10-25)

`glados-data augment --check` labels each of a level's 8 mirrored/turned versions
(`src/augment.rs`, which checks that each is the level moved cell for cell). On 600 ws5 train
levels 99.1% of versions got the original's verdict, plan length and editor flags; the rest
showed the solver depending on which way round a level was built:

- **Bridges through portals** (`dynamics.rs`, `Devices::build`). A light bridge coming out of
  a portal gets a route (a device bit) per group of portal faces that put it over the same
  cells. Routes claimed their cells one after another in coordinate order, and a cell two
  routes cross went to the first: the others were short there. With more routes than device
  bits, the first routes in coordinate order (and the first bridge's) got the bits. Sample line
  221: 14 steps as built, 10 turned 90°, with a plan step that doesn't ground. Now routes of all
  bridges go in an order that doesn't depend on orientation (faces, cells, distances to the
  projector), a tie group at a time, a group only if its bits and its shared cells' fit; a cell
  routes of one bridge cross is a floor while any of them is on; one two bridges' routes cross
  isn't claimed.
- **Any two devices on one cell or face** (same function). Every floor (`stand`) or wall
  (`blocks`) a device made went to the first device in item order (coordinate order), or for
  some walls the last. Now every claim is kept and settled at the end: one condition stays, an
  always wins, and several one-bit conditions become a new bit that is on while any of them
  holds (`Bit::Any`, worked out after the others in `device_bits`); only conditions of more than
  one bit, or bits short for all of them, keep the first claim and mark the model incomplete.
- **An entry door onto a faith plate** (`solver.rs`, `region_of_door`): the door's cell
  launches whoever enters, so it is no region, and the level was "invalid" unless a bridge
  route happened to claim the cell. Now the player starts where the plate lands them. (Sample
  341 then solves in 4 steps whose last, a fall back into a portal moving the other one, doesn't
  ground: the mid-fall portal move, not orientation.)
- **Turret lines of sight through a voxel corner** (`turrets.rs`, `clear_line`): a sample landing
  exactly on a corner (`floor`) was put in the cell above it on each axis, so the line was
  checked past one side cell or the other depending on the orientation. Such samples are now
  skipped, so the crossing is taken as one move on two axes, either order. Sample 582: 8 steps
  in six orientations, 9 in two.

On the 600: versions whose label differs from their level's 18 → 2 (one level, solved at 50k in
some orientations and out of budget in others; all solve in 11 at 500k). 18 of the 600 levels'
own labels changed: 4 uncertain → solvable, 14 uncertain → unsolvable (their models were
incomplete only because of the first-wins claims, and an incomplete model never says
unsolvable). Left: search order near the budget, and claims that stay first-wins as above.
Tests: `tests/augment.rs` (`bridge_routes_and_doors_dont_depend_on_orientation`, `turret_sight_doesnt_depend_on_orientation`).

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
