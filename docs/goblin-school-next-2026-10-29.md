# Goblin self-play: prefix speed-up (done) and the next steps (2026-10-29)

Branch: `goblin-env`. Commit only when the user asks.

## Done: prefix resampling made fast

The try round `runs/school/try/r00/` (256 levels, k=8) took 7,949 s, 7,184 s of it in prefix
resampling, for 1 level. With constrained decoding every step is legal, so the "longest checked
prefix" was the whole failing plan (about 254 tokens, nearly all plans run to the 256 cap), and
each try wrote 256 more. On big levels env then thrashed: one level took 6 GB and over 9 minutes
for 16 plans.

- `train/school.py`: `best_prefix(lv, most)` cuts the prefix to whole steps within
  `PREFIX_WORDS` = (128, 64) words, for passes 1 and 2.
- `train/goblin_eval.py`: `LegalSteps(limit=MAX_CONSTRAINED)` counts the prefix in the 256 cap.
- `src/env.rs`: `forget(keep)` drops the least recently used nodes only, `next` rebuilds from the
  deepest node kept, `MAX_CACHED` 200,000 states (a state took about 6 kB on that level). The
  prefix cut alone fixed the speed; this bounds memory.

Result, `runs/school/verify/r00/` (256 levels, k=8, seed 2733104312; a different draw from
try/r00, since probe levels are now excluded): 1,086 s in all (attempt 265, prefix 441,
liberal 11, polish 359). 26 levels solved, all beyond the Rust search, 4 by prefix resampling.
All 63 buffer plans re-check as solved; no test or probe identity in the buffer.

## After the speed-up: the order of work

Each step needs the one before it. Ask the user before every step marked **ASK**.

### A. Finish Task 5's acceptance checks

These are in docs/goblin-next-steps-2026-10-27.md, Task 5 "Done when". **All passed
(2026-10-29).** `verify` r01 (seed 109745903): 256 new levels, none from r00; 17 solved (3 by
prefix), 1,296 s; levels.jsonl 512 rows; buffer 43 levels, all 100 plans re-check as solved, no
test or probe identity. r01 drew no tried levels: r00's unsolved ones wait 2^tries rounds.

1. Run the unit tests:

   ```bash
   .venv/Scripts/python -m unittest discover -s tests -p test_school.py -v
   ```

   They must pass. Run them before the round as well.
2. Recheck every plan in `runs/school/verify/buffer.jsonl`:
   - one item per row: `{"tokens": row["tokens"], "plans": row["plans"]}`;
   - call `goblin_eval.check(items, mode="conservative")`;
   - every verdict must be `solved`.
3. Check that no `identity` in the buffer appears in `data/goblin/test-*.jsonl` or
   `data/goblin/probe.jsonl`.
4. Run the second draw (a different half of new levels):

   ```bash
   .venv/Scripts/python train/school.py round verify --model runs/goblin-v6init-12h-b/best.pt --round 1 --levels 256 --k 8 > runs/school-verify-r01.log 2>&1
   ```

   Then:
   - r01's `pool.jsonl` must share no new levels with r00's;
   - `runs/school/verify/levels.jsonl` must have grown;
   - repeat checks 2 and 3.
5. **ASK**, then commit `train/school.py`, `train/goblin_eval.py` and this doc on `goblin-env`.

### B. One full self-play round (**ASK**: it holds the GPU for several hours)

The buffer after A is a few dozen levels. That is far too few to train on: data/goblin's
train.jsonl has 223,141 rows (counted 2026-10-30). The plan (goblin-plan "E1") is rounds of 4,096 levels
with k=32:

```bash
.venv/Scripts/python train/school.py round e1 --model runs/goblin-v6init-12h-b/best.pt --round 0 > runs/school-e1-r00.log 2>&1
```

Estimate: the attempt step alone is about 4,096 x 33 plans at about 0.09 s each, so roughly 3.5
hours. Prefix and liberal checking add to that.

- It saves after every chunk. If the user wants the machine back, stop it, and resume later with
  the same command.
- Report the summary table: solved levels, beyond-BFS, buffer size, time per step.

### C. Guarded training on the buffer (**ASK**: needs new code first)

Implemented 2026-10-30: `school.py export` checks the original buffer, exports each plan
with four rotations, preserves identity, and checks the transformed plans before publishing.
`run.py --constrained-eval` uses full constrained evaluation even with `--quick`, reuses
the parent's evaluated levels, and retains the parent if the guard lacks a matching baseline.
The approved first experiment is 30 minutes, roughly 20% buffer (export supplied ten times),
80% base data, from `goblin-v6init-12h-b`. The earlier 30-50% range is deferred.

1. Write `school.py export NAME --out data/school/NAME.jsonl`. It writes the buffer's plans as
   training rows, in the same row format as `data/goblin/train.jsonl`. Each plan becomes x4
   symmetric copies, with `goblin_data.augment` (it calls `glados-data augment --plans`).
2. Train with `run.py`, never `train.py` directly:
   - a `--task goblin --quick --from goblin-v6init-12h-b` run;
   - `--data` naming `data/goblin/train.jsonl` and the export;
   - buffer export ten times alongside one copy of the base data; measure the effective
     share after vocabulary/length filtering and the identity split.
3. Add `--constrained-eval` to `--quick`: compare the same headline levels, constrained
   settings and parent checkpoint, retaining the parent on a drop greater than five points
   or a missing/incompatible baseline. A passed guard does not establish improvement.
4. Judge the result by the constrained `test-ws+v6` headline (57.3% before), the probe and
   `test-frontier` (3/256, 1.2% greedy before). Report solved counts and paired changes;
   five points is a rough headline guideline, not a universal significance threshold.

Round e1/r00 completed cleanly: 421/4096 levels, 1,355 checked plans, 24,022 seconds.
No test/probe identity was in the pool or buffer. The k=8 verification rounds used different
pools, so their similar solved shares do not establish that k=32 adds no value.

Pilot `goblin-school-r00-b` started 2026-10-30 from `goblin-v6init-12h-b`, 0.5 h,
quick learning rates, batch 8 / accumulation 2, constrained evaluation on the parent's
saved evaluation levels. Export: 5,370 checked rows, all 421 identities retained.
Augmentation rejected 42 transformations; independent replay rejected eight more
(saved in `data/school/e1.rejected.jsonl`). No rejected transformation enters training.
After loader filtering and identity split: 212,142 base training rows plus 5,191 school
rows repeated ten times, giving **19.66%** school. Validation: 10,925 base plus 179 school
rows repeated ten times. Audit: `runs/school/e1/training-mix.json`.
Seventeen school/export/guard/reference tests passed.

Result (2026-10-30, guard passed, pilot retained): headline 57.8% vs 57.3% (569 vs 564 of 984),
pass@8 81.8 vs 82.3, frontier greedy 4 vs 3 of 256. Paired greedy flips, pilot wins/losses:
test-gen 108/70 (sign test p ~ 0.004, the one real change), test-ws 25/17, test-v6 66/69,
probe 283/297, test-hard 2/3: noise. Fits the buffer's make-up: 304 of the 421 solved levels
were the level net's, 117 Workshop. Prefix resampling took 7,429 s of 24,022 for 24 levels.
Next: replay the buffer's plans in Portal 2 (`train/stage_school_play.py`, `play.py --plans`)
before training on more of it, then E1 round 1 from the pilot.

Replay (2026-10-30, paused at 92 of 109 staged levels; 89 more of the 198 picked can't be routed
yet): 6 clean, 25 recovered, 50 failed, 11 didn't play. The user played five failures
(`plan?`, careful retry failed too) by hand, out/school-play/e1-r00/PLAYTEST.md; all solvable:
- 00022 (ws 68526134): the faith plate can't be used; jump to the button from a higher
  platform. The plan doesn't say how to reach the button.
- 00048 (ws 98107591): plan viable; the harness probably found no angle to shoot the ceiling.
- 00099 (ws 3423083482): plan viable; the laser must destroy the turrets in its way before the
  door opens, a long delay the harness doesn't wait for.
- 00105 (ws 3769193216): needs timed portal placement to reach a moving platform, then moving
  the portals and walking through quickly.
- 00172 (gen): plan good; probably the harness didn't duck under a laser through the portal.
So the failures are mostly harness gaps (shot angles, waits, timing, lasers), not exploited
model gaps; a failed replay is not evidence that a buffer plan is wrong.

Three of those gaps are closed in the harness (2026-10-31, train/play.py, tests/test_play.py):

- **A shut exit door is waited at** for `--exit-wait` seconds (40 by default) while still
  walking into it, which covers a door that opens late: the laser burning through the turrets
  in its way (00099), a platform arriving. A timed escape (`dash`) doesn't wait, since its door
  shuts again and the time is the point.
- **A laser beam across the way is ducked under or jumped**, by how high the emitters put it
  (`glados_lasers`, `beam_over`): over 40 units above the feet it is ducked under (crouched the
  player is 36 tall), lower it is jumped as before. Burnt again by what was tried, the other
  way is tried once. A beam bent by a reflection cube or sent through a portal is still not
  found, and then the jump comes first, as before (00172).
- **A face whose middle is out of sight is shot at off-centre**: `clear_aim` traces to the
  middle and then to four points 32 units off in the face's plane, and fires at the first the
  shot reaches (00048). A deployed panel's own aim point stands outside the face, so it stays.

Left alone: 00105 (timed portal placement onto a moving platform) and 00022, where the plan
itself doesn't say how the button is reached, so there is nothing for the harness to play.

What they were worth, replayed (2026-11-01, out/school-play/e1-r00/retry-play: the 50 levels of
round 0's replay that had a failed step, played again with the fixes, all 50 played):

- 7 levels came out with no failed step, 10 with fewer, 31 unchanged, 2 worse.
- **Ducking is the win.** 276 beams were met: 142 ducked first, 78 jumped first (146 of them with
  a height read from the emitters), 180 ducks in all, 56 burnt both ways. Laser jumps fell 284 to
  113, deaths 114 to 83, hurt 306 to 280. 00172, the level the user played by hand, is clean.
  Of the 7 clean levels 4 logged ducks; 00061, 00067 and 00107 came clean with no new mechanism
  at all and 2 levels got worse, so run-to-run noise here is about plus or minus 3 levels.
- **The exit-door wait earned nothing.** 18 levels waited, 36 waits of 40 s, and no door opened.
  00099's four steps all passed and its door stayed shut for 80 s, so the hand-play guess (a long
  delay while the laser burns through the turrets) is not what stops it: something keeps the laser
  off the turrets, and that is a solver or converter question, not a harness one. The wait is now
  10 s by default and the careful retry doesn't wait at the same door a second time.
- **Off-centre aiming hardly fires.** Of 142 fire steps 1 aimed off-centre (and that step still
  failed), while 22 found nothing in sight and fell back to the middle, as before: those are most
  likely shots through a portal, which no trace from the eye can reach. 00048, the ceiling shot,
  wasn't in the retry set (its old run had no failed step), so that case is still untested.

The 17 staged levels round 0 never reached were then played with the fixes (2026-11-01, the
same out/school-play/e1-r00/play folder, resumed; that run still had the 40 s wait): 7 clean, 7
with a failed step, and 3 that never played at all — the editor marks 00181 and 00188 invalid (a
turret, a flip panel) and 00196 didn't build, which is a converter or editor-rule gap, not the
model's. 9 of them ducked (56 ducks, 29 jumps). Six waited at a shut exit door and no door opened
there either, so over both runs 24 levels waited and none got out that way; 00184's careful retry
then got out on its own, without waiting. Turning the wait off by default would cost nothing that
has been measured.

Where the 109 staged levels stand, counting each level's latest run: 45 with no failed step, 50
with one, 14 that never played. A failed replay is still not evidence that a buffer plan is
wrong, but after the duck fix it is better evidence than it was.

Next: out/school-play/e1-r01-2755 (ws:132677917, staged but unplayed), 00048 on its own for the
ceiling shot, and the 14 that never played — those are worth a look as converter bugs.

The 14 that never played, looked at offline (2026-11-02). 12 the editor marked invalid, not 2:

- 8 Workshop levels (00038, 00046, 00062, 00064, 00108, 00122, 00129, 00131): our own rules flag
  every one of them too (fields overlapping or across a door, plate targets in the air, rails
  sharing cells). The real maps passed the editor, so these are converter output, and the stager
  should have left them out: `glados-p2c editor` on the staged p2c says so before the game does.
- 4 level-net levels our rules let through, three gaps in src/editor.rs, now closed and each
  scored against every stored in-game check (74,053 levels, runs/*/eval/ingame): a loose cube on
  a turret (00143, 00181; 66 of 66 flags), a door with an open voxel in the wall it is sunk into
  (00174; 476 of 476), an item sunk into the wall behind a rail's track (00188, flip panels on the
  floor above it; 144 of 144). Together 516 more of the game's flags, and no flag the game didn't
  give; door misses fell from 907 to 482. On 4,000 ws5 train levels they newly flag 1 (a rail),
  on 4,000 data/v6 generated levels 29 (0.7%, nearly all doors). Since the grammar keeps items
  clear of these rules, constrained decoding now avoids them too; levels labelled before keep
  their old labels (not rescored).
- 00196 (level net, 105 gel droppers) didn't build: very likely VBSP's entity limit (see the
  "too many items" note), not a placement rule. 00170 (level net, 27x27x25, two funnels and a
  laser) took the game down twice on load; only the game or its crash dumps can say why.

00099 can't be settled offline. The plan sends the laser through a portal and up the x=15 column
to the catcher, through 18 turrets in a row. src/lasers.rs assumes a beam destroys every turret in
its cells at once and reaches the catcher; by hand the door opened only after a long delay, and
not within the harness's 80 s.
The user confirmed it (2026-11-02): a laser takes several seconds to destroy each turret. So the
plan is right and the level solvable; only the time is missing. In the solver that matters only
where a beam is on for a moment: steady beams wait in a settled state (their turrets go down in
`settle_devices`), and timed dashes don't recompute beams at all. A funnel sweep's beam now stops
at a turret still standing (src/solver.rs `beam_hits_with`; tests/playtested.rs, the sweep test
with a turret in the swept beam's way). Left for the harness: play.py waits 10 s at a shut exit
door, which can't cover 18 turrets; it would need the number of turrets a step's beams burn down
(from `route`) and wait several seconds for each.

Round e1/r01 (2026-10-31, from goblin-school-r00-b, `--no-prefix`): 391/4096 solved (142
greedy, 249 by a sample), all beyond BFS; 31 liberal only; buffer 812 levels, 1,256 plans.
Attempt 18,382 s. Polish hung 5 h on one 27x26x25 Workshop level (ws:132677917, ~50-step
plans; staged as glados-school-r01-02755); polish now stops after POLISH_SECONDS (300) a level.

Paused 2026-10-31 by the user, to resume: export both rounds (writes only at the end; it was
stopped part way, nothing written), then a 2-hour guarded run from goblin-v6init-12h-b with the
pilot's settings (`--task goblin --quick --constrained-eval --batch 8 --accum 2`), the export
repeated to keep the school share near 20% (about x5 now, measure with audit_mix.py):

```bash
.venv/Scripts/python train/school.py export e1 --out data/school/e1-r01.jsonl
```
The initial sandbox launch `goblin-school-r00` exited before creating a log; its status
is preserved. The replacement was launched outside the sandbox. Monitor it with an
unrestricted `run.py wait`: sandbox process visibility can falsely report DIED for
an external worker.

### D. In parallel with B or C (**ASK**): search against sampling

This is roadmap step 5. On the same levels and the same model, compare constrained best-of-N
sampling with a bounded step-level beam or best-first search, at equal wall-clock time. Record:

- solve rate;
- plan lengths;
- model and check calls;
- peak memory.

This decides whether a value head is needed. It is a separate piece of code. It must not run
while B holds the GPU.

Result `goblin-school-r01-2h` (2026-10-31, 2 h, rounds 0+1 export x5, 18.9% school, guard
passed): headline 58.9% (580/984) vs parent 57.3% and pilot 57.8%; pass@8 83.6 vs 82.3;
frontier greedy 4 vs 3 of 256. Paired greedy wins/losses vs the parent: test-gen 118/78
(p ~ 0.005, again the one real change), test-ws 27/23, test-v6 65/53, probe 328/291
(p ~ 0.15), test-hard 3/3. Four times the training and twice the buffer did not move Workshop or
frontier levels; the buffer is still mostly level-net levels and the beyond-BFS solves are ones
the model nearly had already. More of the same rounds is unlikely to pay off.

#### The answer (2026-11-01): sampling wins, and the beam's failure says what a value head is for

`train/goblin_search.py` on `runs/goblin-school-r01-2h/best.pt`, 60 s a level a method, on the
72 levels that run's greedy eval missed (24 each of test-ws, test-v6, test-frontier; seed 7,
`out/search/r01-60s/`, 126 minutes). Best-of-N sampling in rounds of 32 against a 4-wide beam
with 8 candidate steps a prefix, both constrained by `glados-score env`, both after a free
greedy plan, both ending rows inside a round when the clock runs out, both restarting rather
than returning the budget unspent.

| | sampling | beam |
|---|---|---|
| solved (72 levels) | 58.3% | 29.2% |
| beyond greedy | 41 of 71 | 20 of 71 |
| test-ws / test-v6 / test-frontier | 83.3 / 87.5 / 4.2% | 33.3 / 50.0 / 4.2% |
| mean s on the levels greedy missed | 33.5 | 73.0 |
| forward passes a solve | 2,267 | 5,910 |
| plan tokens a solve | 55.6k | 69.6k |
| check plans (all 72) | 10,224 | 152 |
| peak GPU memory | 1,089 MB | 1,107 MB |

Sampling solved a strict superset: **every level the beam solved, sampling solved too**, and it
solved 21 the beam did not. It is also cheaper a solve on every axis measured, and both methods
sit near 1.1 GB on the 4 GB card, so memory decides nothing. The beam's one real edge is plan
length: on the 20 levels both solved it wrote 12.7 steps against sampling's 16.8, shorter on 14
of the 20. Ranking by mean instead of total log probability was not tried at length; it would
only sharpen an edge that is not the one we need.

Why the beam loses is the useful part. Of its failures, almost none are dead ends — the
`why` counts read `no step 96–174, the clock 32, dead 0` on level after level, and only one
level reported any `dead` at all. The beam reaches a mean depth of 226 words (ceiling 256) on
the levels it fails: it does not get stuck, it **wanders to the token limit**. Its expansions
also collapse: 8.9 distinct children a round out of 32 sampled steps, so the real branching is
about 2 a parent. A confident policy proposes nearly the same step eight times, and total log
probability then ranks those near-identical children by nothing that correlates with progress.
The env says "dead" too rarely to prune, so a policy-only beam has no signal at all about which
branch is closer to the exit.

That is the finding the roadmap gate wanted: **bounded search over this policy, with no learned
leaf scoring, is worse than simply sampling more plans** — and the reason is missing leaf
evaluation, not missing search. The other half of the gap is that sampling gets 140 solver
verdicts a level while the beam gets 2: the conservative `check` is the only true progress
signal either method has, and sampling spends it freely while the beam asks for it only when
the env already says the plan is at the goal. So the value head stays gated behind rollout
labels (it has nothing to learn from yet), but when those labels exist this is what they buy,
and `train/goblin_search.py --methods beam` is the harness that will re-measure it.

Two caveats. The frontier is untouched by either method (1 of 24 each, the same level), so
nothing here says how to reach levels our BFS cannot. And one row is an outlier: the beam on
`ws:3386989029` took 1,758 s against a 60 s budget, stalled on 2 forward passes with the run's
lowest peak memory (349 MB) and 7 env asks, while the env replays that level's whole greedy
prefix instantly and sampling finished it in 61 s — a machine stall, not the level. The
deadline forces EOS between token steps, so it cannot preempt a single blocked forward pass.
Excluding the level entirely leaves the verdict unchanged.
