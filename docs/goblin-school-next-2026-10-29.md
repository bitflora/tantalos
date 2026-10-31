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

Three of those gaps are closed in the harness (2026-10-31, train/play.py, tests/test_play.py,
none of it tried in the game yet):

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

Next, with the user at the game: replay those five levels and the 17 of 109 staged levels that
round 0 didn't reach, plus out/school-play/e1-r01-2755 (ws:132677917, staged but unplayed), and
see which of the 50 failures the three fixes turn into clean runs.

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
