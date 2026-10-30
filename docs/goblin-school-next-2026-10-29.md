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
train.jsonl has about a million rows. The plan (goblin-plan "E1") is rounds of 4,096 levels
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

This is not built yet. `run.py` has no way to put the school buffer into a Goblin run.

1. Write `school.py export NAME --out data/school/NAME.jsonl`. It writes the buffer's plans as
   training rows, in the same row format as `data/goblin/train.jsonl`. Each plan becomes x4
   symmetric copies, with `goblin_data.augment` (it calls `glados-data augment --plans`).
2. Train with `run.py`, never `train.py` directly:
   - a `--task goblin --quick --from goblin-v6init-12h-b` run;
   - `--data` naming `data/goblin/train.jsonl` and the export;
   - buffer rows 30-50% of what it trains on. How to reach that share with a small buffer (by
     repeating it) is a decision for the user.
3. `--quick` evaluates without constraints and guards against a 5-point drop. This experiment
   needs the constrained eval: ask the user whether to add a flag for it to `run.py`.
4. Judge the result by the constrained `test-ws+v6` headline (57.3% before), the probe and
   `test-frontier` (1.2% greedy before). Differences under 5 points are noise.

### D. In parallel with B or C (**ASK**): search against sampling

This is roadmap step 5. On the same levels and the same model, compare constrained best-of-N
sampling with a bounded step-level beam or best-first search, at equal wall-clock time. Record:

- solve rate;
- plan lengths;
- model and check calls;
- peak memory.

This decides whether a value head is needed. It is a separate piece of code. It must not run
while B holds the GPU.
