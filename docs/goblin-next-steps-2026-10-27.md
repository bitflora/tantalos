# Goblin: next steps (written 2026-10-27)

Goblin is the net that reads a level and writes its solution as plan tokens (it was called the
solver net until 2026-10-27; "the solver" alone means the Rust BFS solver). Why and what for:
docs/goblin-plan-2026-10-27.md. This file is the to-do list, in order, written so each step can
be followed exactly.

Already done and committed (e7d9848 .. 7670dfa): the rename, plan words for every solver action,
plans that move with mirrored/turned levels, fast padded sampling, alternative plans, the data
builder `train/goblin_data.py`, the new eval sets in `train/goblin_eval.py`,
`harvest.py --keep-unsolved`. A smoke run (`runs/goblin-smoke`) went through the whole pipeline.

## Rules for whoever does these tasks

- Read `CLAUDE.md` first and follow it. Use `.venv/Scripts/python`. Run commands from the repo
  root, `C:\Users\timja\code\glados`.
- Do the tasks **in order**. Each says what "done" looks like: check it before going on.
- **Ask the user before each task marked (ASK)**, and wait for a yes in the conversation. Those
  tasks hold the CPU or GPU for hours.
- Before any heavy job, check nothing else is going. This must print `runs: [] broadcasts: []`:
  `.venv/Scripts/python -c "import sys; sys.path.insert(0, 'train'); import run; print('runs:', run.running(), 'broadcasts:', run.broadcasts())"`
  and no data build or harvest of yours may still be running. Only one heavy job at a time (data
  build, harvest, training). While one runs, don't `cargo build`.
- Run long commands with the Bash tool's `run_in_background`, writing a log under `out/`. You
  are told when they end: don't poll, don't sleep.
- Don't touch: `src/bin/reverse.rs`, `train/runs.csv` and `train/goblin-runs.csv` by hand, the
  Portal 2 game, the Puzzle Maker folder.
- If a command fails, or a number is far from what the task expects (the task says how far),
  **stop and report** the output to the user. Don't improvise a fix.
- Commit each finished task on its own (only the files that task changed). `data/`, `out/` and
  `runs/` are not in git; don't add them.
- Report numbers as the scripts print them. Don't round a failure into a success.

## Task 1 (ASK): build data/goblin without the harvest

Why: the supervised data for the first Goblin run. CPU on every thread for several hours.

1. Check nothing heavy is running (rules). `cargo build --release --bins` (must succeed).
2. Start, in the background:
   `.venv/Scripts/python train/goblin_data.py --deep > out/goblin-build-2026-10-27.log 2>&1`
   It plans the parts `ws`, `ws-test`, `ws-deep`, `mut`, `gen` (it prints "v6: no harvested levels
   yet ... skipped": expected), then assembles and prints BUILD.md.
3. When it ends, read the end of the log.

Done when all of these hold (stop and report otherwise):
- No Python traceback in the log; `data/goblin/train.jsonl`, `test-ws.jsonl`, `test-gen.jsonl`,
  `test-mut.jsonl`, `test-hard.jsonl`, `test-frontier.jsonl`, `probe.jsonl` and `BUILD.md` exist.
- The `ws:` counts line says `planned` around 2,600-2,900 (of about 3,000 levels).
- The `ws-deep:` line says `planned` somewhere between 500 and 1,500 (measured 11.5% of 8,089 on a
  sample, so about 930).
- `ws-test:` planned about 380-400.
- BUILD.md: train.jsonl has 200,000 or more rows; `ws` rows are the largest or second largest
  source; the 12+ bucket is at least 18% after balancing; test-frontier has 512 levels;
  probe has at least 1,000 levels.
- Write down BUILD.md's line "Workshop rows longer than a 1024 block": Task 4 needs it.

## Task 2 (ASK): harvest levels from the level net, then plan them

Why: test-v6 (half of the headline score) and training rows in the level net's style, plus the
first self-play pool. GPU (with some CPU) for hours.

1. Check nothing heavy is running. Time a small harvest first:
   `.venv/Scripts/python train/harvest.py runs/v6-fresh-12h/away/best.pt --n 2048 --chunk 1024
   --cond "<verdict:solvable> <src:ws>" --data "data/v5b/levels-*.jsonl" data/ws5/train.jsonl
   --out data/goblin/harvest/v6-a.jsonl --keep-unsolved data/goblin/pool/v6.jsonl`
   Note the seconds in its last progress line ("2048 sampled in Ns"). Expect about 5% kept (~100
   levels) and about 5% written to the pool. If under 2% kept, stop and report.
2. Work out N = the number of levels it samples in 6 hours (2048 × 21600 / seconds, rounded down
   to a multiple of 1024). Tell the user N and the expected kept count (5% of N), then run the
   same command in the background with `--n N --out data/goblin/harvest/v6-b.jsonl` (same
   `--keep-unsolved` file: it appends). Log to `out/goblin-harvest-2026-10-27.log`.
3. When it ends: `.venv/Scripts/python train/goblin_data.py > out/goblin-build2-2026-10-27.log 2>&1`
   (background). This plans the new `v6` part and assembles again; the other parts are kept.

Done when: the second build's log shows a `v6:` line with `planned` close to the harvested count;
BUILD.md lists a `v6` source and `test-v6` with a few hundred levels (1 in 20 of v6 levels).

## Task 3: check the data

No heavy job, a few minutes.

1. Every training plan must check. Run this, which checks 2,000 random train rows:

   ```bash
   .venv/Scripts/python -c "import json,random,subprocess,pathlib;g=str(pathlib.Path('target/release/glados-score.exe').resolve());rows=[json.loads(l) for l in open('data/goblin/train.jsonl',encoding='utf-8')];s=random.Random(1).sample(rows,2000);inp=''.join(json.dumps({'tokens':r['tokens'],'plans':[r['plan']]+r.get('alt_plans',[])})+'\n' for r in s);o=subprocess.run([g,'check'],input=inp,capture_output=True,text=True).stdout;v=[x['verdict'] for l in o.splitlines() for x in json.loads(l)['results']];print(len(v),'plans,',v.count('solved'),'solved')"
   ```

   Done when at least 99.5% are solved. Otherwise stop and report.
2. A same-set baseline: evaluate the old model on the new sets, without touching its own eval:
   `.venv/Scripts/python train/goblin_eval.py runs/solve-fresh-3h/best.pt --plans data/goblin --out runs/solve-fresh-3h/eval-goblin`
   (about 15-25 minutes, GPU: check nothing is running first). Write down from its report.md
   the solved % for test-ws+v6, test-ws, test-v6, test-gen, test-hard and the probe. These are
   the numbers Task 4's run must beat.
3. Add both results (the check share and the baseline numbers) to the "Done" section of
   docs/goblin-plan-2026-10-27.md and commit that file.

## Task 4 (ASK): the first Goblin run

Follow CLAUDE.md's "Training runs" and "Goblin runs" sections exactly; this is step 1 there.

1. If Task 1's "longer than a 1024 block" share is 10% or less:
   `.venv/Scripts/python train/run.py start goblin-v6init-12h --task goblin --from v6-fresh-12h --hours 12 --note "First Goblin run on data/goblin (Workshop-first, 8-way versions with plans, alternatives, primers), started from the v6 level net's weights" -- --layers 10 --batch 8 --accum 2`
   If it is over 10%, add `--block 1280` and use `--batch 4 --accum 4` instead of
   `--batch 8 --accum 2`, and say so in the note.
2. `.venv/Scripts/python train/run.py wait goblin-v6init-12h` with `run_in_background`. Then do
   every NEXT STEPS item it prints. If it ran out of memory in the first minutes (the log says
   "out of memory"), stop and report; don't retry with other settings unasked.
3. Gate. Compare its eval with Task 3's baseline on the same sets. It passes if it is more than
   5 points above the baseline on test-ws+v6 **and** on test-gen, and test-hard is above 0%.
   Tell the user the numbers and whether it passed. Record with `run.py note`.

Done when the run is recorded and the user has the gate result. **If it failed, stop here**: report
the report.md's failing-step counts and the probe table, and wait for the user.

## Task 5 (ASK, only if Task 4 passed): self-play, one round by hand

Why: the first stage of Goblin learning from its own solutions (expert iteration). Build a new
script `train/school.py` with one subcommand, `round`. Keep the code in the style of
`train/harvest.py` and `train/broadcast.py` (docstring at the top that says how to run it, small
functions). Reuse, don't copy: `goblin_eval.plan_prompt`, `goblin_eval.generate_plans`,
`goblin_eval.check`, `sample.load_checkpoint`, `goblin_data.augment`, `goblin_data.identity`,
`data.PLAN_WORDS`.

`.venv/Scripts/python train/school.py round NAME --model CHECKPOINT --round R [--levels 4096] [--k 32]`
writes `runs/school-NAME/rRR/` (RR = two digits) and does, in order:

1. **Pool.** Candidates: `data/goblin/pool/*.jsonl`, plus the records of `data/ws5/train.jsonl`
   whose `solver_verdict` is `uncertain` (never `data/ws5/test.jsonl`; drop any record whose
   `identity` is in a test set of data/goblin). Keep a state file `runs/school-NAME/levels.jsonl`
   with one line per level tried: `tokens`, `identity`, `tries`, `solved_rounds`, `last_round`.
   Draw `--levels`: half from levels tried before and not solved in every try (prefer ones solved
   in some tries; a never-solved level only every 2^tries rounds), half never tried. Save the
   draw as `pool.jsonl`.
2. **Attempt.** For each level: one greedy plan (top_k 1) and `--k` samples (temperature 1.0),
   through `generate_plans` (rows 64). Check all with `check(items, mode="conservative")`.
3. **Prefix resampling** (twice) for levels with no solved plan: take the plan with the largest
   `steps_ok`, cut it to its first `steps_ok` steps (a step starts at a verb: `walk fire enter
   pickup putdown toss throw press dash knock bounce leap plate climb funnel ride dive fetch reach
   loop grab double sweep exit`; numbers, colours and directions are arguments), prompt
   `[BOS] + level + [SEP] + prefix`, sample 16 continuations, and check prefix + continuation.
4. **Liberal.** For levels still unsolved, check all their plans with `mode="liberal"`; any that
   solve go to `liberal-queue.jsonl` (tokens, plan). They are **not** training data until Portal 2
   has played them (a later task, with the user).
5. **Polish.** For each conservatively solved plan, try removing each step in turn; if
   `check` still says solved, keep the shorter one and repeat until no step can go. Keep at most
   4 distinct plans per level, preferring ones whose sets of verbs differ.
6. **Buffer.** Append to `runs/school-NAME/buffer.jsonl` one row per newly solved level:
   `tokens`, `plan` (shortest), `plans`, `alt_plans` (empty), `min_len` (steps of the shortest),
   `src: "school"`, `identity`, `found_by: "goblin"`, `round`, `trusted: "check"`. A level already in
   the buffer gets its row replaced only if the new plan is shorter.
7. **Round summary**: `summary.json` and a printed table: levels drawn, solved greedy, solved by
   any sample, by prefix resampling, liberal-only, newly solved, solved levels the Rust solver
   calls uncertain ("beyond BFS"), the time each step took.

Then check, by hand, on a small round first:
`.venv/Scripts/python train/school.py round try --model runs/goblin-v6init-12h/best.pt --round 0 --levels 256 --k 8`

Done when: the small round runs without errors; every buffer plan checks as solved when run
through `glados-score check` again; `summary.json` numbers add up (solved ≤ drawn); a second
`round try --round 1` draws a different half of new levels and updates `levels.jsonl`. Commit
`train/school.py`. Report the summary table to the user and stop.

## Later (don't start without the user)

In docs/goblin-plan-2026-10-27.md, "Next" 3: the full school loop (rounds with `--quick`
training on the base data plus the buffer, consolidation runs every 5 rounds), Portal 2 checks
of the liberal queue (E3), constrained decoding through a `glados-score env` server (E2), a value
head and tree search (E4), and co-training the level net (E5). Each needs its own plan agreed
with the user first.
