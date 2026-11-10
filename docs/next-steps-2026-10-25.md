# Next steps (written 2026-10-25)

The goal: a level net that writes **varied levels using the mechanisms of highly rated Workshop
levels**. Hard rated levels change the state of the room over time (devices powered, lasers
redirected, panels deployed); generated levels lean on tricks (turret knocks, dashes, latches,
gel). Decisions already made by the user:

- Workshop levels are the main training corpus; well-rated ones are trusted as solvable.
- Levels whose Workshop text mentions bunny hopping or a needed glitch are left out.
- Keep some generated data: only levels whose solution changes the room's state.
- The reverse generator (`src/bin/reverse.rs`) is abandoned. Don't use it.
- The user plays 5-10 levels per training run (run.py stages a blind A/B set).

Already done: star ratings on Workshop records (`stars`, `votes`), mirror/turn augmentation
(`data/v5b/workshop-aug-x2.jsonl`, 129k distinct versions), BEEmod converter (not yet run on
maps), Workshop test loss, the rated-style evaluation (`train/rated.py`, baseline v5-cont-10h-d:
yield 207, strong 8, gap 23.5), val split by puzzle, blind A/B picks (`train/abtest.py`).

## Rules for whoever does these tasks

- Read `CLAUDE.md` first and follow it. Use the repo's `.venv/Scripts/python`.
- Do the tasks **in order**. Each says what "done" looks like; check it before moving on.
- Before overwriting any file under `data/`, copy it to `<name>.before-2026-10-DD` (today's date),
  unless such a copy already exists.
- Never start a training run (task 6) without the user saying yes in the conversation. Nothing
  heavy (downloads, builds, `cargo build`) while a training run or broadcast is going: check
  `.venv/Scripts/python train/run.py list` (it says "running now") first.
- Don't touch: `src/bin/reverse.rs`, `train/runs.csv` by hand, the Portal 2 game, anything in
  the Puzzle Maker folder except through the scripts named here.
- If a command fails or a number looks very different from what a task expects, **stop and
  report** the output to the user. Don't improvise a fix.
- Commit each finished task on its own (only the files that task changed), with a message that
  says what and why. `data/` and `out/` are not in git; don't try to add them.

## Task 1: leave bunny-hop and glitch levels out of the Workshop training file

Why: 82 ws5 levels need bunny hopping or a glitch (31 of them rated 4-5 stars). The list was
made from their Workshop descriptions and lives in an ignored folder.

1. Copy `out/workshop/excluded-tricks-2026-10-25.jsonl` to `tools/workshop/excluded-tricks.jsonl`.
   Each line has `id` (the workshop id), `stars`, `kinds`, `match`, `context`.
2. Remove the line whose `id` is `100189052` (a false positive: "Fixed ... exploit").
3. In `tools/workshop/augment.py`, skip every record whose `workshop_id` is in that file (read
   the ids once at start; print how many were skipped).
4. In `train/rated.py`, point `EXCLUDED` at `ROOT / "tools" / "workshop" / "excluded-tricks.jsonl"`.
5. Rebuild: `.venv/Scripts/python tools/workshop/augment.py` (backs up first, see rules).

Done when: augment.py prints about 70-80 skipped levels (only train levels are in its input), the
new `data/v5b/workshop-aug-x2.jsonl` has a few hundred fewer lines than before (~258k), and
`.venv/Scripts/python -c "import sys; sys.path.insert(0,'train'); import rated; print(len(rated.excluded_ids()))"`
prints 81.

## Task 2: a `<tier:best>` condition for 5-star levels

Why: the model can then be asked for levels like the best-rated ones. `train/rated.py` and
`train/abtest.py` already use `<tier:best>` when a model knows it.

1. In `train/data.py`, change `PROVENANCE["tier"]` to `["<tier:mid>", "<tier:top>", "<tier:best>"]`
   (append; don't reorder).
2. In `conditions()`, where the tier is chosen: a Workshop record with `stars == 5` gets
   `<tier:best>`; otherwise keep the existing subscribers rule (`>= TOP_SUBSCRIBERS` -> top, else mid).
   Update the comment above `PROVENANCE` to say so.
3. Check with a short script: run `conditions()` on the first 2,000 records of
   `data/ws5/train.jsonl` and count the tier tokens.

Done when: the count shows roughly 9-10% `<tier:best>` (5-star levels are ~10% of ws5), and
`.venv/Scripts/python -m py_compile train/data.py` passes. Note in the commit message that the
vocabulary grew by one token (warm starts still work: train.py maps tokens by name).

## Task 3: the generated levels worth keeping

Why: ~500k generated levels mostly teach tricks, and 66% are near-copies of each other. Keep
only solvable ones whose solution changes the room's state, without near-copies.

Write `data/v6/select.py` (data/ is ignored by git, so also save a copy as
`tools/select_generated.py` and commit that). It reads `data/v5b/levels-1.jsonl` ...
`levels-5.jsonl` and `data/v5b/levels-mutants*.jsonl` and writes `data/v6/generated.jsonl`:

1. Keep a record only if `verdict == "solvable"` and, with
   `state, tricks = rated.mechanics(record["metrics"])` (import `train/rated.py`),
   `state >= 2 and tricks <= 1`.
2. Drop exact duplicates (same `tokens`).
3. Drop near-copies: group records by their first four tokens (`size X Y Z`); within a group,
   keep a record only if its token 4-gram Jaccard similarity to every record already kept in
   that group is below 0.9 (`train/evaluate.py` has `shingles` and `jaccard`; reuse them).
4. Print counts after each step, by source file.

Done when: `data/v6/generated.jsonl` exists, step 1 kept roughly 15-25% of the solvable records
(about 53k of 291k from levels-1..5 when this was measured), and the printout is shown to the
user. If step 3 runs longer than about 30 minutes, stop and report.

## Task 4: more Workshop levels from BEEmod maps

Why: the converter now reads BEEmod maps that use only vanilla items (~830 maps, commit
`210ca17`). More distinct Workshop levels is the best cure for memorization.

Check first that no training run is going. Then, one at a time:

```bash
.venv/Scripts/python tools/workshop/fetch.py download --rescreen --workers 8
.venv/Scripts/python tools/workshop/fetch.py build --incremental --budget 500000
.venv/Scripts/python tools/workshop/fetch.py finalize
.venv/Scripts/python tools/workshop/fetch.py ratings --annotate-only
.venv/Scripts/python tools/workshop/ws3.py --src out/workshop/levels.jsonl --out data/ws5
.venv/Scripts/python tools/workshop/fetch.py ratings --annotate-only
.venv/Scripts/python tools/workshop/augment.py
```

(Back up `out/workshop/levels.jsonl`, `data/ws5/` and the aug file first. `ratings
--annotate-only` runs twice because ws3.py rewrites data/ws5.) `fetch.py download` and `build`
can take an hour or more; run them in the background.

Done when: `data/ws5/train.jsonl` has more lines than before (16,214), `data/ws5/test.jsonl`
still has all its old workshop ids (the split keeps old levels where they were: check with a
script), and `data/v5b/workshop-aug-x2.jsonl` was rebuilt. Report the before/after counts. If
the test file lost ids, stop: the hold-out must not change.

## Task 5: assemble data/v6

1. `mkdir data/v6`; copy `data/v5b/vocab.txt` to `data/v6/vocab.txt`.
2. Link or copy `data/v5b/workshop-aug-x2.jsonl` to `data/v6/workshop-aug-x2.jsonl`.
   (`data/v6/generated.jsonl` is there from task 3.)
3. Dry run: `.venv/Scripts/python train/run.py start v6-check --hours 1 --note check --data "data/v6/*.jsonl" --dry-run`
   (prints the commands, starts nothing).
4. Count lines per file and the share of Workshop vs generated.

Done when: the dry run prints a train command whose `--data` lists exactly the two v6 files and
includes `--holdout ...data\ws5\test.jsonl`, and Workshop lines are roughly 75-85% of the total.
Report the counts to the user.

## Task 6: one fresh training run (ask the user first)

Ask the user: "data/v6 is ready. Start a fresh 10-hour run on it now?" Only on a yes:

```bash
.venv/Scripts/python train/run.py start v6-fresh-10h --hours 10 --data "data/v6/*.jsonl" --note "Fresh 10 layers on data/v6: augmented Workshop levels (no tricks) + state-changing generated levels; <tier:best>" -- --layers 10 --batch 8 --accum 2
```

Fresh, not `--from`: the data and vocabulary changed, and chained warm starts multiplied the
Workshop repeats. Then follow CLAUDE.md's steps 2-3 (`wait` in the background, then every NEXT
STEPS item).

Done when: the run finished and you reported the RATED STYLE line (yield, strong, profile gap),
the Workshop test loss, and EVAL solvable %, each next to the baseline (v5-cont-10h-d: yield 207,
strong 8, gap 23.5, Workshop test loss ~1.59, EVAL 32%), and the blind A/B level names.

## Task 7: the user's verdicts

When the user says how the `tantalos-ab-v6-fresh-10h-NN` levels played: follow CLAUDE.md
("When the user reports how the A/B levels played"): fill `runs/v6-fresh-10h/playtest/results.csv`,
run `abtest.py score`, report, commit `train/playtests.jsonl`.

## Not for delegation (needs judgement)

- Why today's generator makes much narrower levels than the one that made data/v5 (lasers in
  6% of levels against 34% at the start of a v5 run). Check before generating any new data.
- The ~4,600 maps that failed on three cheap converter gaps (converter session's notes) and
  wall buttons (a token-format change).
- Upweighting 4-5 star levels in training: only after task 6 shows Workshop test loss falling.

## Progress log

(Being filled in as the work happens.)

### Task 1 — done (2026-10-25)

- Copied `out/workshop/excluded-tricks-2026-10-25.jsonl` to `tools/workshop/excluded-tricks.jsonl`
  and dropped id `100189052` (81 lines left).
- `tools/workshop/augment.py` now reads those ids once and skips those levels; `train/rated.py`'s
  `EXCLUDED` points at the tracked copy.
- Backed up `data/v5b/workshop-aug-x2.jsonl` to `.before-2026-10-25`.
- Rebuild: 69 of 16,214 levels skipped (the other 12 excluded ids are in the test split / not in
  train), 128,602 versions x2 = **257,204 lines** (was 258,308).
- `rated.excluded_ids()` prints **81**.
- Commit: see `git log` (`Skip bunny-hop/glitch Workshop levels in training data`).

### Task 2 — done (2026-10-25)

- `PROVENANCE["tier"]` is now `["<tier:mid>", "<tier:top>", "<tier:best>"]`; `conditions()` gives
  5-star Workshop records `<tier:best>`, the rest keep the `subscribers` rule.
- `py_compile train/data.py` passes. Over all 16,214 ws5 train records: 1,560 `<tier:best>`
  (9.6%), 4,737 `<tier:top>`, 9,917 `<tier:mid>`. **The suggested first-2,000 check shows 25.6%**
  because `data/ws5/train.jsonl` is ordered best-rated first; the whole file is the meaningful
  number, and it matches the ~10% expected.
- Commit: `Add a <tier:best> condition for 5-star Workshop levels`.

### Task 3 — done (2026-10-25)

- Wrote `data/v6/select.py` (copy: `tools/select_generated.py`) and produced
  `data/v6/generated.jsonl`, 78,900 records.
- Step 1 kept 116,916 of 567,728 solvable records (20.6%). From levels-1..5 alone it is 38,289
  of 283,912 (13.5%), **below the ~53k of 291k the task expected**; those files were regenerated
  today at 10:48-13:12 (the `.before-2026-10-25` copies hold 53,509 of 319,449 = 16.8%, exactly
  the measurement the task quotes). Same filter, newer data.
- Step 2: 116,916 left, no exact duplicates. Step 3: 78,900 left (33,016 near-copies dropped),
  277s (well under the 30-minute limit). Per-source counts are in the run's printout.
- Commit: `Select the generated levels worth keeping (data/v6/generated.jsonl)`.

### Task 4 — in progress (2026-10-25)

- No training run or broadcast running when this started (`run.py list` shows none).
- Backed up `out/workshop/levels.jsonl` (30,287,772 bytes), `data/ws5/train.jsonl` (16,214 lines)
  and `data/ws5/test.jsonl` to `*.before-2026-10-25`; `data/v5b/workshop-aug-x2.jsonl` already
  has its `.before-2026-10-25` copy from task 1.
- Started `fetch.py download --rescreen --workers 8` in the background.
- Download finished: 20,774 maps re-screened, **554 ok**, 20,167 still unsupported, 53 errors
  (5,370 MB). Started `fetch.py build --incremental --budget 500000`.

### Task 4 — done (2026-10-25)

- Build converted 162 of the 554 new maps (392 unsupported): 70 solvable, 22 uncertain,
  19 invalid, 50 unsolvable, 1 unparsed. `finalize`: `out/workshop/levels.jsonl`
  22,752 -> **22,914** levels.
- `ratings --annotate-only`, `ws3.py`, `ratings --annotate-only`, `augment.py`, in that order.
- `data/ws5/train.jsonl`: 16,214 -> **16,336** lines. `data/ws5/test.jsonl`: 2,115 -> **2,134**;
  all 2,115 old ids still there (0 lost, 0 moved to train, 19 new).
- `data/v5b/workshop-aug-x2.jsonl` rebuilt: 257,204 -> **259,140** lines (69 levels skipped
  again). Note the new maps produced fewer usable levels than the ~830 the task hoped for:
  554 passed the download screen, 162 converted.
- No tracked files changed (levels and data are under `out/`/`data/`), so nothing to commit.

### Task 5 — done (2026-10-25)

- `data/v6/` is `generated.jsonl` (78,900), `workshop-aug-x2.jsonl` (259,140, copied from v5b),
  `vocab.txt` (copied from v5b) and `select.py`.
- Dry run prints one train command with `--data ...data\v6\generated.jsonl
  ...data\v6\workshop-aug-x2.jsonl` and `--holdout ...data\ws5\test.jsonl`.
- Share: Workshop **76.7%**, generated 23.3% (338,040 lines total).

### Franken / paint-bomb maps re-converted (2026-10-26)

- Task 4's `build --incremental` skipped the maps that had failed before `467ecd7`, so its
  converter fixes never reached the data. Backed up the data files to `*.before-2026-10-26`,
  removed the 4,590 maps whose last error was a Franken cube or paint bomb from
  `out/workshop/report.jsonl`, and ran `build --incremental --budget 500000`: **2,675 converted**,
  1,915 still unsupported (wall buttons, wall gel).
- Scoring them in one `tantalos-score score` pass ran out of memory (16 GB, 12 threads; one paint-bomb
  level asked for 4 GB). The first 1,261 results were kept; the rest were scored one level per
  process, 6 at a time. One level, **3801150618**, still runs out of memory and is left out.
- New verdicts: 434 solvable, 1,072 uncertain, 564 unsolvable, 582 invalid, 21 unparsed, 1 unsupported.
- `finalize`, `ratings --annotate-only`, `ws3.py`, `ratings --annotate-only`, `augment.py`:
  `out/workshop/levels.jsonl` 22,914 -> **25,588**; `data/ws5/train.jsonl` 16,336 -> **18,008**;
  `data/ws5/test.jsonl` 2,134 -> **2,362** (all old ids still there, none moved); aug file
  259,140 -> **285,730** lines (now `data/v5b/levels-workshop-aug-x2.jsonl`, renamed by other
  work in progress), copied to `data/v6/workshop-aug-x2.jsonl`.
- data/v6: 285,730 Workshop + 78,900 generated = 364,630 lines, Workshop **78.4%**. The dry run
  lists exactly the two v6 files and `--holdout ...data\ws5\test.jsonl`.
