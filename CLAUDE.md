# glados

Generates Portal 2 puzzle-maker levels: a Rust generator/solver (`src/`, binaries
`target/release/glados-*.exe`, see below) makes training data; a small GPT (`train/`) learns to write levels.

## Training runs

Use `train/run.py` for every training run. Don't call train.py, evaluate.py, ingame.py, abtest.py pick,
pick.py, retrain.sh or loop.sh yourself, and don't change their settings, unless the user asks
you to. `run.py start` already does every step of a run, in order: build the `glados-*.exe` binaries, train,
evaluate, check the levels in Portal 2's own editor, train 15 minutes away from every level
the game has rejected or a play-tester found unsolvable (`data/ingame/human-unsolvable.jsonl`), and every evaluated
level whose exit can never be reached (`data/ingame/dead.jsonl`, glados-score's `"dead"`) (into `runs/NAME/away`; evaluated and game-checked too, and the final
model unless it scores over 3 points lower), stage a blind A/B play-test (`train/abtest.py`: 4
game-checked levels from the final model and 4 from its parent, or from rated Workshop levels
for a fresh run, shuffled into the Puzzle Maker folder as `glados-ab-NAME-NN`), record in
`train/runs.csv`. You don't run any of those steps yourself. `--from NAME` continues that
run's final model.

When the user reports how the A/B levels played, write each verdict (good, ok, trivial,
boring, unsolvable, broken; plus what the solution used and notes, if they say) into
`runs/NAME/playtest/results.csv`, then run
`.venv/Scripts/python train/abtest.py score NAME --out runs/NAME/playtest`, tell the user the
unblinded result, and commit `train/playtests.jsonl`. Never reveal or read out
`playtest/key.json` before the verdicts are in. Ten levels only show big differences (2 of 10
good against 6 of 10); don't call a run better on a smaller gap.

A training run is exactly these steps:

1. `.venv/Scripts/python train/run.py start NAME --hours H --note "WHAT IT IS FOR"`
   (`--from OLDRUN` continues that run's model; by default it is the most recently trained net,
   see below). If it refuses, tell the user why and stop.
2. `.venv/Scripts/python train/run.py wait NAME`, with the Bash tool's `run_in_background`.
   You are told when it ends. Don't poll; for a progress check, run `status NAME` once.
3. When it ends, read the printed status and do every item under **NEXT STEPS**, in order.
   `wait` exits 0 when the run is complete, 1 if it failed, 2 if the Portal 2 check was
   skipped (the game was already running): then tell the user, and when they say the game is
   closed, run `run.py ingame NAME` and go back to step 2. 3 if it was paused (see Pausing a
   run below).

**Pausing a run** (when the user wants the machine back and the run continued later; never on
your own initiative):

1. `.venv/Scripts/python train/run.py pause NAME`. Only works while it is TRAINING; if it says
   the run is past training, tell the user and let it finish. It waits until training has saved
   and stopped (a minute or so, up to 20 if an eval is under way), then prints PAUSED and how many
   hours are done. If a `wait NAME` was going in the background, it ends with exit code 3
   (paused): that is not a failure.
2. Tell the user it is paused and how many hours are done. Nothing else runs; a paused run holds
   neither the GPU nor the machine.
3. Only when the user says to carry on: `.venv/Scripts/python train/run.py resume NAME`, then
   step 2 above (`wait NAME` with `run_in_background`). It trains for the hours still owed and
   then does every other step as usual. If it refuses (another run going), tell the user why.

**Training in parts.** When the user asks for part of a run now ("the first 5 hours of a 10-hour
run"), start the whole run with `--pause-after`: `run.py start NAME --hours 10 --pause-after 5
--note "..."` (hours of training before it pauses itself; between 0 and `--hours`). Then `wait
NAME` as usual: it exits 3 when the run has paused, and you go on at step 2 above. On `resume`,
`--pause-after H` (hours of training in all, more than are done) makes it pause again, e.g. a
third of 9 hours at a time: `start --hours 9 --pause-after 3`, then `resume NAME --pause-after 6`,
then `resume NAME`.

A paused-and-resumed run equals one unbroken run of the same `--hours`: weights, optimizer and
learning-rate schedule carry on. Two runs chained with `--from` do not (see Training time below).
Don't use `stop` to pause: a stopped run can't be resumed. `stop NAME` on a paused run ends it
for good and records it.

```bash
.venv/Scripts/python train/run.py status NAME
.venv/Scripts/python train/run.py stop NAME
.venv/Scripts/python train/run.py pause NAME
.venv/Scripts/python train/run.py resume NAME
.venv/Scripts/python train/run.py list
.venv/Scripts/python train/run.py ingame NAME
.venv/Scripts/python train/run.py note NAME "WHAT IT SHOWED"
```

- Defaults (data/v5b, learning rates, sizes) are already right: add no other options unless
  the user names them. The default data (every `data/v5b/levels-*.jsonl`) holds the Workshop
  levels as their 8 mirrored/turned versions, twice each (`levels-workshop-aug-x2.jsonl`), since
  2026-10-26; the old 20 plain copies are `data/v5b/workshop-x20.jsonl`, used only if the user
  asks (name all the files with `--data`). Anything after `--` goes to train.py. `--no-ingame` skips the Portal 2
  steps: use it only if the user says not to launch the game.
- **Warm-start by default:** unless the user says otherwise (or names a different `--from`),
  continue the most recently trained net of the same task: pass the final model of the newest
  completed run as `--from` (find it with `list`). Start a run from scratch only when the user
  asks for a fresh model. Note the known risk: recent warm starts have diverged under the warm
  rates (lr 2e-4) and finished well below their init, so report that if it happens again.
- A 10-layer model doesn't fit at batch 16 on the 4 GB card: add `--batch 8 --accum 2` (the same
  effective batch) to any `--layers 10` run. Measured 2026-10-20, when v5-fresh-10h-d ran out of
  memory on its first step.
- The Portal 2 steps launch the game themselves (in the background, no focus needed) and never
  touch a game that is already running.
- Name runs short and dated by topic, e.g. `v5-fresh-2h`, `v5-cont-1h`. The name can't exist yet.
- Only one run at a time (one GPU). `start` refuses if one is running; say so to the user.
- While a run is going, don't rebuild `target/release` (`cargo build --release`) or start
  other heavy jobs: training calls `glados-score.exe` and shares the machine.

Binaries (`src/bin/`, each prints its usage when run without arguments):
`glados-score` (`score`, `check`, `route`: what training, evaluation and play call),
`glados-data` (`dataset`, `mutate`), `glados-p2c` (`info`, `solve`, `tokens`, `vocab`, `editor`,
`roundtrip`, `fixtures`), `glados-view`, `glados-workbench`, `glados-evolve`, and `reverse`.
There is no `glados.exe` any more.

Reading results. The goal (2026-10-25) is varied levels that use the mechanisms of highly rated
Workshop levels, so the headline is the **RATED STYLE** line (`yield` in `list`, `rated_*` in
runs.csv; train/rated.py, eval/report.md "Rated style"): 512 levels asked for Workshop-style
levels at temperature 0.7. **Yield**: valid, novel, not solver-unsolvable levels per 1,000
(higher is better; differences under ~50 are noise). **Strong**: solvable ones whose solution
uses 2+ state-changing mechanics and at most 1 trick, per 1,000. **Profile gap**: mean
difference in points between the samples' item-family mix and 5-star levels' (lower is better).
Baseline, v5-cont-10h-d (2026-10-25, runs/v5-cont-10h-d/rated/): yield 207, strong 8, gap 23.5.
**EVAL solvable %** (share of levels asked for `<verdict:solvable>` our solver solves) is now a
guardrail, not the target; differences under about 10 points are noise. Val loss: lower is
better, but only comparable between runs on the same data; since 2026-10-25 val is split by
puzzle (a Workshop level's copies and mutants stay on one side), so it isn't comparable with
earlier runs, and each eval also logs it per source (gen, mutant, ws). **Workshop test loss**
(`ws` in `list`, `holdout_*` in runs.csv): loss on 256 held-out Workshop levels
(data/ws5/test.jsonl); lower is better, comparable across runs. If it rises while val loss
falls, the model is memorizing Workshop levels: report that. (2026-10-25: v5-cont-10h-d scored
~1.59 here against 0.24 on the Workshop levels it trained on.) The blind A/B verdicts outrank
all of these when they show a clear difference. A run that FAILED or DIED shows the last lines
of its log; report them, don't retry blindly.

Self-training: `train/harvest.py CHECKPOINT --n N` samples levels and keeps the solvable,
portal-requiring, novel ones in `data/harvest/RUN.jsonl`; train on them by adding that file to
`--data` next to the usual files (see its docstring). It uses the GPU and the solver, so not
while a run or data generation is going.

Training time: one run of 2 hours beats two chained 1-hour runs (a paused and resumed run counts
as one run). Each run's learning rate
warms up and decays to its minimum over its own time, so a chained run starts back at a high
rate and spends its first part undoing the fine-tuning the last one ended with. Chain runs only
to change something between them (data, settings) or to look at a model midway.

## Workshop workbench (triage of downloaded levels)

`glados-workbench` is the browser app for the downloaded Workshop levels (`out/workshop/levels.jsonl`
plus `data/ws5/{train,test}.jsonl`). It shows every field, the solver's verdict and rating, and
draws the chamber. The user's triage is server-side in `train/workshop-annotations.jsonl`, not in
the browser. Read or write it with:

```bash
python tools/workshop/annotations.py --get ID
python tools/workshop/annotations.py --excluded
python tools/workshop/annotations.py --json
python tools/workshop/annotations.py --set ID --rating 4 --staged --notes "..."
python tools/workshop/annotations.py --set ID --exclude
```

Fields: `staged` (manual playtest), `rating` (0–5), `notes`, `excluded` (leave out of the ws5 test
split; `tools/workshop/ws3.py` moves it to train on rebuild, `train/plans.py` leaves it out of
`test-ws`). When the user asks for levels to play-test, `--excluded`/`--get` is the shared source
of truth. Do not rewrite the level files for annotations.

## Broadcast (Twitch stream)

`train/broadcast.py` runs the stream's loop in rounds: the level net writes levels, the solver net
solves them, Portal 2 plays them, and both nets train on them (its own `run.py start --quick` runs,
named `bc-NAME-rRR-gen` / `-solve`; never start those yourself). See docs/streaming.md.

1. `.venv/Scripts/python train/broadcast.py start NAME` (e.g. `live-01`; add options only if the
   user names them). If it refuses, tell the user why and stop.
2. `.venv/Scripts/python train/broadcast.py wait NAME` with `run_in_background`. Don't poll; for
   a progress check, run `status NAME` once.
3. When it ends, do every item under **NEXT STEPS** in the printed status.

- `stop NAME` ends it after the current level. Never kill its process (it holds the user's
  editor autosaves).
- No training run can start while a broadcast is going, and the other way round.

## Solver-net runs

The second kind of run trains the solver net: it reads a level and writes its solution as plan
tokens (src/plan.rs). Same steps as above, with `--task solve` in step 1:
`.venv/Scripts/python train/run.py start NAME --task solve --hours H --note "WHAT IT IS FOR"`.
Name them `solve-...`, e.g. `solve-fresh-2h`. `--from` a level run starts from its weights (a
new task, fresh learning rates); `--from` a solve run continues it.

- It needs `data/plans/` (train.jsonl, test-gen.jsonl, test-ws.jsonl), made by `train/plans.py`:
  CPU-heavy (every thread) for about 30-40 minutes, so only when the user asks, and not while a run is going. `start`
  refuses without it; tell the user.
- A solve run does: build, train, evaluate with `solve_eval.py` (about 15 minutes), record. It
  has no Portal 2 steps, no training away and no play-test picks, so `wait` never exits 2.
- Solve runs are recorded in `train/solver-runs.csv`, not runs.csv; see them with
  `.venv/Scripts/python train/run.py list --task solve`. `note` finds the right file itself.
- The main score is **EVAL solved %**: the share of 1,024 held-out generated levels (test-gen)
  whose greedy plan `glados-score check` accepts. Higher is better; differences under about 5 points
  are noise. Also reported: pass@8, optimal %, test-ws (Workshop levels our solver solves) and
  published-liberal (Workshop levels it can't; a weak signal). Don't compare it with the level
  runs' EVAL solvable %.

## Staging "unsolv" Workshop levels for play-testing

When the user asks for N more "unsolvable" Workshop levels, run one command (no Portal 2 needed,
don't start or touch the game; needs `target/release/glados-score.exe` and `glados-p2c.exe`):

```bash
.venv/Scripts/python train/stage_unsolv.py --n 5
```

It picks the smallest levels of `data/ws5/train.jsonl` our solver calls unsolvable (not yet in
`train/unsolv-staged.jsonl`, no `@pre_paint_sprayer` gel, one per item set), re-scores them with the
release build (50k/250k, editor rules off) and `glados-p2c editor`, writes `out/unsolv-DATE/`
(p2c, tokens, scores, MANIFEST.md), copies them into the Puzzle Maker folder as
`glados-ws5b-unsolv-NNNNN` (continuing the highest number; Title `unsolv <title>`) and appends them
to the registry. `--dry-run` shows the picks only. Commit `train/unsolv-staged.jsonl` afterwards.
Then tell the user the file names, titles, Workshop ratings and item sets. Add `--n` only if the
user names a count, and `--order rating` (most stars, then most ratings, first; from
`tools/workshop/fetch.py ratings`) only if they ask for well-rated or popular levels.
If an editor flag shows up, say so; the real-editor check is `train/portal2.py check DIR --out F
--launch` and only when the game is closed. When the user reports how they solved one, follow
the unsolv-playtests memory: fix the solver or document it in docs/solver-gaps-2026-10.md and add
it to tests/playtested.rs.
