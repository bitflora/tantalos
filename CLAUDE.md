# glados

Generates Portal 2 puzzle-maker levels: a Rust generator/solver (`src/`, binary
`target/release/glados.exe`) makes training data; a small GPT (`train/`) learns to write levels.

## Training runs

Use `train/run.py` for every training run. Don't call train.py, evaluate.py, ingame.py,
pick.py, retrain.sh or loop.sh yourself, and don't change their settings, unless the user asks
you to. `run.py start` already does every step of a run, in order: build `glados.exe`, train,
evaluate, check the levels in Portal 2's own editor, pick 5 game-checked levels for
play-testing (copied into the Puzzle Maker folder as `glados-NAME-NNNNN`), record in
`train/runs.csv`. You don't run any of those steps yourself.

A training run is exactly these steps:

1. `.venv/Scripts/python train/run.py start NAME --hours H --note "WHAT IT IS FOR"`
   (add `--from OLDRUN` only to continue an earlier run's model). If it refuses, tell the user
   why and stop.
2. `.venv/Scripts/python train/run.py wait NAME`, with the Bash tool's `run_in_background`.
   You are told when it ends. Don't poll; for a progress check, run `status NAME` once.
3. When it ends, read the printed status and do every item under **NEXT STEPS**, in order.
   `wait` exits 0 when the run is complete, 1 if it failed, 2 if the Portal 2 check was
   skipped (the game was already running): then tell the user, and when they say the game is
   closed, run `run.py ingame NAME` and go back to step 2.

```bash
.venv/Scripts/python train/run.py status NAME
.venv/Scripts/python train/run.py stop NAME
.venv/Scripts/python train/run.py list
.venv/Scripts/python train/run.py ingame NAME
.venv/Scripts/python train/run.py note NAME "WHAT IT SHOWED"
```

- Defaults (data/v5, learning rates, sizes) are already right: add no other options unless
  the user names them. Anything after `--` goes to train.py. `--no-ingame` skips the Portal 2
  steps: use it only if the user says not to launch the game.
- The Portal 2 steps launch the game themselves (in the background, no focus needed) and never
  touch a game that is already running.
- Name runs short and dated by topic, e.g. `v5-fresh-2h`, `v5-cont-1h`. The name can't exist yet.
- Only one run at a time (one GPU). `start` refuses if one is running; say so to the user.
- While a run is going, don't rebuild `target/release` (`cargo build --release`) or start
  other heavy jobs: training calls `glados.exe` and shares the machine.

Reading results: the main score is **EVAL solvable %** (share of sampled levels our solver can
solve; higher is better). Differences under about 10 points are noise. Val loss: lower is
better, but only comparable between runs on the same data. A run that FAILED or DIED shows the
last lines of its log; report them, don't retry blindly.

Self-training: `train/harvest.py CHECKPOINT --n N` samples levels and keeps the solvable,
portal-requiring, novel ones in `data/harvest/RUN.jsonl`; train on them by adding that file to
`--data` next to the usual files (see its docstring). It uses the GPU and the solver, so not
while a run or data generation is going.

Training time: one run of 2 hours beats two chained 1-hour runs. Each run's learning rate
warms up and decays to its minimum over its own time, so a chained run starts back at a high
rate and spends its first part undoing the fine-tuning the last one ended with. Chain runs only
to change something between them (data, settings) or to look at a model midway.
