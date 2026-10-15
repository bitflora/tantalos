# glados

Generates Portal 2 puzzle-maker levels: a Rust generator/solver (`src/`, binary
`target/release/glados.exe`) makes training data; a small GPT (`train/`) learns to write levels.

## Training runs

Use `train/run.py` for every training run. Don't call train.py, retrain.sh or loop.sh
yourself, and don't change their settings, unless the user asks you to.

```bash
.venv/Scripts/python train/run.py start NAME --hours H --note "WHAT IT IS FOR"
.venv/Scripts/python train/run.py start NAME --hours H --from OLDRUN --note "..."
.venv/Scripts/python train/run.py status NAME
.venv/Scripts/python train/run.py wait NAME
.venv/Scripts/python train/run.py stop NAME
.venv/Scripts/python train/run.py list
.venv/Scripts/python train/run.py note NAME "WHAT IT SHOWED"
```

- `start` returns straight away; the run continues in the background and survives this
  session. It trains for `--hours`, picks 5 levels for play-testing (train/pick.py: copied into the Puzzle Maker folder as `glados-RUN-NNNNN`, kept in `runs/NAME/playtest`; tell the user), then evaluates (about 5 more minutes), then records itself
  in `train/runs.csv`. Without `--from` it starts a new model; with `--from OLDRUN` it continues
  `runs/OLDRUN/best.pt`. Defaults (data/v5, learning rates, sizes) are already right: add no
  other options unless the user names them. Anything after `--` goes to train.py.
- Name runs short and dated by topic, e.g. `v5-fresh-2h`, `v5-cont-1h`. The name can't exist yet.
- To be told when it ends: run `wait NAME` with the Bash tool's `run_in_background`. It prints
  the final status. Don't poll in a loop; for a progress check, run `status NAME` once.
- Only one run at a time (one GPU). `start` refuses if one is running; say so to the user.
- When a run ends, report the status output, then record the conclusion with `note`.
- `--ingame` also checks samples in Portal 2. Use it only when the user says the game is free:
  it launches and drives the game, and needs its window focused.
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
