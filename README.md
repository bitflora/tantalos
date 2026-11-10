# TANTaLOS
## Totally Alive Nifty Tester and Loving Operating System

Tooling for generating and solving Portal 2 Puzzle Maker levels. See [plan.md](plan.md).

The project includes:

- `.p2c` reading and writing, a level model, walkable regions and ASCII maps.
- A BFS solver with portal visibility, cubes, buttons, exit logic and conservative/liberal
  rules for uncertain mechanics.
- Level metrics, random chamber generation, mutation and evolutionary search.
- Token sequences, labelled datasets, model-output scoring and a small GPT (`train/`).
- Simulation of glass, grating, fizzlers, laser fields, timed pedestals, turrets and gel
  (repulsion, propulsion, conversion and water).
- Lasers and reflection cubes, faith plates and portal flings, light bridges, excursion
  funnels, panels, stairs, piston and rail platforms, loose cubes and triggered droppers.
  These are represented in the solver, tokens, metrics, generator, routes, game playback
  and Workshop converter. Modelling assumptions appear in module headers; outstanding
  game checks are described in the [reference](docs/level-tooling-reference.md) and [solver gaps](docs/solver-gaps-2026-10.md).

The command-line tools are separate binaries (`src/bin/`; each prints its usage when run
without arguments, and they share `src/cli.rs`):

| binary | commands |
|---|---|
| `tantalos-score` | `score`, `check`, `route`, `grammar`, `env`: scoring, plan checking, routes and legal next tokens |
| `tantalos-data` | `dataset`, `mutate`: making training data |
| `tantalos-p2c` | `info`, `solve`, `tokens`, `vocab`, `editor`, `roundtrip`, `fixtures`: one-off tools for .p2c files |
| `tantalos-view` | the browser app for saved solutions |
| `tantalos-workbench` | the browser app for triaging Workshop levels |
| `tantalos-evolve` | evolutionary search for interesting levels |
| `reverse` | the reverse generator (solvable by construction) |

```bash
cargo test
cargo run --bin tantalos-p2c -- info fixtures/synthetic/worked_example.p2c
cargo run --bin tantalos-p2c -- solve fixtures/synthetic/*.p2c
cargo run --release --bin tantalos-evolve -- --seed 1 --gens 40   # best levels + report.md in out/evolve-1
cargo run --bin tantalos-p2c -- roundtrip fixtures/real/*.p2c
cargo run --bin tantalos-p2c -- fixtures            # regenerate fixtures/synthetic from src/fixtures.rs
cargo run --bin tantalos-p2c -- tokens fixtures/synthetic/worked_example.p2c
```

## Training

Use `train/run.py` for training runs. It builds the Rust tools, trains, evaluates,
checks generated levels in Portal 2, trains away from rejected levels, stages a blind
A/B play-test and records results. Follow [CLAUDE.md](CLAUDE.md) for operating rules,
defaults, continuation and play-test handling.

Install the Python dependencies in a virtual environment:

```powershell
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv/Scripts/pip install -r train/requirements.txt
```

```powershell
.venv/Scripts/python train/run.py list
.venv/Scripts/python train/run.py start NAME --hours 2 --from PARENT --note "Purpose of run"
.venv/Scripts/python train/run.py wait NAME
.venv/Scripts/python train/run.py status NAME
.venv/Scripts/python train/run.py pause NAME
.venv/Scripts/python train/run.py resume NAME
```

Choose the newest completed run of the same task as `PARENT` unless a different parent
or a fresh model is requested. A fresh run omits `--from`. Pausing preserves weights,
optimizer and learning-rate schedule; stopping ends a run permanently. Only one heavy
job may run at a time. Do not rebuild release binaries during training.

Level runs write results under `runs/NAME/` and summaries to `train/runs.csv`.
Rated-style yield, mechanism use and item-family profile describe generated-level quality;
solver results, novelty, holdout loss and editor acceptance are guardrails. Blind human
play-tests take precedence when they show a clear difference.

## Goblin

Goblin reads level tokens and writes solution-plan tokens. The Rust solver independently
checks those plans. Start its runs through the same workflow with `--task goblin`;
results are recorded in `train/goblin-runs.csv`. Goblin runs evaluate plans without
launching Portal 2 or staging A/B levels. See [CLAUDE.md](CLAUDE.md) for data prerequisites.

`tantalos-score score --plan` adds plan tokens to solved levels. `tantalos-score check`
reports `solved`, `illegal`, `unfinished` or `unparsed`; `--route DIR` grounds accepted
plans for playback. `grammar` supplies legal level tokens and `env` legal plan tokens
for constrained decoding.

## Data and simulation

- `src/tokens.rs` encodes chamber geometry, items and connections as token sequences.
  Encoding preserves modelled gameplay rather than the original file bytes.
- `tantalos-data dataset` writes JSONL levels with verdicts, plan lengths and metrics.
  Generation combines random chambers with mutations of solvable levels.
- `tantalos-score score` accepts token lines and returns JSON verdicts and metrics.
- The solver searches abstract states under conservative and liberal rules. A conservative
  solution is provisional game evidence; failure within a search budget is unknown.
- Metrics cover effort, portal use, necessary components, mechanic diversity, dependency
  chains, travel and softlocks. Evolution uses these to select candidate chambers.

The model is a configurable GPT decoder trained on level or plan tokens. Evaluation
checks validity, conditioning, novelty and solver outcomes. Compare results only across
matching datasets and decoding modes.

Synthetic fixtures are built by `src/fixtures.rs`. Real `.p2c` fixtures are local BEE2
preview maps used for round-trip checks; do not redistribute them.

## Viewing and play-testing

`tantalos-view` displays saved `.route.jsonl` solutions alongside their `.p2c` files.
It draws the chamber and animates the recorded route without solving again.
Use `--roots DIR` to select folders, `--no-open` to suppress browser launch and
`--list` to inspect available routes. The viewer can export a self-contained HTML file.

`tantalos-workbench` displays downloaded Workshop levels, ratings, solver results and
chamber geometry. Triage is stored server-side in `train/workshop-annotations.jsonl`:
ratings, staging, notes and exclusion from the Workshop test split.

```powershell
python tools/workshop/annotations.py --get ID
python tools/workshop/annotations.py --excluded
python tools/workshop/annotations.py --set ID --rating 4 --staged --notes "Play-test notes"
```

Use `tantalos-workbench --generated` for the latest level run, `--run NAME` for a
specific run, or `--levels FILE` for token JSONL. Generated annotations are stored in
`train/generated-annotations.jsonl`. Marking a generated level unsolvable or broken
also adds it to the training-away data; clearing the verdict removes that entry.

```powershell
python tools/workshop/annotations.py --generated --get ID
python tools/workshop/annotations.py --generated --set ID --verdict unsolvable --notes "Play-test notes"
```

`tantalos-score route FILE` grounds an abstract solution into walks, shots and cube moves.
`train/play.py LEVEL.p2c --launch` plays that route in Portal 2 and records failed steps
and any recovery. Game automation requires the game to be closed before launch and
restores editor autosaves. See the [reference](docs/level-tooling-reference.md) for details.

## Broadcasting

`train/broadcast.py` coordinates rounds of generation, Goblin solutions, Portal 2 playback
and training, with an OBS browser overlay. See [streaming](docs/streaming.md) and the
operating rules in [CLAUDE.md](CLAUDE.md).

## Further documentation

- [Level tooling, format, metrics and game checks](docs/level-tooling-reference.md)
- [Known solver gaps and game evidence](docs/solver-gaps-2026-10.md)
- [Current roadmap](plan.md)
- [Goblin design](docs/goblin-plan-2026-10-27.md)
- [Goblin execution checklist](docs/goblin-next-steps-2026-10-27.md)
