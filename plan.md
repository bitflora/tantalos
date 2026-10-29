# glados: current roadmap

Updated 2026-10-29. Generate varied Portal 2 Puzzle Maker chambers that work in the game and
use mechanisms found in highly rated Workshop levels. Goblin learns to solve them, including
levels beyond the Rust search budget. The game and blind human play-tests remain the judges.

The [original feasibility proposal](docs/project-proposal-2026-10-06.md) is archived. Its
QLoRA recommendation, greenfield layout and time estimates are historical, not current tasks.

## What exists

- Rust level/token/.p2c tooling, Workshop reconstruction, generation, mutation, an abstract
  simulator and BFS solver, routes, and the Workshop/generated-level workbench.
- A small GPT trained on generated and Workshop chambers, with training/evaluation/game checks
  managed by `train/run.py`. Follow `CLAUDE.md` for every run; do not bypass its workflow.
- Goblin, the solution-policy net; checked plans and symmetry augmentation; identity-based
  data splits; legal-token decoding through `glados-score env`; a manual self-play round in
  `train/school.py`. Automated school training, a value head and tree search are still future work.

## Next milestones

1. **Trustworthy self-play data (repair complete).** Keep test and probe identities out of both new draws and
   resumed rounds. Preserve verdicts grouped by item when shortening plans. Independently
   recheck every retained buffer plan, including old rows, before publishing it. Save JSONL
   state atomically and write the completion marker last. The trial buffer audit recovered
   12 levels and 31 checked plans, excluded one probe identity, and preserved the original buffer.
   Corrections are recorded separately from historical round results.
   Focused regressions: `.venv/Scripts/python -m unittest discover -s tests -p test_school.py -v`.
2. **One verified school round, then a guarded training experiment.** Finish the small-round
   acceptance checks in [the Goblin task list](docs/goblin-next-steps-2026-10-27.md), including
   resumption and a second draw. Keep the level net frozen. Only then try base data plus the
   school buffer through `run.py`, measuring held-out solve rates and mechanism coverage.
3. **Search before more architecture.** At equal wall-clock budgets and on the same fixed
   levels, compare constrained best-of-N sampling with bounded step-level beam or best-first
   search using the existing policy, without a learned value head. Record solved share, plan
   length, model/check calls and peak memory on the 4 GB card. Add a value head only after
   rollouts provide useful labels and the baseline identifies a need for learned leaf scoring.
4. **Validate the simulator in Portal 2.** Queue liberal-only plans for game confirmation;
   conservative checks are provisional too. Fix model/game disagreements with playtested
   regressions. Study refiring portals in flight with the user before extending physics.
5. **Expand only on evidence.** Value-guided search, 3D embeddings, state tokens, co-training
   and learned dynamics remain gated experiments in the [Goblin plan](docs/goblin-plan-2026-10-27.md).
   Do not start long data builds, training or game sessions merely to update this roadmap.

## Measures and constraints

- Level net: rated-style yield, strong solutions, item-family profile gap and blind A/B play.
  Keep solver-solvable share, Workshop holdout loss, novelty and editor acceptance as guardrails.
- Goblin: constrained greedy solve rate on test-ws+v6, pass@8, test-gen/hard, the mechanism
  probe and frontier solves. Compare identical sets and decoding modes; empty test-mut is
  not evidence. A search result must be compared at equal wall-clock cost.
- A bounded search failure is unknown; the abstract simulator is not proof of real-game
  solvability or impossibility. Game-confirmed outcomes outrank simulator verdicts.
- Preserve batched checks, KV caches, row refill and CPU/GPU overlap. Prefer simpler bookkeeping
  around these optimizations; benchmark changes to their execution strategy before replacing them.
- One heavy job at a time. Training and broadcast commands, parameters and game access follow
  `CLAUDE.md`. No new training is part of the code/data-integrity repair.

## Detailed references

- [Goblin design and measured results](docs/goblin-plan-2026-10-27.md)
- [Goblin execution checklist](docs/goblin-next-steps-2026-10-27.md)
- [Known solver gaps and game evidence](docs/solver-gaps-2026-10.md)
- [Operating instructions](CLAUDE.md)
