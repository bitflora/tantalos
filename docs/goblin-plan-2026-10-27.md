# Goblin: data and self-play plan (2026-10-27)

Goblin is the net that reads a level and writes its solution as plan tokens (src/plan.rs). It
was called the solver net (task `solve`, runs `solve-...`) until 2026-10-27; "the solver" alone
still means the Rust BFS solver. The goal: a Goblin that understands Portal's mechanisms, then
learns AlphaGo-style from levels the level net writes, past what the Rust search can solve.

**The framing.** `glados-score check` is Goblin's rules of Go. Goblin can't beat the Rust *model*
of Portal (a plan only counts if the model allows it), but it can beat the Rust *search*: BFS
runs out of budget on big Workshop rooms, a learned policy plus a 0.3 ms check can go deeper.
So self-play's frontier is the levels the solver calls `uncertain`, and the ones only the
liberal model solves (those need Portal 2 to confirm). Every fix to the Rust model raises
Goblin's ceiling, and the replay buffer is re-checked for free.

Where it started: one run, `solve-fresh-3h` (2026-10-18): 30.6% of test-gen solved greedily
(pass@8 39.6%), 14.4% of test-ws, 0% of published Workshop levels; val loss still falling.
54% solved at 3-4 step plans, 23% at 12+; failures mostly `fire`/`walk` steps that fit no move.

User's decisions (2026-10-27): self-play trusts conservative-checked plans, and liberal-only
plans once Portal 2 has played them cleanly; the level net stays frozen first, co-trained
later; overnight CPU and GPU data jobs are fine when nothing is training.

## Done (2026-10-27)

- **Rename** (e7d9848): `--task goblin` (`--task solve`, old checkpoints and runs still work via
  `record.task_name`), `goblin_eval.py`, `goblin_data.py`, `goblin-runs.csv`, broadcast
  `--goblin`/`--goblin-hours` and `bc-NAME-rRR-goblin` runs.
- **Plan language covers every solver action** (96a1136). A 2,000-level sample (700 v5b
  generated, 700 mutants, 600 ws5 train) showed 27% of solvable levels got *no* plan at all,
  mostly portal loops (also double dashes, reaching through a portal): the route can't ground
  them, so they were dropped from the old data/plans. New steps `loop`, `grab`, `double`,
  `sweep`; `plan::encode_actions` writes a plan from the solver's actions when the route can't;
  `check` retries with angled laser spots. Now 13,005 of 13,005 sample plans check as solved.
  Loops appear in 1,508 of the sample's plans.
- **Plans move with augmented levels** (96a1136): `augment::apply_plan`, `glados-data augment
  --plans`. Kept: 100% of generated, 99.7% of mutant, 99.65% of Workshop plans; the rest are
  solver orientation quirks. One found and fixed: a non-reflection cube's facing at a laser spot
  had to match the spot the solver happened to pick, which depends on orientation.
- **Faster sampling** (d459127): mixed-length prompts in one padded batch. Greedy 19x, sampled
  4.4x, about 6x for an eval (greedy + 8 samples); 248/256 greedy plans identical (the rest a
  bf16 near-tie). 64 rows is the most that fits the 4 GB card.
- **Alternatives** (a916813): `score --plan-only --alternatives N`: a face class the first plan
  fires at switched off, solved again, kept if at most 3 actions longer. On 200 ws5 levels: 76
  get one or more (147 in all, all check), 74 with a different set of verbs; 2.7x the time.
- **Data builder** (288bfdc): train/goblin_data.py, data/goblin (see its docstring).
  - Primers are *selected*, not hand-written: the existing generated data and mutants already
    hold, per mechanic, hundreds to tens of thousands of small levels (<= 220 tokens) whose
    solution shows at most 2 mechanics: cube_button 41,646, laser_use 15,434, powered 12,968,
    panel_use 4,305, platform_use 3,990, redirect 3,075, bridge_use 1,523, funnel_use 1,449,
    plate_use 1,356; thin: stand_button 309, bounce 222, conversion 204, leap 41. Up to 2,000
    per mechanic, with looser limits (<= 320 tokens, 3 mechanics) for the thin ones;
    hand-written templates only if a family stays thin.
  - ws-deep timing: 200 ws5 train `uncertain` levels at 1M states took 125 s and 11.5% became
    solvable: all 8,089 take about 85 minutes for some 930 more Workshop levels with plans
    (+30% on the ~3,000).
  - Harvest from v6-fresh-12h/away: with `<verdict:solvable> <src:ws>` 5% of samples are kept
    (solvable, portal-needing, novel), 5.5% go to the self-play pool (`--keep-unsolved`); the
    rated prompt gives 2% solvable.

## Next

Step by step, with commands and checks: docs/goblin-next-steps-2026-10-27.md.

1. Overnight data jobs, one at a time, nothing training: `goblin_data.py --deep` (all parts),
   then `harvest.py runs/v6-fresh-12h/away/best.pt --cond "<verdict:solvable> <src:ws>"
   --out data/goblin/harvest/v6-solvable.jsonl --keep-unsolved data/goblin/pool/v6.jsonl`,
   then `goblin_data.py` again (plans the v6 part, assembles).
2. First Goblin run: one long run, `run.py start goblin-v6init-12h --task goblin --from
   v6-fresh-12h --hours 12 -- --layers 10 --batch 8 --accum 2` (block 1024 unless BUILD.md says
   over 10% of Workshop rows don't fit). Gate: more than 5 points over solve-fresh-3h on
   test-gen (30.6) and test-ws (14.4), and non-zero test-hard.
3. Self-play, `train/school.py` (not built yet), in stages:
   - **E1 expert iteration.** Each round: draw 4,096 levels, half from the frontier (solved by
     1-50% of samples before, or never, with back-off), half new (a fresh v6 harvest's pool,
     ws5 *train* uncertain/published levels; never test-side identities). Greedy + 32 samples,
     then prefix resampling from the longest checked prefix (`steps_ok`). Conservative check:
     trusted; liberal only: the game queue. Shorten each solved plan by dropping steps that
     aren't needed; keep at most 4 per level, different mechanic sets first. Buffer by
     identity, x4 symmetries (`augment --plans`), re-checked after every solver change. Train a
     `--quick` run on the base mix plus buffer (buffer 30-50% of rows), guarded; every 5 rounds
     one long consolidation run. Record per round: pool solved, beyond-BFS solves, frontier
     size, every test set, the probe, plan lengths, the game results, exploits.
   - **E2 constrained decoding** when E1 flattens: `glados-score env` (JSONL server: open a
     level, apply steps, list the legal next steps as plan tokens), sampling only legal steps.
   - **E3 liberal plans in Portal 2** (from round 1, when the game can run): unattended
     play.py batches with `route --plan --mode liberal`; clean plays become trusted; a failed
     conservative plan is a model exploit (docs/solver-gaps-2026-10.md, tests/playtested.rs, fix,
     re-check the buffer).
   - **E4 AlphaGo proper** once E2 is stable: a value head (P(solved | level, prefix)) trained on
     rollouts, best-first/MCTS search over steps with `glados-score env`, the policy trained on
     the search's plans. Game checks stay on.
   - **E5 co-training the level net** after several stable rounds: expert iteration on levels
     at Goblin's frontier, rated prompt kept, judged by blind A/B play-tests.

The reverse generator stays out (pedestals/buttons/goo only, a fixed hall-and-pockets layout
Goblin would learn shortcuts from). Only if the 12+ bucket stays far behind after step 2: a <= 5%
slice in one school round, with the user's go-ahead.
