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

That is the first stage only. Later, a win is reaching the exit **in Portal 2**, not passing
`check`. The Rust model then has two jobs, neither of them judge: the fast simulator that search
runs in (the game can't branch or run thousands of plans), and a cheap filter that decides which
plans are worth game time. Where model and game disagree, the game wins: a game-confirmed plan
the model forbids is a model gap to fix, and a model-checked plan the game rejects is an exploit.
The architecture changes in "Next" 4 are chosen so the model never becomes a hard ceiling.

Where it started: one run, `solve-fresh-3h` (2026-10-18): 30.6% of test-gen solved greedily
(pass@8 39.6%), 14.4% of test-ws, 0% of published Workshop levels; val loss still falling.
54% solved at 3-4 step plans, 23% at 12+; failures mostly `fire`/`walk` steps that fit no move.

User's decisions (2026-10-27): self-play trusts conservative-checked plans, and liberal-only
plans once Portal 2 has played them cleanly; the level net stays frozen first, co-trained
later; overnight CPU and GPU data jobs are fine when nothing is training.

## Done (2026-10-27)

- **data/goblin built, harvest, check, baseline** (2026-10-28). Harvest of runs/v6-fresh-12h/away
  (217,088 sampled in 13,605 s over two sessions, paused once): 11,631 novel kept (5%), 15,001 in
  data/goblin/pool/v6.jsonl. Second build: train.jsonl 223,141 rows (v6 17,679), test-v6 602,
  test-ws 382, test-gen 2,332, test-hard 22, test-mut 0 (empty: look at before relying on it),
  Workshop rows over a 1024 block 0.1%. Check of 2,000 random train rows: 4,205 of 4,205 plans solved.
  **Baseline** (runs/solve-fresh-3h on data/goblin, greedy solved): test-ws+v6 5.7%, test-ws 12.8%,
  test-v6 1.2%, test-gen 23.5% (1,024 sampled), test-hard 0.0%, probe 18.1%. Task 4's run must beat
  test-ws+v6 and test-gen by more than 5 points, and test-hard above 0%.
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
   - **E4 AlphaGo proper** once E2 is stable: the value head and step-level search of "Next" 4
     (items a and b), the policy trained on the search's plans, the value head on the outcomes.
     Game checks stay on.
   - **E5 co-training the level net** after several stable rounds: expert iteration on levels
     at Goblin's frontier, rated prompt kept, judged by blind A/B play-tests.
   - **E6 learned dynamics (MuZero-style)**, only if the game keeps accepting plans the Rust
     model forbids, and only once there are thousands of clean in-game outcomes: "Next" 4e.
4. Architecture changes toward AlphaGo (agreed 2026-10-27). Goblin today is a decoder-only
   transformer (6 layers, 384 wide by default, `--layers 10` for the first run) with a policy
   only, decoded greedily or best-of-N, no search. In order:
   - **a. Value head** (with E4). A scalar head on the shared trunk, read at each step boundary
     (the token before a plan verb, and at `exit`), trained alongside the policy (policy
     cross-entropy plus value loss, as AlphaGo Zero). Two outputs, because there are two judges:
     **P(model accepts | level, prefix)**, labelled by `check` on school rollouts (cheap,
     millions), and **P(game reaches exit | level, prefix)**, labelled only by clean Portal 2
     plays (expensive, authoritative; a harness failure, play.py's `exec`/`plan?` classes, is
     not a label). The gap between the two marks likely exploits and picks which plans get game
     time. Checkpoints without the head load with it freshly initialised.
   - **b. Step-level search** (with E4). Tree nodes are whole steps (`fire C X Y Z D` is one
     move, not five tokens). Candidates are the legal steps `glados-score env` lists (E2), plus
     the policy's own top samples in liberal mode so a step the model forbids can still be
     tried; prior = summed token log-probabilities of the step, leaves scored by the value head;
     best-first first, MCTS (PUCT) if best-first stalls. Search runs in the Rust model; the best
     few plans per level go to the game (E3), and what the game accepts goes to the buffer.
   - **c. 3D position embeddings** (any time after the first Goblin run, step 2, as an A/B run
     against it).
     Today a cell's coordinates are just number tokens with 1D positions; add learned x, y and
     z embeddings to each coordinate token (and to the plan's cell arguments), so the net knows
     which cells are neighbours as AlphaGo's convolutions do. Derived at load time by parsing
     the existing token text (the grammar of `glados-p2c tokens`), so data/goblin is not
     rebuilt. A 3D-conv or patch encoder in front of the decoder only if the probe still shows
     spatial mistakes (wrong walls, off-by-one cells) after this.
   - **d. State tokens** (only if the value head can't learn from history alone). After each
     step, a few coarse tokens for the state the env reaches: player cell, both portals, held
     item, powered buttons/lasers. They come from replaying the existing plans through
     `glados-score env` in `goblin_data.py --assemble` (no re-planning). Kept coarse because
     they are the model's view: where the model is wrong, so are they; the game outcome stays
     the label.
   - **e. Learned dynamics** (E6). The net predicts the next state itself and is trained on
     game outcomes, the Rust model becoming a prior rather than the simulator, so search can
     reach plans the model forbids. Needs the data E3 and E4 collect; not before.
   - Kept as is: x4 symmetry augmentation (AlphaGo used 8), model size (the 4 GB card is the
     limit, and search buys more than parameters).

5. **Portals fired in flight** (asked 2026-10-28). Some levels need a new portal fired while the
   player falls or flies out of another. Goblin can only write what the plan language has words
   for, and a plan only counts if the Rust model allows it, so this is a model and plan-language
   change, not a net change. Today only `loop COLOR X Y Z D X Y Z` has it (back down through a
   floor portal, the other one moved on the way). Missing: firing in flight after a `dive` (out
   of the other portal, a new one placed to catch the player), and with it chained flings, the
   momentum family that blocks most of the ~3,100 movement-bound Workshop levels
   (docs/solver-gaps-2026-10.md, item 2). In order:
   - **a. Look in game first** (with the user): a few of those Workshop levels, and the physics
     of a refire in flight: when the shot can be made, which faces are reachable, how far the
     new portal flings. docs/solver-gaps-2026-10.md says not to model momentum before this.
   - **b. Solver action**, a refire on `dive` (as `Loop` has) and, if a shows they matter,
     chained flings; tests in tests/playtested.rs from the levels looked at.
   - **c. Plan word**: an optional face on `dive` (`dive COLOR [X Y Z D] X Y Z`, as `loop`), or a
     new word appended last to `PLAN_WORDS` so older vocabularies stay a prefix; `encode`,
     `encode_actions`, `check` and `augment::apply_plan` learn it. One new embedding row for
     the net, nothing more.
   - **d. Play harness**: `route` and train/play.py time the shot in flight.
   - **e. Data**: re-plan the levels the new action makes solvable (`goblin_data.py` parts),
     and re-check the school buffer.

None of 4's changes needs data/goblin rebuilt: c is parsed from the existing tokens, a and b take
their labels from school rounds and the game, d is an `--assemble` pass at most. 5 does: the
levels it makes solvable get plans for the first time.

The reverse generator stays out (pedestals/buttons/goo only, a fixed hall-and-pockets layout
Goblin would learn shortcuts from). Only if the 12+ bucket stays far behind after step 2: a <= 5%
slice in one school round, with the user's go-ahead.
