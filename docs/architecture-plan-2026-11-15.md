# Generator architecture bake-off, plan of 2026-11-15

Train one fresh level net per candidate architecture under the same budget, then compare them on
the same scores and in blind play-tests. This merges Claude's plan with Codex's
(`generator-architecture-plan.md`, in the Codex worktree `.codex/worktrees/70b4`): all nine
arms, Codex's frozen manifest, failure accounting and uniform-edit baseline, and Claude's per-run
A/B against the control, the larger bake-off sample and the win thresholds. The user chose the
full set over a shorter budget, so it takes about three weeks.

The reason: the current GPT writes valid levels (RATED STYLE yield ~400 per 1,000) but few
**strong** ones (16–23 per 1,000), and A/B play-tests keep coming back "trivial". Nothing in
today's model plans a solution, sees the chamber it is building, or treats wires as links
between devices. Each arm attacks one of those.

The user asked for these runs: every arm is a **fresh** run, not the usual warm start, and
the steps that override `run.py` defaults (`-- --arch ...`, head-to-head `abtest.py pick`,
the bake-off eval) are part of this plan.

## Fixed rules for every arm

- **Frozen manifest (before any data is derived):** record the input files and their hashes,
  the split seed, vocab, eval settings, solver budgets, editor rules and the `tantalos-*`
  commit in `runs/arch-manifest.json`. Rotations, mirrors, mutants and edit descendants stay
  with their original puzzle. Audit exact and family-level overlap with the Workshop holdout
  once. If the data must be rebuilt mid-series, rerun the control and record the deviation.
- **Data:** the run.py default (`data/v5b/levels-*.jsonl`, ws7 Workshop file). Arms 1–3, 5
  and 7 train on exactly the same levels. Arm 6 trains on the subset with verified plans (and
  reports its size). Arm 8 trains on parent/edit pairs, and arm 9 on the same levels in its
  own form.
- **Budget:** 12 hours of training each (`--hours 12`), measured in wall-clock time after a
  short smoke test. The arms differ in speed (6 layers ran 40k tok/s, 10 layers 16k), so
  wall-clock time is the budget that matters on this machine. Each run also records tokens or
  records processed, trainable parameters and peak GPU memory, so throughput and size stay
  visible.
- **Size:** about 10M parameters (±30%) where the arm allows, so that architecture is what
  changes, not capacity. Each arm must fit batch 16 at block 1024 on the 4 GB card, or use
  `--batch 8 --accum 2`.
- **Decoding:** constrained (`tantalos-score grammar`) wherever the arm can use it. Each arm
  logs its decoding rules and how they differ from the flat grammar. Arms that can't use it
  (run 9) are also compared with the control's *unconstrained* eval.
- **Failure accounting:** every attempt counts in the denominator. Compile failures (arms 5,
  6, 8, 9), plan grounding failures and solver unknowns are reported as such, never dropped.
- **Runs:** one at a time, through `run.py start` and `wait`. Names are `arch-<arm>-12h`,
  e.g. `arch-control-12h`. Each run's `--note` names the arm and this doc.
- **Readiness check:** an arm only gets GPU time once its check (listed per arm) passes. If
  implementation slips, the arm stays pending and the doc says which check is outstanding.
  Other arms go ahead meanwhile.

## Step 0: shared plumbing (before any run)

1. Finish or stop `v6-ft-hard-8h`, which is paused in its Portal 2 check. Nothing else can start
   until it is done.
2. Write the frozen manifest (above).
3. `model.py`: add `Config.arch` (default `"gpt"`, so old checkpoints load unchanged) and a
   `build_model(cfg)` factory. `sample.load_checkpoint` and `train.py` call the factory instead
   of `GPT(...)`.
4. Every arm implements the same sampling interface that `sample.generate` uses
   (`generate_many(prompts, max_new, stop, temperature, top_k, batch, constrain)`) and returns
   standard level tokens. Arms with their own representation compile to tokens before
   returning. Then `evaluate.py`, `rated.py`, `abtest.py`, `harvest.py` and the workbench work
   unchanged.
5. `train.py --arch NAME` (plus any arm-specific flags), passed through `run.py start ... -- --arch NAME`.
6. A unit test per arm: a tiny model overfits 32 levels, saves, reloads, and its samples
   round-trip through `tantalos-score score`.

## The arms, in run order

### Run 1: control, `arch-control-12h`
Today's GPT (6×6×384, QK-norm, block 1024), fresh, 12h. It's needed because data changed after
v7-fresh-16h (plate speed and laser re-convert 2026-11-13), so v7's numbers aren't comparable.
No code changes. It can start as soon as steps 0.1–0.2 are done, while the plumbing is built.
Older runs are context only.

### Run 2: modern backbone, `arch-modern-12h`
RMSNorm, SwiGLU MLP, RoPE in place of learned positions, no dropout, no biases. Width and
depth are set so the parameter count matches the control. This is the "is it just
the backbone?" control for the other arms. Code: about a day, all in `model.py`.
**Ready when:** unit test passes.

### Run 3: spatial embeddings, `arch-spatial-12h`
The control's backbone (not run 2's, so the two effects stay separate), same inputs and
output vocabulary, plus two embeddings added to each token:
- **field-role embedding:** the field this token fills (item type, x, y, z, facing, link
  target, geometry box corner, condition);
- **3D cell embedding:** the summed x/y/z embeddings of the cell the current item occupies, and,
  for link tokens, of the cell the link points at.

Both are computed from already-written tokens only, by a small Python state machine mirroring
`src/grammar.rs`. Sampling keeps the same parser state per row. Code: 2–3 days.
**Ready when:** the parser agrees with the grammar on 10k levels, and teacher forcing and
incremental decoding give identical features at every prefix (a test).

### Run 4: value heads, `arch-value`. Not a full training run
Small MLP heads on the **control's** frozen backbone (run 1's best.pt), trained about 2 hours
on prefixes cut at item boundaries. They predict the finished level's: solvable / unsolvable /
uncertain, log solution length, strong yes/no, and item families used. The labels come from the
scores already in the data.
- **Sampling:** at each item boundary, draw k=4 continuations of the next item, keep the one
  the heads rate as most likely strong (and penalise "unsolvable"), and go on. `--k` trades
  speed for strength.
- **Why it's clean:** it compares *the same weights* sampled two ways, the cleanest test in the
  series.
- **How it runs:** as an `evaluate.py`/`rated.py` option (`--guide heads.pt --k 4`) on run 1's
  checkpoint. Its A/B: guided vs. plain sampling from run 1.

Code: 2–3 days. GPU: about 2h of head training plus evals, so it fits between other runs.
**Ready when:** run 1 is done, and the heads beat a constant-prediction baseline on held-out
prefixes.

### Run 5: device-and-wire Transformer, `arch-wire-12h` (from Codex's plan)
Same chambers as the control, in a typed form: geometry, then one record per device (type,
cell, facing, attributes), then wires. Each wire endpoint is chosen by a **pointer** over the
devices already written, with incompatible endpoints masked (a button can't drive a button).
The output is compiled back to standard level tokens for scoring.
- **Logged:** compile failures, wire counts, and how often the solver's solution uses the
  wired mechanisms.
- **First version:** only records that cover the existing level format.

Code: 4 days.
**Ready when:** every training level round-trips (tokens → records → tokens) with geometry,
device attributes and wire endpoints intact, and generated records compile and reach the
standard evaluator.

### Run 6: solution-first planner and layout decoder, `arch-plan-12h`
Two models, 12 hours in all:
- **Layout decoder (9h):** the control's architecture trained on
  `[conditions] SEP [milestones] SEP [level]`.
- **Milestone planner (3h):** a small model that writes milestones from the conditions.

Milestones are the ordered state-changing steps of an **independently replayed** solution
(checked by `tantalos-score check`), written as mechanic roles and dependencies before any
coordinates exist (e.g. `cube→button bridge-on laser→catcher exit`, at most ~40 tokens).
For rated style, also prompt with plan length (`<planlen:6+>`) to ask for deep levels.
- **Data:** a `tantalos-data` command (or a Python script over `tantalos-score route`) that
  labels v5b rows. It's CPU-heavy, so it runs while no training is going, overnight. Log how
  many distinct puzzle families have verified labels. Gains are interpreted alongside that data
  change.
- **Evaluated two ways:** with held-out milestone prompts (decoder alone), and end to end
  (planner then decoder).
- **Counted:** plan replay, grounding failures and restricted-search bypasses (the solver finds
  a route that skips the declared milestones), unknowns included. Also the share of solvable
  levels whose route uses the declared mechanics.

Code: 4–5 days, mostly data.
**Ready when:** the decoder produces working chambers for several held-out plans in a smoke
run. The planner's 3h starts only after that.

### Run 7: state-conditioned encoder–decoder, `arch-state-12h`
A small voxel encoder (3D conv over a grid at most 32³ of cell type + item type + facing,
about 1M params) reads the chamber built so far. The decoder (control-sized, minus a layer
to stay at ~10M) cross-attends to it. The grid is refreshed at item boundaries only.
- **Training:** for each level, precompute grids at up to 8 random item boundaries; each token
  attends to the most recent grid snapshot before it. Grids come from a new
  `tantalos-score grid` command (Rust already builds this state for the grammar), cached to disk.
- **Sampling:** rebuild the grid of each row when it finishes an item.
- **If memory runs short:** use an 8³-patch encoder at a coarser grid, or drop to batch 8.

Code: 4–5 days.
**Ready when:** the `grid` output matches the grammar's state on 1k levels, and a smoke run
fits in memory.

### Run 8: graph editor, `arch-edit-12h` (both plans; Codex's version)
A small graph encoder reads a verified parent as devices, wires and spatial relations. An
edit decoder proposes one edit per attempt: a rewire, an added prerequisite or a signal swap
(the `src/structural.rs` operations), with an optional target token (`<deeper>`,
`<add:lasers>`).
- **Training data:** verified accepted edits are positives. Checked failures teach a ranker
  which proposals to reject. Examples per parent are capped, and parent provenance is kept.
  Training and evaluation use disjoint parent families.
- **Baseline:** **uniformly sampled legal edits** on the same held-out parents, with the same
  attempts per parent, seeds and solver budgets. The editor has to beat this to count.
- **Scored:** accepted variants per 1,000 attempts, distinct parents with an accepted variant,
  and changed solution structure relative to the parent. A child within 3 items of its parent
  counts as not novel.

It isn't a from-scratch generator, so its RATED STYLE numbers are reported separately next to
its uniform-edit control. It's judged with the other arms on playability and blind ratings.
Code: about 4 days. **Ready when:** the pairs are built and the uniform-edit baseline runs.

### Run 9: masked diffusion over items, `arch-diffusion-12h`
A bidirectional transformer (the control's size, no causal mask) trained MDLM/MaskGIT-style. The
level is a padded set of item slots (geometry stays a prefix, generated autoregressively by the
control, or copied from a Workshop level). Items are masked at a random rate and the model
predicts them.
- **Sampling:** 16–32 unmasking steps, with the lowest-confidence items re-masked.
- **Grammar:** check items for legality after each step and re-mask illegal ones. That isn't
  the same as constrained decoding, so compare it against the control's unconstrained numbers
  too.
- **Bonus test:** **inpainting**, i.e. regenerate the items of 64 rated Workshop rooms with
  their geometry fixed.

Code: about 5 days. The most speculative arm, so it goes last.
**Ready when:** unit test passes and a smoke run unmasks to parseable levels.

## Schedule (about three weeks)

GPU time is about 16 hours per arm with all steps, so 8 runs plus the heads come to about
6 GPU days. Coding is the bottleneck, so runs go as arms become ready.

| Days | GPU | Coding / CPU |
|---|---|---|
| 1 | finish v6-ft-hard-8h; manifest; start run 1 (control) | step 0 plumbing; run 2 backbone |
| 2 | run 2 (modern) | run 3 parser + parity test; run 4 heads |
| 3 | run 4 heads on run 1 (2h), then run 3 (spatial) | run 5 records + round-trip |
| 4–5 | (idle if run 5 isn't ready) | run 5 pointer decoder; milestone label script |
| 5 | milestone labels (CPU, overnight) | |
| 6 | run 5 (device-and-wire) | run 6 decoder + planner |
| 7–8 | run 6 decoder smoke, then run 6 (plan) | `tantalos-score grid`, run 7 model |
| 9–10 | run 7 (state) | edit pairs, ranker, uniform baseline (run 8) |
| 11–12 | run 8 (graph editor) | run 9 diffusion |
| 13–14 | run 9 (diffusion) | bake-off scripts |
| 15–16 | bake-off eval | play-tests |
| 17–21 | follow-up blind round; slack for slips | write-up |

Play-tests happen as each run ends (its A/B), so verdicts come in all through the series, not
only at the end. An arm that fails its readiness check or blows up in training (val loss NaN,
yield under 100) gets one fix. Otherwise it's recorded as failed and skipped.

## How the arms are compared

1. **Main checkpoint first.** All comparisons use each run's `best.pt`. `run.py` also trains
   away from rejected levels; those `away/` results are reported in a separate column, with the
   rejection data each saw. Rejection data grows as the series goes on, so it can't be part of
   the main comparison.
2. **Per run, automatically:** `run.py` records each arm in `train/runs.csv` (EVAL, val loss,
   Workshop test loss, RATED STYLE yield/strong/gap, Portal 2 check). Val loss is only comparable
   between runs 1–3 and 7; the others have different targets or objectives.
3. **Per run, blind A/B:** each arm against the **control's** best.pt, not the usual parent or
   Workshop pairing, with
   `abtest.py pick arch-X-ab runs/arch-X-12h/best.pt runs/arch-control-12h/best.pt --n 5 --constrained --game`
   (10 levels). The control itself gets the usual A/B against rated Workshop levels. The graph
   editor's A/B also includes levels from its uniform-edit baseline.
4. **Bake-off:** once every arm is done, each one samples 2,048 rated-style levels at
   temperature 0.7 (seed 1, and seed 2 for any pair within noise), with the same solver budgets
   and editor checks. At 512 levels, "strong" is only about 8–12 levels, too few to rank on.
   Reported, each with its denominator:
   - rated yield, strong and profile gap;
   - solver-solvable %, editor acceptance, compile/grounding/solver failures;
   - mean solution length of solvable levels;
   - variety: distinct mechanic combinations and item sets per 1,000, exact and
     near-duplicates;
   - generation time per level.
5. **Final blind round:** the top three arms by bake-off "strong", plus the control, with 4
   editor-checked levels each from distinct puzzle families, shuffled into one 16-level batch
   with the key sealed until all verdicts are in. Verdicts record which mechanics the tester
   used. Close contenders get a larger follow-up round.

**What counts as a win:** the play-test verdicts outrank everything else. Per CLAUDE.md, 10–16
levels only show big gaps (2/10 vs 6/10), so a winner needs one. Otherwise the call is "no clear
difference", and an arm only goes ahead on the automatic scores if it beats the control by more
than the noise: yield +50, solvable +10 points, or strong at least doubled (and holding on the
second seed).

**What a win means:** the winning arm becomes the next fresh baseline, and `run.py`'s defaults
switch to it in a separate, explicit change. Runs 3 (spatial) and 4 (value heads) can stack
with any winner and get a combined run if they help on their own.

## Results

One table: run ID, checkpoint, data and compute (hours, tokens or records, parameters, peak
memory), sample counts, automated scores and blind verdicts. Deviations from the manifest go
in Notes.

| Run | Arm | Params | Peak mem | Yield | Strong | Gap | Solv % | Failures | Away yield | A/B vs control | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| arch-control-12h | control | | | | | | | | | (vs Workshop) | |
| arch-modern-12h | modern backbone | | | | | | | | | | |
| arch-spatial-12h | spatial embeddings | | | | | | | | | | |
| arch-value | value heads (on control) | | | | | | | | — | | |
| arch-wire-12h | device-and-wire | | | | | | | | | | |
| arch-plan-12h | planner + layout | | | | | | | | | | |
| arch-state-12h | state encoder | | | | | | | | | | |
| arch-edit-12h | graph editor | | | | | | | | | | vs uniform edits: |
| arch-diffusion-12h | masked diffusion | | | | | | | | | | |
