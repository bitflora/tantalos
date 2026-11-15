# Generator architecture bake-off, plan of 2026-11-15

Over the next one to two weeks, train one fresh level net per candidate architecture under the
same budget, then compare them on the same scores and in blind play-tests. The reason: the
current GPT writes valid levels (RATED STYLE yield ~400 per 1,000) but few **strong** ones
(16–23 per 1,000), and A/B play-tests keep coming back "trivial". Nothing in today's model plans
a solution or sees the chamber it is building. Each arm below attacks that differently.

The user asked for these runs: every arm is a **fresh** run, not the usual warm start, and
the steps that override `run.py` defaults (`-- --arch ...`, head-to-head `abtest.py pick`,
the bake-off eval) are part of this plan.

## Fixed rules for every arm

- **Data:** the run.py default (`data/v5b/levels-*.jsonl`, ws7 Workshop file), unchanged for
  the whole series. If the data is rebuilt mid-series, the control is rerun, because EVAL
  isn't comparable across a relabel. Arm 2 adds plan tokens to the same levels.
- **Budget:** 12 hours of training each (`--hours 12`), measured in wall-clock time, not steps.
  The arms differ in speed (6 layers ran 40k tok/s, 10 layers 16k), so wall-clock time is the
  budget that matters on this machine. With eval, Portal 2 check, away and picking, a run takes
  about 16 hours, so eight runs need about 5–6 days of GPU. That leaves room to implement
  between them.
- **Size:** about 10M parameters (±30%) where the arm allows, so that architecture is what
  changes, not capacity. Each arm must fit batch 16 at block 1024 on the 4 GB card, or use
  `--batch 8 --accum 2`.
- **Decoding:** constrained (`tantalos-score grammar`) wherever the arm can use it. Arms that
  can't (run 7) are also compared with the control's *unconstrained* eval.
- **Runs:** one at a time. Names are `arch-<arm>-12h`, e.g. `arch-control-12h`, `arch-value-12h`.
  Each run's `--note` names the arm and this doc.

## Step 0: shared plumbing (before any run)

1. Finish or stop `v6-ft-hard-8h`, which is paused in its Portal 2 check. Nothing else can start
   until it is done.
2. `model.py`: add `Config.arch` (default `"gpt"`, so old checkpoints load unchanged) and a
   `build_model(cfg)` factory. `sample.load_checkpoint` and `train.py` call the factory instead
   of `GPT(...)`.
3. Every arm implements the same sampling interface that `sample.generate` uses
   (`generate_many(prompts, max_new, stop, temperature, top_k, batch, constrain)`). Then
   `evaluate.py`, `rated.py`, `abtest.py`, `harvest.py` and the workbench work unchanged.
4. `train.py --arch NAME` (plus any arm-specific flags), passed through `run.py start ... -- --arch NAME`.
5. A unit test per arm: a tiny model overfits 32 levels, and its samples round-trip through
   `tantalos-score score`.

## The arms, in run order

### Run 1: control, `arch-control-12h`
Today's GPT (6×6×384, QK-norm), fresh, 12h. It's needed because data changed after
v7-fresh-16h (plate speed and laser re-convert 2026-11-13), so v7's numbers aren't comparable.
No code changes. It can start as soon as step 0.1 is done, and the plumbing work runs meanwhile.

### Run 2: modern backbone, `arch-modern-12h` (proposal 7)
RMSNorm, SwiGLU MLP, RoPE in place of learned positions, no dropout, no biases. Width and
depth are set so the parameter count matches the control. This is the "is it just
the backbone?" control for runs 3–5. Code: about a day, all in `model.py`.

### Run 3: spatial embeddings, `arch-spatial-12h` (proposal 3)
The control's backbone (not run 2's, so the two effects stay separate), plus two embeddings
added to each token:
- **slot embedding:** the field this token fills (item type, x, y, z, facing, link target,
  geometry box corner, condition);
- **3D cell embedding:** the summed x/y/z embeddings of the cell the current item occupies, and,
  for link tokens, of the cell the link points at.

Both are parsed in the data loader from the token stream (a small Python state machine
mirroring `src/grammar.rs`; check it against the grammar on 10k levels). Sampling keeps the
same parser state per row. Code: 2–3 days.

### Run 4: value heads, `arch-value` (proposal 1). Not a full training run
Small MLP heads on the **control's** frozen backbone (run 1's best.pt), trained about 2 hours
on prefixes cut at item boundaries. They predict the finished level's: solvable / unsolvable /
uncertain, log solution length, strong yes/no, and item families used. The labels come from the
scores already in the data. Sampling: at each item boundary, draw k=4 continuations of the
next item, keep the one the heads rate as most likely strong (and penalise "unsolvable"), and
go on. A budget knob trades speed for strength.

This compares *the same weights* sampled two ways, so it is the cleanest test in the series.
It runs as an `evaluate.py`/`rated.py` option (`--guide heads.pt --k 4`) on run 1's checkpoint.
Its A/B: guided vs. plain sampling from run 1. Code: 2–3 days. GPU: about 2h head training plus
evals, so it fits between other runs.

### Run 5: plan-first, `arch-plan-12h` (proposal 2)
The control's architecture, trained on `[conditions] SEP [plan] SEP [level]`.
- **Plan:** a compact dependency sketch made from the solver's solution, *not* the full Goblin
  plan. It's the ordered list of state-changing steps as mechanic tokens (e.g.
  `cube→button bridge-on laser→catcher exit`), at most ~40 tokens. Unsolvable/uncertain levels
  get `<noplan>`, so the data stays the same size.
- **Data:** a `tantalos-data` command (or a Python script over `tantalos-score route`) that adds
  `plan` to each v5b row. CPU-heavy, so it runs while no training is going, ideally overnight
  after run 1 or 2.
- **Sampling:** the plan is sampled freely and the level is constrained as today. For rated
  style, also prompt with plan *length* (`<planlen:6+>`) to ask for deep levels.
- **Extra score:** the share of solvable levels whose solver route uses the mechanics the plan
  declared.

Code: 3–4 days, mostly data.

### Run 6: state-conditioned encoder–decoder, `arch-state-12h` (proposal 4)
A small voxel encoder (3D conv over a grid at most 32³ of cell type + item type + facing,
about 1M params) reads the chamber built so far. The decoder (control-sized, minus a layer
to stay at ~10M) cross-attends to it. The grid is refreshed at item boundaries only.
- **Training:** for each level, precompute grids at up to 8 random item boundaries; each token
  attends to the most recent grid snapshot before it. Grids come from a new
  `tantalos-score grid` command (Rust already builds this state for the grammar), cached to disk.
- **Sampling:** rebuild the grid of each row when it finishes an item.

The most engineering of the autoregressive arms (about 4–5 days), so it goes late. If the
memory doesn't fit, use an 8³-patch encoder at a coarser grid, or drop to batch 8.

### Run 7: masked diffusion over items, `arch-diffusion-12h` (proposal 5)
A bidirectional transformer (the control's size, no causal mask) trained MDLM/MaskGIT-style. The
level is a padded set of item slots (geometry stays a prefix, generated autoregressively by the
control, or copied from a Workshop level). Items are masked at a random rate and the model
predicts them. Sampling: 16–32 unmasking steps, with the lowest-confidence items re-masked.
Grammar: check items for legality after each step and re-mask illegal ones. That isn't the
same as constrained decoding, so compare it against the control's unconstrained numbers too.
Bonus test: **inpainting**, i.e. regenerate the items of 64 rated Workshop rooms with their
geometry fixed. Code: about 5 days. The most speculative arm.

### Run 8: edit model, `arch-edit-12h` (proposal 6)
An encoder–decoder (level in, edit ops out: `del ITEM`, `add ITEM...`, `rewire A B`) with a
target token (`<deeper>`, `<add:lasers>`, ...).
- **Training pairs:** `tantalos-data mutate` and `tantalos-structural` parent→child pairs,
  labelled by the change in solution length and mechanics.
- **Generation:** start from rated ws5-train parents, apply 1–5 edits, and keep the children
  the solver passes.

Novelty is judged against the parent too. A child within 3 items of its parent counts as not
novel. It isn't a from-scratch generator, so its RATED STYLE numbers are reported separately,
and it's judged mainly by play-tests. Code: about 4 days, building on `src/structural.rs`.

## Schedule (GPU, then code)

| Day | GPU | Coding |
|---|---|---|
| 1 | finish v6-ft-hard-8h; start run 1 (control) | step 0 plumbing; run 2 backbone |
| 2 | run 2 (modern) | run 3 spatial parser; run 4 heads |
| 3 | run 4 heads on run 1 (2h), then run 3 (spatial) | plan data script (run 5) |
| 4 | plan data build (CPU, overnight) | plan sampling + extra score |
| 5 | run 5 (plan-first) | `tantalos-score grid`, run 6 model |
| 6–7 | run 6 (state) | run 7 diffusion |
| 8 | run 7 (diffusion) | run 8 edit model + pairs |
| 9 | run 8 (edit) | bake-off scripts |
| 10–11 | bake-off eval | play-tests |

Play-tests happen as each run ends (its usual A/B), so verdicts come in through the
fortnight, not all at the end. An arm that fails its unit test or blows up in training (val
loss NaN, yield under 100) is fixed once, and otherwise recorded as failed and skipped.

## How the arms are compared

1. **Per run, automatically:** `run.py` records each arm in `train/runs.csv` (EVAL, val loss,
   Workshop test loss, RATED STYLE yield/strong/gap, Portal 2 check). Val loss is only comparable
   between runs 1–3 and 6. Run 5's includes plan tokens, and runs 7 and 8 have different objectives.
2. **Per run, blind A/B:** each arm against the **control's** best.pt, not the usual parent or
   Workshop pairing, with
   `abtest.py pick arch-X-ab runs/arch-X-12h/best.pt runs/arch-control-12h/best.pt --n 5 --constrained --game`
   (10 levels). The control itself gets the usual A/B against rated Workshop levels.
3. **Bake-off:** once every arm is done, each best.pt gets `rated.py --n 2048 --constrained
   --temperature 0.7 --seed 1`, the same seed for all. At 512 levels, "strong" is about 8–12
   levels, too few to rank on. The table gets yield, strong, profile gap, mean solution
   length of solvable levels, distinct item sets per 1,000 (variety), and generation time per
   level.
4. **Final blind round:** the top three arms by bake-off "strong", plus the control, with 4
   levels each, shuffled into one 16-level batch. The play-test verdicts outrank everything
   else. Per CLAUDE.md, 10–16 levels only show big gaps (2/10 vs 6/10), so a winner needs one.
   Otherwise the call is "no clear difference", and an arm only goes ahead on the
   automatic scores if it beats the control by more than the noise (yield +50, strong at least
   double).

**What a win means:** the winning arm becomes the next fresh baseline, and `run.py`'s defaults
switch to it in a separate, explicit change. Runs 4 (value heads) and 3 (spatial) can stack
with any winner and get a combined run if they help on their own.

## Results

| Run | Arm | Yield | Strong | Gap | A/B vs control | Notes |
|---|---|---|---|---|---|---|
| arch-control-12h | control | | | | (vs Workshop) | |
| arch-modern-12h | modern backbone | | | | | |
| arch-spatial-12h | spatial embeddings | | | | | |
| arch-value | value heads (on control) | | | | | |
| arch-plan-12h | plan-first | | | | | |
| arch-state-12h | state encoder | | | | | |
| arch-diffusion-12h | masked diffusion | | | | | |
| arch-edit-12h | edit model | | | | | |
