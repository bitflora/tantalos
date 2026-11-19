# Generator architecture bake-off, plan of 2026-11-15

Train one fresh level net per candidate architecture under the same budget, then compare them on
the same scores and in blind play-tests. This merges Claude's plan with Codex's
(`generator-architecture-plan.md`, in the Codex worktree `.codex/worktrees/70b4`): all eight
architecture arms, Codex's frozen manifest, failure accounting and uniform-edit baseline, and
Claude's per-run A/B against a control, the larger bake-off sample and the win thresholds. The
user chose the full set over a shorter budget, so it takes about three weeks.

The reason: the current GPT writes valid levels (RATED STYLE yield ~400 per 1,000) but few
**strong** ones (16–23 per 1,000), and A/B play-tests keep coming back "trivial". Nothing in
today's model plans a solution, sees the chamber it is building, or treats wires as links
between devices. Each arm attacks one of those.

The user asked for these runs: every arm is a **fresh** run, not the usual warm start, and
the steps that override `run.py` defaults (`-- --arch ...`, re-evaluating v7-fresh-16h,
head-to-head `abtest.py pick`, the bake-off eval) are part of this plan.

## The control: v7-fresh-16h now, a matched run only if needed

The comparison baseline is **v7-fresh-16h** (today's GPT, 6×6×384, trained fresh), re-evaluated
with the current scorer in step 0. It saves a day of GPU time, but it isn't a perfect match:
- **Data:** it trained before the 2026-11-13 Workshop patch (plate throw height, lasers
  re-converted low). The arms train on the patched data. Re-evaluating fixes how it is
  scored, not what it learned.
- **Budget:** it trained 16 hours, the arms 12. That favours v7, so an arm that beats it would
  also beat a matched control. An arm that loses narrowly could be losing to the extra hours.
- **Val loss:** its val loss is on the old data, so arms' val loss isn't compared with it.

A matched control (run 9) is the optional last step, trained only when those gaps could change
a conclusion (see run 9).

## Fixed rules for every arm

- **Frozen manifest (before any data is derived):** record the input files and their hashes,
  the split seed, vocab, eval settings, solver budgets, editor rules and the `tantalos-*`
  commit in `runs/arch-manifest.json`. Rotations, mirrors, mutants and edit descendants stay
  with their original puzzle. Audit exact and family-level overlap with the Workshop holdout
  once. Record v7-fresh-16h's deviation from the manifest (its pre-patch data). If the data
  must be rebuilt mid-series, record that too, and the optional control becomes required.
- **Data:** the run.py default (`data/v5b/levels-*.jsonl`, ws7 Workshop file). Runs 1, 2, 4
  and 6 (and the optional control) train on exactly the same levels. Run 5 trains on the
  subset with verified plans (and reports its size). Run 7 trains on parent/edit pairs, and
  run 8 on the same levels in its own form.
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
  (run 8) are also compared with v7's *unconstrained* eval.
- **Failure accounting:** every attempt counts in the denominator. Compile failures (runs 4,
  5, 7, 8), plan grounding failures and solver unknowns are reported as such, never dropped.
- **Runs:** one at a time, through `run.py start` and `wait`. Names are `arch-<arm>-12h`,
  e.g. `arch-modern-12h`. Each run's `--note` names the arm and this doc.
- **Readiness check:** an arm only gets GPU time once its check (listed per arm) passes. If
  implementation slips, the arm stays pending and the doc says which check is outstanding.
  Other arms go ahead meanwhile.

## Step 0: shared plumbing (before any run)

1. Write the frozen manifest (above).
2. Re-evaluate v7-fresh-16h's best.pt with the current scorer, into
   `runs/v7-fresh-16h/eval-arch/`: `evaluate.py --constrained`, `evaluate.py` unconstrained
   (for run 8), and `rated.py --constrained --temperature 0.7`. A few hours of GPU, and it can
   run while the plumbing is written.
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

### Run 1: modern backbone, `arch-modern-12h`
RMSNorm, SwiGLU MLP, RoPE in place of learned positions, no dropout, no biases. Width and
depth are set so the parameter count matches today's GPT. This is the "is it just
the backbone?" check for the other arms. Code: about a day, all in `model.py`.
**Ready when:** unit test passes.

### Run 2: spatial embeddings, `arch-spatial-12h`
Today's GPT backbone (not run 1's, so the two effects stay separate), same inputs and
output vocabulary, plus two embeddings added to each token:
- **field-role embedding:** the field this token fills (item type, x, y, z, facing, link
  target, geometry box corner, condition);
- **3D cell embedding:** the summed x/y/z embeddings of the cell the current item occupies, and,
  for link tokens, of the cell the link points at.

Both are computed from already-written tokens only, by a small Python state machine mirroring
`src/grammar.rs`. Sampling keeps the same parser state per row. Code: 2–3 days.
**Ready when:** the parser agrees with the grammar on 10k levels, and teacher forcing and
incremental decoding give identical features at every prefix (a test).

**Built (2026-11-16):** `train/spatial.py` (parser, 176 roles), `model.Spatial` (`--arch
spatial`), features worked out in the batching on the CPU (~940k tok/s) and by a parser per
row in every sampler. Readiness:
- `tests/test_spatial.py --n 10000`: at every prefix of 10,000 levels (Workshop test and
  generated, 2.37M tokens) the grammar's allowed tokens (editor rules off) are among the
  parser's, and no real token is `junk`. 0 disagreements.
- `tests/test_spatial.py`, `tests/test_arch.py`: incremental parsing equals whole-sequence
  parsing; the cache equals a full pass; greedy samples from all three samplers are the
  argmax of a full pass over the finished sequence, untrained and after overfitting 3 levels.
Start: `run.py start arch-spatial-12h --hours 12 --note "..." -- --arch spatial`.

### Run 3: value heads, `arch-value`. Not a full training run
Small MLP heads on **v7-fresh-16h's** frozen backbone, trained about 2 hours on prefixes cut
at item boundaries. They predict the finished level's: solvable / unsolvable / uncertain, log
solution length, strong yes/no, and item families used. The labels come from the scores
already in the (patched) data.
- **Sampling:** at each item boundary, draw k=4 continuations of the next item, keep the one
  the heads rate as most likely strong (and penalise "unsolvable"), and go on. `--k` trades
  speed for strength.
- **Why it's clean:** it compares *the same weights* sampled two ways, so v7's data and budget
  gaps don't matter here. The cleanest test in the series.
- **How it runs:** as an `evaluate.py`/`rated.py` option (`--guide heads.pt --k 4`) on v7's
  checkpoint. Its A/B: guided vs. plain sampling from v7.

Code: 2–3 days. GPU: about 2h of head training plus evals, so it fits between other runs.
**Ready when:** the heads beat a constant-prediction baseline on held-out prefixes.

**Built (2026-11-16):** `train/value.py`, heads in `runs/arch-value/heads.pt` (report
`heads.md`). Input: v7's final hidden state at the prefix's end, beside its mean over the level
so far. Every prefix is read under the rated prompt, not the level's own conditions, whose
verdict token would give the label away. Data: 102k levels from the default v5b files (Workshop
40k), plus earlier runs' rated eval samples. Held out: a tenth of puzzles by `data.identity`,
and v7's own 512 rated samples. That's 716k prefixes in all; extraction takes 7 minutes and
training 2.
- Readiness on held-out puzzles: every head beats the constant prediction.
  - Verdict CE: 0.80 against 1.00.
  - Log length MSE: 0.22 against 0.41.
  - Strong BCE: 0.47 against 0.64, AUC 0.83.
  - Families BCE: 0.37 against 0.62.
- On v7's own samples, the distribution guiding sees, three of four beat it. Strong ranks better
  than chance (AUC 0.65) but is overconfident: BCE 0.58 against 0.51.
- First third of a level: weaker but still above the constant on held-out puzzles. On v7's
  samples, the verdict head there is no better than the constant (1.12 against 1.10).
- Guided sampling is `sample.generate(guide=..., k=...)`, behind `--guide HEADS --k K` in
  `rated.py` and `evaluate.py`, and `--guide-a`/`--guide-b` in `abtest.py pick`.
  - At each record boundary it draws k drafts of the next record.
  - The net's first draft decides whether the level ends. When the heads could pick an ending
    draft too, levels never ended.
  - Otherwise it keeps the going draft with the highest P(solvable)·P(strong|solvable) −
    0.25·P(unsolvable). With λ = 1 it favoured huge "uncertain" rooms.
- Tests: `tests/test_value.py`.
- Run:
  ```
  python train/rated.py runs/v7-fresh-16h/best.pt --constrained --temperature 0.7 --seed 1 --out runs/arch-value/rated-plain
  python train/rated.py runs/v7-fresh-16h/best.pt --constrained --temperature 0.7 --seed 1 --guide runs/arch-value/heads.pt --k 4 --out runs/arch-value/rated-k4
  ```
  Same seed and same weights, so plain against guided is the comparison. The A/B is then
  `abtest.py pick arch-value-ab runs/v7-fresh-16h/best.pt runs/v7-fresh-16h/best.pt --guide-a runs/arch-value/heads.pt --n 5 --constrained --game`.

### Run 4: device-and-wire Transformer, `arch-wire-12h` (from Codex's plan)
The same chambers, in a typed form: geometry, then one record per device (type, cell, facing,
attributes), then wires. Each wire endpoint is chosen by a **pointer** over the devices
already written, with incompatible endpoints masked (a button can't drive a button). The
output is compiled back to standard level tokens for scoring.
- **Logged:** compile failures, wire counts, and how often the solver's solution uses the
  wired mechanisms.
- **First version:** only records that cover the existing level format.

Code: 4 days.
**Ready when:** every training level round-trips (tokens → records → tokens) with geometry,
device attributes and wire endpoints intact, and generated records compile and reach the
standard evaluator.

**Built (2026-11-17):** `train/wire.py` (the form, `Tracker`, `Legal`, `check`), `model.Wire`
(`--arch wire`), tests in `tests/test_wire.py` and `tests/test_arch.py`.
- **Form:** the flat level with its wires taken out of the sender records and written after all
  the items as `> @s @d`, where `@d` is the d-th item record (every item counts). A wire to a funnel
  adds `funnel` or `polarity`. Wires are grouped by sender in device order, each sender's in
  `encode`'s order. Item records keep the flat tokens unchanged, so the parser of run 2 (spatial)
  finds their boundaries. The training data is rewritten as it is batched (`data.recoded`), not on
  disk; on average the wire form is 0.4% longer.
- **Model:** the GPT backbone (6×6×384), plus a device-slot embedding added to every token of a
  device's record and to a pointer token, and 160 pointer outputs after the words. A pointer's
  score is attention from the current position (a query) to each device's key, taken from the last
  token of its record. Only devices already written that can be that end of the wire count:
  senders (button, pedestal, catcher, relay), then receivers (exit, fizzler or laser field, poured
  gel, laser, bridge, funnel, panel, glass panel, flip, stairs, piston, rail, dropper). Where a
  pointer must come, words are ruled out (EOS at -1e4, so a row with nothing to point at can still
  end). Adds about 0.16M parameters.
- **Decoding:** constrained (`wire.Legal`): items as `tantalos-score grammar` allows them, minus
  their wires; `>` or the end where the grammar lets the level end; then wires from a sender at or
  after the last one, each to a receiver it doesn't already drive. This differs from the flat
  grammar only in where wires go. Sampling compiles every level back to flat tokens
  (`model.Wire.compile_ids`), so evaluation, `rated.py`, `abtest.py` and the workbench are
  unchanged. A level that doesn't compile is written with `<wire-error>` and counts as unparsed.
  `sample.generate` prints `wire: N of M levels didn't compile, W wires per level that did`.
  Whether solutions use the wired mechanisms shows in rated.md's "solved: state changes" row.
- **Not supported:** level prefixes (`--prefix`), value-head guidance and Goblin.
- **Val loss** is over wire-form tokens, so it can't be compared with the flat arms' (Workshop
  test loss neither).

Readiness:
- `train/wire.py check`: all 2,417,554 levels of `data/v5b/levels-*.jsonl` (12.75M wires) go
  flat → wire → flat exactly: 0 errors, 0 mismatches. At most 154 devices in a level (160 slots).
- `tests/test_wire.py` checks:
  - the round trip of every target kind's counting;
  - that `compile` refuses bad wires;
  - that flags follow tokens;
  - that the tracker gives the same features incrementally as over the whole sequence;
  - that the cache gives one full pass's logits, with or without devices in the prompt;
  - that pointers are masked to written devices of the right class;
  - that a tiny model learns wired levels: greedy samples compile and match a full pass, and
    constrained samples all compile and parse.
- `tests/test_arch.py` runs the shared checks on the wire model too.
- Smoke test (400 steps, block 1024, batch 16): 11.0M parameters, ~41k tok/s (as fast as the
  GPT), 3.1 GB of GPU memory. After 400 steps, 12 of 16 unconstrained samples compiled.

Start: `run.py start arch-wire-12h --hours 12 --note "..." -- --arch wire`.

### Run 5: solution-first planner and layout decoder, `arch-plan-12h`
Two models, 12 hours in all:
- **Layout decoder (9h):** today's GPT architecture trained on
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

**Built (2026-11-18).** The code is in `train/milestones.py`, `model.Plan`, and `data.load(...,
milestones=)`; the tests are `tests/test_milestones.py`.
- **Milestones.** A milestone list abstracts a checked solver plan (src/plan.rs steps) and holds
  no coordinates, so mirrored and turned copies share one.
  - `walk`, `fire` and `exit` are dropped. `enter` and the mechanic words (knock, plate, bounce,
    loop, ...) are kept.
  - A cube taken names its kind: `pickup|grab|fetch KIND`.
  - Letting go names where the cube goes: `putdown|toss|throw [button > targets | laser]`.
  - A press names what it drives: `press|dash|reach|double > W...`.
  - Repeated identical groups are written once.
  - 41 words in all, at most 64 per level (5 levels were cut). They are appended to the vocabulary.
- **Labels.** `milestones.py label` → `data/milestones/labels.jsonl` (data/milestones/BUILD.md).
  - Levels: the default data plus `data/ws5/test.jsonl`, solvable rows only.
  - Generated levels: `score --plan-only` at 50k with editor rules on, then `tantalos-score check`
    replays every plan.
  - Workshop puzzles: labelled once, on the r0 copy, at 500k with editor rules off; every
    augmented copy shares that label.
  - Result: 529,385 labelled levels (506k generated, 23k Workshop) from 309,056 distinct puzzles,
    in 84 minutes.
  - Not labelled: 36k generated levels whose solvable label the editor rules now reject, 21k
    uncertain, 4k unsolvable, and 74 that failed the check.
  - Lengths: 232k have 0–7 words, 207k 8–15 words, 70k 16–23 words, 20k 24 or more.
- **One model, not two.** A single GPT (`--arch plan`) trains on `<bos> conds <sep> milestones
  <sep> level <eos>`. Loss covers everything after the first `<sep>`, so the same model plans and
  lays out. This is a deviation from the plan (a 9h decoder and a separate 3h planner): all 12
  hours go to one model.
  - Both evaluations are kept. `milestones.py eval CKPT` runs the decoder alone (prompted with
    held-out validation milestones) and the whole chain (it writes its own). It reports
    verdicts, then whether the solver's plans for the level match the declared milestones
    exactly, or at least use their words. The solver's plans are up to 8 (`--plans 8`).
  - Constrained decoding (`milestones.Legal`) allows milestone words until `<sep>`, with only
    wire targets after `>`. After `<sep>` it applies the level grammar.
- **What changes for this arm, and so isn't comparable:**
  - **Solvable levels only.** It trains on labelled levels only, which drops unsolvable,
    uncertain, `published` and unlabelled rows. That shrinks the Workshop share: about 22.7k
    puzzles × 16 rows ≈ 364k of the file's 1.58M rows.
  - **Losses.** Val, holdout and Workshop test loss include the milestone tokens and are over a
    different set of rows, so they can't be compared with the other arms.
  - **Prompts.** The model never sees `<verdict:published>`, so `rated.py` prompts a plan model
    with `<verdict:solvable>`. `evaluate.py`'s rows for verdicts it never trained on (unsolvable,
    uncertain) mean nothing for this arm.
- **Training away.** Rejected levels get milestones live from the solver, or an empty list
  (`<sep> <sep>`) when it has no plan.
- **Grammar dead end, found while testing.** After a pedestal's `> exit > dropper 0`, the level
  grammar offers `>` but then allows no target, so a sample that takes it ends unparsed. All
  arms can hit this; src/grammar.rs has not been fixed.

### Run 6: state-conditioned encoder–decoder, `arch-state-12h`
A small voxel encoder (3D conv over a grid at most 32³ of cell type + item type + facing,
about 1M params) reads the chamber built so far. The decoder (today's GPT size, minus a layer
to stay at ~10M) cross-attends to it. The grid is refreshed at item boundaries only.
- **Training:** for each level, precompute grids at up to 8 random item boundaries; each token
  attends to the most recent grid snapshot before it. Grids come from a new
  `tantalos-score grid` command (Rust already builds this state for the grammar), cached to disk.
- **Sampling:** rebuild the grid of each row when it finishes an item.
- **If memory runs short:** use an 8³-patch encoder at a coarser grid, or drop to batch 8.

Code: 4–5 days.
**Ready when:** the `grid` output matches the grammar's state on 1k levels, and a smoke run
fits in memory.

**Built (2026-11-19).** The code is in `train/state.py`, `model.State` (`--arch state`) and
`tantalos-score grid`; the tests are `tests/test_state.py` and the shared `tests/test_arch.py`
checks.
- **Grid.** Each cell holds three codes, at most 32 a side, x-major:
  - `cell`: outside, solid, open or goo;
  - `white`: 6 bits, one per side with a white surface;
  - `item`: an item word and its facing, or a part (a glass box's cells, a plate's target, a
    rail's far end); a later item overwrites.

  Python builds the grid from the tokens. `tantalos-score grid` prints Rust's own geometry
  (`Level`), and `train/state.py check` compares the two.
  - Geometry (the grammar's state, the tokens before the first item): 0 of 1,000 levels differ.
  - Whole levels: 118 of 1,000 differ, each by white faces Rust turns off under an item
    mounted there (a door, for instance). The item channel marks those cells anyway.
- **Deviation from the plan: no grid cache on disk.** The plan cached Rust-built grids. Grids
  are built on the fly from the tokens instead, in the batching thread for training and per row
  when sampling: 2.4M levels with 8 dense snapshots each would be far too big to store.
- **Encoder.** 1.04M parameters, 0.2M of them the block embedding.
  - Each voxel's input: its cell one-hot (solid, open or goo; outside is all zeros), its six
    white-side bits, and a 16-wide item embedding placed only where an item stands.
  - Then a 2×2×2 patch convolution (to 64 channels), a 3×3×3 convolution, a stride-2 one (to
    128) and one more. That gives one memory token per 4×4×4 cells, at most 8×8×8.
  - A learned embedding of each token's block, then a LayerNorm.
  - Cells outside the chamber are zeroed after every layer, so a grid encodes the same whether
    it's alone or padded in a batch (tested).
  - The first version embedded every voxel's three codes. Its embedding backward (millions of
    indices into a few rows), together with cross-attention dropout, made a step 5× slower than
    the GPT's.
- **Decoder.** Today's GPT with `--layers 5` (one fewer). After blocks 1, 3 and 4 it
  cross-attends to the memory (no attention dropout), plus a learned "nothing" key every token
  can always see. 11.83M parameters in all, against the GPT's 10.79M (positions not counted).
- **Snapshots.**
  - A snapshot is the grid of the tokens before an item's word.
  - In training, each sequence gets up to **4** of them at random item boundaries, not the
    plan's 8. On real length-bucketed batches (RTX 3050, batch 16), 8 snapshots ran at 20k
    tokens/s and 4 at 36k, against the GPT's 56k. 4 buys 1.8× more training in the same hours. Each token from
    a boundary on sees that snapshot until the next one; tokens before the first item see only
    the "nothing" key.
  - When sampling, a row's grid is rebuilt every time it writes an item word that isn't a wire
    target, so it always sees every earlier item. So sampled grids are fresher than training's
    (which can be up to several items stale); the grid never contradicts the tokens.
  - Validation batches get 4 random snapshots like training ones. A whole sequence fed at once
    (a prompt, the tests) gets every boundary, which is what the samplers give token by token
    (`tests/test_arch.py`: greedy samples match a full pass).
- **Unchanged:** constrained decoding (the level grammar), evaluation, `rated.py`, `abtest.py`
  and training away. Value heads aren't supported.

Readiness:
- `train/state.py check --n 1000`: geometry grids match `tantalos-score grid` on all 1,000
  levels.
- `tests/test_state.py` checks:
  - the geometry against Rust;
  - item and part painting, with wire targets not counted as items;
  - snapshot positions and `which`;
  - that a sampled row's grid equals the snapshot;
  - that tokens before a snapshot are bit-for-bit unchanged when that snapshot's grid changes,
    while tokens after it change;
  - that padding changes no memory.
- `tests/test_arch.py` runs every shared check on the state model too: cache consistency, the
  samplers agreeing, greedy samples matching a full pass, checkpoint reload, and a tiny model's
  constrained samples parsing.
- Smoke runs (batch 16, block 1024):
  - 3,000 steps at 23–28k tokens/s, val loss 1.61 → 1.46;
  - evaluation peaks at 3.2 GB of GPU memory;
  - 8 constrained samples: 2 solvable, 1 uncertain.

Start: `run.py start arch-state-12h --hours 12 --note "..." -- --arch state --layers 5`.

### Run 7: graph editor, `arch-edit-12h` (both plans; Codex's version)
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

### Run 8: masked diffusion over items, `arch-diffusion-12h`
A bidirectional transformer (today's GPT size, no causal mask) trained MDLM/MaskGIT-style. The
level is a padded set of item slots (geometry stays a prefix, written autoregressively by
v7-fresh-16h, or copied from a Workshop level). Items are masked at a random rate and the
model predicts them.
- **Sampling:** 16–32 unmasking steps, with the lowest-confidence items re-masked.
- **Grammar:** check items for legality after each step and re-mask illegal ones. That isn't
  the same as constrained decoding, so compare it against v7's unconstrained numbers too.
- **Bonus test:** **inpainting**, i.e. regenerate the items of 64 rated Workshop rooms with
  their geometry fixed.

Code: about 5 days. The most speculative arm, so it goes last.
**Ready when:** unit test passes and a smoke run unmasks to parseable levels.

### Run 9 (optional): matched control, `arch-control-12h`
Today's GPT (6×6×384, QK-norm, block 1024), fresh, 12h on the manifest's data. No code
changes; about 16 hours of GPU with all steps. **Run it if any of these holds** after the
bake-off:
- an arm loses to v7-fresh-16h by less than twice the noise (yield within 100, strong within a
  factor of 2), so the extra 4 hours or the old data could explain it;
- the winner is within the noise of v7, or the decision between two arms hinges on the
  baseline;
- the data was rebuilt during the series;
- the winner is to become the new default, and the write-up needs a clean baseline.

When it runs, it replaces v7 in the results table and the final blind round, and it gets the
usual A/B against rated Workshop levels. Arms already A/B-tested against v7 keep those verdicts.
New head-to-head A/Bs against the control are staged only for arms whose call it changes.

## Schedule (about three weeks)

GPU time is about 16 hours per arm with all steps, so 7 runs plus the heads come to about
5 GPU days, and 6 with the optional control. Coding is the bottleneck, so runs go as arms
become ready.

| Days | GPU | Coding / CPU |
|---|---|---|
| 1 | manifest; re-evaluate v7-fresh-16h | step 0 plumbing; run 1 backbone |
| 2 | run 1 (modern) | run 2 parser + parity test; run 3 heads |
| 3 | run 3 heads on v7 (2h), then run 2 (spatial) | run 4 records + round-trip |
| 4–5 | (idle if run 4 isn't ready) | run 4 pointer decoder; milestone label script |
| 5 | milestone labels (CPU, overnight) | |
| 6 | run 4 (device-and-wire) | run 5 decoder + planner |
| 7–8 | run 5 decoder smoke, then run 5 (plan) | `tantalos-score grid`, run 6 model |
| 9–10 | run 6 (state) | edit pairs, ranker, uniform baseline (run 7) |
| 11–12 | run 7 (graph editor) | run 8 diffusion |
| 13–14 | run 8 (diffusion) | bake-off scripts |
| 15–16 | bake-off eval | play-tests |
| 17 | run 9 (optional control) | |
| 18–21 | follow-up blind round; slack for slips | write-up |

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
   between runs 1, 2 and 6 (and the optional control); the others have different targets or
   objectives, and v7's is on the old data.
3. **Per run, blind A/B:** each arm against **v7-fresh-16h's** best.pt, not the usual parent or
   Workshop pairing, with
   `abtest.py pick arch-X-ab runs/arch-X-12h/best.pt runs/v7-fresh-16h/best.pt --n 5 --constrained --game`
   (10 levels). v7 already has its A/B against rated Workshop levels (2026-11-12). The graph
   editor's A/B also includes levels from its uniform-edit baseline.
4. **Bake-off:** once every arm is done, each one, and v7-fresh-16h, samples 2,048 rated-style
   levels at temperature 0.7 (seed 1, and seed 2 for any pair within noise), with the same
   solver budgets and editor checks. At 512 levels, "strong" is only about 8–12 levels, too few
   to rank on. Reported, each with its denominator:
   - rated yield, strong and profile gap;
   - solver-solvable %, editor acceptance, compile/grounding/solver failures;
   - mean solution length of solvable levels;
   - variety: distinct mechanic combinations and item sets per 1,000, exact and
     near-duplicates;
   - generation time per level.
5. **Optional control:** decide on run 9 here, by its rules above.
6. **Final blind round:** the top three arms by bake-off "strong", plus the control (v7, or
   run 9 if it ran), with 4 editor-checked levels each from distinct puzzle families, shuffled
   into one 16-level batch with the key sealed until all verdicts are in. Verdicts record which
   mechanics the tester used. Close contenders get a larger follow-up round.

**What counts as a win:** the play-test verdicts outrank everything else. Per CLAUDE.md, 10–16
levels only show big gaps (2/10 vs 6/10), so a winner needs one. Otherwise the call is "no clear
difference", and an arm only goes ahead on the automatic scores if it beats the control by more
than the noise: yield +50, solvable +10 points, or strong at least doubled (and holding on the
second seed).

**What a win means:** the winning arm becomes the next fresh baseline, and `run.py`'s defaults
switch to it in a separate, explicit change. Runs 2 (spatial) and 3 (value heads) can stack
with any winner and get a combined run if they help on their own.

## Results

One table: run ID, checkpoint, data and compute (hours, tokens or records, parameters, peak
memory), sample counts, automated scores and blind verdicts. Deviations from the manifest go
in Notes.

| Run | Arm | Params | Peak mem | Yield | Strong | Gap | Solv % | Failures | Away yield | A/B vs control | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v7-fresh-16h | control (re-evaluated) | | | | | | | | | (vs Workshop, 2026-11-12) | pre-patch data, 16h |
| arch-modern-12h | modern backbone | 10.8M | | 412 | 20 | 6.5 | 39.8 | | 408 | pending | trained in parts; its first 4 counted h held ~1.7 h of data loading (fixed in train.py 2026-11-15), so ~10.3 h of real training |
| arch-spatial-12h | spatial embeddings | 10.9M | | 439 | 20 | 10.5 | 31.6 | | 502 | pending | paused once at 8.57 h; final model away/best.pt |
| arch-value | value heads (on v7) | +0.67M heads | | 494 (plain 320) | 25 (plain 12) | 20.9 (plain 14.1) | | | — | pending | k=4, λ=0.25, seed 1, constrained, 512 each; sampling 3,914 s vs 894 s plain (4.4×). Guided levels are smaller (17 items vs 36, floor 70 vs 99), turrets 4% vs 28%, gel 0%; solvable 27% vs 14% but share of solved that are strong unchanged (16% vs 17%) |
| arch-wire-12h | device-and-wire | 11.0M | 3.1 GB (smoke) | 383 | 8 | 6.5 | 34.8 | | 410 | pending | paused once at 10.93 h; final model best.pt (away EVAL 28% vs 35%); Workshop test loss 0.8083 (away 0.7988), lowest of the arms, but val/ws loss are over wire-form tokens; game-valid 91.2% |
| arch-plan-12h | planner + layout (one joint model) | | | 619 | 35 | 24.8 | 46.1 | | 627 | pending | solvable-labelled data only; rated prompt <verdict:solvable>; paused at 2.0, 2.5, 9.76 h; final model away/best.pt (away strong 41, gap 23.0); game-valid 91.6% (away 90.3%); losses not comparable |
| arch-state-12h | state encoder | | | | | | | | | | |
| arch-edit-12h | graph editor | | | | | | | | | | vs uniform edits: |
| arch-diffusion-12h | masked diffusion | | | | | | | | | | |
| arch-control-12h | matched control (optional) | | | | | | | | | | only if run 9's rules call for it |
