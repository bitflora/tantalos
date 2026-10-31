# Goblin polish: bounded work and fewer replays

The e1/r01 stall was Workshop `132677917` (`size 27 26 25`), work.jsonl
index 2755 (zero based). Its four successful plans were 49, 46, 58 and 52
steps long, counting the optional `exit` word. The old polish repeatedly
built this level's model, checked every single-step deletion, and used only
the first successful deletion in each batch. Model construction alone took
22.9 seconds in a bounded setup measurement. The previously added Python
time limit was checked between batches, so it could not interrupt one.

The replacement is `glados-score polish --seconds 300`, called by
`train/school.py` with a subprocess timeout of the same duration:

- Build one conservative Checker per level and reuse its lazy angled model.
- Try deleting large contiguous blocks, then halve the block size down to one
  step. Accept the first success immediately. This can remove whole loops
  whose individual steps cannot be removed separately.
- Cache candidate verdicts across the level's plans. Never check the same
  candidate twice. A candidate gets at most two seconds between state
  expansions; unknown candidates are never accepted.
- Flush the full plan list initially and after every accepted improvement.
  Python kills and reaps an over-budget process and retains the last complete
  snapshot, even if the final JSON write was interrupted. The hard timeout
  covers model construction and individual expansions too.
- Select up to four plans before polishing, shortest first with verb diversity.
  Previously all successful samples were polished before four were kept.
- Save the round after each solved level, including a timed-out polish.

This is optional optimization: inputs must already have passed conservative
checking. A timeout leaves original plans or verified improvements. It may
leave removable steps; it does not claim a globally shortest solution.
`check` without a deadline keeps its existing behavior and search budget.
Rebuild `glados-score` together with this school.py change; an older binary
does not support the new subcommand.

## Measurements and validation

On the saved stalled level, a 60-second hard budget returned in 60.14 seconds
and shortened the first plan from 49 to 25 steps, retaining the other three.
An independent conservative `check` accepted that 25-step plan (23.81 seconds,
including another model build).
This measures bounded progress, not completion or equivalent final quality
to an unlimited run. A 30-second budget returned in 30.12 seconds with the
originals, before an improvement was found.

On the existing plan.rs regression fixture with twelve redundant pickup /
putdown pairs, the old algorithm took 0.047 seconds and left 38 steps;
the replacement took 0.031 seconds and left 14. Both results independently
passed `glados-score check`. This small fixture demonstrates block deletion,
not a representative throughput benchmark.

Validation: 31 Rust library tests and 20 school Python tests pass. Tests cover
conservative validity of every improvement, no repeated candidates, local
single-step minimality without timeouts, expired checks returning unknown,
and retaining a complete snapshot when timeout interrupts the next write.
Tests use a workspace-local temporary directory because the sandbox denies
atomic replacement in its default Windows temporary directory.
