# Structural mutation experiment, 2026-11-10

The user found both hand-built families solvable, simple, and too similar. This
experiment stops extending those templates. It starts from archived Workshop
levels with stored solutions, checks the solutions against the current model,
and changes existing control dependencies. All outputs are attributed derivatives
of those levels; their geometry and most design decisions belong to the originals.

## Reusable tool

```powershell
cargo build --release --bin tantalos-structural
target/release/tantalos-structural.exe PARENT.tokens.txt PARENT.plan.txt NEW_OUT_DIR
target/release/tantalos-structural.exe audit CANDIDATE.tokens.txt
cargo test --test structural
```

`src/structural.rs` enumerates three edits: replace a mechanism's controlling
signal, add an AND prerequisite from another existing signal, or exchange two
mechanisms' controlling signals. It adds no rooms or items and leaves exit wires
alone. The compiled token roundtrip reconstructs item wiring/counts.

The search rejects editor-placement failures, unchanged original solutions,
short solutions, large losses of estimated dependency depth, ungroundable plans,
and changes with fewer than three resource-operation edits. Ordinary walking,
portal firing and portal entry are excluded from that last comparison. These
are screening heuristics, **not an interestingness score**; independent action
reordering and incidental placement differences can still inflate the distance.
Final selections therefore require inspection beyond this screen.

Witness repair gets eight edits and 10,000 nodes. A separate 100,000-state BFS
looks for shortcuts and measures the shortest solution. A search timeout may
retain an independently replayed repair witness, clearly labeled
`shortest_proven: false`. No selected candidate in this experiment needed that
exception. `shortest_proven` refers only to the conservative discrete model.
All reported lengths are macro-actions, not seconds, human difficulty, or proof
that real-game shortcuts do not exist.

The `audit` command tries conservative and liberal models with and without angled
laser spots at 250,000 states each. Separate restricted conservative searches
test whether portals, laser redirection, timers, funnels, stairs, and cube throws
or reaches can be avoided. An unfinished/incomplete search reports unknown.

## Search and selection

Artifacts: `out/structural-2026-11-10/`.

- Inspected archived, rated Workshop parents and checked their stored solutions.
  A parent whose plan could not replay conservatively was rejected. Parents with
  grounding failures were not used to claim a verified parent solution.
- Ran 463 dependency mutations over six source puzzles and one second-generation
  variant. Seven candidates survived the first screen.
- Tested all eight stored original plans against every initial survivor. None
  still solved it; repeated this against the final three, again zero of 24.
- Rejected the stairs-coupling variant of Button Presser after the liberal model
  found a 12-action shortcut. A second generation coupled the floor button to
  the reflector dropper as well as the timed access field, changing recovery
  order and eliminating that short route in the checked models.
- Picked one candidate per source puzzle. Five initial survivors passed real
  Portal 2 editor load/export checks. Final named copies are checked separately
  in `selected/editor-checks.jsonl`.

| Local name | Original source | Logic change | Conservative / liberal shortest |
|---|---|---|---|
| Crossed Signals | [Sacrifice, 817270317](https://steamcommunity.com/sharedfiles/filedetails/?id=817270317) | Exchange the laser catcher's and floor button's control of two different fields | 23 / 23 |
| Return Circuit | [Recuberation, 131715752](https://steamcommunity.com/sharedfiles/filedetails/?id=131715752) | One cube-operated signal now changes both portal panels, forcing a different recovery and return route | 26 / 25 |
| Recovery Order | ["button presser" test chamber, 3677356906](https://steamcommunity.com/sharedfiles/filedetails/?id=3677356906) | Couple timed field access to the floor button, then move the reflector dropper's control to that same button | 18 / 18 |

The angled models returned the same lengths as their corresponding plain models.
All three final copies passed the real editor check and were staged with previews:
`tantalos-gen-00007.p2c` (Crossed Signals), `tantalos-gen-00008.p2c`
(Return Circuit), and `tantalos-gen-00009.p2c` (Recovery Order). Their Puzzle Maker
titles begin `gen Structural 1`, `2`, and `3`, and explicitly say Workshop remix.

Crossed Signals requires portals, laser redirection, and timed actions in the
conservative model. Return Circuit requires portals and funnel travel. Recovery
Order requires portals, laser redirection, timed actions, and stairs. A mechanic
being present or used by one witness is not automatically called necessary.

`selected/levels.jsonl` records original Workshop IDs/URLs, the corpus symmetry,
source file, mutation history, plans, and provenance. The geometry remains from
the source puzzles. These are local playtest remixes, not original authored maps,
and were not published or added to training. Recorded Workshop ratings were used
only to choose archived parents, not as ratings of these variants.

Real-editor acceptance and model-checked solutions are not in-game playthrough
verification. Human playtests still determine whether this batch is interesting,
too similar to its parents, or exploitable. This experiment demonstrates
structural editing of rich parents, not open-ended invention of rich puzzles.
