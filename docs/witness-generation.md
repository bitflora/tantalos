# Solution-carrying generation prototype

**Playtest outcome:** both template batches were reported solvable but simple and
similar. Work has moved to [structural mutation of verified Workshop parents](structural-generation.md).
The sections below describe the earlier experiments, not evidence of successful
complexity generation.

## Interaction designs (second experiment)

The first playtest found the three staged variants essentially identical. The
new command produces one example of each of three different dependency rules:

```powershell
cargo run --bin tantalos-witness -- interactions out/interactions-trial
cargo test --test interactions
```

- **Exchange:** the reflection cube initially weighs down the bridge button.
  Retrieve the ordinary cube across the gulf, exchange their roles, and use the
  freed reflector plus portals to power the exit while the bridge stays up.
- **Handover:** the laser initially powers one bridge. Retrieve a sphere and use
  its dedicated button to establish a different crossing; only then redirect
  the laser to the exit. A generic button would allow an exchange shortcut, so
  the counterexample test rejects that alteration.
- **Tandem:** the first reflector sends the beam through a relay to maintain
  access. Retrieve the second reflector and route that same beam through portals
  into it, keeping the first relay and final catcher powered together.

The three share a two-bank chamber layout. They are distinct resource dependency
structures, not three unrelated geometries and not unrestricted puzzle invention.
`src/interactions.rs` is a small authored design grammar: it compiles the chosen
rule into devices, wires, a dependency graph and state predicates. Local searches
attain successive graph milestones, retaining the full world state between them.
The graph itself is not yet learned or discovered by an open-ended search.

Each output carries its `.graph.json`, executable plan and grounded route. The
new command rejects bypasses **and unknown dependency audits**, unlike the earlier
single-family experiment. Independent restricted searches require the acquisition
and replacement/cooperation conditions and portal routing. These searches are
model-relative evidence, not a guarantee against game tricks. No full BFS solve
is required to admit a witnessed level. The 18/14/12-action witnesses are upper
bounds, not difficulty scores.

All three exported files in `out/interactions-2026-11-10` passed Portal 2's actual
editor load/export checks (`ingame.jsonl`). That verifies editor acceptance, not
completion of an in-game playthrough. Their cubes have respawning droppers to
recover accidental losses in goo. Nothing was added to training.

## Original single-family experiment

This is independent of `reverse`. It neither imports its construction nor chains
rooms or gadgets. The first family is one chamber with a shared cube and a
change in how the same ledge is reached:

1. Put the cube on the access button to deploy stairs.
2. Climb to the ledge and prepare a portal on the upper side of a parapet.
3. Return downstairs and reclaim the cube, retracting the stairs.
4. Use the prepared portal to carry that same cube up to the exit button.

The dependency is **temporary access → prepared alternative → resource reuse**.
Dimensions, sightlines, surfaces and placements can vary. This remains one puzzle
family; its variants are not evidence of broad creative diversity or enjoyment.
The parapet matters: an exposed upper white wall allowed a five-action shortcut
in the initial prototype. A regression test deliberately restores that shortcut.

## Run

```powershell
cargo run --bin tantalos-witness -- generate out/witness-trial 12 1
cargo run --bin tantalos-witness -- repair changed.p2c parent.plan.txt out/repaired
cargo test --test witness
```

Outputs must go to a new directory. Generation caps attempts at 30 per requested
level and reports any shortfall. Each accepted level has `.p2c`, `.tokens.txt`,
`.plan.txt` and `.route.jsonl` files, plus an entry in `levels.jsonl` recording
its parent, mutation, repair cost and dependency audit. The manifest is an
experimental artifact, intentionally not a training label file. Nothing is
automatically staged in Portal 2 or included in training.

## Acceptance and repair

Each candidate is token-roundtripped and checked for editor placement flags.
Repair searches `(world state, cursor in parent plan)` with three choices:
keep a matching action (zero cost), insert a legal action (one), or delete an
obsolete plan step (one). It preserves both prefix and suffix where possible,
including ambiguous steps that may correspond to different model transitions.
The CLI allows four edits and stores at most 5,000 search nodes. These are a
local repair budget, not a fresh shortest-path search. A returned solution need
not have minimum edit cost if the state cap clipped alternatives.

The result must ground into a concrete route, and that route's encoded plan must
pass conservative rechecking. Plan length is `witness_len`, an upper bound; there
is deliberately no `min_len` or claim of optimality. Verification uses the
existing Rust physics model, so it is not independent evidence of game accuracy.

Generation mixes new parameterized seeds with mutations of accepted pairs. It
uses the existing general-purpose mutation primitives, never their solver-based
admission loop. Add/remove-item mutations are excluded in this first family:
extra pedestals and incidental clutter should not masquerade as new insights.
Imported `.p2c` and plan pairs can use the repair command independently of the
family, with no one-cube restriction or claimed dependency audit.

## Checking the intended dependency

Separate counterexample searches ask whether a level can be solved without ever
putting its sole cube on the access button, or without firing portals. They reject
found shortcuts. Exhaustion of a complete restricted model records
`required-in-model`; a capped search or incomplete model records
`unknown-budget-or-model`. Unknown candidates remain labeled as unknown, not
proven necessary. The check does not establish uniqueness, difficulty, or rule out
unmodeled game tricks. A repair failure likewise means only that no nearby
witness was found, not that the level is unsolvable.

## Next experiment

Play-test representative variants before expanding this family or training on
it. Then add genuinely different dependency contracts (for example a bridge
changing from player transport to cube transport), and accept paired mutations
against those contracts. Compare enjoyment and causal variety, not raw plan
length, state count, or the number of files produced. The reusable repair engine
can also mutate verified Workshop level–solution pairs instead of relying on
this seed family.
