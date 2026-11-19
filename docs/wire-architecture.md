# Device-and-wire generator

`--arch wire` selects `model.WireGPT`. It uses the existing GPT backbone and sampling
engine, including KV caching, batched grammar calls, CPU/GPU overlap and batch refill.
The default 6-layer, 384-wide model has 11,185,920 parameters excluding positions.

## Representation and training

The input files and identity-based train/validation split remain unchanged. Loading a
wire vocabulary converts each level to an unwired geometry/device body, followed by
`<wires>` and zero or more `<wire> <ptr:SOURCE> KIND <ptr:TARGET>` records. Device IDs
refer to their order in the body. `polarity` is distinct from ordinary funnel activation;
field IDs still share the fizzler/laserfield namespace. All device attributes and geometry
tokens are preserved, including legacy geometry. Compilation restores family-local target
indices and inline links. The pointer vocabulary supports 256 devices, more than fit in
the standard 1024-token block. Overlong examples are counted by the existing loader.

Both endpoints use a learned attention pointer over completed device representations,
read at each device's last attribute token. A source must send signals; a target must match
the connection kind (paint splats cannot receive signals). The pointer's cross-entropy is
part of the ordinary next-token loss. Endpoint features use only the prefix already seen.
The existing cached data loader includes the representation code in its cache key, and
the training-away path remaps rejected coordinates after inline connections move.

Checkpoints save `arch`, the extended vocabulary and pointer weights. Old checkpoints
still load as their original architectures. Wire is a level generator, not a Goblin solver.
Flat-level value heads cannot guide its internal representation.

## Sampling and accounting

With `--constrained`, the Rust grammar checks the unwired body. Once it permits the body
to end, the model may emit `<wires>`. The wire phase masks both endpoint types and link
kinds. This differs from flat constrained decoding and is recorded as `wire-constrained`
in the run registry. Without the Rust constraint, the body is free-decoded but connections
remain typed (`wire-typed`). Existing sampling calls return standard level tokens.

An incomplete or incompatible structured result produces `<wire-compile-failure>`, which
the standard scorer counts as unparsed. It is never resampled or omitted. Evaluation JSON
records attempts, compilation failures and reasons, wire counts, and solutions with the
existing `powered` mechanic bit. That bit measures a mechanism being powered along a
solution; it does not prove every generated wire is necessary. Rated-style rows and reports
carry these counts too. Standard solver unknowns and editor failures retain their usual
outcomes. Prefixes must stop before inline connections; compilation that would alter a
supplied prefix is counted as a failure.

## Validation

Completed 2026-11-19. Machine-readable evidence is in
[wire-readiness-2026-11-19.json](wire-readiness-2026-11-19.json).

- All 2,417,554 rows in the ten default files round-trip exactly, with zero failures.
  Their hashes match the frozen bake-off manifest. The largest level has 154 devices.
- At block 1024, 5,672 wire sequences are too long, versus 228 flat sequences. This
  excludes an additional 5,444 rows (0.23% of the input), a representation-specific
  difference to retain in the bake-off interpretation.
- Nine wire tests, eight existing-architecture tests and four value-head tests pass.
- The tiny model overfits 32 distinct wired levels to loss 0.0541 in 301 steps and
  reloads identically. All 32 greedy samples parse; 28/32 stochastic samples parse,
  with four incomplete sequences counted as compilation failures.
- The production-size model fits batch 16 at block 1024, including dense pointer
  decisions and an optimizer step: peak allocated GPU memory 2,333,122,048 bytes
  (2.17 GiB) on the 4 GB RTX 3050, using PyTorch 2.14.1+cu130.
- The normal evaluator and rated-style path complete and account for all 130 + 4
  attempts. The tiny model was overfit with an unspecified-condition prompt; its
  unfamiliar condition prompts produce many failures (86/130, plus 1/4 rated),
  all recorded. These scores are not a bake-off result.

`tests/test_wire.py` covers conversion, source/target masking, gradients, full-vs-cached
decoding, left padding, prefix-only features, checkpoint reload, batch refill, failure
accounting, data split/cache behavior and training-away coordinate remapping.

Run the reproducible readiness checks from a checkout with its training environment:

```text
.venv/Scripts/python -m unittest discover -s tests -p test_wire.py -v
.venv/Scripts/python train/wire_readiness.py --data-root . --out out/wire-audit --audit
.venv/Scripts/python train/wire_readiness.py --data-root . --scorer target/release/tantalos-score.exe --out out/wire-smoke --smoke
```

The audit reads every default `data/v5b/levels-*.jsonl` row, verifies exact round trips,
records input hashes and compares block-length exclusions. The smoke test overfits 32
distinct wired training levels, saves/reloads, sends 32 stochastic and 32 greedy samples
through the standard scorer, and measures a production-size optimizer step at batch 16,
block 1024 with dense pointer decisions. Greedy parsing and overfit are gates; stochastic
failures remain visible in the report. This is implementation validation, not evidence of
level quality or a substitute for the bake-off's full evaluation and blind play-tests.

For the planned fresh bake-off run, after readiness passes:

```text
.venv/Scripts/python train/run.py start arch-wire-12h --hours 12 --note "Device-and-wire arm; docs/architecture-plan-2026-11-15.md" -- --arch wire
.venv/Scripts/python train/run.py wait arch-wire-12h
```

Follow the normal run workflow and its NEXT STEPS. The implementation task does not start
that 12-hour run or launch Portal 2.
