# glados

Tooling for generating and solving Portal 2 Puzzle Maker levels. See [plan.md](plan.md).

**Slice 1 (this):** `.p2c` read/write, level model, walkable regions, ASCII maps.

```bash
cargo test
cargo run -- info fixtures/synthetic/worked_example.p2c
cargo run -- roundtrip fixtures/real/*.p2c
cargo run -- fixtures            # regenerate fixtures/synthetic from src/fixtures.rs
```

## Fixtures
- `fixtures/real/`: two BEE2 style/skybox preview maps from
  [BEEmod/BEE2-items](https://github.com/BEEmod/BEE2-items/tree/master/icons/previews/p2c).
  These were the only public `.p2c` files findable on GitHub/Sourcegraph (2026-10-08).
  They are used to check round-trips and coordinate conventions. Keep them local; don't redistribute.
- `fixtures/synthetic/`: chambers built in code (`src/fixtures.rs`), including the plan's worked
  example. Item properties copy the real files, so they *should* open in the Puzzle Maker.
  **Not yet verified in-game.**

## Format facts (confirmed on the real files)
- Interior = cells `0..ChamberSize` per axis. Voxel layers cover `size + 1` per axis. The extra plane
  holds face bits for the outer walls.
- Rows are `"f" + one 0/1 char per x`, written Y-descending within each `Z` block.
- `PortalN` bit at voxel v means the face between `v - e_N` and v is white.
- `VoxelPos` is the open cell an item occupies. The Source "up" vector of `Angles` is its surface
  normal. Every door, button, dropper and light panel in both real files passes this check.

## Assumptions to verify in-game
- Goo floods every open cell reachable sideways or downward from a goo item.
- A 1-voxel ledge needs a cube to climb; 2+ voxels can't be climbed; any drop is survivable.
- Generated files open in the editor and compile.
