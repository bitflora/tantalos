# glados

Tooling for generating and solving Portal 2 Puzzle Maker levels. See [plan.md](plan.md).

**Slice 1:** `.p2c` read/write, level model, walkable regions, ASCII maps.
**Slice 2:** Tier-0 solver: portal line of sight, face classes, cubes/floor buttons/exit logic,
BFS for the shortest plan.

```bash
cargo test
cargo run -- info fixtures/synthetic/worked_example.p2c
cargo run -- solve fixtures/synthetic/*.p2c
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

## Solver (Tier 0)
`solve` reports **solved** (with the shortest plan), **unsolvable** (every reachable state
explored), **unknown** (state budget hit), or rejects the level as invalid / using items beyond
Tier 0. Search state = (region, portal face class per color, sorted cube locations). Actions:
drop, cube-step, fire, enter portal, pick up, put down (loose or on a button), toss off a ledge,
throw through a portal. Plan length counts these actions; reaching the exit region with the door
open ends the plan (the plan's worked example is 5, not 6, because walking out isn't counted).

| Chamber | Verdict |
|---|---|
| flat_room | 0 |
| goo_crossing | 3: portal over the goo |
| step_ledge_cube | 1: climb on the cube |
| pit_button | 2: toss the cube into the pit (jumping in is a trap) |
| pillar_button | 4: throw the cube through a portal onto the button |
| worked_example | 5: carry the cube through a portal to the button |
| goo_pit, step_ledge, worked_example_black | unsolvable |

## Assumptions to verify in-game
- Goo floods every open cell reachable sideways or downward from a goo item.
- A 1-voxel ledge needs a cube to climb; 2+ voxels can't be climbed; any drop is survivable.
- Generated files open in the editor and compile.
- Line of sight: from the centre of a standable cell (64u eye) to the centre of the face.
  Grazing an edge counts as blocked.
- Wall and floor portals can be walked into from the standable cell in front; ceiling portals
  can't. Coming out, players and cubes drop straight down with no momentum (no flings yet).
- Floor buttons are held only by cubes. Exit door = AND of its buttons, inverted by `Start Open`.
- The portal gun can't fire while carrying a cube.
- All cube and floor button types are interchangeable; cubes are never destroyed.
