//! Hand-designed test chambers. `glados fixtures` writes them to `fixtures/synthetic/`;
//! tests check the committed files match this code.
//!
//! All use the Puzzle Maker convention of a solid floor layer at z=0 (so pits can be dug).

use crate::builder::Builder;
use crate::geom::Dir;
use crate::p2c::Puzzle;

/// One room; entry and exit on the same floor.
pub fn flat_room() -> Puzzle {
    let mut b = Builder::new("Flat room", [6, 6, 4]);
    b.carve([0, 0, 1], [5, 5, 3]);
    b.entry([2, 0, 1], Dir::PosY);
    b.exit([3, 5, 1], Dir::NegY);
    b.build()
}

/// A goo trench splits the room in two; no way across on foot.
pub fn goo_pit() -> Puzzle {
    let mut b = Builder::new("Goo pit", [8, 6, 4]);
    b.carve([0, 0, 1], [7, 5, 3]);
    b.carve([3, 0, 0], [4, 5, 0]);
    b.goo([3, 2, 0]);
    b.entry([1, 0, 1], Dir::PosY);
    b.exit([6, 5, 1], Dir::NegY);
    b.build()
}

/// `goo_pit` with a white wall on each side: portal across.
pub fn goo_crossing() -> Puzzle {
    let mut b = Builder::new("Goo crossing", [8, 6, 4]);
    b.carve([0, 0, 1], [7, 5, 3]);
    b.carve([3, 0, 0], [4, 5, 0]);
    b.goo([3, 2, 0]);
    b.paint([0, 0, 1], [0, 5, 1], Dir::NegX, true);
    b.paint([7, 0, 1], [7, 5, 3], Dir::PosX, true);
    b.entry([1, 0, 1], Dir::PosY);
    b.exit([6, 5, 1], Dir::NegY);
    b.build()
}

/// The back half is raised one voxel: climbable only with a cube, droppable freely.
pub fn step_ledge() -> Puzzle {
    let mut b = Builder::new("Step ledge", [6, 8, 4]);
    b.carve([0, 0, 1], [5, 7, 3]);
    b.fill([0, 4, 1], [5, 7, 1]);
    b.entry([2, 0, 1], Dir::PosY);
    b.exit([2, 7, 2], Dir::NegY);
    b.build()
}

/// `step_ledge` with a cube in the lower half to climb on.
pub fn step_ledge_cube() -> Puzzle {
    let mut b = Builder::new("Step ledge with cube", [6, 8, 4]);
    b.carve([0, 0, 1], [5, 7, 3]);
    b.fill([0, 4, 1], [5, 7, 1]);
    b.entry([2, 0, 1], Dir::PosY);
    b.exit([2, 7, 2], Dir::NegY);
    b.cube_dropper([2, 2, 3]);
    b.build()
}

/// The exit button is at the bottom of a 1x1 pit two voxels deep. Toss the cube in;
/// jumping in with it is a trap.
pub fn pit_button() -> Puzzle {
    let mut b = Builder::new("Pit button", [6, 6, 5]);
    b.carve([0, 0, 2], [5, 5, 4]);
    b.carve([4, 3, 0], [4, 3, 1]);
    b.entry([1, 0, 2], Dir::PosY);
    let exit = b.exit([2, 5, 2], Dir::NegY);
    let button = b.floor_button([4, 3, 0]);
    b.cube_dropper([1, 3, 4]);
    b.connect(button, exit);
    b.build()
}

/// The exit button is on top of a pillar two voxels high, under a white ceiling tile. A
/// white wall in the room is the way in: throw the cube through and it falls onto the button.
pub fn pillar_button() -> Puzzle {
    let mut b = Builder::new("Pillar button", [8, 6, 6]);
    b.carve([0, 0, 1], [7, 5, 5]);
    b.fill([6, 2, 1], [6, 2, 2]);
    b.paint([0, 0, 1], [0, 5, 1], Dir::NegX, true);
    b.paint([6, 2, 5], [6, 2, 5], Dir::PosZ, true);
    b.entry([2, 0, 1], Dir::PosY);
    let exit = b.exit([3, 5, 1], Dir::NegY);
    let button = b.floor_button([6, 2, 3]);
    b.cube_dropper([2, 3, 5]);
    b.connect(button, exit);
    b.build()
}

/// The plan's worked example: room A, exit on ledge B two voxels up. White wall in A,
/// white wall above B that A can see. Exit wired to a button on B; cube dropper in A.
/// Intended solution: get cube, portal A -> B, put cube on button, exit.
pub fn worked_example() -> Puzzle {
    worked(true)
}

/// Same chamber with every surface black: the portal route is gone.
pub fn worked_example_black() -> Puzzle {
    worked(false)
}

fn worked(white: bool) -> Puzzle {
    let title = if white {
        "Worked example"
    } else {
        "Worked example (black)"
    };
    let mut b = Builder::new(title, [8, 8, 6]);
    b.carve([0, 0, 1], [7, 7, 5]);
    b.fill([0, 5, 1], [7, 7, 2]);
    if white {
        b.paint([7, 0, 1], [7, 4, 4], Dir::PosX, true);
        b.paint([0, 5, 3], [0, 7, 5], Dir::NegX, true);
    }
    b.entry([4, 0, 1], Dir::PosY);
    let exit = b.exit([4, 7, 3], Dir::NegY);
    let button = b.floor_button([2, 6, 3]);
    b.cube_dropper([2, 2, 5]);
    b.connect(button, exit);
    b.build()
}

pub fn all() -> Vec<(&'static str, Puzzle)> {
    vec![
        ("flat_room", flat_room()),
        ("goo_pit", goo_pit()),
        ("goo_crossing", goo_crossing()),
        ("step_ledge", step_ledge()),
        ("step_ledge_cube", step_ledge_cube()),
        ("pit_button", pit_button()),
        ("pillar_button", pillar_button()),
        ("worked_example", worked_example()),
        ("worked_example_black", worked_example_black()),
    ]
}
