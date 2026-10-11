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

/// A see-through wall splits the room; white walls on both sides. Glass stops portal shots
/// (unsolvable); grating lets them through (fire, fire, enter).
pub fn window(grating: bool) -> Puzzle {
    let title = if grating {
        "Grating window"
    } else {
        "Glass window"
    };
    let mut b = Builder::new(title, [8, 6, 4]);
    b.carve([0, 0, 1], [7, 5, 3]);
    // Panel on the +X faces of x=3, floor to ceiling, anchored on the floor.
    b.glass([3, 0, 1], Dir::PosX, Dir::PosZ, 5, 0, 2, grating);
    b.paint([0, 0, 1], [0, 5, 1], Dir::NegX, true);
    b.paint([7, 0, 1], [7, 5, 3], Dir::PosX, true);
    b.entry([1, 0, 1], Dir::PosY);
    b.exit([6, 5, 1], Dir::NegY);
    b.build()
}

/// A goo trench bridged by a glass floor: walk across.
pub fn glass_bridge() -> Puzzle {
    let mut b = Builder::new("Glass bridge", [8, 6, 4]);
    b.carve([0, 0, 1], [7, 5, 3]);
    b.carve([3, 0, 0], [4, 5, 0]);
    b.goo([3, 2, 0]);
    b.glass([4, 0, 0], Dir::PosZ, Dir::PosY, 0, 1, 5, false);
    b.entry([1, 0, 1], Dir::PosY);
    b.exit([6, 5, 1], Dir::NegY);
    b.build()
}

/// A fizzler splits the room; the cube is on the entry side, its button by the exit. Carrying
/// the cube through destroys it. A latching pedestal turns the fizzler off.
pub fn fizzler_cube() -> Puzzle {
    let mut b = Builder::new("Fizzler cube", [8, 6, 4]);
    b.carve([0, 0, 1], [7, 5, 3]);
    // Mounted on the y=-1 wall, three cells tall; field runs along +Y on x=3's +X faces.
    let fizzler = b.fizzler([3, 0, 1], Dir::PosY, Dir::PosZ, 3, false, true);
    b.entry([1, 0, 1], Dir::PosY);
    let exit = b.exit([6, 5, 1], Dir::NegY);
    let button = b.floor_button([5, 2, 1]);
    b.connect(button, exit);
    b.cube_dropper([1, 3, 3]);
    let pedestal = b.pedestal([2, 4, 1], None);
    b.connect(pedestal, fizzler);
    b.build()
}

/// `goo_crossing` with a fizzler along the near edge of the trench, between the player and
/// the far white wall, wired to a latching pedestal or (`button`) a floor button on the near
/// side.
fn fizzler_window(title: &str, button: bool) -> Puzzle {
    let mut b = Builder::new(title, [8, 6, 4]);
    b.carve([0, 0, 1], [7, 5, 3]);
    b.carve([3, 0, 0], [4, 5, 0]);
    b.goo([3, 2, 0]);
    b.paint([0, 0, 1], [0, 5, 1], Dir::NegX, true);
    b.paint([7, 0, 1], [7, 5, 3], Dir::PosX, true);
    // Mounted on the y=-1 wall, three cells tall; field on x=2's +X faces.
    let fizzler = b.fizzler([2, 0, 1], Dir::PosY, Dir::PosZ, 3, false, true);
    // Not at x=1: a door can't sit beside the emitters.
    b.entry([0, 0, 1], Dir::PosY);
    b.exit([6, 5, 1], Dir::NegY);
    let switch = if button {
        b.floor_button([1, 3, 1])
    } else {
        b.pedestal([1, 4, 1], None)
    };
    b.connect(switch, fizzler);
    b.build()
}

/// Press a latching pedestal to turn off the fizzler that hides the far wall, then portal.
pub fn pedestal_window() -> Puzzle {
    fizzler_window("Pedestal window", false)
}

/// Stand on the button to turn off the fizzler that hides the far wall and shoot through it.
pub fn button_window() -> Puzzle {
    fizzler_window("Button window", true)
}

/// A laser field splits a long room. A timed pedestal turns it off; run 5 voxels before it
/// comes back. With `seconds` = 6 that's fine (140 u/s); 4 only at 190 u/s; 3 never.
pub fn timed_field(seconds: u32) -> Puzzle {
    let mut b = Builder::new("Timed laser field", [10, 4, 4]);
    b.carve([0, 0, 1], [9, 3, 3]);
    let field = b.fizzler([5, 0, 1], Dir::PosY, Dir::PosZ, 3, true, true);
    b.entry([0, 1, 1], Dir::PosX);
    b.exit([9, 2, 1], Dir::NegX);
    let pedestal = b.pedestal([1, 2, 1], Some(seconds));
    b.connect(pedestal, field);
    b.build()
}

/// Sphere button by the exit; a standard cube and a sphere to choose from.
pub fn sphere_button() -> Puzzle {
    let mut b = Builder::new("Sphere button", [6, 6, 4]);
    b.carve([0, 0, 1], [5, 5, 3]);
    b.entry([1, 0, 1], Dir::PosY);
    let exit = b.exit([3, 5, 1], Dir::NegY);
    let button = b.floor_button_of([3, 3, 1], 2);
    b.connect(button, exit);
    b.cube_dropper_of([1, 2, 3], 0);
    b.cube_dropper_of([4, 1, 3], 3);
    b.build()
}

/// A turret in a corridor faces the exit; the player starts behind it and knocks it over.
pub fn turret_corridor() -> Puzzle {
    let mut b = Builder::new("Turret corridor", [8, 3, 4]);
    b.carve([0, 0, 1], [7, 2, 3]);
    b.entry([0, 1, 1], Dir::PosX);
    b.exit([7, 1, 1], Dir::NegX);
    b.turret([2, 1, 1], 0);
    b.build()
}

/// The exit is on a two-voxel ledge; repulsion gel poured in front of it bounces you up.
pub fn bounce_gel() -> Puzzle {
    let mut b = Builder::new("Bounce gel", [6, 3, 6]);
    b.carve([0, 0, 1], [5, 2, 5]);
    b.fill([3, 0, 1], [5, 2, 2]);
    b.entry([0, 1, 1], Dir::PosX);
    b.exit([5, 1, 3], Dir::NegX);
    b.gel_dropper([2, 1, 1], 0, true);
    b.build()
}

/// A goo trench three wide; propulsion gel poured at its edge lets you leap it.
pub fn speed_gel() -> Puzzle {
    let mut b = Builder::new("Speed gel", [10, 3, 4]);
    b.carve([0, 0, 1], [9, 2, 3]);
    b.carve([4, 0, 0], [6, 2, 0]);
    b.goo([5, 1, 0]);
    b.entry([0, 1, 1], Dir::PosX);
    b.exit([9, 1, 1], Dir::NegX);
    b.gel_dropper([3, 1, 1], 1, true);
    b.build()
}

/// The exit is on a high ledge under a white ceiling. Conversion gel poured on the floor
/// makes it portalable: floor portal in, ceiling portal out.
pub fn conversion_gel() -> Puzzle {
    let mut b = Builder::new("Conversion gel", [8, 3, 6]);
    b.carve([0, 0, 1], [7, 2, 5]);
    b.fill([5, 0, 1], [7, 2, 3]);
    b.paint([5, 0, 5], [7, 2, 5], Dir::PosZ, true);
    b.entry([0, 1, 1], Dir::PosX);
    b.exit([7, 1, 4], Dir::NegX);
    b.gel_dropper([2, 1, 1], 2, true);
    b.build()
}

/// `bounce_gel` with the dropper too far back: put a portal on the white floor under the
/// stream and one on the white ceiling in front of the ledge, and the gel lands there.
pub fn redirect_gel() -> Puzzle {
    let mut b = Builder::new("Redirect gel", [8, 3, 6]);
    b.carve([0, 0, 1], [7, 2, 5]);
    b.fill([5, 0, 1], [7, 2, 2]);
    b.paint([1, 1, 1], [1, 1, 1], Dir::NegZ, true);
    b.paint([4, 1, 5], [4, 1, 5], Dir::PosZ, true);
    b.entry([0, 0, 1], Dir::PosX);
    b.exit([7, 1, 3], Dir::NegX);
    b.gel_dropper([1, 1, 1], 0, true);
    b.build()
}

/// `bounce_gel` with the repulsion gel already on the floor, and water pouring on it.
pub fn washed_gel(water: bool) -> Puzzle {
    let mut b = Builder::new("Washed gel", [6, 3, 6]);
    b.carve([0, 0, 1], [5, 2, 5]);
    b.fill([3, 0, 1], [5, 2, 2]);
    b.entry([0, 1, 1], Dir::PosX);
    b.exit([5, 1, 3], Dir::NegX);
    b.gel_splat([2, 1, 1], Dir::PosZ, 0);
    if water {
        b.gel_dropper([2, 1, 1], 3, true);
    }
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
        ("glass_window", window(false)),
        ("grating_window", window(true)),
        ("glass_bridge", glass_bridge()),
        ("fizzler_cube", fizzler_cube()),
        ("pedestal_window", pedestal_window()),
        ("button_window", button_window()),
        ("turret_corridor", turret_corridor()),
        ("bounce_gel", bounce_gel()),
        ("speed_gel", speed_gel()),
        ("conversion_gel", conversion_gel()),
        ("redirect_gel", redirect_gel()),
        ("washed_gel", washed_gel(true)),
        ("timed_field", timed_field(6)),
        ("sphere_button", sphere_button()),
    ]
}
