//! Hand-designed test chambers. `glados fixtures` writes them to `fixtures/synthetic/`;
//! tests check the committed files match this code.
//!
//! All use the Puzzle Maker convention of a solid floor layer at z=0 (so pits can be dug).

use crate::builder::Builder;
use crate::geom::Dir;
use crate::level::LaserKind;
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
    b.turret([2, 1, 1], 180); // looks +x (Facing + 180)
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

/// A goo trench two wide; propulsion gel poured at its edge lets you leap it (three was too
/// far in-game).
pub fn speed_gel() -> Puzzle {
    let mut b = Builder::new("Speed gel", [10, 3, 4]);
    b.carve([0, 0, 1], [9, 2, 3]);
    b.carve([4, 0, 0], [5, 2, 0]);
    b.goo([4, 1, 0]);
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

/// The exit is wired to a laser catcher on the far wall; the emitter fires across the room.
/// A reflection cube set down in the beam, facing the catcher, powers it.
pub fn laser_cube(cube: bool) -> Puzzle {
    let title = if cube { "Laser cube" } else { "Laser, no cube" };
    let mut b = Builder::new(title, [6, 6, 4]);
    b.carve([0, 0, 1], [5, 5, 3]);
    b.entry([1, 0, 1], Dir::PosY);
    let exit = b.exit([0, 4, 1], Dir::PosX);
    b.laser(LaserKind::Emitter, [5, 2, 1], Dir::NegX, Some(Dir::NegZ), true);
    let catcher = b.laser(LaserKind::Catcher, [3, 5, 1], Dir::NegY, Some(Dir::NegZ), true);
    b.connect(catcher, exit);
    if cube {
        b.cube_dropper_of([2, 4, 3], 2);
    }
    b.build()
}

/// A laser relay in the beam's path powers the exit from the start.
pub fn laser_relay() -> Puzzle {
    let mut b = Builder::new("Laser relay", [6, 6, 4]);
    b.carve([0, 0, 1], [5, 5, 3]);
    b.entry([1, 0, 1], Dir::PosY);
    let exit = b.exit([0, 4, 1], Dir::PosX);
    b.laser(LaserKind::Emitter, [5, 2, 1], Dir::NegX, Some(Dir::NegZ), true);
    let relay = b.laser(LaserKind::Relay, [2, 2, 1], Dir::PosZ, None, true);
    b.connect(relay, exit);
    b.build()
}

/// The exit is on a ledge four voxels up; a faith plate on the floor throws you there.
pub fn faith_plate() -> Puzzle {
    let mut b = Builder::new("Faith plate", [8, 4, 8]);
    b.carve([0, 0, 1], [7, 3, 7]);
    b.fill([5, 0, 1], [7, 3, 4]);
    b.entry([0, 1, 1], Dir::PosX);
    b.exit([7, 1, 5], Dir::NegX);
    b.faith_plate([2, 1, 1], Dir::PosZ, [6, 1, 5], Dir::PosZ);
    b.build()
}

/// A goo trench; a light bridge from the near wall spans it at floor level. Off at the start
/// (`switched`), a latching pedestal turns it on.
pub fn light_bridge(switched: bool) -> Puzzle {
    let title = if switched {
        "Switched bridge"
    } else {
        "Light bridge"
    };
    let mut b = Builder::new(title, [9, 4, 4]);
    b.carve([0, 0, 1], [8, 3, 3]);
    b.carve([3, 0, 0], [5, 3, 0]);
    b.goo([4, 1, 0]);
    b.entry([1, 3, 1], Dir::NegY);
    b.exit([8, 2, 1], Dir::NegX);
    // On the near wall, across the trench at floor level.
    let bridge = b.light_bridge([0, 0, 1], Dir::PosX, Dir::PosZ, !switched);
    if switched {
        let p = b.pedestal([1, 2, 1], None);
        b.connect(p, bridge);
    }
    b.build()
}

/// The bridge projects into a white wall; a portal there and one on the white wall across
/// the trench bring it out spanning the trench.
pub fn bridge_portal() -> Puzzle {
    let mut b = Builder::new("Bridge portal", [9, 6, 5]);
    b.carve([0, 0, 1], [8, 5, 4]);
    b.carve([3, 0, 0], [5, 5, 0]);
    b.goo([4, 2, 0]);
    // Near side: the bridge runs along +y into the white wall at y = 6.
    b.light_bridge([1, 0, 1], Dir::PosY, Dir::PosZ, true);
    b.paint([1, 5, 1], [1, 5, 1], Dir::PosY, true);
    // White wall at x = -1 facing the trench, at floor height on the near side.
    b.paint([0, 3, 1], [0, 3, 1], Dir::NegX, true);
    b.entry([0, 1, 1], Dir::PosX);
    b.exit([8, 3, 1], Dir::NegX);
    b.build()
}

/// The exit is on a ledge three voxels up; a funnel from the floor beside it lifts you.
pub fn funnel_lift() -> Puzzle {
    let mut b = Builder::new("Funnel lift", [7, 3, 7]);
    b.carve([0, 0, 1], [6, 2, 6]);
    b.fill([4, 0, 1], [6, 2, 3]);
    b.entry([0, 1, 1], Dir::PosX);
    b.exit([6, 1, 4], Dir::NegX);
    b.funnel([3, 1, 1], Dir::PosZ, true, false);
    b.build()
}

/// A one-voxel ledge; stairs in front of it, retracted until a cube sits on the button.
pub fn stairs_button() -> Puzzle {
    let mut b = Builder::new("Stairs button", [8, 4, 5]);
    b.carve([0, 0, 1], [7, 3, 4]);
    b.fill([5, 0, 1], [7, 3, 1]);
    b.entry([0, 1, 1], Dir::PosX);
    b.exit([7, 2, 2], Dir::NegX);
    let stairs = b.stairs([3, 2, 1], Dir::PosX, false);
    let button = b.floor_button([1, 3, 1]);
    b.connect(button, stairs);
    b.cube_dropper([2, 0, 4]);
    b.build()
}

/// A two-deep pit cuts the room; a piston platform in it rises flush with the floor when a
/// latching pedestal is pressed. (START_UP set: wired, that starts it down.)
pub fn piston_bridge() -> Puzzle {
    let mut b = Builder::new("Piston bridge", [7, 3, 6]);
    b.carve([0, 0, 3], [6, 2, 5]);
    b.carve([3, 0, 1], [3, 2, 2]);
    b.entry([0, 1, 3], Dir::PosX);
    b.exit([6, 1, 3], Dir::NegX);
    let piston = b.piston([3, 1, 1], Dir::PosZ, 0, 2, true);
    let p = b.pedestal([1, 2, 3], None);
    b.connect(p, piston);
    b.build()
}

/// A goo trench three wide; an oscillating rail platform on its +y wall crosses it.
pub fn rail_ferry() -> Puzzle {
    let mut b = Builder::new("Rail ferry", [9, 3, 4]);
    b.carve([0, 0, 1], [8, 2, 3]);
    b.carve([3, 0, 0], [5, 2, 0]);
    b.goo([4, 1, 0]);
    b.entry([0, 1, 1], Dir::PosX);
    b.exit([8, 1, 1], Dir::NegX);
    b.rail([3, 2, 1], [5, 2, 1], true, true);
    b.build()
}

/// A one-wide corridor; an angled panel deployed at 90° stands across it until a latching
/// pedestal folds it down.
pub fn panel_gate() -> Puzzle {
    let mut b = Builder::new("Panel gate", [7, 3, 4]);
    b.carve([0, 1, 1], [6, 1, 3]);
    b.carve([0, 0, 1], [1, 2, 3]);
    b.entry([0, 0, 1], Dir::PosY);
    b.exit([6, 1, 1], Dir::NegX);
    let panel = b.angled_panel([3, 1, 1], Dir::PosZ, Dir::PosX, 90, true, false, false);
    let p = b.pedestal([1, 2, 1], None);
    b.connect(p, panel);
    b.build()
}

/// `goo_crossing` with the far wall black but for a flip panel, which a latching pedestal
/// turns white side out.
pub fn flip_crossing() -> Puzzle {
    let mut b = Builder::new("Flip crossing", [8, 6, 4]);
    b.carve([0, 0, 1], [7, 5, 3]);
    b.carve([3, 0, 0], [4, 5, 0]);
    b.goo([3, 2, 0]);
    b.paint([0, 0, 1], [0, 5, 1], Dir::NegX, true);
    let panel = b.flip_panel([7, 3, 1], Dir::NegX);
    b.entry([1, 0, 1], Dir::PosY);
    b.exit([6, 5, 1], Dir::NegY);
    let p = b.pedestal([1, 4, 1], None);
    b.connect(p, panel);
    b.build()
}

/// Jump off a high ledge into a floor portal two cells out, come out of a portal high on the
/// wall behind the ledge at speed, and fly over the goo to the exit.
pub fn fling() -> Puzzle {
    let mut b = Builder::new("Fling", [14, 3, 9]);
    b.carve([0, 0, 1], [13, 2, 8]);
    b.fill([0, 0, 1], [2, 2, 6]);
    b.paint([4, 1, 1], [4, 1, 1], Dir::NegZ, true);
    b.paint([0, 1, 8], [0, 1, 8], Dir::NegX, true);
    b.carve([6, 0, 0], [10, 2, 0]);
    b.goo([8, 1, 0]);
    b.entry([1, 2, 7], Dir::NegY);
    b.exit([13, 1, 1], Dir::NegX);
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
        ("laser_cube", laser_cube(true)),
        ("laser_no_cube", laser_cube(false)),
        ("laser_relay", laser_relay()),
        ("faith_plate", faith_plate()),
        ("light_bridge", light_bridge(false)),
        ("switched_bridge", light_bridge(true)),
        ("bridge_portal", bridge_portal()),
        ("funnel_lift", funnel_lift()),
        ("stairs_button", stairs_button()),
        ("piston_bridge", piston_bridge()),
        ("rail_ferry", rail_ferry()),
        ("panel_gate", panel_gate()),
        ("flip_crossing", flip_crossing()),
        ("fling", fling()),
    ]
}
