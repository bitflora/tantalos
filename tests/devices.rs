//! Lasers, faith plates, bridges, funnels, panels, stairs, pistons, rails, flings and loose
//! cubes: each mechanic's fixture is solved the way it is meant to be, and taking the device
//! away (or switching it off) makes the level unsolvable.

use glados::builder::Builder;
use glados::fixtures;
use glados::geom::Dir;
use glados::level::{Level, LaserKind};
use glados::p2c::Puzzle;
use glados::regions::Move;
use glados::solver::{judge, Action, CubeLoc, Judgement, Mode, Model, Verdict};
use glados::tokens::{decode, encode};

const BUDGET: usize = 200_000;

fn plan(p: &Puzzle) -> Vec<Action> {
    match judge(&Level::from_puzzle(p), BUDGET).unwrap().judgement {
        Judgement::Solvable(plan) => plan,
        other => panic!("{}: {other:?}", p.title()),
    }
}

fn unsolvable(p: &Puzzle) {
    let j = judge(&Level::from_puzzle(p), BUDGET).unwrap().judgement;
    assert_eq!(j, Judgement::Unsolvable, "{}", p.title());
}

#[test]
fn reflection_cube_powers_the_catcher() {
    // On the floor its beam passes under the wall catcher; on top of the other cube it doesn't.
    unsolvable(&fixtures::laser_cube(true));
    let plan = plan(&fixtures::laser_cube_stand());
    let at_spots = plan
        .iter()
        .filter(|a| matches!(a, Action::PutDown { at: CubeLoc::Spot(_) }))
        .count();
    assert_eq!(at_spots, 2, "{plan:?}");
    unsolvable(&fixtures::laser_cube(false));
}

#[test]
fn a_relay_in_the_beam_opens_the_exit() {
    assert!(plan(&fixtures::laser_relay()).is_empty());
}

#[test]
fn faith_plate_launches_to_its_target() {
    let plan = plan(&fixtures::faith_plate());
    assert!(matches!(
        plan[..],
        [Action::Move {
            kind: Move::Plate,
            ..
        }]
    ));
}

#[test]
fn light_bridge_spans_the_goo() {
    let on = plan(&fixtures::light_bridge(false));
    assert!(on.iter().all(|a| matches!(a, Action::Move { .. })), "{on:?}");
    let switched = plan(&fixtures::light_bridge(true));
    assert!(matches!(switched[0], Action::Press { .. }), "{switched:?}");
}

#[test]
fn bridge_comes_out_of_a_portal() {
    let plan = plan(&fixtures::bridge_portal());
    let fired = plan
        .iter()
        .filter(|a| matches!(a, Action::Fire { .. }))
        .count();
    assert_eq!(fired, 2, "{plan:?}");
}

#[test]
fn funnel_lifts_to_the_ledge() {
    let plan = plan(&fixtures::funnel_lift());
    assert!(matches!(
        plan[..],
        [Action::Move {
            kind: Move::Funnel,
            ..
        }]
    ));
}

#[test]
fn stairs_need_their_button() {
    let plan = plan(&fixtures::stairs_button());
    assert!(matches!(
        plan.last(),
        Some(Action::Move {
            kind: Move::Stairs,
            ..
        })
    ));
    // Conservative rules don't climb on cubes, so without the stairs there's no way up.
    let mut b = Builder::new("No stairs", [8, 4, 5]);
    b.carve([0, 0, 1], [7, 3, 4]);
    b.fill([5, 0, 1], [7, 3, 1]);
    b.entry([0, 1, 1], Dir::PosX);
    b.exit([7, 2, 2], Dir::NegX);
    let level = Level::from_puzzle(&b.build());
    let m = Model::new(&level, Mode::Conservative).unwrap();
    assert_eq!(m.solve(BUDGET).verdict, Verdict::Unsolvable);
}

#[test]
fn piston_rises_to_bridge_the_pit() {
    let plan = plan(&fixtures::piston_bridge());
    assert!(matches!(plan[0], Action::Press { .. }), "{plan:?}");
}

#[test]
fn rail_ferries_across() {
    // An oscillating platform is a walkway: no action needed.
    assert!(plan(&fixtures::rail_ferry()).is_empty());
}

#[test]
fn panel_gate_opens_with_the_pedestal() {
    let plan = plan(&fixtures::panel_gate());
    assert!(matches!(plan[0], Action::Press { .. }), "{plan:?}");
}

#[test]
fn flip_panel_takes_portals_once_flipped() {
    let plan = plan(&fixtures::flip_crossing());
    let press = plan
        .iter()
        .position(|a| matches!(a, Action::Press { .. }))
        .unwrap();
    // The portal on the panel goes on after it flipped.
    let fire_on_panel = plan
        .iter()
        .rposition(|a| matches!(a, Action::Fire { .. }))
        .unwrap();
    assert!(press < fire_on_panel, "{plan:?}");
}

#[test]
fn fling_over_the_goo() {
    let plan = plan(&fixtures::fling());
    assert!(
        matches!(plan.last(), Some(Action::Dive { height: 6, .. })),
        "{plan:?}"
    );
}

#[test]
fn loose_cube_is_gone_once_fizzled() {
    // A fizzler between a loose cube and its button: carrying the cube through destroys it
    // for good. With a dropper, it comes back but can't get through either.
    let mut b = Builder::new("Loose cube", [8, 6, 4]);
    b.carve([0, 0, 1], [7, 5, 3]);
    b.fizzler([3, 0, 1], Dir::PosY, Dir::PosZ, 3, false, true);
    b.entry([1, 0, 1], Dir::PosY);
    let exit = b.exit([6, 5, 1], Dir::NegY);
    let button = b.floor_button([5, 2, 1]);
    b.connect(button, exit);
    b.loose_cube([1, 3, 1], 0);
    let p = b.build();
    let level = Level::from_puzzle(&p);
    assert!(level.loose_cube(level.first(&glados::level::ItemKind::Cube).unwrap().index));
    let m = Model::new(&level, Mode::Conservative).unwrap();
    assert_eq!(m.start.cubes[0].home, CubeLoc::Gone);
    unsolvable(&p);
}

#[test]
fn waiting_dropper_drops_when_told() {
    // The dropper only drops when the pedestal is pressed; the button by the exit needs the
    // cube.
    let mut b = Builder::new("Waiting dropper", [6, 6, 4]);
    b.carve([0, 0, 1], [5, 5, 3]);
    b.entry([1, 0, 1], Dir::PosY);
    let exit = b.exit([3, 5, 1], Dir::NegY);
    let button = b.floor_button([3, 3, 1]);
    b.connect(button, exit);
    let (_, dropper) = b.cube_dropper([1, 2, 3]);
    b.auto_drop(dropper, false);
    let p = b.pedestal([4, 1, 1], None);
    b.connect(p, dropper);
    let plan = plan(&b.build());
    assert!(matches!(plan[0], Action::Press { .. }), "{plan:?}");
}

#[test]
fn device_fixtures_round_trip_through_tokens() {
    for (name, p) in fixtures::all() {
        let tokens = encode(&p).unwrap_or_else(|e| panic!("{name}: {e}"));
        let back = decode(&tokens.join(" ")).unwrap();
        assert_eq!(encode(&back).unwrap(), tokens, "{name}");
    }
}

#[test]
fn decoder_checks_device_geometry() {
    for bad in [
        "size 4 4 4 s1 o3 | / laser 1 1 1 +z +z on",
        "size 4 4 4 s1 o3 | / panel 1 1 1 +z +x 5 on",
        "size 4 4 4 s1 o3 | / rail 1 1 1 2 2 1 osc",
        "size 4 4 4 s1 o3 | / piston 1 1 1 2 1 on",
        "size 4 4 4 s1 o3 | / stairs 1 1 1 +z on",
        "size 4 4 4 s1 o3 | / button 1 1 1 weighted > laser 3",
    ] {
        assert!(decode(bad).is_err(), "{bad}");
    }
    // Catchers wire like buttons.
    let ok = "size 6 6 4 s1 o3 | / entry 1 0 1 +y exit 0 4 1 +x \
              laser 5 2 1 -x -z on catcher 3 5 1 -y -z > exit";
    let p = decode(ok).unwrap();
    let level = Level::from_puzzle(&p);
    assert_eq!(
        level
            .lasers
            .iter()
            .filter(|l| l.kind == LaserKind::Catcher)
            .count(),
        1
    );
}

#[test]
fn deployed_panel_gate_stops_shots() {
    use glados::portals::{shots, Sight};
    let level = Level::from_puzzle(&fixtures::panel_gate());
    // Standing on its -x hinge edge, the panel at (3, 1, 1) stands across the face to (2, 1, 1).
    assert_eq!(level.face([2, 1, 1], Dir::PosX).panel, Some(0));
    assert_eq!(level.face([3, 1, 1], Dir::NegX).panel, Some(0));
    let sight = |panel_shift| Sight {
        blocking_hazards: 0,
        switchable_hazards: 0,
        strict_grazing: true,
        corner_targets: false,
        panel_shift,
    };
    // From the near side, the far end wall: clear only while the panel is down.
    let far = ([6, 1, 1], Dir::PosX);
    let masks: Vec<u64> = shots(&level, [1, 1, 1], far, sight(Some(5))).collect();
    assert!(!masks.is_empty() && masks.iter().all(|&m| m == 1 << 5), "{masks:?}");
    assert!(shots(&level, [1, 1, 1], far, sight(None)).any(|m| m == 0));
}
