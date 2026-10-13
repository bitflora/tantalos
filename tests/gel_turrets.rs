//! Turrets and gel on chambers built so the answer is known.

use glados::builder::Builder;
use glados::fixtures;
use glados::geom::Dir;
use glados::level::Level;
use glados::p2c::Puzzle;
use glados::portals::Landing;
use glados::regions::{Move, RegionGraph};
use glados::solver::{judge, Action, Color, Judgement, Mode, Model, Verdict, DEFAULT_MAX_STATES};

fn model(p: &Puzzle, mode: Mode) -> Model {
    Model::new(&Level::from_puzzle(p), mode).unwrap()
}

fn plan(p: &Puzzle, mode: Mode) -> Option<Vec<Action>> {
    match model(p, mode).solve(DEFAULT_MAX_STATES).verdict {
        Verdict::Solved(plan) => Some(plan),
        _ => None,
    }
}

#[test]
fn real_turrets_and_gel_decode() {
    let text = std::fs::read_to_string("fixtures/real/1500854111.p2c").unwrap();
    let level = Level::from_puzzle(&Puzzle::parse(&text).unwrap());
    let yaws: Vec<f64> = level.turrets.iter().map(|t| t.yaw).collect();
    assert_eq!(yaws, [90.0, 45.0, 180.0]);
    assert_eq!(level.gels.len(), 1);
    let gel = &level.gels[0];
    assert_eq!(gel.face, ([24, 18, 3], Dir::NegZ));
    assert_eq!(gel.dropper, Some((11, [24, 18, 5])));
    assert!(!gel.start_enabled, "wired to a button, so it starts off");
}

#[test]
fn turrets_are_knocked_over_or_run_past() {
    let p = fixtures::turret_corridor();
    let strict = plan(&p, Mode::Conservative).unwrap();
    assert!(matches!(strict[0], Action::Knock { turret: 0 }), "{strict:?}");
    // Liberal rules run past it (walking out of its sight is the one action).
    let loose = plan(&p, Mode::Liberal).unwrap();
    assert!(!loose.iter().any(|a| matches!(a, Action::Knock { .. })), "{loose:?}");
    // Regions split where the turret can shoot.
    let m = model(&p, Mode::Conservative);
    assert!(m.graph.regions.iter().any(|r| r.danger == 1));
    assert!(m.graph.regions.iter().any(|r| r.danger == 0));
}

#[test]
fn a_turret_facing_the_entry_blocks_it() {
    let mut b = Builder::new("Turret ahead", [8, 3, 4]);
    b.carve([0, 0, 1], [7, 2, 3]);
    b.entry([0, 1, 1], Dir::PosX);
    b.exit([7, 1, 1], Dir::NegX);
    b.turret([5, 1, 1], 0); // looks -x (Facing + 180)
    let p = b.build();
    assert!(plan(&p, Mode::Conservative).is_none());
    // Glass between them stops the bullets; going round it is still in the line of fire.
    let mut b = Builder::new("Turret behind glass", [8, 3, 4]);
    b.carve([0, 0, 1], [7, 2, 3]);
    b.entry([0, 1, 1], Dir::PosX);
    b.exit([3, 0, 1], Dir::PosY);
    b.turret([5, 1, 1], 0); // looks -x (Facing + 180)
    b.glass([4, 0, 1], Dir::PosX, Dir::PosY, 0, 2, 2, false);
    assert!(plan(&b.build(), Mode::Conservative).is_some());
}

#[test]
fn cubes_and_portals_knock_turrets_over() {
    let mut b = Builder::new("Turret on white", [8, 3, 4]);
    b.carve([0, 0, 1], [7, 2, 3]);
    b.paint([5, 1, 1], [5, 1, 1], Dir::NegZ, true);
    b.paint([0, 0, 1], [0, 2, 3], Dir::NegX, true);
    b.entry([0, 1, 1], Dir::PosX);
    b.exit([7, 1, 1], Dir::NegX);
    b.turret([5, 1, 1], 180); // looks +x, away from the player
    let p = b.build();
    let level = Level::from_puzzle(&p);
    let graph = RegionGraph::build(&level);
    let landing = Landing::from_fall(&level, &graph, [5, 1, 3]).unwrap();
    assert_eq!(landing.turret, Some(0), "a cube dropped here lands on the turret");

    let m = model(&p, Mode::Conservative);
    let under = m
        .portals
        .classes
        .iter()
        .find(|c| c.turret == Some(0))
        .expect("the face under the turret is its own class");
    let mut s = m.initial();
    s.portals[0] = Some(under.id);
    let fired = m
        .successors(&s)
        .into_iter()
        .find(|(a, _)| matches!(a, Action::Fire { color: Color::Orange, .. }))
        .expect("a second portal can be fired");
    assert_eq!(fired.1.down, 1, "with both portals open the turret falls through");
}

#[test]
fn repulsion_gel_bounces() {
    let plan = plan(&fixtures::bounce_gel(), Mode::Conservative).unwrap();
    assert_eq!(plan.len(), 1);
    assert!(matches!(plan[0], Action::Move { kind: Move::Bounce, .. }));
}

#[test]
fn propulsion_gel_leaps() {
    let plan = plan(&fixtures::speed_gel(), Mode::Conservative).unwrap();
    assert_eq!(plan.len(), 1);
    assert!(matches!(plan[0], Action::Move { kind: Move::Leap, .. }));
}

#[test]
fn conversion_gel_takes_portals() {
    let p = fixtures::conversion_gel();
    let plan = plan(&p, Mode::Conservative).unwrap();
    assert_eq!(plan.len(), 3);
    let m = model(&p, Mode::Conservative);
    assert!(plan.iter().any(|a| matches!(*a,
        Action::Fire { class, .. } if m.portals.classes[class].needs_paint.is_some())));
}

#[test]
fn gel_goes_through_portals() {
    let p = fixtures::redirect_gel();
    let plan = plan(&p, Mode::Conservative).unwrap();
    assert_eq!(plan.len(), 3, "{plan:?}");
    assert!(matches!(plan[2], Action::Move { kind: Move::Bounce, .. }));
}

#[test]
fn water_washes_gel_off() {
    let judged = |water| judge(&Level::from_puzzle(&fixtures::washed_gel(water)), 100_000);
    assert!(matches!(judged(false).unwrap().judgement, Judgement::Solvable(_)));
    assert_eq!(judged(true).unwrap().judgement, Judgement::Unsolvable);
}

#[test]
fn a_gel_dropper_can_be_wired() {
    let mut b = Builder::new("Wired gel", [6, 3, 6]);
    b.carve([0, 0, 1], [5, 2, 5]);
    b.fill([3, 0, 1], [5, 2, 2]);
    b.entry([0, 1, 1], Dir::PosX);
    b.exit([5, 1, 3], Dir::NegX);
    let dropper = b.gel_dropper([2, 1, 1], 0, false);
    let pedestal = b.pedestal([0, 0, 1], None);
    b.connect(pedestal, dropper);
    let plan = plan(&b.build(), Mode::Conservative).unwrap();
    assert_eq!(plan.len(), 2, "press, then bounce: {plan:?}");
    assert!(matches!(plan[0], Action::Press { .. }));
}
