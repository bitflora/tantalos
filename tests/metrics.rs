//! Metrics on the synthetic chambers, where the right answers are known by design.

use tantalos::fixtures;
use tantalos::level::Level;
use tantalos::metrics::{assess, Assessment, Metrics};
use tantalos::p2c::Puzzle;
use tantalos::solver::{Mode, Model, DEFAULT_MAX_STATES};

fn measure(p: &Puzzle) -> Metrics {
    let mut model = Model::new(&Level::from_puzzle(p), Mode::Conservative).unwrap();
    match assess(&mut model, DEFAULT_MAX_STATES, 7) {
        Assessment::Solved(m) => m,
        other => panic!("{}: {other:?}", p.title()),
    }
}

#[test]
fn worked_example() {
    let m = measure(&fixtures::worked_example());
    assert_eq!(m.min_len, 5);
    assert_eq!(m.portal_trips, 1);
    assert_eq!(m.cubes, 1);
    assert_eq!(m.unnecessary_cubes, 0);
    // Either wall on B works as the exit; the floor-level wall in A is the only way in.
    assert_eq!((m.classes, m.necessary_classes), (4, 1));
    // B always has a way back to A (drop, or portal from B's wall), so no softlocks.
    assert_eq!(m.dead_end_states, 0);
    assert!(m.shortest_plans >= 2, "blue/orange can swap");
}

#[test]
fn pit_button_has_a_softlock() {
    // Jumping into the pit, with or without the cube, can't be undone.
    let m = measure(&fixtures::pit_button());
    assert!(m.dead_end_states > 0);
    assert!(m.dead_end_share() > 0.0 && m.dead_end_share() < 1.0);
}

#[test]
fn flat_room_is_trivial() {
    let m = measure(&fixtures::flat_room());
    assert_eq!(m.min_len, 0);
    assert_eq!(m.random_success, 1.0);
    assert_eq!(m.shortest_plans, 1);
}

#[test]
fn a_spare_cube_makes_both_cubes_unnecessary() {
    let mut p = fixtures::worked_example();
    let mut spare = p
        .items
        .iter()
        .find(|i| i.kind() == "ITEM_CUBE")
        .unwrap()
        .clone();
    spare.set("Index", "99");
    p.items.push(spare);
    let m = measure(&p);
    assert_eq!((m.cubes, m.unnecessary_cubes), (2, 2));
    assert!(m.score() < measure(&fixtures::worked_example()).score());
}

#[test]
fn portal_puzzles_outscore_walking() {
    let walk = measure(&fixtures::pit_button()).score();
    for p in [fixtures::goo_crossing(), fixtures::worked_example()] {
        assert!(measure(&p).score() > walk, "{}", p.title());
    }
}

#[test]
fn random_players_are_seeded() {
    let p = fixtures::worked_example();
    let run = |seed| {
        let mut model = Model::new(&Level::from_puzzle(&p), Mode::Conservative).unwrap();
        match assess(&mut model, DEFAULT_MAX_STATES, seed) {
            Assessment::Solved(m) => m.random_success,
            _ => unreachable!(),
        }
    };
    assert_eq!(run(3), run(3));
}

#[test]
fn worked_example_components_and_distance() {
    let m = measure(&fixtures::worked_example());
    assert_eq!(m.cubes - m.unnecessary_cubes, 1);
    assert_eq!(m.necessary_buttons, 1);
    assert_eq!(m.mechanic_names(), ["portal trip", "cube on a button"]);
    // The two shots don't need each other; walking into the portal needs both, and putting
    // the cube down needs the trip: 1 + 1 + 0.5.
    assert_eq!(m.chain_depth, 2.5);
    assert!(m.travel > 0.0 && m.cube_travel > 0.0);
}

#[test]
fn fields_that_shape_the_plan_are_load_bearing() {
    let m = measure(&fixtures::fizzler_cube());
    assert_eq!((m.necessary_pedestals, m.load_bearing_fizzlers), (1, 1));
    let m = measure(&fixtures::timed_field(6));
    assert_eq!(m.load_bearing_lasers, 1);
    assert!(m.mechanic_names().contains(&"timed dash"));
    // Only there to make the player stand on the button while shooting.
    let m = measure(&fixtures::button_window());
    assert_eq!((m.necessary_buttons, m.load_bearing_fizzlers), (1, 1));
    assert!(m.mechanic_names().contains(&"fire standing on a button"));
}

#[test]
fn flat_room_has_no_components() {
    let m = measure(&fixtures::flat_room());
    assert_eq!(m.necessary_items(), 0);
    assert_eq!(m.mechanics, 0);
    assert_eq!((m.chain_depth, m.travel, m.cube_travel, m.backtrack), (0.0, 0.0, 0.0, 0.0));
}
