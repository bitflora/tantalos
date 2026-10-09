//! Metrics on the synthetic chambers, where the right answers are known by design.

use glados::fixtures;
use glados::level::Level;
use glados::metrics::{assess, Assessment, Metrics};
use glados::p2c::Puzzle;
use glados::solver::{Model, DEFAULT_MAX_STATES};

fn measure(p: &Puzzle) -> Metrics {
    let mut model = Model::new(&Level::from_puzzle(p)).unwrap();
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
    let walk = measure(&fixtures::step_ledge_cube()).score();
    for p in [fixtures::goo_crossing(), fixtures::worked_example()] {
        assert!(measure(&p).score() > walk, "{}", p.title());
    }
}

#[test]
fn random_players_are_seeded() {
    let p = fixtures::worked_example();
    let run = |seed| {
        let mut model = Model::new(&Level::from_puzzle(&p)).unwrap();
        match assess(&mut model, DEFAULT_MAX_STATES, seed) {
            Assessment::Solved(m) => m.random_success,
            _ => unreachable!(),
        }
    };
    assert_eq!(run(3), run(3));
}
