//! Synthetic chambers: committed files match the generator, and regions come out as designed.

use glados::fixtures;
use glados::level::{ItemKind, Level};
use glados::p2c::Puzzle;
use glados::regions::{Cond, Edge, Move, RegionGraph};

fn analyse(p: &Puzzle) -> (Level, RegionGraph) {
    let level = Level::from_puzzle(p);
    let graph = RegionGraph::build(&level);
    (level, graph)
}

fn region(level: &Level, graph: &RegionGraph, kind: ItemKind) -> usize {
    graph.region_of(level.first(&kind).unwrap().cell).unwrap()
}

fn entry_exit(level: &Level, graph: &RegionGraph) -> (usize, usize) {
    (
        region(level, graph, ItemKind::EntryDoor),
        region(level, graph, ItemKind::ExitDoor),
    )
}

#[test]
fn committed_files_are_current_and_reparse() {
    for (name, puzzle) in fixtures::all() {
        let path = format!("fixtures/synthetic/{name}.p2c");
        let on_disk = std::fs::read_to_string(&path).unwrap_or_default();
        let generated = puzzle.to_p2c();
        assert!(
            on_disk == generated,
            "{path} is stale; run `cargo run -- fixtures`"
        );
        assert_eq!(Puzzle::parse(&generated).unwrap(), puzzle, "{name}");
        assert_eq!(
            Level::from_puzzle(&puzzle).mount_problems(),
            Vec::<String>::new(),
            "{name}"
        );
    }
}

#[test]
fn flat_room_is_one_region() {
    let (level, graph) = analyse(&fixtures::flat_room());
    assert_eq!(graph.regions.len(), 1);
    assert_eq!(entry_exit(&level, &graph), (0, 0));
}

#[test]
fn goo_pit_splits_the_room() {
    let (level, graph) = analyse(&fixtures::goo_pit());
    assert_eq!(graph.regions.len(), 2);
    assert!(
        graph.edges.is_empty(),
        "walking into the pit is death, not a move"
    );
    let (a, b) = entry_exit(&level, &graph);
    assert_ne!(a, b);
    assert!(!graph.reachable_on_foot(a)[b]);
    assert_eq!(level.cells().filter(|&c| level.goo(c)).count(), 2 * 6);
}

#[test]
fn step_ledge_needs_a_cube_up_and_drops_down() {
    let (level, graph) = analyse(&fixtures::step_ledge());
    let (a, b) = entry_exit(&level, &graph);
    assert_eq!(
        graph.edges,
        vec![
            Edge {
                from: a,
                to: b,
                kind: Move::Step,
                hazards: 0,
                cond: Cond::ALWAYS,
            },
            Edge {
                from: b,
                to: a,
                kind: Move::Drop { height: 1 },
                hazards: 0,
                cond: Cond::ALWAYS,
            },
        ]
    );
    assert!(!graph.reachable_on_foot(a)[b]);
    assert!(graph.reachable_on_foot(b)[a]);
}

#[test]
fn worked_example_ledge_is_unreachable_on_foot() {
    let (level, graph) = analyse(&fixtures::worked_example());
    let (a, b) = entry_exit(&level, &graph);
    assert_eq!(graph.regions.len(), 2);
    assert_eq!((graph.regions[a].z, graph.regions[b].z), (1, 3));
    assert_eq!(
        graph.edges,
        vec![Edge {
            from: b,
            to: a,
            kind: Move::Drop { height: 2 },
            hazards: 0,
            cond: Cond::ALWAYS,
        }]
    );
    assert!(!graph.reachable_on_foot(a)[b]);

    assert_eq!(region(&level, &graph, ItemKind::Cube), a);
    assert_eq!(region(&level, &graph, ItemKind::FloorButton), b);
    // Wall in A (5 x 4 faces) + wall above B (3 x 3).
    assert_eq!(level.portal_surfaces().len(), 20 + 9);
}

#[test]
fn painting_black_removes_portal_surfaces_but_not_regions() {
    let (white_level, white) = analyse(&fixtures::worked_example());
    let (black_level, black) = analyse(&fixtures::worked_example_black());
    assert_eq!(white.regions, black.regions);
    assert_eq!(white.edges, black.edges);
    assert!(!white_level.portal_surfaces().is_empty());
    assert!(black_level.portal_surfaces().is_empty());
}

#[test]
fn wired_exit_starts_closed() {
    let p = fixtures::worked_example();
    let exit = p
        .items
        .iter()
        .find(|i| i.kind() == "ITEM_EXIT_DOOR")
        .unwrap();
    assert_eq!(exit.get("ITEM_PROPERTY_START_OPEN"), Some("0"));
    assert_eq!(exit.get("ITEM_PROPERTY_CONNECTION_COUNT"), Some("1"));
    let button = p
        .items
        .iter()
        .find(|i| i.kind() == "ITEM_BUTTON_FLOOR")
        .unwrap()
        .index()
        .unwrap();
    assert!(p.connections.iter().any(
        |c| c.get_i32("Sender") == Some(button) && c.get("Type") == Some("CONNECTION_STANDARD")
    ));
}
