//! Tier-0 solver on the synthetic chambers: verdicts, shortest plans, mutations.

use glados::fixtures;
use glados::geom::Dir;
use glados::level::{ItemKind, Level};
use glados::p2c::Puzzle;
use glados::portals::{line_of_sight, Landing};
use glados::regions::Move;
use glados::solver::{
    judge, Action, CubeLoc, Judgement, Mode, Model, Problem, Verdict, DEFAULT_MAX_STATES,
};

fn model(p: &Puzzle) -> Model {
    model_in(p, Mode::Conservative)
}

fn model_in(p: &Puzzle, mode: Mode) -> Model {
    Model::new(&Level::from_puzzle(p), mode).unwrap()
}

fn solve(p: &Puzzle) -> Verdict {
    model(p).solve(DEFAULT_MAX_STATES).verdict
}

/// Solves (conservative), checks the plan replays through `successors` to a goal state.
fn plan(p: &Puzzle) -> Vec<Action> {
    plan_in(p, Mode::Conservative)
}

fn plan_in(p: &Puzzle, mode: Mode) -> Vec<Action> {
    let m = model_in(p, mode);
    let Verdict::Solved(plan) = m.solve(DEFAULT_MAX_STATES).verdict else {
        panic!("{} not solved", p.title());
    };
    let mut s = m.start.clone();
    for a in &plan {
        s = m
            .successors(&s)
            .into_iter()
            .find(|(b, _)| b == a)
            .unwrap_or_else(|| panic!("{a:?} not available"))
            .1;
    }
    assert!(m.is_goal(&s));
    plan
}

#[test]
fn shortest_plan_lengths() {
    let cases = [
        (fixtures::flat_room(), 0),
        (fixtures::goo_crossing(), 3),
        (fixtures::pit_button(), 2),
        (fixtures::pillar_button(), 4),
        (fixtures::worked_example(), 5),
    ];
    for (p, len) in cases {
        assert_eq!(plan(&p).len(), len, "{}", p.title());
    }
}

#[test]
fn unsolvable_chambers() {
    for p in [
        fixtures::goo_pit(),
        fixtures::step_ledge(),
        fixtures::worked_example_black(),
    ] {
        assert_eq!(solve(&p), Verdict::Unsolvable, "{}", p.title());
    }
}

#[test]
fn worked_example_carries_the_cube_through_a_portal() {
    let p = plan(&fixtures::worked_example());
    let enter = p
        .iter()
        .position(|a| matches!(a, Action::Enter { .. }))
        .unwrap();
    let pickup = p
        .iter()
        .position(|a| matches!(a, Action::PickUp { .. }))
        .unwrap();
    assert!(pickup < enter);
    assert!(matches!(
        p.last(),
        Some(Action::PutDown {
            at: CubeLoc::OnButton(0)
        })
    ));
}

#[test]
fn worked_example_face_classes() {
    let m = model(&fixtures::worked_example());
    let (a, b) = (m.start.region, m.exit_region);
    let land = |region| {
        Some(Landing {
            region,
            button: None,
            hazards: 0,
        })
    };
    let mut got: Vec<_> = m
        .portals
        .classes
        .iter()
        .map(|c| (c.entry, c.landing, c.faces.len()))
        .collect();
    got.sort();
    let mut want = vec![
        (None, land(a), 15),   // wall in A above head height
        (None, land(b), 6),    // wall above B, out of reach
        (Some(a), land(a), 5), // wall in A at floor level
        (Some(b), land(b), 3), // wall on B at floor level
    ];
    want.sort();
    assert_eq!(got, want);
    // Every class can be seen from both regions: even B's floor-level wall shows over the
    // ledge lip from A's far corner (see line_of_sight_over_the_ledge for a cell that can't).
    for c in &m.portals.classes {
        assert!(c.visible_from[a] && c.visible_from[b], "class {}", c.id);
    }
}

#[test]
fn line_of_sight_over_the_ledge() {
    let level = Level::from_puzzle(&fixtures::worked_example());
    // Ledge top is z=3. Low on the B wall: the ray clips the ledge. Higher up: clear.
    assert!(!line_of_sight(&level, [4, 1, 1], ([0, 6, 3], Dir::NegX)));
    assert!(line_of_sight(&level, [4, 0, 1], ([0, 6, 4], Dir::NegX)));
    // Own cell's faces are always visible.
    assert!(line_of_sight(&level, [7, 2, 1], ([7, 2, 1], Dir::PosX)));
}

#[test]
fn line_of_sight_through_an_edge_is_blocked() {
    // Eye (0.5, 0.5, 1.5) to the centre of cell (3,2,3)'s +X face, (4, 2.5, 3.5). At t = 1/4 the
    // ray passes exactly through the x-parallel edge at y=1, z=2, inside the x=1 column.
    // Of the four voxels touching that edge, (1,0,2) is the one the ray doesn't enter.
    let room = |blocked: bool| {
        let mut b = glados::builder::Builder::new("edge", [4, 3, 4]);
        b.carve([0, 0, 1], [3, 2, 3]);
        if blocked {
            b.fill([1, 0, 2], [1, 0, 2]);
        }
        Level::from_puzzle(&b.build())
    };
    let face = ([3, 2, 3], Dir::PosX);
    assert!(line_of_sight(&room(false), [0, 0, 1], face));
    assert!(!line_of_sight(&room(true), [0, 0, 1], face));
}

#[test]
fn pillar_button_throws_rather_than_walking_the_cube_up() {
    let p = plan(&fixtures::pillar_button());
    assert!(matches!(
        p.last(),
        Some(Action::Throw {
            to: CubeLoc::OnButton(0),
            ..
        })
    ));
}

#[test]
fn pit_button_tosses_the_cube_in() {
    let p = plan(&fixtures::pit_button());
    assert!(matches!(
        p.as_slice(),
        [
            Action::PickUp { .. },
            Action::Toss {
                to: CubeLoc::OnButton(0)
            }
        ]
    ));
}

#[test]
fn step_ledge_cube_climbs_only_with_liberal_rules() {
    let level = Level::from_puzzle(&fixtures::step_ledge_cube());
    let judged = judge(&level, DEFAULT_MAX_STATES).unwrap();
    assert!(matches!(judged.judgement, Judgement::Uncertain(Some(_))));
    let p = plan_in(&fixtures::step_ledge_cube(), Mode::Liberal);
    assert!(matches!(
        p.as_slice(),
        [Action::Move {
            kind: Move::Step,
            ..
        }]
    ));
}

#[test]
fn removing_the_cube_makes_worked_example_unsolvable() {
    let mut p = fixtures::worked_example();
    p.items.retain(|i| i.kind() != "ITEM_CUBE");
    assert_eq!(solve(&p), Verdict::Unsolvable);
}

#[test]
fn painting_the_ceiling_tile_black_makes_pillar_unsolvable() {
    let mut p = fixtures::pillar_button();
    p.voxels.set_face_portalable([6, 2, 5], Dir::PosZ, false);
    assert_eq!(solve(&p), Verdict::Unsolvable);
}

#[test]
fn start_open_inverts_the_exit_door() {
    // Inverted: the door is open while the button is up, so no cube is needed.
    let mut p = fixtures::worked_example();
    let exit = p
        .items
        .iter_mut()
        .find(|i| i.kind() == "ITEM_EXIT_DOOR")
        .unwrap();
    exit.set("ITEM_PROPERTY_START_OPEN", "1");
    let plan = plan(&p);
    assert_eq!(plan.len(), 3);
    assert!(!plan.iter().any(|a| matches!(a, Action::PickUp { .. })));
}

#[test]
fn state_budget_gives_unknown() {
    let out = model(&fixtures::worked_example()).solve(10);
    assert_eq!(out.verdict, Verdict::Unknown);
    assert_eq!(out.states, 10);
}

#[test]
fn real_files_are_rejected_with_reasons() {
    let load = |path| {
        let p = Puzzle::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
        Model::new(&Level::from_puzzle(&p), Mode::Conservative)
            .err()
            .unwrap()
    };
    let style = load("fixtures/real/1500854111.p2c");
    assert!(style.contains(&Problem::Unsupported(
        "ITEM_TURRET (item 16, 61, 99)".into()
    )));
    assert!(style.iter().all(|p| matches!(p, Problem::Unsupported(_))));
    let skybox = load("fixtures/real/1501698868.p2c");
    assert_eq!(skybox, vec![Problem::Invalid("no exit door".into())]);
}

#[test]
fn cubes_start_where_they_fall() {
    let m = model(&fixtures::worked_example());
    let level = Level::from_puzzle(&fixtures::worked_example());
    let cube = level.first(&ItemKind::Cube).unwrap();
    let locs: Vec<CubeLoc> = m.start.cubes.iter().map(|c| c.loc).collect();
    assert_eq!(
        locs,
        vec![CubeLoc::Loose(m.graph.region_of(cube.cell).unwrap())]
    );
}

#[test]
fn no_portals_while_carrying() {
    let m = model(&fixtures::worked_example());
    let (_, holding) = m
        .successors(&m.start)
        .into_iter()
        .find(|(a, _)| matches!(a, Action::PickUp { .. }))
        .unwrap();
    assert!(!m
        .successors(&holding)
        .iter()
        .any(|(a, _)| matches!(a, Action::Fire { .. })));
}
