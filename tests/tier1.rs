//! Tier 1: glass, grating, fizzlers, laser fields, pedestal buttons, cube and button types,
//! and the two solver modes.

use glados::builder::Builder;
use glados::fixtures;
use glados::geom::{rotate, Dir};
use glados::level::{BarrierKind, CubeKind, HazardKind, ItemKind, Level};
use glados::p2c::Puzzle;
use glados::solver::{judge, Action, CubeLoc, Judgement, Mode, Model, DEFAULT_MAX_STATES};

const STYLE: &str = "fixtures/real/1500854111.p2c";

fn style() -> Puzzle {
    Puzzle::parse(&std::fs::read_to_string(STYLE).unwrap()).unwrap()
}

fn verdict(p: &Puzzle) -> Judgement {
    judge(&Level::from_puzzle(p), DEFAULT_MAX_STATES)
        .unwrap()
        .judgement
}

fn solvable_in(p: &Puzzle) -> Vec<Action> {
    match verdict(p) {
        Judgement::Solvable(plan) => plan,
        other => panic!("{}: {other:?}", p.title()),
    }
}

fn sorted<T: Ord + Clone>(v: &[T]) -> Vec<T> {
    let mut v = v.to_vec();
    v.sort();
    v
}

#[test]
fn style_map_barriers_decode() {
    let level = Level::from_puzzle(&style());
    assert_eq!(level.decode_problems, Vec::<String>::new());
    assert_eq!(level.barriers.len(), 3);
    assert_eq!(level.hazards.len(), 2);

    // Glass 46: hangs from the x=21 wall, three cells along -X at z=5, on their -Y faces.
    let glass = level.barriers.iter().find(|b| b.item == 46).unwrap();
    assert_eq!(glass.kind, BarrierKind::Glass);
    assert_eq!(
        sorted(&glass.faces),
        vec![
            ([18, 23, 5], Dir::NegY),
            ([19, 23, 5], Dir::NegY),
            ([20, 23, 5], Dir::NegY)
        ]
    );
    // Grating 63 lies flat over the pit at z=0, anchored to the y=22 wall.
    let grating = level.barriers.iter().find(|b| b.item == 63).unwrap();
    assert_eq!(grating.kind, BarrierKind::Grating);
    assert_eq!(
        sorted(&grating.faces),
        vec![
            ([13, 21, 0], Dir::PosZ),
            ([14, 21, 0], Dir::PosZ),
            ([15, 21, 0], Dir::PosZ)
        ]
    );
    // Fizzler 54 runs from the x=21 wall to the x=17 wall right under glass 46, in its plane.
    let fizzler = level.hazards.iter().find(|h| h.item == 54).unwrap();
    assert_eq!(fizzler.kind, HazardKind::Fizzler);
    assert_eq!(
        sorted(&fizzler.faces),
        vec![
            ([18, 23, 4], Dir::NegY),
            ([19, 23, 4], Dir::NegY),
            ([20, 23, 4], Dir::NegY)
        ]
    );
    // Laser field 51 spans the gap from the y=21 wall back to the y=18 wall.
    let field = level.hazards.iter().find(|h| h.item == 51).unwrap();
    assert_eq!(field.kind, HazardKind::LaserField);
    assert_eq!(
        sorted(&field.faces),
        vec![([24, 19, 6], Dir::NegX), ([24, 20, 6], Dir::NegX)]
    );
}

/// Same position, same orientation of all three local axes.
fn same_item(ours: &glados::p2c::Props, theirs: &glados::p2c::Props) {
    assert_eq!(ours.kind(), theirs.kind());
    assert_eq!(ours.voxel_pos(), theirs.voxel_pos(), "{}", theirs.kind());
    let (a, b) = (ours.angles().unwrap(), theirs.angles().unwrap());
    for axis in [[1, 0, 0], [0, 1, 0], [0, 0, 1]] {
        assert_eq!(
            rotate(a, axis),
            rotate(b, axis),
            "{} axis {axis:?}",
            theirs.kind()
        );
    }
    let pos = |p: &glados::p2c::Props| p.get_f64s::<3>("LocalPos").unwrap();
    assert_eq!(pos(ours), pos(theirs), "{} LocalPos", theirs.kind());
}

fn extents_of(p: &Puzzle, anchor: i32) -> Vec<glados::p2c::Props> {
    p.connections
        .iter()
        .filter(|c| {
            c.get("Type") == Some("CONNECTION_BARRIER_ANCHOR_TO_EXTENT")
                && c.get_i32("Sender") == Some(anchor)
        })
        .map(|c| p.item(c.get_i32("Receiver").unwrap()).unwrap().clone())
        .collect()
}

#[test]
fn builder_writes_barriers_like_the_editor() {
    let real = style();
    let mut b = Builder::new("copy", [25, 25, 16]);
    let glass = b.glass([20, 23, 5], Dir::NegY, Dir::NegX, 0, 0, 2, false);
    let fizzler = b.fizzler([20, 23, 4], Dir::NegX, Dir::NegZ, 1, false, true);
    let ours = b.build();

    same_item(ours.item(glass).unwrap(), real.item(46).unwrap());
    let (mine, theirs) = (extents_of(&ours, glass), extents_of(&real, 46));
    assert_eq!(mine.len(), theirs.len());
    for (m, t) in mine.iter().zip(&theirs) {
        same_item(m, t);
    }
    same_item(ours.item(fizzler).unwrap(), real.item(54).unwrap());
    let (mine, theirs) = (extents_of(&ours, fizzler), extents_of(&real, 54));
    assert_eq!(mine.len(), theirs.len());
    for (m, t) in mine.iter().zip(&theirs) {
        same_item(m, t);
    }
}

#[test]
fn glass_stops_shots_grating_does_not() {
    assert_eq!(verdict(&fixtures::window(false)), Judgement::Unsolvable);
    assert_eq!(solvable_in(&fixtures::window(true)).len(), 3);
}

#[test]
fn glass_floor_is_a_floor() {
    let level = Level::from_puzzle(&fixtures::glass_bridge());
    let m = Model::new(&level, Mode::Conservative).unwrap();
    assert_eq!(m.graph.regions.len(), 1);
    assert_eq!(solvable_in(&fixtures::glass_bridge()).len(), 0);
}

/// `p` with the wires into its (one) fizzler cut, so it stays on.
fn unwire_fizzler(mut p: Puzzle) -> Puzzle {
    let fizzler = p
        .items
        .iter()
        .find(|i| i.kind() == "ITEM_BARRIER_HAZARD")
        .unwrap()
        .index()
        .unwrap();
    p.connections.retain(|c| {
        c.get_i32("Receiver") != Some(fizzler) || c.get("Type") != Some("CONNECTION_STANDARD")
    });
    p
}

#[test]
fn fizzler_cube_needs_the_pedestal() {
    let plan = solvable_in(&fixtures::fizzler_cube());
    assert_eq!(plan.len(), 4);
    assert!(plan.iter().any(|a| matches!(a, Action::Press { .. })));

    // Unwired, the fizzler stays on and eats the cube on the way through.
    let p = unwire_fizzler(fixtures::fizzler_cube());
    assert_eq!(verdict(&p), Judgement::Unsolvable);

    let m = Model::new(&Level::from_puzzle(&p), Mode::Conservative).unwrap();
    let (_, holding) = m
        .successors(&m.start)
        .into_iter()
        .find(|(a, _)| matches!(a, Action::PickUp { .. }))
        .unwrap();
    let (_, through) = m
        .successors(&holding)
        .into_iter()
        .find(|(a, _)| matches!(a, Action::Move { .. }))
        .unwrap();
    assert_ne!(through.region, m.start.region, "the player gets through");
    assert_eq!(
        through.cubes[0].loc, m.start.cubes[0].loc,
        "the cube respawned"
    );
}

#[test]
fn shots_pass_a_fizzler_once_it_is_off() {
    let plan = solvable_in(&fixtures::pedestal_window());
    assert_eq!(plan.len(), 4);
    // The far wall can only be shot after the press.
    let press = plan
        .iter()
        .position(|a| matches!(a, Action::Press { .. }))
        .unwrap();
    let fires: Vec<usize> = (0..plan.len())
        .filter(|&i| matches!(plan[i], Action::Fire { .. }))
        .collect();
    assert_eq!(fires.len(), 2);
    assert!(fires[1] > press);

    // Left on, it stops shots in both modes.
    let p = unwire_fizzler(fixtures::pedestal_window());
    assert_eq!(verdict(&p), Judgement::Unsolvable);
}

#[test]
fn standing_on_a_button_holds_it_for_a_shot() {
    let plan = solvable_in(&fixtures::button_window());
    assert_eq!(plan.len(), 3);
    assert!(matches!(
        plan[0],
        Action::Fire {
            stand_on: Some(_),
            ..
        }
    ));
    assert!(matches!(
        plan[1],
        Action::Fire {
            stand_on: None,
            ..
        }
    ));

    // A cube-only button ignores the player, and there is no cube.
    let mut p = fixtures::button_window();
    for it in &mut p.items {
        if it.kind() == "ITEM_BUTTON_FLOOR" {
            it.set("ITEM_PROPERTY_BUTTON_TYPE", "1");
        }
    }
    assert_eq!(verdict(&p), Judgement::Unsolvable);
    assert_eq!(
        verdict(&unwire_fizzler(fixtures::button_window())),
        Judgement::Unsolvable
    );
}

#[test]
fn timed_pedestal_races_the_clock() {
    let plan = solvable_in(&fixtures::timed_field(6));
    assert!(matches!(plan.as_slice(), [Action::Dash { .. }]));
    // 5 voxels take 4.6 s at 140 u/s but 3.4 s at 190 u/s.
    assert!(matches!(
        verdict(&fixtures::timed_field(4)),
        Judgement::Uncertain(Some(_))
    ));
    assert_eq!(verdict(&fixtures::timed_field(3)), Judgement::Unsolvable);
}

#[test]
fn sphere_button_takes_only_the_sphere() {
    let p = fixtures::sphere_button();
    let level = Level::from_puzzle(&p);
    let m = Model::new(&level, Mode::Conservative).unwrap();
    let plan = solvable_in(&p);
    assert!(matches!(
        plan.first(),
        Some(Action::PickUp {
            cube: CubeKind::Sphere,
            ..
        })
    ));
    assert_eq!(m.start.cubes.len(), 2);

    let mut no_sphere = p.clone();
    no_sphere
        .items
        .retain(|i| !(i.kind() == "ITEM_CUBE" && i.get("ITEM_PROPERTY_CUBE_TYPE") == Some("3")));
    assert_eq!(verdict(&no_sphere), Judgement::Unsolvable);
}

/// `goo_crossing` with a field across the room on the entry side.
fn guarded_crossing(laser: bool) -> Puzzle {
    let mut b = Builder::new("guarded", [8, 6, 4]);
    b.carve([0, 0, 1], [7, 5, 3]);
    b.carve([3, 0, 0], [4, 5, 0]);
    b.goo([3, 2, 0]);
    b.paint([0, 0, 1], [0, 5, 1], Dir::NegX, true);
    b.paint([7, 0, 1], [7, 5, 3], Dir::PosX, true);
    b.fizzler([1, 0, 1], Dir::PosY, Dir::PosZ, 3, laser, true);
    b.entry([1, 0, 1], Dir::PosY);
    b.exit([6, 5, 1], Dir::NegY);
    b.build()
}

#[test]
fn fizzlers_stop_shots_laser_fields_do_not() {
    assert_eq!(verdict(&guarded_crossing(false)), Judgement::Unsolvable);
    assert_eq!(solvable_in(&guarded_crossing(true)).len(), 3);
}

#[test]
fn laser_fields_stop_the_player_not_cubes() {
    let level = Level::from_puzzle(&guarded_crossing(true));
    let m = Model::new(&level, Mode::Conservative).unwrap();
    // The field splits the near side in two; there is an edge, but it is never taken.
    assert!(m.graph.edges.iter().any(|e| e.hazards != 0));
    for (a, _) in m.successors(&m.start) {
        assert!(!matches!(a, Action::Move { hazards, .. } if hazards != 0));
    }
}

#[test]
fn pedestal_timer_reads_infinity_below_three() {
    let mut b = Builder::new("p", [4, 4, 3]);
    b.carve([0, 0, 1], [3, 3, 2]);
    b.pedestal([1, 1, 1], None);
    b.pedestal([2, 2, 1], Some(10));
    let level = Level::from_puzzle(&b.build());
    let timers: Vec<_> = level
        .of_kind(&ItemKind::PedestalButton)
        .map(|p| p.timer())
        .collect();
    assert_eq!(timers, vec![None, Some(10.0)]);
}

#[test]
fn liberal_sees_at_least_what_conservative_sees() {
    for (name, p) in fixtures::all() {
        let level = Level::from_puzzle(&p);
        let (Ok(c), Ok(l)) = (
            Model::new(&level, Mode::Conservative),
            Model::new(&level, Mode::Liberal),
        ) else {
            continue;
        };
        let seen = |m: &Model| -> usize {
            m.portals
                .classes
                .iter()
                .map(|k| k.faces.len() * k.visible_from.iter().filter(|v| !v.is_empty()).count())
                .sum()
        };
        assert!(seen(&l) >= seen(&c), "{name}");
    }
}

#[test]
fn held_cube_is_tracked_by_kind() {
    let m = Model::new(
        &Level::from_puzzle(&fixtures::sphere_button()),
        Mode::Conservative,
    )
    .unwrap();
    let kinds: Vec<CubeKind> = m.start.cubes.iter().map(|c| c.kind).collect();
    assert_eq!(kinds, vec![CubeKind::Standard, CubeKind::Sphere]);
    assert!(m.start.cubes.iter().all(|c| c.home == c.loc));
    assert!(m
        .start
        .cubes
        .iter()
        .all(|c| matches!(c.loc, CubeLoc::Loose(_))));
}

/// Problems the editor reports for a level written as tokens.
fn field_problems(tokens: &str) -> Vec<String> {
    let level = Level::from_puzzle(&glados::tokens::decode(tokens).unwrap());
    level.mount_problems()
}

#[test]
fn nothing_may_sit_in_a_fields_sweep() {
    // Seen in the editor (2026-10-10): the pedestal and the entry door in the fizzler's sweep,
    // and the exit door beside its emitters on the mounting wall, were marked misplaced and the
    // level wouldn't build. Panels shaped like these built fine on their own.
    const ROOM: &str = "size 6 6 4 s1 o3 | / entry 2 0 1 +y exit 3 5 1 -y";
    let tokens = format!("{ROOM} fizzler 2 5 1 -y +z 3 on pedestal 2 5 1 inf");
    let mut puzzle = glados::tokens::decode(&tokens).unwrap();
    // That pedestal's column stood on the field's side; decoding turns it away (tokens don't
    // say which side it's on), so turn it back.
    let p = Level::from_puzzle(&puzzle).mount_problems();
    assert!(!p.iter().any(|m| m.contains("button")), "{p:?}");
    for it in puzzle.items.iter_mut().filter(|i| i.kind() == "ITEM_BUTTON_PEDESTAL") {
        it.set("Angles", "0 0 0");
    }
    let p = Level::from_puzzle(&puzzle).mount_problems();
    assert!(p.iter().any(|m| m.contains("doorentry") && m.contains("in-field")), "{p:?}");
    assert!(p.iter().any(|m| m.contains("button") && m.contains("in-field")), "{p:?}");
    // Not caught: a door beside a field's emitters, and a grating panel across a field's
    // path (another playtest level that wouldn't build). Over 2,403 levels checked in the
    // game, a door there was flagged 19-33% of the time and a grating across a fizzler's path
    // 29-54%, depending on geometry the editor module doesn't model; the in-game check
    // (train/ingame.py) catches them.
    const OPEN: &str = "size 6 6 4 s1 o3 | / entry 0 0 1 +y exit 5 5 1 -y";
    assert!(field_problems(&format!(
        "{OPEN} fizzler 2 5 1 -y +z 3 on grating 1 2 1 3 2 1 +y +z"
    ))
    .is_empty());
    // The same panel off to the side, and a door beside the far end of the field, are fine
    // (the latter built in the playtested fizzler-cube level).
    assert!(field_problems(&format!(
        "{OPEN} fizzler 2 5 1 -y +z 3 on grating 3 2 1 4 2 1 +y +z"
    ))
    .is_empty());
    assert!(field_problems(
        "size 6 6 4 s1 o3 | / entry 1 0 1 +y exit 5 5 1 -y fizzler 2 5 1 -y +z 3 on"
    )
    .is_empty());
    // Glass across the path always conflicts (211 of 211 in the game).
    let p = field_problems(&format!(
        "{OPEN} fizzler 2 5 1 -y +z 3 on glass 1 2 1 3 2 1 +y +z"
    ));
    assert!(p.iter().any(|m| m.contains("glass-across-field")), "{p:?}");
    for (name, p) in fixtures::all() {
        assert!(Level::from_puzzle(&p).mount_problems().is_empty(), "{name}");
    }
}

#[test]
fn workshop_tricks_the_editor_allows() {
    const OPEN: &str = "size 6 6 4 s1 o3 | / entry 0 0 1 +y exit 5 5 1 -y";
    let rules = |extra: &str| -> Vec<String> { field_problems(&format!("{OPEN} {extra}")) };
    // A fizzler and a laser field stacked in one plane (the game passed both it saw), and
    // two fizzlers in different planes of the same cells.
    let p = rules("fizzler 2 5 1 -y +z 3 on laserfield 2 5 1 -y +z 3 on");
    assert!(!p.iter().any(|m| m.contains("fields-overlap")), "{p:?}");
    let p = rules("fizzler 2 5 1 -y +z 3 on fizzler 2 5 3 -y -z 3 on");
    assert!(!p.iter().any(|m| m.contains("fields-overlap")), "{p:?}");
    // Two fizzlers in one plane still collide (2 of 2 in the game).
    let p = rules("fizzler 2 5 1 -y +z 3 on fizzler 2 5 2 -y +z 2 on");
    assert!(p.iter().any(|m| m.contains("fields-overlap")), "{p:?}");
    // Turrets and gel splats in a field's sweep.
    let p = rules("fizzler 2 5 1 -y +z 3 on turret 2 4 1 0 gel 2 3 1 +z bounce splat");
    assert!(!p.iter().any(|m| m.contains("in-field")), "{p:?}");
    // A turret on a floor panel; a laser on one wall of a cell, its catcher on another.
    let p = rules(
        "turret 3 2 1 0 panel 3 2 1 +z +y 2 off laser 4 3 1 -x center on catcher 4 3 1 -y center",
    );
    assert!(!p.iter().any(|m| m.contains("shared-cell")), "{p:?}");
    // But not a turret on stairs.
    let p = rules("turret 3 2 1 0 stairs 3 2 1 +y off");
    assert!(p.iter().any(|m| m.contains("shared-cell")), "{p:?}");
}
