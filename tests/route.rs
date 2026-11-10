//! Routes: the solver's plans made concrete for playing in the game.

use glados::fixtures;
use glados::geom::{step, Cell, Dir};
use glados::dynamics::Devices;
use glados::level::Level;
use glados::regions::{Geo, Move};
use glados::portals::{shots_from, Sight};
use glados::route::{route, Act, Route};
use glados::solver::{Mode, Model, Verdict, DEFAULT_MAX_STATES};

/// One step on level ground, or off a ledge and down to where a fall can end (devices in any
/// state).
fn legal_step(geo: Geo, a: Cell, b: Cell) -> bool {
    let level = geo.level;
    Dir::HORIZONTAL.iter().any(|&d| {
        let n = step(a, d);
        level.face(a, d).barrier.is_none()
            && !level.solid(n)
            && (n == b && geo.stand(n).is_some() || geo.falls(n).iter().any(|f| f.cell == b))
    })
}

/// A step of a timed dash: one on foot, across a corner (both cells beside it can be stood
/// in, nothing walls its four faces off), or up deployed stairs.
fn legal_dash_step(geo: Geo, a: Cell, b: Cell) -> bool {
    let level = geo.level;
    let corner = [Dir::PosX, Dir::NegX].iter().any(|&dx| {
        [Dir::PosY, Dir::NegY].iter().any(|&dy| {
            let (m1, m2) = (step(a, dx), step(a, dy));
            step(m1, dy) == b
                && [m1, m2].iter().all(|&m| geo.stand(m).is_some())
                && [(a, dx), (m1, dy), (a, dy), (m2, dx)]
                    .iter()
                    .all(|&(f, d)| level.face(f, d).barrier.is_none())
        })
    });
    let stairs = geo
        .dyn_
        .moves
        .iter()
        .any(|m| m.kind == Move::Stairs && m.from.contains(&a) && m.to.iter().any(|t| t.1 == b));
    legal_step(geo, a, b) || corner || stairs
}

fn legal_dash(geo: Geo, from: Cell, cells: &[Cell]) -> bool {
    let mut at = from;
    cells.iter().all(|&c| {
        let ok = geo.stand(c).is_some() && legal_dash_step(geo, at, c);
        at = c;
        ok
    })
}

fn legal_walk(geo: Geo, from: Cell, cells: &[Cell]) -> bool {
    let mut at = from;
    cells.iter().all(|&c| {
        let ok = geo.stand(c).is_some() && legal_step(geo, at, c);
        at = c;
        ok
    })
}

/// What a route must do beyond matching the plan, which `route` checks itself.
fn check(name: &str, level: &Level, r: &Route) {
    let devices = Devices::build(level).unwrap();
    let geo = Geo::new(level, &devices.dyn_);
    // Line of sight as the conservative rules see it, fizzlers and deployed panels aside
    // (`route` checks those).
    let sight = Sight {
        blocking_hazards: 0,
        switchable_hazards: u64::MAX,
        strict_grazing: true,
        corner_targets: false,
        panel_shift: None,
    };
    let mut player = r.start.player;
    assert_eq!(player, level.entry().unwrap().cell, "{name}");
    for (i, leg) in r.legs.iter().enumerate() {
        let at = format!("{name} leg {}: {}", i + 1, leg.text);
        assert!(
            legal_walk(geo, player, &leg.walk),
            "{at}: walk {:?} from {player:?}",
            leg.walk
        );
        let end = leg.walk.last().copied().unwrap_or(player);
        match &leg.act {
            Act::Walk => assert_eq!(leg.after.player, end, "{at}"),
            Act::Fire {
                face, cell, eye, ..
            } => {
                assert_eq!(*cell, end, "{at}");
                assert!(
                    shots_from(level, *cell, *eye, *face, sight).next().is_some(),
                    "{at}: no shot"
                );
            }
            Act::Enter { face, out, .. } => {
                assert_eq!(face.0, end, "{at}: not in front of the portal");
                let lands = geo.falls(out.0).iter().any(|f| f.cell == leg.after.player);
                assert!(lands || !devices.is_empty(), "{at}");
            }
            Act::Throw { out, .. } => assert_eq!(leg.after.player, end, "{at}: {out:?}"),
            Act::Dash { path, .. } => {
                assert!(legal_dash(geo, end, path), "{at}: dash {path:?}");
            }
            Act::Fetch {
                path, cube, carry, ..
            } => {
                assert!(legal_dash(geo, end, path), "{at}: fetch {path:?}");
                assert_eq!(path.last().unwrap_or(&end), cube, "{at}");
                assert!(legal_dash(geo, *cube, carry), "{at}: carry {carry:?}");
                assert_eq!(leg.after.player, *carry.last().unwrap_or(cube), "{at}");
            }
            Act::Reach { button, .. } => {
                assert_eq!(*button, end, "{at}");
                assert_eq!(leg.after.player, end, "{at}");
            }
            Act::Plate { from, to, .. } | Act::Ride { from, to } | Act::Jump { from, to, .. } => {
                assert_eq!(*from, end, "{at}");
                assert_eq!(leg.after.player, *to, "{at}");
            }
            Act::Funnel { enter, to, .. } => {
                assert_eq!(*enter, end, "{at}");
                assert_eq!(leg.after.player, *to, "{at}");
            }
            Act::Climb { path } => assert_eq!(path.last(), Some(&leg.after.player), "{at}"),
            Act::Dive { edge, landing, .. } => {
                assert_eq!(*edge, end, "{at}");
                assert_eq!(leg.after.player, *landing, "{at}");
            }
            Act::Loop { face, landing, .. } => {
                assert_eq!(face.0, end, "{at}");
                assert_eq!(leg.after.player, *landing, "{at}");
            }
            Act::Exit { cell, .. } => {
                assert_eq!(*cell, level.exit().unwrap().cell, "{at}");
                assert_eq!(end, *cell, "{at}");
                assert_eq!(i, r.legs.len() - 1, "{at}: exit isn't last");
            }
            Act::Bounce { gel, to } => {
                assert_eq!(*gel, end, "{at}");
                assert_eq!(leg.after.player, *to, "{at}");
            }
            Act::Leap { run, to, .. } => {
                assert_eq!(run.first(), Some(&end), "{at}");
                assert_eq!(leg.after.player, *to, "{at}");
            }
            Act::PickUp { .. }
            | Act::PutDown { .. }
            | Act::Toss { .. }
            | Act::Press { .. }
            | Act::Knock { .. } => assert_eq!(leg.after.player, end, "{at}"),
        }
        assert!(geo.stand(leg.after.player).is_some(), "{at}");
        player = leg.after.player;
    }
    let escaped = matches!(
        r.legs.last().map(|l| &l.act),
        Some(Act::Dash { escape: true, .. })
    );
    assert!(
        escaped || matches!(r.legs.last().map(|l| &l.act), Some(Act::Exit { .. })),
        "{name}: doesn't end at the exit"
    );
}

fn solvable(level: &Level, budget: usize) -> Option<bool> {
    let m = Model::new(level, Mode::Conservative).ok()?;
    match m.solve(budget).verdict {
        Verdict::Solved(_) => Some(true),
        Verdict::Unsolvable => Some(false),
        Verdict::Unknown => None,
    }
}

#[test]
fn fixtures_route_when_solvable() {
    for (name, p) in fixtures::all() {
        let level = Level::from_puzzle(&p);
        let want = solvable(&level, DEFAULT_MAX_STATES).unwrap_or(false);
        match route(&level, DEFAULT_MAX_STATES) {
            Ok(r) => {
                assert!(want, "{name}: routed but not solvable");
                check(name, &level, &r);
            }
            Err(e) => assert!(!want, "{name}: {e}"),
        }
    }
}

#[test]
fn worked_example_route() {
    let level = Level::from_puzzle(&fixtures::worked_example());
    let r = route(&level, DEFAULT_MAX_STATES).unwrap();
    let acts: Vec<&str> = r
        .legs
        .iter()
        .map(|l| match l.act {
            Act::Fire { .. } => "fire",
            Act::PickUp { .. } => "pickup",
            Act::Enter { .. } => "enter",
            Act::PutDown { button: true, .. } => "button",
            Act::Exit { .. } => "exit",
            _ => "other",
        })
        .collect();
    assert_eq!(acts, ["fire", "fire", "pickup", "enter", "button", "exit"]);
    // Carried through the portal, then put on the button.
    assert_eq!(r.legs[3].after.cubes[0].at, None);
    let lines = r.json_lines("Worked example", level.size);
    assert_eq!(lines.len(), 1 + r.legs.len());
    assert!(lines[0].starts_with("{\"title\":\"Worked example\""));
    assert!(lines[5].contains("\"action\":\"putdown\""));
}
