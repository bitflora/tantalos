//! Routes: the solver's plans made concrete for playing in the game.

use glados::fixtures;
use glados::generator::Genome;
use glados::geom::{step, Cell, Dir};
use glados::level::Level;
use glados::portals::{shots, Sight};
use glados::rng::Rng;
use glados::route::{route, Act, Route};
use glados::solver::{Mode, Model, Verdict, DEFAULT_MAX_STATES};

/// One step on level ground, or off a ledge and down to where the fall ends.
fn legal_step(level: &Level, a: Cell, b: Cell) -> bool {
    Dir::HORIZONTAL.iter().any(|&d| {
        let n = step(a, d);
        level.face(a, d).barrier.is_none()
            && !level.solid(n)
            && if level.standable(n) {
                n == b
            } else {
                level.fall(n) == Some(b)
            }
    })
}

fn legal_walk(level: &Level, from: Cell, cells: &[Cell]) -> bool {
    let mut at = from;
    cells.iter().all(|&c| {
        let ok = level.standable(c) && legal_step(level, at, c);
        at = c;
        ok
    })
}

/// What a route must do beyond matching the plan, which `route` checks itself.
fn check(name: &str, level: &Level, r: &Route) {
    // Line of sight as the conservative rules see it, fizzlers aside (`route` checks those).
    let sight = Sight {
        blocking_hazards: 0,
        switchable_hazards: u64::MAX,
        strict_grazing: true,
        corner_targets: false,
    };
    let mut player = r.start.player;
    assert_eq!(player, level.entry().unwrap().cell, "{name}");
    for (i, leg) in r.legs.iter().enumerate() {
        let at = format!("{name} leg {}: {}", i + 1, leg.text);
        assert!(
            legal_walk(level, player, &leg.walk),
            "{at}: walk {:?} from {player:?}",
            leg.walk
        );
        let end = leg.walk.last().copied().unwrap_or(player);
        match &leg.act {
            Act::Walk => assert_eq!(leg.after.player, end, "{at}"),
            Act::Fire { face, eye, .. } => {
                assert_eq!(*eye, end, "{at}");
                assert!(
                    shots(level, *eye, *face, sight).next().is_some(),
                    "{at}: no shot"
                );
            }
            Act::Enter { face, out, .. } => {
                assert_eq!(face.0, end, "{at}: not in front of the portal");
                assert_eq!(Some(leg.after.player), level.fall(out.0), "{at}");
            }
            Act::Throw { out, .. } => assert_eq!(leg.after.player, end, "{at}: {out:?}"),
            Act::Dash { path, .. } => {
                assert!(legal_walk(level, end, path), "{at}: dash {path:?}");
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
        assert!(level.standable(leg.after.player), "{at}");
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

/// Generated levels, mostly bred from solvable ones (as `glados dataset` does) so that many
/// are solvable and use more mechanics.
#[test]
fn generated_levels_route() {
    let mut rng = Rng::new(17);
    let mut pool: Vec<Genome> = Vec::new();
    let mut kinds = std::collections::BTreeSet::new();
    for _ in 0..250 {
        let g = if !pool.is_empty() && rng.chance(0.7) {
            let mut g = rng.pick(&pool).clone();
            for _ in 0..rng.range(1, 3) {
                g.mutate(&mut rng);
            }
            g
        } else {
            Genome::random(&mut rng)
        };
        let Some(p) = g.build() else { continue };
        let level = Level::from_puzzle(&p);
        if solvable(&level, 20_000) != Some(true) {
            continue;
        }
        let r = route(&level, 20_000).unwrap_or_else(|e| {
            panic!(
                "{e}
{}",
                p.to_p2c()
            )
        });
        check("generated", &level, &r);
        pool.push(g);
        for leg in &r.legs {
            kinds.insert(
                format!("{:?}", leg.act)
                    .split([' ', '{'])
                    .next()
                    .unwrap()
                    .to_string(),
            );
        }
    }
    assert!(pool.len() > 60, "only {} solvable levels", pool.len());
    eprintln!("{} routed; acts: {kinds:?}", pool.len());
}
