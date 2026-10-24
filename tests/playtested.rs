//! Workshop levels our solver called unsolvable until a play-tester showed how
//! (docs/solver-gaps-2026-10.md, "Play-tested small levels").

use glados::level::Level;
use glados::plan::{check, decode, encode};
use glados::route::route;
use glados::solver::{Action, Mode, Model, Verdict};

const BUDGET: usize = 50_000;

/// 3818309212 "???": stairs facing a wall, left sideways off their top step.
const STAIRS_SIDEWAYS: &str = "size 6 8 5 open 0 3 0 2 4 2 0 5 0 0 5 2 0 2 1 2 7 2 1 4 3 1 7 4 1 0 4 5 0 4 1 1 4 1 7 4 4 1 4 5 1 4 0 2 4 2 2 4 entry 1 0 4 +y exit 1 2 1 +y pedestal 0 2 4 inf pedestal 4 1 4 inf > panel 0 plate 1 7 1 -y 1 3 4 +z panel 1 1 4 +z -y 6 on stairs 1 3 0 +x on";

/// 68464713 "Smallest Puzzle in the Multiverse": two fizzlers crossing in a 3x2 room. Fetch the
/// cube across one while a 3 s pedestal holds it off, then reach for it across the other from
/// that one's floor button.
const SMALLEST: &str = "size 3 5 2 open 1 0 0 1 4 0 2 1 0 2 2 0 0 2 0 1 3 0 2 1 1 2 1 1 white +z 1 0 0 2 0 1 white -z 2 2 0 2 4 0 entry 1 0 0 +y exit 1 4 0 -y fizzler 2 1 0 -x +z 1 on fizzler 2 2 0 -y +z 1 on dropper 2 1 1 standard button 1 0 0 weighted > exit button 1 3 0 weighted > field 1 pedestal 1 4 0 3 > field 0";

fn level(tokens: &str) -> Level {
    Level::from_puzzle(&glados::tokens::decode(tokens).unwrap())
}

/// Solves `tokens` with conservative rules; its route's plan tokens check as solved.
fn solved(tokens: &str) -> Vec<Action> {
    let level = level(tokens);
    let model = Model::new(&level, Mode::Conservative).unwrap();
    let Verdict::Solved(plan) = model.solve(BUDGET).verdict else {
        panic!("not solved:\n{tokens}");
    };
    let r = route(&level, BUDGET).unwrap();
    let text = encode(&r);
    let c = check(&model, &decode(&text).unwrap());
    assert!(c.solved, "{text}: {:?}", c.problem);
    plan
}

#[test]
fn stairs_left_sideways() {
    let plan = solved(STAIRS_SIDEWAYS);
    assert!(plan.iter().any(|a| matches!(a, Action::Move { kind: glados::regions::Move::Stairs, .. })));
}

#[test]
fn cube_fetched_in_a_dash_and_reached_for_from_a_button() {
    let plan = solved(SMALLEST);
    assert!(plan.iter().any(|a| matches!(a, Action::Fetch { .. })), "{plan:?}");
    assert!(plan.iter().any(|a| matches!(a, Action::Reach { .. })), "{plan:?}");
}

#[test]
fn no_reach_while_the_button_leaves_the_fizzler_on() {
    // Unwired: standing on the floor button no longer turns the fizzler between it and the
    // cube off (fizzler 1 has no input and stays on).
    let tokens = SMALLEST.replace("button 1 3 0 weighted > field 1", "button 1 3 0 weighted");
    let model = Model::new(&level(&tokens), Mode::Liberal).unwrap();
    let mut s = model.initial();
    // Search everything reachable; no Reach anywhere.
    let mut seen = vec![s.clone()];
    let mut i = 0;
    while i < seen.len() && i < BUDGET {
        s = seen[i].clone();
        for (a, n) in model.successors(&s) {
            assert!(!matches!(a, Action::Reach { .. }), "{a:?}");
            if !seen.contains(&n) {
                seen.push(n);
            }
        }
        i += 1;
    }
}
