//! Plans as tokens: the solver's own plans check as solved, damaged ones fail where they
//! break.

use glados::fixtures;
use glados::generator::Genome;
use glados::level::Level;
use glados::p2c::Puzzle;
use glados::plan::{check, decode, encode, encode_actions, step_text, vocab, Step};
use glados::rng::Rng;
use glados::route::{route, route_plan, route_prefix};
use glados::solver::{Action, Mode, Model, Verdict};

const BUDGET: usize = 50_000;

/// Solvable levels with their models, plan tokens and shortest plan length: the fixtures and
/// some generated levels.
fn plans() -> Vec<(String, Level, String, usize)> {
    let mut puzzles: Vec<(String, Puzzle)> = fixtures::all()
        .into_iter()
        .map(|(n, p)| (n.to_string(), p))
        .collect();
    let mut rng = Rng::new(11);
    for i in 0..200 {
        let mut g = Genome::random(&mut rng);
        for _ in 0..rng.range(0, 15) {
            g.mutate(&mut rng);
        }
        if let Some(p) = g.build() {
            puzzles.push((format!("genome {i}"), p));
        }
    }
    puzzles
        .into_iter()
        .filter_map(|(name, p)| {
            let level = Level::from_puzzle(&p);
            let model = Model::new(&level, Mode::Conservative).ok()?;
            let Verdict::Solved(plan) = model.solve(BUDGET).verdict else {
                return None;
            };
            let r = route(&level, BUDGET).unwrap_or_else(|e| panic!("{name}: {e}"));
            Some((name, level, encode(&r), plan.len()))
        })
        .collect()
}

#[test]
fn solver_plans_check_as_solved() {
    let all = plans();
    assert!(all.len() > 40, "only {} solvable levels", all.len());
    let v = vocab();
    let mut ambiguous = 0;
    for (name, level, tokens, len) in &all {
        for t in tokens.split(' ') {
            assert!(v.contains(&t.to_string()), "{name}: {t:?} missing from vocab");
        }
        let steps = decode(tokens).unwrap_or_else(|e| panic!("{name}: {e}: {tokens}"));
        // Written back, the same tokens (without the closing `exit`).
        let back: Vec<String> = steps.iter().map(step_text).collect();
        assert_eq!(
            back.join(" "),
            tokens.trim_end_matches("exit").trim_end(),
            "{name}"
        );
        let model = Model::new(level, Mode::Conservative).unwrap();
        let c = check(&model, &steps);
        assert!(c.solved, "{name}: {tokens}: {:?}", c.problem);
        assert_eq!(c.plan.len(), *len, "{name}: {tokens}");
        // A step that fits more than one distinct action leaves `check` more than one way
        // through the plan to follow; these plans are where that gets exercised.
        ambiguous += c.ambiguous as usize;
    }
    assert!(ambiguous > 0, "no plan's step fitted more than one distinct action");
}

/// Several shortest plans: the first is `solve`'s, all differ, all are as short, and each
/// grounds and checks as solved.
#[test]
fn shortest_plans_all_solve() {
    let mut more = 0;
    for (name, level, _, len) in plans() {
        let model = Model::new(&level, Mode::Conservative).unwrap();
        let Verdict::Solved(own) = model.solve(BUDGET).verdict else { unreachable!() };
        let (outcome, found) = model.shortest_plans(BUDGET, 6, 3);
        assert!(matches!(outcome.verdict, Verdict::Solved(ref p) if *p == own), "{name}");
        assert_eq!(found[0], own, "{name}");
        for (i, p) in found.iter().enumerate() {
            assert_eq!(p.len(), len, "{name}: plan {i}");
            assert!(!found[..i].contains(p), "{name}: plan {i} twice");
            let tokens = encode(&route_plan(&level, &model, p).unwrap_or_else(|e| panic!("{name}: plan {i}: {e}")));
            let c = check(&model, &decode(&tokens).unwrap());
            assert!(c.solved, "{name}: plan {i}: {tokens}: {:?}", c.problem);
        }
        more += (found.len() > 1) as usize;
    }
    assert!(more > 15, "only {more} levels with more than one shortest plan");
}

/// Levels whose shortest plans need an action the route can't ground yet (data/v5b and
/// data/ws5, 2026-10-27), and the step `encode_actions` writes for it.
const UNGROUNDED: [(&str, &str); 4] = [
    // A portal loop: into the floor portal, back down through it from the wall.
    ("size 16 17 21 open 0 5 0 14 16 20 15 13 3 15 14 3 0 0 4 2 16 11 white +y 1 16 9 1 16 9       white -z 12 5 0 15 9 20 1 8 0 1 15 20 entry 13 16 0 -y exit 1 0 4 +y gel 13 8 0 +z bounce       stream on", "loop"),
    // A loop that moves the other portal on the way down.
    ("size 3 26 25 open 0 9 0 2 11 24 0 0 1 2 24 24 1 25 2 2 25 2 goo 0 5 0 2 7 0 white +x 2 12       21 2 16 23 white +y 1 24 5 1 25 5 1 24 19 1 25 19 white -z 0 9 0 2 25 1 entry 0 23 1 +x       exit 1 0 1 +y turret 0 22 1 3 panel 1 24 5 -y -z 3 on", "loop"),
    // Reaching through a portal for the cube on a button.
    ("size 11 8 5 open 4 0 0 5 1 4 6 0 0 6 0 4 2 3 0 5 4 4 0 0 2 7 7 4 9 0 2 10 7 4 8 6 2 10 7 4       white -x 0 7 2 10 7 4 white +y 4 1 0 10 3 4 white -y 4 0 0 5 4 0 2 3 0 10 4 0 3 3 1 10 4 4       entry 0 2 2 +x exit 10 7 2 -y fizzler 1 7 2 -y +z 2 on fizzler 9 7 2 -y +z 1 on dropper 5       3 4 reflection dropper 7 1 4 reflection button 2 5 2 weighted > exit button 5 0 0       weighted > field 0 button 7 0 2 weighted > exit pedestal 2 1 2 3 > field 1 funnel 2 0 2       +y on", "grab"),
    // Two timed pedestals pressed in turn.
    ("size 5 7 8 open 4 4 1 4 4 7 0 0 2 2 6 7 3 0 2 4 3 7 4 5 2 4 6 7 3 4 5 4 6 7 white -x 0 0 2       4 1 5 0 2 3 4 2 5 white +y 3 3 2 4 5 3 white -y 3 0 2 4 6 3 white -z 2 0 2 3 5 7 entry 4 6       2 -x exit 0 2 2 +x laserfield 3 3 2 -y +z 2 on dropper 3 6 7 companion pedestal 0 4 2 15       > exit pedestal 1 5 2 inf > exit pedestal 2 0 2 10 > glasspanel 0 turret 1 1 2 0 plate 2       3 2 +z 1 3 2 +z plate 4 5 2 +z 3 2 2 +z glasspanel 0 6 2 +z -y 6 off", "double"),
];

/// `encode_actions` writes plans the route can't ground (loops, reaching through a portal,
/// double dashes) and every plan it can: decoded, they check as solved at the same length, and
/// read back to the same tokens.
#[test]
fn plans_encode_from_actions() {
    for (i, (text, verb)) in UNGROUNDED.iter().enumerate() {
        let p = glados::tokens::decode(text).unwrap_or_else(|e| panic!("level {i}: {e}"));
        let level = Level::from_puzzle(&p);
        let model = Model::new(&level, Mode::Conservative).unwrap();
        let (outcome, found) = model.shortest_plans(500_000, 8, 1);
        assert!(matches!(outcome.verdict, Verdict::Solved(_)), "level {i}");
        let mut used = false;
        for plan in &found {
            let tokens = encode_actions(&model, plan);
            let steps = decode(&tokens).unwrap_or_else(|e| panic!("level {i}: {e}: {tokens}"));
            let back: Vec<String> = steps.iter().map(step_text).collect();
            assert_eq!(back.join(" ") + " exit", tokens, "level {i}");
            let c = check(&model, &steps);
            assert!(c.solved, "level {i}: {tokens}: {:?}", c.problem);
            assert_eq!(c.plan.len(), plan.len(), "level {i}: {tokens}");
            let ungrounded = plan.iter().any(|a| {
                matches!(a, Action::Loop { .. } | Action::PortalReach { .. } | Action::DoubleDash { .. })
            });
            if ungrounded {
                assert!(route_plan(&level, &model, plan).is_err(), "level {i}: grounds now: drop it here");
                assert!(tokens.split(' ').any(|w| w == *verb), "level {i}: no {verb}: {tokens}");
                used = true;
            }
        }
        assert!(used, "level {i}: no shortest plan needs {verb}");
    }
    for (name, level, _, len) in plans() {
        let model = Model::new(&level, Mode::Conservative).unwrap();
        let Verdict::Solved(own) = model.solve(BUDGET).verdict else { unreachable!() };
        let tokens = encode_actions(&model, &own);
        let c = check(&model, &decode(&tokens).unwrap());
        assert!(c.solved, "{name}: {tokens}: {:?}", c.problem);
        assert_eq!(c.plan.len(), len, "{name}");
    }
}

/// A cell no level here reaches.
const NOWHERE: [i32; 3] = [31, 31, 31];

#[test]
fn damaged_plans_fail() {
    let mut damaged = 0;
    for (name, level, tokens, _) in plans() {
        let steps = decode(&tokens).unwrap();
        if steps.is_empty() {
            continue;
        }
        let model = Model::new(&level, Mode::Conservative).unwrap();
        // Cut short: every step fits, the goal isn't reached.
        let c = check(&model, &steps[..steps.len() - 1]);
        assert!(!c.solved, "{name}: cut short");
        assert_eq!(c.steps_ok, steps.len() - 1, "{name}: cut short: {:?}", c.problem);
        // A step left out: shortest plans have no step to spare.
        for i in 0..steps.len() {
            let mut fewer = steps.clone();
            fewer.remove(i);
            assert!(!check(&model, &fewer).solved, "{name}: without step {}", i + 1);
        }
        // A step sent somewhere it can't go fails there.
        for (i, s) in steps.iter().enumerate() {
            let bad = match *s {
                Step::Fire(k, _, d) => Step::Fire(k, NOWHERE, d),
                Step::Walk(_) => Step::Walk(NOWHERE),
                Step::PickUp(_) => Step::PickUp(NOWHERE),
                Step::PutDown(_, d) => Step::PutDown(NOWHERE, d),
                Step::Press(_) => Step::Press(NOWHERE),
                Step::Knock(_) => Step::Knock(NOWHERE),
                _ => continue,
            };
            let mut wrong = steps.clone();
            wrong[i] = bad;
            let c = check(&model, &wrong);
            assert!(!c.solved, "{name}: step {} sent nowhere", i + 1);
            assert_eq!(c.steps_ok, i, "{name}: step {} sent nowhere", i + 1);
            damaged += 1;
        }
    }
    assert!(damaged > 50, "only {damaged} damaged steps");
}

#[test]
fn decoder_rejects_garbage() {
    for bad in [
        "fire",
        "fire green 1 2 3 +x",
        "fire blue 1 2 3",
        "walk 1 2",
        "walk 1 2 33",
        "walk 01 2 3",
        "exit walk 1 2 3",
        "size 8 8 6",
        "putdown 1 2 3 +q",
        "loop blue 1 2 3 +x",
        "grab 1 2 3",
        "double 1 2 3 4 5 6",
        "sweep 1 2 3",
    ] {
        assert!(decode(bad).is_err(), "{bad:?} decoded");
    }
    assert_eq!(decode("").unwrap(), vec![]);
    assert_eq!(decode("exit").unwrap(), vec![]);
    assert_eq!(
        decode("putdown 1 2 3 +x enter blue exit").unwrap().len(),
        2
    );
    // A loop's optional face: a direction after the first cell.
    assert_eq!(
        decode("loop blue 1 2 3 +x 4 5 6 loop orange 7 8 9 exit").unwrap(),
        vec![
            Step::Loop(glados::solver::Color::Blue, Some(([1, 2, 3], glados::geom::Dir::PosX)), [4, 5, 6]),
            Step::Loop(glados::solver::Color::Orange, None, [7, 8, 9]),
        ]
    );
}

/// `glados-score route --plan`: a plan that solves the level routes like the solver's own; one sent
/// somewhere it can't go routes the steps before that, then a try at it.
#[test]
fn given_plans_route() {
    let mut tried = 0;
    for (name, level, tokens, _) in plans() {
        let steps = decode(&tokens).unwrap();
        let model = Model::new(&level, Mode::Conservative).unwrap();
        let c = check(&model, &steps);
        let own = route(&level, BUDGET).unwrap();
        let given = route_plan(&level, &model, &c.plan).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(given.legs.len(), own.legs.len(), "{name}");
        assert!(given.attempt.is_none(), "{name}");
        for (i, s) in steps.iter().enumerate() {
            let bad = match *s {
                Step::Fire(k, c, d) => Step::Fire(k, [c[0], c[1], NOWHERE[2]], d),
                Step::Walk(_) => Step::Walk(NOWHERE),
                _ => continue,
            };
            let mut wrong = steps.clone();
            wrong[i] = bad;
            let c = check(&model, &wrong);
            assert!(!c.solved && c.steps_ok == i, "{name}: step {}", i + 1);
            let r = route_prefix(&level, &model, &c.plan, Some((&wrong[i], "fits no move")))
                .unwrap_or_else(|e| panic!("{name}: step {}: {e}", i + 1));
            assert_eq!(r.legs.len(), i, "{name}: step {}", i + 1);
            let t = r.attempt.as_ref().expect("an attempt");
            assert_eq!(t.step, step_text(&wrong[i]));
            assert_eq!(t.fire.is_some(), matches!(bad, Step::Fire(..)), "{name}: step {}", i + 1);
            let lines = r.json_lines("t", level.size);
            assert_eq!(lines.len(), i + 2, "{name}: step {}", i + 1);
            assert!(lines.last().unwrap().contains("\"action\":\"attempt\""));
            tried += 1;
        }
    }
    assert!(tried > 40, "only {tried} attempts");
}
