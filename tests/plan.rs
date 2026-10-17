//! Plans as tokens: the solver's own plans check as solved, damaged ones fail where they
//! break.

use glados::fixtures;
use glados::generator::Genome;
use glados::level::Level;
use glados::p2c::Puzzle;
use glados::plan::{check, decode, encode, step_text, vocab, Step};
use glados::rng::Rng;
use glados::route::route;
use glados::solver::{Mode, Model, Verdict};

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
    ] {
        assert!(decode(bad).is_err(), "{bad:?} decoded");
    }
    assert_eq!(decode("").unwrap(), vec![]);
    assert_eq!(decode("exit").unwrap(), vec![]);
    assert_eq!(
        decode("putdown 1 2 3 +x enter blue exit").unwrap().len(),
        2
    );
}
