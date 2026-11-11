//! Workshop level 110120856, shrunk: six lasers in a ring, each catcher switching the next
//! laser, so the ring keeps switching itself and never settles. `settle` stopped after a few
//! rounds in whatever phase it had reached, so settling a settled state changed it again: the
//! search (which skips settling after quiet moves) found a plan that `metrics::replay`
//! (through `Model::successors`, which settles) couldn't follow, and scoring panicked.

use tantalos::dataset::{label_with, Label};
use tantalos::level::Level;
use tantalos::metrics::replay;
use tantalos::solver::{Mode, Model, Verdict};

const LASER_RING: &str = include_str!("levels/ws-110120856-laser-ring.tokens");
const BUDGET: usize = 50_000;

#[test]
fn laser_ring_plan_replays_or_none() {
    let puzzle = tantalos::tokens::decode(LASER_RING.trim()).unwrap();
    let level = Level::from_puzzle(&puzzle);
    let model = Model::new(&level, Mode::Conservative).unwrap();
    // In the phase the ring settles in, the laser field stays on: unsolvable, but certain.
    match model.explore_to_verdict(BUDGET).verdict() {
        Verdict::Solved(plan) => {
            assert!(replay(&model, &plan).is_some(), "the solver's plan doesn't replay: {plan:?}")
        }
        Verdict::Unsolvable => {}
        Verdict::Unknown => panic!("budget ran out"),
    }
}

#[test]
fn laser_ring_scores() {
    let puzzle = tantalos::tokens::decode(LASER_RING.trim()).unwrap();
    let label = label_with(&puzzle, BUDGET, 0, false);
    assert!(matches!(label, Label::Solvable { .. } | Label::Unsolvable), "{}", label.verdict());
}
