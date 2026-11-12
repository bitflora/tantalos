//! Solution-carrying generation, independent of the reverse generator.
//!
//! A witness is an upper bound on solution length, never a shortest-plan label or
//! proof that its intended insight is necessary. Repair searches near that witness
//! in edit distance, retaining the suffix as well as the prefix after a mutation.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::builder::Builder;
use crate::geom::Dir;
use crate::p2c::Puzzle;
use crate::plan::{self, Step};
use crate::solver::{Action, CubeLoc, Model, State};

#[derive(Debug, PartialEq, Eq)]
pub enum Audit {
    /// A solution bypassing the intended dependency exists in the model.
    Bypass,
    /// Exhausted the restricted model. Not a claim about unmodelled game moves.
    Required,
    Unknown,
}

/// Counterexample search, separate from witness admission. A budget exhaustion
/// is explicitly unknown, never a proof of necessity.
pub fn audit_without(
    model: &Model,
    budget: usize,
    allowed: impl Fn(&Action, &State) -> bool,
) -> Audit {
    if budget == 0 {
        return Audit::Unknown;
    }
    let start = model.initial();
    let mut seen = HashSet::from([start.clone()]);
    let mut states = vec![start];
    let mut cursor = 0;
    let mut clipped = false;
    while cursor < states.len() {
        if model.is_goal(&states[cursor]) {
            return Audit::Bypass;
        }
        for (a, s) in model.successors(&states[cursor]) {
            if !allowed(&a, &s) || seen.contains(&s) {
                continue;
            }
            if model.is_goal(&s) {
                return Audit::Bypass;
            }
            if states.len() >= budget {
                clipped = true;
                continue;
            }
            seen.insert(s.clone());
            states.push(s);
        }
        cursor += 1;
    }
    if clipped || !model.complete {
        Audit::Unknown
    } else {
        Audit::Required
    }
}

/// Does the one-cube chamber require placing its cube on the access button?
pub fn audit_access(model: &Model, button: usize, budget: usize) -> Audit {
    audit_without(model, budget, |_, state| {
        !state
            .cubes
            .iter()
            .any(|c| c.loc == CubeLoc::OnButton(button))
    })
}

#[derive(Clone, Copy, Debug)]
pub struct RepairLimits {
    /// Insertions and deletions; replacing one step costs two.
    pub edits: usize,
    /// Hard cap on stored (world state, witness cursor) pairs.
    pub states: usize,
}

#[derive(Debug)]
pub struct Repaired {
    pub actions: Vec<Action>,
    pub edits: usize,
    pub states: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub enum RepairFailure {
    Budget,
    /// No witness within the edit radius, NOT an unsolvable level.
    NoRepair,
}

struct Node {
    state: State,
    cursor: usize,
    edits: usize,
    parent: Option<(usize, Option<Action>)>,
}

/// Replay for free; insert/delete a bounded number of steps when replay breaks.
/// All transitions come from the supplied model. No full BFS admission test.
pub fn repair(
    model: &Model,
    witness: &[Step],
    limits: RepairLimits,
) -> Result<Repaired, RepairFailure> {
    if limits.states == 0 {
        return Err(RepairFailure::Budget);
    }
    let start = model.initial();
    let mut seen = HashMap::from([((start.clone(), 0), 0)]);
    let mut nodes = vec![Node {
        state: start,
        cursor: 0,
        edits: 0,
        parent: None,
    }];
    let mut queue = BinaryHeap::from([Reverse((0, Reverse(0), 0))]);
    let mut clipped = false;
    while let Some(Reverse((cost, _, id))) = queue.pop() {
        let n = &nodes[id];
        if seen.get(&(n.state.clone(), n.cursor)) != Some(&cost) {
            continue;
        }
        if model.is_goal(&n.state) {
            let mut actions = Vec::new();
            let mut at = id;
            while let Some((parent, action)) = nodes[at].parent {
                if let Some(a) = action {
                    actions.push(a);
                }
                at = parent;
            }
            actions.reverse();
            return Ok(Repaired {
                actions,
                edits: cost,
                states: nodes.len(),
            });
        }
        let (state, cursor, edits) = (n.state.clone(), n.cursor, n.edits);
        let successors = model.successors(&state);
        let mut add = |state: State, cursor: usize, edits: usize, action: Option<Action>| {
            if edits > limits.edits {
                return;
            }
            let key = (state.clone(), cursor);
            if seen.get(&key).is_some_and(|&old| old <= edits) {
                return;
            }
            if nodes.len() >= limits.states {
                clipped = true;
                return;
            }
            seen.insert(key, edits);
            let next = nodes.len();
            nodes.push(Node {
                state,
                cursor,
                edits,
                parent: Some((id, action)),
            });
            queue.push(Reverse((edits, Reverse(cursor), next)));
        };
        // Keep matching transitions first, so a tight budget favors replay.
        if let Some(step) = witness.get(cursor) {
            for (a, s) in &successors {
                if plan::fits(model, &state, a, s, step) {
                    add(s.clone(), cursor + 1, edits, Some(*a));
                }
            }
            add(state.clone(), cursor + 1, edits + 1, None);
        }
        for (a, s) in successors {
            add(s, cursor, edits + 1, Some(a));
        }
    }
    Err(if clipped {
        RepairFailure::Budget
    } else {
        RepairFailure::NoRepair
    })
}

/// One connected chamber, one cube, two roles. A parapet on the ledge
/// hides a portal surface on its upper side from the lower floor. Stairs grant the sightline;
/// the prepared portal replaces the stairs when the cube is reclaimed.
#[derive(Clone, Copy, Debug)]
pub struct ReuseChamber {
    pub lower_width: i32,
    pub ledge_depth: i32,
    pub depth: i32,
    pub lane: i32,
}

impl Default for ReuseChamber {
    fn default() -> Self {
        Self {
            lower_width: 5,
            ledge_depth: 4,
            depth: 7,
            lane: 2,
        }
    }
}

impl ReuseChamber {
    pub fn build(self) -> Result<(Puzzle, Vec<Step>), String> {
        let Self {
            lower_width: split,
            ledge_depth,
            depth,
            lane,
        } = self;
        if !(5..=8).contains(&split)
            || !(3..=6).contains(&ledge_depth)
            || !(7..=10).contains(&depth)
            || !(2..=depth - 4).contains(&lane)
        {
            return Err("reuse chamber dimensions outside the prototype range".into());
        }
        let width = split + ledge_depth;
        let mut b = Builder::new("Borrowed stairs", [width, depth, 5]);
        b.carve([0, 0, 1], [split - 1, depth - 1, 4]);
        b.carve([split, 0, 2], [width - 1, depth - 1, 2]);
        b.fill([split, lane + 1, 2], [split, depth - 1, 2]);
        b.entry([0, depth - 2, 1], Dir::PosX);
        let exit = b.exit([width - 1, depth - 2, 2], Dir::NegX);
        let cube = [2, depth - 2, 1];
        let access = [1, lane, 1];
        let goal = [width - 2, depth - 2, 2];
        b.loose_cube(cube, 0);
        let first = b.floor_button(access);
        let last = b.floor_button(goal);
        let stairs = b.stairs([split - 2, lane, 1], Dir::PosX, false);
        b.connect(first, stairs).connect(last, exit);
        let low = [0, lane, 1];
        let high = [split + 1, lane + 2, 2];
        b.paint(low, low, Dir::NegX, true);
        b.paint(high, high, Dir::NegX, true);
        use crate::solver::Color::{Blue, Orange};
        let witness = vec![
            Step::Fire(Blue, low, Dir::NegX),
            Step::PickUp(cube),
            Step::PutDown(access, None),
            Step::Climb([split, lane, 2]),
            Step::Fire(Orange, high, Dir::NegX),
            Step::Enter(Orange),
            Step::PickUp(access),
            Step::Enter(Blue),
            Step::PutDown(goal, None),
        ];
        Ok((b.build(), witness))
    }
}
