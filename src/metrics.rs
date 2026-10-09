//! What makes a solvable level interesting (plan.md, "Outputs"): computed from the full
//! state space plus ablations, and folded into one score for the generator.

use crate::rng::Rng;
use crate::solver::{Action, Model, Verdict};

#[derive(Clone, Debug, PartialEq)]
pub struct Metrics {
    /// One shortest plan.
    pub plan: Vec<Action>,
    pub min_len: usize,
    /// Trips through a portal in the plan: walking in, or throwing a cube in.
    pub portal_trips: usize,
    /// Distinct shortest plans (saturating). Inflated by independent actions in either order.
    pub shortest_plans: u64,
    pub states: usize,
    /// Reachable states from which the exit can't be reached any more: softlocks.
    pub dead_end_states: usize,
    pub cubes: usize,
    /// Cubes whose removal leaves the level solvable.
    pub unnecessary_cubes: usize,
    pub classes: usize,
    /// Face classes whose removal (painting black) makes the level unsolvable.
    pub necessary_classes: usize,
    /// Share of random players (uniform random action, `RANDOM_STEPS` actions) that get out.
    pub random_success: f64,
}

pub const RANDOM_TRIALS: usize = 200;
pub const RANDOM_STEPS: usize = 60;

#[derive(Clone, Debug, PartialEq)]
pub enum Assessment {
    Solved(Metrics),
    Unsolvable,
    /// The state budget ran out (in the search or one of the ablations).
    Unknown,
}

impl Metrics {
    /// Higher is more interesting. The weights are a first guess, to be tuned on playtests.
    ///
    /// - Effort, with diminishing returns: portal shots and timed dashes count 1, trips through
    ///   a portal 2, latching pedestal presses 0, other actions (walking, cube chores) 0.5.
    ///   A linear length reward bred 30-action staircase slogs; counting presses bred rooms
    ///   full of pedestals wired to the exit.
    /// - Random players failing: the level needs a plan, not just wandering.
    /// - Necessary face classes: specific white surfaces matter.
    /// - Penalties: unnecessary cubes (red herrings or cheese), and the share of states that
    ///   are softlocks.
    pub fn score(&self) -> f64 {
        let effort: f64 = self
            .plan
            .iter()
            .map(|a| match a {
                Action::Fire { .. } | Action::Dash { .. } => 1.0,
                Action::Enter { .. } | Action::Throw { .. } => 2.0,
                Action::Press { .. } => 0.0,
                _ => 0.5,
            })
            .sum();
        8.0 * (1.0 + effort).ln()
            + 4.0 * (1.0 - self.random_success)
            + 2.0 * self.necessary_classes as f64
            - 3.0 * self.unnecessary_cubes as f64
            - 10.0 * self.dead_end_share()
    }

    pub fn dead_end_share(&self) -> f64 {
        self.dead_end_states as f64 / self.states.max(1) as f64
    }
}

/// Solves `model` and measures it. `seed` drives the random players.
pub fn assess(model: &mut Model, budget: usize, seed: u64) -> Assessment {
    let plan = match model.solve(budget).verdict {
        Verdict::Solved(plan) => plan,
        Verdict::Unsolvable => return Assessment::Unsolvable,
        Verdict::Unknown => return Assessment::Unknown,
    };
    let space = model.explore(budget);
    if !space.complete {
        return Assessment::Unknown;
    }
    let n = space.states.len();

    // Shortest plans: count paths along depth-increasing edges.
    let mut paths = vec![0u64; n];
    paths[0] = 1;
    for i in 0..n {
        for &j in &space.next[i] {
            if space.depth[j] == space.depth[i] + 1 {
                paths[j] = paths[j].saturating_add(paths[i]);
            }
        }
    }
    let shortest_plans = (0..n)
        .filter(|&i| space.goal[i] && space.depth[i] == plan.len())
        .fold(0u64, |acc, i| acc.saturating_add(paths[i]));

    // Dead ends: states that can't reach any goal.
    let mut prev = vec![Vec::new(); n];
    for (i, next) in space.next.iter().enumerate() {
        for &j in next {
            prev[j].push(i);
        }
    }
    let mut live = space.goal.clone();
    let mut stack: Vec<usize> = (0..n).filter(|&i| live[i]).collect();
    while let Some(j) = stack.pop() {
        for &i in &prev[j] {
            if !std::mem::replace(&mut live[i], true) {
                stack.push(i);
            }
        }
    }
    let dead_end_states = live.iter().filter(|&&l| !l).count();

    // Random players walk the explored graph.
    let mut rng = Rng::new(seed);
    let wins = (0..RANDOM_TRIALS)
        .filter(|_| {
            let mut i = 0;
            for _ in 0..RANDOM_STEPS {
                if space.goal[i] {
                    return true;
                }
                if space.next[i].is_empty() {
                    return false;
                }
                i = *rng.pick(&space.next[i]);
            }
            space.goal[i]
        })
        .count();

    // Ablations. Any non-solved verdict (including a blown budget) counts as "needed".
    let solvable = |m: &Model| matches!(m.solve(budget).verdict, Verdict::Solved(_));
    let cubes = model.start.cubes.clone();
    let mut unnecessary_cubes = 0;
    for i in 0..cubes.len() {
        let mut fewer = cubes.clone();
        fewer.remove(i);
        model.start.cubes = fewer;
        unnecessary_cubes += solvable(model) as usize;
    }
    model.start.cubes = cubes;
    let classes = model.portals.classes.len();
    let mut necessary_classes = 0;
    for c in 0..classes {
        model.class_enabled[c] = false;
        necessary_classes += !solvable(model) as usize;
        model.class_enabled[c] = true;
    }

    Assessment::Solved(Metrics {
        min_len: plan.len(),
        portal_trips: plan
            .iter()
            .filter(|a| matches!(a, Action::Enter { .. } | Action::Throw { .. }))
            .count(),
        plan,
        shortest_plans,
        states: n,
        dead_end_states,
        cubes: model.start.cubes.len(),
        unnecessary_cubes,
        classes,
        necessary_classes,
        random_success: wins as f64 / RANDOM_TRIALS as f64,
    })
}
