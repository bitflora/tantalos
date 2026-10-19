//! What makes a solvable level interesting (plan.md, "Outputs"): computed from the full
//! state space plus ablations, and folded into one score for the generator.

use crate::geom::Cell;
use crate::level::HazardKind;
use crate::regions::Move;
use crate::rng::Rng;
use crate::solver::{Action, CubeLoc, Model, Signal, State, StateSpace, Verdict};

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
    /// Floor buttons and pedestals that the level can't be solved without.
    pub necessary_buttons: usize,
    pub necessary_pedestals: usize,
    /// Fizzlers and laser fields that change the shortest plan's length when switched off.
    pub load_bearing_fizzlers: usize,
    pub load_bearing_lasers: usize,
    /// Turrets that change the shortest plan's length when removed.
    pub load_bearing_turrets: usize,
    /// Gels the level can't be solved without.
    pub necessary_gels: usize,
    /// Bits of `MECHANICS`: what the plan does.
    pub mechanics: u32,
    /// Longest chain of plan steps that each need an earlier one (see `chain_depth`).
    pub chain_depth: f64,
    /// Voxels the player covers between region centres, portal jumps included.
    pub travel: f64,
    /// Voxels the cubes end up from where they started, summed.
    pub cube_travel: f64,
    /// Voxels the player moves away from the exit along the plan, summed.
    pub backtrack: f64,
}

/// What a plan can do, for `Metrics::mechanics`. Later ones were added with the devices; new
/// ones go at the end, as training conditions follow this order.
pub const MECHANICS: [&str; 24] = [
    "portal trip",
    "cube through a portal",
    "cube on a button",
    "fire standing on a button",
    "toss off a ledge",
    "latching pedestal",
    "timed dash",
    "fizzler",
    "laser field",
    "turret knocked over",
    "bounce on repulsion gel",
    "leap on propulsion gel",
    "portal on conversion gel",
    "gel through a portal",
    "laser redirected by a cube",
    "laser powers a receiver",
    "faith plate",
    "fling",
    "light bridge",
    "funnel ride",
    "stairs",
    "moving platform",
    "angled or flip panel",
    "projection through a portal",
];

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
    /// - Components: necessary items of each kind (cubes, floor buttons, pedestals) and fields
    ///   that shape the plan, with diminishing returns per kind, so ten pedestals wired to the
    ///   exit are worth little more than one; and the number of different mechanics used.
    /// - Chain depth: steps that depend on each other, rather than independent chores.
    /// - Distance: the player's path, how far cubes end up from home, and moving away from the
    ///   exit on the way. Log-scaled, so a big empty room doesn't win.
    ///
    /// Surfaces counted 2 each before components and distance were added; the search then bred
    /// small portal-hopping mazes.
    pub fn score(&self) -> f64 {
        let effort = effort(&self.plan);
        let components: f64 = [
            self.cubes - self.unnecessary_cubes,
            self.necessary_buttons,
            self.necessary_pedestals,
            self.load_bearing_fizzlers,
            self.load_bearing_lasers,
            self.load_bearing_turrets,
            self.necessary_gels,
        ]
        .iter()
        .map(|&n| (1.0 + n as f64).ln())
        .sum();
        6.0 * (1.0 + effort).ln()
            + 4.0 * (1.0 - self.random_success)
            + 1.0 * self.necessary_classes as f64
            + 3.0 * components
            + 2.0 * self.mechanics.count_ones() as f64
            + 4.0 * (1.0 + self.chain_depth).ln()
            + 2.0 * (1.0 + self.travel / 4.0).ln()
            + 1.0 * (1.0 + self.cube_travel / 4.0).ln()
            + 2.0 * (1.0 + self.backtrack / 4.0).ln()
            - 3.0 * self.unnecessary_cubes as f64
            - 10.0 * self.dead_end_share()
    }

    /// Necessary cubes, buttons and pedestals plus load-bearing fields.
    pub fn necessary_items(&self) -> usize {
        self.cubes - self.unnecessary_cubes
            + self.necessary_buttons
            + self.necessary_pedestals
            + self.load_bearing_fizzlers
            + self.load_bearing_lasers
            + self.load_bearing_turrets
            + self.necessary_gels
    }

    /// Names of the mechanics the plan uses.
    pub fn mechanic_names(&self) -> Vec<&'static str> {
        (0..MECHANICS.len())
            .filter(|i| self.mechanics & (1 << i) != 0)
            .map(|i| MECHANICS[i])
            .collect()
    }

    pub fn dead_end_share(&self) -> f64 {
        self.dead_end_states as f64 / self.states.max(1) as f64
    }
}

/// Weighted plan length for `Metrics::score`: portal shots, timed dashes, gel bounces and
/// leaps and knocking turrets over 1, trips through a portal 2, latching presses 0, other
/// actions 0.5.
pub fn effort(plan: &[Action]) -> f64 {
    plan.iter()
        .map(|a| match a {
            Action::Fire { .. } | Action::Dash { .. } | Action::Knock { .. } => 1.0,
            Action::Move {
                kind: Move::Bounce | Move::Leap | Move::Plate | Move::Funnel,
                ..
            } => 1.0,
            Action::Enter { .. } | Action::Throw { .. } | Action::Dive { .. } => 2.0,
            Action::Press { .. } => 0.0,
            _ => 0.5,
        })
        .fold(0.0, |a, b| a + b)
}

/// Solves `model` and measures it. `seed` drives the random players.
pub fn assess(model: &mut Model, budget: usize, seed: u64) -> Assessment {
    let space = model.explore(budget);
    measure(model, &space, budget, seed)
}

/// `assess`, given `model.explore(budget)`: the search finds the plan, and the graph is reused.
pub fn measure(model: &mut Model, space: &StateSpace, budget: usize, seed: u64) -> Assessment {
    let plan = match space.verdict() {
        Verdict::Solved(plan) => plan,
        Verdict::Unsolvable => return Assessment::Unsolvable,
        Verdict::Unknown => return Assessment::Unknown,
    };
    if !space.complete {
        return Assessment::Unknown;
    }
    let n = space.len();

    // Shortest plans: count paths along depth-increasing edges.
    let mut paths = vec![0u64; n];
    paths[0] = 1;
    for i in 0..n {
        for &j in space.next(i) {
            let j = j as usize;
            if space.depth[j] == space.depth[i] + 1 {
                paths[j] = paths[j].saturating_add(paths[i]);
            }
        }
    }
    let shortest_plans = (0..n)
        .filter(|&i| space.goal[i] && space.depth[i] == plan.len())
        .fold(0u64, |acc, i| acc.saturating_add(paths[i]));

    // Dead ends: states that can't reach any goal. Predecessors of `j`:
    // `prev[first[j]..first[j + 1]]`.
    let mut first = vec![0usize; n + 1];
    for i in 0..n {
        for &j in space.next(i) {
            first[j as usize + 1] += 1;
        }
    }
    for j in 0..n {
        first[j + 1] += first[j];
    }
    let mut fill = first.clone();
    let mut prev = vec![0u32; first[n]];
    for i in 0..n {
        for &j in space.next(i) {
            prev[fill[j as usize]] = i as u32;
            fill[j as usize] += 1;
        }
    }
    let mut live = space.goal.clone();
    let mut stack: Vec<usize> = (0..n).filter(|&i| live[i]).collect();
    while let Some(j) = stack.pop() {
        for &i in &prev[first[j]..first[j + 1]] {
            if !std::mem::replace(&mut live[i as usize], true) {
                stack.push(i as usize);
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
                if space.next(i).is_empty() {
                    return false;
                }
                i = *rng.pick(space.next(i)) as usize;
            }
            space.goal[i]
        })
        .count();

    let path = replay(model, &plan).expect("the solver's own plan replays");

    // Ablations. Any non-solved verdict (including a blown budget) counts as "needed". A face
    // class, button or pedestal the plan never uses can't be needed (the plan still works
    // without it), so only the ones it uses are tested. Switching a class or a pedestal off
    // only takes away the actions that use it, so whether the level is still solvable can be
    // read off the explored graph; the rest change states and are solved again.
    let solvable = |m: &Model| matches!(m.solve(budget).verdict, Verdict::Solved(_));
    let cubes = model.start.cubes.clone();
    let mut unnecessary_cubes = 0;
    for i in 0..cubes.len() {
        let mut fewer = cubes.clone();
        fewer.remove(i);
        model.set_start_cubes(fewer);
        unnecessary_cubes += solvable(model) as usize;
    }
    model.set_start_cubes(cubes);
    let classes = model.portals.classes.len();
    let mut necessary_classes = 0;
    let fired = |c: usize| {
        plan.iter()
            .any(|a| matches!(*a, Action::Fire { class, .. } if class == c))
    };
    for c in (0..classes).filter(|&c| fired(c)) {
        necessary_classes += !space.solvable_without_class(c) as usize;
    }
    let pressed = |b: usize| {
        path.iter()
            .any(|s| s.cubes.iter().any(|c| c.loc == CubeLoc::OnButton(b)))
            || plan.iter().any(|a| {
                matches!(*a, Action::Fire { stand_on: Some(on), .. } if on == b)
            })
    };
    let mut button_needed = vec![false; model.buttons.len()];
    for b in (0..model.buttons.len()).filter(|&b| pressed(b)) {
        button_needed[b] = model.exit_needs(Signal::Floor(b)) || {
            model.button_enabled[b] = false;
            let solved = solvable(model);
            model.button_enabled[b] = true;
            !solved
        };
    }
    let used = |p: usize| {
        plan.iter().any(|a| {
            matches!(*a, Action::Press { pedestal } | Action::Dash { pedestal, .. } if pedestal == p)
        })
    };
    let mut pedestal_needed = vec![false; model.pedestals.len()];
    for p in (0..model.pedestals.len()).filter(|&p| used(p)) {
        pedestal_needed[p] = !space.solvable_without_pedestal(p);
    }
    // A field shapes the level if the shortest plan gets shorter or easier without it (walking
    // through a field that's off is still a move, so the length alone can stay the same), or
    // if a necessary button or pedestal drives it: then the field is why that input matters,
    // e.g. standing on a button to shoot through a fizzler. Only plans as short as this one
    // can be the same, so the searches stop there.
    let same_plan = |m: &Model| {
        matches!(m.solve_within(budget, plan.len()).verdict,
            Verdict::Solved(p) if p.len() == plan.len() && effort(&p) == effort(&plan))
    };
    let (mut load_bearing_fizzlers, mut load_bearing_lasers) = (0, 0);
    for h in 0..model.hazards.len() {
        let driven = model.hazard_inputs(h).iter().any(|&i| match i {
            Signal::Floor(b) => button_needed[b],
            Signal::Pedestal(p) => pedestal_needed[p],
            Signal::Laser(_) => false,
        });
        let shapes = driven || {
            model.hazard_enabled[h] = false;
            let same = same_plan(model);
            model.hazard_enabled[h] = true;
            !same
        };
        if shapes {
            match model.hazards[h].kind {
                HazardKind::Fizzler => load_bearing_fizzlers += 1,
                HazardKind::LaserField => load_bearing_lasers += 1,
            }
        }
    }

    // Turrets and gels, like fields and buttons.
    let mut load_bearing_turrets = 0;
    for t in 0..model.turrets.len() {
        model.turret_enabled[t] = false;
        load_bearing_turrets += !same_plan(model) as usize;
        model.turret_enabled[t] = true;
    }
    let mut necessary_gels = 0;
    for g in 0..model.gels.len() {
        model.gel_enabled[g] = false;
        necessary_gels += !solvable(model) as usize;
        model.gel_enabled[g] = true;
    }

    let mut mechanics = 0u32;
    let kinds = model.device_kinds();
    for (i, a) in plan.iter().enumerate() {
        let (before, after) = (&path[i], &path[i + 1]);
        if after.down & !before.down != 0 {
            mechanics |= 1 << 9;
        }
        if model.redirected(after) {
            mechanics |= 1 << 13;
        }
        if after.lit & !before.lit != 0 {
            mechanics |= 1 << 15;
        }
        // Standing somewhere only a device makes: on a bridge, a platform or a piston.
        let needs = model.graph.regions[after.region.min(model.graph.regions.len() - 1)].cond.on;
        if !after.escaped && needs & kinds.bridges != 0 {
            mechanics |= 1 << 18;
        }
        let platform = model.on_platform.get(after.region).copied().unwrap_or(false);
        if !after.escaped && (needs & kinds.platforms != 0 || platform) {
            mechanics |= 1 << 21;
        }
        if after.dev & kinds.routed != 0 {
            mechanics |= 1 << 23;
        }
        let mut set = |bit: usize| mechanics |= 1 << bit;
        match *a {
            Action::Enter { .. } => set(0),
            Action::Dive { .. } => {
                set(0);
                set(17);
            }
            Action::PutDown {
                at: CubeLoc::Spot(_),
            } => set(14),
            Action::Move {
                kind: Move::Plate, ..
            } => set(16),
            Action::Move {
                kind: Move::Funnel, ..
            } => set(19),
            Action::Move {
                kind: Move::Stairs, ..
            } => set(20),
            Action::Move {
                kind: Move::Ride, ..
            } => set(21),
            Action::Fire { class, .. }
                if model.portals.classes[class].deployed.is_some()
                    || !model.portals.classes[class].needs.is_always() =>
            {
                set(22)
            }
            Action::Throw { to, .. } => {
                set(1);
                if matches!(to, CubeLoc::OnButton(_)) {
                    set(2);
                }
            }
            Action::PutDown {
                at: CubeLoc::OnButton(_),
            } => set(2),
            Action::Fire {
                stand_on: Some(_), ..
            } => set(3),
            Action::Toss { to } => {
                set(4);
                if matches!(to, CubeLoc::OnButton(_)) {
                    set(2);
                }
            }
            Action::Press { .. } => set(5),
            Action::Dash { .. } => set(6),
            Action::Move {
                kind: Move::Bounce, ..
            } => set(10),
            Action::Move {
                kind: Move::Leap, ..
            } => set(11),
            Action::Fire { class, .. } if model.portals.classes[class].needs_paint.is_some() => {
                set(12)
            }
            _ => {}
        }
    }
    if load_bearing_fizzlers > 0 {
        mechanics |= 1 << 7;
    }
    if load_bearing_lasers > 0 {
        mechanics |= 1 << 8;
    }
    let (travel, cube_travel, backtrack) = distances(model, &path);
    let chain_depth = chain_depth(model, &plan, &path);

    Assessment::Solved(Metrics {
        min_len: plan.len(),
        portal_trips: plan
            .iter()
            .filter(|a| {
                matches!(
                    a,
                    Action::Enter { .. } | Action::Throw { .. } | Action::Dive { .. }
                )
            })
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
        necessary_buttons: button_needed.iter().filter(|&&n| n).count(),
        necessary_pedestals: pedestal_needed.iter().filter(|&&n| n).count(),
        load_bearing_fizzlers,
        load_bearing_lasers,
        load_bearing_turrets,
        necessary_gels,
        mechanics,
        chain_depth,
        travel,
        cube_travel,
        backtrack,
    })
}

/// The states a plan passes through, from the start: `plan.len() + 1` of them, ending in a
/// goal. `None` if it doesn't replay. One action (a timed dash) can lead to more than one
/// state, so this backtracks over those.
pub fn replay(model: &Model, plan: &[Action]) -> Option<Vec<State>> {
    replay_to(model, plan, true)
}

/// `replay`, for the start of a plan: the states may end anywhere.
pub fn replay_part(model: &Model, plan: &[Action]) -> Option<Vec<State>> {
    replay_to(model, plan, false)
}

fn replay_to(model: &Model, plan: &[Action], goal: bool) -> Option<Vec<State>> {
    fn go(model: &Model, plan: &[Action], goal: bool, path: &mut Vec<State>) -> bool {
        let s = path.last().unwrap().clone();
        let Some(a) = plan.get(path.len() - 1) else {
            return !goal || model.is_goal(&s);
        };
        for (b, n) in model.successors(&s) {
            if b == *a {
                path.push(n);
                if go(model, plan, goal, path) {
                    return true;
                }
                path.pop();
            }
        }
        false
    }
    let mut path = vec![model.initial()];
    go(model, plan, goal, &mut path).then_some(path)
}

/// Longest weighted chain of plan steps where each needs an earlier one. Step `j` needs step
/// `i` if, replaying the plan from just before `i` without it, `j` can't be done. A step
/// counts 1, or 0.5 for walking and cube chores (walk, drop, step, pick up, put down; gel
/// bounces and leaps count 1). Independent steps
/// (two portals fired in either order, chores in any order) don't lengthen the chain.
pub fn chain_depth(model: &Model, plan: &[Action], path: &[State]) -> f64 {
    let n = plan.len();
    let mut needs = vec![Vec::new(); n];
    for i in 0..n {
        // Every state the plan could be in, without step i.
        let mut here = vec![path[i].clone()];
        for j in i + 1..n {
            let mut next: Vec<State> = Vec::new();
            for s in &here {
                for (a, t) in model.successors(s) {
                    if a == plan[j] && !next.contains(&t) {
                        next.push(t);
                    }
                }
            }
            if next.is_empty() {
                needs[j].push(i);
            } else {
                here = next;
            }
        }
    }
    let weight = |a: &Action| match a {
        Action::Move {
            kind: Move::Bounce | Move::Leap | Move::Plate | Move::Funnel,
            ..
        } => 1.0,
        Action::Move { .. } | Action::PickUp { .. } | Action::PutDown { .. } => 0.5,
        _ => 1.0,
    };
    let mut depth = vec![0.0f64; n];
    for j in 0..n {
        let before = needs[j].iter().map(|&i| depth[i]).fold(0.0, f64::max);
        depth[j] = before + weight(&plan[j]);
    }
    depth.into_iter().fold(0.0, f64::max)
}

/// (player travel, cube travel, backtracking) in voxels, along the states of a plan.
fn distances(model: &Model, path: &[State]) -> (f64, f64, f64) {
    let centre: Vec<[f64; 3]> = model
        .graph
        .regions
        .iter()
        .map(|r| {
            let mut c = [0.0; 3];
            for cell in &r.cells {
                for a in 0..3 {
                    c[a] += cell[a] as f64 / r.cells.len() as f64;
                }
            }
            c
        })
        .collect();
    let at = |c: Cell| [c[0] as f64, c[1] as f64, c[2] as f64];
    let dist = |a: [f64; 3], b: [f64; 3]| (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>().sqrt();
    let exit = at(model.exit_cell);

    let (mut travel, mut backtrack) = (0.0, 0.0);
    let pos = |s: &State| centre.get(s.region).copied().unwrap_or(exit);
    for w in path.windows(2) {
        let (a, b) = (pos(&w[0]), pos(&w[1]));
        travel += dist(a, b);
        backtrack += (dist(b, exit) - dist(a, exit)).max(0.0);
    }
    let last = path.last().unwrap();
    let here = centre.get(last.region).copied().unwrap_or(exit);
    let place = |l: CubeLoc| match l {
        CubeLoc::Held | CubeLoc::Gone => here,
        CubeLoc::Loose(r) => centre[r],
        CubeLoc::OnButton(b) => at(model.buttons[b].cell),
        CubeLoc::Spot(k) => at(model.spot_cell(k)),
    };
    let cube_travel = last
        .cubes
        .iter()
        .map(|c| dist(place(c.home), place(c.loc)))
        .fold(0.0, |a, b| a + b);
    (travel, cube_travel, backtrack)
}
