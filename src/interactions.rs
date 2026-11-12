//! Compile resource-dependency graphs into a shared chamber and a witnessed plan.
//! Geometry is a single two-bank arena, not a sequence of independent rooms.
use std::collections::{HashMap, VecDeque};

use crate::{
    builder::Builder,
    geom::{Cell, Dir},
    level::{CubeKind, LaserKind, Level},
    p2c::Puzzle,
    solver::{Action, CubeLoc, Model, State},
    witness::{self, Audit},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Topology {
    Exchange,
    Handover,
    Tandem,
}

impl Topology {
    pub const ALL: [Self; 3] = [Self::Exchange, Self::Handover, Self::Tandem];
    pub fn name(self) -> &'static str {
        match self {
            Self::Exchange => "Exchange",
            Self::Handover => "Handover",
            Self::Tandem => "Tandem",
        }
    }
}

/// Intent names world conditions, not steps or room numbers. Compilation resolves
/// these against the actual model after token roundtrip (which renumbers items).
#[derive(Clone, Debug)]
pub enum Condition {
    OnButton(CubeKind, Cell),
    Held(CubeKind),
    Collected(CubeKind, Cell),
    Laser(Cell),
    Goal,
}

impl Condition {
    pub fn holds(&self, level: &Level, model: &Model, state: &State) -> bool {
        match *self {
            Self::OnButton(kind, cell) => model
                .buttons
                .iter()
                .position(|b| b.cell == cell)
                .is_some_and(|i| {
                    state
                        .cubes
                        .iter()
                        .any(|c| c.kind == kind && c.loc == CubeLoc::OnButton(i))
                }),
            Self::Held(kind) => state
                .cubes
                .iter()
                .any(|c| c.kind == kind && c.loc == CubeLoc::Held),
            Self::Collected(kind, cell) => {
                state
                    .cubes
                    .iter()
                    .any(|c| c.kind == kind && c.loc == CubeLoc::Held)
                    && state.cubes.iter().all(|c| c.loc != CubeLoc::Gone)
                    && model
                        .graph
                        .region_of(cell)
                        .is_some_and(|r| !state.cubes.iter().any(|c| c.loc == CubeLoc::Loose(r)))
            }
            Self::Laser(cell) => model
                .beams
                .receivers
                .iter()
                .enumerate()
                .any(|(i, (id, _))| {
                    level
                        .items
                        .iter()
                        .any(|item| item.index == *id && item.cell == cell)
                        && state.lit & (1 << i) != 0
                }),
            Self::Goal => model.is_goal(state),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Milestone {
    pub name: &'static str,
    pub condition: Condition,
    pub depends_on: Vec<usize>,
}

pub struct Design {
    pub topology: Topology,
    pub puzzle: Puzzle,
    pub graph: Vec<Milestone>,
}

/// Three dependency rules share physical space but differ in substitution,
/// alternative support, or simultaneous cooperation. This is a small authored
/// design grammar, not unrestricted graph synthesis.
pub fn compile(topology: Topology) -> Design {
    use Dir::*;
    let mut b = Builder::new(topology.name(), [11, 8, 5]);
    b.carve([0, 0, 1], [10, 7, 4]);
    b.carve([4, 0, 0], [6, 7, 0]);
    b.goo([5, 4, 0]);
    // Shield the final receiver from direct or diagonal reflector aims.
    b.fill([1, 4, 1], [3, 4, 3]);
    b.fill([4, 4, 1], [4, 5, 3]);
    b.fill([4, 2, 1], [4, 2, 3]);
    b.entry([0, 7, 1], PosX);
    let exit = b.exit([10, 6, 1], NegX);
    b.cube_dropper_of([2, 2, 4], 2);
    let substitute = match topology {
        Topology::Exchange => CubeKind::Standard,
        Topology::Handover => CubeKind::Sphere,
        Topology::Tandem => CubeKind::Reflection,
    };
    b.cube_dropper_of(
        [9, 5, 4],
        match substitute {
            CubeKind::Standard => 0,
            CubeKind::Sphere => 3,
            _ => 2,
        },
    );
    b.laser(LaserKind::Emitter, [0, 2, 1], PosX, Some(NegZ), true);
    let tandem = topology == Topology::Tandem;
    let final_receiver = if tandem { [2, 7, 1] } else { [3, 5, 1] };
    let receiver = b.laser(
        LaserKind::Catcher,
        final_receiver,
        if tandem { NegY } else { NegX },
        Some(NegZ),
        true,
    );
    b.connect(receiver, exit);
    // Both portals stay on the near bank. They route a beam around the shield;
    // they cannot replace the crossing to the far bank.
    let input = if tandem { [3, 0, 1] } else { [1, 0, 1] };
    let output = if tandem { [0, 6, 1] } else { [0, 5, 1] };
    b.paint(input, input, NegY, true);
    b.paint(output, output, NegX, true);
    let access = b.light_bridge([0, 1, 1], PosX, PosZ, false);
    if tandem {
        let relay_cell = [3, 1, 1];
        let relay = b.laser(LaserKind::Relay, relay_cell, PosZ, None, true);
        b.connect(relay, access).connect(relay, exit);
        return Design {
            topology,
            puzzle: b.build(),
            graph: vec![
                Milestone {
                    name: "first reflector sustains access",
                    condition: Condition::Laser(relay_cell),
                    depends_on: vec![],
                },
                Milestone {
                    name: "retrieve second reflector",
                    condition: Condition::Collected(CubeKind::Reflection, [9, 5, 1]),
                    depends_on: vec![0],
                },
                Milestone {
                    name: "complete beam with both reflectors",
                    condition: Condition::Laser(final_receiver),
                    depends_on: vec![1],
                },
                Milestone {
                    name: "cross while both receivers stay powered",
                    condition: Condition::Goal,
                    depends_on: vec![0, 2],
                },
            ],
        };
    }
    let button_cell = [2, 7, 1];
    let button = b.floor_button_of(
        button_cell,
        if topology == Topology::Handover { 2 } else { 0 },
    );
    let first = match topology {
        Topology::Exchange => {
            b.connect(button, access);
            Condition::OnButton(CubeKind::Reflection, button_cell)
        }
        Topology::Handover => {
            let initial_receiver = [3, 0, 1];
            let receiver = b.laser(LaserKind::Catcher, initial_receiver, PosY, Some(NegZ), true);
            b.connect(receiver, access);
            // The alternate bridge occupies a different lane of the SAME gulf.
            // Leave its near-side anchor beyond the beam shield.
            let alternative = b.light_bridge([0, 6, 1], PosX, PosZ, false);
            b.connect(button, alternative);
            Condition::Laser(initial_receiver)
        }
        Topology::Tandem => unreachable!(),
    };
    let graph = vec![
        Milestone {
            name: "commit reflector to access",
            condition: first,
            depends_on: vec![],
        },
        Milestone {
            name: "recover substitute cube",
            condition: Condition::Held(substitute),
            depends_on: vec![0],
        },
        Milestone {
            name: "secure replacement support",
            condition: Condition::OnButton(substitute, button_cell),
            depends_on: vec![1],
        },
        Milestone {
            name: "reclaim reflector",
            condition: Condition::Held(CubeKind::Reflection),
            depends_on: vec![2],
        },
        Milestone {
            name: "reroute beam around shield",
            condition: Condition::Laser(final_receiver),
            depends_on: vec![3],
        },
        Milestone {
            name: "cross with exit powered",
            condition: Condition::Goal,
            depends_on: vec![2, 4],
        },
    ];
    Design {
        topology,
        puzzle: b.build(),
        graph,
    }
}

/// Local milestone planner. It can move portals/cubes while attaining a graph
/// condition; it does not ask global BFS to discover the entire intended plan.
pub fn synthesize(
    design: &Design,
    level: &Level,
    model: &Model,
    budget: usize,
) -> Result<Vec<Action>, String> {
    let mut start = model.initial();
    let mut plan = Vec::new();
    for (index, milestone) in design.graph.iter().enumerate() {
        if milestone.depends_on.iter().any(|&d| d >= index) {
            return Err("dependency graph is not topologically ordered".into());
        }
        let (actions, end) = reach(level, model, start, &milestone.condition, budget)
            .map_err(|e| format!("{}: {e}", milestone.name))?;
        plan.extend(actions);
        start = end;
    }
    if !model.is_goal(&start) {
        return Err("dependency graph ends before the exit".into());
    }
    Ok(plan)
}

fn reach(
    level: &Level,
    model: &Model,
    start: State,
    target: &Condition,
    budget: usize,
) -> Result<(Vec<Action>, State), String> {
    if budget == 0 {
        return Err("milestone budget exhausted".into());
    }
    let mut seen = HashMap::from([(start.clone(), 0)]);
    let mut states = vec![start];
    let mut parents: Vec<Option<(usize, Action)>> = vec![None];
    let mut queue = VecDeque::from([0]);
    while let Some(i) = queue.pop_front() {
        if target.holds(level, model, &states[i]) {
            let mut actions = Vec::new();
            let mut at = i;
            while let Some((p, a)) = parents[at] {
                actions.push(a);
                at = p;
            }
            actions.reverse();
            return Ok((actions, states[i].clone()));
        }
        for (a, s) in model.successors(&states[i]) {
            if seen.contains_key(&s) {
                continue;
            }
            if states.len() >= budget {
                return Err("milestone budget exhausted".into());
            }
            let next = states.len();
            seen.insert(s.clone(), next);
            states.push(s);
            parents.push(Some((i, a)));
            queue.push_back(next);
        }
    }
    Err("milestone unreachable from prior milestone".into())
}

pub fn audit(
    design: &Design,
    level: &Level,
    model: &Model,
    budget: usize,
) -> Vec<(&'static str, Audit)> {
    let mut results = Vec::new();
    let checks: &[usize] = if design.topology == Topology::Tandem {
        &[0, 1, 2]
    } else {
        &[0, 2, 4]
    };
    for &i in checks {
        let m = &design.graph[i];
        results.push((
            m.name,
            witness::audit_without(model, budget, |_, s| !m.condition.holds(level, model, s)),
        ));
    }
    results.push((
        "portal routing",
        witness::audit_without(model, budget, |a, _| !matches!(a, Action::Fire { .. })),
    ));
    results
}
