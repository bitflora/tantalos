//! Tier-0 solver: breadth-first search over macro-states (plan.md, "Solver design").
//!
//! The player's position is a region; walking inside a region is free. Every action is a
//! discrete event: take a drop or cube-step edge, fire a portal at a face class, walk through
//! a portal, pick up / put down / toss a cube, throw a cube through a portal. BFS gives the
//! minimum number of actions.
//!
//! Tier-0 rules (ASSUMPTIONS until checked in-game), on top of `regions` and `portals`:
//! - Floor buttons are held down only by cubes. A player standing on one never helps in
//!   Tier 0: the only thing a button can drive is the exit door, and the player can't stand
//!   on the button and walk out of the door at once.
//! - The exit door is open if nothing is wired to it; otherwise when all inputs are pressed
//!   (AND), inverted if `ITEM_PROPERTY_START_OPEN` is set.
//! - The portal gun can't fire while carrying a cube.
//! - Climbing a one-voxel step needs a loose cube in the lower region; it stays there.
//! - All cube types count as cubes; all floor button types accept any of them.
//! - Cubes are never destroyed, so droppers never matter after the first drop.
//! - A floor portal doesn't cut its region in two, and walking across it is optional.

use std::collections::{BTreeMap, HashMap};

use crate::geom::{step, Dir};
use crate::level::{ItemKind, Level};
use crate::portals::{self, Landing, Portals};
use crate::regions::{Move, RegionGraph};

/// Items that don't affect play.
const DECORATIVE: &[&str] = &[
    "ITEM_COOP_ENTRY_DOOR",
    "ITEM_COOP_EXIT_DOOR",
    "ITEM_LIGHT_PANEL",
    "ITEM_LIGHT_PANEL_WARM",
    "ITEM_OBSERVATION_ROOM",
    "ITEM_SECONDARY_OBSERVATION_ROOM",
    "ITEM_BEE2_ANTLINE",
    "ITEM_BEE2_ANTLINE_CORNER",
];

pub const DEFAULT_MAX_STATES: usize = 1_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Color {
    Blue,
    Orange,
}

impl Color {
    const BOTH: [Color; 2] = [Color::Blue, Color::Orange];

    fn other(self) -> Color {
        match self {
            Color::Blue => Color::Orange,
            Color::Orange => Color::Blue,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Color::Blue => "blue",
            Color::Orange => "orange",
        }
    }
}

/// Where a cube is. Variants order so that sorting puts held cubes first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CubeLoc {
    Held,
    Loose(usize),
    /// Index into `Model::buttons`.
    OnButton(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Drop off a ledge, or climb a step using a cube.
    Move {
        to: usize,
        kind: Move,
    },
    Fire {
        color: Color,
        class: usize,
    },
    /// Walk into the portal of this color.
    Enter {
        color: Color,
    },
    PickUp {
        from: CubeLoc,
    },
    PutDown {
        at: CubeLoc,
    },
    /// Drop the held cube off a ledge of the current region.
    Toss {
        to: CubeLoc,
    },
    /// Throw the held cube into the portal of this color.
    Throw {
        color: Color,
        to: CubeLoc,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct State {
    pub region: usize,
    /// Face class of each portal, `[blue, orange]`.
    pub portals: [Option<usize>; 2],
    /// Sorted, so identical cubes in permuted places are one state.
    pub cubes: Vec<CubeLoc>,
}

impl State {
    fn holding(&self) -> bool {
        self.cubes.first() == Some(&CubeLoc::Held)
    }

    fn with_cube(&self, from: CubeLoc, to: CubeLoc) -> State {
        let mut s = self.clone();
        let i = s.cubes.iter().position(|&c| c == from).unwrap();
        s.cubes[i] = to;
        s.cubes.sort();
        s
    }
}

#[derive(Clone, Debug)]
pub struct Button {
    pub index: i32,
    pub region: Option<usize>,
}

/// Everything the search needs, precomputed from a level.
pub struct Model {
    pub graph: RegionGraph,
    pub portals: Portals,
    pub buttons: Vec<Button>,
    pub start: State,
    pub exit_region: usize,
    /// Buttons wired to the exit door (indices into `buttons`).
    exit_inputs: Vec<usize>,
    exit_inverted: bool,
    /// Per region: where a cube dropped off each of its ledges ends up.
    tosses: Vec<Vec<CubeLoc>>,
    /// Face classes the search may use. Metrics switch one off to test whether it's needed.
    pub class_enabled: Vec<bool>,
}

/// Every state reachable from the start, in BFS order, for metrics.
pub struct StateSpace {
    pub states: Vec<State>,
    /// BFS depth (actions from the start).
    pub depth: Vec<usize>,
    /// Successors of each state, one entry per action (indices into `states`).
    pub next: Vec<Vec<usize>>,
    pub goal: Vec<bool>,
    /// False if the state budget ran out; the graph is then only the nearest part.
    pub complete: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Problem {
    /// The level is broken (no entry, exit door floating, ...).
    Invalid(String),
    /// The level uses something Tier 0 doesn't model.
    Unsupported(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    Solved(Vec<Action>),
    /// Every reachable state was explored.
    Unsolvable,
    /// Hit the state budget.
    Unknown,
}

pub struct Outcome {
    pub verdict: Verdict,
    pub states: usize,
}

impl Model {
    pub fn new(level: &Level) -> Result<Model, Vec<Problem>> {
        let mut problems = Vec::new();
        let graph = RegionGraph::build(level);

        let mut unsupported: BTreeMap<&str, Vec<i32>> = BTreeMap::new();
        for it in &level.items {
            if let ItemKind::Other(t) = &it.kind {
                if !DECORATIVE.contains(&t.as_str()) {
                    unsupported.entry(t).or_default().push(it.index);
                }
            }
        }
        for (t, items) in unsupported {
            let items: Vec<String> = items.iter().map(i32::to_string).collect();
            problems.push(Problem::Unsupported(format!(
                "{t} (item {})",
                items.join(", ")
            )));
        }
        let mut region_of_door = |kind: ItemKind, name: &str| {
            let mut doors = level.items.iter().filter(|i| i.kind == kind);
            let door = doors.next();
            if doors.next().is_some() {
                problems.push(Problem::Invalid(format!("more than one {name} door")));
            }
            match door {
                None => {
                    problems.push(Problem::Invalid(format!("no {name} door")));
                    None
                }
                Some(d) => {
                    let r = graph.region_of(d.cell);
                    if r.is_none() {
                        problems.push(Problem::Invalid(format!(
                            "{name} door at {:?} has no floor to stand on",
                            d.cell
                        )));
                    }
                    r.map(|r| (d, r))
                }
            }
        };
        let entry = region_of_door(ItemKind::EntryDoor, "entry");
        let exit = region_of_door(ItemKind::ExitDoor, "exit");

        let buttons: Vec<Button> = level
            .of_kind(&ItemKind::FloorButton)
            .map(|b| Button {
                index: b.index,
                region: graph.region_of(b.cell),
            })
            .collect();
        let button_slot = |index: i32| buttons.iter().position(|b| b.index == index);

        let mut exit_inputs = Vec::new();
        for link in &level.links {
            let receiver = level.item(link.receiver).map(|i| &i.kind);
            let sender = level.item(link.sender).map(|i| &i.kind);
            let other = |k: Option<&ItemKind>| matches!(k, Some(ItemKind::Other(_)));
            if other(receiver) || other(sender) {
                continue; // Already reported as an unsupported item.
            }
            if receiver != Some(&ItemKind::ExitDoor) {
                problems.push(Problem::Unsupported(format!(
                    "wire {} -> {} drives something other than the exit door",
                    link.sender, link.receiver
                )));
            } else if sender != Some(&ItemKind::FloorButton) {
                problems.push(Problem::Unsupported(format!(
                    "wire {} -> {} comes from something other than a floor button",
                    link.sender, link.receiver
                )));
            } else {
                exit_inputs.push(button_slot(link.sender).unwrap());
            }
        }

        let to_loc = |l: Landing| match l.button.and_then(button_slot) {
            Some(b) => CubeLoc::OnButton(b),
            None => CubeLoc::Loose(l.region),
        };
        let mut cubes = Vec::new();
        for cube in level.of_kind(&ItemKind::Cube) {
            // A cube that would fall into goo is simply gone.
            if let Some(l) = Landing::from_fall(level, &graph, cube.cell) {
                cubes.push(to_loc(l));
            }
        }
        cubes.sort();

        let tosses = graph
            .regions
            .iter()
            .map(|r| {
                let mut locs: Vec<CubeLoc> = r
                    .cells
                    .iter()
                    .flat_map(|&c| Dir::HORIZONTAL.map(|d| step(c, d)))
                    .filter(|&n| !level.solid(n) && !level.standable(n))
                    .filter_map(|n| Landing::from_fall(level, &graph, n))
                    .map(to_loc)
                    .collect();
                locs.sort();
                locs.dedup();
                locs
            })
            .collect();

        let (Some((_, start)), Some((exit_door, exit_region))) = (entry, exit) else {
            return Err(problems);
        };
        if !problems.is_empty() {
            return Err(problems);
        }
        let portals = Portals::build(level, &graph);
        Ok(Model {
            class_enabled: vec![true; portals.classes.len()],
            exit_inverted: !exit_inputs.is_empty() && exit_door.start_open == Some(true),
            graph,
            portals,
            buttons,
            start: State {
                region: start,
                portals: [None, None],
                cubes,
            },
            exit_region,
            exit_inputs,
            tosses,
        })
    }

    pub fn exit_open(&self, s: &State) -> bool {
        if self.exit_inputs.is_empty() {
            return true;
        }
        let all = self
            .exit_inputs
            .iter()
            .all(|&b| s.cubes.contains(&CubeLoc::OnButton(b)));
        all != self.exit_inverted
    }

    pub fn is_goal(&self, s: &State) -> bool {
        s.region == self.exit_region && self.exit_open(s)
    }

    fn landing_loc(&self, l: Landing, s: &State) -> CubeLoc {
        match l
            .button
            .and_then(|i| self.buttons.iter().position(|b| b.index == i))
        {
            Some(b) if !s.cubes.contains(&CubeLoc::OnButton(b)) => CubeLoc::OnButton(b),
            _ => CubeLoc::Loose(l.region),
        }
    }

    /// The portal you'd come out of when entering `color` from the current region.
    fn through(&self, s: &State, color: Color) -> Option<Landing> {
        let [a, b] = match color {
            Color::Blue => s.portals,
            Color::Orange => [s.portals[1], s.portals[0]],
        };
        let (a, b) = (&self.portals.classes[a?], &self.portals.classes[b?]);
        (a.entry == Some(s.region)).then_some(b.landing).flatten()
    }

    pub fn successors(&self, s: &State) -> Vec<(Action, State)> {
        let mut out = Vec::new();
        let r = s.region;
        let held = s.holding();

        for e in self.graph.edges_from(r) {
            let ok = match e.kind {
                Move::Drop { .. } => true,
                Move::Step => s.cubes.contains(&CubeLoc::Loose(r)),
            };
            if ok {
                let mut n = s.clone();
                n.region = e.to;
                out.push((
                    Action::Move {
                        to: e.to,
                        kind: e.kind,
                    },
                    n,
                ));
            }
        }

        for color in Color::BOTH {
            let me = color as usize;
            let other = s.portals[color.other() as usize];
            for class in &self.portals.classes {
                let taken = other == Some(class.id) && class.faces.len() == 1;
                let usable = self.class_enabled[class.id] && class.visible_from[r];
                if !held && usable && s.portals[me] != Some(class.id) && !taken {
                    let mut n = s.clone();
                    n.portals[me] = Some(class.id);
                    out.push((
                        Action::Fire {
                            color,
                            class: class.id,
                        },
                        n,
                    ));
                }
            }
            if let Some(l) = self.through(s, color) {
                let mut n = s.clone();
                n.region = l.region;
                out.push((Action::Enter { color }, n));
                if held {
                    let to = self.landing_loc(l, s);
                    out.push((Action::Throw { color, to }, s.with_cube(CubeLoc::Held, to)));
                }
            }
        }

        if held {
            let mut spots = vec![CubeLoc::Loose(r)];
            spots.extend(
                (0..self.buttons.len())
                    .filter(|&b| self.buttons[b].region == Some(r))
                    .map(CubeLoc::OnButton)
                    .filter(|b| !s.cubes.contains(b)),
            );
            for at in spots {
                out.push((Action::PutDown { at }, s.with_cube(CubeLoc::Held, at)));
            }
            for &to in &self.tosses[r] {
                let to = match to {
                    CubeLoc::OnButton(b) if s.cubes.contains(&to) => {
                        CubeLoc::Loose(self.buttons[b].region.unwrap())
                    }
                    _ => to,
                };
                out.push((Action::Toss { to }, s.with_cube(CubeLoc::Held, to)));
            }
        } else {
            let mut here: Vec<CubeLoc> = s
                .cubes
                .iter()
                .copied()
                .filter(|&c| match c {
                    CubeLoc::Loose(cr) => cr == r,
                    CubeLoc::OnButton(b) => self.buttons[b].region == Some(r),
                    CubeLoc::Held => false,
                })
                .collect();
            here.dedup();
            for from in here {
                out.push((Action::PickUp { from }, s.with_cube(from, CubeLoc::Held)));
            }
        }
        out
    }

    /// Breadth-first search from the start state, exploring at most `max_states` states.
    pub fn solve(&self, max_states: usize) -> Outcome {
        let mut states = vec![self.start.clone()];
        let mut parent: Vec<Option<(usize, Action)>> = vec![None];
        let mut seen = HashMap::from([(self.start.clone(), 0)]);
        let mut next = 0;
        while next < states.len() {
            let s = states[next].clone();
            if self.is_goal(&s) {
                let mut plan = Vec::new();
                let mut i = next;
                while let Some((p, a)) = parent[i] {
                    plan.push(a);
                    i = p;
                }
                plan.reverse();
                return Outcome {
                    verdict: Verdict::Solved(plan),
                    states: states.len(),
                };
            }
            for (a, n) in self.successors(&s) {
                if seen.contains_key(&n) {
                    continue;
                }
                if states.len() >= max_states {
                    return Outcome {
                        verdict: Verdict::Unknown,
                        states: states.len(),
                    };
                }
                seen.insert(n.clone(), states.len());
                states.push(n);
                parent.push(Some((next, a)));
            }
            next += 1;
        }
        Outcome {
            verdict: Verdict::Unsolvable,
            states: states.len(),
        }
    }

    /// Explores every reachable state (up to `max_states`), past goals too: metrics need the
    /// whole graph to find dead ends.
    pub fn explore(&self, max_states: usize) -> StateSpace {
        let mut space = StateSpace {
            states: vec![self.start.clone()],
            depth: vec![0],
            next: Vec::new(),
            goal: Vec::new(),
            complete: true,
        };
        let mut seen = HashMap::from([(self.start.clone(), 0)]);
        let mut i = 0;
        while i < space.states.len() {
            let s = space.states[i].clone();
            space.goal.push(self.is_goal(&s));
            let mut next = Vec::new();
            for (_, n) in self.successors(&s) {
                let j = match seen.get(&n) {
                    Some(&j) => j,
                    None if space.states.len() >= max_states => {
                        space.complete = false;
                        continue;
                    }
                    None => {
                        let j = space.states.len();
                        seen.insert(n.clone(), j);
                        space.states.push(n);
                        space.depth.push(space.depth[i] + 1);
                        j
                    }
                };
                next.push(j);
            }
            space.next.push(next);
            i += 1;
        }
        space
    }

    fn loc_name(&self, l: CubeLoc) -> String {
        match l {
            CubeLoc::Held => "hand".into(),
            CubeLoc::Loose(r) => format!("region {}", RegionGraph::label(r)),
            CubeLoc::OnButton(b) => format!("button {}", self.buttons[b].index),
        }
    }

    pub fn describe(&self, a: &Action) -> String {
        let label = RegionGraph::label;
        match *a {
            Action::Move {
                to,
                kind: Move::Drop { height },
            } => format!("drop {height} to region {}", label(to)),
            Action::Move {
                to,
                kind: Move::Step,
            } => format!("climb onto region {} using a cube", label(to)),
            Action::Fire { color, class } => format!(
                "fire {} at class {class}: {}",
                color.name(),
                portals::describe(&self.portals.classes[class])
            ),
            Action::Enter { color } => format!("walk into the {} portal", color.name()),
            Action::PickUp { from } => format!("pick up the cube in {}", self.loc_name(from)),
            Action::PutDown { at } => format!("put the cube down in {}", self.loc_name(at)),
            Action::Toss { to } => format!("drop the cube off a ledge into {}", self.loc_name(to)),
            Action::Throw { color, to } => format!(
                "throw the cube into the {} portal; it lands in {}",
                color.name(),
                self.loc_name(to)
            ),
        }
    }
}
