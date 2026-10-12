//! Solver: breadth-first search over macro-states (plan.md, "Solver design").
//!
//! The player's position is a region; walking inside a region is free. Every action is a
//! discrete event: take a drop / step / hazard edge, fire a portal at a face class, walk through
//! a portal, pick up / put down / toss a cube, throw a cube through a portal, press a pedestal
//! button (latching, or timed: press and dash on foot before it runs out). BFS gives the
//! minimum number of actions.
//!
//! Two rule sets bracket the truth (plan.md, "Soundness"): `Mode::Conservative` only uses moves
//! we trust, `Mode::Liberal` any plausible one. `judge` runs both: conservative solves it =
//! confidently solvable; liberal can't = confidently unsolvable; otherwise uncertain.
//!
//! Rules (ASSUMPTIONS until checked in-game), on top of `regions` and `portals`:
//! - Floor buttons are held down by cubes the button accepts (weighted: any; cube: not the
//!   sphere; sphere: only the sphere). The player can stand on a weighted one too, but all
//!   they do there is fire a portal from its cell while it is down (`Action::Fire` with
//!   `stand_on`): walking off releases it.
//! - Each receiver (exit door, fizzler, laser field) is on when all its inputs are (AND).
//!   The exit door opens on that, inverted by `Start Open`; with no inputs it is open. A hazard
//!   with inputs is on when `Start Enabled` XOR all inputs on; with none, `Start Enabled`.
//! - Pedestal buttons: infinite timer latches its outputs on; a timed one keeps them on for
//!   its timer, during which the player may only walk and drop (no portals, no cubes).
//!   Walking speed: conservative 140 u/s, liberal 190 u/s; falls take sqrt(2h/600) s.
//! - Fizzlers: walking or falling through an active one clears both portals and destroys a
//!   held cube, which respawns where it started. A cube falling through one respawns too.
//!   Active fizzlers stop portal shots. Laser fields: the player can't pass an active one;
//!   cubes can.
//! - Conservative: no cube steps (unmeasured), no firing while carrying, strict line of
//!   sight. Liberal: the opposite.
//! - A floor portal doesn't cut its region in two, and walking across it is optional.
//! - Turrets (`turrets`): conservative rules never enter a region a live turret can shoot;
//!   liberal rules run through, but do nothing there except move or walk into a portal.
//!   Knocking one over is an action from a region beside it; a cube landing on it or a portal
//!   opening under it (with the other portal open) knocks it over too.
//! - Gel (`gel`): paint is part of the state. A pouring dropper paints where its stream lands
//!   after every action, through a portal if one is under the stream. Splats without a
//!   dropper are painted from the start. Water washes paint off. Conversion gel makes faces
//!   portalable; repulsion and propulsion floors allow bounces and leaps.
//!
//! The search works on states packed into a few words (`Layout`), each stored once in a hash
//! table (`Table`); `State` is the readable form `successors` and `initial` hand out.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};

use crate::gel::{self, GelRules};
use crate::geom::{step, Cell, Dir};
use crate::level::{ButtonKind, CubeKind, HazardKind, ItemKind, Level, PaintKind, MAX_HAZARDS};
use crate::portals::{self, Extras, Face, Landing, Masks, Portals, Sight};
use crate::regions::{Move, RegionGraph};
use crate::turrets::{self, TurretSight};

/// Items that don't affect play.
pub const DECORATIVE: &[&str] = &[
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
const MAX_PEDESTALS: usize = 32;
const GRAVITY: f64 = 600.0;
const VOXEL: f64 = 128.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    Conservative,
    Liberal,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rules {
    /// Climb a one-voxel ledge by standing on a cube.
    pub cube_steps: bool,
    pub fire_while_holding: bool,
    /// Units per second, for timed dashes.
    pub walk_speed: f64,
    pub sight: Sight,
    pub turret_sight: TurretSight,
    /// Run through places a live turret can shoot (doing nothing else there).
    pub run_past_turrets: bool,
    pub gel: GelRules,
}

impl Rules {
    fn turret_sight(mode: Mode) -> TurretSight {
        match mode {
            Mode::Conservative => TurretSight {
                half_angle: 60.0,
                range: 16.0,
                grating_blocks: false,
            },
            Mode::Liberal => TurretSight {
                half_angle: 45.0,
                range: 8.0,
                grating_blocks: true,
            },
        }
    }

    fn gel(mode: Mode) -> GelRules {
        match mode {
            Mode::Conservative => GelRules {
                radius: 1,
                bounce_height: 2,
                bounce_reach: 1,
                leap_run: 2,
                leap_gap: 3,
            },
            Mode::Liberal => GelRules {
                radius: 2,
                bounce_height: 4,
                bounce_reach: 2,
                leap_run: 1,
                leap_gap: 5,
            },
        }
    }
}

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

/// Cubes with the same kind and home are interchangeable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Cube {
    pub kind: CubeKind,
    /// Where it respawns after a fizzler.
    pub home: CubeLoc,
    pub loc: CubeLoc,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Drop off a ledge, climb a step using a cube, or walk through a hazard field.
    Move {
        to: usize,
        kind: Move,
        hazards: u64,
    },
    Fire {
        color: Color,
        class: usize,
        /// Fired standing on this floor button (index into `Model::buttons`), which is held
        /// down meanwhile.
        stand_on: Option<usize>,
    },
    /// Walk into the portal of this color.
    Enter {
        color: Color,
    },
    PickUp {
        cube: CubeKind,
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
    /// Press a latching pedestal button (index into `Model::pedestals`).
    Press {
        pedestal: usize,
    },
    /// Press a timed pedestal and walk to `to` (or out of the exit) before it runs out.
    Dash {
        pedestal: usize,
        to: usize,
        escaped: bool,
    },
    /// Walk up to a turret from where it can't shoot and knock it over.
    Knock {
        turret: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct State {
    pub region: usize,
    /// Face class of each portal, `[blue, orange]`.
    pub portals: [Option<usize>; 2],
    /// Sorted, so interchangeable cubes in permuted places are one state.
    pub cubes: Vec<Cube>,
    /// Bit `i`: latching pedestal `i` has been pressed.
    pub latched: u32,
    /// Left through the exit during a timed dash.
    pub escaped: bool,
    /// Bit `i`: turret `i` has been knocked over.
    pub down: u32,
    /// Painted faces, sorted, each `face << 2 | PaintKind` (face: index into
    /// `Model::paint_faces`).
    pub paint: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct Button {
    pub index: i32,
    pub region: Option<usize>,
    pub kind: ButtonKind,
    pub cell: Cell,
}

#[derive(Clone, Debug)]
pub struct Pedestal {
    pub index: i32,
    pub region: Option<usize>,
    /// Seconds; `None` latches.
    pub timer: Option<f64>,
    pub cell: Cell,
}

/// Something that drives a receiver: a floor button or a pedestal (indices into
/// `Model::buttons` and `Model::pedestals`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Signal {
    Floor(usize),
    Pedestal(usize),
}

#[derive(Clone, Debug)]
pub struct HazardInfo {
    pub index: i32,
    pub kind: HazardKind,
    start_enabled: bool,
    inputs: Vec<Signal>,
}

#[derive(Clone, Debug)]
pub struct TurretInfo {
    pub index: i32,
    pub cell: Cell,
    /// Regions it can be knocked over from: its own cell, or beside it at the same height.
    reach: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct GelInfo {
    /// The splat item.
    pub index: i32,
    pub kind: PaintKind,
    /// The dropper item, if a dropper pours it; otherwise it is painted from the start.
    pub dropper: Option<i32>,
    start_enabled: bool,
    inputs: Vec<Signal>,
    /// Faces (indices into `Model::paint_faces`) it paints where it lands, unredirected.
    spot: Vec<usize>,
}

/// A move off painted floor: every face in `needs` must have `paint`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct GelMove {
    from: usize,
    to: usize,
    kind: Move,
    paint: PaintKind,
    needs: Vec<usize>,
    hazards: u64,
}

/// Standable cells and on-foot moves between them, for timed dashes.
#[derive(Default)]
struct WalkGraph {
    node: HashMap<Cell, usize>,
    region: Vec<usize>,
    /// (to, fall height in voxels, hazards crossed). Every move walks one voxel first.
    moves: Vec<Vec<(usize, i32, u64)>>,
}

/// Everything the search needs, precomputed from a level.
pub struct Model {
    pub mode: Mode,
    pub rules: Rules,
    pub graph: RegionGraph,
    pub portals: Portals,
    pub buttons: Vec<Button>,
    pub pedestals: Vec<Pedestal>,
    pub hazards: Vec<HazardInfo>,
    pub start: State,
    pub exit_region: usize,
    pub exit_cell: Cell,
    exit_inputs: Vec<Signal>,
    exit_inverted: bool,
    /// Per region: where a cube dropped off each of its ledges lands.
    tosses: Vec<Vec<Landing>>,
    walk: WalkGraph,
    /// Per floor button the player can stand on (`[]` for the others), per face class: the
    /// masks of the clear shots from its cell.
    button_sight: Vec<Vec<Masks>>,
    /// Face classes the search may use. Metrics switch one off to test whether it's needed.
    pub class_enabled: Vec<bool>,
    pub turrets: Vec<TurretInfo>,
    /// Turrets that exist. Metrics switch one off to see whether it changes the solution.
    pub turret_enabled: Vec<bool>,
    pub gels: Vec<GelInfo>,
    /// Gels that paint. Metrics switch one off to test whether it's needed.
    pub gel_enabled: Vec<bool>,
    /// Every face paint can land on.
    pub paint_faces: Vec<Face>,
    /// Gels poured by droppers, and per gel where it paints (as `spot_masks`).
    pouring: Vec<usize>,
    gel_masks: Vec<Vec<(usize, u32)>>,
    /// Per face class: where gel coming out of a portal there paints, if it can.
    exit_masks: Vec<Option<Vec<(usize, u32)>>>,
    gel_moves: Vec<GelMove>,
    /// Buttons and pedestals that send a signal. Metrics switch one off to test whether it's
    /// needed: a disabled one never counts as pressed.
    pub button_enabled: Vec<bool>,
    pub pedestal_enabled: Vec<bool>,
    /// Fields that can be on. Metrics switch one off to see whether it changes the solution.
    pub hazard_enabled: Vec<bool>,
    /// Per face class: a turret stands on it or a gel stream lands on it, so a portal there
    /// can make `settle` do something.
    special: Vec<bool>,
    /// Per region, for the search: its slice of `graph.edges` and of `gel_moves` (both sorted
    /// by region), the face classes it may fire at (seen from the region or from a button
    /// there), the weighted buttons it can stand on to fire, its buttons and pedestals, and the
    /// turrets it can knock over.
    edge_range: Vec<(usize, usize)>,
    gel_move_range: Vec<(usize, usize)>,
    fire_from: Vec<Vec<Target>>,
    footholds: Vec<Vec<usize>>,
    buttons_in: Vec<Vec<usize>>,
    pedestals_in: Vec<Vec<usize>>,
    knock_from: Vec<Vec<usize>>,
    /// Hazards that are fizzlers, and those that are laser fields.
    pub fizzlers: u64,
    pub lasers: u64,
}

/// Every state reachable from the start, in BFS order, for metrics.
pub struct StateSpace {
    /// BFS depth (actions from the start).
    pub depth: Vec<usize>,
    pub goal: Vec<bool>,
    /// False if the state budget ran out; the graph is then only the nearest part.
    pub complete: bool,
    /// The state being expanded when the budget first ran out.
    overflow: Option<usize>,
    /// Successors of state `i`: `targets[offsets[i]..offsets[i + 1]]`, one per action, and what
    /// each action uses (`uses`).
    offsets: Vec<u32>,
    targets: Vec<u32>,
    uses: Vec<u32>,
    /// How each state but the start was first reached: (state, action).
    parent: Vec<(u32, Action)>,
}

impl StateSpace {
    /// Number of states.
    pub fn len(&self) -> usize {
        self.goal.len()
    }

    pub fn is_empty(&self) -> bool {
        self.goal.is_empty()
    }

    /// Successors of state `i`, one entry per action (indices of states).
    pub fn next(&self, i: usize) -> &[u32] {
        &self.targets[self.offsets[i] as usize..self.offsets[i + 1] as usize]
    }

    /// What `Model::solve` with the same budget says: the plan to the first goal in BFS order,
    /// unless the budget ran out before the search got there.
    pub fn verdict(&self) -> Verdict {
        match self.goal.iter().position(|&g| g) {
            Some(g) if self.overflow.is_none_or(|o| o >= g) => {
                let mut plan = Vec::new();
                let mut i = g;
                while i > 0 {
                    let (p, a) = self.parent[i - 1];
                    plan.push(a);
                    i = p as usize;
                }
                plan.reverse();
                Verdict::Solved(plan)
            }
            None if self.complete => Verdict::Unsolvable,
            _ => Verdict::Unknown,
        }
    }

    /// Whether a goal can be reached without firing at face class `class`: on a complete space,
    /// whether the level is solvable with that class switched off.
    pub fn solvable_without_class(&self, class: usize) -> bool {
        self.reaches_goal(USES_CLASS | class as u32)
    }

    /// Whether a goal can be reached without pressing pedestal `p`: on a complete space,
    /// whether the level is solvable with that pedestal switched off.
    pub fn solvable_without_pedestal(&self, p: usize) -> bool {
        self.reaches_goal(USES_PEDESTAL | p as u32)
    }

    /// Whether a goal is reachable from the start along actions that don't use `skip`.
    fn reaches_goal(&self, skip: u32) -> bool {
        let mut seen = vec![false; self.len()];
        let mut stack = vec![0];
        seen[0] = true;
        while let Some(i) = stack.pop() {
            if self.goal[i] {
                return true;
            }
            for e in self.offsets[i] as usize..self.offsets[i + 1] as usize {
                let j = self.targets[e] as usize;
                if self.uses[e] != skip && !std::mem::replace(&mut seen[j], true) {
                    stack.push(j);
                }
            }
        }
        false
    }
}

/// `StateSpace::uses` of an action: the face class it fires at or the pedestal it presses.
const USES_CLASS: u32 = 1 << 30;
const USES_PEDESTAL: u32 = 2 << 30;

fn uses(a: &Action) -> u32 {
    match *a {
        Action::Fire { class, .. } => USES_CLASS | class as u32,
        Action::Press { pedestal } | Action::Dash { pedestal, .. } => {
            USES_PEDESTAL | pedestal as u32
        }
        _ => 0,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Problem {
    /// The level is broken (no entry, exit door floating, ...).
    Invalid(String),
    /// The level uses something the solver doesn't model.
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

/// Both modes combined.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Judgement {
    /// Solved with conservative rules.
    Solvable(Vec<Action>),
    /// Not solvable even with liberal rules.
    Unsolvable,
    /// Liberal rules solve it (plan given) and conservative don't, or a budget ran out.
    Uncertain(Option<Vec<Action>>),
}

pub struct Judged {
    pub judgement: Judgement,
    pub conservative: Model,
    pub liberal: Model,
}

/// Solves `level` in both modes.
pub fn judge(level: &Level, budget: usize) -> Result<Judged, Vec<Problem>> {
    let conservative = Model::new(level, Mode::Conservative)?;
    let liberal = Model::new(level, Mode::Liberal)?;
    let judgement = match conservative.solve(budget).verdict {
        Verdict::Solved(plan) => Judgement::Solvable(plan),
        strict => match liberal.solve(budget).verdict {
            Verdict::Unsolvable if strict == Verdict::Unsolvable => Judgement::Unsolvable,
            Verdict::Solved(plan) => Judgement::Uncertain(Some(plan)),
            _ => Judgement::Uncertain(None),
        },
    };
    Ok(Judged {
        judgement,
        conservative,
        liberal,
    })
}

impl Model {
    pub fn new(level: &Level, mode: Mode) -> Result<Model, Vec<Problem>> {
        let mut problems: Vec<Problem> = level
            .decode_problems
            .iter()
            .map(|p| Problem::Invalid(p.clone()))
            .collect();
        // Regions split where the turrets that can shoot change.
        let turret_sight = Rules::turret_sight(mode);
        let danger: HashMap<Cell, u32> = if level.turrets.is_empty() {
            HashMap::new()
        } else {
            level
                .cells()
                .filter(|&c| level.standable(c))
                .map(|c| (c, turrets::danger(level, turret_sight, c)))
                .collect()
        };
        let graph = RegionGraph::build_with(level, |c| danger.get(&c).copied().unwrap_or(0));

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
                kind: b.button_kind(),
                cell: b.cell,
            })
            .collect();
        let pedestals: Vec<Pedestal> = level
            .of_kind(&ItemKind::PedestalButton)
            .map(|p| Pedestal {
                index: p.index,
                region: graph.region_of(p.cell),
                timer: p.timer(),
                cell: p.cell,
            })
            .collect();
        if pedestals.len() > MAX_PEDESTALS {
            problems.push(Problem::Unsupported(format!(
                "more than {MAX_PEDESTALS} pedestal buttons"
            )));
        }
        let mut hazards: Vec<HazardInfo> = level
            .hazards
            .iter()
            .map(|h| HazardInfo {
                index: h.item,
                kind: h.kind,
                start_enabled: h.start_enabled,
                inputs: Vec::new(),
            })
            .collect();
        assert!(hazards.len() <= MAX_HAZARDS);
        let gel_rules = Rules::gel(mode);
        let mut gels: Vec<GelInfo> = level
            .gels
            .iter()
            .map(|g| GelInfo {
                index: g.item,
                kind: g.kind,
                dropper: g.dropper.map(|(d, _)| d),
                start_enabled: g.start_enabled,
                inputs: Vec::new(),
                spot: Vec::new(),
            })
            .collect();

        let mut exit_inputs = Vec::new();
        for link in &level.links {
            let receiver = level.item(link.receiver).map(|i| &i.kind);
            let sender = level.item(link.sender).map(|i| &i.kind);
            let other = |k: Option<&ItemKind>| matches!(k, Some(ItemKind::Other(_)));
            if other(receiver) || other(sender) {
                continue; // Already reported as an unsupported item.
            }
            let signal = match sender {
                Some(ItemKind::FloorButton) => buttons
                    .iter()
                    .position(|b| b.index == link.sender)
                    .map(Signal::Floor),
                Some(ItemKind::PedestalButton) => pedestals
                    .iter()
                    .position(|p| p.index == link.sender)
                    .map(Signal::Pedestal),
                _ => None,
            };
            let Some(signal) = signal else {
                problems.push(Problem::Unsupported(format!(
                    "wire {} -> {} comes from something other than a button",
                    link.sender, link.receiver
                )));
                continue;
            };
            match receiver {
                Some(ItemKind::ExitDoor) => exit_inputs.push(signal),
                Some(ItemKind::Fizzler | ItemKind::LaserField) => {
                    match hazards.iter_mut().find(|h| h.index == link.receiver) {
                        Some(h) => h.inputs.push(signal),
                        None => problems.push(Problem::Invalid(format!(
                            "wire {} -> {} goes to a field that couldn't be decoded",
                            link.sender, link.receiver
                        ))),
                    }
                }
                Some(ItemKind::PaintDropper) => {
                    match gels.iter_mut().find(|g| g.dropper == Some(link.receiver)) {
                        Some(g) => g.inputs.push(signal),
                        None => problems.push(Problem::Invalid(format!(
                            "wire {} -> {} goes to a gel dropper with no splat",
                            link.sender, link.receiver
                        ))),
                    }
                }
                _ => problems.push(Problem::Unsupported(format!(
                    "wire {} -> {} drives something other than the exit door or a field",
                    link.sender, link.receiver
                ))),
            }
        }

        let landing_loc = |l: Landing, kind: CubeKind| match l
            .button
            .and_then(|i| buttons.iter().position(|b| b.index == i))
        {
            Some(b) if buttons[b].kind.accepts(kind) => CubeLoc::OnButton(b),
            _ => CubeLoc::Loose(l.region),
        };
        let mut cubes = Vec::new();
        for cube in level.of_kind(&ItemKind::Cube) {
            // A cube that would fall into goo is simply gone.
            if let Some(l) = Landing::from_fall(level, &graph, cube.cell) {
                let loc = landing_loc(l, cube.cube_kind());
                cubes.push(Cube {
                    kind: cube.cube_kind(),
                    home: loc,
                    loc,
                });
            }
        }
        cubes.sort();

        let tosses = graph
            .regions
            .iter()
            .map(|r| {
                let mut landings: Vec<Landing> = r
                    .cells
                    .iter()
                    .flat_map(|&c| Dir::HORIZONTAL.map(|d| (c, d)))
                    .filter(|&(c, d)| level.face(c, d).barrier.is_none())
                    .filter_map(|(c, d)| {
                        let n = step(c, d);
                        if level.solid(n) || level.standable(n) {
                            return None;
                        }
                        let mut l = Landing::from_fall(level, &graph, n)?;
                        l.hazards |= level.face(c, d).hazards;
                        Some(l)
                    })
                    .collect();
                landings.sort();
                landings.dedup();
                landings
            })
            .collect();

        let (Some((_, start)), Some((exit_door, exit_region))) = (entry, exit) else {
            return Err(problems);
        };
        if !problems.is_empty() {
            return Err(problems);
        }

        // Fizzlers stop shots while on. Every fizzler goes in the shot masks, even one that is
        // always on, so that switching it off (`hazard_enabled`) clears its shots too.
        let blocking_hazards = 0u64;
        let switchable_hazards = hazards
            .iter()
            .enumerate()
            .filter(|(_, h)| h.kind == HazardKind::Fizzler)
            .fold(0u64, |m, (i, _)| m | 1 << i);
        let rules = match mode {
            Mode::Conservative => Rules {
                cube_steps: false,
                fire_while_holding: false,
                walk_speed: 140.0,
                sight: Sight {
                    blocking_hazards,
                    switchable_hazards,
                    strict_grazing: true,
                    corner_targets: false,
                },
                turret_sight,
                run_past_turrets: false,
                gel: gel_rules,
            },
            Mode::Liberal => Rules {
                cube_steps: true,
                fire_while_holding: true,
                walk_speed: 190.0,
                sight: Sight {
                    blocking_hazards,
                    switchable_hazards,
                    strict_grazing: false,
                    corner_targets: true,
                },
                turret_sight,
                run_past_turrets: true,
                gel: gel_rules,
            },
        };

        // Gel: every face paint can land on, by id.
        let mut paint_faces: Vec<Face> = Vec::new();
        let mut face_id: HashMap<Face, usize> = HashMap::new();
        fn intern(
            faces: Vec<Face>,
            paint_faces: &mut Vec<Face>,
            face_id: &mut HashMap<Face, usize>,
        ) -> Vec<usize> {
            faces
                .into_iter()
                .map(|f| {
                    *face_id.entry(f).or_insert_with(|| {
                        paint_faces.push(f);
                        paint_faces.len() - 1
                    })
                })
                .collect()
        }
        let floor_spot = |c: Cell| gel::spot(level, (c, Dir::NegZ), gel_rules.radius);
        // The floor face each stream lands on.
        let mut stream_faces: Vec<Option<Face>> = Vec::new();
        for (g, info) in level.gels.iter().zip(gels.iter_mut()) {
            let (faces, stream) = match g.dropper {
                Some((_, cell)) => match level.fall(cell) {
                    Some(land) => (floor_spot(land), Some((land, Dir::NegZ))),
                    None => (Vec::new(), None),
                },
                None => (gel::spot(level, g.face, gel_rules.radius), None),
            };
            info.spot = intern(faces, &mut paint_faces, &mut face_id);
            stream_faces.push(stream);
        }
        let portal_gel: HashSet<usize> = gels
            .iter()
            .filter(|g| g.kind == PaintKind::Portal)
            .flat_map(|g| g.spot.iter().copied())
            .collect();
        // A stream can go through a portal if the floor it lands on is white or can be made so.
        let redirectable: Vec<usize> = (0..gels.len())
            .filter(|&g| {
                stream_faces[g].is_some_and(|(c, d)| {
                    level.portal_surface(c, d)
                        || face_id.get(&(c, d)).is_some_and(|f| portal_gel.contains(f))
                })
            })
            .collect();
        let mut cell_spots: HashMap<Cell, Vec<usize>> = HashMap::new();
        if !redirectable.is_empty() {
            for (c, d) in level.portal_surfaces() {
                if d != Dir::PosZ {
                    continue;
                }
                if let Some(land) = level.fall(c) {
                    if !cell_spots.contains_key(&land) {
                        let spot = intern(floor_spot(land), &mut paint_faces, &mut face_id);
                        cell_spots.insert(land, spot);
                    }
                }
            }
        }
        // Faces conversion gel can paint: where portal gel lands, and anywhere a stream of it
        // can be sent.
        let mut whitenable: HashSet<usize> = portal_gel;
        if redirectable
            .iter()
            .any(|&g| gels[g].kind == PaintKind::Portal)
        {
            whitenable.extend(cell_spots.values().flatten().copied());
        }
        let mut paintable: Vec<(Face, usize)> = whitenable
            .into_iter()
            .map(|f| (paint_faces[f], f))
            .filter(|&((c, d), _)| !level.portal_surface(c, d))
            .collect();
        paintable.sort();
        let extras = Extras {
            streams: redirectable
                .iter()
                .map(|&g| (stream_faces[g].unwrap(), g))
                .collect(),
            split_ceilings: !redirectable.is_empty(),
            paintable,
        };

        // Moves off painted floors.
        let has = |k: PaintKind| gels.iter().any(|g| g.kind == k);
        let mut gel_moves: HashSet<GelMove> = HashSet::new();
        for (f, &(c, d)) in paint_faces.iter().enumerate() {
            let Some(from) = graph.region_of(c).filter(|_| d == Dir::NegZ) else {
                continue;
            };
            if has(PaintKind::Bounce) {
                for (t, hazards) in gel::bounces(level, c, gel_rules) {
                    if let Some(to) = graph.region_of(t) {
                        gel_moves.insert(GelMove {
                            from,
                            to,
                            kind: Move::Bounce,
                            paint: PaintKind::Bounce,
                            needs: vec![f],
                            hazards,
                        });
                    }
                }
            }
            if has(PaintKind::Speed) {
                for dir in Dir::HORIZONTAL {
                    let Some((run, t, hazards)) = gel::leap(level, c, dir, gel_rules) else {
                        continue;
                    };
                    let needs: Option<Vec<usize>> = run
                        .iter()
                        .map(|&r| face_id.get(&(r, Dir::NegZ)).copied())
                        .collect();
                    if let (Some(needs), Some(to)) = (needs, graph.region_of(t)) {
                        gel_moves.insert(GelMove {
                            from,
                            to,
                            kind: Move::Leap,
                            paint: PaintKind::Speed,
                            needs,
                            hazards,
                        });
                    }
                }
            }
        }
        let mut gel_moves: Vec<GelMove> = gel_moves.into_iter().collect();
        gel_moves.sort_by_key(|m| (m.from, m.to, m.needs.clone(), m.hazards, m.kind));

        let turret_list: Vec<TurretInfo> = level
            .turrets
            .iter()
            .map(|t| {
                let mut reach: Vec<usize> = std::iter::once(t.cell)
                    .chain(
                        Dir::HORIZONTAL
                            .iter()
                            .filter(|&&d| level.face(t.cell, d).barrier.is_none())
                            .map(|&d| step(t.cell, d)),
                    )
                    .filter_map(|c| graph.region_of(c))
                    .collect();
                reach.sort();
                reach.dedup();
                TurretInfo {
                    index: t.item,
                    cell: t.cell,
                    reach,
                }
            })
            .collect();
        let walk = if pedestals.iter().any(|p| p.timer.is_some()) {
            walk_graph(level, &graph)
        } else {
            WalkGraph::default()
        };
        let portals = Portals::build(level, &graph, rules.sight, &extras);
        let button_sight: Vec<Vec<Masks>> = buttons
            .iter()
            .map(|b| match b.region {
                Some(_) if b.kind == ButtonKind::Weighted => portals
                    .classes
                    .iter()
                    .map(|c| portals::shot_masks(level, &[b.cell], &c.faces, rules.sight))
                    .collect(),
                _ => Vec::new(),
            })
            .collect();

        let exit_masks = portals
            .classes
            .iter()
            .map(|c| {
                c.gel_cell
                    .and_then(|cell| cell_spots.get(&cell))
                    .map(|spot| spot_masks(spot))
            })
            .collect();
        let regions = graph.regions.len();
        let by_region = |len: usize, region_of: &dyn Fn(usize) -> Option<usize>| {
            let mut out = vec![Vec::new(); regions];
            for i in 0..len {
                if let Some(r) = region_of(i) {
                    out[r].push(i);
                }
            }
            out
        };
        let ranges = |from: &[usize]| -> Vec<(usize, usize)> {
            (0..regions)
                .map(|r| {
                    let lo = from.partition_point(|&f| f < r);
                    (lo, from.partition_point(|&f| f <= r))
                })
                .collect()
        };
        let edge_from: Vec<usize> = graph.edges.iter().map(|e| e.from).collect();
        let gel_move_from: Vec<usize> = gel_moves.iter().map(|m| m.from).collect();
        let footholds = by_region(buttons.len(), &|b| {
            buttons[b].region.filter(|_| !button_sight[b].is_empty())
        });
        let fire_from = (0..regions)
            .map(|r| {
                portals
                    .classes
                    .iter()
                    .filter(|c| {
                        c.seen_from(r)
                            || footholds[r]
                                .iter()
                                .any(|&b| !button_sight[b][c.id].is_empty())
                    })
                    .map(|c| Target {
                        class: c.id,
                        single: c.faces.len() == 1,
                        needs_paint: c.needs_paint,
                        open: c.visible_from[r].contains(&0),
                    })
                    .collect()
            })
            .collect();
        let knock_from = (0..regions)
            .map(|r| {
                (0..turret_list.len())
                    .filter(|&t| turret_list[t].reach.contains(&r))
                    .collect()
            })
            .collect();
        let kind_mask = |kind: HazardKind| {
            hazards
                .iter()
                .enumerate()
                .filter(|(_, h)| h.kind == kind)
                .fold(0u64, |m, (i, _)| m | 1 << i)
        };
        Ok(Model {
            edge_range: ranges(&edge_from),
            gel_move_range: ranges(&gel_move_from),
            fire_from,
            special: portals
                .classes
                .iter()
                .map(|c| c.turret.is_some() || c.stream.is_some())
                .collect(),
            buttons_in: by_region(buttons.len(), &|b| buttons[b].region),
            pedestals_in: by_region(pedestals.len(), &|p| pedestals[p].region),
            footholds,
            knock_from,
            fizzlers: kind_mask(HazardKind::Fizzler),
            lasers: kind_mask(HazardKind::LaserField),
            exit_masks,
            pouring: (0..gels.len())
                .filter(|&g| gels[g].dropper.is_some())
                .collect(),
            gel_masks: gels.iter().map(|g| spot_masks(&g.spot)).collect(),
            mode,
            rules,
            class_enabled: vec![true; portals.classes.len()],
            turret_enabled: vec![true; turret_list.len()],
            turrets: turret_list,
            gel_enabled: vec![true; gels.len()],
            gels,
            paint_faces,
            gel_moves,
            button_enabled: vec![true; buttons.len()],
            pedestal_enabled: vec![true; pedestals.len()],
            hazard_enabled: vec![true; hazards.len()],
            exit_inverted: !exit_inputs.is_empty() && exit_door.start_open == Some(true),
            exit_cell: exit_door.cell,
            graph,
            portals,
            buttons,
            pedestals,
            hazards,
            start: State {
                region: start,
                portals: [None, None],
                cubes,
                latched: 0,
                escaped: false,
                down: 0,
                paint: Vec::new(),
            },
            exit_region,
            exit_inputs,
            tosses,
            walk,
            button_sight,
        })
    }

    /// The start state: paint from splats without droppers, then whatever the droppers pour.
    pub fn initial(&self) -> State {
        let cx = self.context(&self.start.cubes);
        self.unpack(&cx.lay, &self.initial_packed(&cx))
    }

    pub fn painted(&self, s: &State, face: usize, kind: PaintKind) -> bool {
        s.paint.binary_search(&paint_code(face, kind)).is_ok()
    }

    /// Whether a pouring gel stream goes through a portal in state `s`.
    pub fn redirected(&self, s: &State) -> bool {
        let cx = self.context(&s.cubes);
        let s = self.pack(&cx.lay, s);
        (0..self.gels.len()).any(|g| {
            self.gels[g].dropper.is_some()
                && self.gel_enabled[g]
                && self.gel_on(&cx.lay, &s, g)
                && self.redirect(&s, g).is_some()
        })
    }

    /// Bit `h`: hazard `h` is on in state `s`, with `held` on for the moment as well (a running
    /// timed pedestal, or a floor button the player stands on).
    pub fn hazards_on_in(&self, s: &State, held: Option<Signal>) -> u64 {
        let cx = self.context(&s.cubes);
        self.hazards_on(&cx.lay, &self.pack(&cx.lay, s), held)
    }

    /// The buttons and pedestals wired to hazard `h`.
    pub fn hazard_inputs(&self, h: usize) -> &[Signal] {
        &self.hazards[h].inputs
    }

    /// Whether the exit can only open while `sig` is on: switched off, it keeps the level
    /// unsolvable.
    pub fn exit_needs(&self, sig: Signal) -> bool {
        !self.exit_inverted && self.exit_inputs.contains(&sig)
    }

    pub fn exit_open(&self, s: &State) -> bool {
        let cx = self.context(&s.cubes);
        self.exit_open_with(&cx.lay, &self.pack(&cx.lay, s), None)
    }

    pub fn is_goal(&self, s: &State) -> bool {
        let cx = self.context(&s.cubes);
        self.goal(&cx.lay, &self.pack(&cx.lay, s))
    }

    /// Every action possible in state `s`, with the state it leads to.
    pub fn successors(&self, s: &State) -> Vec<(Action, State)> {
        let cx = self.context(&s.cubes);
        let s = self.pack(&cx.lay, s);
        let mut settled = s.clone();
        self.settle(&cx.lay, &mut settled);
        let mut out = Vec::new();
        self.expand(
            &cx,
            &s,
            settled == s,
            &mut Scratch::default(),
            &mut |a, n| out.push((a, self.unpack(&cx.lay, n))),
        );
        out
    }

    /// Breadth-first search from the start state, exploring at most `max_states` states.
    pub fn solve(&self, max_states: usize) -> Outcome {
        self.search(max_states, None)
    }

    /// Like `solve`, but gives up (`Unknown`) once no plan of at most `max_len` actions is left
    /// to find. A plan it returns is the one `solve` would.
    pub fn solve_within(&self, max_states: usize, max_len: usize) -> Outcome {
        self.search(max_states, Some(max_len))
    }

    fn search(&self, max_states: usize, max_len: Option<usize>) -> Outcome {
        let cx = self.context(&self.start.cubes);
        let mut table = Table::new(cx.lay.len);
        let start = self.initial_packed(&cx);
        table.insert(&start, Table::hash(&start));
        let mut parent: Vec<(u32, Action)> = Vec::new();
        let mut depth: Vec<u32> = vec![0];
        let (mut cur, mut sc) = (Vec::new(), Scratch::default());
        let mut next = 0;
        while next < table.len() {
            cur.clear();
            cur.extend_from_slice(table.get(next));
            if self.goal(&cx.lay, &cur) {
                let mut plan = Vec::new();
                let mut i = next;
                while i > 0 {
                    let (p, a) = parent[i - 1];
                    plan.push(a);
                    i = p as usize;
                }
                plan.reverse();
                return Outcome {
                    verdict: Verdict::Solved(plan),
                    states: table.len(),
                };
            }
            if max_len.is_some_and(|m| depth[next] as usize > m) {
                return Outcome {
                    verdict: Verdict::Unknown,
                    states: table.len(),
                };
            }
            let mut full = false;
            self.expand(&cx, &cur, true, &mut sc, &mut |a, n| {
                let h = Table::hash(n);
                if full || table.find(n, h).is_some() {
                    return;
                }
                if table.len() >= max_states {
                    full = true;
                    return;
                }
                table.insert(n, h);
                parent.push((next as u32, a));
                depth.push(depth[next] + 1);
            });
            if full {
                return Outcome {
                    verdict: Verdict::Unknown,
                    states: table.len(),
                };
            }
            next += 1;
        }
        Outcome {
            verdict: Verdict::Unsolvable,
            states: table.len(),
        }
    }

    /// Explores every reachable state (up to `max_states`), past goals too: metrics need the
    /// whole graph to find dead ends.
    pub fn explore(&self, max_states: usize) -> StateSpace {
        let cx = self.context(&self.start.cubes);
        let mut table = Table::new(cx.lay.len);
        let start = self.initial_packed(&cx);
        table.insert(&start, Table::hash(&start));
        let mut space = StateSpace {
            depth: vec![0],
            goal: Vec::new(),
            complete: true,
            overflow: None,
            offsets: vec![0],
            targets: Vec::new(),
            uses: Vec::new(),
            parent: Vec::new(),
        };
        let (mut cur, mut sc) = (Vec::new(), Scratch::default());
        let mut i = 0;
        while i < table.len() {
            cur.clear();
            cur.extend_from_slice(table.get(i));
            space.goal.push(self.goal(&cx.lay, &cur));
            self.expand(&cx, &cur, true, &mut sc, &mut |a, n| {
                let h = Table::hash(n);
                let j = match table.find(n, h) {
                    Some(j) => j,
                    None if table.len() >= max_states => {
                        space.complete = false;
                        space.overflow.get_or_insert(i);
                        return;
                    }
                    None => {
                        space.depth.push(space.depth[i] + 1);
                        space.parent.push((i as u32, a));
                        table.insert(n, h)
                    }
                };
                space.targets.push(j as u32);
                space.uses.push(uses(&a));
            });
            space.offsets.push(space.targets.len() as u32);
            i += 1;
        }
        space
    }

    /// How states with these cubes are packed, and the turrets that exist.
    fn context(&self, cubes: &[Cube]) -> Ctx {
        // Cubes are sorted, so interchangeable ones are next to each other.
        let ids: Vec<(CubeKind, CubeLoc)> = cubes.iter().map(|c| (c.kind, c.home)).collect();
        let mut group = vec![(0, 0); ids.len()];
        let mut lo = 0;
        for i in 1..=ids.len() {
            if i == ids.len() || ids[i] != ids[lo] {
                group[lo..i].fill((lo, i));
                lo = i;
            }
        }
        let paint = CUBES + ids.len();
        Ctx {
            lay: Layout {
                homes: ids.iter().map(|&(_, home)| self.loc_code(home)).collect(),
                kinds: ids.iter().map(|&(kind, _)| kind).collect(),
                group,
                paint,
                len: paint + self.paint_faces.len().div_ceil(16),
            },
            turrets: self
                .turret_enabled
                .iter()
                .enumerate()
                .filter(|(_, &e)| e)
                .fold(0u32, |m, (i, _)| m | 1 << i),
        }
    }

    fn loc_code(&self, l: CubeLoc) -> u32 {
        match l {
            CubeLoc::Held => HELD,
            CubeLoc::Loose(r) => 1 + r as u32,
            CubeLoc::OnButton(b) => self.button_code(b),
        }
    }

    fn button_code(&self, b: usize) -> u32 {
        (1 + self.graph.regions.len() + b) as u32
    }

    fn code_loc(&self, code: u32) -> CubeLoc {
        let regions = self.graph.regions.len();
        match code as usize {
            0 => CubeLoc::Held,
            c if c <= regions => CubeLoc::Loose(c - 1),
            c => CubeLoc::OnButton(c - 1 - regions),
        }
    }

    fn pack(&self, lay: &Layout, s: &State) -> Vec<u32> {
        let mut w = vec![0; lay.len];
        w[0] = s.region as u32 | if s.escaped { ESCAPED } else { 0 };
        w[BLUE] = s.portals[0].map_or(0, |c| c as u32 + 1);
        w[BLUE + 1] = s.portals[1].map_or(0, |c| c as u32 + 1);
        w[LATCHED] = s.latched;
        w[DOWN] = s.down;
        for (i, c) in s.cubes.iter().enumerate() {
            w[CUBES + i] = self.loc_code(c.loc);
        }
        for &e in &s.paint {
            lay.set_paint(&mut w, e as usize >> 2, (e & 3) + 1);
        }
        w
    }

    fn unpack(&self, lay: &Layout, w: &[u32]) -> State {
        State {
            region: (w[0] & !ESCAPED) as usize,
            portals: [portal(w, 0), portal(w, 1)],
            cubes: (0..lay.kinds.len())
                .map(|i| Cube {
                    kind: lay.kinds[i],
                    home: self.code_loc(lay.homes[i]),
                    loc: self.code_loc(w[CUBES + i]),
                })
                .collect(),
            latched: w[LATCHED],
            escaped: w[0] & ESCAPED != 0,
            down: w[DOWN],
            paint: (0..self.paint_faces.len())
                .filter_map(|f| match lay.paint_at(w, f) {
                    0 => None,
                    v => Some(((f as u32) << 2) | (v - 1)),
                })
                .collect(),
        }
    }

    fn initial_packed(&self, cx: &Ctx) -> Vec<u32> {
        let mut s = self.pack(&cx.lay, &self.start);
        for (g, gel) in self.gels.iter().enumerate() {
            if gel.dropper.is_none() && self.gel_enabled[g] {
                cx.lay.paint(&mut s, &gel.spot, gel.kind);
            }
        }
        self.settle(&cx.lay, &mut s);
        s
    }

    /// `held`: a signal held on for the moment, beyond what the state says: a running timed
    /// pedestal, or a floor button the player stands on.
    fn signal(&self, lay: &Layout, s: &[u32], sig: Signal, held: Option<Signal>) -> bool {
        let enabled = match sig {
            Signal::Floor(b) => self.button_enabled[b],
            Signal::Pedestal(p) => self.pedestal_enabled[p],
        };
        enabled
            && (held == Some(sig)
                || match sig {
                    Signal::Floor(b) => lay.cubes(s).contains(&self.button_code(b)),
                    Signal::Pedestal(p) => s[LATCHED] & (1 << p) != 0,
                })
    }

    fn all_on(&self, lay: &Layout, s: &[u32], inputs: &[Signal], held: Option<Signal>) -> bool {
        inputs.iter().all(|&i| self.signal(lay, s, i, held))
    }

    fn exit_open_with(&self, lay: &Layout, s: &[u32], held: Option<Signal>) -> bool {
        self.exit_inputs.is_empty()
            || self.all_on(lay, s, &self.exit_inputs, held) != self.exit_inverted
    }

    fn goal(&self, lay: &Layout, s: &[u32]) -> bool {
        s[0] & ESCAPED != 0
            || (s[0] as usize == self.exit_region && self.exit_open_with(lay, s, None))
    }

    /// Bit `h`: hazard `h` is on.
    fn hazards_on(&self, lay: &Layout, s: &[u32], held: Option<Signal>) -> u64 {
        let mut on = 0;
        for (h, hz) in self.hazards.iter().enumerate() {
            let lit = self.hazard_enabled[h]
                && if hz.inputs.is_empty() {
                    hz.start_enabled
                } else {
                    hz.start_enabled != self.all_on(lay, s, &hz.inputs, held)
                };
            on |= (lit as u64) << h;
        }
        on
    }

    fn gel_on(&self, lay: &Layout, s: &[u32], g: usize) -> bool {
        let gel = &self.gels[g];
        if gel.inputs.is_empty() {
            gel.start_enabled
        } else {
            gel.start_enabled != self.all_on(lay, s, &gel.inputs, None)
        }
    }

    /// The class of the portal gel stream `g` comes out of, if one is under it.
    fn redirect(&self, s: &[u32], g: usize) -> Option<usize> {
        let (Some(a), Some(b)) = (portal(s, 0), portal(s, 1)) else {
            return None;
        };
        let classes = &self.portals.classes;
        if classes[a].stream == Some(g) {
            Some(b)
        } else if classes[b].stream == Some(g) {
            Some(a)
        } else {
            None
        }
    }

    /// What follows by itself after any action: a turret over an open portal falls through,
    /// and pouring droppers paint where their streams land.
    fn settle(&self, lay: &Layout, s: &mut [u32]) {
        if let (Some(a), Some(b)) = (portal(s, 0), portal(s, 1)) {
            for c in [a, b] {
                if let Some(t) = self.portals.classes[c].turret {
                    s[DOWN] |= 1 << t;
                }
            }
        }
        for &g in &self.pouring {
            if !self.gel_enabled[g] || !self.gel_on(lay, s, g) {
                continue;
            }
            let masks = match self.redirect(s, g) {
                None => &self.gel_masks[g],
                Some(exit) => match &self.exit_masks[exit] {
                    Some(masks) => masks,
                    None => continue,
                },
            };
            lay.paint_masks(s, masks, self.gels[g].kind);
        }
    }

    /// Where the cube in `slot` coming to rest at `l` ends up; `on`: the hazards that are on.
    fn cube_lands(&self, lay: &Layout, s: &[u32], on: u64, l: Landing, slot: usize) -> CubeLoc {
        if l.hazards & on & self.fizzlers != 0 {
            return self.code_loc(lay.homes[slot]);
        }
        match l
            .button
            .and_then(|i| self.buttons.iter().position(|b| b.index == i))
        {
            Some(b)
                if self.buttons[b].kind.accepts(lay.kinds[slot])
                    && !lay.cubes(s).contains(&self.button_code(b)) =>
            {
                CubeLoc::OnButton(b)
            }
            _ => CubeLoc::Loose(l.region),
        }
    }

    /// The portal you'd come out of when entering `color` from the current region.
    fn through(&self, s: &[u32], color: Color) -> Option<Landing> {
        let (a, b) = match color {
            Color::Blue => (portal(s, 0), portal(s, 1)),
            Color::Orange => (portal(s, 1), portal(s, 0)),
        };
        let (a, b) = (&self.portals.classes[a?], &self.portals.classes[b?]);
        (a.entry == Some(s[0] as usize))
            .then_some(b.landing)
            .flatten()
    }

    /// Moves the player, into `n`, along a path crossing `hazards`: whether that crossed an
    /// active fizzler, or `None` if a laser field stops them, or (conservative rules) a live
    /// turret can shoot where they'd go. `on`: the hazards that are on; `alive`: the live
    /// turrets.
    #[allow(clippy::too_many_arguments)]
    fn travel(
        &self,
        lay: &Layout,
        s: &[u32],
        on: u64,
        alive: u32,
        to: usize,
        hazards: u64,
        n: &mut Vec<u32>,
    ) -> Option<bool> {
        if hazards & on & self.lasers != 0 {
            return None;
        }
        if !self.rules.run_past_turrets && self.graph.regions[to].danger & alive != 0 {
            return None;
        }
        n.clear();
        n.extend_from_slice(s);
        n[0] = to as u32;
        let fizzled = hazards & on & self.fizzlers != 0;
        if fizzled {
            lay.fizzle(n);
        }
        Some(fizzled)
    }

    /// Calls `emit` with every action possible in packed state `s` and the (settled) state it
    /// leads to, always in the same order. `settled`: `s` is as `settle` leaves it, as every
    /// state the search finds is.
    fn expand(
        &self,
        cx: &Ctx,
        s: &[u32],
        settled: bool,
        sc: &mut Scratch,
        emit: &mut impl FnMut(Action, &[u32]),
    ) {
        if s[0] & ESCAPED != 0 {
            return;
        }
        let lay = &cx.lay;
        let r = s[0] as usize;
        let cubes = lay.cubes(s);
        let held = cubes.iter().position(|&c| c == HELD);
        let on = self.hazards_on(lay, s, None);
        let alive = cx.turrets & !s[DOWN];
        // Where a live turret can shoot, the player can only keep moving, and only under
        // liberal rules: conservative ones never go there, so this is the start, and fatal.
        let act = self.graph.regions[r].danger & alive == 0;
        if !act && !self.rules.run_past_turrets {
            return;
        }
        // A `quiet` action changes nothing `settle` looks at (cubes on buttons, pedestals,
        // portals on faces under turrets or gel streams), so from a settled state it needs no
        // settling.
        let mut done = |n: &mut Vec<u32>, a: Action, quiet: bool| {
            if !(quiet && settled) {
                self.settle(lay, n);
            }
            emit(a, n);
        };
        let copy = |n: &mut Vec<u32>| {
            n.clear();
            n.extend_from_slice(s);
        };

        let (lo, hi) = self.edge_range[r];
        for e in &self.graph.edges[lo..hi] {
            let ok = match e.kind {
                Move::Step => self.rules.cube_steps && cubes.contains(&(1 + r as u32)),
                _ => true,
            };
            let moved = ok.then(|| self.travel(lay, s, on, alive, e.to, e.hazards, &mut sc.n));
            if let Some(Some(fizzled)) = moved {
                let a = Action::Move {
                    to: e.to,
                    kind: e.kind,
                    hazards: e.hazards,
                };
                done(&mut sc.n, a, !fizzled);
            }
        }
        let (lo, hi) = self.gel_move_range[r];
        for m in &self.gel_moves[lo..hi] {
            let paint = m.paint as u32 + 1;
            if !m.needs.iter().all(|&f| lay.paint_at(s, f) == paint) {
                continue;
            }
            if let Some(fizzled) = self.travel(lay, s, on, alive, m.to, m.hazards, &mut sc.n) {
                let a = Action::Move {
                    to: m.to,
                    kind: m.kind,
                    hazards: m.hazards,
                };
                done(&mut sc.n, a, !fizzled);
            }
        }
        if !act {
            for color in Color::BOTH {
                if let Some(l) = self.through(s, color) {
                    if let Some(fizzled) =
                        self.travel(lay, s, on, alive, l.region, l.hazards, &mut sc.n)
                    {
                        done(&mut sc.n, Action::Enter { color }, !fizzled);
                    }
                }
            }
            return;
        }

        let can_fire = held.is_none() || self.rules.fire_while_holding;
        // Free weighted buttons here the player could stand on, and the hazards on meanwhile.
        sc.foot.clear();
        for &b in &self.footholds[r] {
            if !cubes.contains(&self.button_code(b)) {
                let on = self.hazards_on(lay, s, Some(Signal::Floor(b)));
                sc.foot.push((b, on));
            }
        }
        // Whether one of the shots with these masks gets past the fizzlers.
        let clear = |masks: &Masks, on: u64| masks.iter().any(|&m| m & on & self.fizzlers == 0);
        let plain = |p: Option<usize>| p.is_none_or(|c| !self.special[c]);
        for color in Color::BOTH {
            let me = color as usize;
            let other = portal(s, color.other() as usize);
            let quiet = plain(portal(s, me)) && plain(other);
            for t in self.fire_from[r].iter().filter(|_| can_fire) {
                let c = t.class;
                let taken = other == Some(c) && t.single;
                let unpainted = t
                    .needs_paint
                    .is_some_and(|f| lay.paint_at(s, f) != PaintKind::Portal as u32 + 1);
                if !self.class_enabled[c] || portal(s, me) == Some(c) || taken || unpainted {
                    continue;
                }
                let visible = &self.portals.classes[c].visible_from[r];
                let stand_on = if t.open || clear(visible, on) {
                    Some(None)
                } else {
                    sc.foot
                        .iter()
                        .find(|&&(b, on)| clear(&self.button_sight[b][c], on))
                        .map(|&(b, _)| Some(b))
                };
                if let Some(stand_on) = stand_on {
                    copy(&mut sc.n);
                    sc.n[BLUE + me] = c as u32 + 1;
                    let a = Action::Fire {
                        color,
                        class: c,
                        stand_on,
                    };
                    done(&mut sc.n, a, quiet && !self.special[c]);
                }
            }
            if let Some(l) = self.through(s, color) {
                if let Some(fizzled) =
                    self.travel(lay, s, on, alive, l.region, l.hazards, &mut sc.n)
                {
                    done(&mut sc.n, Action::Enter { color }, !fizzled);
                }
                if let Some(i) = held {
                    let to = self.cube_lands(lay, s, on, l, i);
                    copy(&mut sc.n);
                    lay.move_cube(&mut sc.n, i, self.loc_code(to));
                    knock(&mut sc.n, l.turret);
                    let quiet = !matches!(to, CubeLoc::OnButton(_));
                    done(&mut sc.n, Action::Throw { color, to }, quiet);
                }
            }
        }

        if let Some(i) = held {
            let kind = lay.kinds[i];
            let spots = std::iter::once(CubeLoc::Loose(r)).chain(
                self.buttons_in[r]
                    .iter()
                    .filter(|&&b| {
                        self.buttons[b].kind.accepts(kind) && !cubes.contains(&self.button_code(b))
                    })
                    .map(|&b| CubeLoc::OnButton(b)),
            );
            for at in spots {
                copy(&mut sc.n);
                lay.move_cube(&mut sc.n, i, self.loc_code(at));
                let quiet = !matches!(at, CubeLoc::OnButton(_));
                done(&mut sc.n, Action::PutDown { at }, quiet);
            }
            for &l in &self.tosses[r] {
                let to = self.cube_lands(lay, s, on, l, i);
                copy(&mut sc.n);
                lay.move_cube(&mut sc.n, i, self.loc_code(to));
                knock(&mut sc.n, l.turret);
                let quiet = !matches!(to, CubeLoc::OnButton(_));
                done(&mut sc.n, Action::Toss { to }, quiet);
            }
        } else {
            // Interchangeable cubes in the same place are one choice.
            let mut last = None;
            for (i, &code) in cubes.iter().enumerate() {
                let from = self.code_loc(code);
                let here = match from {
                    CubeLoc::Loose(cr) => cr == r,
                    CubeLoc::OnButton(b) => self.buttons[b].region == Some(r),
                    CubeLoc::Held => false,
                };
                let id = (lay.kinds[i], lay.homes[i], code);
                if !here || last == Some(id) {
                    continue;
                }
                last = Some(id);
                copy(&mut sc.n);
                lay.move_cube(&mut sc.n, i, HELD);
                let a = Action::PickUp {
                    cube: lay.kinds[i],
                    from,
                };
                done(&mut sc.n, a, !matches!(from, CubeLoc::OnButton(_)));
            }
        }

        for &p in &self.pedestals_in[r] {
            if !self.pedestal_enabled[p] {
                continue;
            }
            let ped = &self.pedestals[p];
            match ped.timer {
                None if s[LATCHED] & (1 << p) == 0 => {
                    copy(&mut sc.n);
                    sc.n[LATCHED] |= 1 << p;
                    done(&mut sc.n, Action::Press { pedestal: p }, false);
                }
                None => {}
                Some(t) => self.dash(cx, s, p, ped.cell, t, sc, &mut done),
            }
        }

        for &t in &self.knock_from[r] {
            if alive & (1 << t) != 0 {
                copy(&mut sc.n);
                sc.n[DOWN] |= 1 << t;
                done(&mut sc.n, Action::Knock { turret: t }, true);
            }
        }
    }

    /// Press timed pedestal `p` at `cell` and walk for at most `seconds`: every region reachable
    /// in time, and escaping if the exit door's cell is reached while it's open.
    #[allow(clippy::too_many_arguments)]
    fn dash(
        &self,
        cx: &Ctx,
        s: &[u32],
        p: usize,
        cell: Cell,
        seconds: f64,
        sc: &mut Scratch,
        done: &mut impl FnMut(&mut Vec<u32>, Action, bool),
    ) {
        let Some(&start) = self.walk.node.get(&cell) else {
            return;
        };
        let lay = &cx.lay;
        let timed = Some(Signal::Pedestal(p));
        let on = self.hazards_on(lay, s, timed);
        let alive = cx.turrets & !s[DOWN];
        let step_time = VOXEL / self.rules.walk_speed;
        let fall_time = |h: i32| (2.0 * h as f64 * VOXEL / GRAVITY).sqrt();
        // Dijkstra over (node, fizzled); times as integer microseconds for the heap.
        let key = |node: usize, fizzled: bool| 2 * node + fizzled as usize;
        let best = &mut sc.dist;
        best.clear();
        best.resize(2 * self.walk.region.len(), u64::MAX);
        best[key(start, false)] = 0;
        let heap = &mut sc.heap;
        heap.clear();
        heap.push(Reverse((0u64, start, false)));
        let limit = (seconds * 1e6) as u64;
        while let Some(Reverse((t, node, fizzled))) = heap.pop() {
            if best[key(node, fizzled)] < t {
                continue;
            }
            for &(to, fall, hazards) in &self.walk.moves[node] {
                if hazards & on & self.lasers != 0 {
                    continue;
                }
                let region = self.walk.region[to];
                if !self.rules.run_past_turrets && self.graph.regions[region].danger & alive != 0 {
                    continue;
                }
                let f = fizzled || hazards & on & self.fizzlers != 0;
                let nt = t + ((step_time + fall_time(fall)) * 1e6) as u64;
                if nt <= limit && nt < best[key(to, f)] {
                    best[key(to, f)] = nt;
                    heap.push(Reverse((nt, to, f)));
                }
            }
        }

        let reached =
            |node: usize| best[key(node, false)] < u64::MAX || best[key(node, true)] < u64::MAX;
        let exit = self.walk.node.get(&self.exit_cell).copied();
        if self.exit_open_with(lay, s, timed) && exit.is_some_and(reached) {
            sc.n.clear();
            sc.n.extend_from_slice(s);
            sc.n[0] = self.exit_region as u32 | ESCAPED;
            let a = Action::Dash {
                pedestal: p,
                to: self.exit_region,
                escaped: true,
            };
            done(&mut sc.n, a, false);
        }
        sc.ends.clear();
        for node in 0..self.walk.region.len() {
            for fizzled in [false, true] {
                if best[key(node, fizzled)] < u64::MAX {
                    sc.ends.push((self.walk.region[node], fizzled));
                }
            }
        }
        sc.ends.sort();
        sc.ends.dedup();
        for &(region, fizzled) in &sc.ends {
            sc.n.clear();
            sc.n.extend_from_slice(s);
            sc.n[0] = region as u32;
            if fizzled {
                lay.fizzle(&mut sc.n);
            }
            if sc.n[..] != *s {
                let a = Action::Dash {
                    pedestal: p,
                    to: region,
                    escaped: false,
                };
                done(&mut sc.n, a, false);
            }
        }
    }

    fn loc_name(&self, l: CubeLoc) -> String {
        match l {
            CubeLoc::Held => "hand".into(),
            CubeLoc::Loose(r) => format!("region {}", RegionGraph::label(r)),
            CubeLoc::OnButton(b) => format!("button {}", self.buttons[b].index),
        }
    }

    fn fields(&self, mask: u64) -> String {
        let names: Vec<String> = (0..self.hazards.len())
            .filter(|h| mask & (1 << h) != 0)
            .map(|h| {
                let hz = &self.hazards[h];
                let kind = match hz.kind {
                    HazardKind::Fizzler => "fizzler",
                    HazardKind::LaserField => "laser field",
                };
                format!("{kind} {}", hz.index)
            })
            .collect();
        if names.is_empty() {
            String::new()
        } else {
            format!(" through {}", names.join(", "))
        }
    }

    pub fn describe(&self, a: &Action) -> String {
        let label = RegionGraph::label;
        match *a {
            Action::Move { to, kind, hazards } => {
                let what = match kind {
                    Move::Walk => format!("walk to region {}", label(to)),
                    Move::Drop { height } => format!("drop {height} to region {}", label(to)),
                    Move::Step => format!("climb onto region {} using a cube", label(to)),
                    Move::Bounce => format!("bounce off repulsion gel up to region {}", label(to)),
                    Move::Leap => format!("leap off propulsion gel across to region {}", label(to)),
                };
                format!("{what}{}", self.fields(hazards))
            }
            Action::Fire {
                color,
                class,
                stand_on,
            } => format!(
                "{}fire {} at class {class}: {}{}",
                stand_on.map_or(String::new(), |b| format!(
                    "stand on button {} and ",
                    self.buttons[b].index
                )),
                color.name(),
                portals::describe(&self.portals.classes[class]),
                if self.portals.classes[class].needs_paint.is_some() {
                    " (conversion gel)"
                } else {
                    ""
                }
            ),
            Action::Enter { color } => format!("walk into the {} portal", color.name()),
            Action::PickUp { cube, from } => {
                format!("pick up the {} in {}", cube_name(cube), self.loc_name(from))
            }
            Action::PutDown { at } => format!("put the cube down in {}", self.loc_name(at)),
            Action::Toss { to } => format!("drop the cube off a ledge into {}", self.loc_name(to)),
            Action::Throw { color, to } => format!(
                "throw the cube into the {} portal; it lands in {}",
                color.name(),
                self.loc_name(to)
            ),
            Action::Press { pedestal } => {
                format!("press pedestal button {}", self.pedestals[pedestal].index)
            }
            Action::Dash {
                pedestal,
                to,
                escaped,
            } => {
                let p = &self.pedestals[pedestal];
                let t = p.timer.unwrap_or(0.0);
                if escaped {
                    format!(
                        "press pedestal button {} and run out of the exit within {t} s",
                        p.index
                    )
                } else {
                    format!(
                        "press pedestal button {} and run to region {} within {t} s",
                        p.index,
                        label(to)
                    )
                }
            }
            Action::Knock { turret } => {
                format!("knock over turret {}", self.turrets[turret].index)
            }
        }
    }
}

fn paint_code(face: usize, kind: PaintKind) -> u32 {
    (face as u32) << 2 | kind as u32
}

fn knock(s: &mut [u32], turret: Option<usize>) {
    if let Some(t) = turret {
        s[DOWN] |= 1 << t;
    }
}

// Words of a packed state; see `Layout`.
const BLUE: usize = 1;
const LATCHED: usize = 3;
const DOWN: usize = 4;
const CUBES: usize = 5;
/// Word 0 holds the region, and this bit if the player escaped during a timed dash.
const ESCAPED: u32 = 1 << 31;
/// Cube location codes: `HELD`, `1 + region` loose, `1 + regions + button` on a button, so
/// they sort like `CubeLoc`.
const HELD: u32 = 0;

/// The portal of color `c` (0 blue, 1 orange) in a packed state.
fn portal(s: &[u32], c: usize) -> Option<usize> {
    s[BLUE + c].checked_sub(1).map(|v| v as usize)
}

/// How a search packs states into words: region (and `ESCAPED`), blue and orange portal
/// classes + 1 (0 = none), latched pedestals, knocked-over turrets, then each cube's location
/// code, then 2 bits of paint per face (`PaintKind` + 1, 0 = none). Cubes keep the start's
/// sorted order; interchangeable ones (same kind and home) stay sorted among themselves, so
/// cubes in permuted places are one state, as in `State`.
struct Layout {
    kinds: Vec<CubeKind>,
    /// Location codes of the cubes' homes.
    homes: Vec<u32>,
    /// Per cube: the cubes `lo..hi` interchangeable with it.
    group: Vec<(usize, usize)>,
    /// Word where paint starts.
    paint: usize,
    /// Words per state.
    len: usize,
}

impl Layout {
    fn cubes<'a>(&self, s: &'a [u32]) -> &'a [u32] {
        &s[CUBES..self.paint]
    }

    fn paint_at(&self, s: &[u32], face: usize) -> u32 {
        (s[self.paint + face / 16] >> (face % 16 * 2)) & 3
    }

    fn set_paint(&self, s: &mut [u32], face: usize, v: u32) {
        let w = &mut s[self.paint + face / 16];
        let shift = face % 16 * 2;
        *w = (*w & !(3 << shift)) | (v << shift);
    }

    /// Paints `spot` with `kind` (water washes it).
    fn paint(&self, s: &mut [u32], spot: &[usize], kind: PaintKind) {
        for &f in spot {
            self.set_paint(s, f, paint_value(kind));
        }
    }

    /// Paints the faces of a spot given as `spot_masks`.
    fn paint_masks(&self, s: &mut [u32], masks: &[(usize, u32)], kind: PaintKind) {
        let v = paint_value(kind);
        for &(w, m) in masks {
            let w = &mut s[self.paint + w];
            *w = (*w & !m) | ((m & 0x5555_5555) * v);
        }
    }

    fn move_cube(&self, s: &mut [u32], cube: usize, code: u32) {
        s[CUBES + cube] = code;
        let (lo, hi) = self.group[cube];
        s[CUBES + lo..CUBES + hi].sort_unstable();
    }

    /// A fizzler: portals gone, held cube back to its home.
    fn fizzle(&self, s: &mut [u32]) {
        s[BLUE] = 0;
        s[BLUE + 1] = 0;
        let mut moved = false;
        for i in 0..self.kinds.len() {
            if s[CUBES + i] == HELD {
                s[CUBES + i] = self.homes[i];
                moved = true;
            }
        }
        if moved {
            for &(lo, hi) in self.group.iter().filter(|g| g.1 - g.0 > 1) {
                s[CUBES + lo..CUBES + hi].sort_unstable();
            }
        }
    }
}

/// A face class the player may fire at from a region (`Model::fire_from`).
struct Target {
    class: usize,
    /// It has one face, so it's no use while the other portal is there.
    single: bool,
    needs_paint: Option<usize>,
    /// Some shot at it from the region is clear whatever the fizzlers do.
    open: bool,
}

/// The 2 bits a face painted with `kind` has in a packed state: 0 for water, which washes it.
fn paint_value(kind: PaintKind) -> u32 {
    if kind == PaintKind::Water {
        0
    } else {
        kind as u32 + 1
    }
}

/// The faces of a spot as (word after the cubes, mask of their 2-bit fields), for painting a
/// packed state a word at a time.
fn spot_masks(spot: &[usize]) -> Vec<(usize, u32)> {
    let mut masks: Vec<(usize, u32)> = Vec::new();
    for &f in spot {
        let bits = 3 << (f % 16 * 2);
        match masks.iter_mut().find(|(w, _)| *w == f / 16) {
            Some((_, m)) => *m |= bits,
            None => masks.push((f / 16, bits)),
        }
    }
    masks
}

/// What a search needs besides the model: how it packs states, and the turrets that exist.
struct Ctx {
    lay: Layout,
    turrets: u32,
}

/// Buffers `Model::expand` reuses from one state to the next.
#[derive(Default)]
struct Scratch {
    /// The successor being built.
    n: Vec<u32>,
    /// Weighted buttons the player can stand on, and the hazards on meanwhile.
    foot: Vec<(usize, u64)>,
    /// Timed dashes: best times, the queue, and the (region, fizzled) pairs reached.
    dist: Vec<u64>,
    heap: BinaryHeap<Reverse<(u64, usize, bool)>>,
    ends: Vec<(usize, bool)>,
}

/// The packed states of a search, each stored once, in the order found, with an
/// open-addressing hash index.
struct Table {
    width: usize,
    words: Vec<u32>,
    len: usize,
    /// `hash << 32 | (state index + 1)`, 0 = empty; at most half full.
    slots: Vec<u64>,
}

impl Table {
    fn new(width: usize) -> Table {
        Table {
            width,
            words: Vec::new(),
            len: 0,
            slots: vec![0; 64],
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn get(&self, i: usize) -> &[u32] {
        &self.words[i * self.width..(i + 1) * self.width]
    }

    fn hash(w: &[u32]) -> u32 {
        const K: u64 = 0x517c_c1b7_2722_0a95;
        let mut h = 0u64;
        let mut pairs = w.chunks_exact(2);
        for p in &mut pairs {
            h = (h.rotate_left(5) ^ (p[0] as u64 | (p[1] as u64) << 32)).wrapping_mul(K);
        }
        if let [x] = pairs.remainder() {
            h = (h.rotate_left(5) ^ *x as u64).wrapping_mul(K);
        }
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        (h ^ h >> 33) as u32
    }

    /// The index of state `w` (with hash `h`), if it is here.
    fn find(&self, w: &[u32], h: u32) -> Option<usize> {
        let mask = self.slots.len() - 1;
        let mut i = h as usize & mask;
        loop {
            let e = self.slots[i];
            if e == 0 {
                return None;
            }
            if (e >> 32) as u32 == h {
                let k = (e as u32 - 1) as usize;
                if self.get(k) == w {
                    return Some(k);
                }
            }
            i = (i + 1) & mask;
        }
    }

    /// Adds state `w` (with hash `h`), which isn't here yet, and returns its index.
    fn insert(&mut self, w: &[u32], h: u32) -> usize {
        let k = self.len;
        if 2 * (k + 1) > self.slots.len() {
            let mut slots = vec![0u64; 2 * self.slots.len()];
            let mask = slots.len() - 1;
            for &e in self.slots.iter().filter(|&&e| e != 0) {
                let mut i = (e >> 32) as usize & mask;
                while slots[i] != 0 {
                    i = (i + 1) & mask;
                }
                slots[i] = e;
            }
            self.slots = slots;
        }
        let mask = self.slots.len() - 1;
        let mut i = h as usize & mask;
        while self.slots[i] != 0 {
            i = (i + 1) & mask;
        }
        self.slots[i] = (h as u64) << 32 | (k as u64 + 1);
        self.words.extend_from_slice(w);
        self.len += 1;
        k
    }
}

fn cube_name(k: CubeKind) -> &'static str {
    match k {
        CubeKind::Standard => "cube",
        CubeKind::Companion => "companion cube",
        CubeKind::Reflection => "reflection cube",
        CubeKind::Sphere => "sphere",
        CubeKind::Franken => "frankenturret",
    }
}

fn walk_graph(level: &Level, graph: &RegionGraph) -> WalkGraph {
    let mut w = WalkGraph::default();
    for r in &graph.regions {
        for &c in &r.cells {
            w.node.insert(c, w.region.len());
            w.region.push(r.id);
        }
    }
    w.moves = vec![Vec::new(); w.region.len()];
    for (&c, &i) in &w.node {
        for d in Dir::HORIZONTAL {
            let face = level.face(c, d);
            let n = step(c, d);
            if face.barrier.is_some() || level.solid(n) {
                continue;
            }
            let target = if level.standable(n) {
                Some((n, 0))
            } else {
                level.fall_path(n)
            };
            if let Some((f, fallen)) = target {
                let hazards = face.hazards | fallen;
                if let Some(&j) = w.node.get(&f) {
                    w.moves[i].push((j, n[2] - f[2], hazards));
                }
            }
        }
    }
    w
}
