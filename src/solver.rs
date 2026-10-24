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
//!   cubes can, so a cube can be put down across one (`Action::Toss`).
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
//! - Devices (`dynamics`): stairs, panels, pistons, rails, bridges and funnels are switched by
//!   their wires like fields (start state XOR all inputs on). Their bits and the powered laser
//!   receivers are part of the state, worked out after every action (`settle`); edges and
//!   regions that need a device only exist while it is in that state, and whoever stands in a
//!   region that goes away rides or falls (`Region::fallback`), or dies.
//! - Lasers (`lasers`): catchers and relays are senders, powered while a beam reaches them.
//!   Reflection cubes put down at a spot (cell and facing) redirect beams.
//! - Flings (`flings`): dropping from a ledge into a floor portal (`Action::Dive`) throws the
//!   player out of the other one at speed. Faith plates launch whatever comes onto them.
//! - Cubes placed without a dropper are gone for good when fizzled; a dropper that only drops
//!   when told to (`Auto Drop` off, wired) holds its cube until its inputs are first on.
//!   Whenever a wired dropper's inputs come on (all of them, after not all were), it drops a
//!   new cube and the old one dissolves: its cube goes back to the dropper (item_dropper.vmf).
//! - A model that had to leave things out (`Model::complete`) never calls a level unsolvable.
//!
//! The search works on states packed into a few words (`Layout`), each stored once in a hash
//! table (`Table`); `State` is the readable form `successors` and `initial` hand out.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};

use crate::dynamics::{Bit, Devices};
use crate::gel::{self, GelRules};
use crate::geom::{step, Cell, Dir};
use crate::lasers::{Beams, Event};
use crate::level::{
    ButtonKind, CubeKind, HazardKind, ItemKind, Level, PaintKind, PanelKind,
    ProjectorKind, MAX_HAZARDS,
};
use crate::portals::{self, resolve, Extras, Face, Landing, Landings, Masks, PanelFace, Portals, Sight};
use crate::regions::{Cond, Geo, Move, RegionGraph};
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
    "ITEM_INDICATOR_TOGGLE",
    "ITEM_INDICATOR_PANEL",
    "ITEM_INDICATOR_PANEL_TIMER",
    "ITEM_POINT_LIGHT",
    "ITEM_PLACEMENT_HELPER",
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

    /// Measured in-game 2026-10-18: a splat paints its own cell (the middle of the cell
    /// beside it was bare: a bounce and a portal shot there failed); about one cell of
    /// propulsion gel gets the player to 400 u/s, and a jump from it carries ~290 units at
    /// ledge height (two cells of gap, not three).
    fn gel(mode: Mode) -> GelRules {
        match mode {
            Mode::Conservative => GelRules {
                radius: 0,
                bounce_height: 2,
                bounce_reach: 1,
                leap_run: 1,
                leap_gap: 2,
            },
            Mode::Liberal => GelRules {
                radius: 1,
                bounce_height: 4,
                bounce_reach: 2,
                leap_run: 1,
                leap_gap: 3,
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
    /// Set down at a laser spot (index into `Beams::spots`), facing its way.
    Spot(usize),
    /// Destroyed for good (a loose cube fizzled), or not dropped yet.
    Gone,
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
    /// Drop the held cube off a ledge of the current region, or put it down across a field.
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
    /// Drop `height` voxels off a ledge into the floor portal of this color, and fly out of
    /// the other one to region `to`.
    Dive {
        color: Color,
        height: i32,
        to: usize,
    },
}

impl Action {
    /// The same action with the portal colors the other way round.
    fn flipped(self) -> Action {
        match self {
            Action::Fire {
                color,
                class,
                stand_on,
            } => Action::Fire {
                color: color.other(),
                class,
                stand_on,
            },
            Action::Enter { color } => Action::Enter {
                color: color.other(),
            },
            Action::Throw { color, to } => Action::Throw {
                color: color.other(),
                to,
            },
            Action::Dive { color, height, to } => Action::Dive {
                color: color.other(),
                height,
                to,
            },
            a => a,
        }
    }
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
    /// Bit `i`: the inputs of wired dropper `i` (`Model::redrops`) were on when the state
    /// last settled. Follows from the rest once settled.
    pub redrop: u32,
    /// Painted faces, sorted, each `face << 2 | PaintKind` (face: index into
    /// `Model::paint_faces`).
    pub paint: Vec<u32>,
    /// Bit `k`: laser receiver `k` (`Beams::receivers`) is powered. Follows from the rest.
    pub lit: u32,
    /// The devices' bits (`Devices::bits`). Follows from the rest.
    pub dev: u64,
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

/// Something that drives a receiver: a floor button, a pedestal or a powered laser catcher
/// or relay (indices into `Model::buttons`, `Model::pedestals` and `Beams::receivers`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Signal {
    Floor(usize),
    Pedestal(usize),
    Laser(usize),
}

/// Something wired that is on or off: on when `start` XOR all its inputs are on (`start`
/// alone without inputs).
#[derive(Clone, Debug, Default)]
pub struct Switch {
    pub item: i32,
    pub start: bool,
    pub inputs: Vec<Signal>,
}

/// How a device bit follows from the state.
#[derive(Clone, Copy, Debug)]
enum BitRule {
    /// The switch (index into `Model::switches`) is on.
    Switch(usize),
    /// The projector's switch is on, a portal is on a class its path ends at, and the other
    /// on a class with this route.
    Routed {
        switch: usize,
        projector: usize,
        route: usize,
    },
}

/// A cube that waits in its dropper until the dropper's inputs are first on, or (`redrops`)
/// a wired dropper's cube.
#[derive(Clone, Debug)]
struct Held {
    kind: CubeKind,
    home: CubeLoc,
    inputs: Vec<Signal>,
    /// The turret (index into `Level::turrets`) it lands on, knocking it over.
    turret: Option<usize>,
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
    /// (to, fall height in voxels, hazards crossed, devices needed). Every move walks one
    /// voxel first.
    moves: Vec<Vec<(usize, i32, u64, Cond)>>,
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
    /// How states are packed. It depends only on the model's cubes, so it is built once in
    /// `new`; `set_start_cubes` rebuilds it when the metrics ablate one.
    layout: Layout,
    pub exit_region: usize,
    pub exit_cell: Cell,
    exit_inputs: Vec<Signal>,
    exit_inverted: bool,
    /// Per region: where a cube dropped off each of its ledges lands.
    tosses: Vec<Vec<Landings>>,
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
    /// Per face class: a portal there does nothing on its own. Nothing can go in (no entry,
    /// no dives), and it drives nothing (no turret, gel, projection or beam, not a panel):
    /// it is only somewhere to come out of. Two of them make a pair no one can use.
    inert: Vec<bool>,
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
    /// Per panel that can stop shots: its device bit and its bit in shot masks
    /// (`Sight::panel_shift`).
    panel_shots: Vec<(usize, u64)>,
    /// Devices, their bits, and how each bit follows from the state.
    pub devices: Devices,
    /// Per switch, the device bits that are on only while it is (`BitRule`).
    switch_bits: Vec<u64>,
    /// The bits of `BitRule::Routed`, on only while the portals carry their route too.
    routed: u64,
    /// Per face class, the routed bits whose projector's path ends there (`FaceClass::ends`)
    /// and those whose route comes out of it (`FaceClass::routes`).
    class_ends: Vec<u64>,
    class_routes: Vec<u64>,
    /// Wired things that are on or off: devices' and laser emitters'.
    pub switches: Vec<Switch>,
    /// Laser beams; per emitter (`Beams::emitters`) its switch.
    pub beams: Beams,
    emitter_switch: Vec<usize>,
    /// Per region, the laser spots (`Beams::spots`) in it.
    spots_in: Vec<Vec<usize>>,
    /// Per beam end face (a class of its own), its class.
    beam_end_class: HashMap<Face, usize>,
    /// Per panel (`Level::panels`), the class of its face, if it takes portals.
    panel_class: Vec<Option<usize>>,
    /// Cubes waiting in droppers.
    held_back: Vec<Held>,
    /// Cubes of wired droppers, which drop a new one each time their inputs come on.
    redrops: Vec<Held>,
    /// Devices, lasers or waiting cubes: `settle` has work after any action.
    dynamic: bool,
    /// Where the portals are can switch devices (a bridge or funnel through them) or carry a
    /// laser.
    portals_drive: bool,
    /// Everything in the level was modelled: an exhausted search means unsolvable.
    pub complete: bool,
    projector_kinds: Vec<ProjectorKind>,
    /// Per region: standing there is standing on a moving platform (a piston, or a rail's
    /// track where there's no floor of its own).
    pub on_platform: Vec<bool>,
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
    /// How each state but the start was first reached: (state, action), `SWAPPED` set if
    /// the portals were swapped into `canonical` order on the way.
    parent: Vec<(u32, Action)>,
    /// Each state's region (`u32::MAX` once escaped).
    regions: Vec<u32>,
}

impl StateSpace {
    /// Number of states.
    pub fn len(&self) -> usize {
        self.goal.len()
    }

    pub fn is_empty(&self) -> bool {
        self.goal.is_empty()
    }

    /// The player's region in state `i` (`None` once escaped).
    pub fn region(&self, i: usize) -> Option<usize> {
        let r = self.regions[i];
        (r != u32::MAX).then_some(r as usize)
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
                Verdict::Solved(plan_to(&self.parent, g))
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

    /// Whether a goal can be reached without firing a portal at all: on a complete space,
    /// whether the level is solvable with portals taken away.
    pub fn solvable_without_portals(&self) -> bool {
        self.reaches_goal_unless(|u| u & (3 << 30) == USES_CLASS)
    }

    /// Whether a goal is reachable from the start along actions that don't use `skip`.
    fn reaches_goal(&self, skip: u32) -> bool {
        self.reaches_goal_unless(|u| u == skip)
    }

    fn reaches_goal_unless(&self, skip: impl Fn(u32) -> bool) -> bool {
        let mut seen = vec![false; self.len()];
        let mut stack = vec![0];
        seen[0] = true;
        while let Some(i) = stack.pop() {
            if self.goal[i] {
                return true;
            }
            for e in self.offsets[i] as usize..self.offsets[i + 1] as usize {
                let j = self.targets[e] as usize;
                if !skip(self.uses[e]) && !std::mem::replace(&mut seen[j], true) {
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

/// Solves `level` in both modes, with angled laser spots if the plain model doesn't solve it
/// (as `dataset::label` does).
pub fn judge(level: &Level, budget: usize) -> Result<Judged, Vec<Problem>> {
    let mut conservative = Model::new(level, Mode::Conservative)?;
    let mut verdict = conservative.solve(budget).verdict;
    if !matches!(verdict, Verdict::Solved(_)) && conservative.could_angle() {
        conservative = Model::new_angled(level, Mode::Conservative)?;
        verdict = conservative.solve(budget).verdict;
    }
    let liberal = if conservative.beams.angled {
        Model::new_angled(level, Mode::Liberal)?
    } else {
        Model::new(level, Mode::Liberal)?
    };
    let judgement = match verdict {
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
        Model::build(level, mode, false)
    }

    /// `new`, with angled spots for reflection cubes (`lasers`): a bigger search, for levels
    /// the plain model doesn't solve.
    pub fn new_angled(level: &Level, mode: Mode) -> Result<Model, Vec<Problem>> {
        Model::build(level, mode, true)
    }

    /// Whether `new_angled` would differ: the level has a reflection cube and an emitter, and
    /// this model has no angled spots.
    pub fn could_angle(&self) -> bool {
        self.beams.could_angle && !self.beams.angled
    }

    fn build(level: &Level, mode: Mode, angled: bool) -> Result<Model, Vec<Problem>> {
        let mut problems: Vec<Problem> = level
            .decode_problems
            .iter()
            .map(|p| Problem::Invalid(p.clone()))
            .collect();
        let Some(mut devices) = Devices::build(level) else {
            problems.push(Problem::Unsupported(format!(
                "more than {} device bits",
                crate::dynamics::MAX_BITS
            )));
            return Err(problems);
        };
        let dyn_ = devices.dyn_.clone();
        let geo = Geo::new(level, &dyn_);
        // Regions split where the turrets that can shoot change.
        let turret_sight = Rules::turret_sight(mode);
        let danger: HashMap<Cell, u32> = if level.turrets.is_empty() {
            HashMap::new()
        } else {
            level
                .cells()
                .filter(|&c| geo.stand(c).is_some())
                .map(|c| (c, turrets::danger(level, turret_sight, c)))
                .collect()
        };
        let mut graph = RegionGraph::build_dyn(geo, |c| danger.get(&c).copied().unwrap_or(0));
        let route_moves = devices.funnel_route_moves(geo, |c| graph.region_of(c));
        graph.add_moves(geo, &route_moves);

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

        // Laser beams: emitters, receivers, spots for reflection cubes.
        let reflectors = level
            .of_kind(&ItemKind::Cube)
            .filter(|c| c.cube_kind() == CubeKind::Reflection)
            .count();
        let beams = Beams::build(level, reflectors, angled);

        // Switches: one per device bit that follows a wire, and per laser emitter.
        let mut switches: Vec<Switch> = Vec::new();
        let mut switch_of: HashMap<i32, usize> = HashMap::new();
        let mut polarity_of: HashMap<i32, usize> = HashMap::new();
        let new_switch = |switches: &mut Vec<Switch>, item: i32, start: bool| {
            switches.push(Switch {
                item,
                start,
                inputs: Vec::new(),
            });
            switches.len() - 1
        };
        let mut bit_rules = vec![BitRule::Switch(0); devices.bits.len()];
        for (i, st) in level.stairs.iter().enumerate() {
            let sw = new_switch(&mut switches, st.item, st.start_deployed);
            switch_of.insert(st.item, sw);
            bit_rules[devices.stairs_bit[i]] = BitRule::Switch(sw);
        }
        for (i, pn) in level.panels.iter().enumerate() {
            let sw = new_switch(&mut switches, pn.item, pn.start_deployed);
            switch_of.insert(pn.item, sw);
            bit_rules[devices.panel_bit[i]] = BitRule::Switch(sw);
        }
        for (i, ps) in level.pistons.iter().enumerate() {
            let sw = new_switch(&mut switches, ps.item, ps.start_up);
            switch_of.insert(ps.item, sw);
            bit_rules[devices.piston_bit[i]] = BitRule::Switch(sw);
        }
        for (i, rl) in level.rails.iter().enumerate() {
            let sw = new_switch(&mut switches, rl.item, rl.start_active);
            switch_of.insert(rl.item, sw);
            if let Some(b) = devices.rail_bit[i] {
                bit_rules[b] = BitRule::Switch(sw);
            }
        }
        let mut projector_switch = Vec::new();
        for (i, pj) in level.projectors.iter().enumerate() {
            let sw = new_switch(&mut switches, pj.item, pj.start_enabled);
            switch_of.insert(pj.item, sw);
            projector_switch.push(sw);
            bit_rules[devices.projector_bit[i]] = BitRule::Switch(sw);
            if let Some(b) = devices.reversed_bit[i] {
                let rs = new_switch(&mut switches, pj.item, pj.start_reversed);
                polarity_of.insert(pj.item, rs);
                bit_rules[b] = BitRule::Switch(rs);
            }
        }
        for (b, bit) in devices.bits.iter().enumerate() {
            if let Bit::Routed { projector, route } = *bit {
                bit_rules[b] = BitRule::Routed {
                    switch: projector_switch[projector],
                    projector,
                    route,
                };
            }
        }
        let mut emitter_switch = Vec::new();
        for &(item, _) in &beams.emitters {
            let start = level
                .lasers
                .iter()
                .find(|l| l.item == item)
                .is_some_and(|l| l.start_enabled);
            let sw = new_switch(&mut switches, item, start);
            switch_of.insert(item, sw);
            emitter_switch.push(sw);
        }
        // Droppers that only drop when told to, and their inputs.
        let mut waiting: HashMap<i32, Vec<Signal>> = level
            .of_kind(&ItemKind::CubeDropper)
            .filter(|d| d.props.get_i32("ITEM_PROPERTY_AUTO_DROP_CUBE") == Some(0))
            .map(|d| (d.index, Vec::new()))
            .collect();
        let mut dropper_inputs: HashMap<i32, Vec<Signal>> = HashMap::new();

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
                Some(ItemKind::LaserCatcher | ItemKind::LaserRelay) => beams
                    .receivers
                    .iter()
                    .position(|r| r.0 == link.sender)
                    .map(Signal::Laser),
                _ => None,
            };
            let Some(signal) = signal else {
                problems.push(Problem::Unsupported(format!(
                    "wire {} -> {} comes from something other than a button or laser receiver",
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
                Some(_) if switch_of.contains_key(&link.receiver) => {
                    switches[switch_of[&link.receiver]].inputs.push(signal);
                }
                Some(ItemKind::CubeDropper) => {
                    if let Some(w) = waiting.get_mut(&link.receiver) {
                        w.push(signal);
                    }
                    dropper_inputs.entry(link.receiver).or_default().push(signal);
                }
                _ => problems.push(Problem::Unsupported(format!(
                    "wire {} -> {} drives something that takes no input",
                    link.sender, link.receiver
                ))),
            }
        }
        for link in &level.polarity_links {
            let signal = buttons
                .iter()
                .position(|b| b.index == link.sender)
                .map(Signal::Floor)
                .or_else(|| {
                    pedestals
                        .iter()
                        .position(|p| p.index == link.sender)
                        .map(Signal::Pedestal)
                })
                .or_else(|| {
                    beams
                        .receivers
                        .iter()
                        .position(|r| r.0 == link.sender)
                        .map(Signal::Laser)
                });
            match (signal, polarity_of.get(&link.receiver)) {
                (Some(sig), Some(&sw)) => switches[sw].inputs.push(sig),
                _ => problems.push(Problem::Unsupported(format!(
                    "polarity wire {} -> {} isn't from a sender to a funnel",
                    link.sender, link.receiver
                ))),
            }
        }

        // A piston wired to a button starts the other way round from its START_UP (in-game,
        // piston_bridge: START_UP 0 stood up until its pedestal was pressed, then went down).
        for ps in &level.pistons {
            let w = &mut switches[switch_of[&ps.item]];
            if !w.inputs.is_empty() {
                w.start = !w.start;
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
        let mut held_back = Vec::new();
        let mut redrops = Vec::new();
        // Turrets a cube dropped at the start lands on, knocked over from the start.
        let mut start_down = 0u32;
        for cube in level.of_kind(&ItemKind::Cube) {
            // A cube that would fall into goo is simply gone.
            if let Some(l) = Landing::from_fall(level, &graph, cube.cell) {
                let loc = landing_loc(l, cube.cube_kind());
                let home = if level.loose_cube(cube.index) {
                    CubeLoc::Gone
                } else {
                    loc
                };
                let wait = level
                    .dropper_of
                    .get(&cube.index)
                    .and_then(|d| waiting.get(d))
                    .filter(|inputs| !inputs.is_empty());
                if let Some(inputs) = level.dropper_of.get(&cube.index).and_then(|d| dropper_inputs.get(d)) {
                    if home != CubeLoc::Gone {
                        redrops.push(Held {
                            kind: cube.cube_kind(),
                            home,
                            inputs: inputs.clone(),
                            turret: l.turret,
                        });
                    }
                }
                let loc = match wait {
                    Some(inputs) => {
                        held_back.push(Held {
                            kind: cube.cube_kind(),
                            home,
                            inputs: inputs.clone(),
                            turret: l.turret,
                        });
                        CubeLoc::Gone
                    }
                    None => {
                        if let Some(t) = l.turret.filter(|&t| t < 32) {
                            start_down |= 1 << t;
                        }
                        loc
                    }
                };
                cubes.push(Cube {
                    kind: cube.cube_kind(),
                    home,
                    loc,
                });
            }
        }
        cubes.sort();
        if redrops.len() > 32 {
            problems.push(Problem::Unsupported(format!(
                "{} wired cube droppers (at most 32 re-drop)",
                redrops.len()
            )));
            redrops.truncate(32);
        }

        let tosses = graph
            .regions
            .iter()
            .map(|r| {
                let mut landings: Vec<Landings> = r
                    .cells
                    .iter()
                    .flat_map(|&c| Dir::HORIZONTAL.map(|d| (c, d)))
                    .filter(|&(c, d)| level.face(c, d).barrier.is_none())
                    .filter_map(|(c, d)| {
                        let n = step(c, d);
                        // Across a field, into a floor of another region: through it.
                        let across = level.face(c, d).hazards != 0
                            && graph.region_of(n).is_some_and(|o| o != r.id);
                        if level.solid(n) || (geo.stand(n) == Some(Cond::ALWAYS) && !across) {
                            return None;
                        }
                        let mut ls = Landing::falls(geo, &graph, n);
                        for (_, l) in &mut ls {
                            l.hazards |= level.face(c, d).hazards;
                        }
                        (!ls.is_empty()).then_some(ls)
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
        // Deployed 90° panels stop shots too: their mask bits come after the hazards'.
        let panel_shift = Some(level.hazards.len() as u32);
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
                    panel_shift,
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
                    panel_shift,
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
        let portal_redirectable = redirectable
            .iter()
            .any(|&g| gels[g].kind == PaintKind::Portal);
        // Where gel sprayed out of a wall portal can land on a turret: it leaves the face
        // into the room and falls, so liberal rules let it reach a couple of cells in.
        const WALL_GEL_REACH: i32 = 2;
        fn turret_wall_exits(level: &Level, (c, d): Face) -> Vec<Cell> {
            let mut out = Vec::new();
            let mut at = c;
            for _ in 0..=WALL_GEL_REACH {
                if level.solid(at) {
                    break;
                }
                if let Some(land) = level.fall(at) {
                    if level.turret_at(land).is_some() && !out.contains(&land) {
                        out.push(land);
                    }
                }
                at = step(at, d.opposite());
            }
            out
        }
        let mut cell_spots: HashMap<Cell, Vec<usize>> = HashMap::new();
        let mut exit_cells: HashMap<Face, Vec<Cell>> = HashMap::new();
        if !redirectable.is_empty() {
            // Ceilings first: a cell a ceiling portal's gel falls on gets its whole spot, even
            // if a wall exit reaches it too.
            let mut surfaces: Vec<Face> = level.portal_surfaces().into_iter().collect();
            surfaces.sort_by_key(|f| f.1 != Dir::PosZ);
            for f in surfaces {
                let cells = match f.1 {
                    Dir::PosZ => level.fall(f.0).into_iter().collect(),
                    d if d.axis() != 2 && mode == Mode::Liberal && portal_redirectable => {
                        turret_wall_exits(level, f)
                    }
                    _ => Vec::new(),
                };
                if cells.is_empty() {
                    continue;
                }
                for &land in &cells {
                    if let std::collections::hash_map::Entry::Vacant(slot) =
                        cell_spots.entry(land)
                    {
                        let faces = if f.1 == Dir::PosZ {
                            floor_spot(land)
                        } else {
                            // Liberal wall exits: model only the floor under a turret (so a
                            // portal there can drop it). Painting other floors would add paint
                            // state without knocking anything.
                            vec![(land, Dir::NegZ)]
                        };
                        slot.insert(intern(faces, &mut paint_faces, &mut face_id));
                    }
                }
                exit_cells.insert(f, cells);
            }
        }
        // Faces conversion gel can paint: where portal gel lands, and anywhere a stream of it
        // can be sent.
        let mut whitenable: HashSet<usize> = portal_gel;
        if portal_redirectable {
            whitenable.extend(cell_spots.values().flatten().copied());
        }
        let mut paintable: Vec<(Face, usize)> = whitenable
            .into_iter()
            .map(|f| (paint_faces[f], f))
            .filter(|&((c, d), _)| !level.portal_surface(c, d))
            .collect();
        paintable.sort();
        // Jumping off a ledge into a floor portal: up to two cells out (three, liberal).
        let reach = if mode == Mode::Liberal { 3 } else { 2 };
        let dives = portals::dives(level, &graph, reach);
        let mut fling_heights: Vec<i32> = dives.values().flatten().map(|&(_, h)| h).collect();
        fling_heights.sort();
        fling_heights.dedup();
        let mut flips = Vec::new();
        let mut panel_faces = HashMap::new();
        for (i, pn) in level.panels.iter().enumerate() {
            let face = (pn.cell, pn.normal.opposite());
            let bit = devices.panel_bit[i];
            match pn.kind {
                PanelKind::Flip => {
                    let white = level.portal_surface(face.0, face.1);
                    flips.push((face, if white { Cond::off(bit) } else { Cond::on(bit) }));
                }
                PanelKind::Angled => {
                    panel_faces.insert(
                        face,
                        PanelFace {
                            bit,
                            hinge: pn.hinge(),
                            angle: pn.angle,
                        },
                    );
                }
                PanelKind::Glass => {}
            }
        }
        let extras = Extras {
            streams: redirectable
                .iter()
                .map(|&g| (stream_faces[g].unwrap(), g))
                .collect(),
            split_gel_exits: !redirectable.is_empty(),
            gel_cells: exit_cells,
            paintable,
            flips,
            panels: panel_faces,
            ends: devices.path_ends(),
            routes: devices.route_faces(),
            beam_ends: beams.ends.clone(),
            beam_exits: beams
                .exits
                .iter()
                .map(|(&f, &r)| (f, (r, beams.rays[r].events.clone())))
                .collect(),
            dives,
            fling_heights,
            liberal: mode == Mode::Liberal,
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
            walk_graph(geo, &graph)
        } else {
            WalkGraph::default()
        };
        let portals = Portals::build(geo, &graph, rules.sight, &extras);
        // (device bit, shot mask bit) of each panel that stops shots while deployed.
        let panel_shots: Vec<(usize, u64)> = (0..level.panels.len())
            .filter_map(|i| {
                let k = rules.sight.panel_shift? + i as u32;
                Some((devices.panel_bit[i], 1u64.checked_shl(k)?))
            })
            .collect();
        let beam_end_class: HashMap<Face, usize> = portals
            .classes
            .iter()
            .filter(|c| c.beam_end)
            .map(|c| (c.faces[0], c.id))
            .collect();
        let panel_class: Vec<Option<usize>> = level
            .panels
            .iter()
            .map(|p| {
                let face = (p.cell, p.normal.opposite());
                portals.classes.iter().position(|c| c.faces.contains(&face))
            })
            .collect();
        let regions_n = graph.regions.len();
        let mut spots_in = vec![Vec::new(); regions_n];
        for (k, sp) in beams.spots.iter().enumerate() {
            if let Some(r) = graph.region_of(beams.spot_cells[sp.cell]) {
                spots_in[r].push(k);
            }
        }
        let complete = !devices.incomplete && !beams.incomplete;
        let mut on_platform = vec![false; regions_n];
        let platform_cells = level
            .rails
            .iter()
            .flat_map(|r| r.track())
            .chain(
                level
                    .pistons
                    .iter()
                    .filter(|p| p.normal == Dir::PosZ)
                    .flat_map(|p| [p.stand(false), p.stand(true)]),
            );
        for c in platform_cells {
            if let Some(r) = graph.region_of(c) {
                on_platform[r] |= !graph.regions[r].cond.is_always() || !level.standable(c);
            }
        }
        let dynamic = !devices.is_empty()
            || !beams.is_empty()
            || !held_back.is_empty()
            || !redrops.is_empty();
        let mut switch_bits = vec![0u64; switches.len()];
        let mut routed = 0u64;
        let mut class_ends = vec![0u64; portals.classes.len()];
        let mut class_routes = vec![0u64; portals.classes.len()];
        for (b, rule) in bit_rules.iter().enumerate() {
            match *rule {
                BitRule::Switch(sw) => switch_bits[sw] |= 1 << b,
                BitRule::Routed {
                    switch,
                    projector,
                    route,
                } => {
                    switch_bits[switch] |= 1 << b;
                    routed |= 1 << b;
                    for c in &portals.classes {
                        if c.ends.contains(&projector) {
                            class_ends[c.id] |= 1 << b;
                        }
                        if c.routes.contains(&(projector, route)) {
                            class_routes[c.id] |= 1 << b;
                        }
                    }
                }
            }
        }
        let portals_drive = !beams.is_empty() || routed != 0;
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
                if c.gel_cells.is_empty() {
                    return None;
                }
                let mut spot: Vec<usize> = c
                    .gel_cells
                    .iter()
                    .filter_map(|cell| cell_spots.get(cell))
                    .flatten()
                    .copied()
                    .collect();
                spot.sort_unstable();
                spot.dedup();
                (!spot.is_empty()).then(|| spot_masks(&spot))
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
                            || c.deployed_visible.as_ref().is_some_and(|v| !v[r].is_empty())
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
        let mut model = Model {
            edge_range: ranges(&edge_from),
            gel_move_range: ranges(&gel_move_from),
            fire_from,
            special: portals
                .classes
                .iter()
                .map(|c| c.turret.is_some() || c.stream.is_some() || !c.gel_cells.is_empty())
                .collect(),
            inert: portals
                .classes
                .iter()
                .map(|c| {
                    c.flat.entry.is_none()
                        && c.flat.dives.is_empty()
                        && c.deployed.is_none()
                        && c.turret.is_none()
                        && c.stream.is_none()
                        && c.gel_cells.is_empty()
                        && c.ends.is_empty()
                        && c.routes.is_empty()
                        && !c.beam_end
                        && c.beam_out.is_none()
                })
                .collect(),
            buttons_in: by_region(buttons.len(), &|b| buttons[b].region),
            pedestals_in: by_region(pedestals.len(), &|p| pedestals[p].region),
            footholds,
            knock_from,
            fizzlers: kind_mask(HazardKind::Fizzler),
            panel_shots,
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
                down: start_down,
                redrop: 0,
                paint: Vec::new(),
                lit: 0,
                dev: 0,
            },
            layout: Layout::empty(),
            exit_region,
            exit_inputs,
            tosses,
            walk,
            button_sight,
            devices,
            switch_bits,
            routed,
            class_ends,
            class_routes,
            switches,
            beams,
            emitter_switch,
            spots_in,
            beam_end_class,
            panel_class,
            held_back,
            redrops,
            dynamic,
            portals_drive,
            complete,
            projector_kinds: level.projectors.iter().map(|p| p.kind).collect(),
            on_platform,
        };
        model.layout = model.layout_for(&model.start.cubes);
        Ok(model)
    }

    /// The start state: paint from splats without droppers, then whatever the droppers pour.
    pub fn initial(&self) -> State {
        let cx = self.context();
        self.unpack(&cx.lay, &self.initial_packed(&cx))
    }

    pub fn painted(&self, s: &State, face: usize, kind: PaintKind) -> bool {
        s.paint.binary_search(&paint_code(face, kind)).is_ok()
    }

    /// Whether a pouring gel stream goes through a portal in state `s`.
    pub fn redirected(&self, s: &State) -> bool {
        let cx = self.context();
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
        let cx = self.context();
        self.hazards_on(&cx.lay, &self.pack(&cx.lay, s), held)
    }

    /// `shot_stops` in state `s`: a shot with mask `m` is clear when `m & stops == 0`.
    pub fn shot_stops_in(&self, s: &State, held: Option<Signal>) -> u64 {
        let cx = self.context();
        self.shot_stops(&cx.lay, &self.pack(&cx.lay, s), held)
    }

    /// Which device bits are bridges (direct or through a portal), moving platforms (pistons
    /// and rails), and projections through portals.
    pub fn device_kinds(&self) -> DeviceKinds {
        let mut k = DeviceKinds::default();
        for (b, bit) in self.devices.bits.iter().enumerate() {
            match *bit {
                Bit::Projector(p) | Bit::Routed { projector: p, .. }
                    if self.projector_kinds[p] == ProjectorKind::Bridge =>
                {
                    k.bridges |= 1 << b;
                }
                Bit::Piston(_) | Bit::Rail(_) => k.platforms |= 1 << b,
                _ => {}
            }
            if matches!(bit, Bit::Routed { .. }) {
                k.routed |= 1 << b;
            }
        }
        k
    }

    /// The cell of laser spot `k`.
    pub fn spot_cell(&self, k: usize) -> Cell {
        self.beams.spot_cells[self.beams.spots[k].cell]
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
        let cx = self.context();
        self.exit_open_with(&cx.lay, &self.pack(&cx.lay, s), None)
    }

    pub fn is_goal(&self, s: &State) -> bool {
        let cx = self.context();
        self.goal(&cx.lay, &self.pack(&cx.lay, s))
    }

    /// Every action possible in state `s`, with the state it leads to.
    pub fn successors(&self, s: &State) -> Vec<(Action, State)> {
        let cx = self.context();
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
        let cx = self.context();
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
                return Outcome {
                    verdict: Verdict::Solved(plan_to(&parent, next)),
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
                let swapped = canonical(n);
                let h = Table::hash(n);
                if full || table.find(n, h).is_some() {
                    return;
                }
                if table.len() >= max_states {
                    full = true;
                    return;
                }
                table.insert(n, h);
                parent.push((next as u32 | if swapped { SWAPPED } else { 0 }, a));
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
            verdict: if self.complete {
                Verdict::Unsolvable
            } else {
                Verdict::Unknown
            },
            states: table.len(),
        }
    }

    /// Explores every reachable state (up to `max_states`), past goals too: metrics need the
    /// whole graph to find dead ends.
    pub fn explore(&self, max_states: usize) -> StateSpace {
        let cx = self.context();
        let mut table = Table::new(cx.lay.len);
        let start = self.initial_packed(&cx);
        table.insert(&start, Table::hash(&start));
        let mut space = StateSpace {
            depth: vec![0],
            goal: Vec::new(),
            complete: self.complete,
            overflow: None,
            offsets: vec![0],
            targets: Vec::new(),
            uses: Vec::new(),
            parent: Vec::new(),
            regions: Vec::new(),
        };
        let (mut cur, mut sc) = (Vec::new(), Scratch::default());
        let mut i = 0;
        while i < table.len() {
            cur.clear();
            cur.extend_from_slice(table.get(i));
            space.goal.push(self.goal(&cx.lay, &cur));
            space.regions.push(if cur[0] & ESCAPED != 0 { u32::MAX } else { cur[0] });
            self.expand(&cx, &cur, true, &mut sc, &mut |a, n| {
                let swapped = canonical(n);
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
                        space.parent.push((i as u32 | if swapped { SWAPPED } else { 0 }, a));
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

    /// Replace the model's cubes and repack states to match. The metrics remove one cube at a
    /// time to test whether the level still solves.
    pub fn set_start_cubes(&mut self, cubes: Vec<Cube>) {
        self.start.cubes = cubes;
        let layout = self.layout_for(&self.start.cubes);
        self.layout = layout;
    }

    /// How states with these cubes are packed. Only the cubes' kinds and homes matter, and every
    /// reachable state shares them, so this is built once and cached in `self.layout`.
    fn layout_for(&self, cubes: &[Cube]) -> Layout {
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
        let end = paint + self.paint_faces.len().div_ceil(16);
        let redrop = (!self.redrops.is_empty()).then_some(end);
        Layout {
            homes: ids.iter().map(|&(_, home)| self.loc_code(home)).collect(),
            kinds: ids.iter().map(|&(kind, _)| kind).collect(),
            group,
            paint,
            redrop,
            len: end + redrop.is_some() as usize,
        }
    }

    /// The turrets that exist, as a bit per turret. Metrics switch them off, so this is not part
    /// of the cached layout.
    fn turret_mask(&self) -> u32 {
        self.turret_enabled
            .iter()
            .enumerate()
            .filter(|(_, &e)| e)
            .fold(0u32, |m, (i, _)| m | 1 << i)
    }

    /// How states are packed, and the turrets that exist.
    fn context(&self) -> Ctx<'_> {
        Ctx {
            lay: &self.layout,
            turrets: self.turret_mask(),
        }
    }

    fn loc_code(&self, l: CubeLoc) -> u32 {
        match l {
            CubeLoc::Held => HELD,
            CubeLoc::Loose(r) => 1 + r as u32,
            CubeLoc::OnButton(b) => self.button_code(b),
            CubeLoc::Spot(k) => self.spot_code(k),
            CubeLoc::Gone => self.spot_code(self.beams.spots.len()),
        }
    }

    fn button_code(&self, b: usize) -> u32 {
        (1 + self.graph.regions.len() + b) as u32
    }

    fn spot_code(&self, k: usize) -> u32 {
        (1 + self.graph.regions.len() + self.buttons.len() + k) as u32
    }

    fn code_loc(&self, code: u32) -> CubeLoc {
        let regions = self.graph.regions.len();
        let buttons = self.buttons.len();
        let spots = self.beams.spots.len();
        match code as usize {
            0 => CubeLoc::Held,
            c if c <= regions => CubeLoc::Loose(c - 1),
            c if c <= regions + buttons => CubeLoc::OnButton(c - 1 - regions),
            c if c <= regions + buttons + spots => CubeLoc::Spot(c - 1 - regions - buttons),
            _ => CubeLoc::Gone,
        }
    }

    /// The region a cube at `loc` is in, if it is somewhere.
    fn loc_region(&self, loc: CubeLoc) -> Option<usize> {
        match loc {
            CubeLoc::Loose(r) => Some(r),
            CubeLoc::OnButton(b) => self.buttons[b].region,
            CubeLoc::Spot(k) => self.spot_region(k),
            CubeLoc::Held | CubeLoc::Gone => None,
        }
    }

    fn spot_region(&self, k: usize) -> Option<usize> {
        let cell = self.beams.spot_cells[self.beams.spots[k].cell];
        self.graph.region_of(cell)
    }

    fn pack(&self, lay: &Layout, s: &State) -> Vec<u32> {
        let mut w = vec![0; lay.len];
        w[0] = s.region as u32 | if s.escaped { ESCAPED } else { 0 };
        w[BLUE] = s.portals[0].map_or(0, |c| c as u32 + 1);
        w[BLUE + 1] = s.portals[1].map_or(0, |c| c as u32 + 1);
        w[LATCHED] = s.latched;
        w[DOWN] = s.down;
        w[LIT] = s.lit;
        w[DEV] = s.dev as u32;
        w[DEV + 1] = (s.dev >> 32) as u32;
        for (i, c) in s.cubes.iter().enumerate() {
            w[CUBES + i] = self.loc_code(c.loc);
        }
        for &e in &s.paint {
            lay.set_paint(&mut w, e as usize >> 2, (e & 3) + 1);
        }
        if let Some(r) = lay.redrop {
            w[r] = s.redrop;
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
            redrop: lay.redrop.map_or(0, |r| w[r]),
            lit: w[LIT],
            dev: dev_of(w),
            paint: (0..self.paint_faces.len())
                .filter_map(|f| match lay.paint_at(w, f) {
                    0 => None,
                    v => Some(((f as u32) << 2) | (v - 1)),
                })
                .collect(),
        }
    }

    fn initial_packed(&self, cx: &Ctx<'_>) -> Vec<u32> {
        let mut s = self.pack(&cx.lay, &self.start);
        // A cube that came to rest where a device is at the start (dropped onto a wall piston
        // that starts out) sits on top of it.
        if self.dynamic {
            let dev = self.device_bits(&cx.lay, &s, None);
            for i in 0..cx.lay.kinds.len() {
                let Some(r) = self.loc_region(self.code_loc(s[CUBES + i])) else { continue };
                if self.graph.regions[r].cond.holds(dev) {
                    continue;
                }
                let on_top = self.graph.regions[r].cells.iter().find_map(|&c| {
                    let up = self.graph.region_of(step(c, Dir::PosZ))?;
                    self.graph.regions[up].cond.holds(dev).then_some(up)
                });
                if let Some(up) = on_top {
                    cx.lay.move_cube(&mut s, i, self.loc_code(CubeLoc::Loose(up)));
                }
            }
        }
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
            Signal::Laser(_) => true,
        };
        enabled
            && (held == Some(sig)
                || match sig {
                    Signal::Floor(b) => lay.cubes(s).contains(&self.button_code(b)),
                    Signal::Pedestal(p) => s[LATCHED] & (1 << p) != 0,
                    Signal::Laser(k) => s[LIT] & (1 << k) != 0,
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

    /// Whether switch `sw` is on.
    fn switch_on(&self, lay: &Layout, s: &[u32], sw: usize, held: Option<Signal>) -> bool {
        let w = &self.switches[sw];
        if w.inputs.is_empty() {
            w.start
        } else {
            w.start != self.all_on(lay, s, &w.inputs, held)
        }
    }

    /// The devices' bits, with `held` on for the moment as well.
    fn device_bits(&self, lay: &Layout, s: &[u32], held: Option<Signal>) -> u64 {
        let mut dev = 0u64;
        for (sw, &bits) in self.switch_bits.iter().enumerate() {
            if bits != 0 && self.switch_on(lay, s, sw, held) {
                dev |= bits;
            }
        }
        if self.routed != 0 {
            // A route is on when one portal takes its projector's path and the other is where
            // the route comes out.
            let through = match (portal(s, 0), portal(s, 1)) {
                (Some(a), Some(b)) => {
                    let (ends, routes) = (&self.class_ends, &self.class_routes);
                    ends[a] & routes[b] | ends[b] & routes[a]
                }
                _ => 0,
            };
            dev &= !self.routed | through;
        }
        dev
    }

    /// The laser receivers powered and the turrets beams hit: every beam from an emitter that
    /// is on, through reflection cubes at spots and through portals.
    fn beam_hits(&self, lay: &Layout, s: &[u32]) -> (u32, u32) {
        let (mut lit, mut hit) = (0u32, 0u32);
        // The spot whose cell is `cell`, if a reflection cube is there.
        let cube_at = |cell: usize| {
            lay.cubes(s)
                .iter()
                .zip(&lay.kinds)
                .find_map(|(&code, &kind)| match self.code_loc(code) {
                    CubeLoc::Spot(k)
                        if kind == CubeKind::Reflection && self.beams.spots[k].cell == cell =>
                    {
                        Some(k)
                    }
                    _ => None,
                })
        };
        let (blue, orange) = (portal(s, 0), portal(s, 1));
        // The beam out of the other portal, for one going into a portal on `class`. Not out
        // of a deployed panel (its beam would leave at the panel's angle).
        let through = |class: usize| {
            let out = if blue == Some(class) {
                orange
            } else if orange == Some(class) {
                blue
            } else {
                None
            }?;
            let c = &self.portals.classes[out];
            if c.deployed.as_ref().is_some_and(|&(bit, _)| dev_of(s) & (1 << bit) != 0) {
                return None;
            }
            c.beam_out
        };
        // Each beam goes one way, so follow each emitter's to its end.
        for (e, &(_, ray)) in self.beams.emitters.iter().enumerate() {
            if !self.switch_on(lay, s, self.emitter_switch[e], None) {
                continue;
            }
            let mut next = Some(ray);
            for _ in 0..=8 {
                let Some(ray) = next.take() else { break };
                let r = &self.beams.rays[ray];
                // A cube set down on another beam's line stops this one.
                let mut stopped = false;
                for ev in &r.events {
                    match *ev {
                        Event::Receiver(k) => lit |= 1 << k,
                        Event::Turret(t) => hit |= 1 << t,
                        Event::Panel { p, front } => {
                            if dev_of(s) & (1 << self.devices.panel_bit[p]) == 0 {
                                continue;
                            }
                            // A portal on the panel takes it on; otherwise it stops here.
                            let class = self.panel_class[p].filter(|_| front);
                            next = class.and_then(|c| through(c));
                            stopped = next.is_none();
                            break;
                        }
                        Event::Cube(sc) => {
                            if let Some(k) = cube_at(sc) {
                                let spot = &self.beams.spots[k];
                                if spot.on.is_none_or(|on| on == ray) {
                                    next = Some(spot.ray);
                                } else {
                                    stopped = true;
                                }
                                break;
                            }
                        }
                    }
                }
                if next.is_some() {
                    continue;
                }
                if stopped {
                    break;
                }
                let Some(end) = r.end else { break };
                let Some(&class) = self.beam_end_class.get(&end) else {
                    break;
                };
                next = through(class);
            }
        }
        (lit, hit)
    }

    /// The shot mask bits that stop a shot now: fizzlers that are on and deployed panels.
    fn shot_stops(&self, lay: &Layout, s: &[u32], held: Option<Signal>) -> u64 {
        let mut stops = self.hazards_on(lay, s, held) & self.fizzlers;
        if !self.panel_shots.is_empty() {
            let dev = match held {
                None => dev_of(s),
                Some(_) => self.device_bits(lay, s, held),
            };
            for &(bit, m) in &self.panel_shots {
                if dev & (1 << bit) != 0 {
                    stops |= m;
                }
            }
        }
        stops
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

    /// What follows by itself after any action: devices and laser receivers follow the
    /// buttons, portals and cubes; waiting droppers drop; a portal on a panel that flipped is
    /// gone; whoever stands where a device went rides or falls (or dies); a turret over an open
    /// portal falls through, and pouring droppers paint where their streams land.
    fn settle(&self, lay: &Layout, s: &mut [u32]) {
        if self.dynamic {
            for _ in 0..4 {
                let redrop = |s: &[u32]| lay.redrop.map_or(0, |r| s[r]);
                let before = (s[1], s[2], s[LIT], dev_of(s), redrop(s));
                self.settle_devices(lay, s);
                if before == (s[1], s[2], s[LIT], dev_of(s), redrop(s)) {
                    break;
                }
            }
        }
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

    /// Whether `settle_devices` would leave `n`, a quiet successor of settled `s`, as it is:
    /// nothing devices, lasers or droppers read changed (only the player and cubes moved,
    /// held or loose, or portals that drive nothing), and where they went is there.
    fn unmoved(&self, lay: &Layout, s: &[u32], n: &[u32]) -> bool {
        let dev = dev_of(s);
        let there = |r: u32| self.graph.regions[r as usize].cond.holds(dev);
        let loose = self.graph.regions.len() as u32;
        (n[0] == s[0] || there(n[0]))
            && (!self.portals_drive || n[BLUE..LATCHED] == s[BLUE..LATCHED])
            && n[LATCHED] == s[LATCHED]
            && n[LIT..CUBES] == s[LIT..CUBES]
            && lay.cubes(s).iter().zip(lay.cubes(n)).all(|(&a, &b)| {
                a == b || (a <= loose && b <= loose && (b == HELD || there(b - 1)))
            })
    }

    /// One round of `settle` for devices and lasers.
    fn settle_devices(&self, lay: &Layout, s: &mut [u32]) {
        // Devices and lasers: a few rounds, as receivers can drive emitters and devices.
        for _ in 0..4 {
            let dev = self.device_bits(lay, s, None);
            s[DEV] = dev as u32;
            s[DEV + 1] = (dev >> 32) as u32;
            if self.beams.is_empty() {
                break;
            }
            let (lit, hit) = self.beam_hits(lay, s);
            s[DOWN] |= hit;
            if lit == s[LIT] {
                break;
            }
            s[LIT] = lit;
        }
        let dev = dev_of(s);
        for c in 0..2 {
            if let Some(cl) = portal(s, c) {
                if !self.portals.classes[cl].needs.holds(dev) {
                    s[BLUE + c] = 0;
                }
            }
        }
        // Cubes waiting in droppers whose inputs are on.
        for h in &self.held_back {
            if self.all_on(lay, s, &h.inputs, None) {
                let (home, gone) = (self.loc_code(h.home), self.loc_code(CubeLoc::Gone));
                if let Some(i) = (0..lay.kinds.len()).find(|&i| {
                    lay.kinds[i] == h.kind && lay.homes[i] == home && s[CUBES + i] == gone
                }) {
                    lay.move_cube(s, i, home);
                    knock(s, h.turret);
                }
            }
        }
        // Wired droppers whose inputs just came on: a new cube drops, the old one dissolves.
        if let Some(w) = lay.redrop {
            for (r, h) in self.redrops.iter().enumerate() {
                let on = self.all_on(lay, s, &h.inputs, None);
                if on && s[w] & (1 << r) == 0 {
                    let home = self.loc_code(h.home);
                    if let Some(i) =
                        (0..lay.kinds.len()).find(|&i| lay.kinds[i] == h.kind && lay.homes[i] == home)
                    {
                        lay.move_cube(s, i, home);
                    }
                }
                s[w] = if on { s[w] | 1 << r } else { s[w] & !(1 << r) };
            }
        }
        // Whoever stands where a device went.
        let fall = |r: usize| -> Option<usize> {
            let mut r = r;
            for _ in 0..4 {
                let region = &self.graph.regions[r];
                if region.cond.holds(dev) {
                    return Some(r);
                }
                r = region.fallback.iter().find(|(c, _)| c.holds(dev))?.1;
            }
            None
        };
        if s[0] & ESCAPED == 0 && s[0] != DEAD {
            s[0] = fall(s[0] as usize).map_or(DEAD, |r| r as u32);
        }
        for i in 0..lay.kinds.len() {
            let loc = self.code_loc(s[CUBES + i]);
            let Some(r) = self.loc_region(loc) else { continue };
            if self.graph.regions[r].cond.holds(dev) {
                continue;
            }
            let to = match fall(r) {
                Some(t) => CubeLoc::Loose(t),
                None => self.code_loc(lay.homes[i]),
            };
            lay.move_cube(s, i, self.loc_code(to));
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
        let dev = dev_of(s);
        (a.behavior(dev).entry == Some(s[0] as usize))
            .then(|| resolve(&b.behavior(dev).landings, dev))
            .flatten()
    }

    /// Dives from the current region into the floor portal of `color`: (height, landing).
    fn dives(&self, s: &[u32], color: Color) -> Vec<(i32, Landing)> {
        let (a, b) = match color {
            Color::Blue => (portal(s, 0), portal(s, 1)),
            Color::Orange => (portal(s, 1), portal(s, 0)),
        };
        let (Some(a), Some(b)) = (a, b) else {
            return Vec::new();
        };
        let (a, b) = (&self.portals.classes[a], &self.portals.classes[b]);
        let (dev, r) = (dev_of(s), s[0] as usize);
        let mut out = Vec::new();
        for &(from, h) in &a.behavior(dev).dives {
            if from != r {
                continue;
            }
            for &(c, l) in b.behavior(dev).flings_at(h) {
                if c.holds(dev) && !out.iter().any(|&(_, o): &(i32, Landing)| o.region == l.region) {
                    out.push((h, l));
                }
            }
        }
        out
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
        cx: &Ctx<'_>,
        s: &[u32],
        settled: bool,
        sc: &mut Scratch,
        emit: &mut impl FnMut(Action, &mut [u32]),
    ) {
        if s[0] & ESCAPED != 0 || s[0] == DEAD {
            return;
        }
        let lay = cx.lay;
        let r = s[0] as usize;
        let dev = dev_of(s);
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
        // settling, unless devices could follow it (`unmoved`).
        let mut done = |n: &mut Vec<u32>, a: Action, quiet: bool| {
            if !(quiet && settled && (!self.dynamic || self.unmoved(lay, s, n))) {
                self.settle(lay, n);
            }
            if n[0] != DEAD {
                emit(a, n);
            }
        };
        let copy = |n: &mut Vec<u32>| {
            n.clear();
            n.extend_from_slice(s);
        };

        let (lo, hi) = self.edge_range[r];
        for e in &self.graph.edges[lo..hi] {
            let ok = e.cond.holds(dev)
                && match e.kind {
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
        for color in Color::BOTH {
            for (height, l) in self.dives(s, color) {
                if let Some(fizzled) = self.travel(lay, s, on, alive, l.region, l.hazards, &mut sc.n)
                {
                    let a = Action::Dive {
                        color,
                        height,
                        to: l.region,
                    };
                    done(&mut sc.n, a, !fizzled);
                }
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
                sc.foot.push((b, self.shot_stops(lay, s, Some(Signal::Floor(b)))));
            }
        }
        // Whether one of the shots with these masks gets past the fizzlers and panels.
        let stops = self.shot_stops(lay, s, None);
        let clear = |masks: &Masks, stops: u64| masks.iter().any(|&m| m & stops == 0);
        let plain = |p: Option<usize>| p.is_none_or(|c| !self.special[c]);
        for color in Color::BOTH {
            let me = color as usize;
            let other = portal(s, color.other() as usize);
            let quiet = plain(portal(s, me)) && plain(other);
            // Beside an inert portal, firing at another inert class only makes a pair no one
            // can use: skip it (fire the useful portal first instead).
            let beside_inert = other.is_some_and(|o| self.inert[o]);
            for t in self.fire_from[r].iter().filter(|_| can_fire) {
                let c = t.class;
                if beside_inert && self.inert[c] {
                    continue;
                }
                let taken = other == Some(c) && t.single;
                let unpainted = t
                    .needs_paint
                    .is_some_and(|f| lay.paint_at(s, f) != PaintKind::Portal as u32 + 1);
                if !self.class_enabled[c]
                    || portal(s, me) == Some(c)
                    || taken
                    || unpainted
                    || !self.portals.classes[c].needs.holds(dev)
                {
                    continue;
                }
                let class = &self.portals.classes[c];
                let deployed = class
                    .deployed
                    .as_ref()
                    .is_some_and(|&(b, _)| dev & (1 << b) != 0);
                let (visible, open) = match (&class.deployed_visible, deployed) {
                    (Some(v), true) => (&v[r], v[r].contains(&0)),
                    _ => (&class.visible_from[r], t.open),
                };
                let stand_on = if open || clear(visible, stops) {
                    Some(None)
                } else {
                    sc.foot
                        .iter()
                        .find(|&&(b, stops)| clear(&self.button_sight[b][c], stops))
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
            let reflects = kind == CubeKind::Reflection;
            let spots = std::iter::once(CubeLoc::Loose(r))
                .chain(
                    self.buttons_in[r]
                        .iter()
                        .filter(|&&b| {
                            self.buttons[b].kind.accepts(kind)
                                && !cubes.contains(&self.button_code(b))
                        })
                        .map(|&b| CubeLoc::OnButton(b)),
                )
                .chain(
                    self.spots_in[r]
                        .iter()
                        .filter(|_| reflects)
                        .filter(|&&k| {
                            // One cube to a cell.
                            let cell = self.beams.spots[k].cell;
                            !cubes.iter().any(|&c| {
                                matches!(self.code_loc(c), CubeLoc::Spot(o) if self.beams.spots[o].cell == cell)
                            })
                        })
                        .map(|&k| CubeLoc::Spot(k)),
                );
            for at in spots {
                copy(&mut sc.n);
                lay.move_cube(&mut sc.n, i, self.loc_code(at));
                let quiet = !matches!(at, CubeLoc::OnButton(_) | CubeLoc::Spot(_));
                done(&mut sc.n, Action::PutDown { at }, quiet);
            }
            for ls in &self.tosses[r] {
                let Some(l) = resolve(ls, dev) else { continue };
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
                let here = self.loc_region(from) == Some(r);
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
                done(&mut sc.n, a, !matches!(from, CubeLoc::OnButton(_) | CubeLoc::Spot(_)));
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
        cx: &Ctx<'_>,
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
        let lay = cx.lay;
        let timed = Some(Signal::Pedestal(p));
        let on = self.hazards_on(lay, s, timed);
        let dev = if self.devices.is_empty() {
            0
        } else {
            self.device_bits(lay, s, timed)
        };
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
            for &(to, fall, hazards, cond) in &self.walk.moves[node] {
                if hazards & on & self.lasers != 0 || !cond.holds(dev) {
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
            CubeLoc::Spot(k) => {
                let sp = &self.beams.spots[k];
                let c = self.beams.spot_cells[sp.cell];
                format!(
                    "{:?} facing {:?} (region {})",
                    c,
                    sp.dir,
                    self.spot_region(k).map_or('?', RegionGraph::label)
                )
            }
            CubeLoc::Gone => "nowhere".into(),
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
                    Move::Plate => format!("get launched by a faith plate to region {}", label(to)),
                    Move::Stairs => format!("walk up the stairs to region {}", label(to)),
                    Move::Funnel => format!("float along the funnel to region {}", label(to)),
                    Move::Ride => format!("ride the platform to region {}", label(to)),
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
            Action::Dive { color, height, to } => format!(
                "drop {height} into the {} portal and fly out to region {}",
                color.name(),
                label(to)
            ),
        }
    }
}

/// Set in a parent's state index (`StateSpace::parent`) when the state reached was stored with
/// its portals swapped.
const SWAPPED: u32 = 1 << 31;

/// Puts the portals of packed state `s` in a fixed order, as blue and orange work the same
/// way round: (A, B) and (B, A) are one state, which halves the states a search stores. Returns
/// whether it swapped them; after that, the actions' colors are the other way round.
fn canonical(s: &mut [u32]) -> bool {
    let swap = s[BLUE] > s[BLUE + 1];
    if swap {
        s.swap(BLUE, BLUE + 1);
    }
    swap
}

/// The plan to state `goal` along `parent` links, with each action's color as in the game:
/// flipped wherever the states before it were stored with their portals swapped.
fn plan_to(parent: &[(u32, Action)], goal: usize) -> Vec<Action> {
    let mut steps = Vec::new();
    let mut i = goal;
    while i > 0 {
        let (p, a) = parent[i - 1];
        steps.push((a, p & SWAPPED != 0));
        i = (p & !SWAPPED) as usize;
    }
    let mut flip = false;
    steps
        .into_iter()
        .rev()
        .map(|(a, swapped)| {
            let a = if flip { a.flipped() } else { a };
            flip ^= swapped;
            a
        })
        .collect()
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
const LIT: usize = 5;
const DEV: usize = 6;
const CUBES: usize = 8;
/// Word 0 holds the region, and this bit if the player escaped during a timed dash.
const ESCAPED: u32 = 1 << 31;
/// Word 0 of a state where the player died (a device went from under them over goo).
const DEAD: u32 = ESCAPED - 1;

/// The devices' bits of a packed state.
fn dev_of(s: &[u32]) -> u64 {
    s[DEV] as u64 | (s[DEV + 1] as u64) << 32
}
/// Cube location codes: `HELD`, `1 + region` loose, `1 + regions + button` on a button, so
/// they sort like `CubeLoc`.
const HELD: u32 = 0;

/// The portal of color `c` (0 blue, 1 orange) in a packed state.
fn portal(s: &[u32], c: usize) -> Option<usize> {
    s[BLUE + c].checked_sub(1).map(|v| v as usize)
}

/// How a search packs states into words: region (and `ESCAPED`), blue and orange portal
/// classes + 1 (0 = none), latched pedestals, knocked-over turrets, powered laser receivers,
/// two words of device bits, then each cube's location code, then 2 bits of paint per face
/// (`PaintKind` + 1, 0 = none). Cubes keep the start's
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
    /// Word of `State::redrop`, after the paint, if the model has wired droppers.
    redrop: Option<usize>,
    /// Words per state.
    len: usize,
}

impl Layout {
    /// A layout with no cubes and no paint, before `Model::new` fills it in.
    fn empty() -> Layout {
        Layout {
            kinds: Vec::new(),
            homes: Vec::new(),
            group: Vec::new(),
            paint: CUBES,
            redrop: None,
            len: CUBES,
        }
    }

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

/// Device bits by kind (`Model::device_kinds`).
#[derive(Clone, Copy, Debug, Default)]
pub struct DeviceKinds {
    pub bridges: u64,
    pub platforms: u64,
    pub routed: u64,
}

/// What a search needs besides the model: how it packs states, and the turrets that exist.
struct Ctx<'a> {
    lay: &'a Layout,
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

fn walk_graph(geo: Geo, graph: &RegionGraph) -> WalkGraph {
    let level = geo.level;
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
            let Some(open) = (match geo.block(c, d) {
                None => Some(Cond::ALWAYS),
                Some(b) => b.negate(),
            }) else {
                continue;
            };
            let sn = geo.stand(n);
            let mut targets: Vec<(Cond, Cell, u64)> = Vec::new();
            if let Some(cn) = sn {
                targets.extend(open.and(cn).map(|cc| (cc, n, 0)));
            }
            let rest = match sn {
                None => Some(open),
                Some(cn) => cn.negate().and_then(|x| open.and(x)),
            };
            if let Some(rest) = rest {
                for f in geo.falls(n) {
                    if let Some(cc) = rest.and(f.cond) {
                        targets.push((cc, f.cell, f.hazards));
                    }
                }
            }
            for (cond, f, fallen) in targets {
                if let Some(&j) = w.node.get(&f) {
                    w.moves[i].push((j, n[2] - f[2], face.hazards | fallen, cond));
                }
            }
        }
    }
    // Deployed stairs.
    for m in geo.dyn_.moves.iter().filter(|m| m.kind == Move::Stairs) {
        for f in &m.from {
            let Some(&i) = w.node.get(f) else { continue };
            for &(c, t, h) in &m.to {
                if let (Some(&j), Some(cond)) = (w.node.get(&t), m.cond.and(c)) {
                    w.moves[i].push((j, 0, m.hazards | h, cond));
                }
            }
        }
    }
    w
}
