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

impl State {
    fn held(&self) -> Option<Cube> {
        self.cubes.iter().copied().find(|c| c.loc == CubeLoc::Held)
    }

    fn move_cube(&self, cube: Cube, to: CubeLoc) -> State {
        let mut s = self.clone();
        let i = s.cubes.iter().position(|&c| c == cube).unwrap();
        s.cubes[i].loc = to;
        s.cubes.sort();
        s
    }

    /// A fizzler: portals gone, held cube back to its home.
    fn fizzle(&mut self) {
        self.portals = [None, None];
        for c in &mut self.cubes {
            if c.loc == CubeLoc::Held {
                c.loc = c.home;
            }
        }
        self.cubes.sort();
    }
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
    /// Where gel coming out of a ceiling portal paints, by the cell it lands in.
    cell_spots: HashMap<Cell, Vec<usize>>,
    gel_moves: Vec<GelMove>,
    /// Buttons and pedestals that send a signal. Metrics switch one off to test whether it's
    /// needed: a disabled one never counts as pressed.
    pub button_enabled: Vec<bool>,
    pub pedestal_enabled: Vec<bool>,
    /// Fields that can be on. Metrics switch one off to see whether it changes the solution.
    pub hazard_enabled: Vec<bool>,
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
        let button_sight = buttons
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
        Ok(Model {
            mode,
            rules,
            class_enabled: vec![true; portals.classes.len()],
            turret_enabled: vec![true; turret_list.len()],
            turrets: turret_list,
            gel_enabled: vec![true; gels.len()],
            gels,
            paint_faces,
            cell_spots,
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

    /// `held`: a signal held on for the moment, beyond what the state says: a running timed
    /// pedestal, or a floor button the player stands on.
    fn signal(&self, s: &State, sig: Signal, held: Option<Signal>) -> bool {
        let enabled = match sig {
            Signal::Floor(b) => self.button_enabled[b],
            Signal::Pedestal(p) => self.pedestal_enabled[p],
        };
        enabled
            && (held == Some(sig)
                || match sig {
                    Signal::Floor(b) => s.cubes.iter().any(|c| c.loc == CubeLoc::OnButton(b)),
                    Signal::Pedestal(p) => s.latched & (1 << p) != 0,
                })
    }

    fn all_on(&self, s: &State, inputs: &[Signal], held: Option<Signal>) -> bool {
        inputs.iter().all(|&i| self.signal(s, i, held))
    }

    fn exit_open_with(&self, s: &State, held: Option<Signal>) -> bool {
        self.exit_inputs.is_empty() || self.all_on(s, &self.exit_inputs, held) != self.exit_inverted
    }

    /// The start state: paint from splats without droppers, then whatever the droppers pour.
    pub fn initial(&self) -> State {
        let mut s = self.start.clone();
        for (g, gel) in self.gels.iter().enumerate() {
            if gel.dropper.is_none() && self.gel_enabled[g] {
                paint(&mut s.paint, &gel.spot, gel.kind);
            }
        }
        self.settle(&mut s);
        s
    }

    /// Turrets live in state `s` (not knocked over, not switched off).
    fn alive(&self, s: &State) -> u32 {
        let enabled = self
            .turret_enabled
            .iter()
            .enumerate()
            .filter(|(_, &e)| e)
            .fold(0u32, |m, (i, _)| m | 1 << i);
        enabled & !s.down
    }

    /// Whether a live turret can shoot region `r`.
    fn exposed(&self, s: &State, r: usize) -> bool {
        self.graph.regions[r].danger & self.alive(s) != 0
    }

    fn gel_on(&self, s: &State, g: usize) -> bool {
        let gel = &self.gels[g];
        if gel.inputs.is_empty() {
            gel.start_enabled
        } else {
            gel.start_enabled != self.all_on(s, &gel.inputs, None)
        }
    }

    /// The class of the portal gel stream `g` comes out of, if one is under it.
    fn redirect(&self, s: &State, g: usize) -> Option<usize> {
        let [Some(a), Some(b)] = s.portals else {
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

    /// Whether a pouring gel stream goes through a portal in state `s`.
    pub fn redirected(&self, s: &State) -> bool {
        (0..self.gels.len()).any(|g| {
            self.gels[g].dropper.is_some()
                && self.gel_enabled[g]
                && self.gel_on(s, g)
                && self.redirect(s, g).is_some()
        })
    }

    pub fn painted(&self, s: &State, face: usize, kind: PaintKind) -> bool {
        s.paint.binary_search(&paint_code(face, kind)).is_ok()
    }

    /// What follows by itself after any action: a turret over an open portal falls through,
    /// and pouring droppers paint where their streams land.
    fn settle(&self, s: &mut State) {
        if let [Some(a), Some(b)] = s.portals {
            for c in [a, b] {
                if let Some(t) = self.portals.classes[c].turret {
                    s.down |= 1 << t;
                }
            }
        }
        for (g, gel) in self.gels.iter().enumerate() {
            if gel.dropper.is_none() || !self.gel_enabled[g] || !self.gel_on(s, g) {
                continue;
            }
            let spot: &[usize] = match self.redirect(s, g) {
                None => &gel.spot,
                Some(exit) => match self.portals.classes[exit]
                    .gel_cell
                    .and_then(|c| self.cell_spots.get(&c))
                {
                    Some(spot) => spot,
                    None => continue,
                },
            };
            paint(&mut s.paint, spot, gel.kind);
        }
    }

    /// The buttons and pedestals wired to hazard `h`.
    pub fn hazard_inputs(&self, h: usize) -> &[Signal] {
        &self.hazards[h].inputs
    }

    pub fn exit_open(&self, s: &State) -> bool {
        self.exit_open_with(s, None)
    }

    fn hazard_on(&self, s: &State, h: usize, held: Option<Signal>) -> bool {
        let hz = &self.hazards[h];
        if !self.hazard_enabled[h] {
            false
        } else if hz.inputs.is_empty() {
            hz.start_enabled
        } else {
            hz.start_enabled != self.all_on(s, &hz.inputs, held)
        }
    }

    fn active(&self, s: &State, mask: u64, kind: HazardKind, held: Option<Signal>) -> bool {
        (0..self.hazards.len()).any(|h| {
            mask & (1 << h) != 0 && self.hazards[h].kind == kind && self.hazard_on(s, h, held)
        })
    }

    /// Whether one of the shots with these masks gets past the fizzlers.
    fn clear_shot(&self, s: &State, masks: &Masks, held: Option<Signal>) -> bool {
        masks
            .iter()
            .any(|&m| !self.active(s, m, HazardKind::Fizzler, held))
    }

    pub fn is_goal(&self, s: &State) -> bool {
        s.escaped || (s.region == self.exit_region && self.exit_open(s))
    }

    /// Where a cube coming to rest at `l` ends up.
    fn cube_lands(&self, l: Landing, s: &State, cube: Cube) -> CubeLoc {
        if self.active(s, l.hazards, HazardKind::Fizzler, None) {
            return cube.home;
        }
        match l
            .button
            .and_then(|i| self.buttons.iter().position(|b| b.index == i))
        {
            Some(b)
                if self.buttons[b].kind.accepts(cube.kind)
                    && !s.cubes.iter().any(|c| c.loc == CubeLoc::OnButton(b)) =>
            {
                CubeLoc::OnButton(b)
            }
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

    /// Moves the player along a path crossing `hazards`; `None` if a laser field stops them.
    fn travel(&self, s: &State, to: usize, hazards: u64) -> Option<State> {
        if self.active(s, hazards, HazardKind::LaserField, None) {
            return None;
        }
        if !self.rules.run_past_turrets && self.exposed(s, to) {
            return None;
        }
        let mut n = s.clone();
        n.region = to;
        if self.active(s, hazards, HazardKind::Fizzler, None) {
            n.fizzle();
        }
        Some(n)
    }

    pub fn successors(&self, s: &State) -> Vec<(Action, State)> {
        let mut out = Vec::new();
        if s.escaped {
            return out;
        }
        let r = s.region;
        let held = s.held();
        // Where a live turret can shoot, the player can only keep moving, and only under
        // liberal rules: conservative ones never go there, so this is the start, and fatal.
        let act = !self.exposed(s, r);
        if !act && !self.rules.run_past_turrets {
            return Vec::new();
        }

        for e in self.graph.edges_from(r) {
            let ok = match e.kind {
                Move::Step => {
                    self.rules.cube_steps && s.cubes.iter().any(|c| c.loc == CubeLoc::Loose(r))
                }
                _ => true,
            };
            if let Some(n) = ok.then(|| self.travel(s, e.to, e.hazards)).flatten() {
                let a = Action::Move {
                    to: e.to,
                    kind: e.kind,
                    hazards: e.hazards,
                };
                out.push((a, n));
            }
        }
        for m in self.gel_moves.iter().filter(|m| m.from == r) {
            if !m.needs.iter().all(|&f| self.painted(s, f, m.paint)) {
                continue;
            }
            if let Some(n) = self.travel(s, m.to, m.hazards) {
                let a = Action::Move {
                    to: m.to,
                    kind: m.kind,
                    hazards: m.hazards,
                };
                out.push((a, n));
            }
        }
        if !act {
            if self.rules.run_past_turrets {
                for color in Color::BOTH {
                    if let Some(l) = self.through(s, color) {
                        if let Some(n) = self.travel(s, l.region, l.hazards) {
                            out.push((Action::Enter { color }, n));
                        }
                    }
                }
            }
            for (_, n) in &mut out {
                self.settle(n);
            }
            return out;
        }

        let can_fire = held.is_none() || self.rules.fire_while_holding;
        // Free weighted buttons here the player could stand on.
        let footholds: Vec<usize> = (0..self.buttons.len())
            .filter(|&b| {
                self.buttons[b].region == Some(r)
                    && !self.button_sight[b].is_empty()
                    && !s.cubes.iter().any(|c| c.loc == CubeLoc::OnButton(b))
            })
            .collect();
        for color in Color::BOTH {
            let me = color as usize;
            let other = s.portals[color.other() as usize];
            for class in &self.portals.classes {
                let taken = other == Some(class.id) && class.faces.len() == 1;
                let unpainted = class
                    .needs_paint
                    .is_some_and(|f| !self.painted(s, f, PaintKind::Portal));
                if !can_fire
                    || !self.class_enabled[class.id]
                    || s.portals[me] == Some(class.id)
                    || taken
                    || unpainted
                {
                    continue;
                }
                let stand_on = if self.clear_shot(s, &class.visible_from[r], None) {
                    Some(None)
                } else {
                    footholds
                        .iter()
                        .copied()
                        .find(|&b| {
                            let held = Some(Signal::Floor(b));
                            self.clear_shot(s, &self.button_sight[b][class.id], held)
                        })
                        .map(Some)
                };
                if let Some(stand_on) = stand_on {
                    let mut n = s.clone();
                    n.portals[me] = Some(class.id);
                    let a = Action::Fire {
                        color,
                        class: class.id,
                        stand_on,
                    };
                    out.push((a, n));
                }
            }
            if let Some(l) = self.through(s, color) {
                if let Some(n) = self.travel(s, l.region, l.hazards) {
                    out.push((Action::Enter { color }, n));
                }
                if let Some(cube) = held {
                    let to = self.cube_lands(l, s, cube);
                    let mut n = s.move_cube(cube, to);
                    knock(&mut n, l.turret);
                    out.push((Action::Throw { color, to }, n));
                }
            }
        }

        if let Some(cube) = held {
            let mut spots = vec![CubeLoc::Loose(r)];
            spots.extend(
                (0..self.buttons.len())
                    .filter(|&b| {
                        self.buttons[b].region == Some(r) && self.buttons[b].kind.accepts(cube.kind)
                    })
                    .map(CubeLoc::OnButton)
                    .filter(|&b| !s.cubes.iter().any(|c| c.loc == b)),
            );
            for at in spots {
                out.push((Action::PutDown { at }, s.move_cube(cube, at)));
            }
            for &l in &self.tosses[r] {
                let to = self.cube_lands(l, s, cube);
                let mut n = s.move_cube(cube, to);
                knock(&mut n, l.turret);
                out.push((Action::Toss { to }, n));
            }
        } else {
            let mut here: Vec<Cube> = s
                .cubes
                .iter()
                .copied()
                .filter(|c| match c.loc {
                    CubeLoc::Loose(cr) => cr == r,
                    CubeLoc::OnButton(b) => self.buttons[b].region == Some(r),
                    CubeLoc::Held => false,
                })
                .collect();
            here.dedup();
            for cube in here {
                let a = Action::PickUp {
                    cube: cube.kind,
                    from: cube.loc,
                };
                out.push((a, s.move_cube(cube, CubeLoc::Held)));
            }
        }

        for (p, ped) in self.pedestals.iter().enumerate() {
            if ped.region != Some(r) || !self.pedestal_enabled[p] {
                continue;
            }
            match ped.timer {
                None if s.latched & (1 << p) == 0 => {
                    let mut n = s.clone();
                    n.latched |= 1 << p;
                    out.push((Action::Press { pedestal: p }, n));
                }
                None => {}
                Some(t) => out.extend(self.dash(s, p, ped.cell, t)),
            }
        }

        let alive = self.alive(s);
        for (t, turret) in self.turrets.iter().enumerate() {
            if alive & (1 << t) != 0 && turret.reach.contains(&r) {
                let mut n = s.clone();
                n.down |= 1 << t;
                out.push((Action::Knock { turret: t }, n));
            }
        }
        for (_, n) in &mut out {
            self.settle(n);
        }
        out
    }

    /// Press timed pedestal `p` at `cell` and walk for at most `seconds`: every region reachable
    /// in time, and escaping if the exit door's cell is reached while it's open.
    fn dash(&self, s: &State, p: usize, cell: Cell, seconds: f64) -> Vec<(Action, State)> {
        let Some(&start) = self.walk.node.get(&cell) else {
            return Vec::new();
        };
        let step_time = VOXEL / self.rules.walk_speed;
        let fall_time = |h: i32| (2.0 * h as f64 * VOXEL / GRAVITY).sqrt();
        // Dijkstra over (node, fizzled); times as integer microseconds for the heap.
        let mut best: HashMap<(usize, bool), u64> = HashMap::from([((start, false), 0)]);
        let mut heap = BinaryHeap::from([std::cmp::Reverse((0u64, start, false))]);
        let limit = (seconds * 1e6) as u64;
        while let Some(std::cmp::Reverse((t, node, fizzled))) = heap.pop() {
            if best.get(&(node, fizzled)).is_some_and(|&b| b < t) {
                continue;
            }
            for &(to, fall, hazards) in &self.walk.moves[node] {
                let timed = Some(Signal::Pedestal(p));
                if self.active(s, hazards, HazardKind::LaserField, timed) {
                    continue;
                }
                if !self.rules.run_past_turrets && self.exposed(s, self.walk.region[to]) {
                    continue;
                }
                let f = fizzled || self.active(s, hazards, HazardKind::Fizzler, timed);
                let nt = t + ((step_time + fall_time(fall)) * 1e6) as u64;
                if nt <= limit && best.get(&(to, f)).is_none_or(|&b| nt < b) {
                    best.insert((to, f), nt);
                    heap.push(std::cmp::Reverse((nt, to, f)));
                }
            }
        }

        let mut out = Vec::new();
        let exit = self.walk.node.get(&self.exit_cell);
        if self.exit_open_with(s, Some(Signal::Pedestal(p))) && best.keys().any(|(n, _)| Some(n) == exit) {
            let mut n = s.clone();
            n.region = self.exit_region;
            n.escaped = true;
            let a = Action::Dash {
                pedestal: p,
                to: self.exit_region,
                escaped: true,
            };
            out.push((a, n));
        }
        let mut ends: Vec<(usize, bool)> = best
            .keys()
            .map(|&(n, f)| (self.walk.region[n], f))
            .collect();
        ends.sort();
        ends.dedup();
        for (region, fizzled) in ends {
            let mut n = s.clone();
            n.region = region;
            if fizzled {
                n.fizzle();
            }
            if n != *s {
                let a = Action::Dash {
                    pedestal: p,
                    to: region,
                    escaped: false,
                };
                out.push((a, n));
            }
        }
        out
    }

    /// Breadth-first search from the start state, exploring at most `max_states` states.
    pub fn solve(&self, max_states: usize) -> Outcome {
        let start = self.initial();
        let mut states = vec![start.clone()];
        let mut parent: Vec<Option<(usize, Action)>> = vec![None];
        let mut seen = HashMap::from([(start, 0)]);
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
        let start = self.initial();
        let mut space = StateSpace {
            states: vec![start.clone()],
            depth: vec![0],
            next: Vec::new(),
            goal: Vec::new(),
            complete: true,
        };
        let mut seen = HashMap::from([(start, 0)]);
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

/// Paints `spot` with `kind` (water washes it).
fn paint(list: &mut Vec<u32>, spot: &[usize], kind: PaintKind) {
    for &f in spot {
        list.retain(|&e| e >> 2 != f as u32);
        if kind != PaintKind::Water {
            list.push(paint_code(f, kind));
        }
    }
    list.sort();
}

fn knock(s: &mut State, turret: Option<usize>) {
    if let Some(t) = turret {
        s.down |= 1 << t;
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
