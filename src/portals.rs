//! Portal surfaces as the solver sees them: where a portal can be walked into, where you
//! come out, who can see it, and face classes (faces that behave identically).
//!
//! Rules (ASSUMPTIONS until checked in-game):
//! - The eye is at the centre of a standable cell (64u above the floor). A face can be shot
//!   from a region if the segment from some cell's eye to a target point on the face crosses
//!   only open voxels and no face that stops shots: glass, or a fizzler while it is on.
//!   Grating and laser fields let shots through. Which fizzlers a shot crosses is recorded,
//!   so the solver can check them against the current state.
//! - A wall or floor portal can be walked into from the cell in front of it, if that cell is
//!   standable. Ceiling portals can't be entered.
//! - Coming out of a portal (player or cube) after walking in you fall straight down from the
//!   cell in front of it, with no momentum. Ending in goo means death, so there is no landing.
//!   Dropping onto a floor portal from a ledge is a fling (`flings`).
//! - Gel poured into a floor portal comes out of the other one; out of a ceiling portal it
//!   falls straight down. (Out of a wall or floor portal it isn't modelled: it paints nothing.)
//! - Panels (`dynamics`): a flip panel's face takes portals only while its white side is out;
//!   flipping it clears a portal on it. A portal on an angled panel stays on it as it moves; a
//!   deployed 90° panel's face looks across its hinge edge, a tilted one flings along its tilted
//!   normal.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::flings;
use crate::geom::{step, step_orders, Cell, Dir};
use crate::lasers::Event;
use crate::level::{BarrierKind, Level};
use crate::regions::{Cond, Geo, RegionGraph};

/// How generous line of sight is. The solver's conservative and liberal modes differ here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sight {
    /// Hazards (bits of `Level::hazards`) whose fields always stop portal shots.
    pub blocking_hazards: u64,
    /// Hazards whose fields stop shots only while on: a shot through them is clear, and they
    /// go in its mask.
    pub switchable_hazards: u64,
    /// Passing exactly through a voxel edge or corner: true = every way around it must be
    /// clear, false = one way is enough.
    pub strict_grazing: bool,
    /// Also aim at four points inset from the face's corners, not just its centre.
    pub corner_targets: bool,
}

impl Sight {
    pub const STRICT: Sight = Sight {
        blocking_hazards: u64::MAX,
        switchable_hazards: 0,
        strict_grazing: true,
        corner_targets: false,
    };
}

/// A face, as (open cell, direction to the solid side).
pub type Face = (Cell, Dir);

/// Where something coming to rest at a cell ends up.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Landing {
    pub region: usize,
    /// Index of the floor button at the landing cell, if any. A cube landing here presses it.
    pub button: Option<i32>,
    /// Index (in `Level::turrets`) of a turret at the landing cell: a cube landing on it
    /// knocks it over.
    pub turret: Option<usize>,
    /// Hazard fields fallen through on the way down.
    pub hazards: u64,
}

/// Ways something can come to rest, each with the device condition it needs; the first
/// whose condition holds is the one.
pub type Landings = Vec<(Cond, Landing)>;

impl Landing {
    /// Comes to rest in `cell` (a region's), having crossed `hazards`.
    pub fn at(level: &Level, graph: &RegionGraph, cell: Cell, hazards: u64) -> Option<Landing> {
        Some(Landing {
            hazards,
            region: graph.region_of(cell)?,
            button: level.floor_button_at(cell),
            turret: level.turret_at(cell),
        })
    }

    /// Falls from open cell `c` without devices; `None` if that ends in goo.
    pub fn from_fall(level: &Level, graph: &RegionGraph, c: Cell) -> Option<Landing> {
        let (f, hazards) = level.fall_path(c)?;
        Landing::at(level, graph, f, hazards)
    }

    /// Falls from open cell `c` in the geometry with devices: every way it can end.
    pub fn falls(geo: Geo, graph: &RegionGraph, c: Cell) -> Landings {
        geo.falls(c)
            .into_iter()
            .filter_map(|f| Some((f.cond, Landing::at(geo.level, graph, f.cell, f.hazards)?)))
            .collect()
    }
}

/// The first landing whose condition holds in `dev`.
pub fn resolve(landings: &Landings, dev: u64) -> Option<Landing> {
    landings
        .iter()
        .find(|(c, _)| c.holds(dev))
        .map(|&(_, l)| l)
}

/// How a face class behaves for the player and cubes going through it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Behavior {
    /// The region that can walk into a portal here.
    pub entry: Option<usize>,
    /// Where you come out of a portal here, walking in.
    pub landings: Landings,
    /// Dropping into a portal here from a ledge: (region dropped from, voxels fallen).
    pub dives: Vec<(usize, i32)>,
    /// Coming out of a portal here at the speed of a fall of each height: where you can land.
    pub flings: Vec<(i32, Landings)>,
}

impl Behavior {
    pub fn flings_at(&self, height: i32) -> &[(Cond, Landing)] {
        self.flings
            .iter()
            .find(|(h, _)| *h == height)
            .map_or(&[], |(_, l)| l.as_slice())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FaceClass {
    pub id: usize,
    pub faces: Vec<Face>,
    /// How it behaves, unless `deployed` applies.
    pub flat: Behavior,
    /// An angled panel's bit, and how the class behaves while it is deployed.
    pub deployed: Option<(usize, Behavior)>,
    /// Indexed by region: the masks of its clear shots at faces of this class. Empty if it
    /// can't shoot one at all.
    pub visible_from: Vec<Masks>,
    /// An angled panel's class: the same while it is deployed (shots at the tilted panel).
    pub deployed_visible: Option<Vec<Masks>>,
    /// The floor face under this turret (index in `Level::turrets`): a portal here drops it.
    pub turret: Option<usize>,
    /// Where this gel stream (index in `Level::gels`) lands: a portal here takes the gel.
    pub stream: Option<usize>,
    /// Ceiling faces whose fall lands in this cell: gel coming out of a portal here paints
    /// around it. Only set when a stream can be sent through a portal.
    pub gel_cell: Option<Cell>,
    /// Not white: only usable while this face (the solver's id, from `Extras::paintable`)
    /// has conversion gel on it.
    pub needs_paint: Option<usize>,
    /// Only takes portals while this holds (flip panels).
    pub needs: Cond,
    /// Projectors (indices into `Level::projectors`) whose paths end here: a portal here
    /// takes them on.
    pub ends: Vec<usize>,
    /// (projector, route) of the routes coming out of a portal here (`dynamics::Route`).
    pub routes: Vec<(usize, usize)>,
    /// A laser beam ends here: a portal here takes it.
    pub beam_end: bool,
    /// The beam (index into `Beams::rays`) out of a portal here.
    pub beam_out: Option<usize>,
}

impl FaceClass {
    /// Whether region `r` can shoot this class, at least while no switchable fizzler is on.
    pub fn seen_from(&self, r: usize) -> bool {
        !self.visible_from[r].is_empty()
    }

    /// How it behaves with the devices in `dev`.
    pub fn behavior(&self, dev: u64) -> &Behavior {
        match &self.deployed {
            Some((bit, b)) if dev & (1 << bit) != 0 => b,
            _ => &self.flat,
        }
    }
}

/// The switchable hazards crossed by each of several clear shots, keeping only minimal masks:
/// one that contains another is never the easier shot. A shot works when none of its mask is
/// on.
pub type Masks = Vec<u64>;

/// Adds the mask of one more clear shot.
pub fn add_mask(masks: &mut Masks, m: u64) {
    if masks.iter().any(|&e| e & m == e) {
        return;
    }
    masks.retain(|&e| e & m != m);
    masks.push(m);
}

pub struct Portals {
    pub classes: Vec<FaceClass>,
}

/// An angled panel on a face: its bit, its hinge's side (where its face leans) and its angle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PanelFace {
    pub bit: usize,
    pub hinge: Dir,
    pub angle: i32,
}

/// What besides white surfaces shapes face classes.
#[derive(Clone, Debug, Default)]
pub struct Extras {
    /// The floor face each gel stream lands on (index in `Level::gels`).
    pub streams: Vec<(Face, usize)>,
    /// Split ceiling classes by where their fall lands, for gel coming out of them.
    pub split_ceilings: bool,
    /// Faces that aren't white but conversion gel may paint, with the id the solver knows
    /// each by.
    pub paintable: Vec<(Face, usize)>,
    /// Flip panel faces, with when they take portals (white ones while unflipped, black ones
    /// while flipped).
    pub flips: Vec<(Face, Cond)>,
    /// Angled panel faces.
    pub panels: HashMap<Face, PanelFace>,
    /// Projector paths' ends and routes (`dynamics::Devices::path_ends`, `route_faces`).
    pub ends: HashMap<Face, Vec<usize>>,
    pub routes: HashMap<Face, Vec<(usize, usize)>>,
    /// Laser beam ends and the beams out of portals (`lasers::Beams`), with their events.
    pub beam_ends: HashSet<Face>,
    pub beam_exits: HashMap<Face, (usize, Vec<Event>)>,
    /// Drops onto floor faces: per face, (region, height).
    pub dives: HashMap<Face, Vec<(usize, i32)>>,
    /// Heights of every dive: flings are worked out for these.
    pub fling_heights: Vec<i32>,
    /// Liberal fling landings (`flings::trials`).
    pub liberal: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ClassKey {
    flat: Behavior,
    deployed: Option<(usize, Behavior)>,
    turret: Option<usize>,
    stream: Option<usize>,
    gel_cell: Option<Cell>,
    needs_paint: Option<usize>,
    needs: Cond,
    ends: Vec<usize>,
    routes: Vec<(usize, usize)>,
    beam_end: Option<Face>,
    beam_out: Option<Vec<Event>>,
}

/// How a portal on face `(c, d)` behaves: walked into from `c`, coming out above `c`, and
/// flung out along `e` (`None`: along the face's normal) from the face's own exit point.
fn behavior(
    geo: Geo,
    graph: &RegionGraph,
    (c, d): Face,
    tilt: Option<(Dir, i32)>,
    extras: &Extras,
) -> Behavior {
    let level = geo.level;
    let entry = (d != Dir::PosZ && level.standable(c))
        .then(|| graph.region_of(c))
        .flatten();
    let landings = Landing::falls(geo, graph, c);
    let dives = extras.dives.get(&(c, d)).cloned().unwrap_or_default();
    let e = flings::exit_vector(d, tilt);
    let flings = extras
        .fling_heights
        .iter()
        .map(|&h| {
            let speed = flings::dive_speed(h);
            let trials = flings::landings(level, c, d, e, speed, extras.liberal);
            let mut out: Landings = Vec::new();
            let regions: Vec<Option<Landing>> = trials
                .iter()
                .map(|t| t.and_then(|(cell, hz)| Landing::at(level, graph, cell, hz)))
                .collect();
            if extras.liberal {
                for l in regions.into_iter().flatten() {
                    if !out.iter().any(|&(_, o)| o == l) {
                        out.push((Cond::ALWAYS, l));
                    }
                }
            } else if let Some(&Some(first)) = regions.first() {
                // Conservative: every trial lands the same way.
                if regions.iter().all(|r| r.is_some_and(|l| l.region == first.region)) {
                    out.push((Cond::ALWAYS, first));
                }
            }
            // Landing where walking in lands adds nothing a drop and a walk-in can't do.
            let plain: Vec<usize> = landings.iter().map(|&(_, l)| l.region).collect();
            out.retain(|&(_, l)| !plain.contains(&l.region));
            out.sort();
            (h, out)
        })
        .filter(|(_, out)| !out.is_empty())
        .collect();
    Behavior {
        entry,
        landings,
        dives,
        flings,
    }
}

impl Portals {
    /// Groups every portalable face by how it behaves (entry, landing, dives and flings, and as
    /// a deployed panel), plus what makes faces special: a turret standing on it, a gel stream
    /// landing on it, where gel out of a ceiling portal lands, conversion gel it needs, flip
    /// panels, projections and laser beams going in or coming out. Faces with no entry or
    /// landing are useless and dropped, as are classes no region can see.
    pub fn build(geo: Geo, graph: &RegionGraph, sight: Sight, extras: &Extras) -> Portals {
        let level = geo.level;
        let mut groups: BTreeMap<ClassKey, Vec<Face>> = BTreeMap::new();
        let streams: HashMap<Face, usize> = extras.streams.iter().copied().collect();
        let mut add = |(c, d): Face, needs_paint: Option<usize>, needs: Cond| {
            let flat = behavior(geo, graph, (c, d), None, extras);
            let deployed = extras.panels.get(&(c, d)).map(|p| {
                let b = if p.angle == 90 {
                    // Standing on its hinge edge: the face looks across it.
                    let n = step(c, p.hinge);
                    if level.solid(n) {
                        Behavior::default()
                    } else {
                        behavior(geo, graph, (n, p.hinge.opposite()), None, extras)
                    }
                } else {
                    let mut b = behavior(geo, graph, (c, d), Some((p.hinge, p.angle)), extras);
                    b.entry = flat.entry;
                    b.landings = flat.landings.clone();
                    b
                };
                (p.bit, b)
            });
            let useful = |b: &Behavior| b.entry.is_some() || !b.landings.is_empty();
            if !useful(&flat) && !deployed.as_ref().is_some_and(|(_, b)| useful(b)) {
                return;
            }
            let turret = (d == Dir::NegZ)
                .then(|| level.turrets.iter().position(|t| t.cell == c))
                .flatten();
            let stream = streams.get(&(c, d)).copied();
            let gel_cell = (extras.split_ceilings && d == Dir::PosZ)
                .then(|| level.fall(c))
                .flatten();
            let key = ClassKey {
                flat,
                deployed,
                turret,
                stream,
                gel_cell,
                needs_paint,
                needs,
                ends: extras.ends.get(&(c, d)).cloned().unwrap_or_default(),
                routes: extras.routes.get(&(c, d)).cloned().unwrap_or_default(),
                beam_end: extras.beam_ends.contains(&(c, d)).then_some((c, d)),
                beam_out: extras.beam_exits.get(&(c, d)).map(|(_, e)| e.clone()),
            };
            groups.entry(key).or_default().push((c, d));
        };
        let flips: HashMap<Face, Cond> = extras.flips.iter().copied().collect();
        for f in level.portal_surfaces() {
            add(f, None, flips.get(&f).copied().unwrap_or(Cond::ALWAYS));
        }
        for &(f, needs) in &extras.flips {
            if !level.portal_surface(f.0, f.1) && level.solid(step(f.0, f.1)) {
                add(f, None, needs);
            }
        }
        for &(f, id) in &extras.paintable {
            add(f, Some(id), Cond::ALWAYS);
        }
        let classes = groups
            .into_iter()
            .map(|(key, faces)| {
                let visible_from: Vec<Masks> = graph
                    .regions
                    .iter()
                    .map(|r| shot_masks(level, &r.cells, &faces, sight))
                    .collect();
                (key, faces, visible_from)
            })
            .map(|(key, faces, visible_from)| {
                // A deployed panel is seen where it sticks out (one face per such class).
                let deployed_visible = key.deployed.as_ref().and_then(|_| {
                    let p = extras.panels.get(&faces[0])?;
                    let (cell, point, back) = panel_aim(faces[0], p.hinge, p.angle);
                    Some(
                        graph
                            .regions
                            .iter()
                            .map(|r| point_masks(level, &r.cells, cell, point, back, sight))
                            .collect::<Vec<Masks>>(),
                    )
                });
                (key, faces, visible_from, deployed_visible)
            })
            .filter(|(_, _, vis, dvis)| {
                vis.iter().any(|m| !m.is_empty())
                    || dvis.as_ref().is_some_and(|v| v.iter().any(|m| !m.is_empty()))
            })
            .enumerate()
            .map(|(id, (key, faces, visible_from, deployed_visible))| {
                let beam_out = key
                    .beam_out
                    .as_ref()
                    .and_then(|_| extras.beam_exits.get(&faces[0]).map(|(r, _)| *r));
                FaceClass {
                    id,
                    faces,
                    flat: key.flat,
                    deployed: key.deployed,
                    visible_from,
                    deployed_visible,
                    turret: key.turret,
                    stream: key.stream,
                    gel_cell: key.gel_cell,
                    needs_paint: key.needs_paint,
                    needs: key.needs,
                    ends: key.ends,
                    routes: key.routes,
                    beam_end: key.beam_end.is_some(),
                    beam_out,
                }
            })
            .collect();
        Portals { classes }
    }
}

/// The shortest drop that counts as a fling: one voxel (390 u/s) carries you about a voxel out
/// of a wall portal, which walking in and falling out nearly does too.
pub const MIN_DIVE: i32 = 2;

/// Drops onto white floor faces: for each, the regions a ledge of which drops onto it, and
/// how far (voxels, at least `MIN_DIVE`). Off a ledge the player can fall straight down or
/// jump out up to `reach` cells (over open air at the ledge's height). Without devices.
pub fn dives(level: &Level, graph: &RegionGraph, reach: i32) -> HashMap<Face, Vec<(usize, i32)>> {
    let mut out: HashMap<Face, Vec<(usize, i32)>> = HashMap::new();
    for r in &graph.regions {
        for &c in &r.cells {
            for d in Dir::HORIZONTAL {
                let mut cur = c;
                for _ in 0..reach {
                    let n = step(cur, d);
                    if level.solid(n) || level.standable(n) || level.face(cur, d).barrier.is_some() {
                        break;
                    }
                    cur = n;
                    let Some((f, _)) = level.fall_path(n) else { continue };
                    let h = c[2] - f[2];
                    if h >= MIN_DIVE && level.portal_surface(f, Dir::NegZ) {
                        let v = out.entry((f, Dir::NegZ)).or_default();
                        if !v.contains(&(r.id, h)) {
                            v.push((r.id, h));
                        }
                    }
                }
            }
        }
    }
    for v in out.values_mut() {
        v.sort();
    }
    out
}

/// Where to aim at a deployed angled panel on face `(c, d)`: (the cell the shot ends in, the
/// panel's middle, the panel's facing). It swings out about its `hinge` edge by `angle`.
pub fn panel_aim((c, d): Face, hinge: Dir, angle: i32) -> (Cell, [f64; 3], [f64; 3]) {
    let centre: [f64; 3] = std::array::from_fn(|a| c[a] as f64 + 0.5 + 0.5 * d.vector()[a] as f64);
    let n = d.opposite().vector().map(f64::from);
    let h = hinge.vector().map(f64::from);
    let (s, co) = (angle as f64).to_radians().sin_cos();
    let point: [f64; 3] = std::array::from_fn(|a| {
        centre[a] + 0.5 * h[a] + 0.5 * (-h[a] * co + n[a] * s)
    });
    let facing: [f64; 3] = std::array::from_fn(|a| n[a] * co + h[a] * s);
    // Inside its own cell, or (standing on its hinge edge) on the face to the next one.
    let cell = if angle >= 90 { step(c, hinge) } else { c };
    let point = if angle >= 90 {
        std::array::from_fn(|a| point[a] + 0.01 * h[a])
    } else {
        point
    };
    (cell, point, facing)
}

/// Masks of the clear shots from any of `eyes` at `point` (in cell `cell`), hitting it from the
/// side `facing` points to.
pub fn point_masks(
    level: &Level,
    eyes: &[Cell],
    cell: Cell,
    point: [f64; 3],
    facing: [f64; 3],
    sight: Sight,
) -> Masks {
    let mut masks = Masks::new();
    for &eye in eyes {
        for p0 in eye_points(level, eye) {
            if let Some(m) = point_shot(level, eye, p0, cell, point, facing, sight) {
                add_mask(&mut masks, m);
                if m == 0 {
                    return masks;
                }
            }
        }
    }
    masks
}

/// The mask of a clear shot from eye point `p0` (in cell `eye`) at `point` (in cell `cell`)
/// from the side `facing` points to, if there is one.
pub fn point_shot(
    level: &Level,
    eye: Cell,
    p0: [f64; 3],
    cell: Cell,
    point: [f64; 3],
    facing: [f64; 3],
    sight: Sight,
) -> Option<u64> {
    if (0..3).map(|a| (point[a] - p0[a]) * facing[a]).sum::<f64>() >= 0.0 {
        return None; // From behind.
    }
    segment_hazards(level, eye, p0, point, cell, sight)
}

/// Masks of the clear shots from any of `eyes` at any of `faces` (empty: none is clear).
pub fn shot_masks(level: &Level, eyes: &[Cell], faces: &[Face], sight: Sight) -> Masks {
    let mut masks = Masks::new();
    for &eye in eyes {
        for &face in faces {
            for m in shots(level, eye, face, sight) {
                add_mask(&mut masks, m);
                if m == 0 {
                    return masks;
                }
            }
        }
    }
    masks
}

/// Whether the eye in open cell `eye` can see the centre of `face`, with strict rules.
pub fn line_of_sight(level: &Level, eye: Cell, face: Face) -> bool {
    shots(level, eye, face, Sight::STRICT).next().is_some()
}

/// How far toward a drop the player can stand to look over its edge, in voxels from the
/// middle of the cell (the player's 32u box still on the floor).
pub const EDGE_EYE: f64 = 0.4;

/// Where the player's eye can be in open cell `eye`: the middle of the cell, and over each
/// edge they can look down from (`EDGE_EYE` toward a cell they would fall into). 0.5 voxel
/// over the floor.
pub fn eye_points(level: &Level, eye: Cell) -> Vec<[f64; 3]> {
    let centre = eye.map(|v| v as f64 + 0.5);
    let mut out = vec![centre];
    for d in Dir::HORIZONTAL {
        let n = step(eye, d);
        if !level.solid(n) && !level.standable(n) && level.face(eye, d).barrier.is_none() {
            let v = d.vector();
            out.push(std::array::from_fn(|a| centre[a] + EDGE_EYE * v[a] as f64));
        }
    }
    out
}

/// The masks of the clear shots from the eye in open cell `eye` (`eye_points`) at `face`
/// under `sight`, one per eye point and target point that can be hit.
pub fn shots(level: &Level, eye: Cell, face: Face, sight: Sight) -> impl Iterator<Item = u64> + '_ {
    eye_points(level, eye)
        .into_iter()
        .flat_map(move |p| shots_from(level, eye, p, face, sight))
}

/// `shots` from one eye point `p0` in cell `eye`.
pub fn shots_from(
    level: &Level,
    eye: Cell,
    p0: [f64; 3],
    face: Face,
    sight: Sight,
) -> impl Iterator<Item = u64> + '_ {
    const CORNERS: [(f64, f64); 4] = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)];
    let (target, d) = face;
    let n = d.vector();
    let centre: [f64; 3] = std::array::from_fn(|a| target[a] as f64 + 0.5 + 0.5 * n[a] as f64);
    let (u, v) = match d.axis() {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let corners = if sight.corner_targets { 4 } else { 0 };
    std::iter::once(centre)
        .chain(CORNERS[..corners].iter().map(move |&(du, dv)| {
            let mut p = centre;
            p[u] += 0.3 * du;
            p[v] += 0.3 * dv;
            p
        }))
        .filter_map(move |p| segment_hazards(level, eye, p0, p, target, sight))
}

/// Whether a shot crossing from `c` to its neighbour in `d` gets through, and if so the
/// switchable hazards it crosses.
fn crossing(level: &Level, c: Cell, d: Dir, sight: Sight) -> Option<u64> {
    let n = step(c, d);
    if level.solid(n) {
        return None;
    }
    let f = level.face(c, d);
    let blocked = f.barrier == Some(BarrierKind::Glass) || f.hazards & sight.blocking_hazards != 0;
    (!blocked).then_some(f.hazards & sight.switchable_hazards)
}

/// Walks the voxels the segment from `p0` (in cell `eye`) to `p1` crosses (Amanatides & Woo),
/// until it reaches cell `target`: `None` if something stops it, else the switchable hazards
/// it crosses. Where it passes exactly through an edge or corner, the ways around it are the
/// orders of stepping the tied axes. With `sight.strict_grazing` every way must be clear and
/// the hazards of each count; otherwise one clear way is enough, and only hazards that every
/// clear way crosses count.
fn segment_hazards(
    level: &Level,
    eye: Cell,
    p0: [f64; 3],
    p1: [f64; 3],
    target: Cell,
    sight: Sight,
) -> Option<u64> {
    const EPS: f64 = 1e-9;
    let v: [f64; 3] = std::array::from_fn(|a| p1[a] - p0[a]);

    let mut cur = eye;
    let mut t_max: [f64; 3] = std::array::from_fn(|a| {
        if v[a] == 0.0 {
            f64::INFINITY
        } else {
            let boundary = cur[a] as f64 + if v[a] > 0.0 { 1.0 } else { 0.0 };
            (boundary - p0[a]) / v[a]
        }
    });
    let t_delta: [f64; 3] = v.map(|x| {
        if x == 0.0 {
            f64::INFINITY
        } else {
            1.0 / x.abs()
        }
    });

    let mut crossed = 0;
    while cur != target {
        let t = t_max.iter().cloned().fold(f64::INFINITY, f64::min);
        if t >= 1.0 - EPS {
            return None; // Ran out of segment without reaching the face's cell.
        }
        let mut dirs = [Dir::PosX; 3];
        let mut k = 0;
        for a in 0..3 {
            if t_max[a] - t < EPS {
                dirs[k] = Dir::from_axis(a, v[a] > 0.0);
                k += 1;
            }
        }
        // Masks of the clear ways: all of them OR'd, and AND'd.
        let (mut any, mut every, mut clear) = (0, u64::MAX, false);
        for order in step_orders(k) {
            let mut c = cur;
            let mut mask = Some(0);
            for &i in &order[..k] {
                mask = mask
                    .zip(crossing(level, c, dirs[i], sight))
                    .map(|(a, m)| a | m);
                if mask.is_none() {
                    break;
                }
                c = step(c, dirs[i]);
            }
            match mask {
                Some(m) => {
                    any |= m;
                    every &= m;
                    clear = true;
                }
                None if sight.strict_grazing => return None,
                None => {}
            }
        }
        crossed |= match (sight.strict_grazing, clear) {
            (true, _) => any,
            (false, true) => every,
            (false, false) => return None,
        };
        for &d in &dirs[..k] {
            cur = step(cur, d);
            t_max[d.axis()] += t_delta[d.axis()];
        }
    }
    Some(crossed)
}

/// Short human description of a class: its faces' bounding box and orientation.
pub fn describe(class: &FaceClass) -> String {
    let mut lo = class.faces[0].0;
    let mut hi = lo;
    for (c, _) in &class.faces {
        for a in 0..3 {
            lo[a] = lo[a].min(c[a]);
            hi[a] = hi[a].max(c[a]);
        }
    }
    let mut dirs: Vec<Dir> = class.faces.iter().map(|f| f.1).collect();
    dirs.sort();
    dirs.dedup();
    let surface = match dirs.as_slice() {
        [Dir::NegZ] => "floor".to_string(),
        [Dir::PosZ] => "ceiling".to_string(),
        [d] => format!("{d:?} wall"),
        _ => "mixed".to_string(),
    };
    let range = |a: usize| {
        if lo[a] == hi[a] {
            lo[a].to_string()
        } else {
            format!("{}..{}", lo[a], hi[a])
        }
    };
    format!(
        "{surface} ({},{},{}), {} face{}",
        range(0),
        range(1),
        range(2),
        class.faces.len(),
        if class.faces.len() == 1 { "" } else { "s" }
    )
}
