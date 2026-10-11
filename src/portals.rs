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
//! - Coming out of a portal (player or cube) you fall straight down from the cell in front
//!   of it, with no momentum. Ending in goo means death, so there is no landing.
//! - Gel poured into a floor portal comes out of the other one; out of a ceiling portal it
//!   falls straight down. (Out of a wall or floor portal it isn't modelled: it paints nothing.)

use std::collections::{BTreeMap, HashMap};

use crate::geom::{step, Cell, Dir};
use crate::level::{BarrierKind, ItemKind, Level};
use crate::regions::RegionGraph;

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

impl Landing {
    /// Falls from open cell `c`; `None` if that ends in goo.
    pub fn from_fall(level: &Level, graph: &RegionGraph, c: Cell) -> Option<Landing> {
        let (f, hazards) = level.fall_path(c)?;
        Some(Landing {
            hazards,
            region: graph.region_of(f)?,
            button: level
                .of_kind(&ItemKind::FloorButton)
                .find(|b| b.cell == f)
                .map(|b| b.index),
            turret: level.turrets.iter().position(|t| t.cell == f),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FaceClass {
    pub id: usize,
    pub faces: Vec<Face>,
    /// The region that can walk into a portal here.
    pub entry: Option<usize>,
    /// Where you come out of a portal here.
    pub landing: Option<Landing>,
    /// Indexed by region: the masks of its clear shots at faces of this class. Empty if it
    /// can't shoot one at all.
    pub visible_from: Vec<Masks>,
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
}

impl FaceClass {
    /// Whether region `r` can shoot this class, at least while no switchable fizzler is on.
    pub fn seen_from(&self, r: usize) -> bool {
        !self.visible_from[r].is_empty()
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

/// What besides white surfaces shapes face classes: gel streams that can be sent through a
/// portal, and faces conversion gel can make white.
#[derive(Clone, Debug, Default)]
pub struct Extras {
    /// The floor face each gel stream lands on (index in `Level::gels`).
    pub streams: Vec<(Face, usize)>,
    /// Split ceiling classes by where their fall lands, for gel coming out of them.
    pub split_ceilings: bool,
    /// Faces that aren't white but conversion gel may paint, with the id the solver knows
    /// each by.
    pub paintable: Vec<(Face, usize)>,
}

type ClassKey = (
    Option<usize>,
    Option<Landing>,
    Option<usize>,
    Option<usize>,
    Option<Cell>,
    Option<usize>,
);

impl Portals {
    /// Groups every portalable face by (entry, landing), plus what makes faces special: a
    /// turret standing on it, a gel stream landing on it, where gel out of a ceiling portal
    /// lands, and conversion gel it needs. Faces with no entry or landing are useless and
    /// dropped, as are classes no region can see.
    pub fn build(level: &Level, graph: &RegionGraph, sight: Sight, extras: &Extras) -> Portals {
        let mut groups: BTreeMap<ClassKey, Vec<Face>> = BTreeMap::new();
        let streams: HashMap<Face, usize> = extras.streams.iter().copied().collect();
        let mut add = |(c, d): Face, needs_paint: Option<usize>| {
            let entry = (d != Dir::PosZ && level.standable(c))
                .then(|| graph.region_of(c))
                .flatten();
            let landing = Landing::from_fall(level, graph, c);
            if entry.is_none() && landing.is_none() {
                return;
            }
            let turret = (d == Dir::NegZ)
                .then(|| level.turrets.iter().position(|t| t.cell == c))
                .flatten();
            let stream = streams.get(&(c, d)).copied();
            let gel_cell = (extras.split_ceilings && d == Dir::PosZ)
                .then(|| level.fall(c))
                .flatten();
            let key = (entry, landing, turret, stream, gel_cell, needs_paint);
            groups.entry(key).or_default().push((c, d));
        };
        for f in level.portal_surfaces() {
            add(f, None);
        }
        for &(f, id) in &extras.paintable {
            add(f, Some(id));
        }
        let classes = groups
            .into_iter()
            .map(|(key, faces)| {
                let visible_from = graph
                    .regions
                    .iter()
                    .map(|r| shot_masks(level, &r.cells, &faces, sight))
                    .collect();
                (key, faces, visible_from)
            })
            .filter(|(.., vis): &(_, _, Vec<Masks>)| vis.iter().any(|m| !m.is_empty()))
            .enumerate()
            .map(
                |(id, ((entry, landing, turret, stream, gel_cell, needs_paint), faces, visible_from))| {
                    FaceClass {
                        id,
                        faces,
                        entry,
                        landing,
                        visible_from,
                        turret,
                        stream,
                        gel_cell,
                        needs_paint,
                    }
                },
            )
            .collect();
        Portals { classes }
    }
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
    !shots(level, eye, face, Sight::STRICT).is_empty()
}

/// The masks of the clear shots from the eye in open cell `eye` at `face` under `sight`, one
/// per target point that can be hit.
fn shots(level: &Level, eye: Cell, face: Face, sight: Sight) -> Vec<u64> {
    let (target, d) = face;
    let n = d.vector();
    let centre: [f64; 3] = std::array::from_fn(|a| target[a] as f64 + 0.5 + 0.5 * n[a] as f64);
    let mut points = vec![centre];
    if sight.corner_targets {
        let plane: Vec<usize> = (0..3).filter(|&a| a != d.axis()).collect();
        for (u, v) in [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)] {
            let mut p = centre;
            p[plane[0]] += 0.3 * u;
            p[plane[1]] += 0.3 * v;
            points.push(p);
        }
    }
    points
        .into_iter()
        .filter_map(|p| segment_hazards(level, eye, p, target, sight))
        .collect()
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

/// Walks the voxels the segment from the centre of `eye` to `p1` crosses (Amanatides & Woo),
/// until it reaches cell `target`: `None` if something stops it, else the switchable hazards
/// it crosses. Where it passes exactly through an edge or corner, the ways around it are the
/// orders of stepping the tied axes. With `sight.strict_grazing` every way must be clear and
/// the hazards of each count; otherwise one clear way is enough, and only hazards that every
/// clear way crosses count.
fn segment_hazards(
    level: &Level,
    eye: Cell,
    p1: [f64; 3],
    target: Cell,
    sight: Sight,
) -> Option<u64> {
    const EPS: f64 = 1e-9;
    let p0 = eye.map(|v| v as f64 + 0.5);
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
        let dirs: Vec<Dir> = (0..3)
            .filter(|&a| t_max[a] - t < EPS)
            .map(|a| Dir::from_axis(a, v[a] > 0.0))
            .collect();
        let way = |order: &Vec<Dir>| {
            let mut c = cur;
            let mut mask = 0;
            for &d in order {
                mask |= crossing(level, c, d, sight)?;
                c = step(c, d);
            }
            Some(mask)
        };
        let ways = permutations(&dirs).iter().map(way).collect::<Vec<_>>();
        crossed |= if sight.strict_grazing {
            ways.into_iter().try_fold(0, |a, m| m.map(|m| a | m))?
        } else {
            ways.into_iter().flatten().reduce(|a, m| a & m)?
        };
        for d in dirs {
            cur = step(cur, d);
            t_max[d.axis()] += t_delta[d.axis()];
        }
    }
    Some(crossed)
}

fn permutations(dirs: &[Dir]) -> Vec<Vec<Dir>> {
    if dirs.len() <= 1 {
        return vec![dirs.to_vec()];
    }
    let mut out = Vec::new();
    for i in 0..dirs.len() {
        let mut rest = dirs.to_vec();
        let first = rest.remove(i);
        for mut p in permutations(&rest) {
            p.insert(0, first);
            out.push(p);
        }
    }
    out
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
