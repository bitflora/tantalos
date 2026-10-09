//! Portal surfaces as the solver sees them: where a portal can be walked into, where you
//! come out, who can see it, and face classes (faces that behave identically).
//!
//! Rules (ASSUMPTIONS until checked in-game):
//! - The eye is at the centre of a standable cell (64u above the floor). A face can be shot
//!   from a region if the segment from some cell's eye to a target point on the face crosses
//!   only open voxels and no face that stops shots (glass; fizzlers per `Sight`). Grating and
//!   laser fields let shots through.
//! - A wall or floor portal can be walked into from the cell in front of it, if that cell is
//!   standable. Ceiling portals can't be entered.
//! - Coming out of a portal (player or cube) you fall straight down from the cell in front
//!   of it, with no momentum. Ending in goo means death, so there is no landing.

use std::collections::BTreeMap;

use crate::geom::{step, Cell, Dir};
use crate::level::{BarrierKind, ItemKind, Level};
use crate::regions::RegionGraph;

/// How generous line of sight is. The solver's conservative and liberal modes differ here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sight {
    /// Hazards (bits of `Level::hazards`) whose fields stop portal shots.
    pub blocking_hazards: u64,
    /// Passing exactly through a voxel edge or corner: true = every way around it must be
    /// clear, false = one way is enough.
    pub strict_grazing: bool,
    /// Also aim at four points inset from the face's corners, not just its centre.
    pub corner_targets: bool,
}

impl Sight {
    pub const STRICT: Sight = Sight {
        blocking_hazards: u64::MAX,
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
    /// Indexed by region: can it shoot at least one face of this class.
    pub visible_from: Vec<bool>,
}

pub struct Portals {
    pub classes: Vec<FaceClass>,
}

impl Portals {
    /// Groups every portalable face by (entry, landing). Faces with neither are useless in
    /// Tier 0 and dropped, as are classes no region can see.
    pub fn build(level: &Level, graph: &RegionGraph, sight: Sight) -> Portals {
        let mut groups: BTreeMap<(Option<usize>, Option<Landing>), Vec<Face>> = BTreeMap::new();
        for (c, d) in level.portal_surfaces() {
            let entry = (d != Dir::PosZ && level.standable(c))
                .then(|| graph.region_of(c))
                .flatten();
            let landing = Landing::from_fall(level, graph, c);
            if entry.is_some() || landing.is_some() {
                groups.entry((entry, landing)).or_default().push((c, d));
            }
        }
        let classes = groups
            .into_iter()
            .map(|((entry, landing), faces)| {
                let visible_from = graph
                    .regions
                    .iter()
                    .map(|r| {
                        r.cells
                            .iter()
                            .any(|&eye| faces.iter().any(|&f| visible(level, eye, f, sight)))
                    })
                    .collect();
                (entry, landing, faces, visible_from)
            })
            .filter(|(.., vis): &(_, _, _, Vec<bool>)| vis.iter().any(|&v| v))
            .enumerate()
            .map(|(id, (entry, landing, faces, visible_from))| FaceClass {
                id,
                faces,
                entry,
                landing,
                visible_from,
            })
            .collect();
        Portals { classes }
    }
}

/// Whether the eye in open cell `eye` can see the centre of `face`, with strict rules.
pub fn line_of_sight(level: &Level, eye: Cell, face: Face) -> bool {
    visible(level, eye, face, Sight::STRICT)
}

/// Whether the eye in open cell `eye` can hit `face` under `sight`.
pub fn visible(level: &Level, eye: Cell, face: Face, sight: Sight) -> bool {
    let (target, d) = face;
    let n = d.vector();
    let centre: [f64; 3] = std::array::from_fn(|a| target[a] as f64 + 0.5 + 0.5 * n[a] as f64);
    if segment_clear(level, eye, centre, target, sight) {
        return true;
    }
    if !sight.corner_targets {
        return false;
    }
    let plane: Vec<usize> = (0..3).filter(|&a| a != d.axis()).collect();
    [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
        .iter()
        .any(|&(u, v)| {
            let mut p = centre;
            p[plane[0]] += 0.3 * u;
            p[plane[1]] += 0.3 * v;
            segment_clear(level, eye, p, target, sight)
        })
}

/// Whether a shot crossing from `c` to its neighbour in `d` gets through.
fn crossing_clear(level: &Level, c: Cell, d: Dir, sight: Sight) -> bool {
    let n = step(c, d);
    if level.solid(n) {
        return false;
    }
    let f = level.face(c, d);
    f.barrier != Some(BarrierKind::Glass) && f.hazards & sight.blocking_hazards == 0
}

/// Walks the voxels the segment from the centre of `eye` to `p1` crosses (Amanatides & Woo),
/// until it reaches cell `target`. Where it passes exactly through an edge or corner, the ways
/// around it are the orders of stepping the tied axes; `sight.strict_grazing` decides whether
/// all or one of them must be clear.
fn segment_clear(level: &Level, eye: Cell, p1: [f64; 3], target: Cell, sight: Sight) -> bool {
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

    while cur != target {
        let t = t_max.iter().cloned().fold(f64::INFINITY, f64::min);
        if t >= 1.0 - EPS {
            return false; // Ran out of segment without reaching the face's cell.
        }
        let dirs: Vec<Dir> = (0..3)
            .filter(|&a| t_max[a] - t < EPS)
            .map(|a| Dir::from_axis(a, v[a] > 0.0))
            .collect();
        let path_clear = |order: &[Dir]| {
            let mut c = cur;
            order.iter().all(|&d| {
                let ok = crossing_clear(level, c, d, sight);
                c = step(c, d);
                ok
            })
        };
        let orders = permutations(&dirs);
        let clear = if sight.strict_grazing {
            orders.iter().all(|o| path_clear(o))
        } else {
            orders.iter().any(|o| path_clear(o))
        };
        if !clear {
            return false;
        }
        for d in dirs {
            cur = step(cur, d);
            t_max[d.axis()] += t_delta[d.axis()];
        }
    }
    true
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
