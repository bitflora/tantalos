//! Portal surfaces as the solver sees them: where a portal can be walked into, where you
//! come out, who can see it, and face classes (faces that behave identically).
//!
//! Tier-0 rules (ASSUMPTIONS until checked in-game):
//! - The eye is at the centre of a standable cell (64u above the floor). A face can be shot
//!   from a region if the segment from some cell's eye to the face centre crosses only open
//!   voxels. Grazing an edge or corner of a solid voxel counts as blocked.
//! - A wall or floor portal can be walked into from the cell in front of it, if that cell is
//!   standable. Ceiling portals can't be entered.
//! - Coming out of a portal (player or cube) you fall straight down from the cell in front
//!   of it, with no momentum. Ending in goo means death, so there is no landing.

use std::collections::BTreeMap;

use crate::geom::{Cell, Dir};
use crate::level::{ItemKind, Level};
use crate::regions::RegionGraph;

/// A face, as (open cell, direction to the solid side).
pub type Face = (Cell, Dir);

/// Where something coming to rest at a cell ends up.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Landing {
    pub region: usize,
    /// Index of the floor button at the landing cell, if any. A cube landing here presses it.
    pub button: Option<i32>,
}

impl Landing {
    /// Falls from open cell `c`; `None` if that ends in goo.
    pub fn from_fall(level: &Level, graph: &RegionGraph, c: Cell) -> Option<Landing> {
        let f = level.fall(c)?;
        Some(Landing {
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
    pub fn build(level: &Level, graph: &RegionGraph) -> Portals {
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
                            .any(|&eye| faces.iter().any(|&f| line_of_sight(level, eye, f)))
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

/// Whether the eye in open cell `eye` can see the centre of `face`. Walks the voxels the
/// segment crosses (Amanatides & Woo); where it passes exactly through an edge or corner,
/// every voxel touching that point must be open.
pub fn line_of_sight(level: &Level, eye: Cell, face: Face) -> bool {
    const EPS: f64 = 1e-9;
    let (target, d) = face;
    let p0 = eye.map(|v| v as f64 + 0.5);
    let n = d.vector();
    let p1: [f64; 3] = std::array::from_fn(|a| target[a] as f64 + 0.5 + 0.5 * n[a] as f64);
    let v: [f64; 3] = std::array::from_fn(|a| p1[a] - p0[a]);

    let mut cur = eye;
    let sign: [i32; 3] = v.map(|x| if x > 0.0 { 1 } else { -1 });
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
        let axes: Vec<usize> = (0..3).filter(|&a| t_max[a] - t < EPS).collect();
        // Every voxel touching the crossing point, i.e. each subset of the tied axes stepped.
        for mask in 1..(1u32 << axes.len()) {
            let mut c = cur;
            for (i, &a) in axes.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    c[a] += sign[a];
                }
            }
            if level.solid(c) {
                return false;
            }
        }
        for &a in &axes {
            cur[a] += sign[a];
            t_max[a] += t_delta[a];
        }
    }
    true
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
