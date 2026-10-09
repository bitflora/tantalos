//! Walkable regions: sets of standable cells mutually reachable on foot, plus the
//! one-way moves between them. This is the position abstraction the solver searches over.
//!
//! Movement rules (ASSUMPTIONS until measured in-game):
//! - Walking connects 4-neighbour standable cells at the same height, unless glass or grating
//!   is on the face between them. A fizzler or laser field there makes it a `Walk` edge that
//!   carries the field, so the solver can apply it (laser fields only when on).
//! - Stepping off a ledge falls straight down; landing in goo is death (no edge).
//!   Long-fall boots mean any drop height is survivable. Glass and grating floors stop a fall.
//! - A one-voxel (128u) ledge can't be jumped; it is a `Step` edge that needs a cube.
//!   Two or more voxels can't be climbed at all.

use std::collections::BTreeSet;

use crate::geom::{step, Cell, Dir};
use crate::level::{FaceInfo, Level};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    pub id: usize,
    pub z: i32,
    pub cells: Vec<Cell>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Move {
    /// Walk through a fizzler or laser field to a neighbouring cell at the same height.
    Walk,
    /// Fall `height` voxels.
    Drop { height: i32 },
    /// Climb a one-voxel ledge; needs something to stand on (a cube).
    Step,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub kind: Move,
    /// Hazard fields crossed on the way (bits index `Level::hazards`).
    pub hazards: u64,
}

pub struct RegionGraph {
    pub regions: Vec<Region>,
    pub edges: Vec<Edge>,
    size: [i32; 3],
    cell_region: Vec<Option<usize>>,
}

impl RegionGraph {
    pub fn build(level: &Level) -> RegionGraph {
        let size = level.size;
        let mut g = RegionGraph {
            regions: Vec::new(),
            edges: Vec::new(),
            size,
            cell_region: vec![None; size.iter().product::<i32>() as usize],
        };

        for c in level.cells() {
            if !level.standable(c) || g.region_of(c).is_some() {
                continue;
            }
            let id = g.regions.len();
            let mut cells = Vec::new();
            let mut stack = vec![c];
            while let Some(c) = stack.pop() {
                if !level.standable(c) || g.region_of(c).is_some() {
                    continue;
                }
                let i = g.idx(c).unwrap();
                g.cell_region[i] = Some(id);
                cells.push(c);
                for d in Dir::HORIZONTAL {
                    if level.face(c, d) == FaceInfo::default() {
                        stack.push(step(c, d));
                    }
                }
            }
            cells.sort_by_key(|c| (c[2], c[1], c[0]));
            g.regions.push(Region { id, z: c[2], cells });
        }

        let mut edges = BTreeSet::new();
        for r in &g.regions {
            for &c in &r.cells {
                for d in Dir::HORIZONTAL {
                    let n = step(c, d);
                    let face = level.face(c, d);
                    let mut edge = |to, kind, hazards| {
                        edges.insert(Edge {
                            from: r.id,
                            to,
                            kind,
                            hazards,
                        });
                    };
                    if level.solid(n) {
                        let (head, top) = (step(c, Dir::PosZ), step(n, Dir::PosZ));
                        let over = level.face(head, d);
                        if level.in_interior(n)
                            && level.standable(top)
                            && !level.solid(head)
                            && over.barrier.is_none()
                        {
                            edge(g.region_of(top).unwrap(), Move::Step, over.hazards);
                        }
                    } else if face.barrier.is_some() {
                        // Glass or grating: no way through.
                    } else if level.standable(n) {
                        if face.hazards != 0 {
                            edge(g.region_of(n).unwrap(), Move::Walk, face.hazards);
                        }
                    } else if let Some((f, fallen)) = level.fall_path(n) {
                        edge(
                            g.region_of(f).unwrap(),
                            Move::Drop {
                                height: n[2] - f[2],
                            },
                            face.hazards | fallen,
                        );
                    }
                }
            }
        }
        g.edges = edges.into_iter().collect();
        g
    }

    fn idx(&self, c: Cell) -> Option<usize> {
        let inside = (0..3).all(|a| c[a] >= 0 && c[a] < self.size[a]);
        inside.then(|| ((c[2] * self.size[1] + c[1]) * self.size[0] + c[0]) as usize)
    }

    pub fn region_of(&self, c: Cell) -> Option<usize> {
        self.idx(c).and_then(|i| self.cell_region[i])
    }

    pub fn edges_from(&self, r: usize) -> impl Iterator<Item = &Edge> {
        self.edges.iter().filter(move |e| e.from == r)
    }

    /// Regions reachable from `from` by walking and dropping only (no portals, no cubes).
    pub fn reachable_on_foot(&self, from: usize) -> Vec<bool> {
        let mut seen = vec![false; self.regions.len()];
        let mut stack = vec![from];
        while let Some(r) = stack.pop() {
            if std::mem::replace(&mut seen[r], true) {
                continue;
            }
            stack.extend(
                self.edges_from(r)
                    .filter(|e| matches!(e.kind, Move::Drop { .. }))
                    .map(|e| e.to),
            );
        }
        seen
    }

    /// `A`..`Z`, then `a`..`z`, then `?`.
    pub fn label(id: usize) -> char {
        match id {
            0..=25 => (b'A' + id as u8) as char,
            26..=51 => (b'a' + (id - 26) as u8) as char,
            _ => '?',
        }
    }
}
