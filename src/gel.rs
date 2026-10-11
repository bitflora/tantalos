//! Gel (plan.md tier 3): where paint lands and the moves painted floors allow.
//!
//! Rules (ASSUMPTIONS until measured in-game):
//! - Gel paints the face it lands on and the faces of the same surface within `radius`
//!   (Manhattan) of it. A dropper's stream falls straight down from the dropper.
//! - Repulsion (blue) floor: from a painted cell the player can bounce up `1..=bounce_height`
//!   voxels and drift up to `bounce_reach` cells sideways, through open space, onto a floor.
//! - Propulsion (orange) floor: a straight run of `leap_run` painted cells ending at a drop
//!   lets the player leap across up to `leap_gap` cells to a floor at the same height.

use crate::geom::{step, Cell, Dir};
use crate::level::{FaceInfo, Level};
use crate::portals::Face;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GelRules {
    pub radius: i32,
    pub bounce_height: i32,
    pub bounce_reach: i32,
    pub leap_run: i32,
    pub leap_gap: i32,
}

/// The faces paint lands on around `face` (open cell, direction to the solid side): the same
/// surface within `radius`, wherever it is a surface (solid behind, open in front, no goo).
pub fn spot(level: &Level, face: Face, radius: i32) -> Vec<Face> {
    let (c, d) = face;
    let plane: Vec<usize> = (0..3).filter(|&a| a != d.axis()).collect();
    let mut out = Vec::new();
    for du in -radius..=radius {
        for dv in -radius..=radius {
            if du.abs() + dv.abs() > radius {
                continue;
            }
            let mut n = c;
            n[plane[0]] += du;
            n[plane[1]] += dv;
            if !level.solid(n) && level.solid(step(n, d)) && !level.goo(n) {
                out.push((n, d));
            }
        }
    }
    out
}

/// Where a bounce off repulsion gel under standable cell `c` can land, with the hazard fields
/// crossed on the way.
pub fn bounces(level: &Level, c: Cell, rules: GelRules) -> Vec<(Cell, u64)> {
    let mut out = Vec::new();
    for dz in 1..=rules.bounce_height {
        // Room to rise straight up, with headroom over the landing height.
        let column_clear = (1..=dz + 1).all(|k| {
            let below = [c[0], c[1], c[2] + k - 1];
            let up = [c[0], c[1], c[2] + k];
            !level.solid(up) && level.face(below, Dir::PosZ).barrier.is_none()
        });
        if !column_clear {
            break;
        }
        let top = [c[0], c[1], c[2] + dz];
        for dx in -rules.bounce_reach..=rules.bounce_reach {
            for dy in -rules.bounce_reach..=rules.bounce_reach {
                let reach = dx.abs() + dy.abs();
                if reach == 0 || reach > rules.bounce_reach {
                    continue;
                }
                let t = [top[0] + dx, top[1] + dy, top[2]];
                if !level.standable(t) {
                    continue;
                }
                if let Some(h) = drift(level, top, dx, dy) {
                    out.push((t, h));
                }
            }
        }
    }
    out
}

/// Moving sideways from `from` by (dx, dy) at its height and the one above, x first or y
/// first: the hazards crossed by a clear way, or `None` if both are blocked.
fn drift(level: &Level, from: Cell, dx: i32, dy: i32) -> Option<u64> {
    let xs = std::iter::repeat_n(Dir::from_axis(0, dx > 0), dx.unsigned_abs() as usize);
    let ys = std::iter::repeat_n(Dir::from_axis(1, dy > 0), dy.unsigned_abs() as usize);
    let x_first: Vec<Dir> = xs.clone().chain(ys.clone()).collect();
    let y_first: Vec<Dir> = ys.chain(xs).collect();
    [x_first, y_first].into_iter().find_map(|way| {
        let mut c = from;
        let mut hazards = 0;
        for d in way {
            for lift in 0..2 {
                let at = [c[0], c[1], c[2] + lift];
                let f = level.face(at, d);
                if level.solid(step(at, d)) || f.barrier.is_some() {
                    return None;
                }
                hazards |= f.hazards;
            }
            c = step(c, d);
        }
        Some(hazards)
    })
}

/// A leap off propulsion gel: running along `d` and taking off from standable cell `c`.
/// Returns the run's cells (all need paint), the landing cell and the hazards crossed.
pub fn leap(level: &Level, c: Cell, d: Dir, rules: GelRules) -> Option<(Vec<Cell>, Cell, u64)> {
    let mut run = vec![c];
    for _ in 1..rules.leap_run {
        let prev = *run.last().unwrap();
        let back = step(prev, d.opposite());
        if !level.standable(back) || level.face(prev, d.opposite()) != FaceInfo::default() {
            return None;
        }
        run.push(back);
    }
    let first = step(c, d);
    if level.standable(first) || level.solid(first) {
        return None; // Not a gap: walking does it, or a wall.
    }
    let mut at = c;
    let mut hazards = 0;
    for _ in 0..=rules.leap_gap {
        for lift in 0..2 {
            let p = [at[0], at[1], at[2] + lift];
            let f = level.face(p, d);
            if level.solid(step(p, d)) || f.barrier.is_some() {
                return None;
            }
            hazards |= f.hazards;
        }
        at = step(at, d);
        if at != first && level.standable(at) {
            return Some((run, at, hazards));
        }
    }
    None
}
