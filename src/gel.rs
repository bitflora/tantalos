//! Gel (plan.md tier 3): where paint lands and the moves painted floors allow.
//!
//! Rules (ASSUMPTIONS until measured in-game; the radius and leap were measured 2026-10-18,
//! see `Rules::gel`):
//! - Gel paints the face it lands on and the faces of the same surface within `radius`
//!   (Manhattan) of it. A dropper's stream falls straight down from the dropper.
//! - Repulsion (blue) floor: from a painted cell the player can bounce up `1..=bounce_height`
//!   voxels and drift up to `bounce_reach` cells sideways, through open space, onto a floor.
//! - Propulsion (orange) floor: a straight run of `leap_run` painted cells ending at a drop
//!   lets the player leap across up to `leap_gap` cells to a floor at the same height.
//! - A stream poured into a floor portal comes out of the other one at the speed it fell
//!   (`stream_speed`); out of a wall portal it flies on in an arc (`arc`) until it hits a
//!   surface. Blobs fall like the player (sv_gravity 600) without drag: unmeasured.

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

/// Units per voxel; gravity in units/s².
const VOXEL: f64 = 128.0;
const GRAVITY: f64 = 600.0;
/// How far under the ceiling a dropper lets its blobs go (unmeasured).
const DROPPER_DEPTH: f64 = 32.0;

/// How fast a dropper's stream from cell `top` (under the ceiling) hits the floor of `land`,
/// in units/s.
pub fn stream_speed(top: Cell, land: Cell) -> f64 {
    let height = (top[2] + 1 - land[2]) as f64 * VOXEL - DROPPER_DEPTH;
    (2.0 * GRAVITY * height.max(0.0)).sqrt()
}

/// The face a stream coming out of the portal on `face` (open cell, direction to the solid
/// side) at `speed` (units/s, straight out of the face) hits: the open cell it hits from and
/// the direction to the surface. `None` if it ends in goo or against glass or grating, or
/// out of a floor portal (it falls back in).
pub fn arc(level: &Level, (c, d): Face, speed: f64) -> Option<Face> {
    if d == Dir::NegZ {
        return None;
    }
    let out = d.opposite().vector().map(|v| v as f64);
    let mut p = [0, 1, 2].map(|a| (c[a] as f64 + 0.5 + 0.5 * -out[a]) * VOXEL);
    let mut v = out.map(|o| o * speed);
    let mut at = c;
    const DT: f64 = 1.0 / 256.0;
    for _ in 0..256 * 8 {
        v[2] -= GRAVITY * DT;
        let next = [0, 1, 2].map(|a| p[a] + v[a] * DT);
        let cell = next.map(|x| (x / VOXEL).floor() as i32);
        if cell != at {
            // Through the faces it crosses, one axis at a time (corners are rare enough).
            for a in 0..3 {
                if cell[a] == at[a] {
                    continue;
                }
                let dir = Dir::from_axis(a, cell[a] > at[a]);
                let n = step(at, dir);
                if level.solid(n) {
                    if dir == Dir::NegZ && level.goo(at) {
                        return None;
                    }
                    return Some((at, dir));
                }
                if level.face(at, dir).barrier.is_some() {
                    return None;
                }
                at = n;
            }
        }
        p = next;
        if level.goo(at) && p[2] < at[2] as f64 * VOXEL + VOXEL / 4.0 {
            return None;
        }
    }
    None
}
