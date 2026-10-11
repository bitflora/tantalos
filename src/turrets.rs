//! Where turrets can shoot (plan.md tier 2).
//!
//! Rules (ASSUMPTIONS until measured in-game):
//! - A turret sees and shoots within `half_angle` degrees either side of its facing
//!   (horizontally) and `range` voxels, from 0.5 voxel above its floor to the middle of the
//!   player's cell.
//! - Solid voxels and glass stop bullets; grating does only if `grating_blocks`. Fizzlers and
//!   laser fields don't.
//! - Standing anywhere a live turret can shoot is deadly. A turret is out for good once
//!   knocked over: by the player walking up to it from a cell it can't shoot, by a cube landing
//!   on it, or by a portal opening in the floor under it (it falls through).

use crate::geom::{step, Cell, Dir};
use crate::level::{BarrierKind, Level, Turret};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TurretSight {
    pub half_angle: f64,
    pub range: f64,
    pub grating_blocks: bool,
}

/// Bit `i` of the result: turret `i` of `level.turrets` can shoot a player standing in `c`.
pub fn danger(level: &Level, sight: TurretSight, c: Cell) -> u32 {
    level
        .turrets
        .iter()
        .enumerate()
        .filter(|(_, t)| sees(level, sight, t, c))
        .fold(0, |m, (i, _)| m | 1 << i)
}

fn sees(level: &Level, sight: TurretSight, t: &Turret, c: Cell) -> bool {
    if c == t.cell {
        return false;
    }
    let eye = [
        t.cell[0] as f64 + 0.5,
        t.cell[1] as f64 + 0.5,
        t.cell[2] as f64 + 0.5,
    ];
    let target = [c[0] as f64 + 0.5, c[1] as f64 + 0.5, c[2] as f64 + 0.5];
    let v: [f64; 3] = std::array::from_fn(|a| target[a] - eye[a]);
    let dist = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let flat = (v[0] * v[0] + v[1] * v[1]).sqrt();
    if dist > sight.range || flat == 0.0 {
        return false;
    }
    let yaw = t.yaw.to_radians();
    let cos = (v[0] * yaw.cos() + v[1] * yaw.sin()) / flat;
    if cos < sight.half_angle.to_radians().cos() - 1e-9 {
        return false;
    }
    clear_line(level, eye, target, sight.grating_blocks)
}

/// Whether a bullet from `a` reaches `b`. Samples the segment finely; where it moves to a new
/// cell along more than one axis at once, one clear order of steps is enough.
fn clear_line(level: &Level, a: [f64; 3], b: [f64; 3], grating_blocks: bool) -> bool {
    let len = (0..3).map(|i| (b[i] - a[i]).powi(2)).sum::<f64>().sqrt();
    let n = (len * 16.0).ceil().max(1.0) as usize;
    let cell_of = |p: [f64; 3]| -> Cell { std::array::from_fn(|i| p[i].floor() as i32) };
    let stops = |c: Cell, d: Dir| {
        let f = level.face(c, d);
        level.solid(step(c, d))
            || f.barrier == Some(BarrierKind::Glass)
            || (grating_blocks && f.barrier == Some(BarrierKind::Grating))
    };
    let mut cur = cell_of(a);
    for k in 1..=n {
        let t = k as f64 / n as f64;
        let p: [f64; 3] = std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t);
        let next = cell_of(p);
        if next == cur {
            continue;
        }
        let dirs: Vec<Dir> = (0..3)
            .filter(|&i| next[i] != cur[i])
            .map(|i| Dir::from_axis(i, next[i] > cur[i]))
            .collect();
        let one_way_clear = orders(&dirs).iter().any(|order| {
            let mut c = cur;
            order.iter().all(|&d| {
                let ok = !stops(c, d);
                c = step(c, d);
                ok
            })
        });
        if !one_way_clear {
            return false;
        }
        cur = next;
    }
    true
}

fn orders(dirs: &[Dir]) -> Vec<Vec<Dir>> {
    if dirs.len() <= 1 {
        return vec![dirs.to_vec()];
    }
    let mut out = Vec::new();
    for i in 0..dirs.len() {
        let mut rest = dirs.to_vec();
        let first = rest.remove(i);
        for mut o in orders(&rest) {
            o.insert(0, first);
            out.push(o);
        }
    }
    out
}
