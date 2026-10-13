//! Where turrets can shoot (plan.md tier 2).
//!
//! Rules (ASSUMPTIONS until measured in-game):
//! - A turret sees and shoots within `half_angle` degrees either side of the way it looks
//!   (`Turret::look_yaw`: its .p2c Facing + 180°, checked in the game) horizontally and `range`
//!   voxels, from 0.5 voxel above its floor to the middle of the player's cell.
//! - Solid voxels and glass stop bullets; grating does only if `grating_blocks`. Fizzlers and
//!   laser fields don't.
//! - Standing anywhere a live turret can shoot is deadly. A turret is out for good once
//!   knocked over: by the player walking up to it from a cell it can't shoot, by a cube landing
//!   on it, or by a portal opening in the floor under it (it falls through).

use crate::geom::{step, step_orders, Cell, Dir};
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
    let yaw = t.look_yaw().to_radians();
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
        let mut dirs = [Dir::PosX; 3];
        let mut moved = 0;
        for i in 0..3 {
            if next[i] != cur[i] {
                dirs[moved] = Dir::from_axis(i, next[i] > cur[i]);
                moved += 1;
            }
        }
        let one_way_clear = step_orders(moved).iter().any(|order| {
            let mut c = cur;
            order[..moved].iter().all(|&i| {
                let ok = !stops(c, dirs[i]);
                c = step(c, dirs[i]);
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
