//! Momentum: flings through portals (plan.md tier 2).
//!
//! Rules (ASSUMPTIONS until measured in-game):
//! - Dropping `h` voxels off a ledge onto a floor portal, the player goes in at
//!   sqrt(2 g h) (g = 600 u/s², capped at `MAX_SPEED`) and comes out of the other portal at that
//!   speed along its normal; out of a deployed angled panel, along the panel's tilted normal.
//!   Walking into a portal is too slow to matter: you drop out of the other one (`portals`).
//! - Out of a portal the player is a point at their feet: just in front of a wall portal's
//!   centre, 0.22 voxel below it; on a floor portal; 0.4 voxel under a ceiling one. It flies a
//!   parabola until its feet reach a floor (landing there), hits a wall (losing its speed
//!   along that axis and sliding on), or a ceiling (losing its upward speed). Goo is death.
//!   Glass and grating stop it like walls; fields it flies through apply as when walking.
//!   A faith plate it reaches launches it on.
//! - `Mode::Conservative` trusts a fling only if speeds 10% either side land in the same
//!   region; `Mode::Liberal` takes every landing from 70% to 110% of the speed, with up to
//!   0.3 voxel of air control sideways.
//! - Floor portal to floor portal keeps horizontal speed (play-tested 2026-11-11, "Very easy
//!   challenge", "Fall with cube"): jumping off a ledge into a floor portal at up to walking
//!   speed, the player comes up out of an upward portal at the fall's speed plus that much
//!   sideways, along an axis or at a 1:2 slope (the portals' turn is the player's choice). `carried`
//!   gives these trials; conservative rules trust a direction and speed only if 10% of walking
//!   speed either side, and the fall's speed 10% either side, all land the same way.

use crate::geom::{step, Cell, Dir};
use crate::level::Level;

/// Gravity in voxels per second squared (600 u/s²).
pub const G: f64 = 600.0 / 128.0;
/// Fastest the player goes, voxels per second (2000 u/s).
pub const MAX_SPEED: f64 = 2000.0 / 128.0;
const DT: f64 = 1.0 / 128.0;
const MAX_TIME: f64 = 10.0;

/// Speed in voxels per second after falling `h` voxels.
pub fn dive_speed(h: i32) -> f64 {
    (2.0 * G * h as f64).sqrt().min(MAX_SPEED)
}

/// The way out of a portal on face `(cell, dir)` (`dir` toward the solid side), tilted by an
/// angled panel `(toward, degrees)` if deployed.
pub fn exit_vector(dir: Dir, tilt: Option<(Dir, i32)>) -> [f64; 3] {
    let n = dir.opposite().vector().map(f64::from);
    match tilt {
        None => n,
        Some((t, deg)) => {
            let (s, c) = (deg as f64).to_radians().sin_cos();
            let t = t.vector().map(f64::from);
            std::array::from_fn(|a| n[a] * c + t[a] * s)
        }
    }
}

/// Where the player's feet start coming out of a portal on face `(cell, dir)`.
pub fn exit_point(cell: Cell, dir: Dir) -> [f64; 3] {
    let mut p = [
        cell[0] as f64 + 0.5,
        cell[1] as f64 + 0.5,
        cell[2] as f64,
    ];
    match dir {
        Dir::NegZ => p[2] += 0.02,
        Dir::PosZ => p[2] += 0.4,
        d => {
            let v = d.vector();
            p[0] += 0.35 * v[0] as f64;
            p[1] += 0.35 * v[1] as f64;
            p[2] += 0.22;
        }
    }
    p
}

/// Where a body with its feet at `p` (voxels) moving at `v` (voxels per second) comes to rest,
/// and the fields it crosses. `None` if it ends in goo or doesn't settle.
pub fn fly(level: &Level, p: [f64; 3], v: [f64; 3]) -> Option<(Cell, u64)> {
    let cell_of = |p: [f64; 3]| -> Cell { std::array::from_fn(|a| p[a].floor() as i32) };
    let (mut p, mut v) = (p, v);
    let mut cur = cell_of(p);
    if level.solid(cur) {
        return None;
    }
    let mut hazards = 0;
    let mut t = 0.0;
    while t < MAX_TIME {
        t += DT;
        v[2] -= G * DT;
        let mut next = p;
        for a in 0..3 {
            next[a] += v[a] * DT;
        }
        // Cross into new cells one axis at a time, z last so walls are hit before floors.
        for a in [0, 1, 2] {
            let target = next[a].floor() as i32;
            while target != cur[a] {
                let d = Dir::from_axis(a, target > cur[a]);
                let n = step(cur, d);
                let face = level.face(cur, d);
                if level.solid(n) || face.barrier.is_some() {
                    if d == Dir::NegZ {
                        return settle(level, cur, hazards);
                    }
                    v[a] = 0.0;
                    next[a] = if d.is_positive() {
                        cur[a] as f64 + 0.999
                    } else {
                        cur[a] as f64 + 0.001
                    };
                    break;
                }
                hazards |= face.hazards;
                cur = n;
                if level.goo(cur) {
                    return None;
                }
                if level.plate_at(cur).is_some() {
                    return level.fall_path(cur).map(|(f, h)| (f, hazards | h));
                }
            }
        }
        p = next;
        // Resting on a floor with no speed left upward.
        if v[2] <= 0.0 && p[2] - cur[2] as f64 <= 0.001 && level.floor_under(cur) {
            return settle(level, cur, hazards);
        }
    }
    None
}

/// Coming down in `c`: there if it can be stood in, else wherever falling from it ends.
fn settle(level: &Level, c: Cell, hazards: u64) -> Option<(Cell, u64)> {
    if level.standable(c) {
        Some((c, hazards))
    } else {
        level.fall_path(c).map(|(f, h)| (f, hazards | h))
    }
}

/// Trials of a fling at `speed` out of face `(cell, dir)`: (speed factor, sideways offset).
pub fn trials(liberal: bool) -> Vec<(f64, f64)> {
    if liberal {
        let mut out = Vec::new();
        for f in [0.7, 0.85, 1.0, 1.1] {
            for s in [-0.3, 0.0, 0.3] {
                out.push((f, s));
            }
        }
        out
    } else {
        vec![(0.9, 0.0), (1.0, 0.0), (1.1, 0.0)]
    }
}

/// Every landing of the trials of a fling out of face `(cell, dir)` at `speed` along `e`.
pub fn landings(
    level: &Level,
    cell: Cell,
    dir: Dir,
    e: [f64; 3],
    speed: f64,
    liberal: bool,
) -> Vec<Option<(Cell, u64)>> {
    let p0 = exit_point(cell, dir);
    // Sideways: horizontal and across the way out.
    let side = if e[0].abs() + e[1].abs() > 1e-6 {
        let n = e[0].hypot(e[1]);
        [-e[1] / n, e[0] / n, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    trials(liberal)
        .into_iter()
        .map(|(f, s)| {
            let p: [f64; 3] = std::array::from_fn(|a| p0[a] + side[a] * s);
            let v: [f64; 3] = std::array::from_fn(|a| e[a] * speed * f);
            fly(level, p, v)
        })
        .collect()
}

/// Trials of an upward fling (out of a floor portal, `dir` = `NegZ`, not tilted) at `speed`
/// carrying horizontal speed into the portal and out again, up to `carry` (voxels per second),
/// in groups: conservative rules trust a group's landing only if all its trials agree, liberal
/// rules take every landing. Empty for any other face.
pub fn carried(
    level: &Level,
    cell: Cell,
    dir: Dir,
    speed: f64,
    carry: f64,
    liberal: bool,
) -> Vec<Vec<Option<(Cell, u64)>>> {
    if dir != Dir::NegZ || carry <= 0.0 {
        return Vec::new();
    }
    let p0 = exit_point(cell, dir);
    let dirs = carry_dirs();
    let fly_at = |u: (f64, f64), h: f64, f: f64| {
        fly(level, p0, [u.0 * h * carry, u.1 * h * carry, speed * f])
    };
    let mut out = Vec::new();
    for u in dirs {
        if liberal {
            let mut group = Vec::new();
            for h in [0.25, 0.5, 0.75, 1.0] {
                for f in [0.7, 0.85, 1.0, 1.1] {
                    group.push(fly_at(u, h, f));
                }
            }
            out.push(group);
        } else {
            for h in [0.3, 0.5, 0.7, 0.9] {
                let mut group = Vec::new();
                for dh in [-0.1, 0.0, 0.1] {
                    for f in [0.9, 1.0, 1.1] {
                        group.push(fly_at(u, h + dh, f));
                    }
                }
                out.push(group);
            }
        }
    }
    out
}

/// The ways a carried fling can go sideways (`carried`). Along the axes, and at 1:2 slopes
/// rather than 45°: from a cell's middle those never pass exactly through a voxel corner, where
/// `fly`'s x-before-y crossing would make the landing depend on which way the level is turned.
pub fn carry_dirs() -> [(f64, f64); 12] {
    let (a, b) = (2.0 / 5f64.sqrt(), 1.0 / 5f64.sqrt());
    [
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (a, b),
        (a, -b),
        (-a, b),
        (-a, -b),
        (b, a),
        (b, -a),
        (-b, a),
        (-b, -a),
    ]
}

/// The middle trials of `carried`'s groups at the fall's own speed: sideways velocities
/// (voxels per second), for replaying one (`route`).
pub fn carry_aims(carry: f64, liberal: bool) -> Vec<[f64; 2]> {
    let hs: &[f64] = if liberal { &[0.25, 0.5, 0.75, 1.0] } else { &[0.3, 0.5, 0.7, 0.9] };
    carry_dirs()
        .into_iter()
        .flat_map(|u| hs.iter().map(move |h| [u.0 * h * carry, u.1 * h * carry]))
        .collect()
}
