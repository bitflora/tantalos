//! Voxel coordinates, the six axis directions, and Source engine angle conversion.

/// A voxel position `[x, y, z]`; one voxel is 128 Hammer units, +Z is up.
pub type Cell = [i32; 3];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Dir {
    NegX,
    PosX,
    NegY,
    PosY,
    NegZ,
    PosZ,
}

impl Dir {
    pub const ALL: [Dir; 6] = [
        Dir::NegX,
        Dir::PosX,
        Dir::NegY,
        Dir::PosY,
        Dir::NegZ,
        Dir::PosZ,
    ];
    pub const HORIZONTAL: [Dir; 4] = [Dir::NegX, Dir::PosX, Dir::NegY, Dir::PosY];

    pub fn axis(self) -> usize {
        match self {
            Dir::NegX | Dir::PosX => 0,
            Dir::NegY | Dir::PosY => 1,
            Dir::NegZ | Dir::PosZ => 2,
        }
    }

    pub fn is_positive(self) -> bool {
        matches!(self, Dir::PosX | Dir::PosY | Dir::PosZ)
    }

    pub fn vector(self) -> [i32; 3] {
        let mut v = [0; 3];
        v[self.axis()] = if self.is_positive() { 1 } else { -1 };
        v
    }

    pub fn from_axis(axis: usize, positive: bool) -> Dir {
        match (axis, positive) {
            (0, false) => Dir::NegX,
            (0, true) => Dir::PosX,
            (1, false) => Dir::NegY,
            (1, true) => Dir::PosY,
            (2, false) => Dir::NegZ,
            _ => Dir::PosZ,
        }
    }

    /// `self × other`, or `None` if they are parallel.
    pub fn cross(self, other: Dir) -> Option<Dir> {
        let (a, b) = (self.vector(), other.vector());
        let c = [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ];
        let axis = (0..3).find(|&i| c[i] != 0)?;
        Some(Dir::from_axis(axis, c[axis] > 0))
    }

    pub fn opposite(self) -> Dir {
        match self {
            Dir::NegX => Dir::PosX,
            Dir::PosX => Dir::NegX,
            Dir::NegY => Dir::PosY,
            Dir::PosY => Dir::NegY,
            Dir::NegZ => Dir::PosZ,
            Dir::PosZ => Dir::NegZ,
        }
    }
}

pub fn step(c: Cell, d: Dir) -> Cell {
    let v = d.vector();
    [c[0] + v[0], c[1] + v[1], c[2] + v[2]]
}

/// Every order of taking `k` steps (up to 3) at once, as indices into the steps: the first
/// `k` of each entry.
pub fn step_orders(k: usize) -> &'static [[usize; 3]] {
    const ONE: [[usize; 3]; 1] = [[0, 0, 0]];
    const TWO: [[usize; 3]; 2] = [[0, 1, 0], [1, 0, 0]];
    const THREE: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    match k {
        0 | 1 => &ONE,
        2 => &TWO,
        _ => &THREE,
    }
}

/// The world axis that local axis `v` (Source: +x forward, +y left, +z up) points along after
/// rotating by Source angles `[pitch, yaw, roll]` in degrees. `None` if not axis-aligned.
pub fn rotate(angles: [f64; 3], v: [i32; 3]) -> Option<Dir> {
    let [p, y, r] = angles.map(f64::to_radians);
    let (sp, cp, sy, cy, sr, cr) = (p.sin(), p.cos(), y.sin(), y.cos(), r.sin(), r.cos());
    let forward = [cp * cy, cp * sy, -sp];
    let left = [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, sr * cp];
    let up = [cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp];
    let w: [f64; 3] = std::array::from_fn(|i| {
        v[0] as f64 * forward[i] + v[1] as f64 * left[i] + v[2] as f64 * up[i]
    });
    let axis = (0..3).find(|&a| w[a].abs() > 0.99)?;
    if (0..3).any(|a| a != axis && w[a].abs() > 0.01) {
        return None;
    }
    Some(Dir::from_axis(axis, w[axis] > 0.0))
}

/// The axis an item's "up" points along for Source angles `[pitch, yaw, roll]` in degrees.
/// For Puzzle Maker items this is the surface normal: floor items are +Z, ceiling items -Z,
/// wall items point away from the wall they are mounted on. `None` if not axis-aligned.
pub fn up_from_angles(angles: [f64; 3]) -> Option<Dir> {
    rotate(angles, [0, 0, 1])
}

/// Angles (multiples of 90 degrees) that turn local axis `a` to `wa` and local axis `b` to `wb`.
/// Searches pitch first so wall items come out like the editor's (pitch +-90).
pub fn angles_mapping(a: [i32; 3], wa: Dir, b: [i32; 3], wb: Dir) -> [i32; 3] {
    angles_mapping_pitches([0, -90, 90, 180], a, wa, b, wb)
}

/// `angles_mapping`, trying pitches in the given order.
pub fn angles_mapping_pitches(
    pitches: [i32; 4],
    a: [i32; 3],
    wa: Dir,
    b: [i32; 3],
    wb: Dir,
) -> [i32; 3] {
    const STEPS: [i32; 4] = [0, -90, 90, 180];
    for p in pitches {
        for y in STEPS {
            for r in STEPS {
                let ang = [p, y, r].map(f64::from);
                if rotate(ang, a) == Some(wa) && rotate(ang, b) == Some(wb) {
                    return [p, y, r];
                }
            }
        }
    }
    panic!("no rotation maps {a:?}->{wa:?} and {b:?}->{wb:?}");
}

/// Angles the Puzzle Maker uses for an item whose up is `normal` (inverse of `up_from_angles`).
/// Walls use pitch -90 like the editor's doors; ceilings use the cube dropper's `0 180 180`.
pub fn angles_for_up(normal: Dir) -> [i32; 3] {
    match normal {
        Dir::PosZ => [0, 0, 0],
        Dir::NegZ => [0, 180, 180],
        Dir::PosX => [-90, 180, 0],
        Dir::NegX => [-90, 0, 0],
        Dir::PosY => [-90, -90, 0],
        Dir::NegY => [-90, 90, 0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn angles_seen_in_real_files() {
        assert_eq!(up_from_angles([0.0, 0.0, 0.0]), Some(Dir::PosZ));
        assert_eq!(up_from_angles([0.0, 180.0, 180.0]), Some(Dir::NegZ)); // cube dropper
        assert_eq!(up_from_angles([-90.0, -90.0, 0.0]), Some(Dir::PosY)); // entry door at y=0
        assert_eq!(up_from_angles([90.0, 180.0, 0.0]), Some(Dir::NegX)); // light panel
        assert_eq!(up_from_angles([45.0, 0.0, 0.0]), None);
    }

    #[test]
    fn cross_is_right_handed() {
        assert_eq!(Dir::PosX.cross(Dir::PosY), Some(Dir::PosZ));
        assert_eq!(Dir::PosY.cross(Dir::PosX), Some(Dir::NegZ));
        assert_eq!(Dir::PosZ.cross(Dir::NegZ), None);
    }

    #[test]
    fn rotate_matches_real_items() {
        // Glass 46 in the style map: forward +Y, left -Z, up -X.
        let a = [0.0, 90.0, -90.0];
        assert_eq!(rotate(a, [1, 0, 0]), Some(Dir::PosY));
        assert_eq!(rotate(a, [0, 1, 0]), Some(Dir::NegZ));
        assert_eq!(rotate(a, [0, 0, 1]), Some(Dir::NegX));
    }

    #[test]
    fn angles_mapping_finds_a_rotation() {
        for wa in Dir::ALL {
            for wb in Dir::ALL.into_iter().filter(|b| b.axis() != wa.axis()) {
                let ang = angles_mapping([1, 0, 0], wa, [0, 1, 0], wb).map(f64::from);
                assert_eq!(rotate(ang, [1, 0, 0]), Some(wa));
                assert_eq!(rotate(ang, [0, 1, 0]), Some(wb));
            }
        }
    }

    #[test]
    fn angles_for_up_inverts() {
        for d in Dir::ALL {
            assert_eq!(
                up_from_angles(angles_for_up(d).map(f64::from)),
                Some(d),
                "{d:?}"
            );
        }
    }
}
