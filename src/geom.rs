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

/// The axis an item's "up" points along for Source angles `[pitch, yaw, roll]` in degrees.
/// For Puzzle Maker items this is the surface normal: floor items are +Z, ceiling items -Z,
/// wall items point away from the wall they are mounted on. `None` if not axis-aligned.
pub fn up_from_angles(angles: [f64; 3]) -> Option<Dir> {
    let [p, y, r] = angles.map(f64::to_radians);
    let up = [
        r.cos() * p.sin() * y.cos() + r.sin() * y.sin(),
        r.cos() * p.sin() * y.sin() - r.sin() * y.cos(),
        r.cos() * p.cos(),
    ];
    let axis = (0..3).find(|&a| up[a].abs() > 0.99)?;
    if (0..3).any(|a| a != axis && up[a].abs() > 0.01) {
        return None;
    }
    let positive = up[axis] > 0.0;
    Dir::ALL
        .into_iter()
        .find(|d| d.axis() == axis && d.is_positive() == positive)
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
