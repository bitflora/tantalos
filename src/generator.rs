//! Random chambers as a mutable genome: a heightmap room shaped by platforms and pits, white
//! patches, glass/grating panels, fizzlers/laser fields, cube droppers, floor and pedestal
//! buttons (wired to the exit or to a field), and two doors on the outer walls.
//! `Genome::build` turns it into a puzzle via `Builder`.
//!
//! Layout rules (keep generated files close to what the editor makes):
//! - Floor is two voxels thick (`FLOOR`), so pits can be up to two deep.
//! - Every column is open from its floor up to the ceiling: no overhangs yet.
//! - Doors sit on the outer walls, droppers on the ceiling, so their backs are outside the
//!   chamber where the editor puts corridors and dropper bodies.
//! - Every floor keeps at least two open voxels above it.

use crate::builder::Builder;
use crate::geom::{step, Cell, Dir};
use crate::p2c::Puzzle;
use crate::rng::Rng;

/// Height of the standing cell over an untouched floor.
pub const FLOOR: i32 = 2;
pub const TITLE: &str = "glados generated";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Feature {
    /// Raise the columns in the box by `height`.
    Platform {
        min: [i32; 2],
        max: [i32; 2],
        height: i32,
    },
    /// Lower the columns in the box by `depth`; goo fills the bottom if `goo`.
    Pit {
        min: [i32; 2],
        max: [i32; 2],
        depth: i32,
        goo: bool,
    },
    /// Make every face pointing `dir` from an open cell in the box white.
    White { min: Cell, max: Cell, dir: Dir },
    /// A ceiling cube dropper over this column. `cube`: 0 standard, 1 companion,
    /// 2 reflection, 3 sphere.
    Dropper { at: [i32; 2], cube: i32 },
    /// A floor button in this column. `kind`: 0 weighted, 1 cube, 2 sphere.
    Button {
        at: [i32; 2],
        kind: i32,
        target: Target,
    },
    /// A pedestal button in this column; `timer` seconds, `None` = infinite.
    Pedestal {
        at: [i32; 2],
        timer: Option<u32>,
        target: Target,
    },
    /// A glass (or grating) panel standing on this column's floor, on the cells' `side` faces,
    /// `len` cells wide and `height` tall.
    Glass {
        at: [i32; 2],
        side: Dir,
        len: i32,
        height: i32,
        grating: bool,
    },
    /// A fizzler (or laser field) on the wall behind this column's floor cell (against
    /// `normal`), `height` cells tall, reaching across the room along `normal`.
    Field {
        at: [i32; 2],
        normal: Dir,
        height: i32,
        laser: bool,
        start_on: bool,
    },
}

/// What a button drives: the exit door, or the `n`-th field (modulo the number of fields that
/// were built; the exit if there are none).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Exit,
    Field(usize),
}

/// A door on the outer wall `side`, `along` cells from that wall's low end.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Door {
    pub side: Dir,
    pub along: i32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Genome {
    pub size: [i32; 3],
    pub entry: Door,
    pub exit: Door,
    pub features: Vec<Feature>,
}

fn clip(lo: i32, hi: i32, size: i32) -> Option<(i32, i32)> {
    let (lo, hi) = (lo.min(hi).max(0), lo.max(hi).min(size - 1));
    (lo <= hi).then_some((lo, hi))
}

impl Genome {
    pub fn random(rng: &mut Rng) -> Genome {
        let size = [rng.range(5, 10), rng.range(5, 10), FLOOR + rng.range(3, 5)];
        let mut g = Genome {
            size,
            entry: random_door(rng, size),
            exit: random_door(rng, size),
            features: Vec::new(),
        };
        let counts = [
            rng.range(0, 3),
            rng.range(0, 2),
            rng.range(1, 4),
            rng.range(0, 2),
            rng.range(0, 2),
            rng.range(0, 1),
            rng.range(0, 1),
            rng.range(0, 1),
        ];
        for (kind, &n) in counts.iter().enumerate() {
            for _ in 0..n {
                let f = g.random_feature(rng, kind);
                g.features.push(f);
            }
        }
        g
    }

    fn random_column(&self, rng: &mut Rng) -> [i32; 2] {
        [
            rng.range(0, self.size[0] - 1),
            rng.range(0, self.size[1] - 1),
        ]
    }

    fn random_rect(&self, rng: &mut Rng) -> ([i32; 2], [i32; 2]) {
        let a = self.random_column(rng);
        let w = [rng.range(0, 3), rng.range(0, 3)];
        (a, [a[0] + w[0], a[1] + w[1]])
    }

    /// `kind`: 0 platform, 1 pit, 2 white, 3 dropper, 4 button, 5 glass, 6 field, 7 pedestal.
    fn random_feature(&self, rng: &mut Rng, kind: usize) -> Feature {
        match kind {
            0 => {
                let (min, max) = self.random_rect(rng);
                Feature::Platform {
                    min,
                    max,
                    height: rng.range(1, 3),
                }
            }
            1 => {
                let (min, max) = self.random_rect(rng);
                Feature::Pit {
                    min,
                    max,
                    depth: rng.range(1, 2),
                    goo: rng.chance(0.5),
                }
            }
            2 => {
                let dir = *rng.pick(&Dir::ALL);
                let mut min = [0; 3];
                let mut max = [0; 3];
                for a in 0..3 {
                    let s = self.size[a];
                    let (lo, hi) = (rng.range(0, s - 1), rng.range(0, s - 1));
                    min[a] = lo.min(hi);
                    max[a] = lo.max(hi);
                }
                // Mostly reach the outer surface the patch faces, so it paints something.
                if rng.chance(0.6) {
                    let a = dir.axis();
                    if dir.is_positive() {
                        max[a] = self.size[a] - 1;
                    } else {
                        min[a] = 0;
                    }
                }
                Feature::White { min, max, dir }
            }
            3 => Feature::Dropper {
                at: self.random_column(rng),
                cube: if rng.chance(0.7) { 0 } else { rng.range(1, 3) },
            },
            4 => Feature::Button {
                at: self.random_column(rng),
                kind: if rng.chance(0.7) { 0 } else { rng.range(1, 2) },
                target: random_target(rng),
            },
            5 => Feature::Glass {
                at: self.random_column(rng),
                side: *rng.pick(&Dir::HORIZONTAL),
                len: rng.range(1, 4),
                height: rng.range(1, 3),
                grating: rng.chance(0.4),
            },
            6 => Feature::Field {
                at: self.random_column(rng),
                normal: *rng.pick(&Dir::HORIZONTAL),
                height: rng.range(1, 3),
                laser: rng.chance(0.4),
                start_on: rng.chance(0.8),
            },
            _ => Feature::Pedestal {
                at: self.random_column(rng),
                timer: random_timer(rng),
                target: random_target(rng),
            },
        }
    }

    pub fn mutate(&mut self, rng: &mut Rng) {
        match rng.below(20) {
            0..=5 => {
                let kind = rng.below(8);
                let f = self.random_feature(rng, kind);
                self.features.push(f);
            }
            6..=8 if !self.features.is_empty() => {
                let i = rng.below(self.features.len());
                self.features.remove(i);
            }
            9 => self.entry = random_door(rng, self.size),
            10 => self.exit = random_door(rng, self.size),
            11 => {
                let a = rng.below(2);
                self.size[a] = (self.size[a] + if rng.chance(0.5) { 1 } else { -1 }).clamp(4, 12);
            }
            _ if !self.features.is_empty() => {
                let i = rng.below(self.features.len());
                tweak(&mut self.features[i], rng);
            }
            _ => {}
        }
    }

    /// Floor height of each column (standing z) and whether it's goo, `[x][y]`.
    fn heightmap(&self) -> (Vec<Vec<i32>>, Vec<Vec<bool>>) {
        let [sx, sy, sz] = self.size;
        let top = sz - 1;
        let mut h = vec![vec![FLOOR; sy as usize]; sx as usize];
        let mut goo = vec![vec![false; sy as usize]; sx as usize];
        for f in &self.features {
            let (min, max, delta, wet) = match *f {
                Feature::Platform { min, max, height } => (min, max, height, false),
                Feature::Pit {
                    min,
                    max,
                    depth,
                    goo,
                } => (min, max, -depth, goo),
                _ => continue,
            };
            let (Some((x0, x1)), Some((y0, y1))) =
                (clip(min[0], max[0], sx), clip(min[1], max[1], sy))
            else {
                continue;
            };
            for x in x0..=x1 {
                for y in y0..=y1 {
                    let (x, y) = (x as usize, y as usize);
                    h[x][y] = (h[x][y] + delta).clamp(0, top - 1);
                    goo[x][y] = wet;
                }
            }
        }
        (h, goo)
    }

    /// The open cell in front of a door, or `None` if it would stand in goo.
    fn door_cell(&self, d: Door, h: &[Vec<i32>], goo: &[Vec<bool>]) -> Option<Cell> {
        let [sx, sy, _] = self.size;
        let along = |n: i32| d.along.rem_euclid(n);
        let (x, y) = match d.side {
            Dir::NegX => (0, along(sy)),
            Dir::PosX => (sx - 1, along(sy)),
            Dir::NegY => (along(sx), 0),
            _ => (along(sx), sy - 1),
        };
        let (ux, uy) = (x as usize, y as usize);
        (!goo[ux][uy]).then_some([x, y, h[ux][uy]])
    }

    /// The puzzle, or `None` if items would collide or stand in goo.
    pub fn build(&self) -> Option<Puzzle> {
        let [sx, sy, sz] = self.size;
        let top = sz - 1;
        let (h, goo) = self.heightmap();
        let mut b = Builder::new(TITLE, self.size);
        for x in 0..sx {
            for y in 0..sy {
                b.carve([x, y, h[x as usize][y as usize]], [x, y, top]);
            }
        }
        for f in &self.features {
            if let Feature::White { min, max, dir } = *f {
                let axes: Option<Vec<(i32, i32)>> =
                    (0..3).map(|a| clip(min[a], max[a], self.size[a])).collect();
                if let Some(axes) = axes {
                    let lo = [axes[0].0, axes[1].0, axes[2].0];
                    let hi = [axes[0].1, axes[1].1, axes[2].1];
                    b.paint(lo, hi, dir, true);
                }
            }
        }

        let mut used: Vec<Cell> = Vec::new();
        let mut claim = |c: Cell| {
            let free = !used.contains(&c);
            used.push(c);
            free
        };
        let entry = self.door_cell(self.entry, &h, &goo)?;
        let exit = self.door_cell(self.exit, &h, &goo)?;
        if !claim(entry) || !claim(exit) {
            return None;
        }
        b.entry(entry, self.entry.side.opposite());
        let exit = b.exit(exit, self.exit.side.opposite());

        let column = |at: [i32; 2]| {
            let (x, y) = (at[0].clamp(0, sx - 1), at[1].clamp(0, sy - 1));
            let (ux, uy) = (x as usize, y as usize);
            (!goo[ux][uy]).then_some([x, y, h[ux][uy]])
        };
        // Inside the chamber and above the floor of its column (columns are open to the top).
        let open = |c: Cell| {
            (0..3).all(|a| c[a] >= 0 && c[a] < self.size[a])
                && c[2] >= h[c[0] as usize][c[1] as usize]
        };

        // Panels and fields first: buttons need the fields' item indices. One that doesn't fit
        // is left out rather than spoiling the whole genome.
        let mut fields = Vec::new();
        for f in &self.features {
            match *f {
                Feature::Glass {
                    at,
                    side,
                    len,
                    height,
                    grating,
                } => {
                    let Some(anchor) = column(at) else { continue };
                    let (len, height) = (len.max(1), height.max(1));
                    let along = Dir::PosZ.cross(side.opposite()).unwrap().vector();
                    let fits = (0..len).all(|i| {
                        (0..height).all(|j| {
                            open([
                                anchor[0] + along[0] * i,
                                anchor[1] + along[1] * i,
                                anchor[2] + j,
                            ])
                        })
                    });
                    if fits {
                        b.glass(anchor, side, Dir::PosZ, 0, len - 1, height - 1, grating);
                    }
                }
                Feature::Field {
                    at,
                    normal,
                    height,
                    laser,
                    start_on,
                } => {
                    let Some(anchor) = column(at) else { continue };
                    if open(step(anchor, normal.opposite())) {
                        continue; // Needs a wall behind it.
                    }
                    let height = height.clamp(1, top - anchor[2] + 1);
                    fields.push(b.fizzler(anchor, normal, Dir::PosZ, height, laser, start_on));
                }
                _ => {}
            }
        }
        let target = |t: Target| match t {
            Target::Field(n) if !fields.is_empty() => fields[n % fields.len()],
            _ => exit,
        };

        for f in &self.features {
            match *f {
                Feature::Dropper { at, cube } => {
                    let floor = column(at)?;
                    let dropper = [floor[0], floor[1], top];
                    if floor[2] + 2 > top || !claim(floor) || !claim(dropper) {
                        return None;
                    }
                    b.cube_dropper_of(dropper, cube.clamp(0, 3));
                }
                Feature::Button {
                    at,
                    kind,
                    target: t,
                } => {
                    let cell = column(at)?;
                    if !claim(cell) {
                        return None;
                    }
                    let button = b.floor_button_of(cell, kind.clamp(0, 2));
                    b.connect(button, target(t));
                }
                Feature::Pedestal {
                    at,
                    timer,
                    target: t,
                } => {
                    let cell = column(at)?;
                    if !claim(cell) {
                        return None;
                    }
                    let pedestal = b.pedestal(cell, timer);
                    b.connect(pedestal, target(t));
                }
                _ => {}
            }
        }
        // Goo: one item per goo column, so the result doesn't hinge on how far goo floods.
        for x in 0..sx {
            for y in 0..sy {
                if goo[x as usize][y as usize] {
                    b.goo([x, y, h[x as usize][y as usize]]);
                }
            }
        }
        Some(b.build())
    }
}

fn random_door(rng: &mut Rng, size: [i32; 3]) -> Door {
    Door {
        side: *rng.pick(&Dir::HORIZONTAL),
        along: rng.range(0, size[0].max(size[1]) - 1),
    }
}

fn random_target(rng: &mut Rng) -> Target {
    if rng.chance(0.6) {
        Target::Exit
    } else {
        Target::Field(rng.below(4))
    }
}

fn random_timer(rng: &mut Rng) -> Option<u32> {
    rng.chance(0.5).then(|| rng.range(3, 15) as u32)
}

/// A different horizontal direction.
fn turn(d: Dir, rng: &mut Rng) -> Dir {
    let i = Dir::HORIZONTAL.iter().position(|&h| h == d).unwrap_or(0);
    Dir::HORIZONTAL[(i + 1 + rng.below(3)) % 4]
}

fn nudge(rng: &mut Rng) -> i32 {
    if rng.chance(0.5) {
        1
    } else {
        -1
    }
}

fn tweak(f: &mut Feature, rng: &mut Rng) {
    match f {
        Feature::Platform { min, max, height } => match rng.below(3) {
            0 => *height = (*height + nudge(rng)).clamp(1, 4),
            1 => {
                let (a, d) = (rng.below(2), nudge(rng));
                min[a] += d;
                max[a] += d;
            }
            _ => {
                let a = rng.below(2);
                if rng.chance(0.5) {
                    min[a] += nudge(rng);
                } else {
                    max[a] += nudge(rng);
                }
            }
        },
        Feature::Pit {
            min,
            max,
            depth,
            goo,
        } => match rng.below(4) {
            0 => *depth = (*depth + nudge(rng)).clamp(1, 2),
            1 => *goo = !*goo,
            2 => {
                let (a, d) = (rng.below(2), nudge(rng));
                min[a] += d;
                max[a] += d;
            }
            _ => {
                let a = rng.below(2);
                if rng.chance(0.5) {
                    min[a] += nudge(rng);
                } else {
                    max[a] += nudge(rng);
                }
            }
        },
        Feature::White { min, max, dir } => {
            if rng.chance(0.15) {
                *dir = *rng.pick(&Dir::ALL);
            } else {
                let a = rng.below(3);
                if rng.chance(0.5) {
                    min[a] += nudge(rng);
                } else {
                    max[a] += nudge(rng);
                }
            }
        }
        Feature::Dropper { at, cube } => {
            if rng.chance(0.2) {
                *cube = rng.range(0, 3);
            } else {
                at[rng.below(2)] += nudge(rng);
            }
        }
        Feature::Button { at, kind, target } => match rng.below(5) {
            0 => *kind = rng.range(0, 2),
            1 => *target = random_target(rng),
            _ => at[rng.below(2)] += nudge(rng),
        },
        Feature::Pedestal { at, timer, target } => match rng.below(5) {
            0 => *timer = random_timer(rng),
            1 => {
                if let Some(t) = timer {
                    *t = (*t as i32 + 2 * nudge(rng)).clamp(3, 30) as u32;
                }
            }
            2 => *target = random_target(rng),
            _ => at[rng.below(2)] += nudge(rng),
        },
        Feature::Glass {
            at,
            side,
            len,
            height,
            grating,
        } => match rng.below(6) {
            0 => *side = turn(*side, rng),
            1 => *len = (*len + nudge(rng)).clamp(1, 6),
            2 => *height = (*height + nudge(rng)).clamp(1, 4),
            3 => *grating = !*grating,
            _ => at[rng.below(2)] += nudge(rng),
        },
        Feature::Field {
            at,
            normal,
            height,
            laser,
            start_on,
        } => match rng.below(6) {
            0 => *normal = turn(*normal, rng),
            1 => *height = (*height + nudge(rng)).clamp(1, 4),
            2 => *laser = !*laser,
            3 => *start_on = !*start_on,
            _ => at[rng.below(2)] += nudge(rng),
        },
    }
}
