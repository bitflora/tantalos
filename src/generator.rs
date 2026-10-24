//! Random chambers as a mutable genome: a room shaped by platforms, pits, dividing walls and
//! balconies, with white patches, glass/grating panels, fizzlers/laser fields, cube droppers,
//! turrets, gel droppers, floor and pedestal buttons (wired to the exit, a field, a gel
//! dropper or a device), laser emitters, catchers and relays, faith plates, light bridges,
//! funnels, angled and flip panels, stairs, pistons, rail platforms, loose cubes, and two doors
//! on the outer walls. `Genome::build` turns it into a puzzle via `Builder`.
//!
//! Layout rules (keep generated files close to what the editor makes):
//! - Floor is two voxels thick (`FLOOR`), so pits can be up to two deep.
//! - A column is open from its floor to the ceiling, except for walls (solid to the ceiling)
//!   and balconies (a one-voxel slab with open space above and below).
//! - Items stand on the highest floor of their column: a balcony if there is one.
//! - Doors sit on the outer walls, droppers on the ceiling, so their backs are outside the
//!   chamber where the editor puts corridors and dropper bodies.
//! - Every floor keeps at least two open voxels above it.

use crate::builder::Builder;
use crate::geom::{step, Cell, Dir};
use crate::level::{LaserKind, Level};
use crate::p2c::Puzzle;
use crate::rng::Rng;

/// Height of the standing cell over an untouched floor.
pub const FLOOR: i32 = 2;
pub const TITLE: &str = "glados generated";
/// Room size ranges: width and depth, and height including the floor.
pub const MIN_WIDTH: i32 = 4;
pub const MAX_WIDTH: i32 = 16;
pub const MIN_HEIGHT: i32 = FLOOR + 3;
pub const MAX_HEIGHT: i32 = FLOOR + 8;

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
    /// Columns solid to the ceiling, `len` of them from `at` along `along`, with an opening of
    /// `gap_len` columns starting `gap` columns in (no opening if `gap_len` is 0).
    Wall {
        at: [i32; 2],
        along: Dir,
        len: i32,
        gap: i32,
        gap_len: i32,
    },
    /// A one-voxel slab at height `z` over the box: a floor above, open space below. Columns
    /// without two open voxels both above and below it are left out.
    Balcony { min: [i32; 2], max: [i32; 2], z: i32 },
    /// A turret on this column's floor with `Facing` `facing` × 45° (0 = +x; it looks the
    /// opposite way, see `Turret::look_yaw`).
    Turret { at: [i32; 2], facing: i32 },
    /// Gel poured onto this column's floor from the ceiling above. `kind`: 0 repulsion,
    /// 1 propulsion, 2 conversion, 3 water. It pours from the start unless a button drives it.
    Gel { at: [i32; 2], kind: i32 },
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
    /// A laser emitter on the wall behind this column's floor cell (against `normal`), at cube
    /// height, firing across the room.
    Emitter {
        at: [i32; 2],
        normal: Dir,
        start_on: bool,
    },
    /// A laser catcher on the wall behind this column's floor cell (against `normal`), or a
    /// relay standing on the floor, driving `target`. An `aim`ed catcher goes instead where a
    /// reflection cube in the first emitter's beam can turn the beam into it: along `normal`
    /// from the beam's cell number `at[0] + at[1]` (if `normal` crosses the beam).
    Catcher {
        at: [i32; 2],
        normal: Dir,
        relay: bool,
        aim: bool,
        target: Target,
    },
    /// A faith plate on this column's floor throwing to the floor of column `to`.
    Plate { at: [i32; 2], to: [i32; 2] },
    /// A light bridge (or funnel) on the wall behind the cell `height` over this column's
    /// floor (against `normal`), projecting across the room. A funnel with `normal` up stands
    /// on the floor.
    Projector {
        at: [i32; 2],
        normal: Dir,
        height: i32,
        funnel: bool,
        start_on: bool,
    },
    /// An angled panel (`glass`: the clear one) on this column's floor tilting toward `tilt`
    /// by `angle` degrees; or (`flip`) a flip panel on the wall behind the floor cell against
    /// `tilt`.
    Panel {
        at: [i32; 2],
        tilt: Dir,
        angle: i32,
        deployed: bool,
        glass: bool,
        flip: bool,
    },
    /// Stairs on this column's floor and the next one along `dir`, rising toward `dir`.
    Stairs {
        at: [i32; 2],
        dir: Dir,
        deployed: bool,
    },
    /// A piston platform in this column, up `top` (1..=3) cells over its floor.
    Piston {
        at: [i32; 2],
        top: i32,
        start_up: bool,
    },
    /// A rail platform carrying the player from this column's floor cell (slid to a wall)
    /// `len` cells along `dir`; it moves through the voxels under them, dug out if solid.
    Rail {
        at: [i32; 2],
        dir: Dir,
        len: i32,
        oscillate: bool,
    },
    /// A cube placed without a dropper on this column's floor. `cube` as in `Dropper`.
    LooseCube { at: [i32; 2], cube: i32 },
}

/// What a button drives: the exit door, or the `n`-th field, gel dropper or device (emitter,
/// bridge, funnel, panel, stairs, piston or stopping rail; modulo the number that were built;
/// the exit if there are none).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Exit,
    Field(usize),
    Gel(usize),
    Device(usize),
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
        let size = [
            rng.range(6, 14),
            rng.range(6, 14),
            rng.range(MIN_HEIGHT, MAX_HEIGHT),
        ];
        let mut g = Genome {
            size,
            entry: random_door(rng, size),
            exit: random_door(rng, size),
            features: Vec::new(),
        };
        // Bigger floors get more features: counts are for a 7×7 room.
        let scale = (size[0] * size[1]) as f64 / 49.0;
        let mut upto = |lo: i32, hi: f64| rng.range(lo, lo.max((hi * scale).round() as i32));
        let mut counts = vec![
            upto(0, 3.0),
            upto(0, 2.0),
            upto(1, 4.0),
            upto(0, 2.0).min(3),
            upto(0, 2.0),
            upto(0, 1.0),
            upto(0, 1.0),
            upto(0, 1.0),
            upto(0, 1.0),
            upto(0, 1.0),
            upto(0, 1.0),
            upto(0, 1.0),
        ];
        // The later items: each now and then, so rooms don't fill up with them.
        for _ in 12..KINDS {
            counts.push(rng.chance(0.12) as i32);
        }
        for (kind, &n) in counts.iter().enumerate() {
            for _ in 0..n {
                let f = g.random_feature(rng, kind);
                g.features.push(f);
            }
        }
        // A catcher comes with something to catch: an emitter and a reflection cube.
        let has = |g: &Genome, f: fn(&Feature) -> bool| g.features.iter().any(f);
        if has(&g, |f| matches!(f, Feature::Catcher { .. })) {
            if !has(&g, |f| matches!(f, Feature::Emitter { .. })) {
                let e = g.random_feature(rng, 12);
                g.features.push(e);
            }
            if !has(&g, |f| matches!(f, Feature::Dropper { cube: 2, .. })) {
                let at = g.random_column(rng);
                g.features.push(Feature::Dropper { at, cube: 2 });
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

    /// `kind`: 0 platform, 1 pit, 2 white, 3 dropper, 4 button, 5 glass, 6 field, 7 pedestal,
    /// 8 wall, 9 balcony, 10 turret, 11 gel, 12 emitter, 13 catcher/relay, 14 faith plate,
    /// 15 bridge/funnel, 16 panel, 17 stairs, 18 piston, 19 rail, 20 loose cube.
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
            7 => Feature::Pedestal {
                at: self.random_column(rng),
                timer: random_timer(rng),
                target: random_target(rng),
            },
            8 => {
                let len = rng.range(2, self.size[0].max(self.size[1]));
                Feature::Wall {
                    at: self.random_column(rng),
                    along: *rng.pick(&Dir::HORIZONTAL),
                    len,
                    gap: rng.range(0, len - 1),
                    gap_len: if rng.chance(0.7) { rng.range(1, 2) } else { 0 },
                }
            }
            9 => {
                let (min, max) = self.random_rect(rng);
                Feature::Balcony {
                    min,
                    max,
                    z: rng.range(FLOOR + 2, (FLOOR + 2).max(self.size[2] - 3)),
                }
            }
            10 => Feature::Turret {
                at: self.random_column(rng),
                facing: rng.range(0, 7),
            },
            11 => Feature::Gel {
                at: self.random_column(rng),
                kind: random_paint(rng),
            },
            12 => Feature::Emitter {
                at: self.random_column(rng),
                normal: *rng.pick(&Dir::HORIZONTAL),
                start_on: rng.chance(0.85),
            },
            13 => Feature::Catcher {
                at: self.random_column(rng),
                normal: *rng.pick(&Dir::HORIZONTAL),
                relay: rng.chance(0.3),
                aim: rng.chance(0.6),
                target: random_target(rng),
            },
            14 => Feature::Plate {
                at: self.random_column(rng),
                to: self.random_column(rng),
            },
            15 => {
                let funnel = rng.chance(0.4);
                Feature::Projector {
                    at: self.random_column(rng),
                    normal: if funnel && rng.chance(0.4) {
                        Dir::PosZ
                    } else {
                        *rng.pick(&Dir::HORIZONTAL)
                    },
                    height: if rng.chance(0.6) { 0 } else { rng.range(1, 2) },
                    funnel,
                    start_on: rng.chance(0.7),
                }
            }
            16 => Feature::Panel {
                at: self.random_column(rng),
                tilt: *rng.pick(&Dir::HORIZONTAL),
                angle: random_angle(rng),
                deployed: rng.chance(0.5),
                glass: rng.chance(0.2),
                flip: rng.chance(0.3),
            },
            17 => Feature::Stairs {
                at: self.random_column(rng),
                dir: *rng.pick(&Dir::HORIZONTAL),
                deployed: rng.chance(0.5),
            },
            18 => Feature::Piston {
                at: self.random_column(rng),
                top: rng.range(1, 3),
                start_up: rng.chance(0.5),
            },
            19 => Feature::Rail {
                at: self.random_column(rng),
                dir: *rng.pick(&Dir::HORIZONTAL),
                len: rng.range(2, 4),
                oscillate: rng.chance(0.6),
            },
            _ => Feature::LooseCube {
                at: self.random_column(rng),
                cube: if rng.chance(0.6) { 0 } else { rng.range(1, 3) },
            },
        }
    }

    pub fn mutate(&mut self, rng: &mut Rng) {
        match rng.below(20) {
            0..=5 => {
                let kind = rng.below(KINDS);
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
                let a = rng.below(3);
                let (lo, hi) = if a == 2 {
                    (MIN_HEIGHT, MAX_HEIGHT)
                } else {
                    (MIN_WIDTH, MAX_WIDTH)
                };
                self.size[a] = (self.size[a] + nudge(rng)).clamp(lo, hi);
            }
            _ if !self.features.is_empty() => {
                let i = rng.below(self.features.len());
                tweak(&mut self.features[i], rng);
            }
            _ => {}
        }
    }

    /// The room's voxels: floors from platforms and pits, then walls and balconies.
    fn shape(&self) -> Shape {
        let [sx, sy, sz] = self.size;
        let top = sz - 1;
        let mut ground = vec![vec![FLOOR; sy as usize]; sx as usize];
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
                    ground[x][y] = (ground[x][y] + delta).clamp(0, top - 1);
                    goo[x][y] = wet;
                }
            }
        }
        let mut shape = Shape {
            size: self.size,
            solid: vec![false; (sx * sy * sz) as usize],
            goo,
        };
        for x in 0..sx {
            for y in 0..sy {
                for z in 0..ground[x as usize][y as usize] {
                    shape.set([x, y, z], true);
                }
            }
        }
        for f in &self.features {
            match *f {
                Feature::Wall {
                    at,
                    along,
                    len,
                    gap,
                    gap_len,
                } => {
                    let v = along.vector();
                    for i in (0..len.max(1)).filter(|i| !(gap..gap + gap_len).contains(i)) {
                        let (x, y) = (at[0] + v[0] * i, at[1] + v[1] * i);
                        if x < 0 || y < 0 || x >= sx || y >= sy {
                            continue;
                        }
                        shape.goo[x as usize][y as usize] = false;
                        for z in 0..sz {
                            shape.set([x, y, z], true);
                        }
                    }
                }
                Feature::Balcony { min, max, z } => {
                    let (Some((x0, x1)), Some((y0, y1))) =
                        (clip(min[0], max[0], sx), clip(min[1], max[1], sy))
                    else {
                        continue;
                    };
                    for x in x0..=x1 {
                        for y in y0..=y1 {
                            // Two open voxels below the slab and two above it.
                            let clear =
                                (z - 2..z).chain(z + 1..z + 3).all(|z| shape.open([x, y, z]));
                            if clear {
                                shape.set([x, y, z], true);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        shape
    }

    /// The open cell in front of a door, on the highest floor of its column.
    fn door_cell(&self, d: Door, shape: &Shape) -> Option<Cell> {
        let [sx, sy, _] = self.size;
        let along = |n: i32| d.along.rem_euclid(n);
        let (x, y) = match d.side {
            Dir::NegX => (0, along(sy)),
            Dir::PosX => (sx - 1, along(sy)),
            Dir::NegY => (along(sx), 0),
            _ => (along(sx), sy - 1),
        };
        shape.column([x, y])
    }

    /// The puzzle, or `None` if items would collide or a door has no floor.
    pub fn build(&self) -> Option<Puzzle> {
        let [sx, sy, sz] = self.size;
        let top = sz - 1;
        let shape = self.shape();
        let mut b = Builder::new(TITLE, self.size);
        for x in 0..sx {
            for y in 0..sy {
                for z in 0..sz {
                    if shape.open([x, y, z]) {
                        b.carve([x, y, z], [x, y, z]);
                    }
                }
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

        let entry = self.door_cell(self.entry, &shape)?;
        let exit = self.door_cell(self.exit, &shape)?;
        if entry == exit {
            return None;
        }
        let mut used: Vec<Cell> = vec![entry, exit];
        let doors = [entry, exit];
        b.entry(entry, self.entry.side.opposite());
        let exit = b.exit(exit, self.exit.side.opposite());

        // The highest floor of a column; items there that can't stand are left out.
        let column = |at: [i32; 2]| shape.column([at[0].clamp(0, sx - 1), at[1].clamp(0, sy - 1)]);
        let open = |c: Cell| shape.open(c);

        // Panels, then fields, before everything else: buttons need the fields' item indices.
        // One that doesn't fit is left out rather than spoiling the whole genome. A field fills
        // every cell it sweeps and the editor won't build anything else there, so a field whose
        // sweep meets a door, panel or other field is left out too, as is a later dropper, button
        // or pedestal that would stand in a sweep.
        let mut panels: Vec<Cell> = Vec::new();
        for f in &self.features {
            if let Feature::Glass {
                at,
                side,
                len,
                height,
                grating,
            } = *f
            {
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
                    for i in 0..len {
                        for j in 0..height {
                            panels.push([
                                anchor[0] + along[0] * i,
                                anchor[1] + along[1] * i,
                                anchor[2] + j,
                            ]);
                        }
                    }
                }
            }
        }
        let mut fields = Vec::new();
        let mut swept: Vec<Cell> = Vec::new();
        for f in &self.features {
            if let Feature::Field {
                at,
                normal,
                height,
                laser,
                start_on,
            } = *f
            {
                let Some(mut anchor) = column(at) else {
                    continue;
                };
                // Mounted on a wall: slide back along level floor until there is one.
                loop {
                    let back = step(anchor, normal.opposite());
                    if !open(back) {
                        break;
                    }
                    if column([back[0], back[1]]) != Some(back) {
                        break;
                    }
                    anchor = back;
                }
                if open(step(anchor, normal.opposite())) {
                    continue; // The floor drops or rises before the wall.
                }
                let height = height.clamp(1, top - anchor[2] + 1);
                let mut sweep = Vec::new();
                for j in 0..height {
                    let mut c = [anchor[0], anchor[1], anchor[2] + j];
                    while open(c) {
                        sweep.push(c);
                        c = step(c, normal);
                    }
                }
                if sweep
                    .iter()
                    .any(|c| used.contains(c) || panels.contains(c) || swept.contains(c))
                {
                    continue;
                }
                swept.extend(sweep);
                fields.push(b.fizzler(anchor, normal, Dir::PosZ, height, laser, start_on));
            }
        }
        let mut claim = |c: Cell| {
            let free = !used.contains(&c);
            used.push(c);
            free
        };
        // Gels before buttons, which need their droppers' indices.
        let mut gels = Vec::new();
        for f in &self.features {
            if let Feature::Gel { at, kind } = *f {
                let Some(floor) = column(at) else { continue };
                let mut dropper = floor;
                while open(step(dropper, Dir::PosZ)) {
                    dropper = step(dropper, Dir::PosZ);
                }
                if dropper == floor || swept.contains(&floor) || swept.contains(&dropper) {
                    continue;
                }
                if !claim(floor) || !claim(dropper) {
                    continue;
                }
                gels.push(b.gel_dropper(floor, kind.clamp(0, 3), true));
            }
        }
        // Devices, before the buttons and catchers that drive them.
        let mut devices: Vec<i32> = Vec::new();
        // Emitters built: their beams' cells.
        let mut beams: Vec<Vec<Cell>> = Vec::new();
        let mut rails_built = 0;
        // Back along level floor from this column's floor cell (at `z` over it) to a wall.
        let to_wall = |at: [i32; 2], back: Dir, dz: i32| -> Option<Cell> {
            let mut c = column(at)?;
            loop {
                let b = step(c, back);
                if !open(b) || column([b[0], b[1]]) != Some(b) {
                    break;
                }
                c = b;
            }
            let c = [c[0], c[1], c[2] + dz];
            (open(c) && !open(step(c, back))).then_some(c)
        };
        for f in &self.features {
            match *f {
                Feature::Emitter {
                    at,
                    normal,
                    start_on,
                } => {
                    let Some(c) = to_wall(at, normal.opposite(), 0) else { continue };
                    if swept.contains(&c) || !claim(c) {
                        continue;
                    }
                    devices.push(b.laser(LaserKind::Emitter, c, normal, Some(Dir::NegZ), start_on));
                    let mut beam = vec![c];
                    while open(step(*beam.last().unwrap(), normal)) {
                        beam.push(step(*beam.last().unwrap(), normal));
                    }
                    beams.push(beam);
                }
                Feature::Projector {
                    at,
                    normal,
                    height,
                    funnel,
                    start_on,
                } => {
                    // A bridge goes the way that crosses a gap, if one does.
                    let gap = |n: Dir| {
                        let Some(c) = to_wall(at, n.opposite(), height.clamp(0, 2)) else {
                            return false;
                        };
                        let mut x = c;
                        while open(step(x, n)) {
                            x = step(x, n);
                            if !open(step(x, Dir::NegZ)) {
                                continue;
                            }
                            return true;
                        }
                        false
                    };
                    let normal = if normal != Dir::PosZ && !funnel {
                        [normal, turn_by(normal, 1), turn_by(normal, 2), turn_by(normal, 3)]
                            .into_iter()
                            .find(|&n| gap(n))
                            .unwrap_or(normal)
                    } else {
                        normal
                    };
                    let c = if normal == Dir::PosZ {
                        column(at)
                    } else {
                        to_wall(at, normal.opposite(), height.clamp(0, 2))
                    };
                    let Some(c) = c else { continue };
                    if swept.contains(&c) || !claim(c) {
                        continue;
                    }
                    devices.push(if funnel {
                        b.funnel(c, normal, start_on, false)
                    } else if normal.axis() != 2 {
                        b.light_bridge(c, normal, Dir::PosZ, start_on)
                    } else {
                        continue;
                    });
                }
                Feature::Panel {
                    at,
                    tilt,
                    angle,
                    deployed,
                    glass,
                    flip,
                } => {
                    if flip {
                        let Some(c) = to_wall(at, tilt, 0) else { continue };
                        if swept.contains(&c) || !claim(c) {
                            continue;
                        }
                        devices.push(b.flip_panel(c, tilt.opposite()));
                    } else {
                        let Some(c) = column(at) else { continue };
                        if swept.contains(&c) || !claim(c) {
                            continue;
                        }
                        let white = b.white(c, Dir::NegZ);
                        let angle = random_angle_of(angle);
                        devices.push(b.angled_panel(c, Dir::PosZ, tilt, angle, deployed, white, glass));
                    }
                }
                Feature::Stairs { at, dir, deployed } => {
                    let Some(c) = column(at) else { continue };
                    // Face a one-voxel ledge two cells on, if some way does.
                    let ledge = |d: Dir| {
                        let (s1, s2) = (step(c, d), step(step(c, d), d));
                        column([s1[0], s1[1]]) == Some(s1)
                            && column([s2[0], s2[1]]) == Some(step(s2, Dir::PosZ))
                    };
                    let dir = [dir, turn_by(dir, 1), turn_by(dir, 2), turn_by(dir, 3)]
                        .into_iter()
                        .find(|&d| ledge(d))
                        .unwrap_or(dir);
                    let second = step(c, dir);
                    if column([second[0], second[1]]) != Some(second)
                        || swept.contains(&c)
                        || swept.contains(&second)
                        || !claim(c)
                        || !claim(second)
                    {
                        continue;
                    }
                    devices.push(b.stairs(c, dir, deployed));
                }
                Feature::Piston { at, top, start_up } => {
                    let Some(c) = column(at) else { continue };
                    let top = top.clamp(1, 3);
                    let column_open = (0..=top + 1).all(|l| open([c[0], c[1], c[2] + l]));
                    if !column_open || (0..=top).any(|l| swept.contains(&[c[0], c[1], c[2] + l])) {
                        continue;
                    }
                    if !claim(c) {
                        continue;
                    }
                    devices.push(b.piston(c, Dir::PosZ, 0, top, start_up));
                }
                Feature::Rail {
                    at,
                    dir,
                    len,
                    oscillate,
                } => {
                    // The platform runs along the wall on its left (Builder::rail), which the
                    // editor wants solid the whole way: start from that wall. It is the top of
                    // its voxel (`Rail::stops`), so it moves through the voxels under the cells
                    // it carries the player through: a pit's, or a trench dug for it.
                    let track_of = |d: Dir| -> Option<Vec<Cell>> {
                        let c = to_wall(at, turn_by(d, 1), 0)?;
                        let v = d.vector();
                        let t: Vec<Cell> = (0..=len.clamp(1, 5))
                            .map(|k| [c[0] + v[0] * k, c[1] + v[1] * k, c[2] - 1])
                            .collect();
                        let walled = t.iter().all(|&x| {
                            x[2] >= 0 && open(step(x, Dir::PosZ)) && !open(step(x, turn_by(d, 1)))
                        });
                        // Nor beside or over a door, whose frame takes the wall around it.
                        let near_door = |x: &Cell| {
                            doors.iter().any(|e| (0..3).all(|a| (x[a] - e[a]).abs() <= 1))
                        };
                        (walled && !t.iter().any(near_door)).then_some(t)
                    };
                    let tracks: Vec<Vec<Cell>> = [dir, turn_by(dir, 1), turn_by(dir, 2), turn_by(dir, 3)]
                        .into_iter()
                        .filter_map(track_of)
                        .collect();
                    let over_pit = |t: &Vec<Cell>| t.iter().all(|&x| open(x));
                    let Some(track) = tracks.iter().find(|t| over_pit(t)).or(tracks.first()).cloned()
                    else {
                        continue;
                    };
                    let c = track[0];
                    // It takes its whole track and the cells it carries the player through:
                    // nothing else may stand there.
                    let cells: Vec<Cell> =
                        track.iter().flat_map(|&t| [t, step(t, Dir::PosZ)]).collect();
                    if cells.iter().any(|t| swept.contains(t))
                        || !cells.iter().fold(true, |free, &t| claim(t) && free)
                    {
                        continue;
                    }
                    for &t in track.iter().filter(|&&t| !open(t)) {
                        b.carve(t, t);
                    }
                    let r = b.rail(c, *track.last().unwrap(), oscillate, true);
                    rails_built += 1;
                    if !oscillate {
                        devices.push(r);
                    }
                }
                _ => {}
            }
        }
        let _ = rails_built;
        let target = |t: Target| match t {
            Target::Field(n) if !fields.is_empty() => fields[n % fields.len()],
            Target::Gel(n) if !gels.is_empty() => gels[n % gels.len()],
            Target::Device(n) if !devices.is_empty() => devices[n % devices.len()],
            _ => exit,
        };
        // A driven gel dropper pours only while its buttons are down.
        let wire = |b: &mut Builder, sender: i32, receiver: i32| {
            b.connect(sender, receiver);
            if gels.contains(&receiver) {
                b.gel_start(receiver, false);
            }
        };

        for f in &self.features {
            match *f {
                Feature::Dropper { at, cube } => {
                    let Some(floor) = column(at) else { continue };
                    let dropper = [floor[0], floor[1], top];
                    if swept.contains(&floor) || swept.contains(&dropper) {
                        continue;
                    }
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
                    let Some(cell) = column(at) else { continue };
                    if swept.contains(&cell) {
                        continue;
                    }
                    if !claim(cell) {
                        return None;
                    }
                    let button = b.floor_button_of(cell, kind.clamp(0, 2));
                    wire(&mut b, button, target(t));
                }
                Feature::Pedestal {
                    at,
                    timer,
                    target: t,
                } => {
                    let Some(cell) = column(at) else { continue };
                    if swept.contains(&cell) {
                        continue;
                    }
                    if !claim(cell) {
                        return None;
                    }
                    let pedestal = b.pedestal(cell, timer);
                    wire(&mut b, pedestal, target(t));
                }
                Feature::Turret { at, facing } => {
                    let Some(cell) = column(at) else { continue };
                    if swept.contains(&cell) || !claim(cell) {
                        continue;
                    }
                    b.turret(cell, facing.rem_euclid(8) * 45);
                }
                Feature::Catcher {
                    at,
                    normal,
                    relay,
                    aim,
                    target: t,
                } => {
                    // Aimed: from a cell of the first beam, along `normal` to a wall.
                    let aimed = || -> Option<(Cell, Dir)> {
                        let beam = beams.first()?;
                        let k = (at[0] + at[1]).unsigned_abs() as usize % beam.len();
                        let along = beam
                            .get(1)
                            .map(|&n| (0..3).find(|&a| n[a] != beam[0][a]).unwrap());
                        if along == Some(normal.axis()) || column([beam[k][0], beam[k][1]]) != Some(beam[k]) {
                            return None;
                        }
                        let mut c = beam[k];
                        while open(step(c, normal)) {
                            c = step(c, normal);
                        }
                        (c != beam[k]).then_some((c, normal.opposite()))
                    };
                    let (c, normal) = if relay {
                        (column(at), normal)
                    } else if aim {
                        match aimed() {
                            Some((c, n)) => (Some(c), n),
                            None => (to_wall(at, normal.opposite(), 0), normal),
                        }
                    } else {
                        (to_wall(at, normal.opposite(), 0), normal)
                    };
                    let Some(c) = c else { continue };
                    if swept.contains(&c) || !claim(c) {
                        continue;
                    }
                    let (kind, n) = if relay {
                        (LaserKind::Relay, Dir::PosZ)
                    } else {
                        (LaserKind::Catcher, normal)
                    };
                    let item = b.laser(kind, c, n, Some(if relay { normal } else { Dir::NegZ }), true);
                    wire(&mut b, item, target(t));
                }
                Feature::Plate { at, to } => {
                    let (Some(c), Some(t)) = (column(at), column(to)) else {
                        continue;
                    };
                    if c == t || swept.contains(&c) || !claim(c) {
                        continue;
                    }
                    b.faith_plate(c, Dir::PosZ, t, Dir::PosZ);
                }
                Feature::LooseCube { at, cube } => {
                    let Some(c) = column(at) else { continue };
                    if swept.contains(&c) || !claim(c) {
                        continue;
                    }
                    b.loose_cube(c, cube.clamp(0, 3));
                }
                _ => {}
            }
        }
        // Goo: one item per goo column, so the result doesn't hinge on how far goo floods.
        for x in 0..sx {
            for y in 0..sy {
                if let Some(c) = shape.ground_goo([x, y]) {
                    b.goo(c);
                }
            }
        }
        // Features are placed independently, so a field can sweep through a door, button or
        // panel; the editor won't build those.
        let puzzle = b.build();
        Level::from_puzzle(&puzzle)
            .mount_problems()
            .is_empty()
            .then_some(puzzle)
    }
}

/// A genome's voxels.
struct Shape {
    size: [i32; 3],
    solid: Vec<bool>,
    /// Per column `[x][y]`: goo on its lowest floor.
    goo: Vec<Vec<bool>>,
}

impl Shape {
    fn index(&self, c: Cell) -> Option<usize> {
        let [sx, sy, sz] = self.size;
        let inside = c[0] >= 0 && c[1] >= 0 && c[2] >= 0 && c[0] < sx && c[1] < sy && c[2] < sz;
        inside.then(|| ((c[2] * sy + c[1]) * sx + c[0]) as usize)
    }

    fn set(&mut self, c: Cell, solid: bool) {
        if let Some(i) = self.index(c) {
            self.solid[i] = solid;
        }
    }

    /// Inside the chamber and not solid.
    fn open(&self, c: Cell) -> bool {
        self.index(c).is_some_and(|i| !self.solid[i])
    }

    /// The lowest open cell of a column, if that floor is goo.
    fn ground_goo(&self, at: [i32; 2]) -> Option<Cell> {
        if !self.goo[at[0] as usize][at[1] as usize] {
            return None;
        }
        (0..self.size[2]).map(|z| [at[0], at[1], z]).find(|&c| self.open(c))
    }

    /// The highest cell of a column with a floor below it and room to stand, or `None` if
    /// there is none or it is goo.
    fn column(&self, at: [i32; 2]) -> Option<Cell> {
        let floor = (0..self.size[2]).rev().map(|z| [at[0], at[1], z]).find(|&c| {
            self.open(c) && self.open(step(c, Dir::PosZ)) && !self.open(step(c, Dir::NegZ))
        })?;
        (self.ground_goo(at) != Some(floor)).then_some(floor)
    }
}

fn random_door(rng: &mut Rng, size: [i32; 3]) -> Door {
    Door {
        side: *rng.pick(&Dir::HORIZONTAL),
        along: rng.range(0, size[0].max(size[1]) - 1),
    }
}

fn random_target(rng: &mut Rng) -> Target {
    match rng.below(20) {
        0..=10 => Target::Exit,
        11..=14 => Target::Field(rng.below(4)),
        15..=16 => Target::Gel(rng.below(4)),
        _ => Target::Device(rng.below(4)),
    }
}

/// Feature kinds `random_feature` knows.
const KINDS: usize = 21;

/// A panel's angle: 30, 45, 60 or 90 degrees, 45 and 90 most often.
fn random_angle(rng: &mut Rng) -> i32 {
    match rng.below(6) {
        0 => 30,
        1 | 2 => 45,
        3 => 60,
        _ => 90,
    }
}

/// An angle a genome holds, as one the editor offers.
fn random_angle_of(angle: i32) -> i32 {
    match angle {
        a if a <= 30 => 30,
        a if a <= 45 => 45,
        a if a <= 60 => 60,
        _ => 90,
    }
}

/// Repulsion and conversion most often; water least.
fn random_paint(rng: &mut Rng) -> i32 {
    match rng.below(10) {
        0..=2 => 0,
        3..=5 => 1,
        6..=8 => 2,
        _ => 3,
    }
}

fn random_timer(rng: &mut Rng) -> Option<u32> {
    rng.chance(0.5).then(|| rng.range(3, 15) as u32)
}

/// `d` turned `k` quarter turns.
fn turn_by(d: Dir, k: usize) -> Dir {
    let i = Dir::HORIZONTAL.iter().position(|&h| h == d).unwrap_or(0);
    // HORIZONTAL is -x, +x, -y, +y: quarter turns go +x, +y, -x, -y.
    const RING: [Dir; 4] = [Dir::PosX, Dir::PosY, Dir::NegX, Dir::NegY];
    let r = RING.iter().position(|&h| h == Dir::HORIZONTAL[i]).unwrap();
    RING[(r + k) % 4]
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
        Feature::Wall {
            at,
            along,
            len,
            gap,
            gap_len,
        } => match rng.below(6) {
            0 => *along = turn(*along, rng),
            1 => *len = (*len + nudge(rng)).clamp(1, MAX_WIDTH),
            2 => *gap = (*gap + nudge(rng)).clamp(0, (*len - 1).max(0)),
            3 => *gap_len = (*gap_len + nudge(rng)).clamp(0, 3),
            _ => at[rng.below(2)] += nudge(rng),
        },
        Feature::Turret { at, facing } => {
            if rng.chance(0.4) {
                *facing = (*facing + nudge(rng)).rem_euclid(8);
            } else {
                at[rng.below(2)] += nudge(rng);
            }
        }
        Feature::Gel { at, kind } => {
            if rng.chance(0.25) {
                *kind = random_paint(rng);
            } else {
                at[rng.below(2)] += nudge(rng);
            }
        }
        Feature::Balcony { min, max, z } => match rng.below(4) {
            0 => *z = (*z + nudge(rng)).clamp(FLOOR + 2, MAX_HEIGHT - 3),
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
        Feature::Emitter {
            at,
            normal,
            start_on,
        } => match rng.below(5) {
            0 => *normal = turn(*normal, rng),
            1 => *start_on = !*start_on,
            _ => at[rng.below(2)] += nudge(rng),
        },
        Feature::Catcher {
            at,
            normal,
            relay,
            aim,
            target,
        } => match rng.below(7) {
            0 => *normal = turn(*normal, rng),
            1 => *relay = !*relay,
            2 => *target = random_target(rng),
            3 => *aim = !*aim,
            _ => at[rng.below(2)] += nudge(rng),
        },
        Feature::Plate { at, to } => {
            let a = rng.below(2);
            if rng.chance(0.5) {
                at[a] += nudge(rng);
            } else {
                to[a] += nudge(rng);
            }
        }
        Feature::Projector {
            at,
            normal,
            height,
            funnel,
            start_on,
        } => match rng.below(7) {
            0 => *normal = turn(*normal, rng),
            1 => *height = (*height + nudge(rng)).clamp(0, 2),
            2 => *funnel = !*funnel,
            3 => *start_on = !*start_on,
            _ => at[rng.below(2)] += nudge(rng),
        },
        Feature::Panel {
            at,
            tilt,
            angle,
            deployed,
            glass,
            flip,
        } => match rng.below(8) {
            0 => *tilt = turn(*tilt, rng),
            1 => *angle = random_angle(rng),
            2 => *deployed = !*deployed,
            3 => *glass = !*glass,
            4 => *flip = !*flip,
            _ => at[rng.below(2)] += nudge(rng),
        },
        Feature::Stairs { at, dir, deployed } => match rng.below(4) {
            0 => *dir = turn(*dir, rng),
            1 => *deployed = !*deployed,
            _ => at[rng.below(2)] += nudge(rng),
        },
        Feature::Piston { at, top, start_up } => match rng.below(4) {
            0 => *top = (*top + nudge(rng)).clamp(1, 3),
            1 => *start_up = !*start_up,
            _ => at[rng.below(2)] += nudge(rng),
        },
        Feature::Rail {
            at,
            dir,
            len,
            oscillate,
        } => match rng.below(5) {
            0 => *dir = turn(*dir, rng),
            1 => *len = (*len + nudge(rng)).clamp(1, 5),
            2 => *oscillate = !*oscillate,
            _ => at[rng.below(2)] += nudge(rng),
        },
        Feature::LooseCube { at, cube } => {
            if rng.chance(0.2) {
                *cube = rng.range(0, 3);
            } else {
                at[rng.below(2)] += nudge(rng);
            }
        }
    }
}
