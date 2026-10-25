//! Walkable regions: sets of standable cells mutually reachable on foot, plus the
//! one-way moves between them. This is the position abstraction the solver searches over.
//!
//! Movement rules (ASSUMPTIONS until measured in-game):
//! - Walking connects 4-neighbour standable cells at the same height, unless glass or grating
//!   is on the face between them. A fizzler or laser field there makes it a `Walk` edge that
//!   carries the field, so the solver can apply it (laser fields only when on).
//! - Stepping off a ledge falls straight down; landing in goo is death (no edge).
//!   Long-fall boots mean any drop height is survivable. Glass and grating floors stop a fall.
//!   A faith plate on the way launches whatever falls to its target (`Move::Plate`).
//! - A one-voxel (128u) ledge can't be jumped; it is a `Step` edge that needs a cube.
//! - A running jump clears a one-voxel gap to a floor at the same height (`Move::Jump`):
//!   ~127 units of flight at walking speed (from the measured gel leap, 290 units at 400 u/s),
//!   against 96 for the player's box (32 wide) to get from one floor's edge onto the other's.
//!   Play-tested on "Parkour, Parkour PARKOUR PARKOUR".
//!   Two or more voxels can't be climbed at all.
//! - Cells covered by different sets of turrets are in different regions, joined by `Walk`
//!   edges, so the solver can tell where turrets can shoot.
//!
//! Devices (bridges, pistons, rails, panels, stairs, funnels; see `dynamics`) change the
//! geometry with the state. Their effects come in as a `Dyn`: cells that are standable only
//! under a `Cond` on the devices' bits, faces they block, and moves they add. Cells standable
//! under different conditions are in different regions, and every edge carries the condition
//! it needs, so one graph serves every state. A region whose condition fails takes whoever is
//! in it somewhere else (`Region::fallback`): a moving platform carries them, a vanishing
//! bridge drops them.

use std::collections::{BTreeSet, HashMap};

use crate::geom::{step, Cell, Dir};
use crate::level::{FaceInfo, Level};

/// A condition on the devices' state word: every bit of `on` set and every bit of `off` clear.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Cond {
    pub on: u64,
    pub off: u64,
}

impl Cond {
    pub const ALWAYS: Cond = Cond { on: 0, off: 0 };

    pub fn on(bit: usize) -> Cond {
        Cond {
            on: 1 << bit,
            off: 0,
        }
    }

    pub fn off(bit: usize) -> Cond {
        Cond {
            on: 0,
            off: 1 << bit,
        }
    }

    pub fn holds(self, dev: u64) -> bool {
        dev & self.on == self.on && dev & self.off == 0
    }

    /// Both conditions, or `None` if they contradict each other.
    pub fn and(self, o: Cond) -> Option<Cond> {
        let c = Cond {
            on: self.on | o.on,
            off: self.off | o.off,
        };
        (c.on & c.off == 0).then_some(c)
    }

    /// The opposite condition, if it is one bit (the only kind a conjunction can negate).
    pub fn negate(self) -> Option<Cond> {
        match (self.on.count_ones(), self.off.count_ones()) {
            (0, 0) => None,
            (1, 0) => Some(Cond {
                on: 0,
                off: self.on,
            }),
            (0, 1) => Some(Cond {
                on: self.off,
                off: 0,
            }),
            _ => None,
        }
    }

    pub fn is_always(self) -> bool {
        self == Cond::ALWAYS
    }
}

/// What devices do to the geometry (built by `dynamics`).
#[derive(Clone, Debug, Default)]
pub struct Dyn {
    /// Cells that are standable exactly when the condition holds, overriding
    /// `Level::standable`: a bridge or platform under them, or a panel or piston in them.
    pub stand: HashMap<Cell, Cond>,
    /// Faces blocked while the condition holds, by `face_key`.
    pub blocks: HashMap<(Cell, usize), Cond>,
    /// Moves devices add.
    pub moves: Vec<DynMove>,
    /// Where someone in this cell is taken when its condition fails (a moving platform's other
    /// stop, or off a panel that deploys), instead of falling.
    pub rides: HashMap<Cell, Cell>,
    /// Where a wall piston coming out pushes someone in this cell: the open cell past its end,
    /// to fall from.
    pub pushes: HashMap<Cell, Cell>,
}

/// From any of `from`, to each of `to` (its own extra condition, the cell where it comes to
/// rest, extra hazards), while `cond` holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DynMove {
    pub from: Vec<Cell>,
    pub to: Vec<(Cond, Cell, u64)>,
    pub kind: Move,
    pub cond: Cond,
    pub hazards: u64,
}

/// Faces are stored once, under the cell on their negative side.
pub fn face_key(c: Cell, d: Dir) -> (Cell, usize) {
    if d.is_positive() {
        (c, d.axis())
    } else {
        (step(c, d), d.axis())
    }
}

/// One way a fall can end.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fall {
    /// When it ends this way.
    pub cond: Cond,
    /// Where it comes to rest.
    pub cell: Cell,
    /// Fields fallen through.
    pub hazards: u64,
    /// The last faith plate (index into `Level::plates`) that launched it.
    pub plate: Option<usize>,
}

/// The level's geometry with devices.
#[derive(Clone, Copy)]
pub struct Geo<'a> {
    pub level: &'a Level,
    pub dyn_: &'a Dyn,
}

impl<'a> Geo<'a> {
    pub fn new(level: &'a Level, dyn_: &'a Dyn) -> Geo<'a> {
        Geo { level, dyn_ }
    }

    /// When cell `c` can be stood in (`None`: never).
    pub fn stand(&self, c: Cell) -> Option<Cond> {
        match self.dyn_.stand.get(&c) {
            Some(&cond) => Some(cond),
            None => self.level.standable(c).then_some(Cond::ALWAYS),
        }
    }

    /// When the face between `c` and `step(c, d)` is blocked by a device.
    pub fn block(&self, c: Cell, d: Dir) -> Option<Cond> {
        self.dyn_.blocks.get(&face_key(c, d)).copied()
    }

    /// Running jumps from standable cell `c` across one voxel that can't be stood in, onto a
    /// floor at the same height: the direction, the landing cell, when it can be made (on top
    /// of `c` being there) and the fields crossed.
    pub fn jumps(&self, c: Cell) -> Vec<(Dir, Cell, Cond, u64)> {
        let level = self.level;
        let open = |c: Cell, d: Dir| match self.block(c, d) {
            _ if level.face(c, d).barrier.is_some() => None,
            None => Some(Cond::ALWAYS),
            Some(b) => b.negate(),
        };
        let mut out = Vec::new();
        for d in Dir::HORIZONTAL {
            let (gap, to) = (step(c, d), step(step(c, d), d));
            if level.solid(gap) || level.solid(to) {
                continue;
            }
            let no_floor = match self.stand(gap) {
                None => Some(Cond::ALWAYS),
                Some(g) => g.negate(),
            };
            let cond = [no_floor, open(c, d), open(gap, d), self.stand(to)]
                .into_iter()
                .try_fold(Cond::ALWAYS, |acc, k| k.and_then(|k| acc.and(k)));
            if let Some(cond) = cond {
                let hazards = level.face(c, d).hazards | level.face(gap, d).hazards;
                out.push((d, to, cond, hazards));
            }
        }
        out
    }

    /// Every way something falling from open cell `c` can come to rest, each with the
    /// condition it needs: on the first floor down that is there. Empty if it ends in goo, `c`
    /// is solid, or a device floor's absence can't be written as a condition.
    pub fn falls(&self, c: Cell) -> Vec<Fall> {
        let level = self.level;
        let mut out = Vec::new();
        if level.solid(c) {
            return out;
        }
        let (mut cond, mut f, mut hazards) = (Cond::ALWAYS, c, 0);
        loop {
            if level.plate_at(f).is_some() {
                if let Some((g, h, plate)) = level.fall_route(f) {
                    out.push(Fall {
                        cond,
                        cell: g,
                        hazards: hazards | h,
                        plate,
                    });
                }
                break;
            }
            match self.dyn_.stand.get(&f) {
                Some(&sc) => {
                    if let Some(both) = cond.and(sc) {
                        out.push(Fall {
                            cond: both,
                            cell: f,
                            hazards,
                            plate: None,
                        });
                    }
                    // Without the device, carry on down, unless a floor is here anyway (then
                    // the device fills the cell: nowhere to land).
                    let Some(rest) = sc.negate().and_then(|n| cond.and(n)) else {
                        break;
                    };
                    if level.floor_under(f) || level.goo(f) {
                        break;
                    }
                    cond = rest;
                }
                None => {
                    if level.goo(f) {
                        break;
                    }
                    if level.floor_under(f) {
                        out.push(Fall {
                            cond,
                            cell: f,
                            hazards,
                            plate: None,
                        });
                        break;
                    }
                }
            }
            hazards |= level.face(f, Dir::NegZ).hazards;
            f = step(f, Dir::NegZ);
            if level.solid(f) {
                break;
            }
        }
        out
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    pub id: usize,
    pub z: i32,
    pub cells: Vec<Cell>,
    /// Bit `i`: turret `i` (of `Level::turrets`) can shoot every cell here.
    pub danger: u32,
    /// When the region is there to stand in.
    pub cond: Cond,
    /// Where whoever is here goes when `cond` fails: the first region whose condition holds;
    /// none means they die (or a cube is lost).
    pub fallback: Vec<(Cond, usize)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Move {
    /// Walk to a neighbouring cell at the same height: through a fizzler or laser field, or
    /// into or out of a turret's sight.
    Walk,
    /// Fall `height` voxels.
    Drop { height: i32 },
    /// Climb a one-voxel ledge; needs something to stand on (a cube).
    Step,
    /// Bounce up off repulsion gel.
    Bounce,
    /// Leap a gap at speed from a run of propulsion gel.
    Leap,
    /// Jump across a one-voxel gap at walking speed.
    Jump,
    /// Walk or fall onto a faith plate and be launched to its target.
    Plate,
    /// Walk up deployed stairs.
    Stairs,
    /// Float along an excursion funnel and step out of it.
    Funnel,
    /// Ride a moving platform from one of its stops to another.
    Ride,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub kind: Move,
    /// Hazard fields crossed on the way (bits index `Level::hazards`).
    pub hazards: u64,
    /// When the move can be made.
    pub cond: Cond,
}

pub struct RegionGraph {
    pub regions: Vec<Region>,
    pub edges: Vec<Edge>,
    size: [i32; 3],
    cell_region: Vec<Option<usize>>,
}

impl RegionGraph {
    pub fn build(level: &Level) -> RegionGraph {
        RegionGraph::build_with(level, |_| 0)
    }

    /// Regions that also split where `danger` (the turrets that can shoot a cell) changes.
    pub fn build_with(level: &Level, danger: impl Fn(Cell) -> u32) -> RegionGraph {
        RegionGraph::build_dyn(Geo::new(level, &Dyn::default()), danger)
    }

    /// Regions of the geometry with devices, split by the condition cells are standable under
    /// and by `danger`.
    pub fn build_dyn(geo: Geo, danger: impl Fn(Cell) -> u32) -> RegionGraph {
        let level = geo.level;
        let size = level.size;
        let mut g = RegionGraph {
            regions: Vec::new(),
            edges: Vec::new(),
            size,
            cell_region: vec![None; size.iter().product::<i32>() as usize],
        };
        let open_face = |c: Cell, d: Dir| {
            level.face(c, d) == FaceInfo::default() && geo.block(c, d).is_none()
        };

        for c in level.cells() {
            let Some(cond) = geo.stand(c) else { continue };
            if g.region_of(c).is_some() {
                continue;
            }
            let id = g.regions.len();
            let mask = danger(c);
            let mut cells = Vec::new();
            let mut stack = vec![c];
            while let Some(c) = stack.pop() {
                if geo.stand(c) != Some(cond) || g.region_of(c).is_some() || danger(c) != mask {
                    continue;
                }
                let i = g.idx(c).unwrap();
                g.cell_region[i] = Some(id);
                cells.push(c);
                for d in Dir::HORIZONTAL {
                    if open_face(c, d) {
                        stack.push(step(c, d));
                    }
                }
            }
            cells.sort_by_key(|c| (c[2], c[1], c[0]));
            g.regions.push(Region {
                id,
                z: c[2],
                cells,
                danger: mask,
                cond,
                fallback: Vec::new(),
            });
        }

        let mut edges = BTreeSet::new();
        for r in &g.regions {
            for &c in &r.cells {
                for d in Dir::HORIZONTAL {
                    let n = step(c, d);
                    let face = level.face(c, d);
                    let face_cond = match geo.block(c, d) {
                        None => Cond::ALWAYS,
                        Some(b) => match b.negate() {
                            Some(x) => x,
                            None => continue,
                        },
                    };
                    let mut edge = |to, kind, hazards, cond: Cond| {
                        edges.insert(Edge {
                            from: r.id,
                            to,
                            kind,
                            hazards,
                            cond,
                        });
                    };
                    if level.solid(n) {
                        let (head, top) = (step(c, Dir::PosZ), step(n, Dir::PosZ));
                        let over = level.face(head, d);
                        let top_cond = geo.stand(top).and_then(|t| {
                            let head_block = match geo.block(head, d) {
                                None => Some(Cond::ALWAYS),
                                Some(b) => b.negate(),
                            };
                            head_block.and_then(|h| t.and(h))
                        });
                        if let (true, Some(tc), false, None) = (
                            level.in_interior(n),
                            top_cond,
                            level.solid(head),
                            over.barrier,
                        ) {
                            edge(g.region_of(top).unwrap(), Move::Step, over.hazards, tc);
                        }
                    } else if face.barrier.is_some() {
                        // Glass or grating: no way through.
                    } else {
                        let sn = geo.stand(n);
                        if let Some(cn) = sn {
                            if let Some(cc) = face_cond.and(cn) {
                                let to = g.region_of(n).unwrap();
                                if face.hazards != 0 || to != r.id || !cc.is_always() {
                                    edge(to, Move::Walk, face.hazards, cc);
                                }
                            }
                        }
                        let rest = match sn {
                            None => Some(face_cond),
                            Some(cn) => cn.negate().and_then(|x| face_cond.and(x)),
                        };
                        let Some(rest) = rest else { continue };
                        for fall in geo.falls(n) {
                            let Some(cc) = rest.and(fall.cond) else { continue };
                            let Some(to) = g.region_of(fall.cell) else { continue };
                            let kind = if fall.plate.is_some() {
                                Move::Plate
                            } else {
                                Move::Drop {
                                    height: n[2] - fall.cell[2],
                                }
                            };
                            edge(to, kind, face.hazards | fall.hazards, cc);
                        }
                    }
                }
            }
        }
        for r in &g.regions {
            for &c in &r.cells {
                for (_, t, cond, hazards) in geo.jumps(c) {
                    let to = g.region_of(t).unwrap();
                    if to != r.id {
                        edges.insert(Edge {
                            from: r.id,
                            to,
                            kind: Move::Jump,
                            hazards,
                            cond,
                        });
                    }
                }
            }
        }
        g.move_edges(geo, &geo.dyn_.moves, &mut edges);
        g.edges = edges.into_iter().collect();

        // Where a vanishing region sends whoever is in it.
        let fallbacks: Vec<Vec<(Cond, usize)>> = g
            .regions
            .iter()
            .map(|r| {
                if r.cond.is_always() {
                    return Vec::new();
                }
                let c = r.cells[0];
                let gone = r.cond.negate();
                let mut out = Vec::new();
                if let Some(&p) = geo.dyn_.rides.get(&c) {
                    if let (Some(cp), Some(rp)) = (geo.stand(p), g.region_of(p)) {
                        out.push((cp, rp));
                    }
                }
                let pushed = geo.dyn_.pushes.get(&c).map(|&p| geo.falls(p)).unwrap_or_default();
                for fall in pushed.into_iter().chain(geo.falls(c)) {
                    let cond = match gone {
                        Some(gc) => gc.and(fall.cond),
                        None => Some(fall.cond),
                    };
                    if let (Some(cond), Some(to)) = (cond, g.region_of(fall.cell)) {
                        if to != r.id {
                            out.push((cond, to));
                        }
                    }
                }
                out
            })
            .collect();
        for (r, f) in g.regions.iter_mut().zip(fallbacks) {
            r.fallback = f;
        }
        g
    }

    /// The edges `moves` give between regions.
    fn move_edges(&self, geo: Geo, moves: &[DynMove], edges: &mut BTreeSet<Edge>) {
        for m in moves {
            for &f in &m.from {
                let (Some(cf), Some(rf)) = (geo.stand(f), self.region_of(f)) else {
                    continue;
                };
                for &(tc, t, th) in &m.to {
                    let (Some(ct), Some(rt)) = (geo.stand(t), self.region_of(t)) else {
                        continue;
                    };
                    let cond = m.cond.and(tc).and_then(|c| c.and(ct)).and_then(|c| c.and(cf));
                    if let Some(cond) = cond.filter(|_| rt != rf) {
                        edges.insert(Edge {
                            from: rf,
                            to: rt,
                            kind: m.kind,
                            hazards: m.hazards | th,
                            cond,
                        });
                    }
                }
            }
        }
    }

    /// Adds the edges of more moves (funnels through portals, found once the regions are).
    pub fn add_moves(&mut self, geo: Geo, moves: &[DynMove]) {
        if moves.is_empty() {
            return;
        }
        let mut edges: BTreeSet<Edge> = self.edges.iter().copied().collect();
        self.move_edges(geo, moves, &mut edges);
        self.edges = edges.into_iter().collect();
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

    /// Regions reachable from `from` by walking and dropping only (no portals, no cubes), with
    /// the devices as they are in `dev`.
    pub fn reachable_on_foot(&self, from: usize) -> Vec<bool> {
        let mut seen = vec![false; self.regions.len()];
        let mut stack = vec![from];
        while let Some(r) = stack.pop() {
            if std::mem::replace(&mut seen[r], true) {
                continue;
            }
            stack.extend(
                self.edges_from(r)
                    .filter(|e| matches!(e.kind, Move::Drop { .. }) && e.cond.is_always())
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
