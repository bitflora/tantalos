//! Laser beams (plan.md tier 1): emitters, catchers, relays and reflection cubes.
//!
//! Rules (ASSUMPTIONS until checked in-game):
//! - A beam runs straight from its emitter's cell along the emitter's normal, cell by cell,
//!   until a solid voxel. Glass, grating, fields, bridges and funnels let it through.
//! - A catcher is powered by a beam coming into its cell against its normal (head on); a
//!   relay by one crossing its cell along its surface; the beam goes on through a relay and
//!   stops at a catcher. Both are senders: they drive what they are wired to while powered.
//!   Centred and offset items behave the same: an offset beam is still in its cell.
//! - A turret in a beam's cell is destroyed.
//! - A reflection cube resting on the floor of a cell a beam crosses (sideways, or down onto
//!   it) sends the beam on out of its lens, in whichever horizontal direction the player set
//!   it down facing (from the cell behind it, looking that way: a held cube faces where the
//!   player looks). Such places are "spots" (cell and direction); a cube anywhere else in a
//!   region is out of every beam. Only spots that can lead somewhere useful (a catcher, relay,
//!   turret or portal, perhaps through more spots) are kept.
//! - A beam that ends at a face with a portal comes out of the other portal, along its
//!   normal. A beam coming out of a portal and ending at the other one just loops.
//! - Beams don't hurt the player (crossing one quickly is survivable) and don't stop cubes.

use std::collections::{HashMap, HashSet};

use crate::geom::{step, Cell, Dir};
use crate::level::{ItemKind, LaserKind, Level};
use crate::portals::Face;

/// Most spots kept: each is a place a cube can be, so they multiply states.
pub const MAX_SPOTS: usize = 48;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Event {
    /// The beam crosses a spot cell (index into `Beams::spot_cells`): a cube there redirects
    /// it.
    Cube(usize),
    /// It powers receiver `k` (index into `Beams::receivers`); a catcher stops it.
    Receiver(usize),
    /// It destroys turret `t` (index into `Level::turrets`).
    Turret(usize),
}

/// A straight beam.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Ray {
    pub events: Vec<Event>,
    /// The white face it stops at, if it does: a portal there takes it.
    pub end: Option<Face>,
}

#[derive(Clone, Debug)]
pub struct Spot {
    /// Index into `Beams::spot_cells`.
    pub cell: usize,
    pub dir: Dir,
    /// The beam out of a cube here.
    pub ray: usize,
}

#[derive(Clone, Debug, Default)]
pub struct Beams {
    /// Catchers and relays (item index, kind), the senders `Signal::Laser` names.
    pub receivers: Vec<(i32, LaserKind)>,
    /// Emitters: (item index, ray).
    pub emitters: Vec<(i32, usize)>,
    pub rays: Vec<Ray>,
    pub spot_cells: Vec<Cell>,
    pub spots: Vec<Spot>,
    /// Per white face, the beam out of a portal there (only when some beam can go into a
    /// portal).
    pub exits: HashMap<Face, usize>,
    /// White faces some beam (from an emitter or a cube) ends at.
    pub ends: HashSet<Face>,
    /// Spots had to be dropped (`MAX_SPOTS`).
    pub incomplete: bool,
}

impl Beams {
    pub fn is_empty(&self) -> bool {
        self.emitters.is_empty()
    }

    /// The beams of `level`. `reflectors`: how many reflection cubes it has.
    pub fn build(level: &Level, reflectors: usize) -> Beams {
        let mut b = Beams {
            receivers: level
                .lasers
                .iter()
                .filter(|l| l.kind != LaserKind::Emitter)
                .map(|l| (l.item, l.kind))
                .collect(),
            ..Beams::default()
        };
        let emitters: Vec<_> = level
            .lasers
            .iter()
            .filter(|l| l.kind == LaserKind::Emitter)
            .collect();
        if emitters.is_empty() {
            return b;
        }
        // Spot cells: floors items don't take.
        let taken: HashSet<Cell> = level
            .items
            .iter()
            .filter(|i| {
                matches!(
                    i.kind,
                    ItemKind::FloorButton
                        | ItemKind::PedestalButton
                        | ItemKind::EntryDoor
                        | ItemKind::ExitDoor
                        | ItemKind::Turret
                        | ItemKind::LaserCatcher
                        | ItemKind::LaserRelay
                        | ItemKind::LaserEmitter
                        | ItemKind::FaithPlate
                        | ItemKind::Stairs
                        | ItemKind::Piston
                )
            })
            .map(|i| i.cell)
            .collect();
        let mut tracer = Tracer {
            level,
            taken,
            spot_id: HashMap::new(),
            spot_cells: Vec::new(),
        };
        // Every ray we might need, breadth first by how many cubes redirect into it.
        let mut rays: Vec<Ray> = Vec::new();
        let mut ray_id: HashMap<(Cell, Dir, bool), usize> = HashMap::new();
        let mut depth: Vec<usize> = Vec::new();
        let mut get = |tracer: &mut Tracer, rays: &mut Vec<Ray>, depth: &mut Vec<usize>, c, d, own, dp| {
            *ray_id.entry((c, d, own)).or_insert_with(|| {
                rays.push(tracer.trace(c, d, own));
                depth.push(dp);
                rays.len() - 1
            })
        };
        for e in &emitters {
            let r = get(&mut tracer, &mut rays, &mut depth, e.cell, e.normal, true, 0);
            b.emitters.push((e.item, r));
        }
        // Candidate spots: (cell, dir) -> ray, found from rays at depth below `reflectors`.
        let mut candidates: Vec<(usize, Dir, usize)> = Vec::new();
        let mut seen_cells: HashSet<usize> = HashSet::new();
        let mut i = 0;
        while i < rays.len() {
            if depth[i] < reflectors && rays.len() < 4096 {
                for ev in rays[i].events.clone() {
                    let Event::Cube(sc) = ev else { continue };
                    if !seen_cells.insert(sc) {
                        continue;
                    }
                    let cell = tracer.spot_cells[sc];
                    let next = depth[i] + 1;
                    for d in Dir::HORIZONTAL {
                        let open = !level.solid(step(cell, d)) || level.portal_surface(cell, d);
                        // Set down from the cell behind, facing `d`: a held cube faces where
                        // the player looks.
                        let behind = step(cell, d.opposite());
                        let reach = level.standable(behind)
                            && level.face(cell, d.opposite()).barrier.is_none();
                        if !open || !reach {
                            continue;
                        }
                        let r = get(&mut tracer, &mut rays, &mut depth, cell, d, false, next);
                        candidates.push((sc, d, r));
                    }
                }
            }
            i += 1;
        }
        // Beams out of portals, if any beam can go into one.
        let ends: HashSet<Face> = rays.iter().filter_map(|r| r.end).collect();
        let mut exits: HashMap<Face, usize> = HashMap::new();
        if !ends.is_empty() {
            for f in level.portal_surfaces() {
                let r = get(&mut tracer, &mut rays, &mut depth, f.0, f.1.opposite(), true, 1);
                exits.insert(f, r);
            }
        }
        // Keep what can lead somewhere: rays that power or destroy something, end where a
        // portal could take them on, or cross a useful spot.
        let mut useful_ray = vec![false; rays.len()];
        let mut useful_cell = vec![false; tracer.spot_cells.len()];
        let portals_help = !exits.is_empty();
        loop {
            let mut changed = false;
            for (r, ray) in rays.iter().enumerate() {
                if useful_ray[r] {
                    continue;
                }
                let hit = ray.events.iter().any(|e| match *e {
                    Event::Receiver(_) | Event::Turret(_) => true,
                    Event::Cube(sc) => useful_cell[sc],
                });
                if hit || (portals_help && ray.end.is_some()) {
                    useful_ray[r] = true;
                    changed = true;
                }
            }
            for &(sc, _, r) in &candidates {
                if useful_ray[r] && !useful_cell[sc] {
                    useful_cell[sc] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let mut keep: Vec<(usize, Dir, usize)> = candidates
            .into_iter()
            .filter(|&(_, _, r)| useful_ray[r])
            .collect();
        keep.sort_by_key(|&(_, _, r)| (depth[r], r));
        if keep.len() > MAX_SPOTS {
            keep.truncate(MAX_SPOTS);
            b.incomplete = true;
        }
        let kept_cells: HashSet<usize> = keep.iter().map(|k| k.0).collect();
        // Spot cells with no spot left don't stop beams.
        for ray in &mut rays {
            ray.events
                .retain(|e| !matches!(e, Event::Cube(sc) if !kept_cells.contains(sc)));
        }
        b.spot_cells = tracer.spot_cells;
        b.spots = keep
            .into_iter()
            .map(|(cell, dir, ray)| Spot { cell, dir, ray })
            .collect();
        b.ends = rays
            .iter()
            .enumerate()
            .filter(|&(r, _)| !exits.values().any(|&x| x == r))
            .filter_map(|(_, ray)| ray.end)
            .collect();
        b.exits = exits
            .into_iter()
            .filter(|&(_, r)| !rays[r].events.is_empty())
            .collect();
        b.rays = rays;
        b
    }
}

struct Tracer<'a> {
    level: &'a Level,
    taken: HashSet<Cell>,
    spot_id: HashMap<Cell, usize>,
    spot_cells: Vec<Cell>,
}

impl Tracer<'_> {
    /// The beam from cell `c` along `d`; `own`: whether things in `c` itself count (not for
    /// a beam leaving a cube).
    fn trace(&mut self, c: Cell, d: Dir, own: bool) -> Ray {
        let level = self.level;
        let mut events = Vec::new();
        let mut cur = c;
        let mut first = true;
        for _ in 0..64 {
            if level.solid(cur) {
                break;
            }
            if !first || own {
                let receiver = level
                    .lasers
                    .iter()
                    .filter(|l| l.kind != LaserKind::Emitter)
                    .position(|l| {
                        l.cell == cur
                            && match l.kind {
                                LaserKind::Catcher => l.normal == d.opposite(),
                                _ => l.normal.axis() != d.axis(),
                            }
                    });
                if let Some(k) = receiver {
                    events.push(Event::Receiver(k));
                    let catcher = level
                        .lasers
                        .iter()
                        .filter(|l| l.kind != LaserKind::Emitter)
                        .nth(k)
                        .is_some_and(|l| l.kind == LaserKind::Catcher);
                    if catcher {
                        return Ray { events, end: None };
                    }
                }
                if let Some(t) = level.turrets.iter().position(|t| t.cell == cur) {
                    events.push(Event::Turret(t));
                }
                let restable = d != Dir::PosZ
                    && level.standable(cur)
                    && !self.taken.contains(&cur);
                if restable {
                    let id = *self.spot_id.entry(cur).or_insert_with(|| {
                        self.spot_cells.push(cur);
                        self.spot_cells.len() - 1
                    });
                    events.push(Event::Cube(id));
                }
            }
            first = false;
            let n = step(cur, d);
            if level.solid(n) {
                let end = level.portal_surface(cur, d).then_some((cur, d));
                return Ray { events, end };
            }
            cur = n;
        }
        Ray { events, end: None }
    }
}
