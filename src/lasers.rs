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
//!   A floor button's cell is a spot too: the reflection cube stacked on a cube holding the
//!   button down (or set down beside one) catches the beam passing over it; it doesn't press
//!   the button itself. Beams out of portals have spots as well (straight, and angled at a
//!   receiver), kept only where the beam out of the cube powers a receiver or hits a turret.
//! - Heights (in-game, 2026-10-24): a centred wall emitter, catcher or relay sits at the middle
//!   of its voxel ("high"); one offset toward the floor sits low (`_OFFSET` items are 32 units
//!   off the middle); offset sideways or up counts as high. A reflection cube on the floor is
//!   low: a high beam passes over it. One set on top of another cube (on a button or a spot in
//!   its cell) is high. A cube catches only beams at its height and sends them on at it; a
//!   wall receiver takes only beams at its height; floor and ceiling items take any, and a
//!   beam keeps its height through portals. The solver tracks this (`Beams::high`).
//! - Angled beams: a cube can also be set down anywhere along a beam crossing its cell, turned
//!   to aim at a receiver, turret or portal surface on the same level (`aim_targets`, from
//!   three points along the beam). The beam out of it runs at that angle through the grid
//!   (`Tracer::trace_line`): it powers a floor relay or hits a turret passing within
//!   `NEAR_RELAY` / `NEAR_TURRET` of the cell's centre, a wall catcher landing on its face
//!   within `NEAR_CATCHER` of the centre and `CATCHER_CONE` of head on, and crosses a spot cell
//!   anywhere. Such a spot belongs to the beam it sits on (`Spot::on`); another beam reaching
//!   the cube stops there. Angled spots are a sample of the angles a player can set, so
//!   dropping some (`MAX_ANGLED`) doesn't make the model incomplete.
//! - A beam that ends at a face with a portal comes out of the other portal, along its
//!   normal. A beam coming out of a portal and ending at the other one just loops.
//! - A deployed angled panel (45° or more) stops a straight beam crossing the cell it swings
//!   into, unless the beam runs along its face; one that hits its front (`Event::Panel`) goes
//!   into a portal on it and out of the other one. A 30° panel doesn't reach the beam at the
//!   cell's middle. Angled beams pass panels.
//! - Beams don't stop the player or cubes. (In-game a beam burns and pushes back a player who
//!   walks into it: crossing one takes a running jump over it, which play.py does.)
//! - Funnels (play-tested, "sp_timed_lazer_change"): a reflection cube put into a funnel from
//!   the cell behind it, facing the way the player looks, rides along the flow keeping that
//!   facing, and redirects each beam crossing its way for the moment it passes (`Sweep`). A
//!   funnel going up or down carries it through both heights; along a level one it floats at
//!   the middle of its cells (high). Only sweeps whose beam powers a receiver are kept.

use std::collections::{HashMap, HashSet};

use crate::dynamics::funnel_path;
use crate::geom::{step, Cell, Dir};
use crate::level::{ItemKind, LaserKind, Level, PanelKind, ProjectorKind};
use crate::portals::Face;

/// Most spots kept: each is a place a cube can be, so they multiply states.
pub const MAX_SPOTS: usize = 48;
/// Most angled spots kept, besides those.
pub const MAX_ANGLED: usize = 32;
/// How close (voxels) an angled beam must pass a floor relay's centre to power it.
const NEAR_RELAY: f64 = 0.3;
/// How close it must pass a turret's centre to hit it.
const NEAR_TURRET: f64 = 0.35;
/// How close to the centre of a catcher's face it must land.
const NEAR_CATCHER: f64 = 0.4;
/// Most degrees off head on a catcher takes a beam.
const CATCHER_CONE: f64 = 70.0;
/// Shortest stretch (voxels) of a beam in a cell that a cube there can be set down on.
const MIN_SEG: f64 = 0.2;
/// Most funnel sweeps kept.
pub const MAX_SWEEPS: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Event {
    /// The beam crosses a spot cell (index into `Beams::spot_cells`): a cube there redirects
    /// it.
    Cube(usize),
    /// It powers receiver `k` (index into `Beams::receivers`); a catcher stops it.
    Receiver(usize),
    /// It destroys turret `t` (index into `Level::turrets`).
    Turret(usize),
    /// It crosses where angled panel `p` (index into `Level::panels`) swings out: deployed, the
    /// panel stops it, and a portal on its face takes it on if `front` (the beam hits the
    /// white face).
    Panel { p: usize, front: bool },
    /// It crosses funnel cell `fc` (index into `Beams::funnel_cells`): a reflection cube riding
    /// the funnel redirects it there for a moment (`Beams::sweeps`).
    Funnel(usize),
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
    /// The way it faces (an angled spot: the nearest axis, for setting it down).
    pub dir: Dir,
    /// The beam out of a cube here.
    pub ray: usize,
    /// An angled spot: the beam (index into `Beams::rays`) the cube sits on, which is the
    /// only one it redirects.
    pub on: Option<usize>,
    /// An angled spot: where the cube sits (x, y in voxels) and its yaw in degrees (0 = +x).
    pub aim: Option<([f64; 2], f64)>,
}

/// A reflection cube put into a funnel, crossing a beam on its way.
#[derive(Clone, Debug)]
pub struct Sweep {
    /// Index into `Level::projectors`.
    pub projector: usize,
    /// The flow runs toward the projector.
    pub reversed: bool,
    /// The cell the player stands in, putting the cube into the funnel's cell beside it,
    /// facing `dir`.
    pub from: Cell,
    pub dir: Dir,
    /// Where it crosses the beam (index into `Beams::funnel_cells`).
    pub cell: usize,
    /// Voxels along the funnel from where it goes in to there.
    pub dist: usize,
    /// The beam out of the cube there.
    pub ray: usize,
    /// The funnel goes up or down: the cube catches beams at both heights.
    pub any_height: bool,
}

/// A beam's stretch through a spot cell: (spot cell, entry point, exit point), x and y in
/// voxels.
type Seg = (usize, [f64; 2], [f64; 2]);

#[derive(Clone, Debug, Default)]
pub struct Beams {
    /// Catchers and relays (item index, kind), the senders `Signal::Laser` names.
    pub receivers: Vec<(i32, LaserKind)>,
    /// Per receiver: the height it takes beams at (`height`).
    pub receiver_high: Vec<Option<bool>>,
    /// Emitters: (item index, ray).
    pub emitters: Vec<(i32, usize)>,
    /// Per emitter: the height its beam runs at.
    pub emitter_high: Vec<Option<bool>>,
    pub rays: Vec<Ray>,
    pub spot_cells: Vec<Cell>,
    pub spots: Vec<Spot>,
    pub funnel_cells: Vec<Cell>,
    pub sweeps: Vec<Sweep>,
    /// Per white face, the beam out of a portal there (only when some beam can go into a
    /// portal).
    pub exits: HashMap<Face, usize>,
    /// White faces some beam (from an emitter or a cube) ends at.
    pub ends: HashSet<Face>,
    /// Spots had to be dropped (`MAX_SPOTS`).
    pub incomplete: bool,
    /// Built with angled spots.
    pub angled: bool,
    /// The level has a reflection cube and an emitter, so angled spots could help.
    pub could_angle: bool,
}

impl Beams {
    pub fn is_empty(&self) -> bool {
        self.emitters.is_empty()
    }

    /// The beams of `level`. `reflectors`: how many reflection cubes it has; `angled`: also
    /// angled spots.
    pub fn build(level: &Level, reflectors: usize, angled_spots: bool) -> Beams {
        let mut b = Beams {
            angled: angled_spots,
            receivers: level
                .lasers
                .iter()
                .filter(|l| l.kind != LaserKind::Emitter)
                .map(|l| (l.item, l.kind))
                .collect(),
            receiver_high: level
                .lasers
                .iter()
                .filter(|l| l.kind != LaserKind::Emitter)
                .map(height)
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
        b.could_angle = reflectors > 0;
        // Spot cells: floors items don't take.
        let taken: HashSet<Cell> = level
            .items
            .iter()
            .filter(|i| {
                matches!(
                    i.kind,
                    ItemKind::PedestalButton
                        | ItemKind::EntryDoor
                        | ItemKind::ExitDoor
                        | ItemKind::Turret
                        | ItemKind::FaithPlate
                        | ItemKind::Stairs
                        | ItemKind::Piston
                )
            })
            .map(|i| i.cell)
            // With angled spots, laser items take the floor only when they stand on it: one on
            // a wall leaves room for a cube in front of it (aimed from beside the emitter), and
            // a cube can sit on an emitter in the floor, turning its beam sideways.
            .chain(
                level
                    .lasers
                    .iter()
                    .filter(|l| {
                        !angled_spots || (l.normal == Dir::PosZ && l.kind != LaserKind::Emitter)
                    })
                    .map(|l| l.cell),
            )
            .collect();
        // Funnel sweeps that could be: (projector, reversed, from, dir, cell, dist, any
        // height), for every cell of a funnel's flow after one a cube can be put into.
        let mut templates: Vec<(usize, bool, Cell, Dir, Cell, usize, bool)> = Vec::new();
        let mut funnel: HashMap<Cell, (usize, usize)> = HashMap::new();
        let mut funnel_cells = Vec::new();
        if reflectors > 0 {
            for (i, p) in level.projectors.iter().enumerate() {
                if p.kind != ProjectorKind::Funnel {
                    continue;
                }
                let path = funnel_path(level, p.cell, p.normal);
                for reversed in [false, true] {
                    let mut cells = path.cells.clone();
                    if reversed {
                        cells.reverse();
                    }
                    for (a, &into) in cells.iter().enumerate() {
                        for d in Dir::HORIZONTAL {
                            let from = step(into, d.opposite());
                            if !level.standable(from) || level.face(into, d.opposite()).barrier.is_some() {
                                continue;
                            }
                            for (b, &cell) in cells.iter().enumerate().skip(a + 1) {
                                let vertical = path.dir.axis() == 2;
                                templates.push((i, reversed, from, d, cell, b - a, vertical));
                            }
                        }
                    }
                }
                for &c in &path.cells {
                    funnel.entry(c).or_insert_with(|| {
                        funnel_cells.push(c);
                        (funnel_cells.len() - 1, path.dir.axis())
                    });
                }
            }
        }
        templates.sort_by_key(|t| t.5);
        let mut tracer = Tracer {
            level,
            taken,
            spot_id: HashMap::new(),
            spot_cells: Vec::new(),
            on_emitters: angled_spots,
            funnel,
        };
        // Every ray we might need, breadth first by how many cubes redirect into it.
        let mut rays: Vec<Ray> = Vec::new();
        let mut segs: Vec<Vec<Seg>> = Vec::new();
        let mut ray_id: HashMap<(Cell, Dir, bool), usize> = HashMap::new();
        let mut depth: Vec<usize> = Vec::new();
        let mut get = |tracer: &mut Tracer,
                       rays: &mut Vec<Ray>,
                       segs: &mut Vec<Vec<Seg>>,
                       depth: &mut Vec<usize>,
                       c: Cell,
                       d: Dir,
                       own,
                       dp| {
            *ray_id.entry((c, d, own)).or_insert_with(|| {
                let ray = tracer.trace(c, d, own);
                // Straight beams run through cell centres.
                let v = d.vector();
                let h = [0.5 * v[0] as f64, 0.5 * v[1] as f64];
                segs.push(
                    ray.events
                        .iter()
                        .filter_map(|e| match *e {
                            Event::Cube(sc) => {
                                let x = tracer.spot_cells[sc];
                                let m = [x[0] as f64 + 0.5, x[1] as f64 + 0.5];
                                Some((sc, [m[0] - h[0], m[1] - h[1]], [m[0] + h[0], m[1] + h[1]]))
                            }
                            _ => None,
                        })
                        .collect(),
                );
                rays.push(ray);
                depth.push(dp);
                rays.len() - 1
            })
        };
        for e in &emitters {
            let r = get(&mut tracer, &mut rays, &mut segs, &mut depth, e.cell, e.normal, true, 0);
            b.emitters.push((e.item, r));
            b.emitter_high.push(height(e));
        }
        // Candidate spots: (cell, dir) -> ray, found from rays at depth below `reflectors`.
        let mut candidates: Vec<(usize, Dir, usize)> = Vec::new();
        // Angled ones: (spot cell, nearest facing, ray, ray it sits on, where, yaw).
        let mut angled: Vec<(usize, Dir, usize, usize, [f64; 2], f64)> = Vec::new();
        let mut angled_id: HashSet<(usize, usize, Ray)> = HashSet::new();
        let targets = aim_targets(level);
        let mut seen_cells: HashSet<usize> = HashSet::new();
        let mut i = 0;
        while i < rays.len() {
            if depth[i] < reflectors && rays.len() < 4096 {
                let on_beam = if angled_spots { segs[i].clone() } else { Vec::new() };
                for (sc, p0, p1) in on_beam {
                    let cell = tracer.spot_cells[sc];
                    for f in [0.25, 0.5, 0.75] {
                        let o = [p0[0] + f * (p1[0] - p0[0]), p0[1] + f * (p1[1] - p0[1])];
                        for &(tz, t, want) in &targets {
                            let v = [t[0] - o[0], t[1] - o[1]];
                            let n = (v[0] * v[0] + v[1] * v[1]).sqrt();
                            if tz != cell[2] || n < 0.5 || rays.len() >= 4096 {
                                continue;
                            }
                            let v = [v[0] / n, v[1] / n];
                            // Straight aims are the plain spots.
                            if v[0].abs() > 0.999 || v[1].abs() > 0.999 {
                                continue;
                            }
                            let (ray, ray_segs) = tracer.trace_line(o, cell, v);
                            // Only aims that get where they were aimed.
                            let got = match want {
                                Want::Receiver(k) => ray.events.contains(&Event::Receiver(k)),
                                Want::Turret(t) => ray.events.contains(&Event::Turret(t)),
                                Want::Face(f) => ray.end == Some(f),
                            };
                            if !got {
                                continue;
                            }
                            if !angled_id.insert((i, sc, ray.clone())) {
                                continue;
                            }
                            rays.push(ray);
                            segs.push(ray_segs);
                            depth.push(depth[i] + 1);
                            let yaw = v[1].atan2(v[0]).to_degrees();
                            angled.push((sc, nearest_dir(v), rays.len() - 1, i, o, yaw));
                        }
                    }
                }
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
                        let r =
                            get(&mut tracer, &mut rays, &mut segs, &mut depth, cell, d, false, next);
                        candidates.push((sc, d, r));
                    }
                }
            }
            i += 1;
        }
        // Beams out of portals, if any beam can go into one.
        let ends: HashSet<Face> = rays.iter().filter_map(|r| r.end).collect();
        let into_panel = rays
            .iter()
            .any(|r| r.events.iter().any(|e| matches!(e, Event::Panel { front: true, .. })));
        let mut exits: HashMap<Face, usize> = HashMap::new();
        if !ends.is_empty() || into_panel {
            for f in level.portal_surfaces() {
                let r =
                    get(&mut tracer, &mut rays, &mut segs, &mut depth, f.0, f.1.opposite(), true, 1);
                exits.insert(f, r);
            }
        }
        // Spots on beams out of portals, beside the ones found above. Every portalable face has
        // such a beam, so only spots that go straight to a receiver or turret are kept.
        let mut exit_candidates: Vec<(usize, Dir, usize)> = Vec::new();
        if reflectors > 0 {
            let mut exit_rays: Vec<usize> = exits.values().copied().collect();
            exit_rays.sort_unstable();
            for r in exit_rays {
                // Angled: a cube on the beam turned to aim at a receiver.
                let on_beam = if angled_spots { segs[r].clone() } else { Vec::new() };
                for (sc, p0, p1) in on_beam {
                    let cell = tracer.spot_cells[sc];
                    for f in [0.25, 0.5, 0.75] {
                        let o = [p0[0] + f * (p1[0] - p0[0]), p0[1] + f * (p1[1] - p0[1])];
                        for &(tz, tp, want) in &targets {
                            let Want::Receiver(k) = want else { continue };
                            let v = [tp[0] - o[0], tp[1] - o[1]];
                            let n = (v[0] * v[0] + v[1] * v[1]).sqrt();
                            if tz != cell[2] || n < 0.5 || rays.len() >= 4096 {
                                continue;
                            }
                            let v = [v[0] / n, v[1] / n];
                            if v[0].abs() > 0.999 || v[1].abs() > 0.999 {
                                continue;
                            }
                            let (ray, ray_segs) = tracer.trace_line(o, cell, v);
                            if !ray.events.contains(&Event::Receiver(k))
                                || !angled_id.insert((r, sc, ray.clone()))
                            {
                                continue;
                            }
                            rays.push(ray);
                            segs.push(ray_segs);
                            depth.push(2);
                            let yaw = v[1].atan2(v[0]).to_degrees();
                            angled.push((sc, nearest_dir(v), rays.len() - 1, r, o, yaw));
                        }
                    }
                }
                for ev in rays[r].events.clone() {
                    let Event::Cube(sc) = ev else { continue };
                    if !seen_cells.insert(sc) {
                        continue;
                    }
                    let cell = tracer.spot_cells[sc];
                    for d in Dir::HORIZONTAL {
                        let open = !level.solid(step(cell, d)) || level.portal_surface(cell, d);
                        let behind = step(cell, d.opposite());
                        let reach = level.standable(behind)
                            && level.face(cell, d.opposite()).barrier.is_none();
                        if !open || !reach {
                            continue;
                        }
                        let out = get(&mut tracer, &mut rays, &mut segs, &mut depth, cell, d, false, 2);
                        let hits = rays[out]
                            .events
                            .iter()
                            .any(|e| matches!(e, Event::Receiver(_) | Event::Turret(_)));
                        if hits {
                            exit_candidates.push((sc, d, out));
                        }
                    }
                }
            }
        }
        // Sweeps through a cell some beam crosses, whose beam out powers a receiver.
        let crossed: HashSet<usize> = rays
            .iter()
            .flat_map(|r| &r.events)
            .filter_map(|e| match *e {
                Event::Funnel(fc) => Some(fc),
                _ => None,
            })
            .collect();
        let mut sweeps = Vec::new();
        for &(projector, reversed, from, dir, cell, dist, any_height) in &templates {
            let fc = tracer.funnel[&cell].0;
            if !crossed.contains(&fc) || sweeps.len() >= MAX_SWEEPS || rays.len() >= 4096 {
                continue;
            }
            let ray = get(&mut tracer, &mut rays, &mut segs, &mut depth, cell, dir, false, reflectors);
            if rays[ray].events.iter().any(|e| matches!(e, Event::Receiver(_))) {
                sweeps.push(Sweep {
                    projector,
                    reversed,
                    from,
                    dir,
                    cell: fc,
                    dist,
                    ray,
                    any_height,
                });
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
                    Event::Panel { front, .. } => portals_help && front,
                    Event::Funnel(_) => false,
                });
                if hit || (portals_help && ray.end.is_some()) {
                    useful_ray[r] = true;
                    changed = true;
                }
            }
            let all = candidates.iter().map(|c| (c.0, c.2));
            for (sc, r) in all.chain(angled.iter().map(|a| (a.0, a.2))) {
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
            .chain(exit_candidates)
            .collect();
        keep.sort_by_key(|&(_, _, r)| (depth[r], r));
        if keep.len() > MAX_SPOTS {
            keep.truncate(MAX_SPOTS);
            b.incomplete = true;
        }
        let mut keep_angled: Vec<_> = angled.into_iter().filter(|a| useful_ray[a.2]).collect();
        keep_angled.sort_by_key(|a| (depth[a.2], a.2));
        keep_angled.truncate(MAX_ANGLED);
        let kept_cells: HashSet<usize> =
            keep.iter().map(|k| k.0).chain(keep_angled.iter().map(|a| a.0)).collect();
        // Spot cells with no spot left don't stop beams, and funnel cells with no sweep don't
        // matter.
        let swept: HashSet<usize> = sweeps.iter().map(|s| s.cell).collect();
        for ray in &mut rays {
            ray.events.retain(|e| match e {
                Event::Cube(sc) => kept_cells.contains(sc),
                Event::Funnel(fc) => swept.contains(fc),
                _ => true,
            });
        }
        b.funnel_cells = funnel_cells;
        b.sweeps = sweeps;
        b.spot_cells = tracer.spot_cells;
        b.spots = keep
            .into_iter()
            .map(|(cell, dir, ray)| Spot {
                cell,
                dir,
                ray,
                on: None,
                aim: None,
            })
            .chain(keep_angled.into_iter().map(|(cell, dir, ray, on, o, yaw)| Spot {
                cell,
                dir,
                ray,
                on: Some(on),
                aim: Some((o, yaw)),
            }))
            .collect();
        // Where beams from emitters and cubes end (a beam out of a portal ending at a white
        // face just loops). Not "every ray but the portal ones": an emitter on a white wall
        // fires the same ray a portal behind it would, and its end counts.
        b.ends = b
            .emitters
            .iter()
            .map(|&(_, r)| r)
            .chain(b.spots.iter().map(|s| s.ray))
            .filter_map(|r| rays[r].end)
            .collect();
        b.exits = exits
            .into_iter()
            .filter(|&(_, r)| !rays[r].events.is_empty())
            .collect();
        b.rays = rays;
        b
    }
}

/// The panel a straight beam in `cell` along `d` runs into when deployed, if any: an angled
/// panel of 45° or more swinging into `cell`, not edge on to the beam. Its face leans from
/// its normal toward its hinge as it swings, so the beam hits the front if it runs against
/// that leaning normal.
fn panel_event(level: &Level, cell: Cell, d: Dir) -> Option<Event> {
    let p = level
        .panels
        .iter()
        .position(|p| p.kind == PanelKind::Angled && p.cell == cell && p.angle >= 45)?;
    let panel = &level.panels[p];
    let dot = |a: Dir| {
        let (u, v) = (a.vector(), d.vector());
        u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
    };
    let (along_n, along_t) = (dot(panel.normal), dot(panel.tilt));
    if along_n == 0 && along_t == 0 {
        return None;
    }
    let a = (panel.angle as f64).to_radians();
    let facing = a.cos() * along_n as f64 - a.sin() * along_t as f64;
    let white = level.portal_surface(cell, panel.normal.opposite());
    Some(Event::Panel {
        p,
        front: white && facing < -1e-9,
    })
}

/// The height a laser item takes or sends beams at: `Some(true)` high (the middle of its
/// voxel), `Some(false)` low (on a wall, offset toward the floor), `None` any (on a floor or
/// ceiling).
pub fn height(l: &crate::level::Laser) -> Option<bool> {
    (l.normal.axis() != 2).then_some(l.offset != Some(Dir::NegZ))
}

/// What an angled cube aims at.
#[derive(Clone, Copy)]
enum Want {
    Receiver(usize),
    Turret(usize),
    Face(Face),
}

/// Points an angled cube can aim at, with their level and what they are: floor relays'
/// centres, wall catchers' face centres, turrets, and the centres of white wall faces.
fn aim_targets(level: &Level) -> Vec<(i32, [f64; 2], Want)> {
    let centre = |c: Cell| [c[0] as f64 + 0.5, c[1] as f64 + 0.5];
    let mut out = Vec::new();
    let receivers = level.lasers.iter().filter(|l| l.kind != LaserKind::Emitter);
    for (k, l) in receivers.enumerate() {
        let horizontal = l.normal.axis() != 2;
        match l.kind {
            LaserKind::Relay if !horizontal => {
                out.push((l.cell[2], centre(l.cell), Want::Receiver(k)))
            }
            LaserKind::Catcher if horizontal => {
                let v = l.normal.vector();
                let m = centre(l.cell);
                let f = [m[0] - 0.5 * v[0] as f64, m[1] - 0.5 * v[1] as f64];
                out.push((l.cell[2], f, Want::Receiver(k)));
            }
            _ => {}
        }
    }
    for (t, tur) in level.turrets.iter().enumerate() {
        out.push((tur.cell[2], centre(tur.cell), Want::Turret(t)));
    }
    for (c, d) in level.portal_surfaces() {
        if d.axis() != 2 {
            let v = d.vector();
            let m = centre(c);
            let f = [m[0] + 0.5 * v[0] as f64, m[1] + 0.5 * v[1] as f64];
            out.push((c[2], f, Want::Face((c, d))));
        }
    }
    out
}

/// The horizontal direction nearest to `v`.
fn nearest_dir(v: [f64; 2]) -> Dir {
    if v[0].abs() >= v[1].abs() {
        if v[0] > 0.0 {
            Dir::PosX
        } else {
            Dir::NegX
        }
    } else if v[1] > 0.0 {
        Dir::PosY
    } else {
        Dir::NegY
    }
}

/// Distance from `c` to the segment `a`-`b`.
fn seg_dist(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let l2 = ab[0] * ab[0] + ab[1] * ab[1];
    let t = if l2 == 0.0 {
        0.0
    } else {
        (((c[0] - a[0]) * ab[0] + (c[1] - a[1]) * ab[1]) / l2).clamp(0.0, 1.0)
    };
    let p = [a[0] + t * ab[0], a[1] + t * ab[1]];
    ((c[0] - p[0]).powi(2) + (c[1] - p[1]).powi(2)).sqrt()
}

struct Tracer<'a> {
    level: &'a Level,
    taken: HashSet<Cell>,
    spot_id: HashMap<Cell, usize>,
    spot_cells: Vec<Cell>,
    /// A cube can sit on a floor emitter (its beam going up into the cube).
    on_emitters: bool,
    /// Funnel cells: (index into `funnel_cells`, the axis of the flow).
    funnel: HashMap<Cell, (usize, usize)>,
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
            if let Some(e) = panel_event(level, cur, d) {
                events.push(e);
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
                // A cube riding the funnel crosses a beam across its flow.
                if let Some(&(fc, axis)) = self.funnel.get(&cur) {
                    if axis != d.axis() {
                        events.push(Event::Funnel(fc));
                    }
                }
                let up_from_floor = own && cur == c && self.on_emitters;
                let restable = (d != Dir::PosZ || up_from_floor)
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

    /// The beam from point `o` (x, y in voxels) in cell `start` along the horizontal unit
    /// vector `v`, at `start`'s level, and its stretches through spot cells. Things in `start`
    /// itself don't count (it leaves a cube there).
    fn trace_line(&mut self, o: [f64; 2], start: Cell, v: [f64; 2]) -> (Ray, Vec<Seg>) {
        let level = self.level;
        let (mut events, mut segs) = (Vec::new(), Vec::new());
        let (mut cur, mut p) = (start, o);
        let step_x = if v[0] > 0.0 { Dir::PosX } else { Dir::NegX };
        let step_y = if v[1] > 0.0 { Dir::PosY } else { Dir::NegY };
        let centre = |c: Cell| [c[0] as f64 + 0.5, c[1] as f64 + 0.5];
        // Distance along the beam to the next cell edge across axis `a`.
        let to_edge = |cur: Cell, p: [f64; 2], a: usize| {
            if v[a] == 0.0 {
                f64::INFINITY
            } else {
                let edge = cur[a] as f64 + if v[a] > 0.0 { 1.0 } else { 0.0 };
                (edge - p[a]) / v[a]
            }
        };
        for _ in 0..128 {
            let (tx, ty) = (to_edge(cur, p, 0), to_edge(cur, p, 1));
            let t = tx.min(ty).max(0.0);
            let q = [p[0] + t * v[0], p[1] + t * v[1]];
            let side = if tx <= ty { step_x } else { step_y };
            if cur != start {
                let near = seg_dist(p, q, centre(cur));
                let receivers = level.lasers.iter().filter(|l| l.kind != LaserKind::Emitter);
                for (k, l) in receivers.enumerate() {
                    if l.cell != cur {
                        continue;
                    }
                    let hit = match l.kind {
                        LaserKind::Relay => l.normal.axis() == 2 && near < NEAR_RELAY,
                        LaserKind::Catcher => {
                            // It lands on the face behind the catcher.
                            let n = l.normal.vector();
                            let head_on = -(v[0] * n[0] as f64 + v[1] * n[1] as f64);
                            let m = centre(cur);
                            let f = [m[0] - 0.5 * n[0] as f64, m[1] - 0.5 * n[1] as f64];
                            let off = ((q[0] - f[0]).powi(2) + (q[1] - f[1]).powi(2)).sqrt();
                            l.normal.axis() != 2
                                && side == l.normal.opposite()
                                && head_on >= CATCHER_CONE.to_radians().cos()
                                && off < NEAR_CATCHER
                        }
                        LaserKind::Emitter => false,
                    };
                    if hit {
                        events.push(Event::Receiver(k));
                        if l.kind == LaserKind::Catcher {
                            return (Ray { events, end: None }, segs);
                        }
                    }
                }
                if near < NEAR_TURRET {
                    if let Some(t) = level.turrets.iter().position(|t| t.cell == cur) {
                        events.push(Event::Turret(t));
                    }
                }
                let len = ((q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2)).sqrt();
                if len >= MIN_SEG && level.standable(cur) && !self.taken.contains(&cur) {
                    let id = *self.spot_id.entry(cur).or_insert_with(|| {
                        self.spot_cells.push(cur);
                        self.spot_cells.len() - 1
                    });
                    events.push(Event::Cube(id));
                    segs.push((id, p, q));
                }
            }
            // Through a corner the beam needs both cells beside it open.
            let corner = (tx - ty).abs() < 1e-9;
            let other = if side == step_x { step_y } else { step_x };
            let n = step(cur, side);
            if level.solid(n) || (corner && level.solid(step(cur, other))) {
                let face = if level.solid(n) { side } else { other };
                let end = level.portal_surface(cur, face).then_some((cur, face));
                return (Ray { events, end }, segs);
            }
            cur = if corner { step(n, other) } else { n };
            if level.solid(cur) {
                return (Ray { events, end: None }, segs);
            }
            p = q;
        }
        (Ray { events, end: None }, segs)
    }
}
