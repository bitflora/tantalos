//! Devices: the items that change the geometry while the level is played (plan.md tiers 1-3).
//! Each gets bits in a state word (`Model` derives them from the buttons and portals); their
//! effects reach the region graph as a `Dyn` (see `regions`).
//!
//! Rules (ASSUMPTIONS until checked in-game):
//! - Stairs, deployed, climb from their two cells onto the ledge they face, or a ledge beside
//!   their top step (`Move::Stairs`, `Stairs::exits`).
//!   Deployed, they are a ramp of steps up to most of a voxel high: nobody walks onto or off
//!   them sideways (in-game, a walk across their top end stuck on the steps), so their cells'
//!   side faces are walls; they are walked into from behind.
//! - Angled and glass panels, deployed: on a floor at 45° or 60° their cell is a slope nobody
//!   stands on (whoever is there steps off away from the hinge); at 90° (any surface) the panel
//!   stands on its hinge edge: a wall across that face, or a floor if the hinge is at the top
//!   of a wall panel (for the cell above) or its bottom (for its own cell). At 30° they change
//!   nothing for walking. Flip panels only change which faces take portals (see `portals`).
//! - Piston platforms on a floor: down, their bottom stop is a floor; up, their top stop, and
//!   the column below it is solid. Whoever is on one rides it. On a wall or ceiling, the cells
//!   they push into when up are solid then, and on a wall the cells over them are floors. A
//!   wall piston coming out pushes whatever stands in those cells out past its end, where it
//!   falls.
//! - Rail platforms are the top of their voxel, so a floor for the cell above (`Rail::stops`).
//!   Ones that stop (not oscillating) are that floor over their first cell, or over their last
//!   while active; whoever is on one rides it. Oscillating ones are a walkway along their
//!   track, with rides between stops at different heights: wait for the platform.
//! - Light bridges and funnels project from their cell along their normal until a solid voxel
//!   or a glass/grating panel. A bridge sits 0.375 voxel from the middle of its cells against
//!   its walkable `side` (as the editor's `LocalPos`): with that side up it is the floor of the
//!   cells it crosses, with it down the floor of the cells over them (it runs along their
//!   bottom). One standing on edge is a wall on its `side` face of each. (Measured: out of a
//!   wall portal a bridge sits 24 units over its cells' floor, above a step: getting onto it
//!   from that floor takes a hop, which the player can do, so it is still their floor.)
//! - Through portals: a projection that ends at a white face continues out of the other
//!   portal. A horizontal bridge going wall to wall stays horizontal (`bridge_routes`); other
//!   bridges through portals aren't modelled. Funnels carry on in any direction
//!   (`funnel_route_moves`).
//! - Funnels carry the player from where they step in (the funnel's cell, or one beside it at
//!   the same height) along the flow; they can step out sideways anywhere further along.
//!   Reversed, the flow runs the other way.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::geom::{step, Cell, Dir};
use crate::level::{BarrierKind, Level, PanelKind, ProjectorKind};
use crate::portals::Face;
use crate::regions::{face_key, Cond, Dyn, DynMove, Geo, Move};

/// Most bits a state word holds.
pub const MAX_BITS: usize = 64;

/// What a device bit means.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bit {
    /// `Level::stairs[i]` deployed.
    Stairs(usize),
    /// `Level::panels[i]` deployed (angled, glass) or flipped (flip).
    Panel(usize),
    /// `Level::pistons[i]` up.
    Piston(usize),
    /// `Level::rails[i]` at its end (rails that stop).
    Rail(usize),
    /// `Level::projectors[i]` on.
    Projector(usize),
    /// `Level::projectors[i]` (a funnel) reversed.
    Reversed(usize),
    /// `Level::projectors[i]` on, a portal on the face its path ends at, and the other on a
    /// face of route `route` (index into `Devices::routes[i]`).
    Routed { projector: usize, route: usize },
    /// Any of the bits in `on` on, or any in `off` off: the condition of a cell or face two
    /// devices (or two routes of a bridge) make a floor or a wall, which a conjunction can't
    /// write (`Cond` is one).
    Any { on: u64, off: u64 },
}

/// A projection's straight path.
#[derive(Clone, Debug, PartialEq)]
pub struct Path {
    pub cells: Vec<Cell>,
    pub dir: Dir,
    /// The face it ends at, if that takes portals: where it can go into one.
    pub end: Option<Face>,
}

/// Faces a projection coming out of a portal there does the same thing from.
#[derive(Clone, Debug, PartialEq)]
pub struct Route {
    pub bit: usize,
    pub faces: Vec<Face>,
}

#[derive(Clone, Debug, Default)]
pub struct Devices {
    pub bits: Vec<Bit>,
    pub dyn_: Dyn,
    pub stairs_bit: Vec<usize>,
    pub panel_bit: Vec<usize>,
    pub piston_bit: Vec<usize>,
    /// `None` for oscillating rails.
    pub rail_bit: Vec<Option<usize>>,
    pub projector_bit: Vec<usize>,
    /// Funnels only.
    pub reversed_bit: Vec<Option<usize>>,
    /// Per projector: its path.
    pub paths: Vec<Path>,
    /// Per projector: where it goes on through portals.
    pub routes: Vec<Vec<Route>>,
    /// Something couldn't be modelled (out of bits, overlapping effects, unknown orientations):
    /// the model may miss moves.
    pub incomplete: bool,
}

impl Devices {
    fn bit(&mut self, b: Bit) -> usize {
        self.bits.push(b);
        self.bits.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    /// The devices of `level` and what they do, except funnels through portals
    /// (`funnel_route_moves`, which needs the regions). `None` if there are more than
    /// `MAX_BITS` before any routes.
    pub fn build(level: &Level) -> Option<Devices> {
        let mut d = Devices::default();
        for i in 0..level.stairs.len() {
            let b = d.bit(Bit::Stairs(i));
            d.stairs_bit.push(b);
        }
        for i in 0..level.panels.len() {
            let b = d.bit(Bit::Panel(i));
            d.panel_bit.push(b);
        }
        for i in 0..level.pistons.len() {
            let b = d.bit(Bit::Piston(i));
            d.piston_bit.push(b);
        }
        for (i, r) in level.rails.iter().enumerate() {
            let b = (!r.oscillate).then(|| d.bit(Bit::Rail(i)));
            d.rail_bit.push(b);
        }
        for (i, p) in level.projectors.iter().enumerate() {
            let b = d.bit(Bit::Projector(i));
            d.projector_bit.push(b);
            let r = (p.kind == ProjectorKind::Funnel).then(|| d.bit(Bit::Reversed(i)));
            d.reversed_bit.push(r);
        }
        if d.bits.len() > MAX_BITS {
            return None;
        }
        d.paths = level
            .projectors
            .iter()
            .map(|p| path_with(level, p.cell, p.normal, p.kind == ProjectorKind::Funnel))
            .collect();
        d.routes = vec![Vec::new(); level.projectors.len()];

        let mut stand: HashMap<Cell, Cond> = HashMap::new();
        let mut blocks: HashMap<(Cell, usize), Cond> = HashMap::new();
        let mut moves: Vec<DynMove> = Vec::new();
        let mut rides: HashMap<Cell, Cell> = HashMap::new();
        let mut pushes: HashMap<Cell, Cell> = HashMap::new();
        let mut incomplete = false;
        // Every device's claim to make a cell standable, or a face a wall, under a condition.
        // Until they're settled at the end (`settle`), the first counts (the funnels below see
        // that).
        let mut stand_claims: Vec<(Cell, Cond)> = Vec::new();
        let mut block_claims: Vec<((Cell, usize), Cond)> = Vec::new();
        fn claim(stand: &mut HashMap<Cell, Cond>, claims: &mut Vec<(Cell, Cond)>, c: Cell, cond: Cond) {
            claims.push((c, cond));
            stand.entry(c).or_insert(cond);
        }
        fn block(
            blocks: &mut HashMap<(Cell, usize), Cond>,
            claims: &mut Vec<((Cell, usize), Cond)>,
            k: (Cell, usize),
            cond: Cond,
        ) {
            claims.push((k, cond));
            blocks.entry(k).or_insert(cond);
        }

        for (s, &b) in level.stairs.iter().zip(&d.stairs_bit) {
            for c in [s.cell, step(s.cell, s.dir)] {
                for side in Dir::HORIZONTAL {
                    if side.axis() != s.dir.axis() {
                        block(&mut blocks, &mut block_claims, face_key(c, side), Cond::on(b));
                    }
                }
            }
            // An exit onto a faith plate launches whoever climbs there (Workshop 3718840820
            // "Flying a Timing": stairs up to the first of four plates).
            let (onto_plate, exits): (Vec<_>, Vec<_>) =
                s.exits(level).into_iter().partition(|&(t, _)| level.plate_at(t).is_some());
            let walked = exits.into_iter().map(|(t, h)| (Cond::ALWAYS, t, h)).collect();
            let launched = onto_plate
                .into_iter()
                .filter_map(|(t, h)| level.fall_route(t).map(|(f, fh, _)| (Cond::ALWAYS, f, h | fh)))
                .collect();
            for (to, kind) in [(walked, Move::Stairs), (launched, Move::Plate)] {
                moves.push(DynMove {
                    from: vec![s.cell, step(s.cell, s.dir)],
                    to,
                    kind,
                    cond: Cond::on(b),
                    hazards: 0,
                });
            }
        }
        for (p, &b) in level.panels.iter().zip(&d.panel_bit) {
            if p.kind == PanelKind::Flip {
                continue;
            }
            let (c, h) = (p.cell, p.hinge());
            match (p.normal, p.angle) {
                (Dir::PosZ, 45 | 60) if level.standable(c) => {
                    claim(&mut stand, &mut stand_claims, c, Cond::off(b));
                    let away = [p.tilt, h]
                        .into_iter()
                        .chain(Dir::HORIZONTAL)
                        .map(|d| step(c, d))
                        .find(|&n| level.standable(n) && level.face(c, n_dir(c, n)).barrier.is_none());
                    if let Some(n) = away {
                        rides.insert(c, n);
                    }
                }
                (_, 90) if h.axis() == 2 => {
                    // Lying flat on its hinge edge: a floor, for this cell or the one above.
                    let f = if h == Dir::NegZ { c } else { step(c, Dir::PosZ) };
                    if !level.standable(f) && !level.solid(f) && !level.goo(f) {
                        claim(&mut stand, &mut stand_claims, f, Cond::on(b));
                    }
                }
                (_, 90) => {
                    block(&mut blocks, &mut block_claims, face_key(c, h), Cond::on(b));
                }
                _ => {}
            }
        }
        for (p, &b) in level.pistons.iter().zip(&d.piston_bit) {
            if p.normal == Dir::PosZ {
                let (lo, hi) = (p.stand(false), p.stand(true));
                claim(&mut stand, &mut stand_claims, lo, Cond::off(b));
                claim(&mut stand, &mut stand_claims, hi, Cond::on(b));
                rides.insert(lo, hi);
                rides.insert(hi, lo);
                continue;
            }
            // Out of a wall or ceiling: the cells it fills only when up.
            let past = p.at(p.top);
            for l in p.bottom..p.top {
                let c = p.at(l);
                if p.normal.axis() != 2 && !level.solid(past) {
                    pushes.insert(c, past);
                }
                if level.standable(c) {
                    claim(&mut stand, &mut stand_claims, c, Cond::off(b));
                }
                let over = step(c, Dir::PosZ);
                if p.normal.axis() != 2 && !level.solid(over) && !level.standable(over) {
                    claim(&mut stand, &mut stand_claims, over, Cond::on(b));
                }
            }
            if p.bottom > 0 {
                incomplete = true; // Cells it always fills aren't solid in the level.
            }
        }
        for (r, bit) in level.rails.iter().zip(&d.rail_bit) {
            let track = r.stops();
            match *bit {
                Some(b) => {
                    let (first, last) = (track[0], *track.last().unwrap());
                    if first == last {
                        continue;
                    }
                    for (c, cond) in [(first, Cond::off(b)), (last, Cond::on(b))] {
                        if !level.solid(c) && !level.standable(c) {
                            claim(&mut stand, &mut stand_claims, c, cond);
                        }
                    }
                    rides.insert(first, last);
                    rides.insert(last, first);
                }
                None => {
                    for &c in &track {
                        if !level.solid(c) && !level.standable(c) && !level.goo(c) {
                            claim(&mut stand, &mut stand_claims, c, Cond::ALWAYS);
                        }
                    }
                    for w in track.windows(2) {
                        if w[0][2] != w[1][2] {
                            for (a, z) in [(w[0], w[1]), (w[1], w[0])] {
                                moves.push(DynMove {
                                    from: vec![a],
                                    to: vec![(Cond::ALWAYS, z, 0)],
                                    kind: Move::Ride,
                                    cond: Cond::ALWAYS,
                                    hazards: 0,
                                });
                            }
                        }
                    }
                }
            }
        }
        for (i, p) in level.projectors.iter().enumerate() {
            let b = d.projector_bit[i];
            let path = &d.paths[i];
            match p.kind {
                ProjectorKind::Bridge if p.side.axis() == 2 => {
                    for &c in &path.cells {
                        let f = bridge_floor(c, p.side);
                        if !level.solid(f) && !level.standable(f) && !level.goo(f) {
                            claim(&mut stand, &mut stand_claims, f, Cond::on(b));
                        }
                    }
                }
                ProjectorKind::Bridge => {
                    for &c in &path.cells {
                        block(&mut blocks, &mut block_claims, face_key(c, p.side), Cond::on(b));
                    }
                }
                ProjectorKind::Funnel => {
                    let r = d.reversed_bit[i].unwrap();
                    let base = Dyn {
                        stand: stand.clone(),
                        ..Dyn::default()
                    };
                    let geo = Geo::new(level, &base);
                    for (cells, cond) in [
                        (path.cells.clone(), Cond::on(b).and(Cond::off(r)).unwrap()),
                        (rev(&path.cells), Cond::on(b).and(Cond::on(r)).unwrap()),
                    ] {
                        moves.extend(funnel_moves(geo, &cells, cond));
                    }
                }
            }
        }
        // Bridges through portals: each portal face a bridge can come out of, grouped into
        // routes by the cells it is the floor of. Nothing here may depend on which way round the
        // level was built (claimed route by route in coordinate order, the first route took a
        // shared cell; with bits short, the first routes, or the first bridge's, got them):
        // routes of all bridges go in one order by how many faces and cells they have and how
        // far those are from their projector, routes alike in all of that get bits together or
        // not at all, and so do the cells several routes cross (`RoutedAny`, a floor while any
        // of them is on). A cell routes of two bridges cross isn't claimed: no one condition says
        // which bridge is there.
        type Group = (usize, Vec<Cell>, Vec<Face>);
        let mut all: Vec<Group> = Vec::new();
        for (i, p) in level.projectors.iter().enumerate() {
            let Some(end) = d.paths[i].end else { continue };
            if p.kind != ProjectorKind::Bridge {
                continue;
            }
            if p.side.axis() != 2 || end.1.axis() == 2 {
                incomplete = true; // A bridge on edge, or going into a floor or ceiling portal.
                continue;
            }
            let mut groups: BTreeMap<Vec<Cell>, Vec<Face>> = BTreeMap::new();
            for f in level.portal_surfaces() {
                if f == end || f.1.axis() == 2 {
                    continue;
                }
                let out = path(level, f.0, f.1.opposite());
                let new: Vec<Cell> = out
                    .cells
                    .iter()
                    .map(|&c| bridge_floor(c, p.side))
                    .filter(|&c| !level.solid(c) && !level.standable(c) && !level.goo(c))
                    .collect();
                if !new.is_empty() {
                    groups.entry(new).or_default().push(f);
                }
            }
            all.extend(groups.into_iter().map(|(cells, faces)| (i, cells, faces)));
        }
        let key = |(i, cells, faces): &Group| {
            let origin = level.projectors[*i].cell;
            let dist = |c: &Cell| (0..3).map(|a| (c[a] - origin[a]).abs()).sum::<i32>();
            (
                std::cmp::Reverse(faces.len()),
                std::cmp::Reverse(cells.len()),
                cells.iter().map(dist).sum::<i32>(),
                faces.iter().map(|f| dist(&f.0)).sum::<i32>(),
            )
        };
        all.sort_by_key(key);
        // Per cell: the projector whose routes (of `chosen`, by position) cross it, or `None`
        // if two projectors' do.
        fn cover(all: &[Group], chosen: &[usize]) -> BTreeMap<Cell, Option<(usize, u64)>> {
            let mut covered: BTreeMap<Cell, Option<(usize, u64)>> = BTreeMap::new();
            for (k, &g) in chosen.iter().enumerate() {
                let (i, cells, _) = &all[g];
                for &c in cells {
                    let e = covered.entry(c).or_insert(Some((*i, 0)));
                    *e = match *e {
                        Some((j, m)) if j == *i => Some((j, m | 1 << k)),
                        _ => None,
                    };
                }
            }
            covered
        }
        let shared = |covered: &BTreeMap<Cell, Option<(usize, u64)>>| {
            let sets: std::collections::BTreeSet<u64> = covered
                .values()
                .flatten()
                .map(|&(_, m)| m)
                .filter(|m| m.count_ones() > 1)
                .collect();
            sets
        };
        // Routes alike go in together, if their bits and the shared cells' fit; else on to the
        // next.
        let mut chosen: Vec<usize> = Vec::new();
        let mut k = 0;
        while k < all.len() {
            let tied = all[k..].iter().take_while(|g| key(g) == key(&all[k])).count();
            let with: Vec<usize> = chosen.iter().copied().chain(k..k + tied).collect();
            if d.bits.len() + with.len() + shared(&cover(&all, &with)).len() <= MAX_BITS {
                chosen = with;
            } else {
                incomplete = true;
            }
            k += tied;
        }
        let covered = cover(&all, &chosen);
        // Each chosen route's bit.
        let route_bit: Vec<usize> = chosen
            .iter()
            .map(|&g| {
                let (i, _, faces) = &all[g];
                let route = d.routes[*i].len();
                let b = d.bit(Bit::Routed {
                    projector: *i,
                    route,
                });
                d.routes[*i].push(Route {
                    bit: b,
                    faces: faces.clone(),
                });
                b
            })
            .collect();
        let mut any_bit: BTreeMap<u64, usize> = BTreeMap::new();
        for m in shared(&covered) {
            let on = (0..chosen.len()).filter(|&k| m >> k & 1 == 1).fold(0, |a, k| a | 1 << route_bit[k]);
            let b = d.bit(Bit::Any { on, off: 0 });
            any_bit.insert(m, b);
        }
        for (c, cover) in covered {
            let b = match cover {
                None => {
                    incomplete = true;
                    continue;
                }
                Some((_, m)) if m.count_ones() == 1 => route_bit[m.trailing_zeros() as usize],
                Some((_, m)) => any_bit[&m],
            };
            claim(&mut stand, &mut stand_claims, c, Cond::on(b));
        }
        // Settle the claims: a cell or face more than one device claims is a floor or wall while
        // any of them makes it one (a new bit, `Bit::Any`), and the same for all of them (the
        // first claim used to win, so the item order, which is coordinate order, decided it).
        // Claims that aren't one bit on or off each, or bits short for all the new ones: the
        // first claim, and the model is incomplete.
        let any_of = |conds: &[Cond]| -> Option<(u64, u64)> {
            let literal = |c: &Cond| c.on.count_ones() + c.off.count_ones() == 1;
            conds.iter().all(literal).then(|| conds.iter().fold((0, 0), |(on, off), c| (on | c.on, off | c.off)))
        };
        fn grouped<K: Ord + Copy>(claims: &[(K, Cond)]) -> BTreeMap<K, Vec<Cond>> {
            let mut out: BTreeMap<K, Vec<Cond>> = BTreeMap::new();
            for &(k, c) in claims {
                let v = out.entry(k).or_default();
                if !v.contains(&c) {
                    v.push(c);
                }
            }
            out
        }
        let (stands, walls) = (grouped(&stand_claims), grouped(&block_claims));
        let needed: std::collections::BTreeSet<(u64, u64)> = stands
            .values()
            .chain(walls.values())
            .filter(|v| v.len() > 1 && !v.contains(&Cond::ALWAYS))
            .filter_map(|v| any_of(v))
            .collect();
        let mut any_bit: BTreeMap<(u64, u64), usize> = BTreeMap::new();
        if d.bits.len() + needed.len() <= MAX_BITS {
            for (on, off) in needed {
                any_bit.insert((on, off), d.bit(Bit::Any { on, off }));
            }
        }
        let mut settle = |v: &Vec<Cond>| -> Cond {
            if v.len() == 1 {
                return v[0];
            }
            if v.contains(&Cond::ALWAYS) {
                return Cond::ALWAYS;
            }
            match any_of(v).and_then(|m| any_bit.get(&m)) {
                Some(&b) => Cond::on(b),
                None => {
                    incomplete = true;
                    v[0]
                }
            }
        };
        let stand: HashMap<Cell, Cond> = stands.iter().map(|(&c, v)| (c, settle(v))).collect();
        let blocks: HashMap<(Cell, usize), Cond> = walls.iter().map(|(&k, v)| (k, settle(v))).collect();
        d.dyn_ = Dyn {
            stand,
            blocks,
            moves,
            rides,
            pushes,
        };
        d.incomplete = incomplete;
        Some(d)
    }

    /// Funnels through portals: per funnel with a path ending at a white face, the faces whose
    /// way out carries the player somewhere, grouped by the moves they give (as edges of
    /// `geo`'s regions, via `region_of`), each group with a new bit. Returns the moves; adds the
    /// routes and bits.
    pub fn funnel_route_moves(
        &mut self,
        geo: Geo,
        region_of: impl Fn(Cell) -> Option<usize>,
    ) -> Vec<DynMove> {
        let level = geo.level;
        let mut out = Vec::new();
        for (i, p) in level.projectors.iter().enumerate() {
            let (Some(end), ProjectorKind::Funnel) = (self.paths[i].end, p.kind) else {
                continue;
            };
            let r = self.reversed_bit[i].unwrap();
            let direct = self.paths[i].cells.clone();
            // Cells some other portal's beam runs through: a player in the beam can be lifted
            // into one above them by moving the portal.
            let lift: HashSet<Cell> = level
                .portal_surfaces()
                .into_iter()
                .filter(|&f| f != end)
                .flat_map(|f| funnel_path(level, f.0, f.1.opposite()).cells)
                .collect();
            // Faces whose moves (with a placeholder condition) are the same, as region pairs.
            let mut groups: BTreeMap<Vec<Signature>, (Vec<Face>, Vec<DynMove>)> = BTreeMap::new();
            for f in level.portal_surfaces() {
                if f == end {
                    continue;
                }
                let out_path = funnel_path(level, f.0, f.1.opposite());
                let mut all = direct.clone();
                all.extend(out_path.cells.iter().copied());
                let mut moves = funnel_moves(geo, &all, Cond::on(0));
                let climbs = climb_moves(geo, &out_path.cells, &lift, Cond::on(0));
                let back = funnel_moves(geo, &rev(&all), Cond::off(0));
                moves.extend(climbs);
                moves.extend(back);
                let mut sig: Vec<Signature> = Vec::new();
                for m in &moves {
                    for &from in &m.from {
                        for &(c, to, h) in &m.to {
                            if let (Some(a), Some(b)) = (region_of(from), region_of(to)) {
                                if a != b {
                                    sig.push((a, b, m.hazards | h, m.cond.and(c).unwrap_or(c)));
                                }
                            }
                        }
                    }
                }
                sig.sort();
                sig.dedup();
                if sig.is_empty() {
                    continue;
                }
                let e = groups.entry(sig).or_insert_with(|| (Vec::new(), moves));
                e.0.push(f);
            }
            for (_, (faces, moves)) in groups {
                if self.bits.len() == MAX_BITS {
                    self.incomplete = true;
                    break;
                }
                let route = self.routes[i].len();
                let b = self.bit(Bit::Routed {
                    projector: i,
                    route,
                });
                self.routes[i].push(Route { bit: b, faces });
                // Placeholder bit 0 on = forward, off = reversed.
                for mut m in moves {
                    let forward = m.cond.on & 1 != 0;
                    let dir = if forward { Cond::off(r) } else { Cond::on(r) };
                    m.cond = Cond::on(b).and(dir).unwrap();
                    out.push(m);
                }
            }
        }
        out
    }

    /// Per face: the (projector, route) pairs whose routes come out of it.
    pub fn route_faces(&self) -> HashMap<Face, Vec<(usize, usize)>> {
        let mut out: HashMap<Face, Vec<(usize, usize)>> = HashMap::new();
        for (p, routes) in self.routes.iter().enumerate() {
            for (k, r) in routes.iter().enumerate() {
                for &f in &r.faces {
                    out.entry(f).or_default().push((p, k));
                }
            }
        }
        out
    }

    /// Per face: the projectors whose paths end there.
    pub fn path_ends(&self) -> HashMap<Face, Vec<usize>> {
        let mut out: HashMap<Face, Vec<usize>> = HashMap::new();
        for (p, path) in self.paths.iter().enumerate() {
            if let Some(e) = path.end {
                out.entry(e).or_default().push(p);
            }
        }
        out
    }
}

/// The cell a horizontal bridge crossing cell `c` is the floor of: `c` with its walkable side
/// up, the cell over `c` with it down.
pub fn bridge_floor(c: Cell, side: Dir) -> Cell {
    if side == Dir::NegZ {
        step(c, Dir::PosZ)
    } else {
        c
    }
}

/// A move a funnel through a portal gives, between regions: (from, to, hazards, condition).
type Signature = (usize, usize, u64, Cond);

/// The direction from cell `a` to its neighbour `b`.
fn n_dir(a: Cell, b: Cell) -> Dir {
    let ax = (0..3).find(|&i| a[i] != b[i]).unwrap_or(0);
    Dir::from_axis(ax, b[ax] > a[ax])
}

fn rev(cells: &[Cell]) -> Vec<Cell> {
    cells.iter().rev().copied().collect()
}

/// The cells a projection from open cell `c` along `d` crosses, and the face it ends at if
/// portals can go there.
pub fn path(level: &Level, c: Cell, d: Dir) -> Path {
    path_with(level, c, d, false)
}

/// `path` for a funnel, which (measured by a play-tester, 2026-10-24) goes on through grating
/// but not glass.
pub fn funnel_path(level: &Level, c: Cell, d: Dir) -> Path {
    path_with(level, c, d, true)
}

fn path_with(level: &Level, c: Cell, d: Dir, through_grating: bool) -> Path {
    let mut cells = Vec::new();
    let mut cur = c;
    let mut end = None;
    if !level.solid(cur) {
        loop {
            cells.push(cur);
            let n = step(cur, d);
            if level.solid(n) {
                if level.portal_surface(cur, d) {
                    end = Some((cur, d));
                }
                break;
            }
            let stops = match level.face(cur, d).barrier {
                None => false,
                Some(BarrierKind::Grating) => !through_grating,
                Some(_) => true,
            };
            if stops || cells.len() > 64 {
                break;
            }
            cur = n;
        }
    }
    Path {
        cells,
        dir: d,
        end,
    }
}

/// Rides up a funnel beam that comes out of a portal: the player, in the beam, has the other
/// portal moved to a face whose beam runs through the cell above them (about half a panel
/// higher each time), and so on, stepping out sideways anywhere on the way up. `start` are
/// the cells of the beam out of the portal they step into (as `funnel_moves` does) and `lift`
/// all cells such a beam can run through.
fn climb_moves(geo: Geo, start: &[Cell], lift: &HashSet<Cell>, cond: Cond) -> Vec<DynMove> {
    let level = geo.level;
    let mut out = Vec::new();
    for &f in start {
        // The cells above, up the column, while a beam can reach them.
        let mut risen: Vec<(Cell, u64)> = Vec::new();
        let (mut q, mut hz) = (f, 0);
        loop {
            let up = step(q, Dir::PosZ);
            let face = level.face(q, Dir::PosZ);
            if level.solid(up) || face.barrier.is_some() || !lift.contains(&up) || risen.len() >= 32 {
                break;
            }
            hz |= face.hazards;
            risen.push((up, hz));
            q = up;
        }
        if risen.is_empty() {
            continue;
        }
        let ids: Vec<Cell> = risen.iter().map(|r| r.0).collect();
        let mut to = Vec::new();
        for (k, outs) in funnel_exits(geo, &ids).into_iter().enumerate() {
            for (c, cell, h) in outs {
                to.push((c, cell, h | risen[k].1));
            }
        }
        if to.is_empty() {
            continue;
        }
        // Where they step in, with the fields crossed stepping in.
        let mut from: Vec<(Cell, u64)> = Vec::new();
        if geo.stand(f).is_some() {
            from.push((f, 0));
        }
        for d in Dir::HORIZONTAL {
            let m = step(f, d);
            let face = level.face(f, d);
            if geo.stand(m).is_some() && face.barrier.is_none() && geo.block(f, d).is_none() {
                from.push((m, face.hazards));
            }
        }
        let mut entries: Vec<u64> = from.iter().map(|&(_, h)| h).collect();
        entries.sort();
        entries.dedup();
        for h in entries {
            out.push(DynMove {
                from: from.iter().filter(|e| e.1 == h).map(|e| e.0).collect(),
                to: to.iter().map(|&(c, cell, x)| (c, cell, x | h)).collect(),
                kind: Move::Funnel,
                cond,
                hazards: h,
            });
        }
    }
    out
}

/// Ways out of each funnel cell sideways, with the fields crossed stepping out.
fn funnel_exits(geo: Geo, cells: &[Cell]) -> Vec<Vec<(Cond, Cell, u64)>> {
    let level = geo.level;
    cells
        .iter()
        .map(|&f| {
            let mut out = Vec::new();
            for d in Dir::HORIZONTAL {
                let m = step(f, d);
                let face = level.face(f, d);
                if level.solid(m) || face.barrier.is_some() || geo.block(f, d).is_some() {
                    continue;
                }
                match geo.stand(m) {
                    Some(c) => out.push((c, m, face.hazards)),
                    None => {
                        for fall in geo.falls(m) {
                            out.push((fall.cond, fall.cell, face.hazards | fall.hazards));
                        }
                    }
                }
            }
            out
        })
        .collect()
}

/// Rides along a funnel whose cells are `cells` in the order it carries the player: from
/// where they can step in (a cell itself if they can stand there, or a standable cell beside
/// one at its height) to everywhere they can step out sideways further along.
pub fn funnel_moves(geo: Geo, cells: &[Cell], cond: Cond) -> Vec<DynMove> {
    let level = geo.level;
    let n = cells.len();
    let exits = funnel_exits(geo, cells);
    // Fields crossed between consecutive cells.
    let walled: Vec<bool> = cells
        .windows(2)
        .map(|w| {
            let apart: i32 = (0..3).map(|a| (w[0][a] - w[1][a]).abs()).sum();
            apart == 1 && level.face(w[0], n_dir(w[0], w[1])).barrier.is_some()
        })
        .collect();
    let along: Vec<u64> = cells
        .windows(2)
        .map(|w| {
            let apart: i32 = (0..3).map(|a| (w[0][a] - w[1][a]).abs()).sum();
            if apart == 1 {
                level.face(w[0], n_dir(w[0], w[1])).hazards
            } else {
                0 // Through a portal.
            }
        })
        .collect();
    let mut seen: HashSet<(Cell, u64)> = HashSet::new();
    let mut out = Vec::new();
    for (i, &f) in cells.iter().enumerate() {
        // Where they step in from, with the fields crossed stepping in.
        let mut from: Vec<(Cell, u64)> = Vec::new();
        if geo.stand(f).is_some() {
            from.push((f, 0));
        }
        for d in Dir::HORIZONTAL {
            let m = step(f, d);
            let face = level.face(f, d);
            if geo.stand(m).is_some() && face.barrier.is_none() && geo.block(f, d).is_none() {
                from.push((m, face.hazards));
            }
        }
        // A cell that steps in earlier the same way gets nothing new here.
        from.retain(|&(c, h)| seen.insert((c, h)));
        if from.is_empty() {
            continue;
        }
        let mut to = Vec::new();
        let mut hz = 0;
        for j in i..n {
            if j > i {
                if walled[j - 1] {
                    break;
                }
                hz |= along[j - 1];
            }
            for &(c, cell, h) in &exits[j] {
                to.push((c, cell, h | hz));
            }
        }
        if to.is_empty() {
            continue;
        }
        let mut entries: Vec<u64> = from.iter().map(|&(_, h)| h).collect();
        entries.sort();
        entries.dedup();
        for h in entries {
            out.push(DynMove {
                from: from.iter().filter(|e| e.1 == h).map(|e| e.0).collect(),
                to: to.clone(),
                kind: Move::Funnel,
                cond,
                hazards: h,
            });
        }
    }
    out
}
