//! Plans made concrete, for playing them in the game: where to stand, what to look at, which
//! face each portal goes on and where each cube ends up.
//!
//! `route` replays the solver's conservative plan and grounds each action in cells and faces.
//! It keeps a concrete state (the player's cell, each portal's face, each cube's cell) and
//! checks after every step that it abstracts to the solver's state, so a route does what the
//! plan does. Positions are in voxels: cell `c` spans `c..c + 1` on each axis, so its floor
//! centre is `(x + 0.5, y + 0.5, z)` and the eye of a player standing there is 0.5 above that.
//!
//! Choices the plan leaves open (which cell to fire from, which face of a class, where to put
//! a cube down) go to the nearest cell on foot, avoiding cells with something in the way
//! (cubes, buttons, pedestals, turrets, gel) unless there is no other way.
//!
//! What devices do by themselves after a step (a platform carrying the player, a bridge
//! vanishing from under a cube, a flip panel losing its portal, a dropper dropping) is applied
//! to the concrete state the way the solver's `settle` does (`Grounder::settle`).

use std::collections::{BinaryHeap, HashMap, VecDeque};

use crate::dataset::json_str;
use crate::dynamics;
use crate::flings;
use crate::gel;
use crate::geom::{step, Cell, Dir};
use crate::level::{CubeKind, FaceInfo, ItemKind, Level, PaintKind, PanelKind, ProjectorKind};
use crate::metrics::replay;
use crate::portals::{self, Face};
use crate::regions::{Cond, Geo, Move};
use crate::solver::{Action, Color, Cube, CubeLoc, Mode, Model, Signal, State, Verdict};

const VOXEL: f64 = 128.0;
const GRAVITY: f64 = 600.0;
/// Shots preferably meet the face at least this steeply (cosine to its normal, about 25°).
const MIN_SHOT_COS: f64 = 0.42;

/// Where things are, concretely.
#[derive(Clone, Debug, PartialEq)]
pub struct Here {
    pub player: Cell,
    /// `[blue, orange]`.
    pub portals: [Option<Face>; 2],
    /// In a fixed order for the whole route.
    pub cubes: Vec<Placed>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placed {
    pub kind: CubeKind,
    /// Where it comes back to after a fizzler (where it first came to rest).
    pub home: Cell,
    /// Placed without a dropper: fizzled, it's gone for good.
    pub loose: bool,
    /// `None` while held.
    pub at: Option<Cell>,
    /// Destroyed, or still waiting in its dropper.
    pub gone: bool,
    /// The laser spot (`Beams::spots`) it was set down at, facing its way.
    pub spot: Option<usize>,
}

/// What a leg does once the player has walked to where it starts.
#[derive(Clone, Debug, PartialEq)]
pub enum Act {
    /// Nothing more: the walk is the step (a walk across a field or into new sight lines, or
    /// a drop off a ledge at its end).
    Walk,
    /// Fire from `eye` (a point in the walk's last cell, `cell`) at the centre of `face`, or at
    /// `aim` (a deployed angled panel's middle).
    Fire {
        color: Color,
        face: Face,
        cell: Cell,
        eye: [f64; 3],
        aim: Option<[f64; 3]>,
    },
    /// Walk into the portal on `face` (from the cell in front of it, which the walk ends in),
    /// coming out of the one on `out`.
    Enter {
        color: Color,
        face: Face,
        out: Face,
    },
    PickUp {
        cube: Cell,
    },
    /// Put the held cube down in `at`, next to the walk's end; facing `facing` (from the cell
    /// behind it) to turn a laser.
    PutDown {
        at: Cell,
        button: bool,
        facing: Option<Dir>,
    },
    /// Drop the held cube off the ledge of `edge` (the walk's end) on side `dir`.
    Toss {
        edge: Cell,
        dir: Dir,
        landing: Cell,
    },
    /// Throw the held cube into the portal on `face`; it comes out of `out` and rests at
    /// `landing`.
    Throw {
        color: Color,
        face: Face,
        out: Face,
        landing: Cell,
    },
    Press {
        pedestal: i32,
        cell: Cell,
    },
    /// Press the timed pedestal, then walk `path` (each cell one step or one drop from the
    /// last) before `seconds` run out; `escape`: out through the exit door at its end.
    Dash {
        pedestal: i32,
        cell: Cell,
        path: Vec<Cell>,
        seconds: f64,
        escape: bool,
    },
    Knock {
        turret: i32,
        cell: Cell,
    },
    /// Bounce off the repulsion gel under `gel` (the walk's end) up to `to`.
    Bounce {
        gel: Cell,
        to: Cell,
    },
    /// Run along `run` (painted with propulsion gel, from the walk's end to the take-off cell)
    /// in direction `dir` and leap across to `to`.
    Leap {
        run: Vec<Cell>,
        dir: Dir,
        to: Cell,
    },
    /// Step from `from` (the walk's end) toward `dir` onto the faith plate in `plate`, and
    /// come down in `to`.
    Plate {
        from: Cell,
        dir: Dir,
        plate: Cell,
        to: Cell,
    },
    /// Walk up deployed stairs: `path` is the stairs' second cell and the top.
    Climb {
        path: Vec<Cell>,
    },
    /// Step from `enter` (the walk's end) into the funnel, float along `via` (its cells, in
    /// order) and step out from the last of them into `out`, coming to rest in `to`.
    Funnel {
        enter: Cell,
        via: Vec<Cell>,
        out: Cell,
        to: Cell,
    },
    /// Ride a platform from stop `from` (the walk's end) to stop `to`.
    Ride {
        from: Cell,
        to: Cell,
    },
    /// Jump off the ledge of `edge` (the walk's end) toward `dir`, `out` cells out, into the
    /// floor portal on `face`; fly out of `exit` and land in `landing`.
    Dive {
        edge: Cell,
        dir: Dir,
        out: i32,
        face: Face,
        exit: Face,
        landing: Cell,
    },
    /// Walk out through the exit door in `cell`, against its `normal`.
    Exit {
        cell: Cell,
        normal: Dir,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Leg {
    /// The plan step (`Model::describe`), or "walk out of the exit".
    pub text: String,
    /// Cells to walk through first, in order, each one step from the last (the player's cell
    /// before the first). Every cell of a walk is standable.
    pub walk: Vec<Cell>,
    pub act: Act,
    /// Where things are afterwards.
    pub after: Here,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Route {
    /// The entry door's cell and normal (pointing into the room).
    pub entry: (Cell, Dir),
    pub exit: (Cell, Dir),
    pub start: Here,
    pub legs: Vec<Leg>,
}

/// Grounds the conservative plan of `level` (searching at most `budget` states).
pub fn route(level: &Level, budget: usize) -> Result<Route, String> {
    let model = Model::new(level, Mode::Conservative).map_err(|p| format!("{p:?}"))?;
    let plan = match model.solve(budget).verdict {
        Verdict::Solved(plan) => plan,
        Verdict::Unsolvable => return Err("not solvable under conservative rules".into()),
        Verdict::Unknown => return Err("the state budget ran out".into()),
    };
    let states = replay(&model, &plan).ok_or("the plan doesn't replay")?;
    let g = Grounder::new(level, &model);
    let start = g.settle(g.start()?, &states[0]);
    g.check(&start, &states[0])
        .map_err(|e| format!("start: {e}"))?;
    let mut here = start.clone();
    let mut legs = Vec::new();
    for (i, a) in plan.iter().enumerate() {
        let leg = g
            .leg(&here, &states[i], a, &states[i + 1])
            .ok_or_else(|| format!("step {}: can't ground \"{}\"", i + 1, model.describe(a)))?;
        g.check(&leg.after, &states[i + 1])
            .map_err(|e| format!("step {}: {e}", i + 1))?;
        here = leg.after.clone();
        legs.push(leg);
    }
    let last = states.last().unwrap();
    if !last.escaped {
        legs.push(g.exit(&here, last).ok_or("can't walk to the exit door")?);
    }
    Ok(Route {
        entry: g.entry,
        exit: g.exit_door,
        start,
        legs,
    })
}

struct Grounder<'a> {
    level: &'a Level,
    model: &'a Model,
    geo: Geo<'a>,
    class_of: HashMap<Face, usize>,
    face_id: HashMap<Face, usize>,
    entry: (Cell, Dir),
    exit_door: (Cell, Dir),
}

impl<'a> Grounder<'a> {
    fn new(level: &'a Level, model: &'a Model) -> Grounder<'a> {
        let door = |kind: ItemKind| {
            let d = level.first(&kind).expect("the model checked the doors");
            (d.cell, d.normal.unwrap_or(Dir::PosZ))
        };
        Grounder {
            level,
            model,
            geo: Geo::new(level, &model.devices.dyn_),
            class_of: model
                .portals
                .classes
                .iter()
                .flat_map(|c| c.faces.iter().map(move |&f| (f, c.id)))
                .collect(),
            face_id: model
                .paint_faces
                .iter()
                .enumerate()
                .map(|(i, &f)| (f, i))
                .collect(),
            entry: door(ItemKind::EntryDoor),
            exit_door: door(ItemKind::ExitDoor),
        }
    }

    fn start(&self) -> Result<Here, String> {
        let cubes = self
            .level
            .of_kind(&ItemKind::Cube)
            .filter_map(|c| {
                let home = self.level.fall(c.cell)?;
                let loose = self.level.loose_cube(c.index);
                // A dropper that waits for its inputs holds its cube back (`settle` drops it).
                let waits = self
                    .level
                    .dropper_of
                    .get(&c.index)
                    .and_then(|&d| self.level.item(d))
                    .is_some_and(|d| {
                        d.props.get_i32("ITEM_PROPERTY_AUTO_DROP_CUBE") == Some(0)
                            && self.level.inputs(d.index).next().is_some()
                    });
                Some(Placed {
                    kind: c.cube_kind(),
                    home,
                    loose,
                    at: (!waits).then_some(home),
                    gone: waits,
                    spot: None,
                })
            })
            .collect();
        Ok(Here {
            player: self.entry.0,
            portals: [None, None],
            cubes,
        })
    }

    fn region(&self, c: Cell) -> Option<usize> {
        self.model.graph.region_of(c)
    }

    /// Whether cell `c` can be stood in with the devices in `dev`.
    fn stands(&self, c: Cell, dev: u64) -> bool {
        self.geo.stand(c).is_some_and(|k| k.holds(dev))
    }

    /// Where something falling from open cell `c` comes to rest with the devices in `dev`, the
    /// fields it falls through, and the faith plate that launched it on the way.
    fn fall_at(&self, c: Cell, dev: u64) -> Option<(Cell, u64, Option<usize>)> {
        self.geo
            .falls(c)
            .into_iter()
            .find(|f| f.cond.holds(dev))
            .map(|f| (f.cell, f.hazards, f.plate))
    }

    /// Where whoever is in `c` ends up with the devices in `dev`: there if they can still
    /// stand there, else carried by a platform, or falling.
    fn carry(&self, c: Cell, dev: u64) -> Option<Cell> {
        if self.stands(c, dev) {
            return Some(c);
        }
        if let Some(&p) = self.geo.dyn_.rides.get(&c) {
            if self.stands(p, dev) {
                return Some(p);
            }
        }
        self.fall_at(c, dev).map(|(f, _, _)| f)
    }

    /// The angled panel on face `f` that is deployed in `dev`: (hinge side, angle).
    fn deployed_panel(&self, f: Face, dev: u64) -> Option<(Dir, i32)> {
        let (i, p) = self
            .level
            .panels
            .iter()
            .enumerate()
            .find(|(_, p)| p.kind == PanelKind::Angled && (p.cell, p.normal.opposite()) == f)?;
        (dev & (1 << self.model.devices.panel_bit[i]) != 0).then_some((p.hinge(), p.angle))
    }

    /// The face a portal on `f` really is with the devices in `dev`: a deployed 90° panel's
    /// looks across its hinge edge.
    fn live_face(&self, f: Face, dev: u64) -> Face {
        match self.deployed_panel(f, dev) {
            Some((hinge, 90)) => (step(f.0, hinge), hinge.opposite()),
            _ => f,
        }
    }

    /// What follows by itself after a step, as the solver's `settle` has it in `next`: portals
    /// on flipped panels gone, the player and cubes carried or dropped by devices, cubes
    /// dropped by droppers that were waiting.
    fn settle(&self, mut h: Here, next: &State) -> Here {
        let dev = next.dev;
        for k in 0..2 {
            if next.portals[k].is_none() {
                h.portals[k] = None;
            }
        }
        if !next.escaped {
            if let Some(c) = self.carry(h.player, dev) {
                h.player = c;
            }
        }
        for q in &mut h.cubes {
            if let Some(c) = q.at.filter(|_| !q.gone) {
                match self.carry(c, dev) {
                    Some(n) if n != c => {
                        q.at = Some(n);
                        q.spot = None;
                    }
                    Some(_) => {}
                    None if q.loose => q.gone = true,
                    None => q.at = Some(q.home),
                }
            }
        }
        for kind in [
            CubeKind::Standard,
            CubeKind::Companion,
            CubeKind::Reflection,
            CubeKind::Sphere,
            CubeKind::Franken,
        ] {
            let want = next
                .cubes
                .iter()
                .filter(|c| c.kind == kind && c.loc == CubeLoc::Gone)
                .count();
            let mut have = h
                .cubes
                .iter()
                .filter(|c| c.kind == kind && c.gone)
                .count();
            for q in h.cubes.iter_mut().filter(|c| c.kind == kind && c.gone && !c.loose) {
                if have <= want {
                    break;
                }
                q.gone = false;
                q.at = Some(q.home);
                have -= 1;
            }
        }
        h
    }

    /// `after`, settled (`settle`), if it is what the solver means by `next`.
    fn fits(&self, after: Here, next: &State) -> Option<Here> {
        let after = self.settle(after, next);
        self.matches(&after, next).then_some(after)
    }

    /// The solver's view of a cube at rest in `cell`: on a button that takes it and that no
    /// earlier cube in `claimed` holds, or loose in its region.
    fn loc_at(&self, cell: Cell, kind: CubeKind, claimed: Option<&mut Vec<bool>>) -> CubeLoc {
        let button = self
            .model
            .buttons
            .iter()
            .position(|b| b.cell == cell && b.kind.accepts(kind));
        match (button, claimed) {
            (Some(b), Some(claimed)) if !claimed[b] => {
                claimed[b] = true;
                CubeLoc::OnButton(b)
            }
            (Some(b), None) => CubeLoc::OnButton(b),
            _ => CubeLoc::Loose(self.region(cell).unwrap_or(usize::MAX)),
        }
    }

    /// The solver's cubes for `h`, in `h.cubes`' order (not sorted).
    fn cubes(&self, h: &Here) -> Vec<Cube> {
        let mut claimed = vec![false; self.model.buttons.len()];
        h.cubes
            .iter()
            .map(|c| Cube {
                kind: c.kind,
                home: if c.loose {
                    CubeLoc::Gone
                } else {
                    self.loc_at(c.home, c.kind, None)
                },
                loc: match (c.gone, c.at, c.spot) {
                    (true, ..) => CubeLoc::Gone,
                    (false, None, _) => CubeLoc::Held,
                    (false, Some(_), Some(k)) => CubeLoc::Spot(k),
                    (false, Some(cell), None) => self.loc_at(cell, c.kind, Some(&mut claimed)),
                },
            })
            .collect()
    }

    /// Whether `h` is what the solver means by `s` (paint, pedestals and turrets aside).
    fn check(&self, h: &Here, s: &State) -> Result<(), String> {
        if !s.escaped && self.region(h.player) != Some(s.region) {
            return Err(format!(
                "the player is in region {:?}, the plan has {}",
                self.region(h.player),
                s.region
            ));
        }
        let classes = h
            .portals
            .map(|p| p.and_then(|f| self.class_of.get(&f).copied()));
        if classes != s.portals {
            return Err(format!(
                "portals in classes {classes:?}, the plan has {:?}",
                s.portals
            ));
        }
        let mut cubes = self.cubes(h);
        cubes.sort();
        if cubes != s.cubes {
            return Err(format!("cubes {cubes:?}, the plan has {:?}", s.cubes));
        }
        Ok(())
    }

    fn matches(&self, h: &Here, s: &State) -> bool {
        self.check(h, s).is_ok()
    }

    /// A fizzler: portals gone, a held cube back home (or gone, if it had no dropper).
    fn fizzle(h: &mut Here) {
        h.portals = [None, None];
        for c in &mut h.cubes {
            if c.at.is_none() && !c.gone {
                if c.loose {
                    c.gone = true;
                } else {
                    c.at = Some(c.home);
                }
            }
        }
    }

    fn held(h: &Here) -> Option<usize> {
        h.cubes.iter().position(|c| c.at.is_none() && !c.gone)
    }

    /// Something stands in `c` that the player shouldn't walk into, stand on to act, or put a
    /// cube on: a cube, a floor button, a pedestal or a turret, gel on the floor, an open
    /// portal in the floor (walking over it, the player falls in), or a faith plate.
    fn in_the_way(&self, h: &Here, s: &State, c: Cell) -> bool {
        let open = h.portals.iter().all(Option::is_some);
        (open && h.portals.contains(&Some((c, Dir::NegZ))))
            || h.cubes.iter().any(|q| q.at == Some(c) && !q.gone)
            || self.model.buttons.iter().any(|b| b.cell == c)
            || self.model.pedestals.iter().any(|p| p.cell == c)
            || self.model.turrets.iter().any(|t| t.cell == c)
            || self.level.plate_at(c).is_some()
            || self.face_id.get(&(c, Dir::NegZ)).is_some_and(|&f| {
                [PaintKind::Bounce, PaintKind::Speed]
                    .iter()
                    .any(|&k| self.model.painted(s, f, k))
            })
    }

    /// Neighbours of `c` the player can walk to without leaving its region.
    fn steps_in_region(&self, c: Cell) -> impl Iterator<Item = Cell> + '_ {
        let r = self.region(c);
        Dir::HORIZONTAL.into_iter().filter_map(move |d| {
            let n = step(c, d);
            (self.level.face(c, d) == FaceInfo::default()
                && self.geo.block(c, d).is_none()
                && r.is_some()
                && self.region(n) == r)
                .then_some(n)
        })
    }

    /// The nearest cell on foot in the player's region where `try_at` finds something to do,
    /// with the walk there: first never walking through a cell with something in the way,
    /// then if need be through anything.
    fn find<T>(
        &self,
        h: &Here,
        s: &State,
        mut try_at: impl FnMut(Cell) -> Option<T>,
    ) -> Option<(Vec<Cell>, T)> {
        for avoid in [true, false] {
            let mut prev: HashMap<Cell, Cell> = HashMap::from([(h.player, h.player)]);
            let mut queue = VecDeque::from([h.player]);
            while let Some(c) = queue.pop_front() {
                if let Some(t) = try_at(c) {
                    let mut walk = Vec::new();
                    let mut x = c;
                    while x != h.player {
                        walk.push(x);
                        x = prev[&x];
                    }
                    walk.reverse();
                    return Some((walk, t));
                }
                if avoid && c != h.player && self.in_the_way(h, s, c) {
                    continue;
                }
                for n in self.steps_in_region(c) {
                    if let std::collections::hash_map::Entry::Vacant(e) = prev.entry(n) {
                        e.insert(c);
                        queue.push_back(n);
                    }
                }
            }
        }
        None
    }

    /// Cells next to `c` in its region (to stand in while handling something in `c`).
    fn beside(&self, c: Cell) -> Vec<Cell> {
        self.steps_in_region(c).collect()
    }

    /// The funnels flowing in `s` and where: each one's cells in the order it carries.
    fn funnels(&self, h: &Here, s: &State) -> Vec<Vec<Cell>> {
        let (level, devices) = (self.level, &self.model.devices);
        let mut out = Vec::new();
        for (i, p) in level.projectors.iter().enumerate() {
            if p.kind != ProjectorKind::Funnel || s.dev & (1 << devices.projector_bit[i]) == 0 {
                continue;
            }
            let rev = devices.reversed_bit[i].is_some_and(|b| s.dev & (1 << b) != 0);
            let mut cells = devices.paths[i].cells.clone();
            // On through a portal at its end, out of the other one.
            let end = devices.paths[i].end;
            let k = end.and_then(|e| h.portals.iter().position(|&p| p == Some(e)));
            if let Some(out) = k.and_then(|k| h.portals[1 - k]) {
                cells.extend(dynamics::path(level, out.0, out.1.opposite()).cells);
            }
            if rev {
                cells.reverse();
            }
            out.push(cells);
        }
        out
    }

    fn leg(&self, h: &Here, s: &State, a: &Action, next: &State) -> Option<Leg> {
        let (level, model) = (self.level, self.model);
        let on = model.hazards_on_in(s, None);
        let dev = s.dev;
        let fizzles = |hazards: u64, on: u64| hazards & on & model.fizzlers != 0;
        let rules = model.rules;
        let found: Option<(Vec<Cell>, (Act, Here))> = match *a {
            Action::Move { to, kind, hazards } => match kind {
                Move::Walk | Move::Drop { .. } | Move::Plate => self
                    .find(h, s, |c| {
                        Dir::HORIZONTAL.into_iter().find_map(|d| {
                            let n = step(c, d);
                            let face = level.face(c, d);
                            let blocked = self.geo.block(c, d).is_some_and(|b| b.holds(dev));
                            if face.barrier.is_some() || level.solid(n) || blocked {
                                return None;
                            }
                            let (f, hz, plate) = if self.stands(n, dev) {
                                (n, face.hazards, None)
                            } else {
                                let (f, fallen, plate) = self.fall_at(n, dev)?;
                                (f, face.hazards | fallen, plate)
                            };
                            let fits_kind = match kind {
                                Move::Walk => plate.is_none() && f == n,
                                Move::Drop { height } => {
                                    plate.is_none() && f != n && n[2] - f[2] == height
                                }
                                _ => plate.is_some(),
                            };
                            if !fits_kind || hz != hazards || self.region(f) != Some(to) {
                                return None;
                            }
                            let mut after = h.clone();
                            after.player = f;
                            if fizzles(hz, on) {
                                Self::fizzle(&mut after);
                            }
                            let act = match plate {
                                Some(p) => Act::Plate {
                                    from: c,
                                    dir: d,
                                    plate: level.plates[p].cell,
                                    to: f,
                                },
                                None => Act::Walk,
                            };
                            self.fits(after, next).map(|after| (act, after))
                        })
                    })
                    .map(|(mut walk, (act, after))| {
                        if act == Act::Walk {
                            walk.push(after.player);
                        }
                        (walk, (act, after))
                    }),
                Move::Bounce => self.find(h, s, |c| {
                    let &f = self.face_id.get(&(c, Dir::NegZ))?;
                    if !model.painted(s, f, PaintKind::Bounce) {
                        return None;
                    }
                    gel::bounces(level, c, rules.gel)
                        .into_iter()
                        .find_map(|(t, hz)| {
                            if hz != hazards || self.region(t) != Some(to) {
                                return None;
                            }
                            let mut after = h.clone();
                            after.player = t;
                            if fizzles(hz, on) {
                                Self::fizzle(&mut after);
                            }
                            self.fits(after, next)
                                .map(|after| (Act::Bounce { gel: c, to: t }, after))
                        })
                }),
                // `c` is where the run starts, `leap_run - 1` cells back from the take-off.
                Move::Leap => self.find(h, s, |c| {
                    Dir::HORIZONTAL.into_iter().find_map(|d| {
                        let back = rules.gel.leap_run - 1;
                        let v = d.vector();
                        let off = [c[0] + v[0] * back, c[1] + v[1] * back, c[2]];
                        let (mut run, t, hz) = gel::leap(level, off, d, rules.gel)?;
                        let painted = run.iter().all(|&r| {
                            self.face_id
                                .get(&(r, Dir::NegZ))
                                .is_some_and(|&f| model.painted(s, f, PaintKind::Speed))
                        });
                        if !painted || hz != hazards || self.region(t) != Some(to) {
                            return None;
                        }
                        run.reverse();
                        let mut after = h.clone();
                        after.player = t;
                        if fizzles(hz, on) {
                            Self::fizzle(&mut after);
                        }
                        self.fits(after, next)
                            .map(|after| (Act::Leap { run, dir: d, to: t }, after))
                    })
                }),
                Move::Stairs => {
                    let stairs = level.stairs.iter().enumerate().find(|(i, st)| {
                        dev & (1 << model.devices.stairs_bit[*i]) != 0
                            && self.region(st.top()) == Some(to)
                    });
                    stairs.and_then(|(_, st)| {
                        let second = step(st.cell, st.dir);
                        self.find(h, s, |c| {
                            if c != st.cell && c != second {
                                return None;
                            }
                            let mut after = h.clone();
                            after.player = st.top();
                            let path = if c == second {
                                vec![st.top()]
                            } else {
                                vec![second, st.top()]
                            };
                            self.fits(after, next).map(|after| (Act::Climb { path }, after))
                        })
                    })
                }
                Move::Funnel => {
                    let mut found = None;
                    for cells in self.funnels(h, s) {
                        let moves = dynamics::funnel_moves(self.geo, &cells, Cond::ALWAYS);
                        found = self.find(h, s, |c| {
                            moves.iter().filter(|m| m.from.contains(&c)).find_map(|m| {
                                m.to.iter().find_map(|&(cond, t, th)| {
                                    if !cond.holds(dev)
                                        || m.hazards | th != hazards
                                        || self.region(t) != Some(to)
                                    {
                                        return None;
                                    }
                                    let i = cells.iter().position(|&f| {
                                        f == c || Dir::HORIZONTAL.iter().any(|&d| step(f, d) == c)
                                    })?;
                                    // Where it steps out: a funnel cell further on with `t` beside
                                    // it, or under the cell beside it.
                                    let (j, out) = (i..cells.len()).find_map(|j| {
                                        Dir::HORIZONTAL.iter().find_map(|&d| {
                                            let m = step(cells[j], d);
                                            let lands = m == t
                                                || self.fall_at(m, dev).is_some_and(|f| f.0 == t);
                                            lands.then_some((j, m))
                                        })
                                    })?;
                                    let mut after = h.clone();
                                    after.player = t;
                                    if fizzles(hazards, on) {
                                        Self::fizzle(&mut after);
                                    }
                                    let act = Act::Funnel {
                                        enter: c,
                                        via: cells[i..=j].to_vec(),
                                        out,
                                        to: t,
                                    };
                                    self.fits(after, next).map(|after| (act, after))
                                })
                            })
                        });
                        if found.is_some() {
                            break;
                        }
                    }
                    found
                }
                Move::Ride => {
                    let pairs: Vec<(Cell, Cell)> = level
                        .rails
                        .iter()
                        .flat_map(|r| {
                            let t = r.track();
                            t.windows(2)
                                .flat_map(|w| [(w[0], w[1]), (w[1], w[0])])
                                .collect::<Vec<_>>()
                        })
                        .filter(|&(_, b)| self.region(b) == Some(to))
                        .collect();
                    self.find(h, s, |c| {
                        let &(from, to_cell) = pairs.iter().find(|p| p.0 == c)?;
                        let mut after = h.clone();
                        after.player = to_cell;
                        self.fits(after, next)
                            .map(|after| (Act::Ride { from, to: to_cell }, after))
                    })
                }
                Move::Step => None,
            },

            Action::Fire {
                color,
                class,
                stand_on,
            } => {
                let k = color as usize;
                let on = model.hazards_on_in(s, stand_on.map(Signal::Floor));
                let faces = &model.portals.classes[class].faces;
                let other = h.portals[1 - k];
                // A shot along a wall from close to it rarely lands in the game: first look for
                // a face seen at a fair angle (from a farther cell if need be), then any.
                let at_angle = |min_cos: f64| {
                    self.find(h, s, |c| {
                        match stand_on {
                            Some(b) if c != model.buttons[b].cell => return None,
                            None if self.in_the_way(h, s, c) => return None,
                            _ => {}
                        }
                        // Squared distance from `eye` to the face's centre, and the cosine
                        // between the face's normal and the way back to the eye.
                        let seen = |eye: [f64; 3], &(fc, d): &Face| {
                            let p = face_centre((fc, d));
                            let v: [f64; 3] = std::array::from_fn(|i| eye[i] - p[i]);
                            let d2: f64 = v.iter().map(|x| x * x).sum();
                            let n = d.opposite().vector();
                            let cos = (0..3).map(|i| v[i] * n[i] as f64).sum::<f64>() / d2.sqrt();
                            (d2, cos)
                        };
                        let score = |eye: [f64; 3], f: &Face| {
                            let (d2, cos) = seen(eye, f);
                            d2 / cos.max(0.05).powi(2)
                        };
                        // A deployed angled panel sticks out: aim at its middle.
                        let panel = |f: &Face| {
                            self.deployed_panel(*f, dev)
                                .map(|(hinge, angle)| portals::panel_aim(*f, hinge, angle))
                        };
                        let clear = |eye: [f64; 3], f: &Face| match panel(f) {
                            Some((cell, point, facing)) => portals::point_shot(
                                level, c, eye, cell, point, facing, rules.sight,
                            )
                            .is_some_and(|m| m & on & model.fizzlers == 0),
                            None => portals::shots_from(level, c, eye, *f, rules.sight)
                                .any(|m| m & on & model.fizzlers == 0),
                        };
                        let (eye, face) = portals::eye_points(level, c)
                            .into_iter()
                            .flat_map(|eye| faces.iter().map(move |f| (eye, f)))
                            .filter(|&(eye, f)| {
                                Some(*f) != other && (panel(f).is_some() || seen(eye, f).1 >= min_cos)
                            })
                            .filter(|&(eye, f)| clear(eye, f))
                            .min_by(|a, b| score(a.0, a.1).total_cmp(&score(b.0, b.1)))?;
                        let mut after = h.clone();
                        after.portals[k] = Some(*face);
                        let aim = panel(face).map(|p| p.1);
                        self.fits(after, next).map(|after| {
                            (
                                Act::Fire {
                                    color,
                                    face: *face,
                                    cell: c,
                                    eye,
                                    aim,
                                },
                                after,
                            )
                        })
                    })
                };
                at_angle(MIN_SHOT_COS).or_else(|| at_angle(-1.0))
            }

            Action::Enter { color } => {
                let k = color as usize;
                let (face, out) = (h.portals[k]?, h.portals[1 - k]?);
                let (face_live, out_live) = (self.live_face(face, dev), self.live_face(out, dev));
                let (landing, hz, _) = self.fall_at(out_live.0, dev)?;
                self.find(h, s, |c| {
                    if c != face_live.0 {
                        return None;
                    }
                    let mut after = h.clone();
                    after.player = landing;
                    if fizzles(hz, on) {
                        Self::fizzle(&mut after);
                    }
                    self.fits(after, next).map(|after| {
                        (
                            Act::Enter {
                                color,
                                face: face_live,
                                out: out_live,
                            },
                            after,
                        )
                    })
                })
            }

            Action::Dive { color, height, to } => {
                let k = color as usize;
                let (face, out) = (h.portals[k]?, h.portals[1 - k]?);
                // The way out at the speed of the fall, as the solver's nominal trial has it.
                let (exit, tilt) = match self.deployed_panel(out, dev) {
                    Some((t, 90)) => (self.live_face(out, dev), Some((t, 0))),
                    Some(t) => (out, Some(t)),
                    None => (out, None),
                };
                let e = flings::exit_vector(exit.1, tilt.filter(|t| t.1 != 0));
                let speed = flings::dive_speed(height);
                let v: [f64; 3] = std::array::from_fn(|a| e[a] * speed);
                let (landing, hz) = flings::fly(level, flings::exit_point(exit.0, exit.1), v)?;
                if self.region(landing) != Some(to) {
                    return None;
                }
                self.find(h, s, |c| {
                    Dir::HORIZONTAL.into_iter().find_map(|d| {
                        let mut cur = c;
                        for out_cells in 1..=2 {
                            let n = step(cur, d);
                            if level.solid(n)
                                || level.standable(n)
                                || level.face(cur, d).barrier.is_some()
                            {
                                return None;
                            }
                            cur = n;
                            let Some((f, _)) = level.fall_path(n) else { continue };
                            if (f, Dir::NegZ) != face || c[2] - f[2] != height {
                                continue;
                            }
                            let mut after = h.clone();
                            after.player = landing;
                            if fizzles(hz, on) {
                                Self::fizzle(&mut after);
                            }
                            let act = Act::Dive {
                                edge: c,
                                dir: d,
                                out: out_cells,
                                face,
                                exit,
                                landing,
                            };
                            return self.fits(after, next).map(|after| (act, after));
                        }
                        None
                    })
                })
            }

            Action::Throw { color, .. } => {
                let k = color as usize;
                let i = Self::held(h)?;
                let (face, out) = (h.portals[k]?, h.portals[1 - k]?);
                let (face_live, out_live) = (self.live_face(face, dev), self.live_face(out, dev));
                let (fall, hz, _) = self.fall_at(out_live.0, dev)?;
                let fizzled = fizzles(hz, on);
                // Into a wall portal from in front of it; over a floor portal from beside it.
                let stand = if face_live.1 == Dir::NegZ {
                    self.beside(face_live.0)
                } else {
                    vec![face_live.0]
                };
                self.find(h, s, |c| {
                    if !stand.contains(&c) {
                        return None;
                    }
                    let mut after = h.clone();
                    let q = &mut after.cubes[i];
                    q.spot = None;
                    if fizzled && q.loose {
                        q.gone = true;
                    } else {
                        q.at = Some(if fizzled { q.home } else { fall });
                    }
                    let landing = q.at.unwrap_or(fall);
                    self.fits(after, next).map(|after| {
                        (
                            Act::Throw {
                                color,
                                face: face_live,
                                out: out_live,
                                landing,
                            },
                            after,
                        )
                    })
                })
            }

            Action::PickUp { cube, from } => {
                let locs = self.cubes(h);
                let mut found = None;
                // From beside the cube if possible, else from its own cell.
                for own_cell in [false, true] {
                    found = self.find(h, s, |c| {
                        (0..h.cubes.len()).find_map(|i| {
                            let q = h.cubes[i].at.filter(|_| !h.cubes[i].gone)?;
                            let near = if own_cell {
                                c == q
                            } else {
                                self.beside(q).contains(&c)
                            };
                            if h.cubes[i].kind != cube || locs[i].loc != from || !near {
                                return None;
                            }
                            let mut after = h.clone();
                            after.cubes[i].at = None;
                            after.cubes[i].spot = None;
                            self.fits(after, next)
                                .map(|after| (Act::PickUp { cube: q }, after))
                        })
                    });
                    if found.is_some() {
                        break;
                    }
                }
                found
            }

            Action::PutDown { at } => {
                let i = Self::held(h)?;
                let doors = [self.entry.0, self.exit_door.0];
                if let CubeLoc::Spot(k) = at {
                    // From behind the spot, facing its way: a held cube faces where the
                    // player does.
                    let (x, d) = (model.spot_cell(k), model.beams.spots[k].dir);
                    let behind = step(x, d.opposite());
                    return self
                        .find(h, s, |c| {
                            if c != behind {
                                return None;
                            }
                            let mut after = h.clone();
                            after.cubes[i].at = Some(x);
                            after.cubes[i].spot = Some(k);
                            let act = Act::PutDown {
                                at: x,
                                button: false,
                                facing: Some(d),
                            };
                            self.fits(after, next).map(|after| (act, after))
                        })
                        .map(|(walk, (act, mut after))| {
                            if let Some(&end) = walk.last() {
                                after.player = end;
                            }
                            Leg {
                                text: model.describe(a),
                                walk,
                                act,
                                after,
                            }
                        });
                }
                self.find(h, s, |c| {
                    if self.in_the_way(h, s, c) {
                        return None;
                    }
                    let spots: Vec<Cell> = match at {
                        CubeLoc::OnButton(b) => {
                            let t = model.buttons[b].cell;
                            if !self.beside(t).contains(&c) {
                                return None;
                            }
                            vec![t]
                        }
                        _ => self
                            .beside(c)
                            .into_iter()
                            .filter(|&t| !self.in_the_way(h, s, t) && !doors.contains(&t))
                            .collect(),
                    };
                    spots.into_iter().find_map(|t| {
                        let mut after = h.clone();
                        after.cubes[i].at = Some(t);
                        let button = matches!(at, CubeLoc::OnButton(_));
                        let act = Act::PutDown {
                            at: t,
                            button,
                            facing: None,
                        };
                        self.fits(after, next).map(|after| (act, after))
                    })
                })
            }

            Action::Toss { .. } => {
                let i = Self::held(h)?;
                self.find(h, s, |c| {
                    if self.in_the_way(h, s, c) {
                        return None;
                    }
                    Dir::HORIZONTAL.into_iter().find_map(|d| {
                        let n = step(c, d);
                        let face = level.face(c, d);
                        if face.barrier.is_some() || level.solid(n) || self.stands(n, dev) {
                            return None;
                        }
                        let (fall, fallen, _) = self.fall_at(n, dev)?;
                        let hz = face.hazards | fallen;
                        let mut after = h.clone();
                        let q = &mut after.cubes[i];
                        let fizzled = fizzles(hz, on);
                        if fizzled && q.loose {
                            q.gone = true;
                        } else {
                            q.at = Some(if fizzled { q.home } else { fall });
                        }
                        let landing = q.at.unwrap_or(fall);
                        self.fits(after, next).map(|after| {
                            (
                                Act::Toss {
                                    edge: c,
                                    dir: d,
                                    landing,
                                },
                                after,
                            )
                        })
                    })
                })
            }

            Action::Press { pedestal } => {
                let p = &model.pedestals[pedestal];
                let after = self.fits(h.clone(), next)?;
                let near = self.beside(p.cell);
                self.find(h, s, |c| {
                    (near.contains(&c) && !self.in_the_way(h, s, c)).then_some(())
                })
                .or_else(|| self.find(h, s, |c| (c == p.cell).then_some(())))
                .map(|(walk, ())| {
                    let act = Act::Press {
                        pedestal: p.index,
                        cell: p.cell,
                    };
                    (walk, (act, after.clone()))
                })
            }

            Action::Dash {
                pedestal,
                to,
                escaped,
            } => {
                let p = &model.pedestals[pedestal];
                let seconds = p.timer?;
                let on = model.hazards_on_in(s, Some(Signal::Pedestal(pedestal)));
                let alive = model
                    .turret_enabled
                    .iter()
                    .enumerate()
                    .filter(|&(t, &e)| e && s.down & (1 << t) == 0)
                    .fold(0u32, |m, (t, _)| m | 1 << t);
                // The fastest dash from each stand cell beside the pedestal (or, failing that,
                // from the pedestal's own cell, as the solver times it); the walk goes to the
                // stand cell with the fastest one.
                let near: Vec<Cell> = self
                    .beside(p.cell)
                    .into_iter()
                    .filter(|&c| !self.in_the_way(h, s, c))
                    .chain([p.cell])
                    .collect();
                let mut best: Option<(f64, Cell, Vec<Cell>, Here)> = None;
                for &c in &near {
                    if c == p.cell && best.is_some() {
                        break;
                    }
                    let Some((t, path, after)) = self.dash(c, on, alive, |cell, fizzled| {
                        let mut after = h.clone();
                        after.player = if escaped { self.exit_door.0 } else { cell };
                        // Out of the exit, the level is over: the solver applies no
                        // fizzler crossed on the way.
                        if fizzled && !escaped {
                            Self::fizzle(&mut after);
                        }
                        let arrived = if escaped {
                            cell == self.exit_door.0
                        } else {
                            self.region(cell) == Some(to)
                        };
                        if !arrived {
                            return None;
                        }
                        self.fits(after, next)
                    }) else {
                        continue;
                    };
                    if t <= seconds && best.as_ref().is_none_or(|b| t < b.0) {
                        best = Some((t, c, path, after));
                    }
                }
                let (_, stand, path, after) = best?;
                let act = Act::Dash {
                    pedestal: p.index,
                    cell: p.cell,
                    path,
                    seconds,
                    escape: escaped,
                };
                self.find(h, s, |c| (c == stand).then(|| (act.clone(), after.clone())))
            }

            Action::Knock { turret } => {
                let t = &model.turrets[turret];
                let after = self.fits(h.clone(), next)?;
                let beside: Vec<Cell> = Dir::HORIZONTAL
                    .into_iter()
                    .filter(|&d| level.face(t.cell, d).barrier.is_none())
                    .map(|d| step(t.cell, d))
                    .collect();
                self.find(h, s, |c| beside.contains(&c).then_some(()))
                    .or_else(|| self.find(h, s, |c| (c == t.cell).then_some(())))
                    .map(|(walk, ())| {
                        let act = Act::Knock {
                            turret: t.index,
                            cell: t.cell,
                        };
                        (walk, (act, after.clone()))
                    })
            }
        };
        let (walk, (act, mut after)) = found?;
        // Acting in place leaves the player where the walk ends (unless a device moved them).
        let stays = matches!(
            act,
            Act::Fire { .. }
                | Act::PickUp { .. }
                | Act::PutDown { .. }
                | Act::Toss { .. }
                | Act::Throw { .. }
                | Act::Press { .. }
                | Act::Knock { .. }
        );
        if let (true, Some(&end)) = (stays, walk.last()) {
            if self.region(after.player) == self.region(end) {
                after.player = end;
            }
        }
        Some(Leg {
            text: model.describe(a),
            walk,
            act,
            after,
        })
    }

    /// The quickest walk from `from` (Dijkstra on the solver's dash timing) to a cell where
    /// `arrive(cell, fizzled)` gives the state afterwards: (seconds, the cells after `from`,
    /// that state). Laser fields that are on (`on`) and cells live turrets (`alive`) can
    /// shoot stop it.
    fn dash(
        &self,
        from: Cell,
        on: u64,
        alive: u32,
        arrive: impl Fn(Cell, bool) -> Option<Here>,
    ) -> Option<(f64, Vec<Cell>, Here)> {
        let (level, model) = (self.level, self.model);
        let step_time = VOXEL / model.rules.walk_speed;
        let fall_time = |h: i32| (2.0 * h as f64 * VOXEL / GRAVITY).sqrt();
        // A node is (cell, fizzled on the way). Heap of (time in µs, node); best time and
        // parent per node.
        type Node = (Cell, bool);
        let mut best: HashMap<Node, (u64, Option<Node>)> =
            HashMap::from([((from, false), (0, None))]);
        let mut heap = BinaryHeap::from([(std::cmp::Reverse(0u64), from, false)]);
        while let Some((std::cmp::Reverse(t), c, fizzled)) = heap.pop() {
            if best[&(c, fizzled)].0 < t {
                continue;
            }
            if let Some(after) = arrive(c, fizzled) {
                let mut path = Vec::new();
                let mut x = (c, fizzled);
                while x != (from, false) {
                    path.push(x.0);
                    x = best[&x].1.unwrap();
                }
                path.reverse();
                return Some((t as f64 / 1e6, path, after));
            }
            for d in Dir::HORIZONTAL {
                let face = level.face(c, d);
                let n = step(c, d);
                if face.barrier.is_some() || level.solid(n) {
                    continue;
                }
                let (f, fallen) = if level.standable(n) {
                    (n, 0)
                } else {
                    match level.fall_path(n) {
                        Some(x) => x,
                        None => continue,
                    }
                };
                let hazards = face.hazards | fallen;
                let Some(r) = self.region(f) else { continue };
                if hazards & on & model.lasers != 0
                    || (!model.rules.run_past_turrets && model.graph.regions[r].danger & alive != 0)
                {
                    continue;
                }
                let nf = fizzled || hazards & on & model.fizzlers != 0;
                let nt = t + ((step_time + fall_time(n[2] - f[2])) * 1e6) as u64;
                if best.get(&(f, nf)).is_none_or(|b| nt < b.0) {
                    best.insert((f, nf), (nt, Some((c, fizzled))));
                    heap.push((std::cmp::Reverse(nt), f, nf));
                }
            }
        }
        None
    }

    /// The last leg: walk to the exit door's cell and out through it.
    fn exit(&self, h: &Here, s: &State) -> Option<Leg> {
        let (cell, normal) = self.exit_door;
        let (walk, ()) = self.find(h, s, |c| (c == cell).then_some(()))?;
        let mut after = h.clone();
        after.player = cell;
        Some(Leg {
            text: "walk out of the exit".into(),
            walk,
            act: Act::Exit { cell, normal },
            after,
        })
    }
}

/// The centre of a face (open cell, direction to the solid side), in voxels.
pub fn face_centre((c, d): Face) -> [f64; 3] {
    let n = d.vector();
    std::array::from_fn(|a| c[a] as f64 + 0.5 + 0.5 * n[a] as f64)
}

// --- JSON ---------------------------------------------------------------------------------

fn dir_name(d: Dir) -> &'static str {
    match d {
        Dir::NegX => "-x",
        Dir::PosX => "+x",
        Dir::NegY => "-y",
        Dir::PosY => "+y",
        Dir::NegZ => "-z",
        Dir::PosZ => "+z",
    }
}

fn color_name(c: Color) -> &'static str {
    match c {
        Color::Blue => "blue",
        Color::Orange => "orange",
    }
}

fn cube_name(k: CubeKind) -> &'static str {
    match k {
        CubeKind::Standard => "standard",
        CubeKind::Companion => "companion",
        CubeKind::Reflection => "reflection",
        CubeKind::Sphere => "sphere",
        CubeKind::Franken => "franken",
    }
}

fn cell_json(c: Cell) -> String {
    format!("[{},{},{}]", c[0], c[1], c[2])
}

fn point_json(p: [f64; 3]) -> String {
    let n = |v: f64| {
        let s = format!("{v:.3}");
        let s = s.trim_end_matches('0').trim_end_matches('.');
        if s == "-0" {
            "0".to_string()
        } else {
            s.to_string()
        }
    };
    format!("[{},{},{}]", n(p[0]), n(p[1]), n(p[2]))
}

/// `c` with `dz` added to its floor centre.
fn above(c: Cell, dz: f64) -> [f64; 3] {
    [c[0] as f64 + 0.5, c[1] as f64 + 0.5, c[2] as f64 + dz]
}

fn face_json(f: Face) -> String {
    format!(
        "{{\"cell\":{},\"dir\":\"{}\",\"centre\":{}}}",
        cell_json(f.0),
        dir_name(f.1),
        point_json(face_centre(f))
    )
}

fn cells_json(cells: &[Cell]) -> String {
    let items: Vec<String> = cells.iter().map(|&c| cell_json(c)).collect();
    format!("[{}]", items.join(","))
}

fn here_json(h: &Here) -> String {
    let portals: Vec<String> = h
        .portals
        .iter()
        .map(|p| p.map_or("null".into(), face_json))
        .collect();
    let cubes: Vec<String> = h
        .cubes
        .iter()
        .map(|c| {
            if c.gone {
                "\"gone\"".into()
            } else {
                c.at.map_or("null".into(), cell_json)
            }
        })
        .collect();
    format!(
        "{{\"player\":{},\"portals\":[{}],\"cubes\":[{}]}}",
        cell_json(h.player),
        portals.join(","),
        cubes.join(",")
    )
}

/// Heights above the floor, in voxels, to look at: a cube's centre, a pedestal's button, a
/// turret's middle.
const CUBE_CENTRE: f64 = 0.25;
const PEDESTAL_TOP: f64 = 0.35;
const TURRET_MIDDLE: f64 = 0.3;

fn act_json(act: &Act) -> String {
    match act {
        Act::Walk => "\"action\":\"walk\"".into(),
        Act::Fire {
            color,
            face,
            cell,
            eye,
            aim,
        } => format!(
            "\"action\":\"fire\",\"color\":\"{}\",\"face\":{},\"cell\":{},\"eye\":{}{}",
            color_name(*color),
            face_json(*face),
            cell_json(*cell),
            point_json(*eye),
            aim.map_or(String::new(), |p| format!(",\"aim\":{}", point_json(p)))
        ),
        Act::Enter { color, face, out } => format!(
            "\"action\":\"enter\",\"color\":\"{}\",\"face\":{},\"out\":{}",
            color_name(*color),
            face_json(*face),
            face_json(*out)
        ),
        Act::PickUp { cube } => format!(
            "\"action\":\"pickup\",\"cube\":{},\"look\":{}",
            cell_json(*cube),
            point_json(above(*cube, CUBE_CENTRE))
        ),
        Act::PutDown { at, button, facing } => format!(
            "\"action\":\"putdown\",\"at\":{},\"button\":{button},\"look\":{}{}",
            cell_json(*at),
            point_json(above(*at, 0.0)),
            facing.map_or(String::new(), |d| format!(",\"facing\":\"{}\"", dir_name(d)))
        ),
        Act::Toss { edge, dir, landing } => {
            let over = step(*edge, *dir);
            format!(
                "\"action\":\"toss\",\"edge\":{},\"dir\":\"{}\",\"over\":{},\"landing\":{},\"look\":{}",
                cell_json(*edge),
                dir_name(*dir),
                cell_json(over),
                cell_json(*landing),
                point_json(above(over, 0.0))
            )
        }
        Act::Throw {
            color,
            face,
            out,
            landing,
        } => format!(
            "\"action\":\"throw\",\"color\":\"{}\",\"face\":{},\"out\":{},\"landing\":{}",
            color_name(*color),
            face_json(*face),
            face_json(*out),
            cell_json(*landing)
        ),
        Act::Press { pedestal, cell } => format!(
            "\"action\":\"press\",\"item\":{pedestal},\"cell\":{},\"look\":{}",
            cell_json(*cell),
            point_json(above(*cell, PEDESTAL_TOP))
        ),
        Act::Dash {
            pedestal,
            cell,
            path,
            seconds,
            escape,
        } => format!(
            "\"action\":\"dash\",\"item\":{pedestal},\"cell\":{},\"look\":{},\"path\":{},\"seconds\":{seconds},\"escape\":{escape}",
            cell_json(*cell),
            point_json(above(*cell, PEDESTAL_TOP)),
            cells_json(path)
        ),
        Act::Knock { turret, cell } => format!(
            "\"action\":\"knock\",\"item\":{turret},\"cell\":{},\"look\":{}",
            cell_json(*cell),
            point_json(above(*cell, TURRET_MIDDLE))
        ),
        Act::Bounce { gel, to } => format!(
            "\"action\":\"bounce\",\"gel\":{},\"to\":{}",
            cell_json(*gel),
            cell_json(*to)
        ),
        Act::Leap { run, dir, to } => format!(
            "\"action\":\"leap\",\"run\":{},\"dir\":\"{}\",\"to\":{}",
            cells_json(run),
            dir_name(*dir),
            cell_json(*to)
        ),
        Act::Plate {
            from,
            dir,
            plate,
            to,
        } => format!(
            "\"action\":\"plate\",\"from\":{},\"dir\":\"{}\",\"plate\":{},\"to\":{}",
            cell_json(*from),
            dir_name(*dir),
            cell_json(*plate),
            cell_json(*to)
        ),
        Act::Climb { path } => format!("\"action\":\"climb\",\"path\":{}", cells_json(path)),
        Act::Funnel { enter, via, out, to } => format!(
            "\"action\":\"funnel\",\"enter\":{},\"via\":{},\"out\":{},\"to\":{}",
            cell_json(*enter),
            cells_json(via),
            cell_json(*out),
            cell_json(*to)
        ),
        Act::Ride { from, to } => format!(
            "\"action\":\"ride\",\"from\":{},\"to\":{}",
            cell_json(*from),
            cell_json(*to)
        ),
        Act::Dive {
            edge,
            dir,
            out,
            face,
            exit,
            landing,
        } => format!(
            "\"action\":\"dive\",\"edge\":{},\"dir\":\"{}\",\"out\":{out},\"face\":{},\"exit\":{},\"landing\":{}",
            cell_json(*edge),
            dir_name(*dir),
            face_json(*face),
            face_json(*exit),
            cell_json(*landing)
        ),
        Act::Exit { cell, normal } => format!(
            "\"action\":\"exit\",\"cell\":{},\"normal\":\"{}\"",
            cell_json(*cell),
            dir_name(*normal)
        ),
    }
}

impl Route {
    /// JSON lines: a header (title, size, doors, cubes, start), then one line per leg.
    pub fn json_lines(&self, title: &str, size: [i32; 3]) -> Vec<String> {
        let cubes: Vec<String> = self
            .start
            .cubes
            .iter()
            .map(|c| {
                format!(
                    "{{\"kind\":\"{}\",\"home\":{}}}",
                    cube_name(c.kind),
                    cell_json(c.home)
                )
            })
            .collect();
        let door = |(c, d): (Cell, Dir)| {
            format!(
                "{{\"cell\":{},\"normal\":\"{}\"}}",
                cell_json(c),
                dir_name(d)
            )
        };
        let mut out = vec![format!(
            "{{\"title\":{},\"size\":{},\"entry\":{},\"exit\":{},\"cubes\":[{}],\"start\":{},\"legs\":{}}}",
            json_str(title),
            cell_json(size),
            door(self.entry),
            door(self.exit),
            cubes.join(","),
            here_json(&self.start),
            self.legs.len()
        )];
        for (i, leg) in self.legs.iter().enumerate() {
            out.push(format!(
                "{{\"leg\":{},\"text\":{},\"walk\":{},{},\"after\":{}}}",
                i + 1,
                json_str(&leg.text),
                cells_json(&leg.walk),
                act_json(&leg.act),
                here_json(&leg.after)
            ));
        }
        out
    }
}
