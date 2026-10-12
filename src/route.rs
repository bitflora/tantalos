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

use std::collections::{BinaryHeap, HashMap, VecDeque};

use crate::dataset::json_str;
use crate::gel;
use crate::geom::{step, Cell, Dir};
use crate::level::{CubeKind, FaceInfo, ItemKind, Level, PaintKind};
use crate::metrics::replay;
use crate::portals::{self, Face};
use crate::regions::Move;
use crate::solver::{Action, Color, Cube, CubeLoc, Mode, Model, Signal, State, Verdict};

const VOXEL: f64 = 128.0;
const GRAVITY: f64 = 600.0;

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
    /// `None` while held.
    pub at: Option<Cell>,
}

/// What a leg does once the player has walked to where it starts.
#[derive(Clone, Debug, PartialEq)]
pub enum Act {
    /// Nothing more: the walk is the step (a walk across a field or into new sight lines, or
    /// a drop off a ledge at its end).
    Walk,
    /// Fire from `eye` (the walk's end) at the centre of `face`.
    Fire {
        color: Color,
        face: Face,
        eye: Cell,
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
    /// Put the held cube down in `at`, next to the walk's end.
    PutDown {
        at: Cell,
        button: bool,
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
    let start = g.start()?;
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
                Some(Placed {
                    kind: c.cube_kind(),
                    home,
                    at: Some(home),
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
                home: self.loc_at(c.home, c.kind, None),
                loc: match c.at {
                    None => CubeLoc::Held,
                    Some(cell) => self.loc_at(cell, c.kind, Some(&mut claimed)),
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

    /// A fizzler: portals gone, a held cube back home.
    fn fizzle(h: &mut Here) {
        h.portals = [None, None];
        for c in &mut h.cubes {
            if c.at.is_none() {
                c.at = Some(c.home);
            }
        }
    }

    fn held(h: &Here) -> Option<usize> {
        h.cubes.iter().position(|c| c.at.is_none())
    }

    /// Something stands in `c` that the player shouldn't walk into, stand on to act, or put a
    /// cube on: a cube, a floor button, a pedestal or a turret, gel on the floor, or an open
    /// portal in the floor (walking over it, the player falls in).
    fn in_the_way(&self, h: &Here, s: &State, c: Cell) -> bool {
        let open = h.portals.iter().all(Option::is_some);
        (open && h.portals.contains(&Some((c, Dir::NegZ))))
            || h.cubes.iter().any(|q| q.at == Some(c))
            || self.model.buttons.iter().any(|b| b.cell == c)
            || self.model.pedestals.iter().any(|p| p.cell == c)
            || self.model.turrets.iter().any(|t| t.cell == c)
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
            (self.level.face(c, d) == FaceInfo::default() && r.is_some() && self.region(n) == r)
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

    fn leg(&self, h: &Here, s: &State, a: &Action, next: &State) -> Option<Leg> {
        let (level, model) = (self.level, self.model);
        let on = model.hazards_on_in(s, None);
        let fizzles = |hazards: u64, on: u64| hazards & on & model.fizzlers != 0;
        let rules = model.rules;
        let found: Option<(Vec<Cell>, (Act, Here))> = match *a {
            Action::Move { to, kind, hazards } => match kind {
                Move::Walk | Move::Drop { .. } => self
                    .find(h, s, |c| {
                        Dir::HORIZONTAL.into_iter().find_map(|d| {
                            let n = step(c, d);
                            let face = level.face(c, d);
                            if face.barrier.is_some() || level.solid(n) {
                                return None;
                            }
                            let (f, hz) = match kind {
                                Move::Walk if level.standable(n) => (n, face.hazards),
                                Move::Drop { height } if !level.standable(n) => {
                                    let (f, fallen) = level.fall_path(n)?;
                                    if n[2] - f[2] != height {
                                        return None;
                                    }
                                    (f, face.hazards | fallen)
                                }
                                _ => return None,
                            };
                            if hz != hazards || self.region(f) != Some(to) {
                                return None;
                            }
                            let mut after = h.clone();
                            after.player = f;
                            if fizzles(hz, on) {
                                Self::fizzle(&mut after);
                            }
                            self.matches(&after, next).then_some((Act::Walk, after))
                        })
                    })
                    .map(|(mut walk, (act, after))| {
                        walk.push(after.player);
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
                            self.matches(&after, next)
                                .then_some((Act::Bounce { gel: c, to: t }, after))
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
                        self.matches(&after, next)
                            .then_some((Act::Leap { run, dir: d, to: t }, after))
                    })
                }),
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
                self.find(h, s, |c| {
                    match stand_on {
                        Some(b) if c != model.buttons[b].cell => return None,
                        None if self.in_the_way(h, s, c) => return None,
                        _ => {}
                    }
                    let eye = [c[0] as f64 + 0.5, c[1] as f64 + 0.5, c[2] as f64 + 0.5];
                    let dist = |&(fc, d): &Face| {
                        let p = face_centre((fc, d));
                        (0..3).map(|i| (p[i] - eye[i]).powi(2)).sum::<f64>()
                    };
                    let face = faces
                        .iter()
                        .filter(|&&f| Some(f) != other)
                        .filter(|&&f| {
                            portals::shots(level, c, f, rules.sight)
                                .any(|m| m & on & model.fizzlers == 0)
                        })
                        .min_by(|a, b| dist(a).total_cmp(&dist(b)))?;
                    let mut after = h.clone();
                    after.portals[k] = Some(*face);
                    self.matches(&after, next).then_some((
                        Act::Fire {
                            color,
                            face: *face,
                            eye: c,
                        },
                        after,
                    ))
                })
            }

            Action::Enter { color } => {
                let k = color as usize;
                let (face, out) = (h.portals[k]?, h.portals[1 - k]?);
                let (landing, hz) = level.fall_path(out.0)?;
                self.find(h, s, |c| {
                    if c != face.0 {
                        return None;
                    }
                    let mut after = h.clone();
                    after.player = landing;
                    if fizzles(hz, on) {
                        Self::fizzle(&mut after);
                    }
                    self.matches(&after, next)
                        .then_some((Act::Enter { color, face, out }, after))
                })
            }

            Action::Throw { color, .. } => {
                let k = color as usize;
                let i = Self::held(h)?;
                let (face, out) = (h.portals[k]?, h.portals[1 - k]?);
                let (fall, hz) = level.fall_path(out.0)?;
                let landing = if fizzles(hz, on) {
                    h.cubes[i].home
                } else {
                    fall
                };
                // Into a wall portal from in front of it; over a floor portal from beside it.
                let stand = if face.1 == Dir::NegZ {
                    self.beside(face.0)
                } else {
                    vec![face.0]
                };
                self.find(h, s, |c| {
                    if !stand.contains(&c) {
                        return None;
                    }
                    let mut after = h.clone();
                    after.cubes[i].at = Some(landing);
                    self.matches(&after, next).then_some((
                        Act::Throw {
                            color,
                            face,
                            out,
                            landing,
                        },
                        after,
                    ))
                })
            }

            Action::PickUp { cube, from } => {
                let locs = self.cubes(h);
                let mut found = None;
                // From beside the cube if possible, else from its own cell.
                for own_cell in [false, true] {
                    found = self.find(h, s, |c| {
                        (0..h.cubes.len()).find_map(|i| {
                            let q = h.cubes[i].at?;
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
                            self.matches(&after, next)
                                .then_some((Act::PickUp { cube: q }, after))
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
                        self.matches(&after, next)
                            .then_some((Act::PutDown { at: t, button }, after))
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
                        if face.barrier.is_some() || level.solid(n) || level.standable(n) {
                            return None;
                        }
                        let (fall, fallen) = level.fall_path(n)?;
                        let hz = face.hazards | fallen;
                        let landing = if fizzles(hz, on) {
                            h.cubes[i].home
                        } else {
                            fall
                        };
                        let mut after = h.clone();
                        after.cubes[i].at = Some(landing);
                        self.matches(&after, next).then_some((
                            Act::Toss {
                                edge: c,
                                dir: d,
                                landing,
                            },
                            after,
                        ))
                    })
                })
            }

            Action::Press { pedestal } => {
                let p = &model.pedestals[pedestal];
                self.matches(h, next).then_some(())?;
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
                    (walk, (act, h.clone()))
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
                        (arrived && self.matches(&after, next)).then_some(after)
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
                self.matches(h, next).then_some(())?;
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
                        (walk, (act, h.clone()))
                    })
            }
        };
        let (walk, (act, mut after)) = found?;
        // Acting in place leaves the player where the walk ends.
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
            after.player = end;
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
        .map(|c| c.at.map_or("null".into(), cell_json))
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
        Act::Fire { color, face, eye } => format!(
            "\"action\":\"fire\",\"color\":\"{}\",\"face\":{},\"eye\":{}",
            color_name(*color),
            face_json(*face),
            point_json(above(*eye, 0.5))
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
        Act::PutDown { at, button } => format!(
            "\"action\":\"putdown\",\"at\":{},\"button\":{button},\"look\":{}",
            cell_json(*at),
            point_json(above(*at, 0.0))
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
