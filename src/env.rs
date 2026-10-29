//! The legal next tokens of a plan, for writing plans that `plan::check` accepts step by step
//! (Goblin's constrained decoding, docs/goblin-plan-2026-10-27.md "Next" 3, E2).
//!
//! An `Env` holds one level's model (and its angled-spot model, where the checker would try
//! one too). Given a plan's tokens so far, `next` says which tokens may come next: the words
//! of steps that fit a move from some state the steps so far lead to, as `plan::fits` judges
//! them. A level whose goal is reached allows only `exit`, and then the end. A plan written
//! only from allowed tokens fits every step in at least one of the models, and solves the level
//! whenever it ends with `exit`; it can still run out of tokens (unfinished) or reach a dead
//! end, where no move fits (`next` then allows only the end). `check` reports a plan that
//! doesn't solve by the plain model, so an unfinished plan that only the angled model follows
//! shows as illegal there (2026-10-29: 11 of 1,048 random allowed plans, all of them that).
//!
//! Each step's pattern is exact: every cell, face or direction it allows fits the move, so
//! `next` neither forbids a step `check` would take nor allows one it wouldn't. Two
//! exceptions, both stricter than `check`: `putdown` takes a direction only at a laser spot,
//! and the state sets follow at most `MAX_STATES` states (an ambiguous plan that needs more
//! may lose a way through).

use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;

use crate::geom::{Cell, Dir};
use crate::level::{CubeKind, Level};
use crate::plan::{decode, fits};
use crate::regions::Move;
use crate::solver::{Action, Color, CubeLoc, Mode, Model, Problem, State};
use crate::tokens::{dir_token, parse_dir, MAX_NUM};

/// Most states a node follows, per model (`check` follows up to 100,000; decoding asks a node
/// for every token, so it keeps fewer).
pub const MAX_STATES: usize = 2_000;
/// Most states a level's nodes keep (about 200 bytes each) before all but the first node are
/// dropped; `next` builds what it needs again from the start of the plan.
pub const MAX_CACHED: usize = 1_000_000;

const DIRS: [Dir; 6] = [Dir::PosX, Dir::NegX, Dir::PosY, Dir::NegY, Dir::PosZ, Dir::NegZ];

/// One part of a step's words.
#[derive(Clone, Debug)]
enum Slot {
    Word(&'static str),
    /// Three numbers: one of these cells.
    Cells(Rc<[Cell]>),
    /// Three numbers and a direction: one of these faces.
    Faces(Rc<[(Cell, Dir)]>),
    Dirs(Rc<[Dir]>),
    /// Three numbers, any of them (a cube fizzled on the way, a dash out of the exit).
    AnyCell,
}

impl Slot {
    fn width(&self) -> usize {
        match self {
            Slot::Word(_) | Slot::Dirs(_) => 1,
            Slot::Cells(_) | Slot::AnyCell => 3,
            Slot::Faces(_) => 4,
        }
    }

    /// Whether `words` (`width()` of them) are this slot.
    fn matches(&self, words: &[&str]) -> bool {
        match self {
            Slot::Word(w) => words[0] == *w,
            Slot::Cells(cells) => cell_of(words).is_some_and(|c| cells.contains(&c)),
            Slot::Faces(faces) => match (cell_of(words), parse_dir(words[3])) {
                (Some(c), Some(d)) => faces.contains(&(c, d)),
                _ => false,
            },
            Slot::Dirs(dirs) => parse_dir(words[0]).is_some_and(|d| dirs.contains(&d)),
            Slot::AnyCell => cell_of(words).is_some(),
        }
    }

    /// The tokens that may follow `words` (fewer than `width()`) inside this slot.
    fn extend(&self, words: &[&str], out: &mut BTreeSet<String>) {
        let nums: Option<Vec<i32>> = words.iter().take(3).map(|w| num(w)).collect();
        let Some(nums) = nums else { return };
        let k = nums.len();
        match self {
            Slot::Word(w) => {
                out.insert(w.to_string());
            }
            Slot::Dirs(dirs) => out.extend(dirs.iter().map(|d| dir_token(*d).to_string())),
            Slot::AnyCell => out.extend((0..=MAX_NUM).map(|n| n.to_string())),
            Slot::Cells(cells) => out.extend(
                cells
                    .iter()
                    .filter(|c| c[..k] == nums[..])
                    .map(|c| c[k].to_string()),
            ),
            Slot::Faces(faces) => {
                let fit = faces.iter().filter(|(c, _)| c[..k.min(3)] == nums[..k.min(3)]);
                if k < 3 {
                    out.extend(fit.map(|(c, _)| c[k].to_string()));
                } else {
                    out.extend(fit.map(|(_, d)| dir_token(*d).to_string()));
                }
            }
        }
    }
}

fn num(w: &str) -> Option<i32> {
    match w.parse::<i32>() {
        Ok(n) if (0..=MAX_NUM).contains(&n) && n.to_string() == w => Some(n),
        _ => None,
    }
}

fn cell_of(words: &[&str]) -> Option<Cell> {
    Some([num(words[0])?, num(words[1])?, num(words[2])?])
}

/// Whether `words` (the step so far) fit `pattern`: adds the tokens that may come next to
/// `out`, and says whether the words are the whole pattern.
fn extend(pattern: &[Slot], words: &[&str], out: &mut BTreeSet<String>) -> bool {
    let mut at = 0;
    for slot in pattern {
        let w = slot.width();
        if at + w > words.len() {
            slot.extend(&words[at..], out);
            return false;
        }
        if !slot.matches(&words[at..at + w]) {
            return false;
        }
        at += w;
    }
    at == words.len()
}

/// The states some steps lead to, in each model, and the patterns of the steps that can come
/// next.
struct Node {
    /// Per model: the states. (Their moves are worked out again for each child: keeping them
    /// took gigabytes on big levels.)
    states: Vec<Vec<State>>,
    patterns: Vec<Vec<Slot>>,
    goal: bool,
}

impl Node {
    fn dead(&self) -> bool {
        self.states.iter().all(|s| s.is_empty())
    }

    /// States kept.
    fn size(&self) -> usize {
        self.states.iter().map(Vec::len).sum()
    }
}

/// What may come after a plan's tokens so far.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Next {
    /// The tokens that may come next.
    pub allowed: BTreeSet<String>,
    /// The plan may end here: the goal is reached (or the plan said `exit`), or no step fits.
    pub end: bool,
    /// No step fits any more (or the tokens so far weren't allowed ones).
    pub dead: bool,
    /// The tokens so far are whole steps: a plan that stops here parses (it is unfinished
    /// unless `end`).
    pub whole: bool,
}

/// One level's models and what decoding has asked of them (see the module docs).
pub struct Env {
    models: Vec<Model>,
    /// Per model: the cells of each region.
    regions: Vec<Vec<Rc<[Cell]>>>,
    /// Per model: the faces of each class.
    classes: Vec<Vec<Rc<[(Cell, Dir)]>>>,
    /// Per model: a region's cells a cube of a kind can be put down in loose (no button there
    /// that takes it).
    loose: HashMap<(usize, usize, CubeKind), Rc<[Cell]>>,
    /// Nodes by the words of their steps.
    nodes: HashMap<String, Node>,
    /// The nodes' `size`s, summed.
    cached: usize,
}

impl Env {
    pub fn new(level: &Level, mode: Mode) -> Result<Env, Vec<Problem>> {
        let model = Model::new(level, mode)?;
        let mut models = vec![];
        if model.could_angle() {
            let angled = Model::new_angled(level, mode).ok();
            models.push(model);
            models.extend(angled);
        } else {
            models.push(model);
        }
        let regions = models
            .iter()
            .map(|m| {
                (0..m.graph.regions.len())
                    .map(|r| {
                        m.graph.regions[r]
                            .cells
                            .iter()
                            .filter(|c| m.graph.region_of(**c) == Some(r))
                            .copied()
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let classes = models
            .iter()
            .map(|m| m.portals.classes.iter().map(|c| c.faces.iter().copied().collect()).collect())
            .collect();
        let mut env = Env {
            models,
            regions,
            classes,
            loose: HashMap::new(),
            nodes: HashMap::new(),
            cached: 0,
        };
        let states = env.models.iter().map(|m| vec![m.initial()]).collect();
        let root = env.node(states);
        env.cached = root.size();
        env.nodes.insert(String::new(), root);
        Ok(env)
    }

    /// States the nodes keep (see `MAX_CACHED`).
    pub fn cached(&self) -> usize {
        self.cached
    }

    /// Drops every node but the first: they are built again when asked for.
    pub fn forget(&mut self) {
        self.nodes.retain(|k, _| k.is_empty());
        self.cached = self.nodes.values().map(Node::size).sum();
    }

    /// What may follow `plan` (plan tokens, space-separated).
    pub fn next(&mut self, plan: &str) -> Next {
        if self.cached > MAX_CACHED {
            self.forget();
        }
        let words: Vec<&str> = plan.split_whitespace().collect();
        let dead = Next {
            allowed: BTreeSet::new(),
            end: true,
            dead: true,
            whole: false,
        };
        let Some((ends, partial)) = split(&words) else {
            return dead;
        };
        if partial == usize::MAX {
            // Ended with `exit`.
            return Next {
                allowed: BTreeSet::new(),
                end: true,
                dead: false,
                whole: true,
            };
        }
        // The node after the whole steps.
        let mut key = String::new();
        let mut start = 0;
        for &end in &ends {
            let text = words[start..end].join(" ");
            let child = if key.is_empty() { text.clone() } else { format!("{key} {text}") };
            if !self.child(&key, &child, &text) {
                return dead;
            }
            key = child;
            start = end;
        }
        let rest = &words[partial..];
        let node = &self.nodes[&key];
        if node.dead() {
            return dead;
        }
        if node.goal {
            // Steps after the goal are ignored by `check`: only `exit`.
            return Next {
                allowed: if rest.is_empty() { BTreeSet::from(["exit".to_string()]) } else { BTreeSet::new() },
                end: rest.is_empty(),
                dead: !rest.is_empty(),
                whole: rest.is_empty(),
            };
        }
        let mut out = Next {
            whole: rest.is_empty(),
            ..Next::default()
        };
        let mut whole = false;
        for p in &node.patterns {
            whole |= extend(p, rest, &mut out.allowed);
        }
        // A step that may go on (a direction after `putdown`, a face in `loop`) or end here.
        if whole && !rest.is_empty() {
            let text = rest.join(" ");
            let child = if key.is_empty() { text.clone() } else { format!("{key} {text}") };
            if self.child(&key, &child, &text) {
                out.whole = true;
                let c = &self.nodes[&child];
                if c.goal {
                    out.allowed.insert("exit".into());
                    out.end = true;
                } else if !c.dead() {
                    out.allowed.extend(c.patterns.iter().filter_map(|p| match p.first() {
                        Some(Slot::Word(w)) => Some(w.to_string()),
                        _ => None,
                    }));
                }
            }
        }
        if out.allowed.is_empty() && !out.end {
            return dead;
        }
        out
    }

    /// Makes sure the node `child` (node `parent` and then step `text`) is there. False if the
    /// step isn't one (it doesn't parse).
    fn child(&mut self, parent: &str, child: &str, text: &str) -> bool {
        if self.nodes.contains_key(child) {
            return true;
        }
        let step = match decode(text) {
            Ok(s) if s.len() == 1 => s[0],
            _ => return false,
        };
        let p = &self.nodes[parent];
        let states: Vec<Vec<State>> = (0..self.models.len())
            .map(|m| {
                let model = &self.models[m];
                let mut next: Vec<State> = Vec::new();
                let mut seen = std::collections::HashSet::new();
                for s in &p.states[m] {
                    for (a, n) in model.successors(s) {
                        if next.len() >= MAX_STATES {
                            break;
                        }
                        if fits(model, s, &a, &n, &step) && seen.insert(n.clone()) {
                            next.push(n);
                        }
                    }
                }
                next
            })
            .collect();
        let node = self.node(states);
        self.cached += node.size();
        self.nodes.insert(child.to_string(), node);
        true
    }

    fn node(&mut self, states: Vec<Vec<State>>) -> Node {
        let goal = states
            .iter()
            .enumerate()
            .any(|(m, ss)| ss.iter().any(|s| self.models[m].is_goal(s)));
        let mut patterns = Vec::new();
        if !goal {
            for (m, ss) in states.iter().enumerate() {
                for s in ss {
                    for (a, n) in self.models[m].successors(s) {
                        self.patterns(m, s, &a, &n, &mut patterns);
                    }
                }
            }
        }
        Node { states, patterns, goal }
    }

    /// The cells `plan::loc_fits` takes for a cube at `loc` (`kind`: the cube put there, when
    /// it matters which buttons take it). None for a held cube.
    fn loc_cells(&mut self, m: usize, loc: CubeLoc, kind: Option<CubeKind>) -> Option<Slot> {
        let model = &self.models[m];
        Some(match loc {
            CubeLoc::OnButton(b) => Slot::Cells(Rc::from([model.buttons[b].cell])),
            CubeLoc::Spot(k) => Slot::Cells(Rc::from([model.spot_cell(k)])),
            CubeLoc::Loose(r) => match kind {
                None => Slot::Cells(self.regions[m][r].clone()),
                Some(k) => {
                    let cells = self.regions[m][r].clone();
                    let set = self.loose.entry((m, r, k)).or_insert_with(|| {
                        cells
                            .iter()
                            .filter(|c| !model.buttons.iter().any(|b| b.cell == **c && b.kind.accepts(k)))
                            .copied()
                            .collect()
                    });
                    Slot::Cells(set.clone())
                }
            },
            CubeLoc::Gone => Slot::AnyCell,
            CubeLoc::Held => return None,
        })
    }

    /// The patterns of the steps that fit action `a` from `s` to `n` (`plan::fits`).
    fn patterns(&mut self, m: usize, s: &State, a: &Action, n: &State, out: &mut Vec<Vec<Slot>>) {
        use Slot::Word;
        let held = s.cubes.iter().find(|c| c.loc == CubeLoc::Held).map(|c| c.kind);
        let lands = Slot::Cells(self.regions[m][n.region].clone());
        let color = |c: Color| Word(if c == Color::Blue { "blue" } else { "orange" });
        let cell = |c: Cell| Slot::Cells(Rc::from([c]));
        let p: Vec<Vec<Slot>> = match *a {
            Action::Move { kind, .. } => {
                let verb = match kind {
                    Move::Walk | Move::Drop { .. } | Move::Step | Move::Jump => "walk",
                    Move::Bounce => "bounce",
                    Move::Leap => "leap",
                    Move::Plate => "plate",
                    Move::Stairs => "climb",
                    Move::Funnel => "funnel",
                    Move::Ride => "ride",
                };
                vec![vec![Word(verb), lands]]
            }
            Action::Fire { color: c, class, .. } => {
                vec![vec![Word("fire"), color(c), Slot::Faces(self.classes[m][class].clone())]]
            }
            Action::Enter { color: c } => vec![vec![Word("enter"), color(c)]],
            Action::PickUp { from, .. } => match self.loc_cells(m, from, None) {
                Some(x) => vec![vec![Word("pickup"), x]],
                None => vec![],
            },
            Action::PutDown { at } => {
                let Some(x) = self.loc_cells(m, at, held) else { return };
                let mut v = vec![vec![Word("putdown"), x.clone()]];
                if let CubeLoc::Spot(k) = at {
                    let dirs: Rc<[Dir]> = if held.is_some_and(|k| k != CubeKind::Reflection) {
                        Rc::from(DIRS)
                    } else {
                        Rc::from([self.models[m].beams.spots[k].dir])
                    };
                    v.push(vec![Word("putdown"), x, Slot::Dirs(dirs)]);
                }
                v
            }
            Action::Toss { to } => match self.loc_cells(m, to, held) {
                Some(x) => vec![vec![Word("toss"), x]],
                None => vec![],
            },
            Action::Throw { color: c, to } => match self.loc_cells(m, to, held) {
                Some(x) => vec![vec![Word("throw"), color(c), x]],
                None => vec![],
            },
            Action::Press { pedestal } => vec![vec![Word("press"), cell(self.models[m].pedestals[pedestal].cell)]],
            Action::Dash { pedestal, escaped, .. } => {
                let end = if escaped { Slot::AnyCell } else { lands };
                vec![vec![Word("dash"), cell(self.models[m].pedestals[pedestal].cell), end]]
            }
            Action::DoubleDash { first, second, escaped, .. } => {
                let end = if escaped { Slot::AnyCell } else { lands };
                let (p, q) = (self.models[m].pedestals[first].cell, self.models[m].pedestals[second].cell);
                vec![vec![Word("double"), cell(p), cell(q), end]]
            }
            Action::Fetch { pedestal, from, .. } => {
                let p = self.models[m].pedestals[pedestal].cell;
                match self.loc_cells(m, from, None) {
                    Some(x) => vec![vec![Word("fetch"), cell(p), x, lands]],
                    None => vec![],
                }
            }
            Action::Reach { from, button, .. } => {
                let b = self.models[m].buttons[button].cell;
                match self.loc_cells(m, from, None) {
                    Some(x) => vec![vec![Word("reach"), cell(b), x]],
                    None => vec![],
                }
            }
            Action::Knock { turret } => vec![vec![Word("knock"), cell(self.models[m].turrets[turret].cell)]],
            Action::Dive { color: c, .. } => vec![vec![Word("dive"), color(c), lands]],
            Action::Loop { color: c, refire, .. } => match refire {
                None => vec![vec![Word("loop"), color(c), lands]],
                Some(class) => {
                    vec![vec![Word("loop"), color(c), Slot::Faces(self.classes[m][class].clone()), lands]]
                }
            },
            Action::PortalReach { color: c, from, .. } => match self.loc_cells(m, from, None) {
                Some(x) => vec![vec![Word("grab"), color(c), x]],
                None => vec![],
            },
            Action::Sweep { sweep } => {
                let w = &self.models[m].beams.sweeps[sweep];
                vec![vec![Word("sweep"), Slot::Faces(Rc::from([(w.from, w.dir)]))]]
            }
        };
        out.extend(p);
    }
}

/// A plan's words cut into whole steps: the end of each, and where the step under way starts
/// (`words.len()` if none is; usize::MAX if the plan ended with `exit`). A step that may go on
/// (`putdown X Y Z`, `loop C X Y Z`) is under way. None if the words aren't a plan's.
fn split(words: &[&str]) -> Option<(Vec<usize>, usize)> {
    let mut ends = Vec::new();
    let mut i = 0;
    let n = words.len();
    let is_dir = |j: usize| words.get(j).is_some_and(|w| parse_dir(w).is_some());
    while i < n {
        let len = match words[i] {
            "exit" if i + 1 == n => return Some((ends, usize::MAX)),
            "exit" => return None,
            "enter" => 2,
            "walk" | "pickup" | "toss" | "press" | "knock" | "bounce" | "leap" | "plate"
            | "climb" | "funnel" | "ride" => 4,
            "throw" | "dive" | "grab" | "sweep" => 5,
            "fire" => 6,
            "dash" | "reach" => 7,
            "fetch" | "double" => 10,
            "putdown" if i + 4 >= n => return Some((ends, i)),
            "putdown" if is_dir(i + 4) => 5,
            "putdown" => 4,
            "loop" if i + 5 >= n => return Some((ends, i)),
            "loop" if is_dir(i + 5) => 9,
            "loop" => 5,
            _ => return None,
        };
        if i + len > n {
            return Some((ends, i));
        }
        i += len;
        ends.push(i);
    }
    Some((ends, n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::check;
    use crate::rng::Rng;

    /// plan.rs's test level and its plan.
    const LEVEL: &str = "size 7 9 8 open 3 3 1 4 6 7 0 0 2 6 8 7 white +x 4 4 1 6 8 1 white -x 3 4 1         6 8 7 0 4 2 6 4 5 0 5 4 6 7 5 white +y 1 8 2 6 8 5 entry 0 7 2 +x exit 1 0 2 +y         laserfield 2 8 2 -y +z 1 on dropper 0 6 7 sphere dropper 1 7 7 standard button 1 2 2         sphere > exit button 5 0 2 weighted > exit pedestal 3 3 1 inf > exit turret 4 0 2 3         laser 6 1 2 -x -z on catcher 4 5 1 -x -z > stairs 0 funnel 5 0 3 +y on stairs 5 2 2         +x off";
    const PLAN: &str = "fire blue 2 8 3 +y fire orange 0 4 2 -x pickup 1 7 2 enter orange         knock 4 0 2 walk 5 0 2 putdown 5 0 2 walk 6 1 2 walk 4 3 1 fire blue 3 4 1 -x         press 3 3 1 enter blue pickup 0 6 2 putdown 1 2 2 exit";

    fn env() -> (Level, Env) {
        let level = Level::from_puzzle(&crate::tokens::decode(LEVEL).unwrap());
        let env = Env::new(&level, Mode::Conservative).unwrap();
        (level, env)
    }

    #[test]
    fn split_steps() {
        fn w(s: &str) -> Vec<&str> {
            s.split_whitespace().collect()
        }
        assert_eq!(split(&w("")), Some((vec![], 0)));
        assert_eq!(split(&w("walk 1 2 3 fire blue 1")), Some((vec![4], 4)));
        assert_eq!(split(&w("putdown 1 2 3")), Some((vec![], 0)));
        assert_eq!(split(&w("putdown 1 2 3 +x walk")), Some((vec![5], 5)));
        assert_eq!(split(&w("putdown 1 2 3 walk")), Some((vec![4], 4)));
        assert_eq!(split(&w("loop blue 1 2 3 +x 4 5 6")), Some((vec![9], 9)));
        assert_eq!(split(&w("loop blue 1 2 3 enter")), Some((vec![5], 5)));
        assert_eq!(split(&w("enter blue exit")), Some((vec![2], usize::MAX)));
        assert_eq!(split(&w("exit walk")), None);
        assert_eq!(split(&w("jump")), None);
    }

    /// Every token of a plan that solves is allowed where it comes, `exit` is all that's
    /// allowed once the goal is reached, and then only the end.
    #[test]
    fn a_solving_plan_is_allowed() {
        let (_, mut env) = env();
        let words: Vec<&str> = PLAN.split_whitespace().collect();
        for i in 0..words.len() {
            let next = env.next(&words[..i].join(" "));
            assert!(next.allowed.contains(words[i]), "token {i} ({}) not in {:?}", words[i], next.allowed);
            assert!(!next.dead);
        }
        let last = env.next(&words[..words.len() - 1].join(" "));
        assert_eq!(last.allowed, BTreeSet::from(["exit".to_string()]));
        assert!(last.end);
        let after = env.next(PLAN);
        assert!(after.end && after.allowed.is_empty() && !after.dead);
        // Only cells inside the level (7 wide) start a face.
        let first = env.next("fire blue");
        assert!(!first.allowed.is_empty() && !first.end);
        assert!(first.allowed.iter().all(|t| t.parse::<i32>().is_ok_and(|n| n < 7)), "{:?}", first.allowed);
    }

    /// Dropping the nodes changes no answer: they are built again.
    #[test]
    fn forget_keeps_answers() {
        let (_, mut env) = env();
        let words: Vec<&str> = PLAN.split_whitespace().collect();
        let before: Vec<Next> = (0..=words.len()).map(|i| env.next(&words[..i].join(" "))).collect();
        let full = env.cached();
        env.forget();
        assert!(env.cached() < full);
        let after: Vec<Next> = (0..=words.len()).map(|i| env.next(&words[..i].join(" "))).collect();
        assert_eq!(before, after);
        assert_eq!(env.cached(), full);
    }

    /// Random plans written only from allowed tokens: `check` finds no illegal step in any,
    /// and those that end at the goal are solved.
    #[test]
    fn allowed_plans_check() {
        let (level, mut env) = env();
        let model = Model::new(&level, Mode::Conservative).unwrap();
        let mut rng = Rng::new(7);
        let mut solved = 0;
        for _ in 0..60 {
            let mut plan: Vec<String> = Vec::new();
            let finished = loop {
                let next = env.next(&plan.join(" "));
                if next.allowed.is_empty() || plan.len() > 150 {
                    break next.end && !next.dead;
                }
                let toks: Vec<&String> = next.allowed.iter().collect();
                plan.push(rng.pick(&toks).to_string());
            };
            // A walk cut short ends at its last whole step.
            let words: Vec<&str> = plan.iter().map(String::as_str).collect();
            let (ends, _) = split(&words).unwrap();
            let keep = if finished { words.len() } else { ends.last().copied().unwrap_or(0) };
            let text = words[..keep].join(" ");
            let steps = decode(&text).unwrap();
            let c = check(&model, &steps);
            if finished && plan.last().map(String::as_str) == Some("exit") {
                assert!(c.solved, "{text}: {:?}", c.problem);
                solved += 1;
            } else {
                // Cut short (or a dead end): every whole step still fits.
                assert!(c.solved || c.steps_ok == steps.len(), "{text}: {:?}", c.problem);
            }
        }
        // The walks aren't aimed, so few solve; the check above is what matters.
        eprintln!("{solved} of 60 random allowed plans solved");
    }
}
