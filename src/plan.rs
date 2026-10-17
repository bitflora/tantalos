//! Plans as tokens, for a model that learns to solve levels, and a checker for the plans it
//! writes.
//!
//! A plan is one group of words per step of a route (`route`), in the level's own coordinates
//! (the numbers and directions of `tokens`), without the walking between steps:
//!
//! ```text
//! walk X Y Z                where the player ends up (a drop, a step, a walk across a field)
//! fire blue|orange X Y Z D  the face the portal goes on (cell and direction, as `white` boxes)
//! enter blue|orange         walk into that portal
//! pickup X Y Z              the cube's cell
//! putdown X Y Z [D]         where the cube goes; D: the way it faces, at a laser spot
//! toss X Y Z                where the cube lands, dropped off a ledge
//! throw blue|orange X Y Z   where the cube lands, thrown into that portal
//! press X Y Z               a latching pedestal's cell
//! dash X Y Z X Y Z          a timed pedestal's cell, and where the dash ends
//! knock X Y Z               a turret's cell
//! bounce|leap|plate|climb|funnel|ride X Y Z    where the player ends up
//! dive blue|orange X Y Z    where the player lands, having dived into that floor portal
//! exit                      walk out (optional, last)
//! ```
//!
//! `check` replays such a plan on the solver's model: each step has to fit an action the
//! solver could take next (the face in the action's face class, the cell in the region it
//! leads to, the cube where the action puts it, ...) and the plan has to reach the goal.
//! Any face of the right class is accepted, so a plan may differ from the route's choices
//! and still pass. What happens between the steps (the walks) is the solver's business.

use crate::geom::{Cell, Dir};
use crate::level::CubeKind;
use crate::regions::Move;
use crate::route::{Act, Route};
use crate::solver::{Action, Color, CubeLoc, Model, State};
use crate::tokens::{dir_token, parse_dir, MAX_NUM};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Walk(Cell),
    Fire(Color, Cell, Dir),
    Enter(Color),
    PickUp(Cell),
    PutDown(Cell, Option<Dir>),
    Toss(Cell),
    Throw(Color, Cell),
    Press(Cell),
    Dash(Cell, Cell),
    Knock(Cell),
    Bounce(Cell),
    Leap(Cell),
    Plate(Cell),
    Climb(Cell),
    Funnel(Cell),
    Ride(Cell),
    Dive(Color, Cell),
}

/// Words plans use beyond the level format's numbers and directions.
pub const WORDS: [&str; 19] = [
    "walk", "fire", "enter", "pickup", "putdown", "toss", "throw", "press", "dash", "knock",
    "bounce", "leap", "plate", "climb", "funnel", "ride", "dive", "blue", "orange",
];

/// Every word a plan can use (`exit` and the numbers and directions are in `tokens::vocab`
/// too).
pub fn vocab() -> Vec<String> {
    let mut v: Vec<String> = WORDS.iter().map(|w| w.to_string()).collect();
    v.push("exit".into());
    v.extend((0..=MAX_NUM).map(|n| n.to_string()));
    v.extend(
        [Dir::PosX, Dir::NegX, Dir::PosY, Dir::NegY, Dir::PosZ, Dir::NegZ]
            .map(|d| dir_token(d).to_string()),
    );
    v
}

fn color_word(c: Color) -> &'static str {
    match c {
        Color::Blue => "blue",
        Color::Orange => "orange",
    }
}

/// The route as plan tokens, space-separated.
pub fn encode(route: &Route) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut put = |words: &[&str], cells: &[Cell]| {
        out.extend(words.iter().map(|w| w.to_string()));
        for c in cells {
            out.extend(c.iter().map(|n| n.to_string()));
        }
    };
    let mut before = &route.start;
    for leg in &route.legs {
        let to = leg.after.player;
        match &leg.act {
            Act::Walk => put(&["walk"], &[to]),
            Act::Fire { color, face, .. } => {
                put(&["fire", color_word(*color)], &[face.0]);
                put(&[dir_token(face.1)], &[]);
            }
            Act::Enter { color, .. } => put(&["enter", color_word(*color)], &[]),
            Act::PickUp { cube } => put(&["pickup"], &[*cube]),
            Act::PutDown { at, facing, .. } => {
                put(&["putdown"], &[*at]);
                if let Some(d) = facing {
                    put(&[dir_token(*d)], &[]);
                }
            }
            Act::Toss { landing, .. } => put(&["toss"], &[*landing]),
            Act::Throw { color, landing, .. } => put(&["throw", color_word(*color)], &[*landing]),
            Act::Press { cell, .. } => put(&["press"], &[*cell]),
            Act::Dash { cell, .. } => put(&["dash"], &[*cell, to]),
            Act::Knock { cell, .. } => put(&["knock"], &[*cell]),
            Act::Bounce { .. } => put(&["bounce"], &[to]),
            Act::Leap { .. } => put(&["leap"], &[to]),
            Act::Plate { .. } => put(&["plate"], &[to]),
            Act::Climb { .. } => put(&["climb"], &[to]),
            Act::Funnel { .. } => put(&["funnel"], &[to]),
            Act::Ride { .. } => put(&["ride"], &[to]),
            // The portal dived into is the one on the floor face before the dive.
            Act::Dive { face, .. } => {
                let color = if before.portals[1] == Some(*face) {
                    Color::Orange
                } else {
                    Color::Blue
                };
                put(&["dive", color_word(color)], &[to]);
            }
            Act::Exit { .. } => put(&["exit"], &[]),
        }
        before = &leg.after;
    }
    out.join(" ")
}

struct Reader<'a> {
    words: Vec<&'a str>,
    at: usize,
}

impl<'a> Reader<'a> {
    fn next(&mut self, what: &str) -> Result<&'a str, String> {
        let w = self
            .words
            .get(self.at)
            .copied()
            .ok_or_else(|| format!("token {}: expected {what}, found the end", self.at + 1))?;
        self.at += 1;
        Ok(w)
    }

    fn fail<T>(&self, msg: String) -> Result<T, String> {
        Err(format!("token {}: {msg}", self.at))
    }

    fn num(&mut self) -> Result<i32, String> {
        let t = self.next("a number")?;
        match t.parse::<i32>() {
            Ok(n) if (0..=MAX_NUM).contains(&n) && n.to_string() == t => Ok(n),
            _ => self.fail(format!("expected a number, found {t:?}")),
        }
    }

    fn cell(&mut self) -> Result<Cell, String> {
        Ok([self.num()?, self.num()?, self.num()?])
    }

    fn dir(&mut self) -> Result<Dir, String> {
        let t = self.next("a direction")?;
        parse_dir(t).map_or_else(|| self.fail(format!("expected a direction, found {t:?}")), Ok)
    }

    fn color(&mut self) -> Result<Color, String> {
        match self.next("a portal color")? {
            "blue" => Ok(Color::Blue),
            "orange" => Ok(Color::Orange),
            t => self.fail(format!("expected blue or orange, found {t:?}")),
        }
    }
}

/// The steps of a plan written as tokens. A trailing `exit` is allowed (and dropped).
pub fn decode(text: &str) -> Result<Vec<Step>, String> {
    let mut r = Reader {
        words: text.split_whitespace().collect(),
        at: 0,
    };
    let mut steps = Vec::new();
    while r.at < r.words.len() {
        let verb = r.next("a step")?;
        let step = match verb {
            "walk" => Step::Walk(r.cell()?),
            "fire" => Step::Fire(r.color()?, r.cell()?, r.dir()?),
            "enter" => Step::Enter(r.color()?),
            "pickup" => Step::PickUp(r.cell()?),
            "putdown" => {
                let at = r.cell()?;
                let facing = r.words.get(r.at).and_then(|t| parse_dir(t));
                if facing.is_some() {
                    r.at += 1;
                }
                Step::PutDown(at, facing)
            }
            "toss" => Step::Toss(r.cell()?),
            "throw" => Step::Throw(r.color()?, r.cell()?),
            "press" => Step::Press(r.cell()?),
            "dash" => Step::Dash(r.cell()?, r.cell()?),
            "knock" => Step::Knock(r.cell()?),
            "bounce" => Step::Bounce(r.cell()?),
            "leap" => Step::Leap(r.cell()?),
            "plate" => Step::Plate(r.cell()?),
            "climb" => Step::Climb(r.cell()?),
            "funnel" => Step::Funnel(r.cell()?),
            "ride" => Step::Ride(r.cell()?),
            "dive" => Step::Dive(r.color()?, r.cell()?),
            "exit" if r.at == r.words.len() => break,
            "exit" => return r.fail("exit before the end".into()),
            t => return r.fail(format!("expected a step, found {t:?}")),
        };
        steps.push(step);
    }
    Ok(steps)
}

/// What `check` found.
#[derive(Clone, Debug, PartialEq)]
pub struct Checked {
    /// The goal is reached after `plan.len()` steps (steps after that are ignored).
    pub solved: bool,
    /// The solver's actions for the steps that fit, up to the goal or the step that failed.
    pub plan: Vec<Action>,
    /// Steps that fit, on the longest way through the plan that was found.
    pub steps_ok: usize,
    /// Why not: step `steps_ok + 1` fits no action, or the plan ends short of the goal.
    pub problem: Option<String>,
    /// The state budget ran out before the plan was settled either way: the verdict is unknown,
    /// not a failure of the plan. Then `steps_ok` and `problem` only describe the best path found.
    pub exhausted: bool,
    /// A step fitted more than one distinct action, so `check` had to choose (and back out of
    /// the choices that lead nowhere). Not a problem, just which plans exercise the search.
    pub ambiguous: bool,
}

/// Most states `check` visits: a step can fit more than one action (two ledges down to the
/// same region, a dash that ends in different states), so it backtracks.
const CHECK_BUDGET: usize = 20_000;

/// Replays `steps` on `model` (see the module docs).
pub fn check(model: &Model, steps: &[Step]) -> Checked {
    struct Search<'a> {
        model: &'a Model,
        steps: &'a [Step],
        path: Vec<Action>,
        budget: usize,
        best: (usize, Vec<Action>, String),
        exhausted: bool,
        ambiguous: bool,
    }
    impl Search<'_> {
        fn go(&mut self, s: &State) -> bool {
            let i = self.path.len();
            if self.model.is_goal(s) {
                return true;
            }
            let Some(step) = self.steps.get(i) else {
                if i >= self.best.0 {
                    self.best = (i, self.path.clone(), "the plan ends before the exit".into());
                }
                return false;
            };
            if self.budget == 0 {
                self.exhausted = true;
                return false;
            }
            self.budget -= 1;
            let mut tried: Vec<State> = Vec::new();
            for (a, n) in self.model.successors(s) {
                // Two fitting actions can lead to the same state (two ledges down to one
                // region): only the first is worth exploring. Distinct ones (a drop that
                // fizzles the cube and a walk) all get a turn, and a dead end backs out.
                if !fits(self.model, s, &a, &n, step) || tried.contains(&n) {
                    continue;
                }
                self.ambiguous |= !tried.is_empty();
                tried.push(n.clone());
                self.path.push(a);
                if self.go(&n) {
                    return true;
                }
                self.path.pop();
            }
            if tried.is_empty() && i >= self.best.0 {
                self.best = (
                    i,
                    self.path.clone(),
                    format!("step {} ({}) fits no move", i + 1, step_text(step)),
                );
            }
            false
        }
    }
    let mut search = Search {
        model,
        steps,
        path: Vec::new(),
        budget: CHECK_BUDGET,
        best: (0, Vec::new(), String::new()),
        exhausted: false,
        ambiguous: false,
    };
    let start = model.initial();
    if search.go(&start) {
        return Checked {
            solved: true,
            steps_ok: search.path.len(),
            plan: search.path,
            problem: None,
            exhausted: false,
            ambiguous: search.ambiguous,
        };
    }
    let (steps_ok, plan, mut problem) = search.best;
    if search.exhausted || problem.is_empty() {
        problem = "the check ran out of budget".into();
    }
    Checked {
        solved: false,
        plan,
        steps_ok,
        problem: Some(problem),
        exhausted: search.exhausted,
        ambiguous: search.ambiguous,
    }
}

/// A step as its tokens.
pub fn step_text(step: &Step) -> String {
    let c = |c: &Cell| format!("{} {} {}", c[0], c[1], c[2]);
    match step {
        Step::Walk(x) => format!("walk {}", c(x)),
        Step::Fire(k, x, d) => format!("fire {} {} {}", color_word(*k), c(x), dir_token(*d)),
        Step::Enter(k) => format!("enter {}", color_word(*k)),
        Step::PickUp(x) => format!("pickup {}", c(x)),
        Step::PutDown(x, None) => format!("putdown {}", c(x)),
        Step::PutDown(x, Some(d)) => format!("putdown {} {}", c(x), dir_token(*d)),
        Step::Toss(x) => format!("toss {}", c(x)),
        Step::Throw(k, x) => format!("throw {} {}", color_word(*k), c(x)),
        Step::Press(x) => format!("press {}", c(x)),
        Step::Dash(x, y) => format!("dash {} {}", c(x), c(y)),
        Step::Knock(x) => format!("knock {}", c(x)),
        Step::Bounce(x) => format!("bounce {}", c(x)),
        Step::Leap(x) => format!("leap {}", c(x)),
        Step::Plate(x) => format!("plate {}", c(x)),
        Step::Climb(x) => format!("climb {}", c(x)),
        Step::Funnel(x) => format!("funnel {}", c(x)),
        Step::Ride(x) => format!("ride {}", c(x)),
        Step::Dive(k, x) => format!("dive {} {}", color_word(*k), c(x)),
    }
}

/// The verb of a step (for counting which kinds fail).
pub fn step_verb(step: &Step) -> &'static str {
    match step {
        Step::Walk(_) => "walk",
        Step::Fire(..) => "fire",
        Step::Enter(_) => "enter",
        Step::PickUp(_) => "pickup",
        Step::PutDown(..) => "putdown",
        Step::Toss(_) => "toss",
        Step::Throw(..) => "throw",
        Step::Press(_) => "press",
        Step::Dash(..) => "dash",
        Step::Knock(_) => "knock",
        Step::Bounce(_) => "bounce",
        Step::Leap(_) => "leap",
        Step::Plate(_) => "plate",
        Step::Climb(_) => "climb",
        Step::Funnel(_) => "funnel",
        Step::Ride(_) => "ride",
        Step::Dive(..) => "dive",
    }
}

/// Whether `step` is what action `a` (from `s` to `next`) does.
fn fits(model: &Model, s: &State, a: &Action, next: &State, step: &Step) -> bool {
    let region = |c: &Cell| model.graph.region_of(*c);
    let lands = |c: &Cell| region(c) == Some(next.region);
    let held = s
        .cubes
        .iter()
        .find(|c| c.loc == CubeLoc::Held)
        .map(|c| c.kind);
    match (*step, *a) {
        (Step::Walk(c), Action::Move { kind, .. }) => {
            matches!(kind, Move::Walk | Move::Drop { .. } | Move::Step) && lands(&c)
        }
        (Step::Bounce(c), Action::Move { kind: Move::Bounce, .. })
        | (Step::Leap(c), Action::Move { kind: Move::Leap, .. })
        | (Step::Plate(c), Action::Move { kind: Move::Plate, .. })
        | (Step::Climb(c), Action::Move { kind: Move::Stairs, .. })
        | (Step::Funnel(c), Action::Move { kind: Move::Funnel, .. })
        | (Step::Ride(c), Action::Move { kind: Move::Ride, .. }) => lands(&c),
        (Step::Fire(k, c, d), Action::Fire { color, class, .. }) => {
            k == color && model.portals.classes[class].faces.contains(&(c, d))
        }
        (Step::Enter(k), Action::Enter { color }) => k == color,
        (Step::PickUp(c), Action::PickUp { from, .. }) => loc_fits(model, from, c, None, None),
        (Step::PutDown(c, d), Action::PutDown { at }) => loc_fits(model, at, c, d, held),
        (Step::Toss(c), Action::Toss { to }) => loc_fits(model, to, c, None, held),
        (Step::Throw(k, c), Action::Throw { color, to }) => {
            k == color && loc_fits(model, to, c, None, held)
        }
        (Step::Press(c), Action::Press { pedestal }) => model.pedestals[pedestal].cell == c,
        (Step::Dash(p, end), Action::Dash { pedestal, escaped, .. }) => {
            // Out of the exit, the level is over wherever the dash ends.
            model.pedestals[pedestal].cell == p && (escaped || lands(&end))
        }
        (Step::Knock(c), Action::Knock { turret }) => model.turrets[turret].cell == c,
        (Step::Dive(k, c), Action::Dive { color, .. }) => k == color && lands(&c),
        _ => false,
    }
}

/// Whether a cube in `cell` (facing `dir`, at a laser spot) is at `loc`. `kind`: the cube's
/// kind, when it matters which buttons take it (putting one down: a cube in a button's cell
/// that the button takes is on the button, not loose).
fn loc_fits(
    model: &Model,
    loc: CubeLoc,
    cell: Cell,
    dir: Option<Dir>,
    kind: Option<CubeKind>,
) -> bool {
    match loc {
        CubeLoc::OnButton(b) => model.buttons[b].cell == cell,
        CubeLoc::Spot(k) => {
            model.spot_cell(k) == cell && dir.is_none_or(|d| d == model.beams.spots[k].dir)
        }
        CubeLoc::Loose(r) => {
            model.graph.region_of(cell) == Some(r)
                && !kind.is_some_and(|k| {
                    model
                        .buttons
                        .iter()
                        .any(|b| b.cell == cell && b.kind.accepts(k))
                })
        }
        // A loose cube fizzled on the way: wherever it was going.
        CubeLoc::Gone => true,
        CubeLoc::Held => false,
    }
}
