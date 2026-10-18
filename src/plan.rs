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

use std::collections::HashSet;

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
    /// The goal is reached after `plan.len()` steps (steps after that are ignored): the
    /// shortest part of the plan that can reach it.
    pub solved: bool,
    /// The solver's actions for the steps that fit, up to the goal or the step that failed.
    pub plan: Vec<Action>,
    /// Steps that fit, on the longest way through the plan.
    pub steps_ok: usize,
    /// Why not: step `steps_ok + 1` fits no action, or the plan ends short of the goal.
    pub problem: Option<String>,
    /// Too many ways through the plan to follow (`CHECK_FRONTIER`) before it was settled either
    /// way: the verdict is unknown, not a failure of the plan. Then `steps_ok` counts the steps
    /// that were followed in full.
    pub exhausted: bool,
    /// A step fitted actions to more than one distinct state, so there was more than one way
    /// through the plan to follow. Not a problem, just which plans exercise the search.
    pub ambiguous: bool,
}

/// Most states `check` follows at once: a step can fit more than one action (two ledges down
/// to the same region, one of which fizzles the cube; a dash that ends in different states),
/// and every way through the plan so far is followed until one reaches the goal.
const CHECK_FRONTIER: usize = 100_000;

/// Replays `steps` on `model` (see the module docs). Follows every state the steps so far can
/// lead to, step by step, so an ambiguous step can be settled by any later one.
pub fn check(model: &Model, steps: &[Step]) -> Checked {
    check_from(model, model.initial(), steps)
}

/// `check`, from state `start`.
fn check_from(model: &Model, start: State, steps: &[Step]) -> Checked {
    // The states after `i` steps, each once, and for each step how its states were reached:
    // (index of the state it came from among the step's before, action).
    let mut states = vec![start];
    let mut back: Vec<Vec<(usize, Action)>> = Vec::new();
    // The actions to the state at `k` after the last step.
    let trace = |back: &[Vec<(usize, Action)>], mut k: usize| {
        let mut plan: Vec<Action> = back
            .iter()
            .rev()
            .map(|layer| {
                let (from, a) = layer[k];
                k = from;
                a
            })
            .collect();
        plan.reverse();
        plan
    };
    let mut ambiguous = false;
    let mut i = 0;
    loop {
        if let Some(k) = states.iter().position(|s| model.is_goal(s)) {
            return Checked {
                solved: true,
                plan: trace(&back, k),
                steps_ok: i,
                problem: None,
                exhausted: false,
                ambiguous,
            };
        }
        let failed = |problem: String, exhausted: bool| Checked {
            solved: false,
            plan: trace(&back, 0),
            steps_ok: i,
            problem: Some(problem),
            exhausted,
            ambiguous,
        };
        let Some(step) = steps.get(i) else {
            return failed("the plan ends before the exit".into(), false);
        };
        let mut next: Vec<State> = Vec::new();
        let mut seen: HashSet<State> = HashSet::new();
        let mut layer: Vec<(usize, Action)> = Vec::new();
        for (k, s) in states.iter().enumerate() {
            for (a, n) in model.successors(s) {
                // Two fitting actions can lead to the same state (two ledges down to one
                // region), from one state or two: it is followed once.
                if !fits(model, s, &a, &n, step) || seen.contains(&n) {
                    continue;
                }
                seen.insert(n.clone());
                next.push(n);
                layer.push((k, a));
            }
            if next.len() > CHECK_FRONTIER {
                return failed("the check ran out of budget".into(), true);
            }
        }
        if next.is_empty() {
            return failed(
                format!("step {} ({}) fits no move", i + 1, step_text(step)),
                false,
            );
        }
        ambiguous |= next.len() > 1;
        states = next;
        back.push(layer);
        i += 1;
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
            k == color && model.portals.has_face(class, (c, d))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::level::Level;
    use crate::solver::Mode;

    /// Two droppers leave a sphere and a standard cube in one region, a sphere button opens
    /// the exit (data/plans/test-gen.jsonl).
    const LEVEL: &str = "size 7 9 8 open 3 3 1 4 6 7 0 0 2 6 8 7 white +x 4 4 1 6 8 1 white -x 3 4 1         6 8 7 0 4 2 6 4 5 0 5 4 6 7 5 white +y 1 8 2 6 8 5 entry 0 7 2 +x exit 1 0 2 +y         laserfield 2 8 2 -y +z 1 on dropper 0 6 7 sphere dropper 1 7 7 standard button 1 2 2         sphere > exit button 5 0 2 weighted > exit pedestal 3 3 1 inf > exit turret 4 0 2 3         laser 6 1 2 -x -z on catcher 4 5 1 -x -z > stairs 0 funnel 5 0 3 +y on stairs 5 2 2         +x off";
    const PLAN: &str = "fire blue 2 8 3 +y fire orange 0 4 2 -x pickup 1 7 2 enter orange         knock 4 0 2 walk 5 0 2 putdown 5 0 2 walk 6 1 2 walk 4 3 1 fire blue 3 4 1 -x         press 3 3 1 enter blue pickup 0 6 2 putdown 1 2 2 exit";

    /// Step 3 picks up either cube. The sphere does on the weighted button (step 7) as well as
    /// the standard cube, so with the sphere every later step still fits, until the last puts
    /// the standard cube on the sphere button, where it does nothing: the check has to keep
    /// both ways until the end.
    #[test]
    fn ambiguity_settled_at_the_end() {
        let level = Level::from_puzzle(&crate::tokens::decode(LEVEL).unwrap());
        let model = Model::new(&level, Mode::Conservative).unwrap();
        let steps = decode(PLAN).unwrap();
        let c = check(&model, &steps);
        assert!(c.solved, "{:?}", c.problem);
        assert!(c.ambiguous);
        assert_eq!(c.plan.len(), steps.len());
        let states = crate::metrics::replay(&model, &c.plan).unwrap();
        let wrong: Vec<State> = model
            .successors(&states[2])
            .into_iter()
            .filter(|(a, n)| *n != states[3] && fits(&model, &states[2], a, n, &steps[2]))
            .map(|(_, n)| n)
            .collect();
        assert_eq!(wrong.len(), 1);
        let w = check_from(&model, wrong[0].clone(), &steps[3..]);
        assert!(!w.solved);
        assert_eq!(w.steps_ok, steps.len() - 3, "{:?}", w.problem);
    }

    /// Picking up either cube and putting it back, over and over, splits the way through the
    /// plan in two and joins it again each time: followed one way at a time, that's 2^n ways
    /// by the end, here two states at most. With a last step that fits nothing, the plan is
    /// settled as illegal there.
    #[test]
    fn rejoined_ways_are_followed_once() {
        let level = Level::from_puzzle(&crate::tokens::decode(LEVEL).unwrap());
        let model = Model::new(&level, Mode::Conservative).unwrap();
        let plan = decode(PLAN).unwrap();
        let mut steps = plan[..2].to_vec();
        for _ in 0..40 {
            steps.extend(decode("pickup 1 7 2 putdown 1 7 2").unwrap());
        }
        steps.extend(&plan[2..plan.len() - 1]);
        steps.push(Step::Walk([31, 31, 31]));
        let c = check(&model, &steps);
        assert!(!c.solved && !c.exhausted && c.ambiguous);
        assert_eq!(c.steps_ok, steps.len() - 1, "{:?}", c.problem);
        // Without the bad step, solved.
        steps.pop();
        steps.extend(&plan[plan.len() - 1..]);
        let c = check(&model, &steps);
        assert!(c.solved, "{:?}", c.problem);
    }
}
