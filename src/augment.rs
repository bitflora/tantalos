//! Symmetric copies of a level: mirrored and turned about the vertical (`tantalos-data augment`).
//!
//! Gravity only picks out z, so mirroring a chamber in x and turning it by quarter turns about
//! z gives 8 versions of every level that play the same: the same plan, moved. The Workshop
//! part of the training data repeats each of its levels many times, and the model learns
//! them by heart (2026-10-25: 1.59 nats a token on held-out Workshop levels, 0.24 on trained
//! ones); training on the 8 versions instead gives it distinct sequences.
//!
//! The transform works on tokens (`tokens.rs` documents the format): every cell, box corner and
//! direction is mapped, a turret's facing turned, and the result decoded and encoded again, so
//! items come out in `encode`'s order with their wires renumbered. Two things carry a
//! handedness that a mirror would flip:
//! - a fizzler or laser field sits on the normal × along side of its cells (`Builder::fizzler`),
//!   so a mirrored field starts at its far end and runs back;
//! - a rail platform hangs on the wall left of its travel, if that is solid, else the one on its
//!   right (`tokens::decode`), so a mirrored one finds its wall on the other side.
//!
//! The decoder makes its other choices (a pedestal's column side, which goo cell holds the goo
//! item, a glass panel's anchor) from what the tokens give, and tokens don't record them.

use crate::geom::{Cell, Dir};
use std::collections::BTreeMap;

use crate::editor::editor_flags;
use crate::level::{ItemKind, Level};
use crate::p2c::Puzzle;
use crate::plan::{self, Step};
use crate::tokens::{self, dir_token, parse_dir};

/// One of the 8 symmetries: mirror x (if `mirror`), then `turns` quarter turns about +z
/// (counter-clockwise seen from above: +x goes to +y).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sym {
    pub mirror: bool,
    pub turns: u8,
}

impl Sym {
    pub const IDENTITY: Sym = Sym { mirror: false, turns: 0 };

    /// All 8, the identity first.
    pub const ALL: [Sym; 8] = [
        Sym { mirror: false, turns: 0 },
        Sym { mirror: false, turns: 1 },
        Sym { mirror: false, turns: 2 },
        Sym { mirror: false, turns: 3 },
        Sym { mirror: true, turns: 0 },
        Sym { mirror: true, turns: 1 },
        Sym { mirror: true, turns: 2 },
        Sym { mirror: true, turns: 3 },
    ];

    /// `r90`, `m180`, ...: `r` for a turn only, `m` for mirror then turn.
    pub fn name(self) -> String {
        format!("{}{}", if self.mirror { "m" } else { "r" }, 90 * self.turns as u32)
    }

    pub fn parse(name: &str) -> Option<Sym> {
        Sym::ALL.into_iter().find(|s| s.name() == name)
    }

    /// The symmetry that undoes this one.
    pub fn inverse(self) -> Sym {
        if self.mirror {
            self // M then R^k, twice, is M R^k M R^k = R^-k R^k
        } else {
            Sym { mirror: false, turns: (4 - self.turns) % 4 }
        }
    }

    /// The chamber size afterwards.
    pub fn size(self, size: [i32; 3]) -> [i32; 3] {
        if self.turns % 2 == 1 {
            [size[1], size[0], size[2]]
        } else {
            size
        }
    }

    /// Where cell `c` of a chamber of `size` goes.
    pub fn cell(self, c: Cell, size: [i32; 3]) -> Cell {
        let [mut x, mut y, z] = c;
        let (mut w, mut h) = (size[0], size[1]);
        if self.mirror {
            x = w - 1 - x;
        }
        for _ in 0..self.turns {
            (x, y) = (h - 1 - y, x);
            (w, h) = (h, w);
        }
        [x, y, z]
    }

    pub fn dir(self, d: Dir) -> Dir {
        let [mut x, mut y, z] = d.vector();
        if self.mirror {
            x = -x;
        }
        for _ in 0..self.turns {
            (x, y) = (-y, x);
        }
        let axis = [x, y, z].iter().position(|&v| v != 0).expect("a unit vector");
        Dir::from_axis(axis, [x, y, z][axis] > 0)
    }

    /// A turret's facing, in eighths of a turn from +x toward +y.
    pub fn facing(self, f: i32) -> i32 {
        let f = if self.mirror { 4 - f } else { f };
        (f + 2 * self.turns as i32).rem_euclid(8)
    }
}

/// Reads canonical tokens and writes them moved.
struct Rewrite<'a> {
    toks: &'a [String],
    pos: usize,
    out: Vec<String>,
    sym: Sym,
    size: [i32; 3],
}

impl Rewrite<'_> {
    fn peek(&self) -> Option<&str> {
        self.toks.get(self.pos).map(String::as_str)
    }

    fn take(&mut self) -> Result<&str, String> {
        let t = self.toks.get(self.pos).ok_or("ran out of tokens")?;
        self.pos += 1;
        Ok(t)
    }

    fn copy(&mut self) -> Result<(), String> {
        let t = self.take()?.to_string();
        self.out.push(t);
        Ok(())
    }

    fn copy_if(&mut self, word: &str) {
        if self.peek() == Some(word) {
            self.pos += 1;
            self.out.push(word.to_string());
        }
    }

    fn num(&mut self) -> Result<i32, String> {
        let t = self.take()?.to_string();
        t.parse().map_err(|_| format!("token {}: expected a number, found {t:?}", self.pos - 1))
    }

    fn read_cell(&mut self) -> Result<Cell, String> {
        Ok([self.num()?, self.num()?, self.num()?])
    }

    fn put_cell(&mut self, c: Cell) {
        self.out.extend(c.map(|v| v.to_string()));
    }

    fn cell(&mut self) -> Result<(), String> {
        let c = self.read_cell()?;
        self.put_cell(self.sym.cell(c, self.size));
        Ok(())
    }

    fn read_dir(&mut self) -> Result<Dir, String> {
        let t = self.take()?.to_string();
        parse_dir(&t).ok_or_else(|| format!("token {}: expected a direction, found {t:?}", self.pos - 1))
    }

    fn dir(&mut self) -> Result<(), String> {
        let d = self.read_dir()?;
        self.out.push(dir_token(self.sym.dir(d)).to_string());
        Ok(())
    }

    /// A box: both corners moved, then put back in order.
    fn one_box(&mut self) -> Result<(), String> {
        let (a, b) = (self.read_cell()?, self.read_cell()?);
        let (a, b) = (self.sym.cell(a, self.size), self.sym.cell(b, self.size));
        self.put_cell(std::array::from_fn(|i| a[i].min(b[i])));
        self.put_cell(std::array::from_fn(|i| a[i].max(b[i])));
        Ok(())
    }

    fn boxes(&mut self) -> Result<(), String> {
        while self.peek().is_some_and(|t| t.bytes().all(|c| c.is_ascii_digit())) {
            self.one_box()?;
        }
        Ok(())
    }

    /// `fizzler|laserfield x y z N A L on|off`: the field is on the N × A side of each cell,
    /// so under a mirror it starts at the far end and runs back along -A.
    fn field(&mut self) -> Result<(), String> {
        let anchor = self.read_cell()?;
        let (n, along) = (self.read_dir()?, self.read_dir()?);
        let len = self.num()?;
        let (start, along) = if self.sym.mirror {
            let v = along.vector();
            let far: Cell = std::array::from_fn(|a| anchor[a] + v[a] * (len - 1));
            (far, self.sym.dir(along).opposite())
        } else {
            (anchor, self.sym.dir(along))
        };
        self.put_cell(self.sym.cell(start, self.size));
        self.out.push(dir_token(self.sym.dir(n)).to_string());
        self.out.push(dir_token(along).to_string());
        self.out.push(len.to_string());
        self.copy()
    }

    /// `> target [N]` after a sender: copied as is (item order doesn't change here).
    fn wires(&mut self) -> Result<(), String> {
        while self.peek() == Some(">") {
            self.copy()?;
            self.copy()?;
            if self.peek().is_some_and(|t| t.bytes().all(|c| c.is_ascii_digit())) {
                self.copy()?;
            }
        }
        Ok(())
    }

    fn run(mut self) -> Result<Vec<String>, String> {
        if self.peek() != Some("size") {
            return Err("expected `size`".into());
        }
        self.copy()?;
        self.size = [self.num()?, self.num()?, self.num()?];
        let moved = self.sym.size(self.size);
        self.out.extend(moved.map(|v| v.to_string()));
        while let Some(t) = self.peek() {
            let t = t.to_string();
            self.pos += 1;
            self.out.push(t.clone());
            match t.as_str() {
                "open" | "goo" => self.boxes()?,
                "white" | "black" => {
                    self.dir()?;
                    self.boxes()?;
                }
                "entry" | "exit" => {
                    self.cell()?;
                    self.dir()?;
                    self.copy_if("inv");
                }
                "glass" | "grating" => {
                    self.one_box()?;
                    self.dir()?;
                    self.dir()?;
                }
                "fizzler" | "laserfield" => self.field()?,
                "dropper" => {
                    self.cell()?;
                    self.copy()?;
                    self.copy_if("wait");
                }
                "cube" => {
                    self.cell()?;
                    self.copy()?;
                }
                "laser" | "catcher" | "relay" => {
                    self.cell()?;
                    self.dir()?;
                    if self.peek() == Some("center") {
                        self.copy()?;
                    } else {
                        self.dir()?;
                    }
                    if t == "laser" {
                        self.copy()?;
                    }
                    self.wires()?;
                }
                "plate" => {
                    self.cell()?;
                    self.dir()?;
                    self.cell()?;
                    self.dir()?;
                }
                "bridge" => {
                    self.cell()?;
                    self.dir()?;
                    self.dir()?;
                    self.copy()?;
                }
                "funnel" => {
                    self.cell()?;
                    self.dir()?;
                    self.copy()?;
                    self.copy_if("rev");
                }
                "panel" | "glasspanel" => {
                    self.cell()?;
                    self.dir()?;
                    self.dir()?;
                    self.copy()?;
                    self.copy()?;
                }
                "flip" => {
                    self.cell()?;
                    self.dir()?;
                }
                "stairs" => {
                    self.cell()?;
                    self.dir()?;
                    self.copy()?;
                }
                "piston" => {
                    self.cell()?;
                    self.dir()?;
                    self.copy()?;
                    self.copy()?;
                    self.copy()?;
                }
                "rail" => {
                    self.cell()?;
                    self.cell()?;
                    self.copy()?;
                }
                "button" | "pedestal" => {
                    self.cell()?;
                    self.copy()?;
                    self.wires()?;
                }
                "gel" => {
                    self.cell()?;
                    self.dir()?;
                    self.copy()?;
                    if self.peek() == Some("stream") {
                        self.copy()?;
                    }
                    self.copy()?;
                }
                "turret" => {
                    self.cell()?;
                    let f = self.num()?;
                    self.out.push(self.sym.facing(f).to_string());
                }
                other => return Err(format!("token {}: unexpected {other:?}", self.pos - 1)),
            }
        }
        Ok(self.out)
    }
}

/// `tokens` (as `encode` writes them) moved by `sym`, not yet in canonical order.
pub fn apply(tokens: &[String], sym: Sym) -> Result<Vec<String>, String> {
    Rewrite { toks: tokens, pos: 0, out: Vec::with_capacity(tokens.len()), sym, size: [0; 3] }.run()
}

/// The puzzle moved by `sym`, and its canonical tokens, if it is `p` moved: the same solid,
/// goo, white faces, glass, grating and fields cell for cell (`moved`), no more mounting
/// problems, and the same editor flags. The `Err` says which failed:
/// - "unmounted": a mirrored field is anchored at its far end, and a field whose strip runs
///   past the end of its wall has nothing behind it there;
/// - "editor flags": the decoder's own choices (a pedestal's column side, the goo item's cell)
///   can land an item where our model of the editor objects.
pub fn variant(p: &Puzzle, sym: Sym) -> Result<(Puzzle, Vec<String>), String> {
    let moved = apply(&tokens::encode(p)?, sym)?;
    let q = tokens::decode(&moved.join(" "))?;
    let toks = tokens::encode(&q)?;
    // `encode` renumbers wire targets in its own order, which can push one past the largest
    // number a token holds (a level with over 32 panels), and then the tokens don't read back.
    tokens::decode(&toks.join(" ")).map_err(|e| format!("unreadable in {}: {e}", sym.name()))?;
    let (a, b) = (Level::from_puzzle(p), Level::from_puzzle(&q));
    let name = sym.name();
    let (before, after) = (a.mount_problems_with(false), b.mount_problems_with(false));
    if after.len() > before.len() {
        return Err(format!("unmounted in {name}: {}", after.join("; ")));
    }
    if editor_rules(&a) != editor_rules(&b) {
        return Err(format!("editor flags in {name}: {:?} became {:?}", editor_rules(&a), editor_rules(&b)));
    }
    moved_level(&a, &b, sym).map_err(|e| format!("geometry in {name}: {e}"))?;
    Ok((q, toks))
}

/// A plan for a chamber of `size` (plan tokens, `plan.rs`) moved by `sym`, for the moved level:
/// every cell and direction mapped. A trailing `exit` stays.
pub fn apply_plan(text: &str, sym: Sym, size: [i32; 3]) -> Result<String, String> {
    let c = |x: Cell| sym.cell(x, size);
    let d = |x: Dir| sym.dir(x);
    let steps: Vec<Step> = plan::decode(text)?
        .into_iter()
        .map(|s| match s {
            Step::Walk(x) => Step::Walk(c(x)),
            Step::Fire(k, x, e) => Step::Fire(k, c(x), d(e)),
            Step::Enter(k) => Step::Enter(k),
            Step::PickUp(x) => Step::PickUp(c(x)),
            Step::PutDown(x, e) => Step::PutDown(c(x), e.map(d)),
            Step::Toss(x) => Step::Toss(c(x)),
            Step::Throw(k, x) => Step::Throw(k, c(x)),
            Step::Press(x) => Step::Press(c(x)),
            Step::Dash(x, y) => Step::Dash(c(x), c(y)),
            Step::Fetch(x, y, z) => Step::Fetch(c(x), c(y), c(z)),
            Step::Reach(x, y) => Step::Reach(c(x), c(y)),
            Step::Knock(x) => Step::Knock(c(x)),
            Step::Bounce(x) => Step::Bounce(c(x)),
            Step::Leap(x) => Step::Leap(c(x)),
            Step::Plate(x) => Step::Plate(c(x)),
            Step::Climb(x) => Step::Climb(c(x)),
            Step::Funnel(x) => Step::Funnel(c(x)),
            Step::Ride(x) => Step::Ride(c(x)),
            Step::Dive(k, x) => Step::Dive(k, c(x)),
            Step::Loop(k, f, x) => Step::Loop(k, f.map(|(y, e)| (c(y), d(e))), c(x)),
            Step::Grab(k, x) => Step::Grab(k, c(x)),
            Step::Double(x, y, z) => Step::Double(c(x), c(y), c(z)),
            Step::Sweep(x, e) => Step::Sweep(c(x), d(e)),
        })
        .collect();
    let mut out: Vec<String> = steps.iter().map(plan::step_text).collect();
    if text.split_whitespace().last() == Some("exit") {
        out.push("exit".into());
    }
    Ok(out.join(" "))
}

/// The editor flags' item names and rules, sorted (their positions move).
pub fn editor_rules(level: &Level) -> Vec<String> {
    let mut out: Vec<String> = editor_flags(level)
        .into_iter()
        .flat_map(|f| {
            let name = f.name;
            f.rules.into_iter().map(move |r| format!("{name}:{r}"))
        })
        .collect();
    out.sort();
    out
}

/// Checks that `b` is `a` moved by `sym`: size, solid and goo cells, portal surfaces, glass,
/// grating and field faces, and the items `placed` counts.
pub fn moved_level(a: &Level, b: &Level, sym: Sym) -> Result<(), String> {
    if b.size != sym.size(a.size) {
        return Err(format!("size {:?}, not {:?}", b.size, sym.size(a.size)));
    }
    for c in a.cells() {
        let m = sym.cell(c, a.size);
        if a.solid(c) != b.solid(m) || a.goo(c) != b.goo(m) {
            return Err(format!("cell {c:?}"));
        }
        for d in Dir::ALL {
            let e = sym.dir(d);
            let (fa, fb) = (a.face(c, d), b.face(m, e));
            if a.portal_surface(c, d) != b.portal_surface(m, e)
                || fa.barrier != fb.barrier
                || (fa.hazards != 0) != (fb.hazards != 0)
            {
                return Err(format!("face {c:?} {d:?}"));
            }
        }
    }
    if placed(a, sym) != placed(b, Sym::IDENTITY) {
        return Err("items".into());
    }
    Ok(())
}

/// Items by kind, cell and normal, moved by `sym`. Left out: goo (the decoder picks which cell
/// of a pool holds the item), glass and grating (it picks the anchor along the edge), and
/// fields with their extents (a mirrored field is anchored at its other end); `moved_level`
/// compares their faces instead. A rail between two solid walls hangs on the one left of its
/// travel, which a mirror makes the other one, and tokens don't say which (`tokens::decode`), so
/// only its cells count.
pub fn placed(level: &Level, sym: Sym) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    for it in &level.items {
        if matches!(
            it.kind,
            ItemKind::Goo
                | ItemKind::Glass
                | ItemKind::Grating
                | ItemKind::Fizzler
                | ItemKind::LaserField
                | ItemKind::BarrierExtent
                | ItemKind::HazardExtent
        ) {
            continue;
        }
        let normal = match it.kind {
            ItemKind::Rail | ItemKind::RailExtent => None,
            _ => it.normal.map(|d| sym.dir(d)),
        };
        let key = format!("{:?} {:?} {normal:?}", it.kind, sym.cell(it.cell, level.size));
        *out.entry(key).or_default() += 1;
    }
    out
}
