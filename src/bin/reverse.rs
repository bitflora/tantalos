//! Reverse generator: build a chamber backwards from the goal so it is solvable by
//! construction (plan.md, "self-play" -> "solvability by construction": reverse play).
//!
//! This is a **separate binary** (`reverse`, run with `cargo run --bin reverse`) so both it and
//! the forward `glados dataset` generator live in the crate at once and can be compared on the
//! same data. It only *adds* files: it borrows the library (`builder`, `tokens`, `dataset`) and
//! does not touch the forward generator, the solver, or `main.rs`.
//!
//! # Why reverse
//! The forward `dataset` loop samples random/mutated genomes and keeps the ones the solver can
//! prove solvable. That ties the *supply* of training levels to the solver's search: every kept
//! level passed a BFS, so the corpus is exactly "levels a conservative BFS finds in 50k states"
//! and the hard, human-authored shapes get dropped. Reverse generation flips the dependency:
//! we choose a **solution plan first** and build geometry that requires it, so solvability is a
//! property of the construction, not of a search. The solver is then only a *measurer* (for
//! labels/metrics), never the source of hardness.
//!
//! # What a stage demands
//! Each stage is a **gadget**, drawn from a menu: a latching or timed pedestal on a two-voxel
//! ledge reached by a portal pair (`latch`, `timed`); a floor button on that ledge fed by a
//! cube carried through the portal (`cube_button`); a ledge pedestal *and* a cube on a floor
//! button, so the gate is an AND (`two_buttons`); or a latching pedestal across a goo trench,
//! reached with a floor-level portal pair (`goo_crossing`). Every gadget needs at least one
//! portal trip. Every sender latches except `timed`'s pedestal, whose gate opens only for a
//! dash: the player must be through it before the timer runs out. A stage's button(s) gate the
//! next stage (or the exit).
//!
//! # Layouts
//! Two constructions are available (`--layout`):
//!
//! - `hall` (the default): one **hall** along +x with a sequence of **pockets** hanging off
//!   both sides. Each pocket is a room holding one gadget, sealed by a deployed panel at its
//!   mouth. Pocket `k`'s gadget folds pocket `k + 1`'s gate; the last folds the exit. Every
//!   pocket opens onto the *same* hall, so the forward plan crosses shared space between every
//!   stage instead of walking a fresh room. Pocket pairs are laid out zig-zag (the first pair
//!   at the far end, the next at the near end, ...), so the plan doubles back on itself.
//! - `line`: the original row of rooms along +x, one gadget each, gates in the walls between
//!   them. Kept for comparison.
//!
//! Both grow goal-first (place the exit and its gate, then prepend the stage before it, ...),
//! which is the reverse-play direction: the forward plan is the construction read backwards.
//! Difficulty is a dial (`--rooms`): the minimum plan length grows with the stage count. Big
//! chambers have reachable state spaces far larger than any budget worth paying, so the solver
//! often cannot finish them — but solvability is a property of the construction, not of the
//! search, so such a level is still labelled `solvable` (at `constructed_len`) rather than
//! `uncertain`; see `dataset::label_known`. `min_len` and `metrics` are filled in only when the
//! search *did* finish, i.e. for the smaller chambers.
//!
//! # Output
//! One JSON object per line, the same schema `glados dataset` writes (`tokens`, `n_tokens`,
//! `size`, `verdict`, and for solvable levels `min_len`, `needs_portals`, `metrics`), plus
//! `"src":"reverse"`, `"layout"`, `"constructed_len"` (actions the construction spends; an upper
//! bound on the true minimum, and the length `train/data.py` conditions on), `"rooms"`,
//! `"gates"`, and `"gadget"` (the per-stage gadget names). `train/data.py` reads it as-is; the
//! extra keys are ignored.

use std::io::Write;
use std::path::PathBuf;

use glados::builder::Builder;
use glados::dataset::{self, Label};
use glados::geom::{Cell, Dir};
use glados::p2c::Puzzle;
use glados::rng::Rng;
use glados::tokens;

/// Chamber depth in y for the line layout (a room's perpendicular extent in a pocket).
const D: i32 = 6;
/// Chamber height in z; the floor (z=0) stays solid, z=1..=H-1 is open.
const H: i32 = 6;
/// Room width in x. Narrowed automatically when more rooms must fit the 32-cell limit.
const RW_MAX: i32 = 6;
/// Narrowest room that still holds a ledge and the two portal walls. Below this the solver
/// starts reusing an earlier room's portal, which cuts actions per room instead of adding them.
const RW_MIN: i32 = 4;
/// The doorway's y cell (the door is one cell wide and tall).
const DOOR_Y: i32 = 2;
/// The ledge sits at the back: y=LEDGE_Y..=LEDGE_Y+1.
const LEDGE_Y: i32 = 4;
/// The ledge is this solid tall; the player stands at z=LEDGE_Z.
const LEDGE_H: i32 = 2;
const LEDGE_Z: i32 = LEDGE_H + 1;

/// A pocket's extent across the hall (its local y); the gadget's ledge needs six cells.
const POCKET_W: i32 = D;
/// Most pockets that fit the 32-cell x limit: `xsize = slots * (POCKET_W + 1) - 1`.
const HALL_SLOTS_MAX: i32 = (tokens::MAX_NUM + 1) / (POCKET_W + 1);

/// Which construction to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layout {
    /// A row of rooms along +x, one gadget each (the original).
    Line,
    /// One hall with pockets hanging off both sides.
    Hall,
}

/// A pocket's local frame: local +x runs from the front (door side) to the back wall, local +y
/// across it, local +z up. `origin` is the world cell of local (0, 0, 0); `u` and `v` are the
/// world directions of local +x and +y. This lets one gadget geometry be dropped into a pocket
/// facing any of the four horizontal ways without rewriting it.
#[derive(Clone, Copy)]
struct Frame {
    origin: Cell,
    u: Dir,
    v: Dir,
}

impl Frame {
    /// The identity frame of a room in the line layout, its front (door side) toward -x.
    fn line(s: i32) -> Frame {
        Frame {
            origin: [s, 0, 0],
            u: Dir::PosX,
            v: Dir::PosY,
        }
    }

    /// The world cell of a local cell.
    fn at(&self, local: Cell) -> Cell {
        let (uu, vv) = (self.u.vector(), self.v.vector());
        [
            self.origin[0] + uu[0] * local[0] + vv[0] * local[1],
            self.origin[1] + uu[1] * local[0] + vv[1] * local[1],
            self.origin[2] + local[2],
        ]
    }

    /// The world direction a local axis direction points.
    fn dir(&self, d: Dir) -> Dir {
        match d {
            Dir::PosX => self.u,
            Dir::NegX => self.u.opposite(),
            Dir::PosY => self.v,
            Dir::NegY => self.v.opposite(),
            Dir::PosZ => Dir::PosZ,
            Dir::NegZ => Dir::NegZ,
        }
    }

    /// The axis-aligned world box between two local corners (in either order).
    fn box_at(&self, a: Cell, b: Cell) -> (Cell, Cell) {
        let (p, q) = (self.at(a), self.at(b));
        (
            std::array::from_fn(|i| p[i].min(q[i])),
            std::array::from_fn(|i| p[i].max(q[i])),
        )
    }
}

/// The first x of room `k`; rooms are `rw` wide with a one-cell wall between them.
fn room_start(k: i32, rw: i32) -> i32 {
    k * (rw + 1)
}

/// The widest a room may be so `rooms` of them (plus walls) still fit the 32-cell token limit.
/// Returns `None` if even the narrowest room won't fit.
fn fit_width(rooms: i32) -> Option<i32> {
    if rooms < 1 {
        return None;
    }
    // rooms*(rw+1) - 1 <= 32  =>  rw <= 32/rooms - 1.
    let rw = (tokens::MAX_NUM / rooms - 1).min(RW_MAX);
    (rw >= RW_MIN).then_some(rw)
}

/// x-slots in zig-zag order from the far end inward: `[n-1, 0, n-2, 1, ...]`. Consecutive stage
/// pairs share a slot (one pocket per side), so the plan runs to one end, then the other.
fn zigzag(n: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(n);
    let (mut lo, mut hi) = (0usize, n.saturating_sub(1));
    while out.len() < n {
        if lo > hi {
            break;
        }
        if lo == hi {
            out.push(lo);
            break;
        }
        out.push(hi);
        out.push(lo);
        hi -= 1;
        lo += 1;
    }
    out
}

/// How a stage's button is pressed; all end with the player standing on the ledge (z=LEDGE_Z),
/// except `GooCrossing`, which crosses a goo trench on foot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Gadget {
    /// A latching pedestal on the ledge.
    Latch,
    /// A timed pedestal on the ledge: press it and drop to the doorway before it re-arms.
    Timed,
    /// A floor button on the ledge, fed by a cube carried through the portal.
    CubeButton,
    /// A pedestal on the ledge AND a cube on a floor button; the gate needs both (AND).
    TwoButtons,
    /// A latching pedestal across a two-cell goo trench: portal over the goo, then press.
    GooCrossing,
}

impl Gadget {
    /// All gadgets, in a stable order, for the width-dependent menu.
    const ALL: [Gadget; 5] = [
        Gadget::Latch,
        Gadget::Timed,
        Gadget::CubeButton,
        Gadget::TwoButtons,
        Gadget::GooCrossing,
    ];
    /// Gadgets that fit a narrow room (the goo trench needs a longer room).
    const NARROW: [Gadget; 4] = [
        Gadget::Latch,
        Gadget::Timed,
        Gadget::CubeButton,
        Gadget::TwoButtons,
    ];

    /// Actions the construction itself spends on this stage, ending at its gate (or the exit);
    /// an upper bound on its cost. Walking on through an open gate costs one more, except after
    /// `Timed`, whose dash ends past the gate (see `construct`).
    fn actions(self) -> usize {
        match self {
            Gadget::Latch => 5,
            Gadget::Timed => 4,
            Gadget::CubeButton => 6,
            Gadget::TwoButtons => 7,
            Gadget::GooCrossing => 4,
        }
    }

    /// The gadgets worth drawing from at room width `rw`.
    fn menu(rw: i32) -> &'static [Gadget] {
        if rw >= 5 {
            &Gadget::ALL
        } else {
            &Gadget::NARROW
        }
    }

    /// The menu for a hall pocket: `menu` without `Timed`. A timed pedestal's gate re-arms after
    /// the dash, but a hall pocket's gate leads back into the shared hall the player must cross
    /// again to reach the exit, so the closing gate seals them out (and only sometimes can they
    /// escape through a leftover portal). Every other gadget latches its gate open, so the player
    /// can always come back out.
    fn hall_menu(rw: i32) -> &'static [Gadget] {
        const ALL: [Gadget; 4] = [
            Gadget::Latch,
            Gadget::CubeButton,
            Gadget::TwoButtons,
            Gadget::GooCrossing,
        ];
        const NARROW: [Gadget; 3] = [Gadget::Latch, Gadget::CubeButton, Gadget::TwoButtons];
        if rw >= 5 {
            &ALL
        } else {
            &NARROW
        }
    }

    /// Short name for the JSON output.
    fn name(self) -> &'static str {
        match self {
            Gadget::Latch => "latch",
            Gadget::Timed => "timed",
            Gadget::CubeButton => "cube_button",
            Gadget::TwoButtons => "two_buttons",
            Gadget::GooCrossing => "goo_crossing",
        }
    }
}

/// Seconds a `Timed` pedestal stays on. The dash (drop off the ledge, walk to the doorway) is
/// about 3 s, so this leaves room for the gate panel's fold animation, which the solver doesn't
/// model; short enough that the dash is still the point of the stage. (The hall layout drops
/// `Timed`: its one-way gate would seal the player out of the shared hall they must return
/// through. See `Gadget::hall_menu`.)
const TIMED_SECONDS: u32 = 8;

/// The ledge (a balcony at the back) and the two walls a portal pair needs: a white wall at
/// floor level on the near side and a white wall at ledge level on the far side. Built in the
/// frame's local coordinates, so the door can face any horizontal way.
fn ledge_walls(b: &mut Builder, f: &Frame, rw: i32) {
    let e = rw - 1;
    let (lo, hi) = f.box_at([e - 1, LEDGE_Y, 1], [e, LEDGE_Y + 1, LEDGE_H]);
    b.fill(lo, hi);
    let (lo, hi) = f.box_at([0, LEDGE_Y, 1], [0, LEDGE_Y + 1, LEDGE_H]);
    b.paint(lo, hi, f.dir(Dir::NegX), true);
    let (lo, hi) = f.box_at([e, LEDGE_Y, LEDGE_Z], [e, LEDGE_Y + 1, LEDGE_Z + 1]);
    b.paint(lo, hi, f.dir(Dir::PosX), true);
}

/// Realises one stage's gadget in the frame's local x range `0..rw` and returns the senders that
/// gate the next stage (more than one for `TwoButtons`, so the gate is an AND). The pocket is
/// already carved. `entry_high` is true when the player enters at local x = `rw` (a hall pocket
/// off its back wall) rather than at local x = 0 (a line room, entered from the front).
fn build_room(b: &mut Builder, f: &Frame, rw: i32, gadget: Gadget, entry_high: bool) -> Vec<i32> {
    let e = rw - 1;
    let at = |c: Cell| f.at(c);
    match gadget {
        Gadget::Latch => {
            ledge_walls(b, f, rw);
            vec![b.pedestal(at([e - 1, LEDGE_Y, LEDGE_Z]), None)]
        }
        Gadget::Timed => {
            ledge_walls(b, f, rw);
            vec![b.pedestal(at([e - 1, LEDGE_Y, LEDGE_Z]), Some(TIMED_SECONDS))]
        }
        Gadget::CubeButton => {
            ledge_walls(b, f, rw);
            b.cube_dropper(at([1, LEDGE_Y, H - 1]));
            vec![b.floor_button(at([e - 1, LEDGE_Y, LEDGE_Z]))]
        }
        Gadget::TwoButtons => {
            ledge_walls(b, f, rw);
            b.cube_dropper(at([1, 1, H - 1]));
            let button = b.floor_button(at([1, DOOR_Y, 1]));
            let pedestal = b.pedestal(at([e - 1, LEDGE_Y, LEDGE_Z]), None);
            vec![button, pedestal]
        }
        Gadget::GooCrossing => {
            // A goo trench two cells wide (one cell could be jumped) across the room; white
            // walls on both sides at floor level (the trench's own cells are left solid so they
            // stay surfaces), and the pedestal beyond. The trench goes on the entrance side, so
            // the player must portal over it to reach the pedestal: at local x = 1 in a line
            // room (entered from the front), or against the mouth in a hall pocket (entered from
            // the back).
            let (trench, ped_x) = if entry_high { (rw - 3, 1) } else { (1, e - 1) };
            let (lo, hi) = f.box_at([trench, 0, 0], [trench + 1, D - 1, 0]);
            b.carve(lo, hi);
            b.goo(at([trench, DOOR_Y, 0]));
            for (lo_c, hi_c) in [([0, 0, 1], [0, 1, 2]), ([0, 4, 1], [0, 5, 2])] {
                let (lo, hi) = f.box_at(lo_c, hi_c);
                b.paint(lo, hi, f.dir(Dir::NegX), true);
            }
            for (lo_c, hi_c) in [([e, 0, 1], [e, 1, 2]), ([e, 4, 1], [e, 5, 2])] {
                let (lo, hi) = f.box_at(lo_c, hi_c);
                b.paint(lo, hi, f.dir(Dir::PosX), true);
            }
            // A latching pedestal (a bare floor button would release as the player walks on
            // to the doorway, and there is no cube here to hold it down).
            vec![b.pedestal(at([ped_x, DOOR_Y, 1]), None)]
        }
    }
}

/// A constructed chamber and what the construction claims about it.
struct Built {
    puzzle: Puzzle,
    size: [i32; 3],
    /// Actions the construction itself spends, from the gadget costs. An upper bound on the
    /// true minimum: the solver often finds a shorter route (e.g. reusing an earlier portal).
    constructed: usize,
    gadgets: Vec<Gadget>,
    layout: &'static str,
}

impl Built {
    /// One gate between each pair of adjacent stages.
    fn gates(&self) -> usize {
        self.gadgets.len().saturating_sub(1)
    }
}

/// Builds a chamber of `rooms` stages (>= 1) from `seed` in the chosen layout.
fn construct(seed: u64, rooms: i32, layout: Layout) -> Built {
    assert!(rooms >= 1);
    match layout {
        Layout::Line => construct_line(seed, rooms),
        Layout::Hall => construct_hall(seed, rooms),
    }
}

/// The widest row of `rooms` rooms that fits, then `construct_rw`.
fn construct_line(seed: u64, rooms: i32) -> Built {
    let rw = fit_width(rooms).expect("too many rooms for the 32-cell limit");
    construct_rw(seed, rooms, rw)
}

/// `construct_line` with room width `rw` (RW_MIN..=RW_MAX) instead of the widest that fits.
fn construct_rw(seed: u64, rooms: i32, rw: i32) -> Built {
    assert!(rooms >= 1 && (RW_MIN..=RW_MAX).contains(&rw));
    let mut rng = Rng::new(seed);
    let width = (rooms - 1) * (rw + 1) + rw;
    let mut b = Builder::new(&format!("Reverse {seed}"), [width, D, H]);

    // Carve the rooms.
    for k in 0..rooms {
        let f = Frame::line(room_start(k, rw));
        let (lo, hi) = f.box_at([0, 0, 1], [rw - 1, D - 1, H - 1]);
        b.carve(lo, hi);
    }
    // One-cell doorways through the walls between rooms; a deployed panel fills each.
    for k in 0..rooms - 1 {
        let sep = room_start(k, rw) + rw;
        b.carve([sep, DOOR_Y, 1], [sep, DOOR_Y, 1]);
    }

    let mut gadgets = Vec::new();
    let mut constructed = 0;
    let mut senders: Vec<Vec<i32>> = Vec::new();
    for k in 0..rooms {
        let f = Frame::line(room_start(k, rw));
        let gadget = *rng.pick(Gadget::menu(rw));
        let room_senders = build_room(&mut b, &f, rw, gadget, false);
        constructed += gadget.actions();
        if k < rooms - 1 && gadget != Gadget::Timed {
            constructed += 1;
        }
        gadgets.push(gadget);

        // The gate to the next room: a deployed panel filling the doorway, wired to this
        // room's button(s) so it folds (and the way opens) only once they are pressed.
        if k < rooms - 1 {
            let sep = room_start(k, rw) + rw;
            let panel = b.angled_panel(
                [sep, DOOR_Y, 1],
                Dir::PosZ,
                f.dir(Dir::PosX),
                90,
                true,
                false,
                false,
            );
            for &sender in &room_senders {
                b.connect(sender, panel);
            }
        }
        senders.push(room_senders);
    }

    b.entry([0, DOOR_Y, 1], Dir::PosX);
    let last = room_start(rooms - 1, rw) + rw - 1;
    let exit = b.exit([last, DOOR_Y, 1], Dir::NegX);
    // The last room's button(s) gate the exit itself.
    for &sender in senders.last().expect("at least one room") {
        b.connect(sender, exit);
    }

    Built {
        puzzle: b.build(),
        size: [width, D, H],
        constructed,
        gadgets,
        layout: "line",
    }
}

/// One hall along +x with pockets hanging off both sides, each pocket holding one gadget and
/// sealed by a deployed panel at its mouth. Pocket `k`'s gadget folds pocket `k + 1`'s gate; the
/// last folds the exit. Pockets are placed two per x-slot (one north, one south) in zig-zag slot
/// order, so the forward plan runs from end to end of the same hall.
fn construct_hall(seed: u64, stages: i32) -> Built {
    assert!(stages >= 1);
    let slots = (stages + 1) / 2;
    assert!(
        slots <= HALL_SLOTS_MAX,
        "too many stages for the 32-cell limit"
    );
    let xsize = slots * (POCKET_W + 1) - 1;
    // The hall runs at y = hy, one cell wide; pockets need RW_MAX cells either side of it.
    let hy = RW_MAX + 1;
    let ysize = 2 * (RW_MAX + 1) + 1;
    let mut rng = Rng::new(seed);
    let mut b = Builder::new(&format!("Reverse {seed}"), [xsize, ysize, H]);
    b.carve([0, hy, 1], [xsize - 1, hy, H - 1]);

    let order = zigzag(slots as usize);
    let mut gadgets = Vec::new();
    let mut senders: Vec<Vec<i32>> = Vec::new();
    let mut mouths: Vec<Cell> = Vec::new();
    let mut frames: Vec<Frame> = Vec::new();
    let mut constructed = 0usize;
    for k in 0..stages {
        let slot = order[(k / 2) as usize] as i32;
        let north = k % 2 == 0;
        let rw = rng.range(RW_MIN, RW_MAX);
        let sx0 = slot * (POCKET_W + 1);
        // The mouth is the back wall (where the ledge and the gate are): north pockets face
        // back toward -y, south pockets toward +y.
        let (origin, u) = if north {
            ([sx0, hy + 1 + rw, 0], Dir::NegY)
        } else {
            ([sx0, hy - 1 - rw, 0], Dir::PosY)
        };
        let f = Frame {
            origin,
            u,
            v: Dir::PosX,
        };
        let (lo, hi) = f.box_at([0, 0, 1], [rw - 1, D - 1, H - 1]);
        b.carve(lo, hi);
        let mouth = f.at([rw, DOOR_Y, 1]);
        b.carve(mouth, mouth);

        let gadget = *rng.pick(Gadget::hall_menu(rw));
        let room_senders = build_room(&mut b, &f, rw, gadget, true);
        constructed += gadget.actions();
        gadgets.push(gadget);
        senders.push(room_senders);
        mouths.push(mouth);
        frames.push(f);
    }

    // The gate to the next pocket: a deployed panel filling its mouth, wired to the previous
    // pocket's button(s). Pocket 0's mouth is left open.
    for k in 1..stages as usize {
        let panel = b.angled_panel(mouths[k], Dir::PosZ, frames[k].u, 90, true, false, false);
        for &sender in &senders[k - 1] {
            b.connect(sender, panel);
        }
        constructed += 1;
    }

    b.entry([0, hy, 1], Dir::PosX);
    let exit = b.exit([xsize - 1, hy, 1], Dir::NegX);
    // The last pocket's button(s) gate the exit itself.
    for &sender in senders.last().expect("at least one stage") {
        b.connect(sender, exit);
    }
    constructed += 1;

    Built {
        puzzle: b.build(),
        size: [xsize, ysize, H],
        constructed,
        gadgets,
        layout: "hall",
    }
}

/// Everything one generated level carries out of a worker thread: the label, the tokens and
/// the few `Built` fields the JSONL line needs. The puzzle itself is dropped here (only its
/// `--dump` text is kept), so results waiting to be written in order stay small.
struct Generated {
    dump: Option<String>,
    constructed: usize,
    size: [i32; 3],
    gates: usize,
    gadgets: Vec<Gadget>,
    layout: &'static str,
    tokens: Vec<String>,
    label: Label,
}

/// The JSONL body fields, plus the reverse-specific ones.
fn json_line(g: &Generated) -> String {
    let names: Vec<String> = g
        .gadgets
        .iter()
        .map(|g| format!("\"{}\"", g.name()))
        .collect();
    format!(
        "{{{},\"src\":\"reverse\",\"layout\":\"{}\",\"constructed_len\":{},\"rooms\":{},\"gates\":{},\"gadget\":[{}]}}",
        dataset::json_body(&g.tokens, g.size, &g.label),
        g.layout,
        g.constructed,
        g.gadgets.len(),
        g.gates,
        names.join(","),
    )
}

struct Args {
    seed: u64,
    count: usize,
    rooms: i32,
    rooms_max: Option<i32>,
    budget: usize,
    threads: usize,
    out: Option<PathBuf>,
    dump: Option<PathBuf>,
    require_solvable: bool,
    layout: Layout,
}

const USAGE: &str = "usage: reverse [options]   (cargo run --release --bin reverse -- ...)
  --seed N               base seed (default 0)
  --count N              levels to build (default 100)
  --rooms N              fewest stages per chamber, >= 1 (default 3); the difficulty dial
  --rooms-max N          most stages per chamber (default: same as --rooms); each level draws a
                         count in [--rooms, --rooms-max]
  --layout L             `hall` (default): one hall with pockets off both sides; or `line`: a
                         row of rooms
  --budget N             solver state budget for labelling; a chamber the search can't finish
                         inside it is labelled `solvable` from the construction (default 50000)
  --threads N            levels to label at once (default: all cores)
  --out FILE             write JSONL (default: stdout)
  --dump DIR             also write each chamber as DIR/reverse-<seed>-<i>.p2c
  --require-solvable     only emit chambers the solver proves solvable";

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args {
        seed: 0,
        count: 100,
        rooms: 3,
        rooms_max: None,
        budget: 50_000,
        threads: std::thread::available_parallelism().map_or(4, |n| n.get()),
        out: None,
        dump: None,
        require_solvable: false,
        layout: Layout::Hall,
    };
    let mut i = 0;
    let num = |v: Option<&String>, what: &str| -> Result<u64, String> {
        v.ok_or_else(|| format!("{what} needs a value"))?
            .parse()
            .map_err(|_| format!("{what}: not a number"))
    };
    // A stage count: 1..=32 (more never fit the 32-cell limit; `main` checks the exact bound).
    let rooms = |v: Option<&String>, what: &str| -> Result<i32, String> {
        match num(v, what)? {
            n @ 1..=32 => Ok(n as i32),
            n => Err(format!("{what} {n}: must be 1..=32")),
        }
    };
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                a.seed = num(args.get(i + 1), "--seed")?;
                i += 2;
            }
            "--count" => {
                a.count = num(args.get(i + 1), "--count")? as usize;
                i += 2;
            }
            "--rooms" => {
                a.rooms = rooms(args.get(i + 1), "--rooms")?;
                i += 2;
            }
            "--rooms-max" => {
                a.rooms_max = Some(rooms(args.get(i + 1), "--rooms-max")?);
                i += 2;
            }
            "--layout" => {
                a.layout = match args.get(i + 1).map(String::as_str) {
                    Some("line") => Layout::Line,
                    Some("hall") => Layout::Hall,
                    other => {
                        return Err(format!(
                            "--layout {other:?}: expected `line` or `hall`\n{USAGE}"
                        ))
                    }
                };
                i += 2;
            }
            "--budget" => {
                a.budget = num(args.get(i + 1), "--budget")? as usize;
                i += 2;
            }
            "--threads" => {
                a.threads = num(args.get(i + 1), "--threads")? as usize;
                i += 2;
            }
            "--out" => {
                a.out = Some(PathBuf::from(
                    args.get(i + 1).ok_or("--out needs a value")?,
                ));
                i += 2;
            }
            "--dump" => {
                a.dump = Some(PathBuf::from(
                    args.get(i + 1).ok_or("--dump needs a value")?,
                ));
                i += 2;
            }
            "--require-solvable" => {
                a.require_solvable = true;
                i += 1;
            }
            other => return Err(format!("unknown argument {other:?}\n{USAGE}")),
        }
    }
    Ok(a)
}

#[derive(Default)]
struct Stats {
    built: usize,
    encoded: usize,
    emitted: usize,
    solvable: usize,
    uncertain: usize,
    unsolvable: usize,
    invalid: usize,
    unsupported: usize,
    min_len_sum: usize,
}

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return std::process::ExitCode::SUCCESS;
    }
    let a = match parse(&args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return std::process::ExitCode::from(2);
        }
    };

    if let Some(dir) = &a.dump {
        if let Err(e) = std::fs::create_dir_all(dir) {
            eprintln!("--dump {}: {e}", dir.display());
            return std::process::ExitCode::FAILURE;
        }
    }
    let rooms_max = a.rooms_max.unwrap_or(a.rooms);
    let (rmin, rmax) = (a.rooms.min(rooms_max), a.rooms.max(rooms_max));
    match a.layout {
        Layout::Hall => {
            let max = 2 * HALL_SLOTS_MAX;
            if rmax > max {
                eprintln!("--rooms-max {rmax}: too many stages for the 32-cell limit (max {max})");
                return std::process::ExitCode::from(2);
            }
            let slots = (rmax + 1) / 2;
            eprintln!(
                "reverse: hall, {rmin}..{rmax} stage(s), up to {slots} slot(s), chambers up to {}x{}",
                slots * (POCKET_W + 1) - 1,
                2 * (RW_MAX + 1) + 1
            );
        }
        Layout::Line => match fit_width(rmax) {
            Some(rw) => eprintln!(
                "reverse: line, {rmin}..{rmax} room(s), room width {rw}, chambers up to {} wide",
                (rmax * (rw + 1)) - 1
            ),
            None => {
                let max = (1..=32).rev().find(|r| fit_width(*r).is_some()).unwrap_or(1);
                eprintln!("--rooms-max {rmax}: too many for the 32-cell limit (max {max})");
                return std::process::ExitCode::from(2);
            }
        },
    }
    let mut out: Box<dyn Write> = match &a.out {
        Some(path) => match std::fs::File::create(path) {
            Ok(f) => Box::new(std::io::BufWriter::new(f)),
            Err(e) => {
                eprintln!("{}: {e}", path.display());
                return std::process::ExitCode::FAILURE;
            }
        },
        None => Box::new(std::io::stdout().lock()),
    };

    let mut stats = Stats::default();
    // Levels are independent, and labelling (the solver) is nearly all of the time, so label
    // them on every core and write the lines in input order: the output is the same whatever
    // `threads` is.
    let done: Result<(), std::convert::Infallible> = dataset::par_map_ordered(
        (0..a.count).map(|i| (i, a.seed.wrapping_add(i as u64))),
        a.threads,
        |(i, seed)| {
            let rooms = Rng::new(seed ^ 0x51_ed_0000).range(rmin, rmax);
            let built = construct(seed, rooms, a.layout);
            let made = match tokens::encode(&built.puzzle) {
                Ok(tokens) => {
                    let label = dataset::label_known(
                        &built.puzzle,
                        a.budget,
                        seed,
                        true,
                        Some(built.constructed),
                    );
                    let dump = a.dump.as_ref().map(|_| built.puzzle.to_p2c());
                    Ok(Generated {
                        dump,
                        constructed: built.constructed,
                        size: built.size,
                        gates: built.gates(),
                        gadgets: built.gadgets,
                        layout: built.layout,
                        tokens,
                        label,
                    })
                }
                Err(e) => Err(e),
            };
            (i, seed, made)
        },
        |(i, seed, made)| {
            stats.built += 1;
            let g = match made {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("seed {seed}: does not encode: {e}");
                    return Ok(());
                }
            };
            stats.encoded += 1;
            match &g.label {
                Label::Solvable { min_len, .. } => {
                    stats.solvable += 1;
                    stats.min_len_sum += *min_len;
                }
                Label::Uncertain(_) => stats.uncertain += 1,
                Label::Unsolvable => stats.unsolvable += 1,
                Label::Invalid(_) => stats.invalid += 1,
                Label::Unsupported(_) => stats.unsupported += 1,
            }
            if a.require_solvable && !matches!(g.label, Label::Solvable { .. }) {
                return Ok(());
            }
            if let (Some(dir), Some(text)) = (&a.dump, &g.dump) {
                let path = dir.join(format!("reverse-{seed}-{i}.p2c"));
                if let Err(e) = std::fs::write(&path, text) {
                    eprintln!("{}: {e}", path.display());
                }
            }
            writeln!(out, "{}", json_line(&g)).expect("write output");
            stats.emitted += 1;
            Ok(())
        },
    );
    match done {
        Ok(()) => {}
        Err(never) => match never {},
    }
    out.flush().ok();

    let mean = |n: usize, d: usize| if d == 0 { 0.0 } else { n as f64 / d as f64 };
    eprintln!(
        "reverse: built {} ({:.0}% encoded), emitted {}\n  \
         solvable {} ({:.0}%), uncertain {} ({:.0}%), unsolvable {}, invalid {}, unsupported {}\n  \
         mean min_len {:.1} (over solvable)",
        stats.built,
        100.0 * mean(stats.encoded, stats.built),
        stats.emitted,
        stats.solvable,
        100.0 * mean(stats.solvable, stats.encoded),
        stats.uncertain,
        100.0 * mean(stats.uncertain, stats.encoded),
        stats.unsolvable,
        stats.invalid,
        stats.unsupported,
        mean(stats.min_len_sum, stats.solvable),
    );
    std::process::ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    const LAYOUTS: [Layout; 2] = [Layout::Line, Layout::Hall];

    /// The whole point: a constructed chamber is solvable, and the solver agrees.
    #[test]
    fn constructed_chambers_are_solvable() {
        for layout in LAYOUTS {
            for seed in 0..16 {
                let built = construct(seed, 2, layout);
                tokens::encode(&built.puzzle).expect("a constructed chamber encodes");
                match dataset::label(&built.puzzle, 300_000, seed) {
                    Label::Solvable { min_len, .. } => assert!(min_len > 0),
                    other => {
                        panic!("{layout:?} seed {seed}: expected solvable, got {other:?}")
                    }
                }
            }
        }
    }

    /// A chamber too big to solve inside the budget is still labelled `solvable`, not
    /// `uncertain`: the construction knows the plan, so `constructed_len` stands in. A budget of
    /// one state forces the search to give up before it reaches any goal.
    #[test]
    fn big_chambers_are_labelled_solvable() {
        for layout in LAYOUTS {
            for seed in 0..4 {
                let built = construct(seed, 6, layout);
                tokens::encode(&built.puzzle).expect("encodes");
                match dataset::label_known(&built.puzzle, 1, seed, true, Some(built.constructed)) {
                    Label::Solvable {
                        min_len,
                        needs_portals,
                        metrics,
                    } => {
                        assert_eq!(min_len, built.constructed);
                        assert!(needs_portals);
                        assert!(metrics.is_none());
                    }
                    other => {
                        panic!("{layout:?} seed {seed}: expected solvable, got {other:?}")
                    }
                }
            }
        }
    }

    /// More stages must mean a longer plan, and the gate count must match the stage count.
    #[test]
    fn difficulty_grows_with_rooms() {
        for layout in LAYOUTS {
            let small = construct(3, 1, layout);
            let big = construct(3, 4, layout);
            assert_eq!(small.gates(), 0);
            assert_eq!(big.gates(), 3);
            let label = |b: &Built| match dataset::label(&b.puzzle, 500_000, 3) {
                Label::Solvable { min_len, .. } => min_len,
                other => panic!("{layout:?}: {other:?}"),
            };
            let (l_small, l_big) = (label(&small), label(&big));
            assert!(
                l_big > l_small,
                "{layout:?}: 4 stages ({l_big}) should beat 1 stage ({l_small})"
            );
        }
    }

    /// Every stage's button(s) wire to exactly one gate (the next stage's mouth, or the exit in
    /// the last stage): each sender has one `>`, every gate panel gets one, and the exit gets
    /// the last stage's.
    #[test]
    fn gates_are_wired_in_series() {
        for layout in LAYOUTS {
            for seed in 0..8 {
                for stages in 1..=4 {
                    let built = construct(seed, stages, layout);
                    assert_eq!(built.gadgets.len(), stages as usize);
                    assert_eq!(built.gates(), (stages - 1) as usize);
                    let toks = tokens::encode(&built.puzzle).expect("encodes");
                    let senders = |g: &Gadget| if *g == Gadget::TwoButtons { 2 } else { 1 };
                    let total: usize = built.gadgets.iter().map(senders).sum();
                    let last = senders(built.gadgets.last().unwrap());
                    let arrows: Vec<usize> =
                        (0..toks.len()).filter(|&i| toks[i] == ">").collect();
                    let ctx = format!("{layout:?} seed={seed} stages={stages}");
                    assert_eq!(arrows.len(), total, "{ctx}: one wire per sender");
                    let to_exit = arrows.iter().filter(|&&i| toks[i + 1] == "exit").count();
                    assert_eq!(to_exit, last, "{ctx}: the last stage gates the exit");
                    let panels: std::collections::BTreeSet<&str> = arrows
                        .iter()
                        .filter(|&&i| toks[i + 1] == "panel")
                        .map(|&i| toks[i + 2].as_str())
                        .collect();
                    assert_eq!(
                        panels.len(),
                        (stages - 1) as usize,
                        "{ctx}: every gate panel is wired"
                    );
                }
            }
        }
    }

    /// The hall is genuinely re-traversed: the zig-zag layout makes the forward plan run back
    /// toward the entry between stages, so the solver's `backtrack` is positive.
    #[test]
    fn hall_revisits_the_shared_space() {
        for seed in 0..6 {
            let built = construct(seed, 4, Layout::Hall);
            match dataset::label(&built.puzzle, 500_000, seed) {
                Label::Solvable {
                    metrics: Some(m), ..
                } => assert!(m.backtrack > 0.0, "seed {seed}: the plan never doubles back"),
                other => panic!("seed {seed}: {other:?}"),
            }
        }
    }

    /// A hall chamber is smaller across than the line of the same stage count, because its
    /// pockets share a footprint rather than extending the corridor.
    #[test]
    fn hall_is_compact() {
        let hall = construct(0, 6, Layout::Hall);
        let line = construct(0, 6, Layout::Line);
        assert!(
            hall.size[0] < line.size[0],
            "hall {} isn't narrower than line {}",
            hall.size[0],
            line.size[0]
        );
    }

    /// Every gadget kind is reachable at width `rw`, every chamber is solvable, and the
    /// construction's own plan length bounds the solver's minimum.
    fn check_solvable(rw: i32, seeds: u64) {
        let mut seen = std::collections::HashSet::new();
        for seed in 0..seeds {
            let built = construct_rw(seed, 2, rw);
            seen.extend(built.gadgets.iter().copied());
            tokens::encode(&built.puzzle).expect("encodes");
            match dataset::label(&built.puzzle, 300_000, seed) {
                Label::Solvable { min_len, .. } => {
                    assert!(
                        min_len <= built.constructed,
                        "rw={rw} seed {seed} {:?}: min_len {min_len} > constructed {}",
                        built.gadgets,
                        built.constructed
                    );
                    // The gates are shut, so no shot reaches the last room from the first:
                    // the earlier rooms cost something too.
                    let last = built.gadgets.last().unwrap().actions();
                    assert!(
                        min_len > last,
                        "rw={rw} seed {seed} {:?}: min_len {min_len} skips the earlier rooms",
                        built.gadgets
                    );
                }
                other => panic!("rw={rw} seed {seed} {:?}: {other:?}", built.gadgets),
            }
        }
        for g in Gadget::menu(rw) {
            assert!(seen.contains(g), "rw={rw}: gadget {g:?} was never drawn");
        }
    }

    #[test]
    fn every_gadget_is_solvable() {
        check_solvable(RW_MAX, 40);
    }

    /// The narrow rooms that 5- and 6-room chambers get.
    #[test]
    fn narrow_rooms_are_solvable() {
        check_solvable(RW_MIN, 40);
        check_solvable(RW_MIN + 1, 40);
    }

    /// Every gadget is reachable in a hall pocket too, and its plan fits the construction bound.
    #[test]
    fn every_gadget_is_solvable_in_a_hall() {
        let mut seen = std::collections::HashSet::new();
        for seed in 0..24 {
            let built = construct(seed, 4, Layout::Hall);
            seen.extend(built.gadgets.iter().copied());
            tokens::encode(&built.puzzle).expect("encodes");
            match dataset::label(&built.puzzle, 300_000, seed) {
                Label::Solvable { min_len, .. } => assert!(
                    min_len <= built.constructed,
                    "seed {seed} {:?}: min_len {min_len} > constructed {}",
                    built.gadgets,
                    built.constructed
                ),
                other => panic!("seed {seed} {:?}: {other:?}", built.gadgets),
            }
        }
        for g in Gadget::hall_menu(RW_MAX) {
            assert!(seen.contains(g), "hall: gadget {g:?} was never drawn");
        }
    }
}
