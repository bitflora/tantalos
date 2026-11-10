//! The legal next tokens of a level, for writing levels token by token that `tokens::decode`
//! reads and whose items are mounted on something (the level net's constrained decoding; Goblin's
//! is `env.rs`).
//!
//! `next` reads a level's tokens so far and says which tokens may come next, and whether the
//! level may end there. It follows `tokens::decode` but only its box format, in the order
//! `tokens::encode` writes (which every training level is in): `size`, `open` boxes, optional
//! `goo` boxes, `white`/`black` per direction in the order `+x -x +y -y +z -z`, then the items a
//! section at a time (entry, exit, glass/grating, fields, gels, droppers, buttons, pedestals,
//! turrets, cubes, lasers, catchers, relays, plates, bridges, funnels, panels, glass panels,
//! flips, stairs, pistons, rails). Within a section items may come in any order; a sender's wires
//! come sorted, each once, as `encode` writes them.
//!
//! On top of what `decode` checks:
//! - numbers stay in the chamber, a box's far corner isn't below its near one;
//! - a wire's target exists: one in an earlier section by its count, one in a later section
//!   (a button wired to a panel) is a debt that section must pay before the level goes past it;
//! - one entry and one exit, both before the level ends, on walls;
//! - mounted items (`Level::mount_problems`) stand in an open cell with something solid behind
//!   them; a glass panel's box is flat and hangs from something; stairs' second cell is open
//!   floor; a piston's column is open; a field stays in the chamber; at most `MAX_TURRETS`
//!   turrets. Every token allowed can be followed by one that is, so a level never gets stuck.
//!
//! With the editor's rules on (`Grammar::with_editor_rules`, what `glados-score grammar` serves),
//! the token that settles where an item goes (its last cell number or direction, a field's
//! length, a piston's reach, a rail's far end) may not give it a new flag in our model of the
//! Puzzle Maker's rules (`editor.rs`): two items in one cell, an item on a door frame or in a
//! field, fields or panels overlapping. The level so far, the item finished with any legal
//! tokens and its wires left out, is decoded and its flags compared with the level's before the
//! item. Only the last token is filtered, never the ones already written; when every choice
//! would be flagged they all stay, so a level still never gets stuck.
//!
//! Not checked: anything else between items (a laser hitting nothing, a door's floor, which
//! may be a panel or bridge written after it), and goo.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::rc::Rc;

use crate::editor::{editor_flags, target_name, Flag};
use crate::level::{ItemKind, Level};

use crate::geom::{step, Cell, Dir};
use crate::level::MAX_TURRETS;
use crate::tokens::{dir_token, BUTTON_KINDS, CUBE_KINDS, MAX_NUM, PAINT_KINDS};

const NUMS: [&str; 33] = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16",
    "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32",
];
/// Directions in the order `encode` paints them.
const DIRS: [Dir; 6] = [Dir::PosX, Dir::NegX, Dir::PosY, Dir::NegY, Dir::PosZ, Dir::NegZ];
/// Item words by section, in `encode`'s order (a section may have two words).
const SECTIONS: [&[&str]; 22] = [
    &["entry"],
    &["exit"],
    &["glass", "grating"],
    &["fizzler", "laserfield"],
    &["gel"],
    &["dropper"],
    &["button"],
    &["pedestal"],
    &["turret"],
    &["cube"],
    &["laser"],
    &["catcher"],
    &["relay"],
    &["plate"],
    &["bridge"],
    &["funnel"],
    &["panel"],
    &["glasspanel"],
    &["flip"],
    &["stairs"],
    &["piston"],
    &["rail"],
];
/// Wire targets in `encode`'s order, with the section whose items they count (polarity: funnels).
const TARGETS: [(&str, usize); 14] = [
    ("exit", 1),
    ("field", 3),
    ("gel", 4),
    ("laser", 10),
    ("bridge", 14),
    ("funnel", 15),
    ("polarity", 15),
    ("panel", 16),
    ("glasspanel", 17),
    ("flip", 18),
    ("stairs", 19),
    ("piston", 20),
    ("rail", 21),
    ("dropper", 5),
];
/// Geometries kept between calls (they're cleared all at once past this).
const MAX_CACHED: usize = 4096;

/// What may follow a level's tokens so far: tokens (sorted), and whether it may end there.
#[derive(Debug, PartialEq)]
pub struct Next {
    pub allowed: Vec<&'static str>,
    pub end: bool,
}

/// The chamber's solid cells and where items can be mounted, once the geometry is read.
struct Geo {
    size: [i32; 3],
    solid: Vec<bool>,
    /// Open cells with something solid behind them against each direction (`DIRS` order).
    faces: [Vec<Cell>; 6],
    /// Every (cell, direction) of `faces`.
    all_faces: Vec<(Cell, Dir)>,
    /// The `all_faces` on a wall, where doors go. (Not only over a floor: what a door stands on
    /// may be a glass panel or a bridge written later.)
    doors: Vec<(Cell, Dir)>,
    /// Floor cells and a direction along which the next cell is open floor too.
    stairs: Vec<(Cell, Dir)>,
    /// Where a glass or grating panel's box can start (`glass_starts`), once asked.
    glass: std::cell::OnceCell<Vec<Cell>>,
    /// `pistons`, once asked.
    piston: std::cell::OnceCell<Vec<(Cell, Dir)>>,
}

impl Geo {
    fn new(size: [i32; 3], boxes: &[(Cell, Cell)]) -> Geo {
        let n = (size[0] * size[1] * size[2]) as usize;
        let mut g = Geo {
            size,
            solid: vec![true; n],
            faces: Default::default(),
            all_faces: Vec::new(),
            doors: Vec::new(),
            stairs: Vec::new(),
            glass: Default::default(),
            piston: Default::default(),
        };
        for &(lo, hi) in boxes {
            for c in crate::level::cells_between(lo, hi) {
                let i = g.idx(c).expect("boxes are in the chamber");
                g.solid[i] = false;
            }
        }
        for c in crate::level::cells_between([0, 0, 0], [size[0] - 1, size[1] - 1, size[2] - 1]) {
            if g.solid(c) {
                continue;
            }
            for (k, d) in DIRS.into_iter().enumerate() {
                if g.solid(step(c, d.opposite())) {
                    g.faces[k].push(c);
                    g.all_faces.push((c, d));
                    if d.axis() != 2 {
                        g.doors.push((c, d));
                    }
                }
            }
            if g.solid(step(c, Dir::NegZ)) {
                for d in Dir::HORIZONTAL {
                    let s = step(c, d);
                    if !g.solid(s) && g.solid(step(s, Dir::NegZ)) {
                        g.stairs.push((c, d));
                    }
                }
            }
        }
        g
    }

    fn idx(&self, c: Cell) -> Option<usize> {
        let [sx, sy, sz] = self.size;
        ((0..sx).contains(&c[0]) && (0..sy).contains(&c[1]) && (0..sz).contains(&c[2]))
            .then(|| ((c[2] * sy + c[1]) * sx + c[0]) as usize)
    }

    /// Outside the chamber is solid.
    fn solid(&self, c: Cell) -> bool {
        self.idx(c).is_none_or(|i| self.solid[i])
    }

    /// The `all_faces` a piston fits on: on a floor, with room to stand on it one cell up.
    fn pistons(&self) -> &[(Cell, Dir)] {
        self.piston.get_or_init(|| {
            let fits = |&&(c, n): &&(Cell, Dir)| n != Dir::PosZ || !self.solid(step(c, Dir::PosZ));
            self.all_faces.iter().filter(fits).copied().collect()
        })
    }

    fn glass_starts(&self) -> &[Cell] {
        self.glass.get_or_init(|| glass_starts(self))
    }

    fn floor(&self) -> &[Cell] {
        &self.faces[4]
    }

    fn ceiling(&self) -> &[Cell] {
        &self.faces[5]
    }
}

/// Geometries already worked out, by their tokens.
#[derive(Default)]
pub struct Grammar {
    geos: HashMap<String, Rc<Geo>>,
    /// Whether items may not add editor flags (`editor.rs`).
    editor: bool,
    /// Levels so far to try items in, by their tokens (None: they don't decode).
    probes: HashMap<String, Option<Rc<RefCell<Probe>>>>,
    /// Places tried for an item, by the tokens up to its word: whether it gets no editor flag there.
    tried: HashMap<String, HashMap<Place, bool>>,
}

/// Levels kept to try items in (each holds its chamber: fewer than `MAX_CACHED`).
const MAX_PROBES: usize = 256;

/// A level so far, to try an item in: its editor flags without it.
struct Probe {
    level: Level,
    flags: BTreeSet<Flag>,
}

impl Probe {
    /// Whether the item `word` with the tokens `toks` gets no editor flag in the level. The item
    /// is decoded on its own in an open chamber (over a floor at its height, for stairs) and
    /// added to the level with what `Level` works out of it (a field's faces, a glass panel's),
    /// so that reads the open chamber, not the level's. A faith plate's target is asked about only
    /// if `target` (once it's being placed).
    fn fits(&mut self, word: &str, toks: &[&str], target: bool) -> bool {
        let [x, y, z] = self.level.size;
        let floor = if word == "stairs" { toks[2] } else { "0" };
        let text = format!("size {x} {y} {z} open 0 0 {floor} {} {} {} {word} {}", x - 1, y - 1, z - 1, toks.join(" "));
        let Ok(p) = crate::tokens::decode(&text) else { return true };
        let mut alone = Level::from_puzzle(&p);
        let l = &mut self.level;
        // What the item rests on is the level's, not the open chamber's (as `decode` places it):
        // a rail platform is mounted on whichever surface runs all along its track, a gel
        // dropper hangs from the ceiling over its splat.
        for r in &alone.rails {
            let track = r.track();
            let a = (0..3).find(|&a| r.cell[a] != r.end[a]).unwrap_or(0);
            let left = crate::builder::Builder::rail_up(Dir::from_axis(a, r.end[a] > r.cell[a]));
            let up = [left, left.opposite()]
                .into_iter()
                .chain(Dir::HORIZONTAL)
                .chain([Dir::PosZ, Dir::NegZ])
                .filter(|d| d.axis() != a)
                .find(|&d| track.iter().all(|&c| l.solid(step(c, d.opposite()))));
            if let (Some(up), Some(it)) = (up, alone.items.iter_mut().find(|it| it.index == r.item)) {
                it.normal = Some(up);
            }
        }
        if let Some(splat) = alone.items.iter().find(|it| it.kind == ItemKind::PaintSplat).map(|it| it.cell) {
            let mut top = splat;
            while !l.solid(step(top, Dir::PosZ)) {
                top = step(top, Dir::PosZ);
            }
            for it in alone.items.iter_mut().filter(|it| it.kind == ItemKind::PaintDropper) {
                it.cell = top;
            }
        }
        let lens = (l.items.len(), l.barriers.len(), l.hazards.len(), l.stairs.len(), l.rails.len());
        let next = l.items.iter().map(|i| i.index).max().unwrap_or(-1) + 1;
        let index: HashMap<i32, i32> = alone.items.iter().enumerate().map(|(k, it)| (it.index, next + k as i32)).collect();
        let new = |i: i32| index.get(&i).copied().unwrap_or(-1);
        let mine: HashSet<(&'static str, Cell)> = alone
            .items
            .iter()
            .filter(|it| target || it.kind != ItemKind::PlateTarget)
            .filter_map(|it| Some((target_name(&it.kind)?, it.cell)))
            .collect();
        // A field reaches across the open chamber: each of its lines only as far as the level's
        // first wall, as `Level` would have it there.
        let sweeps: HashMap<i32, Dir> = alone.items.iter().filter_map(|it| Some((it.index, it.normal?))).collect();
        let hazards: Vec<_> = alone
            .hazards
            .into_iter()
            .map(|mut h| {
                let n = sweeps.get(&h.item).copied();
                let (mut kept, mut open, mut prev) = (Vec::new(), true, None);
                for (c, f) in h.faces {
                    let along = n.is_some_and(|n| prev.map(|p| step(p, n)) == Some(c));
                    open = (open || !along) && l.in_interior(c) && !l.solid(c);
                    if open {
                        kept.push((c, f));
                    }
                    prev = Some(c);
                }
                h.faces = kept;
                h.item = new(h.item);
                h
            })
            .collect();
        l.hazards.extend(hazards);
        l.items.extend(alone.items.into_iter().map(|mut it| {
            it.index = new(it.index);
            it
        }));
        l.barriers.extend(alone.barriers.into_iter().map(|mut b| {
            b.item = new(b.item);
            b
        }));
        l.stairs.extend(alone.stairs.into_iter().map(|mut st| {
            st.item = new(st.item);
            st
        }));
        l.rails.extend(alone.rails.into_iter().map(|mut r| {
            r.item = new(r.item);
            r
        }));
        // `decode` keeps goo out of the cells items stand in: off the chamber while it's asked.
        let cells: HashSet<Cell> = mine.iter().map(|m| m.1).collect();
        let goo: Vec<(usize, Cell)> = (0..lens.0)
            .filter(|&i| l.items[i].kind == ItemKind::Goo && cells.contains(&l.items[i].cell))
            .map(|i| (i, l.items[i].cell))
            .collect();
        for &(i, _) in &goo {
            l.items[i].cell = [-1, -1, -1];
        }
        let ok = editor_flags(l).into_iter().all(|f| self.flags.contains(&f) || !mine.contains(&(f.name, f.pos)));
        for &(i, c) in &goo {
            l.items[i].cell = c;
        }
        l.items.truncate(lens.0);
        l.barriers.truncate(lens.1);
        l.hazards.truncate(lens.2);
        l.stairs.truncate(lens.3);
        l.rails.truncate(lens.4);
        ok
    }
}

/// Where an item goes: its cell, and the surface it's on for wall-mounted ones.
type Place = (Cell, Option<Dir>);

/// A level's tokens without its wires, which may name items not written yet and move nothing.
fn unwired<'t>(toks: &[&'t str]) -> Vec<&'t str> {
    let mut kept = Vec::with_capacity(toks.len());
    let mut i = 0;
    while i < toks.len() {
        if toks[i] == ">" {
            i += if toks.get(i + 1) == Some(&"exit") { 2 } else { 3 };
            continue;
        }
        kept.push(toks[i]);
        i += 1;
    }
    kept
}

fn decoded(toks: &[&str]) -> Option<Level> {
    let p = crate::tokens::decode(&unwired(toks).join(" ")).ok()?;
    Some(Level::from_puzzle(&p))
}

impl Grammar {
    pub fn new() -> Grammar {
        Grammar::default()
    }

    /// One whose items also keep clear of the editor's rules (`editor.rs`).
    pub fn with_editor_rules() -> Grammar {
        Grammar {
            editor: true,
            ..Grammar::default()
        }
    }

    /// The level of tokens so far (an unfinished one too) to try items in, if they decode.
    fn probe(&mut self, toks: &[&str]) -> Option<Rc<RefCell<Probe>>> {
        let key = toks.join(" ");
        if let Some(p) = self.probes.get(&key) {
            return p.clone();
        }
        if self.probes.len() >= MAX_PROBES {
            self.probes.clear();
        }
        let p = decoded(toks).map(|level| {
            let flags = editor_flags(&level).into_iter().collect();
            Rc::new(RefCell::new(Probe { level, flags }))
        });
        self.probes.insert(key, p.clone());
        p
    }

    /// Whether an item `word`, after the level's tokens `before`, gets no editor flag
    /// (`Probe::fits`) at one of `places` with some way of finishing it (`tails`). It stops at the
    /// first such place, and remembers the places it tried.
    fn some_clear(&mut self, before: &[&str], word: &str, places: &[Place], tails: Tails) -> bool {
        let Some(probe) = self.probe(before) else { return true };
        let mut probe = probe.borrow_mut();
        let tried = self.tried.entry(format!("{} {word}", before.join(" "))).or_default();
        places.iter().any(|&(c, d)| {
            *tried.entry((c, d)).or_insert_with(|| {
                let at: Vec<&str> = c.iter().map(|&v| NUMS[v as usize]).chain(d.map(dir_token)).collect();
                tails((c, d)).into_iter().any(|tail| probe.fits(word, &[&at[..], &tail[..]].concat(), false))
            })
        })
    }

    /// What may follow `tokens`, or why they aren't the start of a legal level.
    pub fn next(&mut self, tokens: &[&str]) -> Result<Next, String> {
        if self.geos.len() > MAX_CACHED {
            self.geos.clear();
        }
        if self.tried.len() > MAX_CACHED {
            self.tried.clear();
        }
        let mut p = Parser {
            toks: tokens,
            pos: 0,
            grammar: self,
            size: [1, 1, 1],
            boxes: Vec::new(),
            geo: None,
            section: 0,
            count: [0; 22],
            debt: [-1; 22],
            poured: Vec::new(),
            start: 0,
        };
        match p.level() {
            Halt::At(opts, end) => Ok(Next {
                allowed: opts.into_iter().collect(),
                end,
            }),
            Halt::Bad(why) => Err(why),
        }
    }

    /// Whether `tokens` are a whole legal level, or why not.
    pub fn check(&mut self, tokens: &[&str]) -> Result<(), String> {
        match self.next(tokens)? {
            Next { end: true, .. } => Ok(()),
            n => Err(format!("can't end there; expected one of {:?}", n.allowed)),
        }
    }
}

type Opts = BTreeSet<&'static str>;

/// Where reading stops: at the end of the tokens (what may come next), or at one that can't be there.
enum Halt {
    At(Opts, bool),
    Bad(String),
}

type R<T> = Result<T, Halt>;

struct Parser<'a, 'g> {
    toks: &'a [&'a str],
    pos: usize,
    grammar: &'g mut Grammar,
    size: [i32; 3],
    /// `open` and `goo` boxes, while reading them.
    boxes: Vec<(Cell, Cell)>,
    geo: Option<Rc<Geo>>,
    /// Section of the item being read (or last read).
    section: usize,
    /// Items read per section.
    count: [i32; 22],
    /// Highest index wired to per section, -1 for none.
    debt: [i32; 22],
    /// Per gel, whether it's poured (only those take wires).
    poured: Vec<bool>,
    /// Where the item being read starts (its word).
    start: usize,
}

/// The ways to finish an item after a token that places it, each its tokens: legal ones (they
/// don't move it), one per way it may still lie there (a panel's swing, say).
type Tail<'t> = &'t dyn Fn(&str) -> Vec<Vec<&'static str>>;
/// The same after its place.
type Tails<'t> = &'t dyn Fn(Place) -> Vec<Vec<&'static str>>;

fn nums(lo: i32, hi: i32) -> Opts {
    (lo.max(0)..=hi.min(MAX_NUM)).map(|n| NUMS[n as usize]).collect()
}

fn dirs(ds: impl IntoIterator<Item = Dir>) -> Opts {
    ds.into_iter().map(dir_token).collect()
}

/// `white` and `black`, if a direction after `painted` is left.
fn paints(painted: Option<usize>) -> Opts {
    if painted.map_or(0, |p| p + 1) < DIRS.len() {
        ["white", "black"].into_iter().collect()
    } else {
        Opts::new()
    }
}

fn across(n: Dir) -> impl Iterator<Item = Dir> {
    DIRS.into_iter().filter(move |d| d.axis() != n.axis())
}

fn num_of(t: &str) -> i32 {
    t.parse().expect("only numbers were allowed")
}

fn dir_of(t: &str) -> Dir {
    crate::tokens::parse_dir(t).expect("only directions were allowed")
}

impl<'a> Parser<'a, '_> {
    /// The next token, which must be one of `opts`; at the end of the tokens, stops there.
    fn take(&mut self, opts: Opts, end: bool) -> R<&'a str> {
        match self.toks.get(self.pos) {
            None => Err(Halt::At(opts, end)),
            Some(t) if opts.contains(*t) => {
                self.pos += 1;
                Ok(t)
            }
            Some(t) => Err(Halt::Bad(format!(
                "token {}: {t:?} can't come here; expected one of {opts:?}{}",
                self.pos,
                if end { " or the end" } else { "" }
            ))),
        }
    }

    /// One of `opts` (taken), or (`None`, not taken) one of what `rest` gives, which may also
    /// end the level (worked out only when needed: it may build the geometry).
    fn either(&mut self, opts: Opts, rest: impl FnOnce(&mut Self) -> (Opts, bool)) -> R<Option<&'a str>> {
        if let Some(t) = self.toks.get(self.pos) {
            if opts.contains(*t) {
                self.pos += 1;
                return Ok(Some(t));
            }
        }
        let (more, end) = rest(self);
        if self.toks.get(self.pos).is_some_and(|t| more.contains(*t)) {
            return Ok(None);
        }
        let mut all = opts;
        all.extend(more);
        self.take(all, end).map(Some)
    }

    fn num(&mut self, lo: i32, hi: i32) -> R<i32> {
        self.take(nums(lo, hi), false).map(num_of)
    }

    fn word(&mut self, words: &[&'static str]) -> R<&'a str> {
        self.take(words.iter().copied().collect(), false)
    }

    /// A cell with each number between `lo` and `hi`.
    fn cell_between(&mut self, lo: Cell, hi: Cell) -> R<Cell> {
        Ok([
            self.num(lo[0], hi[0])?,
            self.num(lo[1], hi[1])?,
            self.num(lo[2], hi[2])?,
        ])
    }

    fn any_cell(&mut self) -> R<Cell> {
        let s = self.size;
        self.cell_between([0, 0, 0], [s[0] - 1, s[1] - 1, s[2] - 1])
    }

    /// One of `cells`.
    fn cell_in(&mut self, cells: &[Cell]) -> R<Cell> {
        let mut c = [0; 3];
        for k in 0..3 {
            let opts = cells
                .iter()
                .filter(|x| x[..k] == c[..k])
                .map(|x| NUMS[x[k] as usize])
                .collect();
            c[k] = self.take(opts, false).map(num_of)?;
        }
        Ok(c)
    }

    /// One of `places` (a cell, then a direction for faces), each token of it, when it's the
    /// one being written with the editor's rules on, leading to some place where the item
    /// (finished with one of `tails`) gets no editor flag (`Grammar::some_clear`), unless none does.
    fn place_in(&mut self, places: &[Place], tails: Tails) -> R<Place> {
        let mut c = [0; 3];
        for k in 0..3 {
            let group: Vec<Place> = places.iter().filter(|p| p.0[..k] == c[..k]).copied().collect();
            let opts = group.iter().map(|p| NUMS[p.0[k] as usize]).collect();
            let opts = self.leading_clear(opts, |t| group.iter().filter(|p| p.0[k] == num_of(t)).copied().collect(), tails);
            c[k] = self.take(opts, false).map(num_of)?;
        }
        if places.first().is_some_and(|p| p.1.is_none()) {
            return Ok((c, None));
        }
        let here: Vec<Place> = places.iter().filter(|p| p.0 == c).copied().collect();
        let opts = dirs(here.iter().map(|p| p.1.expect("faces have a direction")));
        let opts = self.leading_clear(opts, |t| vec![(c, Some(dir_of(t)))], tails);
        Ok((c, Some(self.take(opts, false).map(dir_of)?)))
    }

    /// Those of `opts` (the token being written) whose `places` have a clear one (`place_in`).
    fn leading_clear(&mut self, opts: Opts, places: impl Fn(&str) -> Vec<Place>, tails: Tails) -> Opts {
        if self.pos < self.toks.len() || !self.grammar.editor {
            return opts;
        }
        let toks = self.toks;
        let (before, word) = (&toks[..self.start], toks[self.start]);
        let clear: Opts = opts
            .iter()
            .copied()
            .filter(|t| self.grammar.some_clear(before, word, &places(t), tails))
            .collect();
        if clear.is_empty() { opts } else { clear }
    }

    /// One of `cells` where the item (finished with one of `tails`) gets no editor flag (`place_in`).
    fn cell_placing(&mut self, cells: &[Cell], tails: Tails) -> R<Cell> {
        let places: Vec<Place> = cells.iter().map(|&c| (c, None)).collect();
        Ok(self.place_in(&places, tails)?.0)
    }

    /// One of `faces` where the item (finished with one of `tails`) gets no editor flag (`place_in`).
    fn face_placing(&mut self, faces: &[(Cell, Dir)], tails: Tails) -> R<(Cell, Dir)> {
        let places: Vec<Place> = faces.iter().map(|&(c, d)| (c, Some(d))).collect();
        let (c, d) = self.place_in(&places, tails)?;
        Ok((c, d.expect("faces have a direction")))
    }

    /// One of `opts`, a token that settles where the item goes. At the end of the tokens, with
    /// the editor's rules on, only those after which the item, finished with one of `tail`, gets
    /// no editor flag (`Probe::fits`), unless that's none of them.
    fn placing(&mut self, opts: Opts, tail: Tail) -> R<&'a str> {
        if self.pos < self.toks.len() || !self.grammar.editor {
            return self.take(opts, false);
        }
        let Some(probe) = self.grammar.probe(&self.toks[..self.start]) else {
            return self.take(opts, false);
        };
        let mut probe = probe.borrow_mut();
        let (word, written) = (self.toks[self.start], &self.toks[self.start + 1..]);
        // A faith plate's face and target cell's first two numbers are written.
        let target = word == "plate" && written.len() >= 6;
        let clear: Opts = opts
            .iter()
            .copied()
            .filter(|t| tail(t).into_iter().any(|end| probe.fits(word, &[written, &[*t], &end[..]].concat(), target)))
            .collect();
        drop(probe);
        self.take(if clear.is_empty() { opts } else { clear }, false)
    }

    /// One of `opts` for which `ok` holds (asked only of the token there, if there is one).
    fn take_if(&mut self, opts: Opts, ok: impl Fn(&str) -> bool) -> R<&'a str> {
        match self.toks.get(self.pos) {
            Some(t) if opts.contains(*t) && ok(t) => {
                self.pos += 1;
                Ok(t)
            }
            _ => self.take(opts.into_iter().filter(|t| ok(t)).collect(), false),
        }
    }

    fn on_off(&mut self) -> R<&'a str> {
        self.word(&["on", "off"])
    }

    fn geo(&mut self) -> Rc<Geo> {
        if let Some(g) = &self.geo {
            return g.clone();
        }
        let key = self.toks[..self.pos].join(" ");
        let g = match self.grammar.geos.get(&key) {
            Some(g) => g.clone(),
            None => {
                let g = Rc::new(Geo::new(self.size, &self.boxes));
                self.grammar.geos.insert(key, g.clone());
                g
            }
        };
        self.geo = Some(g.clone());
        g
    }

    fn level(&mut self) -> Halt {
        match self.read() {
            Ok(()) => unreachable!("reading stops only at the end of the tokens or a bad one"),
            Err(h) => h,
        }
    }

    fn read(&mut self) -> R<()> {
        self.word(&["size"])?;
        self.size = [self.num(1, MAX_NUM)?, self.num(1, MAX_NUM)?, self.num(1, MAX_NUM)?];
        self.word(&["open"])?;
        let mut goo = false;
        loop {
            let b = self.a_box()?;
            self.boxes.push(b);
            self.geo = None;
            // Another box, `goo` boxes, or on to the surfaces.
            let mut words = Opts::new();
            if !goo {
                words.insert("goo");
            }
            let opts: Opts = nums(0, self.size[0] - 1).into_iter().chain(words).collect();
            match self.either(opts, |p| p.after_geometry(None))? {
                Some("goo") => goo = true,
                Some(_) => self.pos -= 1, // The first number of the next box.
                None => break,
            }
        }
        let mut painted: Option<usize> = None;
        loop {
            match self.either(paints(painted), |p| p.after_geometry(painted))? {
                Some(word) => {
                    let k = self.take(dirs(DIRS[painted.map_or(0, |p| p + 1)..].iter().copied()), false)?;
                    painted = Some(DIRS.iter().position(|&d| d == dir_of(k)).unwrap());
                    if word == "white" {
                        self.a_box()?;
                    }
                    loop {
                        match self.either(nums(0, self.size[0] - 1), |p| p.after_geometry(painted))? {
                            Some(_) => {
                                self.pos -= 1;
                                self.a_box()?;
                            }
                            None => break,
                        }
                    }
                }
                None => break,
            }
        }
        loop {
            let (opts, end) = self.items();
            let t = self.take(opts, end)?;
            self.item(t)?;
        }
    }

    /// What may follow the geometry (or a surface group): more surfaces, items.
    fn after_geometry(&mut self, painted: Option<usize>) -> (Opts, bool) {
        self.geo();
        let (mut opts, end) = self.items();
        opts.extend(paints(painted));
        (opts, end)
    }

    fn a_box(&mut self) -> R<(Cell, Cell)> {
        let s = self.size;
        let lo = self.any_cell()?;
        let hi = self.cell_between(lo, [s[0] - 1, s[1] - 1, s[2] - 1])?;
        Ok((lo, hi))
    }

    /// Whether the level may go past section `k`.
    fn closed(&self, k: usize) -> bool {
        match k {
            0 | 1 => self.count[k] == 1,
            _ => self.count[k] > self.debt[k],
        }
    }

    /// Whether an item of section `k` fits somewhere (one more, for turrets).
    fn room(&self, k: usize, g: &Geo) -> bool {
        match k {
            0 | 1 => !g.doors.is_empty(),
            2 => !g.glass_starts().is_empty(),
            5 => !g.ceiling().is_empty(),
            8 if self.count[8] >= MAX_TURRETS as i32 => false,
            6..=8 => !g.floor().is_empty(),
            19 => !g.stairs.is_empty(),
            20 => !g.pistons().is_empty(),
            21 => true,
            _ => !g.all_faces.is_empty(),
        }
    }

    /// The item words that may come next, and whether the level may end instead.
    fn items(&mut self) -> (Opts, bool) {
        let g = self.geo();
        let mut opts = Opts::new();
        for (k, words) in SECTIONS.iter().enumerate().skip(self.section) {
            if k > self.section && !self.closed(k - 1) {
                break;
            }
            if k <= 1 && self.count[k] > 0 {
                continue;
            }
            if self.room(k, &g) {
                opts.extend(words.iter().copied());
            }
        }
        let end = (self.section..SECTIONS.len()).all(|k| self.closed(k));
        (opts, end)
    }

    fn item(&mut self, word: &'a str) -> R<()> {
        let k = SECTIONS.iter().position(|ws| ws.contains(&word)).unwrap();
        self.section = k;
        self.count[k] += 1;
        self.start = self.pos - 1;
        let g = self.geo();
        let s = self.size;
        let none = |_: &str| vec![vec![]];
        let on = |_: &str| vec![vec!["on"]];
        let finished = |tail: &'static [&'static str]| move |_: Place| vec![tail.to_vec()];
        // Every way a panel or bridge may swing out from its face, then `rest`.
        let swings = |rest: &'static [&'static str]| {
            move |(_, n): Place| across(n.unwrap()).map(|d| [&[dir_token(d)][..], rest].concat()).collect()
        };
        match word {
            "entry" | "exit" => {
                self.face_placing(&g.doors, &finished(&[]))?;
                if word == "exit" {
                    self.either(["inv"].into_iter().collect(), |p| p.items())?;
                }
            }
            "glass" | "grating" => {
                let lo = self.cell_in(g.glass_starts())?;
                // The far corner, a number at a time, while some panel over the box still fits.
                let mut hi = lo;
                for k in 0..3 {
                    let ok = |t: &str| {
                        let mut h = hi;
                        h[k] = num_of(t);
                        glass_fits(&g, lo, h, k + 1)
                    };
                    hi[k] = self.take_if(nums(lo[k], s[k] - 1), ok).map(num_of)?;
                }
                let fits = |n: Dir, grow: Dir| grow.axis() != n.axis() && glass_mounted(&g, lo, hi, n, grow);
                let grows = |n: &str| {
                    let n = dir_of(n);
                    DIRS.into_iter().filter(|&gr| fits(n, gr)).map(|gr| vec![dir_token(gr)]).collect()
                };
                let n = self.placing(
                    dirs(DIRS.into_iter()
                        .filter(|n| lo[n.axis()] == hi[n.axis()] && DIRS.iter().any(|&gr| fits(*n, gr)))),
                    &grows,
                )?;
                self.placing(dirs(DIRS.into_iter().filter(|&gr| fits(dir_of(n), gr))), &none)?;
            }
            "fizzler" | "laserfield" => {
                // Placed as a field one cell long, then as long as it gets.
                let short = |(_, n): Place| across(n.unwrap()).map(|d| vec![dir_token(d), "1", "on"]).collect();
                let (c, n) = self.face_placing(&g.all_faces, &short)?;
                let along = dir_of(self.placing(dirs(across(n)), &|_| vec![vec!["1", "on"]])?);
                let room = (0..)
                    .take_while(|&k| {
                        let v = along.vector();
                        g.idx(std::array::from_fn(|a| c[a] + v[a] * k)).is_some()
                    })
                    .count() as i32;
                self.placing(nums(1, room), &on)?;
                self.on_off()?;
            }
            "gel" => {
                let (_, n) = self.face_placing(&g.all_faces, &finished(&["bounce", "splat"]))?;
                self.word(&PAINT_KINDS)?;
                let kinds: &[&'static str] = if n == Dir::PosZ { &["stream", "splat"] } else { &["splat"] };
                let poured_on = |t: &str| vec![if t == "stream" { vec!["on"] } else { vec![] }];
                let poured = self.placing(kinds.iter().copied().collect(), &poured_on)? == "stream";
                self.poured.push(poured);
                if poured {
                    self.on_off()?;
                }
            }
            "dropper" => {
                self.cell_placing(g.ceiling(), &finished(&["standard"]))?;
                self.word(&CUBE_KINDS)?;
                self.either(["wait"].into_iter().collect(), |p| p.items())?;
            }
            "button" => {
                self.cell_placing(g.floor(), &finished(&["weighted"]))?;
                self.word(&BUTTON_KINDS)?;
                self.wires()?;
            }
            "pedestal" => {
                self.cell_placing(g.floor(), &finished(&["inf"]))?;
                let mut opts = nums(3, 30);
                opts.insert("inf");
                self.take(opts, false)?;
                self.wires()?;
            }
            "turret" => {
                self.cell_placing(g.floor(), &finished(&["0"]))?;
                self.num(0, 7)?;
            }
            "cube" => {
                self.num(0, s[0] - 1)?;
                self.num(0, s[1] - 1)?;
                self.placing(nums(0, s[2] - 1), &|_| vec![vec![CUBE_KINDS[0]]])?;
                self.word(&CUBE_KINDS)?;
            }
            "laser" | "catcher" | "relay" => {
                let laser = word == "laser";
                let centred = move |_: Place| vec![if laser { vec!["center", "on"] } else { vec!["center"] }];
                let (_, n) = self.face_placing(&g.all_faces, &centred)?;
                let mut opts = dirs(across(n));
                opts.insert("center");
                let tail = |_: &str| vec![if laser { vec!["on"] } else { vec![] }];
                self.placing(opts, &tail)?;
                if word == "laser" {
                    self.on_off()?;
                } else {
                    self.wires()?;
                }
            }
            "plate" => {
                let target = |(c, _): Place| {
                    let z = if c[2] + 1 < s[2] { c[2] + 1 } else { c[2] - 1 };
                    vec![vec![NUMS[c[0] as usize], NUMS[c[1] as usize], NUMS[z.max(0) as usize], "+z"]]
                };
                let (c, _) = self.face_placing(&g.all_faces, &target)?;
                let x = self.num(0, s[0] - 1)?;
                let y = self.num(0, s[1] - 1)?;
                let facing = |_: &str| DIRS.map(|d| vec![dir_token(d)]).to_vec();
                self.placing(nums(0, s[2] - 1).into_iter().filter(|t| [x, y, num_of(t)] != c).collect(), &facing)?;
                self.placing(dirs(DIRS), &none)?;
            }
            "bridge" => {
                let (_, n) = self.face_placing(&g.all_faces, &swings(&["on"]))?;
                self.placing(dirs(across(n)), &on)?;
                self.on_off()?;
            }
            "funnel" => {
                self.face_placing(&g.all_faces, &finished(&["on"]))?;
                self.on_off()?;
                self.either(["rev"].into_iter().collect(), |p| p.items())?;
            }
            "panel" | "glasspanel" => {
                let (_, n) = self.face_placing(&g.all_faces, &swings(&["2", "on"]))?;
                self.placing(dirs(across(n)), &|_| vec![vec!["2", "on"]])?;
                self.take(["2", "3", "4", "6"].into_iter().collect(), false)?;
                self.on_off()?;
            }
            "flip" => {
                self.face_placing(&g.all_faces, &finished(&[]))?;
            }
            "stairs" => {
                self.face_placing(&g.stairs, &finished(&["on"]))?;
                self.on_off()?;
            }
            "piston" => {
                let (c, n) = self.face_placing(g.pistons(), &finished(&["0", "1", "on"]))?;
                // Open as far as it reaches (on a floor, also to stand on its top): `Level`.
                let v = n.vector();
                let open = |t: i32| {
                    let reach = if n == Dir::PosZ { t } else { t - 1 };
                    (0..=reach).all(|l| !g.solid(std::array::from_fn(|a| c[a] + v[a] * l)))
                };
                let lo = self.take_if(nums(0, 3), |t| open(num_of(t) + 1)).map(num_of)?;
                let reach = nums(lo + 1, 4).into_iter().filter(|t| open(num_of(t))).collect();
                self.placing(reach, &on)?;
                self.on_off()?;
            }
            "rail" => {
                // Placed by its start among the items, on a track one cell long each way from there.
                let all: Vec<Cell> = crate::level::cells_between([0, 0, 0], [s[0] - 1, s[1] - 1, s[2] - 1]).collect();
                let track = |(c, _): Place| {
                    DIRS.into_iter()
                        .map(|d| step(c, d))
                        .filter(|e| (0..3).all(|a| (0..s[a]).contains(&e[a])))
                        .map(|e| vec![NUMS[e[0] as usize], NUMS[e[1] as usize], NUMS[e[2] as usize], "osc"])
                        .collect()
                };
                let c = self.cell_placing(&all, &track)?;
                let mut e = [0; 3];
                for a in 0..3 {
                    let moved = (0..a).filter(|&b| e[b] != c[b]).count();
                    let opts = match (moved, a) {
                        (1, _) => nums(c[a], c[a]),
                        (0, 2) => nums(0, s[a] - 1).into_iter().filter(|t| num_of(t) != c[a]).collect(),
                        _ => nums(0, s[a] - 1),
                    };
                    // The rest of the far end: the start's, or one cell from it along a later
                    // axis if the track hasn't moved yet.
                    let rest = |t: &str| {
                        let mut end = e;
                        end[a] = num_of(t);
                        let moved = (0..=a).any(|b| end[b] != c[b]);
                        let finish = |end: Cell| {
                            let mut toks: Vec<&'static str> = (a + 1..3).map(|b| NUMS[end[b] as usize]).collect();
                            toks.push("osc");
                            toks
                        };
                        let mut end = c;
                        if moved {
                            return vec![finish(end)];
                        }
                        let mut ends = Vec::new();
                        for b in a + 1..3 {
                            for v in [c[b] - 1, c[b] + 1] {
                                if (0..s[b]).contains(&v) {
                                    end[b] = v;
                                    ends.push(finish(end));
                                    end[b] = c[b];
                                }
                            }
                        }
                        ends
                    };
                    e[a] = self.placing(opts, &rest).map(num_of)?;
                }
                self.word(&["osc", "on", "off"])?;
            }
            _ => unreachable!("item words come from SECTIONS"),
        }
        Ok(())
    }

    /// A sender's wires: `> T` each, sorted by target and index, each once.
    fn wires(&mut self) -> R<()> {
        let mut last: Option<(usize, i32)> = None;
        loop {
            if self.either([">"].into_iter().collect(), |p| p.items())?.is_none() {
                return Ok(());
            }
            let opts = (0..TARGETS.len()).filter(|&t| !self.indices(t, last).is_empty());
            let opts = opts.map(|t| TARGETS[t].0).collect();
            let word = self.take(opts, false)?;
            let t = TARGETS.iter().position(|x| x.0 == word).unwrap();
            let i = if word == "exit" {
                0
            } else {
                let opts = self.indices(t, last).into_iter().map(|i| NUMS[i as usize]).collect();
                self.take(opts, false).map(num_of)?
            };
            let k = TARGETS[t].1;
            if k > self.section {
                self.debt[k] = self.debt[k].max(i);
            }
            last = Some((t, i));
        }
    }

    /// The indices a wire to target `t` may name after wire `last` (sorted, each once).
    fn indices(&self, t: usize, last: Option<(usize, i32)>) -> Vec<i32> {
        let from = match last {
            Some((lt, _)) if lt > t => return Vec::new(),
            Some((lt, li)) if lt == t => li + 1,
            _ => 0,
        };
        let (word, k) = TARGETS[t];
        match word {
            "exit" => (from..1).collect(),
            "gel" => (from..(self.poured.len() as i32).min(MAX_NUM + 1))
                .filter(|&i| self.poured[i as usize])
                .collect(),
            // A section not written yet: anything, if its items fit somewhere (to pay the debt).
            _ if k > self.section => match &self.geo {
                Some(g) if self.room(k, g) => (from..=MAX_NUM).collect(),
                _ => Vec::new(),
            },
            _ => (from..self.count[k].min(MAX_NUM + 1)).collect(),
        }
    }
}

/// Whether a glass or grating panel fits the box from `lo` to a far corner whose first `known`
/// numbers are `hi`'s (the rest anything from `lo`'s up): flat across some normal and mounted.
fn glass_fits(g: &Geo, lo: Cell, hi: Cell, known: usize) -> bool {
    if known < 3 {
        return (lo[known]..g.size[known]).any(|v| {
            let mut h = hi;
            h[known] = v;
            glass_fits(g, lo, h, known + 1)
        });
    }
    DIRS.iter().any(|&n| {
        lo[n.axis()] == hi[n.axis()]
            && DIRS.iter().any(|&grow| grow.axis() != n.axis() && glass_mounted(g, lo, hi, n, grow))
    })
}

/// The cells a glass or grating panel's box can start at: some box from there fits
/// (`glass_fits`). Per growth and side axis: the first cell up the side axis from the start with
/// a surface behind it against the growth must be open (`glass_mounted`); a panel growing toward
/// - hangs from the box's far row along the growth, which may be any row past the start.
fn glass_starts(g: &Geo) -> Vec<Cell> {
    let size = g.size;
    let n = g.solid.len();
    let idx = |c: Cell| g.idx(c).expect("in the chamber");
    // Every cell, layer by layer along axis `a`, from the top layer down.
    let down = |a: usize| {
        (0..size[a]).rev().flat_map(move |k| {
            crate::level::cells_between([0, 0, 0], [size[0] - 1, size[1] - 1, size[2] - 1])
                .filter(move |c| c[a] == k)
        })
    };
    let mut ok = vec![false; n];
    for grow in DIRS {
        let ga = grow.axis();
        for sa in (0..3).filter(|&a| a != ga) {
            let up = Dir::from_axis(sa, true);
            let mut f = vec![false; n];
            for c in down(sa) {
                f[idx(c)] = if g.solid(step(c, grow.opposite())) {
                    !g.solid(c)
                } else {
                    g.idx(step(c, up)).is_some_and(|i| f[i])
                };
            }
            if !grow.is_positive() {
                let along = Dir::from_axis(ga, true);
                for c in down(ga) {
                    if let Some(i) = g.idx(step(c, along)) {
                        f[idx(c)] |= f[i];
                    }
                }
            }
            for (o, v) in ok.iter_mut().zip(&f) {
                *o |= v;
            }
        }
    }
    crate::level::cells_between([0, 0, 0], [size[0] - 1, size[1] - 1, size[2] - 1])
        .filter(|&c| ok[idx(c)])
        .collect()
}

/// Whether a glass or grating panel over `lo..=hi` on the `n` faces, grown along `grow`, hangs
/// from something: the cell `decode` anchors it at is open with a solid surface behind it
/// against `grow`.
fn glass_mounted(g: &Geo, lo: Cell, hi: Cell, n: Dir, grow: Dir) -> bool {
    let Some(side) = grow.cross(n.opposite()) else { return false };
    let (ga, sa) = (grow.axis(), side.axis());
    let mut edge = lo;
    if !grow.is_positive() {
        edge[ga] = hi[ga];
    }
    let anchor = (lo[sa]..=hi[sa])
        .map(|k| {
            let mut c = edge;
            c[sa] = k;
            c
        })
        .find(|&c| g.solid(step(c, grow.opposite())))
        .unwrap_or(edge);
    !g.solid(anchor) && g.solid(step(anchor, grow.opposite()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn next(text: &str) -> Next {
        let toks: Vec<&str> = text.split_whitespace().collect();
        Grammar::new().next(&toks).unwrap()
    }

    const ROOM: &str = "size 4 4 4 open 1 1 1 2 2 2";

    #[test]
    fn starts_with_size_and_open() {
        assert_eq!(next("").allowed, vec!["size"]);
        assert_eq!(next("size 4 4").allowed.len(), 32);
        assert_eq!(next("size 4 4 4").allowed, vec!["open"]);
        // The far corner isn't below the near one.
        assert_eq!(next("size 4 4 4 open 1 1 1 2").allowed, vec!["1", "2", "3"]);
    }

    #[test]
    fn doors_stand_on_a_floor_against_a_wall() {
        let n = next(&format!("{ROOM} entry"));
        assert_eq!(n.allowed, vec!["1", "2"]);
        // Any height on a wall.
        assert_eq!(next(&format!("{ROOM} entry 1 1")).allowed, vec!["1", "2"]);
        // In the corner (1, 1, 1): walls at -x and -y, so the door faces +x or +y (or up).
        assert_eq!(next(&format!("{ROOM} entry 1 1 1")).allowed, vec!["+x", "+y"]);
        assert!(!next(&format!("{ROOM} entry 1 1 1 +x")).end);
        assert_eq!(next(&format!("{ROOM} entry 1 1 1 +x")).allowed, vec!["exit"]);
    }

    #[test]
    fn wires_forward_must_be_paid() {
        let base = format!("{ROOM} entry 1 1 1 +x exit 2 2 1 -x button 1 2 1 cube > panel 1");
        let n = next(&base);
        assert!(!n.end);
        assert!(n.allowed.contains(&"panel") && !n.allowed.contains(&"flip"));
        let one = next(&format!("{base} panel 1 1 2 +x +y 2 on"));
        assert!(!one.end && !one.allowed.contains(&"flip"));
        let two = next(&format!("{base} panel 1 1 2 +x +y 2 on panel 2 2 2 -x +y 3 off"));
        assert!(two.end && two.allowed.contains(&"flip"));
    }

    #[test]
    fn wires_backward_name_what_exists() {
        let base = format!("{ROOM} entry 1 1 1 +x exit 2 2 1 -x fizzler 1 2 2 +x +z 1 on button 1 2 1 cube >");
        let n = next(&base);
        assert!(n.allowed.contains(&"field") && !n.allowed.contains(&"gel") && !n.allowed.contains(&"dropper"));
        assert_eq!(next(&format!("{base} field")).allowed, vec!["0"]);
        // Sorted, each once: after `> field 0` only later targets.
        let after = next(&format!("{base} field 0 >"));
        assert!(!after.allowed.contains(&"field") && !after.allowed.contains(&"exit"));
    }

    #[test]
    fn random_levels_never_get_stuck_and_decode() {
        // Random walks over the allowed tokens: each step has a next token or may end, and a
        // level ended where allowed decodes.
        let mut g = Grammar::new();
        let mut rng = crate::rng::Rng::new(7);
        let mut ended = 0;
        for walk in 0..300 {
            let mut toks: Vec<&str> = Vec::new();
            for _ in 0..400 {
                let n = g.next(&toks).unwrap();
                assert!(!n.allowed.is_empty() || n.end, "stuck after {}", toks.join(" "));
                // Short levels: stop as soon as allowed, half the time.
                if n.end && (n.allowed.is_empty() || rng.below(2) == 0) {
                    let text = toks.join(" ");
                    crate::tokens::decode(&text).unwrap_or_else(|e| panic!("walk {walk}: {e}: {text}"));
                    ended += 1;
                    break;
                }
                toks.push(n.allowed[rng.below(n.allowed.len())]);
            }
        }
        assert!(ended > 100, "only {ended} walks ended");
    }

    #[test]
    fn editor_rules_keep_items_off_a_door() {
        let base = "size 8 8 4 open 1 1 1 6 6 2 entry 1 2 1 +x exit 6 6 1 -x plate 1 2";
        let toks: Vec<&str> = base.split_whitespace().collect();
        assert_eq!(Grammar::new().next(&toks).unwrap().allowed, vec!["1", "2"]);
        // A faith plate in the entry door's cell shares it (`editor.rs`); one above doesn't.
        assert_eq!(Grammar::with_editor_rules().next(&toks).unwrap().allowed, vec!["2"]);
        // Levels already written aren't asked: the plate there still reads.
        let whole = format!("{base} 1 +z 3 3 1 +z");
        assert_eq!(Grammar::with_editor_rules().check(&whole.split_whitespace().collect::<Vec<_>>()), Ok(()));
    }

    #[test]
    fn editor_rules_never_get_stuck_and_flag_less() {
        // Random walks as above (past a goo-free geometry), with and without the rules: never
        // stuck, and with them the levels they end (almost) never get an editor flag.
        let flagged = |mut g: Grammar| {
            let mut rng = crate::rng::Rng::new(11);
            let (mut ended, mut flagged) = (0, 0);
            for _ in 0..40 {
                // A small chamber without goo keeps it quick.
                let mut toks: Vec<&str> = "size 7 7 4 open 1 1 1 5 5 2 entry".split_whitespace().collect();
                for _ in 0..300 {
                    let n = g.next(&toks).unwrap();
                    assert!(!n.allowed.is_empty() || n.end, "stuck after {}", toks.join(" "));
                    if n.end && (n.allowed.is_empty() || rng.below(4) == 0) {
                        let p = crate::tokens::decode(&toks.join(" ")).unwrap();
                        let level = crate::level::Level::from_puzzle(&p);
                        ended += 1;
                        flagged += usize::from(!editor_flags(&level).is_empty());
                        break;
                    }
                    toks.push(n.allowed[rng.below(n.allowed.len())]);
                }
            }
            (ended, flagged)
        };
        let (n0, off) = flagged(Grammar::new());
        let (n1, on) = flagged(Grammar::with_editor_rules());
        assert!(n0 > 20 && n1 > 20, "only {n0} and {n1} walks ended");
        assert!(off > 5 && on <= 1, "{on} of {n1} flagged with the rules, {off} of {n0} without");
    }

    #[test]
    fn rejects_what_decode_would() {
        let mut g = Grammar::new();
        let bad = format!("{ROOM} entry 1 1 1 +x exit 2 2 1 -x button 1 2 1 cube > field 0");
        assert!(g.check(&bad.split_whitespace().collect::<Vec<_>>()).is_err());
        let ok = format!("{ROOM} entry 1 1 1 +x exit 2 2 1 -x button 1 2 1 cube > exit");
        assert_eq!(g.check(&ok.split_whitespace().collect::<Vec<_>>()), Ok(()));
    }
}


