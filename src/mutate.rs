//! Random edits to finished levels, for training data the generator can't make itself.
//!
//! `Genome::mutate` (generator.rs) edits the generator's own genomes; these edits work on any
//! decoded `Puzzle`, such as published Workshop levels: move an item (with the items that
//! belong to it: a dropper's cube, a fizzler's extents, a door's coop twin), shift a door,
//! add a cube, button, pedestal, fizzler or turret, remove an item, repaint a patch of wall,
//! or push a patch of wall out or pull it in (growing or shrinking the room there). After
//! each edit the level goes through `tokens::encode` and `decode`, which rebuilds derived
//! items (a moved dropper's cube lands on its new floor) and renumbers items, and an edit is
//! kept only if every item is still mounted (`Level::mount_problems_with`, editor rules off
//! as for Workshop levels). `mutants` then labels the results and keeps the solvable ones
//! that need portals.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::dataset::{json_str, label_portal_solvable, Label};
use crate::geom::{step, Cell, Dir};
use crate::level::Level;
use crate::p2c::Puzzle;
use crate::rng::Rng;
use crate::tokens;

/// Tries at one edit before giving up on it.
const EDIT_TRIES: usize = 30;

#[derive(Clone, Debug)]
pub struct Config {
    pub seed: u64,
    /// Mutants wanted per input level.
    pub per_level: usize,
    /// Most mutants tried per input level, kept or not.
    pub tries: usize,
    /// Longest token sequence kept.
    pub max_tokens: usize,
    /// State budget per solve.
    pub budget: usize,
    /// Tries after which an input level that has given no mutant yet is dropped (0: never).
    /// Most published Workshop levels our solver can't solve give none in all their tries,
    /// and each of those tries runs the search to its budget.
    pub give_up: usize,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            seed: 1,
            per_level: 5,
            tries: 25,
            max_tokens: 1024,
            budget: 50_000,
            give_up: 0,
        }
    }
}

/// A kept mutant.
#[derive(Clone, Debug)]
pub struct Mutant {
    pub tokens: Vec<String>,
    pub size: [i32; 3],
    pub label: Label,
    /// What was done to the parent, in order.
    pub edits: Vec<String>,
}

impl Mutant {
    /// One JSONL line (no newline), in the format of `Sample::json` plus `mutated_from`
    /// (the parent's id) and `edits`.
    pub fn json(&self, parent: &str) -> String {
        let edits: Vec<String> = self.edits.iter().map(|e| json_str(e)).collect();
        format!(
            "{{\"tokens\":{},\"n_tokens\":{},\"size\":[{},{},{}],{},\"mutated_from\":{},\"edits\":[{}]}}",
            json_str(&self.tokens.join(" ")),
            self.tokens.len(),
            self.size[0],
            self.size[1],
            self.size[2],
            self.label.json_fields(),
            json_str(parent),
            edits.join(",")
        )
    }
}

/// Up to `cfg.per_level` mutants of the level `parent` (a token sequence), each 1–3 edits
/// away, solvable only with portals, at most `cfg.max_tokens` long and different from the
/// parent and each other. Also counts what happened to every try (`stats`).
pub fn mutants(parent: &str, cfg: &Config, seed: u64, stats: &mut BTreeMap<String, usize>) -> Vec<Mutant> {
    let mut count = |k: &str| *stats.entry(k.to_string()).or_default() += 1;
    let Ok(puzzle) = tokens::decode(parent) else {
        count("parent unparsed");
        return Vec::new();
    };
    let Some((puzzle, parent_tokens)) = normalize(&puzzle) else {
        count("parent invalid");
        return Vec::new();
    };
    let mut rng = Rng::new(seed);
    let mut seen: HashSet<Vec<String>> = HashSet::from([parent_tokens]);
    let mut out = Vec::new();
    for t in 0..cfg.tries {
        if out.len() == cfg.per_level {
            break;
        }
        if out.is_empty() && cfg.give_up > 0 && t == cfg.give_up {
            count("gave up");
            break;
        }
        count("tries");
        let n = rng.range(1, 3) as usize;
        let Some((child, edits)) = mutate(&puzzle, &mut rng, n) else {
            count("no valid edit");
            continue;
        };
        let Ok(toks) = tokens::encode(&child) else {
            count("no valid edit");
            continue;
        };
        if toks.len() > cfg.max_tokens {
            count("too long");
            continue;
        }
        if !seen.insert(toks.clone()) {
            count("same as parent or sibling");
            continue;
        }
        let label = label_portal_solvable(&child, cfg.budget, rng.next_u64(), false);
        match &label {
            Label::Solvable {
                needs_portals: true,
                ..
            } => count("kept"),
            Label::Solvable { .. } => {
                count("solvable without portals");
                continue;
            }
            other => {
                count(other.verdict());
                continue;
            }
        }
        out.push(Mutant {
            tokens: toks,
            size: child.chamber_size(),
            label,
            edits,
        });
    }
    out
}

/// `parent` after `n` random edits, each one that left every item mounted, and what they
/// were; `None` if an edit found nothing valid to do. The result is normalized (decoded from
/// its own tokens).
pub fn mutate(parent: &Puzzle, rng: &mut Rng, n: usize) -> Option<(Puzzle, Vec<String>)> {
    let mut p = parent.clone();
    let mut edits = Vec::new();
    for _ in 0..n {
        let (next, what) = (0..EDIT_TRIES).find_map(|_| {
            let (edited, what) = edit(&p, rng)?;
            let (normal, _) = normalize(&edited)?;
            Some((normal, what))
        })?;
        p = next;
        edits.push(what);
    }
    Some((p, edits))
}

/// The puzzle decoded from its own tokens, and the tokens, if it encodes and every item is
/// mounted.
pub fn normalize(p: &Puzzle) -> Option<(Puzzle, Vec<String>)> {
    let toks = tokens::encode(p).ok()?;
    let decoded = tokens::decode(&toks.join(" ")).ok()?;
    if !Level::from_puzzle(&decoded).mount_problems_with(false).is_empty() {
        return None;
    }
    Some((decoded, toks))
}

/// One random edit (maybe an invalid one) and its description.
fn edit(p: &Puzzle, rng: &mut Rng) -> Option<(Puzzle, String)> {
    match rng.below(20) {
        0..=4 => move_group(p, rng, false),
        5..=6 => move_group(p, rng, true),
        7..=10 => add_item(p, rng),
        11..=13 => remove_group(p, rng),
        14..=16 => repaint(p, rng),
        _ => push_wall(p, rng),
    }
}

/// Items that belong together, by index: those joined by anything but a wire (a dropper and
/// its cube, a panel or fizzler and its extents, a door and its coop twin, a faith plate and
/// its target, a gel dropper and its paint).
fn groups(p: &Puzzle) -> Vec<Vec<i32>> {
    let ids: Vec<i32> = p.items.iter().filter_map(|i| i.index()).collect();
    let mut parent: HashMap<i32, i32> = ids.iter().map(|&i| (i, i)).collect();
    fn root(parent: &HashMap<i32, i32>, mut i: i32) -> i32 {
        while parent[&i] != i {
            i = parent[&i];
        }
        i
    }
    for c in &p.connections {
        if matches!(
            c.get("Type"),
            Some("CONNECTION_STANDARD" | "CONNECTION_TBEAM_POLARITY")
        ) {
            continue;
        }
        let (Some(a), Some(b)) = (c.get_i32("Sender"), c.get_i32("Receiver")) else {
            continue;
        };
        if !parent.contains_key(&a) || !parent.contains_key(&b) {
            continue;
        }
        let (ra, rb) = (root(&parent, a), root(&parent, b));
        parent.insert(ra, rb);
    }
    let mut by_root: BTreeMap<i32, Vec<i32>> = BTreeMap::new();
    for &i in &ids {
        by_root.entry(root(&parent, i)).or_default().push(i);
    }
    by_root.into_values().collect()
}

fn kind_of(p: &Puzzle, index: i32) -> &str {
    p.item(index).map_or("", |i| i.kind())
}

/// A short name for a group: its main item's type, lower case without `ITEM_`.
fn group_name(p: &Puzzle, g: &[i32]) -> String {
    const MINOR: [&str; 4] = [
        "ITEM_COOP_ENTRY_DOOR",
        "ITEM_COOP_EXIT_DOOR",
        "ITEM_BARRIER_EXTENT",
        "ITEM_BARRIER_HAZARD_EXTENT",
    ];
    let main = g
        .iter()
        .map(|&i| kind_of(p, i))
        .find(|k| !MINOR.contains(k))
        .unwrap_or_else(|| kind_of(p, g[0]));
    main.trim_start_matches("ITEM_").to_lowercase()
}

fn is_door(p: &Puzzle, g: &[i32]) -> bool {
    g.iter()
        .any(|&i| matches!(kind_of(p, i), "ITEM_ENTRY_DOOR" | "ITEM_EXIT_DOOR"))
}

fn is_goo(p: &Puzzle, g: &[i32]) -> bool {
    g.iter().any(|&i| kind_of(p, i) == "ITEM_GOO")
}

/// Moves a random item group (a door if `doors`) a voxel or few.
fn move_group(p: &Puzzle, rng: &mut Rng, doors: bool) -> Option<(Puzzle, String)> {
    let groups: Vec<Vec<i32>> = groups(p)
        .into_iter()
        .filter(|g| !is_goo(p, g) && is_door(p, g) == doors)
        .collect();
    if groups.is_empty() {
        return None;
    }
    let g = rng.pick(&groups).clone();
    let mut delta = [0; 3];
    let axis = if doors || rng.chance(0.8) {
        rng.below(2)
    } else {
        2
    };
    let dist = if rng.chance(0.6) { 1 } else { rng.range(2, 3) };
    delta[axis] = if rng.chance(0.5) { dist } else { -dist };
    let mut q = p.clone();
    for it in q.items.iter_mut() {
        if !it.index().is_some_and(|i| g.contains(&i)) {
            continue;
        }
        let c = it.voxel_pos()?;
        let moved = [c[0] + delta[0], c[1] + delta[1], c[2] + delta[2]];
        // `Level::from_puzzle` expects items inside the chamber.
        let size = p.chamber_size();
        if (0..3).any(|a| moved[a] < 0 || moved[a] >= size[a]) {
            return None;
        }
        it.set("VoxelPos", format!("{} {} {}", moved[0], moved[1], moved[2]));
    }
    let what = format!(
        "move {} by {} {} {}",
        group_name(p, &g),
        delta[0],
        delta[1],
        delta[2]
    );
    Some((q, what))
}

/// Removes a random item group other than a door or goo, and its wires.
fn remove_group(p: &Puzzle, rng: &mut Rng) -> Option<(Puzzle, String)> {
    let groups: Vec<Vec<i32>> = groups(p)
        .into_iter()
        .filter(|g| !is_goo(p, g) && !is_door(p, g))
        .collect();
    if groups.is_empty() {
        return None;
    }
    let g = rng.pick(&groups).clone();
    let gone = |i: Option<i32>| i.is_some_and(|i| g.contains(&i));
    let mut q = p.clone();
    q.items.retain(|it| !gone(it.index()));
    q.connections
        .retain(|c| !gone(c.get_i32("Sender")) && !gone(c.get_i32("Receiver")));
    Some((q, format!("remove {}", group_name(p, &g))))
}

fn solid(level: &Level, c: Cell) -> bool {
    level.solid(c)
}

/// Adds a cube dropper, loose cube, floor button, pedestal, fizzler or turret, written as a
/// token clause appended to the level's tokens (so new targets don't renumber old ones).
fn add_item(p: &Puzzle, rng: &mut Rng) -> Option<(Puzzle, String)> {
    let level = Level::from_puzzle(p);
    let open: Vec<Cell> = level
        .cells()
        .filter(|&c| !solid(&level, c) && !level.goo(c))
        .collect();
    if open.is_empty() {
        return None;
    }
    let c = *rng.pick(&open);
    let floor = |mut c: Cell| {
        while !solid(&level, step(c, Dir::NegZ)) {
            c = step(c, Dir::NegZ);
        }
        c
    };
    let xyz = |c: Cell| format!("{} {} {}", c[0], c[1], c[2]);
    let fields = level.hazards.len();
    let target = |rng: &mut Rng| {
        if fields > 0 && (level.exit().is_none() || rng.chance(0.25)) {
            format!("field {}", rng.below(fields))
        } else {
            "exit".to_string()
        }
    };
    let cube = |rng: &mut Rng| *rng.pick(&["standard", "standard", "companion", "sphere", "reflection"]);
    let clause = match rng.below(6) {
        0 => {
            let mut top = c;
            while !solid(&level, step(top, Dir::PosZ)) {
                top = step(top, Dir::PosZ);
            }
            format!("dropper {} {}", xyz(top), cube(rng))
        }
        1 => format!("cube {} {}", xyz(floor(c)), cube(rng)),
        2 => {
            let kind = *rng.pick(&["weighted", "weighted", "weighted", "cube", "sphere"]);
            format!("button {} {kind} > {}", xyz(floor(c)), target(rng))
        }
        3 => {
            let timer = if rng.chance(0.5) {
                "inf".to_string()
            } else {
                rng.range(3, 15).to_string()
            };
            format!("pedestal {} {timer} > {}", xyz(floor(c)), target(rng))
        }
        4 => {
            // Floor to ceiling, on a wall beside `c`.
            let n = *rng.pick(&Dir::HORIZONTAL);
            let base = floor(c);
            if !solid(&level, step(base, n.opposite())) {
                return None;
            }
            let mut len = 1;
            while !solid(&level, [base[0], base[1], base[2] + len]) {
                len += 1;
            }
            let kind = if rng.chance(0.8) { "fizzler" } else { "laserfield" };
            format!(
                "{kind} {} {} +z {len} on",
                xyz(base),
                tokens_dir(n)
            )
        }
        _ => format!("turret {} {}", xyz(floor(c)), rng.below(8)),
    };
    let mut text = tokens::encode(p).ok()?.join(" ");
    text.push(' ');
    text.push_str(&clause);
    let q = tokens::decode(&text).ok()?;
    Some((q, format!("add {clause}")))
}

fn tokens_dir(d: Dir) -> &'static str {
    match d {
        Dir::PosX => "+x",
        Dir::NegX => "-x",
        Dir::PosY => "+y",
        Dir::NegY => "-y",
        Dir::PosZ => "+z",
        Dir::NegZ => "-z",
    }
}

/// A random surface: an open cell and a direction toward a solid one.
fn random_surface(level: &Level, rng: &mut Rng) -> Option<(Cell, Dir)> {
    let surfaces: Vec<(Cell, Dir)> = level
        .cells()
        .filter(|&c| !solid(level, c))
        .flat_map(|c| Dir::ALL.map(|d| (c, d)))
        .filter(|&(c, d)| solid(level, step(c, d)))
        .collect();
    (!surfaces.is_empty()).then(|| *rng.pick(&surfaces))
}

/// Cells of a random rectangle around `c` in the plane across `d`, up to 5 × 5, with their
/// offsets.
fn patch(c: Cell, d: Dir, rng: &mut Rng) -> (Vec<Cell>, [i32; 2]) {
    let axes: Vec<usize> = (0..3).filter(|&a| a != d.axis()).collect();
    let r = [rng.range(0, 2), rng.range(0, 2)];
    let mut cells = Vec::new();
    for i in -r[0]..=r[0] {
        for j in -r[1]..=r[1] {
            let mut q = c;
            q[axes[0]] += i;
            q[axes[1]] += j;
            cells.push(q);
        }
    }
    (cells, [2 * r[0] + 1, 2 * r[1] + 1])
}

/// Paints a patch of one surface white or black (the opposite of its colour at the middle).
fn repaint(p: &Puzzle, rng: &mut Rng) -> Option<(Puzzle, String)> {
    let level = Level::from_puzzle(p);
    let (c, d) = random_surface(&level, rng)?;
    let white = !p.voxels.face_portalable(c, d);
    let (cells, dims) = patch(c, d, rng);
    let mut q = p.clone();
    for f in cells {
        if level.in_interior(f) && !solid(&level, f) && solid(&level, step(f, d)) {
            q.voxels.set_face_portalable(f, d, white);
        }
    }
    let colour = if white { "white" } else { "black" };
    let what = format!(
        "paint {} {}x{} {colour} at {} {} {}",
        tokens_dir(d),
        dims[0],
        dims[1],
        c[0],
        c[1],
        c[2]
    );
    Some((q, what))
}

/// Pushes a patch of a surface 1–2 voxels back (carving the room bigger there) or pulls it
/// in (filling). New faces copy the colour of the face they replace; carved side walls copy
/// the sides they extend.
fn push_wall(p: &Puzzle, rng: &mut Rng) -> Option<(Puzzle, String)> {
    let level = Level::from_puzzle(p);
    let (c, d) = random_surface(&level, rng)?;
    let (cells, dims) = patch(c, d, rng);
    let depth = rng.range(1, 2);
    let out = rng.chance(0.5);
    let face = |c: Cell, d: Dir| p.voxels.face_portalable(c, d);
    let mut q = p.clone();
    let mut paints: Vec<(Cell, Dir, bool)> = Vec::new();
    let mut changed = 0;
    for f in cells {
        if !level.in_interior(f) || solid(&level, f) || !solid(&level, step(f, d)) {
            continue;
        }
        let back = face(f, d);
        if out {
            let mut n = f;
            for _ in 0..depth {
                n = step(n, d);
                if !level.in_interior(n) {
                    break;
                }
                q.voxels.set_solid(n, false);
                changed += 1;
                for s in Dir::ALL {
                    if s == d.opposite() {
                        continue;
                    }
                    let colour = if s != d && solid(&level, step(f, s)) {
                        face(f, s)
                    } else {
                        back
                    };
                    paints.push((n, s, colour));
                }
            }
        } else {
            let mut n = f;
            for _ in 0..depth {
                if solid(&level, n) {
                    break;
                }
                q.voxels.set_solid(n, true);
                changed += 1;
                for s in Dir::ALL {
                    paints.push((n, s, back));
                }
                n = step(n, d.opposite());
            }
        }
    }
    if changed == 0 {
        return None;
    }
    // Only faces that are now surfaces matter (`encode` drops the rest), but a face may be
    // listed twice: one between two new cells is not a surface, so order doesn't matter.
    for (n, s, colour) in paints {
        let qs = |c: Cell| !level.in_interior(c) || q.voxels.solid(c).unwrap_or(true);
        if qs(n) != qs(step(n, s)) {
            q.voxels.set_face_portalable(n, s, colour);
        }
    }
    let what = format!(
        "{} {} {}x{} by {depth} at {} {} {}",
        if out { "push" } else { "pull" },
        tokens_dir(d),
        dims[0],
        dims[1],
        c[0],
        c[1],
        c[2]
    );
    Some((q, what))
}

/// The value of `"key"` in a JSON object line: a string (unescaped) or a bare scalar.
/// Enough for the flat records our data files hold.
pub fn json_field(line: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let mut at = 0;
    while let Some(i) = line[at..].find(&pat) {
        let rest = line[at + i + pat.len()..].trim_start();
        at += i + pat.len();
        let Some(rest) = rest.strip_prefix(':') else {
            continue;
        };
        let rest = rest.trim_start();
        if rest.starts_with('"') {
            return json_string(rest).map(|(s, _)| s);
        }
        let end = rest.find([',', '}']).unwrap_or(rest.len());
        return Some(rest[..end].trim().to_string());
    }
    None
}

/// The strings of a JSON array value, `"key": ["a", "b", ...]`, unescaped.
pub fn json_array(line: &str, key: &str) -> Option<Vec<String>> {
    let pat = format!("\"{key}\"");
    let i = line.find(&pat)?;
    let rest = line[i + pat.len()..].trim_start();
    let mut rest = rest.strip_prefix(':')?.trim_start();
    rest = rest.strip_prefix('[')?.trim_start();
    let mut out = Vec::new();
    loop {
        if rest.starts_with(']') {
            return Some(out);
        }
        let (s, after) = json_string(rest)?;
        out.push(s);
        rest = after.trim_start();
        if let Some(after) = rest.strip_prefix(',') {
            rest = after.trim_start();
        }
    }
}

/// A JSON string at the start of `s` (which starts with `"`): its unescaped value and the rest
/// of `s` after the closing quote.
fn json_string(s: &str) -> Option<(String, &str)> {
    let s = s.strip_prefix('"')?;
    let mut out = String::new();
    let mut chars = s.char_indices();
    while let Some((i, ch)) = chars.next() {
        match ch {
            '"' => return Some((out, &s[i + 1..])),
            '\\' => match chars.next()?.1 {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                'r' => out.push('\r'),
                'u' => {
                    let mut hex = String::new();
                    for _ in 0..4 {
                        hex.push(chars.next()?.1);
                    }
                    out.push(char::from_u32(u32::from_str_radix(&hex, 16).ok()?)?);
                }
                c => out.push(c),
            },
            c => out.push(c),
        }
    }
    None
}
