//! Levels as flat token sequences, for training sequence models (plan.md, "Representation").
//!
//! Tokens are whitespace-free words; a level is written with single spaces between them.
//! Coordinates and counts are the words `0`..`32`. Directions are `+x -x +y -y +z -z`.
//!
//! ```text
//! size X Y Z
//! <row>*               one per y: columns (x = 0, 1, ...) then `/`. A column is its runs
//!                      bottom-up then `|`, or `=` (same as the column before it, also across
//!                      rows). Runs: `s<n>` solid, `o<n>` open, `g<n>` goo; they add up to Z.
//!                      Columns left out at the end of a row repeat its last one; a row that
//!                      is just `/` repeats the row before, and rows left out at the end
//!                      repeat the last one. So only changes cost tokens, and the model
//!                      never has to count columns or rows exactly.
//! white D x0 y0 z0 x1 y1 z1        the D faces of open cells in the box are white surfaces
//! entry x y z D                    D = the door's normal (into the room)
//! exit x y z D [inv]               inv = starts open, closes when its inputs are on
//! glass|grating x0 y0 z0 x1 y1 z1 N G
//!                                  panel on the N faces of the cells in the box, hung from
//!                                  the surface behind it against G
//! fizzler|laserfield x y z N A L on|off
//!                                  mounted on the wall against N, L cells from (x, y, z)
//!                                  along A; the field is on the N × A side of each cell
//! dropper x y z K                  ceiling cube dropper, its cube on the floor below.
//!                                  K: standard companion reflection sphere franken
//! button x y z K [> T]*            floor button. K: weighted cube sphere
//! pedestal x y z S|inf [> T]*      pedestal button with an S-second timer, or latching
//! gel x y z N K stream on|off      gel K (bounce speed portal water) poured onto the floor of
//!                                  (x, y, z) by a dropper in the ceiling above, pouring from
//!                                  the start or not; N is +z
//! gel x y z N K splat              gel K painted from the start on the surface against N
//! turret x y z F                   turret on the floor, facing F × 45° (0 = +x, 2 = +y)
//! cube x y z K                     a cube placed without a dropper, resting in (x, y, z)
//! laser x y z N O on|off           laser emitter on the surface against N, firing along N;
//!                                  O: `center`, or the side of the face it's offset toward
//! catcher|relay x y z N O [> T]*   laser catcher or relay, driving its targets while powered
//! plate x y z N x y z N            faith plate on the surface against N, and its target
//! bridge x y z N S on|off          light bridge projected along N, walkable side S
//! funnel x y z N on|off [rev]      excursion funnel along N; rev = pulling back at the start
//! panel|glasspanel x y z N F A on|off
//!                                  angled panel on the face behind (x, y, z) (against N),
//!                                  tilting toward F by A × 15° (2, 3, 4 or 6) when deployed,
//!                                  deployed from the start or not; white if its face is
//! flip x y z N                     flip panel on the face behind (x, y, z) (against N)
//! stairs x y z F on|off            stairs on the floors of (x, y, z) and the next cell along F,
//!                                  rising toward F
//! piston x y z N B T on|off        piston platform pushing out of the surface behind (x, y, z)
//!                                  along N: B or T cells out, up (on) or down at the start
//! rail x y z x y z osc|on|off      rail platform travelling between the two cells (along one
//!                                  axis): back and forth, or to the far one while active (on:
//!                                  from the start)
//! ```
//! A dropper may end in `wait`: it drops its cube only when its inputs are first on.
//! Targets `T`: `exit`, `field N` (the N-th fizzler/laser field in the sequence), `gel N`
//! (the N-th gel, which must be poured), or the N-th `laser`, `bridge`, `funnel`, `panel`,
//! `glasspanel`, `flip`, `stairs`, `piston`, `rail` or `dropper`; `polarity N` reverses the
//! N-th funnel.
//!
//! `encode` writes items in a fixed order (white, doors, panels, fields, gels, droppers,
//! buttons, pedestals, turrets, then the later items in the order above; each sorted), so
//! `encode(decode(encode(p))) == encode(p)`. The encoding keeps
//! what play depends on (the `Level` view), not the file's bytes: decorative items, property
//! order and face bits that aren't on a surface are dropped. `decode` accepts items in any
//! order and rejects anything malformed with an error instead of panicking, so it can read
//! model output.

use std::collections::{BTreeSet, HashSet};

use crate::builder::Builder;
use crate::generator::TITLE;
use crate::geom::{rotate, step, Cell, Dir};
use crate::level::{
    ButtonKind, CubeKind, HazardKind, ItemKind, LaserKind, Level, PaintKind, PanelKind,
    PlacedItem, ProjectorKind,
};
use crate::p2c::Puzzle;
use crate::solver::DECORATIVE;

/// Largest coordinate, size or count a token can hold.
pub const MAX_NUM: i32 = 32;

const CUBE_KINDS: [&str; 5] = ["standard", "companion", "reflection", "sphere", "franken"];
const BUTTON_KINDS: [&str; 3] = ["weighted", "cube", "sphere"];
const PAINT_KINDS: [&str; 4] = ["bounce", "speed", "portal", "water"];
/// Words added after the first models were trained: they go at the end of the vocabulary,
/// so earlier tokens keep their ids.
const LATER_WORDS: [&str; 4] = ["turret", "gel", "stream", "splat"];
/// Words for the items added after turrets and gel, after those (`cube` was already there,
/// as a button kind).
const DEVICE_WORDS: [&str; 17] = [
    "laser",
    "catcher",
    "relay",
    "center",
    "plate",
    "bridge",
    "funnel",
    "rev",
    "panel",
    "glasspanel",
    "flip",
    "stairs",
    "piston",
    "rail",
    "osc",
    "polarity",
    "wait",
];
/// Wire targets after `exit`, in the order `encode` writes them.
const TARGETS: [&str; 13] = [
    "field",
    "gel",
    "laser",
    "bridge",
    "funnel",
    "polarity",
    "panel",
    "glasspanel",
    "flip",
    "stairs",
    "piston",
    "rail",
    "dropper",
];
const DIRS: [(Dir, &str); 6] = [
    (Dir::PosX, "+x"),
    (Dir::NegX, "-x"),
    (Dir::PosY, "+y"),
    (Dir::NegY, "-y"),
    (Dir::PosZ, "+z"),
    (Dir::NegZ, "-z"),
];
const KEYWORDS: [&str; 20] = [
    "size",
    "|",
    "/",
    "=",
    "white",
    "entry",
    "exit",
    "inv",
    "glass",
    "grating",
    "fizzler",
    "laserfield",
    "on",
    "off",
    "dropper",
    "button",
    "pedestal",
    "inf",
    ">",
    "field",
];

/// Every token `encode` can write, in a fixed order.
pub fn vocab() -> Vec<String> {
    let mut v: Vec<String> = KEYWORDS.iter().map(|s| s.to_string()).collect();
    v.extend((0..=MAX_NUM).map(|n| n.to_string()));
    for run in ["s", "o", "g"] {
        v.extend((1..=MAX_NUM).map(|n| format!("{run}{n}")));
    }
    v.extend(DIRS.iter().map(|(_, s)| s.to_string()));
    for k in CUBE_KINDS.iter().chain(&BUTTON_KINDS) {
        if !v.iter().any(|t| t == k) {
            v.push(k.to_string());
        }
    }
    v.extend(LATER_WORDS.iter().chain(&PAINT_KINDS).map(|s| s.to_string()));
    v.extend(DEVICE_WORDS.iter().map(|s| s.to_string()));
    v
}

fn dir_token(d: Dir) -> &'static str {
    DIRS.iter().find(|(x, _)| *x == d).unwrap().1
}

fn num(n: i32) -> String {
    n.to_string()
}

fn cell_tokens(c: Cell) -> [String; 3] {
    c.map(num)
}

/// The level as tokens, or why it can't be written (items the format doesn't cover, or
/// layouts it can't express).
pub fn encode(p: &Puzzle) -> Result<Vec<String>, String> {
    let level = Level::from_puzzle(p);
    if let Some(problem) = level.decode_problems.first() {
        return Err(problem.clone());
    }
    let [sx, sy, sz] = level.size;
    if level.size.iter().any(|&s| s > MAX_NUM) {
        return Err(format!("chamber {:?} is bigger than {MAX_NUM}", level.size));
    }
    for it in &level.items {
        if let ItemKind::Other(t) = &it.kind {
            if !DECORATIVE.contains(&t.as_str()) {
                return Err(format!("unsupported item {t} (item {})", it.index));
            }
        }
    }
    let mut out: Vec<String> = vec!["size".into()];
    out.extend(cell_tokens(level.size));

    let column = |x, y| -> Vec<String> {
        let mut runs: Vec<(char, i32)> = Vec::new();
        for z in 0..sz {
            let c = [x, y, z];
            let k = if level.solid(c) {
                's'
            } else if level.goo(c) {
                'g'
            } else {
                'o'
            };
            match runs.last_mut() {
                Some((last, n)) if *last == k => *n += 1,
                _ => runs.push((k, 1)),
            }
        }
        runs.iter().map(|(k, n)| format!("{k}{n}")).collect()
    };
    let rows: Vec<Vec<Vec<String>>> = (0..sy)
        .map(|y| (0..sx).map(|x| column(x, y)).collect())
        .collect();
    // Rows after the last change repeat it; the decoder fills them in.
    let last_change = (1..rows.len())
        .rev()
        .find(|&y| rows[y] != rows[y - 1])
        .unwrap_or(0);
    let mut prev_col: Option<Vec<String>> = None;
    let mut prev_row: Option<&Vec<Vec<String>>> = None;
    for row in &rows[..=last_change] {
        if prev_row != Some(row) {
            // Columns after the last change repeat it; `/` fills them in.
            let keep = (1..row.len())
                .rev()
                .find(|&x| row[x] != row[x - 1])
                .map_or(1, |x| x + 1);
            for col in &row[..keep] {
                if prev_col.as_ref() == Some(col) {
                    out.push("=".into());
                } else {
                    out.extend(col.iter().cloned());
                    out.push("|".into());
                }
                prev_col = Some(col.clone());
            }
        }
        out.push("/".into());
        prev_col = row.last().cloned();
        prev_row = Some(row);
    }

    for (d, lo, hi) in white_rects(&level) {
        out.push("white".into());
        out.push(dir_token(d).into());
        out.extend(cell_tokens(lo));
        out.extend(cell_tokens(hi));
    }

    let single = |kind: ItemKind, name: &str| -> Result<Option<&PlacedItem>, String> {
        let mut doors = level.items.iter().filter(|i| i.kind == kind);
        let door = doors.next();
        if doors.next().is_some() {
            return Err(format!("more than one {name} door"));
        }
        Ok(door)
    };
    let normal = |it: &PlacedItem| {
        it.normal
            .ok_or_else(|| format!("item {} {:?}: angles not axis-aligned", it.index, it.kind))
    };
    let exit = single(ItemKind::ExitDoor, "exit")?;
    if let Some(e) = single(ItemKind::EntryDoor, "entry")? {
        out.push("entry".into());
        out.extend(cell_tokens(e.cell));
        out.push(dir_token(normal(e)?).into());
    }
    if let Some(e) = exit {
        out.push("exit".into());
        out.extend(cell_tokens(e.cell));
        out.push(dir_token(normal(e)?).into());
        if e.start_open == Some(true) && level.inputs(e.index).next().is_some() {
            out.push("inv".into());
        }
    }

    let axis = |it: &PlacedItem, v: [i32; 3]| {
        it.angles
            .and_then(|a| rotate(a, v))
            .ok_or_else(|| format!("item {} {:?}: angles not axis-aligned", it.index, it.kind))
    };
    let mut panels = Vec::new();
    for b in &level.barriers {
        let it = level.item(b.item).expect("barriers come from items");
        let (lo, hi) = extent_box(p, &level, it);
        let kind = if it.kind == ItemKind::Glass {
            "glass"
        } else {
            "grating"
        };
        let mut t = vec![kind.to_string()];
        t.extend(cell_tokens(lo));
        t.extend(cell_tokens(hi));
        t.push(dir_token(axis(it, [-1, 0, 0])?).into());
        t.push(dir_token(axis(it, [0, 0, 1])?).into());
        panels.push(((lo, hi), t));
    }
    panels.sort();
    out.extend(panels.into_iter().flat_map(|(_, t)| t));

    // Fields, sorted; buttons refer to them by position.
    let mut fields = Vec::new();
    for h in &level.hazards {
        let it = level.item(h.item).expect("hazards come from items");
        let (n, along) = (axis(it, [0, 0, 1])?, axis(it, [0, 1, 0])?);
        let (lo, hi) = extent_box(p, &level, it);
        let anchor = if along.is_positive() { lo } else { hi };
        let len = hi[along.axis()] - lo[along.axis()] + 1;
        let kind = match h.kind {
            HazardKind::Fizzler => "fizzler",
            HazardKind::LaserField => "laserfield",
        };
        let mut t = vec![kind.to_string()];
        t.extend(cell_tokens(anchor));
        t.push(dir_token(n).into());
        t.push(dir_token(along).into());
        t.push(num(len));
        t.push(if h.start_enabled { "on" } else { "off" }.into());
        fields.push(((anchor, n, along, len), t, h.item));
    }
    fields.sort();
    let field_items: Vec<i32> = fields.iter().map(|f| f.2).collect();
    out.extend(fields.into_iter().flat_map(|(_, t, _)| t));

    // Gels, sorted; buttons refer to poured ones by position.
    let mut gels = Vec::new();
    for g in &level.gels {
        let (cell, d) = g.face;
        let n = d.opposite();
        let mut t = vec!["gel".to_string()];
        t.extend(cell_tokens(cell));
        t.push(dir_token(n).into());
        t.push(PAINT_KINDS[paint_index(g.kind)].into());
        match g.dropper {
            Some((dropper, at)) => {
                let mut top = cell;
                while !level.solid(step(top, Dir::PosZ)) {
                    top = step(top, Dir::PosZ);
                }
                if at != top {
                    return Err(format!(
                        "gel dropper {dropper} isn't in the ceiling above its splat"
                    ));
                }
                t.push("stream".into());
                t.push(if g.start_enabled { "on" } else { "off" }.into());
            }
            None => t.push("splat".into()),
        }
        gels.push(((cell, n), t, g.dropper.map(|(d, _)| d)));
    }
    gels.sort();
    let gel_items: Vec<Option<i32>> = gels.iter().map(|g| g.2).collect();
    out.extend(gels.into_iter().flat_map(|(_, t, _)| t));

    // Droppers carry their cube; every cube must sit under one.
    let mut droppers = Vec::new();
    for d in level.of_kind(&ItemKind::CubeDropper) {
        if normal(d)? != Dir::NegZ {
            return Err(format!("dropper {} isn't on a ceiling", d.index));
        }
        droppers.push(d);
    }
    let mut dropped = Vec::new();
    let mut loose = Vec::new();
    for cube in level.of_kind(&ItemKind::Cube) {
        if level.loose_cube(cube.index) {
            loose.push((cube.cell, cube.cube_kind()));
            continue;
        }
        let dropper = p
            .connections
            .iter()
            .filter(|c| c.get("Type") == Some("CONNECTION_BOX_DROPPER"))
            .filter(|c| c.get_i32("Sender") == Some(cube.index))
            .find_map(|c| {
                droppers
                    .iter()
                    .find(|d| Some(d.index) == c.get_i32("Receiver"))
            })
            .ok_or_else(|| format!("cube {} has no dropper", cube.index))?;
        let mut floor = dropper.cell;
        while !level.solid(step(floor, Dir::NegZ)) {
            floor = step(floor, Dir::NegZ);
        }
        if cube.cell != floor {
            return Err(format!(
                "cube {} isn't on the floor under its dropper",
                cube.index
            ));
        }
        dropped.push((dropper.cell, cube.cube_kind(), dropper.index));
    }
    if dropped.len() != droppers.len() {
        return Err("a dropper has no cube, or two".into());
    }
    dropped.sort();
    let waits = |d: i32| {
        level
            .item(d)
            .is_some_and(|d| d.props.get_i32("ITEM_PROPERTY_AUTO_DROP_CUBE") == Some(0))
    };
    let dropper_items: Vec<i32> = dropped.iter().map(|d| d.2).collect();
    for (cell, kind, d) in dropped {
        out.push("dropper".into());
        out.extend(cell_tokens(cell));
        out.push(CUBE_KINDS[cube_index(kind)].into());
        if waits(d) {
            out.push("wait".into());
        }
    }

    // The later items, as tokens with what sorts them, and their items in that order.
    let later = later_items(&level, &loose)?;
    // What each wire can drive: (target kind: index into `TARGETS` + 1, 0 = exit; position).
    let mut receivers: std::collections::HashMap<i32, (usize, usize)> = Default::default();
    if let Some(e) = exit {
        receivers.insert(e.index, (0, 0));
    }
    for (i, &f) in field_items.iter().enumerate() {
        receivers.insert(f, (1, i));
    }
    for (i, g) in gel_items.iter().enumerate() {
        if let Some(g) = g {
            receivers.insert(*g, (2, i));
        }
    }
    for (kind, items) in &later.receivers {
        let k = 1 + TARGETS.iter().position(|t| t == kind).unwrap();
        for (i, &item) in items.iter().enumerate() {
            receivers.insert(item, (k, i));
        }
    }
    for (i, &d) in dropper_items.iter().enumerate() {
        receivers.insert(d, (TARGETS.len(), i));
    }
    let polarity: Vec<(i32, (usize, usize))> = later
        .receivers
        .iter()
        .filter(|(k, _)| *k == "funnel")
        .flat_map(|(_, items)| items.iter().enumerate())
        .map(|(i, &f)| (f, (1 + TARGETS.iter().position(|t| *t == "polarity").unwrap(), i)))
        .collect();
    // Every wire must run from a sender to something that takes input.
    for l in &level.links {
        let from = level.item(l.sender).map(|i| &i.kind);
        if !from.is_some_and(ItemKind::is_sender) || !receivers.contains_key(&l.receiver) {
            return Err(format!(
                "wire {} -> {} isn't sender -> receiver",
                l.sender, l.receiver
            ));
        }
    }
    for l in &level.polarity_links {
        let from = level.item(l.sender).map(|i| &i.kind);
        if !from.is_some_and(ItemKind::is_sender) || !polarity.iter().any(|p| p.0 == l.receiver) {
            return Err(format!("polarity wire {} -> {}", l.sender, l.receiver));
        }
    }
    let targets = |sender: i32| -> Vec<String> {
        let mut ts: BTreeSet<(usize, usize)> = BTreeSet::new();
        for l in level.links.iter().filter(|l| l.sender == sender) {
            ts.insert(receivers[&l.receiver]);
        }
        for l in level.polarity_links.iter().filter(|l| l.sender == sender) {
            ts.insert(polarity.iter().find(|p| p.0 == l.receiver).unwrap().1);
        }
        ts.into_iter()
            .flat_map(|(kind, i)| match kind {
                0 => vec![">".to_string(), "exit".into()],
                k => vec![">".to_string(), TARGETS[k - 1].into(), num(i as i32)],
            })
            .collect()
    };

    let mut buttons = Vec::new();
    for b in level.of_kind(&ItemKind::FloorButton) {
        if normal(b)? != Dir::PosZ {
            return Err(format!("button {} isn't on a floor", b.index));
        }
        let kind = match b.button_kind() {
            ButtonKind::Weighted => 0,
            ButtonKind::Cube => 1,
            ButtonKind::Sphere => 2,
        };
        let mut t = vec!["button".to_string()];
        t.extend(cell_tokens(b.cell));
        t.push(BUTTON_KINDS[kind].into());
        t.extend(targets(b.index));
        buttons.push((b.cell, t));
    }
    buttons.sort();
    out.extend(buttons.into_iter().flat_map(|(_, t)| t));

    let mut pedestals = Vec::new();
    for ped in level.of_kind(&ItemKind::PedestalButton) {
        if normal(ped)? != Dir::PosZ {
            return Err(format!("pedestal {} isn't on a floor", ped.index));
        }
        let timer = match ped.timer() {
            None => "inf".to_string(),
            Some(s) if s.round() <= 30.0 => num(s.round() as i32),
            Some(s) => return Err(format!("pedestal {}: timer {s} too long", ped.index)),
        };
        let mut t = vec!["pedestal".to_string()];
        t.extend(cell_tokens(ped.cell));
        t.push(timer);
        t.extend(targets(ped.index));
        pedestals.push((ped.cell, t));
    }
    pedestals.sort();
    out.extend(pedestals.into_iter().flat_map(|(_, t)| t));

    let mut turrets = Vec::new();
    for t in &level.turrets {
        let yaw = t.yaw.rem_euclid(360.0);
        let f = (yaw / 45.0).round();
        if (yaw - f * 45.0).abs() > 1e-6 {
            return Err(format!("turret {} faces {yaw}°, not a multiple of 45", t.item));
        }
        let mut tok = vec!["turret".to_string()];
        tok.extend(cell_tokens(t.cell));
        tok.push(num(f as i32 % 8));
        turrets.push(tok);
    }
    turrets.sort();
    out.extend(turrets.into_iter().flatten());

    for (tokens, sender) in later.tokens {
        out.extend(tokens);
        if let Some(s) = sender {
            out.extend(targets(s));
        }
    }
    Ok(out)
}

/// The items added after turrets and gel, ready to write.
struct Later {
    /// Each item's tokens (without wires), in order, and the item to write the wires of.
    tokens: Vec<(Vec<String>, Option<i32>)>,
    /// Per target word: its items in the order written.
    receivers: Vec<(&'static str, Vec<i32>)>,
}

fn later_items(level: &Level, loose: &[(crate::geom::Cell, CubeKind)]) -> Result<Later, String> {
    let dir = |d: Dir| dir_token(d).to_string();
    let on = |b: bool| if b { "on" } else { "off" }.to_string();
    // (sort key = the tokens, item, whether it sends) per section.
    type Section = Vec<(Vec<String>, i32, bool)>;
    let mut sections: Vec<(&'static str, Section)> = Vec::new();
    let mut cubes: Section = loose
        .iter()
        .map(|&(c, k)| {
            let mut t = vec!["cube".to_string()];
            t.extend(cell_tokens(c));
            t.push(CUBE_KINDS[cube_index(k)].into());
            (t, -1, false)
        })
        .collect();
    cubes.sort();
    sections.push(("cube", cubes));
    for (word, kind) in [
        ("laser", LaserKind::Emitter),
        ("catcher", LaserKind::Catcher),
        ("relay", LaserKind::Relay),
    ] {
        let mut sec: Section = Vec::new();
        for l in level.lasers.iter().filter(|l| l.kind == kind) {
            let mut t = vec![word.to_string()];
            t.extend(cell_tokens(l.cell));
            t.push(dir(l.normal));
            t.push(l.offset.map_or("center".into(), dir));
            if kind == LaserKind::Emitter {
                t.push(on(l.start_enabled));
            }
            sec.push((t, l.item, kind != LaserKind::Emitter));
        }
        sec.sort();
        sections.push((word, sec));
    }
    let mut plates: Section = Vec::new();
    for pl in &level.plates {
        let mut t = vec!["plate".to_string()];
        t.extend(cell_tokens(pl.cell));
        t.push(dir(pl.normal));
        t.extend(cell_tokens(pl.target.1));
        t.push(dir(pl.target.2));
        plates.push((t, pl.item, false));
    }
    plates.sort();
    sections.push(("plate", plates));
    for (word, kind) in [("bridge", ProjectorKind::Bridge), ("funnel", ProjectorKind::Funnel)] {
        let mut sec: Section = Vec::new();
        for pj in level.projectors.iter().filter(|p| p.kind == kind) {
            let mut t = vec![word.to_string()];
            t.extend(cell_tokens(pj.cell));
            t.push(dir(pj.normal));
            if kind == ProjectorKind::Bridge {
                t.push(dir(pj.side));
            }
            t.push(on(pj.start_enabled));
            if kind == ProjectorKind::Funnel && pj.start_reversed {
                t.push("rev".into());
            }
            sec.push((t, pj.item, false));
        }
        sec.sort();
        sections.push((word, sec));
    }
    for (word, kind) in [
        ("panel", PanelKind::Angled),
        ("glasspanel", PanelKind::Glass),
        ("flip", PanelKind::Flip),
    ] {
        let mut sec: Section = Vec::new();
        for pn in level.panels.iter().filter(|p| p.kind == kind) {
            let mut t = vec![word.to_string()];
            t.extend(cell_tokens(pn.cell));
            t.push(dir(pn.normal));
            if kind != PanelKind::Flip {
                t.push(dir(pn.tilt));
                t.push(num(pn.angle / 15));
                t.push(on(pn.start_deployed));
            }
            sec.push((t, pn.item, false));
        }
        sec.sort();
        sections.push((word, sec));
    }
    let mut stairs: Section = Vec::new();
    for st in &level.stairs {
        let mut t = vec!["stairs".to_string()];
        t.extend(cell_tokens(st.cell));
        t.push(dir(st.dir));
        t.push(on(st.start_deployed));
        stairs.push((t, st.item, false));
    }
    stairs.sort();
    sections.push(("stairs", stairs));
    let mut pistons: Section = Vec::new();
    for ps in &level.pistons {
        let mut t = vec!["piston".to_string()];
        t.extend(cell_tokens(ps.cell));
        t.push(dir(ps.normal));
        t.push(num(ps.bottom));
        t.push(num(ps.top));
        t.push(on(ps.start_up));
        pistons.push((t, ps.item, false));
    }
    pistons.sort();
    sections.push(("piston", pistons));
    let mut rails: Section = Vec::new();
    for r in &level.rails {
        let mut t = vec!["rail".to_string()];
        t.extend(cell_tokens(r.cell));
        t.extend(cell_tokens(r.end));
        t.push(if r.oscillate {
            "osc".into()
        } else {
            on(r.start_active)
        });
        rails.push((t, r.item, false));
    }
    rails.sort();
    sections.push(("rail", rails));

    let mut out = Later {
        tokens: Vec::new(),
        receivers: Vec::new(),
    };
    for (word, sec) in sections {
        if TARGETS.contains(&word) {
            out.receivers.push((word, sec.iter().map(|s| s.1).collect()));
        }
        for (t, item, sends) in sec {
            out.tokens.push((t, sends.then_some(item)));
        }
    }
    Ok(out)
}

fn paint_index(k: PaintKind) -> usize {
    PaintKind::ALL.iter().position(|&p| p == k).unwrap()
}

fn cube_index(k: CubeKind) -> usize {
    match k {
        CubeKind::Standard => 0,
        CubeKind::Companion => 1,
        CubeKind::Reflection => 2,
        CubeKind::Sphere => 3,
        CubeKind::Franken => 4,
    }
}

/// The box spanned by an anchor item and its extents.
fn extent_box(p: &Puzzle, level: &Level, it: &PlacedItem) -> (Cell, Cell) {
    let mut corners = vec![it.cell];
    for c in &p.connections {
        if c.get("Type") == Some("CONNECTION_BARRIER_ANCHOR_TO_EXTENT")
            && c.get_i32("Sender") == Some(it.index)
        {
            if let Some(e) = c.get_i32("Receiver").and_then(|r| level.item(r)) {
                corners.push(e.cell);
            }
        }
    }
    let lo = std::array::from_fn(|a| corners.iter().map(|c| c[a]).min().unwrap());
    let hi = std::array::from_fn(|a| corners.iter().map(|c| c[a]).max().unwrap());
    (lo, hi)
}

/// White surfaces as boxes: greedy rectangles in each plane, per direction.
fn white_rects(level: &Level) -> Vec<(Dir, Cell, Cell)> {
    let mut out = Vec::new();
    for (d, _) in DIRS {
        let white: HashSet<Cell> = level
            .cells()
            .filter(|&c| !level.solid(c) && level.portal_surface(c, d))
            .collect();
        let (u, v) = match d.axis() {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        let mut used: HashSet<Cell> = HashSet::new();
        let free = |c: Cell, used: &HashSet<Cell>| white.contains(&c) && !used.contains(&c);
        let at = |c: Cell, a: usize, k: i32| {
            let mut c = c;
            c[a] += k;
            c
        };
        for c in level.cells() {
            if !free(c, &used) {
                continue;
            }
            let mut du = 1;
            while free(at(c, u, du), &used) {
                du += 1;
            }
            let mut dv = 1;
            while (0..du).all(|i| free(at(at(c, u, i), v, dv), &used)) {
                dv += 1;
            }
            for i in 0..du {
                for j in 0..dv {
                    used.insert(at(at(c, u, i), v, j));
                }
            }
            out.push((d, c, at(at(c, u, du - 1), v, dv - 1)));
        }
    }
    out
}

struct Reader<'a> {
    toks: Vec<&'a str>,
    pos: usize,
}

impl<'a> Reader<'a> {
    fn peek(&self) -> Option<&'a str> {
        self.toks.get(self.pos).copied()
    }

    fn next(&mut self, what: &str) -> Result<&'a str, String> {
        let t = self
            .peek()
            .ok_or_else(|| format!("ran out of tokens, expected {what}"))?;
        self.pos += 1;
        Ok(t)
    }

    fn fail<T>(&self, msg: impl std::fmt::Display) -> Result<T, String> {
        Err(format!("token {}: {msg}", self.pos.saturating_sub(1)))
    }

    fn expect(&mut self, word: &str) -> Result<(), String> {
        match self.next(word)? {
            t if t == word => Ok(()),
            t => self.fail(format!("expected {word:?}, found {t:?}")),
        }
    }

    fn num(&mut self, lo: i32, hi: i32) -> Result<i32, String> {
        let t = self.next("a number")?;
        match t.parse::<i32>() {
            Ok(n) if n.to_string() == t && (lo..=hi).contains(&n) => Ok(n),
            _ => self.fail(format!("expected a number in {lo}..={hi}, found {t:?}")),
        }
    }

    fn cell(&mut self, size: [i32; 3]) -> Result<Cell, String> {
        Ok([
            self.num(0, size[0] - 1)?,
            self.num(0, size[1] - 1)?,
            self.num(0, size[2] - 1)?,
        ])
    }

    fn dir(&mut self) -> Result<Dir, String> {
        let t = self.next("a direction")?;
        match DIRS.iter().find(|(_, s)| *s == t) {
            Some((d, _)) => Ok(*d),
            None => self.fail(format!("expected a direction, found {t:?}")),
        }
    }

    fn word(&mut self, words: &[&str]) -> Result<usize, String> {
        let t = self.next("a keyword")?;
        match words.iter().position(|w| *w == t) {
            Some(i) => Ok(i),
            None => self.fail(format!("expected one of {words:?}, found {t:?}")),
        }
    }
}

enum Target {
    Exit,
    Field(i32),
    Gel(i32),
    /// A later target word (index into `TARGETS`) and position.
    Other(usize, i32),
}

/// Builds the puzzle a token sequence describes. Errors name the offending token.
/// The fewest goo items whose flood (sideways and down, see `Level`) covers exactly
/// `goo` (cells as (z, y, x)): one per group of goo cells joined sideways at one height that
/// goo from above doesn't already flow into, on the group's first cell with a floor under it
/// (the editor only takes goo on a floor) and, if it can, no item in it (`taken`). Items the
/// flood reaches, like a door or button, then aren't standing in a goo item.
fn goo_sources(
    goo: &BTreeSet<(i32, i32, i32)>,
    solid: &HashSet<Cell>,
    taken: &HashSet<Cell>,
) -> Vec<Cell> {
    let mut seen: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut out = Vec::new();
    for &start in goo {
        if !seen.insert(start) {
            continue;
        }
        let mut group = vec![start];
        let mut i = 0;
        while i < group.len() {
            let (z, y, x) = group[i];
            for (dy, dx) in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
                let n = (z, y + dy, x + dx);
                if goo.contains(&n) && seen.insert(n) {
                    group.push(n);
                }
            }
            i += 1;
        }
        if group.iter().any(|&(z, y, x)| goo.contains(&(z + 1, y, x))) {
            continue;
        }
        group.sort();
        let floored = |&&(z, y, x): &&(i32, i32, i32)| z == 0 || solid.contains(&[x, y, z - 1]);
        let free = |c: &&(i32, i32, i32)| !taken.contains(&[c.2, c.1, c.0]);
        let (z, y, x) = *group
            .iter()
            .find(|c| floored(c) && free(c))
            .or_else(|| group.iter().find(floored))
            .unwrap_or(&group[0]);
        out.push([x, y, z]);
    }
    out
}

pub fn decode(text: &str) -> Result<Puzzle, String> {
    let mut r = Reader {
        toks: text.split_whitespace().collect(),
        pos: 0,
    };
    r.expect("size")?;
    let size = [r.num(1, MAX_NUM)?, r.num(1, MAX_NUM)?, r.num(1, MAX_NUM)?];
    let [sx, sy, sz] = size;
    let mut b = Builder::new(TITLE, size);
    let mut goo_cells: BTreeSet<(i32, i32, i32)> = BTreeSet::new(); // (z, y, x)
    let mut solid_cells: HashSet<Cell> = HashSet::new();
    let mut prev_col: Option<Vec<(char, i32)>> = None;
    let mut rows: Vec<Vec<Vec<(char, i32)>>> = Vec::new();
    let is_geometry = |t: Option<&str>| {
        t.is_some_and(|t| {
            matches!(t, "/" | "=" | "|")
                || (t.len() > 1
                    && t.starts_with(['s', 'o', 'g'])
                    && t[1..].bytes().all(|c| c.is_ascii_digit()))
        })
    };
    for y in 0..sy {
        if y > 0 && !is_geometry(r.peek()) {
            break; // The rest repeat the last row.
        }
        let mut row: Vec<Vec<(char, i32)>> = Vec::new();
        loop {
            match r.peek() {
                Some("/") => {
                    r.pos += 1;
                    break;
                }
                Some("=") => {
                    r.pos += 1;
                    match &prev_col {
                        Some(p) => row.push(p.clone()),
                        None => return r.fail("`=` before any column"),
                    }
                }
                _ => {
                    let mut runs = Vec::new();
                    let mut total = 0;
                    loop {
                        let t = r.next("a run or `|`")?;
                        if t == "|" {
                            break;
                        }
                        let mut chars = t.chars();
                        let kind = chars.next().unwrap();
                        let n = chars.as_str().parse::<i32>().ok();
                        match (kind, n) {
                            ('s' | 'o' | 'g', Some(n))
                                if n >= 1 && n.to_string() == chars.as_str() =>
                            {
                                total += n;
                                runs.push((kind, n));
                            }
                            _ => return r.fail(format!("expected a run or `|`, found {t:?}")),
                        }
                    }
                    if total != sz {
                        return r.fail(format!("column in row {y} is {total} tall, not {sz}"));
                    }
                    row.push(runs);
                }
            }
            if row.len() > sx as usize {
                return r.fail(format!("row {y} has more than {sx} columns"));
            }
            prev_col = row.last().cloned();
        }
        if row.is_empty() {
            match rows.last() {
                Some(p) => row = p.clone(),
                None => return r.fail("the first row is empty"),
            }
        }
        while row.len() < sx as usize {
            row.push(row.last().unwrap().clone());
        }
        prev_col = row.last().cloned();
        rows.push(row);
    }
    for y in 0..sy {
        let row = rows
            .get(y as usize)
            .or(rows.last())
            .expect("row 0 was read");
        for (x, runs) in row.iter().enumerate() {
            let x = x as i32;
            let mut z = 0;
            for &(kind, n) in runs {
                if kind != 's' {
                    b.carve([x, y, z], [x, y, z + n - 1]);
                }
                for zz in z..z + n {
                    match kind {
                        'g' => {
                            goo_cells.insert((zz, y, x));
                        }
                        's' => {
                            solid_cells.insert([x, y, zz]);
                        }
                        _ => {}
                    }
                }
                z += n;
            }
        }
    }

    let (mut entry, mut exit, mut inverted) = (None, None, false);
    let mut fields: Vec<i32> = Vec::new();
    // Each gel's dropper, if poured.
    let mut gels: Vec<Option<i32>> = Vec::new();
    let mut wires: Vec<(i32, Target)> = Vec::new();
    // Items of the later target kinds, by `TARGETS` index.
    let mut later: Vec<Vec<i32>> = vec![Vec::new(); TARGETS.len()];
    let slot = |word: &str| TARGETS.iter().position(|t| *t == word).unwrap();
    // Two directions that must be perpendicular.
    fn across(r: &Reader, a: Dir, b: Dir) -> Result<(), String> {
        if a.axis() == b.axis() {
            return r.fail("directions aren't perpendicular");
        }
        Ok(())
    }
    fn wired(r: &mut Reader, sender: i32, wires: &mut Vec<(i32, Target)>) -> Result<(), String> {
        while r.peek() == Some(">") {
            r.pos += 1;
            let mut words = vec!["exit", "field", "gel"];
            words.extend(&TARGETS[2..]);
            let target = match r.word(&words)? {
                0 => Target::Exit,
                1 => Target::Field(r.num(0, MAX_NUM)?),
                2 => Target::Gel(r.num(0, MAX_NUM)?),
                k => Target::Other(k - 1, r.num(0, MAX_NUM)?),
            };
            wires.push((sender, target));
        }
        Ok(())
    }
    while let Some(t) = r.peek() {
        r.pos += 1;
        match t {
            "white" => {
                let d = r.dir()?;
                let (lo, hi) = (r.cell(size)?, r.cell(size)?);
                if (0..3).any(|a| lo[a] > hi[a]) {
                    return r.fail("white box corners are out of order");
                }
                b.paint(lo, hi, d, true);
            }
            "entry" | "exit" => {
                let (cell, d) = (r.cell(size)?, r.dir()?);
                if t == "entry" {
                    if entry.is_some() {
                        return r.fail("second entry door");
                    }
                    entry = Some(b.entry(cell, d));
                } else {
                    if exit.is_some() {
                        return r.fail("second exit door");
                    }
                    exit = Some(b.exit(cell, d));
                    if r.peek() == Some("inv") {
                        r.pos += 1;
                        inverted = true;
                    }
                }
            }
            "glass" | "grating" => {
                let (lo, hi) = (r.cell(size)?, r.cell(size)?);
                let (n, grow) = (r.dir()?, r.dir()?);
                if (0..3).any(|a| lo[a] > hi[a]) {
                    return r.fail("panel corners are out of order");
                }
                if lo[n.axis()] != hi[n.axis()] || grow.axis() == n.axis() {
                    return r.fail("panel isn't flat across its normal, or grows along it");
                }
                // Anchor on the edge it hangs from: the first cell there with something
                // solid behind it, so panels over pits stay mounted.
                let side = grow.cross(n.opposite()).expect("perpendicular");
                let (g, s) = (grow.axis(), side.axis());
                let mut edge = lo;
                if !grow.is_positive() {
                    edge[g] = hi[g];
                }
                let anchor = (lo[s]..=hi[s])
                    .map(|k| {
                        let mut c = edge;
                        c[s] = k;
                        c
                    })
                    .find(|&c| b.solid(step(c, grow.opposite())))
                    .unwrap_or(edge);
                let (below, above) = (anchor[s] - lo[s], hi[s] - anchor[s]);
                let (left, right) = if side.is_positive() {
                    (below, above)
                } else {
                    (above, below)
                };
                b.glass(anchor, n, grow, left, right, hi[g] - lo[g], t == "grating");
            }
            "fizzler" | "laserfield" => {
                let anchor = r.cell(size)?;
                let (n, along) = (r.dir()?, r.dir()?);
                let len = r.num(1, MAX_NUM)?;
                let on = r.word(&["on", "off"])? == 0;
                if n.axis() == along.axis() {
                    return r.fail("field runs along its own normal");
                }
                let v = along.vector();
                let far: Cell = std::array::from_fn(|a| anchor[a] + v[a] * (len - 1));
                if (0..3).any(|a| far[a] < 0 || far[a] >= size[a]) {
                    return r.fail("field leaves the chamber");
                }
                fields.push(b.fizzler(anchor, n, along, len, t == "laserfield", on));
            }
            "dropper" => {
                let cell = r.cell(size)?;
                let kind = r.word(&CUBE_KINDS)?;
                let (_, dropper) = b.cube_dropper_of(cell, kind as i32);
                if r.peek() == Some("wait") {
                    r.pos += 1;
                    b.auto_drop(dropper, false);
                }
                later[slot("dropper")].push(dropper);
            }
            "cube" => {
                let cell = r.cell(size)?;
                let kind = r.word(&CUBE_KINDS)?;
                b.loose_cube(cell, kind as i32);
            }
            "laser" | "catcher" | "relay" => {
                let (cell, n) = (r.cell(size)?, r.dir()?);
                let offset = if r.peek() == Some("center") {
                    r.pos += 1;
                    None
                } else {
                    let o = r.dir()?;
                    across(&r, n, o)?;
                    Some(o)
                };
                let kind = match t {
                    "laser" => LaserKind::Emitter,
                    "catcher" => LaserKind::Catcher,
                    _ => LaserKind::Relay,
                };
                let on = kind != LaserKind::Emitter || r.word(&["on", "off"])? == 0;
                let item = b.laser(kind, cell, n, offset, on);
                if kind == LaserKind::Emitter {
                    later[slot("laser")].push(item);
                } else {
                    wired(&mut r, item, &mut wires)?;
                }
            }
            "plate" => {
                let (cell, n) = (r.cell(size)?, r.dir()?);
                let (target, tn) = (r.cell(size)?, r.dir()?);
                if cell == target {
                    return r.fail("faith plate targets its own cell");
                }
                b.faith_plate(cell, n, target, tn);
            }
            "bridge" | "funnel" => {
                let (cell, n) = (r.cell(size)?, r.dir()?);
                if t == "bridge" {
                    let side = r.dir()?;
                    across(&r, n, side)?;
                    let on = r.word(&["on", "off"])? == 0;
                    let item = b.light_bridge(cell, n, side, on);
                    later[slot("bridge")].push(item);
                } else {
                    let on = r.word(&["on", "off"])? == 0;
                    let rev = r.peek() == Some("rev");
                    if rev {
                        r.pos += 1;
                    }
                    let item = b.funnel(cell, n, on, rev);
                    later[slot("funnel")].push(item);
                    later[slot("polarity")].push(item);
                }
            }
            "panel" | "glasspanel" => {
                let (cell, n, tilt) = (r.cell(size)?, r.dir()?, r.dir()?);
                across(&r, n, tilt)?;
                let angle = match r.num(2, 6)? {
                    a @ (2 | 3 | 4 | 6) => a * 15,
                    a => return r.fail(format!("panel angle {a} isn't 2, 3, 4 or 6")),
                };
                let on = r.word(&["on", "off"])? == 0;
                let white = b.white(cell, n.opposite());
                let glass = t == "glasspanel";
                let item = b.angled_panel(cell, n, tilt, angle, on, white, glass);
                later[slot(t)].push(item);
            }
            "flip" => {
                let (cell, n) = (r.cell(size)?, r.dir()?);
                let item = b.flip_panel(cell, n);
                later[slot("flip")].push(item);
            }
            "stairs" => {
                let (cell, d) = (r.cell(size)?, r.dir()?);
                if d.axis() == 2 {
                    return r.fail("stairs rise along the floor");
                }
                let on = r.word(&["on", "off"])? == 0;
                let item = b.stairs(cell, d, on);
                later[slot("stairs")].push(item);
            }
            "piston" => {
                let (cell, n) = (r.cell(size)?, r.dir()?);
                let (lo, hi) = (r.num(0, 3)?, r.num(1, 4)?);
                if lo >= hi {
                    return r.fail("piston's bottom isn't below its top");
                }
                let on = r.word(&["on", "off"])? == 0;
                let item = b.piston(cell, n, lo, hi, on);
                later[slot("piston")].push(item);
            }
            "rail" => {
                let (cell, end) = (r.cell(size)?, r.cell(size)?);
                if (0..3).filter(|&a| cell[a] != end[a]).count() != 1 {
                    return r.fail("rail track isn't one straight line");
                }
                let mode = r.word(&["osc", "on", "off"])?;
                let item = b.rail(cell, end, mode == 0, mode != 2);
                later[slot("rail")].push(item);
            }
            "button" | "pedestal" => {
                let cell = r.cell(size)?;
                let sender = if t == "button" {
                    let kind = r.word(&BUTTON_KINDS)?;
                    b.floor_button_of(cell, kind as i32)
                } else {
                    let timer = if r.peek() == Some("inf") {
                        r.pos += 1;
                        None
                    } else {
                        Some(r.num(3, 30)? as u32)
                    };
                    b.pedestal(cell, timer)
                };
                wired(&mut r, sender, &mut wires)?;
            }
            "gel" => {
                let (cell, n) = (r.cell(size)?, r.dir()?);
                let kind = r.word(&PAINT_KINDS)? as i32;
                if b.solid(cell) || !b.solid(step(cell, n.opposite())) {
                    return r.fail("gel isn't on a surface");
                }
                if r.word(&["stream", "splat"])? == 0 {
                    let on = r.word(&["on", "off"])? == 0;
                    if n != Dir::PosZ {
                        return r.fail("poured gel must be on a floor");
                    }
                    gels.push(Some(b.gel_dropper(cell, kind, on)));
                } else {
                    b.gel_splat(cell, n, kind);
                    gels.push(None);
                }
            }
            "turret" => {
                let cell = r.cell(size)?;
                let f = r.num(0, 7)?;
                b.turret(cell, f * 45);
            }
            other => return r.fail(format!("unexpected {other:?}")),
        }
    }
    // Cells other items stand in, which the editor won't let goo share.
    let taken: HashSet<Cell> = Level::from_puzzle(&b.build())
        .items
        .iter()
        .filter(|i| i.kind.is_mounted())
        .map(|i| i.cell)
        .collect();
    for cell in goo_sources(&goo_cells, &solid_cells, &taken) {
        b.goo(cell);
    }
    for (sender, target) in wires {
        let receiver = match target {
            Target::Exit => exit.ok_or("a button is wired to the exit, but there is none")?,
            Target::Field(i) => *fields.get(i as usize).ok_or(format!(
                "a button is wired to field {i}, which doesn't exist"
            ))?,
            Target::Gel(i) => gels.get(i as usize).copied().flatten().ok_or(format!(
                "a button is wired to gel {i}, which doesn't exist or isn't poured"
            ))?,
            Target::Other(k, i) => {
                let item = *later[k].get(i as usize).ok_or(format!(
                    "a wire goes to {} {i}, which doesn't exist",
                    TARGETS[k]
                ))?;
                if TARGETS[k] == "polarity" {
                    b.connect_polarity(sender, item);
                    continue;
                }
                item
            }
        };
        b.connect(sender, receiver);
    }
    if let (Some(e), true) = (exit, inverted) {
        b.set_prop(e, "ITEM_PROPERTY_START_OPEN", "1");
    }
    Ok(b.build())
}
