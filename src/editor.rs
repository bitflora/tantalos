//! The Puzzle Maker's placement rules: which items the editor marks as invalid (red) and
//! refuses to build. Predictions use the game's own names and positions, so they can be
//! compared item for item with what Portal 2 logs (`train/portal2.py`):
//! `INVALID ITEM:  <TargetName> ( x , y , z )`, with the item's `VoxelPos`.
//!
//! Footprints come from `portal2_dlc2/scripts/editoritems.txt`. Glass, grating and fields have
//! none there (their rules are in the game's code), so theirs were fitted to in-game checks of
//! 2,403 generated levels (2026-10-10; `tools/ingame/rules.py`). Each rule below agreed with
//! the game on every item it flagged unless noted. Collisions the game decides by finer
//! geometry than this models (grating across a field, a panel on a field's own face, crossing
//! fields other than a fizzler with a laser field, fields side by side, a field beside a
//! pedestal) are left to the in-game check. Turrets and gel (added later) follow
//! editoritems.txt where it says something (turrets can't anchor on goo, splats can; the paint
//! dropper's embedded voxels are commented out) and are otherwise assumed to collide with
//! fields like floor buttons, until in-game checks say otherwise.

use crate::geom::{rotate, step, Cell, Dir};
use crate::level::{BarrierKind, ItemKind, Level, PlacedItem};
use std::collections::{BTreeMap, HashMap, HashSet};

/// An item the editor would mark invalid.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Flag {
    /// The item's `TargetName` in editoritems.txt, as the game logs it.
    pub name: &'static str,
    pub pos: Cell,
    pub rules: Vec<&'static str>,
}

/// The editor's name for an item kind (glass and grating are both "target", like the
/// faith plate target, in editoritems.txt).
pub fn target_name(kind: &ItemKind) -> Option<&'static str> {
    Some(match kind {
        ItemKind::EntryDoor => "doorentry",
        ItemKind::ExitDoor => "doorexit",
        ItemKind::Cube => "cube",
        ItemKind::CubeDropper => "cubedropper",
        ItemKind::FloorButton | ItemKind::PedestalButton => "button",
        ItemKind::Goo => "goo",
        ItemKind::Glass | ItemKind::Grating => "target",
        ItemKind::Fizzler | ItemKind::LaserField => "barrierhazard",
        ItemKind::Turret => "turret",
        ItemKind::PaintSplat => "paintsplat",
        ItemKind::PaintDropper => "paintdropper",
        _ => return None,
    })
}

#[derive(Default)]
struct Flags(BTreeMap<(&'static str, Cell), Vec<&'static str>>);

impl Flags {
    fn add(&mut self, item: &PlacedItem, rule: &'static str) {
        if let Some(name) = target_name(&item.kind) {
            let rules = self.0.entry((name, item.cell)).or_default();
            if !rules.contains(&rule) {
                rules.push(rule);
            }
        }
    }

    fn pair(&mut self, a: &PlacedItem, b: &PlacedItem, rule: &'static str) {
        self.add(a, rule);
        self.add(b, rule);
    }
}

/// Two directions perpendicular to `n`.
fn across(n: Dir) -> [Dir; 2] {
    match n {
        Dir::NegX | Dir::PosX => [Dir::PosY, Dir::PosZ],
        Dir::NegY | Dir::PosY => [Dir::PosX, Dir::PosZ],
        Dir::NegZ | Dir::PosZ => [Dir::PosX, Dir::PosY],
    }
}

fn offset(c: Cell, d: Dir, k: i32) -> Cell {
    let (d, k) = if k < 0 { (d.opposite(), -k) } else { (d, k) };
    (0..k).fold(c, |c, _| step(c, d))
}

/// Voxels an item is sunk into (`EmbeddedVoxels`), in world cells. Local (0,0,0) is the
/// solid voxel behind the item's surface; local -z goes deeper.
fn embedded(item: &PlacedItem) -> Vec<Cell> {
    let Some(n) = item.normal else { return Vec::new() };
    let wall = step(item.cell, n.opposite());
    let [a, b] = across(n);
    match item.kind {
        // Volume (-1,-1,0)..(1,1,-3): 3x3 across the wall, 4 deep. Two doors on one wall
        // closer than 3 cells both ways overlap.
        ItemKind::EntryDoor | ItemKind::ExitDoor => (0..4)
            .flat_map(|depth| (-1..=1).flat_map(move |i| (-1..=1).map(move |j| (depth, i, j))))
            .map(|(depth, i, j)| offset(offset(offset(wall, n, -depth), a, i), b, j))
            .collect(),
        ItemKind::CubeDropper => vec![wall],
        _ => Vec::new(),
    }
}

/// The side of its cell a pedestal button's column stands against: local -x (editoritems.txt
/// gives the pedestal surfaces on one side; the game confirms it is this one).
fn pedestal_side(item: &PlacedItem) -> Option<Dir> {
    item.angles.and_then(|a| rotate(a, [-1, 0, 0]))
}

/// Whether an item in a field's swept cell collides with the field, which lies in that cell
/// next to face `f`.
fn hit_by_field(item: &PlacedItem, f: Dir) -> bool {
    match item.kind {
        // Turrets and gel: not checked in-game yet; assumed to collide like a floor button.
        ItemKind::FloorButton
        | ItemKind::CubeDropper
        | ItemKind::Turret
        | ItemKind::PaintSplat
        | ItemKind::PaintDropper => true,
        // Unless the field runs across the door's front.
        ItemKind::EntryDoor | ItemKind::ExitDoor => item.normal != Some(f),
        ItemKind::PedestalButton => pedestal_side(item) == Some(f),
        _ => false,
    }
}

/// Everything the editor would flag in this level.
pub fn editor_flags(level: &Level) -> Vec<Flag> {
    let mut flags = Flags::default();
    let goo_items: HashSet<Cell> = level.of_kind(&ItemKind::Goo).map(|i| i.cell).collect();

    let mut owner: HashMap<Cell, usize> = HashMap::new();
    for (i, it) in level.items.iter().enumerate() {
        for c in embedded(it) {
            match owner.get(&c) {
                Some(&j) if j != i => flags.pair(&level.items[j], it, "embedded-overlap"),
                Some(_) => {}
                None => {
                    owner.insert(c, i);
                }
            }
        }
    }

    for it in &level.items {
        match it.kind {
            // Goo is placed on a floor.
            ItemKind::Goo if !level.solid(step(it.cell, Dir::NegZ)) => {
                flags.add(it, "goo-not-on-floor")
            }
            // Only cubes and gel splats may anchor on goo (CanAnchorOnGoo).
            ItemKind::FloorButton
            | ItemKind::PedestalButton
            | ItemKind::CubeDropper
            | ItemKind::EntryDoor
            | ItemKind::ExitDoor
            | ItemKind::Turret
            | ItemKind::PaintDropper
                if goo_items.contains(&it.cell) =>
            {
                flags.add(it, "in-goo")
            }
            _ => {}
        }
    }

    // Glass and grating on a face an item occupies (only cubes may anchor on them).
    for b in &level.barriers {
        let Some(panel) = level.item(b.item) else { continue };
        for it in &level.items {
            let Some(n) = it.normal else { continue };
            for &(c, d) in &b.faces {
                if c != it.cell {
                    continue;
                }
                let rule = match it.kind {
                    ItemKind::EntryDoor | ItemKind::ExitDoor if d == n.opposite() => "door-on-panel",
                    ItemKind::CubeDropper => "dropper-in-panel",
                    ItemKind::PedestalButton if Some(d) == pedestal_side(it) => "pedestal-against-panel",
                    // Not seen in the checked levels; follows CanAnchorOnBarriers.
                    ItemKind::FloorButton | ItemKind::PedestalButton if d == n.opposite() => {
                        "button-on-panel"
                    }
                    _ => continue,
                };
                flags.pair(it, panel, rule);
            }
        }
    }

    // Fields fill the cells they sweep, from the mounting wall to the far wall, next to face `f`.
    for (i, h) in level.hazards.iter().enumerate() {
        let Some(field) = level.item(h.item) else { continue };
        let (Some(&(_, f)), Some(n)) = (h.faces.first(), field.normal) else { continue };
        let swept: HashSet<Cell> = h.faces.iter().map(|&(c, _)| c).collect();
        for it in &level.items {
            if swept.contains(&it.cell) && hit_by_field(it, f) {
                flags.pair(it, field, "in-field");
            }
        }
        // Glass across the field's path (grating mostly lets it through; not modelled).
        for b in level.barriers.iter().filter(|b| b.kind == BarrierKind::Glass) {
            let Some(panel) = level.item(b.item) else { continue };
            let blocks = b.faces.iter().any(|&(c, d)| {
                (d == n || d == n.opposite()) && (swept.contains(&c) || swept.contains(&step(c, d)))
            });
            if blocks {
                flags.pair(panel, field, "glass-across-field");
            }
        }
        // Fields through the same cells: parallel ones (any kinds), or a fizzler crossing a
        // laser field (8 of 10 and 15 of 17 flags agreed).
        for h2 in &level.hazards[i + 1..] {
            let Some(other) = level.item(h2.item) else { continue };
            let Some(&(_, f2)) = h2.faces.first() else { continue };
            if !h2.faces.iter().any(|(c, _)| swept.contains(c)) {
                continue;
            }
            let parallel = f2 == f || f2 == f.opposite();
            if parallel || h.kind != h2.kind {
                flags.pair(field, other, "fields-overlap");
            }
        }
    }

    flags
        .0
        .into_iter()
        .map(|((name, pos), rules)| Flag { name, pos, rules })
        .collect()
}

/// Editor flags as problem messages, for `Level::mount_problems`.
pub fn editor_problems(level: &Level) -> Vec<String> {
    editor_flags(level)
        .into_iter()
        .map(|f| format!("editor: {} at {:?} ({})", f.name, f.pos, f.rules.join(", ")))
        .collect()
}
