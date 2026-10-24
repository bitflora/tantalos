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
//! fields like floor buttons, until in-game checks say otherwise. So are the items added after
//! them (lasers, faith plates, bridges, funnels, panels, stairs, pistons, rails), which also
//! can't anchor on goo, and two of which can't share a cell.
//!
//! Published Workshop levels all passed the real editor, so they show what it allows: where
//! they break a rule here and the in-game checks don't contradict them (2026-10-16), the rule
//! was narrowed (turrets, splats and relays in fields; turrets and wall lasers sharing cells;
//! parallel fields). Not where the in-game checks disagree with the Workshop or
//! can't tell the cases apart (a floor button sharing a cell with a light bridge, a field on
//! the door side of an exit door's cell, a pedestal in a field's end cell, pedestals against
//! glass).

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
        ItemKind::Glass | ItemKind::Grating | ItemKind::PlateTarget => "target",
        ItemKind::Fizzler | ItemKind::LaserField => "barrierhazard",
        ItemKind::Turret => "turret",
        ItemKind::PaintSplat => "paintsplat",
        ItemKind::PaintDropper => "paintdropper",
        ItemKind::LaserEmitter => "laseremit",
        ItemKind::LaserCatcher => "lasercatch",
        ItemKind::LaserRelay => "laserrelay",
        ItemKind::FaithPlate => "catapult",
        ItemKind::LightBridge => "bridge",
        ItemKind::Funnel => "tbeam",
        ItemKind::AngledPanel => "angledPanel",
        ItemKind::GlassPanel => "angledClearPanel",
        ItemKind::FlipPanel => "flipPanel",
        ItemKind::Stairs => "stairs",
        ItemKind::Piston => "pistonlift",
        ItemKind::Rail => "raillift",
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
        // Turrets, gel splats and laser relays: the game flagged none of the 11 in fields
        // (in-game check of runs/v5-fresh-8h, 2026-10-16), and 43 Workshop levels have some.
        ItemKind::Turret | ItemKind::PaintSplat | ItemKind::LaserRelay => false,
        // Items that stay on their surface are clear of a field on the cell's far face, like
        // doors (in 22 Workshop levels; ASSUMPTION, none in the in-game checks).
        ItemKind::FloorButton
        | ItemKind::FaithPlate
        | ItemKind::LaserEmitter
        | ItemKind::LaserCatcher => item.normal != Some(f),
        // Other later items: assumed to collide like a floor button.
        ItemKind::CubeDropper
        | ItemKind::PaintDropper
        | ItemKind::LightBridge
        | ItemKind::Funnel
        | ItemKind::AngledPanel
        | ItemKind::GlassPanel
        | ItemKind::FlipPanel
        | ItemKind::Stairs
        | ItemKind::Piston => true,
        // Unless the field runs across the door's front.
        ItemKind::EntryDoor | ItemKind::ExitDoor => item.normal != Some(f),
        ItemKind::PedestalButton => pedestal_side(item) == Some(f),
        _ => false,
    }
}

/// Pairs of items that may share a cell after all. Turrets collide only with physics
/// (editoritems.txt) and share cells with floor panels, pistons, faith plates and wall items
/// in 95 Workshop levels; the game flagged a turret only on stairs and rails. Laser emitters
/// and catchers sit in their wall: on another face of the cell they leave lasers, doors and
/// pedestal buttons alone (121 Workshop levels; in-game, 0 of 2 pedestals flagged), but not
/// pistons or stairs, which the game flagged.
fn share_ok(a: &PlacedItem, b: &PlacedItem) -> bool {
    let ok = |a: &PlacedItem, b: &PlacedItem| match a.kind {
        ItemKind::Turret => !matches!(b.kind, ItemKind::Stairs | ItemKind::Rail),
        ItemKind::LaserEmitter | ItemKind::LaserCatcher => {
            a.normal != b.normal
                && matches!(
                    b.kind,
                    ItemKind::LaserEmitter
                        | ItemKind::LaserCatcher
                        | ItemKind::LaserRelay
                        | ItemKind::EntryDoor
                        | ItemKind::ExitDoor
                        | ItemKind::PedestalButton
                )
        }
        _ => false,
    };
    ok(a, b) || ok(b, a)
}

/// A face, named the same way from both cells it separates.
fn face_key((c, d): (Cell, Dir)) -> (Cell, Dir) {
    if d.is_positive() {
        (c, d)
    } else {
        (step(c, d), d.opposite())
    }
}

/// Whether `it` is glass or grating lying on top of goo: every face on a goo cell's +z side,
/// with no goo above.
fn on_goo_surface(level: &Level, it: &PlacedItem) -> bool {
    level.barriers.iter().filter(|b| b.item == it.index).any(|b| {
        b.faces
            .iter()
            .all(|&(c, d)| d == Dir::PosZ && !level.goo(step(c, Dir::PosZ)))
    })
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

    // Items mounted on a voxel of a door's frame (beside or above it on the same wall). In
    // in-game checks (2026-10-16) the game flagged all 152 lasers, catchers, funnels, flip
    // panels and rails there, but only 5 of 51 light bridges, so bridges are left out.
    for door in level
        .items
        .iter()
        .filter(|d| matches!(d.kind, ItemKind::EntryDoor | ItemKind::ExitDoor))
    {
        let frame: HashSet<Cell> = embedded(door).into_iter().collect();
        for it in &level.items {
            if matches!(
                it.kind,
                ItemKind::EntryDoor
                    | ItemKind::ExitDoor
                    | ItemKind::LightBridge
                    | ItemKind::Glass
                    | ItemKind::Grating
                    | ItemKind::Fizzler
                    | ItemKind::LaserField
                    | ItemKind::PlateTarget
            ) {
                continue;
            }
            let Some(n) = it.normal else { continue };
            if target_name(&it.kind).is_some()
                && it.normal == door.normal
                && frame.contains(&step(it.cell, n.opposite()))
            {
                flags.pair(door, it, "on-door-frame");
            }
        }
    }

    for it in &level.items {
        match it.kind {
            // Goo is placed on a floor, not on goo: the game flagged all 7 goo items stacked
            // on goo items (2026-10-16). How Workshop levels get deep goo (1,085 bodies, up
            // to 21 deep) is unknown.
            ItemKind::Goo if !level.solid(step(it.cell, Dir::NegZ)) => {
                flags.add(it, "goo-not-on-floor")
            }
            // Only cubes and gel splats may anchor on goo (CanAnchorOnGoo). Not just in a goo
            // item's own cell: anywhere its flood reaches. In-game checks of
            // runs/v5-cont-10h-b (2026-10-20) flagged 118 of 119 turrets in goo cells, glass
            // 19/19, grating 80/85, fields 42/42, pedestals 33/34, floor buttons 39/42, faith
            // plates 14/15, lasers 9/9, funnels 10/10, stairs 9/9, doors 16/18. Glass or
            // grating lying on the goo's surface is a floor over it (3 of 5 passed, and 53 in
            // Workshop levels). Workshop levels also have turrets (22) and fields (7) in goo,
            // but their goo comes from compiled maps and is unreliable (see goo-not-on-floor).
            ItemKind::FloorButton
            | ItemKind::PedestalButton
            | ItemKind::EntryDoor
            | ItemKind::ExitDoor
            | ItemKind::Turret
            | ItemKind::LaserEmitter
            | ItemKind::LaserCatcher
            | ItemKind::LaserRelay
            | ItemKind::FaithPlate
            | ItemKind::Funnel
            | ItemKind::Stairs
            | ItemKind::Glass
            | ItemKind::Grating
            | ItemKind::Fizzler
            | ItemKind::LaserField
                if (level.goo(it.cell) || goo_items.contains(&it.cell))
                    && !on_goo_surface(level, it) =>
            {
                flags.add(it, "in-goo")
            }
            // Surface items flooded by goo were mostly accepted (pistons 3 of 16 flagged,
            // panels 5 of 23), so only the goo item's own cell counts for them. So for light
            // bridges: the game flagged 5 of 6, but 130 Workshop levels have them in goo.
            ItemKind::CubeDropper
            | ItemKind::LightBridge
            | ItemKind::PaintDropper
            | ItemKind::AngledPanel
            | ItemKind::GlassPanel
            | ItemKind::FlipPanel
            | ItemKind::Piston
                if goo_items.contains(&it.cell) =>
            {
                flags.add(it, "in-goo")
            }
            _ => {}
        }
        // A faith plate's target lies on a surface: the game flagged all 43 targets with open
        // space behind them, and none with a solid voxel there (in-game checks of
        // v5-fresh-10h-e and v5-cont-10h-d, 2026-10-22). Workshop levels have many in the air,
        // but their targets are placed by the map converter, not the editor. One in the air is
        // flagged with VERTICAL_ALIGNMENT 1 too ("Replica" and a test chamber, 2026-10-24);
        // a vertical plate's target lies on the plate.
        if let (ItemKind::PlateTarget, Some(n)) = (&it.kind, it.normal) {
            if !level.solid(step(it.cell, n.opposite())) {
                flags.add(it, "target-not-on-surface");
            }
        }
        // A piston is sunk two voxels into its surface (EmbeddedVoxels 0 0 0 and 0 0 -1): the
        // game flagged all 64 pistons whose second voxel was open, and no piston with it
        // solid (2026-10-20).
        if let (ItemKind::Piston, Some(n)) = (&it.kind, it.normal) {
            let back = n.opposite();
            if !level.solid(step(step(it.cell, back), back)) {
                flags.add(it, "piston-not-embedded");
            }
        }
    }

    // A rail platform runs along a wall: the voxel behind it (against its up) must be solid
    // at every cell of its track. In-game checks of 873 rails (2026-10-16, runs/v5-fresh-8h):
    // all 775 with a gap in that wall were flagged, and every rail the game accepted had it.
    for r in &level.rails {
        let Some(it) = level.item(r.item) else { continue };
        let Some(n) = it.normal else { continue };
        if r.track().iter().any(|&c| !level.solid(step(c, n.opposite()))) {
            flags.add(it, "rail-off-wall");
        }
    }

    // Later items each take their cell (stairs two): two can't share one, nor stand where a
    // door, button, dropper or turret does. Items mounted on different faces of a cell (a
    // laser on a wall over a floor panel) aren't told apart; ASSUMPTION until checked in-game.
    let mut taken: HashMap<Cell, usize> = HashMap::new();
    for (i, it) in level.items.iter().enumerate() {
        let cells: Vec<Cell> = match it.kind {
            ItemKind::Stairs => match level.stairs.iter().find(|s| s.item == it.index) {
                Some(s) => vec![s.cell, step(s.cell, s.dir)],
                None => vec![it.cell],
            },
            // A rail platform takes its whole track.
            ItemKind::Rail => match level.rails.iter().find(|r| r.item == it.index) {
                Some(r) => r.track(),
                None => vec![it.cell],
            },
            ItemKind::LaserEmitter
            | ItemKind::LaserCatcher
            | ItemKind::LaserRelay
            | ItemKind::FaithPlate
            | ItemKind::LightBridge
            | ItemKind::Funnel
            | ItemKind::AngledPanel
            | ItemKind::GlassPanel
            | ItemKind::FlipPanel
            | ItemKind::Piston
            | ItemKind::EntryDoor
            | ItemKind::ExitDoor
            | ItemKind::FloorButton
            | ItemKind::PedestalButton
            | ItemKind::CubeDropper
            | ItemKind::Turret => vec![it.cell],
            _ => continue,
        };
        for c in cells {
            match taken.get(&c) {
                Some(&j) if j != i => {
                    let later = |k: &ItemKind| {
                        !matches!(
                            k,
                            ItemKind::EntryDoor
                                | ItemKind::ExitDoor
                                | ItemKind::FloorButton
                                | ItemKind::PedestalButton
                                | ItemKind::CubeDropper
                                | ItemKind::Turret
                        )
                    };
                    // Older pairs are the existing rules' business.
                    if (later(&it.kind) || later(&level.items[j].kind))
                        && !share_ok(&level.items[j], it)
                    {
                        flags.pair(&level.items[j], it, "shared-cell");
                    }
                }
                Some(_) => {}
                None => {
                    taken.insert(c, i);
                }
            }
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
                    // Paint droppers too: 43 of 43 flagged (2026-10-20).
                    ItemKind::CubeDropper | ItemKind::PaintDropper => "dropper-in-panel",
                    // On the face a wall item is mounted on (24 of 24 flagged, 2026-10-20).
                    // Elsewhere in their cells mostly fine (lasers 0 of 20 flagged, funnels 2 of
                    // 22). Light bridges' other faces: 12 of 12 flagged, but 18 Workshop levels
                    // have glass there.
                    ItemKind::LaserEmitter
                    | ItemKind::LaserCatcher
                    | ItemKind::LaserRelay
                    | ItemKind::LightBridge
                    | ItemKind::Funnel
                        if d == n.opposite() =>
                    {
                        "panel-on-mount"
                    }
                    // An off-centre relay against the panel beside it: 19 of 19 flagged
                    // (2026-10-22). On its cell's other faces 0 of 14.
                    ItemKind::LaserRelay if it.laser_offset() == Some(d) => "panel-beside-offset",
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

    // Glass and grating on the same face: the game flagged 20 of 20 such panels (2026-10-22).
    // Two panels of one kind there are fine (44 passed).
    for (i, a) in level.barriers.iter().enumerate() {
        let faces: HashSet<(Cell, Dir)> = a.faces.iter().map(|&f| face_key(f)).collect();
        for b in level.barriers[i + 1..].iter().filter(|b| b.kind != a.kind) {
            if b.faces.iter().any(|&f| faces.contains(&face_key(f))) {
                if let (Some(pa), Some(pb)) = (level.item(a.item), level.item(b.item)) {
                    flags.pair(pa, pb, "panels-overlap");
                }
            }
        }
    }

    // Glass and grating in a stairs cell, on any of its sides (167 of 167 flags agreed with
    // the game, 2026-10-16). Panels on a neighbour's face against the stairs are fine.
    for s in &level.stairs {
        let Some(st) = level.item(s.item) else { continue };
        let cells = [s.cell, step(s.cell, s.dir)];
        for b in &level.barriers {
            let Some(panel) = level.item(b.item) else { continue };
            if b.faces.iter().any(|(c, _)| cells.contains(c)) {
                flags.pair(st, panel, "panel-in-stairs");
            }
        }
    }
    // Likewise in the cell of a piston or a surface panel: the game flagged pistons 66 of 66,
    // angled panels 24/24, glass panels 17/17 and flip panels 15/15 (2026-10-20). Not laser
    // relays (11 of 16).
    for it in level.items.iter().filter(|i| {
        matches!(
            i.kind,
            ItemKind::Piston | ItemKind::AngledPanel | ItemKind::GlassPanel | ItemKind::FlipPanel
        )
    }) {
        for b in &level.barriers {
            let Some(panel) = level.item(b.item) else { continue };
            if b.faces.iter().any(|(c, _)| *c == it.cell) {
                flags.pair(it, panel, "panel-in-item");
            }
        }
    }

    // A rail platform's track can't hold glass or grating (on any side of a cell) or a field,
    // nor run over a door's frame. In-game checks of runs/v5-fresh-8h and of its levels with
    // rails moved to other surfaces (2026-10-16): 222 of 222, 80 of 80 and 83 of 83 flags
    // agreed with the game.
    for r in &level.rails {
        let Some(rail) = level.item(r.item) else { continue };
        let track = r.track();
        for b in &level.barriers {
            let Some(panel) = level.item(b.item) else { continue };
            if b.faces.iter().any(|(c, _)| track.contains(c)) {
                flags.pair(rail, panel, "panel-in-rail-track");
            }
        }
        for h in &level.hazards {
            let Some(field) = level.item(h.item) else { continue };
            if h.faces.iter().any(|(c, _)| track.contains(c)) {
                flags.pair(rail, field, "field-in-rail-track");
            }
        }
        if let Some(n) = rail.normal {
            for door in level
                .items
                .iter()
                .filter(|d| matches!(d.kind, ItemKind::EntryDoor | ItemKind::ExitDoor))
            {
                let frame: HashSet<Cell> = embedded(door).into_iter().collect();
                if track.iter().any(|&c| frame.contains(&step(c, n.opposite()))) {
                    flags.pair(rail, door, "rail-on-door-frame");
                }
            }
        }
    }

    // Fields fill the cells they sweep, from the mounting wall to the far wall, next to face `f`.
    for (i, h) in level.hazards.iter().enumerate() {
        let Some(field) = level.item(h.item) else { continue };
        let (Some(&(_, f)), Some(n)) = (h.faces.first(), field.normal) else { continue };
        let swept: HashSet<Cell> = h.faces.iter().map(|&(c, _)| c).collect();
        // Where the mounting wall has a gap, the editor's field also runs back through it, to
        // the next wall. In-game checks of 2026-10-22 flagged the items there as in a field:
        // stairs 24 of 24, faith plates 12/12, floor buttons 11/11, panels, pistons, funnels
        // and doors 22/26; not turrets, cubes, plate targets or pedestals beside it (0 of 71).
        let mut reach = swept.clone();
        for &c in swept.iter().filter(|&&c| !swept.contains(&step(c, n.opposite()))) {
            let mut b = step(c, n.opposite());
            while level.in_interior(b) && !level.solid(b) {
                reach.insert(b);
                b = step(b, n.opposite());
            }
        }
        for it in &level.items {
            if reach.contains(&it.cell) && hit_by_field(it, f) {
                flags.pair(it, field, "in-field");
            }
        }
        // Stairs take two cells: the game flagged 48 of 48 with only the second in a field.
        for s in &level.stairs {
            let Some(st) = level.item(s.item) else { continue };
            if reach.contains(&step(s.cell, s.dir)) {
                flags.pair(st, field, "in-field");
            }
        }
        // Glass across the field's path, also behind a gap in its wall (7 of 7 flagged,
        // 2026-10-22), or between two of its cells whichever way it lies (2 of 2). Grating
        // mostly lets it through; not modelled.
        for b in level.barriers.iter().filter(|b| b.kind == BarrierKind::Glass) {
            let Some(panel) = level.item(b.item) else { continue };
            let blocks = b.faces.iter().any(|&(c, d)| {
                let (here, there) = (reach.contains(&c), reach.contains(&step(c, d)));
                ((d == n || d == n.opposite()) && (here || there)) || (here && there)
            });
            if blocks {
                flags.pair(panel, field, "glass-across-field");
            }
        }
        // Fields through the same cells: parallel ones, or a fizzler crossing a laser field
        // (15 of 17 flags agreed).
        for h2 in &level.hazards[i + 1..] {
            let Some(other) = level.item(h2.item) else { continue };
            let Some(&(_, f2)) = h2.faces.first() else { continue };
            if !h2.faces.iter().any(|(c, _)| swept.contains(c)) {
                continue;
            }
            let parallel = f2 == f || f2 == f.opposite();
            // Parallel ones collide on the same side of one cell, whatever their kinds (in-game
            // checks of 2026-10-24: a fizzler and a laser field there, all of it or one cell
            // of it, 3 of 3 flagged), or if they're the same kind in the same plane. A fizzler
            // and a laser field on the two sides of one face pass (1 of 1), and fields in
            // different planes of the same cells don't touch. (The Workshop's fizzler and
            // laser field in one plane are one of them in the middle of the voxel, which
            // tokens can't say; the converter puts it on the voxel's other side.)
            let faces: HashSet<(Cell, Dir)> = h.faces.iter().copied().collect();
            let same_side = h2.faces.iter().any(|f| faces.contains(f));
            let planes: HashSet<(Cell, Dir)> = h.faces.iter().map(|&f| face_key(f)).collect();
            let same_plane = h2.faces.iter().any(|&f| planes.contains(&face_key(f)));
            let collide = if parallel {
                same_side || (h.kind == h2.kind && same_plane)
            } else {
                h.kind != h2.kind
            };
            if collide {
                flags.pair(field, other, "fields-overlap");
            }
        }
    }

    // Voxels the later items are sunk into (EmbeddedVoxels in editoritems.txt: the one behind
    // the surface; a piston's two; under each step of stairs). Two can't share one: the game
    // flagged 75 of 75 such pairs (a laser in the wall a faith plate stands on, say), 2026-10-20.
    let mut sunk: HashMap<Cell, usize> = HashMap::new();
    for (i, it) in level.items.iter().enumerate() {
        let Some(n) = it.normal else { continue };
        let wall = step(it.cell, n.opposite());
        let cells = match it.kind {
            ItemKind::Funnel
            | ItemKind::FaithPlate
            | ItemKind::LaserEmitter
            | ItemKind::LaserCatcher
            | ItemKind::LaserRelay
            | ItemKind::AngledPanel
            | ItemKind::GlassPanel
            | ItemKind::FlipPanel
            | ItemKind::CubeDropper => vec![wall],
            ItemKind::Piston => vec![wall, step(wall, n.opposite())],
            ItemKind::Stairs => match level.stairs.iter().find(|s| s.item == it.index) {
                Some(s) => vec![step(s.cell, Dir::NegZ), step(step(s.cell, s.dir), Dir::NegZ)],
                None => vec![wall],
            },
            _ => continue,
        };
        for c in cells {
            match sunk.get(&c) {
                Some(&j) if j != i => flags.pair(&level.items[j], it, "embedded-overlap"),
                Some(_) => {}
                None => {
                    sunk.insert(c, i);
                }
            }
        }
    }

    // A field mounted on the voxel a faith plate is sunk into: 16 of 18 flagged (2026-10-22).
    // Only the field's first cell counts (its others: 3 of 7).
    for f in level
        .items
        .iter()
        .filter(|i| matches!(i.kind, ItemKind::Fizzler | ItemKind::LaserField))
    {
        let Some(n) = f.normal else { continue };
        let mount = step(f.cell, n.opposite());
        for p in level.of_kind(&ItemKind::FaithPlate) {
            if p.normal.is_some_and(|m| step(p.cell, m.opposite()) == mount) {
                flags.pair(f, p, "embedded-overlap");
            }
        }
    }

    // Two buttons in one cell (13 of 13 flagged), or a cube and a paint dropper (16 of 16),
    // 2026-10-20. Not a door and a button: the game flagged 28 of 108. In-game checks of
    // 2026-10-22 add: a turret on a floor button (27 of 27; pedestals 0 of 32) or on a laser
    // relay on the same surface (6 of 6), a pedestal button whose column stands against an
    // exit door (9 of 9; across the door 1 of 15), a loose cube on a floor button (8 of 8; a
    // dropper's cube there is fine).
    for (i, a) in level.items.iter().enumerate() {
        for b in &level.items[i + 1..] {
            if a.cell != b.cell {
                continue;
            }
            let button = |x: &PlacedItem| matches!(x.kind, ItemKind::FloorButton | ItemKind::PedestalButton);
            let dropper = |x: &PlacedItem| matches!(x.kind, ItemKind::CubeDropper | ItemKind::PaintDropper);
            let collide = |a: &PlacedItem, b: &PlacedItem| match (&a.kind, &b.kind) {
                (ItemKind::Turret, ItemKind::FloorButton) => true,
                (ItemKind::Turret, ItemKind::LaserRelay) => a.normal == b.normal,
                (ItemKind::PedestalButton, ItemKind::ExitDoor) => {
                    b.normal.is_some_and(|n| pedestal_side(a) == Some(n.opposite()))
                }
                (ItemKind::Cube, ItemKind::FloorButton) => level.loose_cube(a.index),
                _ => false,
            };
            if (button(a) && button(b)) || (dropper(a) && dropper(b)) || collide(a, b) || collide(b, a) {
                flags.pair(a, b, "shared-cell");
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
