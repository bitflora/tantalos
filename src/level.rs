//! Semantic view of a puzzle for play: solidity, goo, portalable surfaces, Tier-0 items.

use crate::geom::{step, up_from_angles, Cell, Dir};
use crate::p2c::{Puzzle, Voxels};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ItemKind {
    EntryDoor,
    ExitDoor,
    Cube,
    CubeDropper,
    FloorButton,
    Goo,
    Other(String),
}

impl ItemKind {
    pub fn from_type(t: &str) -> ItemKind {
        match t {
            "ITEM_ENTRY_DOOR" => ItemKind::EntryDoor,
            "ITEM_EXIT_DOOR" => ItemKind::ExitDoor,
            "ITEM_CUBE" => ItemKind::Cube,
            "ITEM_DROPPER_CUBE" => ItemKind::CubeDropper,
            "ITEM_BUTTON_FLOOR" => ItemKind::FloorButton,
            "ITEM_GOO" => ItemKind::Goo,
            other => ItemKind::Other(other.to_string()),
        }
    }

    /// Items that sit against a surface: their cell is open and the cell behind them is solid.
    pub fn is_mounted(&self) -> bool {
        matches!(
            self,
            ItemKind::EntryDoor
                | ItemKind::ExitDoor
                | ItemKind::CubeDropper
                | ItemKind::FloorButton
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlacedItem {
    pub index: i32,
    pub kind: ItemKind,
    pub cell: Cell,
    /// Surface normal (the item's up), if its angles are axis-aligned.
    pub normal: Option<Dir>,
    /// `ITEM_PROPERTY_START_OPEN`, for items that have it (the exit door).
    pub start_open: Option<bool>,
}

/// A `CONNECTION_STANDARD` wire: `sender` drives `receiver` (item indices).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Link {
    pub sender: i32,
    pub receiver: i32,
}

pub struct Level {
    pub size: [i32; 3],
    voxels: Voxels,
    goo: Vec<bool>,
    pub items: Vec<PlacedItem>,
    pub links: Vec<Link>,
}

impl Level {
    pub fn from_puzzle(p: &Puzzle) -> Level {
        let items = p
            .items
            .iter()
            .filter_map(|it| {
                Some(PlacedItem {
                    index: it.index()?,
                    kind: ItemKind::from_type(it.kind()),
                    cell: it.voxel_pos()?,
                    normal: it.angles().and_then(up_from_angles),
                    start_open: it.get_i32("ITEM_PROPERTY_START_OPEN").map(|v| v != 0),
                })
            })
            .collect();
        let links = p
            .connections
            .iter()
            .filter(|c| c.get("Type") == Some("CONNECTION_STANDARD"))
            .filter_map(|c| {
                Some(Link {
                    sender: c.get_i32("Sender")?,
                    receiver: c.get_i32("Receiver")?,
                })
            })
            .collect();
        let size = p.chamber_size();
        let n = size.iter().product::<i32>() as usize;
        let mut level = Level {
            size,
            voxels: p.voxels.clone(),
            goo: vec![false; n],
            items,
            links,
        };
        level.flood_goo();
        level
    }

    fn idx(&self, c: Cell) -> Option<usize> {
        self.in_interior(c)
            .then(|| ((c[2] * self.size[1] + c[1]) * self.size[0] + c[0]) as usize)
    }

    pub fn in_interior(&self, c: Cell) -> bool {
        (0..3).all(|a| c[a] >= 0 && c[a] < self.size[a])
    }

    /// Everything outside the interior counts as solid.
    pub fn solid(&self, c: Cell) -> bool {
        !self.in_interior(c) || self.voxels.solid(c).unwrap_or(true)
    }

    pub fn goo(&self, c: Cell) -> bool {
        self.idx(c).is_some_and(|i| self.goo[i])
    }

    /// Open cell with a floor under it that isn't goo: somewhere the player can stand.
    pub fn standable(&self, c: Cell) -> bool {
        !self.solid(c) && self.solid(step(c, Dir::NegZ)) && !self.goo(c)
    }

    /// Face between `c` and `step(c, d)` is a surface (one side solid) and is white.
    pub fn portal_surface(&self, c: Cell, d: Dir) -> bool {
        self.solid(c) != self.solid(step(c, d)) && self.voxels.face_portalable(c, d)
    }

    /// Where something falling from open cell `c` comes to rest: the first cell down with a
    /// floor under it. `None` if it ends in goo or `c` is solid.
    pub fn fall(&self, c: Cell) -> Option<Cell> {
        if self.solid(c) {
            return None;
        }
        let mut f = c;
        while !self.goo(f) && !self.solid(step(f, Dir::NegZ)) {
            f = step(f, Dir::NegZ);
        }
        (!self.goo(f)).then_some(f)
    }

    /// Every portalable surface, as (open cell, direction to the solid side).
    pub fn portal_surfaces(&self) -> Vec<(Cell, Dir)> {
        let mut out = Vec::new();
        for c in self.cells() {
            if self.solid(c) {
                continue;
            }
            for d in Dir::ALL {
                if self.solid(step(c, d)) && self.voxels.face_portalable(c, d) {
                    out.push((c, d));
                }
            }
        }
        out
    }

    /// Interior cells in z, y, x order.
    pub fn cells(&self) -> impl Iterator<Item = Cell> + '_ {
        let [sx, sy, sz] = self.size;
        (0..sz).flat_map(move |z| (0..sy).flat_map(move |y| (0..sx).map(move |x| [x, y, z])))
    }

    pub fn first(&self, kind: &ItemKind) -> Option<&PlacedItem> {
        self.items.iter().find(|i| &i.kind == kind)
    }

    pub fn of_kind<'a>(&'a self, kind: &'a ItemKind) -> impl Iterator<Item = &'a PlacedItem> + 'a {
        self.items.iter().filter(move |i| &i.kind == kind)
    }

    pub fn item(&self, index: i32) -> Option<&PlacedItem> {
        self.items.iter().find(|i| i.index == index)
    }

    /// Senders wired into item `receiver`.
    pub fn inputs(&self, receiver: i32) -> impl Iterator<Item = i32> + '_ {
        self.links
            .iter()
            .filter(move |l| l.receiver == receiver)
            .map(|l| l.sender)
    }

    pub fn entry(&self) -> Option<&PlacedItem> {
        self.first(&ItemKind::EntryDoor)
    }

    pub fn exit(&self) -> Option<&PlacedItem> {
        self.first(&ItemKind::ExitDoor)
    }

    /// Mounted items whose cell isn't open or that have nothing solid behind them.
    pub fn mount_problems(&self) -> Vec<String> {
        self.items
            .iter()
            .filter(|i| i.kind.is_mounted())
            .filter_map(|i| {
                let Some(n) = i.normal else {
                    return Some(format!(
                        "item {} {:?}: angles not axis-aligned",
                        i.index, i.kind
                    ));
                };
                if self.solid(i.cell) {
                    Some(format!(
                        "item {} {:?} at {:?} is inside a solid voxel",
                        i.index, i.kind, i.cell
                    ))
                } else if !self.solid(step(i.cell, n.opposite())) {
                    Some(format!(
                        "item {} {:?} at {:?} has nothing behind it ({:?})",
                        i.index,
                        i.kind,
                        i.cell,
                        n.opposite()
                    ))
                } else {
                    None
                }
            })
            .collect()
    }

    /// ASSUMPTION (verify in-game): goo fills every open cell reachable from a goo item
    /// sideways or downward, i.e. it floods the whole pit at its level.
    fn flood_goo(&mut self) {
        let mut stack: Vec<Cell> = self
            .items
            .iter()
            .filter(|i| i.kind == ItemKind::Goo)
            .map(|i| i.cell)
            .collect();
        while let Some(c) = stack.pop() {
            let Some(i) = self.idx(c) else { continue };
            if self.goo[i] || self.solid(c) {
                continue;
            }
            self.goo[i] = true;
            for d in [Dir::NegX, Dir::PosX, Dir::NegY, Dir::PosY, Dir::NegZ] {
                stack.push(step(c, d));
            }
        }
    }
}
