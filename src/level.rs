//! Semantic view of a puzzle for play: solidity, goo, portalable surfaces, items, and the
//! thin barriers that sit on voxel faces (glass, grating, fizzlers, laser fields).
//!
//! Barrier layout (from the BEE2 style map and Kyle0654/Portal2.Puzzle; ASSUMPTIONS until
//! checked in the editor):
//! - Glass/grating (`ITEM_BARRIER`): local -x is the panel normal, local +z points away from
//!   the surface it is anchored to. Its `ITEM_BARRIER_EXTENT`s (linked by
//!   `CONNECTION_BARRIER_ANCHOR_TO_EXTENT`) are corners of the rectangle of cells. The panel is
//!   the face of each of those cells in the normal direction.
//! - Fizzler/laser field (`ITEM_BARRIER_HAZARD`): local +z is the normal (away from the wall it
//!   is mounted on), local +y the axis its two extents lie along. For each cell between the
//!   extents the field runs along the normal until it hits a wall. It is a plane across the
//!   axis normal × right, drawn on each cell's face on the positive side of that axis (the
//!   editor offsets it 0.375 voxel that way, or centres it).
//!
//! Turrets and gel (from the community style map, fixtures/real/1500854111.p2c):
//! - `ITEM_TURRET` stands on a floor; `Facing` holds its yaw (pitch, yaw, roll in degrees).
//! - `ITEM_PAINT_SPLAT` marks where gel lands, on the surface against its up axis;
//!   `ITEM_PROPERTY_PAINT_TYPE` 0 repulsion, 1 propulsion, 2 conversion, 3 cleansing water.
//!   With `ITEM_PROPERTY_DROPPER_ENABLED` an `ITEM_DROPPER_PAINT` on the ceiling above
//!   (linked by `CONNECTION_PAINT_DROPPER`) pours it; buttons wire to the dropper, whose
//!   `ITEM_PROPERTY_START_ENABLED` says whether it pours from the start. Without a dropper the
//!   splat is paint that is there from the start.

use crate::geom::{rotate, step, up_from_angles, Cell, Dir};
use crate::p2c::{Props, Puzzle, Voxels};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ItemKind {
    EntryDoor,
    ExitDoor,
    Cube,
    CubeDropper,
    FloorButton,
    PedestalButton,
    Goo,
    Glass,
    Grating,
    BarrierExtent,
    Fizzler,
    LaserField,
    HazardExtent,
    Turret,
    PaintSplat,
    PaintDropper,
    Other(String),
}

/// Gel, by `ITEM_PROPERTY_PAINT_TYPE`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PaintKind {
    /// Repulsion (blue): bounces.
    Bounce,
    /// Propulsion (orange): speed.
    Speed,
    /// Conversion (white): portals can go on it.
    Portal,
    /// Cleansing water: washes paint off.
    Water,
}

impl PaintKind {
    pub const ALL: [PaintKind; 4] = [
        PaintKind::Bounce,
        PaintKind::Speed,
        PaintKind::Portal,
        PaintKind::Water,
    ];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CubeKind {
    Standard,
    Companion,
    Reflection,
    Sphere,
    Franken,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ButtonKind {
    /// Pressed by the player or any cube.
    Weighted,
    /// Any cube but the sphere.
    Cube,
    Sphere,
}

impl ButtonKind {
    pub fn accepts(self, cube: CubeKind) -> bool {
        match self {
            ButtonKind::Weighted => true,
            ButtonKind::Cube => cube != CubeKind::Sphere,
            ButtonKind::Sphere => cube == CubeKind::Sphere,
        }
    }
}

impl ItemKind {
    pub fn of(item: &Props) -> ItemKind {
        let alt = |key| item.get_i32(key).unwrap_or(0) != 0;
        match item.kind() {
            "ITEM_ENTRY_DOOR" => ItemKind::EntryDoor,
            "ITEM_EXIT_DOOR" => ItemKind::ExitDoor,
            "ITEM_CUBE" => ItemKind::Cube,
            "ITEM_DROPPER_CUBE" => ItemKind::CubeDropper,
            "ITEM_BUTTON_FLOOR" => ItemKind::FloorButton,
            "ITEM_BUTTON_PEDESTAL" => ItemKind::PedestalButton,
            "ITEM_GOO" => ItemKind::Goo,
            "ITEM_BARRIER" if alt("ITEM_PROPERTY_BARRIER_TYPE") => ItemKind::Grating,
            "ITEM_BARRIER" => ItemKind::Glass,
            "ITEM_BARRIER_EXTENT" => ItemKind::BarrierExtent,
            "ITEM_BARRIER_HAZARD" if alt("ITEM_PROPERTY_BARRIER_HAZARD_TYPE") => {
                ItemKind::LaserField
            }
            "ITEM_BARRIER_HAZARD" => ItemKind::Fizzler,
            "ITEM_BARRIER_HAZARD_EXTENT" => ItemKind::HazardExtent,
            "ITEM_TURRET" => ItemKind::Turret,
            "ITEM_PAINT_SPLAT" => ItemKind::PaintSplat,
            "ITEM_DROPPER_PAINT" => ItemKind::PaintDropper,
            other => ItemKind::Other(other.to_string()),
        }
    }

    /// Items that sit against a surface: their cell is open and the cell behind them (against
    /// their up axis) is solid.
    pub fn is_mounted(&self) -> bool {
        matches!(
            self,
            ItemKind::EntryDoor
                | ItemKind::ExitDoor
                | ItemKind::CubeDropper
                | ItemKind::FloorButton
                | ItemKind::PedestalButton
                | ItemKind::Glass
                | ItemKind::Grating
                | ItemKind::Fizzler
                | ItemKind::LaserField
                | ItemKind::Turret
                | ItemKind::PaintSplat
                | ItemKind::PaintDropper
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
    pub angles: Option<[f64; 3]>,
    pub props: Props,
}

impl PlacedItem {
    fn flag(&self, key: &str) -> Option<bool> {
        self.props.get_i32(key).map(|v| v != 0)
    }

    pub fn cube_kind(&self) -> CubeKind {
        match self.props.get_i32("ITEM_PROPERTY_CUBE_TYPE").unwrap_or(0) {
            1 => CubeKind::Companion,
            2 => CubeKind::Reflection,
            3 => CubeKind::Sphere,
            4 => CubeKind::Franken,
            _ => CubeKind::Standard,
        }
    }

    pub fn button_kind(&self) -> ButtonKind {
        match self.props.get_i32("ITEM_PROPERTY_BUTTON_TYPE").unwrap_or(0) {
            1 => ButtonKind::Cube,
            2 => ButtonKind::Sphere,
            _ => ButtonKind::Weighted,
        }
    }

    /// Pedestal timer in seconds; `None` = infinite (latches on). The editor offers 3..=30
    /// or infinity. ASSUMPTION: infinity is stored as a value below 3.
    pub fn timer(&self) -> Option<f64> {
        let t = self
            .props
            .get_f64s::<1>("ITEM_PROPERTY_TIMER_DELAY")
            .map_or(3.0, |[t]| t);
        (t >= 3.0).then_some(t)
    }

    /// Fizzlers, laser fields and gel droppers: `ITEM_PROPERTY_START_ENABLED` (default on).
    pub fn start_enabled(&self) -> bool {
        self.flag("ITEM_PROPERTY_START_ENABLED").unwrap_or(true)
    }

    pub fn paint_kind(&self) -> PaintKind {
        match self.props.get_i32("ITEM_PROPERTY_PAINT_TYPE").unwrap_or(0) {
            1 => PaintKind::Speed,
            2 => PaintKind::Portal,
            3 => PaintKind::Water,
            _ => PaintKind::Bounce,
        }
    }

    /// Gel splats: poured by a dropper (default), or painted from the start.
    pub fn dropper_enabled(&self) -> bool {
        self.flag("ITEM_PROPERTY_DROPPER_ENABLED").unwrap_or(true)
    }

    /// Yaw in degrees from `Facing` (turrets).
    pub fn yaw(&self) -> Option<f64> {
        self.props.get_f64s::<3>("Facing").map(|[_, yaw, _]| yaw)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Turret {
    pub item: i32,
    pub cell: Cell,
    /// Degrees; 0 faces +x, 90 faces +y.
    pub yaw: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gel {
    /// The splat item.
    pub item: i32,
    pub kind: PaintKind,
    /// The surface it paints: (open cell, direction to the solid side).
    pub face: Face,
    /// The dropper pouring it (the item buttons wire to), if any, and the cell it is in.
    pub dropper: Option<(i32, Cell)>,
    /// Whether the dropper pours from the start.
    pub start_enabled: bool,
}

/// A `CONNECTION_STANDARD` wire: `sender` drives `receiver` (item indices).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Link {
    pub sender: i32,
    pub receiver: i32,
}

/// A face, as (cell, direction to the neighbour across it).
pub type Face = (Cell, Dir);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BarrierKind {
    Glass,
    Grating,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HazardKind {
    /// Lets the player through but clears their portals and destroys cubes; blocks shots.
    Fizzler,
    /// Kills the player; cubes and shots pass.
    LaserField,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Barrier {
    pub item: i32,
    pub kind: BarrierKind,
    pub faces: Vec<Face>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Hazard {
    pub item: i32,
    pub kind: HazardKind,
    pub start_enabled: bool,
    pub faces: Vec<Face>,
}

/// What sits on one face.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FaceInfo {
    pub barrier: Option<BarrierKind>,
    /// Bit `i` = `Level::hazards[i]` has a field here.
    pub hazards: u64,
}

pub const MAX_HAZARDS: usize = 64;
pub const MAX_TURRETS: usize = 32;

pub struct Level {
    pub size: [i32; 3],
    voxels: Voxels,
    goo: Vec<bool>,
    pub items: Vec<PlacedItem>,
    pub links: Vec<Link>,
    pub barriers: Vec<Barrier>,
    pub hazards: Vec<Hazard>,
    pub turrets: Vec<Turret>,
    pub gels: Vec<Gel>,
    /// What sits on each face, at `face_slot`.
    faces: Vec<FaceInfo>,
    /// Items that couldn't be decoded (bad extents, too many hazards).
    pub decode_problems: Vec<String>,
}

/// Faces are stored once, under the cell on their negative side.
fn face_key(c: Cell, d: Dir) -> (Cell, usize) {
    if d.is_positive() {
        (c, d.axis())
    } else {
        (step(c, d), d.axis())
    }
}

impl Level {
    pub fn from_puzzle(p: &Puzzle) -> Level {
        let items: Vec<PlacedItem> = p
            .items
            .iter()
            .filter_map(|it| {
                Some(PlacedItem {
                    index: it.index()?,
                    kind: ItemKind::of(it),
                    cell: it.voxel_pos()?,
                    normal: it.angles().and_then(up_from_angles),
                    start_open: it.get_i32("ITEM_PROPERTY_START_OPEN").map(|v| v != 0),
                    angles: it.angles(),
                    props: it.clone(),
                })
            })
            .collect();
        let wire = |c: &Props| {
            Some(Link {
                sender: c.get_i32("Sender")?,
                receiver: c.get_i32("Receiver")?,
            })
        };
        let links = p
            .connections
            .iter()
            .filter(|c| c.get("Type") == Some("CONNECTION_STANDARD"))
            .filter_map(wire)
            .collect();
        let extent_links: Vec<Link> = p
            .connections
            .iter()
            .filter(|c| c.get("Type") == Some("CONNECTION_BARRIER_ANCHOR_TO_EXTENT"))
            .filter_map(wire)
            .collect();
        let paint_links: Vec<Link> = p
            .connections
            .iter()
            .filter(|c| c.get("Type") == Some("CONNECTION_PAINT_DROPPER"))
            .filter_map(wire)
            .collect();
        let size = p.chamber_size();
        let n = size.iter().product::<i32>() as usize;
        let mut level = Level {
            size,
            voxels: p.voxels.clone(),
            goo: vec![false; n],
            items,
            links,
            barriers: Vec::new(),
            hazards: Vec::new(),
            turrets: Vec::new(),
            gels: Vec::new(),
            faces: vec![
                FaceInfo::default();
                3 * size.map(|s| s as usize + 1).iter().product::<usize>()
            ],
            decode_problems: Vec::new(),
        };
        level.flood_goo();
        level.decode_barriers(&extent_links);
        level.decode_turrets_and_gel(&paint_links);
        level
    }

    fn decode_turrets_and_gel(&mut self, paint_links: &[Link]) {
        for it in self.items.clone() {
            let problem = |what: &str| format!("item {} {:?}: {what}", it.index, it.kind);
            match it.kind {
                ItemKind::Turret => {
                    if it.normal != Some(Dir::PosZ) {
                        self.decode_problems.push(problem("turret not on a floor"));
                        continue;
                    }
                    self.turrets.push(Turret {
                        item: it.index,
                        cell: it.cell,
                        yaw: it.yaw().unwrap_or(0.0),
                    });
                }
                ItemKind::PaintSplat => {
                    let Some(n) = it.normal else {
                        self.decode_problems.push(problem("angles not axis-aligned"));
                        continue;
                    };
                    let mut gel = Gel {
                        item: it.index,
                        kind: it.paint_kind(),
                        face: (it.cell, n.opposite()),
                        dropper: None,
                        start_enabled: true,
                    };
                    if it.dropper_enabled() {
                        let dropper = paint_links
                            .iter()
                            .filter(|l| l.sender == it.index)
                            .find_map(|l| self.item(l.receiver))
                            .filter(|d| d.kind == ItemKind::PaintDropper);
                        let Some(d) = dropper else {
                            self.decode_problems.push(problem("gel splat has no dropper"));
                            continue;
                        };
                        if n != Dir::PosZ {
                            self.decode_problems
                                .push(problem("gel dropper over a splat that isn't on a floor"));
                            continue;
                        }
                        gel.dropper = Some((d.index, d.cell));
                        gel.start_enabled = d.start_enabled();
                    }
                    self.gels.push(gel);
                }
                _ => {}
            }
        }
        if self.turrets.len() > MAX_TURRETS {
            self.decode_problems
                .push(format!("more than {MAX_TURRETS} turrets"));
            self.turrets.truncate(MAX_TURRETS);
        }
    }

    fn decode_barriers(&mut self, extent_links: &[Link]) {
        let items = self.items.clone();
        for it in &items {
            let extents: Vec<Cell> = extent_links
                .iter()
                .filter(|l| l.sender == it.index)
                .filter_map(|l| self.item(l.receiver))
                .map(|e| e.cell)
                .collect();
            let mut corners = extents.clone();
            corners.push(it.cell);
            let lo: Cell = std::array::from_fn(|a| corners.iter().map(|c| c[a]).min().unwrap());
            let hi: Cell = std::array::from_fn(|a| corners.iter().map(|c| c[a]).max().unwrap());
            let axis_dir = |v| it.angles.and_then(|a| rotate(a, v));
            let problem = |what: &str| format!("item {} {:?}: {what}", it.index, it.kind);
            let is_barrier = matches!(
                it.kind,
                ItemKind::Glass | ItemKind::Grating | ItemKind::Fizzler | ItemKind::LaserField
            );
            if is_barrier && corners.iter().any(|&c| !self.in_interior(c)) {
                self.decode_problems
                    .push(problem("extent outside the chamber"));
                continue;
            }
            match it.kind {
                ItemKind::Glass | ItemKind::Grating => {
                    let Some(n) = axis_dir([-1, 0, 0]) else {
                        self.decode_problems
                            .push(problem("angles not axis-aligned"));
                        continue;
                    };
                    if lo[n.axis()] != hi[n.axis()] {
                        self.decode_problems
                            .push(problem("extents not in the panel's plane"));
                        continue;
                    }
                    let faces = cells_between(lo, hi).map(|c| (c, n)).collect();
                    let kind = if it.kind == ItemKind::Glass {
                        BarrierKind::Glass
                    } else {
                        BarrierKind::Grating
                    };
                    self.barriers.push(Barrier {
                        item: it.index,
                        kind,
                        faces,
                    });
                }
                ItemKind::Fizzler | ItemKind::LaserField => {
                    let (Some(n), Some(r)) = (axis_dir([0, 0, 1]), axis_dir([0, 1, 0])) else {
                        self.decode_problems
                            .push(problem("angles not axis-aligned"));
                        continue;
                    };
                    let f = n.cross(r).expect("rotation keeps axes perpendicular");
                    if (0..3).any(|a| a != r.axis() && lo[a] != hi[a]) {
                        self.decode_problems
                            .push(problem("extents not along the field's axis"));
                        continue;
                    }
                    if self.hazards.len() == MAX_HAZARDS {
                        self.decode_problems
                            .push(problem("more than 64 fizzlers/laser fields"));
                        continue;
                    }
                    let mut faces = Vec::new();
                    for base in cells_between(lo, hi) {
                        let mut c = base;
                        while self.in_interior(c) && !self.solid(c) {
                            faces.push((c, f));
                            c = step(c, n);
                        }
                    }
                    let kind = if it.kind == ItemKind::Fizzler {
                        HazardKind::Fizzler
                    } else {
                        HazardKind::LaserField
                    };
                    self.hazards.push(Hazard {
                        item: it.index,
                        kind,
                        start_enabled: it.start_enabled(),
                        faces,
                    });
                }
                _ => {}
            }
        }
        for b in &self.barriers {
            for &(c, d) in &b.faces {
                if let Some(i) = self.face_slot(c, d) {
                    self.faces[i].barrier = Some(b.kind);
                }
            }
        }
        for (i, h) in self.hazards.iter().enumerate() {
            for &(c, d) in &h.faces {
                if let Some(f) = self.face_slot(c, d) {
                    self.faces[f].hazards |= 1 << i;
                }
            }
        }
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

    /// Barriers and hazard fields on the face between `c` and `step(c, d)`.
    pub fn face(&self, c: Cell, d: Dir) -> FaceInfo {
        self.face_slot(c, d)
            .map_or_else(FaceInfo::default, |i| self.faces[i])
    }

    /// Where `faces` keeps the face between `c` and `step(c, d)`: its cells on the negative
    /// side run from -1 to `size - 1` on each axis.
    fn face_slot(&self, c: Cell, d: Dir) -> Option<usize> {
        let (k, axis) = face_key(c, d);
        let [x, y, z] = k.map(|v| v + 1);
        let [sx, sy, sz] = self.size.map(|s| s + 1);
        let inside = x >= 0 && y >= 0 && z >= 0 && x < sx && y < sy && z < sz;
        inside.then(|| 3 * ((z * sy + y) * sx + x) as usize + axis)
    }

    /// Something the player and cubes can stand on or are stopped by: solid, glass, grating.
    fn floor_under(&self, c: Cell) -> bool {
        self.solid(step(c, Dir::NegZ)) || self.face(c, Dir::NegZ).barrier.is_some()
    }

    /// Open cell with a floor (or a glass/grating panel) under it that isn't goo: somewhere the
    /// player can stand.
    pub fn standable(&self, c: Cell) -> bool {
        !self.solid(c) && self.floor_under(c) && !self.goo(c)
    }

    /// Face between `c` and `step(c, d)` is a surface (one side solid) and is white.
    pub fn portal_surface(&self, c: Cell, d: Dir) -> bool {
        self.solid(c) != self.solid(step(c, d)) && self.voxels.face_portalable(c, d)
    }

    /// Where something falling from open cell `c` comes to rest, and the hazard fields it
    /// falls through. `None` if it ends in goo or `c` is solid.
    pub fn fall_path(&self, c: Cell) -> Option<(Cell, u64)> {
        if self.solid(c) {
            return None;
        }
        let mut f = c;
        let mut hazards = 0;
        while !self.goo(f) && !self.floor_under(f) {
            hazards |= self.face(f, Dir::NegZ).hazards;
            f = step(f, Dir::NegZ);
        }
        (!self.goo(f)).then_some((f, hazards))
    }

    /// Where something falling from open cell `c` comes to rest. `None` if it ends in goo or
    /// `c` is solid.
    pub fn fall(&self, c: Cell) -> Option<Cell> {
        self.fall_path(c).map(|(f, _)| f)
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

    /// Mounted items whose cell isn't open or that have nothing solid behind them, plus
    /// barriers that couldn't be decoded.
    pub fn mount_problems(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .items
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
            .collect();
        // Goo is placed on a floor (seen in-game: one on top of goo is invalid).
        for it in self.items.iter().filter(|i| i.kind == ItemKind::Goo) {
            if !self.solid(step(it.cell, Dir::NegZ)) {
                out.push(format!(
                    "item {} Goo at {:?} isn't on a floor",
                    it.index, it.cell
                ));
            }
        }
        out.extend(self.field_problems());
        out.extend(self.decode_problems.iter().cloned());
        out
    }

    /// A fizzler or laser field fills every cell it sweeps, and the editor marks any other
    /// item there as misplaced and won't build (seen in-game: a pedestal button and both doors
    /// in a fizzler's sweep). Panels and other fields count too, to be safe.
    fn field_problems(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (i, h) in self.hazards.iter().enumerate() {
            let swept: std::collections::HashSet<Cell> = h.faces.iter().map(|&(c, _)| c).collect();
            let what = format!("{:?} {}", h.kind, h.item);
            for it in &self.items {
                let blocks = matches!(
                    it.kind,
                    ItemKind::EntryDoor
                        | ItemKind::ExitDoor
                        | ItemKind::Cube
                        | ItemKind::CubeDropper
                        | ItemKind::FloorButton
                        | ItemKind::PedestalButton
                        | ItemKind::Turret
                        | ItemKind::PaintSplat
                        | ItemKind::PaintDropper
                );
                if blocks && swept.contains(&it.cell) {
                    out.push(format!(
                        "item {} {:?} at {:?} is in the sweep of {what}",
                        it.index, it.kind, it.cell
                    ));
                }
            }
            // The emitters sit on the mounting wall and are wider than the field: an item on the
            // same wall in the cell either side of them doesn't fit (seen in-game: an exit door
            // there). A door beside the field's far end built fine.
            let n = self.item(h.item).and_then(|it| it.normal);
            let side = h.faces.first().map(|&(_, f)| f);
            if let (Some(n), Some(side)) = (n, side) {
                for &(c, _) in &h.faces {
                    if !self.solid(step(c, n.opposite())) {
                        continue;
                    }
                    for beside in [step(c, side), step(c, side.opposite())] {
                        for it in &self.items {
                            if it.kind.is_mounted()
                                && it.index != h.item
                                && it.normal == Some(n)
                                && it.cell == beside
                                && !swept.contains(&beside)
                            {
                                out.push(format!(
                                    "item {} {:?} at {:?} is beside the emitters of {what}",
                                    it.index, it.kind, it.cell
                                ));
                            }
                        }
                    }
                }
            }
            for b in &self.barriers {
                if b.faces.iter().any(|(c, _)| swept.contains(c)) {
                    out.push(format!(
                        "{:?} {} crosses the sweep of {what}",
                        b.kind, b.item
                    ));
                }
            }
            for other in &self.hazards[i + 1..] {
                if other.faces.iter().any(|(c, _)| swept.contains(c)) {
                    out.push(format!(
                        "{:?} {} crosses the sweep of {what}",
                        other.kind, other.item
                    ));
                }
            }
        }
        out
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

/// Cells of the inclusive box `lo..=hi`.
pub fn cells_between(lo: Cell, hi: Cell) -> impl Iterator<Item = Cell> {
    (lo[2]..=hi[2]).flat_map(move |z| {
        (lo[1]..=hi[1]).flat_map(move |y| (lo[0]..=hi[0]).map(move |x| [x, y, z]))
    })
}
