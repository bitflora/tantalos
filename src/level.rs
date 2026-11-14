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
//!
//! The other Puzzle Maker items (from `portal2_dlc2/scripts/editoritems.txt`, the instances in
//! `sdk_content/maps/instances/p2editor` and the community style map; ASSUMPTIONS where noted):
//! - Loose cubes: an `ITEM_CUBE` whose `ITEM_DROPPER_CUBE` (still linked by
//!   `CONNECTION_BOX_DROPPER`) has `ITEM_PROPERTY_DROPPER_ENABLED` 0. A fizzled one is gone.
//! - Laser emitters, catchers and relays (`_CENTER` and `_OFFSET`): in the open cell in front of
//!   their surface; up is the normal, along which an emitter fires. `_OFFSET` ones sit off the
//!   middle of the face toward local -x (editoritems' EmbedFace), still in the same cell.
//! - Faith plates (`ITEM_CATAPULT`) launch whatever lands on them to their target
//!   (`ITEM_CATAPULT_TARGET`, linked by `CONNECTION_CATAPULT_TARGET`).
//! - Light bridges and excursion funnels (`ITEM_LIGHT_BRIDGE`, `ITEM_TBEAM`) project along their
//!   up. A bridge's walkable side faces its local +x (the style map's wall bridge has +x up; its
//!   `LocalPos` puts it 0.375 voxel low, so it is the floor of the cells it crosses).
//! - Angled panels (`ITEM_PANEL_ANGLED`, glass `ITEM_PANEL_CLEAR`) cover the face behind their
//!   cell and, deployed, swing out by `ITEM_PROPERTY_ANGLED_PANEL_ANIMATION` (ramp_N_deg_open):
//!   local +x points to the edge that swings out, the hinge is on the opposite edge, and the face
//!   leans toward the hinge (ASSUMPTION; Workshop wall panels point local +x down, so they lean up
//!   and out, the way flings off them go). Flip panels (`ITEM_PANEL_FLIP`) swap
//!   the face's portalability when flipped (ASSUMPTION: unflipped is the tile's own).
//! - Stairs (`ITEM_PANEL_STAIRS`) take two floor cells, the item's and the next along local +x,
//!   and deployed climb one voxel toward local +x (stairs.vmf: four steps of 32 over 256 units).
//! - Piston platforms (`ITEM_PISTON_PLATFORM`) push out of the surface behind the item's cell
//!   along its up: at level L they fill the L cells from the item's (`BOTTOM_LEVEL`/`TOP_LEVEL`,
//!   `START_UP`). On a floor, that makes the cell L above the item's a floor. Wired to a
//!   button, a piston starts the other way round from `START_UP` (measured in-game; the
//!   solver flips it).
//! - Rail platforms (`ITEM_RAIL_PLATFORM`) travel from the item's cell to the cell of the extent
//!   farthest from it (their angles don't follow the up = normal rule). The platform is the top
//!   face of the voxel it is in: whoever rides it stands in the cell above (in all 9,264 rails of
//!   the compiled Workshop maps, any way up, the platform's top is the top of its voxel, and
//!   the editor puts every item's instance at the middle of its voxel; Workshop 3718840820
//!   "Flying a Timing", play-tested, has a lift from a goo voxel up to a walkway a voxel over
//!   its track).

use std::collections::{HashMap, HashSet};

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
    /// A cube dropper switched off: its cube is a loose cube.
    IdleDropper,
    LaserEmitter,
    LaserCatcher,
    LaserRelay,
    FaithPlate,
    PlateTarget,
    LightBridge,
    Funnel,
    AngledPanel,
    GlassPanel,
    FlipPanel,
    Stairs,
    Piston,
    PistonExtent,
    Rail,
    RailExtent,
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
            "ITEM_DROPPER_CUBE" if item.get_i32("ITEM_PROPERTY_DROPPER_ENABLED") == Some(0) => {
                ItemKind::IdleDropper
            }
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
            "ITEM_LASER_EMITTER_CENTER" | "ITEM_LASER_EMITTER_OFFSET" => ItemKind::LaserEmitter,
            "ITEM_LASER_CATCHER_CENTER" | "ITEM_LASER_CATCHER_OFFSET" => ItemKind::LaserCatcher,
            "ITEM_LASER_RELAY_CENTER" | "ITEM_LASER_RELAY_OFFSET" => ItemKind::LaserRelay,
            "ITEM_CATAPULT" => ItemKind::FaithPlate,
            "ITEM_CATAPULT_TARGET" => ItemKind::PlateTarget,
            "ITEM_LIGHT_BRIDGE" => ItemKind::LightBridge,
            "ITEM_TBEAM" => ItemKind::Funnel,
            "ITEM_PANEL_ANGLED" => ItemKind::AngledPanel,
            "ITEM_PANEL_CLEAR" => ItemKind::GlassPanel,
            "ITEM_PANEL_FLIP" => ItemKind::FlipPanel,
            "ITEM_PANEL_STAIRS" => ItemKind::Stairs,
            "ITEM_PISTON_PLATFORM" => ItemKind::Piston,
            "ITEM_PISTON_PLATFORM_EXTENT" => ItemKind::PistonExtent,
            "ITEM_RAIL_PLATFORM" => ItemKind::Rail,
            "ITEM_RAIL_PLATFORM_EXTENT" => ItemKind::RailExtent,
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
                | ItemKind::LaserEmitter
                | ItemKind::LaserCatcher
                | ItemKind::LaserRelay
                | ItemKind::FaithPlate
                | ItemKind::LightBridge
                | ItemKind::Funnel
                | ItemKind::AngledPanel
                | ItemKind::GlassPanel
                | ItemKind::FlipPanel
                | ItemKind::Stairs
                | ItemKind::Piston
        )
    }

    /// Items that can drive others over a `CONNECTION_STANDARD` wire.
    pub fn is_sender(&self) -> bool {
        matches!(
            self,
            ItemKind::FloorButton
                | ItemKind::PedestalButton
                | ItemKind::LaserCatcher
                | ItemKind::LaserRelay
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

    /// The world direction of the item's local axis `v`, if axis-aligned.
    pub fn axis(&self, v: [i32; 3]) -> Option<Dir> {
        self.angles.and_then(|a| rotate(a, v))
    }

    /// Panels and stairs: `ITEM_PROPERTY_START_DEPLOYED` (default on).
    pub fn start_deployed(&self) -> bool {
        self.flag("ITEM_PROPERTY_START_DEPLOYED").unwrap_or(true)
    }

    /// Funnels: `ITEM_PROPERTY_START_REVERSED`.
    pub fn start_reversed(&self) -> bool {
        self.flag("ITEM_PROPERTY_START_REVERSED").unwrap_or(false)
    }

    /// Angled panels: `ITEM_PROPERTY_PORTALABLE` (default on).
    pub fn portalable(&self) -> bool {
        self.flag("ITEM_PROPERTY_PORTALABLE").unwrap_or(true)
    }

    /// Angled panels: the tilt in degrees, from `ramp_N_deg_open` (default 45).
    pub fn panel_angle(&self) -> i32 {
        self.props
            .get("ITEM_PROPERTY_ANGLED_PANEL_ANIMATION")
            .and_then(|a| a.strip_prefix("ramp_"))
            .and_then(|a| a.split('_').next())
            .and_then(|n| n.parse().ok())
            .unwrap_or(45)
    }

    /// Laser items: where an `_OFFSET` one sits on its face (local -x); `None` when centred.
    pub fn laser_offset(&self) -> Option<Dir> {
        if self.props.kind().ends_with("_OFFSET") {
            self.axis([-1, 0, 0])
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Turret {
    pub item: i32,
    pub cell: Cell,
    /// The item's `Facing` yaw in degrees (0 = +x, 90 = +y), as in the .p2c and the tokens.
    /// The turret looks the other way: see `look_yaw`.
    pub yaw: f64,
}

impl Turret {
    /// Where the turret looks and shoots, in degrees. The Puzzle Maker's turret instance
    /// (sdk_content/maps/instances/p2editor/turret.vmf) turns its npc 180° from the item's
    /// Facing; checked in the game 2026-10-13: Facing 0 shoots a player 3 cells towards -x and
    /// ignores one towards +x.
    pub fn look_yaw(&self) -> f64 {
        self.yaw + 180.0
    }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum LaserKind {
    Emitter,
    Catcher,
    Relay,
}

/// A laser emitter, catcher or relay.
#[derive(Clone, Debug, PartialEq)]
pub struct Laser {
    pub item: i32,
    pub kind: LaserKind,
    /// The open cell in front of its surface.
    pub cell: Cell,
    /// The surface's normal: an emitter fires along it, a catcher takes a beam coming against
    /// it, a relay takes one crossing its cell along the surface.
    pub normal: Dir,
    /// `_OFFSET` items: the side of the face they sit toward (`None`: centred).
    pub offset: Option<Dir>,
    /// Emitters: whether they fire from the start.
    pub start_enabled: bool,
}

/// A faith plate and its target.
#[derive(Clone, Debug, PartialEq)]
pub struct Plate {
    pub item: i32,
    pub cell: Cell,
    pub normal: Dir,
    /// The target item: its index, its cell (whatever is launched comes down from there) and
    /// its surface's normal. For a plate throwing straight up, the cell the throw reaches
    /// (the target item itself lies on the plate).
    pub target: (i32, Cell, Dir),
    /// Whether a simple arc from the plate to the target misses solid voxels (`clear_arc`).
    pub clear: bool,
    /// The cell the tokens name as the target: `target.1`, except for a plate throwing straight
    /// up past the ceiling over it, where it is the cell the throw would reach (`target.1` is
    /// then the top of the open column, where the solver lands the throw).
    pub written: Cell,
}

/// How many voxels up a floor plate throwing straight up with `CatapultHeight` `h` reaches:
/// the game throws from a trigger 56 units over the floor at √(76800 h) units/s, rising 64 h
/// units (in-game exports of 2026-10-24, h = 1..8).
pub fn vertical_reach(h: f64) -> i32 {
    ((56.0 + 64.0 * h) / 128.0).floor() as i32
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ProjectorKind {
    Bridge,
    Funnel,
}

/// A light bridge or excursion funnel emitter.
#[derive(Clone, Debug, PartialEq)]
pub struct Projector {
    pub item: i32,
    pub kind: ProjectorKind,
    pub cell: Cell,
    /// It projects along this.
    pub normal: Dir,
    /// Bridges: the walkable side of the bridge (perpendicular to `normal`).
    pub side: Dir,
    pub start_enabled: bool,
    /// Funnels: pulling toward the emitter from the start.
    pub start_reversed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PanelKind {
    Angled,
    Glass,
    Flip,
}

/// An angled panel (white/black or glass) or a flip panel, on the face behind `cell`.
#[derive(Clone, Debug, PartialEq)]
pub struct Panel {
    pub item: i32,
    pub kind: PanelKind,
    pub cell: Cell,
    pub normal: Dir,
    /// Angled panels: their local +x, toward the edge that swings out (see `hinge`).
    pub tilt: Dir,
    /// Angled panels: degrees (30, 45, 60 or 90); flip panels 180.
    pub angle: i32,
    /// Angled panels: deployed from the start. Flip panels: never flipped at the start.
    pub start_deployed: bool,
}

impl Panel {
    /// The edge an angled panel swings on, and the way its face leans when deployed.
    pub fn hinge(&self) -> Dir {
        self.tilt.opposite()
    }
}

/// Stairs on the floors of `cell` and the next cell along `dir`, rising toward `dir`.
#[derive(Clone, Debug, PartialEq)]
pub struct Stairs {
    pub item: i32,
    pub cell: Cell,
    pub dir: Dir,
    pub start_deployed: bool,
}

impl Stairs {
    /// Where the deployed stairs lead: the cell over the solid voxel past their second cell.
    pub fn top(&self) -> Cell {
        step(step(step(self.cell, self.dir), self.dir), Dir::PosZ)
    }

    /// Where the deployed stairs can be left, with the hazards crossed on the way: `top`, and
    /// sideways off the top step onto a ledge level with it (Workshop 3818309212 "???":
    /// stairs facing a wall, the exit ledge beside their top end; a play-tester walked it).
    pub fn exits(&self, level: &Level) -> Vec<(Cell, u64)> {
        let over = step(step(self.cell, self.dir), Dir::PosZ);
        let mut out = vec![(self.top(), 0)];
        if !level.solid(over) {
            for side in Dir::HORIZONTAL {
                let (t, face) = (step(over, side), level.face(over, side));
                if side.axis() != self.dir.axis() && face.barrier.is_none() && level.standable(t) {
                    out.push((t, face.hazards));
                }
            }
        }
        out
    }
}

/// A piston platform pushing out of the surface behind `cell` along `normal`: down, it fills
/// the `bottom` cells from `cell` on; up, the `top` cells.
#[derive(Clone, Debug, PartialEq)]
pub struct Piston {
    pub item: i32,
    pub cell: Cell,
    pub normal: Dir,
    pub bottom: i32,
    pub top: i32,
    pub start_up: bool,
}

impl Piston {
    /// The cell `l` out from `cell` along `normal`.
    pub fn at(&self, l: i32) -> Cell {
        let v = self.normal.vector();
        std::array::from_fn(|a| self.cell[a] + v[a] * l)
    }

    /// On a floor: the cell standing on the platform at its bottom or top.
    pub fn stand(&self, up: bool) -> Cell {
        self.at(if up { self.top } else { self.bottom })
    }
}

/// A rail platform: the top of voxel `cell`, travelling to `end` (along one axis).
#[derive(Clone, Debug, PartialEq)]
pub struct Rail {
    pub item: i32,
    pub cell: Cell,
    pub end: Cell,
    /// Back and forth for as long as it's active.
    pub oscillate: bool,
    pub start_active: bool,
    /// Units per second (`ITEM_PROPERTY_RAIL_SPEED`, the editor's 100 by default).
    pub speed: f64,
}

impl Rail {
    /// The voxels it moves through, from `cell` to `end`.
    pub fn track(&self) -> Vec<Cell> {
        let a = (0..3).find(|&a| self.cell[a] != self.end[a]).unwrap_or(0);
        let d = (self.end[a] - self.cell[a]).signum();
        let n = (self.end[a] - self.cell[a]).abs();
        (0..=n)
            .map(|k| {
                let mut c = self.cell;
                c[a] += d * k;
                c
            })
            .collect()
    }

    /// Where whoever rides it stands at each cell of `track`: the cell above.
    pub fn stops(&self) -> Vec<Cell> {
        self.track().into_iter().map(|c| step(c, Dir::PosZ)).collect()
    }
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
    /// The angled or glass panel (index in `Level::panels`) that, deployed at 90°, stands
    /// across this face (on its hinge edge; `dynamics`). It stops shots while deployed.
    pub panel: Option<usize>,
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
    /// Laser emitters, catchers and relays, in item order.
    pub lasers: Vec<Laser>,
    pub plates: Vec<Plate>,
    pub projectors: Vec<Projector>,
    pub panels: Vec<Panel>,
    pub stairs: Vec<Stairs>,
    pub pistons: Vec<Piston>,
    pub rails: Vec<Rail>,
    /// `CONNECTION_TBEAM_POLARITY` wires: they reverse a funnel.
    pub polarity_links: Vec<Link>,
    /// Each cube's dropper, if it has one that is switched on.
    pub dropper_of: HashMap<i32, i32>,
    /// Faith plate cells (index into `plates`): whatever comes to them is launched.
    launch: HashMap<Cell, usize>,
    /// The first floor button in each cell (its item index), so `Landing::at` doesn't scan
    /// every item for each landing it builds.
    button_at: HashMap<Cell, i32>,
    /// The first turret in each cell (index into `turrets`), for the same reason.
    turret_at: HashMap<Cell, usize>,
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
        let typed = |t: &str| -> Vec<Link> {
            p.connections
                .iter()
                .filter(|c| c.get("Type") == Some(t))
                .filter_map(wire)
                .collect()
        };
        let paint_links = typed("CONNECTION_PAINT_DROPPER");
        let plate_links = typed("CONNECTION_CATAPULT_TARGET");
        let polarity_links = typed("CONNECTION_TBEAM_POLARITY");
        let box_links = typed("CONNECTION_BOX_DROPPER");
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
            lasers: Vec::new(),
            plates: Vec::new(),
            projectors: Vec::new(),
            panels: Vec::new(),
            stairs: Vec::new(),
            pistons: Vec::new(),
            rails: Vec::new(),
            polarity_links,
            dropper_of: HashMap::new(),
            launch: HashMap::new(),
            button_at: HashMap::new(),
            turret_at: HashMap::new(),
            faces: vec![
                FaceInfo::default();
                3 * size.map(|s| s as usize + 1).iter().product::<usize>()
            ],
            decode_problems: Vec::new(),
        };
        level.flood_goo();
        level.decode_barriers(&extent_links);
        level.decode_turrets_and_gel(&paint_links);
        level.decode_devices(&extent_links, &plate_links, &box_links);
        level.mark_panel_faces();
        level.index_cells();
        level
    }

    /// `FaceInfo::panel`: the face each 90° angled or glass panel stands across when deployed.
    fn mark_panel_faces(&mut self) {
        for (i, p) in self.panels.iter().enumerate() {
            if p.kind == PanelKind::Flip || p.angle != 90 {
                continue;
            }
            if let Some(f) = self.face_slot(p.cell, p.hinge()) {
                self.faces[f].panel.get_or_insert(i);
            }
        }
    }

    /// Index the floor buttons and turrets by cell, for the per-landing lookups the solver's
    /// model building makes (`Landing::at`).
    fn index_cells(&mut self) {
        for it in &self.items {
            if it.kind == ItemKind::FloorButton {
                self.button_at.entry(it.cell).or_insert(it.index);
            }
        }
        for (i, t) in self.turrets.iter().enumerate() {
            self.turret_at.entry(t.cell).or_insert(i);
        }
    }

    /// Lasers, faith plates, bridges, funnels, panels, stairs, pistons, rails and loose cubes.
    fn decode_devices(&mut self, extent_links: &[Link], plate_links: &[Link], box_links: &[Link]) {
        for l in box_links {
            let on = self
                .item(l.receiver)
                .is_some_and(|d| d.kind == ItemKind::CubeDropper);
            if on {
                self.dropper_of.insert(l.sender, l.receiver);
            }
        }
        for it in self.items.clone() {
            let problem = |what: &str| format!("item {} {:?}: {what}", it.index, it.kind);
            let Some(n) = it.normal else {
                continue; // `mount_problems` reports it.
            };
            match it.kind {
                ItemKind::LaserEmitter | ItemKind::LaserCatcher | ItemKind::LaserRelay => {
                    let kind = match it.kind {
                        ItemKind::LaserEmitter => LaserKind::Emitter,
                        ItemKind::LaserCatcher => LaserKind::Catcher,
                        _ => LaserKind::Relay,
                    };
                    self.lasers.push(Laser {
                        item: it.index,
                        kind,
                        cell: it.cell,
                        normal: n,
                        offset: it.laser_offset(),
                        start_enabled: it.start_enabled(),
                    });
                }
                ItemKind::FaithPlate => {
                    let target = plate_links
                        .iter()
                        .filter(|l| l.sender == it.index)
                        .find_map(|l| self.item(l.receiver))
                        .filter(|t| t.kind == ItemKind::PlateTarget);
                    let Some(t) = target else {
                        self.decode_problems.push(problem("faith plate without a target"));
                        continue;
                    };
                    let Some(tn) = t.normal else {
                        self.decode_problems.push(problem("target angles not axis-aligned"));
                        continue;
                    };
                    if self.solid(t.cell) {
                        self.decode_problems.push(problem("target isn't in an open cell"));
                        continue;
                    }
                    // A floor plate marked vertical throws straight up wherever its target is
                    // (in-game checks, 2026-10-24), as high as `CatapultHeight` says. Like any
                    // target, that cell may be over an obstacle (`clear` says).
                    let vertical = it.props.get_i32("ITEM_PROPERTY_VERTICAL_ALIGNMENT") == Some(1);
                    let height = it.props.get_f64s::<1>("CatapultHeight").map(|[h]| h);
                    let apex = height.map(|h| {
                        let mut c = it.cell;
                        c[2] += vertical_reach(h);
                        c
                    });
                    // A throw past the ceiling over the plate stops there, as far as the solver
                    // goes (a player drifting off the column on the way up can climb on; the
                    // tokens keep the full height in `written`).
                    let top = apex.filter(|c| vertical && n == Dir::PosZ && c[2] > it.cell[2]).map(|c| {
                        let mut top = it.cell;
                        while top[2] < c[2] && !self.solid([top[0], top[1], top[2] + 1]) {
                            top[2] += 1;
                        }
                        (top, c)
                    });
                    let (tc, tn, written) = match top {
                        Some((top, c)) if top != it.cell => (top, Dir::PosZ, c),
                        _ => (t.cell, tn, t.cell),
                    };
                    let clear = self.clear_arc(it.cell, tc);
                    self.plates.push(Plate {
                        item: it.index,
                        cell: it.cell,
                        normal: n,
                        target: (t.index, tc, tn),
                        clear,
                        written,
                    });
                }
                ItemKind::LightBridge | ItemKind::Funnel => {
                    let kind = if it.kind == ItemKind::LightBridge {
                        ProjectorKind::Bridge
                    } else {
                        ProjectorKind::Funnel
                    };
                    let side = it.axis([1, 0, 0]).filter(|s| s.axis() != n.axis());
                    let Some(side) = side else {
                        self.decode_problems.push(problem("side not axis-aligned"));
                        continue;
                    };
                    self.projectors.push(Projector {
                        item: it.index,
                        kind,
                        cell: it.cell,
                        normal: n,
                        side,
                        start_enabled: it.start_enabled(),
                        start_reversed: it.start_reversed(),
                    });
                }
                ItemKind::AngledPanel | ItemKind::GlassPanel | ItemKind::FlipPanel => {
                    let kind = match it.kind {
                        ItemKind::AngledPanel => PanelKind::Angled,
                        ItemKind::GlassPanel => PanelKind::Glass,
                        _ => PanelKind::Flip,
                    };
                    let tilt = it.axis([1, 0, 0]).filter(|t| t.axis() != n.axis());
                    let tilt = match (kind, tilt) {
                        (PanelKind::Flip, t) => t.unwrap_or(across_of(n)),
                        (_, Some(t)) => t,
                        (_, None) => {
                            self.decode_problems.push(problem("tilt not axis-aligned"));
                            continue;
                        }
                    };
                    let angle = match kind {
                        PanelKind::Flip => 180,
                        _ => it.panel_angle(),
                    };
                    if !matches!(angle, 30 | 45 | 60 | 90 | 180) {
                        self.decode_problems
                            .push(problem(&format!("unknown panel angle {angle}")));
                        continue;
                    }
                    self.panels.push(Panel {
                        item: it.index,
                        kind,
                        cell: it.cell,
                        normal: n,
                        tilt,
                        angle,
                        start_deployed: kind != PanelKind::Flip && it.start_deployed(),
                    });
                }
                ItemKind::Stairs => {
                    let dir = it.axis([1, 0, 0]).filter(|d| d.axis() != 2);
                    let (Some(dir), Dir::PosZ) = (dir, n) else {
                        self.decode_problems.push(problem("stairs not flat on a floor"));
                        continue;
                    };
                    let second = step(it.cell, dir);
                    if self.solid(second) || !self.solid(step(second, Dir::NegZ)) {
                        self.decode_problems
                            .push(problem("stairs' second cell isn't open floor"));
                        continue;
                    }
                    self.stairs.push(Stairs {
                        item: it.index,
                        cell: it.cell,
                        dir,
                        start_deployed: it.start_deployed(),
                    });
                }
                ItemKind::Piston => {
                    let level = |k: &str| it.props.get_i32(k);
                    let (b, t) = (
                        level("ITEM_PROPERTY_PISTON_LIFT_BOTTOM_LEVEL").unwrap_or(0),
                        level("ITEM_PROPERTY_PISTON_LIFT_TOP_LEVEL").unwrap_or(1),
                    );
                    if !(0..=3).contains(&b) || !(1..=4).contains(&t) || b >= t {
                        self.decode_problems.push(problem("bad piston levels"));
                        continue;
                    }
                    let v = n.vector();
                    // Open as far as it reaches (and, on a floor, to stand on its top).
                    let reach = if n == Dir::PosZ { t } else { t - 1 };
                    let open = (0..=reach)
                        .all(|l| !self.solid(std::array::from_fn(|a| it.cell[a] + v[a] * l)));
                    if !open {
                        self.decode_problems.push(problem("piston's column isn't open"));
                        continue;
                    }
                    self.pistons.push(Piston {
                        item: it.index,
                        cell: it.cell,
                        normal: n,
                        bottom: b,
                        top: t,
                        start_up: it.flag("ITEM_PROPERTY_PISTON_LIFT_START_UP").unwrap_or(false),
                    });
                }
                ItemKind::Rail => {
                    let end = extent_links
                        .iter()
                        .filter(|l| l.sender == it.index)
                        .filter_map(|l| self.item(l.receiver))
                        .map(|e| e.cell)
                        .max_by_key(|e| (0..3).map(|a| (e[a] - it.cell[a]).abs()).sum::<i32>())
                        .unwrap_or(it.cell);
                    if (0..3).filter(|&a| end[a] != it.cell[a]).count() > 1 {
                        self.decode_problems.push(problem("rail track isn't straight"));
                        continue;
                    }
                    let rail = Rail {
                        item: it.index,
                        cell: it.cell,
                        end,
                        oscillate: it.flag("ITEM_PROPERTY_RAIL_OSCILLATE").unwrap_or(true),
                        start_active: it.flag("ITEM_PROPERTY_RAIL_START_ACTIVE").unwrap_or(true),
                        speed: it
                            .props
                            .get_f64s::<1>("ITEM_PROPERTY_RAIL_SPEED")
                            .map_or(100.0, |[v]| v)
                            .max(1.0),
                    };
                    if rail.track().iter().any(|&c| self.solid(c)) {
                        self.decode_problems.push(problem("rail track isn't open"));
                        continue;
                    }
                    self.rails.push(rail);
                }
                _ => {}
            }
        }
        for (i, p) in self.plates.iter().enumerate() {
            self.launch.insert(p.cell, i);
        }
        // Panels cover the face behind them: an angled one is white or black as it says, a
        // glass one is never white.
        for panel in self.panels.clone() {
            let (c, d) = (panel.cell, panel.normal.opposite());
            match panel.kind {
                PanelKind::Angled => {
                    let white = self.item(panel.item).is_some_and(|it| it.portalable());
                    self.voxels.set_face_portalable(c, d, white);
                }
                PanelKind::Glass => self.voxels.set_face_portalable(c, d, false),
                PanelKind::Flip => {}
            }
        }
        // Doors cover the face behind them: no portal goes on a door.
        let doors: Vec<(Cell, Dir)> = self
            .items
            .iter()
            .filter(|i| matches!(i.kind, ItemKind::EntryDoor | ItemKind::ExitDoor))
            .filter_map(|i| Some((i.cell, i.normal?.opposite())))
            .collect();
        for (c, d) in doors {
            self.voxels.set_face_portalable(c, d, false);
        }
    }

    /// Whether an arc from the middle of cell `a` up to a voxel over the higher of the two and
    /// down to the middle of cell `b` misses solid voxels (sampled every 1/16 voxel; the ends'
    /// own cells don't count).
    pub fn clear_arc(&self, a: Cell, b: Cell) -> bool {
        let p0 = a.map(|v| v as f64 + 0.5);
        let p1 = b.map(|v| v as f64 + 0.5);
        let rise = p0[2].max(p1[2]) + 1.0 - p0[2];
        let dz = p1[2] - p0[2];
        let len = (p1[0] - p0[0]).hypot(p1[1] - p0[1]) + 2.0 * rise;
        let n = (len * 16.0).ceil().max(1.0) as usize;
        (1..n).all(|k| {
            let t = k as f64 / n as f64;
            let z = p0[2] + dz * t + 4.0 * (rise - dz / 2.0).max(0.0) * t * (1.0 - t);
            let c = [
                (p0[0] + (p1[0] - p0[0]) * t).floor() as i32,
                (p0[1] + (p1[1] - p0[1]) * t).floor() as i32,
                z.floor() as i32,
            ];
            c == a || c == b || !self.solid(c)
        })
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
    pub fn floor_under(&self, c: Cell) -> bool {
        self.solid(step(c, Dir::NegZ)) || self.face(c, Dir::NegZ).barrier.is_some()
    }

    /// Open cell with a floor (or a glass/grating panel) under it that isn't goo or a faith
    /// plate: somewhere the player can stand.
    pub fn standable(&self, c: Cell) -> bool {
        !self.solid(c) && self.floor_under(c) && !self.goo(c) && !self.launch.contains_key(&c)
    }

    /// The faith plate (index into `plates`) in cell `c`, if any.
    pub fn plate_at(&self, c: Cell) -> Option<usize> {
        self.launch.get(&c).copied()
    }

    /// The item index of the floor button in cell `c`, if any (the first, in item order).
    pub fn floor_button_at(&self, c: Cell) -> Option<i32> {
        self.button_at.get(&c).copied()
    }

    /// The index into `turrets` of the turret in cell `c`, if any (the first).
    pub fn turret_at(&self, c: Cell) -> Option<usize> {
        self.turret_at.get(&c).copied()
    }

    /// Whether cube item `cube` has no working dropper (it was placed loose; fizzled, it's gone).
    pub fn loose_cube(&self, cube: i32) -> bool {
        !self.dropper_of.contains_key(&cube)
    }

    /// Face between `c` and `step(c, d)` is a surface (one side solid) and is white.
    pub fn portal_surface(&self, c: Cell, d: Dir) -> bool {
        self.solid(c) != self.solid(step(c, d)) && self.voxels.face_portalable(c, d)
    }

    /// Where something falling from open cell `c` comes to rest, and the hazard fields it
    /// falls through. `None` if it ends in goo or `c` is solid.
    pub fn fall_path(&self, c: Cell) -> Option<(Cell, u64)> {
        self.fall_route(c).map(|(f, h, _)| (f, h))
    }

    /// `fall_path`, and the last faith plate (index into `plates`) that launched it on the way:
    /// a plate in any cell it falls through sends it to its target, to fall from there. `None`
    /// also for a plate whose arc is blocked (`Plate::clear`), and for plates launching in a
    /// loop.
    pub fn fall_route(&self, c: Cell) -> Option<(Cell, u64, Option<usize>)> {
        let mut f = c;
        let mut hazards = 0;
        let mut plate = None;
        for _ in 0..8 {
            if self.solid(f) {
                return None;
            }
            loop {
                if let Some(&i) = self.launch.get(&f) {
                    if !self.plates[i].clear {
                        return None;
                    }
                    plate = Some(i);
                    break;
                }
                if self.goo(f) {
                    return None;
                }
                if self.floor_under(f) {
                    return Some((f, hazards, plate));
                }
                hazards |= self.face(f, Dir::NegZ).hazards;
                f = step(f, Dir::NegZ);
            }
            f = self.plates[plate.unwrap()].target.1;
        }
        None
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

    /// Whether `b` can be reached from `a` through open voxels at all, whatever is in the way
    /// (glass, fields, goo, devices). Nothing but walls stops it, so two cells it says aren't
    /// connected are walled apart for good: no player, cube, beam or portal shot gets across.
    pub fn open_connected(&self, a: Cell, b: Cell) -> bool {
        let mut seen = HashSet::from([a]);
        let mut stack = vec![a];
        while let Some(c) = stack.pop() {
            if c == b {
                return true;
            }
            for d in Dir::ALL {
                let n = step(c, d);
                if !self.solid(n) && seen.insert(n) {
                    stack.push(n);
                }
            }
        }
        false
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

    /// Mounted items whose cell isn't open or that have nothing solid behind them, items the
    /// editor would mark misplaced (`editor`), and barriers that couldn't be decoded.
    pub fn mount_problems(&self) -> Vec<String> {
        self.mount_problems_with(true)
    }

    /// `mount_problems`, optionally without our model of the editor's rules: for levels the
    /// game's own editor already accepted (published Workshop puzzles), where its flags are
    /// only false positives.
    pub fn mount_problems_with(&self, editor_rules: bool) -> Vec<String> {
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
        // What the Puzzle Maker itself would refuse (misplaced items it shows in red).
        if editor_rules {
            out.extend(crate::editor::editor_problems(self));
        }
        out.extend(self.decode_problems.iter().cloned());
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

/// A direction perpendicular to `n`.
fn across_of(n: Dir) -> Dir {
    if n.axis() == 0 {
        Dir::PosY
    } else {
        Dir::PosX
    }
}

/// Cells of the inclusive box `lo..=hi`.
pub fn cells_between(lo: Cell, hi: Cell) -> impl Iterator<Item = Cell> {
    (lo[2]..=hi[2]).flat_map(move |z| {
        (lo[1]..=hi[1]).flat_map(move |y| (lo[0]..=hi[0]).map(move |x| [x, y, z]))
    })
}
