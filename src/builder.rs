//! Build puzzles in code. Item property lists copy what the retail editor writes
//! (taken from real v14 files), so the output should open in the Puzzle Maker.

use crate::geom::{
    angles_for_up, angles_mapping, angles_mapping_pitches, step, up_from_angles, Cell, Dir,
};
use crate::level::LaserKind;
use crate::p2c::{Connection, Item, Props, Puzzle, Section, Voxels};

/// Gravity, units per second squared.
const GRAVITY: f64 = 600.0;

/// Yaw in degrees of a horizontal direction.
fn yaw(d: Dir) -> i32 {
    match d {
        Dir::PosX => 0,
        Dir::PosY => 90,
        Dir::NegX => 180,
        _ => 270,
    }
}

pub struct Builder {
    title: String,
    size: [i32; 3],
    voxels: Voxels,
    items: Vec<Item>,
    connections: Vec<Connection>,
}

fn props(pairs: &[(&str, String)]) -> Props {
    Props(
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    )
}

fn vec3(v: [i32; 3]) -> String {
    format!("{} {} {}", v[0], v[1], v[2])
}

fn cells_in(min: Cell, max: Cell) -> impl Iterator<Item = Cell> {
    (min[2]..=max[2]).flat_map(move |z| {
        (min[1]..=max[1]).flat_map(move |y| (min[0]..=max[0]).map(move |x| [x, y, z]))
    })
}

impl Builder {
    /// Starts fully solid and black (non-portalable).
    pub fn new(title: &str, size: [i32; 3]) -> Builder {
        Builder {
            title: title.to_string(),
            size,
            voxels: Voxels::new(size.map(|s| s as usize + 1), true, false),
            items: Vec::new(),
            connections: Vec::new(),
        }
    }

    /// Makes the inclusive box `min..=max` open air.
    pub fn carve(&mut self, min: Cell, max: Cell) -> &mut Self {
        cells_in(min, max).for_each(|c| self.voxels.set_solid(c, false));
        self
    }

    pub fn fill(&mut self, min: Cell, max: Cell) -> &mut Self {
        cells_in(min, max).for_each(|c| self.voxels.set_solid(c, true));
        self
    }

    /// Whether `c` is solid so far; anything off the voxel lattice is.
    pub fn solid(&self, c: Cell) -> bool {
        self.voxels.solid(c).unwrap_or(true)
    }

    /// Sets a property of an item added earlier.
    pub fn set_prop(&mut self, item: i32, key: &str, value: &str) -> &mut Self {
        self.items[item as usize].set(key, value);
        self
    }

    /// Whether the `d` face of cell `c` is white so far.
    pub fn white(&self, c: Cell, d: Dir) -> bool {
        self.voxels.face_portalable(c, d)
    }

    /// Sets portalability of the `d` face of every cell in the box.
    pub fn paint(&mut self, min: Cell, max: Cell, d: Dir, white: bool) -> &mut Self {
        cells_in(min, max).for_each(|c| self.voxels.set_face_portalable(c, d, white));
        self
    }

    /// `extra` goes before the common `Facing`/`ConnectionVisiblity` pair and `tail` after it,
    /// matching the editor's property order.
    fn add(
        &mut self,
        kind: &str,
        cell: Cell,
        normal: Dir,
        extra: &[(&str, String)],
        tail: &[(&str, String)],
    ) -> i32 {
        let angles = angles_for_up(normal);
        self.add_raw(kind, cell, angles, "0 0 0", true, extra, tail)
    }

    #[allow(clippy::too_many_arguments)]
    fn add_raw(
        &mut self,
        kind: &str,
        cell: Cell,
        angles: [i32; 3],
        local_pos: &str,
        deletable: bool,
        extra: &[(&str, String)],
        tail: &[(&str, String)],
    ) -> i32 {
        let index = self.items.len() as i32;
        let mut p = props(&[
            ("Index", index.to_string()),
            ("Type", kind.to_string()),
            ("Deletable", if deletable { "1" } else { "0" }.into()),
            ("VoxelPos", vec3(cell)),
            ("LocalPos", local_pos.into()),
            ("Angles", vec3(angles)),
        ]);
        p.0.extend(props(extra).0);
        p.0.extend(
            props(&[
                ("Facing", "0 0 0".into()),
                ("ConnectionVisiblity", "0".into()),
            ])
            .0,
        );
        p.0.extend(props(tail).0);
        self.items.push(p);
        index
    }

    /// A corner marker of a glass panel or hazard field: local +x along `forward`, +z `up`.
    fn extent(&mut self, kind: &str, anchor: i32, cell: Cell, forward: Dir, up: Dir) {
        let angles = angles_mapping([1, 0, 0], forward, [0, 0, 1], up);
        let e = self.add_raw(kind, cell, angles, "0 0 0", false, &[], &[]);
        self.link(anchor, e, "CONNECTION_BARRIER_ANCHOR_TO_EXTENT");
    }

    /// Glass (or grating) on the `normal` face of a rectangle of cells. The panel is anchored
    /// to the surface behind `anchor` (against `grow`) and reaches `out` more cells along
    /// `grow`; sideways it covers `left` cells one way and `right` the other along
    /// `grow × -normal`.
    #[allow(clippy::too_many_arguments)]
    pub fn glass(
        &mut self,
        anchor: Cell,
        normal: Dir,
        grow: Dir,
        left: i32,
        right: i32,
        out: i32,
        grating: bool,
    ) -> i32 {
        let side = grow
            .cross(normal.opposite())
            .expect("grow must lie in the panel");
        let angles = angles_mapping([1, 0, 0], normal.opposite(), [0, 1, 0], side);
        let extra = [
            ("ITEM_PROPERTY_CONNECTION_COUNT", "0".to_string()),
            ("ITEM_PROPERTY_START_REVERSED", "0".into()),
            (
                "ITEM_PROPERTY_BARRIER_TYPE",
                if grating { "1" } else { "0" }.into(),
            ),
        ];
        let g = self.add_raw("ITEM_BARRIER", anchor, angles, "0 0 0", true, &extra, &[]);
        let at = |d: Dir, k: i32| {
            let v = d.vector();
            [
                anchor[0] + v[0] * k,
                anchor[1] + v[1] * k,
                anchor[2] + v[2] * k,
            ]
        };
        let up = normal.opposite();
        let kind = "ITEM_BARRIER_EXTENT";
        self.extent(kind, g, at(side.opposite(), left), side.opposite(), up);
        self.extent(kind, g, at(grow, out), grow, up);
        self.extent(kind, g, at(side, right), side, up);
        g
    }

    /// A fizzler (or laser field) mounted on the wall behind `anchor` (against `normal`),
    /// covering `anchor` and `len - 1` more cells along `along`. In game the field runs out
    /// along `normal` to the opposite wall. Offset 0.375 voxel toward `normal × along`, as the
    /// editor does by default.
    pub fn fizzler(
        &mut self,
        anchor: Cell,
        normal: Dir,
        along: Dir,
        len: i32,
        laser: bool,
        start_enabled: bool,
    ) -> i32 {
        let angles = angles_mapping([0, 0, 1], normal, [0, 1, 0], along);
        let f = normal
            .cross(along)
            .expect("along must lie in the wall")
            .vector();
        let local = format!(
            "{} {} {}",
            0.375 * f[0] as f64,
            0.375 * f[1] as f64,
            0.375 * f[2] as f64
        );
        let extra = [
            ("ITEM_PROPERTY_CONNECTION_COUNT", "0".to_string()),
            (
                "ITEM_PROPERTY_BARRIER_HAZARD_TYPE",
                if laser { "1" } else { "0" }.into(),
            ),
            (
                "ITEM_PROPERTY_START_ENABLED",
                if start_enabled { "1" } else { "0" }.into(),
            ),
        ];
        let h = self.add_raw(
            "ITEM_BARRIER_HAZARD",
            anchor,
            angles,
            &local,
            true,
            &extra,
            &[],
        );
        let v = along.vector();
        let far = [
            anchor[0] + v[0] * (len - 1),
            anchor[1] + v[1] * (len - 1),
            anchor[2] + v[2] * (len - 1),
        ];
        let (lo, hi) = if along.is_positive() {
            (anchor, far)
        } else {
            (far, anchor)
        };
        let axis = Dir::from_axis(along.axis(), true);
        let up = normal.cross(axis).expect("perpendicular");
        let kind = "ITEM_BARRIER_HAZARD_EXTENT";
        self.extent(kind, h, lo, axis.opposite(), up);
        self.extent(kind, h, hi, axis, up);
        h
    }

    /// A pedestal button on the floor of `cell`; `timer` in seconds (3..=30), `None` = infinite.
    pub fn pedestal(&mut self, cell: Cell, timer: Option<u32>) -> i32 {
        self.add(
            "ITEM_BUTTON_PEDESTAL",
            cell,
            Dir::PosZ,
            &[
                ("ITEM_PROPERTY_TIMER_DELAY", timer.unwrap_or(0).to_string()),
                ("ITEM_PROPERTY_TIMER_SOUND", "0".into()),
            ],
            &[],
        )
    }

    /// Turns pedestal button `item` so that its column stands against side `side` of its cell
    /// (local -x; `pedestal` puts it against -x).
    pub fn turn_pedestal(&mut self, item: i32, side: Dir) -> &mut Self {
        let angles = angles_mapping([-1, 0, 0], side, [0, 0, 1], Dir::PosZ);
        self.set_prop(item, "Angles", &vec3(angles))
    }

    fn link(&mut self, sender: i32, receiver: i32, kind: &str) {
        self.connections.push(props(&[
            ("Sender", sender.to_string()),
            ("Receiver", receiver.to_string()),
            ("Type", kind.to_string()),
        ]));
    }

    /// Entry door in open `cell`, against the wall opposite `facing`. Adds the disabled coop door too.
    pub fn entry(&mut self, cell: Cell, facing: Dir) -> i32 {
        let extra = [("ITEM_PROPERTY_DOOR_IS_COOP", "0".to_string())];
        let tail = |enabled: &str| {
            [
                ("DoorInstanceID", "0.000000".to_string()),
                ("Enabled", enabled.to_string()),
            ]
        };
        let sp = self.add("ITEM_ENTRY_DOOR", cell, facing, &extra, &tail("1"));
        let coop = self.add("ITEM_COOP_ENTRY_DOOR", cell, facing, &extra, &tail("0"));
        self.link(sp, coop, "CONNECTION_BARRIER_ANCHOR_TO_EXTENT");
        sp
    }

    /// Exit door, open until something is wired to it with `connect`.
    pub fn exit(&mut self, cell: Cell, facing: Dir) -> i32 {
        let sp = self.add(
            "ITEM_EXIT_DOOR",
            cell,
            facing,
            &[
                ("ITEM_PROPERTY_CONNECTION_COUNT", "0".into()),
                ("ITEM_PROPERTY_START_OPEN", "1".into()),
                ("ITEM_PROPERTY_DOOR_IS_COOP", "0".into()),
            ],
            &[
                ("DoorInstanceID", "3.000000".into()),
                ("Enabled", "1".into()),
            ],
        );
        let coop = self.add(
            "ITEM_COOP_EXIT_DOOR",
            cell,
            facing,
            &[
                ("ITEM_PROPERTY_CONNECTION_COUNT", "0".into()),
                ("ITEM_PROPERTY_DOOR_IS_COOP", "0".into()),
                ("ITEM_PROPERTY_COOP_EXIT_STARTS_LOCKED", "0".into()),
            ],
            &[
                ("DoorInstanceID", "3.000000".into()),
                ("Enabled", "0".into()),
            ],
        );
        self.link(sp, coop, "CONNECTION_BARRIER_ANCHOR_TO_EXTENT");
        sp
    }

    /// Ceiling cube dropper in open `cell`; its cube sits on the first floor below.
    /// Returns (cube, dropper).
    pub fn cube_dropper(&mut self, cell: Cell) -> (i32, i32) {
        self.cube_dropper_of(cell, 0)
    }

    /// `cube_type`: 0 standard, 1 companion, 2 reflection, 3 sphere, 4 franken.
    pub fn cube_dropper_of(&mut self, cell: Cell, cube_type: i32) -> (i32, i32) {
        let mut floor = cell;
        while !self.voxels.solid(step(floor, Dir::NegZ)).unwrap_or(true) {
            floor = step(floor, Dir::NegZ);
        }
        let shared = |delay: &str| {
            vec![
                ("ITEM_PROPERTY_CUBE_TYPE", cube_type.to_string()),
                ("ITEM_PROPERTY_DROPPER_ENABLED", "1".into()),
                ("ITEM_PROPERTY_AUTO_DROP_CUBE", "1".into()),
                ("ITEM_PROPERTY_AUTO_RESPAWN_CUBE", "1".into()),
                ("ITEM_PROPERTY_TIMER_DELAY", delay.into()),
            ]
        };
        let cube = self.add("ITEM_CUBE", floor, Dir::PosZ, &shared("0"), &[]);
        let mut dropper_props = vec![("ITEM_PROPERTY_CONNECTION_COUNT", "0".to_string())];
        dropper_props.extend(shared("4"));
        dropper_props.insert(3, ("ITEM_PROPERTY_DROPPER_FALL_STRAIGHT_DOWN", "1".into()));
        let dropper = self.add("ITEM_DROPPER_CUBE", cell, Dir::NegZ, &dropper_props, &[]);
        self.link(cube, dropper, "CONNECTION_BOX_DROPPER");
        (cube, dropper)
    }

    pub fn floor_button(&mut self, cell: Cell) -> i32 {
        self.floor_button_of(cell, 0)
    }

    /// `button_type`: 0 weighted, 1 cube, 2 sphere.
    pub fn floor_button_of(&mut self, cell: Cell, button_type: i32) -> i32 {
        self.add(
            "ITEM_BUTTON_FLOOR",
            cell,
            Dir::PosZ,
            &[("ITEM_PROPERTY_BUTTON_TYPE", button_type.to_string())],
            &[],
        )
    }

    pub fn goo(&mut self, cell: Cell) -> i32 {
        self.add("ITEM_GOO", cell, Dir::PosZ, &[], &[])
    }

    /// A turret on the floor of `cell` with `Facing` yaw `yaw` degrees (0 = +x, 90 = +y); in
    /// the game it looks the opposite way (`Turret::look_yaw`). Copies the style map's
    /// turrets: no properties, the yaw in `Facing`.
    pub fn turret(&mut self, cell: Cell, yaw: i32) -> i32 {
        let t = self.add("ITEM_TURRET", cell, Dir::PosZ, &[], &[]);
        self.items[t as usize].set("Facing", format!("0 {yaw} 0"));
        t
    }

    fn paint_props(paint_type: i32, dropper: bool, start_on: bool) -> Vec<(&'static str, String)> {
        // ASSUMPTION: the export type follows the paint type (the style map's one gel, type
        // 0, has 0). The editor may recompute it.
        vec![
            ("ITEM_PROPERTY_PAINT_TYPE", paint_type.to_string()),
            ("ITEM_PROPERTY_DROPPER_ENABLED", (dropper as i32).to_string()),
            ("ITEM_PROPERTY_ALLOW_STREAK_PAINT", "1".into()),
            ("ITEM_PROPERTY_PAINT_FLOW_TYPE", "1".into()),
            ("ITEM_PROPERTY_START_ENABLED", (start_on as i32).to_string()),
            ("ITEM_PROPERTY_PAINT_EXPORT_TYPE", paint_type.to_string()),
        ]
    }

    /// Gel poured onto the floor of `cell` by a dropper in the ceiling above it (the top open
    /// cell of the column), pouring from the start if `start_on`. `paint_type`: 0 repulsion,
    /// 1 propulsion, 2 conversion, 3 water. Returns the dropper: wire buttons to it.
    pub fn gel_dropper(&mut self, cell: Cell, paint_type: i32, start_on: bool) -> i32 {
        let mut top = cell;
        while !self.solid(step(top, Dir::PosZ)) {
            top = step(top, Dir::PosZ);
        }
        let props = Self::paint_props(paint_type, true, start_on);
        let splat = self.add("ITEM_PAINT_SPLAT", cell, Dir::PosZ, &props, &[]);
        self.items[splat as usize].set("ConnectionVisiblity", "2");
        let mut dropper_props = vec![("ITEM_PROPERTY_CONNECTION_COUNT", "0".to_string())];
        dropper_props.extend(props);
        let dropper = self.add("ITEM_DROPPER_PAINT", top, Dir::NegZ, &dropper_props, &[]);
        self.link(splat, dropper, "CONNECTION_PAINT_DROPPER");
        dropper
    }

    /// Whether the gel dropper `dropper` (and its splat) pours from the start.
    pub fn gel_start(&mut self, dropper: i32, on: bool) -> &mut Self {
        let splats: Vec<i32> = self
            .connections
            .iter()
            .filter(|c| c.get("Type") == Some("CONNECTION_PAINT_DROPPER"))
            .filter(|c| c.get_i32("Receiver") == Some(dropper))
            .filter_map(|c| c.get_i32("Sender"))
            .collect();
        for item in splats.into_iter().chain([dropper]) {
            self.items[item as usize].set("ITEM_PROPERTY_START_ENABLED", (on as i32).to_string());
        }
        self
    }

    /// Gel that is there from the start, on the surface behind open `cell` against `normal`.
    /// ASSUMPTION: a splat with its dropper disabled needs no dropper item (unchecked).
    pub fn gel_splat(&mut self, cell: Cell, normal: Dir, paint_type: i32) -> i32 {
        let props = Self::paint_props(paint_type, false, true);
        let splat = self.add("ITEM_PAINT_SPLAT", cell, normal, &props, &[]);
        self.items[splat as usize].set("ConnectionVisiblity", "2");
        splat
    }

    /// A cube placed without a dropper, resting in open `cell`. The editor still writes a
    /// dropper for it, switched off, at the top of the chamber (as in the style map).
    /// `cube_type` as in `cube_dropper_of`.
    pub fn loose_cube(&mut self, cell: Cell, cube_type: i32) -> i32 {
        let shared = |delay: &str| {
            vec![
                ("ITEM_PROPERTY_CUBE_TYPE", cube_type.to_string()),
                ("ITEM_PROPERTY_DROPPER_ENABLED", "0".into()),
                ("ITEM_PROPERTY_AUTO_DROP_CUBE", "1".into()),
                ("ITEM_PROPERTY_AUTO_RESPAWN_CUBE", "1".into()),
                ("ITEM_PROPERTY_TIMER_DELAY", delay.into()),
            ]
        };
        let cube = self.add("ITEM_CUBE", cell, Dir::PosZ, &shared("0"), &[]);
        let mut dropper_props = vec![("ITEM_PROPERTY_CONNECTION_COUNT", "0".to_string())];
        dropper_props.extend(shared("0"));
        dropper_props.insert(3, ("ITEM_PROPERTY_DROPPER_FALL_STRAIGHT_DOWN", "1".into()));
        let top = [cell[0], cell[1], self.size[2] - 1];
        let dropper = self.add("ITEM_DROPPER_CUBE", top, Dir::NegZ, &dropper_props, &[]);
        self.link(cube, dropper, "CONNECTION_BOX_DROPPER");
        cube
    }

    /// A laser emitter, catcher or relay in open `cell` on the surface behind it (against
    /// `normal`). `offset`: the `_OFFSET` item, sitting toward that side of the face; `None`
    /// for the centred one. Emitters fire along `normal`; `start_on` only matters for them.
    pub fn laser(
        &mut self,
        kind: LaserKind,
        cell: Cell,
        normal: Dir,
        offset: Option<Dir>,
        start_on: bool,
    ) -> i32 {
        let name = match kind {
            LaserKind::Emitter => "EMITTER",
            LaserKind::Catcher => "CATCHER",
            LaserKind::Relay => "RELAY",
        };
        let kind_name = format!(
            "ITEM_LASER_{name}_{}",
            if offset.is_some() { "OFFSET" } else { "CENTER" }
        );
        let angles = match offset {
            Some(o) => angles_mapping([0, 0, 1], normal, [-1, 0, 0], o),
            None => angles_for_up(normal),
        };
        let mut extra = vec![("ITEM_PROPERTY_CONNECTION_COUNT", "0".to_string())];
        match kind {
            LaserKind::Emitter => {
                extra.push(("ITEM_PROPERTY_CUBE_TYPE", "0".into()));
                extra.push(("ITEM_PROPERTY_START_ENABLED", (start_on as i32).to_string()));
            }
            LaserKind::Catcher => extra.push(("ITEM_PROPERTY_CUBE_TYPE", "0".into())),
            LaserKind::Relay => {}
        }
        self.add_raw(&kind_name, cell, angles, "0 0 0", true, &extra, &[])
    }

    /// A faith plate in open `cell` on the surface against `normal`, launching to a target in
    /// open `target` on the surface against `target_normal`. Its speed is what a lob over the
    /// higher of the two by two voxels needs (ASSUMPTION: the compiled trigger aims at the
    /// target with this speed). A floor target straight over a floor plate (in the air: how
    /// the Workshop converter writes one) means a plate throwing straight up, to its height. In-game checks of 2026-10-24: `VERTICAL_ALIGNMENT` 1
    /// on the plate makes the editor throw straight up wherever its target is, and the editor
    /// rejects a target in the air but takes one on the plate itself, so the target goes there.
    /// The throw rises `CatapultHeight` × 64 units over a trigger 56 units above the floor,
    /// so twice the voxels up reaches `target` (`level::vertical_reach`). Returns (plate,
    /// target).
    pub fn faith_plate(
        &mut self,
        cell: Cell,
        normal: Dir,
        target: Cell,
        target_normal: Dir,
    ) -> (i32, i32) {
        let dz = (target[2] - cell[2]) as f64 * 128.0;
        let flat = (((target[0] - cell[0]).pow(2) + (target[1] - cell[1]).pow(2)) as f64).sqrt()
            * 128.0;
        let rise = dz.max(0.0) + 256.0;
        let vz = (2.0 * GRAVITY * rise).sqrt();
        let t = vz / GRAVITY + (2.0 * (rise - dz) / GRAVITY).sqrt();
        let speed = (vz * vz + (flat / t).powi(2)).sqrt();
        let up = normal == Dir::PosZ
            && target_normal == Dir::PosZ
            && target[..2] == cell[..2]
            && target[2] > cell[2];
        let (height, vertical, target) = if up {
            (2.0 * (target[2] - cell[2]) as f64, "1", cell)
        } else {
            ((rise / 128.0).round(), "0", target)
        };
        let plate = self.add(
            "ITEM_CATAPULT",
            cell,
            normal,
            &[
                ("ITEM_PROPERTY_CONNECTION_COUNT", "0".into()),
                ("ITEM_PROPERTY_START_ENABLED", "1".into()),
                ("ITEM_PROPERTY_START_REVERSED", "0".into()),
                ("ITEM_PROPERTY_CATAPULT_SPEED", format!("{speed:.6}")),
                ("ITEM_PROPERTY_TARGET_NAME", "target".into()),
                ("ITEM_PROPERTY_VERTICAL_ALIGNMENT", vertical.into()),
            ],
            &[("CatapultHeight", format!("{height:.6}"))],
        );
        let t = self.add(
            "ITEM_CATAPULT_TARGET",
            target,
            target_normal,
            &[("ITEM_PROPERTY_VERTICAL_ALIGNMENT", vertical.into())],
            &[],
        );
        self.link(plate, t, "CONNECTION_CATAPULT_TARGET");
        (plate, t)
    }

    /// A light bridge emitter in open `cell` on the surface against `normal`, projecting
    /// along it, walkable side `side`. Like the style map's, it sits 0.375 voxel toward
    /// `-side`, so a bridge with `side` up is the floor of the cells it crosses.
    pub fn light_bridge(&mut self, cell: Cell, normal: Dir, side: Dir, start_on: bool) -> i32 {
        let angles = angles_mapping([0, 0, 1], normal, [1, 0, 0], side);
        let v = side.vector();
        let local = format!(
            "{} {} {}",
            -0.375 * v[0] as f64,
            -0.375 * v[1] as f64,
            -0.375 * v[2] as f64
        );
        self.add_raw(
            "ITEM_LIGHT_BRIDGE",
            cell,
            angles,
            &local,
            true,
            &[
                ("ITEM_PROPERTY_CONNECTION_COUNT", "0".into()),
                ("ITEM_PROPERTY_START_ENABLED", (start_on as i32).to_string()),
            ],
            &[],
        )
    }

    /// An excursion funnel emitter in open `cell` on the surface against `normal`.
    pub fn funnel(&mut self, cell: Cell, normal: Dir, start_on: bool, reversed: bool) -> i32 {
        self.add(
            "ITEM_TBEAM",
            cell,
            normal,
            &[
                ("ITEM_PROPERTY_CONNECTION_COUNT", "0".into()),
                ("ITEM_PROPERTY_CONNECTION_COUNT_POLARITY", "0".into()),
                ("ITEM_PROPERTY_START_ENABLED", (start_on as i32).to_string()),
                ("ITEM_PROPERTY_START_REVERSED", (reversed as i32).to_string()),
            ],
            &[],
        )
    }

    /// An angled panel (`glass`: the clear one) on the face behind open `cell` (against
    /// `normal`), tilting toward `tilt` by `angle` degrees (30, 45, 60 or 90) when deployed.
    /// `ITEM_PROPERTY_ANGLED_PANEL_TYPE` only picks the editor's model: editoritems lists them
    /// 90°, 60°, 45°, 30°, so 90° = 0 ... 30° = 3 (the default type 2 goes with the default
    /// 45° animation). Past 3 the editor reads a model index from beyond its table and
    /// crashes drawing a retracted panel. The compiled level follows the animation only.
    #[allow(clippy::too_many_arguments)]
    pub fn angled_panel(
        &mut self,
        cell: Cell,
        normal: Dir,
        tilt: Dir,
        angle: i32,
        start_deployed: bool,
        white: bool,
        glass: bool,
    ) -> i32 {
        let angles = angles_mapping([0, 0, 1], normal, [1, 0, 0], tilt);
        let kind = if glass {
            "ITEM_PANEL_CLEAR"
        } else {
            "ITEM_PANEL_ANGLED"
        };
        let ty = match angle {
            30 => 3,
            45 => 2,
            60 => 1,
            _ => 0,
        };
        self.add_raw(
            kind,
            cell,
            angles,
            "0 0 0",
            true,
            &[
                ("ITEM_PROPERTY_CONNECTION_COUNT", "0".into()),
                (
                    "ITEM_PROPERTY_START_DEPLOYED",
                    (start_deployed as i32).to_string(),
                ),
                ("ITEM_PROPERTY_PORTALABLE", ((white && !glass) as i32).to_string()),
                ("ITEM_PROPERTY_ANGLED_PANEL_TYPE", ty.to_string()),
                (
                    "ITEM_PROPERTY_ANGLED_PANEL_ANIMATION",
                    format!("ramp_{angle}_deg_open"),
                ),
            ],
            &[],
        )
    }

    /// A flip panel on the face behind open `cell` (against `normal`).
    pub fn flip_panel(&mut self, cell: Cell, normal: Dir) -> i32 {
        self.add(
            "ITEM_PANEL_FLIP",
            cell,
            normal,
            &[("ITEM_PROPERTY_CONNECTION_COUNT", "0".into())],
            &[],
        )
    }

    /// Stairs on the floors of open `cell` and the next cell along `dir`, rising toward `dir`.
    pub fn stairs(&mut self, cell: Cell, dir: Dir, start_deployed: bool) -> i32 {
        self.add_raw(
            "ITEM_PANEL_STAIRS",
            cell,
            [0, yaw(dir), 0],
            "0 0 0",
            true,
            &[
                ("ITEM_PROPERTY_CONNECTION_COUNT", "0".into()),
                (
                    "ITEM_PROPERTY_START_DEPLOYED",
                    (start_deployed as i32).to_string(),
                ),
            ],
            &[],
        )
    }

    /// A piston platform pushing out of the surface behind `cell` along `normal`, `bottom`
    /// (down) or `top` (up) cells out.
    pub fn piston(
        &mut self,
        cell: Cell,
        normal: Dir,
        bottom: i32,
        top: i32,
        start_up: bool,
    ) -> i32 {
        let p = self.add(
            "ITEM_PISTON_PLATFORM",
            cell,
            normal,
            &[
                ("ITEM_PROPERTY_CONNECTION_COUNT", "0".into()),
                ("ITEM_PROPERTY_CUBE_TYPE", "0".into()),
                ("ITEM_PROPERTY_AUTO_DROP_CUBE", "1".into()),
                ("ITEM_PROPERTY_PISTON_LIFT_BOTTOM_LEVEL", bottom.to_string()),
                ("ITEM_PROPERTY_PISTON_LIFT_TOP_LEVEL", top.to_string()),
                ("ITEM_PROPERTY_PISTON_LIFT_START_UP", (start_up as i32).to_string()),
            ],
            &[],
        );
        let v = normal.vector();
        for (level, handle) in [(top, 0), (bottom, 1)] {
            let c = std::array::from_fn(|a| cell[a] + v[a] * level);
            let e = self.add_raw(
                "ITEM_PISTON_PLATFORM_EXTENT",
                c,
                angles_for_up(normal),
                "0 0 0",
                false,
                &[],
                &[("EndHandle", handle.to_string())],
            );
            self.link(p, e, "CONNECTION_BARRIER_ANCHOR_TO_EXTENT");
        }
        p
    }

    /// A rail platform: the top of open voxel `cell` (a floor for the cell above), travelling
    /// along a straight track to `end`.
    /// Angles and extents copy the style map's rail (track toward -y), turned about z; a
    /// vertical one keeps those and travels up or down (ASSUMPTION: `TRAVEL_DIRECTION` pitch).
    /// It runs along the wall on its left (a vertical one: its up is -x).
    pub fn rail(&mut self, cell: Cell, end: Cell, oscillate: bool, start_active: bool) -> i32 {
        self.rail_on(cell, end, oscillate, start_active, None)
    }

    /// The up `rail` gives a platform travelling along `t`: away from the wall it runs on.
    pub fn rail_up(t: Dir) -> Dir {
        let turn = if t.axis() == 2 { 0 } else { yaw(t) - 270 };
        up_from_angles([90.0, f64::from(180 + turn), 0.0]).expect("axis-aligned")
    }

    /// `rail`, mounted on the surface against `up` (perpendicular to the track) instead:
    /// Workshop rails run on either wall, floors and ceilings. In-game, the editor took 192 of
    /// 312 such rails, and all but 5 of the rest break rules in editor.rs (2026-10-16).
    /// ASSUMPTION: `TRAVEL_DIRECTION` as in `rail` (`Level` reads the track from the extents).
    pub fn rail_on(
        &mut self,
        cell: Cell,
        end: Cell,
        oscillate: bool,
        start_active: bool,
        up: Option<Dir>,
    ) -> i32 {
        let a = (0..3).find(|&a| end[a] != cell[a]).unwrap_or(0);
        let t = Dir::from_axis(a, end[a] > cell[a]);
        let turn = if a == 2 { 0 } else { yaw(t) - 270 };
        // The platform and its four extents, each pointing along the mounting surface (the
        // one at `end` along the track), as (cell, angles). The style map's, turned about z;
        // on another surface, built the way the style map's are: a wall platform has pitch 90
        // and its up away from the wall (as on every Workshop rail), extents have that up too.
        let style = [
            (cell, [90, 180, 0]),
            (cell, [90, 180, 0]),
            (cell, [0, 90, -90]),
            (cell, [-90, 0, 0]),
            (end, [0, -90, 90]),
        ];
        // The style map's rail runs level: turned about z, its parts don't fit a lift, which
        // the editor never stops flagging (repeating its invalid sound, 2026-10-16). A lift is
        // built like any other surface's rail, on the wall `rail_up` puts it on by default.
        let up = if a == 2 {
            Some(up.unwrap_or(Self::rail_up(t)))
        } else {
            up.filter(|&n| n != Self::rail_up(t) && n.axis() != a)
        };
        let parts: Vec<(Cell, [i32; 3])> = match up {
            None => style.map(|(c, g)| (c, [g[0], g[1] + turn, g[2]])).to_vec(),
            Some(n) => {
                let point = |forward: Dir| {
                    angles_mapping_pitches([0, 90, -90, 180], [1, 0, 0], forward, [0, 0, 1], n)
                };
                // For the style map's rail this is down: its extents go down, back, up, along.
                let side = n.cross(t).expect("perpendicular").opposite();
                let platform = if n.axis() == 2 { t } else { Dir::NegZ };
                vec![
                    (cell, point(platform)),
                    (cell, point(side)),
                    (cell, point(t.opposite())),
                    (cell, point(side.opposite())),
                    (end, point(t)),
                ]
            }
        };
        let dist = (end[a] - cell[a]).abs() * 128;
        let direction = match t {
            Dir::PosZ => "-90 0 0".to_string(),
            Dir::NegZ => "90 0 0".to_string(),
            _ => format!("0 {} 0", (yaw(t) + 180) % 360),
        };
        let r = self.add_raw(
            "ITEM_RAIL_PLATFORM",
            cell,
            parts[0].1,
            "0 0 0",
            true,
            &[
                ("ITEM_PROPERTY_CONNECTION_COUNT", "0".into()),
                ("ITEM_PROPERTY_CUBE_TYPE", "0".into()),
                ("ITEM_PROPERTY_AUTO_DROP_CUBE", "0".into()),
                ("ITEM_PROPERTY_RAIL_OSCILLATE", (oscillate as i32).to_string()),
                ("ITEM_PROPERTY_RAIL_STARTING_POSITION", "0".into()),
                ("ITEM_PROPERTY_RAIL_TRAVEL_DISTANCE", dist.to_string()),
                ("ITEM_PROPERTY_RAIL_SPEED", "100".into()),
                ("ITEM_PROPERTY_RAIL_TRAVEL_DIRECTION", direction),
                (
                    "ITEM_PROPERTY_RAIL_START_ACTIVE",
                    (start_active as i32).to_string(),
                ),
            ],
            &[],
        );
        for &(c, angles) in &parts[1..] {
            let e = self.add_raw("ITEM_RAIL_PLATFORM_EXTENT", c, angles, "0 0 0", false, &[], &[]);
            self.link(r, e, "CONNECTION_BARRIER_ANCHOR_TO_EXTENT");
        }
        r
    }

    /// Wires `sender` to reverse funnel `funnel`.
    pub fn connect_polarity(&mut self, sender: i32, funnel: i32) -> &mut Self {
        self.link(sender, funnel, "CONNECTION_TBEAM_POLARITY");
        let r = &mut self.items[funnel as usize];
        let count = r
            .get_i32("ITEM_PROPERTY_CONNECTION_COUNT_POLARITY")
            .unwrap_or(0)
            + 1;
        r.set("ITEM_PROPERTY_CONNECTION_COUNT_POLARITY", count.to_string());
        self
    }

    /// Sets whether cube dropper `dropper` drops its cube by itself (or waits for its inputs).
    pub fn auto_drop(&mut self, dropper: i32, on: bool) -> &mut Self {
        let cubes: Vec<i32> = self
            .connections
            .iter()
            .filter(|c| c.get("Type") == Some("CONNECTION_BOX_DROPPER"))
            .filter(|c| c.get_i32("Receiver") == Some(dropper))
            .filter_map(|c| c.get_i32("Sender"))
            .collect();
        for item in cubes.into_iter().chain([dropper]) {
            self.items[item as usize].set("ITEM_PROPERTY_AUTO_DROP_CUBE", (on as i32).to_string());
        }
        self
    }

    /// Wires `sender` to activate `receiver`. A wired exit door starts closed.
    pub fn connect(&mut self, sender: i32, receiver: i32) -> &mut Self {
        self.link(sender, receiver, "CONNECTION_STANDARD");
        let r = &mut self.items[receiver as usize];
        let count = r.get_i32("ITEM_PROPERTY_CONNECTION_COUNT").unwrap_or(0) + 1;
        r.set("ITEM_PROPERTY_CONNECTION_COUNT", count.to_string());
        if r.kind() == "ITEM_EXIT_DOOR" {
            r.set("ITEM_PROPERTY_START_OPEN", "0");
        }
        self
    }

    pub fn build(&self) -> Puzzle {
        let prop = |k: &str, v: &str| Section::Prop(k.to_string(), v.to_string());
        // Fixed timestamps keep generated fixtures byte-stable.
        let sections = vec![
            prop("AppID", "644"),
            prop("Version", "14"),
            prop("FileID", "0x0000000000000000"),
            prop("Timestamp_Created", "0x0000000068E60000"),
            prop("Timestamp_Modified", "0x0000000068E60000"),
            prop("CompileTime", "0.000000"),
            prop("Coop", "0"),
            prop("Title", &self.title),
            prop("Description", "Generated by tantalos"),
            prop("PreviewDirty", "1"),
            prop("ChamberSize", &vec3(self.size)),
            Section::Voxels,
            Section::Items,
            Section::Connections,
        ];
        Puzzle {
            sections,
            voxels: self.voxels.clone(),
            items: self.items.clone(),
            connections: self.connections.clone(),
        }
    }
}
