//! Build puzzles in code. Item property lists copy what the retail editor writes
//! (taken from real v14 files), so the output should open in the Puzzle Maker.

use crate::geom::{angles_for_up, angles_mapping, step, Cell, Dir};
use crate::p2c::{Connection, Item, Props, Puzzle, Section, Voxels};

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

    /// A turret on the floor of `cell`, facing `yaw` degrees (0 = +x, 90 = +y). Copies the
    /// style map's turrets: no properties, the yaw in `Facing`.
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
            prop("Description", "Generated by glados"),
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
