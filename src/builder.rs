//! Build puzzles in code. Item property lists copy what the retail editor writes
//! (taken from real v14 files), so the output should open in the Puzzle Maker.

use crate::geom::{angles_for_up, step, Cell, Dir};
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
        let index = self.items.len() as i32;
        let mut p = props(&[
            ("Index", index.to_string()),
            ("Type", kind.to_string()),
            ("Deletable", "1".into()),
            ("VoxelPos", vec3(cell)),
            ("LocalPos", "0 0 0".into()),
            ("Angles", vec3(angles_for_up(normal))),
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
        let mut floor = cell;
        while !self.voxels.solid(step(floor, Dir::NegZ)).unwrap_or(true) {
            floor = step(floor, Dir::NegZ);
        }
        let shared = |delay: &str| {
            vec![
                ("ITEM_PROPERTY_CUBE_TYPE", "0".to_string()),
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
        self.add(
            "ITEM_BUTTON_FLOOR",
            cell,
            Dir::PosZ,
            &[("ITEM_PROPERTY_BUTTON_TYPE", "0".into())],
            &[],
        )
    }

    pub fn goo(&mut self, cell: Cell) -> i32 {
        self.add("ITEM_GOO", cell, Dir::PosZ, &[], &[])
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
