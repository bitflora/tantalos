//! `.p2c` Puzzle Maker files, format version 14 (what the retail game writes).
//!
//! Layout, as found in real files:
//! - `ChamberSize "X Y Z"`: the playable interior is cells `0..X`, `0..Y`, `0..Z`.
//! - `Voxels` holds four layers (`Solid`, `Portal0..2`), each `"Zz" { "Yy" "f<bits>" }`,
//!   one char per x. Layers cover a lattice of `size + 1` per axis: the extra plane is
//!   the boundary, which exists so it can carry face bits for the interior's outer faces.
//! - `PortalN` bit at voxel `v` = the face between `v - e_N` and `v` is portalable (white).
//! - `Items` / `Connections`: flat property lists, kept as ordered strings.
//!
//! Every section and property keeps its position, so `parse(s)?.to_p2c() == s` for game files.

use std::fmt;

use crate::geom::{Cell, Dir};
use crate::kv::{self, Pair, Value};

#[derive(Debug, Clone, PartialEq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<kv::ParseError> for Error {
    fn from(e: kv::ParseError) -> Self {
        Error(format!("KeyValues syntax: {e}"))
    }
}

fn err<T>(msg: impl Into<String>) -> Result<T, Error> {
    Err(Error(msg.into()))
}

/// Ordered string properties of an item or connection.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Props(pub Vec<(String, String)>);

pub type Item = Props;
pub type Connection = Props;

impl Props {
    pub fn get(&self, key: &str) -> Option<&str> {
        self.0
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// Replaces the value in place, or appends the key if it isn't present.
    pub fn set(&mut self, key: &str, value: impl Into<String>) {
        let value = value.into();
        match self.0.iter_mut().find(|(k, _)| k == key) {
            Some((_, v)) => *v = value,
            None => self.0.push((key.to_string(), value)),
        }
    }

    pub fn get_i32(&self, key: &str) -> Option<i32> {
        self.get(key)?.trim().parse().ok()
    }

    pub fn get_f64s<const N: usize>(&self, key: &str) -> Option<[f64; N]> {
        let nums: Vec<f64> = self
            .get(key)?
            .split_whitespace()
            .map(str::parse)
            .collect::<Result<_, _>>()
            .ok()?;
        nums.try_into().ok()
    }

    pub fn index(&self) -> Option<i32> {
        self.get_i32("Index")
    }

    pub fn kind(&self) -> &str {
        self.get("Type").unwrap_or("")
    }

    pub fn voxel_pos(&self) -> Option<Cell> {
        let [x, y, z] = self.get_f64s::<3>("VoxelPos")?;
        Some([x as i32, y as i32, z as i32])
    }

    pub fn angles(&self) -> Option<[f64; 3]> {
        self.get_f64s::<3>("Angles")
    }

    fn from_block(children: &[Pair], what: &str) -> Result<Props, Error> {
        children
            .iter()
            .map(|p| match &p.value {
                Value::Str(v) => Ok((p.key.clone(), v.clone())),
                Value::Block(_) => err(format!("unexpected block \"{}\" in {what}", p.key)),
            })
            .collect::<Result<_, _>>()
            .map(Props)
    }

    fn to_block(&self) -> Vec<Pair> {
        self.0
            .iter()
            .map(|(k, v)| Pair::str(k.clone(), v.clone()))
            .collect()
    }
}

/// The four voxel layers over the `size + 1` lattice.
#[derive(Clone, Debug, PartialEq)]
pub struct Voxels {
    pub dims: [usize; 3],
    solid: Vec<bool>,
    portal: [Vec<bool>; 3],
}

const LAYERS: [&str; 4] = ["Solid", "Portal0", "Portal1", "Portal2"];

impl Voxels {
    pub fn new(dims: [usize; 3], solid: bool, portalable: bool) -> Voxels {
        let n = dims.iter().product();
        Voxels {
            dims,
            solid: vec![solid; n],
            portal: [
                vec![portalable; n],
                vec![portalable; n],
                vec![portalable; n],
            ],
        }
    }

    fn idx(&self, c: Cell) -> Option<usize> {
        let in_bounds = (0..3).all(|a| c[a] >= 0 && (c[a] as usize) < self.dims[a]);
        in_bounds.then(|| {
            let [x, y, z] = c.map(|v| v as usize);
            (z * self.dims[1] + y) * self.dims[0] + x
        })
    }

    pub fn solid(&self, c: Cell) -> Option<bool> {
        self.idx(c).map(|i| self.solid[i])
    }

    pub fn set_solid(&mut self, c: Cell, v: bool) {
        let i = self.idx(c).expect("voxel out of range");
        self.solid[i] = v;
    }

    /// Raw `Portal{axis}` bit at lattice voxel `c` (the face on its negative side).
    pub fn portal_bit(&self, axis: usize, c: Cell) -> Option<bool> {
        self.idx(c).map(|i| self.portal[axis][i])
    }

    /// Whether the face between `c` and its neighbour in direction `d` is portalable.
    pub fn face_portalable(&self, c: Cell, d: Dir) -> bool {
        self.portal_bit(d.axis(), face_owner(c, d)).unwrap_or(false)
    }

    pub fn set_face_portalable(&mut self, c: Cell, d: Dir, v: bool) {
        let i = self.idx(face_owner(c, d)).expect("face out of range");
        self.portal[d.axis()][i] = v;
    }

    fn parse(children: &[Pair], dims: [usize; 3]) -> Result<Voxels, Error> {
        let mut vox = Voxels::new(dims, false, false);
        let mut seen = vec![false; LAYERS.len() * dims[2] * dims[1]];
        for layer in children {
            let Some(li) = LAYERS.iter().position(|&n| n == layer.key) else {
                return err(format!("unknown voxel layer \"{}\"", layer.key));
            };
            for zb in as_block(layer, "voxel layer")? {
                let z = coord(&zb.key, 'Z', dims[2])?;
                for row in as_block(zb, "voxel Z block")? {
                    let y = coord(&row.key, 'Y', dims[1])?;
                    let Value::Str(bits) = &row.value else {
                        return err(format!(
                            "{}/{}/{} is not a string",
                            layer.key, zb.key, row.key
                        ));
                    };
                    let Some(bits) = bits.strip_prefix('f') else {
                        return err(format!(
                            "{}/{}/{}: row lacks 'f' prefix",
                            layer.key, zb.key, row.key
                        ));
                    };
                    if bits.len() != dims[0] {
                        return err(format!(
                            "{}/{}/{}: {} voxels, expected {}",
                            layer.key,
                            zb.key,
                            row.key,
                            bits.len(),
                            dims[0]
                        ));
                    }
                    for (x, ch) in bits.chars().enumerate() {
                        let v = match ch {
                            '0' => false,
                            '1' => true,
                            _ => {
                                return err(format!(
                                    "{}/{}/{}: bad char {ch:?}",
                                    layer.key, zb.key, row.key
                                ))
                            }
                        };
                        let i = vox.idx([x as i32, y as i32, z as i32]).unwrap();
                        match li {
                            0 => vox.solid[i] = v,
                            a => vox.portal[a - 1][i] = v,
                        }
                    }
                    seen[(li * dims[2] + z) * dims[1] + y] = true;
                }
            }
        }
        if let Some(missing) = seen.iter().position(|s| !s) {
            let (li, rest) = (missing / (dims[2] * dims[1]), missing % (dims[2] * dims[1]));
            return err(format!(
                "voxel row {}/Z{}/Y{} missing",
                LAYERS[li],
                rest / dims[1],
                rest % dims[1]
            ));
        }
        Ok(vox)
    }

    fn to_block(&self) -> Vec<Pair> {
        let [nx, ny, nz] = self.dims;
        LAYERS
            .iter()
            .enumerate()
            .map(|(li, name)| {
                let layer = if li == 0 {
                    &self.solid
                } else {
                    &self.portal[li - 1]
                };
                let zs = (0..nz)
                    .map(|z| {
                        // The game writes rows from the highest Y down.
                        let rows = (0..ny)
                            .rev()
                            .map(|y| {
                                let mut bits = String::with_capacity(nx + 1);
                                bits.push('f');
                                for x in 0..nx {
                                    bits.push(if layer[(z * ny + y) * nx + x] {
                                        '1'
                                    } else {
                                        '0'
                                    });
                                }
                                Pair::str(format!("Y{y}"), bits)
                            })
                            .collect();
                        Pair::block(format!("Z{z}"), rows)
                    })
                    .collect();
                Pair::block(*name, zs)
            })
            .collect()
    }
}

/// The lattice voxel that stores the portal bit for face (`c`, `d`).
fn face_owner(c: Cell, d: Dir) -> Cell {
    if d.is_positive() {
        crate::geom::step(c, d)
    } else {
        c
    }
}

fn as_block<'a>(p: &'a Pair, what: &str) -> Result<&'a [Pair], Error> {
    match &p.value {
        Value::Block(children) => Ok(children),
        Value::Str(_) => err(format!("{what} \"{}\" should be a block", p.key)),
    }
}

fn coord(key: &str, prefix: char, limit: usize) -> Result<usize, Error> {
    match key
        .strip_prefix(prefix)
        .and_then(|n| n.parse::<usize>().ok())
    {
        Some(n) if n < limit => Ok(n),
        _ => err(format!(
            "bad voxel key \"{key}\" (expected {prefix}0..{prefix}{})",
            limit - 1
        )),
    }
}

/// Top-level entries in file order; the three big sections are stored typed on `Puzzle`.
#[derive(Clone, Debug, PartialEq)]
pub enum Section {
    Prop(String, String),
    Voxels,
    Items,
    Connections,
    Other(Pair),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Puzzle {
    pub sections: Vec<Section>,
    pub voxels: Voxels,
    pub items: Vec<Item>,
    pub connections: Vec<Connection>,
}

impl Puzzle {
    pub fn parse(src: &str) -> Result<Puzzle, Error> {
        let top = kv::parse(src)?;
        let [root] = top.as_slice() else {
            return err(format!("expected one top-level block, found {}", top.len()));
        };
        if root.key != "portal2_puzzle" {
            return err(format!(
                "top-level block is \"{}\", not \"portal2_puzzle\"",
                root.key
            ));
        }
        let children = as_block(root, "top level")?;
        let prop = |k: &str| {
            children.iter().find_map(|p| match &p.value {
                Value::Str(v) if p.key == k => Some(v.as_str()),
                _ => None,
            })
        };
        match prop("Version") {
            Some("14") => {}
            v => {
                return err(format!(
                    "unsupported .p2c version {v:?} (only 14 is supported)"
                ))
            }
        }
        let size = parse_size(prop("ChamberSize").ok_or(Error("missing ChamberSize".into()))?)?;
        let dims = size.map(|s| s as usize + 1);

        let mut sections = Vec::new();
        let mut voxels = None;
        let mut items = Vec::new();
        let mut connections = Vec::new();
        for p in children {
            match (&p.value, p.key.as_str()) {
                (Value::Str(v), k) => sections.push(Section::Prop(k.to_string(), v.clone())),
                (Value::Block(b), "Voxels") => {
                    voxels = Some(Voxels::parse(b, dims)?);
                    sections.push(Section::Voxels);
                }
                (Value::Block(b), "Items") => {
                    items = list(b, "Item")?;
                    sections.push(Section::Items);
                }
                (Value::Block(b), "Connections") => {
                    connections = list(b, "Connection")?;
                    sections.push(Section::Connections);
                }
                _ => sections.push(Section::Other(p.clone())),
            }
        }
        let voxels = voxels.ok_or(Error("missing Voxels".into()))?;
        Ok(Puzzle {
            sections,
            voxels,
            items,
            connections,
        })
    }

    pub fn to_p2c(&self) -> String {
        let children = self
            .sections
            .iter()
            .map(|s| match s {
                Section::Prop(k, v) => Pair::str(k.clone(), v.clone()),
                Section::Voxels => Pair::block("Voxels", self.voxels.to_block()),
                Section::Items => Pair::block(
                    "Items",
                    self.items
                        .iter()
                        .map(|i| Pair::block("Item", i.to_block()))
                        .collect(),
                ),
                Section::Connections => Pair::block(
                    "Connections",
                    self.connections
                        .iter()
                        .map(|c| Pair::block("Connection", c.to_block()))
                        .collect(),
                ),
                Section::Other(p) => p.clone(),
            })
            .collect();
        kv::write(&[Pair::block("portal2_puzzle", children)])
    }

    pub fn prop(&self, key: &str) -> Option<&str> {
        self.sections.iter().find_map(|s| match s {
            Section::Prop(k, v) if k == key => Some(v.as_str()),
            _ => None,
        })
    }

    pub fn title(&self) -> &str {
        self.prop("Title").unwrap_or("")
    }

    /// Interior size in voxels. Always consistent with `voxels.dims` (validated on parse).
    pub fn chamber_size(&self) -> [i32; 3] {
        self.voxels.dims.map(|d| d as i32 - 1)
    }

    pub fn item(&self, index: i32) -> Option<&Item> {
        self.items.iter().find(|i| i.index() == Some(index))
    }
}

fn parse_size(s: &str) -> Result<[i32; 3], Error> {
    let nums: Vec<i32> = s
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<_, _>>()
        .map_err(|_| Error(format!("bad ChamberSize {s:?}")))?;
    match nums.as_slice() {
        &[x, y, z] if x > 0 && y > 0 && z > 0 => Ok([x, y, z]),
        _ => err(format!("bad ChamberSize {s:?}")),
    }
}

fn list(children: &[Pair], key: &str) -> Result<Vec<Props>, Error> {
    children
        .iter()
        .map(|p| match &p.value {
            Value::Block(b) if p.key == key => Props::from_block(b, key),
            _ => err(format!("expected \"{key}\" block, found \"{}\"", p.key)),
        })
        .collect()
}
