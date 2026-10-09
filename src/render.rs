//! Top-down ASCII dump of a level, one block per Z layer, +Y up the screen.

use std::fmt::Write;

use crate::level::{ItemKind, Level};
use crate::regions::{Move, RegionGraph};

pub const LEGEND: &str = "# solid  ~ goo  . open air  A-Z region floor  \
E entry  X exit  c cube  D dropper  b button  (+Y is up)";

pub fn render(level: &Level, graph: &RegionGraph) -> String {
    let [sx, sy, sz] = level.size;
    let mut out = String::new();
    for z in 0..sz {
        if (0..sy).all(|y| (0..sx).all(|x| level.solid([x, y, z]))) {
            continue;
        }
        writeln!(out, "z={z}").unwrap();
        for y in (0..sy).rev() {
            let row: String = (0..sx).map(|x| glyph(level, graph, [x, y, z])).collect();
            writeln!(out, "  {row}").unwrap();
        }
    }
    out
}

fn glyph(level: &Level, graph: &RegionGraph, c: [i32; 3]) -> char {
    let item = level.items.iter().find_map(|i| {
        if i.cell != c {
            return None;
        }
        match i.kind {
            ItemKind::EntryDoor => Some('E'),
            ItemKind::ExitDoor => Some('X'),
            ItemKind::Cube => Some('c'),
            ItemKind::CubeDropper => Some('D'),
            ItemKind::FloorButton => Some('b'),
            _ => None,
        }
    });
    if let Some(g) = item {
        g
    } else if level.solid(c) {
        '#'
    } else if level.goo(c) {
        '~'
    } else if let Some(r) = graph.region_of(c) {
        RegionGraph::label(r)
    } else {
        '.'
    }
}

/// One line per region with its outgoing moves.
pub fn describe_regions(graph: &RegionGraph) -> String {
    let mut out = String::new();
    for r in &graph.regions {
        let moves: Vec<String> = graph
            .edges_from(r.id)
            .map(|e| {
                let to = RegionGraph::label(e.to);
                match e.kind {
                    Move::Drop { height } => format!("drop {height} -> {to}"),
                    Move::Step => format!("step (needs cube) -> {to}"),
                }
            })
            .collect();
        let moves = if moves.is_empty() {
            "no exits on foot".to_string()
        } else {
            moves.join(", ")
        };
        writeln!(
            out,
            "  {}: z={}, {} cells; {}",
            RegionGraph::label(r.id),
            r.z,
            r.cells.len(),
            moves
        )
        .unwrap();
    }
    out
}
