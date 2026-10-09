//! Real Puzzle Maker files (BEE2-items preview maps, the only public .p2c files found).

use glados::geom::step;
use glados::level::{ItemKind, Level};
use glados::p2c::Puzzle;
use glados::regions::RegionGraph;

const SKYBOX: &str = "fixtures/real/1501698868.p2c";
const STYLE: &str = "fixtures/real/1500854111.p2c";

fn load(path: &str) -> (String, Puzzle) {
    let src = std::fs::read_to_string(path).unwrap();
    let puzzle = Puzzle::parse(&src).unwrap();
    (src, puzzle)
}

#[test]
fn round_trip_is_byte_identical() {
    for path in [SKYBOX, STYLE] {
        let (src, puzzle) = load(path);
        assert!(puzzle.to_p2c() == src, "{path} did not round-trip");
    }
}

#[test]
fn mounted_items_sit_against_solid_voxels() {
    // Confirms the coordinate reading: interior is 0..ChamberSize, VoxelPos is the open
    // cell an item occupies, and Angles' up vector points away from its surface.
    for path in [SKYBOX, STYLE] {
        let (_, puzzle) = load(path);
        let level = Level::from_puzzle(&puzzle);
        assert_eq!(level.mount_problems(), Vec::<String>::new(), "{path}");
        let panels: Vec<_> = level
            .items
            .iter()
            .filter(|i| i.kind == ItemKind::Other("ITEM_LIGHT_PANEL".into()))
            .collect();
        for p in panels {
            let n = p.normal.expect("light panel angles are axis-aligned");
            assert!(
                !level.solid(p.cell) && level.solid(step(p.cell, n.opposite())),
                "{path}: {p:?}"
            );
        }
    }
}

#[test]
fn skybox_map_geometry() {
    let (_, puzzle) = load(SKYBOX);
    assert_eq!(puzzle.chamber_size(), [9, 24, 5]);
    assert_eq!(puzzle.title(), "Skybox Preview Map");
    let level = Level::from_puzzle(&puzzle);

    // One goo item floods the whole 3-wide trench (x 3..=5, y 1..=23).
    let goo: Vec<_> = level.cells().filter(|&c| level.goo(c)).collect();
    assert_eq!(goo.len(), 3 * 23);
    assert!(goo
        .iter()
        .all(|c| c[2] == 0 && (3..=5).contains(&c[0]) && c[1] >= 1));

    // Both sides of the trench join along the y=0 strip: a single region.
    let graph = RegionGraph::build(&level);
    assert_eq!(graph.regions.len(), 1);
    assert_eq!(graph.regions[0].cells.len(), 9 + 23 * 6);
    assert!(graph.edges.is_empty());
    assert_eq!(graph.region_of(level.entry().unwrap().cell), Some(0));
}

#[test]
fn style_map_has_entry_and_exit() {
    let (_, puzzle) = load(STYLE);
    let level = Level::from_puzzle(&puzzle);
    let graph = RegionGraph::build(&level);
    assert!(graph.region_of(level.entry().unwrap().cell).is_some());
    assert!(graph.region_of(level.exit().unwrap().cell).is_some());
}
