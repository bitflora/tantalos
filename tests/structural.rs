use tantalos::{
    interactions::{self, Topology},
    plan::{self, Step},
    structural, tokens,
};

#[test]
fn mutations_change_controls_without_adding_rooms_or_items() {
    let parent = interactions::compile(Topology::Handover).puzzle;
    let children = structural::mutations(&parent);
    assert!(!children.is_empty());
    let exits: Vec<_> = parent
        .connections
        .iter()
        .filter(|c| {
            c.get_i32("Receiver")
                .and_then(|id| parent.item(id))
                .is_some_and(|i| i.kind() == "ITEM_EXIT_DOOR")
        })
        .cloned()
        .collect();
    for m in children {
        assert_eq!(m.puzzle.items.len(), parent.items.len());
        assert_eq!(m.puzzle.voxels, parent.voxels);
        assert_ne!(m.puzzle.connections, parent.connections);
        for e in &exits {
            assert!(m.puzzle.connections.contains(e));
        }
        tokens::decode(&tokens::encode(&m.puzzle).unwrap().join(" ")).unwrap();
    }
}

#[test]
fn movement_and_portal_aims_do_not_count_as_resource_changes() {
    let a = plan::decode("fire blue 0 0 1 -x walk 1 2 1 pickup 2 2 1 putdown 3 3 1").unwrap();
    let b = plan::decode("fire orange 0 4 1 -x enter orange walk 8 2 1 pickup 2 2 1 putdown 3 3 1")
        .unwrap();
    assert_eq!(
        structural::edit_distance(&structural::operations(&a), &structural::operations(&b)),
        0
    );
    let mut changed = b.clone();
    changed.push(Step::Press([1, 1, 1]));
    assert_eq!(
        structural::edit_distance(
            &structural::operations(&a),
            &structural::operations(&changed)
        ),
        1
    );
}
