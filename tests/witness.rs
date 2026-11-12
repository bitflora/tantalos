use tantalos::{
    editor,
    level::Level,
    plan,
    solver::{Mode, Model},
    witness::*,
};

#[test]
fn shared_cube_witness_replays_and_grounds() {
    let (p, steps) = ReuseChamber::default().build().unwrap();
    let level = Level::from_puzzle(&p);
    assert!(
        editor::editor_flags(&level).is_empty(),
        "{:?}",
        editor::editor_flags(&level)
    );
    let model = Model::new(&level, Mode::Conservative).unwrap();
    let checked = plan::check(&model, &steps);
    assert!(checked.solved, "{checked:?}");
    tantalos::route::route_plan(&level, &model, &checked.plan).unwrap();
    let result = repair(
        &model,
        &steps,
        RepairLimits {
            edits: 0,
            states: 100,
        },
    )
    .unwrap();
    assert_eq!(result.edits, 0);
    assert_eq!(audit_access(&model, 0, 5_000), Audit::Required);
    assert_eq!(
        audit_without(&model, 5_000, |a, _| !matches!(
            a,
            tantalos::solver::Action::Fire { .. }
        )),
        Audit::Required
    );
}

#[test]
fn removing_the_sightline_barrier_is_detected_as_a_shortcut() {
    let (mut p, _) = ReuseChamber::default().build().unwrap();
    // Add an exposed white surface on the upper back wall. The original witness
    // still works, but the access cube is no longer necessary.
    p.voxels
        .set_face_portalable([8, 2, 2], tantalos::geom::Dir::PosX, true);
    let model = Model::new(&Level::from_puzzle(&p), Mode::Conservative).unwrap();
    assert_eq!(audit_access(&model, 0, 5_000), Audit::Bypass);
}

#[test]
fn an_inserted_step_repairs_the_middle_and_preserves_the_suffix() {
    let (p, mut steps) = ReuseChamber::default().build().unwrap();
    steps.remove(3); // Missing the climb, before the hidden portal shot.
    let level = Level::from_puzzle(&p);
    let model = Model::new(&level, Mode::Conservative).unwrap();
    assert!(!plan::check(&model, &steps).solved);
    let result = repair(
        &model,
        &steps,
        RepairLimits {
            edits: 1,
            states: 1000,
        },
    )
    .unwrap();
    assert_eq!(result.edits, 1);
    assert_eq!(result.actions.len(), 9);
    tantalos::route::route_plan(&level, &model, &result.actions).unwrap();
}

#[test]
fn chamber_dimensions_and_symmetries_keep_the_dependency() {
    use tantalos::{
        augment::{self, Sym},
        tokens,
    };
    for lower_width in [5, 7] {
        for ledge_depth in [3, 5] {
            for lane in [2, 3] {
                let (p, steps) = ReuseChamber {
                    lower_width,
                    ledge_depth,
                    depth: 8,
                    lane,
                }
                .build()
                .unwrap();
                let text = steps
                    .iter()
                    .map(plan::step_text)
                    .collect::<Vec<_>>()
                    .join(" ");
                for sym in Sym::ALL {
                    let transformed = augment::apply(&tokens::encode(&p).unwrap(), sym).unwrap();
                    let p2 = tokens::decode(&transformed.join(" ")).unwrap();
                    let level = Level::from_puzzle(&p2);
                    assert!(editor::editor_flags(&level).is_empty());
                    let model = Model::new(&level, Mode::Conservative).unwrap();
                    let text = augment::apply_plan(&text, sym, p.chamber_size()).unwrap();
                    let checked = plan::check(&model, &plan::decode(&text).unwrap());
                    assert!(
                        checked.solved,
                        "{lower_width} {ledge_depth} {lane} {sym:?}: {checked:?}"
                    );
                    tantalos::route::route_plan(&level, &model, &checked.plan).unwrap();
                    let access = model.buttons.iter().position(|b| b.cell[2] == 1).unwrap();
                    assert_eq!(audit_access(&model, access, 5_000), Audit::Required);
                }
            }
        }
    }
}

#[test]
fn repair_replaces_a_moved_button_step_and_keeps_the_plan() {
    let (mut p, steps) = ReuseChamber::default().build().unwrap();
    let item = p
        .items
        .iter_mut()
        .find(|i| i.kind() == "ITEM_BUTTON_FLOOR" && i.voxel_pos() == Some([7, 5, 2]))
        .unwrap();
    item.set("VoxelPos", "6 5 2");
    let level = Level::from_puzzle(&p);
    let model = Model::new(&level, Mode::Conservative).unwrap();
    assert!(!plan::check(&model, &steps).solved);
    let result = repair(
        &model,
        &steps,
        RepairLimits {
            edits: 2,
            states: 5_000,
        },
    )
    .unwrap();
    let encoded = plan::encode_actions(&model, &result.actions);
    assert!(plan::check(&model, &plan::decode(&encoded).unwrap()).solved);
    tantalos::route::route_plan(&level, &model, &result.actions).unwrap();
    assert!(result.edits > 0 && result.edits <= 2);
}

#[test]
fn failures_are_not_unsolvability_claims() {
    let (p, steps) = ReuseChamber::default().build().unwrap();
    let model = Model::new(&Level::from_puzzle(&p), Mode::Conservative).unwrap();
    assert_eq!(
        repair(
            &model,
            &steps,
            RepairLimits {
                edits: 0,
                states: 1
            }
        )
        .unwrap_err(),
        RepairFailure::Budget
    );
    assert_eq!(
        repair(
            &model,
            &[],
            RepairLimits {
                edits: 0,
                states: 100
            }
        )
        .unwrap_err(),
        RepairFailure::NoRepair
    );
    assert_eq!(audit_access(&model, 0, 1), Audit::Unknown);
}
