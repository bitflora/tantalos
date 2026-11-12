use tantalos::{
    editor,
    interactions::{self, Topology},
    level::Level,
    plan, route,
    solver::{Mode, Model},
    tokens,
};

#[test]
fn interaction_graphs_have_grounded_witnesses() {
    for topology in Topology::ALL {
        let mut design = interactions::compile(topology);
        design.puzzle = tokens::decode(&tokens::encode(&design.puzzle).unwrap().join(" ")).unwrap();
        let level = Level::from_puzzle(&design.puzzle);
        assert!(
            editor::editor_flags(&level).is_empty(),
            "{topology:?}: {:?}",
            editor::editor_flags(&level)
        );
        let model = Model::new(&level, Mode::Conservative).unwrap();
        let actions = interactions::synthesize(&design, &level, &model, 20_000).unwrap();
        let grounded = route::route_plan(&level, &model, &actions).unwrap();
        assert!(plan::check(&model, &plan::decode(&plan::encode(&grounded)).unwrap()).solved);
        let audits = interactions::audit(&design, &level, &model, 20_000);
        assert!(
            audits
                .iter()
                .all(|(_, a)| *a == tantalos::witness::Audit::Required),
            "{topology:?}: {audits:?}"
        );
    }
}

#[test]
fn handover_rejects_substitution_shortcut() {
    let mut design = interactions::compile(Topology::Handover);
    // A generic button allows the reflector to power the alternate bridge
    // directly: it collapses the intended laser-to-bridge dependency.
    let button = design
        .puzzle
        .items
        .iter_mut()
        .find(|i| i.kind() == "ITEM_BUTTON_FLOOR")
        .unwrap();
    button.set("ITEM_PROPERTY_BUTTON_TYPE", "0");
    let level = Level::from_puzzle(&design.puzzle);
    let model = Model::new(&level, Mode::Conservative).unwrap();
    let audits = interactions::audit(&design, &level, &model, 20_000);
    assert_eq!(audits[0].1, tantalos::witness::Audit::Bypass);
}

#[test]
fn uncertain_audits_and_incomplete_graphs_are_not_successes() {
    let mut design = interactions::compile(Topology::Exchange);
    let level = Level::from_puzzle(&design.puzzle);
    let model = Model::new(&level, Mode::Conservative).unwrap();
    assert!(interactions::audit(&design, &level, &model, 0)
        .iter()
        .all(|(_, a)| *a == tantalos::witness::Audit::Unknown));
    assert!(interactions::synthesize(&design, &level, &model, 0).is_err());
    design.graph.truncate(1);
    assert!(interactions::synthesize(&design, &level, &model, 20_000)
        .unwrap_err()
        .contains("ends before"));
    design.graph[0].depends_on.push(0);
    assert!(interactions::synthesize(&design, &level, &model, 20_000)
        .unwrap_err()
        .contains("topologically"));
}

#[test]
fn rotated_and_mirrored_graphs_preserve_the_interactions() {
    use tantalos::{
        augment::{self, Sym},
        interactions::Condition,
    };
    for topology in Topology::ALL {
        for sym in Sym::ALL.into_iter().skip(1) {
            let mut design = interactions::compile(topology);
            let size = design.puzzle.chamber_size();
            let encoded = augment::apply(&tokens::encode(&design.puzzle).unwrap(), sym).unwrap();
            design.puzzle = tokens::decode(&encoded.join(" ")).unwrap();
            for milestone in &mut design.graph {
                milestone.condition = match milestone.condition {
                    Condition::OnButton(k, c) => Condition::OnButton(k, sym.cell(c, size)),
                    Condition::Laser(c) => Condition::Laser(sym.cell(c, size)),
                    Condition::Collected(k, c) => Condition::Collected(k, sym.cell(c, size)),
                    ref other => other.clone(),
                };
            }
            let level = Level::from_puzzle(&design.puzzle);
            assert!(editor::editor_flags(&level).is_empty());
            let model = Model::new(&level, Mode::Conservative).unwrap();
            let actions = interactions::synthesize(&design, &level, &model, 20_000).unwrap();
            let grounded = route::route_plan(&level, &model, &actions).unwrap();
            assert!(plan::check(&model, &plan::decode(&plan::encode(&grounded)).unwrap()).solved);
            let audits = interactions::audit(&design, &level, &model, 20_000);
            assert!(
                audits
                    .iter()
                    .all(|(_, a)| *a == tantalos::witness::Audit::Required),
                "{topology:?} {sym:?}: {audits:?}"
            );
        }
    }
}
