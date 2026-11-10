//! Edits to finished levels (`tantalos-data mutate`).

use std::collections::{BTreeMap, HashSet};

use tantalos::dataset::{label_with, Label};
use tantalos::fixtures;
use tantalos::level::Level;
use tantalos::mutate::{json_field, mutants, mutate, normalize, Config};
use tantalos::p2c::Puzzle;
use tantalos::rng::Rng;
use tantalos::tokens::{decode, encode};

fn fixture(name: &str) -> Puzzle {
    fixtures::all()
        .into_iter()
        .find(|(n, _)| *n == name)
        .unwrap_or_else(|| panic!("no fixture {name}"))
        .1
}

#[test]
fn mutated_levels_are_canonical_and_mounted() {
    for (name, p) in fixtures::all() {
        let Some((parent, parent_tokens)) = normalize(&p) else {
            continue;
        };
        let mut rng = Rng::new(7);
        let mut changed = 0;
        for _ in 0..10 {
            let Some((child, edits)) = mutate(&parent, &mut rng, 2) else {
                continue;
            };
            assert_eq!(edits.len(), 2, "{name}");
            let tokens = encode(&child).unwrap_or_else(|e| panic!("{name} {edits:?}: {e}"));
            let back = decode(&tokens.join(" ")).unwrap();
            assert_eq!(encode(&back).unwrap(), tokens, "{name} {edits:?}: not canonical");
            let problems = Level::from_puzzle(&child).mount_problems_with(false);
            assert!(problems.is_empty(), "{name} {edits:?}: {problems:?}");
            changed += usize::from(tokens != parent_tokens);
        }
        assert!(changed > 0, "{name}: no edit changed anything");
    }
}

#[test]
fn mutation_depends_only_on_the_seed() {
    let (p, _) = normalize(&fixture("worked_example")).unwrap();
    let run = |seed| {
        let mut rng = Rng::new(seed);
        (0..5)
            .map(|_| mutate(&p, &mut rng, 3).map(|(c, e)| (encode(&c).unwrap(), e)))
            .collect::<Vec<_>>()
    };
    assert_eq!(run(3), run(3));
    assert_ne!(run(3), run(4));
}

#[test]
fn every_kind_of_edit_happens() {
    let mut verbs = HashSet::new();
    for name in ["worked_example", "fizzler_cube", "pedestal_window"] {
        let (p, _) = normalize(&fixture(name)).unwrap();
        let mut rng = Rng::new(11);
        for _ in 0..80 {
            if let Some((_, edits)) = mutate(&p, &mut rng, 1) {
                let e = &edits[0];
                let verb = e.split(' ').next().unwrap();
                // Doors are moved by their own edit.
                let verb = if verb == "move" && e.contains("_door") { "shift" } else { verb };
                verbs.insert(verb.to_string());
                if verb == "add" {
                    verbs.insert(format!("add {}", e.split(' ').nth(1).unwrap()));
                }
            }
        }
    }
    for v in [
        "move", "shift", "add", "remove", "paint", "push", "pull", "add dropper", "add button",
        "add fizzler",
    ] {
        assert!(verbs.contains(v), "never saw {v:?}: {verbs:?}");
    }
}

#[test]
fn removing_an_item_drops_its_wires() {
    // pedestal_window: removing the pedestal must not leave a wire to a missing item.
    let (p, _) = normalize(&fixture("pedestal_window")).unwrap();
    let mut rng = Rng::new(1);
    let mut removed = 0;
    for _ in 0..200 {
        if let Some((child, edits)) = mutate(&p, &mut rng, 1) {
            if edits[0].starts_with("remove") {
                removed += 1;
                let ids: HashSet<i32> = child.items.iter().filter_map(|i| i.index()).collect();
                for c in &child.connections {
                    for end in ["Sender", "Receiver"] {
                        let i = c.get_i32(end).unwrap();
                        assert!(ids.contains(&i), "{edits:?}: wire to missing item {i}");
                    }
                }
            }
        }
    }
    assert!(removed > 0);
}

#[test]
fn kept_mutants_need_portals_fit_and_are_new() {
    let parent = encode(&fixture("worked_example")).unwrap().join(" ");
    let cfg = Config {
        per_level: 3,
        tries: 30,
        max_tokens: 300,
        budget: 20_000,
        ..Config::default()
    };
    let mut stats = BTreeMap::new();
    let found = mutants(&parent, &cfg, 5, &mut stats);
    assert!(!found.is_empty(), "{stats:?}");
    assert!(found.len() <= cfg.per_level);
    assert_eq!(stats["kept"], found.len());
    let mut seen = HashSet::new();
    for m in &found {
        assert!(m.tokens.len() <= cfg.max_tokens);
        assert_ne!(m.tokens.join(" "), parent);
        assert!(seen.insert(m.tokens.clone()), "duplicate mutant");
        assert!(
            matches!(m.label, Label::Solvable { needs_portals: true, .. }),
            "{:?}",
            m.label
        );
        // The label is the one the tokens get on their own.
        let again = label_with(&decode(&m.tokens.join(" ")).unwrap(), cfg.budget, 0, false);
        assert_eq!(again.verdict(), "solvable");

        let line = m.json("12345");
        assert_eq!(json_field(&line, "mutated_from").as_deref(), Some("12345"));
        assert_eq!(json_field(&line, "tokens"), Some(m.tokens.join(" ")));
        assert_eq!(json_field(&line, "verdict").as_deref(), Some("solvable"));
        assert_eq!(json_field(&line, "n_tokens"), Some(m.tokens.len().to_string()));
    }
}

#[test]
fn bad_parents_give_nothing() {
    let mut stats = BTreeMap::new();
    assert!(mutants("size 4 4 nonsense", &Config::default(), 1, &mut stats).is_empty());
    assert_eq!(stats["parent unparsed"], 1);
}

#[test]
fn json_fields_are_read() {
    let line = r#"{"n_tokens": 12, "tokens": "size 4 \"4\" 4\\n", "workshop_id": "100650838", "x":{"y":1}}"#;
    assert_eq!(json_field(line, "n_tokens").as_deref(), Some("12"));
    assert_eq!(json_field(line, "tokens").as_deref(), Some("size 4 \"4\" 4\\n"));
    assert_eq!(json_field(line, "workshop_id").as_deref(), Some("100650838"));
    assert_eq!(json_field(line, "missing"), None);
    let compact = r#"{"tokens":"a b","verdict":"solvable"}"#;
    assert_eq!(json_field(compact, "verdict").as_deref(), Some("solvable"));
    assert_eq!(json_field(r#"{"u":"é"}"#, "u").as_deref(), Some("é"));
}
