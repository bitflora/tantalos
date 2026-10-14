//! Token format round trips, decoder robustness, and the labelled dataset.

use glados::dataset::{self, generate, label, Config, Label};
use glados::fixtures;
use glados::generator::Genome;
use glados::level::Level;
use glados::p2c::Puzzle;
use glados::rng::Rng;
use glados::solver::{judge, Judgement};
use glados::tokens::{decode, encode, vocab};

const BUDGET: usize = 50_000;

/// Verdict and plan length; plans themselves name items by index, which the format renumbers.
fn outcome(p: &Puzzle) -> String {
    match judge(&Level::from_puzzle(p), BUDGET) {
        Err(problems) => format!("problems {problems:?}"),
        Ok(j) => match j.judgement {
            Judgement::Solvable(plan) => format!("solvable in {}", plan.len()),
            Judgement::Unsolvable => "unsolvable".into(),
            Judgement::Uncertain(plan) => format!("uncertain {:?}", plan.map(|p| p.len())),
        },
    }
}

/// Encode, decode, encode again: same tokens, same verdict, no new mount problems.
fn round_trip(name: &str, p: &Puzzle) -> Vec<String> {
    let tokens = encode(p).unwrap_or_else(|e| panic!("{name}: {e}"));
    let back = decode(&tokens.join(" ")).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(encode(&back).unwrap(), tokens, "{name}: not canonical");
    assert_eq!(outcome(&back), outcome(p), "{name}: verdict changed");
    assert_eq!(
        Level::from_puzzle(&back).mount_problems().is_empty(),
        Level::from_puzzle(p).mount_problems().is_empty(),
        "{name}: mounting changed"
    );
    tokens
}

fn mutated_genomes(seed: u64, n: usize) -> Vec<Genome> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let mut g = Genome::random(&mut rng);
            for _ in 0..rng.range(0, 15) {
                g.mutate(&mut rng);
            }
            g
        })
        .collect()
}

#[test]
fn fixtures_round_trip() {
    for (name, p) in fixtures::all() {
        round_trip(name, &p);
    }
}

#[test]
fn worked_example_tokens() {
    let tokens = encode(&fixtures::worked_example()).unwrap().join(" ");
    assert_eq!(
        tokens,
        "size 8 8 6 open 0 0 1 7 4 5 0 5 3 7 7 5 \
         white +x 7 0 1 7 4 4 white -x 0 5 3 7 7 5 \
         entry 4 0 1 +y exit 4 7 3 -y dropper 2 2 5 standard button 2 6 3 weighted > exit"
    );
}

#[test]
fn real_files_round_trip_or_refuse() {
    // The preview map with goo and a checkerboard encodes; the style map uses bridges,
    // turrets and the like, which the format leaves out on purpose.
    let goo =
        Puzzle::parse(&std::fs::read_to_string("fixtures/real/1501698868.p2c").unwrap()).unwrap();
    round_trip("1501698868", &goo);
    let style =
        Puzzle::parse(&std::fs::read_to_string("fixtures/real/1500854111.p2c").unwrap()).unwrap();
    assert!(encode(&style).unwrap_err().contains("unsupported item"));
}

#[test]
fn generated_levels_round_trip() {
    let mut encoded = 0;
    for (i, g) in mutated_genomes(3, 150).iter().enumerate() {
        let Some(p) = g.build() else { continue };
        round_trip(&format!("genome {i}: {g:?}"), &p);
        encoded += 1;
    }
    assert!(encoded > 60, "only {encoded} built");
}

#[test]
fn vocab_covers_every_token() {
    let v = vocab();
    let mut sorted = v.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), v.len(), "duplicate tokens");
    let mut levels: Vec<Puzzle> = fixtures::all().into_iter().map(|(_, p)| p).collect();
    levels.extend(mutated_genomes(4, 200).iter().filter_map(Genome::build));
    for p in &levels {
        for t in encode(p).unwrap() {
            assert!(v.contains(&t), "{t:?} missing from vocab");
        }
    }
}

#[test]
fn decoder_rejects_garbage_without_panicking() {
    let v = vocab();
    let mut rng = Rng::new(8);
    let valid: Vec<Vec<String>> = mutated_genomes(5, 60)
        .iter()
        .filter_map(Genome::build)
        .map(|p| encode(&p).unwrap())
        .collect();
    for _ in 0..3000 {
        let mut t = rng.pick(&valid).clone();
        match rng.below(4) {
            0 => t.truncate(rng.below(t.len())),
            1 => {
                let i = rng.below(t.len());
                t[i] = rng.pick(&v).clone();
            }
            2 => {
                let i = rng.below(t.len());
                t.remove(i);
            }
            _ => {
                let n = rng.below(80);
                t = (0..n).map(|_| rng.pick(&v).clone()).collect();
                if rng.chance(0.5) {
                    t.insert(0, "size".into());
                }
            }
        }
        // Must not panic; anything it accepts must be writable and re-readable.
        if let Ok(p) = decode(&t.join(" ")) {
            let text = p.to_p2c();
            assert_eq!(Puzzle::parse(&text).unwrap(), p);
            let _ = label(&p, 5_000, 0);
        }
    }
    assert!(decode("").is_err());
    assert!(decode("size 2 2 2 s1 o1 | =").is_err());
    assert!(decode("size 1 1 2 s1 o1 | / white +z 0 0 1 0 0 0").is_err());
    assert!(decode("size 1 1 2 s1 o1 | / button 0 0 1 weighted > field 0").is_err());
    assert!(decode("size 1 1 2 s1 o01 | /").is_err());
    assert!(
        decode("size 2 1 2 s1 o1 | = = /").is_err(),
        "too many columns"
    );
    assert!(decode("size 1 2 2 / /").is_err(), "empty first row");
    assert!(decode("size 2 2 2 open").is_err(), "empty box group");
    assert!(decode("size 2 2 2 open 0 0 0 1 1").is_err(), "short box");
    assert!(decode("size 2 2 2 open 1 0 0 0 1 1").is_err(), "corners out of order");
    assert!(decode("size 2 2 2 open 0 0 0 1 1 1 white +z").is_err(), "white without a box");
    // The earlier run format still reads: short rows repeat their last column, an empty row
    // repeats the row before, and rows left out at the end repeat the last one.
    let canonical = |t: &str| encode(&decode(t).unwrap()).unwrap().join(" ");
    assert_eq!(
        canonical("size 3 2 2 s1 o1 | o2 | / /"),
        "size 3 2 2 open 1 0 0 2 1 1 0 0 1 2 1 1"
    );
    assert_eq!(
        canonical("size 3 3 2 o2 | / s1 o1 | / entry 0 2 1 +y"),
        "size 3 3 2 open 0 0 0 2 0 1 0 1 1 2 2 1 entry 0 2 1 +y"
    );
    // A face is shared: `+z` over the solid layer paints the open layer's floors, which are
    // its `-z` surfaces. A `black` group with no boxes: every face that way is white.
    assert_eq!(
        canonical("size 2 2 2 open 0 0 1 1 1 1 white +z 0 0 0 1 1 0"),
        "size 2 2 2 open 0 0 1 1 1 1 black -z"
    );
    assert_eq!(
        canonical("size 2 2 2 open 0 0 1 1 1 1 black +z 1 0 0 1 0 0"),
        "size 2 2 2 open 0 0 1 1 1 1 white +z 0 0 1 1 1 1 black -z 1 0 1 1 0 1"
    );
}

#[test]
fn labels_match_the_solver() {
    let solvable = label(&fixtures::worked_example(), BUDGET, 0);
    assert!(
        matches!(
            solvable,
            Label::Solvable {
                min_len: 5,
                metrics: Some(_)
            }
        ),
        "{solvable:?}"
    );
    assert!(solvable
        .json_fields()
        .starts_with("\"verdict\":\"solvable\",\"min_len\":5,\"metrics\":{"));
    assert_eq!(label(&fixtures::goo_pit(), BUDGET, 0), Label::Unsolvable);
}

#[test]
fn json_strings_escape() {
    assert_eq!(dataset::json_str("a\"b\\c\nd"), "\"a\\\"b\\\\c\\nd\"");
}

fn small(threads: usize) -> Config {
    Config {
        seed: 2,
        count: 60,
        budget: 20_000,
        threads,
        fresh: 0.3,
        pool: 50,
    }
}

#[test]
fn dataset_is_distinct_and_deterministic() {
    let run = |threads| {
        let mut lines = Vec::new();
        generate(&small(threads), |s| lines.push(s.json()));
        lines
    };
    let one = run(1);
    assert_eq!(one.len(), 60);
    let mut distinct = one.clone();
    distinct.sort();
    distinct.dedup();
    assert_eq!(distinct.len(), 60);
    assert_eq!(one, run(4));
    assert!(one.iter().any(|l| l.contains("\"verdict\":\"solvable\"")));
}

#[test]
fn goo_items_sit_on_the_floor_of_their_run() {
    // A two-deep goo column beside a one-deep pit: the goo item rests on solid ground and
    // floods the same cells.
    let goo_problems = |line: &str| -> Vec<String> {
        let level = Level::from_puzzle(&decode(line).unwrap());
        level.mount_problems().into_iter().filter(|p| p.contains("goo")).collect()
    };
    let line = "size 3 1 4 s1 o3 | s1 g1 o2 | g2 o2 | / entry 0 0 1 +x exit 0 0 2 +x";
    assert_eq!(goo_problems(line), Vec::<String>::new());
    let level = Level::from_puzzle(&decode(line).unwrap());
    assert!(level.goo([2, 0, 0]) && level.goo([2, 0, 1]) && level.goo([1, 0, 1]));
    // Goo over open air has no floor to stand on.
    let floating = "size 2 1 4 s1 o3 | o1 g1 o2 | / entry 0 0 1 +x exit 0 0 2 +x";
    assert!(!goo_problems(floating).is_empty());
}
