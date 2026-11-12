//! Experimental solution-carrying generation. Never feeds training automatically.
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::Path;

use tantalos::{
    dataset::json_str,
    editor,
    interactions::{self, Topology},
    level::Level,
    mutate,
    p2c::Puzzle,
    plan::{self, Step},
    rng::Rng,
    route,
    solver::{Action, Mode, Model},
    tokens,
    witness::{self, Audit, RepairLimits, ReuseChamber},
};

const USAGE: &str = "Usage:\n  tantalos-witness interactions OUT_DIR\n  tantalos-witness generate OUT_DIR [COUNT=12] [SEED=1]  (original single-family experiment)\n  tantalos-witness repair LEVEL.p2c PLAN.txt OUT_DIR\nOutputs are experimental model-checked witnesses, not game-verified or shortest plans.\nExisting output directories are never overwritten.";

fn main() -> std::process::ExitCode {
    match run(&std::env::args().skip(1).collect::<Vec<_>>()) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            std::process::ExitCode::FAILURE
        }
    }
}

struct Candidate {
    puzzle: Puzzle,
    steps: Vec<Step>,
    tokens: String,
    plan: String,
    route: String,
    edits: usize,
    states: usize,
}

fn verify(puzzle: Puzzle, steps: &[Step]) -> Result<Candidate, String> {
    // Check the exact token-roundtripped object that downstream tools will see.
    let encoded = tokens::encode(&puzzle)?.join(" ");
    let title = puzzle.title().to_string();
    let mut puzzle = tokens::decode(&encoded)?;
    puzzle.set_title(&title);
    let level = Level::from_puzzle(&puzzle);
    let flags = editor::editor_flags(&level);
    if !flags.is_empty() {
        return Err("editor flags".into());
    }
    let model = Model::new(&level, Mode::Conservative).map_err(|_| "invalid model")?;
    let repaired = witness::repair(
        &model,
        steps,
        RepairLimits {
            edits: 4,
            states: 5_000,
        },
    )
    .map_err(|e| format!("repair {e:?}"))?;
    let grounded =
        route::route_plan(&level, &model, &repaired.actions).map_err(|_| "route failed")?;
    let text = plan::encode(&grounded);
    let steps = plan::decode(&text)?;
    if !plan::check(&model, &steps).solved {
        return Err("grounded replay failed".into());
    }
    let route = grounded
        .json_lines(puzzle.title(), puzzle.chamber_size())
        .join("\n");
    Ok(Candidate {
        puzzle,
        steps,
        tokens: encoded,
        plan: text,
        route,
        edits: repaired.edits,
        states: repaired.states,
    })
}

fn save(
    out: &Path,
    id: usize,
    c: &Candidate,
    parent: Option<usize>,
    edits: &[String],
    audit: &str,
) -> Result<String, String> {
    let name = format!("witness-{id:03}");
    for (ext, text) in [
        ("p2c", c.puzzle.to_p2c()),
        ("tokens.txt", c.tokens.clone()),
        ("plan.txt", c.plan.clone()),
        ("route.jsonl", c.route.clone()),
    ] {
        fs::write(out.join(format!("{name}.{ext}")), format!("{text}\n"))
            .map_err(|e| e.to_string())?;
    }
    Ok(format!("{{\"id\":{},\"src\":\"witness-prototype\",\"tokens\":{},\"plan\":{},\"witness_len\":{},\"verification\":\"conservative-and-grounded\",\"game_verified\":false,\"dependency_audit\":{},\"repair_edits\":{},\"repair_states\":{},\"parent\":{},\"mutations\":[{}]}}", json_str(&name), json_str(&c.tokens), json_str(&c.plan), c.steps.len(), json_str(audit), c.edits, c.states, parent.map_or("null".into(), |n| json_str(&format!("witness-{n:03}"))), edits.iter().map(|s| json_str(s)).collect::<Vec<_>>().join(",")))
}

fn output_dir(path: &str) -> Result<&Path, String> {
    let path = Path::new(path);
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::create_dir(path).map_err(|e| format!("Cannot create new output directory: {e}"))?;
    Ok(path)
}

fn run(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("interactions") if args.len() == 2 => {
            // Construct and check all three before publishing a reviewable set.
            let mut checked = Vec::new();
            for topology in Topology::ALL {
                let mut design = interactions::compile(topology);
                design.puzzle = tokens::decode(&tokens::encode(&design.puzzle)?.join(" "))?;
                design
                    .puzzle
                    .set_title(&format!("Interaction {}", topology.name()));
                let level = Level::from_puzzle(&design.puzzle);
                let model = Model::new(&level, Mode::Conservative).map_err(|e| format!("{e:?}"))?;
                let actions = interactions::synthesize(&design, &level, &model, 20_000)?;
                let audits = interactions::audit(&design, &level, &model, 20_000);
                if audits.iter().any(|(_, a)| *a != Audit::Required) {
                    return Err(format!(
                        "{} dependency audit failed: {audits:?}",
                        topology.name()
                    ));
                }
                let steps = plan::decode(&plan::encode_actions(&model, &actions))?;
                let candidate = verify(design.puzzle.clone(), &steps)?;
                let nodes = design
                    .graph
                    .iter()
                    .enumerate()
                    .map(|(id, m)| {
                        format!(
                            "{{\"id\":{id},\"name\":{},\"condition\":{},\"depends_on\":{:?}}}",
                            json_str(m.name),
                            json_str(&format!("{:?}", m.condition)),
                            m.depends_on
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                let audit_json = audits
                    .iter()
                    .map(|(name, a)| {
                        format!(
                            "{{\"condition\":{},\"result\":{}}}",
                            json_str(name),
                            json_str(&format!("{a:?}"))
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                checked.push((
                    topology,
                    candidate,
                    format!(
                        "{{\"topology\":{},\"milestones\":[{nodes}],\"audits\":[{audit_json}]}}",
                        json_str(topology.name())
                    ),
                ));
            }
            let out = output_dir(&args[1])?;
            let mut rows = Vec::new();
            for (id, (topology, candidate, graph)) in checked.iter().enumerate() {
                rows.push(save(
                    out,
                    id,
                    candidate,
                    None,
                    &[format!("dependency topology {}", topology.name())],
                    "required-in-model",
                )?);
                fs::write(out.join(format!("witness-{id:03}.graph.json")), graph)
                    .map_err(|e| e.to_string())?;
                println!(
                    "{}: {} checked actions; all four dependency audits required",
                    topology.name(),
                    candidate.steps.len()
                );
            }
            fs::write(out.join("levels.jsonl"), rows.join("\n") + "\n")
                .map_err(|e| e.to_string())?;
            fs::write(out.join("README.md"), "# Interaction playtest set\n\nThree dependency structures, one representative each.\n\n- witness-000: Exchange — one resource replaces another.\n- witness-001: Handover — secure an alternate route before redirecting its power source.\n- witness-002: Tandem — keep both resources working together.\n\nEach has an explicit dependency graph, a locally synthesized plan, conservative replay, a grounded route, and four required-in-model counterexample checks. Unknown audits are not accepted. No global BFS solve was required for admission. They share a two-bank arena; these are three authored dependency rules, not unrestricted puzzle invention. Plan lengths are upper bounds, not difficulty scores. Game verification is recorded separately in ingame.jsonl if run.\n").map_err(|e| e.to_string())?;
            Ok(())
        }
        Some("generate") if (2..=4).contains(&args.len()) => {
            let count: usize = args
                .get(2)
                .map_or(Ok(12), |s| s.parse())
                .map_err(|_| "invalid count")?;
            let seed: u64 = args
                .get(3)
                .map_or(Ok(1), |s| s.parse())
                .map_err(|_| "invalid seed")?;
            if !(1..=100).contains(&count) {
                return Err("count must be 1..100".into());
            }
            let out = output_dir(&args[1])?;
            let mut rng = Rng::new(seed);
            let mut pool: Vec<Candidate> = Vec::new();
            let mut seen = HashSet::new();
            let mut rows = Vec::new();
            let mut stats = BTreeMap::<String, usize>::new();
            for attempt in 0..count * 30 {
                if pool.len() == count {
                    break;
                }
                let (p, steps, parent, changes) = if pool.is_empty() || attempt % 3 == 0 {
                    let dims = ReuseChamber {
                        lower_width: rng.range(5, 7),
                        ledge_depth: rng.range(3, 5),
                        depth: rng.range(7, 9),
                        lane: rng.range(2, 3),
                    };
                    let (p, steps) = dims.build()?;
                    (p, steps, None, vec![format!("shared chamber {dims:?}")])
                } else {
                    let id = rng.below(pool.len());
                    let Some((p, changes)) = mutate::mutate(&pool[id].puzzle, &mut rng, 1) else {
                        *stats.entry("no mutation".into()).or_default() += 1;
                        continue;
                    };
                    // This experiment varies geometry and placements, not extra
                    // chores or decorative items. Richer interaction families need
                    // their own causal contract, rather than a length reward.
                    if changes
                        .iter()
                        .any(|s| s.starts_with("add ") || s.starts_with("remove "))
                    {
                        *stats
                            .entry("item-count mutation excluded".into())
                            .or_default() += 1;
                        continue;
                    }
                    (p, pool[id].steps.clone(), Some(id), changes)
                };
                let candidate = match verify(p, &steps) {
                    Ok(c) => c,
                    Err(reason) => {
                        *stats.entry(reason).or_default() += 1;
                        continue;
                    }
                };
                if !seen.insert(candidate.tokens.clone()) {
                    *stats.entry("duplicate".into()).or_default() += 1;
                    continue;
                }
                let model = Model::new(&Level::from_puzzle(&candidate.puzzle), Mode::Conservative)
                    .map_err(|_| "model changed")?;
                // Mutations must retain the single shared resource and its two roles.
                if model.start.cubes.len() != 1 || model.buttons.len() != 2 {
                    *stats.entry("changed resource contract".into()).or_default() += 1;
                    continue;
                }
                let access = model
                    .buttons
                    .iter()
                    .position(|b| b.cell[2] == 1)
                    .ok_or("missing access button")?;
                let reuse = witness::audit_access(&model, access, 5_000);
                let portals =
                    witness::audit_without(&model, 5_000, |a, _| !matches!(a, Action::Fire { .. }));
                if reuse == Audit::Bypass || portals == Audit::Bypass {
                    *stats.entry("dependency shortcut".into()).or_default() += 1;
                    continue;
                }
                let audit = if reuse == Audit::Required && portals == Audit::Required {
                    "required-in-model"
                } else {
                    "unknown-budget-or-model"
                };
                rows.push(save(out, pool.len(), &candidate, parent, &changes, audit)?);
                *stats
                    .entry(
                        if parent.is_some() {
                            "kept mutant"
                        } else {
                            "kept seed"
                        }
                        .into(),
                    )
                    .or_default() += 1;
                pool.push(candidate);
            }
            fs::write(out.join("levels.jsonl"), rows.join("\n") + "\n")
                .map_err(|e| e.to_string())?;
            let report = format!("# Shared-resource prototype\n\nSeed: {seed}. Kept {} of {count} requested.\n\nOne puzzle family, not a diversity benchmark: the cube powers access, a prepared portal replaces that access, then the same cube powers the exit. Each level is a single connected chamber.\n\nEvery saved plan was conservatively replayed, grounded, and replayed again after token roundtrip. No game checks have run. Witness length is not minimum length. Dependency audits are model-relative; unknown is not proof. Nothing was added to training.\n\nStatistics: {stats:#?}\n", pool.len());
            fs::write(out.join("README.md"), report).map_err(|e| e.to_string())?;
            println!(
                "Saved {} level/solution pairs to {}\n{stats:#?}",
                pool.len(),
                out.display()
            );
            if pool.len() < count {
                return Err("Attempt limit reached; partial results and report saved.".into());
            }
            Ok(())
        }
        Some("repair") if args.len() == 4 => {
            let p = Puzzle::parse(&fs::read_to_string(&args[1]).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
            let steps = plan::decode(&fs::read_to_string(&args[2]).map_err(|e| e.to_string())?)?;
            let candidate = verify(p, &steps)?;
            let out = output_dir(&args[3])?;
            let row = save(out, 0, &candidate, None, &[], "not-run")?;
            fs::write(out.join("levels.jsonl"), row + "\n").map_err(|e| e.to_string())?;
            println!(
                "Saved checked plan; {} edits, {} states",
                candidate.edits, candidate.states
            );
            Ok(())
        }
        _ => Err(USAGE.into()),
    }
}
