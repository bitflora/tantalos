use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::Path,
};
use tantalos::{
    dataset::json_str,
    editor,
    level::Level,
    metrics, plan, route,
    solver::{Mode, Model, Verdict},
    structural, tokens,
    witness::{repair, RepairLimits},
};

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
fn run() -> Result<(), String> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    if a.first().map(String::as_str) == Some("recover") {
        if a.len() != 3 {
            return Err(
                "usage: tantalos-structural recover PARENT.tokens.txt PARENT.plans.txt".into(),
            );
        }
        let puzzle = tokens::decode(&fs::read_to_string(&a[1]).map_err(|e| e.to_string())?)?;
        let level = Level::from_puzzle(&puzzle);
        let model = Model::new(&level, Mode::Conservative).map_err(|e| format!("{e:?}"))?;
        let archive = fs::read_to_string(&a[2]).map_err(|e| e.to_string())?;
        for text in archive.lines().filter(|line| !line.trim().is_empty()) {
            let Ok(steps) = plan::decode(text) else {
                continue;
            };
            let checked = plan::check(&model, &steps);
            if checked.solved && route::route_plan(&level, &model, &checked.plan).is_ok() {
                eprintln!("recovered from archive");
                println!("{text}");
                return Ok(());
            }
        }
        let (_, alternatives) = model.shortest_plans(250_000, 8, 0);
        for actions in alternatives {
            if actions.len() < 12 {
                continue;
            }
            if let Ok(route) = route::route_plan(&level, &model, &actions) {
                let text = plan::encode(&route);
                if plan::check(&model, &plan::decode(&text)?).solved {
                    eprintln!("recovered from fresh search");
                    println!("{text}");
                    return Ok(());
                }
            }
        }
        return Err("no grounded parent solution found".into());
    }
    if a.first().map(String::as_str) == Some("audit") {
        if a.len() != 2 {
            return Err("usage: tantalos-structural audit LEVEL.tokens.txt".into());
        }
        let p = tokens::decode(&fs::read_to_string(&a[1]).map_err(|e| e.to_string())?)?;
        let l = Level::from_puzzle(&p);
        for (name, mode, angled) in [
            ("conservative", Mode::Conservative, false),
            ("liberal", Mode::Liberal, false),
            ("angled-conservative", Mode::Conservative, true),
            ("angled-liberal", Mode::Liberal, true),
        ] {
            let model = if angled {
                Model::new_angled(&l, mode)
            } else {
                Model::new(&l, mode)
            }
            .map_err(|e| format!("{e:?}"))?;
            let result = model.solve(250_000);
            let (verdict, len) = match &result.verdict {
                Verdict::Solved(a) => ("solved", a.len()),
                Verdict::Unknown => ("unknown", 0),
                Verdict::Unsolvable => ("unsolvable", 0),
            };
            println!(
                "{{\"model\":{},\"verdict\":{},\"actions\":{len},\"states\":{},\"complete\":{}}}",
                json_str(name),
                json_str(verdict),
                result.states,
                model.complete
            );
            if name == "conservative" {
                use tantalos::{
                    regions::Move,
                    solver::{Action, CubeLoc},
                    witness::{audit_without, Audit},
                };
                let checks: [(&str, fn(&Action) -> bool); 6] = [
                    ("portals", |a| matches!(a, Action::Fire { .. })),
                    ("laser redirection", |a| {
                        matches!(
                            a,
                            Action::PutDown {
                                at: CubeLoc::Spot(_)
                            }
                        )
                    }),
                    ("timed actions", |a| {
                        matches!(
                            a,
                            Action::Dash { .. } | Action::Fetch { .. } | Action::DoubleDash { .. }
                        )
                    }),
                    ("funnel travel", |a| {
                        matches!(
                            a,
                            Action::Move {
                                kind: Move::Funnel,
                                ..
                            }
                        )
                    }),
                    ("stairs", |a| {
                        matches!(
                            a,
                            Action::Move {
                                kind: Move::Stairs,
                                ..
                            }
                        )
                    }),
                    ("cube transport through portals", |a| {
                        matches!(a, Action::Throw { .. } | Action::PortalReach { .. })
                    }),
                ];
                for (label, forbidden) in checks {
                    let audit = audit_without(&model, 100_000, |a, _| !forbidden(a));
                    let outcome = match audit {
                        Audit::Required => "required",
                        Audit::Bypass => "not-required",
                        Audit::Unknown => "unknown",
                    };
                    println!(
                        "{{\"mechanic\":{},\"necessity\":{}}}",
                        json_str(label),
                        json_str(outcome)
                    );
                }
            }
        }
        return Ok(());
    }
    if a.len() != 3 {
        return Err(
            "usage: tantalos-structural PARENT.tokens.txt PARENT.plan.txt NEW_OUT_DIR".into(),
        );
    }
    let p = tokens::decode(&fs::read_to_string(&a[0]).map_err(|e| e.to_string())?)?;
    let steps = plan::decode(&fs::read_to_string(&a[1]).map_err(|e| e.to_string())?)?;
    let level = Level::from_puzzle(&p);
    let model = Model::new(&level, Mode::Conservative).map_err(|e| format!("{e:?}"))?;
    let original = plan::check(&model, &steps);
    if !original.solved {
        return Err("parent plan does not solve conservative model".into());
    }
    let ground = route::route_plan(&level, &model, &original.plan)?;
    let parent_steps = plan::decode(&plan::encode(&ground))?;
    let parent_ops = structural::operations(&parent_steps);
    let parent_chain = metrics::chain_depth(
        &model,
        &original.plan,
        &metrics::replay(&model, &original.plan).unwrap(),
    );
    let out = Path::new(&a[2]);
    fs::create_dir(out).map_err(|e| e.to_string())?;
    let mut stats = BTreeMap::<String, usize>::new();
    let mut seen = HashSet::new();
    let mut count = 0;
    let mut records = Vec::new();
    let mutations = structural::mutations(&p);
    eprintln!(
        "{} dependency mutations; parent {} actions, chain {parent_chain}",
        mutations.len(),
        original.plan.len()
    );
    for (attempt, m) in mutations.into_iter().enumerate() {
        let status = (|| -> Result<(), String> {
            let text = tokens::encode(&m.puzzle)?.join(" ");
            if !seen.insert(text.clone()) {
                return Err("duplicate".into());
            }
            let child = tokens::decode(&text)?;
            let l = Level::from_puzzle(&child);
            if !editor::editor_flags(&l).is_empty() {
                return Err("editor flags".into());
            }
            let model = Model::new(&l, Mode::Conservative).map_err(|_| "invalid model")?;
            if plan::check(&model, &steps).solved {
                return Err("parent solution unchanged".into());
            }
            let repaired = repair(
                &model,
                &steps,
                RepairLimits {
                    edits: 8,
                    states: 10_000,
                },
            )
            .ok();
            // Global search here measures shortcuts; admission still has a witness
            // when BFS is unknown. Such candidates are saved separately for review.
            let result = model.solve(100_000);
            let (actions, optimal) = match result.verdict {
                Verdict::Solved(p) => (p, true),
                Verdict::Unsolvable => return Err("unsolvable in model".into()),
                Verdict::Unknown => match &repaired {
                    Some(r) => (r.actions.clone(), false),
                    None => return Err("no witness within budget".into()),
                },
            };
            if actions.len() < 16 || actions.len() * 5 < original.plan.len() * 4 {
                return Err("short solution".into());
            }
            let path = metrics::replay(&model, &actions).ok_or("replay failed")?;
            let chain = metrics::chain_depth(&model, &actions, &path);
            if chain < 8.0 || chain < parent_chain * 0.8 {
                return Err("dependency depth collapsed".into());
            }
            let grounded =
                route::route_plan(&l, &model, &actions).map_err(|_| "grounding failed")?;
            let encoded = plan::encode(&grounded);
            let next = plan::decode(&encoded)?;
            if !plan::check(&model, &next).solved {
                return Err("grounded replay failed".into());
            }
            let distance = structural::edit_distance(&parent_ops, &structural::operations(&next));
            if distance < 3 {
                return Err("resource operations too similar".into());
            }
            let name = format!("candidate-{count:03}");
            fs::write(out.join(format!("{name}.p2c")), child.to_p2c())
                .map_err(|e| e.to_string())?;
            fs::write(out.join(format!("{name}.plan.txt")), &encoded).map_err(|e| e.to_string())?;
            fs::write(out.join(format!("{name}.tokens.txt")), &text).map_err(|e| e.to_string())?;
            fs::write(
                out.join(format!("{name}.route.jsonl")),
                grounded.json_lines(&name, child.chamber_size()).join("\n") + "\n",
            )
            .map_err(|e| e.to_string())?;
            records.push(format!("{{\"id\":{},\"tokens\":{},\"plan\":{},\"change\":{},\"actions\":{},\"parent_actions\":{},\"chain\":{chain},\"parent_chain\":{parent_chain},\"operation_distance\":{distance},\"shortest_proven\":{optimal},\"repair_found\":{},\"search_states\":{}}}",json_str(&name),json_str(&text),json_str(&encoded),json_str(&m.description),actions.len(),original.plan.len(),repaired.is_some(),result.states));
            eprintln!(
                "{name}: len {}, chain {chain}, changed operations {distance}; {}",
                actions.len(),
                m.description
            );
            count += 1;
            Ok(())
        })();
        *stats
            .entry(status.err().unwrap_or("retained for review".into()))
            .or_default() += 1;
        if attempt % 20 == 19 {
            eprintln!("{} checked, {count} retained", attempt + 1);
        }
    }
    fs::write(out.join("candidates.jsonl"), records.join("\n") + "\n")
        .map_err(|e| e.to_string())?;
    fs::write(out.join("report.txt"), format!("{stats:#?}\n")).map_err(|e| e.to_string())?;
    eprintln!("{stats:#?}");
    Ok(())
}
