//! `glados-score`: solving and checking for training, evaluation and play.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use glados::cli::{self, flags, load, parse, Result};
use glados::level::Level;
use glados::p2c::Puzzle;
use glados::solver::{Mode, Model, Problem, DEFAULT_MAX_STATES};
use glados::{dataset, plan, route, tokens};

const USAGE: &str = "usage:
  glados-score score [--budget N] [--p2c DIR] [--p2c-all DIR] [--editor-rules on|off] [--plan] [--plan-only] [--plans K]
                     [--alternatives N] [FILE]
                                decode and solve one token sequence per line of FILE (or
                                stdin); JSONL to stdout. --p2c writes the solvable ones,
                                --p2c-all every one that parsed. --editor-rules off: don't
                                call levels invalid for our model of the editor's rules
                                (for levels the game's editor accepted, e.g. the Workshop).
                                --plan: add each solvable level's plan as plan tokens
                                (\"plan\", src/plan.rs), or why it has none (\"plan_error\").
                                --plan-only: just the plan and min_len, for levels already
                                labelled solvable: no metrics, portal check or liberal model.
                                --plans K (with --plan-only): also \"plans\", up to K
                                different shortest plans, \"plan\" first. --alternatives N
                                (with --plan-only): also \"alt_plans\", up to N plans that
                                solve it another way (a face class the first plan fires at
                                switched off; at most 3 actions longer)
  glados-score check [--mode conservative|liberal] [--route DIR] [FILE]
                                check plans: JSONL with \"tokens\" (a level) and \"plan\" (plan
                                tokens) or \"plans\" (a list of them, all against one model),
                                optionally \"min_len\", from FILE or stdin; one JSON line per
                                input, with a \"results\" list of one verdict per plan, in order
                                (solved, illegal, unfinished, unknown, unparsed, invalid,
                                unsupported). A plan that fails is checked again with angled
                                laser spots, as the solver retries such levels.
                                --route DIR: NNNNN.p2c and NNNNN.route.jsonl (as
                                route writes) for the first solved plan of a line
  glados-score env              the legal next plan tokens, for decoding plans step by step
                                (src/env.rs): a JSONL server on stdin/stdout, one answer
                                line per request line, flushed.
                                {\"op\":\"open\",\"id\":ID,\"tokens\":LEVEL[,\"mode\":\"liberal\"]}
                                  -> {\"ok\":true} or {\"error\":...,\"verdict\":\"invalid\"|...}
                                {\"op\":\"next\",\"ids\":[ID,...],\"plans\":[PLAN,...]}
                                  -> {\"results\":[{\"allowed\":[TOKEN,...],\"end\":B,\"dead\":B,\"whole\":B},...]}
                                  (end: the plan solves if it stops here; dead: no step
                                  fits; whole: the plan so far is whole steps)
                                {\"op\":\"close\",\"ids\":[ID,...]} -> {\"ok\":true}
  glados-score grammar [--check FILE]
                                the legal next level tokens, for the level net's constrained
                                decoding (src/grammar.rs): a JSONL server on stdin/stdout, one
                                answer line per request line, flushed.
                                {\"levels\":[TOKENS,...]} (each a level's tokens so far)
                                  -> {\"results\":[{\"allowed\":[TOKEN,...],\"end\":B} or {\"error\":...},...]}
                                  (end: the level may end here)
                                --check FILE: whether each line's \"tokens\" (JSONL) is a whole
                                legal level; one JSON line per input, {\"ok\":B[,\"error\":...]}
  glados-score route [--plan TOKENS [--mode conservative|liberal]] <file.p2c>
                                the conservative plan made concrete for playing it in the game:
                                where to stand, look, fire and walk (JSON lines, src/route.rs).
                                --plan: route that plan instead (plan tokens, src/plan.rs); one
                                that doesn't solve the level is routed as far as its steps fit,
                                then an \"attempt\" leg tries the step that fits no move, and
                                the head line's \"net\" says what the check found. --mode
                                liberal: check and route the plan by the liberal rules (a plan
                                only they allow, to try in the game)";

fn main() -> ExitCode {
    cli::main(USAGE, |args| match args.first().map(String::as_str) {
        Some("score") => Some(run_score(&args[1..])),
        Some("check") => Some(run_check(&args[1..])),
        Some("route") => Some(run_route(&args[1..])),
        Some("env") => Some(run_env(&args[1..])),
        Some("grammar") => Some(run_grammar(&args[1..])),
        _ => None,
    })
}

fn run_route(args: &[String]) -> Result<()> {
    let mut plan = None;
    let mut mode = Mode::Conservative;
    let mut rest = Vec::new();
    for (flag, value) in flags(args, &mut rest)? {
        match flag {
            "--plan" => plan = Some(value),
            "--mode" => {
                mode = match value {
                    "conservative" => Mode::Conservative,
                    "liberal" => Mode::Liberal,
                    _ => return Err(format!("--mode: conservative or liberal, not {value}").into()),
                }
            }
            _ => return Err(format!("unknown flag {flag}\n{USAGE}").into()),
        }
    }
    if plan.is_none() && mode == Mode::Liberal {
        return Err("--mode liberal goes with --plan (the solver's own route is conservative)".into());
    }
    match rest.as_slice() {
        [file] => print_route(Path::new(file), plan, mode),
        _ => Err(format!("route takes one .p2c file\n{USAGE}").into()),
    }
}

/// `route FILE [--plan TOKENS [--mode M]]`: the solver's plan, or the given one, made concrete.
/// A given plan that doesn't solve the level is routed as far as it fits, then a try at the
/// step that fits no move; the head line says what its check found (`"net"`). `mode`: the
/// rules the given plan is checked and routed by (liberal: a plan only those rules allow, to
/// be tried in the game).
fn print_route(path: &Path, plan: Option<&str>, mode: Mode) -> Result<()> {
    let (_, puzzle) = load(path)?;
    let level = Level::from_puzzle(&puzzle);
    let at = |e: String| format!("{}: {e}", path.display());
    let r = match plan {
        None => route::route(&level, DEFAULT_MAX_STATES).map_err(at)?,
        Some(text) => {
            let model = Model::new(&level, mode).map_err(|p| at(format!("{p:?}")))?;
            let (verdict, steps, mut r) = match plan::decode(text) {
                Err(e) => {
                    let mut r = route::route_prefix(&level, &model, &[], None).map_err(at)?;
                    r.net = Some(route::NetCheck {
                        verdict: "unparsed".into(),
                        steps: 0,
                        steps_ok: 0,
                        problem: Some(format!("plan: {e}")),
                    });
                    (None, 0, r)
                }
                Ok(steps) => {
                    let c = plan::check(&model, &steps);
                    let verdict = if c.solved {
                        "solved"
                    } else if c.steps_ok == steps.len() {
                        "unfinished"
                    } else if c.exhausted {
                        "unknown"
                    } else {
                        "illegal"
                    };
                    let r = if c.solved {
                        route::route_plan(&level, &model, &c.plan)
                    } else {
                        let failed = (verdict == "illegal")
                            .then(|| (&steps[c.steps_ok], c.problem.as_deref().unwrap_or("")));
                        route::route_prefix(&level, &model, &c.plan, failed)
                    }
                    .map_err(at)?;
                    (Some((verdict, c.steps_ok, c.problem)), steps.len(), r)
                }
            };
            if let Some((verdict, steps_ok, problem)) = verdict {
                r.net = Some(route::NetCheck {
                    verdict: verdict.into(),
                    steps,
                    steps_ok,
                    problem,
                });
            }
            r
        }
    };
    for line in r.json_lines(puzzle.title(), level.size) {
        println!("{line}");
    }
    Ok(())
}

fn run_score(args: &[String]) -> Result<()> {
    let mut budget = 50_000;
    let mut p2c_dir = None;
    let mut all = false;
    let mut editor_rules = true;
    let mut with_plan = false;
    let mut plan_only = false;
    let mut k = 1;
    let mut alternatives = 0;
    let mut rest = Vec::new();
    // `--plan` and `--plan-only` take no value.
    let args: Vec<String> = args
        .iter()
        .filter(|a| {
            let plan = *a == "--plan";
            let only = *a == "--plan-only";
            with_plan |= plan;
            plan_only |= only;
            !plan && !only
        })
        .cloned()
        .collect();
    for (flag, value) in flags(&args, &mut rest)? {
        match flag {
            "--budget" => budget = parse(flag, value)?,
            "--plans" => k = parse(flag, value)?,
            "--alternatives" => alternatives = parse(flag, value)?,
            "--editor-rules" => {
                editor_rules = match value {
                    "on" => true,
                    "off" => false,
                    _ => return Err(format!("--editor-rules: on or off, not {value}").into()),
                }
            }
            "--p2c" => p2c_dir = Some(PathBuf::from(value)),
            "--p2c-all" => {
                p2c_dir = Some(PathBuf::from(value));
                all = true;
            }
            _ => return Err(format!("unknown flag {flag}\n{USAGE}").into()),
        }
    }
    // Streamed: levels are read as threads need them and written, in order, as they're done.
    let lines = cli::input_lines(&rest, "score", USAGE)?;
    if let Some(dir) = &p2c_dir {
        std::fs::create_dir_all(dir)?;
    }
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    dataset::par_map_ordered(
        lines,
        cli::threads(),
        |(i, line)| {
            let line = line?;
            let line = line.trim();
            let n = line.split_whitespace().count();
            Ok(match tokens::decode(line) {
                Err(e) => (
                    i,
                    format!(
                        "{{\"line\":{i},\"n_tokens\":{n},\"verdict\":\"unparsed\",\"problem\":{}}}",
                        dataset::json_str(&e)
                    ),
                    None,
                ),
                Ok(p) => {
                    let normal: Vec<&str> = line.split_whitespace().collect();
                    let encoded = tokens::encode(&p).map(|t| t.join(" "));
                    let canonical = encoded.as_ref().is_ok_and(|t| *t == normal.join(" "));
                    // The length of the tokens written, which can differ from the input's.
                    let n = encoded.as_ref().map_or(n, |t| t.split(' ').count());
                    let tokens = encoded.map_or(String::new(), |t| {
                        format!("\"tokens\":{},", dataset::json_str(&t))
                    });
                    if plan_only {
                        let list = |ts: &[String]| {
                            ts.iter().map(|t| dataset::json_str(t)).collect::<Vec<_>>().join(",")
                        };
                        let fields = match dataset::plan_only(&p, budget, editor_rules, k, i as u64, alternatives) {
                            Ok(found) => {
                                let mut f = format!(
                                    "\"verdict\":\"solvable\",\"min_len\":{},\"states\":{},\"plan\":{}",
                                    found.min_len,
                                    found.states,
                                    dataset::json_str(&found.plans[0])
                                );
                                if k > 1 {
                                    f += &format!(",\"plans\":[{}]", list(&found.plans));
                                }
                                if alternatives > 0 {
                                    f += &format!(",\"alt_plans\":[{}]", list(&found.alt_plans));
                                }
                                f
                            }
                            Err(label) => label.json_fields(),
                        };
                        return Ok((
                            i,
                            format!("{{\"line\":{i},\"n_tokens\":{n},\"canonical\":{canonical},{tokens}{fields}}}"),
                            None,
                        ));
                    }
                    let (label, plan) = if with_plan {
                        dataset::label_with_plan(&p, budget, i as u64, editor_rules)
                    } else {
                        (dataset::label_with(&p, budget, i as u64, editor_rules), None)
                    };
                    let plan = match plan {
                        Some(Ok(t)) => format!(",\"plan\":{}", dataset::json_str(&t)),
                        Some(Err(e)) => format!(",\"plan_error\":{}", dataset::json_str(&e)),
                        None => String::new(),
                    };
                    let keep =
                        (all || matches!(label, dataset::Label::Solvable { .. })).then_some(p);
                    (
                        i,
                        format!(
                            "{{\"line\":{i},\"n_tokens\":{n},\"canonical\":{canonical},{tokens}{}{plan}}}",
                            label.json_fields()
                        ),
                        keep,
                    )
                }
            })
        },
        |scored: std::io::Result<(usize, String, Option<Puzzle>)>| -> Result<()> {
            let (i, json, puzzle) = scored?;
            writeln!(out, "{json}")?;
            if let (Some(dir), Some(mut p)) = (&p2c_dir, puzzle) {
                p.set_title(&format!("glados sample {i}"));
                std::fs::write(dir.join(format!("{i:05}.p2c")), p.to_p2c())?;
            }
            Ok(())
        },
    )?;
    out.flush()?;
    Ok(())
}

fn run_env(args: &[String]) -> Result<()> {
    use glados::env::Env;
    use glados::mutate::{json_array, json_field};
    use std::io::BufRead;
    if !args.is_empty() {
        return Err(format!("env takes no arguments
{USAGE}").into());
    }
    let mut envs: std::collections::HashMap<String, Env> = std::collections::HashMap::new();
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let error = |why: &str, verdict: &str| {
            format!("{{\"error\":{},\"verdict\":\"{verdict}\"}}", dataset::json_str(why))
        };
        let answer = match json_field(&line, "op").as_deref() {
            Some("open") => match (json_field(&line, "id"), json_field(&line, "tokens")) {
                (Some(id), Some(toks)) => {
                    let mode = match json_field(&line, "mode").as_deref() {
                        Some("liberal") => Mode::Liberal,
                        _ => Mode::Conservative,
                    };
                    match tokens::decode(&toks) {
                        Err(e) => error(&format!("level: {e}"), "unparsed"),
                        Ok(puzzle) => match Env::new(&Level::from_puzzle(&puzzle), mode) {
                            Ok(env) => {
                                envs.insert(id, env);
                                "{\"ok\":true}".to_string()
                            }
                            Err(problems) => {
                                let invalid = problems.iter().any(|p| matches!(p, Problem::Invalid(_)));
                                error(&format!("{problems:?}"), if invalid { "invalid" } else { "unsupported" })
                            }
                        },
                    }
                }
                _ => error("open needs \"id\" and \"tokens\"", "unparsed"),
            },
            Some("next") => match (json_array(&line, "ids"), json_array(&line, "plans")) {
                (Some(ids), Some(plans)) if ids.len() == plans.len() => {
                    let results: Vec<String> = ids
                        .iter()
                        .zip(&plans)
                        .map(|(id, p)| match envs.get_mut(id) {
                            None => error(&format!("no level {id} open"), "unparsed"),
                            Some(env) => {
                                let n = env.next(p);
                                let allowed: Vec<String> = n.allowed.iter().map(|t| dataset::json_str(t)).collect();
                                format!(
                                    "{{\"allowed\":[{}],\"end\":{},\"dead\":{},\"whole\":{}}}",
                                    allowed.join(","),
                                    n.end,
                                    n.dead,
                                    n.whole
                                )
                            }
                        })
                        .collect();
                    // All the open levels together: past this, each drops its older half.
                    if envs.values().map(Env::cached).sum::<usize>() > 4 * glados::env::MAX_CACHED {
                        envs.values_mut().for_each(|e| e.forget(e.cached() / 2));
                    }
                    format!("{{\"results\":[{}]}}", results.join(","))
                }
                _ => error("next needs \"ids\" and \"plans\", as many of each", "unparsed"),
            },
            Some("close") => {
                for id in json_array(&line, "ids").unwrap_or_default() {
                    envs.remove(&id);
                }
                "{\"ok\":true}".to_string()
            }
            _ => error("\"op\" is open, next or close", "unparsed"),
        };
        writeln!(out, "{answer}")?;
        out.flush()?;
    }
    Ok(())
}

fn run_grammar(args: &[String]) -> Result<()> {
    use glados::grammar::Grammar;
    use glados::mutate::{json_array, json_field};
    use std::io::BufRead;
    let mut check = None;
    let mut rest = Vec::new();
    for (flag, value) in flags(args, &mut rest)? {
        match flag {
            "--check" => check = Some(value),
            _ => return Err(format!("unknown flag {flag}
{USAGE}").into()),
        }
    }
    if !rest.is_empty() {
        return Err(format!("grammar takes no arguments
{USAGE}").into());
    }
    let mut grammar = Grammar::new();
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    if let Some(file) = check {
        for line in std::io::BufReader::new(std::fs::File::open(file)?).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let toks = json_field(&line, "tokens").unwrap_or_default();
            match grammar.check(&toks.split_whitespace().collect::<Vec<_>>()) {
                Ok(()) => writeln!(out, "{{\"ok\":true}}")?,
                Err(e) => writeln!(out, "{{\"ok\":false,\"error\":{}}}", dataset::json_str(&e))?,
            }
        }
        out.flush()?;
        return Ok(());
    }
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let results: Vec<String> = json_array(&line, "levels")
            .unwrap_or_default()
            .iter()
            .map(|level| match grammar.next(&level.split_whitespace().collect::<Vec<_>>()) {
                Ok(n) => {
                    let allowed: Vec<String> = n.allowed.iter().map(|t| dataset::json_str(t)).collect();
                    format!("{{\"allowed\":[{}],\"end\":{}}}", allowed.join(","), n.end)
                }
                Err(e) => format!("{{\"error\":{}}}", dataset::json_str(&e)),
            })
            .collect();
        writeln!(out, "{{\"results\":[{}]}}", results.join(","))?;
        out.flush()?;
    }
    Ok(())
}

fn run_check(args: &[String]) -> Result<()> {
    use glados::mutate::{json_array, json_field};
    let mut mode = Mode::Conservative;
    let mut route_dir = None;
    let mut rest = Vec::new();
    for (flag, value) in flags(args, &mut rest)? {
        match flag {
            "--mode" => {
                mode = match value {
                    "conservative" => Mode::Conservative,
                    "liberal" => Mode::Liberal,
                    _ => return Err(format!("--mode: conservative or liberal, not {value}").into()),
                }
            }
            "--route" => route_dir = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown flag {flag}\n{USAGE}").into()),
        }
    }
    let lines = cli::input_lines(&rest, "check", USAGE)?;
    if let Some(dir) = &route_dir {
        std::fs::create_dir_all(dir)?;
    }
    let want_route = route_dir.is_some();
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    // A line's verdicts as a JSON `results` array (after "line"), and the level with its
    // route's lines if one of its plans solved and --route was given.
    type Done = (String, Option<(Puzzle, Vec<String>)>);
    let one = |line: &str| -> Done {
        // The level itself can't be checked (or read), so every plan of the line says so.
        let fail = |n: usize, verdict: &str, why: &str| {
            let one = format!(
                "{{\"verdict\":\"{verdict}\",\"problem\":{}}}",
                dataset::json_str(why)
            );
            (format!("\"results\":[{}]", vec![one; n].join(",")), None)
        };
        // `plans`: a list of plans for one level, checked against one model. `plan`: one.
        let plans = match json_array(line, "plans") {
            Some(p) if !p.is_empty() => p,
            _ => match json_field(line, "plan") {
                Some(p) => vec![p],
                None => return fail(1, "unparsed", "the line needs \"plan\" or \"plans\""),
            },
        };
        let n = plans.len();
        let Some(toks) = json_field(line, "tokens") else {
            return fail(n, "unparsed", "the line needs \"tokens\"");
        };
        let min_len: Option<usize> = json_field(line, "min_len").and_then(|v| v.parse().ok());
        let puzzle = match tokens::decode(&toks) {
            Ok(p) => p,
            Err(e) => return fail(n, "unparsed", &format!("level: {e}")),
        };
        let level = Level::from_puzzle(&puzzle);
        let mut checker = match plan::Checker::new(&level, mode) {
            Ok(m) => m,
            Err(problems) => {
                let invalid = problems.iter().any(|p| matches!(p, Problem::Invalid(_)));
                let verdict = if invalid { "invalid" } else { "unsupported" };
                return fail(n, verdict, &format!("{problems:?}"));
            }
        };
        let mut routed = None;
        let mut results = Vec::with_capacity(n);
        for plan_text in &plans {
            let steps = match plan::decode(plan_text) {
                Ok(s) => s,
                Err(e) => {
                    results.push(format!(
                        "{{\"verdict\":\"unparsed\",\"problem\":{}}}",
                        dataset::json_str(&format!("plan: {e}"))
                    ));
                    continue;
                }
            };
            let (c, used) = checker.check(&steps);
            // `steps_ok` is measured against `steps.len()`, so a plan that fits whole and
            // still doesn't reach the goal is unfinished, budget or not.
            let verdict = if c.solved {
                "solved"
            } else if c.steps_ok == steps.len() {
                "unfinished"
            } else if c.exhausted {
                "unknown"
            } else {
                "illegal"
            };
            let mut json = format!(
                "\"verdict\":\"{verdict}\",\"steps\":{},\"steps_ok\":{}",
                steps.len(),
                c.steps_ok
            );
            if c.solved {
                json.push_str(&format!(",\"len\":{}", c.plan.len()));
                if let Some(m) = min_len {
                    json.push_str(&format!(",\"optimal\":{}", c.plan.len() <= m));
                }
            } else if !c.exhausted {
                if let Some(s) = steps.get(c.steps_ok) {
                    json.push_str(&format!(",\"fail_verb\":\"{}\"", plan::step_verb(s)));
                }
            }
            if let Some(why) = &c.problem {
                json.push_str(&format!(",\"problem\":{}", dataset::json_str(why)));
            }
            if c.solved && want_route && routed.is_none() {
                match route::route_plan(&level, used, &c.plan) {
                    Ok(r) => {
                        let lines = r.json_lines(puzzle.title(), level.size);
                        routed = Some((puzzle.clone(), lines));
                    }
                    Err(e) => json.push_str(&format!(",\"route_error\":{}", dataset::json_str(&e))),
                }
            }
            results.push(format!("{{{json}}}"));
        }
        (format!("\"results\":[{}]", results.join(",")), routed)
    };
    dataset::par_map_ordered(
        lines,
        cli::threads(),
        |(i, line)| line.map(|l| (i, one(l.trim()))),
        |done: std::io::Result<(usize, Done)>| -> Result<()> {
            let (i, (json, routed)) = done?;
            writeln!(out, "{{\"line\":{i},{json}}}")?;
            if let (Some(dir), Some((mut p, lines))) = (&route_dir, routed) {
                p.set_title(&format!("glados plan {i}"));
                std::fs::write(dir.join(format!("{i:05}.p2c")), p.to_p2c())?;
                std::fs::write(dir.join(format!("{i:05}.route.jsonl")), lines.join("\n") + "\n")?;
            }
            Ok(())
        },
    )?;
    out.flush()?;
    Ok(())
}
