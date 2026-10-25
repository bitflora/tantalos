use std::path::{Path, PathBuf};
use std::process::ExitCode;

use glados::evolve::{self, Candidate, Config};
use glados::level::Level;
use glados::metrics::{assess, Assessment, Metrics};
use glados::p2c::Puzzle;
use glados::regions::RegionGraph;
use glados::solver::{judge, Judged, Judgement, Mode, Model, Problem, DEFAULT_MAX_STATES};
use glados::{dataset, fixtures, portals, render, route, tokens};

const USAGE: &str = "usage:
  glados info <file.p2c>...     summary, region graph and ASCII map
  glados solve <file.p2c>...    solvable / unsolvable / uncertain, shortest plan, metrics
  glados evolve [--seed N] [--pop N] [--gens N] [--budget N] [--min-area N] [--keep N] [--out DIR]
                                search for interesting levels; writes the best + report.md
                                (default out/evolve-<seed>)
  glados roundtrip <file.p2c>...  check parse -> write reproduces the file byte for byte
  glados fixtures [dir]         write the synthetic test chambers (default fixtures/synthetic)
  glados tokens <file.p2c>...   the level as a token sequence (format: src/tokens.rs)
  glados vocab                  every token the format uses, one per line
  glados dataset [--seed N] [--count N] [--budget N] [--fresh P] [--out FILE]
                 [--portals-required true|false] [--max-unsolved P]
                                labelled token sequences as JSONL, and vocab.txt beside it
                                (default data/levels-<seed>.jsonl). --portals-required true:
                                drop levels solvable without portals; --max-unsolved: most
                                share of levels that aren't solvable
  glados score [--budget N] [--p2c DIR] [--p2c-all DIR] [--editor-rules on|off] [--plan] [--plan-only] [--plans K] [FILE]
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
                                different shortest plans, \"plan\" first
  glados check [--mode conservative|liberal] [--route DIR] [FILE]
                                check plans: JSONL with \"tokens\" (a level) and \"plan\" (plan
                                tokens) or \"plans\" (a list of them, all against one model),
                                optionally \"min_len\", from FILE or stdin; one JSON line per
                                input, with a \"results\" list of one verdict per plan, in order
                                (solved, illegal, unfinished, unknown, unparsed, invalid,
                                unsupported). --route DIR: NNNNN.p2c and NNNNN.route.jsonl (as
                                glados route writes) for the first solved plan of a line
  glados mutate [--seed N] [--per-level N] [--tries N] [--max-tokens N] [--budget N]
                [--give-up N] [--threads N] [--out FILE] [FILE]
                                1-3 random edits to each level of FILE (JSONL with \"tokens\",
                                or stdin), scored with editor rules off; writes the mutants
                                that need portals to solve, fit --max-tokens (default 1024)
                                and differ from their parent, as JSONL with \"mutated_from\"
                                (the parent's workshop_id, or its line) and \"edits\".
                                --give-up N: drop a level with no mutant after N tries
  glados editor <file.p2c|dir>...
                                items the Puzzle Maker would mark invalid, one JSON line per file
  glados route [--plan TOKENS] <file.p2c>
                                the conservative plan made concrete for playing it in the game:
                                where to stand, look, fire and walk (JSON lines, src/route.rs).
                                --plan: route that plan instead (plan tokens, src/plan.rs); one
                                that doesn't solve the level is routed as far as its steps fit,
                                then an \"attempt\" leg tries the step that fits no move, and
                                the head line's \"net\" says what the check found
  glados view [--roots DIR]... [--port N] [--no-open] [--list]
                                a browser app for saved solutions: scans the roots for
                                <name>.route.jsonl with <name>.p2c beside it, serves them and
                                draws the level in 2.5D with the plan playing back. --list just
                                prints what it found. (default roots: fixtures, out, runs)
  glados workbench [--all FILE] [--train FILE] [--test FILE] [--ratings FILE]
                   [--items FILE] [--fetched FILE] [--annotations FILE]
                   [--port N] [--no-open] [--list]
                                a browser app for the downloaded Workshop levels: every field
                                we have, the solver's verdict, the Workshop rating and the
                                level in 2.5D. Rate, note, stage into the Puzzle Maker, or
                                exclude from test data; saved in
                                train/workshop-annotations.jsonl.";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("info") if args.len() > 1 => args[1..].iter().try_for_each(|f| info(Path::new(f))),
        Some("solve") if args.len() > 1 => args[1..].iter().try_for_each(|f| solve(Path::new(f))),
        Some("evolve") => run_evolve(&args[1..]),
        Some("tokens") if args.len() > 1 => args[1..]
            .iter()
            .try_for_each(|f| print_tokens(Path::new(f))),
        Some("vocab") => {
            println!("{}", tokens::vocab().join("\n"));
            Ok(())
        }
        Some("dataset") => run_dataset(&args[1..]),
        Some("score") => run_score(&args[1..]),
        Some("check") => run_check(&args[1..]),
        Some("mutate") => run_mutate(&args[1..]),
        Some("editor") if args.len() > 1 => editor_check(&args[1..]),
        Some("route") => run_route(&args[1..]),
        Some("view") => run_view(&args[1..]),
        Some("workbench") => glados::workbench::run(&args[1..]).map_err(Into::into),
        Some("roundtrip") if args.len() > 1 => {
            args[1..].iter().try_for_each(|f| roundtrip(Path::new(f)))
        }
        Some("fixtures") => write_fixtures(
            &args
                .get(1)
                .map_or("fixtures/synthetic".into(), PathBuf::from),
        ),
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn load(path: &Path) -> Result<(String, Puzzle)> {
    let src = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let puzzle = Puzzle::parse(&src).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok((src, puzzle))
}

fn info(path: &Path) -> Result<()> {
    let (_, puzzle) = load(path)?;
    let level = Level::from_puzzle(&puzzle);
    let graph = RegionGraph::build(&level);
    let region = |c| graph.region_of(c).map_or('-', RegionGraph::label);

    println!("{} — \"{}\"", path.display(), puzzle.title());
    println!(
        "  size {:?}, {} items, {} connections",
        level.size,
        puzzle.items.len(),
        puzzle.connections.len()
    );
    for (name, door) in [("entry", level.entry()), ("exit", level.exit())] {
        match door {
            Some(d) => println!("  {name} at {:?} (region {})", d.cell, region(d.cell)),
            None => println!("  {name}: none"),
        }
    }
    if let (Some(entry), Some(exit)) = (level.entry(), level.exit()) {
        if let (Some(a), Some(b)) = (graph.region_of(entry.cell), graph.region_of(exit.cell)) {
            let on_foot = graph.reachable_on_foot(a)[b];
            println!(
                "  exit reachable on foot from entry: {}",
                if on_foot { "yes" } else { "no" }
            );
        }
    }
    println!("  portalable surfaces: {}", level.portal_surfaces().len());
    for b in &level.barriers {
        println!(
            "  {:?} panel (item {}): {} faces",
            b.kind,
            b.item,
            b.faces.len()
        );
    }
    for h in &level.hazards {
        let state = if h.start_enabled { "on" } else { "off" };
        println!(
            "  {:?} (item {}, starts {state}): {} faces",
            h.kind,
            h.item,
            h.faces.len()
        );
    }
    for p in level.mount_problems() {
        println!("  WARNING: {p}");
    }
    println!("  regions:");
    print!("{}", render::describe_regions(&graph));
    println!("  {}", render::LEGEND);
    print!("{}", render::render(&level, &graph));
    println!();
    Ok(())
}

fn solve(path: &Path) -> Result<()> {
    let (_, puzzle) = load(path)?;
    let level = Level::from_puzzle(&puzzle);
    println!("{} — \"{}\"", path.display(), puzzle.title());
    let start = std::time::Instant::now();
    let judged = match judge(&level, DEFAULT_MAX_STATES) {
        Ok(j) => j,
        Err(problems) => {
            for p in problems {
                match p {
                    Problem::Invalid(m) => println!("  invalid: {m}"),
                    Problem::Unsupported(m) => println!("  unsupported: {m}"),
                }
            }
            println!();
            return Ok(());
        }
    };
    let Judged {
        judgement,
        conservative: mut model,
        liberal,
    } = judged;
    println!(
        "  {} regions, {} fizzlers/laser fields, {} portal face classes (conservative):",
        model.graph.regions.len(),
        model.hazards.len(),
        model.portals.classes.len()
    );
    if !model.beams.is_empty() {
        let spots: Vec<String> = model
            .beams
            .spots
            .iter()
            .enumerate()
            .map(|(k, s)| format!("{:?} {:?}", model.spot_cell(k), s.dir))
            .collect();
        println!(
            "  lasers: {} emitters, {} receivers, {} rays, spots: {}",
            model.beams.emitters.len(),
            model.beams.receivers.len(),
            model.beams.rays.len(),
            spots.join(", ")
        );
        if !model.beams.sweeps.is_empty() {
            let sweeps: Vec<String> = model
                .beams
                .sweeps
                .iter()
                .map(|s| format!("{:?} {:?} -> {:?}", s.from, s.dir, model.beams.funnel_cells[s.cell]))
                .collect();
            println!("  funnel sweeps: {}", sweeps.join(", "));
        }
    }
    if !model.devices.is_empty() {
        println!("  device bits: {:?}", model.devices.bits);
    }
    if !model.complete {
        println!("  (some devices or laser spots couldn't be modelled)");
    }
    for c in &model.portals.classes {
        let seen: String = (0..model.graph.regions.len())
            .filter(|&r| c.seen_from(r))
            .map(RegionGraph::label)
            .collect();
        let region = |r: Option<usize>| r.map_or('-', RegionGraph::label);
        let landing = c.flat.landings.first().map(|&(_, l)| l);
        let mut extra = String::new();
        if !c.flat.dives.is_empty() {
            extra += &format!(", dives {:?}", c.flat.dives);
        }
        if c.deployed.is_some() {
            extra += ", on a panel";
        }
        if !c.needs.is_always() {
            extra += ", flips";
        }
        if !c.ends.is_empty() || !c.routes.is_empty() {
            extra += &format!(", projections in {:?} out {:?}", c.ends, c.routes);
        }
        if c.beam_end || c.beam_out.is_some() {
            extra += ", laser";
        }
        println!(
            "    {}: {}; enter from {}, land in {}{}, seen from {}{}",
            c.id,
            portals::describe(c),
            region(c.flat.entry),
            region(landing.map(|l| l.region)),
            landing
                .and_then(|l| l.button)
                .map_or(String::new(), |b| format!(" on button {b}")),
            seen,
            extra
        );
    }
    match &judgement {
        Judgement::Solvable(_) => match assess(&mut model, DEFAULT_MAX_STATES, 0) {
            Assessment::Solved(m) => {
                println!("  SOLVABLE (conservative rules) in {} actions:", m.min_len);
                for (i, a) in m.plan.iter().enumerate() {
                    println!("    {}. {}", i + 1, model.describe(a));
                }
                for line in metric_lines(&m) {
                    println!("  {line}");
                }
            }
            _ => println!("  SOLVABLE, but metrics ran out of state budget"),
        },
        Judgement::Unsolvable => {
            println!("  UNSOLVABLE (even with liberal rules)");
            print_reach(&liberal);
        }
        Judgement::Uncertain(None) => {
            if liberal.complete {
                println!("  UNCERTAIN (state budget {DEFAULT_MAX_STATES} hit)");
            } else {
                println!("  UNCERTAIN (not everything could be modelled)");
            }
            print_reach(&liberal);
        }
        Judgement::Uncertain(Some(plan)) => {
            println!(
                "  UNCERTAIN: only liberal rules solve it, in {} actions:",
                plan.len()
            );
            for (i, a) in plan.iter().enumerate() {
                println!("    {}. {}", i + 1, liberal.describe(a));
            }
        }
    }
    println!("  judged and measured in {:.1?}", start.elapsed());
    println!();
    Ok(())
}

/// Where the player can get to under liberal rules, for seeing why a level isn't solved.
fn print_reach(model: &glados::solver::Model) {
    let space = model.explore(DEFAULT_MAX_STATES);
    let mut seen = vec![false; model.graph.regions.len()];
    for i in 0..space.len() {
        if let Some(r) = space.region(i) {
            seen[r] = true;
        }
    }
    let reached: String = (0..seen.len())
        .filter(|&r| seen[r])
        .map(RegionGraph::label)
        .collect();
    println!(
        "  liberal rules reach regions {reached} in {} states{}; the exit is in {}",
        space.len(),
        if space.complete { "" } else { " (budget hit)" },
        RegionGraph::label(model.exit_region)
    );
}

fn run_route(args: &[String]) -> Result<()> {
    let mut plan = None;
    let mut rest = Vec::new();
    for (flag, value) in flags(args, &mut rest)? {
        match flag {
            "--plan" => plan = Some(value),
            _ => return Err(format!("unknown flag {flag}
{USAGE}").into()),
        }
    }
    match rest.as_slice() {
        [file] => print_route(Path::new(file), plan),
        _ => Err(format!("route takes one .p2c file
{USAGE}").into()),
    }
}

/// `glados route FILE [--plan TOKENS]`: the solver's plan, or the given one, made concrete.
/// A given plan that doesn't solve the level is routed as far as it fits, then a try at the
/// step that fits no move; the head line says what its check found (`"net"`).
fn print_route(path: &Path, plan: Option<&str>) -> Result<()> {
    use glados::plan;
    let (_, puzzle) = load(path)?;
    let level = Level::from_puzzle(&puzzle);
    let at = |e: String| format!("{}: {e}", path.display());
    let r = match plan {
        None => route::route(&level, DEFAULT_MAX_STATES).map_err(at)?,
        Some(text) => {
            let model = Model::new(&level, Mode::Conservative).map_err(|p| at(format!("{p:?}")))?;
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

/// `glados view`: a browser app for the saved solutions under the roots.
fn run_view(args: &[String]) -> Result<()> {
    let mut roots: Vec<PathBuf> = Vec::new();
    let mut port = 0u16;
    let mut open = true;
    let mut list = false;
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--roots" | "--dir" => {
                let value = it.next().ok_or(format!("{flag} needs a value"))?;
                roots.push(PathBuf::from(value));
            }
            "--port" => {
                let value = it.next().ok_or("--port needs a value")?;
                port = value.parse().map_err(|_| format!("--port: not a number: {value}"))?;
            }
            "--no-open" => open = false,
            "--list" => list = true,
            _ => return Err(format!("unknown flag {flag}\n{USAGE}").into()),
        }
    }
    if roots.is_empty() {
        roots = ["fixtures", "out", "runs"].iter().map(PathBuf::from).collect();
    }
    roots.retain(|p| p.exists());
    if roots.is_empty() {
        return Err("none of the roots exist; pass --roots DIR".into());
    }
    if list {
        let found = glados::view::scan(&roots);
        for f in &found {
            println!(
                "{}\t{}x{}x{}\t{} legs\t{}",
                if f.group.is_empty() { "." } else { &f.group },
                f.size[0], f.size[1], f.size[2], f.legs,
                f.route.display()
            );
        }
        println!("{} solutions", found.len());
        return Ok(());
    }
    glados::view::serve(roots, port, open).map_err(Into::into)
}

fn metric_lines(m: &Metrics) -> Vec<String> {
    vec![
        format!("score {:.2}", m.score()),
        format!(
            "portal trips {}, shortest plans {}, random player wins {:.0}%",
            m.portal_trips,
            m.shortest_plans,
            100.0 * m.random_success
        ),
        format!(
            "{} states, {} dead ends; {} cubes ({} unnecessary); {} face classes ({} necessary)",
            m.states,
            m.dead_end_states,
            m.cubes,
            m.unnecessary_cubes,
            m.classes,
            m.necessary_classes
        ),
        format!(
            "necessary: {} buttons, {} pedestals, {} gels; load-bearing: {} fizzlers, {} laser fields, {} turrets",
            m.necessary_buttons,
            m.necessary_pedestals,
            m.necessary_gels,
            m.load_bearing_fizzlers,
            m.load_bearing_lasers,
            m.load_bearing_turrets
        ),
        format!(
            "chain depth {:.1}; travel {:.1}, cube travel {:.1}, backtrack {:.1} voxels; mechanics: {}",
            m.chain_depth,
            m.travel,
            m.cube_travel,
            m.backtrack,
            m.mechanic_names().join(", ")
        ),
    ]
}

fn run_evolve(args: &[String]) -> Result<()> {
    let mut cfg = Config::default();
    let mut keep = 5;
    let mut out = None;
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let value = it.next().ok_or(format!("{flag} needs a value"))?;
        let num = || -> Result<usize> {
            value
                .parse()
                .map_err(|_| format!("{flag}: not a number: {value}").into())
        };
        match flag.as_str() {
            "--seed" => cfg.seed = num()? as u64,
            "--pop" => cfg.population = num()?.max(2),
            "--gens" => cfg.generations = num()?,
            "--budget" => cfg.budget = num()?,
            "--min-area" => cfg.min_area = num()? as i32,
            "--keep" => keep = num()?,
            "--out" => out = Some(PathBuf::from(value)),
            _ => {
                return Err(format!(
                    "unknown flag {flag}
{USAGE}"
                )
                .into())
            }
        }
    }
    let out = out.unwrap_or_else(|| PathBuf::from(format!("out/evolve-{}", cfg.seed)));
    println!(
        "evolve: seed {}, population {}, {} generations, budget {}, {} threads",
        cfg.seed, cfg.population, cfg.generations, cfg.budget, cfg.threads
    );
    let start = std::time::Instant::now();
    let pop = evolve::evolve(&cfg, |s, _| {
        println!(
            "  gen {:>3}: {:>4} evaluated, {:>4} solved; best {:>6.2}, mean {:>6.2}",
            s.generation, s.evaluated, s.solved, s.best, s.mean
        );
    });
    println!("  took {:.1?}", start.elapsed());
    if pop.is_empty() {
        return Err("no solvable level found".into());
    }
    std::fs::create_dir_all(&out)?;
    let mut report = format!(
        "# glados evolve, seed {}

population {}, {} generations, budget {}
",
        cfg.seed, cfg.population, cfg.generations, cfg.budget
    );
    for (i, c) in pop.iter().take(keep).enumerate() {
        let name = format!("{:02}.p2c", i + 1);
        std::fs::write(out.join(&name), c.puzzle.to_p2c())?;
        report.push_str(&describe_candidate(i + 1, &name, c));
    }
    std::fs::write(out.join("report.md"), &report)?;
    println!(
        "wrote {} levels and report.md to {}",
        keep.min(pop.len()),
        out.display()
    );
    Ok(())
}

fn describe_candidate(rank: usize, file: &str, c: &Candidate) -> String {
    let level = Level::from_puzzle(&c.puzzle);
    let model = Model::new(&level, Mode::Conservative).expect("candidates are valid");
    let mut s = format!(
        "
## {rank}. `{file}`: score {:.2} (born in generation {})

size {:?}, ",
        c.score, c.born, level.size
    );
    s.push_str(&metric_lines(&c.metrics)[1..].join("; "));
    s.push_str(
        "

Plan:
",
    );
    for (i, a) in c.metrics.plan.iter().enumerate() {
        s.push_str(&format!(
            "{}. {}
",
            i + 1,
            model.describe(a)
        ));
    }
    s.push_str(&format!(
        "
Regions:
```
{}```

```
{}
{}```
",
        render::describe_regions(&model.graph),
        render::LEGEND,
        render::render(&level, &model.graph)
    ));
    s
}

fn roundtrip(path: &Path) -> Result<()> {
    let (src, puzzle) = load(path)?;
    let out = puzzle.to_p2c();
    if out == src {
        println!("{}: OK", path.display());
        return Ok(());
    }
    let line = src
        .lines()
        .zip(out.lines())
        .position(|(a, b)| a != b)
        .unwrap_or(src.lines().count().min(out.lines().count()));
    Err(format!(
        "{}: output differs from input at line {}",
        path.display(),
        line + 1
    )
    .into())
}

fn print_tokens(path: &Path) -> Result<()> {
    let (_, puzzle) = load(path)?;
    match tokens::encode(&puzzle) {
        Ok(t) => println!("{}: {} tokens\n{}\n", path.display(), t.len(), t.join(" ")),
        Err(e) => println!("{}: can't encode: {e}\n", path.display()),
    }
    Ok(())
}

/// `--flag value` pairs; `positional` collects anything else.
fn flags<'a>(args: &'a [String], positional: &mut Vec<&'a str>) -> Result<Vec<(&'a str, &'a str)>> {
    let mut out = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a.starts_with("--") {
            let value = it.next().ok_or(format!("{a} needs a value"))?;
            out.push((a.as_str(), value.as_str()));
        } else {
            positional.push(a);
        }
    }
    Ok(out)
}

fn parse<T: std::str::FromStr>(flag: &str, value: &str) -> Result<T> {
    value
        .parse()
        .map_err(|_| format!("{flag}: bad value {value}").into())
}

fn run_dataset(args: &[String]) -> Result<()> {
    use std::io::Write;
    let mut cfg = dataset::Config::default();
    let mut out = None;
    let mut rest = Vec::new();
    for (flag, value) in flags(args, &mut rest)? {
        match flag {
            "--seed" => cfg.seed = parse(flag, value)?,
            "--count" => cfg.count = parse(flag, value)?,
            "--budget" => cfg.budget = parse(flag, value)?,
            "--fresh" => cfg.fresh = parse(flag, value)?,
            "--portals-required" => cfg.portals_required = parse(flag, value)?,
            "--max-unsolved" => cfg.max_unsolved = parse(flag, value)?,
            "--out" => out = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown flag {flag}\n{USAGE}").into()),
        }
    }
    if !rest.is_empty() {
        return Err(format!("unexpected {rest:?}\n{USAGE}").into());
    }
    let out = out.unwrap_or_else(|| PathBuf::from(format!("data/levels-{}.jsonl", cfg.seed)));
    let dir = out.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join("vocab.txt"), tokens::vocab().join("\n") + "\n")?;
    let mut file = std::io::BufWriter::new(std::fs::File::create(&out)?);
    println!(
        "dataset: seed {}, {} levels, budget {}, {:.0}% fresh, portals required {}, at most {:.0}% unsolved, {} threads -> {}",
        cfg.seed,
        cfg.count,
        cfg.budget,
        100.0 * cfg.fresh,
        cfg.portals_required,
        100.0 * cfg.max_unsolved,
        cfg.threads,
        out.display()
    );
    let start = std::time::Instant::now();
    let mut counts: std::collections::BTreeMap<&str, usize> = Default::default();
    let (mut n, mut tokens_total, mut tokens_max) = (0, 0, 0);
    let mut io_error = None;
    dataset::generate(&cfg, |s| {
        if io_error.is_none() {
            io_error = writeln!(file, "{}", s.json()).err();
        }
        *counts.entry(s.label.verdict()).or_default() += 1;
        n += 1;
        tokens_total += s.tokens.len();
        tokens_max = tokens_max.max(s.tokens.len());
        if n % 1000 == 0 || n == cfg.count {
            println!(
                "  {n:>7} levels in {:>6.1?}: {counts:?}; tokens mean {}, max {tokens_max}",
                start.elapsed(),
                tokens_total / n
            );
        }
    });
    if let Some(e) = io_error {
        return Err(e.into());
    }
    file.flush()?;
    Ok(())
}

fn run_score(args: &[String]) -> Result<()> {
    use std::io::{BufRead, Write};
    let mut budget = 50_000;
    let mut p2c_dir = None;
    let mut all = false;
    let mut editor_rules = true;
    let mut with_plan = false;
    let mut plan_only = false;
    let mut k = 1;
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
    let input: Box<dyn BufRead + Send> = match rest.as_slice() {
        [] | ["-"] => Box::new(std::io::BufReader::new(std::io::stdin())),
        [file] => Box::new(std::io::BufReader::new(
            std::fs::File::open(file).map_err(|e| format!("{file}: {e}"))?,
        )),
        _ => return Err(format!("score takes one input file\n{USAGE}").into()),
    };
    let lines = input
        .lines()
        .enumerate()
        .map(|(i, l)| (i + 1, l))
        .filter(|(_, l)| !l.as_ref().is_ok_and(|l| l.trim().is_empty()));
    if let Some(dir) = &p2c_dir {
        std::fs::create_dir_all(dir)?;
    }
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    dataset::par_map_ordered(
        lines,
        threads,
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
                        let fields = match dataset::plan_only(&p, budget, editor_rules, k, i as u64) {
                            Ok((len, Ok(ts))) => {
                                let mut f = format!(
                                    "\"verdict\":\"solvable\",\"min_len\":{len},\"plan\":{}",
                                    dataset::json_str(&ts[0])
                                );
                                if k > 1 {
                                    let all: Vec<String> = ts.iter().map(|t| dataset::json_str(t)).collect();
                                    f += &format!(",\"plans\":[{}]", all.join(","));
                                }
                                f
                            }
                            Ok((len, Err(e))) => format!(
                                "\"verdict\":\"solvable\",\"min_len\":{len},\"plan_error\":{}",
                                dataset::json_str(&e)
                            ),
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

fn run_check(args: &[String]) -> Result<()> {
    use glados::mutate::{json_array, json_field};
    use glados::plan;
    use std::io::{BufRead, Write};
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
    let input: Box<dyn BufRead + Send> = match rest.as_slice() {
        [] | ["-"] => Box::new(std::io::BufReader::new(std::io::stdin())),
        [file] => Box::new(std::io::BufReader::new(
            std::fs::File::open(file).map_err(|e| format!("{file}: {e}"))?,
        )),
        _ => return Err(format!("check takes one input file\n{USAGE}").into()),
    };
    let lines = input
        .lines()
        .enumerate()
        .map(|(i, l)| (i + 1, l))
        .filter(|(_, l)| !l.as_ref().is_ok_and(|l| l.trim().is_empty()));
    if let Some(dir) = &route_dir {
        std::fs::create_dir_all(dir)?;
    }
    let want_route = route_dir.is_some();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
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
        let model = match Model::new(&level, mode) {
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
            let c = plan::check(&model, &steps);
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
                match route::route_plan(&level, &model, &c.plan) {
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
        threads,
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

fn run_mutate(args: &[String]) -> Result<()> {
    use glados::mutate;
    use std::io::{BufRead, Write};
    let mut cfg = mutate::Config::default();
    let mut out_path = None;
    let mut threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut rest = Vec::new();
    for (flag, value) in flags(args, &mut rest)? {
        match flag {
            "--seed" => cfg.seed = parse(flag, value)?,
            "--threads" => threads = parse(flag, value)?,
            "--per-level" => cfg.per_level = parse(flag, value)?,
            "--tries" => cfg.tries = parse(flag, value)?,
            "--max-tokens" => cfg.max_tokens = parse(flag, value)?,
            "--budget" => cfg.budget = parse(flag, value)?,
            "--give-up" => cfg.give_up = parse(flag, value)?,
            "--out" => out_path = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown flag {flag}
{USAGE}").into()),
        }
    }
    let input: Box<dyn BufRead + Send> = match rest.as_slice() {
        [] | ["-"] => Box::new(std::io::BufReader::new(std::io::stdin())),
        [file] => Box::new(std::io::BufReader::new(
            std::fs::File::open(file).map_err(|e| format!("{file}: {e}"))?,
        )),
        _ => return Err(format!("mutate takes one input file
{USAGE}").into()),
    };
    let mut out: Box<dyn Write> = match &out_path {
        Some(p) => Box::new(std::io::BufWriter::new(std::fs::File::create(p)?)),
        None => Box::new(std::io::BufWriter::new(std::io::stdout().lock())),
    };
    let lines = input
        .lines()
        .enumerate()
        .map(|(i, l)| (i + 1, l))
        .filter(|(_, l)| !l.as_ref().is_ok_and(|l| l.trim().is_empty()));
    let start = std::time::Instant::now();
    let mut stats: std::collections::BTreeMap<String, usize> = Default::default();
    let (mut parents, mut kept) = (0, 0);
    dataset::par_map_ordered(
        lines,
        threads,
        |(i, line)| -> std::io::Result<_> {
            let line = line?;
            let mut stats = Default::default();
            let parent = mutate::json_field(&line, "workshop_id").unwrap_or(format!("line {i}"));
            let Some(toks) = mutate::json_field(&line, "tokens") else {
                return Ok((parent, Vec::new(), stats));
            };
            let seed = cfg.seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let found = mutate::mutants(&toks, &cfg, seed, &mut stats);
            Ok((parent, found, stats))
        },
        |r| -> Result<()> {
            let (parent, found, s) = r?;
            for m in &found {
                writeln!(out, "{}", m.json(&parent))?;
            }
            for (k, v) in s {
                *stats.entry(k).or_default() += v;
            }
            parents += 1;
            kept += found.len();
            if parents % 50 == 0 {
                eprintln!("  {parents} levels, {kept} mutants in {:.1?}", start.elapsed());
            }
            Ok(())
        },
    )?;
    out.flush()?;
    eprintln!("mutate: {parents} levels -> {kept} mutants in {:.1?}", start.elapsed());
    for (k, v) in &stats {
        eprintln!("  {k}: {v}");
    }
    Ok(())
}

fn editor_check(args: &[String]) -> Result<()> {
    let mut files = Vec::new();
    for a in args {
        let p = PathBuf::from(a);
        if p.is_dir() {
            let mut found: Vec<PathBuf> = std::fs::read_dir(&p)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|f| f.extension().is_some_and(|x| x == "p2c"))
                .collect();
            found.sort();
            files.extend(found);
        } else {
            files.push(p);
        }
    }
    for f in files {
        let (_, puzzle) = load(&f)?;
        let flags = glados::editor::editor_flags(&Level::from_puzzle(&puzzle));
        let items: Vec<String> = flags
            .iter()
            .map(|fl| {
                format!(
                    "{{\"name\":\"{}\",\"pos\":[{},{},{}],\"rules\":[{}]}}",
                    fl.name,
                    fl.pos[0],
                    fl.pos[1],
                    fl.pos[2],
                    fl.rules
                        .iter()
                        .map(|r| format!("\"{r}\""))
                        .collect::<Vec<_>>()
                        .join(",")
                )
            })
            .collect();
        println!(
            "{{\"file\":{},\"flags\":[{}]}}",
            dataset::json_str(&f.display().to_string()),
            items.join(",")
        );
    }
    Ok(())
}

fn write_fixtures(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    for (name, puzzle) in fixtures::all() {
        let path = dir.join(format!("{name}.p2c"));
        std::fs::write(&path, puzzle.to_p2c())?;
        println!("wrote {}", path.display());
    }
    Ok(())
}
