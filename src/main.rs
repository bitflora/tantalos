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
                                labelled token sequences as JSONL, and vocab.txt beside it
                                (default data/levels-<seed>.jsonl)
  glados score [--budget N] [--p2c DIR] [--p2c-all DIR] [--editor-rules on|off] [FILE]
                                decode and solve one token sequence per line of FILE (or
                                stdin); JSONL to stdout. --p2c writes the solvable ones,
                                --p2c-all every one that parsed. --editor-rules off: don't
                                call levels invalid for our model of the editor's rules
                                (for levels the game's editor accepted, e.g. the Workshop)
  glados editor <file.p2c|dir>...
                                items the Puzzle Maker would mark invalid, one JSON line per file
  glados route <file.p2c>       the conservative plan made concrete for playing it in the game:
                                where to stand, look, fire and walk (JSON lines, src/route.rs)";

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
        Some("editor") if args.len() > 1 => editor_check(&args[1..]),
        Some("route") if args.len() == 2 => print_route(Path::new(&args[1])),
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
    for c in &model.portals.classes {
        let seen: String = (0..model.graph.regions.len())
            .filter(|&r| c.seen_from(r))
            .map(RegionGraph::label)
            .collect();
        let region = |r: Option<usize>| r.map_or('-', RegionGraph::label);
        println!(
            "    {}: {}; enter from {}, land in {}{}, seen from {}",
            c.id,
            portals::describe(c),
            region(c.entry),
            region(c.landing.map(|l| l.region)),
            c.landing
                .and_then(|l| l.button)
                .map_or(String::new(), |b| format!(" on button {b}")),
            seen
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
        Judgement::Unsolvable => println!("  UNSOLVABLE (even with liberal rules)"),
        Judgement::Uncertain(None) => {
            println!("  UNCERTAIN (state budget {DEFAULT_MAX_STATES} hit)")
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

fn print_route(path: &Path) -> Result<()> {
    let (_, puzzle) = load(path)?;
    let level = Level::from_puzzle(&puzzle);
    let r = route::route(&level, DEFAULT_MAX_STATES).map_err(|e| format!("{}: {e}", path.display()))?;
    for line in r.json_lines(puzzle.title(), level.size) {
        println!("{line}");
    }
    Ok(())
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
        "dataset: seed {}, {} levels, budget {}, {:.0}% fresh, {} threads -> {}",
        cfg.seed,
        cfg.count,
        cfg.budget,
        100.0 * cfg.fresh,
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
    let mut rest = Vec::new();
    for (flag, value) in flags(args, &mut rest)? {
        match flag {
            "--budget" => budget = parse(flag, value)?,
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
                    let tokens = encoded.map_or(String::new(), |t| {
                        format!("\"tokens\":{},", dataset::json_str(&t))
                    });
                    let label = dataset::label_with(&p, budget, i as u64, editor_rules);
                    let keep =
                        (all || matches!(label, dataset::Label::Solvable { .. })).then_some(p);
                    (
                        i,
                        format!(
                            "{{\"line\":{i},\"n_tokens\":{n},\"canonical\":{canonical},{tokens}{}}}",
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
