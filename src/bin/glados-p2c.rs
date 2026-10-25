//! `glados-p2c`: one-off tools for .p2c files: what a level is, whether it's solvable, its
//! tokens, what the editor would flag; and the synthetic test chambers.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use glados::cli::{self, load, metric_lines, Result};
use glados::level::Level;
use glados::metrics::{assess, Assessment};
use glados::regions::RegionGraph;
use glados::solver::{judge, Judged, Judgement, Problem, DEFAULT_MAX_STATES};
use glados::{dataset, fixtures, portals, render, tokens};

const USAGE: &str = "usage:
  glados-p2c info <file.p2c>...     summary, region graph and ASCII map
  glados-p2c solve <file.p2c>...    solvable / unsolvable / uncertain, shortest plan, metrics
  glados-p2c tokens <file.p2c>...   the level as a token sequence (format: src/tokens.rs)
  glados-p2c vocab                  every token the format uses, one per line
  glados-p2c editor <file.p2c|dir>...
                                    items the Puzzle Maker would mark invalid, one JSON line per file
  glados-p2c roundtrip <file.p2c>...  check parse -> write reproduces the file byte for byte
  glados-p2c fixtures [dir]         write the synthetic test chambers (default fixtures/synthetic)";

fn main() -> ExitCode {
    cli::main(USAGE, |args| {
        let files = || args[1..].iter().map(Path::new);
        Some(match args.first().map(String::as_str) {
            Some("info") if args.len() > 1 => files().try_for_each(info),
            Some("solve") if args.len() > 1 => files().try_for_each(solve),
            Some("tokens") if args.len() > 1 => files().try_for_each(print_tokens),
            Some("vocab") => {
                println!("{}", tokens::vocab().join("\n"));
                Ok(())
            }
            Some("editor") if args.len() > 1 => editor_check(&args[1..]),
            Some("roundtrip") if args.len() > 1 => files().try_for_each(roundtrip),
            Some("fixtures") => write_fixtures(
                &args
                    .get(1)
                    .map_or("fixtures/synthetic".into(), PathBuf::from),
            ),
            _ => return None,
        })
    })
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
