use std::path::{Path, PathBuf};
use std::process::ExitCode;

use glados::level::Level;
use glados::p2c::Puzzle;
use glados::regions::RegionGraph;
use glados::{fixtures, render};

const USAGE: &str = "usage:
  glados info <file.p2c>...     summary, region graph and ASCII map
  glados roundtrip <file.p2c>...  check parse -> write reproduces the file byte for byte
  glados fixtures [dir]         write the synthetic test chambers (default fixtures/synthetic)";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("info") if args.len() > 1 => args[1..].iter().try_for_each(|f| info(Path::new(f))),
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

fn write_fixtures(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    for (name, puzzle) in fixtures::all() {
        let path = dir.join(format!("{name}.p2c"));
        std::fs::write(&path, puzzle.to_p2c())?;
        println!("wrote {}", path.display());
    }
    Ok(())
}
