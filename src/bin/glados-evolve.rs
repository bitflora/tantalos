//! `glados-evolve`: evolutionary search for interesting levels (`src/evolve.rs`).

use std::path::PathBuf;
use std::process::ExitCode;

use glados::cli::{self, metric_lines, Result};
use glados::evolve::{self, Candidate, Config};
use glados::level::Level;
use glados::render;
use glados::solver::{Mode, Model};

const USAGE: &str = "usage:
  glados-evolve [--seed N] [--pop N] [--gens N] [--budget N] [--min-area N] [--keep N] [--out DIR]
                                search for interesting levels; writes the best + report.md
                                (default out/evolve-<seed>)";

fn main() -> ExitCode {
    cli::main(USAGE, |args| Some(run(args)))
}

fn run(args: &[String]) -> Result<()> {
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
            _ => return Err(format!("unknown flag {flag}\n{USAGE}").into()),
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
        "# glados evolve, seed {}\n\npopulation {}, {} generations, budget {}\n",
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
        "\n## {rank}. `{file}`: score {:.2} (born in generation {})\n\nsize {:?}, ",
        c.score, c.born, level.size
    );
    s.push_str(&metric_lines(&c.metrics)[1..].join("; "));
    s.push_str("\n\nPlan:\n");
    for (i, a) in c.metrics.plan.iter().enumerate() {
        s.push_str(&format!("{}. {}\n", i + 1, model.describe(a)));
    }
    s.push_str(&format!(
        "\nRegions:\n```\n{}```\n\n```\n{}\n{}```\n",
        render::describe_regions(&model.graph),
        render::LEGEND,
        render::render(&level, &model.graph)
    ));
    s
}
