//! `glados-data`: making training data, generated levels and mutants of existing ones.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use glados::cli::{self, flags, parse, Result};
use glados::{augment, dataset, editor, level, mutate, tokens};

const USAGE: &str = "usage:
  glados-data dataset [--seed N] [--count N] [--budget N] [--fresh P] [--out FILE]
                      [--portals-required true|false] [--max-unsolved P]
                                labelled token sequences as JSONL, and vocab.txt beside it
                                (default data/levels-<seed>.jsonl). --portals-required true:
                                drop levels solvable without portals; --max-unsolved: most
                                share of levels that aren't solvable
  glados-data mutate [--seed N] [--per-level N] [--tries N] [--max-tokens N] [--budget N]
                     [--give-up N] [--threads N] [--out FILE] [FILE]
                                1-3 random edits to each level of FILE (JSONL with \"tokens\",
                                or stdin), scored with editor rules off; writes the mutants
                                that need portals to solve, fit --max-tokens (default 1024)
                                and differ from their parent, as JSONL with \"mutated_from\"
                                (the parent's workshop_id, or its line) and \"edits\".
                                --give-up N: drop a level with no mutant after N tries
  glados-data augment [--syms r0,r90,...|all] [--check] [--budget N] [--editor-rules on|off]
                      [--threads N] [--out FILE] [FILE]
                                the 8 mirrored/turned versions of each level of FILE (JSONL
                                with \"tokens\", or stdin; src/augment.rs), one JSON line each:
                                \"line\" (1-based), \"variant\" (r0 = the level itself, re-encoded),
                                \"tokens\", \"n_tokens\", or \"error\". A version with the same tokens
                                as an earlier one is left out. --check: label the level and each
                                version (default budget 50000, editor rules off) and add both
                                labels (\"a\", \"b\") and editor flag rules (\"editor_a\", \"editor_b\")";

fn main() -> ExitCode {
    cli::main(USAGE, |args| match args.first().map(String::as_str) {
        Some("dataset") => Some(run_dataset(&args[1..])),
        Some("mutate") => Some(run_mutate(&args[1..])),
        Some("augment") => Some(run_augment(&args[1..])),
        _ => None,
    })
}

fn run_dataset(args: &[String]) -> Result<()> {
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

fn run_mutate(args: &[String]) -> Result<()> {
    let mut cfg = mutate::Config::default();
    let mut out_path = None;
    let mut threads = cli::threads();
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
            _ => return Err(format!("unknown flag {flag}\n{USAGE}").into()),
        }
    }
    let lines = cli::input_lines(&rest, "mutate", USAGE)?;
    let mut out: Box<dyn Write> = match &out_path {
        Some(p) => Box::new(std::io::BufWriter::new(std::fs::File::create(p)?)),
        None => Box::new(std::io::BufWriter::new(std::io::stdout().lock())),
    };
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

fn run_augment(args: &[String]) -> Result<()> {
    let mut syms = augment::Sym::ALL.to_vec();
    let (mut check, mut budget, mut editor_rules) = (false, 50_000, false);
    let mut out_path = None;
    let mut threads = cli::threads();
    let mut rest = Vec::new();
    let mut positional = Vec::new();
    for a in args {
        if a == "--check" {
            check = true;
        } else {
            rest.push(a.clone());
        }
    }
    for (flag, value) in flags(&rest, &mut positional)? {
        match flag {
            "--syms" if value != "all" => {
                syms = value
                    .split(',')
                    .map(|s| augment::Sym::parse(s).ok_or(format!("--syms: no symmetry {s:?}")))
                    .collect::<std::result::Result<_, _>>()?;
            }
            "--syms" => {}
            "--budget" => budget = parse(flag, value)?,
            "--editor-rules" => editor_rules = value == "on",
            "--threads" => threads = parse(flag, value)?,
            "--out" => out_path = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown flag {flag}\n{USAGE}").into()),
        }
    }
    let lines = cli::input_lines(&positional, "augment", USAGE)?;
    let mut out: Box<dyn Write> = match &out_path {
        Some(p) => Box::new(std::io::BufWriter::new(std::fs::File::create(p)?)),
        None => Box::new(std::io::BufWriter::new(std::io::stdout().lock())),
    };
    let label = |p: &glados::p2c::Puzzle| {
        let flags: Vec<String> = editor::editor_flags(&level::Level::from_puzzle(p))
            .into_iter()
            .flat_map(|f| { let name = f.name; f.rules.into_iter().map(move |r| format!("{name}:{r}")) })
            .collect();
        let flags: Vec<String> = flags.iter().map(|f| dataset::json_str(f)).collect();
        let l = dataset::label_with(p, budget, 1, editor_rules);
        (format!("{{{}}}", l.json_fields()), format!("[{}]", flags.join(",")))
    };
    let start = std::time::Instant::now();
    let (mut levels, mut written, mut errors, mut same) = (0, 0, 0, 0);
    dataset::par_map_ordered(
        lines,
        threads,
        |(i, line)| -> std::io::Result<Vec<String>> {
            let line = line?;
            let Some(toks) = mutate::json_field(&line, "tokens") else {
                return Ok(vec![format!("{{\"line\":{i},\"error\":\"no tokens\"}}")]);
            };
            let original = match tokens::decode(&toks) {
                Ok(p) => p,
                Err(e) => return Ok(vec![format!("{{\"line\":{i},\"error\":{}}}", dataset::json_str(&e))]),
            };
            let base = check.then(|| label(&original));
            let mut seen = std::collections::HashSet::new();
            let mut rows = Vec::new();
            for &sym in &syms {
                let head = format!("{{\"line\":{i},\"variant\":\"{}\"", sym.name());
                match augment::variant(&original, sym) {
                    Err(e) => rows.push(format!("{head},\"error\":{}}}", dataset::json_str(&e))),
                    Ok((p, t)) => {
                        let text = t.join(" ");
                        if !seen.insert(text.clone()) {
                            continue;
                        }
                        let mut row = format!(
                            "{head},\"tokens\":{},\"n_tokens\":{}",
                            dataset::json_str(&text),
                            t.len()
                        );
                        if let Some((a, ea)) = &base {
                            let (b, eb) = label(&p);
                            row.push_str(&format!(",\"a\":{a},\"b\":{b},\"editor_a\":{ea},\"editor_b\":{eb}"));
                        }
                        row.push('}');
                        rows.push(row);
                    }
                }
            }
            Ok(rows)
        },
        |r| -> Result<()> {
            let rows = r?;
            levels += 1;
            for row in &rows {
                writeln!(out, "{row}")?;
                if row.contains("\"error\":") {
                    errors += 1;
                } else {
                    written += 1;
                }
            }
            same += syms.len().saturating_sub(rows.len());
            if levels % 500 == 0 {
                eprintln!("  {levels} levels, {written} versions in {:.1?}", start.elapsed());
            }
            Ok(())
        },
    )?;
    out.flush()?;
    eprintln!(
        "augment: {levels} levels -> {written} versions, {same} left out as duplicates, {errors} errors, in {:.1?}",
        start.elapsed()
    );
    Ok(())
}
