//! `tantalos-view`: a browser app for saved solutions (`src/view.rs`).

use std::path::PathBuf;
use std::process::ExitCode;

use tantalos::cli::{self, Result};

const USAGE: &str = "usage:
  tantalos-view [--roots DIR]... [--port N] [--no-open] [--list]
                                a browser app for saved solutions: scans the roots for
                                <name>.route.jsonl with <name>.p2c beside it, serves them and
                                draws the level in 2.5D with the plan playing back. --list just
                                prints what it found. (default roots: fixtures, out, runs)";

fn main() -> ExitCode {
    cli::main(USAGE, |args| Some(run(args)))
}

fn run(args: &[String]) -> Result<()> {
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
        let found = tantalos::view::scan(&roots);
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
    tantalos::view::serve(roots, port, open).map_err(Into::into)
}
