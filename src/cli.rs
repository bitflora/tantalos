//! What the command-line binaries (`src/bin/glados-*.rs`) share: dispatch, flag parsing,
//! reading .p2c files and JSONL input, and printing a level's metrics.

use std::io::BufRead;
use std::path::Path;
use std::process::ExitCode;

use crate::metrics::Metrics;
use crate::p2c::Puzzle;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// A binary's `main`: runs `command` on the arguments; `None` (an unknown command or missing
/// arguments) prints `usage` and exits 2, an error prints it and exits 1.
pub fn main(usage: &str, command: impl FnOnce(&[String]) -> Option<Result<()>>) -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match command(&args) {
        None => {
            eprintln!("{usage}");
            ExitCode::from(2)
        }
        Some(Ok(())) => ExitCode::SUCCESS,
        Some(Err(e)) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

pub fn load(path: &Path) -> Result<(String, Puzzle)> {
    let src = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let puzzle = Puzzle::parse(&src).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok((src, puzzle))
}

/// `--flag value` pairs; `positional` collects anything else.
pub fn flags<'a>(args: &'a [String], positional: &mut Vec<&'a str>) -> Result<Vec<(&'a str, &'a str)>> {
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

pub fn parse<T: std::str::FromStr>(flag: &str, value: &str) -> Result<T> {
    value
        .parse()
        .map_err(|_| format!("{flag}: bad value {value}").into())
}

/// The non-blank lines of FILE (or stdin, for none or `-`), numbered from 1; `command` names
/// the command in the error for more than one file.
pub fn input_lines(
    rest: &[&str],
    command: &str,
    usage: &str,
) -> Result<impl Iterator<Item = (usize, std::io::Result<String>)> + Send> {
    let input: Box<dyn BufRead + Send> = match rest {
        [] | ["-"] => Box::new(std::io::BufReader::new(std::io::stdin())),
        [file] => Box::new(std::io::BufReader::new(
            std::fs::File::open(file).map_err(|e| format!("{file}: {e}"))?,
        )),
        _ => return Err(format!("{command} takes one input file\n{usage}").into()),
    };
    Ok(input
        .lines()
        .enumerate()
        .map(|(i, l)| (i + 1, l))
        .filter(|(_, l)| !l.as_ref().is_ok_and(|l| l.trim().is_empty())))
}

pub fn threads() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get())
}

pub fn metric_lines(m: &Metrics) -> Vec<String> {
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
