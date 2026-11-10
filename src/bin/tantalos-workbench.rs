//! `tantalos-workbench`: a browser app for triaging the downloaded Workshop levels
//! (`src/workbench.rs`, which has the flags and usage).

use std::process::ExitCode;

fn main() -> ExitCode {
    tantalos::cli::main("", |args| Some(tantalos::workbench::run(args).map_err(Into::into)))
}
