//! `glados-workbench`: a browser app for triaging the downloaded Workshop levels
//! (`src/workbench.rs`, which has the flags and usage).

use std::process::ExitCode;

fn main() -> ExitCode {
    glados::cli::main("", |args| Some(glados::workbench::run(args).map_err(Into::into)))
}
