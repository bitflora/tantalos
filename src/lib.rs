//! glados: tooling for generating and solving Portal 2 Puzzle Maker levels.
//!
//! Slice 1: `.p2c` I/O (`kv`, `p2c`), level semantics (`level`), walkable regions
//! (`regions`), ASCII maps (`render`), and code-built test chambers (`builder`, `fixtures`).

pub mod builder;
pub mod fixtures;
pub mod geom;
pub mod kv;
pub mod level;
pub mod p2c;
pub mod regions;
pub mod render;
