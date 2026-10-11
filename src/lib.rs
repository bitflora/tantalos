//! glados: tooling for generating and solving Portal 2 Puzzle Maker levels.
//!
//! Slice 1: `.p2c` I/O (`kv`, `p2c`), level semantics (`level`), walkable regions
//! (`regions`), ASCII maps (`render`), and code-built test chambers (`builder`, `fixtures`).
//! Slice 2: portal surfaces and face classes (`portals`), Tier-0 BFS solver (`solver`).
//! Slice 3: level metrics (`metrics`), random chamber genomes (`generator`), evolutionary
//! search (`evolve`).
//! Slice 5: levels as token sequences (`tokens`), labelled training data and scoring of model
//! output (`dataset`).
//! Turrets (`turrets`) and gel (`gel`): where turrets shoot, where paint lands, and gel moves.

pub mod builder;
pub mod dataset;
pub mod evolve;
pub mod fixtures;
pub mod gel;
pub mod generator;
pub mod geom;
pub mod kv;
pub mod level;
pub mod metrics;
pub mod p2c;
pub mod portals;
pub mod regions;
pub mod render;
pub mod rng;
pub mod solver;
pub mod tokens;
pub mod turrets;
