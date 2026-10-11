//! Training data for sequence models, and scoring of what they write.
//!
//! `generate` samples genomes (fresh random ones, and mutations of earlier solvable ones so
//! the data isn't dominated by unsolvable rooms), encodes each level as tokens and labels it
//! with the solver's verdict and metrics. `label` is also what scores model output.
//! Results depend only on the seed, not on the thread count.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use crate::generator::Genome;
use crate::level::Level;
use crate::metrics::{assess, Assessment, Metrics};
use crate::p2c::Puzzle;
use crate::rng::Rng;
use crate::solver::{judge, Judgement, Problem};
use crate::tokens;

/// What the solver says about a level.
#[derive(Clone, Debug, PartialEq)]
pub enum Label {
    /// Broken: items not mounted, no doors, bad wiring.
    Invalid(String),
    /// Uses something the solver doesn't model.
    Unsupported(String),
    Unsolvable,
    /// Only liberal rules solve it (in this many actions), or the budget ran out (`None`).
    Uncertain(Option<usize>),
    /// Conservative rules solve it in `min_len` actions. `metrics` is `None` if measuring
    /// ran out of budget.
    Solvable {
        min_len: usize,
        metrics: Option<Metrics>,
    },
}

impl Label {
    pub fn verdict(&self) -> &'static str {
        match self {
            Label::Invalid(_) => "invalid",
            Label::Unsupported(_) => "unsupported",
            Label::Unsolvable => "unsolvable",
            Label::Uncertain(_) => "uncertain",
            Label::Solvable { .. } => "solvable",
        }
    }

    /// JSON object members (no braces), starting with `"verdict"`.
    pub fn json_fields(&self) -> String {
        let mut s = format!("\"verdict\":\"{}\"", self.verdict());
        match self {
            Label::Invalid(why) | Label::Unsupported(why) => {
                s.push_str(&format!(",\"problem\":{}", json_str(why)));
            }
            Label::Unsolvable => {}
            Label::Uncertain(len) => {
                if let Some(n) = len {
                    s.push_str(&format!(",\"liberal_len\":{n}"));
                }
            }
            Label::Solvable { min_len, metrics } => {
                s.push_str(&format!(",\"min_len\":{min_len}"));
                if let Some(m) = metrics {
                    s.push_str(&format!(
                        ",\"metrics\":{{\"score\":{:.4},\"portal_trips\":{},\"shortest_plans\":{},\
                         \"states\":{},\"dead_end_states\":{},\"cubes\":{},\"unnecessary_cubes\":{},\
                         \"classes\":{},\"necessary_classes\":{},\"random_success\":{:.4},\
                         \"necessary_buttons\":{},\"necessary_pedestals\":{},\
                         \"load_bearing_fizzlers\":{},\"load_bearing_lasers\":{},\
                         \"necessary_items\":{},\"mechanics\":{},\"chain_depth\":{:.2},\
                         \"travel\":{:.2},\"cube_travel\":{:.2},\"backtrack\":{:.2}}}",
                        m.score(),
                        m.portal_trips,
                        m.shortest_plans,
                        m.states,
                        m.dead_end_states,
                        m.cubes,
                        m.unnecessary_cubes,
                        m.classes,
                        m.necessary_classes,
                        m.random_success,
                        m.necessary_buttons,
                        m.necessary_pedestals,
                        m.load_bearing_fizzlers,
                        m.load_bearing_lasers,
                        m.necessary_items(),
                        m.mechanics.count_ones(),
                        m.chain_depth,
                        m.travel,
                        m.cube_travel,
                        m.backtrack
                    ));
                }
            }
        }
        s
    }
}

/// Judges `puzzle` in both modes and, if it's solvable, measures it. `seed` drives the random
/// players in the metrics.
pub fn label(puzzle: &Puzzle, budget: usize, seed: u64) -> Label {
    let level = Level::from_puzzle(puzzle);
    let problems = level.mount_problems();
    if !problems.is_empty() {
        return Label::Invalid(problems.join("; "));
    }
    let judged = match judge(&level, budget) {
        Ok(j) => j,
        Err(problems) => {
            let unsupported: Vec<&str> = problems
                .iter()
                .filter_map(|p| match p {
                    Problem::Unsupported(m) => Some(m.as_str()),
                    _ => None,
                })
                .collect();
            let invalid: Vec<&str> = problems
                .iter()
                .filter_map(|p| match p {
                    Problem::Invalid(m) => Some(m.as_str()),
                    _ => None,
                })
                .collect();
            return if invalid.is_empty() {
                Label::Unsupported(unsupported.join("; "))
            } else {
                Label::Invalid(invalid.join("; "))
            };
        }
    };
    match judged.judgement {
        Judgement::Solvable(plan) => {
            let mut model = judged.conservative;
            let metrics = match assess(&mut model, budget, seed) {
                Assessment::Solved(m) => Some(m),
                _ => None,
            };
            Label::Solvable {
                min_len: plan.len(),
                metrics,
            }
        }
        Judgement::Unsolvable => Label::Unsolvable,
        Judgement::Uncertain(plan) => Label::Uncertain(plan.map(|p| p.len())),
    }
}

pub fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// `f` over `items` on `threads` threads, results in input order.
pub fn par_map<T: Sync, R: Send>(
    items: &[T],
    threads: usize,
    f: impl Fn(&T) -> R + Sync,
) -> Vec<R> {
    let chunk = items.len().div_ceil(threads.max(1)).max(1);
    let f = &f;
    std::thread::scope(|scope| {
        let handles: Vec<_> = items
            .chunks(chunk)
            .map(|part| scope.spawn(move || part.iter().map(f).collect::<Vec<_>>()))
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("worker thread panicked"))
            .collect()
    })
}

#[derive(Clone, Debug)]
pub struct Config {
    pub seed: u64,
    /// Levels to emit (distinct token sequences).
    pub count: usize,
    /// State budget per solve.
    pub budget: usize,
    pub threads: usize,
    /// Share of jobs that are fresh random genomes; the rest mutate solvable ones.
    pub fresh: f64,
    /// Solvable genomes kept as parents, biased toward high `Metrics::score`.
    pub pool: usize,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            seed: 1,
            count: 1000,
            budget: 50_000,
            threads: std::thread::available_parallelism().map_or(4, |n| n.get()),
            fresh: 0.3,
            pool: 2000,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Sample {
    pub genome: Genome,
    pub tokens: Vec<String>,
    pub label: Label,
}

impl Sample {
    /// One JSONL line (no newline).
    pub fn json(&self) -> String {
        format!(
            "{{\"tokens\":{},\"n_tokens\":{},\"size\":[{},{},{}],{}}}",
            json_str(&self.tokens.join(" ")),
            self.tokens.len(),
            self.genome.size[0],
            self.genome.size[1],
            self.genome.size[2],
            self.label.json_fields()
        )
    }
}

fn sample(genome: &Genome, budget: usize, seed: u64) -> Option<Sample> {
    let puzzle = genome.build()?;
    let tokens = tokens::encode(&puzzle).ok()?;
    Some(Sample {
        genome: genome.clone(),
        label: label(&puzzle, budget, seed),
        tokens,
    })
}

/// Emits `cfg.count` distinct levels in a seed-determined order.
pub fn generate(cfg: &Config, mut emit: impl FnMut(&Sample)) {
    let mut rng = Rng::new(cfg.seed);
    let mut pool: Vec<(Genome, f64)> = Vec::new();
    let mut seen = HashSet::new();
    let mut emitted = 0;
    // Fixed, so the job stream (and pool updates) don't depend on the thread count.
    const BATCH: usize = 64;
    while emitted < cfg.count {
        let jobs: Vec<(Genome, u64)> = (0..BATCH)
            .map(|_| {
                let g = if pool.is_empty() || rng.chance(cfg.fresh) {
                    Genome::random(&mut rng)
                } else {
                    let mut g = rng.pick(&pool).0.clone();
                    for _ in 0..rng.range(1, 4) {
                        g.mutate(&mut rng);
                    }
                    g
                };
                (g, rng.next_u64())
            })
            .collect();
        let results = par_map(&jobs, cfg.threads, |(g, seed)| sample(g, cfg.budget, *seed));
        for s in results.into_iter().flatten() {
            let mut h = DefaultHasher::new();
            s.tokens.hash(&mut h);
            if !seen.insert(h.finish()) {
                continue;
            }
            // Parents drift toward interesting levels: a newcomer replaces a random member
            // if it scores at least as well (or 1 time in 4 anyway, to stay diverse).
            if let Label::Solvable {
                metrics: Some(m), ..
            } = &s.label
            {
                let entry = (s.genome.clone(), m.score());
                if pool.len() < cfg.pool {
                    pool.push(entry);
                } else {
                    let i = rng.below(pool.len());
                    if entry.1 >= pool[i].1 || rng.chance(0.25) {
                        pool[i] = entry;
                    }
                }
            }
            emit(&s);
            emitted += 1;
            if emitted == cfg.count {
                break;
            }
        }
    }
}
