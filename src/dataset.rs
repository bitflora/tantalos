//! Training data for sequence models, and scoring of what they write.
//!
//! `generate` samples genomes (fresh random ones, and mutations of earlier solvable ones so
//! the data isn't dominated by unsolvable rooms), encodes each level as tokens and labels it
//! with the solver's verdict and metrics. `label` is also what scores model output.
//! Results depend only on the seed, not on the thread count.

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Mutex};

use crate::generator::Genome;
use crate::level::Level;
use crate::metrics::{measure, Assessment, Metrics};
use crate::p2c::Puzzle;
use crate::rng::Rng;
use crate::solver::{Action, Mode, Model, Problem, Verdict};
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
    /// Conservative rules solve it in `min_len` actions; `needs_portals` if they can't
    /// without firing a portal. `metrics` is `None` if measuring ran out of budget.
    Solvable {
        min_len: usize,
        needs_portals: bool,
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
            Label::Solvable {
                min_len,
                needs_portals,
                metrics,
            } => {
                s.push_str(&format!(
                    ",\"min_len\":{min_len},\"needs_portals\":{needs_portals}"
                ));
                if let Some(m) = metrics {
                    s.push_str(&format!(
                        ",\"metrics\":{{\"score\":{:.4},\"portal_trips\":{},\"shortest_plans\":{},\
                         \"states\":{},\"dead_end_states\":{},\"cubes\":{},\"unnecessary_cubes\":{},\
                         \"classes\":{},\"necessary_classes\":{},\"random_success\":{:.4},\
                         \"necessary_buttons\":{},\"necessary_pedestals\":{},\
                         \"load_bearing_fizzlers\":{},\"load_bearing_lasers\":{},\
                         \"load_bearing_turrets\":{},\"necessary_gels\":{},\
                         \"necessary_items\":{},\"mechanics\":{},\"mechanic_bits\":{},\"chain_depth\":{:.2},\
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
                        m.load_bearing_turrets,
                        m.necessary_gels,
                        m.necessary_items(),
                        m.mechanics.count_ones(),
                        m.mechanics,
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
    label_with(puzzle, budget, seed, true)
}

/// `label`, optionally skipping our editor rules (see `Level::mount_problems_with`).
pub fn label_with(puzzle: &Puzzle, budget: usize, seed: u64, editor_rules: bool) -> Label {
    label_known(puzzle, budget, seed, editor_rules, None)
}

/// `label_with`, plus `known`: a solution length the caller has established by construction (the
/// reverse generator does — see `bin/reverse.rs`). If the conservative search overflows `budget`
/// before it reaches a goal, the level is reported `Solvable` at `known` rather than `Uncertain`.
/// Big levels are exactly that case: their reachable state space runs to millions of states, far
/// past any budget worth paying, yet the construction already knows the way out. A level the
/// search actually *proves* unsolvable is still `Unsolvable`; only an unfinished search is
/// overridden, and then the liberal model isn't consulted.
pub fn label_known(
    puzzle: &Puzzle,
    budget: usize,
    seed: u64,
    editor_rules: bool,
    known: Option<usize>,
) -> Label {
    label_full(puzzle, budget, seed, editor_rules, known).0
}

/// `label_with`, plus the conservative plan of a solvable level, as plan tokens (`plan`), or
/// why it couldn't be grounded (`route`).
pub fn label_with_plan(
    puzzle: &Puzzle,
    budget: usize,
    seed: u64,
    editor_rules: bool,
) -> (Label, Option<Result<String, String>>) {
    let (label, plan) = label_full(puzzle, budget, seed, editor_rules, None);
    let plan = plan.map(|plan| {
        let level = Level::from_puzzle(puzzle);
        let model = Model::new(&level, Mode::Conservative).map_err(|p| format!("{p:?}"))?;
        crate::route::route_plan(&level, &model, &plan).map(|r| crate::plan::encode(&r))
    });
    (label, plan)
}

/// Only the conservative plan of a level already labelled solvable: its length and plan tokens
/// (or why it couldn't be grounded), without the metrics, the portal check or the liberal model
/// that labelling pays for. A level that no longer solves gets its label instead: `Invalid`,
/// `Unsupported`, `Unsolvable`, or `Uncertain(None)` if the budget ran out.
pub fn plan_only(
    puzzle: &Puzzle,
    budget: usize,
    editor_rules: bool,
) -> Result<(usize, Result<String, String>), Label> {
    let level = Level::from_puzzle(puzzle);
    let problems = level.mount_problems_with(editor_rules);
    if !problems.is_empty() {
        return Err(Label::Invalid(problems.join("; ")));
    }
    let model = Model::new(&level, Mode::Conservative).map_err(|problems| {
        let (invalid, unsupported): (Vec<_>, Vec<_>) =
            problems.iter().partition(|p| matches!(p, Problem::Invalid(_)));
        let text = |ps: Vec<&Problem>| {
            ps.iter()
                .map(|p| match p {
                    Problem::Invalid(m) | Problem::Unsupported(m) => m.as_str(),
                })
                .collect::<Vec<_>>()
                .join("; ")
        };
        if invalid.is_empty() {
            Label::Unsupported(text(unsupported))
        } else {
            Label::Invalid(text(invalid))
        }
    })?;
    match model.solve(budget).verdict {
        Verdict::Solved(plan) => Ok((
            plan.len(),
            crate::route::route_plan(&level, &model, &plan).map(|r| crate::plan::encode(&r)),
        )),
        Verdict::Unsolvable => Err(Label::Unsolvable),
        Verdict::Unknown => Err(Label::Uncertain(None)),
    }
}

/// `label_known`, with the conservative plan when the search found one.
fn label_full(
    puzzle: &Puzzle,
    budget: usize,
    seed: u64,
    editor_rules: bool,
    known: Option<usize>,
) -> (Label, Option<Vec<Action>>) {
    let level = Level::from_puzzle(puzzle);
    let problems = level.mount_problems_with(editor_rules);
    if !problems.is_empty() {
        return (Label::Invalid(problems.join("; ")), None);
    }
    // `judge`, with the conservative search shared with the metrics, and the liberal model
    // built only when needed (both models have the same problems).
    let mut model = match Model::new(&level, Mode::Conservative) {
        Ok(m) => m,
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
            let label = if invalid.is_empty() {
                Label::Unsupported(unsupported.join("; "))
            } else {
                Label::Invalid(invalid.join("; "))
            };
            return (label, None);
        }
    };
    let space = model.explore(budget);
    let strict = match space.verdict() {
        Verdict::Solved(plan) => {
            let metrics = match measure(&mut model, &space, budget, seed) {
                Assessment::Solved(m) => Some(m),
                _ => None,
            };
            let label = Label::Solvable {
                min_len: plan.len(),
                needs_portals: !space.solvable_without_portals(),
                metrics,
            };
            return (label, Some(plan));
        }
        strict => strict,
    };
    // The conservative search ran out of budget before it reached a goal. A caller that built the
    // level knows a solution, so trust it; every reverse gadget needs at least one portal trip.
    if let (Verdict::Unknown, Some(n)) = (&strict, known) {
        let label = Label::Solvable {
            min_len: n,
            needs_portals: true,
            metrics: None,
        };
        return (label, None);
    }
    drop(space);
    let liberal = Model::new(&level, Mode::Liberal).expect("same problems as conservative");
    let label = match liberal.solve(budget).verdict {
        Verdict::Unsolvable if strict == Verdict::Unsolvable => Label::Unsolvable,
        Verdict::Solved(plan) => Label::Uncertain(Some(plan.len())),
        _ => Label::Uncertain(None),
    };
    (label, None)
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

/// `f` over `items` on `threads` threads, results in input order. Each thread takes the next
/// item when it's free, so one slow item doesn't hold up a fixed share of the rest.
pub fn par_map<T: Sync, R: Send>(
    items: &[T],
    threads: usize,
    f: impl Fn(&T) -> R + Sync,
) -> Vec<R> {
    let mut out = Vec::with_capacity(items.len());
    let done = par_map_ordered(items.iter(), threads, f, |r| {
        out.push(r);
        Ok::<(), std::convert::Infallible>(())
    });
    match done {
        Ok(()) => out,
        Err(never) => match never {},
    }
}

/// Maps `f` over `items` on `threads` threads and hands each result to `emit` in input order,
/// as soon as it and everything before it are done. Threads take one item at a time, so a slow
/// item holds up only its own thread, and `items` is read as threads need it, so the input
/// needn't fit in memory. If `emit` fails, threads stop after their current item and the error
/// is returned.
pub fn par_map_ordered<I, R, E>(
    items: I,
    threads: usize,
    f: impl Fn(I::Item) -> R + Sync,
    mut emit: impl FnMut(R) -> Result<(), E>,
) -> Result<(), E>
where
    I: Iterator + Send,
    I::Item: Send,
    R: Send,
{
    use std::sync::atomic::{AtomicBool, Ordering};
    let source = std::sync::Mutex::new(items.enumerate());
    let stop = AtomicBool::new(false);
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            let (tx, source, stop, f) = (tx.clone(), &source, &stop, &f);
            scope.spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let Some((i, item)) = source.lock().expect("input poisoned").next() else {
                        break;
                    };
                    if tx.send((i, f(item))).is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx);
        // Results that arrived ahead of an earlier one still being worked on.
        let mut waiting = std::collections::HashMap::new();
        let mut next = 0;
        for (i, r) in rx {
            waiting.insert(i, r);
            while let Some(r) = waiting.remove(&next) {
                next += 1;
                if let Err(e) = emit(r) {
                    stop.store(true, Ordering::Relaxed);
                    return Err(e);
                }
            }
        }
        Ok(())
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
    /// Drop solvable levels that can be solved without portals (and don't breed from them).
    pub portals_required: bool,
    /// Most share of the emitted levels that aren't solvable; the rest are dropped.
    pub max_unsolved: f64,
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
            portals_required: false,
            max_unsolved: 1.0,
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
        format!("{{{}}}", json_body(&self.tokens, self.genome.size, &self.label))
    }
}

/// The fields of a dataset line, without the braces: `tokens`, `n_tokens`, `size`, then the
/// label's. Other writers (src/bin/reverse.rs) append their own fields after it.
pub fn json_body(tokens: &[String], size: [i32; 3], label: &Label) -> String {
    format!(
        "\"tokens\":{},\"n_tokens\":{},\"size\":[{},{},{}],{}",
        json_str(&tokens.join(" ")),
        tokens.len(),
        size[0],
        size[1],
        size[2],
        label.json_fields()
    )
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

/// Parents kept per mechanic (bit of `Metrics::mechanics`), so rare mechanics keep breeding.
const RESERVE: usize = 100;
/// Share of mutations whose parent comes from a mechanic's reserve, picked evenly over the
/// mechanics seen so far, rather than from the score-biased pool.
const RESERVE_SHARE: f64 = 0.4;

/// Job `j` takes its parent from the pool as it stood once the results of the jobs before
/// `j - lag(j)` were in, so up to `lag` jobs run at once and a slow level holds up nothing
/// until the rest are that far ahead. The lag starts at `BATCH` so the pool fills early.
const BATCH: usize = 64;
const WINDOW: usize = 512;

fn lag(j: usize) -> usize {
    (j / 2).clamp(BATCH, WINDOW)
}

/// Emits `cfg.count` distinct levels in a seed-determined order: jobs are made, and their
/// results taken in, in a fixed order whatever the thread count.
pub fn generate(cfg: &Config, mut emit: impl FnMut(&Sample)) {
    let mut rng = Rng::new(cfg.seed);
    let mut pool: Vec<(Genome, f64)> = Vec::new();
    let mut reserves: Vec<Vec<Genome>> = vec![Vec::new(); crate::metrics::MECHANICS.len()];
    let mut seen = HashSet::new();
    let (mut emitted, mut unsolved) = (0, 0);
    if cfg.count == 0 {
        return;
    }
    let (job_tx, job_rx) = mpsc::channel::<(usize, Genome, u64)>();
    let (result_tx, result_rx) = mpsc::channel::<(usize, std::thread::Result<Option<Sample>>)>();
    let job_rx = Mutex::new(job_rx);
    let stop = AtomicBool::new(false);
    std::thread::scope(|scope| {
        for _ in 0..cfg.threads.max(1) {
            let (job_rx, stop, result_tx) = (&job_rx, &stop, result_tx.clone());
            scope.spawn(move || loop {
                let job = job_rx.lock().unwrap_or_else(|e| e.into_inner()).recv();
                let Ok((j, genome, seed)) = job else { break };
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                // A panic goes to the coordinator, which would otherwise wait for this job.
                let sample = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    sample(&genome, cfg.budget, seed)
                }));
                if result_tx.send((j, sample)).is_err() {
                    break;
                }
            });
        }
        drop(result_tx);
        // However this ends, jobs still queued are dropped (`_stop` goes first).
        let job_tx = job_tx;
        let _stop = StopOnDrop(&stop);

        let mut waiting: HashMap<usize, Option<Sample>> = HashMap::new();
        let mut next_job: usize = 0;
        for taken in 0.. {
            // Every job whose parent pool is complete now.
            while next_job.saturating_sub(lag(next_job)) <= taken {
                let stocked: Vec<usize> = (0..reserves.len())
                    .filter(|&i| !reserves[i].is_empty())
                    .collect();
                let g = if pool.is_empty() || rng.chance(cfg.fresh) {
                    Genome::random(&mut rng)
                } else {
                    let mut g = if !stocked.is_empty() && rng.chance(RESERVE_SHARE) {
                        let i = *rng.pick(&stocked);
                        rng.pick(&reserves[i]).clone()
                    } else {
                        rng.pick(&pool).0.clone()
                    };
                    for _ in 0..rng.range(1, 4) {
                        g.mutate(&mut rng);
                    }
                    g
                };
                job_tx
                    .send((next_job, g, rng.next_u64()))
                    .expect("workers wait for jobs");
                next_job += 1;
            }
            let result = loop {
                if let Some(r) = waiting.remove(&taken) {
                    break r;
                }
                let (j, r) = result_rx.recv().expect("workers outlive the job queue");
                let r = r.unwrap_or_else(|panic| std::panic::resume_unwind(panic));
                waiting.insert(j, r);
            };
            let Some(s) = result else { continue };
            let mut h = DefaultHasher::new();
            s.tokens.hash(&mut h);
            if !seen.insert(h.finish()) {
                continue;
            }
            match &s.label {
                Label::Solvable { needs_portals, .. } => {
                    if cfg.portals_required && !needs_portals {
                        continue;
                    }
                }
                _ => {
                    if (unsolved + 1) as f64 > cfg.max_unsolved * (emitted + 1) as f64 {
                        continue;
                    }
                    unsolved += 1;
                }
            }
            // Parents drift toward interesting levels: a newcomer replaces a random member
            // if it scores at least as well (or 1 time in 4 anyway, to stay diverse).
            if let Label::Solvable {
                metrics: Some(m), ..
            } = &s.label
            {
                for (i, reserve) in reserves.iter_mut().enumerate() {
                    if m.mechanics & (1 << i) == 0 {
                        continue;
                    }
                    if reserve.len() < RESERVE {
                        reserve.push(s.genome.clone());
                    } else {
                        let j = rng.below(RESERVE);
                        reserve[j] = s.genome.clone();
                    }
                }
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
    });
}

/// Tells the workers to stop when dropped.
struct StopOnDrop<'a>(&'a AtomicBool);

impl Drop for StopOnDrop<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn par_map_keeps_input_order_with_uneven_work() {
        let items: Vec<u64> = (0..200).collect();
        let slow = |&x: &u64| {
            if x % 17 == 0 {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            x * x
        };
        let expected: Vec<u64> = items.iter().map(slow).collect();
        for threads in [1, 3, 8] {
            assert_eq!(par_map(&items, threads, slow), expected);
        }
        assert!(par_map(&[] as &[u64], 4, slow).is_empty());
    }

    #[test]
    fn par_map_ordered_stops_when_emit_fails() {
        let taken = std::sync::atomic::AtomicUsize::new(0);
        let mut seen = Vec::new();
        let result = par_map_ordered(
            0..1_000_000u64,
            4,
            |x| {
                taken.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                x
            },
            |x| {
                seen.push(x);
                if x == 10 {
                    Err("full")
                } else {
                    Ok(())
                }
            },
        );
        assert_eq!(result, Err("full"));
        assert_eq!(seen, (0..=10).collect::<Vec<_>>());
        assert!(
            taken.into_inner() < 1_000_000,
            "threads kept going after the error"
        );
    }
}
