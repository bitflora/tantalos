//! Generate-and-test search: an elitist evolutionary loop over `Genome`s, scored by
//! `metrics`. Only solved levels enter the population. Evaluation runs on all cores; results
//! depend only on the seed, not on the thread count.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use crate::generator::Genome;
use crate::level::Level;
use crate::metrics::{assess, Assessment, Metrics};
use crate::p2c::Puzzle;
use crate::rng::Rng;
use crate::solver::Model;

#[derive(Clone, Debug)]
pub struct Config {
    pub seed: u64,
    pub population: usize,
    pub generations: usize,
    /// State budget per solve.
    pub budget: usize,
    pub threads: usize,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            seed: 1,
            population: 48,
            generations: 40,
            budget: 50_000,
            threads: std::thread::available_parallelism().map_or(4, |n| n.get()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Candidate {
    pub genome: Genome,
    pub puzzle: Puzzle,
    pub metrics: Metrics,
    pub score: f64,
    /// Generation it was born in (0 = initial random population).
    pub born: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub generation: usize,
    pub evaluated: usize,
    pub solved: usize,
    pub best: f64,
    pub mean: f64,
}

/// Builds, solves and measures one genome. `None` unless it builds, is valid Tier 0, and is
/// solved within the budget.
pub fn evaluate(genome: &Genome, budget: usize, seed: u64, born: usize) -> Option<Candidate> {
    let puzzle = genome.build()?;
    let level = Level::from_puzzle(&puzzle);
    if !level.mount_problems().is_empty() {
        return None;
    }
    let mut model = Model::new(&level).ok()?;
    match assess(&mut model, budget, seed) {
        Assessment::Solved(metrics) => Some(Candidate {
            genome: genome.clone(),
            puzzle,
            score: metrics.score(),
            metrics,
            born,
        }),
        _ => None,
    }
}

fn evaluate_all(jobs: &[(Genome, u64)], cfg: &Config, born: usize) -> Vec<Option<Candidate>> {
    let chunk = jobs.len().div_ceil(cfg.threads.max(1)).max(1);
    std::thread::scope(|scope| {
        let handles: Vec<_> = jobs
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    part.iter()
                        .map(|(g, seed)| evaluate(g, cfg.budget, *seed, born))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("evaluation thread panicked"))
            .collect()
    })
}

fn key(p: &Puzzle) -> u64 {
    let mut h = DefaultHasher::new();
    p.to_p2c().hash(&mut h);
    h.finish()
}

/// Ranks by score (ties: older, then fewer states), drops behavioural duplicates, keeps `n`.
/// Mutations that only touch unused space give a new file with identical metrics; without
/// this the population fills with clones of one lineage.
fn cull(pop: &mut Vec<Candidate>, n: usize) {
    pop.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then(a.born.cmp(&b.born))
            .then(a.metrics.states.cmp(&b.metrics.states))
    });
    let mut kinds = HashSet::new();
    pop.retain(|c| {
        let m = &c.metrics;
        kinds.insert((
            c.genome.size,
            m.min_len,
            m.portal_trips,
            m.states,
            m.dead_end_states,
            m.classes,
            m.necessary_classes,
            m.cubes,
            m.unnecessary_cubes,
        ))
    });
    pop.truncate(n);
}

/// Runs the search; `progress` sees the ranked population after each generation.
pub fn evolve(cfg: &Config, mut progress: impl FnMut(&Stats, &[Candidate])) -> Vec<Candidate> {
    let mut rng = Rng::new(cfg.seed);
    let mut seen = HashSet::new();
    let mut pop: Vec<Candidate> = Vec::new();

    let mut admit = |pop: &mut Vec<Candidate>, results: Vec<Option<Candidate>>| {
        let mut solved = 0;
        for c in results.into_iter().flatten() {
            solved += 1;
            if seen.insert(key(&c.puzzle)) {
                pop.push(c);
            }
        }
        solved
    };
    let stats = |generation, evaluated, solved, pop: &[Candidate]| Stats {
        generation,
        evaluated,
        solved,
        best: pop.first().map_or(0.0, |c| c.score),
        mean: pop.iter().map(|c| c.score).sum::<f64>() / pop.len().max(1) as f64,
    };

    // Random levels until the population is full (or we give up).
    let (mut evaluated, mut solved) = (0, 0);
    for _ in 0..50 {
        if pop.len() >= cfg.population {
            break;
        }
        let jobs: Vec<_> = (0..cfg.population * 2)
            .map(|_| (Genome::random(&mut rng), rng.next_u64()))
            .collect();
        evaluated += jobs.len();
        solved += admit(&mut pop, evaluate_all(&jobs, cfg, 0));
    }
    cull(&mut pop, cfg.population);
    progress(&stats(0, evaluated, solved, &pop), &pop);
    if pop.is_empty() {
        return pop;
    }

    for generation in 1..=cfg.generations {
        let jobs: Vec<_> = (0..cfg.population)
            .map(|_| {
                let mut child = tournament(&pop, &mut rng, 3).genome.clone();
                for _ in 0..rng.range(1, 3) {
                    child.mutate(&mut rng);
                }
                (child, rng.next_u64())
            })
            .collect();
        let solved = admit(&mut pop, evaluate_all(&jobs, cfg, generation));
        cull(&mut pop, cfg.population);
        progress(&stats(generation, jobs.len(), solved, &pop), &pop);
    }
    pop
}

fn tournament<'a>(pop: &'a [Candidate], rng: &mut Rng, k: usize) -> &'a Candidate {
    // `pop` is ranked, so the lowest index drawn wins.
    (0..k)
        .map(|_| rng.below(pop.len()))
        .min()
        .map(|i| &pop[i])
        .unwrap()
}
