//! Random genomes, mutation and the evolutionary search.

use glados::evolve::{evolve, Config};
use glados::generator::Genome;
use glados::level::Level;
use glados::p2c::Puzzle;
use glados::rng::Rng;

#[test]
fn rng_range_is_inclusive_and_seeded() {
    let mut a = Rng::new(42);
    let mut b = Rng::new(42);
    let xs: Vec<i32> = (0..1000).map(|_| a.range(-2, 2)).collect();
    assert!(xs.iter().all(|x| (-2..=2).contains(x)));
    assert!((-2..=2).all(|v| xs.contains(&v)));
    assert_eq!(xs, (0..1000).map(|_| b.range(-2, 2)).collect::<Vec<_>>());
}

#[test]
fn built_genomes_are_well_formed() {
    let mut rng = Rng::new(5);
    let mut built = 0;
    for _ in 0..300 {
        let mut g = Genome::random(&mut rng);
        for _ in 0..rng.range(0, 20) {
            g.mutate(&mut rng);
        }
        let Some(p) = g.build() else { continue };
        built += 1;
        let text = p.to_p2c();
        assert_eq!(Puzzle::parse(&text).unwrap(), p);
        let level = Level::from_puzzle(&p);
        assert_eq!(level.mount_problems(), Vec::<String>::new(), "{g:?}");
        assert!(level.entry().is_some() && level.exit().is_some());
    }
    assert!(built > 100, "only {built} of 300 genomes built");
}

#[test]
fn genomes_are_seeded() {
    assert_eq!(
        Genome::random(&mut Rng::new(9)),
        Genome::random(&mut Rng::new(9))
    );
}

fn small(threads: usize) -> Config {
    Config {
        seed: 11,
        population: 8,
        generations: 4,
        budget: 50_000,
        threads,
        min_area: 0,
    }
}

#[test]
fn evolve_respects_min_area() {
    let cfg = Config {
        min_area: 100,
        ..small(4)
    };
    let pop = evolve(&cfg, |_, pop| {
        assert!(pop.iter().all(|c| c.genome.size[0] * c.genome.size[1] >= 100));
    });
    assert!(!pop.is_empty());
}

#[test]
fn evolve_is_deterministic_across_thread_counts() {
    let summary = |threads| {
        evolve(&small(threads), |_, _| {})
            .iter()
            .map(|c| (c.puzzle.to_p2c(), c.score.to_bits()))
            .collect::<Vec<_>>()
    };
    let one = summary(1);
    assert!(!one.is_empty());
    assert_eq!(one, summary(3));
}

#[test]
fn evolve_keeps_its_best() {
    let mut best = Vec::new();
    let pop = evolve(&small(2), |s, pop| {
        assert!(pop.windows(2).all(|w| w[0].score >= w[1].score), "ranked");
        best.push(s.best);
    });
    assert_eq!(best.len(), 5);
    assert!(best.windows(2).all(|w| w[1] >= w[0]), "{best:?}");
    assert!(pop.len() <= 8);
    for c in &pop {
        assert_eq!(c.score, c.metrics.score());
    }
}

#[test]
fn genomes_produce_tier1_items() {
    use glados::level::ItemKind;
    let mut rng = Rng::new(21);
    let (mut barriers, mut hazards, mut pedestals) = (0, 0, 0);
    for _ in 0..300 {
        let mut g = Genome::random(&mut rng);
        for _ in 0..10 {
            g.mutate(&mut rng);
        }
        let Some(p) = g.build() else { continue };
        let level = Level::from_puzzle(&p);
        assert_eq!(level.decode_problems, Vec::<String>::new());
        barriers += level.barriers.len();
        hazards += level.hazards.len();
        pedestals += level.of_kind(&ItemKind::PedestalButton).count();
    }
    assert!(
        barriers > 20 && hazards > 20 && pedestals > 20,
        "{barriers} {hazards} {pedestals}"
    );
}

#[test]
fn genomes_make_big_rooms_walls_and_balconies() {
    let mut rng = Rng::new(33);
    let (mut big, mut walls, mut balconies) = (0, 0, 0);
    for _ in 0..300 {
        let mut g = Genome::random(&mut rng);
        for _ in 0..10 {
            g.mutate(&mut rng);
        }
        let Some(p) = g.build() else { continue };
        let level = Level::from_puzzle(&p);
        assert_eq!(level.mount_problems(), Vec::<String>::new(), "{g:?}");
        let [sx, sy, sz] = level.size;
        big += (sx > 10 || sy > 10) as usize;
        for x in 0..sx {
            for y in 0..sy {
                let c: Vec<bool> = (0..sz).map(|z| level.solid([x, y, z])).collect();
                // Solid all the way up: a wall. Open under solid: a balcony.
                walls += c.iter().all(|&s| s) as usize;
                balconies += c.windows(2).any(|w| !w[0] && w[1]) as usize;
            }
        }
    }
    assert!(big > 50 && walls > 50 && balconies > 50, "{big} {walls} {balconies}");
}
