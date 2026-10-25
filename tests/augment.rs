//! Mirrored and turned copies of levels (`glados-data augment`, src/augment.rs): each of the 8
//! versions is the same level moved, cell for cell and face for face, and plays the same.

use glados::augment::{editor_rules, moved_level, placed, variant, Sym};
use glados::dataset::label_with_plan;
use glados::fixtures;
use glados::generator::Genome;
use glados::geom::{step, Dir};
use glados::level::Level;
use glados::p2c::Puzzle;
use glados::rng::Rng;
use glados::solver::{judge, Judgement};
use glados::tokens::{decode, encode};
use glados::turrets::{danger, TurretSight};

const BUDGET: usize = 50_000;

/// Fixtures and generated levels that the token format can write.
fn levels() -> Vec<(String, Puzzle)> {
    let mut out: Vec<(String, Puzzle)> = fixtures::all()
        .into_iter()
        .filter(|(_, p)| encode(p).is_ok())
        .map(|(n, p)| (n.to_string(), p))
        .collect();
    let mut rng = Rng::new(11);
    for i in 0..300 {
        let mut g = Genome::random(&mut rng);
        for _ in 0..rng.range(0, 15) {
            g.mutate(&mut rng);
        }
        if let Some(p) = g.build().filter(|p| encode(p).is_ok()) {
            out.push((format!("genome {i}"), p));
        }
    }
    out
}

/// The level re-read from its own tokens, as `variant` starts from.
fn canonical(p: &Puzzle) -> (Puzzle, Vec<String>) {
    let t = encode(p).unwrap();
    let q = decode(&t.join(" ")).unwrap();
    (q, t)
}

fn outcome(p: &Puzzle) -> String {
    match judge(&Level::from_puzzle(p), BUDGET) {
        Err(problems) => format!("problems {}", problems.len()),
        Ok(j) => match j.judgement {
            Judgement::Solvable(plan) => format!("solvable in {}", plan.len()),
            Judgement::Unsolvable => "unsolvable".into(),
            Judgement::Uncertain(plan) => format!("uncertain {:?}", plan.map(|p| p.len())),
        },
    }
}

#[test]
fn symmetries_compose() {
    let size = [5, 3, 4];
    for s in Sym::ALL {
        let back = s.inverse();
        assert_eq!(back.size(s.size(size)), size);
        for d in Dir::ALL {
            assert_eq!(back.dir(s.dir(d)), d, "{}", s.name());
            assert_eq!(s.dir(d).opposite(), s.dir(d.opposite()));
        }
        for c in [[0, 0, 0], [4, 2, 3], [1, 2, 0]] {
            let moved = s.cell(c, size);
            let m = s.size(size);
            assert!((0..3).all(|a| moved[a] >= 0 && moved[a] < m[a]), "{} {c:?}", s.name());
            assert_eq!(back.cell(moved, m), c, "{} {c:?}", s.name());
            // A step along d moves with d.
            for d in Dir::HORIZONTAL {
                let n = step(c, d);
                if (0..3).all(|a| n[a] >= 0 && n[a] < size[a]) {
                    assert_eq!(s.cell(n, size), step(moved, s.dir(d)), "{} {c:?} {d:?}", s.name());
                }
            }
        }
        // A turret facing +x, +y, -x, -y (eighths 0, 2, 4, 6) turns with those directions.
        for (f, d) in [(0, Dir::PosX), (2, Dir::PosY), (4, Dir::NegX), (6, Dir::NegY)] {
            let to = Dir::HORIZONTAL.iter().find(|&&e| e == s.dir(d)).unwrap();
            let want = [(Dir::PosX, 0), (Dir::PosY, 2), (Dir::NegX, 4), (Dir::NegY, 6)]
                .into_iter()
                .find(|(e, _)| e == to)
                .unwrap()
                .1;
            assert_eq!(s.facing(f), want, "{} facing {f}", s.name());
        }
    }
    assert_eq!(Sym::ALL.iter().map(|s| s.name()).collect::<Vec<_>>().join(" "),
               "r0 r90 r180 r270 m0 m90 m180 m270");
}

#[test]
fn versions_are_the_level_moved() {
    let all = levels();
    assert!(all.len() > 150, "only {} levels", all.len());
    let (mut mirrored, mut refused) = (0, 0);
    for (name, p) in &all {
        let (orig, tokens) = canonical(p);
        let a = Level::from_puzzle(&orig);
        for s in Sym::ALL {
            let what = format!("{name} {}", s.name());
            mirrored += usize::from(s.mirror);
            let (q, t) = match variant(&orig, s) {
                Ok(v) => v,
                // Only a mirror moves a field's anchor to its other end.
                Err(e) if s.mirror && e.starts_with("unmounted") || e.starts_with("editor flags") => {
                    refused += 1;
                    continue;
                }
                Err(e) => panic!("{what}: {e}"),
            };
            // Canonical, and undone by the inverse.
            assert_eq!(encode(&decode(&t.join(" ")).unwrap()).unwrap(), t, "{what}: not canonical");
            assert_eq!(variant(&q, s.inverse()).unwrap().1, tokens, "{what}: inverse differs");
            if s == Sym::IDENTITY {
                assert_eq!(t, tokens, "{what}");
            }
            let b = Level::from_puzzle(&q);
            assert_eq!(b.size, s.size(a.size), "{what}");
            for c in a.cells() {
                let m = s.cell(c, a.size);
                assert_eq!(a.solid(c), b.solid(m), "{what}: solid {c:?}");
                assert_eq!(a.goo(c), b.goo(m), "{what}: goo {c:?}");
                for d in Dir::ALL {
                    let e = s.dir(d);
                    assert_eq!(a.portal_surface(c, d), b.portal_surface(m, e), "{what}: white {c:?} {d:?}");
                    let (fa, fb) = (a.face(c, d), b.face(m, e));
                    assert_eq!(fa.barrier, fb.barrier, "{what}: glass/grating {c:?} {d:?}");
                    assert_eq!(fa.hazards != 0, fb.hazards != 0, "{what}: field {c:?} {d:?}");
                }
            }
            assert_eq!(placed(&a, s), placed(&b, Sym::IDENTITY), "{what}: items");
            // (The messages name positions, which move.)
            let problems = |l: &Level| l.mount_problems_with(false).len();
            assert_eq!(
                problems(&a),
                problems(&b),
                "{what}: mount problems {:?} vs {:?}
{}
{}",
                a.mount_problems_with(false),
                b.mount_problems_with(false),
                tokens.join(" "),
                t.join(" ")
            );
            assert_eq!(editor_rules(&a), editor_rules(&b), "{what}: editor flags");
            moved_level(&a, &b, s).unwrap_or_else(|e| panic!("{what}: {e}"));
        }
    }
    assert!(refused * 20 < mirrored, "{refused} of {mirrored} mirrored versions refused");
}

#[test]
fn versions_play_the_same() {
    for (name, p) in levels().iter().step_by(3) {
        let (orig, _) = canonical(p);
        let want = outcome(&orig);
        for s in Sym::ALL.into_iter().skip(1) {
            let Ok((q, _)) = variant(&orig, s) else { continue };
            assert_eq!(outcome(&q), want, "{name} {}", s.name());
        }
    }
}

/// Workshop levels whose solver results depended on which way round they were built
/// (2026-10-25, `glados-data augment --check` on ws5 train), and the cut-down levels that
/// still showed it. Bridges through portals: routes claimed the cells they share in
/// coordinate order, and with bits short the first routes (or bridges) in that order got
/// them; an entry door onto a faith plate had no region unless a route happened to claim it.
const ROUTE_GEOMETRY: &str = "size 9 6 9 open 6 0 1 8 1 8 1 0 2 4 1 5 4 2 2 4 5 4 6 2 2 8 5 5 5 3 \
    2 8 5 4 5 0 3 8 0 4 1 2 3 2 5 4 3 3 3 8 5 4 0 0 5 4 1 5 1 2 5 2 2 5 goo 6 0 0 8 1 0 white +x \
    8 3 2 8 5 8 8 1 4 8 1 5 white +y 6 5 2 6 5 8 8 5 2 8 5 8 7 5 3 8 5 8 1 5 4 1 5 8 white -y \
    8 0 1 8 1 2 white -z 6 3 2 8 5 5 entry 2 0 2 +y exit 7 5 2 -y";
const TALL_ROOM: &str = "size 8 13 8 open 0 1 3 7 4 7 0 7 3 7 11 4 1 12 3 7 12 7 0 0 4 7 5 7 \
    7 6 4 7 12 7 0 7 5 2 12 7 3 7 5 7 8 7 4 9 5 7 12 7 3 10 5 7 12 7 3 9 6 7 12 7 black +x 7 12 3 \
    7 12 5 2 9 5 6 12 7 black -x 3 9 5 7 12 7 black +y 5 12 3 7 12 5 3 8 5 7 11 7 0 12 7 0 12 7 \
    black -y 0 0 4 0 5 4 3 9 5 7 12 7 white -z 0 1 3 7 11 4 1 12 3 4 12 7 0 0 4 7 12 4 \
    entry 0 0 4 +y exit 0 12 7 -y";

fn oriented() -> Vec<(&'static str, String, bool)> {
    // (what, tokens, whether its verdict is far enough inside the budget to compare)
    vec![
        ("sample 221's bridge alone", format!("{ROUTE_GEOMETRY} bridge 6 1 4 +x +z on"), true),
        (
            "sample 221",
            format!(
                "{ROUTE_GEOMETRY} glass 5 0 3 5 0 4 -x +z laserfield 1 2 5 +x -z 3 on fizzler 3 5 4 -y -z 2 on \
                 laserfield 8 2 5 -x -z 4 on dropper 8 0 8 standard button 1 5 3 weighted > exit \
                 bridge 6 1 4 +x +z on"
            ),
            true,
        ),
        ("exit over crossing bridge routes", format!("{TALL_ROOM} bridge 7 9 6 -x +z on"), true),
        (
            "turret looking diagonally past angled panels (sample 582 cut down)",
            "size 11 12 8 open 2 10 0 10 10 0 2 10 1 2 10 6 10 10 1 10 10 4 0 5 3 1 10 6 2 5 3 7 10 3              8 8 3 10 8 4 10 9 3 10 10 4 9 3 4 9 5 5 3 5 4 7 10 4 8 5 4 9 5 5 2 6 4 7 10 6 7 0 5 9 0 6              9 1 5 9 3 6 2 5 5 2 10 6 3 11 5 3 11 5 3 0 6 4 10 7 5 0 6 8 0 7 5 5 6 7 10 6 8 7 6 8 8 6              entry 3 10 3 -y exit 7 5 6 +y glass 2 8 3 2 8 3 -x +z dropper 0 5 6 standard              pedestal 9 1 5 inf > panel 2 > panel 3 > panel 6 > panel 7              pedestal 9 2 5 inf > panel 0 > panel 1 > panel 4 > panel 5 turret 0 5 3 5              panel 3 2 6 +z +y 6 on panel 3 2 7 -z +y 6 on panel 3 4 6 +z +y 6 on panel 3 4 7 -z +y 6 on              panel 4 2 6 +z +y 6 on panel 4 2 7 -z +y 6 on panel 4 4 6 +z +y 6 on panel 4 4 7 -z +y 6 on              stairs 6 5 3 +x on stairs 8 0 5 -x on stairs 9 4 4 -y on"
                .to_string(),
            true,
        ),
        (
            "entry door onto a faith plate",
            format!("{TALL_ROOM} plate 0 0 4 +z 0 5 4 +z bridge 7 9 6 -x +z on"),
            false,
        ),
    ]
}

#[test]
fn bridge_routes_and_doors_dont_depend_on_orientation() {
    for (what, tokens, verdict) in oriented() {
        let p = decode(&tokens).unwrap_or_else(|e| panic!("{what}: {e}"));
        let mut first = None;
        for s in Sym::ALL {
            let (q, _) = variant(&p, s).unwrap_or_else(|e| panic!("{what} {}: {e}", s.name()));
            let level = Level::from_puzzle(&q);
            let j = judge(&level, BUDGET).unwrap_or_else(|e| panic!("{what} {}: {e:?}", s.name()));
            let m = &j.conservative;
            let mut got = format!("{} regions, {} classes", m.graph.regions.len(), m.portals.classes.len());
            if verdict {
                got += &format!(", {}", outcome(&q));
                let (_, plan) = label_with_plan(&q, BUDGET, 1, false);
                if let Some(plan) = plan {
                    assert!(plan.is_ok(), "{what} {}: plan doesn't ground: {plan:?}", s.name());
                }
            }
            match &first {
                None => first = Some(got),
                Some(f) => assert_eq!(&got, f, "{what} {}", s.name()),
            }
        }
    }
}

/// Where turrets can shoot moves with the level: a line of sight through a voxel corner was
/// checked past one side cell or the other depending on the orientation (`turrets::clear_line`
/// put a sample exactly on the corner in the cell above it).
#[test]
fn turret_sight_doesnt_depend_on_orientation() {
    let sights = [
        TurretSight { half_angle: 60.0, range: 16.0, grating_blocks: false },
        TurretSight { half_angle: 45.0, range: 8.0, grating_blocks: true },
    ];
    let mut checked = 0;
    for (name, p) in levels() {
        let (orig, _) = canonical(&p);
        let a = Level::from_puzzle(&orig);
        if a.turrets.is_empty() {
            continue;
        }
        checked += 1;
        for s in Sym::ALL.into_iter().skip(1) {
            let Ok((q, _)) = variant(&orig, s) else { continue };
            let b = Level::from_puzzle(&q);
            for sight in sights {
                for c in a.cells().filter(|&c| !a.solid(c)) {
                    assert_eq!(
                        danger(&a, sight, c) != 0,
                        danger(&b, sight, s.cell(c, a.size)) != 0,
                        "{name} {} {c:?} (half angle {})",
                        s.name(),
                        sight.half_angle
                    );
                }
            }
        }
    }
    assert!(checked > 10, "only {checked} levels with turrets");
}
