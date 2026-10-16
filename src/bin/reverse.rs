//! Reverse generator: build a chamber backwards from the goal so it is solvable by
//! construction (plan.md, "self-play" -> "solvability by construction": reverse play).
//!
//! This is a **separate binary** (`reverse`, run with `cargo run --bin reverse`) so both it and
//! the forward `glados dataset` generator live in the crate at once and can be compared on the
//! same data. It only *adds* files: it borrows the library (`builder`, `tokens`, `dataset`) and
//! does not touch the forward generator, the solver, or `main.rs`.
//!
//! # Why reverse
//! The forward `dataset` loop samples random/mutated genomes and keeps the ones the solver can
//! prove solvable. That ties the *supply* of training levels to the solver's search: every kept
//! level passed a BFS, so the corpus is exactly "levels a conservative BFS finds in 50k states"
//! and the hard, human-authored shapes get dropped. Reverse generation flips the dependency:
//! we choose a **solution plan first** and build geometry that requires it, so solvability is a
//! property of the construction, not of a search. The solver is then only a *measurer* (for
//! labels/metrics), never the source of hardness.
//!
//! # The construction
//! A chamber is a row of `rooms` along +x, each `rw` wide, separated by a solid wall with one
//! one-cell doorway. A deployed angled panel fills each doorway, wired to a button in the room
//! *before* it: a deployed panel is solid, so it blocks the player and portal shots until the
//! room's button folds it, and every further panel is still deployed. Inside a room the button
//! sits on a two-voxel ledge reachable only through a portal pair (a white wall at floor level
//! and a white wall at ledge level), so each room needs at least one portal trip. The last
//! room's button is wired to the exit door.
//!
//! Growing goal-first (place the exit and its gate, then prepend the room before it, ...) is the
//! reverse-play direction: the forward plan is the construction read backwards. Difficulty is a
//! dial (`--rooms`): the minimum plan length grows with the room count, and for large chambers
//! the solver may run out of budget while the level is still provably solvable — exactly the
//! hard-but-labelled data the forward loop cannot produce.
//!
//! # Output
//! One JSON object per line, the same schema `glados dataset` writes (`tokens`, `n_tokens`,
//! `size`, `verdict`, and for solvable levels `min_len`, `needs_portals`, `metrics`), plus
//! `"src":"reverse"` and `"constructed_len"` (actions the construction spends; an upper bound on
//! the true minimum). `train/data.py` reads it as-is; the extra keys are ignored.

use std::io::Write;
use std::path::PathBuf;

use glados::builder::Builder;
use glados::dataset::{self, Label};
use glados::geom::Dir;
use glados::p2c::Puzzle;
use glados::rng::Rng;
use glados::tokens;

/// Chamber depth in y.
const D: i32 = 6;
/// Chamber height in z; the floor (z=0) stays solid, z=1..=H-1 is open.
const H: i32 = 6;
/// Room width in x. Narrowed automatically when more rooms must fit the 32-cell limit.
const RW_MAX: i32 = 6;
/// Narrowest room that still holds a ledge and the two portal walls. Below this the solver
/// starts reusing an earlier room's portal, which cuts actions per room instead of adding them.
const RW_MIN: i32 = 4;
/// The doorway's y cell (the door is one cell wide and tall).
const DOOR_Y: i32 = 2;
/// The ledge sits at the back: y=LEDGE_Y..=LEDGE_Y+1.
const LEDGE_Y: i32 = 4;
/// The ledge is this solid tall; the player stands at z=LEDGE_Z.
const LEDGE_H: i32 = 2;
const LEDGE_Z: i32 = LEDGE_H + 1;

/// The first x of room `k`; rooms are `rw` wide with a one-cell wall between them.
fn room_start(k: i32, rw: i32) -> i32 {
    k * (rw + 1)
}

/// The widest a room may be so `rooms` of them (plus walls) still fit the 32-cell token limit.
/// Returns `None` if even the narrowest room won't fit.
fn fit_width(rooms: i32) -> Option<i32> {
    if rooms < 1 {
        return None;
    }
    // rooms*(rw+1) - 1 <= 32  =>  rw <= 32/rooms - 1.
    let rw = (32 / rooms - 1).min(RW_MAX);
    (rw >= RW_MIN).then_some(rw)
}

/// How a room's button is pressed; both end with the player standing on the ledge (z=LEDGE_Z).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Gadget {
    /// A latching pedestal on the ledge: fire, fire, enter, press, drop.
    Latch,
    /// A floor button on the ledge fed by a cube lifted from the floor: fire, fire, pick up,
    /// enter, put down, drop.
    CubeButton,
}

impl Gadget {
    fn actions(self) -> usize {
        match self {
            Gadget::Latch => 5,
            Gadget::CubeButton => 6,
        }
    }
}

/// A constructed chamber and what the construction claims about it.
struct Built {
    puzzle: Puzzle,
    size: [i32; 3],
    /// Actions the construction itself spends, from the gadget costs. An upper bound on the
    /// true minimum: the solver often finds a shorter route (e.g. reusing an earlier portal).
    constructed: usize,
    gadgets: Vec<Gadget>,
}

impl Built {
    /// One gate between each pair of adjacent rooms.
    fn gates(&self) -> usize {
        self.gadgets.len().saturating_sub(1)
    }
}

/// Builds a chamber of `rooms` rooms (>= 1) from `seed`.
fn construct(seed: u64, rooms: i32) -> Built {
    assert!(rooms >= 1);
    let rw = fit_width(rooms).expect("too many rooms for the 32-cell limit");
    let mut rng = Rng::new(seed);
    let width = (rooms - 1) * (rw + 1) + rw;
    let mut b = Builder::new("Reverse chamber", [width, D, H]);

    // Carve the rooms.
    for k in 0..rooms {
        let s = room_start(k, rw);
        b.carve([s, 0, 1], [s + rw - 1, D - 1, H - 1]);
    }
    // One-cell doorways through the walls between rooms; a deployed panel fills each.
    for k in 0..rooms - 1 {
        let sep = room_start(k, rw) + rw;
        b.carve([sep, DOOR_Y, 1], [sep, DOOR_Y, 1]);
    }

    let mut gadgets = Vec::new();
    let mut constructed = 0;
    let mut senders: Vec<i32> = Vec::new();
    for k in 0..rooms {
        let s = room_start(k, rw);
        let e = s + rw - 1;

        // The ledge (a balcony at the back) and the two walls a portal pair needs: a white wall
        // at floor level on the near side and a white wall at ledge level on the far side.
        b.fill([e - 1, LEDGE_Y, 1], [e, LEDGE_Y + 1, LEDGE_H]);
        b.paint([s, LEDGE_Y, 1], [s, LEDGE_Y + 1, LEDGE_H], Dir::NegX, true);
        b.paint([e, LEDGE_Y, LEDGE_Z], [e, LEDGE_Y + 1, LEDGE_Z + 1], Dir::PosX, true);

        let gadget = if rng.chance(0.5) {
            Gadget::Latch
        } else {
            Gadget::CubeButton
        };
        let sender = match gadget {
            Gadget::Latch => b.pedestal([e - 1, LEDGE_Y, LEDGE_Z], None),
            Gadget::CubeButton => {
                b.cube_dropper([s + 1, LEDGE_Y, H - 1]);
                b.floor_button([e - 1, LEDGE_Y, LEDGE_Z])
            }
        };
        constructed += gadget.actions();
        gadgets.push(gadget);
        senders.push(sender);

        // The gate to the next room: a deployed panel filling the doorway, wired to this
        // room's button so it folds (and the way opens) only once the button is pressed.
        if k < rooms - 1 {
            let sep = e + 1;
            let panel = b.angled_panel([sep, DOOR_Y, 1], Dir::PosZ, Dir::PosX, 90, true, false, false);
            b.connect(sender, panel);
        }
    }

    b.entry([0, DOOR_Y, 1], Dir::PosX);
    let last = room_start(rooms - 1, rw) + rw - 1;
    let exit = b.exit([last, DOOR_Y, 1], Dir::NegX);
    // The last room's button gates the exit itself.
    b.connect(*senders.last().expect("at least one room"), exit);

    Built {
        puzzle: b.build(),
        size: [width, D, H],
        constructed,
        gadgets,
    }
}

/// The JSONL body fields, plus the reverse-specific ones.
fn json_line(built: &Built, tokens: &[String], label: &Label) -> String {
    format!(
        "{{\"tokens\":{},\"n_tokens\":{},\"size\":[{},{},{}],{},\"src\":\"reverse\",\
         \"constructed_len\":{},\"rooms\":{},\"gates\":{}}}",
        dataset::json_str(&tokens.join(" ")),
        tokens.len(),
        built.size[0],
        built.size[1],
        built.size[2],
        label.json_fields(),
        built.constructed,
        built.gadgets.len(),
        built.gates(),
    )
}

struct Args {
    seed: u64,
    count: usize,
    rooms: i32,
    rooms_max: i32,
    budget: usize,
    out: Option<PathBuf>,
    dump: Option<PathBuf>,
    require_solvable: bool,
}

const USAGE: &str = "usage: glados-reverse [options]
  --seed N               base seed (default 0)
  --count N              levels to build (default 100)
  --rooms N              fewest rooms per chamber, >= 1 (default 3); the difficulty dial
  --rooms-max N          most rooms per chamber (default: same as --rooms); each level draws a
                         count in [--rooms, --rooms-max]
  --budget N             solver state budget for labelling (default 50000)
  --out FILE             write JSONL (default: stdout)
  --dump DIR             also write each chamber as DIR/reverse-<seed>-<i>.p2c
  --require-solvable     only emit chambers the solver proves solvable";

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args {
        seed: 0,
        count: 100,
        rooms: 3,
        rooms_max: 3,
        budget: 50_000,
        out: None,
        dump: None,
        require_solvable: false,
    };
    let mut i = 0;
    let num = |v: Option<&String>, what: &str| -> Result<u64, String> {
        v.ok_or_else(|| format!("{what} needs a value"))?
            .parse()
            .map_err(|_| format!("{what}: not a number"))
    };
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                a.seed = num(args.get(i + 1), "--seed")?;
                i += 2;
            }
            "--count" => {
                a.count = num(args.get(i + 1), "--count")? as usize;
                i += 2;
            }
            "--rooms" => {
                a.rooms = num(args.get(i + 1), "--rooms")?.max(1) as i32;
                i += 2;
            }
            "--rooms-max" => {
                a.rooms_max = num(args.get(i + 1), "--rooms-max")?.max(1) as i32;
                i += 2;
            }
            "--budget" => {
                a.budget = num(args.get(i + 1), "--budget")? as usize;
                i += 2;
            }
            "--out" => {
                a.out = Some(PathBuf::from(
                    args.get(i + 1).ok_or("--out needs a value")?,
                ));
                i += 2;
            }
            "--dump" => {
                a.dump = Some(PathBuf::from(
                    args.get(i + 1).ok_or("--dump needs a value")?,
                ));
                i += 2;
            }
            "--require-solvable" => {
                a.require_solvable = true;
                i += 1;
            }
            other => return Err(format!("unknown argument {other:?}\n{USAGE}")),
        }
    }
    Ok(a)
}

#[derive(Default)]
struct Stats {
    built: usize,
    encoded: usize,
    emitted: usize,
    solvable: usize,
    uncertain: usize,
    unsolvable: usize,
    invalid: usize,
    unsupported: usize,
    min_len_sum: usize,
}

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return std::process::ExitCode::SUCCESS;
    }
    let a = match parse(&args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return std::process::ExitCode::from(2);
        }
    };

    if let Some(dir) = &a.dump {
        if let Err(e) = std::fs::create_dir_all(dir) {
            eprintln!("--dump {}: {e}", dir.display());
            return std::process::ExitCode::FAILURE;
        }
    }
    let (rmin, rmax) = (a.rooms.min(a.rooms_max), a.rooms.max(a.rooms_max));
    match fit_width(rmax) {
        Some(rw) => eprintln!(
            "reverse: {rmin}..{rmax} room(s), room width {rw}, chambers up to {} wide",
            (rmax * (rw + 1)) - 1
        ),
        None => {
            let max = (1..=32).rev().find(|r| fit_width(*r).is_some()).unwrap_or(1);
            eprintln!("--rooms-max {rmax}: too many for the 32-cell limit (max {max})");
            return std::process::ExitCode::from(2);
        }
    }
    let mut out: Box<dyn Write> = match &a.out {
        Some(path) => match std::fs::File::create(path) {
            Ok(f) => Box::new(std::io::BufWriter::new(f)),
            Err(e) => {
                eprintln!("{}: {e}", path.display());
                return std::process::ExitCode::FAILURE;
            }
        },
        None => Box::new(std::io::stdout().lock()),
    };

    let mut stats = Stats::default();
    for i in 0..a.count {
        let seed = a.seed.wrapping_add(i as u64);
        let rooms = Rng::new(seed ^ 0x51_ed_0000).range(rmin, rmax);
        let built = construct(seed, rooms);
        stats.built += 1;
        let tokens = match tokens::encode(&built.puzzle) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("seed {seed}: does not encode: {e}");
                continue;
            }
        };
        stats.encoded += 1;
        let label = dataset::label(&built.puzzle, a.budget, seed);
        match &label {
            Label::Solvable { min_len, .. } => {
                stats.solvable += 1;
                stats.min_len_sum += *min_len;
            }
            Label::Uncertain(_) => stats.uncertain += 1,
            Label::Unsolvable => stats.unsolvable += 1,
            Label::Invalid(_) => stats.invalid += 1,
            Label::Unsupported(_) => stats.unsupported += 1,
        }
        if a.require_solvable && !matches!(label, Label::Solvable { .. }) {
            continue;
        }
        if let Some(dir) = &a.dump {
            let path = dir.join(format!("reverse-{seed}-{i}.p2c"));
            if let Err(e) = std::fs::write(&path, built.puzzle.to_p2c()) {
                eprintln!("{}: {e}", path.display());
            }
        }
        writeln!(out, "{}", json_line(&built, &tokens, &label)).expect("write output");
        stats.emitted += 1;
    }
    out.flush().ok();

    let mean = |n: usize, d: usize| if d == 0 { 0.0 } else { n as f64 / d as f64 };
    eprintln!(
        "reverse: built {} ({:.0}% encoded), emitted {}\n  \
         solvable {} ({:.0}%), uncertain {} ({:.0}%), unsolvable {}, invalid {}, unsupported {}\n  \
         mean min_len {:.1} (over solvable)",
        stats.built,
        100.0 * mean(stats.encoded, stats.built),
        stats.emitted,
        stats.solvable,
        100.0 * mean(stats.solvable, stats.encoded),
        stats.uncertain,
        100.0 * mean(stats.uncertain, stats.encoded),
        stats.unsolvable,
        stats.invalid,
        stats.unsupported,
        mean(stats.min_len_sum, stats.solvable),
    );
    std::process::ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point: a constructed chamber is solvable, and the solver agrees.
    #[test]
    fn constructed_chambers_are_solvable() {
        for seed in 0..16 {
            let built = construct(seed, 2);
            tokens::encode(&built.puzzle).expect("a constructed chamber encodes");
            match dataset::label(&built.puzzle, 200_000, seed) {
                Label::Solvable { min_len, .. } => assert!(min_len > 0),
                other => panic!("seed {seed}: expected solvable, got {other:?}"),
            }
        }
    }

    /// More rooms must mean a longer plan, and the gate count must match the room count.
    #[test]
    fn difficulty_grows_with_rooms() {
        let small = construct(3, 1);
        let big = construct(3, 4);
        assert_eq!(small.gates(), 0);
        assert_eq!(big.gates(), 3);
        let l_small = match dataset::label(&small.puzzle, 200_000, 3) {
            Label::Solvable { min_len, .. } => min_len,
            other => panic!("{other:?}"),
        };
        let l_big = match dataset::label(&big.puzzle, 200_000, 3) {
            Label::Solvable { min_len, .. } => min_len,
            other => panic!("{other:?}"),
        };
        assert!(
            l_big > l_small,
            "4 rooms ({l_big}) should beat 1 room ({l_small})"
        );
    }

    /// Every room's button is wired to exactly one gate (the panel to the next room, or the
    /// exit in the last room), so the chamber is a chain with no way around: one `>` wire and
    /// one `> panel` per non-final room, plus one `> exit`.
    #[test]
    fn gates_are_wired_in_series() {
        for rooms in 1..=4 {
            let built = construct(1, rooms);
            assert_eq!(built.gadgets.len(), rooms as usize);
            assert_eq!(built.gates(), (rooms - 1) as usize);
            let toks = tokens::encode(&built.puzzle).expect("encodes");
            let arrows = toks.iter().filter(|t| *t == ">").count();
            let to_panel = toks
                .windows(2)
                .filter(|w| w[0] == ">" && w[1] == "panel")
                .count();
            let to_exit = toks
                .windows(2)
                .filter(|w| w[0] == ">" && w[1] == "exit")
                .count();
            assert_eq!(arrows, rooms as usize, "rooms={rooms}: one wire per room");
            assert_eq!(to_panel, (rooms - 1) as usize, "rooms={rooms}");
            assert_eq!(to_exit, 1, "rooms={rooms}");
        }
    }
}
