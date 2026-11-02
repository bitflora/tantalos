//! Workshop levels our solver called unsolvable until a play-tester showed how
//! (docs/solver-gaps-2026-10.md, "Play-tested small levels").

use glados::level::Level;
use glados::plan::{check, decode, encode};
use glados::route::route;
use glados::solver::{Action, Mode, Model, Verdict};

const BUDGET: usize = 50_000;

/// 3818309212 "???": stairs facing a wall, left sideways off their top step.
const STAIRS_SIDEWAYS: &str = "size 6 8 5 open 0 3 0 2 4 2 0 5 0 0 5 2 0 2 1 2 7 2 1 4 3 1 7 4 1 0 4 5 0 4 1 1 4 1 7 4 4 1 4 5 1 4 0 2 4 2 2 4 entry 1 0 4 +y exit 1 2 1 +y pedestal 0 2 4 inf pedestal 4 1 4 inf > panel 0 plate 1 7 1 -y 1 3 4 +z panel 1 1 4 +z -y 6 on stairs 1 3 0 +x on";

/// 68464713 "Smallest Puzzle in the Multiverse": two fizzlers crossing in a 3x2 room. Fetch the
/// cube across one while a 3 s pedestal holds it off, then reach for it across the other from
/// that one's floor button.
const SMALLEST: &str = "size 3 5 2 open 1 0 0 1 4 0 2 1 0 2 2 0 0 2 0 1 3 0 2 1 1 2 1 1 white +z 1 0 0 2 0 1 white -z 2 2 0 2 4 0 entry 1 0 0 +y exit 1 4 0 -y fizzler 2 1 0 -x +z 1 on fizzler 2 2 0 -y +z 1 on dropper 2 1 1 standard button 1 0 0 weighted > exit button 1 3 0 weighted > field 1 pedestal 1 4 0 3 > field 0";

/// 3094057883 "Simple Test 01": momentum. Walk into a floor portal under a wall portal, and
/// move the wall one to a tilted panel while falling back; later loops carry you (and the
/// cube, caught as it falls past into goo) up to the button and the exit.
const SIMPLE_TEST_01: &str = "size 10 11 11 open 6 0 1 6 6 4 2 1 1 9 6 10 1 3 1 9 3 3 0 3 2 9 4 3 1 5 6 9 6 7 6 7 6 9 9 10 6 10 8 9 10 10 7 0 9 7 10 9 1 3 9 9 4 9 goo 6 0 0 6 0 0 white +x 9 1 1 9 4 5 9 5 2 9 10 5 white -x 2 2 1 9 10 4 2 1 3 9 4 5 white +y 2 6 2 9 10 4 5 6 5 9 10 5 white -y 5 1 1 9 6 5 2 1 3 9 6 5 white -z 5 1 1 9 4 10 2 2 1 9 2 10 3 3 1 9 4 10 2 4 1 9 4 10 entry 3 1 1 +y exit 7 10 8 -y grating 2 5 5 4 6 5 +z +x glass 6 7 6 6 7 10 -y +z grating 6 9 7 9 9 7 +z +x dropper 6 0 4 standard button 6 8 6 weighted > exit > panel 1 > panel 3 > stairs 1 pedestal 1 5 6 inf > panel 0 pedestal 5 1 1 inf > dropper 0 panel 8 1 5 +y -z 3 off panel 9 5 6 -x -z 6 off panel 9 6 3 -x -z 4 on panel 9 6 6 -x -z 6 off stairs 2 3 1 -x on stairs 9 8 6 -y off";

/// 3543260079 "My First Test Chamber": a turret sees the entrance from behind a laser field;
/// a piston (up until its button is pressed) lifts the player to the second cube once they
/// reach through a portal from it and take the cube off that button.
const MY_FIRST: &str = "size 10 10 4 open 0 0 0 8 6 3 0 7 0 4 7 3 8 7 0 8 9 3 6 7 1 6 8 3 9 3 3 9 4 3 white +x 8 0 0 8 9 3 4 7 0 4 9 3 black -x 6 7 1 7 9 3 white +y 0 7 0 9 7 3 8 9 0 9 9 3 white -y 0 0 0 9 2 3 black -z 6 6 0 6 9 3 9 3 3 9 9 3 entry 1 0 0 +y exit 2 7 0 -y laserfield 8 6 0 +z +x 1 on dropper 1 3 3 standard dropper 6 8 3 companion button 0 7 0 weighted > exit button 4 7 0 weighted > exit button 8 9 0 weighted > piston 0 turret 8 7 0 2 piston 6 6 0 +z 0 1 off";

/// 1498933548 "Laser Management": a reflection cube set down on the floor button holds it and
/// sends the beam (through portals) to the exit's receptacle.
const LASER_MANAGEMENT: &str = "size 8 6 3 open 1 0 0 7 5 1 0 2 1 7 3 1 1 0 2 3 5 2 5 0 2 7 5 2 white +x 7 0 0 7 5 2 white -x 1 0 0 3 1 2 1 4 0 3 5 2 white +y 5 5 0 7 5 2 white -y 1 0 0 3 5 2 entry 2 5 0 -y exit 7 4 0 -x fizzler 4 0 0 +y +z 2 on dropper 6 1 2 reflection button 2 2 0 weighted > exit > field 0 pedestal 5 1 0 inf > dropper 0 laser 6 0 0 +y -z on catcher 2 5 2 -y -z > field 0 catcher 7 2 0 -x -z > exit";

/// 68655995 "The Grate Escape": the funnel beam out of a wall portal passes grating to the
/// player's column; moving that portal up a panel at a time lifts them to the exit ledge.
const GRATE_ESCAPE: &str = "size 3 3 9 open 1 0 0 2 2 6 1 0 7 1 2 8 2 0 7 2 0 8 2 2 7 2 2 8 0 0 8 2 1 8 white -y 1 0 0 2 0 8 white +z 2 1 6 2 2 6 entry 1 1 0 +x exit 2 1 8 -x grating 1 0 0 2 0 6 +y +z grating 1 0 7 1 0 8 +y -z grating 1 2 0 2 2 6 -y +z grating 1 2 7 1 2 8 -y -z grating 2 0 8 2 0 8 +y -z grating 2 1 0 2 1 6 -x +z grating 2 2 8 2 2 8 -y -z funnel 2 1 0 +z on";

/// 68651531 "Mari0 - Test chamber 04 (Update 1)": gel dropped through ceiling portals covers the
/// floor; bounces off it, fed back in through the portals, carry the player up to the exit ledge.
const MARI0: &str = "size 2 19 13 open 1 1 0 1 12 7 0 4 2 1 5 2 1 6 8 1 13 12 1 14 8 1 18 9 1 0 9 1 3 10 1 4 10 1 13 12 1 14 10 1 14 10 1 17 10 1 18 10 1 3 11 1 13 12 white +z 1 5 12 1 9 12 white -z 1 2 0 1 3 7 entry 1 0 9 +y exit 1 18 8 -y gel 1 2 0 +z bounce stream on stairs 1 2 9 +y on";

/// 3718840820 "Flying a Timing": portal the laser to its catcher (the laser field goes off),
/// fetch the cube, put it on the button that raises the stairs; up them onto the first of four
/// faith plates, which throw the player across the goo. There, press the 15 s pedestal that
/// swings a walkway out under the respawning cube dropper, ride the lift up, take the cube off
/// the walkway and come back down with it to the exit's button.
const FLYING_A_TIMING: &str = "size 27 8 7 s7 | s1 o4 s2 | s1 o4 s2 | s1 o4 s2 | g1 o4 s2 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | s1 o4 s2 | s1 o4 s2 | s7 | / s7 | s1 o4 s2 | s1 o4 s2 | s1 o4 s2 | g1 o4 s2 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | s1 o4 s2 | s1 o4 s2 | s7 | / s4 o1 s2 | s1 o4 s2 | s1 o4 s2 | s1 o4 s2 | s2 o3 s1 o1 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | s1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | s1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | s1 o4 s1 o1 | s1 o4 s2 | s4 o1 s2 | / s7 | s1 o4 s2 | s1 o4 s2 | s1 o4 s2 | g1 o4 s2 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | s1 o4 s2 | s1 o4 s2 | s4 o1 s2 | / s7 | s1 o4 s2 | s1 o4 s2 | s1 o4 s2 | g1 o4 s2 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | g1 o6 | s1 o4 s2 | s1 o4 s2 | s7 | / s7 | s7 | s1 o1 s5 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | / s7 | s7 | s1 o1 s5 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | / s7 | s7 | s1 o1 s5 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | s7 | / white +z 1 4 4 1 4 4 white -z 3 4 1 3 4 1 entry 1 2 1 +x exit 25 2 1 -x laserfield 2 5 1 -x +z 1 on dropper 19 0 6 companion button 1 0 1 weighted > stairs 0 button 25 4 1 weighted > exit pedestal 25 0 1 15 > panel 0 > panel 1 > panel 2 > panel 3 pedestal 24 0 1 inf > dropper 0 cube 2 7 1 companion laser 1 4 1 +z +y on catcher 3 4 4 -z +y > field 0 plate 14 4 2 -y 17 2 1 +z plate 17 2 1 +z 24 2 1 +z plate 4 2 2 +z 10 2 1 +z plate 10 2 1 +z 14 4 2 -y panel 19 0 3 +y -z 6 off panel 20 0 3 +y -z 6 off panel 21 0 3 +y -z 6 off panel 22 0 3 +y -z 6 off panel 24 4 1 +z -x 6 on panel 25 4 2 -x +z 6 on stairs 2 2 1 +x off rail 23 0 0 23 0 3 osc";

fn level(tokens: &str) -> Level {
    Level::from_puzzle(&glados::tokens::decode(tokens).unwrap())
}

/// Solves `tokens` with conservative rules; its route's plan tokens check as solved.
fn solved(tokens: &str) -> Vec<Action> {
    let level = level(tokens);
    let model = Model::new(&level, Mode::Conservative).unwrap();
    let Verdict::Solved(plan) = model.solve(BUDGET).verdict else {
        panic!("not solved:\n{tokens}");
    };
    let r = route(&level, BUDGET).unwrap();
    let text = encode(&r);
    let c = check(&model, &decode(&text).unwrap());
    assert!(c.solved, "{text}: {:?}", c.problem);
    plan
}

#[test]
fn stairs_left_sideways() {
    let plan = solved(STAIRS_SIDEWAYS);
    assert!(plan.iter().any(|a| matches!(a, Action::Move { kind: glados::regions::Move::Stairs, .. })));
}

#[test]
fn cube_fetched_in_a_dash_and_reached_for_from_a_button() {
    let plan = solved(SMALLEST);
    assert!(plan.iter().any(|a| matches!(a, Action::Fetch { .. })), "{plan:?}");
    assert!(plan.iter().any(|a| matches!(a, Action::Reach { .. })), "{plan:?}");
}

#[test]
fn no_reach_while_the_button_leaves_the_fizzler_on() {
    // Unwired: standing on the floor button no longer turns the fizzler between it and the
    // cube off (fizzler 1 has no input and stays on).
    let tokens = SMALLEST.replace("button 1 3 0 weighted > field 1", "button 1 3 0 weighted");
    let model = Model::new(&level(&tokens), Mode::Liberal).unwrap();
    let mut s = model.initial();
    // Search everything reachable; no Reach anywhere.
    let mut seen = vec![s.clone()];
    let mut i = 0;
    while i < seen.len() && i < BUDGET {
        s = seen[i].clone();
        for (a, n) in model.successors(&s) {
            assert!(!matches!(a, Action::Reach { .. }), "{a:?}");
            if !seen.contains(&n) {
                seen.push(n);
            }
        }
        i += 1;
    }
}

#[test]
fn portal_loops_and_a_caught_cube() {
    let model = Model::new(&level(SIMPLE_TEST_01), Mode::Liberal).unwrap();
    let Verdict::Solved(plan) = model.solve(BUDGET).verdict else {
        panic!("not solved");
    };
    assert!(
        plan.iter().any(|a| matches!(a, Action::Loop { refire: Some(_), .. })),
        "{plan:?}"
    );
    // Conservative rules don't trust the 4-voxel fling out of the tilted panel (its trials
    // land in different places), so the level is uncertain, not solvable.
}

#[test]
fn step_out_of_a_turret_and_reach_through_a_portal() {
    let level = level(MY_FIRST);
    let model = Model::new(&level, Mode::Conservative).unwrap();
    let Verdict::Solved(plan) = model.solve(BUDGET).verdict else {
        panic!("not solved");
    };
    assert!(plan.iter().any(|a| matches!(a, Action::PortalReach { .. })), "{plan:?}");
}

#[test]
fn reflection_cube_on_a_button_holds_it_and_redirects() {
    let plan = solved(LASER_MANAGEMENT);
    assert!(plan.iter().any(|a| matches!(a, Action::PutDown { at: glados::solver::CubeLoc::Spot(_) })), "{plan:?}");
}

#[test]
fn funnel_beam_lifts_the_player_as_its_portal_moves_up() {
    // Not routed yet (`glados-score route` can't ground a ride up).
    let model = Model::new(&level(GRATE_ESCAPE), Mode::Conservative).unwrap();
    let Verdict::Solved(plan) = model.solve(BUDGET).verdict else {
        panic!("not solved");
    };
    assert!(plan.iter().any(|a| matches!(a, Action::Move { kind: glados::regions::Move::Funnel, .. })), "{plan:?}");
}

#[test]
fn bounce_gel_under_ceiling_portals_carries_the_player_up() {
    // Liberal rules only: the conservative ones don't trust the rebound height.
    let model = Model::new(&level(MARI0), Mode::Liberal).unwrap();
    let Verdict::Solved(plan) = model.solve(BUDGET).verdict else {
        panic!("not solved");
    };
    assert!(plan.iter().any(|a| matches!(a, Action::Move { kind: glados::regions::Move::Bounce, .. })), "{plan:?}");
}

#[test]
fn stairs_onto_a_plate_and_a_timed_lift_to_a_respawning_cube() {
    use glados::regions::Move;
    let level = level(FLYING_A_TIMING);
    // The lift's platform is the top of its voxels: it carries the player from the floor of the
    // exit side, a voxel over its track's bottom, up to the walkway.
    assert_eq!(level.rails[0].stops().first(), Some(&[23, 0, 1]));
    assert_eq!(level.rails[0].stops().last(), Some(&[23, 0, 4]));
    // Liberal rules only: the dash rides the lift as soon as it's there, and the cube is on the
    // walkway once the walkway is out.
    let model = Model::new(&level, Mode::Liberal).unwrap();
    let Verdict::Solved(plan) = model.solve(BUDGET).verdict else {
        panic!("not solved");
    };
    assert!(plan.iter().any(|a| matches!(a, Action::Move { kind: Move::Plate, .. })), "{plan:?}");
    assert!(plan.iter().any(|a| matches!(a, Action::Fetch { .. })), "{plan:?}");
    // With the exit's button unwired, the stairs and plates alone get there: conservative
    // rules solve it, and the route climbs the stairs onto the plate.
    let open_exit = FLYING_A_TIMING.replace("button 25 4 1 weighted > exit", "button 25 4 1 weighted");
    let plan = solved(&open_exit);
    assert!(plan.iter().any(|a| matches!(a, Action::Move { kind: Move::Plate, .. })), "{plan:?}");
}

/// 3372907955 "SUPER SIMPLE #2": two 3 s pedestals side by side both open the exit. Press one,
/// then the other, and walk out while both are on.
const SUPER_SIMPLE_2: &str = "size 3 6 2 s2 | o2 | s2 | / s2 | o2 | s2 | / s2 | o2 | s1 o1 | / s2 | o2 | s1 o1 | / o1 s1 | o2 | s2 | / s2 | o2 | s2 | / entry 1 0 0 +y exit 1 5 0 -y pedestal 0 4 0 3 > exit pedestal 1 4 0 3 > exit";

#[test]
fn press_two_timed_pedestals_and_run_out() {
    // Not routed yet (the play harness has no leg for a second press).
    let model = Model::new(&level(SUPER_SIMPLE_2), Mode::Conservative).unwrap();
    let Verdict::Solved(plan) = model.solve(BUDGET).verdict else {
        panic!("not solved");
    };
    assert!(matches!(plan.as_slice(), [Action::DoubleDash { escaped: true, .. }]), "{plan:?}");
    // Not if the second pedestal is too far to reach in time.
    let far = SUPER_SIMPLE_2.replace("pedestal 1 4 0 3", "pedestal 1 0 0 3");
    let model = Model::new(&level(&far), Mode::Liberal).unwrap();
    assert_eq!(model.solve(BUDGET).verdict, Verdict::Unsolvable);
}

/// 3817082781 "sp_timed_lazer_change": the laser crosses the funnel's column in mid-air. Put the
/// reflection cube into the funnel facing the catcher and walk to the exit while it rises: it
/// turns the beam onto the catcher for the moment it passes.
const TIMED_LAZER_CHANGE: &str = "size 4 6 4 o3 s1 | o4 | o4 | s4 | / o4 | o4 | o4 | s4 | / o4 | o4 | o4 | s3 o1 | / o4 | o4 | o4 | s3 o1 | / o4 | o4 | o4 | s4 | / o4 | o4 | o4 | s4 | / entry 1 0 0 +y exit 2 5 0 -y cube 0 0 0 reflection laser 0 3 2 +x center on catcher 1 0 2 +y center > exit funnel 1 3 0 +z on";

#[test]
fn reflection_cube_rising_in_a_funnel_sweeps_the_beam_onto_the_catcher() {
    // Liberal rules only: the cube reaches the beam 1 s after it goes in (at the funnel's
    // assumed 250 u/s), which a conservative walk to the exit doesn't beat. Not routed yet.
    let model = Model::new(&level(TIMED_LAZER_CHANGE), Mode::Liberal).unwrap();
    let Verdict::Solved(plan) = model.solve(BUDGET).verdict else {
        panic!("not solved");
    };
    assert!(matches!(plan.as_slice(), [Action::PickUp { .. }, Action::Sweep { .. }]), "{plan:?}");
    let model = Model::new(&level(TIMED_LAZER_CHANGE), Mode::Conservative).unwrap();
    assert!(!matches!(model.solve(BUDGET).verdict, Verdict::Solved(_)));
    // Not with the funnel off, nor pulling down.
    for other in ["funnel 1 3 0 +z off", "funnel 1 3 0 +z on rev"] {
        let tokens = TIMED_LAZER_CHANGE.replace("funnel 1 3 0 +z on", other);
        let model = Model::new(&level(&tokens), Mode::Liberal).unwrap();
        assert_eq!(model.solve(BUDGET).verdict, Verdict::Unsolvable, "{other}");
    }
    // Nor with a turret in the swept beam's way to the catcher (on a block at (1,1,1)): a
    // laser takes several seconds to destroy a turret (play-tested, 2026-11-02), and the cube
    // crosses the beam for a moment. Beside the beam it's no obstacle.
    let block = |t: &str| {
        TIMED_LAZER_CHANGE
            .replacen("o4 | o4 | o4 | s4 | / o4", "o4 | o1 s1 o2 | o4 | s4 | / o4", 1)
            .replace("funnel 1 3 0 +z on", &format!("funnel 1 3 0 +z on {t}"))
    };
    let model = Model::new(&level(&block("turret 0 1 0 3")), Mode::Liberal).unwrap();
    assert!(matches!(model.solve(BUDGET).verdict, Verdict::Solved(_)));
    let model = Model::new(&level(&block("turret 1 1 2 3")), Mode::Liberal).unwrap();
    assert_eq!(model.solve(BUDGET).verdict, Verdict::Unsolvable);
}

/// 252742140 "Parkour, Parkour PARKOUR PARKOUR": strips of floor between one-voxel channels of
/// goo; a running jump clears each one.
const PARKOUR: &str = "size 4 7 5 open 0 0 1 2 6 4 3 2 4 3 3 4 goo 0 1 0 2 1 0 0 3 0 2 3 0 0 5 0 2 5 0 entry 1 0 1 +y exit 1 6 1 -y pedestal 2 6 1 inf > exit";

#[test]
fn jump_one_voxel_gaps() {
    let plan = solved(PARKOUR);
    let jumps = plan
        .iter()
        .filter(|a| matches!(a, Action::Move { kind: glados::regions::Move::Jump, .. }))
        .count();
    assert_eq!(jumps, 3, "{plan:?}");
    // Not two voxels.
    let wide = PARKOUR.replace("goo 0 1 0 2 1 0", "goo 0 1 0 2 2 0");
    let model = Model::new(&level(&wide), Mode::Liberal).unwrap();
    assert_eq!(model.solve(BUDGET).verdict, Verdict::Unsolvable);
}

/// 3666439290 "First chamber": a laser field cuts the player off from the exit; the only white
/// face is high on the wall on their side, and a conversion gel stream pours on the far side.
/// Portal the stream out of the wall: it paints the wall across the room, which sent on paints
/// the floor; a portal there drops the player out under the stream.
const FIRST_CHAMBER: &str = "size 9 6 4 open 0 0 0 7 5 3 8 2 3 8 3 3 white +y 0 5 3 0 5 3 entry 1 0 0 +y exit 6 5 0 -y laserfield 3 5 0 -y +z 3 on gel 7 0 0 +z portal stream on";

#[test]
fn conversion_gel_sent_on_through_painted_walls() {
    let plan = solved(FIRST_CHAMBER);
    assert!(plan.iter().any(|a| matches!(a, Action::Enter { .. })), "{plan:?}");
    // Not with the dropper off.
    let off = FIRST_CHAMBER.replace("stream on", "stream off");
    let model = Model::new(&level(&off), Mode::Liberal).unwrap();
    assert_eq!(model.solve(BUDGET).verdict, Verdict::Unsolvable);
}
