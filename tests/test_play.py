"""Play-harness regressions: .venv/Scripts/python -m unittest discover -s tests -p test_play.py

The parts that decide what the player does about a laser beam, a face whose middle it can't
see, and a shut exit door. Nothing here talks to the game: `Game` is made without a console,
and the few methods that would reach the game are stubbed.
"""

from pathlib import Path
import sys
import time as real_time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import play


class Clock:
    """Time that only moves when it is asked for, so a wait takes none."""

    def __init__(self):
        self.t = 1000.0

    def time(self):
        self.t += 0.1
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def bare_game(**attrs):
    """A Game with no console: only what the method under test reads."""
    g = object.__new__(play.Game)
    g.events = []
    g.leg_no = 0
    g.last = None
    g.held = set()
    g.jumped = False
    g.hurt_at = 0.0
    g.exit_wait = play.EXIT_WAIT
    g.top_speed = 175.0
    g.event = lambda kind, text, shot=True, **extra: g.events.append((kind, text))
    for k, v in attrs.items():
        setattr(g, k, v)
    return g


class HeadingTests(unittest.TestCase):
    def test_cardinal_angles(self):
        self.assertEqual([round(v, 6) for v in play.heading([0, 0, 0])], [1.0, 0.0, 0.0])
        self.assertEqual([round(v, 6) for v in play.heading([0, 90, 0])], [0.0, 1.0, 0.0])
        self.assertEqual([round(v, 6) for v in play.heading([0, 180, 0])], [-1.0, 0.0, 0.0])
        # Source pitch is down-positive: -90 points straight up.
        self.assertEqual([round(v, 6) for v in play.heading([-90, 0, 0])], [0.0, 0.0, 1.0])


class BeamOverTests(unittest.TestCase):
    def test_height_over_the_feet_of_a_beam_across_the_way(self):
        lasers = [{"o": [0.0, 0.0, 164.0], "a": [0.0, 0.0, 0.0]}]
        self.assertAlmostEqual(play.beam_over(lasers, [300.0, 0.0, 100.0]), 64.0)

    def test_a_beam_pointing_away_or_past_is_not_found(self):
        away = [{"o": [0.0, 0.0, 164.0], "a": [0.0, 180.0, 0.0]}]
        self.assertIsNone(play.beam_over(away, [300.0, 0.0, 100.0]))
        past = [{"o": [0.0, 0.0, 164.0], "a": [0.0, 0.0, 0.0]}]
        self.assertIsNone(play.beam_over(past, [300.0, 200.0, 100.0]))

    def test_a_beam_straight_up_is_not_found(self):
        up = [{"o": [300.0, 0.0, 0.0], "a": [-90.0, 0.0, 0.0]}]
        self.assertIsNone(play.beam_over(up, [300.0, 0.0, 0.0]))

    def test_the_lowest_crossing_beam_wins(self):
        lasers = [{"o": [0.0, 0.0, 164.0], "a": [0.0, 0.0, 0.0]},
                  {"o": [0.0, 0.0, 126.0], "a": [0.0, 0.0, 0.0]}]
        self.assertAlmostEqual(play.beam_over(lasers, [300.0, 0.0, 100.0]), 26.0)

    def test_a_beam_on_a_slope_is_measured_where_it_crosses(self):
        lasers = [{"o": [0.0, 0.0, 200.0], "a": [45.0, 0.0, 0.0]}]  # 45 degrees down, +x
        self.assertAlmostEqual(play.beam_over(lasers, [100.0, 0.0, 0.0]), 100.0, places=3)


class AimPointsTests(unittest.TestCase):
    def test_the_middle_comes_first_and_the_rest_stay_on_the_face(self):
        points = play.aim_points([64.0, 128.0, 640.0], "+z", spread=32.0)
        self.assertEqual(points[0], (64.0, 128.0, 640.0))
        self.assertEqual(len(points), 5)
        for q in points:
            self.assertEqual(q[2], 640.0)                        # the face's own plane
            self.assertLessEqual(abs(q[0] - 64.0), 32.0 + 1e-9)  # inside the voxel
            self.assertLessEqual(abs(q[1] - 128.0), 32.0 + 1e-9)

    def test_a_wall_face_is_offset_along_the_wall_and_up(self):
        points = play.aim_points([0.0, 128.0, 640.0], "-y")
        self.assertTrue(all(q[1] == 128.0 for q in points))
        self.assertEqual({round(q[0]) for q in points}, {0, -32, 32})
        self.assertEqual({round(q[2] - 640.0) for q in points}, {0, -32, 32})


class ClearAimTests(unittest.TestCase):
    FACE = {"cell": [6, 4, 10], "dir": "+z", "centre": [832.0, 576.0, 1408.0]}
    EYE = [800.0, 576.0, 100.0]

    def test_an_offset_point_is_taken_when_the_middle_is_blocked(self):
        tried = []

        def trace_line(a, b):
            tried.append(tuple(round(v) for v in b))
            return 0.4 if len(tried) == 1 else 1.0

        g = bare_game(trace_line=trace_line)
        aim = g.clear_aim(self.FACE, list(self.FACE["centre"]), self.EYE)
        self.assertNotEqual(list(aim), self.FACE["centre"])
        self.assertEqual(aim[2], self.FACE["centre"][2])
        self.assertEqual(len(tried), 2)
        # Traced to just inside the room, not to the face itself.
        self.assertEqual(tried[0][2], round(self.FACE["centre"][2] - 4))

    def test_the_middle_stands_when_it_is_in_sight(self):
        g = bare_game(trace_line=lambda a, b: 1.0)
        target = list(self.FACE["centre"])
        self.assertEqual(g.clear_aim(self.FACE, target, self.EYE), target)

    def test_the_middle_stands_when_no_trace_comes_back(self):
        g = bare_game(trace_line=lambda a, b: None)
        target = list(self.FACE["centre"])
        self.assertEqual(g.clear_aim(self.FACE, target, self.EYE), target)

    def test_nothing_in_sight_falls_back_to_the_middle_and_says_so(self):
        g = bare_game(trace_line=lambda a, b: 0.3)
        target = list(self.FACE["centre"])
        self.assertEqual(g.clear_aim(self.FACE, target, self.EYE), target)
        self.assertEqual([k for k, _ in g.events], ["aim"])


class BeamPassTests(unittest.TestCase):
    FEET = {"o": [300.0, 0.0, 100.0], "v": [0.0, 0.0, 0.0]}
    TO = [500.0, 0.0, 100.0]

    def game(self, lasers, burns=()):
        """A game whose duck and jump record themselves; `burns` names the ways that get
        burnt again (the hurt moves on while they run)."""
        g = bare_game(lasers=lambda: lasers, obs=lambda timeout=3: {"o": [300.0, 0.0, 100.0]})
        g.ways = []

        def way(name):
            def go(s, p, **kw):
                g.ways.append(name)
                if name in burns:
                    g.hurt_at += 10.0
            return go

        g.duck_beam, g.jump_beam = way("duck"), way("jump")
        return g

    def test_a_high_beam_is_ducked_under(self):
        g = self.game([{"o": [0.0, 0.0, 164.0], "a": [0.0, 0.0, 0.0]}])
        g.beam_pass(self.FEET, self.TO)
        self.assertEqual(g.ways, ["duck"])

    def test_a_low_beam_is_jumped(self):
        g = self.game([{"o": [0.0, 0.0, 126.0], "a": [0.0, 0.0, 0.0]}])
        g.beam_pass(self.FEET, self.TO)
        self.assertEqual(g.ways, ["jump"])

    def test_a_beam_that_cannot_be_found_is_jumped_first_as_before(self):
        g = self.game([])
        g.beam_pass(self.FEET, self.TO)
        self.assertEqual(g.ways, ["jump"])
        self.assertIn("not found", g.events[0][1])

    def test_burnt_again_tries_the_other_way(self):
        g = self.game([{"o": [0.0, 0.0, 164.0], "a": [0.0, 0.0, 0.0]}], burns=("duck",))
        g.beam_pass(self.FEET, self.TO)
        self.assertEqual(g.ways, ["duck", "jump"])

    def test_burnt_both_ways_gives_up_after_two(self):
        g = self.game([], burns=("duck", "jump"))
        g.beam_pass(self.FEET, self.TO)
        self.assertEqual(g.ways, ["jump", "duck"])
        self.assertEqual(g.events[-1][1], "burnt ducking and jumping it")


class ExitDoorTests(unittest.TestCase):
    def tearDown(self):
        play.time = real_time

    def exit_game(self, opens_after=None, **attrs):
        """A game at a shut exit door that opens on the `opens_after`th look (the first 15 s
        are 150 looks on this clock)."""
        looks = [0]

        def obs(timeout=3):
            looks[0] += 1
            if opens_after is not None and looks[0] >= opens_after:
                g.jumped = True
            return {"o": [100.0, 0.0, 0.0], "v": [0.0, 0.0, 0.0], "eye": [100.0, 0.0, 64.0]}

        g = bare_game(obs=obs, turn=lambda pitch, yaw: None,
                      floor=lambda cell: [100.0, 0.0, 0.0],  # the player is at the door
                      exit_cell=[1, 0, 0], exit_normal="+x",
                      con=type("Con", (), {"send": lambda self, *a: None})(), **attrs)
        g.hold = lambda key: g.held.add(key)
        g.release = lambda *keys: [g.held.discard(k) for k in keys]
        g.looks = looks
        play.time = Clock()
        return g

    def test_a_door_that_opens_late_is_waited_for(self):
        g = self.exit_game(opens_after=200)
        self.assertTrue(g.act_exit({"normal": "+x"}))
        self.assertEqual([k for k, _ in g.events], ["waiting"])

    def test_a_door_that_stays_shut_is_still_stuck(self):
        g = self.exit_game(opens_after=None)
        with self.assertRaises(play.Stuck) as caught:
            g.act_exit({"normal": "+x"})
        self.assertIn("at its door (shut?)", str(caught.exception))
        self.assertEqual([k for k, _ in g.events], ["waiting"])

    def test_a_timed_escape_does_not_wait(self):
        g = self.exit_game(opens_after=200)
        with self.assertRaises(play.Stuck):
            g.act_exit({"normal": "+x", "patient": False})
        self.assertEqual(g.events, [])
        self.assertLess(g.looks[0], 200)

    def test_no_wait_asked_for_does_not_wait(self):
        g = self.exit_game(opens_after=200, exit_wait=0.0)
        with self.assertRaises(play.Stuck):
            g.act_exit({"normal": "+x"})
        self.assertEqual(g.events, [])


if __name__ == "__main__":
    unittest.main()
