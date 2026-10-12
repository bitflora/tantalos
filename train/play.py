"""Play a level's solution in Portal 2 the way a player would, to watch it.

    python train/play.py fixtures/synthetic/goo_crossing.p2c --launch
    python train/play.py LEVEL.p2c... --port PORT --password PW     # a game you started yourself
    python train/play.py --spike fixtures/synthetic/worked_example.p2c --launch

`glados route` turns the solver's shortest plan into cells, faces and points (src/route.rs).
This loads the level into the Puzzle Maker (train/portal2.py), builds it, and plays it with the
player's own controls: `setang` to look (a perfect mouse, turning smoothly so it can be watched;
--fast snaps it, for validation runs) and +forward, +jump, +attack (blue),
+attack2 (orange) and +use, steering by the positions a small VScript reports each tick. The
step being played shows on screen. Each step's result is checked against the route; one that
didn't work out is put right by teleporting (setpos, a cube's SetOrigin, portal_place) and
counted as recovered, and the run carries on. The terminal gets one line per step.

The game is driven over its TCP console. --launch starts it (refusing if it's already running)
with -insecure and -netconport/-netconpassword, under portal2.py's safety rules, and quits it at
the end unless --keep. For a game you started yourself, add
`-insecure -netconport PORT -netconpassword PW` to its launch options and pass --port and
--password. --spike checks the console, VScript, coordinates, movement and the commands the
player uses on the first level, and prints what it found instead of playing.
"""
import argparse, json, math, re, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import portal2  # noqa: E402
from portal2 import Console, GameUnavailable, open_editor  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VOXEL = 128.0
SCRIPT = portal2.GAME / "portal2" / "scripts" / "vscripts" / "glados_play.nut"
LEVEL = portal2.STAGE / "play.p2c"
COLORS = ("blue", "orange")
FIRE_KEY = {"blue": "attack", "orange": "attack2"}
TURN_RATE = 270.0  # degrees a second, turning to look or to walk somewhere
MIN_SPEED = 50.0   # slowest walk while easing into a stop
PEDESTAL_TOP = 40.0  # a pedestal button's button, over the entity's origin (its base)
MOVES = ("forward", "back", "moveleft", "moveright")
DIRS = {"-x": (-1, 0, 0), "+x": (1, 0, 0), "-y": (0, -1, 0), "+y": (0, 1, 0), "-z": (0, 0, -1), "+z": (0, 0, 1)}

VSCRIPT = r"""// Written by train/play.py and deleted when it quits: reports what the player steers by.
// One line per thing, as long lines can be cut short.
function glados_v(v) { return "[" + v.x + "," + v.y + "," + v.z + "]" }
function glados_list(n, kind, cls) {
	local e = null
	while ((e = Entities.FindByClassname(e, cls)) != null)
		printl("GLADOS_ENT " + n + " " + kind + " {\"id\":" + e.entindex() + ",\"o\":" + glados_v(e.GetOrigin()) + ",\"a\":" + glados_v(e.GetAngles()) + "}")
}
function glados_obs(n) {
	local p = GetPlayer()
	if (p == null) {
		printl("GLADOS_OBS " + n + " null")
	} else {
		printl("GLADOS_OBS " + n + " {\"t\":" + Time() + ",\"o\":" + glados_v(p.GetOrigin()) + ",\"eye\":" + glados_v(p.EyePosition()) + ",\"v\":" + glados_v(p.GetVelocity()) + ",\"hp\":" + p.GetHealth() + "}")
		glados_list(n, "portals", "prop_portal")
		glados_list(n, "cubes", "prop_weighted_cube")
		glados_list(n, "turrets", "npc_portal_turret_floor")
		glados_list(n, "pedestals", "prop_button")
		glados_list(n, "pedestals", "prop_under_button")
	}
	printl("GLADOS_END " + n)
}
function glados_put(id, x, y, z) {
	local e = null
	while ((e = Entities.FindByClassname(e, "prop_weighted_cube")) != null) {
		if (e.entindex() == id) {
			e.SetOrigin(Vector(x, y, z))
			e.SetVelocity(Vector(0, 0, 0))
			return
		}
	}
}
function glados_names() {
	foreach (cls in ["linked_portal_door", "prop_testchamber_door", "prop_floor_button", "prop_floor_cube_button", "prop_floor_ball_button", "prop_button", "prop_under_button", "prop_weighted_cube", "npc_portal_turret_floor", "prop_portal"]) {
		local e = null
		while ((e = Entities.FindByClassname(e, cls)) != null)
			printl("GLADOS_NAME " + cls + " " + e.entindex() + " " + e.GetName() + " " + glados_v(e.GetOrigin()))
	}
}
function glados_arrival() {
	foreach (cls in ["trigger_once", "linked_portal_door", "func_tracktrain"]) {
		local e = null
		while ((e = Entities.FindByClassname(e, cls)) != null) {
			local name = e.GetName()
			if (name.find("leaving_elevator_trigger") != null || name == "@entrance_portal_elevator_side" || name.find("arrival_elevator-elevator_1") != null)
				printl("GLADOS_ARRIVAL " + name + " " + glados_v(e.GetOrigin()))
		}
	}
}
// The level's own start (what the editor's Build & Play does): the relay that teleports the
// player to the entry corridor and opens the entry door.
function glados_start() {
	local e = null
	while ((e = Entities.FindByClassname(e, "logic_relay")) != null) {
		local n = e.GetName()
		if (n.len() >= 15 && n.slice(n.len() - 15) == "player_start_rl") {
			DoEntFire(n, "Trigger", "", 0, null, null)
			printl("GLADOS_START " + n)
		}
	}
}
printl("GLADOS_SCRIPT loaded")
"""


class Stuck(Exception):
    """A move that didn't get where it was going."""


# --- geometry ------------------------------------------------------------------------------

def hdist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle_to(eye, target):
    """(pitch, yaw) in degrees looking from `eye` at `target`."""
    dx, dy, dz = (target[i] - eye[i] for i in range(3))
    return -math.degrees(math.atan2(dz, math.hypot(dx, dy))), math.degrees(math.atan2(dy, dx))


def yaw_of(d):
    return math.degrees(math.atan2(d[1], d[0]))


def fmt(p):
    return "(" + ", ".join(f"{v:.0f}" for v in p[:3]) + ")"


# --- the game ------------------------------------------------------------------------------

class Game:
    """The player in a running level: what it sees (via glados_play.nut) and its controls."""

    def __init__(self, con, origin=(0.0, 0.0, 0.0), turn_rate=TURN_RATE):
        self.con = con
        self.turn_rate = turn_rate  # degrees a second; math.inf snaps (--fast)
        self.origin = origin
        self.n = 0
        self.held = set()
        self.pitch = self.yaw = 0.0
        self.last = None
        self.jumped = False
        self.portal_ids = {}
        self.held_id = None     # the cube picked up, until it's let go
        self.hold_dist = 75.0   # how far in front of the eye a held cube hangs (measured)
        self.text_made = False
        out = con.ask(["cl_forwardspeed"], 5) or ""
        m = re.search(r'"cl_forwardspeed" = "([\d.]+)"', out)
        self.top_speed = float(m.group(1)) if m else 175.0

    # positions: route voxels -> world units
    def world(self, p):
        return [VOXEL * p[i] + self.origin[i] for i in range(3)]

    def floor(self, cell):
        return self.world([cell[0] + 0.5, cell[1] + 0.5, cell[2]])

    def in_cell(self, o, cell, slack=12):
        f = self.floor(cell)
        return (abs(o[0] - f[0]) <= 64 + slack and abs(o[1] - f[1]) <= 64 + slack
                and -16 <= o[2] - f[2] <= 48)

    # seeing
    def obs(self, timeout=3):
        """Where the player, portals, cubes and turrets are. Sets `jumped` when the player
        moved further than walking could since the last look (a portal or a linked door)."""
        self.n += 1
        n = self.n
        self.con.send(f"script glados_obs({n})")
        got = self.con.until(lambda l: l.strip() == f"GLADOS_END {n}", timeout)
        if got is None:
            raise GameUnavailable("the game stopped reporting positions")
        head = next((l for l in got[0] if l.startswith(f"GLADOS_OBS {n} ")), None)
        s = json.loads(head.split(" ", 2)[2]) if head else None
        if s is None:
            raise Stuck("no player")
        s.update(portals=[], cubes=[], turrets=[], pedestals=[])
        for l in got[0]:
            if l.startswith(f"GLADOS_ENT {n} "):
                _, _, kind, item = l.split(" ", 3)
                s[kind].append(json.loads(item))
        if self.last is not None:
            dt = max(s["t"] - self.last["t"], 0.0)
            if math.dist(s["o"], self.last["o"]) > 64 + 600 * dt:
                self.jumped = True
        self.last = s
        return s

    @staticmethod
    def speed(s):
        return math.hypot(s["v"][0], s["v"][1])

    def forward(self):
        p, y = math.radians(self.pitch), math.radians(self.yaw)
        return (math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), -math.sin(p))

    def holding(self, s):
        """The cube the player is holding, if any: the one picked up, while it hangs where a
        held cube does in front of the view, or near the eye and off the floor. (Looking down
        at a button, a held cube hangs low, so height alone doesn't tell; and a cube lying
        where the player looks isn't held, so only the picked-up one counts.)"""
        if self.held_id is None:
            return None
        f = self.forward()
        expect = [s["eye"][i] + f[i] * self.hold_dist for i in range(3)]
        for c in s["cubes"]:
            if c["id"] == self.held_id and (math.dist(c["o"], expect) < 64 or (
                    math.dist(c["o"], s["eye"]) < 110 and c["o"][2] > s["o"][2] + 20)):
                return c
        return None

    def let_go(self):
        """Drops the held cube (use), checking it went; True if it did."""
        for _ in range(2):
            self.tap("use", 0.15)
            time.sleep(1.2)
            if self.holding(self.obs()) is None:
                self.held_id = None
                return True
        return False

    def nearest_cube(self, s, p, within=1e9, skip=()):
        best = min((c for c in s["cubes"] if c["id"] not in skip),
                   key=lambda c: math.dist(c["o"], p), default=None)
        return best if best is not None and math.dist(best["o"], p) <= within else None

    def portal_on(self, color, face, s):
        """Whether a portal (this color's, once known) sits on `face`."""
        centre = self.world(face["centre"])
        axis = "xyz".index(face["dir"][1])
        for p in s["portals"]:
            if color in self.portal_ids and p["id"] != self.portal_ids[color]:
                continue
            o = p["o"]
            off = abs(o[axis] - centre[axis])
            along = math.sqrt(sum((o[i] - centre[i]) ** 2 for i in range(3) if i != axis))
            if off <= 8 and along <= 64:
                return True
        return False

    # controls
    def hold(self, key):
        if key not in self.held:
            self.con.send(f"+{key}")
            self.held.add(key)

    def release(self, *keys):
        for key in keys:
            if key in self.held:
                self.con.send(f"-{key}")
                self.held.discard(key)

    def release_all(self):
        self.release(*list(self.held))

    def tap(self, key, seconds=0.08):
        self.con.send(f"+{key}")
        time.sleep(seconds)
        self.con.send(f"-{key}")

    def look(self, pitch, yaw):
        """Sets the view at once (also how turning works, a step at a time)."""
        self.pitch, self.yaw = pitch, yaw
        self.con.send(f"setang {pitch:.3f} {yaw:.3f} 0")

    def turn(self, pitch, yaw):
        """Turns the view to (pitch, yaw) at `turn_rate` degrees a second, so it can be seen
        (at once with --fast)."""
        rate = self.turn_rate
        if math.isinf(rate):
            self.look(pitch, yaw)
            return
        while True:
            dy = (yaw - self.yaw + 180) % 360 - 180
            dp = pitch - self.pitch
            step = rate / 60
            if abs(dy) <= step and abs(dp) <= step:
                self.look(pitch, yaw)
                return
            k = min(1.0, step / max(abs(dy), abs(dp)))
            self.look(self.pitch + dp * k, self.yaw + dy * k)
            time.sleep(1 / 60)

    def sync_view(self):
        """Reads where the player is actually looking (getpos), so turns start from there."""
        out = self.con.ask(["getpos"], 5) or ""
        m = re.search(r"setang\s+([-\d.e]+)\s+([-\d.e]+)", out)
        if m:
            self.pitch, self.yaw = float(m.group(1)), float(m.group(2))

    def look_at(self, target, eye=None):
        self.turn(*angle_to(eye or self.obs()["eye"], target))

    def steer(self, yaw, dt):
        """One tick of turning toward `yaw` (and level) while moving; the yaw error left."""
        dy = (yaw - self.yaw + 180) % 360 - 180
        step = self.turn_rate * min(dt, 0.1)
        dp = -self.pitch
        if abs(dy) > 0.5 or abs(dp) > 0.5:
            self.look(self.pitch + max(-step, min(step, dp)), self.yaw + max(-step, min(step, dy)))
        return abs(dy) - min(abs(dy), step)

    def set_speed(self, v):
        """Walking speed (cl_forwardspeed), for easing into a stop."""
        v = round(max(MIN_SPEED, min(self.top_speed, v)) / 10) * 10
        if v != getattr(self, "speed_now", None):
            self.con.send(f"cl_forwardspeed {v}")
            self.speed_now = v

    def show(self, text):
        """Puts `text` on screen (a game_text the harness creates)."""
        text = re.sub(r'[";:,\\]', " ", text)
        if not self.text_made:
            self.con.send('ent_create game_text targetname glados_text x 0.02 y 0.92 channel 3 '
                          'holdtime 9999 fadein 0 fadeout 0 color "255 220 120"')
            self.text_made = True
        self.con.send(f'ent_fire glados_text addoutput "message {text}"', "ent_fire glados_text display")

    def teleport(self, feet):
        self.release_all()
        self.con.send(f"setpos {feet[0]:.1f} {feet[1]:.1f} {feet[2] + 1:.1f}")
        time.sleep(0.3)
        self.last = None
        self.obs()
        self.jumped = False

    # moving
    def settle(self, timeout=3):
        """Waits for the player to stand still on the ground."""
        self.release(*MOVES)
        end = time.time() + timeout
        still = None
        while time.time() < end:
            s = self.obs()
            if math.hypot(*s["v"]) < 10:
                still = still or time.time()
                if time.time() - still > 0.2:
                    return s
            else:
                still = None
        return self.obs()

    def walk(self, points, tol=8.0, stop=True, timeout=None, on_jump="stop"):
        """Walks through world points (xy), turning smoothly toward each and walking on once
        roughly facing it. Intermediate points are passed within about a quarter voxel; the
        last within `tol`, easing into a stop if `stop`. A jump (portal, linked door) ends the
        walk when `on_jump` is "stop". Raises Stuck if no progress is made for 2.5 s."""
        self.jumped = False
        tick = time.time()
        try:
            for k, p in enumerate(points):
                last = k == len(points) - 1
                reach = tol if last else 28.0
                best, best_t = 1e9, time.time()
                end = time.time() + (timeout or 10 + hdist(self.obs()["o"], p) / 100)
                while True:
                    s = self.obs()
                    now = time.time()
                    dt, tick = now - tick, now
                    if self.jumped and on_jump == "stop":
                        return s
                    d = hdist(s["o"], p)
                    if d < reach:
                        if not (last and stop) or self.speed(s) <= 30:
                            break
                        self.release("forward")  # there: coast to a stop
                        continue
                    if d < best - 2:
                        best, best_t = d, now
                    if now - best_t > 2.5 or now > end:
                        raise Stuck(f"stuck at {fmt(s['o'])} walking to {fmt(p)}")
                    yaw = math.degrees(math.atan2(p[1] - s["o"][1], p[0] - s["o"][0]))
                    off = self.steer(yaw, dt)
                    # Slow for the stop at the end, and while turning a corner.
                    self.set_speed(3.0 * d if last and stop else self.top_speed * (1 - off / 90))
                    if off < 45:
                        self.hold("forward")
                    else:
                        self.release("forward")
        finally:
            self.release(*MOVES)
            self.set_speed(self.top_speed)
        return self.settle() if stop else self.obs()

    def walk_cells(self, cells, tol=8.0):
        """Walks the route's cells (each one step or a drop from the last), straight lines
        merged."""
        if not cells:
            return self.obs()
        pts = [self.floor(c) for c in cells]
        keep = [p for i, p in enumerate(pts)
                if i in (0, len(pts) - 1) or not self._collinear(cells[i - 1], cells[i], cells[i + 1])]
        return self.walk(keep, tol)

    @staticmethod
    def _collinear(a, b, c):
        return a[2] == b[2] == c[2] and (b[0] - a[0], b[1] - a[1]) == (c[0] - b[0], c[1] - b[1])

    def shift_to(self, goal, tol=4.0, timeout=5):
        """Moves the player's feet to `goal` (xy) without turning, by forward/back/strafe
        pulses: for lining up a held cube."""
        end = time.time() + timeout
        while time.time() < end:
            s = self.obs()
            dx, dy = goal[0] - s["o"][0], goal[1] - s["o"][1]
            if math.hypot(dx, dy) < tol and self.speed(s) < 20:
                return True
            if self.speed(s) > 40:
                continue
            yaw = math.radians(self.yaw)
            f = dx * math.cos(yaw) + dy * math.sin(yaw)
            r = dx * math.sin(yaw) - dy * math.cos(yaw)
            for key, other, comp in (("forward", "back", f), ("moveright", "moveleft", r)):
                if abs(comp) >= tol / 2:
                    self.tap(key if comp > 0 else other, min(0.1, 0.02 + abs(comp) / 600))
        return False

    def wait_landed(self, timeout=8):
        """Waits until the player is back on the ground after a fall, bounce or portal."""
        end = time.time() + timeout
        steady = None
        while time.time() < end:
            s = self.obs()
            if abs(s["v"][2]) < 5:
                steady = steady or time.time()
                if time.time() - steady > 0.25:
                    return s
            else:
                steady = None
        return self.obs()

    # --- the route's actions -----------------------------------------------------------
    # Each runs after the leg's walk, from where it ends, and returns whether it worked.

    def act_walk(self, leg):
        return True

    def act_fire(self, leg):
        color, face = leg["color"], leg["face"]
        target = self.world(face["centre"])
        for _ in range(2):
            s = self.settle()
            self.look_at(target, s["eye"])
            time.sleep(0.15)
            before = {p["id"]: p for p in self.obs()["portals"]}
            self.tap(FIRE_KEY[color])
            time.sleep(0.5)
            s = self.obs()
            moved = [p for p in s["portals"] if before.get(p["id"]) != p]
            if moved:  # the first shot can make both portal entities: take the one there
                self.portal_ids[color] = min(moved, key=lambda p: math.dist(p["o"], target))["id"]
            if self.portal_on(color, face, s):
                return True
        return False

    def act_enter(self, leg):
        face = leg["face"]
        if not self.jumped and face["dir"] != "-z":
            # A wall portal: walk into it.
            self.turn(0, yaw_of(DIRS[face["dir"]]))
            self.hold("forward")
            end = time.time() + 5
            while not self.jumped and time.time() < end:
                self.obs()
            self.release("forward")
        elif not self.jumped:
            # A floor portal: the walk ends on it; step onto its middle.
            self.walk([self.floor(face["cell"])], tol=4, stop=False, timeout=4)
        went = self.jumped
        self.wait_landed()
        return went

    def act_pickup(self, leg):
        home = self.world(leg["look"])
        for _ in range(2):
            s = self.obs()
            cube = self.nearest_cube(s, home, within=96)
            if cube is None:
                return False
            if hdist(s["o"], cube["o"]) > 80:
                self._approach(cube["o"], 64)
            self.look_at(cube["o"])
            time.sleep(0.15)
            self.tap("use")
            time.sleep(0.4)
            s = self.obs()
            now = next((c for c in s["cubes"] if c["id"] == cube["id"]), None)
            # Picked up: it moved (lifted toward the view) and hangs in front of the eye.
            if now is not None and math.dist(now["o"], cube["o"]) > 8 and math.dist(now["o"], s["eye"]) < 120:
                self.held_id = now["id"]
                self.hold_dist = math.dist(now["o"], s["eye"])
                return True
        return False

    def _approach(self, p, gap):
        """Walks straight toward `p` until `gap` short of it."""
        s = self.obs()
        d = hdist(s["o"], p)
        if d > gap:
            t = (d - gap) / d
            self.walk([[s["o"][0] + (p[0] - s["o"][0]) * t, s["o"][1] + (p[1] - s["o"][1]) * t]], tol=8)

    def place_cube(self, target, tol, look, keep_inside=None):
        """Lines the held cube up over `target` (xy) by moving the player, then lets go.
        `keep_inside`: (cell, margin) the player's feet must stay in (by margin units)."""
        self.look_at(look)
        time.sleep(0.3)
        for _ in range(8):
            s = self.obs()
            cube = self.holding(s)
            if cube is None:
                return False
            err = (target[0] - cube["o"][0], target[1] - cube["o"][1])
            if math.hypot(*err) < tol:
                break
            goal = [s["o"][0] + err[0], s["o"][1] + err[1]]
            if keep_inside:
                cell, margin = keep_inside
                f = self.floor(cell)
                for i in range(2):
                    goal[i] = min(max(goal[i], f[i] - 64 + margin), f[i] + 64 - margin)
            self.shift_to(goal)
            self.look_at(look)
            time.sleep(0.2)
        return self.let_go()

    def act_putdown(self, leg):
        target = self.floor(leg["at"])
        return self.place_cube(target, 14 if leg["button"] else 40, self.world(leg["look"]))

    def act_toss(self, leg):
        edge, d = leg["edge"], DIRS[leg["dir"]]
        f = self.floor(edge)
        # Over the drop, a little past the ledge.
        target = [f[0] + d[0] * 104, f[1] + d[1] * 104]
        return self.place_cube(target, 24, self.world(leg["look"]), keep_inside=(edge, 20))

    def act_throw(self, leg):
        face = leg["face"]
        if face["dir"] == "-z":
            c = self.world(face["centre"])
            return self.place_cube(c, 20, c)
        # Into a wall portal: walk at it until the cube goes through, then let go.
        self.turn(0, yaw_of(DIRS[face["dir"]]))
        s = self.obs()
        cube = self.holding(s)
        if cube is None:
            return False
        self.hold("forward")
        end = time.time() + 5
        went = False
        while time.time() < end and not went:
            s = self.obs()
            now = next((c for c in s["cubes"] if c["id"] == cube["id"]), None)
            went = now is None or math.dist(now["o"], cube["o"]) > 160
            if now is not None and not went:
                cube = now
        self.release("forward")
        return self.let_go() and went

    def act_press(self, leg):
        """Presses a pedestal button: close up, looking at its button (on top of the column,
        about PEDESTAL_TOP over its base) rather than a guess at the cell."""
        p = self.world(leg["look"])
        s = self.obs()
        ped = min(s["pedestals"], key=lambda e: hdist(e["o"], p), default=None)
        if ped is not None and hdist(ped["o"], p) < 96:
            p = [ped["o"][0], ped["o"][1], ped["o"][2] + PEDESTAL_TOP]
        if hdist(s["o"], p) > 64:
            self._approach(p, 48)
        self.look_at(p)
        time.sleep(0.2)
        self.tap("use", 0.15)
        time.sleep(0.4)
        return True

    def act_dash(self, leg):
        self.act_press(leg)
        start = time.time()
        self.walk_cells(leg["path"], tol=24)
        if leg["escape"]:
            self.act_exit({"normal": self.exit_normal})
        took = time.time() - start
        print(f"    dash took {took:.1f} s of {leg['seconds']} s", flush=True)
        return took <= leg["seconds"]

    def act_knock(self, leg):
        p = self.world(leg["look"])
        if hdist(self.obs()["o"], p) > 80:
            self._approach(p, 64)
        self.look_at(p)
        time.sleep(0.15)
        self.tap("use")
        time.sleep(0.5)
        self.turn(-30, self.yaw + 90)
        time.sleep(0.3)
        self.tap("use")
        time.sleep(1.5)
        s = self.obs()
        t = min(s["turrets"], key=lambda t: hdist(t["o"], p), default=None)
        return t is not None and (abs(t["a"][0]) > 30 or abs(t["a"][2]) > 30)

    def act_bounce(self, leg):
        to = self.floor(leg["to"])
        s = self.obs()
        self.turn(0, math.degrees(math.atan2(to[1] - s["o"][1], to[0] - s["o"][0])))
        self.tap("jump")
        end = time.time() + 6
        while time.time() < end:
            s = self.obs()
            if hdist(s["o"], to) > 20:
                self.look(0, math.degrees(math.atan2(to[1] - s["o"][1], to[0] - s["o"][0])))
                self.hold("forward")
            else:
                self.release("forward")
            if s["o"][2] > to[2] - 8 and abs(s["v"][2]) < 5 and hdist(s["o"], to) < 64:
                break
        self.release("forward")
        return self.in_cell(self.wait_landed()["o"], leg["to"])

    def act_leap(self, leg):
        d = DIRS[leg["dir"]]
        take_off = self.floor(leg["run"][-1])
        to = self.floor(leg["to"])
        self.turn(0, yaw_of(d))
        self.hold("forward")
        jumped, end = False, time.time() + 8
        while time.time() < end:
            s = self.obs()
            ahead = (s["o"][0] - take_off[0]) * d[0] + (s["o"][1] - take_off[1]) * d[1]
            if not jumped and ahead > 40:
                self.tap("jump")
                jumped = True
            if jumped and hdist(s["o"], to) < 30:
                break
        self.release("forward")
        return self.in_cell(self.wait_landed()["o"], leg["to"])

    def act_exit(self, leg):
        """Out through the exit door: done at its linked door, or when the level restarts
        (in preview, the exit corridor restarts the level, which stops the reports)."""
        self.turn(0, yaw_of(tuple(-v for v in DIRS[leg["normal"]])))
        self.hold("forward")
        end = time.time() + 15
        try:
            while not self.jumped and time.time() < end:
                self.obs(timeout=3)
        except GameUnavailable:
            self.held.discard("forward")
            return True
        self.release("forward")
        return self.jumped

    # --- checking and putting right ----------------------------------------------------

    def problems(self, after, slack=12):
        """How the game differs from what the route expects after a leg. `slack`: how far
        (units) the player may be outside their cell."""
        s = self.obs()
        out = []
        if not self.in_cell(s["o"], after["player"], slack):
            out.append(f"player at {fmt(s['o'])}, not in cell {after['player']}")
        for color, face in zip(COLORS, after["portals"]):
            if face and not self.portal_on(color, face, s):
                out.append(f"no {color} portal on {face['cell']} {face['dir']}")
        held = self.holding(s)
        free = [c for c in s["cubes"] if held is None or c["id"] != held["id"]]
        for want in after["cubes"]:
            if want is None:
                if held is None:
                    out.append("not holding the cube")
                continue
            here = next((c for c in free if self.in_cell(c["o"], want, slack=16)), None)
            if here is None:
                out.append(f"no cube in cell {want}")
            else:
                free.remove(here)
        return out

    def recover(self, after):
        """Teleports things to where the route expects them after a leg."""
        s = self.obs()
        if not self.in_cell(s["o"], after["player"]):
            self.teleport(self.floor(after["player"]))
        s = self.obs()
        for k, (color, face) in enumerate(zip(COLORS, after["portals"])):
            if face and not self.portal_on(color, face, s):
                c = self.world(face["centre"])
                n = [-v for v in DIRS[face["dir"]]]
                pitch = -90 if n[2] > 0 else 90 if n[2] < 0 else 0
                self.con.send(f"portal_place 0 {k} {c[0] + n[0]:.1f} {c[1] + n[1]:.1f} "
                              f"{c[2] + n[2]:.1f} {pitch} {yaw_of(n) if not n[2] else 0:.0f} 0")
                self.portal_ids.pop(color, None)
        s = self.obs()
        held = self.holding(s)
        used = {held["id"]} if held else set()
        for want in after["cubes"]:
            if want is None:
                if held is None:
                    cube = self.nearest_cube(s, s["eye"], skip=used)
                    if cube is not None:
                        p = s["eye"]
                        yaw = math.radians(self.yaw)
                        self.con.send(f"script glados_put({cube['id']}, {p[0] + 60 * math.cos(yaw):.1f}, "
                                      f"{p[1] + 60 * math.sin(yaw):.1f}, {p[2] - 10:.1f})")
                        time.sleep(0.2)
                        self.tap("use")
                        self.held_id = cube["id"]
                        used.add(cube["id"])
                continue
            here = next((c for c in s["cubes"] if c["id"] not in used
                         and self.in_cell(c["o"], want, slack=16)), None)
            if here is None:
                f = self.floor(want)
                cube = self.nearest_cube(s, f, skip=used)
                if cube is not None:
                    self.con.send(f"script glados_put({cube['id']}, {f[0]:.1f}, {f[1]:.1f}, {f[2] + 24:.1f})")
                    here = cube
            if here is not None:
                used.add(here["id"])
        time.sleep(1.0)
        return self.problems(after)


# --- a level ---------------------------------------------------------------------------------

def glados_exe(path=None):
    if path:
        return Path(path)
    for p in (ROOT / "target" / "release" / "glados.exe", ROOT / "target" / "debug" / "glados.exe"):
        if p.exists():
            return p
    sys.exit("no glados.exe: run `cargo build --release`")


def route(exe, level):
    out = subprocess.run([str(exe), "route", str(level)], capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        sys.exit(f"{level}: {out.stderr.strip() or out.stdout.strip()}")
    lines = [json.loads(l) for l in out.stdout.splitlines() if l.strip()]
    return lines[0], lines[1:]


def calibrate(head):
    """World position of voxel 0, from the entry door's instance in the exported VMF: its
    origin is the floor centre of the door's cell."""
    if not portal2.VMF.exists():
        return None
    text = portal2.VMF.read_text(encoding="utf-8", errors="replace")
    m = re.search(r'"targetname" "doorentry\d*"\s*"file" "[^"]*"\s*"origin" "([-\d.e]+) ([-\d.e]+) ([-\d.e]+)"', text)
    if m is None:
        m = re.search(r'"targetname" "doorentry\d*".{0,200}?"origin" "([-\d.e]+) ([-\d.e]+) ([-\d.e]+)"', text, re.S)
    if m is None:
        return None
    c = head["entry"]["cell"]
    o = [float(v) for v in m.groups()]
    return [o[0] - VOXEL * (c[0] + 0.5), o[1] - VOXEL * (c[1] + 0.5), o[2] - VOXEL * c[2]]


def load_level(con, path, timeout=240):
    """Loads a .p2c into the editor, exports it (for calibration) and builds and starts it.
    Returns the export's text."""
    portal2.STAGE.mkdir(parents=True, exist_ok=True)
    LEVEL.write_bytes(Path(path).read_bytes())
    out = con.ask([f'puzzlemaker_load_dev "{LEVEL.as_posix()}"'], 30) or ""
    if "Loaded puzzle" not in out:
        raise GameUnavailable(f"the editor didn't load {path}: {out[-300:]}")
    portal2.VMF.unlink(missing_ok=True)
    exported = con.ask(["puzzlemaker_export glados_check"], 30) or ""
    con.send("puzzlemaker_compile_and_preview")
    if con.until(lambda l: "launching map!" in l, timeout) is None:
        raise GameUnavailable("the level didn't build (no \"launching map!\" after compile_and_preview)")
    if con.until(lambda l: "END loading assets" in l, timeout) is None:
        raise GameUnavailable("the level didn't load")
    # Built from the console, the level first loads as a transition from the editor's map and
    # comes up half started: unlit, fields unpowered, floor buttons that ignore cubes, the
    # player left in the elevator. Restarting it gives the start Build & Play gives: the
    # player in the entry corridor, the entry door opening.
    time.sleep(2)
    con.send("restart_level")
    if con.until(lambda l: "END loading assets" in l, timeout) is None:
        raise GameUnavailable("the level didn't restart")
    end = time.time() + timeout
    while time.time() < end:
        out = con.ask(["sv_cheats 1", "script_execute glados_play"], 10) or ""
        if "GLADOS_SCRIPT loaded" in out:
            return exported
        time.sleep(1)
    raise GameUnavailable("the level didn't finish loading")


def arrive(game, head):
    """To the entry door's cell from the level's start, normally just outside the entry door
    (see load_level). Failing that, fire the level's start relay, which puts the player there;
    failing that, the elevator: through its linked door, or teleported to the triggers
    outside it."""
    game.show("Arriving")
    entry = game.floor(head["entry"]["cell"])
    s = game.obs()
    if hdist(s["o"], entry) < 4 * VOXEL and abs(s["o"][2] - entry[2]) < 64:
        # The restarted level's start: just outside the entry door. The level's start relay
        # opens it; after a restart that doesn't always happen, so open it as that relay does.
        game.con.send("ent_fire @entrance_door_open_rl trigger")
        game.sync_view()
        for attempt in range(5):
            try:
                game.walk([entry], tol=12)
                return
            except Stuck:
                if attempt == 4:
                    raise
                time.sleep(1.5)
    out = game.con.ask(["script glados_start()"], 5) or ""
    if "GLADOS_START" in out:
        time.sleep(1.8)  # teleported after 0.5 s, the door opens after 1.5 s
        game.last = None
        game.obs()
        game.sync_view()
        game.walk([game.floor(head["entry"]["cell"])], tol=12)
        return
    end = time.time() + 25
    still = None
    while time.time() < end:  # the elevator ride: wait for it to stop
        s = game.obs()
        if math.hypot(*s["v"]) < 5:
            still = still or time.time()
            if time.time() - still > 1.5:
                break
        else:
            still = None
    if not walk_out(game, 8):
        points = {}
        for l in (game.con.ask(["script glados_arrival()"], 5) or "").splitlines():
            if l.startswith("GLADOS_ARRIVAL "):
                _, name, xyz = l.split(" ", 2)
                points.setdefault(name, []).append(json.loads(xyz))
        door = points.get("@entrance_portal_elevator_side", [None])[0]
        lift = next((v[0] for k, v in points.items() if "arrival_elevator-elevator_1" in k), None)
        triggers = next((v for k, v in points.items() if "leaving_elevator_trigger" in k), [])
        if door is None or not triggers:
            raise Stuck("never left the elevator, and found no corridor to put the player in")
        first = min(triggers, key=lambda t: hdist(t, lift or t))
        print("  arrival: the elevator didn't take the player; "
              f"teleporting to its corridor at {fmt(first)}", flush=True)
        game.teleport(first)
        game.look(0, math.degrees(math.atan2(door[1] - first[1], door[0] - first[0])))
        if not walk_out(game, 15):
            raise Stuck(f"never got through the elevator's linked door (stopped at {fmt(game.obs()['o'])})")
    game.walk([game.floor(head["entry"]["cell"])], tol=12)


def wait_for_cubes(game, head, timeout=15):
    """Droppers drop their cubes a little after the puzzle starts: wait for a cube in every
    cube's starting cell."""
    end = time.time() + timeout
    while time.time() < end:
        s = game.obs()
        if all(any(game.in_cell(c["o"], want["home"], slack=16) for c in s["cubes"])
               for want in head["cubes"]):
            return
        time.sleep(0.3)


def walk_out(game, seconds):
    """Holds forward, keeping the heading, until the player goes through a linked door."""
    game.jumped = False
    game.hold("forward")
    end = time.time() + seconds
    while not game.jumped and time.time() < end:
        game.obs()
    game.release("forward")
    return game.jumped


def play(game, head, legs):
    """Plays a route; returns one (status, text, problems) per leg. Status: ok, unconfirmed
    (the action couldn't see that it worked, e.g. a turret still standing, but nothing the
    route tracks is wrong), recovered (put right by teleporting) or failed."""
    game.exit_normal = head["exit"]["normal"]
    results = []
    try:
        arrive(game, head)
    except Stuck as e:
        print(f"  arrival: {e}; teleporting in", flush=True)
        game.teleport(game.floor(head["entry"]["cell"]))
    wait_for_cubes(game, head)
    for leg in legs:
        game.show(f"{leg['leg']}/{len(legs)}: {leg['text']}")
        ok, why = False, ""
        try:
            game.walk_cells(leg["walk"])
            ok = getattr(game, "act_" + leg["action"])(leg)
        except Stuck as e:
            why = str(e)
        finally:
            game.release_all()
        exit_leg = leg["action"] == "exit" or leg.get("escape")
        # Lining up a cube or reaching for something can take the player half a cell over.
        slack = 64 if leg["action"] in ("pickup", "putdown", "toss", "throw", "press", "knock") else 12
        problems = [] if exit_leg else game.problems(leg["after"], slack)
        if ok and not problems:
            status = "ok"
        elif not problems and not why:
            status = "unconfirmed"  # the action couldn't see it work; nothing to put right
        elif exit_leg:
            status = "failed"
        else:
            problems = [why] * bool(why) + problems
            left = game.recover(leg["after"])
            status = "recovered" if not left else "failed"
            problems += [f"after recovery: {p}" for p in left]
        results.append((status, leg["text"], problems))
        print(f"  {status:9} {leg['text']}" + "".join(f"\n            {p}" for p in problems), flush=True)
    return results


# --- the spike -----------------------------------------------------------------------------

def spike(game, head, exported):
    """Checks what playing relies on and prints what it found."""
    con = game.con
    say = lambda k, v: print(f"  {k:24} {v}", flush=True)
    say("calibration", f"voxel 0 at {game.origin} (expected [0, 0, 0])")
    say("export said", " | ".join(l for l in exported.splitlines() if "PUZZLEMAKER" in l)[:300])
    s = game.obs()
    say("player", f"feet {fmt(s['o'])}, eye {fmt(s['eye'])}, hp {s['hp']}")
    say("entities", f"{len(s['portals'])} portals, {len(s['cubes'])} cubes, {len(s['turrets'])} turrets")
    for l in (con.ask(["script glados_names()"], 5) or "").splitlines():
        if l.startswith("GLADOS_NAME "):
            say("  entity", l.split(" ", 1)[1][:150])
    t0 = time.time()
    for _ in range(20):
        game.obs()
    say("observation rate", f"{20 / (time.time() - t0):.0f} Hz")
    out = con.ask(["portal_place"], 5) or ""
    say("portal_place usage", " | ".join(out.splitlines())[:300])
    game.show("glados play: spike")
    say("game_text", "shown (look at the bottom left of the screen)")
    try:
        arrive(game, head)
        say("arrival", f"in the entry cell at {fmt(game.obs()['o'])}")
    except Stuck as e:
        say("arrival", f"FAILED: {e}")
        game.teleport(game.floor(head["entry"]["cell"]))
    s0 = game.settle()
    game.look(0, game.yaw)
    game.tap("forward", 0.5)
    s1 = game.settle()
    say("+forward 0.5 s", f"moved {hdist(s0['o'], s1['o']):.0f} units")
    target = game.floor(head["entry"]["cell"])
    game.teleport(target)
    s = game.obs()
    say("setpos", f"asked feet {fmt(target)}, got feet {fmt(s['o'])}, eye {fmt(s['eye'])}")
    print("  console lines seen while checking:", flush=True)
    for l in con.drain()[-20:]:
        print("    " + l[:160])


# --- main ----------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("levels", nargs="+", help=".p2c files")
    ap.add_argument("--launch", action="store_true", help="start the game (and quit it at the end)")
    ap.add_argument("--keep", action="store_true", help="leave a launched game running")
    ap.add_argument("--port", type=int, help="console port of a game you started")
    ap.add_argument("--password", default="", help="its -netconpassword")
    ap.add_argument("--slow", type=float, default=1.0, help="host_timescale while playing")
    ap.add_argument("--fast", action="store_true",
                    help="snap the view instead of turning smoothly (for validation runs)")
    ap.add_argument("--spike", action="store_true", help="check the harness on the first level")
    ap.add_argument("--glados", help="glados.exe (default: target/release, else target/debug)")
    a = ap.parse_args()
    exe = glados_exe(a.glados)
    routes = [(p, *route(exe, p)) for p in a.levels]  # refuse unplayable levels up front
    saved = {p: p.read_bytes() for p in portal2.autosaves()}
    pid = con = None
    summary = []
    try:
        SCRIPT.parent.mkdir(parents=True, exist_ok=True)
        SCRIPT.write_text(VSCRIPT, encoding="ascii")
        if a.launch:
            pid, con = portal2.launch(saved)
        else:
            proc = portal2.game_process()
            if proc is None or "-insecure" not in proc[1].split():
                raise GameUnavailable("no game started with -insecure (or use --launch)")
            if not a.port:
                raise GameUnavailable("pass --port (and --password) for a game you started")
            con = Console(a.port, a.password)
            open_editor(con)  # from wherever the game is, e.g. its main menu
        for i, (path, head, legs) in enumerate(routes):
            if i:
                open_editor(con)
            print(f"{path}: \"{head['title']}\", {len(legs)} legs", flush=True)
            exported = load_level(con, path)
            game = Game(con, calibrate(head) or (0.0, 0.0, 0.0), math.inf if a.fast else TURN_RATE)
            if a.slow != 1.0:
                con.send(f"host_timescale {a.slow}")
            if a.spike:
                spike(game, head, exported)
                break
            results = play(game, head, legs)
            summary.append((path, results))
            time.sleep(2)
    except GameUnavailable as e:
        print(f"Portal 2 unavailable: {e}", file=sys.stderr)
        sys.exit(portal2.UNAVAILABLE)
    finally:
        SCRIPT.unlink(missing_ok=True)
        if pid is not None and not a.keep:
            portal2.quit_game(saved, pid, con)
        elif pid is None:
            portal2.VMF.unlink(missing_ok=True)
    for path, results in summary:
        counts = {k: sum(r[0] == k for r in results) for k in ("ok", "unconfirmed", "recovered", "failed")}
        print(f"{path}: " + ", ".join(f"{v} {k}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
