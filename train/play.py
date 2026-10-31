"""Play a level's solution in Portal 2 the way a player would, to watch it.

    python train/play.py fixtures/synthetic/goo_crossing.p2c --launch
    python train/play.py LEVEL.p2c... --port PORT --password PW     # a game you started yourself
    python train/play.py --spike fixtures/synthetic/worked_example.p2c --launch

`glados-score route` turns the solver's shortest plan into cells, faces and points (src/route.rs).
This loads the level into the Puzzle Maker (train/portal2.py), builds it, and plays it with the
player's own controls: `setang` to look (a perfect mouse, turning smoothly so it can be watched;
--fast snaps it, for validation runs) and +forward, +jump, +attack (blue),
+attack2 (orange) and +use, steering by the positions a small VScript reports each tick. The
step being played shows on screen. Each step's result is checked against the route.

The game is saved before each step. A step that didn't work out is screenshotted, loaded back
and tried again carefully (the route's own cells walked, the player put exactly where the
action starts, the view snapped); what that shows classifies it: exec (the careful retry worked:
the harness played it badly), route (the route's path crosses a live field it didn't plan to),
blocked (the route's cells can't be walked), plan? (done carefully it still fails: the game may
disagree with the solver). Only then is it put right by teleporting (setpos, a cube's
SetOrigin, portal_place) and counted as recovered, and the run carries on. The terminal gets
one line per step.

Some of what the game asks of a player the harness does itself: a laser beam across the way is
ducked under or jumped, by how high the emitters put it; a face whose middle is out of sight is
shot at off-centre; and a shut exit door is waited at (--exit-wait), since one can open late (a
laser burning through the turrets in its way, a platform arriving). A timed escape never waits.

    python train/play.py --list LEVELS.txt --launch --fast --out out/play/NAME
    python train/play.py --plans PLANS.jsonl --launch --fast --out out/play/NAME

--plans plays given plans (Goblin's) instead of the solver's: JSONL rows with "p2c" (the level,
relative to the file's folder or absolute), "plan" (plan tokens) and optionally "mode"
(liberal routes a plan only the liberal model accepts).

--out keeps the evidence, per level: legs.jsonl (each step's tries, problems, events, probes),
trace.jsonl (every look: player, portals, cubes), events.jsonl (what happened unasked: a portal
lost, a field crossed by the player or a cube, hurt, died, a cube gone) and screenshots; and
summary.jsonl for the batch (levels already in it are skipped; a file STOP there ends the batch
after the level being played). With --launch the game is restarted every --restart-every
levels and after it goes down. train/playreport.py sums a batch up.

The game is driven over its TCP console. --launch starts it (refusing if it's already running)
with -insecure and -netconport/-netconpassword, under portal2.py's safety rules, and quits it at
the end unless --keep. For a game you started yourself, add
`-insecure -netconport PORT -netconpassword PW` to its launch options and pass --port and
--password. --spike checks the console, VScript, coordinates, movement and the commands the
player uses on the first level, and prints what it found instead of playing.

Keep the game window focused while it plays: in the background, a level built from the console
never starts (unlit, entry door shut, droppers holding their cubes), and arrival gets stuck.
"""
import argparse, json, math, re, shutil, subprocess, sys, time
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
EXIT_WAIT = 40.0   # longest wait at a shut exit door: the laser burning through the turrets in
                   # its way, a platform arriving (a timed escape never waits)
DUCK_HEIGHT = 40.0  # a beam higher than this over the feet is ducked under, not jumped (the
                    # player is 72 tall standing, 36 crouched)
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
// How far a straight line from a to b gets before something solid stops it (0..1).
function glados_trace(ax, ay, az, bx, by, bz) {
	printl("GLADOS_TRACE " + TraceLine(Vector(ax, ay, az), Vector(bx, by, bz), GetPlayer()))
}
// The laser emitters and where they point (what burnt the player).
function glados_lasers() {
	local e = null
	while ((e = Entities.FindByClassname(e, "env_portal_laser")) != null)
		printl("GLADOS_LASER {\"o\":" + glados_v(e.GetOrigin()) + ",\"a\":" + glados_v(e.GetAngles()) + "}")
	printl("GLADOS_LASER_END")
}
// What is within r of a point: class, name, origin (what a blocked walk ran into).
function glados_near(x, y, z, r) {
	local e = null
	while ((e = Entities.FindInSphere(e, Vector(x, y, z), r)) != null) {
		local c = e.GetClassname()
		if (c != "player" && c != "weapon_portalgun" && c != "predicted_viewmodel" && c != "info_target")
			printl("GLADOS_NEAR " + c + " " + e.GetName() + " " + glados_v(e.GetOrigin()) + " " + glados_v(e.GetAngles()))
	}
}
printl("GLADOS_SCRIPT loaded")
"""


class Stuck(Exception):
    """A move that didn't get where it was going."""


class InvalidLevel(Exception):
    """A level the editor won't build."""


# --- geometry ------------------------------------------------------------------------------

def hdist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle_to(eye, target):
    """(pitch, yaw) in degrees looking from `eye` at `target`."""
    dx, dy, dz = (target[i] - eye[i] for i in range(3))
    return -math.degrees(math.atan2(dz, math.hypot(dx, dy))), math.degrees(math.atan2(dy, dx))


def yaw_of(d):
    return math.degrees(math.atan2(d[1], d[0]))


def heading(angles):
    """The way an entity's angles (pitch, yaw, roll) point."""
    pitch, yaw = math.radians(angles[0]), math.radians(angles[1])
    return (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch))


def beam_over(lasers, feet, reach=40.0):
    """How high over `feet` a laser beam crosses: of the beams in `lasers` (each {"o", "a"},
    from `glados_lasers`) whose line passes within `reach` of the column over `feet`, the
    lowest one's height over the feet, or None if none does. Only the straight beam an
    emitter makes is found this way: one bent by a reflection cube or sent through a portal
    is not, and nor is one the level has switched off."""
    best = None
    for l in lasers:
        o, u = l["o"], heading(l["a"])
        flat = math.hypot(u[0], u[1])
        if flat < 1e-3:
            continue  # straight up or down: not something to duck under or jump
        # Where the beam's line is nearest the column over the feet, along the beam.
        along = ((feet[0] - o[0]) * u[0] + (feet[1] - o[1]) * u[1]) / (flat * flat)
        if along < 0:
            continue  # behind the emitter: it points the other way
        x, y = o[0] + u[0] * along, o[1] + u[1] * along
        if math.hypot(x - feet[0], y - feet[1]) > reach:
            continue
        h = o[2] + u[2] * along - feet[2]
        if best is None or h < best:
            best = h
    return best


def aim_points(centre, face_dir, spread=32.0):
    """Points on a face to aim at, its centre first, then points offset in its plane: a shot
    at the middle can be blocked (a pane's frame, a panel's edge) where the face is open."""
    out = [tuple(float(v) for v in centre)]
    for i in (i for i in range(3) if i != "xyz".index(face_dir[1])):
        for away in (-spread, spread):
            q = [float(v) for v in centre]
            q[i] += away
            out.append(tuple(q))
    return out


def fmt(p):
    return "(" + ", ".join(f"{v:.0f}" for v in p[:3]) + ")"


# --- the game ------------------------------------------------------------------------------

class Game:
    """The player in a running level: what it sees (via glados_play.nut) and its controls."""

    def __init__(self, con, origin=(0.0, 0.0, 0.0), turn_rate=TURN_RATE, exit_wait=EXIT_WAIT):
        self.con = con
        self.turn_rate = turn_rate  # degrees a second; math.inf snaps (--fast)
        self.exit_wait = exit_wait  # seconds to wait at a shut exit door
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
        # Evidence for the report (see play): every look, what happened, screenshots.
        self.leg_no = 0
        self.trace = []         # one compact record per look
        self.events = []        # what happened that the route didn't ask for
        self.shots = []         # screenshot names taken
        self.shot_prefix = "glados"
        self.firing = False     # portals are expected to move (a shot, portal_place)
        self.hazards = []       # [(index, kind, axis, plane, lo, hi)] in world units
        self.fields_on = set()  # hazard indices on during this leg
        self.last_event_shot = 0.0
        self.last_fire = None
        self.intended = set()   # hazard indices the route's walk crosses this leg
        self.crossed = set()    # (leg, cube, hazard) crossings already noted
        out = con.ask(["cl_forwardspeed"], 5) or ""
        m = re.search(r'"cl_forwardspeed" = "([\d.]+)"', out)
        self.top_speed = float(m.group(1)) if m else 175.0
        out = con.ask(["cl_sidespeed"], 5) or ""
        m = re.search(r'"cl_sidespeed" = "([\d.]+)"', out)
        self.side_speed = float(m.group(1)) if m else 175.0

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
        jump = False
        if self.last is not None:
            dt = max(s["t"] - self.last["t"], 0.0)
            if math.dist(s["o"], self.last["o"]) > 64 + 600 * dt:
                self.jumped = jump = True
                # Going through a portal turns the view: read where it looks now, or turns
                # and shuffles (shift_to) work from the old heading.
                self.sync_view()
            self._events(self.last, s, jump)
        self.trace.append({"leg": self.leg_no, "t": round(s["t"], 2), "o": [round(v) for v in s["o"]],
                           "v": [round(v) for v in s["v"]], "hp": s["hp"],
                           "p": [[p["id"]] + [round(v) for v in p["o"]] for p in s["portals"]],
                           "c": [[c["id"]] + [round(v) for v in c["o"]] for c in s["cubes"]]})
        self.last = s
        return s

    def event(self, kind, text, shot=True, **extra):
        """Notes something that happened (and screenshots it, at most once a second)."""
        e = {"leg": self.leg_no, "t": round(self.last["t"], 2) if self.last else None,
             "kind": kind, "text": text, **extra}
        if shot and time.time() - self.last_event_shot > 1.0:
            e["shot"] = self.screenshot(kind.replace(" ", "-"))
            self.last_event_shot = time.time()
        self.events.append(e)
        print(f"    event: {kind}: {text}", flush=True)

    def _events(self, a, b, jump):
        """What changed between two looks that the harness didn't do itself."""
        if b["hp"] < a["hp"]:
            self.hurt_at = time.time()
            if b["hp"] <= 0:
                self.event("died", f"hp {a['hp']} -> 0 at {fmt(b['o'])}")
            elif time.time() - getattr(self, "hurt_logged", 0) > 3:
                self.hurt_logged = time.time()
                self.event("hurt", f"hp {a['hp']} -> {b['hp']} at {fmt(b['o'])}")
        ids = {c["id"] for c in b["cubes"]}
        for c in a["cubes"]:
            if c["id"] not in ids:
                self.event("cube gone", f"cube {c['id']} last at {fmt(c['o'])}")
        if not self.firing:
            now = {p["id"]: p for p in b["portals"]}
            for p in a["portals"]:
                q = now.get(p["id"])
                if q is None or math.dist(q["o"], p["o"]) > 8:
                    color = next((k for k, v in self.portal_ids.items() if v == p["id"]), "a")
                    self.event("portal lost", f"{color} portal {p['id']} at {fmt(p['o'])} "
                               + ("gone" if q is None else f"now at {fmt(q['o'])}")
                               + f"; player at {fmt(b['o'])}")
        if not jump:
            mid = lambda o: [o[0], o[1], o[2] + 36]  # the middle of the player, not the feet
            hit = self.field_between(mid(a["o"]), mid(b["o"]))
            if hit:
                h, kind, x = hit
                self.event("crossed field", f"{kind} {h} at {fmt(x)}", hazard=h)
            else:
                # The player's box reaches 16 either side of the middle: within a few units
                # of a field it may touch it (a fizzler then takes the portals, unseen: a
                # fizzled portal entity stays where it was).
                near = self.field_near(mid(b["o"]), 24)
                if near and (self.leg_no, "player", near[0]) not in self.crossed:
                    self.crossed.add((self.leg_no, "player", near[0]))
                    h, kind, d = near
                    self.event("near field", f"{kind} {h}: player {d:.0f} from it at {fmt(b['o'])}", hazard=h)
        before = {c["id"]: c for c in a["cubes"]}
        for c in b["cubes"]:
            if c["id"] in before:
                hit = self.field_between(before[c["id"]]["o"], c["o"])
                if hit and (self.leg_no, c["id"], hit[0]) not in self.crossed:
                    self.crossed.add((self.leg_no, c["id"], hit[0]))
                    h, kind, x = hit
                    self.event("cube crossed field", f"cube {c['id']} through {kind} {h} at {fmt(x)}"
                               + (" (held)" if c["id"] == self.held_id else ""), hazard=h)

    def field_between(self, a, b, skip=()):
        """The first live hazard field the straight line a-b goes through: (index, kind,
        where), or None. `skip`: hazard indices to leave out."""
        for h, kind, axis, plane, lo, hi in self.hazards:
            if h not in self.fields_on or h in skip:
                continue
            pa, pb = a[axis] - plane, b[axis] - plane
            if pa * pb >= 0:
                continue
            k = pa / (pa - pb)
            x = [a[i] + (b[i] - a[i]) * k for i in range(3)]
            if all(lo[i] - 16 <= x[i] <= hi[i] + 16 for i in range(3) if i != axis):
                return h, kind, x
        return None

    def away_from_fields(self, s, yaw, reach=64.0):
        """`yaw` bent away from live fields the route doesn't cross that the player is within
        `reach` of (a laser field kills on touch; a fizzler takes the portals)."""
        dx, dy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        mid = [s["o"][0], s["o"][1], s["o"][2] + 36]
        bent = False
        for h, kind, axis, plane, lo, hi in self.hazards:
            if axis == 2 or h not in self.fields_on or h in self.intended:
                continue
            d = mid[axis] - plane
            if abs(d) >= reach or not all(lo[i] - 16 <= mid[i] <= hi[i] + 16 for i in range(3) if i != axis):
                continue
            push = 1.5 * (reach - abs(d)) / reach * (1 if d > 0 else -1)
            if axis == 0:
                dx += push
            else:
                dy += push
            bent = True
        return math.degrees(math.atan2(dy, dx)) if bent else yaw

    def guarding(self):
        """Whether a held cube has fields to keep out of (live ones the route doesn't cross)."""
        return self.held_id is not None and bool(self.fields_on - self.intended)

    def cube_clear(self, s, yaw):
        """Whether the held cube, hanging in front of a view heading `yaw`, stays out of the
        fields the route doesn't cross: not across one, nor within a cube's size of one."""
        if not self.guarding():
            return True
        here = [s["o"][0], s["o"][1], s["eye"][2] - 20]
        r = math.radians(yaw)
        c = [here[0] + self.hold_dist * math.cos(r), here[1] + self.hold_dist * math.sin(r), here[2]]
        if self.field_between(here, c, skip=self.intended):
            return False
        for h, kind, axis, plane, lo, hi in self.hazards:
            # A cube carried at an angle reaches its half-diagonal (about 45) out, and the
            # field has some thickness: 48 from the plane was seen to fizzle one.
            if (h in self.fields_on and h not in self.intended and abs(c[axis] - plane) < 72
                    and all(lo[i] - 48 <= c[i] <= hi[i] + 48 for i in range(3) if i != axis)):
                return False
        return True

    def field_near(self, p, reach):
        """A live field the route doesn't cross within `reach` of point p (beside it, not
        across): (index, kind, distance), or None."""
        for h, kind, axis, plane, lo, hi in self.hazards:
            if h not in self.fields_on or h in self.intended:
                continue
            d = abs(p[axis] - plane)
            if d < reach and all(lo[i] <= p[i] <= hi[i] for i in range(3) if i != axis):
                return h, kind, d
        return None

    def cube_yaw(self, s, yaw):
        """A view heading near `yaw` that keeps the held cube out of fields (`cube_clear`):
        `yaw` itself if that is clear (or nothing is)."""
        if not self.guarding():
            return yaw
        for k in range(0, 13):
            for y in ((yaw,) if k == 0 else (yaw + 15 * k, yaw - 15 * k)):
                if self.cube_clear(s, y):
                    return y
        return yaw

    def screenshot(self, tag):
        """Writes portal2/screenshots/<prefix>-<leg>-<tag>-<n>.jpg; returns its name."""
        name = f"{self.shot_prefix}-{self.leg_no:02d}-{tag}-{len(self.shots):03d}"
        self.con.send(f"jpeg {name} 80")
        self.shots.append(name)
        return name

    def set_hazards(self, head):
        """The level's hazard fields (`glados-score route`'s head), as planes in world units."""
        self.hazards = []
        for h, hz in enumerate(head.get("hazards", [])):
            for face in hz["faces"]:
                axis = "xyz".index(face["dir"][1])
                c = self.world(face["centre"])
                lo = [c[i] - VOXEL / 2 for i in range(3)]
                hi = [c[i] + VOXEL / 2 for i in range(3)]
                self.hazards.append((h, hz["kind"], axis, c[axis], lo, hi))

    def trace_line(self, a, b):
        """How far (0..1) a straight line from a to b gets before something solid."""
        out = self.con.ask([f"script glados_trace({a[0]:.1f},{a[1]:.1f},{a[2]:.1f},"
                            f"{b[0]:.1f},{b[1]:.1f},{b[2]:.1f})"], 5) or ""
        m = re.search(r"GLADOS_TRACE ([-\d.e]+)", out)
        return float(m.group(1)) if m else None

    def lasers(self):
        """Every laser emitter: {"o": origin, "a": angles}, for `beam_over`."""
        out = self.con.ask(["script glados_lasers()"], 5) or ""
        found = []
        for l in out.splitlines():
            if l.startswith("GLADOS_LASER {"):
                try:
                    found.append(json.loads(l.split(" ", 1)[1]))
                except ValueError:
                    pass
        return found

    def near(self, p, r):
        """Entities within r of p: [class name origin angles], nearest first."""
        out = self.con.ask([f"script glados_near({p[0]:.1f},{p[1]:.1f},{p[2]:.1f},{r})"], 5) or ""
        found = [l.split(" ", 1)[1] for l in out.splitlines() if l.startswith("GLADOS_NEAR ")]
        return found[:20]

    # saving and loading: a leg is retried from where it started
    def save(self):
        out = self.con.ask(["save glados_leg"], 20) or ""
        return "SAVEGAME" in out

    def load(self, before):
        """Loads the save from the start of the leg and finds the portals and the held cube
        again (a load gives entities new numbers). `before`: the route's state then."""
        self.release_all()
        self.con.send("load glados_leg")
        if self.con.until(lambda l: "END loading assets" in l, 60) is None:
            raise GameUnavailable("the save didn't load")
        time.sleep(1.0)
        end = time.time() + 20
        while "GLADOS_SCRIPT loaded" not in (self.con.ask(["script_execute glados_play"], 10) or ""):
            if time.time() > end:
                raise GameUnavailable("the script didn't load after loading the save")
            time.sleep(0.5)
        self.last = None
        self.jumped = False
        self.sync_view()
        s = self.obs()
        self.portal_ids = {}
        for color, face in zip(COLORS, before["portals"]):
            if face:
                c = self.world(face["centre"])
                p = min(s["portals"], key=lambda p: math.dist(p["o"], c), default=None)
                if p is not None and math.dist(p["o"], c) < 64:
                    self.portal_ids[color] = p["id"]
        self.held_id = None
        if None in before["cubes"]:
            c = self.nearest_cube(s, s["eye"], within=120)
            if c is not None:
                self.held_id = c["id"]
                self.hold_dist = math.dist(c["o"], s["eye"])
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
        """Drops the held cube (use), checking it went; True if it did. Where the cube ends up
        doesn't tell (one let go onto a button rests about where a held one hangs, and pressing
        use again would pick it back up), so glance up: a held cube follows the view."""
        for _ in range(2):
            self.tap("use", 0.15)
            time.sleep(0.3)
            if self.held_id is None:
                return True
            cube = lambda: next((c for c in self.obs()["cubes"] if c["id"] == self.held_id), None)
            a = cube()
            time.sleep(0.3)
            b = cube()
            if a is None or b is None:  # gone (fizzled, dissolved): not held any more
                self.held_id = None
                return True
            if b["o"][2] < a["o"][2] - 8:  # falling (glancing would see it move too)
                self.held_id = None
                time.sleep(0.6)
                return True
            time.sleep(0.4)
            if not self.follows_view(self.held_id):
                self.held_id = None
                return True
        return False

    def follows_view(self, cube_id):
        """Whether the player holds this cube: glance up and see if it comes along (where a
        cube is doesn't tell; a held one can hang just where a loose one lies)."""
        before = next((c for c in self.obs()["cubes"] if c["id"] == cube_id), None)
        if before is None:
            return False
        pitch, yaw = self.pitch, self.yaw
        self.look(pitch - 30, yaw)
        time.sleep(0.4)
        after = next((c for c in self.obs()["cubes"] if c["id"] == cube_id), None)
        self.look(pitch, yaw)
        time.sleep(0.4)  # a held cube swings back down
        return after is not None and math.dist(after["o"], before["o"]) >= 12

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
        (at once with --fast). A held cube swings round with the view: it goes the way that
        keeps the cube out of fields, and with --fast in steps the cube can follow."""
        rate = self.turn_rate
        dy = (yaw - self.yaw + 180) % 360 - 180
        guard = self.guarding() and self.last is not None
        if guard:
            s = self.last
            arc = lambda d: all(self.cube_clear(s, self.yaw + d * k / 12) for k in range(13))
            if dy and not arc(dy) and arc(dy - math.copysign(360, dy)):
                dy -= math.copysign(360, dy)
            if math.isinf(rate):
                rate = 360.0
        if math.isinf(rate):
            self.look(pitch, yaw)
            return
        goal = self.yaw + dy
        while True:
            dy, dp = goal - self.yaw, pitch - self.pitch
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
        if getattr(self, "stream", False):  # the overlay says the rest
            text = text[:60]
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

    def walk(self, points, tol=8.0, stop=True, timeout=None, on_jump="stop", launched=False):
        """Walks through world points (xy), turning smoothly toward each and walking on once
        roughly facing it. Intermediate points are passed within about a quarter voxel; the
        last within `tol`, easing into a stop if `stop`. A jump (portal, linked door) ends the
        walk when `on_jump` is "stop"; going up fast (a faith plate) ends it when `launched`.
        Raises Stuck if no progress is made for 2.5 s."""
        self.jumped = False
        tick = time.time()
        try:
            for k, p in enumerate(points):
                last = k == len(points) - 1
                reach = tol if last else 28.0
                best, best_t, hopped = 1e9, time.time(), False
                floor_t, waiting, beam_t = 0.0, 0.0, 0.0
                end = time.time() + (timeout or 10 + hdist(self.obs()["o"], p) / 100)
                while True:
                    s = self.obs()
                    now = time.time()
                    dt, tick = now - tick, now
                    if self.jumped and on_jump == "stop":
                        return s
                    if launched and s["v"][2] > 150:
                        return s  # thrown: hands off (steering in the air spoils the arc)
                    # Burnt (a laser beam across the way, which also pushes back): duck
                    # under it or jump it.
                    if now - getattr(self, "hurt_at", 0.0) < 0.3 and now - beam_t > 3.0 and abs(s["v"][2]) < 5:
                        beam_t = now
                        self.beam_pass(s, p)
                        best, best_t = 1e9, time.time()
                        continue
                    d = hdist(s["o"], p)
                    if d < reach:
                        if not (last and stop) or self.speed(s) <= 30:
                            break
                        self.release(*MOVES)  # there: coast to a stop
                        continue
                    if d < best - 2:
                        best, best_t, hopped = d, now, False
                    if now - best_t > 2.5 or now > end:
                        raise Stuck(f"stuck at {fmt(s['o'])} walking to {fmt(p)}")
                    if now - best_t > 1.0 and not hopped and abs(s["v"][2]) < 5:
                        # Pushing against something low (a bridge out of a portal sits 24 up,
                        # higher than a step): hop up onto it.
                        self.tap("jump")
                        hopped = True
                        self.event("hop", f"no headway at {fmt(s['o'])}: jumped", shot=False)
                    # Level ahead with no floor (a rail platform's track while the platform is
                    # elsewhere): wait for the floor to come, then go on (riding it).
                    if len(p) > 2 and abs(p[2] - s["o"][2]) < 20 and now - floor_t > 0.25:
                        floor_t = now
                        rise = self.ground_ahead(s, p)
                        if rise is not None and 18 < rise <= 40 and not hopped and abs(s["v"][2]) < 5:
                            # A low ledge ahead (a bridge out of a portal sits 24 up): hop.
                            self.tap("jump")
                            hopped = True
                            self.event("hop", f"{rise:.0f} up ahead of {fmt(s['o'])}: jumped", shot=False)
                        if rise is None:
                            if not waiting:
                                waiting = now
                                self.event("waiting", f"no floor ahead of {fmt(s['o'])}", shot=False)
                            if now - waiting > 30:
                                raise Stuck(f"no floor ahead of {fmt(s['o'])} for 30 s")
                            self.release(*MOVES)
                            best_t, end = now, end + 0.25
                            continue
                        waiting = 0.0
                    yaw = self.away_from_fields(s, math.degrees(math.atan2(p[1] - s["o"][1], p[0] - s["o"][0])))
                    view = self.cube_yaw(s, yaw)
                    if view != yaw:
                        # Face away from the field and walk sideways (or backwards) there.
                        self.steer(view, dt)
                        self.move_toward(yaw, 3.0 * d if last and stop else self.top_speed)
                        continue
                    self.release("back", "moveleft", "moveright")
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
            self.set_side_speed(self.side_speed)
        return self.settle() if stop else self.obs()

    def ground_ahead(self, s, p, ahead=40.0):
        """How high the ground is `ahead` units on toward p (or at p, if nearer), over the
        player's feet: None if there is none down to 24 under them. (Unknown counts as 0.)"""
        d = hdist(s["o"], p)
        k = min(1.0, ahead / max(d, 1.0))
        x, y = s["o"][0] + (p[0] - s["o"][0]) * k, s["o"][1] + (p[1] - s["o"][1]) * k
        frac = self.trace_line([x, y, s["o"][2] + 44], [x, y, s["o"][2] - 24])
        if frac is None:
            return 0.0
        return None if frac >= 1.0 else 44 - 68 * frac

    def jump_beam(self, s, p, back=80.0):
        """Over a laser beam across the way to p: back off from where it burnt, run at it and
        jump just before it (it pushes back and burns on touch; a running jump clears it)."""
        burnt = s["o"][:2]
        d = hdist(burnt, p)
        u = ((p[0] - burnt[0]) / max(d, 1), (p[1] - burnt[1]) / max(d, 1))
        self.event("laser jump", f"burnt at {fmt(s['o'])}: running jump", shot=False)
        self.release(*MOVES)
        self.turn(0, math.degrees(math.atan2(u[1], u[0])))
        self.shift_to([burnt[0] - u[0] * back, burnt[1] - u[1] * back], tol=8, timeout=4)
        self.set_speed(self.top_speed)
        self.hold("forward")
        jumped, end = False, time.time() + 4
        try:
            while time.time() < end:
                q = self.obs()
                ahead = (q["o"][0] - burnt[0]) * u[0] + (q["o"][1] - burnt[1]) * u[1]
                if not jumped and ahead > -36:
                    self.tap("jump")
                    jumped = True
                if jumped and ahead > 48 and abs(q["v"][2]) < 5:
                    break
        finally:
            self.release(*MOVES)

    def duck_beam(self, s, p, back=64.0):
        """Under a laser beam across the way to p: back off, crouch and walk through. A beam
        at chest height can't be jumped; crouched the player is 36 tall, so it passes over."""
        burnt = s["o"][:2]
        d = hdist(burnt, p)
        u = ((p[0] - burnt[0]) / max(d, 1), (p[1] - burnt[1]) / max(d, 1))
        self.event("laser duck", f"burnt at {fmt(s['o'])}: ducking under", shot=False)
        self.release(*MOVES)
        self.turn(0, math.degrees(math.atan2(u[1], u[0])))
        self.shift_to([burnt[0] - u[0] * back, burnt[1] - u[1] * back], tol=8, timeout=4)
        self.hold("duck")
        time.sleep(0.2)
        self.set_speed(self.top_speed)
        self.hold("forward")
        try:
            end = time.time() + 8  # crouched the player walks at about a third of the speed
            while time.time() < end:
                q = self.obs()
                ahead = (q["o"][0] - burnt[0]) * u[0] + (q["o"][1] - burnt[1]) * u[1]
                if ahead > 56:
                    break
        finally:
            self.release(*MOVES)
            self.release("duck")

    def beam_pass(self, s, p):
        """Past a laser beam across the way to p: ducked under where it crosses high, else
        jumped (the emitters say which, `beam_over`; a bent beam or one through a portal isn't
        found, and then the jump is tried first, as before). Burnt again by what was tried,
        the other way is tried once."""
        try:
            high = beam_over(self.lasers(), s["o"])
        except GameUnavailable:
            raise
        except (OSError, ValueError):
            high = None
        duck_first = high is not None and high > DUCK_HEIGHT
        self.event("laser", f"burnt at {fmt(s['o'])}: beam "
                   + (f"{high:.0f} over the feet" if high is not None else "not found")
                   + f": {'duck' if duck_first else 'jump'} first", shot=False)
        for way in ((self.duck_beam, self.jump_beam) if duck_first else (self.jump_beam, self.duck_beam)):
            hurt = getattr(self, "hurt_at", 0.0)
            way(s, p)
            self.obs()
            if getattr(self, "hurt_at", 0.0) <= hurt:
                return  # through it untouched
        self.event("laser", "burnt ducking and jumping it", shot=False)

    def move_toward(self, yaw, speed):
        """Moves toward heading `yaw` whatever way the view faces: forward/back and the strafe
        keys, at speeds in proportion."""
        phi = math.radians(yaw - self.yaw)
        f, l = math.cos(phi), math.sin(phi)
        self.set_speed(speed * abs(f))
        self.set_side_speed(speed * abs(l))
        for key, other, comp in (("forward", "back", f), ("moveleft", "moveright", l)):
            if abs(comp) < 0.2:
                self.release(key, other)
            else:
                self.release(other if comp > 0 else key)
                self.hold(key if comp > 0 else other)

    def set_side_speed(self, v):
        v = round(max(MIN_SPEED, min(self.side_speed, v)) / 10) * 10
        if v != getattr(self, "side_now", None):
            self.con.send(f"cl_sidespeed {v}")
            self.side_now = v

    def walk_cells(self, cells, tol=8.0):
        """Walks the route's cells (each one step or a drop from the last), straight lines
        merged."""
        if not cells:
            return self.obs()
        pts = [self.floor(c) for c in cells]
        keep = [p for i, p in enumerate(pts)
                if i in (0, len(pts) - 1) or not self._collinear(cells[i - 1], cells[i], cells[i + 1])]
        return self.walk(keep, tol)

    def walk_points(self, points, cells, tol=8.0):
        """Walks a leg's straightened walk (`points`: cells to head for in turn, straight
        lines checked clear by `glados-score route`), or its `cells` in routes made before those."""
        if points is None:
            return self.walk_cells(cells, tol)
        if not points:
            return self.obs()
        return self.walk([self.floor(c) for c in points], tol)

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

    def clear_aim(self, face, target, eye):
        """Where on `face` to aim from `eye`: its middle, or, with something solid in the way
        (a pane's frame, a panel's edge), a point offset in the face's plane that the shot
        reaches. Its middle where no trace comes back, as before."""
        n = DIRS[face["dir"]]
        for q in aim_points(target, face["dir"]):
            # 4 units out from the face, on the room's side of it: a clear line gets there.
            frac = self.trace_line(eye, [q[i] - n[i] * 4 for i in range(3)])
            if frac is None:
                return target
            if frac > 0.999:
                return list(q)
        self.event("aim", f"nothing on {face['cell']} {face['dir']} is in sight from "
                   f"{fmt(eye)}: firing at its middle", shot=False)
        return target

    def act_fire(self, leg):
        color, face = leg["color"], leg["face"]
        # A deployed angled panel sticks out of its face: aim at its middle.
        target = self.world(leg["aim"] if leg.get("aim") else face["centre"])
        # From over a ledge's edge, to look down past it: shuffle there first.
        eye = self.world(leg["eye"])
        if hdist(self.obs()["o"], eye) > 10:
            self.shift_to(eye)
        for k in range(2):
            # Devices take a moment: a field switched off by the button stood on, a panel.
            time.sleep(1.0 if "stand on" in leg["text"] and k == 0 else 1.5 * k)
            s = self.settle()
            # The middle of a face can be out of sight (a pane in the way) where the rest of
            # it is open; a panel's own aim point stands outside the face, so it stays.
            aim = target if leg.get("aim") else self.clear_aim(face, target, s["eye"])
            self.look_at(aim, s["eye"])
            time.sleep(0.15)
            before = {p["id"]: p for p in self.obs()["portals"]}
            self.firing = True
            try:
                self.tap(FIRE_KEY[color])
                time.sleep(0.5)
                s = self.obs()
            finally:
                self.firing = False
            self.last_fire = {"eye": [round(v, 1) for v in s["eye"]], "target": [round(v, 1) for v in aim],
                              "pitch": round(self.pitch, 2), "yaw": round(self.yaw, 2)}
            moved = [p for p in s["portals"] if before.get(p["id"]) != p]
            if moved:  # the first shot can make both portal entities: take the one there
                self.portal_ids[color] = min(moved, key=lambda p: math.dist(p["o"], aim))["id"]
            if self.portal_on(color, face, s) or (leg.get("aim") and any(
                    math.dist(p["o"], aim) < 64 for p in s["portals"]
                    if self.portal_ids.get(color) in (None, p["id"]))):
                return True
        return False

    def act_enter(self, leg):
        face = leg["face"]
        if not self.jumped and face["dir"] != "-z":
            # A wall portal: walk into it (looking a little aside if a held cube would
            # otherwise hang in a field beside it).
            yaw = yaw_of(DIRS[face["dir"]])
            self.turn(0, self.cube_yaw(self.obs(), yaw))
            speed = self.entry_speed(leg)
            end = time.time() + 5 + 150 / speed
            try:
                while not self.jumped and time.time() < end:
                    self.obs()
                    self.move_toward(yaw, speed)
            finally:
                self.release(*MOVES)
                self.set_speed(self.top_speed)
                self.set_side_speed(self.side_speed)
        elif not self.jumped:
            # A floor portal: the walk ends on it; step onto its middle.
            self.walk([self.floor(face["cell"])], tol=4, stop=False, timeout=4)
        went = self.jumped
        if went:
            self.land_clear()
        elif face["dir"] != "-z":
            s = self.obs()
            c = self.world(face["centre"])
            if hdist(s["o"], c) < 48:
                raise Stuck(f"at the portal on {face['cell']} {face['dir']} but not through it: "
                            "its partner is gone (fizzled?)")
        self.wait_landed()
        return went

    def entry_speed(self, leg):
        """How fast to walk into a wall portal: out of a wall portal the walk's speed carries
        on, and a fall after it drifts that far (into a field beside the landing, say). So:
        the speed whose fall out of the other portal ends in the middle of the landing cell."""
        out = leg.get("out")
        if not out or out["dir"] in ("-z", "+z"):
            return self.top_speed
        n = [-v for v in DIRS[out["dir"]]]  # out of the exit portal, into the room
        c = self.world(out["centre"])
        land = self.floor(leg["after"]["player"])
        # The player comes out with its middle 16 in front of the portal, feet ~36 under it.
        start = [c[0] + n[0] * 16, c[1] + n[1] * 16, c[2] - 36]
        dist = (land[0] - start[0]) * n[0] + (land[1] - start[1]) * n[1]
        height = start[2] - land[2]
        if height < 16:
            return self.top_speed
        t = math.sqrt(2 * height / 600.0)
        return max(MIN_SPEED, min(self.top_speed, dist / t))

    def land_clear(self, timeout=3):
        """Out of a portal: while falling, if the drift would carry the player across a live
        field the route doesn't cross, brake (in the air that snaps the speed down)."""
        end = time.time() + timeout
        while time.time() < end:
            s = self.obs()
            if abs(s["v"][2]) < 5 and self.speed(s) < 5:
                return
            # Where the drift goes in the next half second.
            ahead = [s["o"][0] + s["v"][0] * 0.5, s["o"][1] + s["v"][1] * 0.5, s["o"][2] + 36]
            here = [s["o"][0], s["o"][1], s["o"][2] + 36]
            hit = self.field_between(here, ahead, skip=self.intended) or self.field_near(ahead, 32)
            if hit and self.speed(s) > 20:
                self.move_toward(math.degrees(math.atan2(-s["v"][1], -s["v"][0])), self.top_speed)
            else:
                self.release(*MOVES)
        self.release(*MOVES)
        self.set_speed(self.top_speed)
        self.set_side_speed(self.side_speed)

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
            # Picked up: it comes along when the view moves. (Moving a little isn't enough: a
            # cube just dropped, or bumped by the player, moves on its own.)
            if now is not None and self.follows_view(now["id"]):
                now = next((c for c in self.obs()["cubes"] if c["id"] == cube["id"]), now)
                self.held_id = now["id"]
                self.hold_dist = math.dist(now["o"], self.obs()["eye"])
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
        return self.set_down(target, 14 if leg["button"] else 40)

    def set_down(self, target, tol, height=36.0):
        """Puts the held cube down on `target` (a floor point) from a held cube's reach away,
        facing it and looking down only so far that the cube hangs `height` over it. (Lining
        it up by looking at the target itself walks the player onto it, looking steeply down,
        and a cube held against the player that way won't let go.)"""
        for _ in range(8):
            s = self.obs()
            cube = self.holding(s)
            if cube is None:
                return False
            if hdist(cube["o"], target) < tol:
                break
            d = hdist(s["o"], target)
            if d > 16:
                u = ((target[0] - s["o"][0]) / d, (target[1] - s["o"][1]) / d)
            else:  # on top of it: back off the way we're facing
                u = (math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw)))
            dz = s["eye"][2] - (target[2] + height)
            reach = math.sqrt(max(self.hold_dist ** 2 - dz ** 2, 1.0))
            self.turn(math.degrees(math.atan2(dz, reach)), math.degrees(math.atan2(u[1], u[0])))
            self.shift_to([target[0] - u[0] * reach, target[1] - u[1] * reach])
            time.sleep(0.2)
        return self.let_go()

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
        # Into a wall portal: square in front of it, a held cube's reach and a bit back, and
        # walk straight at it (from aside, the cube can wedge beside the portal, or come out
        # moving sideways).
        d = DIRS[face["dir"]]
        c = self.world(face["centre"])
        s = self.obs()
        cube = self.holding(s)
        if cube is None:
            return False
        self.turn(0, yaw_of(d))
        time.sleep(0.3)
        back = self.hold_dist + 60
        self.shift_to([c[0] - d[0] * back, c[1] - d[1] * back], tol=4, timeout=6)
        self.look(0, yaw_of(d))
        time.sleep(0.3)
        self.hold("forward")
        end = time.time() + 5
        went = False
        try:
            while time.time() < end and not went:
                s = self.obs()
                now = next((q for q in s["cubes"] if q["id"] == cube["id"]), None)
                went = now is None or math.dist(now["o"], cube["o"]) > 160
                if now is not None and not went:
                    cube = now
        finally:
            self.release("forward")
        if not went:
            return False
        # Held through the portal, it hangs out of the other one, about over where it lands:
        # line it up there, then let go (a cube dropped short of a portal can stick halfway
        # through it).
        time.sleep(0.3)
        if leg.get("landing"):
            self.align_held(cube["id"], self.floor(leg["landing"]))
        self.tap("use", 0.15)
        self.held_id = None
        time.sleep(1.5)
        return True

    def align_held(self, cube_id, target, tol=10.0, tries=6):
        """Moves a cube held through a portal over `target` (xy) before it is let go. How the
        cube moves when the player steps sideways or looks up depends on the portals, so it is
        measured: a small step and a small tilt, then corrections solved from those."""
        def where():
            c = next((q for q in self.obs()["cubes"] if q["id"] == cube_id), None)
            return None if c is None else c["o"]

        p0 = where()
        if p0 is None:
            return
        if math.hypot(target[0] - p0[0], target[1] - p0[1]) < tol:
            return
        yaw = math.radians(self.yaw)
        right = (math.sin(yaw), -math.cos(yaw))
        o = self.obs()["o"]
        self.shift_to([o[0] + right[0] * 12, o[1] + right[1] * 12], tol=3, timeout=3)
        time.sleep(0.4)
        p1 = where()
        self.look(self.pitch - 6, self.yaw)
        time.sleep(0.4)
        p2 = where()
        if p1 is None or p2 is None:
            return
        o1 = self.obs()["o"]
        step = right[0] * (o1[0] - o[0]) + right[1] * (o1[1] - o[1])
        if abs(step) < 3:
            return
        # Columns: cube movement per unit of sideways step, per degree of looking up.
        a, c = (p1[0] - p0[0]) / step, (p1[1] - p0[1]) / step
        b, d = (p2[0] - p1[0]) / 6, (p2[1] - p1[1]) / 6
        det = a * d - b * c
        if abs(det) < 1e-3:
            print(f"    align: no handle on the cube (moves {a:.2f},{c:.2f} / {b:.2f},{d:.2f})", flush=True)
            return
        for _ in range(tries):
            p = where()
            if p is None:
                return
            ex, ey = target[0] - p[0], target[1] - p[1]
            if math.hypot(ex, ey) < tol:
                break
            u = max(-48.0, min(48.0, (d * ex - b * ey) / det))
            v = max(-20.0, min(20.0, (a * ey - c * ex) / det))
            o = self.obs()["o"]
            self.shift_to([o[0] + right[0] * u, o[1] + right[1] * u], tol=3, timeout=3)
            self.look(max(-80.0, min(80.0, self.pitch - v)), self.yaw)
            time.sleep(0.4)
        p = where()
        if p is not None:
            print(f"    align: cube {math.hypot(target[0] - p[0], target[1] - p[1]):.0f} from the landing", flush=True)

    def act_press(self, leg):
        """Presses a pedestal button: close up, looking at its button (on top of the column,
        about PEDESTAL_TOP over its base) rather than a guess at the cell. Holding a cube, use
        would drop it rather than press: it is put down first and picked up again after."""
        cube = self.holding(self.obs()) if self.held_id is not None else None
        if cube is not None:
            self.turn(30, self.yaw + 180)  # behind, out of the way of the button
            time.sleep(0.3)
            if not self.let_go():
                return False
            time.sleep(0.5)
        ok = self._press(leg)
        if cube is not None:
            s = self.obs()
            near = self.nearest_cube(s, s["o"], within=160)
            if near is None:
                return False
            self.look_at(near["o"])
            time.sleep(0.15)
            self.tap("use")
            time.sleep(0.4)
            if not self.follows_view(near["id"]):
                return False
            self.held_id = near["id"]
            self.hold_dist = math.dist(next(c for c in self.obs()["cubes"] if c["id"] == near["id"])["o"],
                                       self.obs()["eye"])
        return ok

    def _press(self, leg):
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
        took = time.time() - start
        print(f"    dash took {took:.1f} s of {leg['seconds']} s", flush=True)
        if leg["escape"]:
            # The timer only has to last to the door; what counts is getting out.
            return self.act_exit({"normal": self.exit_normal, "patient": False})
        return took <= leg["seconds"]

    def act_fetch(self, leg):
        """Presses a timed pedestal, runs to the cube at the path's end, picks it up and carries
        it along `carry` before the timer runs out (solver.rs `Action::Fetch`). Unverified in
        game: the solver's timing has the cube across the field with little to spare."""
        self.act_press(leg)
        start = time.time()
        if leg["path"][:-1]:
            self.walk_cells(leg["path"][:-1], tol=24)
        if not self.act_pickup({"look": leg["cube_look"]}):
            return False
        picked = time.time() - start
        self.walk_cells(leg["carry"], tol=24)
        took = time.time() - start
        print(f"    fetch: cube picked up at {picked:.1f} s, carried by {took:.1f} s of {leg['seconds']} s",
              flush=True)
        return True

    def act_reach(self, leg):
        """Stands on the floor button in `cell`, on its edge nearest the cube, and picks the cube
        up across the fizzler the button switches off (solver.rs `Action::Reach`). Unverified
        in game: done by hand in Workshop 68464713, with the cube put down close to the field."""
        button, at = self.floor(leg["cell"]), self.world(leg["look"])
        d = max(hdist(button, at), 1.0)
        u = ((at[0] - button[0]) / d, (at[1] - button[1]) / d)
        self.shift_to([button[0] + u[0] * 40, button[1] + u[1] * 40])
        time.sleep(0.5)  # the button goes down, and the fizzler off
        cube = self.nearest_cube(self.obs(), at, within=128)
        if cube is None:
            return False
        self.look_at(cube["o"])
        time.sleep(0.15)
        self.tap("use")
        time.sleep(0.4)
        if not self.follows_view(cube["id"]):
            return False
        self.held_id = cube["id"]
        now = next((c for c in self.obs()["cubes"] if c["id"] == cube["id"]), cube)
        self.hold_dist = math.dist(now["o"], self.obs()["eye"])
        return True

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
        if t is not None and abs(t["a"][0]) <= 30 and abs(t["a"][2]) <= 30:
            # Still standing, it shoots: that's a failure to retry, not an unknown.
            raise Stuck(f"turret at {fmt(t['o'])} still standing")
        return t is not None

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

    def act_jump(self, leg):
        """Runs off the edge of `from` toward `dir` and jumps the gap onto `to`, jumping as late
        as it dares: at walking speed a one-voxel gap leaves ~30 units to spare."""
        d = DIRS[leg["dir"]]
        take_off = self.floor(leg["from"])
        to = self.floor(leg["to"])
        self.turn(0, yaw_of(d))
        self.hold("forward")
        jumped, end = False, time.time() + 6
        while time.time() < end:
            s = self.obs()
            ahead = (s["o"][0] - take_off[0]) * d[0] + (s["o"][1] - take_off[1]) * d[1]
            if not jumped and ahead > 52:
                self.tap("jump")
                jumped = True
            if jumped and hdist(s["o"], to) < 30:
                break
        self.release("forward")
        return self.in_cell(self.wait_landed()["o"], leg["to"])

    def act_plate(self, leg):
        """Steps onto the faith plate and waits to come down where it throws."""
        try:
            self.walk([self.floor(leg["plate"])], tol=16, stop=False, timeout=6, on_jump="ignore",
                      launched=True)
        except Stuck:
            pass  # thrown before getting to its middle
        time.sleep(0.5)
        return self.in_cell(self.wait_landed(timeout=10)["o"], leg["to"])

    def act_climb(self, leg):
        self.walk_cells(leg["path"], tol=16)
        return self.in_cell(self.obs()["o"], leg["path"][-1])

    def act_funnel(self, leg):
        """Steps into the funnel, floats along it to its last cell on the way, then steps
        out toward `out`."""
        via = leg["via"]
        try:
            self.walk([self.floor(via[0])], tol=24, stop=False, timeout=6, on_jump="ignore")
        except Stuck:
            pass  # caught by the funnel before getting there
        last = self.world([via[-1][0] + 0.5, via[-1][1] + 0.5, via[-1][2] + 0.3])
        end = time.time() + 30
        while time.time() < end:
            if math.dist(self.obs()["o"], last) < 48:
                break
            time.sleep(0.05)
        try:
            self.walk([self.floor(leg["out"])], tol=24, timeout=8, on_jump="ignore")
        except Stuck:
            pass
        return self.in_cell(self.wait_landed()["o"], leg["to"])

    def act_ride(self, leg):
        """Waits on the platform until it has carried the player to its other stop."""
        end = time.time() + 20
        while time.time() < end:
            if self.in_cell(self.obs()["o"], leg["to"]):
                return True
            time.sleep(0.1)
        return False

    def act_dive(self, leg):
        """Off the ledge (with a running jump if the portal is two cells out) into the floor
        portal, and out of the other one at speed. The run-up speed is worked out so the fall
        ends in the portal's middle (on its rim the player just lands): in the air the keys
        hardly help (braking snaps the speed down to about 73 and it can't be got back)."""
        d = DIRS[leg["dir"]]
        edge = self.floor(leg["edge"])
        portal = self.floor(leg["face"]["cell"])
        jump = leg["out"] > 1
        height = edge[2] - portal[2]
        dist = (portal[0] - edge[0]) * d[0] + (portal[1] - edge[1]) * d[1]
        # Where the fall starts: the player's box (16 either side) off the edge, or the jump
        # (taken 40 past the edge cell's middle, 226 up).
        start = 40.0 if jump else VOXEL / 2 + 16
        vz = 226.0 if jump else 0.0
        t = (vz + math.sqrt(vz * vz + 2 * 600.0 * height)) / 600.0
        speed = max(MIN_SPEED, min(self.top_speed, (dist - start) / t))
        print(f"    dive: {dist - start:.0f} to go over a {height:.0f} fall: run at {speed:.0f}", flush=True)
        self.turn(0, yaw_of(d))
        # Back up to the far side of the edge cell for a run-up.
        self.shift_to([edge[0] - d[0] * 40, edge[1] - d[1] * 40])
        self.set_speed(speed)
        self.hold("forward")
        jumped, end = False, time.time() + 10
        try:
            while time.time() < end and not self.jumped:
                s = self.obs()
                ahead = (s["o"][0] - edge[0]) * d[0] + (s["o"][1] - edge[1]) * d[1]
                if jump and not jumped and ahead > 40:
                    self.tap("jump")
                    jumped = True
                if s["o"][2] < edge[2] - 8:
                    self.release(*MOVES)  # falling: hands off
        finally:
            self.release(*MOVES)
            self.set_speed(self.top_speed)
        went = self.jumped
        return went and self.in_cell(self.wait_landed(timeout=10)["o"], leg["landing"])

    def air_steer(self, s, target, gravity=600.0):
        """One tick of air control toward landing on `target` (a floor point): the velocity
        that would get there by the time the fall reaches its height, pushed for with the
        move keys whatever way the view faces. In the air the keys only add speed up to 30
        units a second along where they push (Source's air control), so in practice this
        brakes: it aims a little long (better to brake late than fall short) and only pushes
        where that does something."""
        z, vz = s["o"][2] - target[2], s["v"][2]
        # Time until the fall comes down to the target's height: z + vz t - g t^2 / 2 = 0.
        t = (vz + math.sqrt(max(vz * vz + 2 * gravity * max(z, 0.0), 0.0))) / gravity
        t = max(t, 0.1)
        v = s["v"][:2]
        sp = math.hypot(*v)
        long = [v[i] / sp * 10 if sp > 1 else 0.0 for i in range(2)]
        want = [(target[i] + long[i] - s["o"][i]) / t for i in range(2)]
        dv = [want[i] - v[i] for i in range(2)]
        n = math.hypot(*dv)
        if n < 20 or (v[0] * dv[0] + v[1] * dv[1]) / n >= 30:
            self.release(*MOVES)  # close enough, or pushing that way can't add speed
        else:
            self.move_toward(math.degrees(math.atan2(dv[1], dv[0])), self.top_speed)

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
        if not self.jumped:
            s = self.obs()
            door = self.floor(self.exit_cell)
            near = hdist(s["o"], door) < 1.5 * VOXEL
            if near and leg.get("patient", True) and self.exit_wait > 0:
                # The door may be about to open: a laser burning through the turrets in its
                # way, a platform arriving. Wait at it, still walking into it. (A timed
                # escape doesn't wait: its door shuts again, and the time is the point.)
                self.event("waiting", f"at the exit door, shut: waiting up to {self.exit_wait:.0f} s",
                           shot=False)
                self.hold("forward")
                end = time.time() + self.exit_wait
                try:
                    while not self.jumped and time.time() < end:
                        self.obs(timeout=3)
                except GameUnavailable:
                    self.held.discard("forward")
                    return True
                self.release("forward")
            if not self.jumped:
                s = self.obs()
                raise Stuck(f"not out of the exit: stopped at {fmt(s['o'])}"
                            + (", at its door (shut?)" if near
                               else f", {hdist(s['o'], door):.0f} from its door"))
        if self.jumped:
            # Out: leave before the elevator ends the level. A preview built from the console
            # has no editor session to go back to, so the level change shuts the server down
            # and leaves a modal "Disconnected" dialog that blocks reopening the editor.
            self.con.send("disconnect")
        return self.jumped

    def act_attempt(self, leg):
        """Goblin's step that fits no move (`glados-score route --plan`): after the walk
        toward it, a shot at the face it names (from the nearest cell that sees it, else from
        here), or nothing more. Never counts as done."""
        if leg.get("verb") == "fire" and leg.get("face"):
            target = self.world(leg["face"]["centre"])
            eye = self.world(leg["eye"])
            if hdist(self.obs()["o"], eye) > 10:
                try:
                    self.shift_to(eye)
                except Stuck:
                    pass
            s = self.settle()
            self.look_at(target, s["eye"])
            time.sleep(0.15)
            self.firing = True
            try:
                self.tap(FIRE_KEY[leg["color"]])
                time.sleep(0.5)
                self.obs()
            finally:
                self.firing = False
        time.sleep(2.0)
        self.screenshot("net-failed")
        return False

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
            if want == "gone":
                continue
            if want is None:
                if held is None or not self.follows_view(held["id"]):
                    out.append("not holding the cube")
                continue
            here = next((c for c in free if self.in_cell(c["o"], want, slack=16)), None)
            if here is None:
                f = self.floor(want)
                near = min(free, key=lambda c: math.dist(c["o"], f), default=None)
                out.append(f"no cube in cell {want}" + (
                    "" if near is None else f" (nearest at {fmt(near['o'])}, cell floor {fmt(f)})"))
            else:
                free.remove(here)
        return out

    def recover(self, after):
        """Teleports things to where the route expects them after a leg."""
        s = self.obs()
        if None not in after["cubes"]:
            near = self.nearest_cube(s, s["eye"], within=120)
            if near is not None and self.follows_view(near["id"]):
                self.held_id = near["id"]
                self.let_go()
            self.held_id = None
            s = self.obs()
        if not self.in_cell(s["o"], after["player"]):
            self.teleport(self.floor(after["player"]))
        s = self.obs()
        for k, (color, face) in enumerate(zip(COLORS, after["portals"])):
            if face and not self.portal_on(color, face, s):
                c = self.world(face["centre"])
                n = [-v for v in DIRS[face["dir"]]]
                pitch = -90 if n[2] > 0 else 90 if n[2] < 0 else 0
                self.firing = True
                self.con.send(f"portal_place 0 {k} {c[0] + n[0]:.1f} {c[1] + n[1]:.1f} "
                              f"{c[2] + n[2]:.1f} {pitch} {yaw_of(n) if not n[2] else 0:.0f} 0")
                time.sleep(0.3)
                self.obs()
                self.firing = False
                self.portal_ids.pop(color, None)
        s = self.obs()
        held = self.holding(s)
        used = {held["id"]} if held else set()
        for want in after["cubes"]:
            if want == "gone":
                continue
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
    for p in (ROOT / "target" / "release" / "glados-score.exe", ROOT / "target" / "debug" / "glados-score.exe"):
        if p.exists():
            return p
    sys.exit("no glados-score.exe: run `cargo build --release`")


class NoRoute(Exception):
    """`glados-score route` can't play the level."""


def route(exe, level, plan=None, mode=None):
    """(head, legs) of `glados-score route` for the level: the solver's plan, or with `plan` (plan
    tokens) that one, as far as it fits, routed by the `mode` model (conservative, the default,
    or liberal) (src/bin/glados-score.rs print_route)."""
    cmd = ([str(exe), "route"] + (["--plan", plan] if plan is not None else [])
           + (["--mode", mode] if plan is not None and mode else []) + [str(level)])
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        raise NoRoute(f"{level}: {out.stderr.strip() or out.stdout.strip()}")
    return route_lines(out.stdout.splitlines())


def route_lines(lines):
    """(head, legs) from `glados-score route`'s JSON lines."""
    rows = [json.loads(l) for l in lines if l.strip()]
    return rows[0], rows[1:]


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
    invalid = [l.split("INVALID ITEM:", 1)[1].strip() for l in exported.splitlines() if "INVALID ITEM:" in l]
    if invalid:
        # Building it would only stop at a modal "Build Failed" dialog, which blocks the editor
        # until someone clicks it.
        raise InvalidLevel("the editor marks items invalid: " + "; ".join(invalid))
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
    cube's starting cell, and for the cubes to come to rest (one can bounce or roll on)."""
    end = time.time() + timeout
    still = None
    prev = None
    while time.time() < end:
        s = game.obs()
        here = {c["id"]: c["o"] for c in s["cubes"]}
        moving = prev is None or prev.keys() != here.keys() or any(
            math.dist(here[k], prev[k]) > 1 for k in here)
        prev = here
        homed = all(any(game.in_cell(c["o"], want["home"], slack=16) for c in s["cubes"])
                    for want in head["cubes"])
        if moving or not homed and len(s["cubes"]) < len(head["cubes"]):
            still = None
        else:
            still = still or time.time()
            if time.time() - still > 1.0:
                return
        time.sleep(0.2)


def walk_out(game, seconds):
    """Holds forward, keeping the heading, until the player goes through a linked door."""
    game.jumped = False
    game.hold("forward")
    end = time.time() + seconds
    while not game.jumped and time.time() < end:
        game.obs()
    game.release("forward")
    return game.jumped


# Where a leg's action starts matters (aim, reach, a run-up): a careful retry puts the player
# exactly there after walking. For the rest the walk is the action, or a device moves them.
PLACED = ("fire", "pickup", "putdown", "toss", "throw", "press", "knock", "enter", "leap", "dive",
          "bounce", "fetch", "reach", "jump")
# Lining up a cube or reaching for something can take the player half a cell over.
SLACK = ("pickup", "putdown", "toss", "throw", "press", "knock", "funnel", "plate", "dive", "ride",
         "reach")


def intended_fields(game, before, leg):
    """Hazards the route's own walk crosses (cell to cell): crossing those is the plan."""
    faces = {}
    for h, hz in enumerate(game.head.get("hazards", [])):
        for f in hz["faces"]:
            d = DIRS[f["dir"]]
            faces[(tuple(f["cell"]), f["dir"])] = h
            other = tuple(f["cell"][i] + d[i] for i in range(3))
            faces[(other, ("+" if f["dir"][0] == "-" else "-") + f["dir"][1])] = h
    out = set()
    cells = [before["player"]] + leg["walk"]
    for a, b in zip(cells, cells[1:]):
        step_ = tuple(b[i] - a[i] for i in range(3))
        for name, d in DIRS.items():
            if step_[:2] == d[:2] and d[2] == 0 and (tuple(a), name) in faces:
                out.add(faces[(tuple(a), name)])
    return out


def attempt(game, leg, before, careful=False):
    """One go at a leg. Careful: the route's own cells walked (not straightened), the view
    snapped, and the player put exactly where the action starts. Returns what happened."""
    t0, n_events = time.time(), len(game.events)
    ok, why = False, ""
    rate = game.turn_rate
    game.last_fire = None
    intended = game.intended = intended_fields(game, before, leg)
    try:
        if careful:
            game.turn_rate = math.inf
            game.walk_cells(leg["walk"])
            if leg["action"] in PLACED and (leg["walk"] or leg["action"] == "fire"):
                if leg["action"] == "fire":
                    e = game.world(leg["eye"])
                    feet = [e[0], e[1], game.floor(leg["cell"])[2]]
                else:
                    feet = game.floor(leg["walk"][-1])
                held = game.held_id
                game.teleport(feet)
                game.held_id = held
                game.settle()
        else:
            game.walk_points(leg.get("points"), leg["walk"])
        ok = getattr(game, "act_" + leg["action"])(leg)
    except Stuck as e:
        why = str(e)
    finally:
        game.release_all()
        game.turn_rate = rate
    if leg["walk"] and leg["after"]["player"] != leg["walk"][-1] and leg["action"] in (
            "press", "putdown", "fire", "pickup", "toss", "throw", "knock"):
        time.sleep(2.0)  # a device carried the player: let it get there
        game.wait_landed()
    exit_leg = leg["action"] == "exit" or leg.get("escape")
    problems = [] if exit_leg else game.problems(leg["after"], 64 if leg["action"] in SLACK else 12)
    # A cube for a button has to rest near its middle (in its cell but on the plate's rim, it
    # doesn't press it; that only shows later, as a door that stays shut).
    target = leg.get("at") if leg["action"] == "putdown" and leg.get("button") else         leg.get("landing") if leg["action"] in ("throw", "toss") and "button" in leg["text"] else None
    if target and not problems:
        f = game.floor(target)
        s = game.obs()
        near = min(s["cubes"], key=lambda c: hdist(c["o"], f), default=None)
        if near is not None and hdist(near["o"], f) > 30:
            problems.append(f"cube {hdist(near['o'], f):.0f} from the middle of the button in {target}")
    events = game.events[n_events:]
    for e in events:
        if e["kind"] == "crossed field":
            e["intended"] = e.get("hazard") in intended
    r = {"ok": bool(ok), "why": why, "problems": problems, "events": events,
         "seconds": round(time.time() - t0, 1)}
    if game.last_fire:
        r["fire"] = game.last_fire
    return r


def good(r, exit_leg):
    return r["ok"] if exit_leg else (r["ok"] or not r["why"]) and not r["problems"]


def classify(game, leg, first, retry):
    """Why a leg failed: exec (the careful retry worked: the harness's playing is at fault),
    route (the route's own path crosses a field it doesn't plan to), blocked (the route's own
    cells can't be walked: the route or the solver's idea of the level is wrong; look),
    plan? (done carefully it still fails: the game may disagree with the solver; look at the
    evidence) or unknown (no retry)."""
    if retry is None:
        return "unknown"
    if good(retry, leg["action"] == "exit" or leg.get("escape")):
        return "exec"
    if any(e["kind"] == "crossed field" and not e["intended"] for e in retry["events"]):
        return "route"
    if retry["why"].startswith("stuck") and "walking" in retry["why"]:
        return "blocked"
    return "plan?"


def probe_target(game, leg):
    """What a failed leg was about, to look at: a world point."""
    a = leg["action"]
    if a in ("fire", "enter", "throw") and leg.get("face"):
        return game.world(leg["aim"] if leg.get("aim") else leg["face"]["centre"])
    if a in ("bounce", "leap") and leg.get("gel"):
        return game.floor(leg["gel"])
    if a == "putdown":
        return game.floor(leg["at"])
    if leg.get("look"):
        return game.world(leg["look"])
    if leg.get("to"):
        return game.floor(leg["to"])
    if leg["walk"]:
        s = game.obs()
        # the first cell of the walk not reached
        for c in leg["walk"]:
            if not game.in_cell(s["o"], c):
                return game.floor(c)
    return None


def probe(game, leg, r):
    """Evidence about the geometry for a leg that failed done carefully: a screenshot looking
    at what the leg was about, and for a shot, how far a straight line gets."""
    out = {}
    if r.get("why", "").startswith("stuck"):
        o = game.obs()["o"]
        out["near"] = game.near([o[0], o[1], o[2] + 36], 96)
    target = probe_target(game, leg)
    if target is not None:
        game.look_at(target)
        time.sleep(0.3)
        out["look_shot"] = game.screenshot("look")
    if leg["action"] == "fire":
        eye = r.get("fire", {}).get("eye") or game.world(leg["eye"])
        target = game.world(leg["aim"] if leg.get("aim") else leg["face"]["centre"])
        hit = game.field_between(eye, target)
        if hit:
            out["through_field"] = f"{hit[1]} {hit[0]} (on, as the route has it) at {fmt(hit[2])}"
        frac = game.trace_line(eye, target)
        if frac is not None:
            d = math.dist(eye, target)
            out["line_of_sight"] = {"fraction": round(frac, 3), "short_by": round(d * (1 - frac), 1)}
    return out


def play(game, head, legs, deadline=None, report=None, on_leg=None):
    """Plays a route; returns one record per leg (also printed, and passed to `report`).

    Before each leg the game is saved. A leg that doesn't work out is screenshotted, loaded
    back and tried once more carefully (`attempt`); what the retry shows classifies it
    (`classify`). Only if that fails too is it put right by teleporting (recover), so the rest
    can be played. Status: ok, unconfirmed (the action couldn't see that it worked, e.g. a
    turret still standing, but nothing the route tracks is wrong), retried (worked carefully),
    recovered (put right by teleporting), failed, or skipped (out of time). An "attempt" leg
    (Goblin's step that fits no move) is played once, as is, and is net-failed.
    `on_leg(event, rec)`: "start" before each leg, "done" with its record after."""
    game.exit_normal = head["exit"]["normal"]
    game.exit_cell = head["exit"]["cell"]
    game.head = head
    game.set_hazards(head)
    results = []
    game.leg_no = 0
    try:
        arrive(game, head)
        game.arrival = "ok"
    except Stuck as e:
        print(f"  arrival: {e}; teleporting in", flush=True)
        game.arrival = f"teleported: {e}"
        game.teleport(game.floor(head["entry"]["cell"]))
    wait_for_cubes(game, head)
    before = head["start"]
    start = game.problems(before, 12)
    if start:
        left = game.recover(before)
        rec = {"leg": 0, "action": "start", "text": "the level as built", "status": "fixed" if not left else "off",
               "problems": start, "after_recovery": left}
        results.append(rec)
        print("  start differs from the route's: " + "; ".join(start)
              + ("" if not left else "; after recovery: " + "; ".join(left)), flush=True)
        if report:
            report(rec)
    for leg in legs:
        game.leg_no = leg["leg"]
        game.fields_on = set(leg.get("fields", []))
        rec = {"leg": leg["leg"], "action": leg["action"], "text": leg["text"]}
        if deadline and time.time() > deadline:
            rec.update(status="skipped", cls=None)
            results.append(rec)
            if report:
                report(rec)
            continue
        if getattr(game, "pid", None) and not portal2.focused(game.pid):
            game.event("focus lost", f"refocused: {portal2.focus(game.pid)}", shot=False)
        if on_leg:
            on_leg("start", dict(rec))
        game.show(leg["text"] if getattr(game, "stream", False) else f"{leg['leg']}/{len(legs)}: {leg['text']}")
        if leg["action"] == "attempt":
            rec["first"] = attempt(game, leg, before)
            rec.update(status="net-failed", verb=leg.get("verb"), step=leg.get("step"), problem=leg.get("problem"))
            results.append(rec)
            print(f"  net-failed  {leg['text']}: {leg.get('problem')}", flush=True)
            if report:
                report(rec)
            if on_leg:
                on_leg("done", rec)
            continue
        saved = game.save()
        exit_leg = leg["action"] == "exit" or leg.get("escape")
        first = attempt(game, leg, before)
        rec["first"] = first
        retry = None
        if good(first, exit_leg):
            status = "ok" if first["ok"] or exit_leg else "unconfirmed"
        else:
            rec["fail_shot"] = game.screenshot("failed")
            time.sleep(0.3)
            if saved and not (exit_leg and first["ok"]):
                try:
                    game.load(before)
                    retry = attempt(game, leg, before, careful=True)
                except Stuck as e:
                    retry = {"ok": False, "why": f"load: {e}", "problems": [], "events": [], "seconds": 0}
                rec["retry"] = retry
            if retry is not None and good(retry, exit_leg):
                status = "retried"
            elif exit_leg:
                status = "failed"
            else:
                if retry is not None:
                    rec["retry_shot"] = game.screenshot("retry-failed")
                    rec["probe"] = probe(game, leg, retry)
                left = game.recover(leg["after"])
                status = "recovered" if not left else "failed"
                rec["after_recovery"] = left
            rec["cls"] = classify(game, leg, first, retry)
            # After a leg was put right by teleporting, the game may not be quite what the
            # route has (a button not held, a door shut): later failures may come from that.
            rec["tainted"] = any(r["status"] in ("recovered", "failed") for r in results)
        rec["status"] = status
        results.append(rec)
        why = [first["why"]] * bool(first["why"]) + first["problems"]
        print(f"  {status:11} {leg['text']}" + (f"  [{rec['cls']}]" if rec.get("cls") else "")
              + "".join(f"\n              {p}" for p in why)
              + ("" if retry is None else "\n              careful retry: "
                 + ("worked" if good(retry, exit_leg) else "; ".join([retry["why"]] * bool(retry["why"]) + retry["problems"]) or "failed")),
              flush=True)
        if report:
            report(rec)
        if on_leg:
            on_leg("done", rec)
        before = leg["after"]
    return results


# --- the spike -----------------------------------------------------------------------------

def spike_check(game, head, exported):
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

def level_key(path):
    """A name for a level's report folder: its path under the repository, flattened."""
    path = Path(path).resolve()
    try:
        rel = path.relative_to(ROOT)
    except ValueError:
        rel = Path(path.name)
    return re.sub(r"[^\w.-]+", "_", str(rel.with_suffix("")).replace("\\", "/").replace("/", "_"))


def collect_shots(game, folder):
    """Moves the level's screenshots from the game's folder into `folder`."""
    src = portal2.GAME / "portal2" / "screenshots"
    end = time.time() + 3
    for name in game.shots:
        f = src / f"{name}.jpg"
        while not f.exists() and time.time() < end:
            time.sleep(0.2)
        if f.exists():
            f.replace(folder / f.name)


def tally(results):
    keys = ("ok", "unconfirmed", "retried", "recovered", "failed", "skipped", "net-failed")
    counts = {k: sum(r["status"] == k for r in results) for k in keys}
    classes = {}
    for r in results:
        if r.get("cls"):
            classes[r["cls"]] = classes.get(r["cls"], 0) + 1
    return counts, classes


class Session:
    """A game to play levels in, one after another: launched (restarted every `restart_every`
    levels and after it goes down), or one started with -insecure and attached on `port`.
    Keeps the editor autosaves the levels overwrite and puts them back on close (so never kill
    the process using it: stop it, or kill portal2.exe first). Use it in a with block.

        with Session(launch=True) as s:
            row, results = s.play_level(path, *route(exe, path), folder)
    """

    MAX_RESTARTS = 6  # game restarts after it went down, over the session

    def __init__(self, launch=False, port=None, password="", restart_every=20, turn_rate=TURN_RATE,
                 slow=1.0, keep=False, stream=False, exit_wait=EXIT_WAIT):
        self.launch, self.port, self.password = launch, port, password
        self.restart_every, self.turn_rate, self.slow = restart_every, turn_rate, slow
        self.exit_wait = exit_wait
        self.keep, self.stream = keep, stream
        self.pid = self.con = None
        self.played = 0  # levels since the game started
        self.downs = 0

    def __enter__(self):
        self.saved = {p: p.read_bytes() for p in portal2.autosaves()}
        SCRIPT.parent.mkdir(parents=True, exist_ok=True)
        SCRIPT.write_text(VSCRIPT, encoding="ascii")
        return self

    def __exit__(self, *exc):
        SCRIPT.unlink(missing_ok=True)
        if self.pid is not None and not self.keep:
            portal2.quit_game(self.saved, self.pid, self.con)
        elif self.pid is None:
            portal2.VMF.unlink(missing_ok=True)
        self.pid = self.con = None
        return False

    def ready(self):
        """A game with the editor open."""
        if self.con is None:
            if self.launch:
                self.pid, self.con = portal2.launch(self.saved)
            else:
                proc = portal2.game_process()
                if proc is None or "-insecure" not in proc[1].split():
                    raise GameUnavailable("no game started with -insecure (or use --launch)")
                if not self.port:
                    raise GameUnavailable("pass --port (and --password) for a game you started")
                self.con = Console(self.port, self.password)
                open_editor(self.con)  # from wherever the game is, e.g. its main menu
            self.played = 0
        elif self.played:
            open_editor(self.con)

    def restart(self):
        """Ends the game this session launched (it is launched again for the next level). A game
        this session didn't launch (someone is playing) is never touched."""
        if self.pid is not None:
            portal2.quit_game(self.saved, self.pid, self.con)
        self.pid = self.con = None

    def play_level(self, path, head, legs, folder=None, on_leg=None, timeout=900, spike=False, retry=None):
        """Loads, builds and plays one level's route; returns (summary row, leg records). With
        `folder`, keeps the evidence there (route, legs, trace, events, screenshots). A level the
        editor marks invalid comes back as an error row. If the game goes down, a launched one
        is restarted and the level played again once (`retry`, default: when launching); a
        level that brings it down twice is an error row. A game this session didn't launch
        (someone already playing) is never restarted: that raises GameUnavailable."""
        retry = self.launch if retry is None else retry
        key = level_key(path)
        for tries in (1, 2):
            try:
                return self._play(path, head, legs, folder, on_leg, timeout, spike, key)
            except GameUnavailable as e:
                # A game someone else is playing: this session didn't launch it, so it must not
                # restart (kill) it, and retrying would only spin. Refuse instead.
                if not retry or (self.pid is None and portal2.game_process() is not None):
                    raise
                self.downs += 1
                print(f"{path}: Portal 2 unavailable ({e}); restarting it", flush=True)
                self.restart()
                if self.downs > self.MAX_RESTARTS:
                    raise
                if tries == 2:
                    return {"key": key, "level": str(path), "error": f"game unavailable: {e}"}, []

    def _play(self, path, head, legs, folder, on_leg, timeout, spike, key):
        self.ready()
        print(f"{path}: \"{head['title']}\", {len(legs)} legs", flush=True)
        t0 = time.time()
        try:
            exported = load_level(self.con, path)
        except InvalidLevel as e:
            print(f"{path}: invalid in the editor: {e}", flush=True)
            self.played += 1
            return {"key": key, "level": str(path), "error": f"invalid: {e}"}, []
        if self.pid is not None and not portal2.focused(self.pid):
            print(f"  game window not in front; focus(): {portal2.focus(self.pid)}", flush=True)
        game = Game(self.con, calibrate(head) or (0.0, 0.0, 0.0), self.turn_rate, self.exit_wait)
        game.pid = self.pid
        game.stream = self.stream
        if self.slow != 1.0:
            self.con.send(f"host_timescale {self.slow}")
        if spike:
            spike_check(game, head, exported)
            return None, []
        report = None
        if folder:
            folder = Path(folder)
            if folder.exists():
                shutil.rmtree(folder)  # a level cut short: play it again from the start
            folder.mkdir(parents=True)
            (folder / "route.jsonl").write_text(
                "\n".join(json.dumps(x) for x in [head] + legs) + "\n", encoding="utf-8")
            game.shot_prefix = f"glados-{key[-40:]}"

            def report(rec, folder=folder):
                with open(folder / "legs.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
        results = play(game, head, legs, deadline=time.time() + timeout, report=report, on_leg=on_leg)
        counts, classes = tally(results)
        row = {"key": key, "level": str(path), "title": head["title"], "legs": len(legs),
               "counts": counts, "classes": classes, "arrival": game.arrival,
               "events": len(game.events), "seconds": round(time.time() - t0)}
        if "net" in head:
            row["net"] = head["net"]
        if folder:
            with open(folder / "trace.jsonl", "w", encoding="utf-8") as f:
                for t in game.trace:
                    f.write(json.dumps(t) + "\n")
            with open(folder / "events.jsonl", "w", encoding="utf-8") as f:
                for e in game.events:
                    f.write(json.dumps(e) + "\n")
            collect_shots(game, folder)
        self.played += 1
        time.sleep(2)
        if self.launch and self.played >= self.restart_every:
            self.restart()
        return row, results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("levels", nargs="*", help=".p2c files (or folders of them)")
    ap.add_argument("--list", help="a file naming more .p2c files, one a line")
    ap.add_argument("--plans", help="JSONL of levels with the plan to play: p2c, plan, mode")
    ap.add_argument("--launch", action="store_true", help="start the game (and quit it at the end)")
    ap.add_argument("--keep", action="store_true", help="leave a launched game running")
    ap.add_argument("--port", type=int, help="console port of a game you started")
    ap.add_argument("--password", default="", help="its -netconpassword")
    ap.add_argument("--slow", type=float, default=1.0, help="host_timescale while playing")
    ap.add_argument("--fast", action="store_true",
                    help="snap the view instead of turning smoothly (for validation runs)")
    ap.add_argument("--spike", action="store_true", help="check the harness on the first level")
    ap.add_argument("--glados", help="glados-score.exe (default: target/release, else target/debug)")
    ap.add_argument("--out", help="folder for reports: per level legs.jsonl, trace.jsonl and "
                    "screenshots, and summary.jsonl (levels already in it are skipped); "
                    "a file named STOP there ends the batch after the level being played")
    ap.add_argument("--restart-every", type=int, default=20,
                    help="with --launch, restart the game after this many levels")
    ap.add_argument("--level-timeout", type=float, default=900, help="seconds per level")
    ap.add_argument("--exit-wait", type=float, default=EXIT_WAIT,
                    help="seconds to wait at a shut exit door (a laser burning through the "
                         "turrets in its way, a platform arriving); 0 doesn't wait")
    ap.add_argument("--stream", action="store_true",
                    help="for watching on stream: smooth turning, short on-screen step text")
    a = ap.parse_args()
    exe = glados_exe(a.glados)
    names = a.levels + (Path(a.list).read_text().split() if a.list else [])
    given = {}  # level path -> (plan, mode)
    if a.plans:
        base = Path(a.plans).parent
        for line in Path(a.plans).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                p = Path(r["p2c"]) if Path(r["p2c"]).is_absolute() else base / r["p2c"]
                given[p] = (r["plan"], r.get("mode"))
    if not names and not given:
        ap.error("no levels")
    paths = [g for f in map(Path, names) for g in (sorted(f.glob("*.p2c")) if f.is_dir() else [f])]
    paths += [p for p in given if p not in paths]
    out = Path(a.out) if a.out else None
    done = set()
    if out:
        out.mkdir(parents=True, exist_ok=True)
        summary_file = out / "summary.jsonl"
        if summary_file.exists():
            done = {json.loads(l)["key"] for l in summary_file.read_text().splitlines() if l.strip()}

    def summarize(row):
        if out:
            with open(out / "summary.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

    routes = []
    for p in paths:
        if level_key(p) in done:
            continue
        try:
            routes.append((p, *route(exe, p, *given.get(p, (None, None)))))
        except NoRoute as e:
            if not out:
                sys.exit(str(e))  # refuse unplayable levels up front
            print(f"{e}", flush=True)
            summarize({"key": level_key(p), "level": str(p), "error": f"no route: {e}"})
    summary = []
    try:
        with Session(launch=a.launch, port=a.port, password=a.password, restart_every=a.restart_every,
                     turn_rate=math.inf if a.fast else TURN_RATE, slow=a.slow, keep=a.keep,
                     stream=a.stream, exit_wait=a.exit_wait) as session:
            for path, head, legs in routes:
                if out and (out / "STOP").exists():
                    print(f"{out / 'STOP'} found: stopping", flush=True)
                    break
                folder = out / level_key(path) if out else None
                row, results = session.play_level(path, head, legs, folder, timeout=a.level_timeout,
                                                  spike=a.spike, retry=bool(a.launch and out))
                if a.spike:
                    break
                summarize(row)
                if "error" not in row:
                    summary.append((path, results))
    except GameUnavailable as e:
        print(f"Portal 2 unavailable: {e}", file=sys.stderr)
        sys.exit(portal2.UNAVAILABLE)
    for path, results in summary:
        counts, classes = tally(results)
        print(f"{path}: " + ", ".join(f"{v} {k}" for k, v in counts.items() if v)
              + ("" if not classes else "  (" + ", ".join(f"{v} {k}" for k, v in classes.items()) + ")"))


if __name__ == "__main__":
    main()
