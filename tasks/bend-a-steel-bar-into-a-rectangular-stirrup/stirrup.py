"""On the station table: a steel bending table (a plate 160 x 290 x 15 mm) 28 cm in front of the arm bases, a little
right of the midline. A 16 mm bending pin stands on it, with three 10 mm guide pins and two low guide strips that
make a channel for the bar, and a red bending lever pivots on the bending pin (a roller near the pivot, a black handle
150 mm out). A straight 3 mm mild steel bar, 255 mm long, lies in the channel with three white chalk marks; its free
end lies 88 mm past the bending pin, towards the robot's right. The robot must bend it into a rectangular stirrup,
80 x 50 mm inside, making each bend at its chalk mark, all the same way, so that the two ends meet.

Run it:  inspect-robots run --embodiment yam_arena -E task=stirrup.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "..." --scorer task_score ...
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rodkit  # noqa: E402
from rodkit import Hinge, body, box, prism12, rod  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

rodkit.install()
SCENE = "station"
TOP = TABLE_TOP[SCENE]
MM = 0.001
STEEL = (0.55, 0.56, 0.58)
DARK = (0.22, 0.22, 0.24)
RED = (0.72, 0.15, 0.12)

# -- the bending table (fixed; frame: origin on the table under the bending pin's centre) --------------------------
PLATE_H = 15.0 * MM
PIN = (0.28, -0.06)  # the bending pin's centre (world x, y)
PIN_R = 8.0 * MM
PIN_H = 22.0 * MM
ROD_T = 3.0 * MM
# WHY 1.3 mm off the pin (2026-11-09, sim trace): the 12-sided pin's corners stand 0.28 mm proud of its flats; at
# 0.3 mm the wrapping rod ran into them and could not wrap, and the joints beside the bend bent back instead.
GAP = float(os.environ.get("PIN_GAP", "1.3")) * MM
ROD_X = PIN[0] - (PIN_R + ROD_T / 2 + GAP)  # the rod runs along y on the robot's side of the pin
STOP_R = 5.0 * MM
# WHY a tight guide (2026-11-09, sim trace): with 1.3 mm each side at the back and front pins the feed side swung 6
# degrees as the next bend was made, and the left fingertip then met the rod's end 11 mm off centre and kinked it.
GUIDE_GAP = float(os.environ.get("GUIDE_GAP", "0.4")) * MM
STOP = (ROD_X - (ROD_T / 2 + STOP_R + GUIDE_GAP), PIN[1] + 30.0 * MM)  # behind the rod, holds its feed side
# WHY a back pin beside the bending pin (2026-11-09, sim trace): with only the stop pin 40 mm back, a third of each
# bend went into turning the rod's feed side the other way about the stop pin; between the bending pin and this pin
# the rod sits in a channel, as on a real bending table.
BACK = (ROD_X - (ROD_T / 2 + STOP_R + GUIDE_GAP), PIN[1] + 12.0 * MM)
# WHY a front pin (2026-11-09, sim trace): for the third bend only about 37 mm of rod is left past the pin, and with
# nothing on its far side the swing of the formed U turned the whole stirrup 20 degrees about the bending pin. The front
# and stop pins sit 24 and 30 mm back, inside the finished hoop: its first side closes about 41 mm back, clear of
# both. (A guide pin 90 mm back was removed: the left fingertips met it on the second push.)
FRONT = (ROD_X + (ROD_T / 2 + STOP_R + GUIDE_GAP), PIN[1] + 24.0 * MM)
# WHY low guide strips along the feed side (2026-11-09, sim trace): beyond the last pin the rod's tail ran free for
# 90 mm; pushed from its end, it turned a few degrees in the pins' clearance, the end slid 8 mm sideways off the
# fingertip, and the passing finger bent the tail. Two strips 0.8 mm high either side of the rod keep it straight; the
# fingertip pushes the rod's end above them. They start 60 mm back so the last side, which closes at about 41 mm,
# never meets them.
STRIP_H, STRIP_W, STRIP_Y = 0.8 * MM, 3.0 * MM, (60.0 * MM, 170.0 * MM)
STRIP_X = tuple(ROD_X + k * (ROD_T / 2 + GUIDE_GAP + STRIP_W / 2) for k in (-1, 1))
PLATE_X = (-0.03, 0.13)  # the plate's extent, relative to the pin
PLATE_Y = (-0.10, 0.19)


def _rel(p):
    return (p[0] - PIN[0], p[1] - PIN[1])


# WHY friction 0.02 on the table and the rod (2026-11-09, sim traces): at 0.1 to 0.4, PhysX's contact friction on the
# rod's 52 short links resisted its sliding over the plate far more than its 14 g weight would; the bend then turned the
# rod's feed side, or kinked it beside the bend, instead of swinging the free leg. At 0.02 the bend is clean and the
# rod does not slip along. A real table would be greased; it is listed in the fidelity notes.
TABLE_MU = float(os.environ.get("PLATE_MU", "0.02"))
BENDER = body("bender", pos=(*PIN, TOP), friction=TABLE_MU, shapes=[
    box((sum(PLATE_X) / 2, sum(PLATE_Y) / 2, PLATE_H / 2), (PLATE_X[1] - PLATE_X[0], PLATE_Y[1] - PLATE_Y[0], PLATE_H),
        STEEL),
    *prism12((0, 0, PLATE_H + PIN_H / 2), PIN_R, PIN_H, DARK),
    *prism12((*_rel(STOP), PLATE_H + PIN_H / 2), STOP_R, PIN_H, DARK),
    *prism12((*_rel(BACK), PLATE_H + PIN_H / 2), STOP_R, PIN_H, DARK),
    *prism12((*_rel(FRONT), PLATE_H + PIN_H / 2), STOP_R, PIN_H, DARK),
    *(box((x - PIN[0], sum(STRIP_Y) / 2, PLATE_H + STRIP_H / 2), (STRIP_W, STRIP_Y[1] - STRIP_Y[0], STRIP_H), DARK)
      for x in STRIP_X),
])

# -- the bending lever (hinged on the bending pin) -----------------------------------------------------------------
ROLLER_R = 6.0 * MM
# WHY the roller 25 mm past the pin (2026-11-09, sim trace): 7 mm past it, the rod had to be squeezed with about 50 N
# between roller and pin to bend, and the squeeze kinked it the other way just past the bend.
# WHY a 2.5 mm clearance (2026-11-09, sim trace): at 0.3 mm, any small lag in the rod's wrap let the roller's arc cut
# into it, and the rod was wedged between roller and pin with forces far beyond the arm's; it bent the wrong way
# beyond the roller, and the solver later failed. A real bender's roller also stands off the bar.
ROLLER_GAP = float(os.environ.get("ROLLER_GAP", "2.5")) * MM
# WHY the roller reaches 0.2 mm from the plate (2026-11-09, sim trace): its foot 0.6 mm above the rod's underside let it
# ride onto the rod's top edge and pinch it down on the plate; the friction dragged the free end and bent it back.
ROLLER_LOW = float(os.environ.get("ROLLER_LOW", "0.2")) * MM
ROLLER = (ROD_X - (ROD_T / 2 + ROLLER_R + ROLLER_GAP), PIN[1] - float(os.environ.get("ROLLER_OFF", "25")) * MM)
ROLL_ANG = math.atan2(*reversed(_rel(ROLLER)))  # the roller's direction from the pivot; the lever runs this way
ARM_Z = PLATE_H + PIN_H + 6.0 * MM
LEVER_LEN = 150.0 * MM
HANDLE_R, HANDLE_H = 10.0 * MM, 45.0 * MM


def _lever_shapes() -> list:
    c, s = math.cos(ROLL_ANG), math.sin(ROLL_ANG)
    hx, hy = LEVER_LEN * c, LEVER_LEN * s
    rx, ry = _rel(ROLLER)
    return [
        box((hx / 2, hy / 2, ARM_Z), (LEVER_LEN + 20.0 * MM, 16.0 * MM, 6.0 * MM), RED, yaw=ROLL_ANG),
        *prism12((rx, ry, (PLATE_H + ROLLER_LOW + ARM_Z) / 2), ROLLER_R, ARM_Z - PLATE_H - ROLLER_LOW, STEEL),
        *prism12((hx, hy, ARM_Z + 3.0 * MM + HANDLE_H / 2), HANDLE_R, HANDLE_H, DARK),
    ]


# WHY low friction on the lever (2026-11-09, sim trace): its roller is a fixed pin here, not a roller that turns; at
# friction 0.6 it dragged along the rod and kinked the leg 13 degrees the other way just past the bend.
# WHY the roller is seen only (2026-11-09, sim traces): pushing a 3 mm rod of 5 mm links round the pin by contact, the
# solver kinked it the wrong way beside the roller in every variant tried (roller position, clearance, height, link
# length, link inertia). The bend is formed by rodkit.Bender instead: the rod wraps the pin as the lever turns.
LEVER = body("lever", pos=(*PIN, TOP), shapes=_lever_shapes(), mass=0.20, visual_only=tuple(range(1, 7)),
             friction=float(os.environ.get("LEVER_MU", "0.05")),
             contact_offset=1.0 * MM, solver_iterations=32,
             com=(LEVER_LEN / 2 * math.cos(ROLL_ANG), LEVER_LEN / 2 * math.sin(ROLL_ANG), ARM_Z))
LEVER_HINGE = rodkit.add_hinge(Hinge("bender", "lever", pivot=(0.0, 0.0, ARM_Z), axis=(0.0, 0.0, 1.0), start=0.0,
                                     lower=-10.0, upper=200.0))

# -- the rod (the free end first; frame: origin at the free end, +y along the rod) --------------------------------
# The stirrup: 80 x 50 mm inside its corners. Each chalk mark is where the rod must sit at the pin (its tangent point)
# for that bend: the rod's free end starts FIRST beyond the pin, and is fed on by FEEDS between bends. Measured in the
# sim (2026-11-09, scripted demo): with the free end 88 mm past the pin and feeds of 50 and 79 mm, the sides come out
# 50 and 80 mm inside and the ends meet, the free end butting against the last side; the last 38 mm of the 255 mm bar
# (51 links of 5 mm) is the fourth side.
INSIDE = (80.0 * MM, 50.0 * MM)
FIRST = float(os.environ.get("ROD_FIRST", "88")) * MM
FEEDS = tuple(float(v) * MM for v in os.environ.get("ROD_FEEDS", "50,79").split(","))
# WHY 5 mm links (2026-11-09, sim trace): a 90-degree bend round the 8 mm pin takes about 15 mm of rod; with 10 mm
# links it could only form as one or two sharp joints, which kinked the leg beside it.
LINK = float(os.environ.get("ROD_LINK", "5")) * MM
ROD_LEN = float(os.environ.get("ROD_LEN", "255")) * MM  # the last side ends just short of the first
ROD_LINKS = int(round(ROD_LEN / LINK))
MARKS = (FIRST, FIRST + FEEDS[0], FIRST + FEEDS[0] + FEEDS[1])
ROD = rod("rod", pos=(ROD_X, PIN[1] - FIRST, TOP + PLATE_H + ROD_T / 2 + 0.05 * MM), links=ROD_LINKS, link_len=LINK,
          mass=float(os.environ.get("ROD_MASS", str(0.0555 * ROD_LEN))),
          link_inertia=float(os.environ["ROD_I"]) if os.environ.get("ROD_I") else None, stiffness=0.795 / LINK,
          friction=float(os.environ.get("ROD_MU", str(TABLE_MU))), yield_moment=float(os.environ.get("ROD_MY", "0.8")),
          marks=MARKS)  # 3 mm steel: E I = 0.795 N m2

OBJECTS = [BENDER, LEVER, ROD]
rodkit.register(*OBJECTS)
WRAP_R = PIN_R + ROD_T / 2 + GAP  # the rod's centreline round the pin
SPRINGBACK = 3.0  # deg
BENDER_MODEL = rodkit.add_bender(rodkit.Bender("rod", "lever", pin=PIN, wrap_r=WRAP_R, moment=0.8,
                                               springback=SPRINGBACK))

# -- scoring -------------------------------------------------------------------------------------------------------
# Measured as on a real table: lay the bar on the drawing; read each corner with a protractor, the sides with a ruler
# to the corners' outer lines (here: the fitted sides' centrelines meet at each corner, less the bar's thickness), and
# check the two ends meet at the fourth corner.
SCORE_OVER = "end"
RUN_LIMIT_STEPS = 1600
CORNER_MIN = 20.0  # deg: a bend this sharp or more counts as a corner
KINK = 3.0  # deg: joints bent this much or more belong to a bend
ANGLE_TOL = 5.0  # deg
SIDE_TOL = 5.0 * MM
CLOSE_TOL = 5.0 * MM  # each end within this of the other end's side (the hoop closes)
STRAIGHT_TOL = 3.0 * MM  # no point of a side further than this off its line
FLAT_TOL = 5.0 * MM  # the whole bar within this height band (flat on the plate)
WRONG_WAY = 30.0  # deg: bends this sharp made both ways scrap the bar (latched)
CLEAR = 10.0 * MM


def _line(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A least-squares line through 2-D points: (a point on it, unit direction)."""
    c = pts.mean(axis=0)
    u = np.linalg.svd(pts - c)[2][0]
    return c, u


def _meet(l1, l2) -> np.ndarray:
    (p, u), (q, v) = l1, l2
    a = np.array([u, -v]).T
    if abs(np.linalg.det(a)) < 1e-9:
        return (p + q) / 2
    t = np.linalg.solve(a, q - p)
    return p + t[0] * u


def _off(pt, line) -> float:
    c, u = line
    d = pt - c
    return float(abs(d[0] * u[1] - d[1] * u[0]))


def measure(points: np.ndarray, angles: np.ndarray) -> dict:
    """The bar's shape from its centreline ``points`` (links + 1, 3) and joint ``angles`` (rad), free end first.

    ``bends``: each bent stretch's total angle (deg; negative is the right way), in order from the free end;
    ``corners``: those of CORNER_MIN or more. With exactly three corners, also ``sides`` (the inside lengths between
    corners 1-2 and 2-3, m), ``gaps`` (the free end's distance from the last side's line and the feed end's from the
    first side's, m), ``straight`` (the worst point's distance off its side's line, m). ``flat``: the height band
    (m)."""
    a = np.degrees(np.asarray(angles, float))
    xy = np.asarray(points, float)[:, :2]
    runs, cur = [], []
    for j, v in enumerate(a):
        if abs(v) >= KINK:  # a run of bent joints, one straight joint apart at most, all bent the same way
            if cur and (j - cur[-1] > 2 or np.sign(v) != np.sign(a[cur[-1]])):
                runs.append(cur)
                cur = []
            cur.append(j)
    if cur:
        runs.append(cur)
    # a bend's angle includes the joints either side of its run (a bend that starts part way along a link leaves a
    # few degrees on its first or last joint)
    bends = [(r[0], r[-1], float(a[max(0, r[0] - 1):r[-1] + 2].sum())) for r in runs]
    corners = [b for b in bends if abs(b[2]) >= CORNER_MIN]
    out = {"bends": [b[2] for b in bends], "corners": [c[2] for c in corners],
           "flat": float(np.ptp(np.asarray(points, float)[:, 2]))}
    if len(corners) != 3:
        return out
    # joint j sits at points[j + 1]; each side runs between its corners' outer joints
    cut = [0] + [x for c in corners for x in (c[0] + 1, c[1] + 1)] + [len(xy) - 1]
    sides = [xy[cut[2 * k]:cut[2 * k + 1] + 1] for k in range(4)]
    if min(len(sd) for sd in sides) < 2:
        return out
    lines = [_line(sd) for sd in sides]
    # each side's direction from the free end on; the corners as a protractor reads them, between the sides
    lines = [(c, u if np.dot(sd[-1] - sd[0], u) >= 0 else -u) for (c, u), sd in zip(lines, sides)]
    out["corners"] = [math.degrees(math.atan2(lines[k][1][0] * lines[k + 1][1][1] - lines[k][1][1] * lines[k + 1][1][0],
                                              float(np.dot(lines[k][1], lines[k + 1][1])))) for k in range(3)]
    apex = [_meet(lines[k], lines[k + 1]) for k in range(3)]
    out["sides"] = [float(np.linalg.norm(apex[1] - apex[0])) - ROD_T, float(np.linalg.norm(apex[2] - apex[1])) - ROD_T]
    out["gaps"] = [_off(xy[0], lines[3]), _off(xy[-1], lines[0])]
    out["straight"] = max(max(_off(p, ln) for p in sd) for sd, ln in zip(sides, lines))
    return out


def _good(c: float) -> bool:
    """A right-angle corner either way (bent all the other way, the stirrup is the same part turned over)."""
    return abs(abs(c) - 90.0) <= ANGLE_TOL


_STATE: dict = {}


def _reset_state() -> None:
    _STATE.update(violation=None)


_reset_state()


def fingers_to_rod(points: np.ndarray) -> float:
    """Nearest distance (m) from either gripper's hulls to the bar's surface. inf offline."""
    pts = rodkit.finger_points()
    if not len(pts):
        return math.inf
    d = np.linalg.norm(pts[:, None, :] - np.asarray(points, float)[None, :, :], axis=2)
    return float(d.min()) - ROD_T / 2


def _update(objects) -> tuple[float, bool]:
    r = objects["rod"]
    m = measure(r.points, r.angles)
    both_ways = max(m["bends"], default=0.0) >= WRONG_WAY and min(m["bends"], default=0.0) <= -WRONG_WAY
    if _STATE["violation"] is None and both_ways:
        _STATE["violation"] = f"bent both ways ({min(m['bends']):.0f} and {max(m['bends']):.0f} deg)"
    if _STATE["violation"]:
        return 0.0, False
    good = sum(_good(c) for c in m["corners"])
    if len(m["corners"]) == 3 and good == 3:
        full = ("sides" in m
                and all(abs(a - b) <= SIDE_TOL for a, b in zip(sorted(m["sides"]), sorted(INSIDE)))
                and max(m["gaps"]) <= CLOSE_TOL and m["straight"] <= STRAIGHT_TOL and m["flat"] <= FLAT_TOL)
        if full and fingers_to_rod(r.points) >= CLEAR:
            return 1.0, bool(r.resting())
        return 0.5, False
    return (0.25 if good >= 1 else 0.0), False


def success(objects) -> bool:
    """Three right-angle bends the same way, the sides 50 and 80 mm inside, the hoop closed, straight and flat,
    released and still."""
    return _update(objects)[1]


def score(objects) -> float:
    """1.0 / 0.5 / 0.25 / 0 as in the README; bends made both ways latch 0."""
    return _update(objects)[0]


# -- scripted demo -------------------------------------------------------------------------------------------------
# Lever degrees for each bend: 90 plus the springback and the arm's steady lag (sim traces, 2026-11-09: at 104 each,
# the bends came out 92.6, 91.1 and 93.5 degrees).
SWINGS = tuple(float(v) for v in os.environ.get("ROD_SWINGS", "101.5,103,100.5").split(","))
FEED_PAD = float(os.environ.get("ROD_FEED_PAD", "0")) * MM  # extra push, for the arm's lag


def handle_at(deg: float, up: float = 0.0) -> tuple:
    a = ROLL_ANG + math.radians(deg)
    return (PIN[0] + LEVER_LEN * math.cos(a), PIN[1] + LEVER_LEN * math.sin(a), TOP + ARM_Z + 3.0 * MM + 30.0 * MM + up)


GRIP_YAW = ROLL_ANG + math.pi / 2  # fingers close across the lever's length


def _swing(deg_to: float, deg_from: float, n: int, dur: float, label: str) -> list:
    return [Waypoint(dur / n, handle_at(deg_from + (deg_to - deg_from) * i / n), FINGER_CLOSED, label,
                     GRIP_YAW + math.radians(deg_from + (deg_to - deg_from) * i / n)) for i in range(1, n + 1)]


def bend(swing: float) -> list:
    """Swing the lever ``swing`` degrees, hold it there, swing it back.

    WHY the hold (2026-11-09, sim traces): forming, the lever resists with the bar's 0.8 N m, and the soft arm lags
    behind its target. Turned back at once, the bend depended on how far the arm had caught up (2 degrees more swing
    gave 5 degrees more bend); held for a second, the lever settles at a steady lag."""
    out = _swing(swing, 0.0, 8, 5.0, "bend")
    return out + [_wait(out, 1.0, "hold the bend")] + _swing(0.0, swing, 4, 2.4, "lever back")


TIP_Z = TOP + PLATE_H + STRIP_H + 0.6 * MM  # the left fingertips' lowest point pushing the rod's end: above the strips


def push(end_y: float, dist: float) -> list:
    """The left arm's closed fingertips come down behind the rod's end (at world y ``end_y``) and push it ``dist``
    along -y, then lift away. Fingers closing along y: their outer face (15 mm wide) meets the rod's end."""
    tcp_z = TIP_Z + PAD_TIP_PAST_TCP
    off = 4.0 * MM + 2.0 * MM  # the closed fingertips' half thickness plus a clearance
    y0, y1 = end_y + off + 8.0 * MM, end_y - dist - FEED_PAD + off
    return [
        Waypoint(1.5, (ROD_X, y0, tcp_z + 0.05), FINGER_CLOSED, "over the rod's end", 0.0),
        Waypoint(1.0, (ROD_X, y0, tcp_z), FINGER_CLOSED, "down behind it", 0.0),
        Waypoint(max(1.0, dist / 0.03), (ROD_X, y1, tcp_z), FINGER_CLOSED, "push the rod along", 0.0),
        Waypoint(0.6, (ROD_X, y1 + 3.0 * MM, tcp_z), FINGER_CLOSED, "ease off", 0.0),
        Waypoint(1.0, (ROD_X, y1 + 3.0 * MM, tcp_z + 0.06), FINGER_CLOSED, "up", 0.0),
    ]


def _t(w) -> float:
    return sum(p.duration for p in w)


def _wait(w: list, seconds: float, label: str = "wait") -> Waypoint:
    last = w[-1]
    return Waypoint(seconds, last.pos, last.finger, label, last.yaw)


def no_feed_attempt() -> dict:
    """A shortcut check (``STIRRUP_DEMO=nofeed``): the right arm swings the lever three times and the bar is never fed
    on, so every bend lands on the first mark."""
    right = [Waypoint(3.0, handle_at(0.0, 0.05), FINGER_OPEN, "over the handle", GRIP_YAW),
             Waypoint(0.8, handle_at(0.0, 0.05), FINGER_OPEN, "line up", GRIP_YAW),
             Waypoint(1.5, handle_at(0.0), FINGER_OPEN, "down to it", GRIP_YAW),
             Waypoint(1.8, handle_at(0.0), FINGER_CLOSED, "grip the handle", GRIP_YAW)]
    for swing in SWINGS:
        right += bend(swing)
    right += [Waypoint(1.8, handle_at(0.0), FINGER_OPEN, "let go", GRIP_YAW),
              Waypoint(1.0, handle_at(0.0, 0.06), FINGER_OPEN, "up", GRIP_YAW)]
    return {"right": right}


def demo():
    if os.environ.get("STIRRUP_DEMO") == "nofeed":
        return no_feed_attempt()
    right = [Waypoint(3.0, handle_at(0.0, 0.05), FINGER_OPEN, "over the handle", GRIP_YAW),
             Waypoint(0.8, handle_at(0.0, 0.05), FINGER_OPEN, "line up", GRIP_YAW),
             Waypoint(1.5, handle_at(0.0), FINGER_OPEN, "down to it", GRIP_YAW),
             Waypoint(1.8, handle_at(0.0), FINGER_CLOSED, "grip the handle", GRIP_YAW)]
    left_home = (ROD_X - 0.02, PIN[1] + 0.22, TOP + 0.15)
    left = [Waypoint(3.0, left_home, FINGER_CLOSED, "ready", 0.0)]
    end_y = PIN[1] - FIRST + ROD.length  # the rod's feed end (world y)
    right += bend(SWINGS[0])
    for k, dist in enumerate(FEEDS):
        left.append(_wait(left, _t(right) - _t(left), "wait for the bend"))
        p = push(end_y, dist)
        left += p
        end_y -= dist + FEED_PAD
        left.append(Waypoint(1.2, left_home, FINGER_CLOSED, "clear", 0.0))
        right.append(_wait(right, _t(left) - _t(right), "wait for the feed"))
        right += bend(SWINGS[k + 1])
    right += [Waypoint(1.8, handle_at(0.0), FINGER_OPEN, "let go", GRIP_YAW),
              Waypoint(1.0, handle_at(0.0, 0.06), FINGER_OPEN, "up", GRIP_YAW)]
    left.append(_wait(left, _t(right) - _t(left), "done"))
    return {"left": left, "right": right}
