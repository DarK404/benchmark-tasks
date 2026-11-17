"""On the station table: an exterior door's head jamb, lying on its back and taped down, its stop's kerf facing the
robot. A new kerf-in weatherstrip, cut to the jamb's length, lies loose on the rabbet floor in front of the kerf,
not pressed in, slid 20 mm too far to the right (its right-hand end overhangs the jamb's end).

The robot must fit the weatherstrip into the kerf: both ends flush with the jamb's ends and the fin pressed home all
along. The strip is limp and 81 cm long: it is pressed in a little at a time, working away from one end, and its
ends must be lined up first (once a fin is home, it stays where it is).

Run it:  inspect-robots run --embodiment yam_arena -E task=weatherstrip.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "Fit the weatherstrip into the head jamb's kerf." ...
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import weatherkit  # noqa: E402
from weatherkit import Bend, Drag, Kerf, body, box  # noqa: E402
from yam_arena.analysis import box_signed_distance  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

weatherkit.install()
SCENE = "station"
TOP = TABLE_TOP[SCENE]
MM = 0.001
PRIMER = (0.88, 0.88, 0.85)  # factory-primed poplar
BLACK = (0.07, 0.07, 0.08)  # Pemko finish BL

# -- sizes (env frame: x forward, y left, z up from the table top) ---------------------------------------------------
# the jamb: Pease Doors exterior head jamb, SKU TJH324, 32 in (for a 31-3/4 in slab), 4-9/16 in deep, primed poplar;
# 1-1/4 in thick with the stop, 3/4 in without; the stop 1/2 in high; a 1/8 x 3/8 in kerf in the stop's face
JL = 813.0 * MM
JY0, JY1 = -JL / 2, JL / 2
JX0, JX1 = 0.242, 0.358  # its front (robot side) and back edges: 116 mm (4-9/16 in)
JT = 19.05 * MM  # the rabbet floor's height (3/4 in)
STOP_X = 0.290  # the stop's face; the rabbet floor in front of it is 48 mm wide (a 1-3/4 in door and its play)
STOP_H = 12.7 * MM
KERF_W, KERF_D = 3.2 * MM, 9.5 * MM
# the strip: Pemko Q102BL foam kerf-in weatherstrip, 5/8 in reach (the bulb), 1 in total with the fin
BULB_X, BULB_Z = 15.9 * MM, 12.7 * MM
FIN_X, FIN_Z = 9.5 * MM, 1.5 * MM
KERF_Z = JT + BULB_Z / 2  # the kerf's centre: at the fin's height when the strip lies on the rabbet floor
N_LINKS = 34
LINK = JL / N_LINKS  # 23.9 mm links that bend (weatherkit.Bend)
SEAT = 8.0 * MM  # the fin at least this far in: home (the bulb's base within 1.5 mm of the stop's face)
FLUSH = 2.0 * MM  # each end within this of the jamb's end
LOOSE = 6.0 * MM  # laid in the rabbet, not pressed: its fin's tip this far short of the stop's face
STRIP_X = STOP_X - LOOSE - FIN_X - BULB_X / 2
SHIFT0 = -20.0 * MM  # and slid this far along: its right-hand end overhangs the jamb's end by 20 mm
STRIP_MASS = 0.025  # kg, about 31 g/m (our estimate)


def _jamb_shapes() -> list:
    zl, zh = KERF_Z - KERF_W / 2, KERF_Z + KERF_W / 2  # the kerf's lower and upper walls
    sx = (STOP_X + JX1) / 2
    return [
        box(((JX0 + JX1) / 2, 0.0, JT / 2), (JX1 - JX0, JL, JT), PRIMER),  # the board
        box((sx, 0.0, (JT + zl) / 2), (JX1 - STOP_X, JL, zl - JT), PRIMER),  # the stop, below the kerf
        box((sx, 0.0, (zh + JT + STOP_H) / 2), (JX1 - STOP_X, JL, JT + STOP_H - zh), PRIMER),  # above it
        box(((STOP_X + KERF_D + JX1) / 2, 0.0, KERF_Z), (JX1 - STOP_X - KERF_D, JL, KERF_W), PRIMER),  # behind it
    ]


JAMB = body("head_jamb", pos=(0.0, 0.0, TOP), shapes=_jamb_shapes(), friction=0.5, contact_offset=0.0003, mass=2.0)
weatherkit.kinematic(JAMB.name)  # (a rigid body nothing moves, so the strip's friction joints can hang on it)

# the strip lying loose on the rabbet floor along y, its fin towards the kerf (+x); link frames: x along the strip (+y), the fin
# along -y (world +x: the links are turned 90 degrees); link 0 at the right-hand end (-y)
FIN_TIP = (0.0, -(BULB_X / 2 + FIN_X), 0.0)
STRIP = []
for i in range(N_LINKS):
    shapes = [box((0.0, 0.0, 0.0), (LINK - 0.5 * MM, BULB_X, BULB_Z), BLACK),
              box((0.0, -(BULB_X + FIN_X) / 2, 0.0), (LINK - 0.5 * MM, FIN_X, FIN_Z), BLACK)]
    STRIP.append(body(f"strip{i:02d}", pos=(STRIP_X, JY0 + SHIFT0 + (i + 0.5) * LINK, TOP + JT + BULB_Z / 2 + 0.3 * MM),
                      yaw=math.pi / 2, mass=STRIP_MASS / N_LINKS, friction=0.5, shapes=shapes, contact_offset=0.0003,
                      inertia=(4e-8, 1.5e-7, 1.5e-7), solver_iterations=32))
OBJECTS = [JAMB, *STRIP]
weatherkit.register(*OBJECTS)
for _s in STRIP:
    weatherkit.damped(_s.name, 0.5, 2.0)
for i in range(1, N_LINKS):
    weatherkit.add_bend(Bend(STRIP[i - 1].name, STRIP[i].name, anchor=(LINK / 2, 0.0, 0.0), stiffness=0.01,
                             damping=2e-5))
HOME_X = STOP_X - BULB_X / 2  # a link home: its bulb against the stop's face
# the fin's barbs in the kerf: each 23.9 mm of strip resists being pushed in (and drawn out) by up to INSERT_F
# newtons, a Coulomb friction across the stop's face (our estimate of a firm thumb press: about 170 N per metre)
INSERT_F = 4.0
for _s in STRIP:
    weatherkit.add_drag(Drag(JAMB.name, _s.name, axis="X", force=INSERT_F))
MODEL = weatherkit.add_model(Kerf([s.name for s in STRIP], FIN_TIP, STOP_X, TOP + KERF_Z, JY0 - FLUSH, JY1 + FLUSH,
                                  SEAT, home=(HOME_X, TOP + KERF_Z, weatherkit.yaw_quat(math.pi / 2)), hold=False))

# -- scoring -------------------------------------------------------------------------------------------------------
# 1.0   every link's fin home in the kerf; both ends flush with the jamb's ends (within 2 mm); no gripper within
#       1 cm of the strip; the strip still, for 1 s
# 0.5   at least 90% of the links home, but not all, or an end not flush (or not let go, or not still)
# 0.25  at least 25% of the links home
# 0     anything else
SCORE_OVER = "end"
RUN_LIMIT_STEPS = 2400
CLEAR_DIST = 10.0 * MM
STILL_STEPS = 20
_STATE: dict = {}


@weatherkit.on_reset
def _reset_state() -> None:
    _STATE.update(still=0)


_reset_state()


def _local(objects, link, point) -> np.ndarray:
    return objects[link.name].to_world(point) - np.array([0.0, 0.0, TOP])


def fin_depth(objects, link) -> float:
    """How far the link's fin is into the kerf (m; negative: short of the stop's face), 0 if not in line with it."""
    tip = _local(objects, link, FIN_TIP)
    if abs(tip[2] - KERF_Z) > KERF_W / 2 or not JY0 - FLUSH <= tip[1] <= JY1 + FLUSH:
        return -1.0
    return float(tip[0] - STOP_X)


def seated(objects) -> list[bool]:
    return [fin_depth(objects, s) >= SEAT for s in STRIP]


def ends(objects) -> tuple[float, float]:
    """The strip's two cut ends' y (the right-hand end's, then the left-hand end's)."""
    return (float(_local(objects, STRIP[0], (-LINK / 2, 0.0, 0.0))[1]),
            float(_local(objects, STRIP[-1], (LINK / 2, 0.0, 0.0))[1]))


def flush(objects) -> bool:
    e0, e1 = ends(objects)
    return abs(e0 - JY0) <= FLUSH and abs(e1 - JY1) <= FLUSH


def fingers_clear(objects) -> bool:
    pts = weatherkit.finger_points()
    if not len(pts):
        return True
    size = np.array([LINK, BULB_X + FIN_X, BULB_Z])
    centre = np.array([0.0, -FIN_X / 2, 0.0])
    for s in STRIP:
        st = objects[s.name]
        if box_signed_distance(pts, st.to_world(centre), st.rotation(), size).min() < CLEAR_DIST:
            return False
    return True


def _update(objects) -> tuple[float, bool]:
    home = sum(seated(objects))
    if home == N_LINKS and flush(objects):
        if all(objects[s.name].resting() for s in STRIP) and fingers_clear(objects):
            return 1.0, True
        return 0.5, False
    if home >= 0.9 * N_LINKS:
        return 0.5, False
    if home >= 0.25 * N_LINKS:
        return 0.25, False
    return 0.0, False


def success(objects) -> bool:
    ok = _update(objects)[1]
    _STATE["still"] = _STATE["still"] + 1 if ok else 0
    return ok


def score(objects) -> float:
    tier = _update(objects)[0]
    if tier == 1.0 and _STATE["still"] < STILL_STEPS:
        return 0.5
    return tier


# -- scripted demo -------------------------------------------------------------------------------------------------
# Open loop, from the fixed layout only. Fingers close along x (yaw pi/2 for the right arm, -pi/2 for the left: the
# other way round the right arm's wrist runs out of reach near the middle); closed, the two pads are 8 mm thick along
# x and 16 mm wide along y, which fits the 16 mm of rabbet floor behind the loose strip's bulb.
TIP = 13.0 * MM  # the TCP is this far above the pad tips
YAW_R, YAW_L = math.pi / 2, -math.pi / 2
RHOME, LHOME = (0.16, -0.22, TOP + 0.15), (0.16, 0.22, TOP + 0.15)
SEATED_X = HOME_X
PUSH_IN = 28.0 * MM  # the press aims this far past home (the arm is soft: about 8 N)
PRESS_Z = 2.0 * MM  # pads' tips this far above the rabbet floor while pressing
DRAG_Z = BULB_Z - 6.0 * MM  # pads' tips this far above the floor when dragging the strip along: 6 mm into its top
SLIDE = -SHIFT0  # drag it this far towards the left-hand end
DRAG_LAG = 8.0 * MM  # aimed this much further (the soft arm trails the target)
SPLIT = N_LINKS // 2  # the right hand presses links 0..16, the left 17..33


def _link_y(i: int) -> float:
    return JY0 + (i + 0.5) * LINK


def _tcp_z(above_floor: float) -> float:
    return TOP + JT + above_floor + TIP


def slide(yaw: float) -> list:
    """Press down on the top of the strip's bulb near its left-hand end and drag it 20 mm towards that end: the
    right-hand end comes back flush with the jamb's end (dragged from the far end, the strip runs straight)."""
    y0 = JY1 + SHIFT0 - 0.05
    x = STRIP_X
    zup, zd = _tcp_z(BULB_Z + 15.0 * MM), _tcp_z(DRAG_Z)
    return [Waypoint(2.0, (x, y0, zup), FINGER_CLOSED, "over the strip's left end", yaw),
            Waypoint(0.8, (x, y0, zd), FINGER_CLOSED, "down on it", yaw),
            Waypoint(2.0, (x, y0 + SLIDE + DRAG_LAG, zd), FINGER_CLOSED, "drag it along", yaw),
            Waypoint(0.8, (x, y0 + SLIDE + DRAG_LAG, zd), FINGER_CLOSED, "hold", yaw),
            Waypoint(0.6, (x, y0 + SLIDE, zup), FINGER_CLOSED, "up off it", yaw)]


def press(i: int, yaw: float, first: bool = False) -> list:
    """Press link ``i``'s fin home: pads down on the rabbet floor behind its bulb, push towards the stop, back off."""
    y = _link_y(i)
    behind = STRIP_X - BULB_X / 2 - 4.0 * MM - 2.5 * MM  # the pads' centre, 2.5 mm behind the loose bulb
    zp, zup = _tcp_z(PRESS_Z), _tcp_z(BULB_Z + 4.0 * MM)
    home = SEATED_X - BULB_X / 2 - 4.0 * MM + PUSH_IN
    return [Waypoint(1.5 if first else 0.4, (behind, y, zup), FINGER_CLOSED, f"over link {i}", yaw),
            Waypoint(0.3, (behind, y, zp), FINGER_CLOSED, "down behind it", yaw),
            Waypoint(0.5, (home, y, zp), FINGER_CLOSED, "press it home", yaw),
            Waypoint(0.3, (behind, y, zp), FINGER_CLOSED, "back off", yaw),
            Waypoint(0.2, (behind, y, zup), FINGER_CLOSED, "up", yaw)]


def demo():
    """``WS_DEMO=noslide`` (a shortcut check): press it in where it lies, without lining its ends up first."""
    left = slide(YAW_L) + [Waypoint(1.5, LHOME, FINGER_CLOSED, "aside", 0.0)]
    if os.environ.get("WS_DEMO") == "noslide":
        left = [Waypoint(sum(w.duration for w in left), LHOME, FINGER_CLOSED, "wait", 0.0)]
    t_slid = sum(w.duration for w in left[:-1])
    right = [Waypoint(t_slid, RHOME, FINGER_CLOSED, "wait", 0.0)]
    for i in range(SPLIT):
        right += press(i, YAW_R, first=i == 0)
    right += [Waypoint(2.0, RHOME, FINGER_OPEN, "home", 0.0)]
    # the left hand presses on from the middle once the right hand has gone home
    t_right = sum(w.duration for w in right)
    left.append(Waypoint(t_right - sum(w.duration for w in left), LHOME, FINGER_CLOSED, "wait", 0.0))
    for i in range(SPLIT, N_LINKS):
        left += press(i, YAW_L, first=i == SPLIT)
    left += [Waypoint(2.0, LHOME, FINGER_OPEN, "home", 0.0)]
    tr, tl = sum(w.duration for w in right), sum(w.duration for w in left)
    right.append(Waypoint(max(0.0, tl - tr) + 1e-3, RHOME, FINGER_OPEN, "wait", 0.0))
    for plan in (left, right):
        plan.append(Waypoint(2.0, plan[-1].pos, plan[-1].finger, "done", plan[-1].yaw))
    return {"right": right, "left": left}
