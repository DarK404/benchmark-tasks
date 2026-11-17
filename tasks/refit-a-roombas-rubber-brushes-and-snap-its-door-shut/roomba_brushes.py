"""On the station table: an iRobot Roomba (e/i/j-series) lies upside down, its cleaning head's door hinged open and
its two rubber brushes out, lying on the table either side of it: one with a square peg on its drive end, the other
with a hex peg; both with a grey bearing cap on their other end.

The robot must refit both brushes and close the door, as iRobot says: each brush goes in the seat whose drive socket
matches its peg's shape, the peg end first, then the bearing end down into its cradle; then the door is closed over
them until its catches snap. A brush whose peg is not in its socket does not turn; a door left unlatched lets the
brushes fall out.

Run it:  inspect-robots run --embodiment yam_arena -E task=roomba_brushes.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "Refit both brushes in the robot vacuum's cleaning head and close its door." ...
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import brushkit  # noqa: E402
from brushkit import Hinge, Spring, body, box, tilted  # noqa: E402
from yam_arena.analysis import box_signed_distance  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

brushkit.install()
SCENE = "station"
TOP = TABLE_TOP[SCENE]
MM = 0.001
CHARCOAL = (0.16, 0.16, 0.17)  # the robot's body
DARKGREY = (0.30, 0.30, 0.32)  # the cleaning head's frame
GREY = (0.62, 0.62, 0.64)  # bearing caps
GREEN = (0.30, 0.62, 0.30)  # the door's tab and the brushes' fins
RUBBER = (0.20, 0.22, 0.20)
YELLOW = (0.90, 0.75, 0.15)

# -- sizes (env frame: x forward, y left, z up; the table top at TOP) ------------------------------------------------
# The robot: iRobot Roomba i3, 340 mm across and 92 mm tall, upside down, its cleaning head across it (along y)
RX, RY = 0.27, 0.0
R_RAD, R_H = 170.0 * MM, 92.0 * MM
# the cleaning head's pocket, in the robot's frame (origin on the table under its middle)
PK_X0, PK_X1 = -42.0 * MM, 90.0 * MM  # front (towards the robot) and rear edges of the opening
PK_Y0, PK_Y1 = -125.0 * MM, 127.0 * MM
AXIS_Z = 75.0 * MM  # a seated brush's axis above the table
SEAT_X = {"front": -12.0 * MM, "rear": 34.0 * MM}  # the two seats' axes (x in the robot's frame)
FLOOR_Z = AXIS_Z - 17.0 * MM

# The brushes: iRobot dual multi-surface rubber brushes, about 30 mm across over their fins and 215 mm long, about 60
# g each (our estimates). In a brush's frame: origin on its axis at its middle, +y towards its drive (peg) end.
BR_R, BR_L = 15.0 * MM, 215.0 * MM
BR_MASS = 0.06
PEG_L = 12.0 * MM
SQUARE = 10.0 * MM  # the square peg's side
HEX_AF = 11.0 * MM  # the hex peg's width across flats
CAP_R, CAP_L = 6.0 * MM, 12.0 * MM  # the bearing cap
# each socket's clearance round its own peg, a side; neither peg fits the other's socket at any turn: the hex peg
# needs a square hole at least 12.3 mm across (the square socket is 11.6), and the largest square that passes the hex
# socket (13.4 across flats) is 9.8 mm (the square peg is 10)
HOLE_CLEAR = {"front": 0.8 * MM, "rear": 1.2 * MM}
PLATE_Y0, PLATE_Y1 = 110.0 * MM, 112.0 * MM  # each seat's drive socket plate (in the seat frame)
BACK_Y = 121.5 * MM  # behind it, the drive's end stop
FUNNEL, FUNNEL_L = 3.5 * MM, 4.0 * MM  # each socket's mouth: this much wider a side, tapering over this depth
CRADLE_Y0, CRADLE_Y1 = -117.0 * MM, -114.0 * MM  # the bearing cradle plate, its U notch open at the top

# The door: hinged along the pocket's rear edge, 136 mm front to back, closed flat over the pocket on the robot's
# bottom face; two sprung catches under its front edge snap under a lip on the frame's front edge. It rests open at
# 110 degrees (20 degrees past upright, on its hinge stop: clear of the hand at the rear seat).
DOOR_L, DOOR_W, DOOR_T = 136.0 * MM, 250.0 * MM, 3.0 * MM
HINGE_X = PK_X1 + 2.0 * MM
OPEN_DEG = 110.0
DOOR_MASS = 0.04
CATCH_Y = 80.0 * MM  # the two catches, either side of the middle
CATCH_T, CATCH_H = 2.5 * MM, 8.0 * MM
CATCH_MASS = 0.003
CATCH_K, CATCH_PRELOAD, CATCH_TRAVEL = 1000.0, 1.0, 3.0 * MM
LIP_X0, LIP_X1, LIP_Z0, LIP_Z1 = PK_X0, PK_X0 + 2.5 * MM, R_H - 7.0 * MM, R_H - 4.5 * MM


def _prism_y(yc, length, r, colour, x=0.0, z=0.0, n=12) -> list:
    """A regular prism about the y axis: ``n`` 12 (six boxes 30 degrees apart) or 6 (three boxes 60 degrees apart);
    ``r`` is its inradius (the half width across flats)."""
    k_n = n // 2
    side = 2 * r * math.tan(math.pi / n)
    return [box((x, yc, z), (2 * r, length, side), colour, tilt_y=k * 2 * math.pi / n) for k in range(k_n)]


def _ring_y(yc, length, r_in, r_out, colour, segments, x=0.0, z=0.0) -> list:
    """A plate about the y axis with a regular ``segments``-sided hole (inradius ``r_in``): 4 a square, 6 a hex."""
    r_mid, w = (r_in + r_out) / 2, r_out - r_in
    chord = 2 * r_out * math.tan(math.pi / segments) + 0.0003  # (exactly the outer polygon: nothing sticks out)
    out = []
    for k in range(segments):
        a = 2 * math.pi * k / segments
        out.append(box((x + r_mid * math.cos(a), yc, z - r_mid * math.sin(a)), (w, length, chord), colour, tilt_y=a))
    return out


def _seat_shapes(name: str) -> list:
    """One seat in the robot's frame: its drive socket plate (square or hex hole), the drive's end stop behind it and
    the bearing cradle (a plate with a U notch, open at the top) at the other end."""
    x, z = SEAT_X[name], AXIS_Z
    sides, r_in = (4, SQUARE / 2 + HOLE_CLEAR[name]) if name == "front" else (6, HEX_AF / 2 + HOLE_CLEAR[name])
    colour = YELLOW if name == "front" else GREEN
    out = _ring_y((PLATE_Y0 + PLATE_Y1) / 2, PLATE_Y1 - PLATE_Y0, r_in, 15.0 * MM, colour, sides, x=x, z=z)
    # the socket's mouth: four plates sloping in from FUNNEL wider to the hole, over FUNNEL_L in front of the plate
    t, run, dep = 0.8 * MM, FUNNEL, FUNNEL_L
    slope = math.hypot(run, dep)
    a = math.atan2(run, dep)
    mid_y = PLATE_Y0 - dep / 2
    # the throat: the hole's own edge on each side (the hex hole's flats face +/-x, its corners +/-z)
    rx_t = r_in
    rz_t = r_in if sides == 4 else r_in / math.cos(math.pi / 6)
    w = 2 * (max(rx_t, rz_t) + run) + 2 * MM
    nr, ny = dep / slope, run / slope  # the guiding face's normal: radial outward, and towards -y
    for sgn in (-1.0, 1.0):
        # each plate lies behind its guiding face (outwards and towards +y), keeping the face on the line
        cy = mid_y + ny * t / 2
        cx_r = rx_t + run / 2 + nr * t / 2
        cz_r = rz_t + run / 2 + nr * t / 2
        out.append(box((x + sgn * cx_r, cy, z), (t, slope, w), colour, yaw=sgn * a))  # the +/-x faces
        out.append(box((x, cy, z + sgn * cz_r), (w, slope, t), colour, tilt_x=-sgn * a))  # the +/-z faces
    out.append(box((x, (BACK_Y + PK_Y1) / 2, z), (24.0 * MM, PK_Y1 - BACK_Y, 24.0 * MM), DARKGREY))
    ny, nt = (CRADLE_Y0 + CRADLE_Y1) / 2, CRADLE_Y1 - CRADLE_Y0
    half = CAP_R + 0.4 * MM
    for s in (-1.0, 1.0):
        out.append(box((x + s * (half + 16.0 * MM) / 2, ny, (FLOOR_Z + z + 8 * MM) / 2),
                       (16.0 * MM - half, nt, z + 8 * MM - FLOOR_Z), GREY))
    out.append(box((x, ny, (FLOOR_Z + z - CAP_R) / 2), (2 * half, nt, z - CAP_R - FLOOR_Z), GREY))
    return out


def _robot_shapes() -> tuple[list, int]:
    """The robot upside down: a round body (drawn), the frame round the cleaning head's opening and its floor (which
    collide), the two seats and the catch lip. Returns (shapes, number of drawn-only shapes at the front)."""
    side = 2 * R_RAD * math.tan(math.pi / 12)
    drawn = [box((0.0, 0.0, R_H / 2), (2 * R_RAD, side, R_H), CHARCOAL, yaw=k * math.pi / 6) for k in range(6)]
    top0 = R_H - 34.0 * MM
    out = [box((0.0, 0.0, top0 / 2), (0.30, 0.30, top0), CHARCOAL)]  # the body under the pocket
    # the frame round the opening, from the pocket's floor up to the bottom face
    h, zc = R_H - FLOOR_Z, (R_H + FLOOR_Z) / 2
    out.append(box(((PK_X0 - 0.14) / 2, 0.0, zc), (0.14 + PK_X0, 0.30, h), CHARCOAL))
    out.append(box(((PK_X1 + 0.14) / 2, 0.0, zc), (0.14 - PK_X1, 0.30, h), CHARCOAL))
    out.append(box(((PK_X0 + PK_X1) / 2, (PK_Y1 + 0.15) / 2, zc), (PK_X1 - PK_X0, 0.15 - PK_Y1, h), CHARCOAL))
    out.append(box(((PK_X0 + PK_X1) / 2, (PK_Y0 - 0.15) / 2, zc), (PK_X1 - PK_X0, 0.15 + PK_Y0, h), CHARCOAL))
    out += _seat_shapes("front") + _seat_shapes("rear")
    # the catch lip: a square-bottomed ledge under the front edge, its top sloping so a catch rides over it
    out.append(box(((LIP_X0 + LIP_X1) / 2, 0.0, (LIP_Z0 + LIP_Z1) / 2), (LIP_X1 - LIP_X0, 2 * (CATCH_Y + 15 * MM),
                                                                    LIP_Z1 - LIP_Z0), DARKGREY))
    t = 0.8 * MM
    rise = (R_H - 1.5 * MM) - LIP_Z1  # (it ends 1.5 mm under the frame's top, so a catch's nose meets its face)
    run = LIP_X1 - LIP_X0
    slope = math.hypot(run, rise)
    n = np.array([rise, run]) / slope  # (x, z) outward normal of the face rising from the lip's tip to the frame's edge
    c = np.array([(LIP_X0 + LIP_X1) / 2, (LIP_Z1 + R_H - 1.5 * MM) / 2]) - n * t / 2
    out.append(box((c[0], 0.0, c[1]), (t, 2 * (CATCH_Y + 15 * MM), slope), DARKGREY,
                   tilt_y=-math.atan2(run, rise)))
    return drawn + out, len(drawn)


_RS, _N_DRAWN = _robot_shapes()
ROBOT = body("roomba", pos=(RX, RY, TOP), shapes=_RS, mass=3.4, friction=0.5, contact_offset=0.0005,
             visual_only=tuple(range(_N_DRAWN)))
brushkit.kinematic(ROBOT.name)  # fixed (it lies on the table), and the door's hinge hangs on it


def _brush_shapes(peg: str) -> list:
    out = []
    out += _prism_y(0.0, BR_L, BR_R, RUBBER)
    # the two chevron fins (drawn)
    out.append(box((0.0, 0.0, BR_R + 0.4 * MM), (3.0 * MM, BR_L - 10 * MM, 0.8 * MM), GREEN))
    out.append(box((0.0, 0.0, -BR_R - 0.4 * MM), (3.0 * MM, BR_L - 10 * MM, 0.8 * MM), GREEN))
    yc = BR_L / 2 + PEG_L / 2
    if peg == "square":
        out.append(box((0.0, yc, 0.0), (SQUARE, PEG_L, SQUARE), YELLOW))
    else:
        out += _prism_y(yc, PEG_L, HEX_AF / 2, GREEN, n=6)
    out += _prism_y(-BR_L / 2 - CAP_L / 2, CAP_L, CAP_R, GREY)
    return out


# the brushes lie on the table either side of the robot, along y, drive ends towards +y, their pegs' flats square
BRUSH_START = {"square": np.array([0.24, -0.30, TOP + BR_R]), "hex": np.array([0.24, 0.30, TOP + BR_R])}
BRUSHES = {}
for _peg, _seat in (("square", "front"), ("hex", "rear")):
    _bs = _brush_shapes(_peg)
    BRUSHES[_peg] = body(f"brush_{_peg}", pos=tuple(BRUSH_START[_peg]), shapes=_bs, mass=BR_MASS, friction=0.8,
                         contact_offset=0.0005, solver_iterations=32, visual_only=(6, 7))
SEAT_OF = {"square": "front", "hex": "rear"}


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


# the door: its frame on the hinge axis, closed it lies along -x from there; spawned open
_HINGE_W = np.array([RX + HINGE_X, RY, TOP + R_H + DOOR_T / 2])
_R_OPEN = _ry(math.radians(OPEN_DEG))


def _q(r):
    return brushkit.matrix_to_quat_rel(r)


_door_shapes = [box((-DOOR_L / 2, 0.0, 0.0), (DOOR_L, DOOR_W, DOOR_T), DARKGREY),
                box((-DOOR_L + 6 * MM, 0.0, DOOR_T / 2 + 1.0 * MM), (10 * MM, 40 * MM, 2.0 * MM), GREEN)]
DOOR = body("brush_door", pos=tuple(_HINGE_W), shapes=_door_shapes, mass=DOOR_MASS, friction=0.5,
            contact_offset=0.0005, solver_iterations=32, quat=_q(_R_OPEN), com=(-DOOR_L / 2, 0.0, 0.0))


# each catch: an upright bar just behind the lip's tip, with a hooked nose at its foot reaching forward under the lip
CATCH_X0, CATCH_X1 = LIP_X1 + 0.3 * MM, LIP_X1 + 0.3 * MM + CATCH_T  # the bar (robot frame, closed)
NOSE_X0, NOSE_Z0, NOSE_Z1 = LIP_X0 + 1.0 * MM, LIP_Z0 - 3.0 * MM, LIP_Z0 - 0.5 * MM  # the nose, under the lip


def _catch(sign: float):
    """One of the door's two sprung catches (closed: its nose 2 mm under the lip's square underside)."""
    cx, cz = (CATCH_X0 + CATCH_X1) / 2, (NOSE_Z1 + R_H) / 2  # the catch's frame: the bar's middle
    local = np.array([cx - HINGE_X, sign * CATCH_Y, cz - (R_H + DOOR_T / 2)])
    pos = _HINGE_W + _R_OPEN @ local
    shapes = [box((0.0, 0.0, 0.0), (CATCH_T, 20.0 * MM, R_H - NOSE_Z1), GREEN),
              box(((NOSE_X0 + CATCH_X1) / 2 - cx, 0.0, (NOSE_Z0 + NOSE_Z1) / 2 - cz),
                  (CATCH_X1 - NOSE_X0, 20.0 * MM, NOSE_Z1 - NOSE_Z0), GREEN)]
    return body(f"catch_{'a' if sign > 0 else 'b'}", pos=tuple(pos), shapes=shapes, mass=CATCH_MASS,
                friction=0.3, contact_offset=0.0005, solver_iterations=32, quat=_q(_R_OPEN))


CATCH_A, CATCH_B = _catch(+1.0), _catch(-1.0)
OBJECTS = [ROBOT, BRUSHES["square"], BRUSHES["hex"], DOOR, CATCH_A, CATCH_B]
brushkit.register(*OBJECTS)
brushkit.add_hinge(Hinge(ROBOT.name, DOOR.name, anchor=(HINGE_X, 0.0, R_H + DOOR_T / 2), axis=(0.0, 1.0, 0.0),
                         damping=0.002, lower=math.radians(-OPEN_DEG), upper=0.0))
_pre = CATCH_PRELOAD / CATCH_K
for _c in (CATCH_A, CATCH_B):
    # sprung towards the front (the door's -x) against its stop; the lip pushes it back (+x) as the door shuts
    brushkit.add_spring(Spring(DOOR.name, _c.name, axis="X", stiffness=CATCH_K, damping=0.5, target=-_pre,
                               lower=0.0, upper=CATCH_TRAVEL))

# -- scoring -------------------------------------------------------------------------------------------------------
# 1.0   both brushes seated in the seats that match their pegs (peg through its drive socket, bearing cap in its
#       cradle: within 1.5 mm and 2 degrees of seated), the door shut and latched (within 2 degrees of closed); no
#       gripper within 1 cm of the brushes or the door; all still; for 1 s
# 0.5   both brushes seated, the door not latched (or a hand on it, or not yet still)
# 0.25  one brush seated
# 0     anything else
SCORE_OVER = "end"
RUN_LIMIT_STEPS = 1800
STILL_STEPS = 20
CLEAR_DIST = 10.0 * MM
SEAT_TOL, SEAT_DEG, DOOR_DEG = 1.5 * MM, 2.0, 2.0
_STATE: dict = {}


@brushkit.on_reset
def _reset_state() -> None:
    _STATE.update(still=0)


_reset_state()


def seat_origin(seat: str) -> np.ndarray:
    return np.array([RX + SEAT_X[seat], RY, TOP + AXIS_Z])


AXIAL_SHORT = 5.0 * MM  # the brush may sit this far short of the drive's end stop (its axial play) and still have
#                        its peg at least 2 mm through the socket plate


def seated(st, seat: str) -> bool:
    """Peg through its socket, bearing cap down in its cradle, square: within SEAT_TOL across the axis, between
    AXIAL_SHORT short of home and 2 mm past it along it, and within SEAT_DEG."""
    d = np.asarray(st.pos) - seat_origin(seat)
    axis = st.rotation() @ np.array([0.0, 1.0, 0.0])
    ang = math.degrees(math.acos(min(1.0, float(axis[1]))))
    return bool(abs(d[0]) <= SEAT_TOL and abs(d[2]) <= SEAT_TOL and -AXIAL_SHORT <= d[1] <= 2.0 * MM
                and ang <= SEAT_DEG)


def door_deg(st) -> float:
    """How far (degrees) the door stands open from shut."""
    free = st.rotation() @ np.array([-1.0, 0.0, 0.0])  # from the hinge to its front edge
    return math.degrees(math.atan2(float(free[2]), -float(free[0])))


CATCH_REST_X = (CATCH_X0 + CATCH_X1) / 2 - HINGE_X  # a catch's place along the door, at rest
LATCH_TOL = 0.6 * MM


def latched(objects) -> bool:
    """The door shut (within DOOR_DEG) and both catches sprung back to rest, their noses under the lip."""
    door = objects[DOOR.name]
    if door_deg(door) > DOOR_DEG:
        return False
    for c in (CATCH_A, CATCH_B):
        x = float((door.rotation().T @ (np.asarray(objects[c.name].pos) - np.asarray(door.pos)))[0])
        if abs(x - CATCH_REST_X) > LATCH_TOL:
            return False
    return True


def hands_clear(objects) -> bool:
    pts = brushkit.finger_points()
    if not len(pts):
        return True
    for peg, b in BRUSHES.items():
        st = objects[b.name]
        if box_signed_distance(pts, st.pos, st.rotation(), np.array([2 * BR_R, BR_L + 2 * PEG_L, 2 * BR_R])).min() \
                < CLEAR_DIST:
            return False
    st = objects[DOOR.name]
    centre = st.to_world((-DOOR_L / 2, 0.0, -CATCH_H / 2))
    return box_signed_distance(pts, centre, st.rotation(), np.array([DOOR_L, DOOR_W, DOOR_T + CATCH_H])).min() \
        >= CLEAR_DIST


def _update(objects) -> tuple[float, bool]:
    n = sum(seated(objects[b.name], SEAT_OF[p]) for p, b in BRUSHES.items())
    if n == 2:
        # the door hangs on its hinge, so its read-back velocity is not physical: its stillness is from its angle
        ang = door_deg(objects[DOOR.name])
        door_still = abs(ang - _STATE.get("door", ang)) < 0.2
        _STATE["door"] = ang
        still = all(objects[b.name].resting() for b in BRUSHES.values()) and door_still
        if latched(objects) and still and hands_clear(objects):
            return 1.0, True
        return 0.5, False
    return (0.25 if n == 1 else 0.0), False


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
# Open loop, from the fixed layout only. Each hand takes the brush on its side across its middle, tips it bearing end
# up, slides its peg through the drive socket of its seat (the right hand the square-peg brush to the front seat, then
# the left hand the hex-peg brush to the rear seat), lets go so it drops level, and presses its bearing end down into
# the cradle. Then the right hand pushes the door over its upright stop so it swings shut, and presses its front edge
# down until the catches snap.
TIP = 13.0 * MM
RHOME, LHOME = (0.14, -0.22, TOP + 0.17), (0.14, 0.22, TOP + 0.17)
GRIP_DZ = float(os.environ.get("RB_GRIPZ", "-4")) * MM  # pad tips this far above the brush's axis
OPEN_W = -(BR_R + 12.0 * MM)
HOLD = -(BR_R - 4.0 * MM)
TILT = math.radians(float(os.environ.get("RB_TILT", "10")))
SHORT = 22.0 * MM  # the peg lined up this far short of home, along the brush's axis (clear of the socket's mouth)
PUSH_PAST = 3.0 * MM  # ... and pushed in aiming this far past home (the soft arm)
CARRY_Z = TOP + 0.135
# each hand's steady error at the lined-up pose (measured in the sim), corrected: RB_CAL_RIGHT / RB_CAL_LEFT, mm
_CAL_MM = {"right": "0.4,0,0.9", "left": "0.5,-3.0,0.9"}
CAL = {arm: np.array([float(v) for v in os.environ.get(f"RB_CAL_{arm.upper()}", _CAL_MM[arm]).split(",")]) * MM
       for arm in ("right", "left")}
PRESS_Y = -90.0 * MM  # the bearing end's press point along the brush
PRESS_DEPTH = 40.0 * MM


def _grip(arm: str) -> float:
    """Straight down, fingers closing along x (the right hand turned +90 degrees, the left -90: each within its wrist's
    range)."""
    return tilted(math.pi / 2 if arm == "right" else -math.pi / 2, math.pi / 2)


def _grip_tipped(arm: str) -> float:
    """Its -y (bearing) end tipped up TILT: the same approach either way, the hand turned as in :func:`_grip`."""
    if arm == "right":
        return tilted(math.pi / 2, math.pi / 2 + TILT)
    return tilted(-math.pi / 2, math.pi / 2 - TILT)


def _brush_tipped(seat: str, out: float) -> tuple[np.ndarray, np.ndarray]:
    """The brush's (origin, rotation) tipped TILT bearing end up about its drive socket, ``out`` (m) from home along
    its own axis."""
    s = seat_origin(seat)
    c, sn = math.cos(-TILT), math.sin(-TILT)
    r = np.array([[1.0, 0.0, 0.0], [0.0, c, -sn], [0.0, sn, c]])  # -y end up
    piv = np.array([0.0, (PLATE_Y0 + PLATE_Y1) / 2, 0.0])
    return s + piv - r @ piv + r @ np.array([0.0, out, 0.0]), r


def _tcp(origin, r, arm) -> tuple[float, float, float]:
    p = np.asarray(origin) + r @ np.array([0.0, 0.0, GRIP_DZ + TIP]) + CAL[arm]
    return tuple(float(v) for v in p)


def fit_brush(peg: str, arm: str, home) -> list:
    """``RB_DEMO=swap`` (a shortcut check): each brush is offered to the other seat."""
    seat = SEAT_OF[peg]
    if os.environ.get("RB_DEMO") == "swap":
        seat = "rear" if seat == "front" else "front"
    b0 = BRUSH_START[peg]
    at = _tcp(b0, np.eye(3), arm)
    at0 = tuple(np.array(at) - CAL[arm])
    g, gt = _grip(arm), _grip_tipped(arm)
    o1, r1 = _brush_tipped(seat, -SHORT)
    o2, r2 = _brush_tipped(seat, PUSH_PAST)
    t1, t2 = _tcp(o1, r1, arm), _tcp(o2, r2, arm)
    return [Waypoint(2.0, (at0[0], at0[1], at0[2] + 0.05), OPEN_W, f"over the {peg} brush", g),
            Waypoint(1.2, at0, OPEN_W, "round its middle", g),
            Waypoint(1.0, at0, HOLD, "take it", g),
            Waypoint(1.2, (at0[0], at0[1], CARRY_Z), HOLD, "lift it", g),
            Waypoint(2.0, (t1[0], t1[1], CARRY_Z), HOLD, f"over the {seat} seat", gt),
            Waypoint(1.5, t1, HOLD, "peg lined up", gt),
            Waypoint(1.0, t1, HOLD, "settle", gt),
            Waypoint(1.5, t2, HOLD, "peg through the socket", gt),
            Waypoint(1.0, t2, OPEN_W, "let it drop", gt),
            Waypoint(1.0, (t2[0], t2[1], t2[2] + 0.03), OPEN_W, "up", gt),
            Waypoint(2.5, home, FINGER_OPEN, "home", 0.0)]


def press_bearings() -> list:
    """The right hand presses each brush's bearing end down into its cradle."""
    g = _grip("right")
    out = []
    for seat in ("front", "rear"):
        s = seat_origin(seat)
        p = (s[0], s[1] + PRESS_Y, s[2] + BR_R + TIP)
        out += [Waypoint(2.0, (p[0], p[1], p[2] + 0.03), FINGER_CLOSED, f"over the {seat} brush's bearing end", g),
                Waypoint(1.2, (p[0], p[1], p[2] - PRESS_DEPTH), FINGER_CLOSED, "press it into the cradle", g),
                Waypoint(1.0, (p[0], p[1], p[2] + 0.03), FINGER_CLOSED, "lift off", g)]
    return out


def close_door() -> list:
    # the door's face, 25 mm up from its hinge, as it rests open; pushed towards the front past upright
    p = _HINGE_W + _R_OPEN @ np.array([-25.0 * MM, 0.0, DOOR_T / 2 + 2.0 * MM])
    behind = (p[0] + 0.018, -0.10, p[2] + TIP)  # near the door's right-hand end: in the right hand's reach
    over = (p[0] - 0.10, -0.10, p[2] + TIP)
    front = np.array([RX + PK_X0 + 14.0 * MM, RY, TOP + R_H + DOOR_T])  # 16 mm inside the door's front edge
    g = _grip("right")
    return [Waypoint(2.0, (behind[0], behind[1], behind[2] + 0.015), FINGER_CLOSED, "behind the door", g),
            Waypoint(1.0, behind, FINGER_CLOSED, "at its face", g),
            Waypoint(1.5, over, FINGER_CLOSED, "push it over", g),
            # out of the door's swing (it is 250 mm wide: y +/-125 mm) while it falls shut
            Waypoint(1.2, (over[0] - 0.02, -0.17, over[2]), FINGER_CLOSED, "out of its way", g),
            Waypoint(2.5, (over[0] - 0.02, -0.17, over[2]), FINGER_CLOSED, "let it fall shut", g),
            Waypoint(1.5, (front[0], front[1], front[2] + TIP + 0.03), FINGER_CLOSED, "over its front edge", g),
            Waypoint(1.2, (front[0], front[1], front[2] + TIP - 45.0 * MM), FINGER_CLOSED, "press it shut", g),
            Waypoint(1.0, (front[0], front[1], front[2] + TIP + 0.04), FINGER_CLOSED, "lift off", g),
            Waypoint(2.5, RHOME, FINGER_OPEN, "home", 0.0)]


def demo():
    right = [Waypoint(1.0, RHOME, FINGER_OPEN, "wait", 0.0)] + fit_brush("square", "right", RHOME)
    t_right = sum(w.duration for w in right)
    left = [Waypoint(t_right, LHOME, FINGER_OPEN, "wait", 0.0)] + fit_brush("hex", "left", LHOME)
    t_left = sum(w.duration for w in left)
    right += [Waypoint(t_left - t_right, RHOME, FINGER_OPEN, "wait for the left", 0.0)]
    right += press_bearings() + close_door()
    left.append(Waypoint(sum(w.duration for w in right) - t_left, LHOME, FINGER_OPEN, "wait", 0.0))
    for plan in (left, right):
        plan.append(Waypoint(2.0, plan[-1].pos, plan[-1].finger, "done", plan[-1].yaw))
    return {"right": right, "left": left}
