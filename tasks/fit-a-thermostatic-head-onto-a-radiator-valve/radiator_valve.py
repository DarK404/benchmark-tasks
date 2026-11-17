"""On the station table: a small steel panel radiator stands on its floor brackets along the far side, its thermostatic
radiator valve at its right-hand end, the valve's spindle pointing to the robot's right, its protective cap off. A new
Danfoss RA 2000 snap-on thermostatic sensor (head), set to 5 as it comes, lies on its side on the table in front.

The robot must fit the head to the valve as Danfoss says: offered with its scale pointer at the top, pushed straight
onto the valve until its snap fitting clicks home behind the valve's collar. A head left part way on is loose and
does not work the valve.

Run it:  inspect-robots run --embodiment yam_arena -E task=radiator_valve.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "Fit the thermostatic head onto the radiator valve, its scale pointer at the top." ...
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import valvekit  # noqa: E402
from valvekit import Spring, body, box  # noqa: E402
from yam_arena.analysis import box_signed_distance  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

valvekit.install()
SCENE = "station"
TOP = TABLE_TOP[SCENE]
MM = 0.001
WHITE = (0.93, 0.93, 0.91)  # radiator enamel and the head's housing
BRASS = (0.78, 0.64, 0.32)  # the valve body
CHROME = (0.75, 0.76, 0.78)
DARK = (0.20, 0.20, 0.22)
RED = (0.80, 0.10, 0.10)  # the scale pointer
valvekit.METALLIC.update({BRASS: 0.35, CHROME: 0.2})

# -- sizes (env frame: x forward, y left, z up; the table top at TOP) ------------------------------------------------
# The valve: its seat frame has its origin on the axis at the face of the valve's shoulder (where the head's base
# stops), +y back into the valve; the collar (the RA snap connection) stands 16 mm out towards -y.
VX, VS_Y, AXIS_Z = 0.34, -0.120, 80.0 * MM  # the shoulder's centre (x, y) and the spindle's height above the table
COLLAR_L = 16.0 * MM
CORE_R = 10.0 * MM  # the collar's core
# its snap ridge, square at the back (the shoulder side): the teeth drop in behind it 0.6 mm before the base meets the
# shoulder
RIDGE_R, RIDGE_Y0, RIDGE_Y1 = 12.0 * MM, -10.0 * MM, -6.6 * MM
LEAD_DEG = 20.0  # the ridge's lead-in face (towards the tip) leans this far from the axis
TIP_R, TIP_L = 8.0 * MM, 2.0 * MM  # the collar's chamfered tip

# The head: Danfoss RA 2000 (RA 2990) snap-on sensor, about 49 mm across and 90 mm long, 0.15 kg (our estimate from the
# data sheet's drawing). Its frame: origin at the middle of its base face, +y out of the base (towards the valve).
HEAD_R, HEAD_L = 24.5 * MM, 90.0 * MM
HEAD_MASS = 0.15
BORE_R, BORE_D = 13.0 * MM, 18.0 * MM  # the bore in its base that takes the collar
# the snap fitting: two sprung teeth either side (+/-x) of the bore, 2-6 mm in from the base face, their inner faces
# 11 mm from the axis at rest: the collar's 12 mm ridge spreads them 1 mm, and they drop in behind it
TOOTH_IN, TOOTH_T, TOOTH_Y0, TOOTH_Y1, TOOTH_H = 11.0 * MM, 2.0 * MM, -6.0 * MM, -2.0 * MM, 8.0 * MM
TOOTH_MASS = 0.004
SNAP_K = 2000.0  # N/m
SNAP_PRELOAD = float(os.environ.get("RV_PRELOAD", "3"))  # N each, against its stop
SNAP_TRAVEL = 2.5 * MM
SNAP_MU = 0.3

# The radiator: a Type 11 steel panel, 300 mm high and 400 mm long, on floor brackets, its bottom 100 mm up
RAD_X0, RAD_X1 = VX - 25.0 * MM, VX + 25.0 * MM
RAD_Y0, RAD_Y1 = VS_Y + 45.0 * MM, VS_Y + 445.0 * MM
RAD_Z0, RAD_Z1 = 100.0 * MM, 400.0 * MM


def _prism_y(yc, length, r, colour, x=0.0, z=0.0) -> list:
    """A 12-sided prism about the y axis (inradius ``r``), from six boxes turned 30 degrees apart about y."""
    side = 2 * r * math.tan(math.pi / 12)
    return [box((x, yc, z), (2 * r, length, side), colour, tilt_y=k * math.pi / 6) for k in range(6)]


def _ring_y(y0, y1, r_in, r_out, colour, segments=12) -> list:
    """A ring about the y axis from ``y0`` to ``y1``, as ``segments`` boxes."""
    r_mid, w = (r_in + r_out) / 2, r_out - r_in
    chord = 2 * r_out * math.tan(math.pi / segments) + 0.0003
    out = []
    for k in range(segments):
        a = 2 * math.pi * k / segments
        out.append(box((r_mid * math.cos(a), (y0 + y1) / 2, -r_mid * math.sin(a)), (w, y1 - y0, chord), colour,
                       tilt_y=a))
    return out


def _valve_shapes() -> list:
    """The valve in its seat frame: the collar (core, chamfered tip, snap ridge at +/-x with its lead-in faces), the
    shoulder and body, and its tails."""
    out = []
    core_len = COLLAR_L - TIP_L
    out += _prism_y(-core_len / 2, core_len, CORE_R, CHROME)
    out += _prism_y(-COLLAR_L + TIP_L / 2, TIP_L, TIP_R, CHROME)
    lead = (RIDGE_R - CORE_R) / math.tan(math.radians(LEAD_DEG))
    t = 0.8 * MM
    for s in (-1.0, 1.0):
        # the ridge's square-backed band and its sloping lead-in face (towards the tip)
        # (the band stands 0.2 mm lower than the slope's crest, so a tooth rides over the crest and drops onto it
        # rather than catching the band's front edge)
        band_r = RIDGE_R - 0.2 * MM
        out.append(box((s * (CORE_R + band_r) / 2, (RIDGE_Y0 + RIDGE_Y1) / 2, 0.0),
                       (band_r - CORE_R, RIDGE_Y1 - RIDGE_Y0, 8.0 * MM), CHROME))
        slope = math.hypot(RIDGE_R - CORE_R, lead)
        n = np.array([s * lead, -(RIDGE_R - CORE_R)]) / slope  # outward normal (x, y)
        c = np.array([s * (CORE_R + RIDGE_R) / 2, RIDGE_Y0 - lead / 2]) - n * t / 2
        out.append(box((c[0], c[1], 0.0), (t, slope, 8.0 * MM), CHROME,
                       yaw=-s * math.atan2(RIDGE_R - CORE_R, lead)))
        # the tip's chamfer: it steers an off-centre tooth outwards
        tc = math.hypot(CORE_R - TIP_R, TIP_L)
        n2 = np.array([s * TIP_L, -(CORE_R - TIP_R)]) / tc
        c2 = np.array([s * (CORE_R + TIP_R) / 2, -COLLAR_L + TIP_L / 2]) - n2 * t / 2
        out.append(box((c2[0], c2[1], 0.0), (t, tc, 8.0 * MM), CHROME, yaw=-s * math.atan2(CORE_R - TIP_R, TIP_L)))
    # the shoulder (a flange the head's base stops against), the body, and the tails to the radiator and the floor
    out += _prism_y(2.0 * MM, 4.0 * MM, 18.0 * MM, BRASS)
    out.append(box((0.0, 20.0 * MM, 0.0), (32.0 * MM, 32.0 * MM, 32.0 * MM), BRASS))
    out.append(box((0.0, 40.0 * MM, (RAD_Z0 + 20 * MM - AXIS_Z) / 2), (16.0 * MM, 16.0 * MM,
                                                                      RAD_Z0 + 20 * MM - AXIS_Z + 16 * MM), CHROME))
    out.append(box((0.0, 20.0 * MM, -AXIS_Z / 2), (15.0 * MM, 15.0 * MM, AXIS_Z), CHROME))
    return out


def _radiator_shapes() -> list:
    """The radiator panel and its two floor brackets, in the seat frame."""
    cx, cy = (RAD_X0 + RAD_X1) / 2 - VX, (RAD_Y0 + RAD_Y1) / 2 - VS_Y
    out = [box((cx, cy, (RAD_Z0 + RAD_Z1) / 2 - AXIS_Z), (RAD_X1 - RAD_X0, RAD_Y1 - RAD_Y0, RAD_Z1 - RAD_Z0), WHITE)]
    for fy in (RAD_Y0 + 40 * MM, RAD_Y1 - 40 * MM):
        out.append(box((cx, fy - VS_Y, RAD_Z0 / 2 - AXIS_Z), (40.0 * MM, 20.0 * MM, RAD_Z0), WHITE))
    return out


_SEAT = np.array([VX, VS_Y, TOP + AXIS_Z])
_VALVE_SHAPES = _valve_shapes()
VALVE = body("radiator", pos=tuple(_SEAT), shapes=_VALVE_SHAPES + _radiator_shapes(), friction=0.4,
             contact_offset=0.0005)
for _i in range(len(_VALVE_SHAPES)):
    valvekit.shape_friction(VALVE.name, _i, SNAP_MU)


def _head_shapes() -> list:
    out = []
    solid = HEAD_L - BORE_D
    out += _prism_y(-BORE_D - solid / 2, solid, HEAD_R, WHITE)
    out += _ring_y(-BORE_D, 0.0, BORE_R, HEAD_R, WHITE)
    # the scale pointer (on top, by the base) and the dial's grip ribs near the far end
    out.append(box((0.0, -8.0 * MM, HEAD_R + 0.3 * MM), (3.0 * MM, 8.0 * MM, 0.6 * MM), RED))
    out.append(box((0.0, -HEAD_L + 15.0 * MM, HEAD_R + 0.3 * MM), (14.0 * MM, 2.0 * MM, 0.6 * MM), DARK))
    return out


# the new head: lying on its side on the table, its base towards the valve (+y), its scale pointer up
HEAD_START = np.array([0.24, -0.26, TOP + HEAD_R])
_HS = _head_shapes()
HEAD = body("thermostat_head", pos=tuple(HEAD_START), shapes=_HS, mass=HEAD_MASS, friction=0.7,
            contact_offset=0.0005, solver_iterations=32, visual_only=(len(_HS) - 2, len(_HS) - 1),
            com=(0.0, -HEAD_L / 2, 0.0))


def _tooth(sign: float):
    """One of the snap fitting's two teeth (sign +1: on the head's +x side), at rest against its stop."""
    local = np.array([sign * (TOOTH_IN + TOOTH_T / 2), (TOOTH_Y0 + TOOTH_Y1) / 2, 0.0])
    return body(f"snap_{'a' if sign > 0 else 'b'}", pos=tuple(HEAD_START + local),
                shapes=[box((0.0, 0.0, 0.0), (TOOTH_T, TOOTH_Y1 - TOOTH_Y0, TOOTH_H), DARK)],
                mass=TOOTH_MASS, friction=SNAP_MU, contact_offset=0.0005, solver_iterations=32)


SNAP_A, SNAP_B = _tooth(+1.0), _tooth(-1.0)
OBJECTS = [VALVE, HEAD, SNAP_A, SNAP_B]
valvekit.register(*OBJECTS)
_pre = SNAP_PRELOAD / SNAP_K
valvekit.add_spring(Spring(HEAD.name, SNAP_A.name, axis="X", stiffness=SNAP_K, damping=2.0, target=-_pre,
                           lower=0.0, upper=SNAP_TRAVEL))
valvekit.add_spring(Spring(HEAD.name, SNAP_B.name, axis="X", stiffness=SNAP_K, damping=2.0, target=+_pre,
                           lower=-SNAP_TRAVEL, upper=0.0))

# -- scoring -------------------------------------------------------------------------------------------------------
# Measured from the head's pose against its seated pose on the valve:
# 1.0   snapped home (its base within 1.0 mm of the valve's shoulder, so its teeth are behind the collar's ridge),
#       on the axis (within 1.5 mm, square within 3 degrees), its scale pointer at the top (within 20 degrees); no
#       gripper within 1 cm of it; still; for 1 s
# 0.5   snapped home but the pointer not at the top, or a hand still on it / not yet still
# 0.25  on the collar past its ridge's start but not snapped home (part way on)
# 0     anything else
SCORE_OVER = "end"
RUN_LIMIT_STEPS = 1200
STILL_STEPS = 20
CLEAR_DIST = 10.0 * MM
SEAT_TOL = 1.0 * MM
AXIS_TOL = 1.5 * MM
SQUARE_DEG = 3.0
POINTER_DEG = 20.0
_STATE: dict = {}


@valvekit.on_reset
def _reset_state() -> None:
    _STATE.update(still=0)


_reset_state()


def to_seat(st) -> tuple[np.ndarray, np.ndarray]:
    """The head's (origin, rotation) in the valve's seat frame."""
    return np.asarray(st.pos) - _SEAT, st.rotation()


def depth(st) -> float:
    """How far (m) the collar is in the head's bore: 16 mm seated."""
    p, r = to_seat(st)
    return COLLAR_L + float(p[1])


def off_axis(st) -> tuple[float, float]:
    """The head's base centre's distance (m) from the valve's axis, and its axis's angle (degrees) from it."""
    p, r = to_seat(st)
    axis = r @ np.array([0.0, 1.0, 0.0])
    return float(math.hypot(p[0], p[2])), math.degrees(math.acos(min(1.0, abs(float(axis[1])))))


def pointer_deg(st) -> float:
    """How far (degrees) the scale pointer is turned from the top, about the head's axis."""
    up = st.rotation() @ np.array([0.0, 0.0, 1.0])
    return math.degrees(math.atan2(abs(float(up[0])), float(up[2])))


def snapped(st) -> bool:
    off, ang = off_axis(st)
    return depth(st) >= COLLAR_L - SEAT_TOL and off <= AXIS_TOL and ang <= SQUARE_DEG


def started(st) -> bool:
    """On the collar with its teeth at or past the ridge's start."""
    off, ang = off_axis(st)
    return depth(st) >= COLLAR_L + RIDGE_Y0 + (-TOOTH_Y1) and off <= 3.0 * MM and ang <= 10.0


def hands_clear(objects) -> bool:
    pts = valvekit.finger_points()
    if not len(pts):
        return True
    st = objects[HEAD.name]
    centre = st.to_world((0.0, -HEAD_L / 2, 0.0))
    size = np.array([2 * HEAD_R, HEAD_L, 2 * HEAD_R])
    return box_signed_distance(pts, centre, st.rotation(), size).min() >= CLEAR_DIST


def _update(objects) -> tuple[float, bool]:
    st = objects[HEAD.name]
    if snapped(st):
        if pointer_deg(st) <= POINTER_DEG and st.resting() and hands_clear(objects):
            return 1.0, True
        return 0.5, False
    if started(st):
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
# Open loop, from the fixed layout only. The right hand takes the head across its middle (fingers closing along x),
# lifts it, lines its bore up with the valve's collar, pushes it straight on (+y) until it snaps home, lets go and
# draws back.
TIP = 13.0 * MM  # the TCP is this far above the pad tips
RHOME, LHOME = (0.14, -0.22, TOP + 0.17), (0.14, 0.22, TOP + 0.17)
GRIP_Y = -55.0 * MM  # along the head: 55 mm from its base, so the hand (about 100 mm wide along y) clears the valve
GRIP_DZ = float(os.environ.get("RV_GRIPZ", "-4")) * MM  # the pad tips this far above the head's axis
OPEN_W = -(HEAD_R + 12.0 * MM)
HOLD = -(HEAD_R - 6.0 * MM)  # commanded 6 mm inside: a firm grip
APPROACH = 25.0 * MM  # lined up this far short of the collar's tip
PUSH_PAST = float(os.environ.get("RV_PUSH", "40")) * MM  # aimed this far past seated (the soft arm)
CAL = np.array([float(v) for v in os.environ.get("RV_CAL", "1.7,0,2.7").split(",")]) * MM
YAW = math.pi / 2  # fingers closing along x


def _tcp(head_origin) -> tuple[float, float, float]:
    p = np.asarray(head_origin) + np.array([0.0, GRIP_Y, GRIP_DZ + TIP]) + CAL
    return tuple(float(v) for v in p)


def take() -> list:
    at = _tcp(HEAD_START - np.array([0.0, 0.0, 0.0]))
    at0 = (at[0] - CAL[0], at[1] - CAL[1], at[2] - CAL[2])
    return [Waypoint(2.0, (at0[0], at0[1], at0[2] + 0.05), OPEN_W, "over the head", YAW),
            Waypoint(1.2, at0, OPEN_W, "round its middle", YAW),
            Waypoint(1.0, at0, HOLD, "take it", YAW),
            Waypoint(1.2, (at0[0], at0[1], at0[2] + 0.05), HOLD, "lift it", YAW)]


def fit() -> list:
    """``RV_DEMO=halfway`` (a shortcut check): push it only 8 mm onto the collar, short of the snap, and let go."""
    lined = _tcp(_SEAT + np.array([0.0, -COLLAR_L - APPROACH, 0.0]))
    past = -8.0 * MM if os.environ.get("RV_DEMO") == "halfway" else PUSH_PAST
    home = _tcp(_SEAT + np.array([0.0, past, 0.0]))
    seat = _tcp(_SEAT)
    back = (seat[0], seat[1] - 0.06, seat[2] + 0.04)
    return [Waypoint(2.0, (lined[0], lined[1], lined[2] + 0.03), HOLD, "over the valve", YAW),
            Waypoint(1.2, lined, HOLD, "lined up with the collar", YAW),
            Waypoint(1.2, lined, HOLD, "settle", YAW),
            Waypoint(2.0, home, HOLD, "push it on", YAW),
            Waypoint(1.0, home, HOLD, "hold", YAW),
            Waypoint(1.0, seat, HOLD, "ease off", YAW),
            Waypoint(1.2, seat, OPEN_W, "let go", YAW),
            Waypoint(1.2, (seat[0], seat[1], seat[2] + 0.05), OPEN_W, "up", YAW),
            Waypoint(1.5, RHOME, FINGER_OPEN, "home", 0.0)]


def demo():
    right = [Waypoint(1.0, RHOME, FINGER_OPEN, "wait", 0.0)] + take() + fit()
    left = [Waypoint(sum(w.duration for w in right), LHOME, FINGER_OPEN, "wait", 0.0)]
    for plan in (left, right):
        plan.append(Waypoint(2.0, plan[-1].pos, plan[-1].finger, "done", plan[-1].yaw))
    return {"right": right, "left": left}
