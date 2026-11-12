"""On the station table: a cabinet side stands upright in an assembly jig with two connecting bolts and two dowels
standing out of its face; a shelf lies flat in front of it, its two cam locks turned the wrong way; a stubby
screwdriver stands in a holder either side.

The robot must join the shelf to the side with its cam locks: turn each cam's opening to the shelf's edge, push the
shelf onto the bolts, turn the cams to draw the joint tight, and put the screwdrivers back. Pushed onto a cam that is
still closed, the shelf stops 9 mm short of the side.

Run it:  inspect-robots run --embodiment yam_arena -E task=cam_lock.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "..." --scorer task_score ...
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import camkit  # noqa: E402
from camkit import CamLockJoint, body, box, cyl, prism12  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

camkit.install()
SCENE = "station"
TOP = TABLE_TOP[SCENE]
MM = 0.001
MELAMINE = (0.93, 0.93, 0.91)  # white melamine faced particleboard
CHIPBOARD = (0.78, 0.68, 0.52)  # its raw edge in the drilled holes
ZINC = (0.74, 0.75, 0.77)
DARK = (0.08, 0.08, 0.09)
STEEL = (0.50, 0.51, 0.53)
WOOD = (0.55, 0.40, 0.25)
BEECH = (0.80, 0.66, 0.45)  # the dowels
HANDLE_R, HANDLE_L = (0.85, 0.15, 0.10), (0.95, 0.75, 0.10)  # the two screwdrivers' handles: red, yellow
camkit.METALLIC.update({ZINC: 0.3, STEEL: 0.35})

# -- the cabinet side, upright in its jig ----------------------------------------------------------------------------
# WHY 0.40 (2026-11-12): with the cams 24 mm in from the edge (was 34) this keeps them where the demo's paths were
# proven; at 0.41 the hand above a cam, 15 cm up, was 4 mm out of reach.
FACE_X = 0.40  # the side's inner face (the shelf's edge meets it here)
PANEL_T = 16.0 * MM
SIDE_W, SIDE_H = 0.32, 0.12
BOLT_Y = (-0.09, 0.09)  # the connecting bolts (right, left), at the shelf's mid-thickness
DOWEL_Y = (-0.115, 0.115)
BOLT_LEN = 25.0 * MM  # out of the face to the head's end
SHANK_R, HEAD_R, HEAD_L = 3.5 * MM, 5.0 * MM, 5.0 * MM
DOWEL_R, DOWEL_OUT = 4.0 * MM, 12.0 * MM


def _side_shapes() -> tuple[list, list]:
    mid = PANEL_T / 2
    solid = [
        box((PANEL_T / 2, 0.0, SIDE_H / 2), (PANEL_T, SIDE_W, SIDE_H), MELAMINE),  # the side
        box((PANEL_T + 15.0 * MM, -0.13, 10.0 * MM), (30.0 * MM, 30.0 * MM, 20.0 * MM), STEEL),  # the jig's two shoes
        box((PANEL_T + 15.0 * MM, 0.13, 10.0 * MM), (30.0 * MM, 30.0 * MM, 20.0 * MM), STEEL),
        box((PANEL_T + 2.0 * MM, -0.13, 50.0 * MM), (4.0 * MM, 30.0 * MM, 100.0 * MM), STEEL),  # and their uprights
        box((PANEL_T + 2.0 * MM, 0.13, 50.0 * MM), (4.0 * MM, 30.0 * MM, 100.0 * MM), STEEL),
    ]
    seen = []
    for y in BOLT_Y:
        seen += [cyl((-(BOLT_LEN - HEAD_L) / 2, y, mid), SHANK_R, BOLT_LEN - HEAD_L, ZINC, axis="X"),
                 cyl((-(BOLT_LEN - HEAD_L / 2), y, mid), HEAD_R, HEAD_L, ZINC, axis="X")]
    for y in DOWEL_Y:
        seen.append(cyl((-DOWEL_OUT / 2, y, mid), DOWEL_R, DOWEL_OUT, BEECH, axis="X"))
    return solid, seen


_SS, _SV = _side_shapes()
SIDE = body("side", pos=(FACE_X, 0.0, TOP), friction=0.6, shapes=_SS + _SV,
            visual_only=tuple(range(len(_SS), len(_SS) + len(_SV))))

# -- the shelf -----------------------------------------------------------------------------------------------------
# Its frame: origin at its middle, x towards its joining edge, z up.
SHELF_D, SHELF_W = 0.20, 0.26
HALF = SHELF_D / 2
# A Minifix-type 15 mm cam connector, drilling distance B = 24 mm: the cam's housing in a 15 mm bore whose centre is 24
# mm in from the joining edge, and the connecting bolt made for B = 24 mm standing out of the side's face, so that its
# head reaches the housing's middle. WHY 24 (2026-11-12, review): 34 mm is the other standard distance B, which takes
# a longer bolt; a cam 34 mm in could not reach the 25 mm bolt.
CAM_IN = 24.0 * MM  # the cams' middles in from the joining edge (drilling distance B)
CAM_R = 7.5 * MM  # a 15 mm cam, in a 15 mm bore
CAM_XY = [(HALF - CAM_IN, y) for y in BOLT_Y]
START_GAP = 30.0 * MM  # the shelf's edge from the side's face, as it lies at the start
SHELF_X0 = FACE_X - START_GAP - HALF
PHI0 = (math.radians(90.0), math.radians(270.0))  # the cams as found: clockwise from open (right, left)


def _shelf_shapes() -> tuple[list, list]:
    t = PANEL_T
    solid = [box((0.0, 0.0, 0.0), (SHELF_D, SHELF_W, t), MELAMINE)]
    seen = []
    for (cx, cy), y in zip(CAM_XY, BOLT_Y):  # the holes in the joining edge, and the cams' bores round the cams
        seen.append(box((HALF + 0.1 * MM, y, 0.0), (0.2 * MM, 2 * SHANK_R + 1.0 * MM, 2 * SHANK_R + 1.0 * MM), DARK))
        seen += prism12((cx, cy, t / 2 + 0.05 * MM), CAM_R + 0.6 * MM, 0.1 * MM, CHIPBOARD)
    for y in DOWEL_Y:
        seen.append(box((HALF + 0.1 * MM, y, 0.0), (0.2 * MM, 2 * DOWEL_R, 2 * DOWEL_R), CHIPBOARD))
    return solid, seen


_HS, _HV = _shelf_shapes()
SHELF = body("shelf", pos=(SHELF_X0, 0.0, TOP + PANEL_T / 2), mass=0.50, friction=0.4, shapes=_HS + _HV,
             visual_only=tuple(range(len(_HS), len(_HS) + len(_HV))))


def _cam_shapes() -> list:
    """A cam seen from above, in its own frame (origin on the shelf's top face at its middle): zinc, a cross recess, an
    arrow pointing to its opening (+x)."""
    return [
        *prism12((0.0, 0.0, 0.15 * MM), CAM_R, 0.3 * MM, ZINC),
        box((0.0, 0.0, 0.35 * MM), (7.0 * MM, 1.4 * MM, 0.1 * MM), DARK),  # the cross recess
        box((0.0, 0.0, 0.35 * MM), (1.4 * MM, 7.0 * MM, 0.1 * MM), DARK),
        box((5.2 * MM, 0.0, 0.35 * MM), (3.0 * MM, 1.0 * MM, 0.1 * MM), DARK),  # the arrow's shaft
        box((6.2 * MM, 0.9 * MM, 0.35 * MM), (1.8 * MM, 0.8 * MM, 0.1 * MM), DARK, yaw=math.radians(-35.0)),
        box((6.2 * MM, -0.9 * MM, 0.35 * MM), (1.8 * MM, 0.8 * MM, 0.1 * MM), DARK, yaw=math.radians(35.0)),
    ]


_CS = _cam_shapes()
CAMS = []
for k, (cx, cy) in enumerate(CAM_XY):
    yaw0 = -PHI0[k]
    c = body(f"cam_{'rl'[k]}", pos=(SHELF_X0 + cx, cy, TOP + PANEL_T), mass=1e-3, inertia=(1e-8, 1e-8, 1e-8),
             quat=(0.0, 0.0, math.sin(yaw0 / 2), math.cos(yaw0 / 2)), shapes=_CS, visual_only=tuple(range(len(_CS))))
    camkit.no_gravity(c.name)
    CAMS.append(c)

# -- the screwdrivers in their holders -------------------------------------------------------------------------------
# A PH2 x 100 screwdriver: its frame's origin at its tip, z up its axis. (Long enough that the hand turning it stays
# above the cabinet side's top edge.)
BLADE_L, HANDLE_LEN, HANDLE_IR = 100.0 * MM, 45.0 * MM, 14.0 * MM
HOLDER_XY = ((0.24, -0.20), (0.24, 0.20))  # right, left
HOLDER_H, HOLE = 30.0 * MM, 9.0 * MM  # a 9 mm hole for the 6 mm blade, its mouth chamfered


def _driver_shapes(color) -> list:
    return [
        *prism12((0.0, 0.0, BLADE_L / 2 + 1.0 * MM), 3.0 * MM, BLADE_L - 2.0 * MM, STEEL),  # the blade
        *prism12((0.0, 0.0, BLADE_L + HANDLE_LEN / 2), HANDLE_IR, HANDLE_LEN, color),  # the handle
        box((0.0, 0.0, 0.5 * MM), (3.0 * MM, 3.0 * MM, 1.0 * MM), DARK),  # the tip
    ]


DRIVERS = []
for k, (hx, hy) in enumerate(HOLDER_XY):
    d = body(f"driver_{'rl'[k]}", pos=(hx, hy, TOP + 0.5 * MM), mass=0.12, com=(0.0, 0.0, 0.10), friction=1.0,
             shapes=_driver_shapes((HANDLE_R, HANDLE_L)[k]), solver_iterations=32)
    DRIVERS.append(d)


def _holder_shapes() -> list:
    w = 40.0 * MM
    side = (w - HOLE) / 2
    out = []
    for hx, hy in HOLDER_XY:
        for dx, dy, sx, sy in [(0.0, (HOLE + side) / 2, w, side), (0.0, -(HOLE + side) / 2, w, side),
                               ((HOLE + side) / 2, 0.0, side, HOLE), (-(HOLE + side) / 2, 0.0, side, HOLE)]:
            out.append(box((hx + dx, hy + dy, HOLDER_H / 2), (sx, sy, HOLDER_H), WOOD))
    return out


HOLDERS = body("holders", pos=(0.0, 0.0, TOP), friction=0.6, shapes=_holder_shapes())

OBJECTS = [SIDE, SHELF, *CAMS, *DRIVERS, HOLDERS]
camkit.register(*OBJECTS)
MODEL = camkit.add_model(CamLockJoint(
    "shelf", [d.name for d in DRIVERS], [c.name for c in CAMS], FACE_X, HALF, BOLT_Y, CAM_XY, PANEL_T / 2,
    bolt_len=BOLT_LEN, phi0=PHI0, holes=[(*xy, TOP + HOLDER_H) for xy in HOLDER_XY]))


# -- scoring -------------------------------------------------------------------------------------------------------
SCORE_OVER = "end"
RUN_LIMIT_STEPS = 2000
CLOSED = 0.5 * MM
CLEAR = 10.0 * MM
STILL = 0.01


def fingers_to(state, spec) -> float:
    """Nearest distance (m) from either gripper's hulls to a body's bounding box (in its own frame). inf offline."""
    pts = camkit.finger_points()
    if not len(pts):
        return math.inf
    lo, hi = spec.bounds_local()
    local = (pts - state.pos) @ state.rotation()
    d = np.maximum(np.maximum(lo - local, local - hi), 0.0)
    return float(np.linalg.norm(d, axis=1).min())


def joined() -> bool:
    """Both cams locked on their bolts and the joint shut at both."""
    m = MODEL
    return all(m.locked(k) and m.gap[k] <= CLOSED for k in range(2))


def in_holder(driver, k) -> bool:
    return bool(np.hypot(*(np.asarray(driver.pos[:2]) - HOLDER_XY[k])) < 4.0 * MM and driver.pos[2] < TOP + 6.0 * MM
                and driver.tilt_deg() < 10.0)


def _update(objects) -> tuple[float, bool]:
    m = MODEL
    if joined():
        drivers = [objects[d.name] for d in DRIVERS]
        if all(in_holder(drv, k) for k, drv in enumerate(drivers)) and min(
                [fingers_to(objects["shelf"], SHELF)] + [fingers_to(drv, spec) for drv, spec in zip(drivers, DRIVERS)]
        ) >= CLEAR:
            return 1.0, bool(objects["shelf"].resting(max_speed=STILL)
                             and all(drv.resting(max_speed=STILL) for drv in drivers))
        return 0.5, False
    if all(m.bolt_in):
        return 0.25, False
    return 0.0, False


STILL_STEPS = 20  # 1 s at the 20 Hz control rate
_STATE: dict = {}


@camkit.on_reset
def _reset_state() -> None:
    _STATE.update(still=0)


_reset_state()


def success(objects) -> bool:
    """The shelf joined (both cams locked, the joint shut), the screwdrivers back in their holders, released and still
    (held 1 s by the run). It also counts the steps the full-score state has held, for score()."""
    ok = _update(objects)[1]
    _STATE["still"] = _STATE["still"] + 1 if ok else 0
    return ok


def score(objects) -> float:
    """1.0 / 0.5 / 0.25 / 0 as in the README. 1.0 only once the full-score state has held for 1 s (counted by success(),
    which the run calls first every step); until then 0.5."""
    tier = _update(objects)[0]
    if tier == 1.0 and _STATE["still"] < STILL_STEPS:
        return 0.5
    return tier


# -- scripted demo -------------------------------------------------------------------------------------------------
# Open loop, from the fixed layout. Each hand takes its screwdriver, turns its cam's arrow to the shelf's edge (the
# right cam a quarter turn anticlockwise, the left a quarter turn clockwise) and puts the screwdriver back; both hands
# push the shelf onto the bolts; each takes its screwdriver again and turns its cam half a round clockwise, in two
# quarter turns with a regrip between them, drawing the joint shut; both put the screwdrivers back.
RHOME, LHOME = (0.16, -0.22, TOP + 0.15), (0.16, 0.22, TOP + 0.15)
TIP_PAST_TCP = 13.0 * MM
DRIVER_TCP = BLADE_L + 10.0 * MM + TIP_PAST_TCP  # the TCP above the screwdriver's tip, holding it by its handle
HOLD = FINGER_OPEN * 18.0 / 94.0  # closed to 18 mm on the 28 mm handle
PUSH_TCP_Z = 5.0 * MM + TIP_PAST_TCP  # closed fingertips 5 mm above the table, on the shelf's near edge
PUSH_BEHIND = 12.0 * MM  # the TCP behind the edge it pushes
LIFT = HOLDER_H + 15.0 * MM  # out of the holder: the blade's tip clears its rim
CLEAR_Z = 0.17  # above the screwdrivers standing in their holders
PUSH_PAST = float(os.environ.get("PUSH_PAST", "0.012"))  # the hands push this far past home
TWIST = math.radians(90.0)
YAW_R, YAW_L = math.radians(90.0), math.radians(-90.0)  # each wrist's middle; a twist runs +-45 degrees about it
DEMO_MODE = os.environ.get("CAM_DEMO", "")  # shortcut checks: "noset" skips setting the cams; "force" also shoves


def cam_at(k, gap):
    """A cam's middle (world, on the shelf's top) with the shelf's edge ``gap`` from the side's face."""
    x = FACE_X - gap - HALF + CAM_XY[k][0]
    return np.array([x, CAM_XY[k][1], TOP + PANEL_T])


def _w(dur, pos, finger, label, yaw):
    return Waypoint(dur, tuple(float(v) for v in pos), finger, label, yaw)


def _t(w) -> float:
    return sum(p.duration for p in w)


def _fetch(k, mid):
    """Take screwdriver ``k`` out of its holder, the wrist turned to ``mid``."""
    h = np.array([*HOLDER_XY[k], TOP + 0.5 * MM])
    up = np.array([0.0, 0.0, 1.0])
    return [
        _w(1.8, h + (DRIVER_TCP + 0.03) * up, FINGER_OPEN, "over the screwdriver", mid),
        _w(1.2, h + DRIVER_TCP * up, FINGER_OPEN, "round it", mid),
        _w(1.6, h + DRIVER_TCP * up, HOLD, "grip", mid),
        _w(1.4, h + (DRIVER_TCP + LIFT) * up, HOLD, "lift it out", mid),
    ]


def _put_back(k, at, yaw):
    h = np.array([*HOLDER_XY[k], TOP + 0.5 * MM])
    up = np.array([0.0, 0.0, 1.0])
    return [
        _w(1.0, at + (DRIVER_TCP + 0.015) * up, HOLD, "lift off", yaw),
        _w(1.8, h + (DRIVER_TCP + LIFT) * up, HOLD, "over the holder", yaw),
        _w(1.0, h + (DRIVER_TCP + 0.015) * up, HOLD, "steady", yaw),
        _w(1.2, h + (DRIVER_TCP + 0.002) * up, HOLD, "stand it in", yaw),
        _w(1.6, h + (DRIVER_TCP + 0.002) * up, FINGER_OPEN, "let go", yaw),
        _w(1.0, h + (DRIVER_TCP + 0.05) * up, FINGER_OPEN, "up", yaw),
    ]


def _engage(at, yaw):
    up = np.array([0.0, 0.0, 1.0])
    return [
        _w(1.8, at + (DRIVER_TCP + 0.015) * up, HOLD, "over the cam", yaw),
        _w(1.0, at + (DRIVER_TCP + 0.008) * up, HOLD, "steady", yaw),
        _w(1.2, at + (DRIVER_TCP - 1.0 * MM) * up, HOLD, "into the recess", yaw),
    ]


def _turn(at, yaw_from, yaw_to, label, regrip=False):
    up = np.array([0.0, 0.0, 1.0])
    out = []
    if regrip:  # let go of the handle (the screwdriver stands in the recess), turn the wrist back, grip again
        out += [_w(1.6, at + (DRIVER_TCP - 1.0 * MM) * up, FINGER_OPEN, "let go", yaw_to),
                _w(1.0, at + (DRIVER_TCP - 1.0 * MM) * up, FINGER_OPEN, "wrist back", yaw_from),
                _w(1.6, at + (DRIVER_TCP - 1.0 * MM) * up, HOLD, "grip", yaw_from)]
    out.append(_w(1.6, at + (DRIVER_TCP - 1.0 * MM) * up, HOLD, label, yaw_to))
    return out


def demo():
    right, left = [], []
    gap0, home = START_GAP, MODEL.g_home
    # 1. each hand sets its cam open (arrow to the edge)
    for k, side, mid, sign in [(0, right, YAW_R, -1.0), (1, left, YAW_L, 1.0)]:
        # right cam: a quarter turn anticlockwise (the wrist from mid - 45 to mid + 45); left: clockwise
        a, b = (mid - TWIST / 2, mid + TWIST / 2) if k == 0 else (mid + TWIST / 2, mid - TWIST / 2)
        at = cam_at(k, gap0)
        side += _fetch(k, a)
        if DEMO_MODE in ("noset", "force"):
            side += [_w(4.0, at + (DRIVER_TCP + 0.015) * np.array([0, 0, 1.0]), HOLD, "look", a)]
            side += _put_back(k, at, a)
        else:
            side += _engage(at, a)
            side += _turn(at, a, b, "turn it open")
            side += _put_back(k, at, b)
    end = max(_t(right), _t(left)) + 2.0  # both home, fingers closed for pushing
    right.append(_w(end - _t(right), RHOME, FINGER_CLOSED, "wait", 0.0))
    left.append(_w(end - _t(left), LHOME, FINGER_CLOSED, "wait", 0.0))
    # 2. both push the shelf onto the bolts, square
    near0 = FACE_X - gap0 - SHELF_D
    past = 0.09 if DEMO_MODE == "force" else PUSH_PAST  # "force": shoved on hard
    for side, k in [(right, 0), (left, 1)]:
        y = BOLT_Y[k]
        behind = near0 - PUSH_BEHIND
        side += [
            _w(1.8, (behind - 0.01, y, TOP + PUSH_TCP_Z + 0.04), FINGER_CLOSED, "behind the shelf", 0.0),
            _w(1.2, (behind - 0.01, y, TOP + PUSH_TCP_Z), FINGER_CLOSED, "down", 0.0),
            _w(1.0, (behind, y, TOP + PUSH_TCP_Z), FINGER_CLOSED, "on its edge", 0.0),
            _w(3.0, (behind + gap0 - home + past, y, TOP + PUSH_TCP_Z), FINGER_CLOSED, "push it on", 0.0),
            _w(0.8, (behind + gap0 - home + past, y, TOP + PUSH_TCP_Z), FINGER_CLOSED, "hold", 0.0),
            _w(1.0, (behind + gap0 - home - 0.01, y, TOP + PUSH_TCP_Z + 0.04), FINGER_CLOSED, "back off", 0.0),
            _w(2.0, (behind + gap0 - home - 0.01, y, TOP + CLEAR_Z), FINGER_OPEN, "up", 0.0),  # clear of the
        ]  # screwdrivers' handles before going over to them
    # 3. each hand locks its cam: half a round clockwise, in two quarter turns
    for k, side, mid in [(0, right, YAW_R), (1, left, YAW_L)]:
        a, b = mid + TWIST / 2, mid - TWIST / 2  # clockwise: the wrist's yaw falls
        at = cam_at(k, home)
        side += _fetch(k, a)
        side += _engage(at, a)
        side += _turn(at, a, b, "lock it, 1")
        side += _turn(at, a, b, "lock it, 2", regrip=True)
        side += _put_back(k, at, b)
        side += [_w(1.5, (RHOME, LHOME)[k], FINGER_OPEN, "home", 0.0)]
    end = max(_t(right), _t(left)) + 2.5
    right.append(_w(end - _t(right), RHOME, FINGER_OPEN, "done", 0.0))
    left.append(_w(end - _t(left), LHOME, FINGER_OPEN, "done", 0.0))
    return {"right": right, "left": left}
