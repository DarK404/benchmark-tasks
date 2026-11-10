"""On the station table: an M10 hex bolt stands tight in a steel bracket on a block, and a 3/8" ratchet with a 17 mm
socket lies in its tray beside it, its direction lever set to tighten.

The robot must undo the bolt with the ratchet and put the ratchet back: see that the ratchet is set to tighten and flip
its lever to loosen, seat the socket on the bolt, and swing the handle until the bolt has come a full turn out.

Run it:  inspect-robots run --embodiment yam_arena -E task=ratchet_bolt.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "..." --scorer task_score ...
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ratchetkit  # noqa: E402
from ratchetkit import Hinge, RatchetBolt, body, box, prism12  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

ratchetkit.install()
SCENE = "station"
TOP = TABLE_TOP[SCENE]
MM = 0.001
CHROME = (0.80, 0.81, 0.83)
STEEL = (0.50, 0.51, 0.53)
ZINC_YELLOW = (0.78, 0.70, 0.35)  # the bolt's zinc-yellow plating
GRIP = (0.12, 0.12, 0.13)
WOOD = (0.55, 0.40, 0.25)
DARK = (0.08, 0.08, 0.09)
RED = (0.75, 0.12, 0.10)
ratchetkit.METALLIC.update({CHROME: 0.12, STEEL: 0.35, ZINC_YELLOW: 0.3})


def hexagon(centre, across_flats, height, color) -> list:
    """An upright hexagonal prism as three boxes turned 60 degrees apart."""
    side = across_flats / math.sqrt(3)
    return [box(centre, (across_flats, side, height), color, yaw=k * math.pi / 3 + math.pi / 2) for k in range(3)]


# -- the bracket and its bolt --------------------------------------------------------------------------------------
BOLT_XY = (0.27, 0.0)
BLOCK_H = 30.0 * MM
PLATE_T = 5.0 * MM
SEAT = BLOCK_H + PLATE_T  # the bracket's top, where the bolt's washer face sits
WASHER_T = 2.0 * MM  # the bolt's washer (M10, 20 mm across)
HEAD_AF, HEAD_H = 17.0 * MM, 6.4 * MM  # M10 hex head (DIN 933): 17 mm across flats, 6.4 mm high
HEAD_TOP = SEAT + WASHER_T + HEAD_H
BRACKET = body("bracket", pos=(*BOLT_XY, TOP), friction=0.7, shapes=[
    box((0.0, 0.0, BLOCK_H / 2), (0.08, 0.08, BLOCK_H), WOOD),
    box((0.0, 0.0, BLOCK_H + PLATE_T / 2), (0.06, 0.07, PLATE_T), STEEL),  # the steel bracket
    box((0.0, 0.035, BLOCK_H + 0.020), (0.06, 0.005, 0.040), STEEL),  # its upstand
    *prism12((0.0, 0.0, SEAT + 0.1 * MM), 9.5 * MM, 0.2 * MM, DARK),  # the bolt's shadow ring on the plate
], visual_only=(3, 4, 5, 6, 7, 8))
HEAD = body("bolt_head", pos=(*BOLT_XY, TOP + SEAT + WASHER_T + HEAD_H / 2), mass=1e-3,
            inertia=(1e-8, 1e-8, 1e-8), shapes=[
                *hexagon((0.0, 0.0, 0.0), HEAD_AF, HEAD_H, ZINC_YELLOW),
                *prism12((0.0, 0.0, -HEAD_H / 2 - WASHER_T / 2), 10.0 * MM, WASHER_T, STEEL),  # the washer
                box((0.0, HEAD_AF / 4, HEAD_H / 2 + 0.05 * MM), (0.8 * MM, HEAD_AF / 2, 0.1 * MM), RED),  # a paint mark
            ], visual_only=tuple(range(3 + 6 + 1)))
ratchetkit.no_gravity(HEAD.name)
# the head's collider, just inside the drawn head: a tight bolt, solid to the fingers (the model meets the socket);
# heavy and held in place every substep, so a push does not move it
SOLID = body("bolt_solid", pos=(*BOLT_XY, TOP + SEAT + WASHER_T + HEAD_H / 2), mass=20.0,
             inertia=(1e-2, 1e-2, 1e-2),
             friction=0.8, shapes=hexagon((0.0, 0.0, 0.0), HEAD_AF - 0.4 * MM, HEAD_H - 0.4 * MM, ZINC_YELLOW))
ratchetkit.no_gravity(SOLID.name)

# -- the ratchet in its tray ---------------------------------------------------------------------------------------
# Its frame: origin at the middle of the socket's mouth, z up the socket's axis, the handle along +x.
SOCKET_R, SOCKET_L = 11.0 * MM, 26.0 * MM
HEAD_Z = (SOCKET_L, SOCKET_L + 18.0 * MM)  # the ratchet's round head
RHEAD_R = 19.0 * MM
HANDLE_Z = (29.5 * MM, 40.5 * MM)
HANDLE_X = (40.0 * MM, 170.0 * MM)
HANDLE_W = 15.0 * MM
LEVER_AT = (-8.0 * MM, 0.0, HEAD_Z[1])  # the direction lever's pivot on the head's top
LEVER_L = 16.0 * MM
LEVER_DEG = 30.0  # the lever's two positions either side of the handle's line: + loosen, - tighten
TRAY_XY = (0.22, -0.06)  # the socket's axis when the ratchet lies in its tray
TRAY_YAW = -math.pi / 2  # its handle towards -y
TRAY_FLOOR = 5.0 * MM


def _ratchet_shapes() -> tuple[list, int]:
    hx = (HANDLE_X[0] + HANDLE_X[1]) / 2
    hz = (HANDLE_Z[0] + HANDLE_Z[1]) / 2
    solid = [
        *prism12((0.0, 0.0, SOCKET_L / 2), SOCKET_R, SOCKET_L, CHROME),  # the socket
        *prism12((0.0, 0.0, (HEAD_Z[0] + HEAD_Z[1]) / 2), RHEAD_R, HEAD_Z[1] - HEAD_Z[0], CHROME),  # the head
        box(((RHEAD_R - 2.0 * MM + HANDLE_X[0]) / 2, 0.0, hz), (HANDLE_X[0] - RHEAD_R + 2.0 * MM, 12.0 * MM, 10.0 * MM),
            CHROME),  # the neck
        box((hx, 0.0, hz), (HANDLE_X[1] - HANDLE_X[0], HANDLE_W, HANDLE_Z[1] - HANDLE_Z[0]), CHROME),  # the handle
    ]
    seen = [
        box((125.0 * MM, 0.0, hz), (85.0 * MM, HANDLE_W + 0.4 * MM, HANDLE_Z[1] - HANDLE_Z[0] + 0.4 * MM), GRIP),
        *prism12((0.0, 0.0, 0.1 * MM), 8.5 * MM, 0.2 * MM, DARK),  # the socket's bore
        *prism12((0.0, 0.0, HEAD_Z[1] + 0.1 * MM), 6.0 * MM, 0.2 * MM, STEEL),  # the release button
        box((-3.0 * MM, 11.0 * MM, HEAD_Z[1] + 0.05 * MM), (5.0 * MM, 1.0 * MM, 0.1 * MM), DARK),  # "L" and "R" marks
        box((-3.0 * MM, -11.0 * MM, HEAD_Z[1] + 0.05 * MM), (5.0 * MM, 1.0 * MM, 0.1 * MM), RED),
    ]
    return solid + seen, len(solid)


_RSH, _NRS = _ratchet_shapes()
_c, _s = math.cos(TRAY_YAW / 2), math.sin(TRAY_YAW / 2)
RATCHET = body("ratchet", pos=(*TRAY_XY, TOP + TRAY_FLOOR), mass=0.28, friction=1.0, quat=(0.0, 0.0, _s, _c),
               com=(0.035, 0.0, 0.032), shapes=_RSH, visual_only=tuple(range(_NRS, len(_RSH))))
# the lever: spawned on tighten (-30 degrees), its arm pointing back from the pivot (away from the handle)
_lv = TRAY_YAW - math.radians(LEVER_DEG)
_pivot = np.array(TRAY_XY) + np.array([[math.cos(TRAY_YAW), -math.sin(TRAY_YAW)],
                                       [math.sin(TRAY_YAW), math.cos(TRAY_YAW)]]) @ np.array(LEVER_AT[:2])
LEVER = body("lever", pos=(*_pivot, TOP + TRAY_FLOOR + LEVER_AT[2]), mass=0.005, friction=0.8,
             quat=(0.0, 0.0, math.sin(_lv / 2), math.cos(_lv / 2)), inertia=(3e-7, 3e-7, 3e-7), shapes=[
                 box((-LEVER_L / 2, 0.0, 3.0 * MM), (LEVER_L, 6.0 * MM, 6.0 * MM), CHROME)])
ratchetkit.add_hinge(Hinge("ratchet", "lever", anchor=LEVER_AT, axis=(0.0, 0.0, 1.0), damping=2e-4,
                           lower=math.radians(-2.0), upper=math.radians(2 * LEVER_DEG + 2.0)))
# the tray: a base, a cradle round the socket and a rest under the handle's end
_hy = lambda x: TRAY_XY[1] - x  # noqa: E731  (world y of a point x along the handle, the handle towards -y)
TRAY = body("tray", pos=(TRAY_XY[0], 0.0, TOP), friction=0.8, shapes=[
    box((0.0, _hy(0.08), TRAY_FLOOR / 2), (0.06, 0.22, TRAY_FLOOR), WOOD),
    box((0.0195, TRAY_XY[1], TRAY_FLOOR + 6.0 * MM), (0.010, 0.030, 12.0 * MM), WOOD),  # the cradle's two cheeks
    box((-0.0195, TRAY_XY[1], TRAY_FLOOR + 6.0 * MM), (0.010, 0.030, 12.0 * MM), WOOD),
    box((0.0, _hy(0.150), (TRAY_FLOOR + HANDLE_Z[0]) / 2 + TRAY_FLOOR / 2),
        (0.030, 0.025, TRAY_FLOOR + HANDLE_Z[0] - TRAY_FLOOR), WOOD),  # the handle's rest
])

OBJECTS = [BRACKET, HEAD, SOLID, TRAY, RATCHET, LEVER]
ratchetkit.register(*OBJECTS)
MODEL = ratchetkit.add_model(RatchetBolt(
    "ratchet", "lever", "bolt_head", BOLT_XY, TOP + HEAD_TOP, HEAD_AF / 2 / math.cos(math.pi / 6), SOCKET_R,
    TOP + SEAT + WASHER_T + HEAD_H / 2, TOP - 0.3, lever_angle=math.radians(LEVER_DEG), depth=HEAD_H,
    solid=SOLID.name))


# -- scoring -------------------------------------------------------------------------------------------------------
SCORE_OVER = "end"
RUN_LIMIT_STEPS = 1600
BROKEN_LOOSE = math.radians(30.0)
CLEAR = 10.0 * MM
STILL = 0.01


def fingers_to(state, spec) -> float:
    """Nearest distance (m) from either gripper's hulls to a body's bounding box (in its own frame). inf offline."""
    pts = ratchetkit.finger_points()
    if not len(pts):
        return math.inf
    lo, hi = spec.bounds_local()
    local = (pts - state.pos) @ state.rotation()
    d = np.maximum(np.maximum(lo - local, local - hi), 0.0)
    return float(np.linalg.norm(d, axis=1).min())


def in_tray(ratchet) -> bool:
    """Lying back in its tray: the socket in the cradle, the handle on its rest, level."""
    off = np.hypot(*(np.asarray(ratchet.pos[:2]) - TRAY_XY))
    yaw = math.atan2(ratchet.rotation()[1, 0], ratchet.rotation()[0, 0])
    return bool(off < 8.0 * MM and abs(ratchet.pos[2] - (TOP + TRAY_FLOOR)) < 4.0 * MM and ratchet.tilt_deg() < 8.0
                and abs(math.remainder(yaw - TRAY_YAW, 2 * math.pi)) < math.radians(15.0))


def _update(objects) -> tuple[float, bool]:
    m = MODEL
    if m.stretched:
        return 0.0, False
    if m.turns() >= 1.0:
        r = objects["ratchet"]
        if in_tray(r) and min(fingers_to(r, RATCHET), fingers_to(objects["lever"], LEVER)) >= CLEAR:
            return 1.0, bool(r.resting(max_speed=STILL))
        return 0.5, False
    if m.turned >= BROKEN_LOOSE:
        return 0.25, False
    return 0.0, False


def success(objects) -> bool:
    """The bolt backed off a full turn and never tightened past 20 degrees; the ratchet back in its tray, released and
    still for 1 s."""
    return _update(objects)[1]


def score(objects) -> float:
    """1.0 / 0.5 / 0.25 / 0 as in the README; a stretched bolt latches 0."""
    return _update(objects)[0]


# -- scripted demo -------------------------------------------------------------------------------------------------
# Open loop, from the fixed layout. The left hand flips the direction lever from tighten to loosen while the ratchet
# lies in its tray; the right hand takes the ratchet by its handle near its balance point, seats the socket on the
# bolt and swings the handle anticlockwise (seen from above) in 50 degree strokes, back-swinging between them as the
# pawl clicks, until the bolt is a full turn out; then it lays the ratchet back in its tray.
RHOME, LHOME = (0.16, -0.22, TOP + 0.15), (0.16, 0.22, TOP + 0.15)
TIP_PAST_TCP = 13.0 * MM
BACK_SAG = (6.0 * MM, 4.0 * MM)  # the arm, holding the ratchet out over the tray, sags this much short of its target
FLIP_TO = float(os.environ.get("FLIP_TO", "0.004"))  # the left finger stops this far past the handle's line
GRIP_X = 50.0 * MM  # where the hand holds the handle, from the socket's axis (the fingertips are 38 mm wide)
GRIP_TCP_Z = HANDLE_Z[0] - 2.0 * MM + TIP_PAST_TCP  # the TCP above the socket's mouth, the fingertips under the handle
HOLD = FINGER_OPEN * 4.0 / 94.0  # closed to 4 mm on the 15 mm handle: a firm grip
STROKE = (-115.0, -65.0)  # the handle's direction (degrees from +x) at the ends of each drive stroke
STROKES = 9
DEMO_MODE = os.environ.get("RATCHET_DEMO", "")  # shortcut checks: "noflip" leaves the lever on tighten and swings


def handle_tcp(mouth, heading_deg: float) -> tuple:
    """The TCP holding the handle, the socket's mouth at ``mouth`` (world), the handle pointing ``heading_deg``."""
    a = math.radians(heading_deg)
    return (mouth[0] + GRIP_X * math.cos(a), mouth[1] + GRIP_X * math.sin(a), mouth[2] + GRIP_TCP_Z)


def _w(dur, pos, finger, label, yaw):
    return Waypoint(dur, tuple(pos), finger, label, yaw)


def _t(w) -> float:
    return sum(p.duration for p in w)


def demo():
    tray_mouth = np.array([*TRAY_XY, TOP + TRAY_FLOOR])
    tray_back = tray_mouth + np.array([*BACK_SAG, 0.0])  # aimed past the arm's sag under the ratchet (measured)
    bolt_on = np.array([*BOLT_XY, TOP + SEAT])  # the socket's mouth pressed down past the washer, over the head
    up = np.array([0.0, 0.0, 1.0])
    heading0 = math.degrees(TRAY_YAW)
    # the fingers close across the handle; a grip across is the same turned half round, and the right wrist reaches
    # it in its own range (the handle pointing -65 to -115 degrees: the hand turned +115 to +65)
    yaw = lambda h: math.radians(h + 180.0)  # noqa: E731
    left = []
    if DEMO_MODE != "noflip":
        # the lever's tip is 7 mm to +x of the handle's line (tighten); pushed across to -x it snaps to loosen
        tip_y = TRAY_XY[1] - LEVER_AT[0] + LEVER_L * 0.9  # the lever's tip along the handle's line (world y)
        z = TOP + TRAY_FLOOR + HEAD_Z[1] + 1.5 * MM + TIP_PAST_TCP
        start = (TRAY_XY[0] + 0.016, tip_y, z)
        left += [
            _w(2.0, (start[0], start[1], z + 0.04), FINGER_CLOSED, "over the lever", -math.pi / 2),
            _w(1.0, start, FINGER_CLOSED, "down beside it", -math.pi / 2),
            _w(1.2, (TRAY_XY[0] - FLIP_TO, tip_y, z), FINGER_CLOSED, "flip it to loosen", -math.pi / 2),
            _w(0.8, (TRAY_XY[0] - FLIP_TO, tip_y, z + 0.04), FINGER_CLOSED, "up", -math.pi / 2),
            _w(1.5, LHOME, FINGER_CLOSED, "home", 0.0),
        ]
    right = [_w(max(0.1, _t(left)), RHOME, FINGER_OPEN, "wait", 0.0)]
    right += [
        _w(2.0, np.array(handle_tcp(tray_mouth, heading0)) + 0.04 * up, FINGER_OPEN, "over the handle", yaw(heading0)),
        _w(1.2, handle_tcp(tray_mouth, heading0), FINGER_OPEN, "round it", yaw(heading0)),
        _w(1.6, handle_tcp(tray_mouth, heading0), HOLD, "grip", yaw(heading0)),
        _w(1.2, handle_tcp(tray_mouth + 0.05 * up, heading0), HOLD, "lift it out", yaw(heading0)),
        _w(2.0, handle_tcp(bolt_on + 0.022 * up, STROKE[0]), HOLD, "over the bolt", yaw(STROKE[0])),
        _w(1.5, handle_tcp(bolt_on - 0.001 * up, STROKE[0]), HOLD, "socket on", yaw(STROKE[0])),
    ]
    # (noflip: the same strokes; on tighten, the pulls free-wheel and the back-swings tighten the bolt)
    for k in range(STROKES):
        a, b = STROKE
        right += [
            _w(1.6, handle_tcp(bolt_on - 0.001 * up, b), HOLD, f"pull {k + 1}", yaw(b)),
            _w(1.0, handle_tcp(bolt_on - 0.001 * up, a), HOLD, f"swing back {k + 1}", yaw(a)),
        ]
    right += [
        _w(1.0, handle_tcp(bolt_on + 0.022 * up, STROKE[0]), HOLD, "lift off", yaw(STROKE[0])),
        _w(2.0, handle_tcp(tray_back + 0.05 * up, heading0), HOLD, "over the tray", yaw(heading0)),
        _w(1.0, handle_tcp(tray_back + 0.025 * up, heading0), HOLD, "steady", yaw(heading0)),
        _w(1.5, handle_tcp(tray_back + 0.001 * up, heading0), HOLD, "lay it in", yaw(heading0)),
        _w(1.6, handle_tcp(tray_mouth + 0.003 * up, heading0), FINGER_OPEN, "let go", yaw(heading0)),
        _w(1.0, np.array(handle_tcp(tray_mouth, heading0)) + 0.05 * up, FINGER_OPEN, "up", yaw(heading0)),
        _w(1.5, RHOME, FINGER_OPEN, "home", 0.0),
    ]
    end = max(_t(left), _t(right)) + 2.5
    left.append(_w(end - _t(left), LHOME, FINGER_CLOSED, "done", 0.0))
    right.append(_w(end - _t(right), RHOME, FINGER_OPEN, "done", 0.0))
    return {"right": right, "left": left}
