"""On the station table: a ball valve's 15 mm compression inlet faces up from its mounting block; a short copper pipe
tail stands in a holder, the olive and the compression nut lie in a parts tray, and a 22 mm open-ended spanner lies
in its tray.

The robot must join the pipe to the valve with the compression fitting, the way a plumber does: the pipe pushed home
into the inlet's socket, the olive dropped over it before the nut (the olive must sit between the nut and the
fitting), the nut run onto the thread and turned 3/4 to 1 1/4 turns past snug with the spanner. A missing or
misplaced olive, or a pipe short of its stop, leaks; turned much further, the olive and the pipe are crushed.

Run it:  inspect-robots run --embodiment yam_arena -E task=compression_joint.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "..." --scorer task_score ...
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fittingkit  # noqa: E402
from fittingkit import CompressionJoint, body, box, cyl, prism12, ring  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

fittingkit.install()
SCENE = "station"
TOP = TABLE_TOP[SCENE]
MM = 0.001
COPPER = (0.78, 0.45, 0.28)
BRASS = (0.80, 0.66, 0.32)
CHROME = (0.80, 0.81, 0.83)
DARK = (0.08, 0.08, 0.09)
RED = (0.72, 0.12, 0.09)
WOOD = (0.55, 0.40, 0.25)
GRAPHITE = (0.20, 0.20, 0.22)  # the pencil mark on the pipe
fittingkit.METALLIC.update({COPPER: 0.3, BRASS: 0.3, CHROME: 0.12})


def hexagon(centre, across_flats, height, color) -> list:
    """An upright hexagonal prism as three boxes turned 60 degrees apart (two flats facing +-x)."""
    side = across_flats / math.sqrt(3)
    return [box(centre, (across_flats, side, height), color, yaw=k * math.pi / 3) for k in range(3)]


# -- the valve, fixed: its compression inlet faces up ----------------------------------------------------------------
AX, AY = 0.30, -0.03  # the joint's vertical axis
BODY_TOP = 30.0 * MM  # the valve body's top: the socket's stop
STUB = BODY_TOP + 12.0 * MM  # the thread stub's top
SOCKET = STUB - BODY_TOP  # the pipe goes 12 mm into the stub
STUB_R = 10.5 * MM


def _valve_shapes() -> tuple[list, list]:
    solid = [
        box((AX, AY, 2.0 * MM), (60.0 * MM, 40.0 * MM, 4.0 * MM), WOOD),  # its mounting block
        box((AX, AY, (4.0 * MM + BODY_TOP) / 2), (26.0 * MM, 26.0 * MM, BODY_TOP - 4.0 * MM), BRASS),  # the body
    ]
    seen = [
        cyl((AX, AY, BODY_TOP + 6.0 * MM), STUB_R, 12.0 * MM, BRASS),  # the threaded stub
        *[cyl((AX, AY, BODY_TOP + (1.5 + 1.8 * k) * MM), STUB_R + 0.3 * MM, 0.5 * MM, DARK) for k in range(6)],
        cyl((AX, AY, STUB + 0.1 * MM), 7.6 * MM, 0.2 * MM, DARK),  # the socket's mouth
        # the snug mark: a red paint line on the body's top, where the nut's paint mark comes at snug
        box((AX + 12.0 * MM, AY, BODY_TOP + 0.05 * MM), (2.0 * MM, 3.0 * MM, 0.1 * MM), RED),
        box((AX + 13.05 * MM, AY, BODY_TOP - 3.0 * MM), (0.1 * MM, 3.0 * MM, 6.0 * MM), RED),
        cyl((AX + 0.020, AY, 17.0 * MM), STUB_R, 14.0 * MM, BRASS, axis="X"),  # the outlet
        box((AX, AY - 0.016, 17.0 * MM), (6.0 * MM, 6.0 * MM, 6.0 * MM), CHROME),  # the spindle
        box((AX, AY - 0.020, 17.0 * MM + 12.0 * MM), (8.0 * MM, 4.0 * MM, 30.0 * MM), RED),  # the lever
    ]
    return solid, seen


_VS, _VV = _valve_shapes()
VALVE = body("valve", pos=(0.0, 0.0, TOP), friction=0.6, shapes=_VS + _VV,
             visual_only=tuple(range(len(_VS), len(_VS) + len(_VV))))

# -- the pipe tail, the olive and the nut ----------------------------------------------------------------------------
PIPE_R, PIPE_L = 7.5 * MM, 60.0 * MM
PIPE_XY = (0.20, 0.17)  # standing in its holder
NUT_AF, NUT_LEN = 22.0 * MM, 12.0 * MM
OLIVE_LEN = 7.0 * MM
TRAY_TOP = 10.0 * MM
OLIVE_XY, NUT_XY = (0.27, 0.06), (0.27, 0.11)  # clear of the holder for the open fingers


def _pipe_shapes() -> tuple[list, list]:
    """In the pipe's frame: origin at its bottom end, z up its axis."""
    solid = prism12((0.0, 0.0, PIPE_L / 2), PIPE_R, PIPE_L, COPPER)
    seen = [cyl((0.0, 0.0, PIPE_L / 2), PIPE_R + 0.15 * MM, PIPE_L, COPPER),
            cyl((0.0, 0.0, SOCKET), PIPE_R + 0.25 * MM, 0.6 * MM, GRAPHITE),  # the insertion mark
            cyl((0.0, 0.0, PIPE_L + 0.1 * MM), PIPE_R - 0.8 * MM, 0.2 * MM, DARK)]  # its bore, seen from above
    return solid, seen


def _olive_shapes() -> tuple[list, list]:
    solid = ring(-OLIVE_LEN / 2, OLIVE_LEN / 2, 8.8 * MM, 9.6 * MM, BRASS, segments=12)  # its bore slides freely
    seen = [cyl((0.0, 0.0, 0.0), 8.9 * MM, OLIVE_LEN - 2.0 * MM, BRASS)]
    return solid, seen


def _nut_shapes() -> tuple[list, list]:
    """Its upper half is solid (the bore round the pipe); the lower half, its threaded mouth, is open (a thin rim at
    its bottom face, wide enough for the olive to pass inside)."""
    solid = [*ring(0.0, NUT_LEN / 2, 9.0 * MM, NUT_AF / 2, BRASS, segments=12),
             *ring(-NUT_LEN / 2, -NUT_LEN / 2 + 1.0 * MM, 10.2 * MM, NUT_AF / 2, BRASS, segments=12)]
    seen = [*hexagon((0.0, 0.0, 0.0), NUT_AF, NUT_LEN, BRASS),
            cyl((0.0, 0.0, NUT_LEN / 2 + 0.05 * MM), 8.0 * MM, 0.1 * MM, DARK),  # its bore, seen from above
            box((NUT_AF / 2 + 0.05 * MM, 0.0, 0.0), (0.1 * MM, 3.0 * MM, NUT_LEN), RED),  # a paint mark on a flat
            box((9.6 * MM, 0.0, NUT_LEN / 2 + 0.05 * MM), (2.6 * MM, 2.0 * MM, 0.1 * MM), RED)]  # and on its top
    return solid, seen


_PS, _PV = _pipe_shapes()
PIPE = body("pipe", pos=(*PIPE_XY, TOP + 1.0 * MM), mass=0.05, friction=0.4, com=(0.0, 0.0, PIPE_L / 2),
            shapes=_PS + _PV, visual_only=tuple(range(len(_PS), len(_PS) + len(_PV))), solver_iterations=32)
_OS, _OV = _olive_shapes()
OLIVE = body("olive", pos=(*OLIVE_XY, TOP + TRAY_TOP + OLIVE_LEN / 2), mass=0.03, friction=0.6, shapes=_OS + _OV,
             visual_only=tuple(range(len(_OS), len(_OS) + len(_OV))), solver_iterations=32)
_NS, _NV = _nut_shapes()
NUT = body("nut", pos=(*NUT_XY, TOP + TRAY_TOP + NUT_LEN / 2), mass=0.05, friction=0.6, shapes=_NS + _NV,
           visual_only=tuple(range(len(_NS), len(_NS) + len(_NV))), solver_iterations=32)


def _parts_shapes() -> list:
    """The pipe's holder (a block with a hole) and the parts tray (a raised board with low lips)."""
    hx, hy = PIPE_XY
    hole, w, h = 19.0 * MM, 44.0 * MM, 25.0 * MM
    side = (w - hole) / 2
    out = [box((hx + dx, hy + dy, h / 2), (sx, sy, h), WOOD)
           for dx, dy, sx, sy in [(0.0, (hole + side) / 2, w, side), (0.0, -(hole + side) / 2, w, side),
                                  ((hole + side) / 2, 0.0, side, hole), (-(hole + side) / 2, 0.0, side, hole)]]
    tx, ty = OLIVE_XY[0], (OLIVE_XY[1] + NUT_XY[1]) / 2
    out += [box((tx, ty, TRAY_TOP / 2), (60.0 * MM, 110.0 * MM, TRAY_TOP), WOOD),
            box((tx + 32.0 * MM, ty, TRAY_TOP + 1.0 * MM), (4.0 * MM, 110.0 * MM, 2.0 * MM), WOOD),
            box((tx - 32.0 * MM, ty, TRAY_TOP + 1.0 * MM), (4.0 * MM, 110.0 * MM, 2.0 * MM), WOOD)]
    return out


PARTS = body("parts", pos=(0.0, 0.0, TOP), friction=0.6, shapes=_parts_shapes())

# -- the spanner in its tray ---------------------------------------------------------------------------------------
# Its frame: origin at the middle of its jaw, x along its handle, z up.
SPAN_T = 5.0 * MM
SPAN_JAW = (0.27, -0.17)  # the jaw's middle as it lies in its tray, the handle pointing to -x
SPAN_TRAY_TOP = 6.0 * MM


def _spanner_shapes() -> tuple[list, list]:
    solid = [box((0.0725, 0.0, 0.0), (0.105, 14.0 * MM, SPAN_T), CHROME),  # the handle
             box((0.022, 0.0, 0.0), (16.0 * MM, 34.0 * MM, SPAN_T), CHROME)]  # the head behind the jaw
    seen = [box((0.0, y, 0.0), (28.0 * MM, 6.0 * MM, SPAN_T), CHROME) for y in (-14.0 * MM, 14.0 * MM)]  # the jaws
    seen.append(box((0.07, 0.0, SPAN_T / 2 + 0.05 * MM), (20.0 * MM, 8.0 * MM, 0.1 * MM), DARK))  # "22" stamped
    return solid, seen


_SS, _SV = _spanner_shapes()
SPANNER = body("spanner", pos=(*SPAN_JAW, TOP + SPAN_TRAY_TOP + SPAN_T / 2), mass=0.10, com=(0.05, 0.0, 0.0), yaw=math.pi,
               friction=1.0, shapes=_SS + _SV, visual_only=tuple(range(len(_SS), len(_SS) + len(_SV))),
               solver_iterations=32)
SPAN_TRAY = body("spanner_tray", pos=(0.0, 0.0, TOP), friction=0.6, shapes=[
    box((SPAN_JAW[0] - 0.05, SPAN_JAW[1], SPAN_TRAY_TOP / 2), (0.17, 50.0 * MM, SPAN_TRAY_TOP), WOOD)])  # a flat board

OBJECTS = [VALVE, PIPE, OLIVE, NUT, PARTS, SPANNER, SPAN_TRAY]
fittingkit.register(*OBJECTS)
MODEL = fittingkit.add_model(CompressionJoint("pipe", "olive", "nut", "spanner", AX, AY, TOP + STUB, socket=SOCKET,
                                            nut_len=NUT_LEN, olive_len=OLIVE_LEN, pipe_len=PIPE_L))


# -- scoring -------------------------------------------------------------------------------------------------------
SCORE_OVER = "end"
RUN_LIMIT_STEPS = 2400
CLEAR = 10.0 * MM
STILL = 0.01


def fingers_to(state, spec) -> float:
    """Nearest distance (m) from either gripper's hulls to a body's bounding box (in its own frame). inf offline."""
    pts = fittingkit.finger_points()
    if not len(pts):
        return math.inf
    lo, hi = spec.bounds_local()
    local = (pts - state.pos) @ state.rotation()
    d = np.maximum(np.maximum(lo - local, local - hi), 0.0)
    return float(np.linalg.norm(d, axis=1).min())


def in_tray(spanner) -> bool:
    """Lying back in its tray: its jaw within 10 mm of its place, flat, the handle along the tray within 15 degrees."""
    yaw = math.atan2(spanner.rotation()[1, 0], spanner.rotation()[0, 0])
    return bool(np.hypot(*(np.asarray(spanner.pos[:2]) - SPAN_JAW)) < 10.0 * MM
                and spanner.pos[2] < TOP + SPAN_TRAY_TOP + SPAN_T + 3.0 * MM and spanner.tilt_deg() < 10.0
                and abs(math.remainder(yaw - math.pi, 2 * math.pi)) < math.radians(15.0))


OVER = 1.25  # turns past snug: further than this the olive is over-compressed (backing off does not undo it)
STILL_STEPS = 20  # 1 s at the 20 Hz control rate
_STATE: dict = {}


@fittingkit.on_reset
def _reset_state() -> None:
    _STATE.update(still=0)


_reset_state()


def _update(objects) -> tuple[float, bool]:
    """(tier, whether the full-score state holds this step). The full score also needs it held for 1 s. The olive's
    compression is the furthest the nut has been turned past snug, not where it ends: backing off does not undo it."""
    m = MODEL
    if m.crushed or m.furthest > OVER:
        return 0.0, False
    if m.sealed():
        sp = objects["spanner"]
        clear = min(fingers_to(objects[n], spec) for n, spec in (("spanner", SPANNER), ("nut", NUT), ("pipe", PIPE)))
        if in_tray(sp) and clear >= CLEAR:
            return 1.0, bool(sp.resting(max_speed=STILL) and objects["pipe"].resting(max_speed=STILL))
        return 0.5, False
    if m.threaded and m.olive_in and m.pipe_home() and m.turns_past() < 0.75:
        return 0.25, False
    return 0.0, False


def success(objects) -> bool:
    """The joint made up: the pipe home, the olive under the nut, the nut 3/4 to 1 1/4 turns past snug (and never
    further); the spanner back in its tray, released and still (held 1 s by the run). It also counts the steps the
    full-score state has held, for score()."""
    ok = _update(objects)[1]
    _STATE["still"] = _STATE["still"] + 1 if ok else 0
    return ok


def score(objects) -> float:
    """1.0 / 0.5 / 0.25 / 0 as in the README; past 1 1/4 turns 0, past 1 1/2 latched (the olive crushed). 1.0 only once
    the full-score state has held for 1 s (counted by success(), which the run calls first every step); until then
    0.5."""
    tier = _update(objects)[0]
    if tier == 1.0 and _STATE["still"] < STILL_STEPS:
        return 0.5
    return tier


# -- scripted demo -------------------------------------------------------------------------------------------------
# Open loop, from the fixed layout. The left hand stands the pipe tail in the valve's inlet, then drops the olive and
# then the nut over its top; the right hand meanwhile takes the spanner from its tray, and once the nut is down, puts
# the jaw on its flats and swings the handle clockwise in 60 degree strokes, sliding it off, back and on again
# between them, until the nut is about a turn past snug; then it lays the spanner back in its tray.
RHOME, LHOME = (0.16, -0.22, TOP + 0.15), (0.16, 0.22, TOP + 0.15)
TIP_PAST_TCP = 13.0 * MM
LYAW = math.radians(-90.0)  # the left fingers closing along x
PIPE_GRIP = 45.0 * MM  # the fingertips this far up the pipe from its bottom
HOLD_PIPE = FINGER_OPEN * 9.0 / 94.0
HOLD_OLIVE = FINGER_OPEN * 13.0 / 94.0
HOLD_NUT = FINGER_OPEN * 16.0 / 94.0
HOLD_SPANNER = FINGER_OPEN * 9.0 / 94.0  # across the 14 mm handle
SPAN_R = 70.0 * MM  # the TCP along the handle from the jaw
SPAN_GRIP_Z = -SPAN_T / 2 + 1.5 * MM + TIP_PAST_TCP  # the TCP above the jaw's middle as it is gripped in its tray
SPAN_SAG = 6.0 * MM  # the jaw end droops in the grip and the arm sags: the hand aims this much higher (measured)
H0, H1 = -60.0, -120.0  # each stroke swings the handle's heading (degrees, seen from above) from H0 to H1: clockwise
STROKE_PAST = float(os.environ.get("STROKE_PAST", "2.0"))  # degrees past H1: the arm lags
STROKES = int(os.environ.get("STROKES", "12"))
OFF = 15.0 * MM  # how far the spanner slides off along its handle between strokes
DEMO_MODE = os.environ.get("JOINT_DEMO", "")  # shortcut checks: "noolive", "swap" (nut first), "over"


def _w(dur, pos, finger, label, yaw):
    return Waypoint(dur, tuple(float(v) for v in pos), finger, label, yaw)


def _t(w) -> float:
    return sum(p.duration for p in w)


def span_tcp(heading_deg, r=SPAN_R, jaw_z=None):
    """The TCP holding the spanner, its jaw on the nut's axis at ``jaw_z``, its handle pointing ``heading_deg``."""
    h = math.radians(heading_deg)
    z = (TOP + STUB + 5.0 * MM if jaw_z is None else jaw_z) + SPAN_GRIP_Z + SPAN_SAG
    return (AX + r * math.cos(h), AY + r * math.sin(h), z)


def span_yaw(heading_deg):
    return math.radians(heading_deg + 180.0)  # the fingers across the handle (the right wrist's own range)


def _drop_ring(xy, centre_z, length, hold, label):
    """Take a ring from the tray and drop it over the pipe's top (it falls down the pipe)."""
    tcp_dz = TIP_PAST_TCP  # fingertips at the ring's middle
    top = TOP + BODY_TOP + PIPE_L  # the pipe's top, home in the socket
    over = top + length / 2 + 5.0 * MM  # the ring's middle above the top
    on = top + length / 2 - 4.0 * MM  # the top 4 mm into its bore
    return [
        _w(1.8, (*xy, centre_z + tcp_dz + 0.04), FINGER_OPEN, f"over the {label}", LYAW),
        _w(1.2, (*xy, centre_z + tcp_dz), FINGER_OPEN, f"round the {label}", LYAW),
        _w(1.6, (*xy, centre_z + tcp_dz), hold, f"grip the {label}", LYAW),
        _w(1.2, (*xy, centre_z + tcp_dz + 0.04), hold, f"lift the {label}", LYAW),
        _w(2.0, (AX, AY, over + tcp_dz + 0.02), hold, f"{label} over the pipe", LYAW),
        _w(1.0, (AX, AY, over + tcp_dz), hold, "steady", LYAW),
        _w(1.2, (AX, AY, on + tcp_dz), hold, f"{label} onto its top", LYAW),
        _w(1.6, (AX, AY, on + tcp_dz), FINGER_OPEN, f"drop the {label}", LYAW),
        _w(1.0, (AX, AY, on + tcp_dz + 0.03), FINGER_OPEN, "up", LYAW),
    ]


def demo():
    left, right = [], []
    up = 0.04
    # the left hand stands the pipe tail in the valve's inlet
    pz0 = TOP + 1.0 * MM + PIPE_GRIP + TIP_PAST_TCP
    pz_home = TOP + BODY_TOP + PIPE_GRIP + TIP_PAST_TCP
    left += [
        _w(1.8, (*PIPE_XY, pz0 + up), FINGER_OPEN, "over the pipe", LYAW),
        _w(1.2, (*PIPE_XY, pz0), FINGER_OPEN, "round the pipe", LYAW),
        _w(1.6, (*PIPE_XY, pz0), HOLD_PIPE, "grip the pipe", LYAW),
        _w(1.4, (*PIPE_XY, pz_home + 0.025), HOLD_PIPE, "lift it out", LYAW),
        _w(2.0, (AX, AY, pz_home + 0.025), HOLD_PIPE, "pipe over the inlet", LYAW),
        _w(1.0, (AX, AY, pz_home + 0.016), HOLD_PIPE, "steady", LYAW),
        _w(1.6, (AX, AY, pz_home - 1.0 * MM), HOLD_PIPE, "push it home", LYAW),
        _w(1.6, (AX, AY, pz_home - 1.0 * MM), FINGER_OPEN, "let go", LYAW),
        _w(1.0, (AX, AY, pz_home + 0.04), FINGER_OPEN, "up", LYAW),
    ]
    rings = [(OLIVE_XY, TOP + TRAY_TOP + OLIVE_LEN / 2, OLIVE_LEN, HOLD_OLIVE, "olive"),
             (NUT_XY, TOP + TRAY_TOP + NUT_LEN / 2, NUT_LEN, HOLD_NUT, "nut")]
    if DEMO_MODE == "swap":
        rings = rings[::-1]
    if DEMO_MODE == "noolive":
        rings = rings[1:]
    for xy, cz, length, hold, label in rings:
        left += _drop_ring(xy, cz, length, hold, label)
    left.append(_w(1.5, LHOME, FINGER_OPEN, "home", 0.0))
    # the right hand takes the spanner from its tray and waits beside the valve
    tray = (SPAN_JAW[0] - SPAN_R, SPAN_JAW[1], TOP + SPAN_TRAY_TOP + 1.5 * MM + TIP_PAST_TCP)  # fingertips off the board
    # (at the tray the wrist is at 0: a grip across the handle is the same turned half round, and 0 is in its range)
    right += [
        _w(2.0, (tray[0], tray[1], tray[2] + up), FINGER_OPEN, "over the spanner", 0.0),
        _w(1.2, tray, FINGER_OPEN, "round its handle", 0.0),
        _w(1.6, tray, HOLD_SPANNER, "grip the spanner", 0.0),
        _w(1.2, (tray[0], tray[1], tray[2] + up), HOLD_SPANNER, "lift it", 0.0),
        _w(1.5, (0.18, -0.20, TOP + 0.08), HOLD_SPANNER, "back clear", 0.0),
    ]
    # it waits clear of the valve until the left hand has gone home (the two grippers are ~10 cm across)
    right.append(_w(max(0.1, _t(left) - _t(right)), (0.18, -0.20, TOP + 0.08), HOLD_SPANNER, "wait", 0.0))
    right += [
        _w(2.5, np.array(span_tcp(H0, SPAN_R + OFF)) + [0, 0, up], HOLD_SPANNER, "beside the valve", span_yaw(H0)),
        _w(1.0, span_tcp(H0, SPAN_R + OFF), HOLD_SPANNER, "down level with the nut", span_yaw(H0)),
    ]
    strokes = STROKES + (6 if DEMO_MODE == "over" else 0)
    for k in range(strokes):
        right += [
            _w(0.8, span_tcp(H0), HOLD_SPANNER, f"onto the nut {k + 1}", span_yaw(H0)),
            _w(1.4, span_tcp(H1 - STROKE_PAST), HOLD_SPANNER, f"stroke {k + 1}", span_yaw(H1 - STROKE_PAST)),
            _w(0.6, span_tcp(H1 - STROKE_PAST, SPAN_R + OFF), HOLD_SPANNER, f"slide off {k + 1}",
               span_yaw(H1 - STROKE_PAST)),
            _w(1.0, span_tcp(H0, SPAN_R + OFF), HOLD_SPANNER, f"back {k + 1}", span_yaw(H0)),
        ]
    right += [
        _w(1.0, np.array(span_tcp(H0, SPAN_R + OFF)) + [0, 0, up], HOLD_SPANNER, "lift it", span_yaw(H0)),
        _w(2.5, (tray[0], tray[1], tray[2] + up), HOLD_SPANNER, "over its tray", 0.0),
        _w(1.2, tray, HOLD_SPANNER, "lay it in", 0.0),
        _w(1.6, tray, FINGER_OPEN, "let go", 0.0),
        _w(1.0, (tray[0], tray[1], tray[2] + up), FINGER_OPEN, "up", 0.0),
        _w(1.5, RHOME, FINGER_OPEN, "home", 0.0),
    ]
    end = max(_t(left), _t(right)) + 2.5
    left.append(_w(end - _t(left), LHOME, FINGER_OPEN, "done", 0.0))
    right.append(_w(end - _t(right), RHOME, FINGER_OPEN, "done", 0.0))
    return {"left": left, "right": right}
