"""On the station table: a small machinist's vise on a rubber mat, on the midline 24 cm in front of the arm bases, its
screw handle towards the robot, clamps an aluminium block (60 x 60 x 40 mm) whose top is 85 mm above the table. An M10
hand tap stands in a hole drilled in the block's top, started square, with a few threads cut. A bar tap wrench clamps
the tap's square end: a round steel centre with two straight round handles with knurled grips, 280 mm across,
125 mm above the table, pointing back-left and front-right. A white paint mark on the back-left handle's end and a
felt-pen line on the block's top show how far it has turned.

Run it:  inspect-robots run --embodiment yam_arena -E task=tap_wrench.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "Start the tap: turn it a quarter turn in with both hands on the wrench, then back it off at
           least a tenth of a turn to break the chip, and let go." --scorer task_score ...
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tapkit  # noqa: E402
from tapkit import Thread, body, box, cyl  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

tapkit.install()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
MM = 0.001

# -- the vise and the block (fixed) ---------------------------------------------------------------------------------
# A small machinist's vise (75 mm jaws) on a rubber mat, its screw and T-bar handle towards the robot; the block is
# clamped in its jaws. Vise frame: origin on the table under the block's centre, x forward, y left.
# WHY 24 cm out (2026-11-07, offline IK maps): both arms, each on its own side, can hold opposite handles 120 mm out,
# 125 mm up, and turn them together from 155 to 25 degrees (the left-held end's direction); the demo uses 140 to 40.
TAP_XY = (0.24, 0.0)
CAST_IRON = (0.24, 0.31, 0.40)
MACHINED = (0.64, 0.65, 0.67)
CHROME = (0.82, 0.82, 0.84)
RUBBER_MAT = (0.12, 0.12, 0.13)
ALU = (0.76, 0.77, 0.79)
INK = (0.08, 0.08, 0.10)
WHITE = (0.95, 0.95, 0.93)
SLIDE_TOP = 45.0 * MM  # the block sits on the vise's slide
BLOCK = (60.0 * MM, 60.0 * MM, 40.0 * MM)
BLOCK_TOP = SLIDE_TOP + BLOCK[2]  # 85 mm above the table
JAW_TOP = 70.0 * MM  # the jaws hold the block's lower 25 mm


def _vise() -> tuple[list, int]:
    """(shapes, how many collide): the main castings collide; the jaw plates, screw, handle and mat are seen only."""
    hb = BLOCK[0] / 2
    jaw_h = JAW_TOP - SLIDE_TOP
    collide = [
        box((-5.0 * MM, 0, 7.5 * MM), (120.0 * MM, 90.0 * MM, 15.0 * MM), CAST_IRON),  # base
        box((-10.0 * MM, 0, 30.0 * MM), (180.0 * MM, 50.0 * MM, 30.0 * MM), CAST_IRON),  # slide
        box((hb + 12.5 * MM, 0, SLIDE_TOP + jaw_h / 2), (25.0 * MM, 90.0 * MM, jaw_h), CAST_IRON),  # fixed jaw
        box((-hb - 12.5 * MM, 0, SLIDE_TOP + jaw_h / 2), (25.0 * MM, 90.0 * MM, jaw_h), CAST_IRON),  # moving jaw
    ]
    look = [
        box((hb + 1.5 * MM, 0, SLIDE_TOP + jaw_h / 2), (3.0 * MM, 76.0 * MM, jaw_h), MACHINED),  # jaw plates
        box((-hb - 1.5 * MM, 0, SLIDE_TOP + jaw_h / 2), (3.0 * MM, 76.0 * MM, jaw_h), MACHINED),
        cyl((-hb - 25.0 * MM - 32.0 * MM, 0, 35.0 * MM), 8.0 * MM, 64.0 * MM, MACHINED, axis="X"),  # screw
        cyl((-hb - 25.0 * MM - 64.0 * MM, 0, 35.0 * MM), 5.0 * MM, 120.0 * MM, CHROME, axis="Y"),  # T-bar handle
        cyl((-hb - 25.0 * MM - 64.0 * MM, +60.0 * MM, 35.0 * MM), 8.0 * MM, 10.0 * MM, CHROME, axis="Y"),
        cyl((-hb - 25.0 * MM - 64.0 * MM, -60.0 * MM, 35.0 * MM), 8.0 * MM, 10.0 * MM, CHROME, axis="Y"),
        box((-20.0 * MM, 0, 0.75 * MM), (280.0 * MM, 170.0 * MM, 1.5 * MM), RUBBER_MAT),  # mat
    ]
    return collide + look, len(collide)


_vise_shapes, _vise_n = _vise()
VISE = body("vise", pos=(*TAP_XY, TOP), shapes=_vise_shapes, visual_only=tuple(range(_vise_n, len(_vise_shapes))),
            friction=0.6)


def _block() -> list:
    """The block (it collides) with the drilled hole round the tap and a felt-pen line pointing to where the paint
    mark starts (seen only). Block frame: origin on the table under its centre."""
    t = math.radians(START_YAW)
    r0, r1 = 16.0 * MM, 28.0 * MM
    rm = (r0 + r1) / 2
    return [
        box((0, 0, SLIDE_TOP + BLOCK[2] / 2), BLOCK, ALU),
        cyl((0, 0, BLOCK_TOP + 0.1 * MM), 6.5 * MM, 0.2 * MM, INK),  # the hole's dark ring round the tap
        box((rm * math.cos(t), rm * math.sin(t), BLOCK_TOP + 0.1 * MM), (r1 - r0, 1.0 * MM, 0.2 * MM), INK, yaw=t),
    ]


START_YAW = 140.0  # deg: the handle with the paint mark (body +x) points back-left
BLOCK_BODY = body("block", pos=(*TAP_XY, TOP), shapes=_block(), visual_only=(1, 2), friction=0.6)

# -- the tap and its wrench (one body; frame: origin on the tap's axis at the block's top, z up the axis, handles
#    along x) -----------------------------------------------------------------------------------------------------
STEEL = (0.70, 0.71, 0.73)
DARK = (0.30, 0.30, 0.32)
# WHY square bars inside round handles (2026-11-07): the handles collide as 9 mm square bars (PhysX custom cylinders
# miss thin contacts) inside round 12.8 mm handles that are only seen; the bars' corners stay inside the round.
BAR = 9.0 * MM
HANDLE_R = 6.4 * MM
BAR_UP = 40.0 * MM  # the handles' centre above the block's top: 125 mm above the table
# WHY a round centre and polished-steel friction (2026-11-07): a square centre can be clamped across its flats and
# turned with one wrist like a spanner, with no side load, sidestepping the two-handed push. Round (28 mm across, as on
# real adjustable tap wrenches) and at friction 0.35, a full 50 N squeeze can twist it with at most about 0.5 N m,
# under the 0.9 N m the tap needs.
CENTRE_R, CENTRE_H = 14.0 * MM, 20.0 * MM
# WHY 280 mm (2026-11-07, sim trace): a 180 mm wrench put the hands 160 mm apart and the two wrists met above the tap
# with both handles sideways. 280 mm is the usual length of a size-2 bar tap wrench (M5 to M12).
SPAN = 280.0 * MM  # handle end to handle end
KNURL = (80.0 * MM, 140.0 * MM)  # the knurled grip sleeves, from and to this far out
WRENCH_MASS = 0.15  # kg: wrench 0.12, tap 0.03
# WHY these inertias (2026-11-07): the thread's stiff springs act explicitly each substep, so the wrench's real moments
# (about 0.0004 kg m2) would make them unstable; 0.01 about the axis (stick-slip spring 100 N m/rad) and 0.002 for
# tilt keep each spring well inside the stable limit. At the demo's turning rate the extra inertia costs about
# 0.03 N m. (At 0.1 kg m2 it lagged the wrench 9 degrees behind the hands.)
SPIN_INERTIA = 0.01
TILT_INERTIA = 2.0e-3


BRIGHT = (0.86, 0.86, 0.88)  # ground high-speed steel
FLUTE = (0.18, 0.18, 0.20)
THREAD_SHOWN = 16.0 * MM  # the tap's cutting part still above the block (a few threads are in the hole)
TAP_PITCH = 1.5 * MM


def _tap_look() -> list:
    """The M10 tap as seen (it is part of the wrench's body, so it turns with it): the cutting part above the hole,
    four lands of thread crests 1.5 mm apart with dark flutes between them, then the 8 mm shank up into the wrench,
    and the 6.3 mm square drive showing above the wrench's centre. Threads are drawn as rings, not a helix."""
    out = [cyl((0, 0, THREAD_SHOWN / 2), 3.6 * MM, THREAD_SHOWN, FLUTE)]  # the core, seen in the flutes
    n = int(THREAD_SHOWN / TAP_PITCH)
    for k in range(n):
        z = (k + 0.5) * TAP_PITCH
        for j in range(4):
            t = math.radians(45.0 + 90.0 * j)
            out.append(box((4.3 * MM * math.cos(t), 4.3 * MM * math.sin(t), z), (1.4 * MM, 3.8 * MM, 0.8 * MM), BRIGHT,
                           yaw=t))
    top = BAR_UP - CENTRE_H / 2
    out.append(cyl((0, 0, (THREAD_SHOWN + top + 1.0 * MM) / 2), 4.0 * MM, top + 1.0 * MM - THREAD_SHOWN, BRIGHT))
    out.append(box((0, 0, BAR_UP + CENTRE_H / 2 + 2.5 * MM), (6.3 * MM, 6.3 * MM, 5.0 * MM), BRIGHT))  # square drive
    return out


def _wrench_shapes() -> tuple[list, int]:
    hl = SPAN / 2 - CENTRE_R
    collide = [
        cyl((0, 0, BAR_UP), CENTRE_R, CENTRE_H, STEEL),
        box((+(CENTRE_R + hl / 2), 0, BAR_UP), (hl, BAR, BAR), STEEL),
        box((-(CENTRE_R + hl / 2), 0, BAR_UP), (hl, BAR, BAR), STEEL),
    ]
    kl = KNURL[1] - KNURL[0]
    km = (KNURL[0] + KNURL[1]) / 2
    look = [
        cyl((+(CENTRE_R + hl / 2), 0, BAR_UP), HANDLE_R, hl, STEEL, axis="X"),
        cyl((-(CENTRE_R + hl / 2), 0, BAR_UP), HANDLE_R, hl, STEEL, axis="X"),
        cyl((+km, 0, BAR_UP), HANDLE_R + 0.6 * MM, kl, DARK, axis="X"),  # knurled sleeves
        cyl((-km, 0, BAR_UP), HANDLE_R + 0.6 * MM, kl, DARK, axis="X"),
        cyl((0, +CENTRE_R + 4.0 * MM, BAR_UP), 4.0 * MM, 8.0 * MM, STEEL, axis="Y"),  # the jaw screw's boss
        *_tap_look(),
        cyl((SPAN / 2 - 6.0 * MM, 0, BAR_UP), HANDLE_R + 0.9 * MM, 6.0 * MM, WHITE, axis="X"),  # paint mark
    ]
    return collide + look, len(collide)


_wrench, _wrench_n = _wrench_shapes()
WRENCH = body("wrench", pos=(*TAP_XY, TOP + BLOCK_TOP), yaw=math.radians(START_YAW), shapes=_wrench,
              visual_only=tuple(range(_wrench_n, len(_wrench))), mass=WRENCH_MASS, com=(0.0, 0.0, BAR_UP),
              friction=0.35, contact_offset=1.0 * MM, solver_iterations=32,
              inertia=(TILT_INERTIA, TILT_INERTIA, SPIN_INERTIA))

OBJECTS = [VISE, BLOCK_BODY, WRENCH]
tapkit.register(*OBJECTS)

# The thread: M10 x 1.5. WHY 0.9 N m to cut (2026-11-07, sim trace): the simulated arms are soft, about 0.3 N/mm at
# the gripper, so the push per hand is kept small: 0.9 N m with the hands 120 mm out is 3.75 N each. Turning by one
# handle alone loads the tap sideways by its whole push, 7.5 N, over the 5 N limit. A real tap starting a thread in
# aluminium takes about this much.
TORQUE_IN = float(os.environ.get("TAP_TORQUE", "0.9"))
SIDE_LIMIT = 5.0
THREAD = tapkit.add_model(Thread("wrench", origin=(*TAP_XY, TOP + BLOCK_TOP),
                                 pitch=1.5 * MM, torque_in=TORQUE_IN, torque_out=0.3, side_limit=SIDE_LIMIT,
                                 tilt_limit=2.0, axial_limit=60.0))


# -- scoring -------------------------------------------------------------------------------------------------------
# 1.0   the tap turned a quarter turn in (90 degrees clockwise seen from above) and then backed off at least a tenth
#       of a turn (36 degrees) from the furthest it went in; no part of either gripper within 10 mm of the wrench;
#       the wrench still (under 1 cm/s and 0.5 rad/s) for 1 s
# 0.5   a quarter turn in, not backed off as for 1.0
# 0.25  a tenth of a turn in (36 degrees)
# 0     anything else. Also 0, latched for the rest of the trial: the tap snapped (pushed sideways more than 5 N,
#       smoothed over 0.1 s, or tilted more than 2 degrees), or its thread stripped (pushed or pulled along its axis
#       more than 60 N). Backing off counts only after the quarter turn in.
SCORE_OVER = "end"
RUN_LIMIT_STEPS = 1200
IN_TURNS = 0.25
BACK_TURNS = 0.10
START_TURNS = 0.10
CLEAR = 10.0 * MM

_STATE: dict = {}


@tapkit.on_reset
def _reset_state() -> None:
    _STATE.update(violation=None, deepest=0.0, quarter=False, backed=False)


_reset_state()


def fingers_to(state, spec) -> float:
    """Nearest distance (m) from either gripper's hulls to a body's bounding box (in its own frame). inf offline."""
    pts = tapkit.finger_points()
    if not len(pts):
        return math.inf
    lo, hi = spec.bounds_local()
    local = (pts - state.pos) @ state.rotation()
    d = np.maximum(np.maximum(lo - local, local - hi), 0.0)
    return float(np.linalg.norm(d, axis=1).min())


def _update(objects) -> tuple[float, bool]:
    if _STATE["violation"] is None and THREAD.snapped:
        _STATE["violation"] = THREAD.snapped
    turns = THREAD.turns
    _STATE["deepest"] = max(_STATE["deepest"], turns)
    _STATE["quarter"] |= _STATE["deepest"] >= IN_TURNS
    _STATE["backed"] |= _STATE["quarter"] and _STATE["deepest"] - turns >= BACK_TURNS
    if _STATE["violation"]:
        return 0.0, False
    wrench = objects["wrench"]
    if _STATE["backed"] and fingers_to(wrench, WRENCH) >= CLEAR:
        return 1.0, bool(wrench.resting())
    if _STATE["quarter"]:
        return 0.5, False
    return (0.25 if _STATE["deepest"] >= START_TURNS else 0.0), False


def success(objects) -> bool:
    """A quarter turn in, then backed off a tenth, the tap never side-loaded or tilted past its limits, both grippers
    off the wrench, still for 1 s."""
    return _update(objects)[1]


def score(objects) -> float:
    """1.0 / 0.5 / 0.25 / 0 as in the table above; a snapped tap latches 0."""
    return _update(objects)[0]


# -- scripted demo -------------------------------------------------------------------------------------------------
# Both arms hold opposite handles 120 mm out, the left the back-left one, the right the front-right one, each with its
# gripper heading towards the tap (fingers across the bar), and turn the wrench together 100 degrees in, ease off,
# then 50 degrees back, ease off, let go. Angles: a = the left-held handle's direction from the tap (deg; 0 forward,
# 90 left); the right holds a - 180.
R_GRIP = 120.0 * MM
BAR_Z0 = TOP + BLOCK_TOP + BAR_UP
PITCH = 1.5 * MM
A_START, A_IN, A_BACK = START_YAW, START_YAW - 100.0, START_YAW - 100.0 + 50.0
STEP = 25.0  # deg between waypoints: a 25-degree chord cuts 3.4 mm inside the circle, the same at both handles
TURN_RATE = 35.0  # deg/s
# WHY a gentle grip (2026-11-07 design and trace): two position-controlled arms clamped to one rigid wrench fight
# through their tracking errors; squeezed with about 2.5 N a finger (0.5 mm short of the bar's width, servo 5000 N/m),
# the bar slides in the grip instead of pushing the tap sideways, while the turning push goes through the pads' faces.
FINGER_HOLD = -(BAR / 2 - 0.5 * MM)
FINGER_LOOSE = -20.0 * MM  # a 40 mm gap to come down round the bar
LIFT = 30.0 * MM
# WHY a lead (2026-11-07, sim trace): pushing the handles round, each arm sits a steady distance behind its target
# (the simulated arms give about 0.3 N/mm). While turning, the hands are commanded this far ahead along the circle (in
# the direction of the turn); at the end of each stroke they are commanded back to where the handles are. Open loop,
# from the trace: 12 mm turning in (3.75 N a hand), 4 mm backing out (1.25 N).
LEAD_IN = float(os.environ.get("TAP_LEAD", "12")) * MM
LEAD_OUT = float(os.environ.get("TAP_LEAD_OUT", "4")) * MM


def end_at(a: float, turns: float, up: float = 0.0, r: float = R_GRIP) -> tuple:
    t = math.radians(a)
    return (TAP_XY[0] + r * math.cos(t), TAP_XY[1] + r * math.sin(t), BAR_Z0 - PITCH * turns + up)


def heading(a: float) -> float:
    """The gripper's heading holding the handle end at ``a``: towards the tap, fingers across the bar."""
    return math.radians(a + 180.0)


def _turns_at(a: float) -> float:
    return (A_START - a) / 360.0


def _stroke(a0: float, a1: float, lead_mm: float):
    """Both arms' waypoints turning the left-held end from ``a0`` to ``a1`` (deg), then easing off at ``a1``."""
    n = max(1, math.ceil(abs(a1 - a0) / STEP))
    sign = 1.0 if a1 > a0 else -1.0
    lead = sign * math.degrees(lead_mm / R_GRIP)
    left, right = [], []
    for i in range(1, n + 1):
        a = a0 + (a1 - a0) * i / n
        dur = abs(a1 - a0) / n / TURN_RATE
        left.append(Waypoint(dur, end_at(a + lead, _turns_at(a)), FINGER_HOLD, "turn", heading(a + lead)))
        right.append(Waypoint(dur, end_at(a - 180.0 + lead, _turns_at(a)), FINGER_HOLD, "turn",
                              heading(a - 180.0 + lead)))
    left.append(Waypoint(0.5, end_at(a1, _turns_at(a1)), FINGER_HOLD, "ease off", heading(a1)))
    right.append(Waypoint(0.5, end_at(a1 - 180.0, _turns_at(a1)), FINGER_HOLD, "ease off", heading(a1 - 180.0)))
    return left, right


def _approach(a: float, finger_hold: float = None) -> list:
    hold = FINGER_HOLD if finger_hold is None else finger_hold
    return [
        Waypoint(3.0, end_at(a, 0.0, LIFT), FINGER_LOOSE, "over the handle", heading(a)),
        Waypoint(0.8, end_at(a, 0.0, LIFT), FINGER_LOOSE, "line up", heading(a)),
        Waypoint(1.5, end_at(a, 0.0), FINGER_LOOSE, "down to it", heading(a)),
        Waypoint(0.8, end_at(a, 0.0), hold, "take hold", heading(a)),
    ]


def one_hand_attempt() -> dict:
    """Dev check: the left arm alone turns the wrench by its handle, the stroke the demo makes with two."""
    lw, _ = _stroke(A_START, A_IN, LEAD_IN)
    return {"left": _approach(A_START) + lw}


def centre_twist_attempt() -> dict:
    """Dev check: the right arm clamps the wrench's round centre with its full squeeze and turns its wrist 100 degrees
    clockwise, like a spanner, with no side load."""
    z = BAR_Z0
    h0 = math.radians(A_START - 180.0 + 90.0)
    out = [
        Waypoint(3.0, (TAP_XY[0], TAP_XY[1], z + LIFT), FINGER_LOOSE, "over the centre", h0),
        Waypoint(1.5, (TAP_XY[0], TAP_XY[1], z), FINGER_LOOSE, "down to it", h0),
        Waypoint(1.8, (TAP_XY[0], TAP_XY[1], z), 0.0, "clamp the centre", h0),
    ]
    for i in range(1, 5):
        out.append(Waypoint(0.8, (TAP_XY[0], TAP_XY[1], z), 0.0, "twist", h0 - math.radians(25.0 * i)))
    return {"right": out}


def demo():
    if os.environ.get("TAP_DEMO") == "onehand":
        return one_hand_attempt()
    if os.environ.get("TAP_DEMO") == "centre":
        return centre_twist_attempt()
    plans = {}
    # WHY a pause above each handle and a slow descent (2026-11-07, sim trace): coming straight down, the soft arms were
    # still 17 mm off along the circle and the fingers landed on top of the handles.
    for side, a in (("left", A_START), ("right", A_START - 180.0)):
        plans[side] = _approach(a)
    for a0, a1, lead in ((A_START, A_IN, LEAD_IN), (A_IN, A_BACK, LEAD_OUT)):
        lw, rw = _stroke(a0, a1, lead)
        plans["left"] += lw
        plans["right"] += rw
    for plan in plans.values():
        last = plan[-1]
        plan.append(Waypoint(0.6, last.pos, FINGER_LOOSE, "let go", last.yaw))
        plan.append(Waypoint(1.0, (last.pos[0], last.pos[1], last.pos[2] + 0.06), FINGER_LOOSE, "away", last.yaw))
    return plans
