"""On the station table: a window sash lies flat, a white-painted pine frame 464 mm wide and 264 mm deep, its far
half 39 to 52 cm in front of the arm bases, centred on the midline. A glass pane 400 x 200 x 4 mm lies in its
rebate on a felt bed, 1 mm below the frame's top, with a 2 mm gap all round: there is no edge to take hold of. Two
hand suction lifters lie face down on the table, one in front of each arm, 30 cm out. Each has a black rubber cup
115 mm across, an aluminium handle bar across its back and a red lever beside the handle: turned down flat, the
lever seals the cup; lifted, it lets go. Two felt-topped pine bearers stand between the sash and the robot.

Run it:  inspect-robots run --embodiment yam_arena -E task=glass_pane.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "Take the glass pane out of the window sash with the two suction lifters, lay it flat on
           the two felt bearers, then unlock the lifters." --scorer task_score ...
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glazekit  # noqa: E402
from glazekit import Detent, Hinge, Suction, body, box, prism12  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

glazekit.install()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
MM = 0.001

# -- the sash (fixed): frame and felt bed -----------------------------------------------------------------------
SASH_XY = (0.39, 0.0)
REBATE = (204.0 * MM, 404.0 * MM)  # inside (x, y): the pane plus a 2 mm gap all round
FRAME_W = 30.0 * MM
FRAME_H = 40.0 * MM
BED_TOP = 35.0 * MM  # the felt bed the pane lies on; the pane's top is then 1 mm below the frame's top
WHITE_PAINT = (0.92, 0.91, 0.88)
FELT = (0.20, 0.32, 0.24)


def _sash() -> list:
    rx, ry = REBATE
    ox, oy = rx + 2 * FRAME_W, ry + 2 * FRAME_W
    return [
        box((+(rx / 2 + FRAME_W / 2), 0, FRAME_H / 2), (FRAME_W, oy, FRAME_H), WHITE_PAINT),
        box((-(rx / 2 + FRAME_W / 2), 0, FRAME_H / 2), (FRAME_W, oy, FRAME_H), WHITE_PAINT),
        box((0, +(ry / 2 + FRAME_W / 2), FRAME_H / 2), (rx, FRAME_W, FRAME_H), WHITE_PAINT),
        box((0, -(ry / 2 + FRAME_W / 2), FRAME_H / 2), (rx, FRAME_W, FRAME_H), WHITE_PAINT),
        box((0, 0, BED_TOP / 2), (rx, ry, BED_TOP), FELT),
    ]


SASH = body("sash", pos=(*SASH_XY, TOP), shapes=_sash(), friction=0.6)

# -- the pane (pane frame: origin at the centre of its underside, z up) ------------------------------------------
PANE_SIZE = (200.0 * MM, 400.0 * MM, 4.0 * MM)  # x (depth), y (width), thickness
PANE_MASS = 0.80  # kg: 4 mm float glass, 2.5 g/cm3
GLASS = (0.72, 0.84, 0.82)
PANE = body(
    "pane",
    pos=(*SASH_XY, TOP + BED_TOP),
    shapes=[box((0, 0, PANE_SIZE[2] / 2), PANE_SIZE, GLASS)],
    mass=PANE_MASS,
    com=(0.0, 0.0, PANE_SIZE[2] / 2),
    friction=0.5,
    contact_offset=2.0 * MM,
    solver_iterations=32,
)

# -- the bearers (fixed): two felt-topped battens between the sash and the robot ----------------------------------
BEARERS_X = 0.17  # midway between the two battens; the pane laid on them is centred here
BEARER_GAP = 0.12  # centre to centre (x)
BEARER = (40.0 * MM, 360.0 * MM, 50.0 * MM)
PINE = (0.80, 0.66, 0.46)
BEARERS = body(
    "bearers",
    pos=(BEARERS_X, 0.0, TOP),
    shapes=[box((s * BEARER_GAP / 2, 0, BEARER[2] / 2), BEARER, PINE) for s in (-1, 1)]
    + [box((s * BEARER_GAP / 2, 0, BEARER[2] - 0.5 * MM), (BEARER[0], BEARER[1], 1.0 * MM), FELT) for s in (-1, 1)],
    visual_only=(2, 3),
    friction=0.7,
)

# -- the suction lifters (lifter frame: origin at the centre of the cup's face, z out of its back) -----------------
CUP_R = 57.5 * MM
CUP_T = 8.0 * MM
HANDLE_Z = 51.0 * MM  # the handle bar's centre above the cup's face: where the gripper holds it
LIFTER_MASS = 0.28
LEVER_MASS = 0.03
RUBBER = (0.08, 0.08, 0.09)
ALU = (0.70, 0.71, 0.73)
RED = (0.78, 0.14, 0.12)
LEVER_PIVOT = (0.0, 30.0 * MM, 23.0 * MM)  # beside the handle, on the cup's outer side; axis along the handle
LEVER_LEN = 64.0 * MM
LEVER_OPEN = 60.0  # deg: how far it stands up when unlocked
LIFTER_START = 0.31  # |y| of each lifter at the start, x LIFTER_X
LIFTER_X = 0.30


def _lifter_shapes() -> list:
    return [
        *prism12((0, 0, CUP_T / 2), CUP_R, CUP_T, RUBBER),
        *prism12((0, 0, 14.0 * MM), 40.0 * MM, 12.0 * MM, ALU),
        box((+45.0 * MM, 0, 30.0 * MM), (12.0 * MM, 22.0 * MM, 20.0 * MM), ALU),
        box((-45.0 * MM, 0, 30.0 * MM), (12.0 * MM, 22.0 * MM, 20.0 * MM), ALU),
        box((0, 0, HANDLE_Z), (110.0 * MM, 22.0 * MM, 22.0 * MM), ALU),
    ]


def _lever_shapes() -> list:
    """In the lifter's frame, turned down flat (the joint at 0): it runs outwards from the pivot, past the cup's rim."""
    px, py, pz = LEVER_PIVOT
    return [box((px, py - 2.0 * MM + LEVER_LEN / 2, pz), (14.0 * MM, LEVER_LEN, 6.0 * MM), RED)]


def _lifter(side: str, s: float):
    """``s`` +1 for the left lifter (lever towards +y), -1 for the right one (turned half round)."""
    yaw = 0.0 if s > 0 else math.pi
    pos = (LIFTER_X, s * LIFTER_START, TOP + 0.5 * MM)
    lifter = body(f"lifter_{side}", pos=pos, yaw=yaw, shapes=_lifter_shapes(), mass=LIFTER_MASS,
                  com=(0.0, 0.0, 20.0 * MM), friction=0.9, contact_offset=2.0 * MM, solver_iterations=32)
    hinge = glazekit.add_hinge(Hinge(lifter.name, f"lever_{side}", pivot=LEVER_PIVOT, start=LEVER_OPEN, lower=-3.0,
                                     upper=LEVER_OPEN + 5.0))
    lpos, lquat = hinge.child_pose(pos, yaw)
    lever = body(hinge.child, pos=lpos, quat=lquat, shapes=_lever_shapes(), mass=LEVER_MASS,
                 com=(0.0, LEVER_PIVOT[1] - 2.0 * MM + LEVER_LEN / 2, LEVER_PIVOT[2]), friction=0.9,
                 contact_offset=1.0 * MM, solver_iterations=32)
    return lifter, lever, hinge


LIFTER_L, LEVER_L, HINGE_L = _lifter("l", +1.0)
LIFTER_R, LEVER_R, HINGE_R = _lifter("r", -1.0)

OBJECTS = [SASH, BEARERS, PANE, LIFTER_L, LEVER_L, LIFTER_R, LEVER_R]
_SPEC = {o.name: o for o in OBJECTS}
glazekit.register(*OBJECTS)

DETENTS = [glazekit.add_model(Detent(h, low=0.0, high=LEVER_OPEN + 3.0, mid=25.0)) for h in (HINGE_L, HINGE_R)]
CUPS = {
    side: glazekit.add_model(Suction(f"lifter_{side}", h, "pane", part_half=(PANE_SIZE[0] / 2, PANE_SIZE[1] / 2),
                                     part_top=PANE_SIZE[2], radius=CUP_R))
    for side, h in (("l", HINGE_L), ("r", HINGE_R))
}


# -- scoring -------------------------------------------------------------------------------------------------------
# 1.0   the pane lies flat on both bearers (tilted under 3 degrees, underside within 3 mm of their tops, centre within
#       20 mm of the point midway between them), after a lift made with both lifters locked on it; both levers up
#       (45 degrees or more: unlocked); no part of either gripper within 20 mm of the glass or 10 mm of a lifter; all
#       still (under 1 cm/s) for 1 s
# 0.5   the pane was lifted 10 mm clear of the sash with both lifters locked on it, but is not as for 1.0
# 0.25  a lifter was locked on the pane at some point
# 0     anything else. Also 0, latched for the rest of the trial: the pane dropped (falling faster than 0.3 m/s, or
#       lower than the sash's bed), tilted more than 20 degrees once off the sash, or touched by a gripper.
SCORE_OVER = "end"
RUN_LIMIT_STEPS = 2400
LIFT_CLEAR = 10.0 * MM
FLAT_DEG = 3.0
ON_BEARERS = 3.0 * MM
CENTRE_TOL = 20.0 * MM
FINGERS_CLEAR = 20.0 * MM
TOOLS_CLEAR = 10.0 * MM
LEVER_UP = 45.0  # deg: a lever this far up has snapped open past its catch
TOUCH = 1.0 * MM
TOOLS = ("lifter_l", "lever_l", "lifter_r", "lever_r")
DROP_SPEED = 0.30
CARRY_TILT = 20.0
BEARER_TOP = BEARER[2]

_STATE: dict = {}


@glazekit.on_reset
def _reset_state() -> None:
    _STATE.update(lifted=False, violation=None)


_reset_state()


def tilt_deg(state) -> float:
    return float(np.degrees(np.arccos(np.clip(state.rotation()[2, 2], -1.0, 1.0))))


def fingers_to_pane(pane) -> float:
    """Nearest distance (m) from either gripper's hulls (palm and fingers) to the pane's box. inf offline."""
    pts = glazekit.finger_points()
    if not len(pts):
        return math.inf
    local = (pts - pane.pos) @ pane.rotation()
    half = np.array(PANE_SIZE) / 2
    centre = np.array([0.0, 0.0, PANE_SIZE[2] / 2])
    d = np.maximum(np.abs(local - centre) - half, 0.0)
    return float(np.linalg.norm(d, axis=1).min())


def fingers_to(state, spec) -> float:
    """Nearest distance (m) from either gripper's hulls to a body's bounding box (in its own frame). inf offline.
    WHY (2026-11-07, sim trace): with only the glass checked, success fired while the fingers still held both levers
    part-way up; placed must mean let go."""
    pts = glazekit.finger_points()
    if not len(pts):
        return math.inf
    lo, hi = spec.bounds_local()
    local = (pts - state.pos) @ state.rotation()
    d = np.maximum(np.maximum(lo - local, local - hi), 0.0)
    return float(np.linalg.norm(d, axis=1).min())


def lever_angles(objects) -> tuple[float, float]:
    return (HINGE_L.angle(objects["lifter_l"].rotation(), objects["lever_l"].rotation()),
            HINGE_R.angle(objects["lifter_r"].rotation(), objects["lever_r"].rotation()))


def _update(objects) -> tuple[float, bool]:
    pane = objects["pane"]
    under = float(pane.pos[2] - TOP)  # the underside's centre above the table
    tilt = tilt_deg(pane)
    off_sash = under > BED_TOP + LIFT_CLEAR
    if _STATE["violation"] is None:
        if pane.lin_vel[2] < -DROP_SPEED or under < BED_TOP - 5.0 * MM:
            _STATE["violation"] = "dropped"
        elif off_sash and tilt > CARRY_TILT:
            _STATE["violation"] = "tilted"
        elif fingers_to_pane(pane) < TOUCH:
            _STATE["violation"] = "touched"
    both = all(c.sealed for c in CUPS.values())
    _STATE["lifted"] |= bool(off_sash and both)
    if _STATE["violation"]:
        return 0.0, False
    centre = pane.to_world((0.0, 0.0, 0.0))
    on = (tilt < FLAT_DEG and abs(under - BEARER_TOP) < ON_BEARERS
          and np.hypot(centre[0] - BEARERS_X, centre[1]) < CENTRE_TOL)
    unlocked = not any(c.sealed for c in CUPS.values()) and min(lever_angles(objects)) >= LEVER_UP
    clear = fingers_to_pane(pane) >= FINGERS_CLEAR and all(
        fingers_to(objects[n], _SPEC[n]) >= TOOLS_CLEAR for n in TOOLS)
    if _STATE["lifted"] and on and unlocked and clear:
        still = all(objects[n].resting() for n in ("pane", "lifter_l", "lifter_r"))
        return 1.0, bool(still)
    if _STATE["lifted"]:
        return 0.5, False
    return (0.25 if any(c.ever_sealed for c in CUPS.values()) else 0.0), False


def success(objects) -> bool:
    """Pane lifted with both lifters, laid flat on the bearers, lifters unlocked, grippers clear, still for 1 s."""
    return _update(objects)[1]


def score(objects) -> float:
    """1.0 / 0.5 / 0.25 / 0 as in the table above; violations latch."""
    return _update(objects)[0]


# -- scripted demo -------------------------------------------------------------------------------------------------
# Each arm sets its lifter on its own half of the pane, locks it, takes its handle again; then both lift together,
# draw the pane back over the bearers, lay it down, and unlock.
# WHY on the pane's centre line (2026-11-07, sim trace): set 30 mm nearer the robot, the cups carried the pane's weight
# off-centre and it turned the lifters 3 degrees in the grippers' pads (a parallel grip holds a bar against turning
# about the grip's own line only by friction).
CUP_AT = (SASH_XY[0], 0.12)  # x, |y|: the cups' centres; 20 mm inside the pane's ends with their rims
PANE_TOP = TOP + BED_TOP + PANE_SIZE[2]
LIFT = 50.0 * MM  # the pane's underside goes from 35 to 85 mm: over the sash's frame (40) and the bearers (50)
CARRY_DX = BEARERS_X - SASH_XY[0]
PRESS = 1.0 * MM
GRIP_S = 1.8  # WHY 1.8 s for a full gripper stroke: the run's delta-limit guard allows 5% of its range a step
# WHY a 6 mm target gap on the 14 mm lever (2026-11-07, sim trace): commanded to exactly its width, the fingers held it
# with no squeeze, and it slipped out after 5 degrees. 4 mm short each side is about 20 N per finger (servo 5000 N/m).
LEVER_FINGER = -3.0 * MM


def wait(seconds: float, last: Waypoint, label: str = "wait") -> Waypoint:
    return Waypoint(seconds, last.pos, last.finger, label, last.yaw)


def _t(wps) -> float:
    return sum(w.duration for w in wps)


def _lever_point(cup_xy, s: float, d: float, angle_deg: float, face_z: float) -> np.ndarray:
    """World point ``d`` along the lever from its pivot, the lever at ``angle_deg``, the cup's face at ``face_z``."""
    a = math.radians(angle_deg)
    px, py, pz = LEVER_PIVOT
    return np.array([cup_xy[0] + px, s * (cup_xy[1] + py + d * math.cos(a)), face_z + pz + d * math.sin(a)])


def _tcp_for_tip(tip) -> tuple:
    """TCP for fingertips at ``tip``, pointing straight down."""
    return (float(tip[0]), float(tip[1]), float(tip[2] + PAD_TIP_PAST_TCP))


def place_and_lock(s: float) -> list:
    x0, y0 = LIFTER_X, s * LIFTER_START
    cx, cy = CUP_AT
    on_glass = PANE_TOP + HANDLE_Z - PRESS
    above_lever = _lever_point((cx, cy), s, 45.0 * MM, LEVER_OPEN, PANE_TOP) + np.array([0, s * 10.0 * MM, 25.0 * MM])
    pushed = _lever_point((cx, cy), s, 52.0 * MM, 0.0, PANE_TOP) + np.array([0, 0, 6.0 * MM + 1.0 * MM])
    return [
        Waypoint(2.0, (x0, y0, TOP + HANDLE_Z + 0.06), FINGER_OPEN, "over the lifter", 0.0),
        Waypoint(1.2, (x0, y0, TOP + HANDLE_Z), FINGER_OPEN, "down to the handle", 0.0),
        Waypoint(GRIP_S, (x0, y0, TOP + HANDLE_Z), FINGER_CLOSED, "grip the handle", 0.0),
        Waypoint(1.5, (x0, y0, TOP + HANDLE_Z + 0.10), FINGER_CLOSED, "lift the lifter", 0.0),
        Waypoint(2.0, (cx, s * cy, PANE_TOP + HANDLE_Z + 0.04), FINGER_CLOSED, "over the pane", 0.0),
        Waypoint(1.5, (cx, s * cy, on_glass), FINGER_CLOSED, "cup onto the glass", 0.0),
        Waypoint(GRIP_S, (cx, s * cy, on_glass), FINGER_OPEN, "let go of the handle", 0.0),
        Waypoint(1.0, (cx, s * cy, on_glass + 0.06), FINGER_OPEN, "up from the handle", 0.0),
        Waypoint(GRIP_S, _tcp_for_tip(above_lever), FINGER_CLOSED, "over the lever", 0.0),
        Waypoint(1.5, _tcp_for_tip(pushed), FINGER_CLOSED, "push the lever down", 0.0),
        Waypoint(0.8, _tcp_for_tip(pushed + np.array([0, 0, 0.04])), FINGER_CLOSED, "up from the lever", 0.0),
        Waypoint(GRIP_S, (cx, s * cy, PANE_TOP + HANDLE_Z + 0.04), FINGER_OPEN, "over the handle again", 0.0),
        Waypoint(1.2, (cx, s * cy, PANE_TOP + HANDLE_Z), FINGER_OPEN, "down to the handle again", 0.0),
        Waypoint(GRIP_S, (cx, s * cy, PANE_TOP + HANDLE_Z), FINGER_CLOSED, "grip the handle again", 0.0),
    ]


def carry(s: float) -> list:
    cx, cy = CUP_AT
    h0 = PANE_TOP + HANDLE_Z
    down = TOP + BEARER_TOP + PANE_SIZE[2] + HANDLE_Z - PRESS
    return [
        Waypoint(2.0, (cx, s * cy, h0 + LIFT), FINGER_CLOSED, "lift the pane", 0.0),
        Waypoint(3.0, (cx + CARRY_DX, s * cy, h0 + LIFT), FINGER_CLOSED, "carry the pane", 0.0),
        Waypoint(2.5, (cx + CARRY_DX, s * cy, down), FINGER_CLOSED, "lay it on the bearers", 0.0),
    ]


def unlock(s: float) -> list:
    cx, cy = CUP_AT[0] + CARRY_DX, CUP_AT[1]
    face = TOP + BEARER_TOP + PANE_SIZE[2]
    yaw = -math.pi / 2 * s  # fingers close along x, across the lever's width (the left arm reaches -90, the right +90)
    end = 58.0 * MM  # along the lever from its pivot, 6 mm from its end
    flat = _lever_point((cx, cy), s, end, 0.0, face)
    pts = [_lever_point((cx, cy), s, end, a, face) for a in (10.0, 20.0, 35.0)]
    hand = (cx, s * cy, face + HANDLE_Z)
    return [
        Waypoint(GRIP_S, hand, FINGER_OPEN, "let go of the handle", 0.0),
        Waypoint(1.0, (hand[0], hand[1], hand[2] + 0.05), FINGER_OPEN, "up from the handle", 0.0),
        Waypoint(1.5, _tcp_for_tip(flat + np.array([0, 0, 0.03])), FINGER_OPEN, "over the lever's end", yaw),
        Waypoint(1.2, _tcp_for_tip(flat + np.array([0, 0, -2.0 * MM])), FINGER_OPEN, "round the lever's end", yaw),
        Waypoint(GRIP_S, _tcp_for_tip(flat + np.array([0, 0, -2.0 * MM])), LEVER_FINGER, "pinch the lever", yaw),
        *[Waypoint(0.6, _tcp_for_tip(p + np.array([0, 0, -2.0 * MM])), LEVER_FINGER, "lift the lever", yaw) for p in pts],
        Waypoint(GRIP_S, _tcp_for_tip(pts[-1] + np.array([0, 0, -2.0 * MM])), FINGER_OPEN, "let go of the lever", yaw),
        Waypoint(1.5, _tcp_for_tip(pts[-1] + np.array([0, 0, 0.08])), FINGER_OPEN, "away", yaw),
    ]


def edge_attempt() -> list:
    """Dev check for a loophole: the left gripper aims its fingertips into the 2 mm gap at the pane's left end,
    squeezes and pulls up, without the lifters."""
    x, y = SASH_XY[0], PANE_SIZE[1] / 2 + 1.0 * MM
    tip = np.array([x, y, PANE_TOP - 2.0 * MM])
    return [
        Waypoint(2.5, _tcp_for_tip(tip + np.array([0, 0, 0.06])), FINGER_OPEN, "over the pane's end", math.pi / 2),
        Waypoint(1.5, _tcp_for_tip(tip), FINGER_OPEN, "fingers into the gap", math.pi / 2),
        Waypoint(GRIP_S, _tcp_for_tip(tip), FINGER_CLOSED, "squeeze", math.pi / 2),
        Waypoint(1.5, _tcp_for_tip(tip + np.array([0, 0, 0.05])), FINGER_CLOSED, "pull up", math.pi / 2),
        Waypoint(1.0, _tcp_for_tip(tip + np.array([0, 0, 0.05])), FINGER_CLOSED, "hold", math.pi / 2),
    ]


def demo():
    if os.environ.get("GLASS_DEMO") == "edge":
        return {"left": edge_attempt()}
    plans = {}
    for side, s in (("left", 1.0), ("right", -1.0)):
        plans[side] = place_and_lock(s)
    # both arms start the lift together
    t = max(_t(p) for p in plans.values())
    for side in plans:
        if _t(plans[side]) < t:
            plans[side].append(wait(t - _t(plans[side]), plans[side][-1], "wait for the other arm"))
    for side, s in (("left", 1.0), ("right", -1.0)):
        plans[side] += carry(s) + unlock(s)
    return plans
