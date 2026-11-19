"""On the station table: a Christmas cracker (a standard 12 in cracker: a card tube wrapped in paper, crimped near each
end, a snap strip, a paper hat and a small gift inside) lies across the table in front of the robot.

The robot must pull the cracker, one end in each hand, until it cracks and comes apart, then put the pieces down on
the table.

Run it:  inspect-robots run --embodiment yam_arena -E task=christmas_cracker.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "Pull the Christmas cracker with both hands until it cracks, then put the pieces down." ...
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crackerkit  # noqa: E402
from crackerkit import Glue, body, box  # noqa: E402
from yam_arena.analysis import box_signed_distance  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

crackerkit.install()
SCENE = "station"
TOP = TABLE_TOP[SCENE]
MM = 0.001
RED = (0.72, 0.08, 0.10)  # the cracker's foil paper
GOLD = (0.85, 0.68, 0.22)  # its crimp bands
GREEN = (0.10, 0.42, 0.20)
WHITE = (0.93, 0.93, 0.90)
crackerkit.METALLIC.update({GOLD: 0.5, RED: 0.2})

# -- the cracker (env frame: x forward, y left, z up): along y, lying on the table --------------------------------------
CX, CY = 0.30, 0.0
TUBE_R = 25.0 * MM  # 50 mm across
NECK_R = 12.0 * MM  # the crimps squeeze it to about 24 mm
BODY_HALF = 70.0 * MM  # the middle: 140 mm long
NECK_L = 15.0 * MM
END_L = 65.0 * MM  # each end: 65 mm long (the whole cracker 300 mm)
M_END, M_BODY, M_GIFT = 0.010, 0.020, 0.012
# The crimps hold the ends on until they tear: a fixed joint each, breaking past BREAK_F (N) of pull or BREAK_T (N m)
# of bending. One is a little weaker (no two crimps are alike), so it tears first and the other end keeps the middle.
BREAK_F = {"left": float(os.environ.get("XC_BREAK", "20")), "right": float(os.environ.get("XC_BREAK", "20")) * 0.9}
BREAK_T = 0.25


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


_R_ALONG_Y = _rx(-math.pi / 2)  # a body's z axis along env +y
_Q_ALONG_Y = crackerkit.matrix_to_quat_rel(_R_ALONG_Y)


def _tube(z0: float, z1: float, r: float, colour) -> list:
    """A tube (a 12-sided prism of boxes) from z0 to z1 along the body's z."""
    return crackerkit.prism12((0.0, 0.0, (z0 + z1) / 2), r, z1 - z0, colour)


def _end(side: str):
    """One end: its crimped neck and its end tube (z along the cracker, outwards; origin at the crimp's middle)."""
    sgn = 1.0 if side == "left" else -1.0
    neck_mid = sgn * (BODY_HALF + NECK_L / 2)
    shapes = _tube(-NECK_L / 2, NECK_L / 2, NECK_R, GOLD) + _tube(NECK_L / 2, NECK_L / 2 + END_L, TUBE_R, RED)
    r = _R_ALONG_Y if side == "left" else _rx(math.pi / 2)  # z outwards on each side
    return body(f"cracker_{side}", pos=(CX, CY + neck_mid, TOP + TUBE_R), shapes=shapes, mass=M_END, friction=0.7,
                contact_offset=0.0005, solver_iterations=32, quat=crackerkit.matrix_to_quat_rel(r),
                com=(0.0, 0.0, NECK_L / 2 + END_L / 2))


_MIDDLE_SHAPES = crackerkit.ring(-BODY_HALF, BODY_HALF, TUBE_R - 1.5 * MM, TUBE_R, RED, segments=16)  # a card tube
MIDDLE = body("cracker_middle", pos=(CX, CY, TOP + TUBE_R), shapes=_MIDDLE_SHAPES
              + [box((0.0, TUBE_R + 0.3 * MM, 0.0), (20 * MM, 0.6 * MM, 60 * MM), GREEN)],
              mass=M_BODY, friction=0.7, contact_offset=0.0005, solver_iterations=32, quat=_Q_ALONG_Y,
              visual_only=(len(_MIDDLE_SHAPES),))
ENDS = {"left": _end("left"), "right": _end("right")}
# the gift inside: a small boxed toy, loose in the middle (it falls out when an end comes off)
GIFT = body("cracker_gift", pos=(CX, CY, TOP + 1.5 * MM + 7.5 * MM), shapes=[box((0.0, 0.0, 0.0), (22 * MM, 50 * MM, 14 * MM), WHITE)],
            mass=M_GIFT, friction=0.5, contact_offset=0.0005)

OBJECTS = [MIDDLE, ENDS["left"], ENDS["right"], GIFT]
crackerkit.register(*OBJECTS)
for _side, _e in ENDS.items():
    crackerkit.add_glue(Glue(MIDDLE.name, _e.name, break_force=BREAK_F[_side], break_torque=BREAK_T))
    crackerkit.no_collide(_e.name, GIFT.name)  # (the gift lies in the middle; the ends' necks would trap it)

# -- scoring -------------------------------------------------------------------------------------------------------
# 1.0   cracked (an end torn off the middle: the crimp more than 10 mm apart) and every piece down on the table,
#       no gripper within 2 cm of any piece, all still, for 1 s
# 0.5   cracked, the pieces not yet down and let go
# 0.25  held up off the table by both ends, not cracked
# 0     anything else
SCORE_OVER = "end"
RUN_LIMIT_STEPS = 1200
STILL_STEPS = 20
CLEAR_DIST = 20.0 * MM
TORN = 10.0 * MM
_STATE: dict = {}


@crackerkit.on_reset
def _reset_state() -> None:
    _STATE.clear()
    _STATE.update(still=0)


_reset_state()


def crimp_gap(objects, side: str) -> float:
    """How far (m) an end's crimp has moved from where it joins the middle."""
    m, e = objects[MIDDLE.name], objects[ENDS[side].name]
    sgn = 1.0 if side == "left" else -1.0
    join = m.to_world((0.0, 0.0, sgn * (BODY_HALF + NECK_L / 2)))
    return float(np.linalg.norm(np.asarray(e.pos) - join))


def cracked(objects) -> bool:
    return any(crimp_gap(objects, s) > TORN for s in ENDS)


def _down(st) -> bool:
    return bool(np.asarray(st.pos)[2] < TOP + TUBE_R + 15 * MM)


def held_up(objects) -> bool:
    return all(np.asarray(objects[e.name].pos)[2] > TOP + TUBE_R + 30 * MM for e in ENDS.values())


def hands_clear(objects) -> bool:
    pts = crackerkit.finger_points()
    if not len(pts):
        return True
    for spec in (MIDDLE, *ENDS.values()):
        st = objects[spec.name]
        lo, hi = spec.bounds_local()
        c = st.to_world((lo + hi) / 2)
        if box_signed_distance(pts, c, st.rotation(), hi - lo).min() < CLEAR_DIST:
            return False
    return True


def _update(objects) -> tuple[float, bool]:
    if cracked(objects):
        pieces = [objects[s.name] for s in (MIDDLE, *ENDS.values())]
        if all(_down(p) and p.resting() for p in pieces) and hands_clear(objects):
            return 1.0, True
        return 0.5, False
    return (0.25 if held_up(objects) else 0.0), False


def success(objects) -> bool:
    """True only once the full-score state has held for STILL_STEPS control steps in a row (1 s), as score() is 1.0
    only then; yam_arena ends the episode after success() has then held for its own SUCCESS_HOLD_S (1 s)."""
    ok = _update(objects)[1]
    _STATE["still"] = _STATE["still"] + 1 if ok else 0
    return _STATE["still"] >= STILL_STEPS


def score(objects) -> float:
    tier = _update(objects)[0]
    if tier == 1.0 and _STATE["still"] < STILL_STEPS:
        return 0.5
    return tier


# -- scripted demo -------------------------------------------------------------------------------------------------
# Open loop. Each hand takes one end from above (fingers closing across it), both lift it together, then both pull
# outwards along it, aimed well past where it will part (the soft arms pull until the weaker crimp tears); then each
# hand lowers its piece to the table and lets go.
TIP = 13.0 * MM
RHOME, LHOME = (0.14, -0.22, TOP + 0.17), (0.14, 0.22, TOP + 0.17)
GRIP_Y = BODY_HALF + NECK_L + END_L / 2  # each hand at the middle of its end
GRIP_DZ = float(os.environ.get("XC_GZ", "-6")) * MM  # pad tips this far below the tube's axis
OPEN_W = -(TUBE_R + 12 * MM)
HOLD = -(TUBE_R - float(os.environ.get("XC_SQ", "5")) * MM)  # commanded this far inside the tube: a firm grip
LIFT = 0.06
PULL = float(os.environ.get("XC_PULL", "80")) * MM  # each hand aims this far outwards (the soft arms)


def hand(side: str) -> list:
    sgn = 1.0 if side == "left" else -1.0
    home = LHOME if side == "left" else RHOME
    y = CY + sgn * GRIP_Y
    z = TOP + TUBE_R + GRIP_DZ + TIP
    g = math.pi / 2  # fingers closing along x, across the tube
    return [Waypoint(1.0, home, FINGER_OPEN, "wait", 0.0),
            Waypoint(2.0, (CX, y, z + 0.05), OPEN_W, "over its end", g),
            Waypoint(1.2, (CX, y, z), OPEN_W, "round its end", g),
            Waypoint(1.2, (CX, y, z), HOLD, "grip it", g),
            Waypoint(1.5, (CX, y, z + LIFT), HOLD, "lift it", g),
            Waypoint(0.8, (CX, y, z + LIFT), HOLD, "steady", g),
            Waypoint(1.5, (CX, y + sgn * PULL, z + LIFT), HOLD, "pull", g),
            Waypoint(1.0, (CX, y + sgn * PULL, z + LIFT), HOLD, "hold", g),
            Waypoint(1.5, (CX, y + sgn * PULL * 0.6, z + 0.004), HOLD, "down to the table", g),
            Waypoint(1.2, (CX, y + sgn * PULL * 0.6, z + 0.004), OPEN_W, "let go", g),
            Waypoint(1.0, (CX, y + sgn * PULL * 0.6, z + 0.06), OPEN_W, "up", g),
            Waypoint(2.0, home, FINGER_OPEN, "home", 0.0),
            Waypoint(2.0, home, FINGER_OPEN, "done", 0.0)]


def demo():
    """``XC_DEMO=onehand`` (a shortcut check): only the right hand grips and pulls; the left stays home."""
    right = hand("right")
    if os.environ.get("XC_DEMO") == "onehand":
        left = [Waypoint(sum(w.duration for w in right), LHOME, FINGER_OPEN, "wait", 0.0)]
    else:
        left = hand("left")
    return {"right": right, "left": left}
