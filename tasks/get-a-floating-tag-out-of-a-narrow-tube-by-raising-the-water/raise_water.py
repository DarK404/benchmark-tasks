"""On the station table: a clear acrylic tube (40 mm bore, 135 mm deep inside, its rim 140 mm above the table)
stands on a steel base plate fixed to the table 20 cm in front of the arm bases, on the midline, with 71 mm of water
in it. A cork ring float hugs the bore, kept from turning by a thin rib up the tube's far wall, with a red plastic tag
25 mm tall standing on its near side; the tag's top is 33 mm below the rim, out of reach. On the left, a wooden rack
holds six round slugs 27 mm across and 35 mm long, standing upright: four aluminium, two cork. On the right lies a
shallow grey tray.

The robot must get the tag out and put it in the tray. Dropping aluminium slugs into the tube raises the water (each
by 15 mm) and the float with it; three bring the tag's top 13 mm above the rim, where it can be pinched and the float
lifted out (two leave it 2 mm below the rim). Cork slugs float and hardly raise the water.

Run it:  inspect-robots run --embodiment yam_arena -E task=raise_water.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "..." --scorer task_score ...
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import waterkit  # noqa: E402
from waterkit import Floater, Water, body, box, cyl, prism12, ring  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

waterkit.install()
SCENE = "station"
TOP = TABLE_TOP[SCENE]
MM = 0.001
STEEL = (0.55, 0.56, 0.58)
BASE_C = (0.30, 0.32, 0.36)
ACRYLIC = (0.92, 0.96, 0.98, 0.2)  # a fourth component below 1 makes it see-through (waterkit: glass)
WATER_C = (0.25, 0.50, 0.95, 0.6)
CORK = (0.72, 0.55, 0.36)
WOOD = (0.62, 0.45, 0.28)
RED = (0.82, 0.12, 0.10)
GREY = (0.45, 0.47, 0.50)

# -- the tube (fixed; origin on the table under the bore's axis) --------------------------------------------------
# WHY everything low (2026-11-09, IK map and sim traces): pointing straight down, neither arm reaches higher than about
# 200 mm above the table anywhere near here (joint 4 runs onto its limit; above that the gripper tilts and ends up
# 40 mm or more from its target, and the slugs fell outside the rim). Every straight-down pose stays below 190 mm:
# the rim is 140 mm up, the slugs 35 mm long, the tag 25 mm.
TUBE_XY = (0.20, 0.0)
BASE_H = 5.0 * MM
BORE_R = 20.0 * MM
WALL = 4.0 * MM
DEPTH = 135.0 * MM  # inside, from the floor to the rim
FLOOR = BASE_H
RIM = BASE_H + DEPTH  # 140 mm above the table
WATER0 = 71.0 * MM  # the water's starting depth (74 mm with the float in it)
AREA = math.pi * BORE_R ** 2
RIB = 3.0 * MM  # the float's guide rib, up the far (+x) wall
TUBE = body("tube", pos=(*TUBE_XY, TOP), friction=0.3, shapes=[
    box((0, 0, BASE_H / 2), (0.08, 0.08, BASE_H), BASE_C),
    box((BORE_R - RIB / 2 + 0.3 * MM, 0, FLOOR + DEPTH / 2), (RIB, RIB, DEPTH), (0.85, 0.88, 0.9)),
    *ring(BASE_H, RIM, BORE_R, BORE_R + WALL, ACRYLIC, segments=24),
])

# the water: drawn as a column whose top the water model keeps at the level (no gravity, no collisions)
COLUMN = 0.25
SURFACE = body("water", pos=(*TUBE_XY, TOP + FLOOR + WATER0), mass=0.05, visual_only=(0,), inertia=(1e-4,) * 3,
               shapes=[cyl((0, 0, -COLUMN / 2), BORE_R - 0.4 * MM, COLUMN, WATER_C)])
waterkit.no_gravity("water")

# -- the float: a cork ring with a notch for the rib, a red tag on its near (-x) side; origin at its bottom centre --
F_RIN, F_ROUT, F_H = 15.0 * MM, 19.5 * MM, 12.0 * MM
F_SEGMENTS = 16
F_RING = [s for k, s in enumerate(ring(0.0, F_H, F_RIN, F_ROUT, CORK, segments=F_SEGMENTS)) if k != 0]  # notch at +x
TAG_H, TAG_W, TAG_T = 25.0 * MM, 12.0 * MM, 3.0 * MM
TAG_R = (F_RIN + F_ROUT) / 2
F_MASS = 0.004
F_AREA = math.pi * (F_ROUT ** 2 - F_RIN ** 2) * (F_SEGMENTS - 1) / F_SEGMENTS
F_SINK = F_MASS / (1000.0 * F_AREA)  # how deep it floats: 9 mm of its 12
FLOAT = body("tagfloat", pos=(*TUBE_XY, TOP + FLOOR + WATER0 - F_SINK), mass=F_MASS, friction=0.2,
             com=(0.0, 0.0, 0.004), inertia=(2e-6, 2e-6, 2e-6), shapes=[
                 *F_RING,
                 box((-TAG_R, 0.0, F_H + TAG_H / 2), (TAG_T, TAG_W, TAG_H), RED),
             ])

# -- the slugs, standing in a rack on the left -------------------------------------------------------------------
SLUG_R, SLUG_H = 13.0 * MM, 35.0 * MM  # 12-sided, 13 mm to the flats
SLUG_AREA = 12 * SLUG_R ** 2 * math.tan(math.pi / 12)
SLUG_R12 = SLUG_R / math.cos(math.pi / 12)
# WHY 60 mm apart (2026-11-09, sim trace): turned to close across the rack, the gripper's body spans about +-51 mm
# along it; at 35 mm it came down on the neighbouring slugs.
RACK_X, RACK_Y0, RACK_DY = 0.18, 0.12, 0.060
RACK_H, POCKET = 15.0 * MM, 8.0 * MM  # the slugs stand in pockets 8 mm deep
KINDS = os.environ.get("WATER_SLUGS", "SCSSCS")  # S aluminium (sinks) / C cork (floats), from the midline outwards
SLUG_Z = TOP + RACK_H - POCKET + SLUG_H / 2


def _rack_shapes() -> list:
    n, w = len(KINDS), 30.0 * MM
    length = (n - 1) * RACK_DY + 40.0 * MM
    out = [box((0, (n - 1) * RACK_DY / 2, (RACK_H - POCKET) / 2), (40.0 * MM, length, RACK_H - POCKET), WOOD)]
    for i in range(n + 1):  # walls between pockets
        out.append(box((0, i * RACK_DY - RACK_DY / 2, RACK_H - POCKET / 2), (40.0 * MM, RACK_DY - w, POCKET), WOOD))
    for sx in (-1, 1):  # front and back walls
        out.append(box((sx * 17.5 * MM, (n - 1) * RACK_DY / 2, RACK_H - POCKET / 2), (5.0 * MM, length, POCKET), WOOD))
    return out


RACK = body("rack", pos=(RACK_X, RACK_Y0, TOP), friction=0.6, shapes=_rack_shapes())


# WHY aluminium (2026-11-09, sim trace): pinched near its top, a 208 g steel slug swung about the pads' line as it was
# carried and its bottom caught the rim; aluminium (73 g) still sinks and raises the water just as much.
def _slug(i: int, kind: str):
    metal = kind == "S"
    name = f"{'metal' if metal else 'cork'}{i}"
    m = SLUG_AREA * SLUG_H * (2700.0 if metal else 240.0)
    return body(name, pos=(RACK_X, RACK_Y0 + i * RACK_DY, SLUG_Z), mass=m, friction=0.6,
                shapes=prism12((0, 0, 0), SLUG_R, SLUG_H, STEEL if metal else CORK))


SLUGS = [_slug(i, k) for i, k in enumerate(KINDS)]
METAL_SLOTS = [i for i, k in enumerate(KINDS) if k == "S"]
CORK_SLOTS = [i for i, k in enumerate(KINDS) if k == "C"]

TRAY_XY = (0.20, -0.15)
TRAY = body("tray", pos=(*TRAY_XY, TOP), friction=0.6, shapes=[
    box((0, 0, 1.5 * MM), (0.12, 0.08, 3.0 * MM), GREY),
    box((0, 0.0385, 7.5 * MM), (0.12, 3.0 * MM, 15.0 * MM), GREY),
    box((0, -0.0385, 7.5 * MM), (0.12, 3.0 * MM, 15.0 * MM), GREY),
    box((0.0585, 0, 7.5 * MM), (3.0 * MM, 0.08, 15.0 * MM), GREY),
    box((-0.0585, 0, 7.5 * MM), (3.0 * MM, 0.08, 15.0 * MM), GREY),
])

OBJECTS = [TUBE, SURFACE, FLOAT, RACK, TRAY, *SLUGS]
waterkit.register(*OBJECTS)

# -- the water model --------------------------------------------------------------------------------------------
# WHY these drags (2026-11-09, sim trace): about critical damping for the floats' bobbing (stiffness rho g A); the
# aluminium slugs sink at about the speed a 51 g slug falls through water.
FLOATERS = [Floater("tagfloat", 0.0, F_H, F_AREA, F_ROUT, drag=0.25, spin_drag=2e-5)]
FLOATERS += [Floater(s.name, -SLUG_H / 2, SLUG_H, SLUG_AREA, SLUG_R12,
                     drag=1.5 if s.name.startswith("metal") else 0.4,
                     spin_drag=1e-4 if s.name.startswith("metal") else 5e-5) for s in SLUGS]
WATER = waterkit.add_model(Water(TUBE_XY, BORE_R, TOP + FLOOR, AREA * WATER0, FLOATERS, surface="water"))
waterkit.on_reset(WATER.clear)

# -- scoring -------------------------------------------------------------------------------------------------------
SCORE_OVER = "end"
RUN_LIMIT_STEPS = 1600
CLEAR = 10.0 * MM
TAG_LOCAL_TOP = np.array([-TAG_R, 0.0, F_H + TAG_H])


def _tag_top(state) -> np.ndarray:
    return state.to_world(TAG_LOCAL_TOP)


def _out_of_tube(state) -> bool:
    """The float wholly above the rim."""
    return float(state.low[2]) > TOP + RIM


def _in_tray(state) -> bool:
    lo, hi = state.low, state.high
    return (abs(state.pos[0] - TRAY_XY[0]) < 0.06 - 0.003 and abs(state.pos[1] - TRAY_XY[1]) < 0.04 - 0.003
            and float(lo[2]) < TOP + 15.0 * MM)


_STATE: dict = {}


def _reset_state() -> None:
    _STATE.update(raised=False, out=False)


_reset_state()
waterkit.on_reset(_reset_state)


def fingers_to(state, spec) -> float:
    """Nearest distance (m) from either gripper's hulls to a body's bounding box (in its own frame). inf offline."""
    pts = waterkit.finger_points()
    if not len(pts):
        return math.inf
    lo, hi = spec.bounds_local()
    local = (pts - state.pos) @ state.rotation()
    d = np.maximum(np.maximum(lo - local, local - hi), 0.0)
    return float(np.linalg.norm(d, axis=1).min())


def _update(objects) -> tuple[float, bool]:
    f = objects["tagfloat"]
    _STATE["raised"] |= bool(_tag_top(f)[2] > TOP + RIM)
    _STATE["out"] |= _out_of_tube(f)
    if _in_tray(f) and fingers_to(f, FLOAT) >= CLEAR:
        return 1.0, bool(f.resting())
    if _STATE["out"]:
        return 0.5, False
    return (0.25 if _STATE["raised"] else 0.0), False


def success(objects) -> bool:
    """The float with its tag resting in the tray, released, still for 1 s."""
    return _update(objects)[1]


def score(objects) -> float:
    """1.0 in the tray / 0.5 taken out of the tube / 0.25 its tag raised above the rim / 0."""
    return _update(objects)[0]


# -- scripted demo -------------------------------------------------------------------------------------------------
# Open loop, from the fixed layout. The left arm drops three aluminium slugs into the tube, one at a time; then the
# right arm pinches the tag above the rim, lifts the float out and lays it in the tray.
# WHY so near the top (2026-11-09, sim trace): the fingers' collision hulls fill in about 25 mm above the fingertips;
# held 18 mm down (its top 31 mm above the tips) the gripper landed on the slug's top and pushed it over.
GRIP_DOWN = 10.0 * MM  # the TCP this far below the slug's top: its top 23 mm above the fingertips
LIFT_Z = TOP + RIM + SLUG_H - GRIP_DOWN + 14.0 * MM  # carried with its bottom 14 mm above the rim
DROP_Z = TOP + RIM + SLUG_H - GRIP_DOWN + 8.0 * MM  # let go with its bottom 8 mm above the rim
LEFT_HOME = (0.18, 0.20, TOP + 0.17)
RIGHT_HOME = (0.18, -0.20, TOP + 0.17)
TAG_PINCH = 8.0 * MM  # fingertips this far above the rim, pinching the tag's top 13 mm


# WHY a correction over the tube (2026-11-09, sim trace): holding a slug there, the left gripper sat a steady
# (+28, -28) mm from its target, the same for every slug, and the slugs were let go outside the rim. The demo aims
# that much the other way (open loop: a fixed calibration of the arm, as on a real robot).
DROP_FIX = (float(os.environ.get("WATER_FIX_X", "0")) * MM, float(os.environ.get("WATER_FIX_Y", "0")) * MM)
DROP_XY = (TUBE_XY[0] + DROP_FIX[0], TUBE_XY[1] + DROP_FIX[1])
HALF_OPEN = FINGER_OPEN  # fully open: closing across the rack, nothing is in the way (half open caught the slug top)
# WHY across the rack (2026-11-09, sim trace): with the fingers closing along the rack, fully open they reached the
# neighbouring slugs 35 mm away and knocked the one being taken over; closing across it, nothing is in the way.
# WHY -90 degrees, not +90 (2026-11-09, sim trace and offline IK): at +90 the left wrist's roll (joint 6) sat on its
# limit (-2.09 rad) and the gripper ended 40 mm from where it was sent, letting the slugs go outside the rim. The
# fingers close the same way at -90, with the wrist well inside its range.
SLUG_YAW = -math.pi / 2


def drop_slug(slot: int) -> list:
    x, y = RACK_X, RACK_Y0 + slot * RACK_DY
    top = SLUG_Z + SLUG_H / 2
    grip = top - GRIP_DOWN
    over = (x, y, grip + 0.08)
    return [
        Waypoint(2.0, over, HALF_OPEN, "over a slug", SLUG_YAW),
        Waypoint(1.2, (x, y, grip), HALF_OPEN, "down to it", SLUG_YAW),
        Waypoint(1.6, (x, y, grip), FINGER_CLOSED, "grip", SLUG_YAW),
        Waypoint(1.8, (x, y, LIFT_Z), FINGER_CLOSED, "lift", SLUG_YAW),
        Waypoint(3.0, (*DROP_XY, LIFT_Z), FINGER_CLOSED, "over the tube", SLUG_YAW),
        Waypoint(1.2, (*DROP_XY, DROP_Z), FINGER_CLOSED, "down to the rim", SLUG_YAW),
        Waypoint(1.0, (*DROP_XY, DROP_Z), FINGER_CLOSED, "steady", SLUG_YAW),
        Waypoint(1.6, (*DROP_XY, DROP_Z), HALF_OPEN, "drop it", SLUG_YAW),
        Waypoint(0.8, (*DROP_XY, LIFT_Z + 0.02), HALF_OPEN, "up", SLUG_YAW),
    ]


def _t(w) -> float:
    return sum(p.duration for p in w)


def _wait(w: list, seconds: float, label: str = "wait") -> Waypoint:
    last = w[-1]
    return Waypoint(seconds, last.pos, last.finger, label, last.yaw)


def demo():
    """The scripted demo, or a shortcut check: ``WATER_DEMO=cork`` drops the two cork slugs instead (they float and
    hardly raise the water), ``WATER_DEMO=reach`` drops nothing and reaches into the tube for the tag."""
    mode = os.environ.get("WATER_DEMO", "")
    left = [Waypoint(2.0, LEFT_HOME, HALF_OPEN, "ready", SLUG_YAW)]
    slots = {"cork": CORK_SLOTS, "reach": []}.get(mode, METAL_SLOTS[:3])
    for slot in slots:
        left += drop_slug(slot)
    left += [Waypoint(2.0, LEFT_HOME, HALF_OPEN, "clear", SLUG_YAW)]
    tx, ty = TUBE_XY[0] - TAG_R, TUBE_XY[1]
    pinch_z = TOP + RIM + 15.0 * MM  # fingertips 2 mm above the rim, round the top 10 mm of the raised tag
    if mode == "reach":  # as deep as the fingers go into the bore, slightly open round where the tag would be
        pinch_z = TOP + RIM - 12.0 * MM
    out_z = TOP + RIM + 45.0 * MM  # the float's bottom clear of the rim
    opening = FINGER_OPEN * 0.25 if mode == "reach" else FINGER_OPEN
    right = [Waypoint(2.0, RIGHT_HOME, FINGER_OPEN, "ready", math.pi / 2)]
    right.append(_wait(right, _t(left) - _t(right), "wait for the water"))
    right += [
        Waypoint(2.0, (tx, ty, TOP + RIM + 40.0 * MM), opening, "over the tag", math.pi / 2),
        Waypoint(1.2, (tx, ty, pinch_z), opening, "down to it", math.pi / 2),
        Waypoint(1.6, (tx, ty, pinch_z), FINGER_CLOSED, "pinch the tag", math.pi / 2),
        Waypoint(2.0, (tx, ty, out_z), FINGER_CLOSED, "lift the float out", math.pi / 2),
        Waypoint(2.5, (*TRAY_XY, out_z), FINGER_CLOSED, "over the tray", math.pi / 2),
        Waypoint(1.5, (*TRAY_XY, TOP + 0.055), FINGER_CLOSED, "lower it", math.pi / 2),
        Waypoint(1.6, (*TRAY_XY, TOP + 0.055), FINGER_OPEN, "let go", math.pi / 2),
        Waypoint(1.2, (*TRAY_XY, TOP + 0.12), FINGER_OPEN, "up", math.pi / 2),
    ]
    left.append(_wait(left, max(0.0, _t(right) - _t(left)), "done"))
    return {"left": left, "right": right}
