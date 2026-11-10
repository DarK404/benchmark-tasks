"""On the station table: a small machinist's vise 24 cm in front of the arm bases, on the midline, clamps a mild steel
block whose top (60 x 16 mm, along the table's y axis) stands 1 mm above a line scribed round it. A 150 mm flat file
with a wooden handle lies on a wooden rest to the right, along y, its handle towards the right arm.

The robot must file the block's top down to the line, keeping it flat, and put the file back on its rest.

Run it:  inspect-robots run --embodiment yam_arena -E task=file_flat.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "..." --scorer task_score ...
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import filekit  # noqa: E402
from filekit import Filing, body, box, cyl  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

filekit.install()
SCENE = "station"
TOP = TABLE_TOP[SCENE]
MM = 0.001
CAST_IRON = (0.24, 0.31, 0.40)  # the vise, painted blue-grey
MACHINED = (0.64, 0.65, 0.67)
CHROME = (0.82, 0.82, 0.84)
RUBBER_MAT = (0.12, 0.12, 0.13)
MILL_SCALE = (0.33, 0.34, 0.36)  # the bar as it comes, below the jaws
MARKING_BLUE = (0.10, 0.20, 0.60)  # layout fluid on the faces above the jaws
BRIGHT = (0.84, 0.85, 0.87)  # freshly filed steel, and the scribed line through the blue
FILE_STEEL = (0.20, 0.21, 0.23)
TEETH = (0.07, 0.07, 0.08)
BRASS = (0.78, 0.60, 0.25)
BEECH = (0.70, 0.50, 0.30)  # the file's handle
WOOD = (0.55, 0.40, 0.25)
filekit.METALLIC.update({MACHINED: 0.3, CHROME: 0.15, BRIGHT: 0.25, FILE_STEEL: 0.55, BRASS: 0.3})

# -- the vise and the block (fixed; frame: origin on the table under the block's centre) -------------------------
# A small machinist's vise (75 mm jaws) on a rubber mat, its screw and T-bar handle towards the robot, clamping the
# bar across x; the bar's top runs along y, the way it is filed.
VISE_XY = (0.24, 0.0)
BLOCK_W, BLOCK_L = 16.0 * MM, 60.0 * MM  # across (x) and along (y) the stroke
LINE = 85.0 * MM  # the scribed line, above the table
H0 = LINE + float(os.environ.get("FILE_STOCK", "1.0")) * MM  # the block's top to start
# WHY 8 mm (2026-11-10, sim trace): pressed hard (16 N), the drawn file sank past a solid part 2.5 mm down and stood
# on the simulator's contact there, so pressing harder cut no faster; at 8 mm our springs carry it all.
SOLID = LINE - 8.0 * MM  # the block collides up to here; above, its top is our model's
JAW_TOP = LINE - 10.0 * MM
SLIDE_TOP = 45.0 * MM


def _vise() -> tuple[list, int]:
    """(shapes, how many collide): the castings and the bar collide; jaw plates, screw, handle, mat and the scribed
    line are seen only."""
    hb = BLOCK_W / 2
    jaw_h = JAW_TOP - SLIDE_TOP
    collide = [
        box((-5.0 * MM, 0, 7.5 * MM), (120.0 * MM, 90.0 * MM, 15.0 * MM), CAST_IRON),  # base
        box((-10.0 * MM, 0, 30.0 * MM), (180.0 * MM, 50.0 * MM, 30.0 * MM), CAST_IRON),  # slide
        box((hb + 12.5 * MM, 0, SLIDE_TOP + jaw_h / 2), (25.0 * MM, 90.0 * MM, jaw_h), CAST_IRON),  # fixed jaw
        box((-hb - 12.5 * MM, 0, SLIDE_TOP + jaw_h / 2), (25.0 * MM, 90.0 * MM, jaw_h), CAST_IRON),  # moving jaw
        box((0.0, 0.0, (SLIDE_TOP + SOLID) / 2), (BLOCK_W, BLOCK_L, SOLID - SLIDE_TOP), MILL_SCALE),  # the bar
    ]
    look = [
        box((hb + 1.5 * MM, 0, SLIDE_TOP + jaw_h / 2), (3.0 * MM, 76.0 * MM, jaw_h), MACHINED),  # jaw plates
        box((-hb - 1.5 * MM, 0, SLIDE_TOP + jaw_h / 2), (3.0 * MM, 76.0 * MM, jaw_h), MACHINED),
        cyl((-hb - 25.0 * MM - 32.0 * MM, 0, 35.0 * MM), 8.0 * MM, 64.0 * MM, MACHINED, axis="X"),  # screw
        cyl((-hb - 25.0 * MM - 64.0 * MM, 0, 35.0 * MM), 5.0 * MM, 120.0 * MM, CHROME, axis="Y"),  # T-bar handle
        cyl((-hb - 25.0 * MM - 64.0 * MM, +60.0 * MM, 35.0 * MM), 8.0 * MM, 10.0 * MM, CHROME, axis="Y"),
        cyl((-hb - 25.0 * MM - 64.0 * MM, -60.0 * MM, 35.0 * MM), 8.0 * MM, 10.0 * MM, CHROME, axis="Y"),
        box((-20.0 * MM, 0, 0.75 * MM), (280.0 * MM, 170.0 * MM, 1.5 * MM), RUBBER_MAT),  # mat
        # the scribed line, bright through the marking blue, round the bar
        box((-(hb + 0.15 * MM), 0.0, LINE - 0.15 * MM), (0.3 * MM, BLOCK_L, 0.3 * MM), BRIGHT),
        box((hb + 0.15 * MM, 0.0, LINE - 0.15 * MM), (0.3 * MM, BLOCK_L, 0.3 * MM), BRIGHT),
        box((0.0, -(BLOCK_L / 2 + 0.15 * MM), LINE - 0.15 * MM), (BLOCK_W, 0.3 * MM, 0.3 * MM), BRIGHT),
        box((0.0, BLOCK_L / 2 + 0.15 * MM, LINE - 0.15 * MM), (BLOCK_W, 0.3 * MM, 0.3 * MM), BRIGHT),
    ]
    return collide + look, len(collide)


_vise_shapes, _vise_n = _vise()
VISE = body("vise", pos=(*VISE_XY, TOP), friction=0.6, shapes=_vise_shapes,
            visual_only=tuple(range(_vise_n, len(_vise_shapes))))

# the top above SOLID: four sections along the stroke, each drawn as a blued box with a bright filed face, its top
# set by the filing model
N_SECTIONS = 4
SEC_L = BLOCK_L / N_SECTIONS
SECTION_Y = [-BLOCK_L / 2 + (i + 0.5) * SEC_L for i in range(N_SECTIONS)]
SURF_DEPTH = 0.05
FACE = 0.4 * MM
SURFACES = [body(f"top{i}", pos=(VISE_XY[0], VISE_XY[1] + y, TOP + H0), mass=0.01, visual_only=(0, 1),
                 inertia=(1e-5,) * 3,
                 shapes=[box((0.0, 0.0, -FACE / 2), (BLOCK_W - 0.2 * MM, SEC_L - 0.05 * MM, FACE), BRIGHT),
                         box((0.0, 0.0, -FACE - SURF_DEPTH / 2), (BLOCK_W - 0.2 * MM, SEC_L - 0.05 * MM, SURF_DEPTH),
                             MARKING_BLUE)])
            for i, y in enumerate(SECTION_Y)]
for s in SURFACES:
    filekit.no_gravity(s.name)

# -- the file (origin at the middle of the blade's flat bottom; +y towards the tip) ------------------------------
BLADE_L, BLADE_W, BLADE_T = 150.0 * MM, 20.0 * MM, 5.0 * MM
HANDLE_L, HANDLE_W = 80.0 * MM, 20.0 * MM  # the handle collides as a 20 mm square bar inside the round one seen
HANDLE_Z = 6.0 * MM  # the handle's axis above the blade's bottom
HANDLE_Y = -(BLADE_L / 2 + HANDLE_L / 2)
REST_XY = (0.17, -0.14)  # the blade's middle when the file lies on its rest
REST_H = 30.0 * MM
CUT_ANGLE = math.radians(25.0)  # a single-cut file's teeth, slanted across the blade


def _file_shapes() -> list:
    collide = [
        box((0.0, 0.0, BLADE_T / 2), (BLADE_W, BLADE_L, BLADE_T), FILE_STEEL),
        box((0.0, HANDLE_Y, HANDLE_Z), (HANDLE_W, HANDLE_L, HANDLE_W), BEECH),
    ]
    look = [
        cyl((0.0, HANDLE_Y, HANDLE_Z), 14.5 * MM, HANDLE_L + 0.5 * MM, BEECH, axis="Y"),  # round handle
        cyl((0.0, -BLADE_L / 2 - 4.0 * MM, HANDLE_Z), 14.8 * MM, 8.0 * MM, BRASS, axis="Y"),  # ferrule
    ]
    n = int(BLADE_L / (2.5 * MM)) - 2
    for k in range(n):  # the teeth, drawn on the blade's top
        y = -BLADE_L / 2 + (k + 1.5) * 2.5 * MM
        look.append(box((0.0, y, BLADE_T + 0.05 * MM), (BLADE_W * 0.9, 0.6 * MM, 0.1 * MM), TEETH, yaw=CUT_ANGLE))
    return collide + look


_file = _file_shapes()
FILE = body("file", pos=(*REST_XY, TOP + REST_H), mass=0.14, friction=0.6, com=(0.0, -20.0 * MM, 3.0 * MM),
            shapes=_file, visual_only=tuple(range(2, len(_file))))
REST = body("rest", pos=(*REST_XY, TOP), friction=0.6, shapes=[
    box((0.0, -45.0 * MM, REST_H / 2), (0.04, 0.02, REST_H), WOOD),
    box((0.0, 45.0 * MM, REST_H / 2), (0.04, 0.02, REST_H), WOOD),
])

OBJECTS = [VISE, *SURFACES, FILE, REST]
filekit.register(*OBJECTS)

# -- the cutting model -------------------------------------------------------------------------------------------
SECTIONS = [(VISE_XY[0] - BLOCK_W / 2, VISE_XY[0] + BLOCK_W / 2, VISE_XY[1] + y - SEC_L / 2, VISE_XY[1] + y + SEC_L / 2)
            for y in SECTION_Y]
FILING = filekit.add_model(Filing("file", (-BLADE_L / 2, BLADE_L / 2), BLADE_W / 2, SECTIONS, TOP + H0,
                                  k=float(os.environ.get("FILE_K", "2000")), damping=6.0,
                                  mu=0.5, wear=float(os.environ.get("FILE_WEAR", "6e-4")),
                                  surfaces=[s.name for s in SURFACES], floor=TOP + SOLID))

# -- scoring -------------------------------------------------------------------------------------------------------
# Measured as on a real bench: the top against the scribed line (a straightedge across it, a feeler gauge), its
# flatness (the straightedge along it), and the file on its rest.
SCORE_OVER = "end"
RUN_LIMIT_STEPS = 1600
CLEAR = 10.0 * MM
OVERCUT = 0.05 * MM  # any section this far below the line: the part is spoilt (latched)
FINISHED = 0.20 * MM  # every section at most this far above the line
FLAT = 0.15 * MM  # highest and lowest sections within this
NEARLY = 0.40 * MM
STARTED = 0.30 * MM  # removed at least this much everywhere
REST_TOL = 25.0 * MM


def heights() -> np.ndarray:
    """Each section's height above the line (m)."""
    return FILING.h - (TOP + LINE)


def _on_rest(state) -> bool:
    off = np.hypot(*(np.asarray(state.pos[:2]) - REST_XY))
    return bool(off < REST_TOL and abs(state.pos[2] - (TOP + REST_H)) < 5.0 * MM and state.tilt_deg() < 5.0)


_STATE: dict = {}


def _reset_state() -> None:
    _STATE.update(spoilt=False)


_reset_state()
filekit.on_reset(_reset_state)


def fingers_to(state, spec) -> float:
    """Nearest distance (m) from either gripper's hulls to a body's bounding box (in its own frame). inf offline."""
    pts = filekit.finger_points()
    if not len(pts):
        return math.inf
    lo, hi = spec.bounds_local()
    local = (pts - state.pos) @ state.rotation()
    d = np.maximum(np.maximum(lo - local, local - hi), 0.0)
    return float(np.linalg.norm(d, axis=1).min())


def _update(objects) -> tuple[float, bool]:
    h = heights()
    _STATE["spoilt"] |= bool(h.min() < -OVERCUT)
    if _STATE["spoilt"]:
        return 0.0, False
    f = objects["file"]
    if h.max() <= FINISHED and h.max() - h.min() <= FLAT:
        if _on_rest(f) and fingers_to(f, FILE) >= CLEAR:
            return 1.0, bool(f.resting())
        return 0.5, False
    if h.max() <= NEARLY:
        return 0.5, False
    removed = (H0 - LINE) - h.max()  # H0 and LINE are heights above the table
    return (0.25 if removed >= STARTED else 0.0), False


def success(objects) -> bool:
    """The top within 0.2 mm above the line everywhere and flat within 0.15 mm, never below it, the file back on its
    rest, released and still for 1 s."""
    return _update(objects)[1]


def score(objects) -> float:
    """1.0 / 0.5 / 0.25 / 0 as in the README; filing below the line latches 0."""
    return _update(objects)[0]


# -- scripted demo -------------------------------------------------------------------------------------------------
RIGHT_HOME = (0.16, -0.22, TOP + 0.15)
LEFT_HOME = (0.16, 0.22, TOP + 0.15)
GRIP_YAW = math.pi / 2  # right fingers close across the handle (along x)
# WHY the left hand pinches the tip (2026-11-10, sim trace): pressing down on the blade with closed fingertips, the
# file (turned 4 degrees in the right grip as it was picked up) slid out from under them; then only the handle hand
# pushed, the file pivoted in its grip and cut the near end alone. Pinched across the blade near its tip (always
# beyond the block's far end, so the fingers clear the work), the file is held straight and pushed down through the
# grip.
PRESS_Y = 68.0 * MM  # where the left fingers pinch the blade, from its middle
TIP_YAW = -math.pi / 2  # left fingers close across the blade (along x); -90 keeps the left wrist off its limit
TIP_PAST_TCP = 13.0 * MM
STROKE = (-40.0 * MM, 40.0 * MM)  # the blade's middle relative to the block's centre, along y
# Each stroke's push (N, both hands together): three full strokes, then two light ones to finish near the line, as
# a fitter eases off. Measured (2026-11-10): an 8 N stroke takes about 0.21 mm off, evenly.
PUSHES = [float(v) for v in os.environ.get("FILE_PUSHES", "8,8,8,4,4").split(",")]
# Shortcut checks (FILE_DEMO): "nostop" keeps filing with full strokes instead of easing off near the line;
# "onehand" works the file by its handle alone, the tip not held.
DEMO_MODE = os.environ.get("FILE_DEMO", "")
if DEMO_MODE == "nostop":
    PUSHES = [8.0] * 7
PUSH = float(os.environ.get("FILE_PUSH", "8")) * 1.0  # N, both hands together
ARM_K = 0.23  # N per mm a hand pushes for each mm it is sent below the contact (simulated arms are soft)
# WHY the tip hand goes further down (2026-11-10, sim traces): sent down equally, the arms pushed unequally in this
# pose, the file pitched 2.5 degrees tip-up and cut the handle end five times faster. With the tip hand sent 16 mm
# further down (at 8 N), three strokes cut all four sections within 0.01 mm of each other.
TILT = float(os.environ.get("FILE_TILT", "16")) * MM  # at 8 N; scaled with the push
def leads(blade_mid: float, push: float = None) -> tuple[float, float]:
    """(handle, tip) leads in m: how far below the contact each hand is sent, for ``push`` N in all. Both ends of
    the file are held, so equal leads keep it level and its pressure even along the block (``blade_mid``, the
    blade's middle along y from the block's centre, is kept for a pressure shift, which the held file does not
    need)."""
    push = PUSH if push is None else push
    base, tilt = push / 2 / ARM_K * MM, TILT * push / 8.0
    return base - tilt / 2, base + tilt / 2


def handle_tcp(blade_x: float, blade_y: float, blade_z: float) -> tuple:
    return (blade_x, blade_y + HANDLE_Y, blade_z + HANDLE_Z)


def press_tcp(blade_x: float, blade_y: float, blade_z: float) -> tuple:
    """The left TCP pinching the blade near its tip: the fingertips 2 mm below the blade's bottom."""
    return (blade_x, blade_y + PRESS_Y, blade_z + TIP_PAST_TCP - 2.0 * MM)


def _w(dur, pos, finger, label, yaw):
    return Waypoint(dur, tuple(pos), finger, label, yaw)


def _one_hand(right: list) -> dict:
    """The shortcut check "onehand": the right arm alone files, pushing the handle down with the whole push."""
    bx, by = VISE_XY
    top = TOP + H0
    for push in PUSHES:
        lead = push / ARM_K * MM
        right.append(_w(0.6, handle_tcp(bx, by + STROKE[0], top - lead), FINGER_CLOSED, "bear down", GRIP_YAW))
        right.append(_w(2.0, handle_tcp(bx, by + STROKE[1], top - lead), FINGER_CLOSED, "push stroke", GRIP_YAW))
        right.append(_w(0.5, handle_tcp(bx, by + STROKE[1], top + 0.015), FINGER_CLOSED, "lift", GRIP_YAW))
        right.append(_w(1.2, handle_tcp(bx, by + STROKE[0], top + 0.015), FINGER_CLOSED, "back", GRIP_YAW))
    rx, ry = REST_XY
    rest_z = TOP + REST_H
    right += [
        _w(1.5, handle_tcp(bx, by + STROKE[0], top + 0.04), FINGER_CLOSED, "lift the file", GRIP_YAW),
        _w(2.0, handle_tcp(rx, ry, rest_z + 0.03), FINGER_CLOSED, "over the rest", GRIP_YAW),
        _w(1.0, handle_tcp(rx, ry, rest_z + 0.003), FINGER_CLOSED, "onto it", GRIP_YAW),
        _w(1.6, handle_tcp(rx, ry, rest_z + 0.003), FINGER_OPEN, "let go", GRIP_YAW),
        _w(1.0, handle_tcp(rx, ry, rest_z + 0.06), FINGER_OPEN, "up", GRIP_YAW),
    ]
    return {"right": right}


def demo():
    bx, by = VISE_XY
    top = TOP + H0
    rx, ry = REST_XY
    rest_z = TOP + REST_H
    right = [
        _w(2.0, (rx, ry + HANDLE_Y, rest_z + HANDLE_Z + 0.06), FINGER_OPEN, "over the handle", GRIP_YAW),
        _w(1.2, handle_tcp(rx, ry, rest_z), FINGER_OPEN, "down to it", GRIP_YAW),
        _w(1.6, handle_tcp(rx, ry, rest_z), FINGER_CLOSED, "grip", GRIP_YAW),
        _w(1.2, handle_tcp(rx, ry, rest_z + 0.04), FINGER_CLOSED, "lift", GRIP_YAW),
        _w(2.0, handle_tcp(bx, by + STROKE[0], top + 0.02), FINGER_CLOSED, "over the block", GRIP_YAW),
        _w(1.0, handle_tcp(bx, by + STROKE[0], top + 0.005), FINGER_CLOSED, "onto it", GRIP_YAW),
    ]
    if DEMO_MODE == "onehand":
        return _one_hand(right)
    left = [_w(2.0, LEFT_HOME, FINGER_OPEN, "ready", TIP_YAW)]
    left.append(_w(sum(w.duration for w in right) - 2.0 - 1.0, press_tcp(bx, by + STROKE[0], top + 0.05),
                   FINGER_OPEN, "over the tip", TIP_YAW))
    left.append(_w(1.0, press_tcp(bx, by + STROKE[0], top), FINGER_OPEN, "down to the tip", TIP_YAW))
    left.append(_w(1.6, press_tcp(bx, by + STROKE[0], top), FINGER_CLOSED, "pinch the tip", TIP_YAW))
    right.append(_w(1.6, right[-1].pos, FINGER_CLOSED, "wait for the pinch", GRIP_YAW))
    steps = 4
    for push in PUSHES:
        lr, ll = leads(STROKE[0], push)
        right.append(_w(0.6, handle_tcp(bx, by + STROKE[0], top - lr), FINGER_CLOSED, "bear down", GRIP_YAW))
        left.append(_w(0.6, press_tcp(bx, by + STROKE[0], top - ll), FINGER_CLOSED, "bear down", TIP_YAW))
        for j in range(1, steps + 1):
            ym = STROKE[0] + (STROKE[1] - STROKE[0]) * j / steps
            lr, ll = leads(ym, push)
            right.append(_w(2.0 / steps, handle_tcp(bx, by + ym, top - lr), FINGER_CLOSED, "push stroke", GRIP_YAW))
            left.append(_w(2.0 / steps, press_tcp(bx, by + ym, top - ll), FINGER_CLOSED, "push stroke", TIP_YAW))
        for arm, f, yaw in ((right, handle_tcp, GRIP_YAW), (left, press_tcp, TIP_YAW)):
            arm.append(_w(0.5, f(bx, by + STROKE[1], top + 0.015), FINGER_CLOSED, "lift", yaw))
            arm.append(_w(1.2, f(bx, by + STROKE[0], top + 0.015), FINGER_CLOSED, "back", yaw))
    left.append(_w(1.6, left[-1].pos, FINGER_OPEN, "let go of the tip", TIP_YAW))
    right.append(_w(1.6, right[-1].pos, FINGER_CLOSED, "wait", GRIP_YAW))
    left.append(_w(1.5, LEFT_HOME, FINGER_OPEN, "clear", TIP_YAW))
    right += [
        _w(1.5, handle_tcp(bx, by + STROKE[0], top + 0.04), FINGER_CLOSED, "lift the file", GRIP_YAW),
        _w(2.0, handle_tcp(rx, ry, rest_z + 0.03), FINGER_CLOSED, "over the rest", GRIP_YAW),
        _w(1.0, handle_tcp(rx, ry, rest_z + 0.003), FINGER_CLOSED, "onto it", GRIP_YAW),
        _w(1.6, handle_tcp(rx, ry, rest_z + 0.003), FINGER_OPEN, "let go", GRIP_YAW),
        _w(1.0, handle_tcp(rx, ry, rest_z + 0.06), FINGER_OPEN, "up", GRIP_YAW),
    ]
    tl, tr = sum(w.duration for w in left), sum(w.duration for w in right)
    left.append(_w(max(0.1, tr - tl), LEFT_HOME, FINGER_OPEN, "done", TIP_YAW))
    return {"left": left, "right": right}
