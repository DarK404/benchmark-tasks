"""On the station table: a Square D Homeline 4-space load center (unwired) lies on its back, its dead front off, its
top away from the robot. Spaces 1 and 3 (the left-hand column) hold breakers; spaces 2 and 4 (the right-hand column)
are empty. A new Homeline HOM120 breaker (1-pole, 20 A, OFF) stands on its back on the table to the right.

The robot must install the new breaker in space 4, as the maker says: its load (wire terminal) end hooked under the
mounting rail at the outer edge, then the line end pushed in until its plug-on jaw is fully on the bus stab. A
breaker left part way on its stab arcs and overheats; one not hooked on the rail works loose.

Run it:  inspect-robots run --embodiment yam_arena -E task=breaker.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "Install the new breaker in space 4 of the unpowered, unwired load center." ...
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import breakerkit  # noqa: E402
from breakerkit import Spring, body, box, tilted  # noqa: E402
from yam_arena.analysis import box_signed_distance  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

breakerkit.install()
SCENE = "station"
TOP = TABLE_TOP[SCENE]
MM = 0.001
GREY = (0.62, 0.63, 0.64)  # the enclosure (painted steel)
BASE_GREY = (0.22, 0.23, 0.25)  # the interior's plastic base
TINNED = (0.78, 0.76, 0.70)  # the bus and its stabs (tin-plated)
BLACK = (0.08, 0.08, 0.09)  # breaker housings
WHITE = (0.92, 0.92, 0.90)
breakerkit.METALLIC.update({TINNED: 0.35})

# -- sizes (env frame: x forward, y left, z up; the table top at TOP) ------------------------------------------------
# The load center: Square D HOM48L125GC, 376 x 226 x 97 mm (14.8 x 8.9 x 3.8 in), on its back, its top (main lugs)
# at +x. Seen from the robot its left-hand column (odd spaces) is at +y, the right-hand one (even) at -y.
PX, PY = 0.32, -0.06  # the middle of the enclosure (and of the bus)
ENC_L, ENC_W, ENC_D, WALL = 376.0 * MM, 226.0 * MM, 97.0 * MM, 1.6 * MM
ZP = 30.0 * MM  # the interior base's top (the breakers' mounting face), above the table
PITCH = 25.4 * MM  # Homeline: one space is 1 in
ROW_X = {1: PX + PITCH / 2, 2: PX - PITCH / 2}  # rows 1 (spaces 1, 2) and 2 (spaces 3, 4)

# The breaker: Square D Homeline HOM120 (1-pole 20 A), 25.4 W x 79.5 H x 75.7 D mm (1 x 3.13 x 2.98 in). In its own
# frame: origin at the middle of its mounting face, +y to its line (jaw) end, z up from the mounting face.
BW, BL = 25.4 * MM, 79.5 * MM
B_BODY = 60.0 * MM  # the housing's height; the face and the handle stand above it
B_FACE = 6.0 * MM  # the raised face, 40 mm long
B_HANDLE = 8.0 * MM  # the handle above the face (OFF: towards the load end)
B_MASS = 0.11  # kg, our estimate (Square D lists only the boxed weight, 0.28-0.32 kg)
SHOE_MASS = 0.006
GAP = 1.0 * MM  # each jaw shoe's inner face from the middle, at rest (the jaw's free gap: 2.0 mm)
# the jaw slot in the line end (x, y, z in the breaker's frame)
SLOT_X, SLOT_Y0, SLOT_Y1, SLOT_Z = 6.0 * MM, 24.0 * MM, 37.0 * MM, 20.0 * MM
SHOE_T, SHOE_Y0, SHOE_Y1, SHOE_Z0, SHOE_Z1 = 4.0 * MM, 25.5 * MM, 36.5 * MM, 1.0 * MM, 19.0 * MM
FLARE = 2.0 * MM  # the shoes' flared mouth: 45 degrees, 2 mm out from their inner faces (it catches a stab off-centre)
# the load end's hook: a foot under a 4.5 mm high slot that takes the mounting rail's lip
FOOT_Z, HOOK_Z1, HOOK_Y1 = 3.5 * MM, 8.0 * MM, -35.0 * MM  # the slot spans z FOOT_Z..HOOK_Z1, y -BL/2..HOOK_Y1

# The bus stab (in the frame of the breaker seated in its space): a 3.2 mm tinned copper blade, 10 mm long under the
# middle of the jaw slot, standing STAB_H up from the base, its top tapered to a ridge (the lead-in) whose faces lean
# RIDGE_DEG from the vertical.
STAB_T, STAB_Y0, STAB_Y1, STAB_H = 3.2 * MM, 26.0 * MM, 36.0 * MM, 11.0 * MM
RIDGE_DEG = float(os.environ.get("BK_RIDGE", "15"))
RIDGE_H = (STAB_T / 2) / math.tan(math.radians(RIDGE_DEG))
STAB_TOP = STAB_H + RIDGE_H  # the ridge's crest
# The mounting rail (the same frame): a plastic bar along the outer edge with a lip reaching 4.5 mm in at 4.5-7.0 mm
RAIL_Y0, RAIL_Y1, RAIL_H = -46.0 * MM, -40.5 * MM, 10.0 * MM
LIP_Y1, LIP_Z0, LIP_Z1 = -36.0 * MM, 4.5 * MM, 7.0 * MM

# The jaw's spring: each shoe is held against its stop (2.0 mm apart) with JAW_PRELOAD and is pushed apart 0.6 mm
# by the stab, so the jaw grips the stab with about 14 N a side; with JAW_MU between the tinned faces it slides on
# against about 2 x 0.3 x 14 = 9 N, and up the ridge's faces (15 degrees) against about 18 N: our estimate (Square D
# publishes no seating force; the instructions say to push the breaker in until the jaws plug on securely).
JAW_K = 4000.0  # N/m
JAW_PRELOAD = float(os.environ.get("BK_PRELOAD", "12"))  # N
JAW_TRAVEL = 3.0 * MM
JAW_MU = 0.3

TARGET = 4  # the space to install it in
SPACES = {1: (ROW_X[1], +1), 2: (ROW_X[1], -1), 3: (ROW_X[2], +1), 4: (ROW_X[2], -1)}  # x, column side (+1 left)
SEAT_Y = BL / 2 + 5.0 * MM  # a seated breaker's middle from the bus's middle (line ends 10 mm apart over the bus)


def seat_pose(space: int) -> tuple[np.ndarray, float]:
    """The pose (origin, yaw) of a breaker seated in ``space``: its line end towards the bus."""
    x, side = SPACES[space]
    yaw = 0.0 if side < 0 else math.pi  # right-hand column: +y (its line end) points at the bus
    return np.array([x, PY + side * SEAT_Y, TOP + ZP]), yaw


def _rot(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _breaker_housing(colour=BLACK, jaw_slot: bool = True) -> list:
    """The breaker's housing shapes in its own frame (a seated neighbour has no jaw slot to model)."""
    h, w, l2 = B_BODY, BW, BL / 2
    out = []
    # the main block, from the hook's back wall to the jaw slot
    out.append(box((0.0, (HOOK_Y1 + SLOT_Y0) / 2, h / 2), (w, SLOT_Y0 - HOOK_Y1, h), colour))
    # the load end: the foot, and the block above the hook's slot
    out.append(box((0.0, (-l2 + HOOK_Y1) / 2, FOOT_Z / 2), (w, HOOK_Y1 + l2, FOOT_Z), colour))
    out.append(box((0.0, (-l2 + HOOK_Y1) / 2, (HOOK_Z1 + h) / 2), (w, HOOK_Y1 + l2, h - HOOK_Z1), colour))
    if jaw_slot:
        # the line end round the jaw slot: the block above it, its two sides and its end wall
        out.append(box((0.0, (SLOT_Y0 + l2) / 2, (SLOT_Z + h) / 2), (w, l2 - SLOT_Y0, h - SLOT_Z), colour))
        for s in (-1.0, 1.0):
            out.append(box((s * (SLOT_X + w / 2) / 2, (SLOT_Y0 + l2) / 2, SLOT_Z / 2),
                           (w / 2 - SLOT_X, l2 - SLOT_Y0, SLOT_Z), colour))
        out.append(box((0.0, (SLOT_Y1 + l2) / 2, SLOT_Z / 2), (2 * SLOT_X, l2 - SLOT_Y1, SLOT_Z), colour))
    else:
        out.append(box((0.0, (SLOT_Y0 + l2) / 2, h / 2), (w, l2 - SLOT_Y0, h), colour))
    # the raised face and the handle (OFF: thrown towards the load end), and its white rating label
    out.append(box((0.0, 0.0, h + B_FACE / 2), (w, 40.0 * MM, B_FACE), colour))
    out.append(box((0.0, -4.0 * MM, h + B_FACE + B_HANDLE / 2), (8.0 * MM, 9.0 * MM, B_HANDLE), colour))
    out.append(box((0.0, 13.0 * MM, h + B_FACE + 0.2 * MM), (14.0 * MM, 8.0 * MM, 0.4 * MM), WHITE))
    return out


def _in_panel(shapes, space: int) -> list:
    """Shapes given in the frame of a breaker seated in ``space``, moved into the panel's frame."""
    origin, yaw = seat_pose(space)
    off = origin - np.array([PX, PY, TOP])
    r = _rot(yaw)
    out = []
    for s in shapes:
        c = off + r @ np.asarray(s.centre)
        out.append(box(tuple(c), s.size, s.color, yaw=s.yaw + yaw, tilt_y=s.tilt_y))
    return out


def _space_fittings(space: int) -> list:
    """The stab and the rail section of ``space``, in the seated breaker's frame."""
    ym, ly = (STAB_Y0 + STAB_Y1) / 2, STAB_Y1 - STAB_Y0
    out = [box((0.0, ym, STAB_H / 2), (STAB_T, ly, STAB_H), TINNED)]
    # the ridge: two thin plates leaning together, their outer faces from the blade's edges up to the crest
    t, slope = 0.8 * MM, math.hypot(STAB_T / 2, RIDGE_H)
    for s in (-1.0, 1.0):
        n = np.array([s * RIDGE_H, STAB_T / 2]) / slope  # the face's outward normal (x, z)
        cx, cz = np.array([s * STAB_T / 4, STAB_H + RIDGE_H / 2]) - n * t / 2
        out.append(box((cx, ym, cz), (t, ly, slope), TINNED, tilt_y=-s * math.atan2(STAB_T / 2, RIDGE_H)))
    return out + [
            box((0.0, (RAIL_Y0 + RAIL_Y1) / 2, RAIL_H / 2), (PITCH, RAIL_Y1 - RAIL_Y0, RAIL_H), BASE_GREY),
            box((0.0, (RAIL_Y1 + LIP_Y1) / 2, (LIP_Z0 + LIP_Z1) / 2), (PITCH, LIP_Y1 - RAIL_Y1, LIP_Z1 - LIP_Z0),
                BASE_GREY)]


N_STAB_SHAPES = 3  # the first three fittings of each space are its stab (own friction)


def _panel_shapes() -> tuple[list, list]:
    """The enclosure, the interior base, the bus, and each space's stab and rail; the seated breakers of spaces 1
    and 3. Returns (shapes, indices of the stab shapes)."""
    L, W, D, t = ENC_L, ENC_W, ENC_D, WALL
    out = [box((0.0, 0.0, t / 2), (L, W, t), GREY),
           box((L / 2 - t / 2, 0.0, D / 2), (t, W, D), GREY), box((-L / 2 + t / 2, 0.0, D / 2), (t, W, D), GREY),
           box((0.0, W / 2 - t / 2, D / 2), (L, t, D), GREY), box((0.0, -W / 2 + t / 2, D / 2), (L, t, D), GREY),
           # the interior's plastic base: the breakers' mounting face at ZP
           box((-10.0 * MM, 0.0, (t + ZP) / 2), (150.0 * MM, 200.0 * MM, ZP - t), BASE_GREY),
           # the main lugs at the top, and the neutral bars along both sides (seen, not touched)
           box((110.0 * MM, 0.0, ZP + 12.0 * MM), (40.0 * MM, 60.0 * MM, 24.0 * MM), TINNED),
           box((-30.0 * MM, 95.0 * MM, ZP + 6.0 * MM), (110.0 * MM, 10.0 * MM, 12.0 * MM), TINNED),
           box((-30.0 * MM, -95.0 * MM, ZP + 6.0 * MM), (110.0 * MM, 10.0 * MM, 12.0 * MM), TINNED)]
    stab_idx = []
    for space in (1, 2, 3, 4):
        k = len(out)
        out += _in_panel(_space_fittings(space), space)
        stab_idx += [k + i for i in range(N_STAB_SHAPES)]
    for space in (1, 3):
        out += _in_panel(_breaker_housing(jaw_slot=False), space)
    return out, stab_idx


_PANEL_SHAPES, STAB_SHAPES = _panel_shapes()
PANEL = body("load_center", pos=(PX, PY, TOP), shapes=_PANEL_SHAPES, friction=0.4, contact_offset=0.0005)
for _i in STAB_SHAPES:
    breakerkit.shape_friction(PANEL.name, _i, JAW_MU)

# the new breaker: on its back on the table, its line end pointing away from the robot (+x)
START_XY = (0.20, -0.24)
START_YAW = -math.pi / 2  # its +y (the line end) along world +x
_start = np.array([START_XY[0], START_XY[1], TOP])
BREAKER = body("breaker", pos=tuple(_start), shapes=_breaker_housing(), yaw=START_YAW, mass=B_MASS,
               friction=0.5, contact_offset=0.0005, solver_iterations=32, com=(0.0, 0.0, 30.0 * MM))


def _shoe(sign: float):
    """One of the jaw's two shoes (sign +1: on the breaker's +x side), at rest against its stop."""
    xc = sign * (GAP + SHOE_T / 2)
    yc, zc = (SHOE_Y0 + SHOE_Y1) / 2, (SHOE_Z0 + SHOE_Z1) / 2
    local = np.array([xc, yc, zc])
    pos = _start + _rot(START_YAW) @ local
    # its flat face from FLARE up, and below that a 45-degree flared mouth (the bottom of a square bar turned 45
    # degrees) from its inner face out to its outer edge: it steers a stab that comes down off-centre into the jaw
    ly = SHOE_Y1 - SHOE_Y0
    flat = SHOE_Z1 - SHOE_Z0 - FLARE
    shapes = [box((0.0, 0.0, (SHOE_Z0 + FLARE + SHOE_Z1) / 2 - zc), (SHOE_T, ly, flat), TINNED),
              box((sign * SHOE_T / 2, 0.0, SHOE_Z0 + FLARE - zc), (FLARE * math.sqrt(2.0), ly, FLARE * math.sqrt(2.0)),
                  TINNED, tilt_y=math.pi / 4)]
    return body(f"jaw_{'a' if sign > 0 else 'b'}", pos=tuple(pos), shapes=shapes,
                yaw=START_YAW, mass=SHOE_MASS, friction=JAW_MU, contact_offset=0.0005, solver_iterations=32)


JAW_A, JAW_B = _shoe(+1.0), _shoe(-1.0)
OBJECTS = [PANEL, BREAKER, JAW_A, JAW_B]
breakerkit.register(*OBJECTS)
_pre = JAW_PRELOAD / JAW_K  # the spring's target lies this far beyond the stop
breakerkit.add_spring(Spring(BREAKER.name, JAW_A.name, axis="X", stiffness=JAW_K, damping=2.0, target=-_pre,
                             lower=0.0, upper=JAW_TRAVEL))
breakerkit.add_spring(Spring(BREAKER.name, JAW_B.name, axis="X", stiffness=JAW_K, damping=2.0, target=+_pre,
                             lower=-JAW_TRAVEL, upper=0.0))

# -- scoring -------------------------------------------------------------------------------------------------------
# Measured from the breaker's pose against its seated pose in space 4:
# 1.0   in space 4, the jaw fully on the stab (its line end within 1.0 mm of seated), its load end hooked under the
#       rail's lip and within 1.0 mm of seated, square (within 2 degrees); no gripper within 1 cm; still; for 1 s
# 0.5   hooked on the rail of space 4 and the jaw at least partly on the stab (line end not yet within 1.0 mm), or
#       seated with a hand on it / not yet still
# 0.25  hooked on the rail of space 4, the jaw not on the stab
# 0     anything else
SCORE_OVER = "end"
RUN_LIMIT_STEPS = 1200
STILL_STEPS = 20
CLEAR_DIST = 10.0 * MM
SEAT_TOL = 1.0 * MM
SQUARE_DEG = 2.0
ENGAGED = RIDGE_H + 0.5 * MM  # the crest this far above the shoes' flat faces: the jaw is on the stab's full blade
_STATE: dict = {}


@breakerkit.on_reset
def _reset_state() -> None:
    _STATE.update(still=0)


_reset_state()


def to_seat(st, space: int = TARGET) -> tuple[np.ndarray, np.ndarray]:
    """The breaker's pose in the frame of a breaker seated in ``space``: (origin, rotation)."""
    o, yaw = seat_pose(space)
    r = _rot(yaw)
    return r.T @ (np.asarray(st.pos) - o), r.T @ st.rotation()


def heights(st, space: int = TARGET) -> tuple[float, float]:
    """How far (m) the breaker's mounting face stands above the base at its line end (the jaw) and its load end."""
    p, r = to_seat(st, space)
    line = p + r @ np.array([0.0, (SHOE_Y0 + SHOE_Y1) / 2, 0.0])
    load = p + r @ np.array([0.0, -BL / 2 + 2.0 * MM, 0.0])
    return float(line[2]), float(load[2])


HOOK_MIN = 1.0 * MM  # the lip over the foot by at least this much: hooked


def hooked(st, space: int = TARGET) -> bool:
    """The rail's lip reaches at least HOOK_MIN into the load end's hook slot, over its foot (seen in the breaker's
    frame)."""
    p, r = to_seat(st, space)
    tip = np.array([0.0, LIP_Y1, (LIP_Z0 + LIP_Z1) / 2])
    q = r.T @ (tip - p)
    return abs(q[0]) < BW / 2 and q[1] - (-BL / 2) >= HOOK_MIN and q[1] < HOOK_Y1 and FOOT_Z < q[2] < HOOK_Z1


def engaged(st, space: int = TARGET) -> bool:
    """The stab's crest is at least ENGAGED above the bottom of the shoes' flat faces, inside the jaw slot: the jaw
    grips the full blade."""
    p, r = to_seat(st, space)
    crest = np.array([0.0, (STAB_Y0 + STAB_Y1) / 2, STAB_TOP])
    q = r.T @ (crest - p)
    return abs(q[0]) < SLOT_X and SLOT_Y0 < q[1] < SLOT_Y1 and q[2] - (SHOE_Z0 + FLARE) >= ENGAGED


def seated(st, space: int = TARGET) -> bool:
    p, r = to_seat(st, space)
    line, load = heights(st, space)
    yaw_err = math.degrees(abs(math.atan2(r[1, 0], r[0, 0])))
    return (hooked(st, space) and abs(line) <= SEAT_TOL and abs(load) <= SEAT_TOL and abs(p[0]) <= 1.5 * MM
            and yaw_err <= SQUARE_DEG and st.tilt_deg() <= SQUARE_DEG)


def hands_clear(objects) -> bool:
    pts = breakerkit.finger_points()
    if not len(pts):
        return True
    st = objects[BREAKER.name]
    centre = st.to_world((0.0, 0.0, (B_BODY + B_FACE + B_HANDLE) / 2))
    size = np.array([BW, BL, B_BODY + B_FACE + B_HANDLE])
    return box_signed_distance(pts, centre, st.rotation(), size).min() >= CLEAR_DIST


def _update(objects) -> tuple[float, bool]:
    st = objects[BREAKER.name]
    if seated(st):
        # the breaker still (its jaw's shoes are inside it; joint-held, their read-back velocities are not physical)
        still = objects[BREAKER.name].resting()
        if still and hands_clear(objects):
            return 1.0, True
        return 0.5, False
    if hooked(st):
        return (0.5 if engaged(st) else 0.25), False
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
# Open loop, from the fixed layout only. The right hand takes the breaker by the top of its sides (fingers closing
# across its width), lifts it and turns it a quarter turn so its line end points at the bus, tips it line end up,
# slides its load end's hook under the rail's lip, lets it down onto the stab and lets go; then presses its closed
# fingertips on the top of the line end until the jaw is fully on the stab.
TIP = 13.0 * MM  # the TCP is this far above the pad tips
RHOME, LHOME = (0.14, -0.22, TOP + 0.17), (0.14, 0.22, TOP + 0.17)
GRIP_Z = 50.0 * MM  # the pad tips this high up the breaker's sides
OPEN_W = -(BW / 2 + 12.0 * MM)
HOLD = -(BW / 2 - 2.5 * MM)  # commanded 2.5 mm inside each side: about 12 N a pad
TILT = math.radians(float(os.environ.get("BK_TILT", "18")))  # line end up while hooking
SLIDE = 11.0 * MM  # the hook is slid this far out under the lip
HOOK_PAST = 4.0 * MM  # ... aimed this far past home, so the soft arm holds it out against the lip
PRESS_Y = 30.0 * MM  # the press: over the jaw
PRESS_OUT = 4.0 * MM  # ... aimed this far towards the rail, so it drags the hook home against the lip
PRESS_YAW = float(os.environ.get("BK_PYAW", "0"))  # fingers closing along y: the wrist camera clear of space 3
PRESS_DEPTH = float(os.environ.get("BK_PRESS", "200")) * MM  # commanded this far below the seated top (soft arm)
CARRY_Z = TOP + 0.17  # carried this high: its mounting face clears the enclosure's 97 mm walls
HELD_DZ = 0.0  # held, the breaker hangs this much lower in the pads (calibrate in the sim)
CAL = np.array([float(v) for v in os.environ.get("BK_CAL", "0,2.5,0").split(",")]) * MM  # free-space correction


def _hook_pivot() -> np.ndarray:
    """The breaker's frame point it is tipped about while hooking: the foot's outer bottom edge."""
    return np.array([0.0, -BL / 2, 0.0])


def _breaker_pose_tipped(tilt: float, out: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """The breaker's (origin, rotation) tipped ``tilt`` line end up about its foot's outer edge on the base, the
    hook ``out`` short of home (m, along the breaker's own y)."""
    o, yaw = seat_pose(TARGET)
    r0 = _rot(yaw)
    c, s = math.cos(tilt), math.sin(tilt)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    r = r0 @ rx
    piv = o + r0 @ _hook_pivot()
    origin = piv - r @ _hook_pivot() + r @ np.array([0.0, out, 0.0])
    return origin, r


def _tcp(origin: np.ndarray, r: np.ndarray) -> tuple[float, float, float]:
    """The TCP for a breaker held at (origin, r): over its middle, the pad tips GRIP_Z up its sides."""
    p = origin + r @ np.array([0.0, 0.0, GRIP_Z + TIP + HELD_DZ]) + CAL
    return tuple(float(v) for v in p)


def take() -> list:
    x, y = START_XY
    at = (x, y, TOP + GRIP_Z + TIP)
    return [Waypoint(2.0, (x, y, at[2] + 0.04), OPEN_W, "over the breaker", 0.0),
            Waypoint(1.2, at, OPEN_W, "round its top", 0.0),
            Waypoint(1.0, at, HOLD, "take it", 0.0),
            Waypoint(1.5, (x, y, CARRY_Z), HOLD, "lift it over the enclosure", 0.0)]


def hook() -> list:
    g = tilted(math.pi / 2, math.pi / 2 - TILT)  # pads closing along x, its line end (+y) tipped up
    o1, r1 = _breaker_pose_tipped(TILT, SLIDE)
    o2, r2 = _breaker_pose_tipped(TILT, -HOOK_PAST)
    a1 = _tcp(o1 + np.array([0.0, 0.0, 0.03]), r1)
    t1, t2 = _tcp(o1, r1), _tcp(o2, r2)
    return [Waypoint(2.5, (a1[0], a1[1], CARRY_Z), HOLD, "turned, over space 4", g),
            Waypoint(1.5, a1, HOLD, "into the enclosure", g),
            Waypoint(1.5, t1, HOLD, "down by the rail", g),
            Waypoint(1.5, t2, HOLD, "hook under the lip", g),
            Waypoint(1.2, t2, OPEN_W, "let go", g),
            Waypoint(1.0, (t2[0], t2[1], t2[2] + 0.05), OPEN_W, "up", g)]


def press() -> list:
    o, yaw = seat_pose(TARGET)
    top = o + _rot(yaw) @ np.array([0.0, PRESS_Y, B_BODY])
    above = (top[0], top[1], top[2] + TIP + 0.03)
    down = (top[0], top[1] - PRESS_OUT, top[2] + TIP - PRESS_DEPTH)
    return [Waypoint(1.2, above, FINGER_CLOSED, "over the line end", PRESS_YAW),
            Waypoint(1.0, (top[0], top[1], top[2] + TIP + 0.025), FINGER_CLOSED, "pause", PRESS_YAW),
            Waypoint(1.5, down, FINGER_CLOSED, "push it home", PRESS_YAW),
            Waypoint(1.0, down, FINGER_CLOSED, "hold", PRESS_YAW),
            Waypoint(1.0, above, FINGER_CLOSED, "lift off", PRESS_YAW),
            Waypoint(2.0, RHOME, FINGER_OPEN, "home", 0.0)]


def demo():
    """``BK_DEMO=nohook`` (a shortcut check): set it down level on the stab without hooking, then press."""
    right = [Waypoint(1.0, RHOME, FINGER_OPEN, "wait", 0.0)] + take()
    if os.environ.get("BK_DEMO") == "nohook":
        o, yaw = seat_pose(TARGET)
        r = _rot(yaw)
        p = _tcp(o + np.array([0.0, 0.0, STAB_TOP + 1.0 * MM]), r)
        right += [Waypoint(2.5, (p[0], p[1], p[2] + 0.03), HOLD, "over space 4", math.pi / 2),
                  Waypoint(1.5, p, HOLD, "onto the stab", math.pi / 2),
                  Waypoint(1.2, p, OPEN_W, "let go", math.pi / 2),
                  Waypoint(1.0, (p[0], p[1], p[2] + 0.05), OPEN_W, "up", math.pi / 2)]
    else:
        right += hook()
    right += press()
    left = [Waypoint(sum(w.duration for w in right), LHOME, FINGER_OPEN, "wait", 0.0)]
    for plan in (left, right):
        plan.append(Waypoint(2.0, plan[-1].pos, plan[-1].finger, "done", plan[-1].yaw))
    return {"right": right, "left": left}

