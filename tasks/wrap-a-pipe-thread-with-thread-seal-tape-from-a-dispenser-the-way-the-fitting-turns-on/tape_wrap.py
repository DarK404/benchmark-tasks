"""On the station table: a 3/4" galvanised pipe nipple stands upright in a floor flange on a wooden block, its male
thread at the top, and a handheld dispenser of thread-seal (PTFE) tape lies on a mat beside it.

The robot must wrap the thread with the tape, ready for a fitting to be screwed on, and put the dispenser back:
wound the way the fitting will turn on (clockwise seen from the thread's end), 3 to 6 turns over the thread with the
first thread left bare, torn off and its tail pressed down.

Run it:  inspect-robots run --embodiment yam_arena -E task=tape_wrap.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "..." --scorer task_score ...
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tapekit  # noqa: E402
from tapekit import TapeWrap, body, box, prism12, ring  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

tapekit.install()
SCENE = "station"
TOP = TABLE_TOP[SCENE]
MM = 0.001
GALV = (0.62, 0.64, 0.66)  # galvanised pipe
CUT = (0.50, 0.51, 0.53)  # the thread, cut through the zinc (darker than the white tape on it)
GROOVE = (0.28, 0.29, 0.31)
CAST_IRON = (0.16, 0.16, 0.17)
WOOD = (0.55, 0.40, 0.25)
PTFE = (0.96, 0.96, 0.95)
SPOOL = (0.90, 0.78, 0.20)  # the roll's plastic core
tapekit.METALLIC.update({GALV: 0.35, CUT: 0.25, CAST_IRON: 0.6})

# -- the pipe: a 3/4" BSP nipple (26.4 mm across) in a floor flange on a block ------------------------------------
PIPE_XY = (0.26, 0.0)
R_PIPE = 13.2 * MM
BLOCK_H = 20.0 * MM
FLANGE_H = 8.0 * MM
HUB_H = 14.0 * MM
PIPE_TOP = 100.0 * MM  # above the table
THREAD_L = 18.0 * MM
Z_SHOULDER = PIPE_TOP - THREAD_L  # where the thread starts below its end
PITCH = 1.814 * MM  # 3/4" BSP: 14 threads per inch


def _stand_shapes() -> list:
    base = BLOCK_H + FLANGE_H
    shapes = [box((0.0, 0.0, BLOCK_H / 2), (0.09, 0.09, BLOCK_H), WOOD)]
    shapes += prism12((0.0, 0.0, BLOCK_H + FLANGE_H / 2), 0.034, FLANGE_H, CAST_IRON)
    shapes += prism12((0.0, 0.0, base + HUB_H / 2), 0.0185, HUB_H, CAST_IRON)
    body_len = Z_SHOULDER - base
    shapes += prism12((0.0, 0.0, base + body_len / 2), R_PIPE, body_len, GALV)
    shapes += prism12((0.0, 0.0, Z_SHOULDER + THREAD_L / 2), R_PIPE, THREAD_L, CUT)
    n_solid = len(shapes)
    for k in range(int(THREAD_L / PITCH)):  # the thread's grooves, seen only
        z = Z_SHOULDER + (k + 0.5) * PITCH
        shapes += ring(z - 0.25 * MM, z + 0.25 * MM, R_PIPE - 0.4 * MM, R_PIPE + 0.15 * MM, GROOVE, segments=12)
    shapes += prism12((0.0, 0.0, PIPE_TOP - 0.2 * MM), R_PIPE - 2.6 * MM, 0.6 * MM, (0.08, 0.08, 0.09))  # its bore
    return shapes, n_solid


_SHAPES, _N_SOLID = _stand_shapes()
PIPE = body("pipe", pos=(*PIPE_XY, TOP), friction=0.6, shapes=_SHAPES,
             visual_only=tuple(range(_N_SOLID, len(_SHAPES))))

# -- the tape in its dispenser, lying on a mat --------------------------------------------------------------------
# A handheld thread-seal tape dispenser: the roll turns on an upright spindle, braked; an arm runs from the spindle's
# top to a handle. Its frame: origin on the spindle's axis at the roll's middle height, the handle towards -y.
# WHY a dispenser (2026-11-10): pinched by its rim in a parallel gripper the roll cannot turn, and no tape could come
# off it; held by the dispenser's handle the roll turns as the tape is drawn off, and the handle passes over the pipe's
# end while the roll goes round the thread.
ROLL_R = 15.0 * MM
ROLL_H = 12.0 * MM  # the tape's width
HOLE_R = 6.0 * MM
SPINDLE_DOWN = 9.0 * MM  # the spindle's foot below the roll's middle
# WHY the handle stands 26 mm above the roll's middle (2026-11-10, sim trace): at 16 mm it passed 1 mm over the pipe's
# end with the roll at the bottom of the wrap, caught it and tipped the dispenser in the hand
ARM_Z = (27.0 * MM, 33.0 * MM)  # the arm from the spindle's top to the handle
HANDLE_Y = (-25.0 * MM, -95.0 * MM)
HANDLE_W, HANDLE_Z = 16.0 * MM, 34.0 * MM  # the handle's width (gripped across) and its middle's height
GRIP_Y = -60.0 * MM  # where the hand grips the handle, from the spindle
BRAKE = 2.5e-3  # N m s/rad: the spindle's brake (a steady wrap draws the tape off at about 0.3 N)
TEAR_STRETCH = 5.0 * MM  # the tape tears stretched this far: at 1.5 N (300 N/m), as PTFE tape tears by hand
DISPENSER_ORANGE = (0.88, 0.45, 0.10)
GRIP_BLACK = (0.10, 0.10, 0.11)
MAT_XY = (0.20, -0.12)
MAT_T = 2.0 * MM
REST_XY = (MAT_XY[0], MAT_XY[1] + 0.045)  # the spindle's axis when the dispenser lies on its mat
REST_Z = MAT_T + SPINDLE_DOWN + 0.5 * MM  # the roll's middle above the table, the dispenser on its feet on the mat


def _dispenser_shapes() -> list:
    hy = (HANDLE_Y[0] + HANDLE_Y[1]) / 2
    foot_h = HANDLE_Z - HANDLE_W / 2 + SPINDLE_DOWN
    return [
        box((0.0, 0.0, (ARM_Z[1] - SPINDLE_DOWN) / 2), (0.006, 0.006, ARM_Z[1] + SPINDLE_DOWN), DISPENSER_ORANGE),
        box((0.0, (0.004 + HANDLE_Y[0]) / 2, sum(ARM_Z) / 2), (0.012, 0.004 - HANDLE_Y[0], ARM_Z[1] - ARM_Z[0]),
            DISPENSER_ORANGE),
        box((0.0, hy, HANDLE_Z), (HANDLE_W, HANDLE_Y[0] - HANDLE_Y[1], HANDLE_W), GRIP_BLACK),
        box((0.0, HANDLE_Y[1] + 0.008, HANDLE_Z - HANDLE_W / 2 - foot_h / 2), (0.012, 0.012, foot_h), DISPENSER_ORANGE),
        *prism12((0.0, 0.0, -ROLL_H / 2 - 1.5 * MM), 0.010, 1.0 * MM, DISPENSER_ORANGE),  # the washer under the roll
    ]


_DSH = _dispenser_shapes()
DISPENSER = body("dispenser", pos=(*REST_XY, TOP + REST_Z), mass=0.06, friction=0.8,
                 com=(0.0, -0.05, 0.02), shapes=_DSH, visual_only=tuple(range(4, len(_DSH))))
ROLL = body("roll", pos=(*REST_XY, TOP + REST_Z), mass=0.02, friction=0.8, inertia=(1.5e-6, 1.5e-6, 2.6e-6), shapes=[
    *ring(-ROLL_H / 2, ROLL_H / 2, HOLE_R, ROLL_R, PTFE, segments=12),
    *ring(ROLL_H / 2, ROLL_H / 2 + 0.3 * MM, HOLE_R, HOLE_R + 2.0 * MM, SPOOL, segments=12),
    *ring(-ROLL_H / 2 - 0.3 * MM, -ROLL_H / 2, HOLE_R, HOLE_R + 2.0 * MM, SPOOL, segments=12),
], visual_only=tuple(range(12, 36)))
tapekit.add_hinge(tapekit.Hinge("dispenser", "roll", anchor=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), damping=BRAKE))
MAT = body("mat", pos=(*MAT_XY, TOP), friction=0.9, shapes=[
    box((0.0, 0.0, MAT_T / 2), (0.06, 0.15, MAT_T), (0.14, 0.14, 0.15))])  # MAT_SIZE

# -- the tape: bodies we pose (seen only, no gravity) -------------------------------------------------------------
N_SEGMENTS = 28  # quarter turns drawn: up to 7 turns
TAPE_T = 0.15 * MM
HIDDEN_Z = TOP - 0.30  # under the table


def _segment(k: int):
    r = R_PIPE + 0.35 * MM + (k // 4) * TAPE_T  # each turn over the last
    chord = 2 * r * math.tan(math.radians(15)) + 0.2 * MM
    shapes = [box((r * math.cos(a), r * math.sin(a), 0.0), (TAPE_T, chord, ROLL_H), PTFE, yaw=a)
              for a in (math.radians(d) for d in (-30, 0, 30))]
    return body(f"tape{k}", pos=(*PIPE_XY, HIDDEN_Z), mass=1e-3, inertia=(1e-7, 1e-7, 1e-7), shapes=shapes,
                visual_only=(0, 1, 2))


SEGMENTS = [_segment(k) for k in range(N_SEGMENTS)]
SPAN = body("tape_span", pos=(*PIPE_XY, HIDDEN_Z), mass=1e-3, inertia=(1e-7, 1e-7, 1e-7),
            shapes=[box((0.0175, 0.0, 0.0), (0.035, TAPE_T, ROLL_H), PTFE)], visual_only=(0,))
TAIL = body("tape_tail", pos=(*PIPE_XY, HIDDEN_Z), mass=1e-3, inertia=(1e-7, 1e-7, 1e-7),
            shapes=[box((0.009, 0.0, 0.0), (0.018, TAPE_T, ROLL_H), PTFE)], visual_only=(0,))
for _s in [*SEGMENTS, SPAN, TAIL]:
    tapekit.no_gravity(_s.name)

OBJECTS = [PIPE, MAT, DISPENSER, ROLL, *SEGMENTS, SPAN, TAIL]
tapekit.register(*OBJECTS)
MODEL = tapekit.add_model(TapeWrap(
    "roll", "dispenser", [s.name for s in SEGMENTS], SPAN.name, TAIL.name, PIPE_XY, R_PIPE, TOP + Z_SHOULDER, TOP + PIPE_TOP,
    ROLL_R, ROLL_H, HIDDEN_Z, break_stretch=TEAR_STRETCH))


# -- scoring -------------------------------------------------------------------------------------------------------
SCORE_OVER = "end"
RUN_LIMIT_STEPS = 1600
TURNS = (3.0, 6.0)
COVER = (Z_SHOULDER + 1.0 * MM, PIPE_TOP - 4.0 * MM)  # every part of the thread between these heights taped
END_CLEAR = 1.0 * MM  # no tape within this of the pipe's end (the first thread left bare)
CLEAR = 10.0 * MM  # no part of either gripper this near the roll: released
MAT_SIZE = (0.06, 0.15)
STILL = 0.01


def taped_heights() -> tuple[float, float, bool]:
    """The tape's lowest and highest edge above the table, and whether it covers every part of the thread between
    COVER's heights."""
    bands = [(lo - TOP, hi - TOP) for lo, hi in MODEL.bands()]
    if not bands:
        return math.nan, math.nan, False
    grid = np.arange(COVER[0], COVER[1] + 1e-9, 0.25 * MM)
    covered = all(any(lo <= z <= hi for lo, hi in bands) for z in grid)
    return min(b[0] for b in bands), max(b[1] for b in bands), covered


def wound_right(min_turns: float) -> bool:
    """On the thread, wound clockwise seen from its end, at least ``min_turns`` turns."""
    return bool(MODEL.anchored and MODEL.turns() >= min_turns)


def fingers_to(state, spec) -> float:
    """Nearest distance (m) from either gripper's hulls to a body's bounding box (in its own frame). inf offline."""
    pts = tapekit.finger_points()
    if not len(pts):
        return math.inf
    lo, hi = spec.bounds_local()
    local = (pts - state.pos) @ state.rotation()
    d = np.maximum(np.maximum(lo - local, local - hi), 0.0)
    return float(np.linalg.norm(d, axis=1).min())


def on_mat(dispenser) -> bool:
    """Back on its feet on its mat: its spindle upright and over the mat, resting on it."""
    off = np.abs(np.asarray(dispenser.pos[:2]) - MAT_XY)
    return bool(off[0] <= MAT_SIZE[0] / 2 and off[1] <= MAT_SIZE[1] / 2
                and abs(dispenser.pos[2] - (TOP + REST_Z)) < 4.0 * MM and dispenser.tilt_deg() < 10.0)


def _update(objects) -> tuple[float, bool]:
    if not wound_right(1.0):
        return 0.0, False
    lo, hi, covered = taped_heights()
    standard = (TURNS[0] <= MODEL.turns() <= TURNS[1] and covered and hi <= PIPE_TOP - END_CLEAR
                and MODEL.torn)
    if not standard:
        return 0.25, False
    d = objects["dispenser"]
    if MODEL.tail_down and on_mat(d) and min(fingers_to(d, DISPENSER), fingers_to(objects["roll"], ROLL)) >= CLEAR:
        return 1.0, bool(d.resting(max_speed=STILL))
    return 0.5, False


def success(objects) -> bool:
    """Wound clockwise seen from the thread's end, 3 to 6 turns, covering the thread from 1 mm above its shoulder to
    4 mm below its end and clear of the end by 1 mm, torn off with its tail pressed down, the dispenser back on its
    mat, released and still for 1 s."""
    return _update(objects)[1]


def score(objects) -> float:
    """1.0 / 0.5 / 0.25 / 0 as in the README."""
    return _update(objects)[0]


# -- scripted demo -------------------------------------------------------------------------------------------------
# Open loop, from the fixed layout. The right hand takes the dispenser by its handle, carries it over the pipe's end
# and touches the roll to the thread on the pipe's left side (+y); it carries the roll four turns clockwise round the
# pipe (seen from above), spiralling down, the handle passing over the pipe's end, the first turn binding the tape's
# end, and the roll turning on its spindle as the tape comes off; it yanks the roll straight out, which tears the tape,
# and lays the dispenser back on its mat; then the left hand presses the torn tail down on the wrap from its own side.
# WHY one hand at a time (2026-11-10, sim traces): with the left hand pinning the tape's end while the right carried
# the roll round, the two arms' wrists met above the pipe, though the hands were 6 cm apart.
RHOME, LHOME = (0.16, -0.22, TOP + 0.15), (0.16, 0.22, TOP + 0.15)
TIP_PAST_TCP = 13.0 * MM
# the fingers closed to 4 mm on the 16 mm handle: a firm grip
# WHY firm (2026-11-10, sim traces): the fingers hold the handle across it, so the dispenser can pitch about the line
# between the pads, held only by their friction; closed to 12 or 8 mm, the tape's pull on the roll, 34 mm below the
# handle, tipped the dispenser in the hand as the tape was torn
HOLD = FINGER_OPEN * 4.0 / 94.0
# the arm sags under the dispenser: the roll is sent this much higher than it is meant to go (measured in sim)
SAG_Z = 2.7 * MM
# where the roll's middle sits from the TCP, the handle held (the fingertips 2 mm below the handle)
ROLL_IN_GRIP = np.array([0.0, -GRIP_Y, -(HANDLE_Z - HANDLE_W / 2 - 2.0 * MM + TIP_PAST_TCP)])
GRIP_YAW = math.pi / 2  # the right fingers close along x: across the handle, which runs along y
PIN_YAW = -math.pi / 2  # the left fingers close along x too (-90 keeps the left wrist off its limit)
START_DEG = 90.0  # where the tape starts round the pipe (from +x, anticlockwise seen from above)
ORBIT_R = 45.0 * MM
TOUCH_R = R_PIPE + ROLL_R + 1.0 * MM
Z_START = PIPE_TOP - 2.5 * MM - ROLL_H / 2  # the tape's top edge 2.5 mm below the pipe's end
Z_END = Z_SHOULDER - 1.5 * MM + ROLL_H / 2  # ...and its bottom edge 1.5 mm below the thread's shoulder at the end
WRAPS = 4.0  # ends where it started, on the pipe's left side, where the left hand can reach the tail
STEP_DEG = 30.0
STEP_S = 0.25
PRESS = 2.0 * MM  # the left fingers sent this far into the pipe's side
FINGER_HALF = 7.7 * MM  # half a fingertip's width
DEMO_MODE = os.environ.get("TAPE_DEMO", "")  # shortcut checks: "backwards" winds anticlockwise; "nopress" leaves
# the torn tail standing


def _roll_at(angle_deg: float, r: float, z: float) -> tuple:
    """The TCP that puts the held roll's middle ``r`` from the pipe's axis at ``angle_deg``, ``z`` above the table."""
    a = math.radians(angle_deg)
    return tuple(np.array([PIPE_XY[0] + r * math.cos(a), PIPE_XY[1] + r * math.sin(a), TOP + z + SAG_Z])
                 - ROLL_IN_GRIP)


def _w(dur, pos, finger, label, yaw=GRIP_YAW):
    return Waypoint(dur, tuple(pos), finger, label, yaw)


def _t(w) -> float:
    return sum(p.duration for p in w)


def _orbit(w0: float, w1: float, label: str, sign: float) -> list:
    """The roll carried round the pipe from ``w0`` to ``w1`` degrees wound (clockwise for sign +1), spiralling down
    from Z_START to Z_END over the whole wrap."""
    out = []
    n = max(1, int(round((w1 - w0) / STEP_DEG)))
    for k in range(1, n + 1):
        w = w0 + (w1 - w0) * k / n
        z = Z_START + (Z_END - Z_START) * min(1.0, w / (WRAPS * 360.0))
        out.append(_w(STEP_S * (w1 - w0) / n / STEP_DEG, _roll_at(START_DEG - sign * w, ORBIT_R, z), HOLD,
                      label))
    return out


def press_tcp(z_tip: float) -> tuple:
    """The left TCP with its closed fingertips' edge pressed on the pipe's left side, the tips at ``z_tip``."""
    return (PIPE_XY[0], PIPE_XY[1] + R_PIPE + FINGER_HALF - PRESS, TOP + z_tip + TIP_PAST_TCP)


def demo():
    sign = -1.0 if DEMO_MODE == "backwards" else 1.0
    over = PIPE_TOP + 16.0 * MM  # the roll's middle when it is carried over the pipe's end

    def roll_tcp(x, y, z):  # the TCP that puts the roll's middle at (x, y, z)
        return tuple(np.array([x, y, z]) - ROLL_IN_GRIP)

    rest = roll_tcp(REST_XY[0], REST_XY[1], TOP + REST_Z)
    above = lambda p, dz: (p[0], p[1], p[2] + dz)  # noqa: E731
    right = [
        _w(2.0, above(rest, 0.04), FINGER_OPEN, "over the handle"),
        _w(1.2, rest, FINGER_OPEN, "round it"),
        _w(1.6, rest, HOLD, "grip"),
        _w(1.2, roll_tcp(REST_XY[0], REST_XY[1], TOP + over), HOLD, "lift it"),
        # carried over the pipe's end, clear of the thread (touched on the way, the tape would start there)
        _w(2.0, _roll_at(START_DEG, 0.065, over), HOLD, "to the pipe"),
        _w(1.0, _roll_at(START_DEG, 0.065, Z_START), HOLD, "down beside it"),
        _w(1.0, _roll_at(START_DEG, TOUCH_R, Z_START), HOLD, "touch the thread"),
        _w(0.5, _roll_at(START_DEG, TOUCH_R, Z_START), HOLD, "stick the end"),
        _w(0.5, _roll_at(START_DEG, ORBIT_R, Z_START), HOLD, "out"),
    ]
    right += _orbit(0.0, WRAPS * 360.0, "wrap", sign)
    end = right[-1].pos
    a_end = START_DEG - sign * WRAPS * 360.0
    right += [
        _w(0.8, end, HOLD, "wrapped"),  # the arm, lagging under the dispenser, catches up before the yank
        # a yank straight out from the pipe: faster than the braked roll can let the tape off, so it tears
        _w(0.3, _roll_at(a_end, 0.105, Z_END), HOLD, "tear it off"),
        _w(1.0, _roll_at(a_end, 0.105, over), HOLD, "up"),
        _w(2.0, roll_tcp(REST_XY[0], REST_XY[1], TOP + over), HOLD, "over the mat"),
        _w(1.5, roll_tcp(REST_XY[0], REST_XY[1], TOP + REST_Z + 0.5 * MM), HOLD, "lay it down"),
        _w(1.6, roll_tcp(REST_XY[0], REST_XY[1], TOP + REST_Z + 0.5 * MM), FINGER_OPEN, "let go"),
        _w(1.0, above(rest, 0.04), FINGER_OPEN, "up"),
        _w(1.5, RHOME, FINGER_OPEN, "home", 0.0),
    ]
    t_clear = _t(right[:-4])  # the dispenser is over its mat: the right hand is clear of the pipe
    tail = press_tcp(Z_END - 2.0 * MM)
    above_tail = (tail[0], tail[1] + 0.02, tail[2] + 0.04)
    left = [_w(t_clear, LHOME, FINGER_CLOSED, "wait", 0.0)]
    if DEMO_MODE != "nopress":
        left += [
            _w(1.5, above_tail, FINGER_CLOSED, "over the tail", PIN_YAW),
            _w(0.8, tail, FINGER_CLOSED, "press it down", PIN_YAW),
            _w(0.6, tail, FINGER_CLOSED, "smooth it", PIN_YAW),
            _w(0.8, above_tail, FINGER_CLOSED, "off", PIN_YAW),
            _w(1.5, LHOME, FINGER_CLOSED, "home", 0.0),
        ]
    end_t = max(_t(left), _t(right)) + 2.5
    left.append(_w(end_t - _t(left), LHOME, FINGER_CLOSED, "done", 0.0))
    right.append(_w(end_t - _t(right), RHOME, FINGER_OPEN, "done", 0.0))
    return {"right": right, "left": left}
