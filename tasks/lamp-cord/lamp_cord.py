"""On the station table: a column table lamp being rewired lies on its side, its old cord out, its socket and harp off.
The new lamp cord lies on the table at its base, its end cut ready.

The robot must thread the new cord through the lamp, leaving about 3 inches out of the top for the socket: the
cord's end goes in at the lamp pipe's bottom end (in the base) and is fed up the 1/8 IP pipe (12 in, 7.6 mm bore)
until it comes out of the threaded top, then drawn out to length. The lamp's base stops the hand short of the
pipe's mouth, so the cord is fed hand over hand: pushed in, let go, taken again where it hangs, pushed again.

Run it:  inspect-robots run --embodiment yam_arena -E task=lamp_cord.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "Thread the new cord through the lamp, leaving about 3 inches out of the top." ...
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lampkit  # noqa: E402
from lampkit import Bend, body, box, cyl, ring  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

lampkit.install()
SCENE = "station"
TOP = TABLE_TOP[SCENE]
MM = 0.001
E = os.environ
ZINC = (0.72, 0.74, 0.76)
CERAMIC = (0.86, 0.88, 0.90)
FELT = (0.30, 0.22, 0.16)
BROWN = (0.36, 0.22, 0.12)  # brown lamp cord
lampkit.METALLIC.update({ZINC: 0.4})

# -- sizes ---------------------------------------------------------------------------------------------------------
# the pipe: Westinghouse 7027200 all-thread lamp pipe, 1/8 IP (3/8 in outside), 12-1/16 in; bore 7.6 mm (ours: the
# maker gives none); the lamp: a ceramic column 60 mm across (ours), lying on its side, the pipe on its axis
PIPE_X = 0.30
LAMP_R = 30.0 * MM
PIPE_Z = LAMP_R  # the pipe's axis above the table (the lamp lying on its side)
PIPE_LEN = 306.0 * MM
PIPE_Y0, PIPE_Y1 = -PIPE_LEN / 2, PIPE_LEN / 2  # its bottom end (flush with the base: the cord goes in) and its top
TOP_OUT = 30.0 * MM  # the threaded top stands this far out of the lamp's neck (for the harp and the socket cap)
PIPE_ID, PIPE_OD = 7.6 * MM, 9.5 * MM
# the cord: Southwire 49908744, 18/2 SPT-1, 2.8 x 5.5 mm; in 12 mm links that bend (lampkit.Bend)
CORD_W, CORD_T = 5.5 * MM, 2.8 * MM
LINK = 12.0 * MM
N_LINKS = 42  # 50 cm of it (the rest of the 8 ft length beyond the table edge is not modelled)
END_Y = PIPE_Y0 - 0.03  # the cord's end on the table, 3 cm short of the pipe's mouth
STUB = (60.0 * MM, 100.0 * MM)  # out of the top: about 3 in (76 mm), for tying the knot and wiring the socket
LAMP_Q = (-math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4))  # the lamp's z axis (the pipe) along +y


def _lamp_shapes() -> tuple[list, list]:
    """In the lamp's frame: z along the pipe (from its middle, to +y), the bore round it."""
    half = PIPE_LEN / 2
    solid = ring(-half, half, PIPE_ID / 2, PIPE_OD / 2, ZINC, segments=12)  # the pipe
    solid += ring(-half, half - TOP_OUT, PIPE_OD / 2, LAMP_R, CERAMIC, segments=24)  # the column round it
    seen = [cyl((0.0, 0.0, -half - 1.0 * MM), LAMP_R, 2.0 * MM, FELT)]  # the felt on the base
    return solid, seen


_LS, _LV = _lamp_shapes()
LAMP = body("table_lamp", pos=(PIPE_X, 0.0, TOP + PIPE_Z), quat=LAMP_Q, friction=0.6, contact_offset=0.0003,
            shapes=_LS + _LV, visual_only=tuple(range(len(_LS), len(_LS) + len(_LV))))

# the new cord lying straight on the table along y, its end towards the pipe; link frames' x runs away from the end
TAPER = 5.0 * MM  # its end cut to a point (a lead-in)
CORD = []
for i in range(N_LINKS):
    y = END_Y - (i + 0.5) * LINK
    if i == 0:
        shapes = [box((TAPER / 2, 0.0, 0.0), (LINK - TAPER - 0.5 * MM, CORD_W, CORD_T), BROWN),
                  box((-LINK / 2 + TAPER / 2, 0.0, 0.0), (TAPER, 3.0 * MM, 1.8 * MM), BROWN)]
    else:
        shapes = [box((0.0, 0.0, 0.0), (LINK - 0.5 * MM, CORD_W, CORD_T), BROWN)]
    CORD.append(body(f"cord{i:02d}", pos=(PIPE_X, y, TOP + CORD_T / 2 + 0.3 * MM), yaw=-math.pi / 2, mass=0.0005,
                     friction=0.3, shapes=shapes, contact_offset=0.0003, inertia=(2e-8, 1.5e-7, 1.5e-7),
                     solver_iterations=32))
OBJECTS = [LAMP, *CORD]
lampkit.register(*OBJECTS)
for _c in CORD:
    lampkit.damped(_c.name, 0.5, 2.0)
for i in range(1, N_LINKS):
    lampkit.add_bend(Bend(CORD[i - 1].name, CORD[i].name, anchor=(LINK / 2, 0.0, 0.0), stiffness=0.01,
                          damping=2e-5))

# -- scoring -------------------------------------------------------------------------------------------------------
# 1.0   the cord through the pipe, its end 60-100 mm (about 3 in) out of the top; no gripper within 1 cm of it; still
# 0.5   through, its end out of the top, but outside that length (or not yet let go or still)
# 0.25  its end at least half way up the pipe, not out of the top
# 0     anything else
SCORE_OVER = "end"
RUN_LIMIT_STEPS = 2400
CLEAR_DIST = 10.0 * MM
STILL_STEPS = 20
_STATE: dict = {}


@lampkit.on_reset
def _reset_state() -> None:
    _STATE.update(still=0)


_reset_state()


def cord_tip(objects) -> np.ndarray:
    """The cord's end (table frame): the end link's leading face."""
    s = objects[CORD[0].name]
    return np.asarray(s.pos, float) - s.rotation()[:, 0] * (LINK / 2) - np.array([0.0, 0.0, TOP])


def in_pipe(p) -> bool:
    return PIPE_Y0 <= p[1] <= PIPE_Y1 and math.hypot(p[0] - PIPE_X, p[2] - PIPE_Z) < PIPE_ID / 2 + 1.0 * MM


def through(objects) -> bool:
    """The cord runs through the pipe: a link inside it near each end."""
    pts = [np.asarray(objects[c.name].pos, float) - np.array([0.0, 0.0, TOP]) for c in CORD]
    near0 = any(in_pipe(p) and p[1] < PIPE_Y0 + 0.05 for p in pts)
    near1 = any(in_pipe(p) and p[1] > PIPE_Y1 - 0.05 for p in pts)
    return near0 and near1


def fingers_clear(objects) -> bool:
    pts = lampkit.finger_points()
    if not len(pts):
        return True
    cs = np.array([np.asarray(objects[c.name].pos, float) for c in CORD])
    d = np.linalg.norm(pts[:, None, :] - cs[None, :, :], axis=2)
    return float(d.min()) >= CLEAR_DIST + LINK / 2


def _update(objects) -> tuple[float, bool]:
    tip = cord_tip(objects)
    out = tip[1] - PIPE_Y1
    if out > 0 and through(objects):
        # still: the cord outside the pipe (inside it, the bore holds it; a link there can buzz against its walls)
        still = all(objects[c.name].resting() for c in CORD[:16]
                    if not in_pipe(np.asarray(objects[c.name].pos, float) - np.array([0.0, 0.0, TOP])))
        if STUB[0] <= out <= STUB[1] and still and fingers_clear(objects):
            return 1.0, True
        return 0.5, False
    if in_pipe(tip) and tip[1] >= (PIPE_Y0 + PIPE_Y1) / 2:
        return 0.25, False
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
TIP = 13.0 * MM
RHOME, LHOME = (0.16, -0.22, TOP + 0.15), (0.16, 0.22, TOP + 0.15)
GRIP = -(CORD_W / 2 - 1.0 * MM)
OPEN = -(CORD_W / 2 + 8.0 * MM)
STROKE = 20.0 * MM
SAG_PER_MM = 0.47  # the cord droops this far below the pipe's axis per mm out of its mouth (seen in the sim)
HAND_MAX = PIPE_Y0 - 20.0 * MM  # the right fingers stop this short of the pipe's mouth (the gripper clears the base)
FIRST = 50.0 * MM  # the first grip, behind the cord's end: carried in, its end goes 30 mm up the pipe
FED_TO = PIPE_Y1 + 0.05  # the right hand's strokes add up to its end 5 cm out of the top (it slips a little: ~3 cm)
TAKE_Y = PIPE_Y1 + 0.015  # the left hand takes the cord here, just out of the top
DRAW = 0.060  # and draws it this far (it slips about a third): its end ends about 80 mm out


def _za() -> float:
    return TOP + PIPE_Z + TIP - 1.0 * MM


def right_feed() -> list:
    """Take the cord behind its end, carry the end into the pipe's mouth, then feed it on a stroke at a time,
    taking it again where it hangs out of the mouth each time (the lamp's base stops the hand short of it)."""
    yaw = -math.pi / 2  # fingers close along x (across the cord), the wrist camera on the side away from the lamp
    yg = END_Y - FIRST
    zt = TOP + CORD_T / 2 + TIP - 0.8 * MM  # fingertips 0.8 mm under the cord's middle
    za = _za()
    out = [Waypoint(2.0, (PIPE_X, yg, zt + 0.05), OPEN, "over the cord", yaw),
           Waypoint(1.2, (PIPE_X, yg, zt), OPEN, "round it", yaw),
           Waypoint(1.2, (PIPE_X, yg, zt), GRIP, "take it", yaw),
           Waypoint(1.5, (PIPE_X, yg, za), GRIP, "up to the pipe", yaw),
           Waypoint(1.5, (PIPE_X, HAND_MAX - 0.03, za), GRIP, "its end to the mouth", yaw),
           Waypoint(2.0, (PIPE_X, HAND_MAX, za), GRIP, "its end into the pipe", yaw),
           Waypoint(0.6, (PIPE_X, HAND_MAX, za), OPEN, "let go", yaw)]
    end = END_Y + (HAND_MAX - yg)
    while end < FED_TO - 1e-6:
        step = min(STROKE, FED_TO - end)
        sag = SAG_PER_MM * (PIPE_Y0 - (HAND_MAX - step))
        out += [Waypoint(0.6, (PIPE_X, HAND_MAX, za + 0.012), OPEN, "up off it", yaw),
                Waypoint(0.8, (PIPE_X, HAND_MAX - step, za + 0.012), OPEN, "back along it", yaw),
                Waypoint(0.6, (PIPE_X, HAND_MAX - step, za - sag), OPEN, "down round it", yaw),
                Waypoint(0.6, (PIPE_X, HAND_MAX - step, za - sag), GRIP, "take it again", yaw),
                Waypoint(1.0, (PIPE_X, HAND_MAX, za - 0.002), GRIP, "feed it in", yaw),
                Waypoint(0.6, (PIPE_X, HAND_MAX, za - 0.002), OPEN, "let go", yaw)]
        end += step
    out += [Waypoint(0.8, (PIPE_X, HAND_MAX, za + 0.03), OPEN, "up off it", yaw),
            Waypoint(1.5, RHOME, OPEN, "home", 0.0)]
    return out


def left_draw() -> list:
    """Take the cord's end where it shows out of the top and draw it out to length."""
    yaw = math.pi / 2  # the wrist camera on the side away from the lamp
    za = _za()
    y0 = TAKE_Y
    return [Waypoint(2.0, (PIPE_X, y0, za + 0.04), OPEN, "over the cord's end", yaw),
            Waypoint(1.2, (PIPE_X, y0, za - 0.002), OPEN, "round it", yaw),
            Waypoint(1.2, (PIPE_X, y0, za - 0.002), GRIP, "take it", yaw),
            Waypoint(2.0, (PIPE_X, y0 + DRAW, za), GRIP, "draw it out", yaw),
            Waypoint(0.6, (PIPE_X, y0 + DRAW, za), OPEN, "let go", yaw),
            Waypoint(1.0, (PIPE_X, y0 + DRAW, za + 0.04), OPEN, "up off it", yaw),
            Waypoint(1.5, LHOME, OPEN, "home", 0.0)]


def demo():
    right = right_feed()
    t = sum(w.duration for w in right[:-2])
    left = [Waypoint(t, LHOME, FINGER_OPEN, "wait", 0.0)] + left_draw()
    tr, tl = sum(w.duration for w in right), sum(w.duration for w in left)
    if tr < tl:
        right.append(Waypoint(tl - tr, right[-1].pos, right[-1].finger, "wait", right[-1].yaw))
    else:
        left.append(Waypoint(tr - tl, left[-1].pos, left[-1].finger, "wait", left[-1].yaw))
    for plan in (left, right):
        plan.append(Waypoint(2.0, plan[-1].pos, plan[-1].finger, "done", plan[-1].yaw))
    return {"right": right, "left": left}
