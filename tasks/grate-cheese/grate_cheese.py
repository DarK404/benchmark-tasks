"""On the station table: a flat coarse grater lies across a glass dish, its handle to the robot's left; a block of
cheddar sits on a cutting board to the right.

The robot must grate about 20 g of the cheese into the dish: the block is stroked along the blades, pressed, away
from the handle, while the other hand steadies the grater by its handle; the fingers holding the block never go onto
the blades.

Run it:  inspect-robots run --embodiment yam_arena -E task=grate_cheese.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "Grate about 20 g of the cheese into the dish." --scorer task_score ...
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gratekit  # noqa: E402
from gratekit import Grating, body, box, cyl, prism12, ring  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

gratekit.install()
SCENE = "station"
TOP = TABLE_TOP[SCENE]
MM = 0.001
E = os.environ
STEEL = (0.78, 0.79, 0.80)
BLACK = (0.12, 0.12, 0.13)
GLASS = (0.80, 0.88, 0.92)
WOOD = (0.72, 0.56, 0.38)
CHEDDAR = (0.98, 0.72, 0.22)
gratekit.METALLIC.update({STEEL: 0.3})

# -- sizes ---------------------------------------------------------------------------------------------------------
DISH_XY = (0.30, -0.02)
DISH_R, DISH_WALL, DISH_H = 100.0 * MM, 6.0 * MM, 45.0 * MM  # a 20 cm glass dish, 45 mm deep (ours)
# the grater: Microplane Professional coarse paddle grater 38000: 13 x 2-5/8 x 1 in, grating surface 5-1/4 x
# 2-5/16 in (133 x 59 mm); its frame rests on the dish's rim, tip end and handle
WIN_W, WIN_L = 59.0 * MM, 133.0 * MM
TIP_END = 45.0 * MM  # the frame beyond the window at the tip (it rests on the rim)
HANDLE_L, HANDLE_W, HANDLE_H = 140.0 * MM, 30.0 * MM, 18.0 * MM
RAIL_H = 3.0 * MM
SURF_Z = RAIL_H + 8.0 * MM  # the grating surface stands 8 mm above the frame's rails (in the grater's frame), so the
# block, worn into the surface, never sits on them (ours: on the real grater the frame's edges are nearly flush)
GRATER_Y = DISH_XY[1] - 0.006  # the window's middle: its tip end on the dish's rim at -y
# the cheese: half of a Tillamook sharp cheddar 8 oz block (5.5 x 2.25 x 1.13 in): 70 x 57 x 29 mm, 113 g; it lies
# along the grater, its 57 mm width inside the 59 mm grating surface
CH_L, CH_W, CH_H = 70.0 * MM, 57.0 * MM, 29.0 * MM
SOLID_FROM = 14.0 * MM  # drawn from here up; below, a wear layer the model moves up as the underside is grated
BOARD_XY, BOARD_T = (0.17, -0.20), 12.0 * MM
TARGET = (0.015, 0.025)  # kg grated for full marks (about 20 g)


def _dish_shapes() -> list:
    ax, ay = DISH_XY
    return [*ring(0.0, DISH_H, DISH_R - DISH_WALL, DISH_R, GLASS, segments=24, centre=DISH_XY),
            *prism12((ax, ay, 2.5 * MM), DISH_R - DISH_WALL, 5.0 * MM, GLASS)]


DISH = body("dish", pos=(0.0, 0.0, TOP), friction=0.2, shapes=_dish_shapes())  # glass: slippery
BOARD = body("board", pos=(*BOARD_XY, TOP), friction=0.7,
             shapes=[box((0.0, 0.0, BOARD_T / 2), (0.25, 0.15, BOARD_T), WOOD)])


def _grater_shapes() -> tuple[list, list]:
    """In the grater's frame: origin at the window's middle on the rails' undersides, y towards the handle (+y),
    the tip at -y."""
    y0, y1 = -WIN_L / 2, WIN_L / 2
    solid = [box((s * (WIN_W / 2 + 2.0 * MM), (y0 - TIP_END + y1) / 2, RAIL_H / 2), (4.0 * MM, WIN_L + TIP_END, RAIL_H),
                 STEEL) for s in (-1, 1)]
    solid += [box((0.0, y0 - TIP_END + 4.0 * MM, RAIL_H / 2), (WIN_W + 8.0 * MM, 8.0 * MM, RAIL_H), STEEL),
              box((0.0, y1 + HANDLE_L / 2, HANDLE_H / 2), (HANDLE_W, HANDLE_L, HANDLE_H), BLACK)]
    seen = [box((0.0, 0.0, SURF_Z - 0.5 * MM), (WIN_W, WIN_L, 1.0 * MM), STEEL)]  # the blades' surface
    for k in range(9):  # rows of cutting teeth (drawn)
        seen.append(box((0.0, y0 + (k + 0.5) * WIN_L / 9, SURF_Z + 0.3 * MM), (WIN_W - 4.0 * MM, 2.0 * MM, 0.6 * MM),
                        (0.55, 0.56, 0.58)))
    return solid, seen


_GS, _GV = _grater_shapes()
GRATER_Z = DISH_H  # the rails rest on the rim
GRATER = body("grater", pos=(DISH_XY[0], GRATER_Y, TOP + GRATER_Z), mass=0.11, friction=0.2, shapes=_GS + _GV,
              visual_only=tuple(range(len(_GS), len(_GS) + len(_GV))), com=(0.0, 0.06, 0.006),
              inertia=(1.2e-3, 5e-5, 1.2e-3), solver_iterations=32)

# its frame: origin on its underside as it came, x across it (its width), y along it; it collides with the board
# and the table through a full, undrawn box; on the grater only the model's contact acts (the grating surface
# does not collide)
CHEESE = body("cheese", pos=(*BOARD_XY, TOP + BOARD_T), mass=0.113, friction=0.6,
              shapes=[box((0.0, 0.0, CH_H / 2), (CH_W, CH_L, CH_H), CHEDDAR),
                      box((0.0, 0.0, (SOLID_FROM + CH_H) / 2), (CH_W, CH_L, CH_H - SOLID_FROM), CHEDDAR)],
              visual_only=(1,), com=(0.0, 0.0, CH_H / 2), inertia=(5.4e-5, 3.9e-5, 7.7e-5), solver_iterations=32)
WEAR = body("cheese_under", pos=(BOARD_XY[0], BOARD_XY[1], TOP + BOARD_T + SOLID_FROM / 2), mass=0.001,
            shapes=[box((0.0, 0.0, 0.0), (CH_W, CH_L, SOLID_FROM), CHEDDAR)], visual_only=(0,))
SHREDS = body("shreds", pos=(DISH_XY[0], GRATER_Y, TOP - 0.02), mass=0.001,
              shapes=[cyl((0.0, 0.0, 0.0), 30.0 * MM, 32.0 * MM, CHEDDAR)], visual_only=(0,))
OBJECTS = [DISH, BOARD, GRATER, CHEESE, WEAR, SHREDS]
gratekit.register(*OBJECTS)
gratekit.hidden("cheese", 0)
for _n in ("cheese_under", "shreds"):
    gratekit.no_gravity(_n)

_GRID = [(x, y) for x in (-25.0 * MM, -12.0 * MM, 0.0, 12.0 * MM, 25.0 * MM)
         for y in (-30.0 * MM, -15.0 * MM, 0.0, 15.0 * MM, 30.0 * MM)]
_bx, _by = BOARD_XY
MODEL = gratekit.add_model(Grating(
    "cheese", "grater", _GRID, SOLID_FROM, SURF_Z, WIN_W / 2, (-WIN_L / 2, WIN_L / 2),
    planes=[], area=CH_W * CH_L, wear=float(E.get("GRATE_WEAR", "8e-6")), wear_layer="cheese_under", layer_h=SOLID_FROM,
    shreds="shreds", pile_at=(DISH_XY[0], GRATER_Y, TOP + 5.0 * MM), dish_xy=(DISH_XY[0], GRATER_Y)))

# -- scoring -------------------------------------------------------------------------------------------------------
# 1.0   15-25 g grated into the dish, the grater still across it, the cheese back on its board, hands clear, still
# 0.5   5 g or more grated, but outside 15-25 g, or the cheese not back on the board (or not yet still)
# 0.25  some grated (under 5 g)
# 0     nothing grated; or, latched, a finger on the blades or the grater off the dish
SCORE_OVER = "end"
RUN_LIMIT_STEPS = 2400
STILL_STEPS = 20
_STATE: dict = {}


@gratekit.on_reset
def _reset_state() -> None:
    _STATE.update(still=0)


_reset_state()


def cheese_on_board(objects) -> bool:
    s = objects["cheese"]
    p = np.asarray(s.pos, float) - np.array([0.0, 0.0, TOP])
    return abs(p[0] - BOARD_XY[0]) < 0.10 and abs(p[1] - BOARD_XY[1]) < 0.06 and s.tilt_deg() < 15.0


def hands_clear(objects) -> bool:
    pts = gratekit.finger_points()
    if not len(pts):
        return True
    s = objects["cheese"]
    loc = (pts - np.asarray(s.pos, float)) @ s.rotation()
    out = np.maximum(np.abs(loc) - np.array([CH_L / 2, CH_W / 2, CH_H / 2]), 0.0)
    return float(np.linalg.norm(out, axis=1).min()) >= 0.01


def _update(objects) -> tuple[float, bool]:
    m = MODEL
    if m.knuckles or m.off_dish:
        return 0.0, False
    g = m.grated
    if g >= 0.005:
        ok = (TARGET[0] <= g <= TARGET[1] and cheese_on_board(objects) and objects["cheese"].resting()
              and objects["grater"].resting() and hands_clear(objects))
        return (1.0, True) if ok else (0.5, False)
    if g > 0.0:
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
GRIP_UP = 17.0 * MM + TIP  # the TCP above the block's underside: fingertips 17 mm up its sides
GRIP = -(CH_W / 2 - 3.0 * MM)  # across its width (x)
OPEN = -(CH_W / 2 + 10.0 * MM)
HOLD = -(HANDLE_W / 2 - 4.0 * MM)
PRESS = 30.0 * MM  # the push aims this far into the grater (the soft arm gives about 0.3 N/mm: ~9 N)
N_STROKES = int(E.get("GRATE_STROKES", "8"))


def left_hold() -> list:
    x, y = DISH_XY[0], GRATER_Y + WIN_L / 2 + 0.08
    z = TOP + GRATER_Z + HANDLE_H / 2 + TIP - 2.0 * MM
    yaw = math.pi / 2  # fingers close along x, across the handle
    return [Waypoint(2.0, (x, y, z + 0.05), -(HANDLE_W / 2 + 10.0 * MM), "over the handle", yaw),
            Waypoint(1.2, (x, y, z), -(HANDLE_W / 2 + 10.0 * MM), "round it", yaw),
            Waypoint(1.2, (x, y, z), HOLD, "hold the grater", yaw)]


def right_grate() -> list:
    bx, by = BOARD_XY
    z0 = TOP + BOARD_T + GRIP_UP
    zs = TOP + GRATER_Z + SURF_Z + GRIP_UP  # the block's underside on the grating surface
    x = DISH_XY[0]
    lag = 12.0 * MM  # the soft arm trails the target this much against the friction: aim past each end
    y_hi = GRATER_Y + WIN_L / 2 - CH_L / 2 - 3.0 * MM + lag  # the stroke starts at the handle end of the window
    y_lo = GRATER_Y - WIN_L / 2 + CH_L / 2 + 3.0 * MM - lag  # and ends at its tip end
    yaw = math.pi / 2  # fingers close along x: across the block's width
    out = [Waypoint(2.0, (bx, by, z0 + 0.05), OPEN, "over the cheese", yaw),
           Waypoint(1.2, (bx, by, z0), OPEN, "round it", yaw),
           Waypoint(1.4, (bx, by, z0), GRIP, "take the cheese", yaw),
           Waypoint(1.2, (bx, by, zs + 0.04), GRIP, "lift it", yaw),
           Waypoint(2.0, (x, y_hi, zs + 0.03), GRIP, "over the grater", yaw),
           Waypoint(1.0, (x, y_hi, zs + 0.002), GRIP, "onto the blades", yaw)]
    for _ in range(N_STROKES):
        out += [Waypoint(0.5, (x, y_hi, zs - PRESS), GRIP, "press", yaw),
                Waypoint(2.0, (x, y_lo, zs - PRESS), GRIP, "grate: stroke away from the handle", yaw),
                Waypoint(0.4, (x, y_lo, zs + 0.006), GRIP, "ease off", yaw),
                Waypoint(0.9, (x, y_hi, zs + 0.006), GRIP, "back", yaw)]
    out += [Waypoint(1.0, (x, y_hi, zs + 0.04), GRIP, "lift it off", yaw),
            Waypoint(2.0, (bx, by, z0 + 0.04), GRIP, "back over the board", yaw),
            Waypoint(1.2, (bx, by, z0 + 0.001), GRIP, "down on the board", yaw),
            Waypoint(1.4, (bx, by, z0 + 0.001), OPEN, "let go", yaw),
            Waypoint(1.2, (bx, by, z0 + 0.05), OPEN, "up", yaw),
            Waypoint(1.5, RHOME, FINGER_OPEN, "home", 0.0)]
    return out


def demo():
    right = right_grate()
    left = left_hold()
    t_hold = sum(w.duration for w in right[:-6]) - sum(w.duration for w in left)
    left.append(Waypoint(max(0.05, t_hold), left[-1].pos, left[-1].finger, "steady the grater", left[-1].yaw))
    lx, ly, lz = left[-1].pos
    left += [Waypoint(1.2, (lx, ly, lz), -(HANDLE_W / 2 + 10.0 * MM), "let go", left[-1].yaw),
             Waypoint(1.0, (lx, ly, lz + 0.05), -(HANDLE_W / 2 + 10.0 * MM), "up", left[-1].yaw),
             Waypoint(1.5, LHOME, FINGER_OPEN, "home", 0.0)]
    tr, tl = sum(w.duration for w in right), sum(w.duration for w in left)
    if tr < tl:
        right.append(Waypoint(tl - tr, right[-1].pos, right[-1].finger, "wait", right[-1].yaw))
    else:
        left.append(Waypoint(tr - tl, left[-1].pos, left[-1].finger, "wait", left[-1].yaw))
    for plan in (left, right):
        plan.append(Waypoint(2.0, plan[-1].pos, plan[-1].finger, "done", plan[-1].yaw))
    return {"right": right, "left": left}
