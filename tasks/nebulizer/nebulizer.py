"""On the station table: a compressor nebulizer's parts, washed and dried after a treatment. The compressor stands on
the right with its air tube on its connector, the tube lying coiled in front of it. The kit's medication tank stands in
its kit stand; the inhalation top, the cap and the mouthpiece lie on a drying mat.

The robot must put the nebulizer together, ready for the next treatment: the inhalation top into the tank, lined up
by its tab and turned clockwise until it clicks (with the tank held, or it turns with the top); the cap on the top's
air inlet; the mouthpiece on its outlet; the air tube's free end pushed onto the nozzle under the tank; and the kit
stood back in its stand.

Run it:  inspect-robots run --embodiment yam_arena -E task=nebulizer.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "Put the nebulizer together, ready for the next treatment." --scorer task_score ...
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nebkit  # noqa: E402
from nebkit import Ball, Nebulizer, body, box, cyl, prism12, tilted  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

nebkit.install()
SCENE = "station"
TOP = TABLE_TOP[SCENE]
MM = 0.001
E = os.environ

WHITE = (0.92, 0.93, 0.94)
CLEAR = (0.80, 0.86, 0.90)
BLUE = (0.30, 0.55, 0.85)
GREY = (0.62, 0.64, 0.66)
DARK = (0.18, 0.19, 0.21)
MAT = (0.35, 0.60, 0.70)
WOOD = (0.62, 0.47, 0.30)
nebkit.METALLIC.update({GREY: 0.2})

# -- sizes (the maker publishes the compressor's, not the kit's: the kit's below are our estimates) -------------------
TANK_R, TANK_H = 22.0 * MM, 42.0 * MM  # the medication tank's body below its rim
SHOULDER_R, SHOULDER_Z = 30.0 * MM, -38.0 * MM  # a moulded shoulder that rests in the stand (its underside)
NOZZLE_R, NOZZLE_L = 3.0 * MM, 10.0 * MM  # the air nozzle under the tank
TOP_R, TOP_H = 20.0 * MM, 30.0 * MM  # the inhalation top's body
INLET_R, INLET_TOP = 6.0 * MM, 42.0 * MM  # its air inlet, upright on its top (the cap goes on it)
CAP_R, CAP_H, CAP_DEPTH = 8.0 * MM, 14.0 * MM, 10.0 * MM
OUTLET_R, OUTLET_LEN, OUTLET_Z = 7.0 * MM, 20.0 * MM, 15.0 * MM  # its outlet, out of its side (the mouthpiece)
LOCK_DEG = 40.0  # the turn from the tab's notch to the click
MP_L, MP_W, MP_H, MP_DEPTH = 65.0 * MM, 24.0 * MM, 16.0 * MM, 12.0 * MM
CUFF_L, CUFF_W, CUFF_DEPTH = 24.0 * MM, 10.0 * MM, 8.0 * MM

# -- the stand (fixed): a wooden fork whose prongs take the tank's shoulder; open underneath for the tube -------------
# WHY here (2026-11-15, sim trace): a level grip reaches only along x 0.17 (its TCP); its fingertips are 12.6 mm
# beyond the TCP and its pads about 8 mm behind them, so the pads centre on a tank whose axis is 4.6 mm beyond.
STAND_XY = (0.1746, 0.08)  # the tank's axis in the stand
# WHY 55 mm (2026-11-15, sim traces): the tube's cuff and the link behind it hang 44 mm under the nozzle; lower, they
# met the table, propped the kit up and tipped it over when it was let go.
PRONG_TOP = 55.0 * MM
PRONG_T = 5.0 * MM
GAP = TANK_R + 2.0 * MM  # half the gap between the prongs


def _stand_shapes() -> list:
    zp = PRONG_TOP - PRONG_T / 2
    return [
        box((0.0, +(GAP + 6.0 * MM), zp), (70.0 * MM, 12.0 * MM, PRONG_T), WOOD),  # the prongs, towards the robot
        box((0.0, -(GAP + 6.0 * MM), zp), (70.0 * MM, 12.0 * MM, PRONG_T), WOOD),
        box((41.0 * MM, 0.0, PRONG_TOP / 2), (12.0 * MM, 2 * GAP + 24.0 * MM, PRONG_TOP), WOOD),  # the back
    ]


STAND = body("stand", pos=(*STAND_XY, TOP), shapes=_stand_shapes(), friction=0.6)

# -- the medication tank (origin at the middle of its rim) ----------------------------------------------------------
TANK_Z0 = PRONG_TOP - SHOULDER_Z  # its rim above the table, standing in the stand


def _tank_shapes() -> tuple[list, list]:
    solid = [*prism12((0.0, 0.0, -TANK_H / 2), TANK_R, TANK_H, CLEAR),
             *prism12((0.0, 0.0, SHOULDER_Z + 1.0 * MM), SHOULDER_R, 2.0 * MM, CLEAR)]
    seen = [cyl((0.0, 0.0, -TANK_H - NOZZLE_L / 2), NOZZLE_R, NOZZLE_L, CLEAR),  # the air nozzle (drawn only)
            box((TANK_R + 0.6 * MM, 0.0, -2.0 * MM), (1.2 * MM, 3.0 * MM, 4.0 * MM), BLUE),  # the notch's mark
            cyl((0.0, 0.0, -TANK_H + 6.0 * MM), TANK_R - 2.0 * MM, 0.2 * MM, BLUE)]  # the 7 ml line, seen inside
    return solid, seen


_TS, _TV = _tank_shapes()
# WHY 45 g: the parts once on are drawn by the model and weigh nothing in the simulation; the tank carries the
# assembled kit's weight, so the tube hanging from it cannot pull it over.
TANK = body("tank", pos=(*STAND_XY, TOP + TANK_Z0), mass=0.045, friction=0.8, shapes=_TS + _TV,
            visual_only=tuple(range(len(_TS), len(_TS) + len(_TV))), solver_iterations=32)

# -- the drying mat and the parts on it -----------------------------------------------------------------------------
MAT_XY, MAT_SIZE = (0.265, 0.035), (0.11, 0.12, 0.003)
MAT_BODY = body("mat", pos=(*MAT_XY, TOP), shapes=[box((0.0, 0.0, MAT_SIZE[2] / 2), MAT_SIZE, MAT)], friction=0.8)
ON_MAT = TOP + MAT_SIZE[2]
OUT_YAW = math.radians(LOCK_DEG)  # the outlet's direction in the top's frame: +x once the top is turned to the click
TOP_XY, CAP_XY = (0.245, 0.005), (0.245, 0.06)
MP_XY = (0.27, 0.055)  # the mouthpiece's mouth; it lies along +x


def _top_shapes() -> tuple[list, list]:
    c, s = math.cos(OUT_YAW), math.sin(OUT_YAW)
    solid = [*prism12((0.0, 0.0, TOP_H / 2), TOP_R, TOP_H, WHITE)]
    seen = [cyl((0.0, 0.0, (TOP_H + INLET_TOP) / 2), INLET_R, INLET_TOP - TOP_H, WHITE),  # the air inlet
            box(((TOP_R + OUTLET_LEN / 2) * c, (TOP_R + OUTLET_LEN / 2) * s, OUTLET_Z),
                (OUTLET_LEN + 2.0 * MM, 2 * OUTLET_R, 2 * OUTLET_R), WHITE, yaw=OUT_YAW),  # the outlet
            box((TOP_R + 0.6 * MM, 0.0, 2.0 * MM), (1.2 * MM, 3.0 * MM, 4.0 * MM), BLUE),  # the tab
            cyl((0.0, 0.0, INLET_TOP - 0.1 * MM), INLET_R - 1.5 * MM, 0.2 * MM, DARK)]  # the inlet's bore
    return solid, seen


_OS, _OV = _top_shapes()
TOP_PART = body("top", pos=(*TOP_XY, ON_MAT + 0.5 * MM), mass=0.012, friction=0.8, shapes=_OS + _OV,
                visual_only=tuple(range(len(_OS), len(_OS) + len(_OV))), solver_iterations=32)
CAP = body("cap", pos=(*CAP_XY, ON_MAT + 0.5 * MM), mass=0.003, friction=0.8,
           shapes=[*prism12((0.0, 0.0, CAP_H / 2), CAP_R, CAP_H, BLUE)], solver_iterations=32)


def _mp_shapes() -> tuple[list, list]:
    """The mouthpiece, along its x axis from its mouth (the socket that goes over the outlet)."""
    solid = [box((MP_DEPTH + (MP_L - MP_DEPTH) / 2 + 2.0 * MM, 0.0, 0.0), (MP_L - MP_DEPTH - 4.0 * MM, MP_W, MP_H), WHITE)]
    seen = [box((MP_DEPTH / 2 + 1.0 * MM, 0.0, 0.0), (MP_DEPTH + 2.0 * MM, 2 * OUTLET_R + 4.0 * MM, 2 * OUTLET_R + 4.0 * MM),
                WHITE),
            box((MP_L - 1.0 * MM, 0.0, 0.0), (2.0 * MM, MP_W - 6.0 * MM, 4.0 * MM), DARK)]  # its lip opening
    return solid, seen


_MS, _MV = _mp_shapes()
# WHY its centre of mass mid-way (2026-11-15, sim trace): left at its origin (the mouth, which has no collider so it
# can slide over the outlet) the mouthpiece tipped up onto its end as soon as the run started.
MP = body("mp", pos=(MP_XY[0], MP_XY[1], ON_MAT + MP_H / 2 + 0.5 * MM), mass=0.008, com=(MP_L / 2 + 5.0 * MM, 0.0, 0.0),
          friction=0.8, shapes=_MS + _MV,
          visual_only=tuple(range(len(_MS), len(_MS) + len(_MV))), solver_iterations=32)

# -- the compressor (fixed) and its air tube -------------------------------------------------------------------------
COMP_XY, COMP_SIZE = (0.37, -0.15), (0.098, 0.142, 0.072)  # NE-C801: 142 W x 72 H x 98 D mm
CONNECTOR = np.array([COMP_XY[0] - COMP_SIZE[0] / 2, -0.10, 0.030])  # its air connector, on its front face


def _comp_shapes() -> tuple[list, list]:
    solid = [box((0.0, 0.0, COMP_SIZE[2] / 2), COMP_SIZE, WHITE)]
    seen = [box((-COMP_SIZE[0] / 2 - 0.5 * MM, -0.035, 0.045), (1.0 * MM, 0.020, 0.012), BLUE),  # the switch
            box((-COMP_SIZE[0] / 2 - 4.0 * MM, CONNECTOR[1] - COMP_XY[1], CONNECTOR[2]), (8.0 * MM, 6.0 * MM, 6.0 * MM),
                GREY)]  # the air connector
    return solid, seen


_CS, _CV = _comp_shapes()
COMPRESSOR = body("compressor", pos=(*COMP_XY, TOP), shapes=_CS + _CV,
                  visual_only=tuple(range(len(_CS), len(_CS) + len(_CV))), friction=0.6)

N_LINKS = 25
LINK = 1.0 / N_LINKS  # the 100 cm tube, in 4 cm links
TUBE_R = 3.25 * MM  # PVC tube, 6.5 mm across


def _tube_path() -> list:
    """The tube's centre line from the compressor's connector to its free end, as N_LINKS + 1 points (table frame):
    off the connector, down to the table, in rows along y, the last row running to +y and ending at the free end."""
    z = TUBE_R + 0.5 * MM
    c = CONNECTOR
    line = [np.array([c[0] - 0.006, c[1], c[2]]), np.array([c[0] - 0.006 - LINK, c[1], c[2]]),
            np.array([0.255, c[1], z])]
    line += [np.array([*xy, z]) for xy in ((0.255, -0.24), (0.230, -0.24), (0.230, -0.07), (0.205, -0.07),
                                           (0.205, -0.24), (0.180, -0.24), (0.180, -0.07), (0.155, -0.07),
                                           (0.155, -0.24), (0.130, -0.24), (0.130, 0.20))]
    pts, j, t0, cur = [line[0]], 0, 0.0, line[0]
    while len(pts) < N_LINKS + 1:
        # the first point along the line, past (j, t0), exactly LINK from cur
        found = None
        for jj in range(j, len(line) - 1):
            a, b = line[jj], line[jj + 1]
            d, f = b - a, a - cur
            A, B, C = d @ d, 2 * f @ d, f @ f - LINK * LINK
            disc = B * B - 4 * A * C
            if disc < 0:
                continue
            for t in sorted(((-B - math.sqrt(disc)) / (2 * A), (-B + math.sqrt(disc)) / (2 * A))):
                if 0.0 <= t <= 1.0 and (jj > j or t > t0 + 1e-9):
                    found = (jj, t)
                    break
            if found:
                break
        j, t0 = found
        cur = line[j] + t0 * (line[j + 1] - line[j])
        pts.append(cur)
    return pts


def _quat_x_to(d) -> tuple:
    """xyzw quaternion turning +x onto the unit vector ``d``."""
    d = np.asarray(d, float) / np.linalg.norm(d)
    x = np.array([1.0, 0.0, 0.0])
    c = float(x @ d)
    if c > 1 - 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    if c < -1 + 1e-9:
        return (0.0, 0.0, 1.0, 0.0)
    ax = np.cross(x, d)
    ax /= np.linalg.norm(ax)
    a = math.acos(c)
    s = math.sin(a / 2)
    return (ax[0] * s, ax[1] * s, ax[2] * s, math.cos(a / 2))


PATH = _tube_path()
CUFF_XY = (float(PATH[-1][0]), float(PATH[-1][1]))  # the tube's free end on the table: its mouth, pointing to +y
LINKS = []
for i in range(N_LINKS):
    a, b = PATH[i], PATH[i + 1]
    last = i == N_LINKS - 1
    if last:  # the free end: origin at its cuff's mouth (b), x axis pointing back into the tube
        shapes = [box((CUFF_L / 2, 0.0, 0.0), (CUFF_L, CUFF_W, CUFF_W), GREY),
                  box((CUFF_L + (LINK - CUFF_L - 1.0 * MM) / 2, 0.0, 0.0), (LINK - CUFF_L - 1.0 * MM, 2 * TUBE_R, 2 * TUBE_R),
                      CLEAR)]
        LINKS.append(body("cuff", pos=tuple(b), quat=_quat_x_to(a - b), mass=0.003, friction=0.8, shapes=shapes,
                          inertia=(1e-7, 4e-7, 4e-7), solver_iterations=32))
    else:
        shapes = [box((0.0, 0.0, 0.0), (LINK - 1.0 * MM, 2 * TUBE_R, 2 * TUBE_R), CLEAR)]
        if i == 0:
            shapes.append(box((-LINK / 2 + 6.0 * MM, 0.0, 0.0), (12.0 * MM, CUFF_W, CUFF_W), GREY))
        LINKS.append(body(f"tube{i:02d}", pos=tuple((a + b) / 2 + np.array([0.0, 0.0, TOP])), quat=_quat_x_to(b - a),
                          mass=0.0012, friction=0.8, shapes=shapes, inertia=(2e-8, 2e-7, 2e-7), solver_iterations=32))
# the cuff's spawn position is in table coordinates like the rest
LINKS[-1] = body("cuff", pos=tuple(PATH[-1] + np.array([0.0, 0.0, TOP])), quat=_quat_x_to(PATH[-2] - PATH[-1]), mass=0.003,
                 friction=0.8, shapes=LINKS[-1].shapes, inertia=(1e-7, 4e-7, 4e-7), solver_iterations=32)

OBJECTS = [STAND, TANK, MAT_BODY, TOP_PART, CAP, MP, COMPRESSOR, *LINKS]
nebkit.register(*OBJECTS)
for i in range(1, N_LINKS):
    parent, child = LINKS[i - 1], LINKS[i]
    nebkit.add_ball(Ball(parent.name, child.name, anchor=(LINK / 2, 0.0, 0.0)))

MODEL = nebkit.add_model(Nebulizer(
    "tank", "top", "cap", "mp", "cuff", rim_r=TANK_R, top_r=TOP_R, tank_r=TANK_R,
    inlet=(0.0, 0.0), inlet_top=INLET_TOP, cap_depth=CAP_DEPTH,
    outlet=((TOP_R + OUTLET_LEN) * math.cos(OUT_YAW), (TOP_R + OUTLET_LEN) * math.sin(OUT_YAW), OUTLET_Z),
    outlet_dir=(math.cos(OUT_YAW), math.sin(OUT_YAW), 0.0), mp_depth=MP_DEPTH,
    nozzle=(0.0, 0.0, -TANK_H - NOZZLE_L), nozzle_dir=(0.0, 0.0, -1.0), cuff_depth=CUFF_DEPTH,
    fixed_end=(LINKS[0].name, np.array(LINKS[0].pos), LINKS[0].quat), lock_deg=LOCK_DEG))

# -- scoring -------------------------------------------------------------------------------------------------------
# 1.0   the inhalation top locked in the tank (turned to the click), the cap on its air inlet, the mouthpiece on its
#       outlet, the tube's cuff on the tank's nozzle, the kit standing upright in its stand (its shoulder on the prongs,
#       within 10 degrees of upright); no part of either gripper within 1 cm of the kit; the kit still for 1 s
# 0.5   all of that put together but the kit not back in its stand (or not yet still for 1 s)
# 0.25  the top locked in the tank
# 0     anything else
SCORE_OVER = "end"
RUN_LIMIT_STEPS = 2400
CLEAR_DIST = 10.0 * MM
STILL_STEPS = 20  # 1 s at 20 Hz
KIT = ("tank", "top", "cap", "mp")
_STATE: dict = {}


@nebkit.on_reset
def _reset_state() -> None:
    _STATE.update(still=0)


_reset_state()


def in_stand(tank) -> bool:
    off = np.hypot(*(np.asarray(tank.pos[:2]) - STAND_XY))
    return bool(off < 6.0 * MM and abs(tank.pos[2] - (TOP + TANK_Z0)) < 3.0 * MM and tank.tilt_deg() < 10.0)


def fingers_to(state, spec) -> float:
    pts = nebkit.finger_points()
    if not len(pts):
        return math.inf
    lo, hi = spec.bounds_local()
    local = (pts - state.pos) @ state.rotation()
    d = np.maximum(np.maximum(lo - local, local - hi), 0.0)
    return float(np.linalg.norm(d, axis=1).min())


SPECS = {s.name: s for s in OBJECTS}


def _update(objects) -> tuple[float, bool]:
    m = MODEL
    if not m.locked:
        return 0.0, False
    if not (m.cap_on and m.mp_on and m.cuff_on):
        return 0.25, False
    tank = objects["tank"]
    if in_stand(tank) and min(fingers_to(objects[n], SPECS[n]) for n in KIT) >= CLEAR_DIST:
        return 1.0, bool(tank.resting())
    return 0.5, False


def success(objects) -> bool:
    """Put together and back in its stand, let go, still (held 1 s by the run). Also counts the steps the full-score
    state has held, for score()."""
    ok = _update(objects)[1]
    _STATE["still"] = _STATE["still"] + 1 if ok else 0
    return ok


def score(objects) -> float:
    """1.0 / 0.5 / 0.25 / 0 as above; 1.0 only once the full-score state has held for 1 s (counted by success(),
    which the run calls first every step); until then 0.5."""
    tier = _update(objects)[0]
    if tier == 1.0 and _STATE["still"] < STILL_STEPS:
        return 0.5
    return tier


# -- scripted demo -------------------------------------------------------------------------------------------------
# Open loop, from the fixed layout. The right hand fits the top, the cap and the mouthpiece; the left holds the tank
# for the top's turn, then lifts the kit and lays it on its side so the right can push the tube onto its nozzle, and
# stands it back.
TIP = 13.0 * MM  # the fingertips below a straight-down TCP
LEVEL_BACK = 4.6 * MM  # a level grip's pads' middle beyond its TCP (fingertips 12.6 mm beyond, measured)
RHOME, LHOME = (0.16, -0.22, TOP + 0.15), (0.16, 0.22, TOP + 0.15)
OPEN_TOP = -32.0 * MM  # 64 mm between the fingertips, round a 40 mm part
OPEN_SMALL = -18.0 * MM
# WHY gentle grips (2026-11-15, sim trace): closed fully (50 N) on the 12 g top, the fingers shot it 55 mm out of the
# grip. Each part is held with the fingers commanded 1 mm inside its width: about 5 N a finger (servo 5000 N/m).
SQUEEZE = 1.0 * MM
HOLD_TOP = -(TOP_R - SQUEEZE)
HOLD_CAP = -(CAP_R - SQUEEZE)
HOLD_MP = -(MP_W / 2 - SQUEEZE)
HOLD_CUFF = -(CUFF_W / 2 - SQUEEZE)
HOLD_TANK = -(TANK_R - SQUEEZE)
L_ROLL = math.radians(float(E.get("NEB_ROLL", "-90")))  # the left hand's roll that lays the kit nozzle towards the right
# WHY (2026-11-15, sim trace): 15 mm down, the fingers' lower edges met the stand's prongs and stopped 70 mm apart.
GRIP_TANK_Z = -12.0 * MM  # the left hand closes on the tank this far below its rim


def down(yaw_deg: float) -> float:
    return math.radians(yaw_deg)


def _t(w) -> float:
    return sum(p.duration for p in w)


def wait(t: float, last: Waypoint, label: str) -> Waypoint:
    return Waypoint(max(0.05, t), last.pos, last.finger, label, last.yaw)


def right_top() -> list:
    """Take the top from the mat (fingers across it, clear of its outlet), set it on the tank by its tab."""
    x, y = TOP_XY
    zg = ON_MAT + 10.0 * MM + TIP
    sx, sy = STAND_XY
    zs = TOP + TANK_Z0 + 0.5 * MM + 10.0 * MM + TIP
    yaw = down(LOCK_DEG)  # fingers across the top, at right angles to its outlet
    return [
        Waypoint(2.5, (x, y, zg + 0.06), OPEN_TOP, "over the top", yaw),
        Waypoint(1.5, (x, y, zg), OPEN_TOP, "round the top", yaw),
        Waypoint(1.6, (x, y, zg), HOLD_TOP, "take the top", yaw),
        Waypoint(1.5, (x, y, zg + 0.07), HOLD_TOP, "lift it", yaw),
        Waypoint(2.5, (sx, sy, zs + 0.05), HOLD_TOP, "over the tank", yaw),
        Waypoint(1.5, (sx, sy, zs + 0.002), HOLD_TOP, "set it on the tank", yaw),
    ]


def right_turn() -> list:
    sx, sy = STAND_XY
    zs = TOP + TANK_Z0 + 0.5 * MM + 10.0 * MM + TIP
    out = []
    n = 4
    for k in range(1, n + 1):
        out.append(Waypoint(0.6, (sx, sy, zs), HOLD_TOP, "turn it to the click", down(LOCK_DEG * (1 - k / n) - 3.0 * k / n)))
    out += [Waypoint(1.6, (sx, sy, zs), OPEN_TOP, "let go of the top", down(-3.0)),
            Waypoint(1.0, (sx, sy, zs + 0.05), OPEN_TOP, "up", down(-3.0))]
    return out


def right_cap() -> list:
    x, y = CAP_XY
    low = 3.0 * MM  # the fingertips this far above the cap's base: it stands on them a little above the table
    zg = ON_MAT + low + TIP
    sx, sy = STAND_XY
    zc = TOP + TANK_Z0 + 0.5 * MM + INLET_TOP - 2.0 * MM + low + TIP
    return [
        Waypoint(2.0, (x, y, zg + 0.05), OPEN_SMALL, "over the cap", 0.0),
        Waypoint(1.2, (x, y, zg), OPEN_SMALL, "round the cap", 0.0),
        Waypoint(1.6, (x, y, zg), HOLD_CAP, "take the cap", 0.0),
        Waypoint(1.2, (x, y, zg + 0.06), HOLD_CAP, "lift it", 0.0),
        Waypoint(2.0, (sx, sy, zc + 0.012), HOLD_CAP, "over the inlet", 0.0),
        Waypoint(1.2, (sx, sy, zc), HOLD_CAP, "press it on", 0.0),
        Waypoint(1.6, (sx, sy, zc), OPEN_SMALL, "let go of the cap", 0.0),
        Waypoint(1.0, (sx, sy - 0.04, zc + 0.005), OPEN_SMALL, "aside", 0.0),
    ]


def right_mouthpiece() -> list:
    x, y = MP_XY
    grab = 30.0 * MM  # the TCP this far along the mouthpiece from its mouth
    zg = ON_MAT + MP_H / 2 + TIP
    sx, sy = STAND_XY
    tip_x = sx + TOP_R + OUTLET_LEN  # the outlet's end once the top is locked (pointing +x)
    zo = TOP + TANK_Z0 + 0.5 * MM + OUTLET_Z + TIP
    on_x = tip_x - MP_DEPTH + grab
    return [
        Waypoint(2.0, (x + grab, y, zg + 0.05), OPEN_TOP, "over the mouthpiece", 0.0),
        Waypoint(1.2, (x + grab, y, zg), OPEN_TOP, "round the mouthpiece", 0.0),
        Waypoint(1.6, (x + grab, y, zg), HOLD_MP, "take the mouthpiece", 0.0),
        Waypoint(1.2, (x + grab, y, zg + 0.06), HOLD_MP, "lift it", 0.0),
        Waypoint(2.0, (on_x + 0.02, sy, zo), HOLD_MP, "in line with the outlet", 0.0),
        Waypoint(1.5, (on_x + 0.002, sy, zo), HOLD_MP, "push it on", 0.0),
        Waypoint(1.6, (on_x + 0.002, sy, zo), OPEN_TOP, "let go of it", 0.0),
        Waypoint(1.0, (on_x + 0.012, sy, zo + 0.008), OPEN_TOP, "back off", 0.0),
        Waypoint(1.5, (on_x + 0.012, sy - 0.07, zo + 0.008), OPEN_TOP, "clear of the kit", 0.0),
    ]


def left_hold() -> list:
    sx, sy = STAND_XY
    z = TOP + TANK_Z0 + GRIP_TANK_Z
    lv = tilted(0.0, 0.0)
    return [
        Waypoint(2.5, (sx - LEVEL_BACK, sy + 0.05, z + 0.06), FINGER_OPEN, "turn the hand level", lv),
        Waypoint(1.5, (sx - LEVEL_BACK, sy, z + 0.06), FINGER_OPEN, "above the tank", lv),
        Waypoint(1.5, (sx - LEVEL_BACK, sy, z), FINGER_OPEN, "round the tank", lv),
        Waypoint(1.6, (sx - LEVEL_BACK, sy, z), HOLD_TANK, "hold the tank", lv),
    ]


LIFT = float(E.get("NEB_LIFT", "30")) * MM


def left_lay_over() -> list:
    sx, sy = STAND_XY
    z = TOP + TANK_Z0 + GRIP_TANK_Z
    lv, rolled = tilted(0.0, 0.0), tilted(0.0, 0.0, L_ROLL)
    return [
        Waypoint(1.5, (sx - LEVEL_BACK, sy, z + LIFT), HOLD_TANK, "lift the kit out", lv),
        Waypoint(2.0, (sx - LEVEL_BACK, sy, z + LIFT), HOLD_TANK, "lay it on its side", rolled),
    ]


def right_tube(nozzle_y: float, z_axis: float) -> list:
    """Take the tube's free end by its cuff (it lies along y, mouth to +y), push it onto the nozzle pointing -y."""
    x, y = CUFF_XY
    zg = TOP + CUFF_W / 2 + TIP
    yaw = down(90.0)  # fingers along x, across the cuff
    grab = 12.0 * MM  # the TCP this far back from the cuff's mouth
    on_y = nozzle_y + CUFF_DEPTH + 0.003 - grab
    zt = z_axis + TIP
    return [
        Waypoint(2.0, (x, y - grab, zg + 0.05), OPEN_SMALL, "over the cuff", yaw),
        Waypoint(1.2, (x, y - grab, zg), OPEN_SMALL, "round the cuff", yaw),
        Waypoint(1.6, (x, y - grab, zg), HOLD_CUFF, "take the cuff", yaw),
        Waypoint(1.5, (x, y - grab - 0.02, zt), HOLD_CUFF, "lift it", yaw),
        Waypoint(1.5, (STAND_XY[0], on_y - 0.025, zt), HOLD_CUFF, "in line with the nozzle", yaw),
        Waypoint(1.5, (STAND_XY[0], on_y, zt), HOLD_CUFF, "push it on", yaw),
        Waypoint(1.6, (STAND_XY[0], on_y, zt), OPEN_SMALL, "let go of the cuff", yaw),
        Waypoint(1.2, (STAND_XY[0], on_y - 0.04, zt + 0.04), OPEN_SMALL, "away", yaw),
    ]


def left_stand_back() -> list:
    sx, sy = STAND_XY
    z = TOP + TANK_Z0 + GRIP_TANK_Z
    lv = tilted(0.0, 0.0)
    return [
        Waypoint(2.0, (sx - LEVEL_BACK, sy, z + LIFT), HOLD_TANK, "stand it up", lv),
        Waypoint(2.0, (sx - LEVEL_BACK, sy, z + 0.002), HOLD_TANK, "into the stand", lv),
        Waypoint(1.6, (sx - LEVEL_BACK, sy, z + 0.002), FINGER_OPEN, "let go of the kit", lv),
        Waypoint(1.5, (sx - LEVEL_BACK, sy, z + 0.06), FINGER_OPEN, "up from the kit", lv),
        Waypoint(1.8, (sx - LEVEL_BACK, sy + 0.08, z + 0.03), FINGER_CLOSED, "aside", lv),
    ]


DEMO_MODE = E.get("NEB_DEMO", "")  # shortcut check: "nohold" turns the top with nothing holding the tank


def demo():
    right = right_top()
    if DEMO_MODE == "nohold":  # the left hand comes to the tank only after the top has been turned
        right += right_turn()
        left = [wait(_t(right), Waypoint(0.1, LHOME, FINGER_OPEN, "", 0.0), "wait")] + left_hold()
        right += right_cap() + right_mouthpiece()
    else:
        left = [wait(_t(right) - 6.0, Waypoint(0.1, LHOME, FINGER_OPEN, "", 0.0), "wait")] + left_hold()
        right.append(wait(_t(left) - _t(right) + 0.3, right[-1], "wait for the tank to be held"))
        right += right_turn() + right_cap() + right_mouthpiece()
    left.append(wait(_t(right) - _t(left), left[-1], "hold the tank"))
    lay = left_lay_over()
    left += lay
    # where the nozzle is with the kit on its side: below the grip along the tank's axis, turned by the roll
    grip = np.array([STAND_XY[0], STAND_XY[1], TOP + TANK_Z0 + GRIP_TANK_Z + LIFT])
    to_tip = -TANK_H - NOZZLE_L - GRIP_TANK_Z  # along the tank's axis
    turn = nebkit.tool_frame(0.0, 0.0, L_ROLL) @ nebkit.tool_frame(0.0, 0.0, 0.0).T
    nozzle_y = float(grip[1] + (turn @ np.array([0.0, 0.0, to_tip]))[1])
    tube = right_tube(nozzle_y, grip[2])
    right.append(wait(_t(left) - _t(right) - _t(tube[:4]), right[-1], "wait for the kit"))
    right += tube
    left.append(wait(_t(right) - _t(left) + 0.2, left[-1], "hold it for the tube"))
    left += left_stand_back()
    right.append(Waypoint(2.0, RHOME, OPEN_SMALL, "home", 0.0))
    if _t(left) > _t(right):
        right.append(wait(_t(left) - _t(right), right[-1], "done"))
    else:
        left.append(wait(_t(right) - _t(left), left[-1], "done"))
    for plan in (left, right):
        plan.append(wait(2.0, plan[-1], "done"))
    return {"right": right, "left": left}
