"""On the station table: a cordless drill (Milwaukee 3601-20, M18) lies on its side on the table with its battery
pack (M18 CP2.0, 48-11-1820) on.

The robot must take the battery pack off the drill. The pack is held on by two release buttons,
one on each side: it slides off forwards only while both are pushed in.

Run it:  inspect-robots run --embodiment yam_arena -E task=drill_battery.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "Take the battery pack off the drill." ...
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drillkit  # noqa: E402
from drillkit import Spring, body, box, cyl  # noqa: E402
from yam_arena.analysis import box_signed_distance  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

drillkit.install()
SCENE = "station"
TOP = TABLE_TOP[SCENE]
MM = 0.001
RED = (0.72, 0.07, 0.07)  # Milwaukee red
BLACK = (0.09, 0.09, 0.10)
GREY = (0.35, 0.36, 0.38)
STEEL = (0.70, 0.71, 0.72)
TRAY_C = (0.12, 0.30, 0.62)
drillkit.METALLIC.update({STEEL: 0.3})


def _env(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


# The drill lies on its side on the table, its pack still on, as left on a bench. In the pack's and the drill's own
# frame x is forwards (the way the pack slides off), y across (the release buttons face +y and -y), z up the drill.
# Laid down: x points to the robot's right (world -y), y points down (so one button faces the table, the other up),
# z points away from the robot (world +x), tipped TIP_DEG about world y: the pack (79 mm) is wider than the housing
# (56 mm), so the drill leans down onto its housing.
TIP_DEG = _env("DB_TIP", 4.5)  # WHY: atan(11.5 mm drop over the 146 mm from the pack's far edge to the housing's)


def _ry(a: float) -> np.ndarray:
    c, s_ = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]])


R0 = _ry(math.radians(TIP_DEG)) @ np.column_stack([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
Q0 = None  # filled in below (after the kit is imported)


def W(xl: float, yl: float = 0.0, zl: float = 0.0) -> tuple:
    """World position of a point in the pack's (= the seated drill's, offset along z) frame."""
    return tuple((np.array([PX, PY, PZ]) + R0 @ np.array([xl, yl, zl])).tolist())


# -- the pack: M18 CP2.0 (48-11-1820), 118 x 79 x 55 mm, 0.43 kg; frame origin at its bottom centre, +x forwards ----
PX, PY = 0.18, 0.03  # the pack frame's origin (its bottom centre) on the table
PZ = TOP + 46.1 * MM  # the down button's cap (the lowest point, 42.5 mm out and 44 mm up the tipped pack) on the table
Q0 = drillkit.matrix_to_quat_rel(R0)
PK_L, PK_W, PK_H = 118.0 * MM, 79.0 * MM, 55.0 * MM
BASE_H = 45.0 * MM  # the cell housing; above it the tongue the drill's foot slides on
TONGUE_L = 80.0 * MM  # the tongue runs from x -40 to +40 mm
TONGUE_LOW_W = 36.0 * MM  # its neck (z 45-50 mm)
TONGUE_LIP_W = 44.0 * MM  # its lips (z 50-55 mm): the foot's rails hook under them
TONGUE_X0 = -10.0 * MM  # the tongue's centre along the pack (it sits back from the front)
M_PACK = 0.43
_pshapes = [box((0, 0, BASE_H / 2), (PK_L, PK_W, BASE_H), BLACK),
            box((TONGUE_X0, 0, BASE_H + 2.5 * MM), (TONGUE_L, TONGUE_LOW_W, 5.0 * MM), RED),
            box((TONGUE_X0, 0, BASE_H + 7.5 * MM), (TONGUE_L, TONGUE_LIP_W, 5.0 * MM), RED)]
PACK = body("pack", pos=W(0.0), shapes=_pshapes, quat=Q0, mass=M_PACK, com=(0, 0, 22 * MM), friction=0.5,
            solver_iterations=32)

# -- the release buttons: one on each side face, 3 mm proud, 3 mm of travel, sprung; each carries the latch pawl -------
BTN_X = 22.0 * MM  # along the pack (pack frame)
BTN_Z = (30.0 * MM, 44.0 * MM)
BTN_W = 16.0 * MM  # along x
BTN_PROUD = 3.0 * MM
BTN_TRAVEL = 3.0 * MM
BTN_K = _env("DB_BK", 1500.0)  # N/m
BTN_PRELOAD = _env("DB_BP", 6.0)  # N at rest, 10.5 N fully in: WHY our estimate of a firm two-finger squeeze
PAWL_X = 27.0 * MM  # each button's latch pawl, along the pack (pack frame); it rises through the pack's top
PAWL_Y = (30.0 * MM, 34.0 * MM)  # from the middle, at rest
PAWL_Z = (40.0 * MM, 52.0 * MM)
LATCH_OVERLAP = 1.5 * MM  # the drill's stops overlap the resting pawls by this much (of 3 mm travel)
def _button(side: int):
    """side +1: the left face (+y), -1: the right face."""
    y_face = side * PK_W / 2
    zc = (BTN_Z[0] + BTN_Z[1]) / 2
    py = (PAWL_Y[0] + PAWL_Y[1]) / 2
    shapes = [box((0, side * BTN_PROUD / 2, 0), (BTN_W, BTN_PROUD, BTN_Z[1] - BTN_Z[0]), GREY),  # the cap
              # inside the housing (drawn hidden): an arm from the cap to the pawl, which rises through a slot
              box((PAWL_X - BTN_X, side * -(PK_W / 2 - PAWL_Y[0]) / 2, 42.0 * MM - zc), (6.0 * MM, PK_W / 2 - PAWL_Y[0], 4.0 * MM), GREY),
              box((PAWL_X - BTN_X, side * (py - PK_W / 2), (PAWL_Z[0] + PAWL_Z[1]) / 2 - zc),
                  (6.0 * MM, PAWL_Y[1] - PAWL_Y[0], PAWL_Z[1] - PAWL_Z[0]), RED)]
    return body("button_l" if side > 0 else "button_r",
                pos=W(BTN_X, y_face, (BTN_Z[0] + BTN_Z[1]) / 2), shapes=shapes, mass=0.005, quat=Q0,
                friction=0.8, solver_iterations=32)


BUTTONS = {+1: _button(+1), -1: _button(-1)}
for _side, _b in BUTTONS.items():
    # sprung outwards against its stop (offset 0 = at rest); pushed in up to BTN_TRAVEL
    lo, hi = (-BTN_TRAVEL, 0.0) if _side > 0 else (0.0, BTN_TRAVEL)
    drillkit.hidden(_b.name, 1)  # the arm is inside the housing
    drillkit.add_spring(Spring(PACK.name, _b.name, axis="Y", stiffness=BTN_K, damping=2.0,
                               target=_side * BTN_PRELOAD / BTN_K, lower=lo, upper=hi))

# -- the drill: Milwaukee 3601-20, 145 mm long, 202 mm tall on its pack, 56 mm wide, 1.04 kg bare --------------------
# Frame origin: the middle of its foot's underside, which bears on the tongue's top; +x towards the chuck.
FOOT_Z = TOP + PK_H
M_DRILL = 1.04
RAIL_IN, RAIL_OUT = 18.5 * MM, 21.5 * MM  # the rails under the lips: y from 18.5 to 21.5 mm each side
RAIL_Z = (-9.5 * MM, -5.5 * MM)  # in the grooves under the lips (z 45.5-49.5 mm on the pack)
WALL_IN, WALL_OUT = 22.5 * MM, 26.0 * MM
# The latch: a stop under each side of the foot, just in front of that side's pawl, overlapping the resting pawl by
# LATCH_OVERLAP. The pack slides forwards only once both buttons are pushed in past it.
STOP_X = PAWL_X + 3.0 * MM + 0.5 * MM + 2.5 * MM - TONGUE_X0  # drill frame: 0.5 mm in front of the pawl
STOP = (5.0 * MM, 3.5 * MM)  # x, y
STOP_Y = PAWL_Y[1] - LATCH_OVERLAP + STOP[1] / 2  # centre: its inner face 1.5 mm inside the resting pawl's outer face (y 32.5-36 mm)
STOP_Z = (46.0 * MM - PK_H, 3.0 * MM)  # drill frame: from 9 mm under the foot up into its web
HANDLE_X = -12.0 * MM
HANDLE = (40.0 * MM, 34.0 * MM, 82.0 * MM)
_dshapes = [box((0, 0, 8 * MM), (100 * MM, 56 * MM, 16 * MM), RED),  # the foot
            box((HANDLE_X, 0, 16 * MM + HANDLE[2] / 2), HANDLE, BLACK),  # the grip
            box((10 * MM, 0, 98 * MM + 24 * MM), (110 * MM, 56 * MM, 48 * MM), RED),  # the motor housing
            cyl((65 * MM + 22 * MM, 0, 117 * MM), 20 * MM, 44 * MM, BLACK, axis="X"),  # the chuck
            box((-45 * MM, 0, 60 * MM), (10 * MM, 30 * MM, 30 * MM), GREY)]  # the belt clip boss (back)
for _s in (-1, 1):
    _dshapes += [box((0, _s * (RAIL_IN + RAIL_OUT) / 2, sum(RAIL_Z) / 2), (70 * MM, RAIL_OUT - RAIL_IN, RAIL_Z[1] - RAIL_Z[0]), RED),
                 box((0, _s * (WALL_IN + WALL_OUT) / 2, -4.75 * MM), (70 * MM, WALL_OUT - WALL_IN, 9.5 * MM), RED),
                 box((STOP_X, _s * STOP_Y, sum(STOP_Z) / 2), (STOP[0], STOP[1], STOP_Z[1] - STOP_Z[0]), RED),
                 # its web, from the foot's side out over the pawl's slot, clear above the pack (z 55.5-59 mm)
                 box((STOP_X, _s * (27 * MM + STOP_Y) / 2, 2.25 * MM), (STOP[0], STOP_Y - 27 * MM, 3.5 * MM), RED)]
DRILL = body("drill", pos=W(TONGUE_X0, 0.0, PK_H), shapes=_dshapes, quat=Q0, mass=M_DRILL, com=(5 * MM, 0, 75 * MM),
             friction=0.6, solver_iterations=32)

# The battery terminals' grip: a sprung shoe in the foot presses on the tongue's top (the pack slides against its
# friction). WHY (2026-11-20): our estimate of an M18 pack's sliding drag is about 6 N: the 6 N preload rubs at the
# shoe (mu 0.45) and again where it pulls the rails up under the lips (mu 0.5). At 15 N the drag dragged the held drill.
SHOE = body("terminal_shoe", pos=W(TONGUE_X0, 0.0, PK_H + 1.5 * MM), quat=Q0,
            shapes=[box((0, 0, 0), (30 * MM, 20 * MM, 3 * MM), STEEL)], mass=0.01, friction=0.45, solver_iterations=32)
SHOE_K = 3000.0
SHOE_PRELOAD = _env("DB_SHOE", 6.0)
drillkit.add_spring(Spring(DRILL.name, SHOE.name, axis="Z", stiffness=SHOE_K, damping=5.0,
                           target=-SHOE_PRELOAD / SHOE_K, lower=-4.0 * MM, upper=0.0))
drillkit.shape_friction(PACK.name, 2, 0.45)

OBJECTS = [PACK, BUTTONS[+1], BUTTONS[-1], DRILL, SHOE]
drillkit.register(*OBJECTS)

# -- scoring --------------------------------------------------------------------------------------------------------
# On the drill: the pack's tongue still under the foot's rails (within 5 mm of the rails' line across and up, and slid
#   less than OFF_SLIDE forwards). Off: otherwise.
# Down: the part resting on the tabletop (its lowest point within FLOOR_TOL of it, any way up), inside the table's
#   edges; the drill also within DRILL_MOVE of where it lay.
# 1.0   the pack off the drill and down, the drill down, no gripper within CLEAR_DIST of the pack or the drill, both
#       still; held STILL_STEPS (1 s)
# 0.5   the pack off the drill, anything else (still held, in the air, moving; the drill still held)
# 0.25  the pack slid SLID_MIN or more forwards along the rails, still on the drill
# 0     anything else; and from then on if the pack or the drill ever falls off the table (latched)
SCORE_OVER = "end"
RUN_LIMIT_STEPS = 2400
STILL_STEPS = 20
OFF_SLIDE = TONGUE_L / 2 + 35.0 * MM  # 75 mm: the tongue's back end has left the rails' front end
SLID_MIN = 10.0 * MM
FLOOR_TOL = 3.0 * MM
DRILL_MOVE = 0.15
CLEAR_DIST = 20.0 * MM
STILL_MM = 0.5 * MM
FALL_Z = TOP - 0.02
_STATE: dict = {}
DRILL_START = np.array(DRILL.pos)


@drillkit.on_reset
def _reset_state() -> None:
    _STATE.clear()
    _STATE.update(still=0, n=0, fell=False, prev=None, calm=False)


_reset_state()


def pack_offset(objects) -> np.ndarray:
    """The pack's place relative to its seat, in the drill's frame (m): [slid forwards, across, up]."""
    d, p = objects[DRILL.name], objects[PACK.name]
    return np.asarray(d.to_local(p.pos), float) - np.array([-TONGUE_X0, 0.0, -PK_H])


def on_drill(objects) -> bool:
    o = pack_offset(objects)
    return bool(abs(o[1]) < 5 * MM and abs(o[2]) < 5 * MM and o[0] < OFF_SLIDE)


def button_in(objects, side: int) -> float:
    p, b = objects[PACK.name], objects[BUTTONS[side].name]
    return float(side * (side * PK_W / 2 - p.to_local(b.pos)[1]))


def _corners(spec, st, which=(0,)) -> np.ndarray:
    pts = []
    for i in which:
        sh = spec.shapes[i]
        c, h = np.asarray(sh.centre), sh.half_extent()
        pts += [st.to_world(c + h * np.array([a, b, e])) for a in (-1, 1) for b in (-1, 1) for e in (-1, 1)]
    return np.array(pts)


def pack_down(objects) -> bool:
    """The pack resting on the tabletop: its lowest point (its housing's or a button cap's, which stands 3 mm proud)
    within FLOOR_TOL of it."""
    low = min(_corners(sp, objects[sp.name], which=[0] if sp is not PACK else range(len(PACK.shapes)))[:, 2].min()
              for sp in (PACK, BUTTONS[1], BUTTONS[-1]))
    return bool(abs(low - TOP) < FLOOR_TOL)


def drill_down(objects) -> bool:
    d = objects[DRILL.name]
    pts = _corners(DRILL, d, which=[i for i, sh in enumerate(DRILL.shapes) if sh.kind == "box"])
    moved = np.linalg.norm(np.asarray(d.pos)[:2] - DRILL_START[:2])
    return bool(abs(pts[:, 2].min() - TOP) < FLOOR_TOL and moved < DRILL_MOVE)


def hands_clear(objects) -> bool:
    pts = drillkit.finger_points()
    if not len(pts):
        return True
    for spec in (PACK, DRILL):
        st = objects[spec.name]
        for sh in spec.shapes:
            if box_signed_distance(pts, st.to_world(sh.centre), st.rotation(), np.asarray(sh.size if sh.kind == "box"
                                   else (2 * sh.size[1], 2 * sh.size[0], 2 * sh.size[0]))).min() < CLEAR_DIST:
                return False
    return True


def _track(objects) -> bool:
    now = np.concatenate([np.asarray(objects[s.name].pos, float) for s in (PACK, DRILL)])
    prev, _STATE["prev"] = _STATE["prev"], now
    return prev is not None and bool(np.all(np.abs(now - prev) < STILL_MM))


def _update(objects) -> tuple[float, bool]:
    if any(np.asarray(objects[s.name].pos)[2] < FALL_Z for s in (PACK, DRILL)):
        _STATE["fell"] = True
    if _STATE["fell"]:
        return 0.0, False
    if on_drill(objects):
        return (0.25 if pack_offset(objects)[0] >= SLID_MIN else 0.0), False
    if pack_down(objects) and drill_down(objects) and hands_clear(objects) and _STATE["calm"]:
        return 1.0, True
    return 0.5, False


def _log(objects) -> None:
    _STATE["n"] += 1
    if _STATE["n"] % 10 == 0 and os.environ.get("DB_LOG"):
        d, p = objects[DRILL.name], objects[PACK.name]
        print(f"[drill] n={_STATE['n']} drill={np.round(d.pos, 4)} tilt={d.tilt_deg():.1f} pack={np.round(p.pos, 4)}"
              f" off={np.round(pack_offset(objects) * 1000, 1)} btn=({button_in(objects, 1)*1000:.1f},"
              f"{button_in(objects, -1)*1000:.1f}) on={on_drill(objects)} pdown={pack_down(objects)}"
              f" down={drill_down(objects)} clear={hands_clear(objects)} still={_STATE['still']}"
              f" tier={_update(objects)[0]}", flush=True)
        if os.environ.get("DB_LOG") == "2":
            for k, side in ((0, "left"), (1, "right")):
                for link in ("tip_left", "tip_right"):
                    pts = drillkit.finger_points(robots=(drillkit.ROBOTS[k],), links=(link,))
                    if len(pts):
                        lo = pts[np.argsort(pts[:, 2])[:20]].mean(axis=0)
                        print(f"[drill]   {side} {link} lowest={np.round(lo, 4)}", flush=True)
            for sd in (1, -1):
                print(f"[drill]   cap {sd} at {np.round(objects[BUTTONS[sd].name].pos, 4)}", flush=True)


def success(objects) -> bool:
    """True only once the full-score state has held for STILL_STEPS control steps in a row (1 s), as score() is 1.0
    only then; yam_arena ends the episode after success() has then held for its own SUCCESS_HOLD_S (1 s)."""
    _STATE["calm"] = _track(objects)
    _STATE["still"] = _STATE["still"] + 1 if _update(objects)[1] else 0
    _log(objects)
    return _STATE["still"] >= STILL_STEPS


def score(objects) -> float:
    tier = _update(objects)[0]
    if tier == 1.0 and _STATE["still"] < STILL_STEPS:
        return 0.5
    return tier


# -- scripted demo ---------------------------------------------------------------------------------------------------
# Open loop. The left hand takes the drill's grip from above (fingers closing across it) and lifts the drill and its
# pack 5 cm. The right hand comes at the pack from the robot's side, nearly level, fingers above and below it on the
# two release buttons; it squeezes them in and slides the pack forwards off the foot. The left hand lays the drill back
# down and lets go; the right hand sets the pack down on the table and lets go.
TIP = 13.0 * MM
RHOME, LHOME = (0.14, -0.22, TOP + 0.17), (0.14, 0.22, TOP + 0.17)
GRIP = W(TONGUE_X0 + HANDLE_X, 0.0, PK_H + 16 * MM + HANDLE[2] / 2)  # the grip's middle
H_SQ = -(HANDLE[0] / 2 - _env("DB_HSQ", 6.0) * MM)  # across its 40 mm (along world y)
LIFT = _env("DB_LIFT", 50.0) * MM
SAG = _env("DB_SAG", 40.0) * MM  # WHY: the soft arm sags under the 1.5 kg drill and pack; lead the lift
BP = math.radians(_env("DB_BPITCH", 5.0))  # WHY (2026-11-20): tipped 30 deg, the fingers' up-down span (94 mm x cos)
# was narrower than the 85 mm across the caps and they stopped on the pack's edge; at 5 deg it is 93.6 mm
BG = drillkit.tilted(0.0, BP, -math.pi / 2)  # pointing +x, nearly level, fingers closing up-down
B_SQ = -(PK_W / 2 - _env("DB_BSQ", 4.0) * MM)
B_DX = _env("DB_BDX", 0.6) * MM  # WHY (run c_db6): where the lifted pack's buttons hang, against the laid-down layout
B_DY = _env("DB_BDY", -5.6) * MM
B_DZ = _env("DB_BDZ", -2.0) * MM
SLIDE = _env("DB_SLIDE", 110.0) * MM


def _tcp(tip, heading: float, pitch: float):
    d = np.array([math.cos(heading) * math.cos(pitch), math.sin(heading) * math.cos(pitch), -math.sin(pitch)])
    return tuple((np.asarray(tip, float) - TIP * d).tolist())


def demo():
    g = (GRIP[0], GRIP[1], GRIP[2] + TIP)
    up = (g[0], g[1], g[2] + LIFT + SAG)
    left = [Waypoint(1.0, LHOME, FINGER_OPEN, "wait", 0.0),
            Waypoint(2.0, (g[0], g[1], g[2] + 0.06), FINGER_OPEN, "over the grip", 0.0),
            Waypoint(1.5, g, FINGER_OPEN, "round it", 0.0),
            Waypoint(1.8, g, H_SQ, "grip it", 0.0),
            Waypoint(2.0, up, H_SQ, "lift the drill", 0.0)]
    t_up = sum(w.duration for w in left)
    br = lambda tip: _tcp(tip, 0.0, BP)  # noqa: E731
    cap = W(BTN_X, 0.0, (BTN_Z[0] + BTN_Z[1]) / 2)
    cap = (cap[0] + B_DX, cap[1] + B_DY, cap[2] + LIFT + B_DZ)  # where the lifted pack's buttons are
    near = (cap[0] - 0.07, cap[1], cap[2])
    off = (cap[0], cap[1] - SLIDE, cap[2] + 0.005)
    right = [Waypoint(t_up, RHOME, FINGER_OPEN, "wait", 0.0),
             Waypoint(2.0, br(near), FINGER_OPEN, "level with the pack", BG),
             Waypoint(1.5, br(cap), FINGER_OPEN, "fingers over its buttons", BG),
             Waypoint(2.0, br(cap), B_SQ, "squeeze both", BG),
             Waypoint(3.0, br(off), B_SQ, "slide it off", BG),
             Waypoint(1.0, br(off), B_SQ, "clear", BG)]
    t_off = sum(w.duration for w in right)
    down = (g[0], g[1], g[2] + _env("DB_DOWN", 15.0) * MM)
    left += [Waypoint(t_off - t_up, up, H_SQ, "hold it up", 0.0),
             Waypoint(2.0, down, H_SQ, "lay the drill down", 0.0),
             Waypoint(1.8, down, FINGER_OPEN, "let go", 0.0),
             Waypoint(1.2, (down[0], down[1], down[2] + 0.06), FINGER_OPEN, "up", 0.0),
             Waypoint(2.0, LHOME, FINGER_OPEN, "home", 0.0)]
    t_l = sum(w.duration for w in left)
    # set the pack down beside the drill: lowered until the lower finger is just over the table, then let go (it
    # drops off the finger, about 5 mm)
    low = (off[0], off[1], TOP + PK_W / 2 + BTN_PROUD + _env("DB_SETZ", 1.0) * MM)
    rise = (off[0], off[1], low[2] + 0.05)  # straight up first: backing out low dragged the pack along
    back = (off[0] - 0.045, off[1], rise[2])
    right += [Waypoint(max(t_l - t_off - 2.0, 0.5), br(off), B_SQ, "hold the pack", BG),
              Waypoint(1.5, br(low), B_SQ, "down to the table", BG),
              Waypoint(1.8, br(low), FINGER_OPEN, "let go", BG),
              Waypoint(1.2, br(rise), FINGER_OPEN, "up", BG),
              Waypoint(2.0, br(back), FINGER_OPEN, "back out", BG),
              Waypoint(2.0, RHOME, FINGER_OPEN, "home", 0.0),
              Waypoint(1.5, RHOME, FINGER_OPEN, "done", 0.0)]
    t_r = sum(w.duration for w in right)
    left.append(Waypoint(max(t_r - sum(w.duration for w in left), 0.1), LHOME, FINGER_OPEN, "done", 0.0))
    if os.environ.get("DB_DEMO") == "nosqueeze":  # a shortcut check: the right hand pulls without pressing the buttons
        right = [Waypoint(w.duration, w.pos, -(PK_W / 2 + BTN_PROUD - 0.5 * MM) if w.finger == B_SQ else w.finger,
                          w.label, w.yaw) for w in right]
    return {"right": right, "left": left}
