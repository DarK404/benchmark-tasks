"""On the station table: a Square D Homeline 4-space load center (an unpowered training panel, not wired to any supply)
is mounted upright on a plywood wall board, as it is installed on a wall, its door open and its dead front (cover) on.
Four Homeline HOM120 breakers show their handles through the dead front's slots. Three are ON; one has tripped: its
handle stands midway between ON and OFF.

The robot must reset the tripped breaker, as Square D says: push its handle all the way to OFF, then all the way back
to ON. A tripped breaker's handle pushed straight towards ON springs back to the middle; and the other breakers must
be left ON.

Run it:  inspect-robots run --embodiment yam_arena -E task=breaker_reset.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "One breaker in this unpowered training panel has tripped. Reset it." ...
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resetkit  # noqa: E402
from resetkit import Hinge, Spring, body, box  # noqa: E402
from yam_arena.analysis import box_signed_distance  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

resetkit.install()
SCENE = "station"
TOP = TABLE_TOP[SCENE]
MM = 0.001
GREY = (0.62, 0.63, 0.64)  # the enclosure and the dead front (painted steel)
BASE_GREY = (0.22, 0.23, 0.25)
BLACK = (0.08, 0.08, 0.09)  # breaker housings and handles
WHITE = (0.92, 0.92, 0.90)
PLY = (0.78, 0.66, 0.48)
RED = (0.75, 0.12, 0.10)  # the trip indicator window on each handle's base (drawn)
resetkit.METALLIC.update({GREY: 0.3})

# -- the load center (laid out in the PANEL frame, then stood upright: see breaker install's notes) --------------------
# PANEL frame: the enclosure lying on its back, x along it towards its top, y across it, z out of its back; origin
# (PX, PY, TOP) at the middle of its back. Stood upright on the wall board: panel x -> env z, panel z -> env -x.
PX, PY = 0.32, -0.06
ENC_L, ENC_W, ENC_D, WALL = 376.0 * MM, 226.0 * MM, 97.0 * MM, 1.6 * MM
ZP = 30.0 * MM  # the interior's breaker-mounting face, above the enclosure's back
PITCH = 25.4 * MM
ROW_X = {1: PX + PITCH / 2, 2: PX - PITCH / 2}
PANEL_ROT = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
BACK_X = 0.42
PANEL_ORIGIN = np.array([BACK_X, PY, TOP + 10.0 * MM + ENC_L / 2])
BOARD_T, BOARD_W, BOARD_TOP = 19.0 * MM, 0.60, 0.50

# the breakers: Square D HOM120, 25.4 W x 79.5 H mm; in a breaker's frame, y to its line end (the bus), z out of its
# mounting face; the raised face's top 66 mm out
BW, BL = 25.4 * MM, 79.5 * MM
B_BODY, B_FACE = 60.0 * MM, 6.0 * MM
SPACES = {1: (ROW_X[1], +1), 2: (ROW_X[1], -1), 3: (ROW_X[2], +1), 4: (ROW_X[2], -1)}
SEAT_Y = BL / 2 + 5.0 * MM
TRIPPED = int(os.environ.get("BR_TRIPPED", "4"))  # the tripped breaker's space
DF_Z0, DF_T = B_BODY + B_FACE, 1.5 * MM  # the dead front lies on the breakers' faces
SLOT_HALF = 11.0 * MM  # its slots: 22 mm wide across each column of handles

# -- the handle and its mechanism (a breaker's frame; the handle's pivot at PIV_Z) ----------------------------------
# The handle turns on a pin across the breaker (its x). phi > 0 throws its lever towards the load end: OFF; phi < 0
# towards the line end, at the middle of the panel: ON. It stops at +-25 degrees. Under the pivot an arm reaches 12 mm
# into the breaker; two parts of its tip press on two springs:
# - the toggle plunger (every breaker): pushing up on the arm's tip, it throws the handle to whichever end it is
#   nearer: ON or OFF, never in between (an over-centre toggle);
# - the trip cradle (the tripped breaker): unlatched, it presses a notch down on a pin at the other part of the tip
#   with a stiffer spring, centring the handle midway (tripped). Pushed towards ON, the pin rides the notch's shallow
#   side and the cradle springs it back. Pushed firmly to OFF, the pin rides the steep side and lifts the cradle 3 mm,
#   where a sprung pawl slips in under a ledge on it and holds it up (relatched): the notch is then clear of the pin
#   and the handle toggles OFF and ON like any other.
PIV_Z = 64.0 * MM
PHI_MAX = math.radians(25.0)
LEVER_TOP, LEVER_W, LEVER_T = 20.0 * MM, 8.0 * MM, 5.0 * MM  # the lever: 20 mm above the pivot, 16.5 mm through the dead front
ARM_L, ARM_T = 12.0 * MM, 3.0 * MM
ARM_A_X, ARM_B_X = -2.0 * MM, 2.0 * MM  # the arm's two parts (on the cradle; on the toggle plunger), 3 mm wide each
H_MASS = 0.004
PL_K, PL_F, PL_UP = 2000.0, 3.0, 2.0 * MM  # the toggle plunger: N/m, N at the top of its rise (7 N at mid), its rise
CR_K, CR_F, CR_UP = 1000.0, float(os.environ.get("BR_CRF", "5")), 5.0 * MM  # the trip cradle's spring (pressing down)
V_OFF_DEG, V_ON_DEG = 35.0, 15.0  # the notch's sides (from horizontal): steep on the OFF side, shallow on the ON side
LATCH_D = 3.0 * MM  # the cradle relatches this far up
PIN_X0, PIN_X1, PIN_H = -6.5 * MM, -3.5 * MM, 2.0 * MM  # the pin beside the arm's tip, under the notch
PAWL_K, PAWL_F, PAWL_TRAVEL = 1000.0, 3.0, 2.5 * MM
PART_MASS = 0.003


def seat_pose(space: int) -> tuple[np.ndarray, float]:
    x, side = SPACES[space]
    yaw = 0.0 if side < 0 else math.pi
    return np.array([x, PY + side * SEAT_Y, TOP + ZP]), yaw


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def to_env(p_panel) -> np.ndarray:
    return PANEL_ORIGIN + PANEL_ROT @ (np.asarray(p_panel, float) - np.array([PX, PY, TOP]))


def breaker_frame(space: int) -> tuple[np.ndarray, np.ndarray]:
    """The env pose (origin, rotation) of the breaker in ``space`` (origin: the middle of its mounting face)."""
    o, yaw = seat_pose(space)
    return to_env(o), PANEL_ROT @ _rz(yaw)


def b_to_panel(space: int, p_b) -> np.ndarray:
    o, yaw = seat_pose(space)
    return o + _rz(yaw) @ np.asarray(p_b, float)


def _breaker_housing() -> list:
    h, w, l2 = B_BODY, BW, BL / 2
    return [box((0.0, 0.0, h / 2), (w, BL, h), BLACK),
            box((0.0, -l2 + 9.0 * MM, h + B_FACE / 2), (w, 14.0 * MM, B_FACE), BLACK),
            box((0.0, l2 - 9.0 * MM, h + B_FACE / 2), (w, 14.0 * MM, B_FACE), BLACK),
            box((0.0, 13.0 * MM, h + B_FACE + 0.2 * MM), (14.0 * MM, 8.0 * MM, 0.4 * MM), WHITE)]


def _panel_shapes() -> list:
    L, W, D, t = ENC_L, ENC_W, ENC_D, WALL
    out = [box((0.0, 0.0, t / 2), (L, W, t), GREY),
           box((L / 2 - t / 2, 0.0, D / 2), (t, W, D), GREY), box((-L / 2 + t / 2, 0.0, D / 2), (t, W, D), GREY),
           box((0.0, W / 2 - t / 2, D / 2), (L, t, D), GREY), box((0.0, -W / 2 + t / 2, D / 2), (L, t, D), GREY),
           box((-10.0 * MM, 0.0, (t + ZP) / 2), (150.0 * MM, 200.0 * MM, ZP - t), BASE_GREY)]
    x0 = TOP - PANEL_ORIGIN[2]
    out.append(box(((x0 + x0 + BOARD_TOP) / 2, 0.0, -BOARD_T / 2), (BOARD_TOP, BOARD_W, BOARD_T), PLY))
    for space in (1, 2, 3, 4):
        o, yaw = seat_pose(space)
        off = o - np.array([PX, PY, TOP])
        for s in _breaker_housing():
            c = off + _rz(yaw) @ np.asarray(s.centre)
            out.append(box(tuple(c), s.size, s.color, yaw=s.yaw + yaw))
    # the dead front: a steel plate on the breakers' faces, with a slot down each column for the handles
    zc = ZP + DF_Z0 + DF_T / 2
    inner_l, inner_w = L - 2 * t, W - 2 * t
    ys = sorted([SEAT_Y - SLOT_HALF, SEAT_Y + SLOT_HALF, -SEAT_Y - SLOT_HALF, -SEAT_Y + SLOT_HALF])
    edges = [-inner_w / 2, ys[0], ys[1], ys[2], ys[3], inner_w / 2]
    for k, (y0, y1) in enumerate(zip(edges[:-1], edges[1:])):
        if k in (1, 3):  # a slot column: plate above and below the handles only
            for xa, xb in ((-inner_l / 2, -PITCH - 4 * MM), (PITCH + 4 * MM, inner_l / 2)):
                out.append(box(((xa + xb) / 2, (y0 + y1) / 2, zc), (xb - xa, y1 - y0, DF_T), GREY))
        else:
            out.append(box((0.0, (y0 + y1) / 2, zc), (inner_l, y1 - y0, DF_T), GREY))
    return out


PANEL = body("load_center", pos=tuple(PANEL_ORIGIN), shapes=_panel_shapes(), mass=4.0, friction=0.4,
             contact_offset=0.0005, quat=resetkit.matrix_to_quat_rel(PANEL_ROT))
resetkit.kinematic(PANEL.name)  # fixed on its wall board (a kinematic rigid body, so the breakers' parts can be jointed to it)


def _plate_yz(p0, p1, t, colour, x, w, outward) -> object:
    """A thin box whose ``outward`` face lies on the line p0-p1 in the y-z plane, ``w`` long in x."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    if np.dot(n, outward) < 0:
        n = -n
    c = (p0 + p1) / 2 - n * t / 2
    return box((x, c[0], c[1]), (w, float(np.linalg.norm(d)), t), colour, tilt_x=math.atan2(d[1], d[0]))


def _handle_shapes() -> list:
    """The handle in its frame (at the pivot, the breaker's axes, upright): the lever above, the arm's two parts below."""
    return [box((0.0, 0.0, (2.0 * MM + LEVER_TOP) / 2), (LEVER_W, LEVER_T, LEVER_TOP - 2.0 * MM), BLACK),
            box((0.0, 0.0, 0.0), (LEVER_W, 6.0 * MM, 4.0 * MM), BLACK),  # its round base (in the slot)
            box((ARM_A_X, 0.0, -ARM_L / 2), (3.0 * MM, ARM_T, ARM_L), BLACK),
            box((ARM_B_X, 0.0, -ARM_L / 2), (3.0 * MM, ARM_T, ARM_L), BLACK),
            box(((PIN_X0 + PIN_X1) / 2, 0.0, -ARM_L), (PIN_X1 - PIN_X0, PIN_H, PIN_H), BLACK)]  # the pin


def phi_start(space: int) -> float:
    return 0.0 if space == TRIPPED else -PHI_MAX  # tripped: midway; the others ON


def _body_rot(space: int, phi: float = 0.0) -> np.ndarray:
    _, yaw = seat_pose(space)
    return PANEL_ROT @ _rz(yaw) @ _rx(phi)


HANDLES, PLUNGERS = {}, {}
for _s in (1, 2, 3, 4):
    _piv = to_env(b_to_panel(_s, (0.0, 0.0, PIV_Z)))
    HANDLES[_s] = body(f"handle_{_s}", pos=tuple(_piv), shapes=_handle_shapes(), mass=H_MASS, friction=0.5,
                       contact_offset=0.0003, solver_iterations=32,
                       quat=resetkit.matrix_to_quat_rel(_body_rot(_s, phi_start(_s))))
    _top = PIV_Z - ARM_L  # the plunger's top under the arm's tip (at mid)
    PLUNGERS[_s] = body(f"toggle_{_s}", pos=tuple(to_env(b_to_panel(_s, (ARM_B_X, 0.0, _top - 4.0 * MM)))),
                        shapes=[box((0.0, 0.0, 0.0), (3.0 * MM, 6.0 * MM, 8.0 * MM), BLACK)], mass=PART_MASS,
                        friction=0.2, contact_offset=0.0003, solver_iterations=32,
                        quat=resetkit.matrix_to_quat_rel(_body_rot(_s)))


def _cradle_shapes() -> list:
    """The trip cradle in its frame (origin at the notch's apex, the breaker's axes): the notch's two sides facing down,
    its block above them, and the ledge the pawl slips under."""
    off_end = (7.0 * MM, -7.0 * MM * math.tan(math.radians(V_OFF_DEG)))
    on_end = (-12.0 * MM, -12.0 * MM * math.tan(math.radians(V_ON_DEG)))
    return [_plate_yz((0.0, 0.0), off_end, 3.0 * MM, BLACK, 0.0, 3.0 * MM, (-1.0, -1.0)),
            _plate_yz((0.0, 0.0), on_end, 3.0 * MM, BLACK, 0.0, 3.0 * MM, (1.0, -1.0)),
            box((0.0, 0.5 * MM, 4.5 * MM), (3.0 * MM, 17.0 * MM, 7.0 * MM), BLACK),
            box((0.0, 10.0 * MM, 8.0 * MM), (3.0 * MM, 2.0 * MM, 4.0 * MM), BLACK)]  # the ledge: underside at +6 mm


_X_CR = (PIN_X0 + PIN_X1) / 2
_V0 = (_X_CR, 0.0, PIV_Z - ARM_L + PIN_H / 2)  # the notch's apex (on the pin's top), in the tripped breaker's frame
CRADLE = body("trip_cradle", pos=tuple(to_env(b_to_panel(TRIPPED, _V0))), shapes=_cradle_shapes(), mass=PART_MASS,
              friction=0.2, contact_offset=0.0003, solver_iterations=32,
              quat=resetkit.matrix_to_quat_rel(_body_rot(TRIPPED)))
# the pawl: its front face against the ledge's outer face (y 11 mm), its top LATCH_D under the ledge's underside
_PAWL0 = np.array([_X_CR, 11.0 * MM + 1.5 * MM, _V0[2] + 6.0 * MM + LATCH_D - 2.5 * MM])
PAWL = body("trip_pawl", pos=tuple(to_env(b_to_panel(TRIPPED, _PAWL0))),
            shapes=[box((0.0, 0.0, 0.0), (3.0 * MM, 3.0 * MM, 5.0 * MM), BLACK)], mass=PART_MASS, friction=0.2,
            contact_offset=0.0003, solver_iterations=32, quat=resetkit.matrix_to_quat_rel(_body_rot(TRIPPED)))

OBJECTS = [PANEL, *HANDLES.values(), *PLUNGERS.values(), CRADLE, PAWL]
resetkit.register(*OBJECTS)
for _s, _h in HANDLES.items():
    _, _yaw = seat_pose(_s)
    sgn = 1.0 if _yaw == 0.0 else -1.0  # the breaker's x along the panel's +x or -x
    lo, hi = sorted((sgn * (-PHI_MAX - phi_start(_s)), sgn * (PHI_MAX - phi_start(_s))))
    resetkit.add_hinge(Hinge(PANEL.name, _h.name, anchor=tuple(b_to_panel(_s, (0.0, 0.0, PIV_Z)) - [PX, PY, TOP]),
                             axis=(1.0, 0.0, 0.0), damping=0.0005, lower=lo, upper=hi))
    resetkit.add_spring(Spring(PANEL.name, PLUNGERS[_s].name, axis="Z", stiffness=PL_K, damping=1.0,
                               target=PL_UP + PL_F / PL_K, lower=-4.0 * MM, upper=PL_UP))
    for _o in HANDLES.values():
        if _o is not _h:
            resetkit.no_collide(_h.name, _o.name)
resetkit.add_spring(Spring(PANEL.name, CRADLE.name, axis="Z", stiffness=CR_K, damping=2.0, target=-CR_F / CR_K,
                           lower=0.0, upper=CR_UP))
resetkit.add_spring(Spring(PANEL.name, PAWL.name, axis="Y", stiffness=PAWL_K, damping=0.5,
                           target=-(PAWL_TRAVEL + PAWL_F / PAWL_K), lower=-PAWL_TRAVEL, upper=0.0,
                           frame_yaw=seat_pose(TRIPPED)[1]))

# -- scoring -------------------------------------------------------------------------------------------------------
# 1.0   the tripped breaker reset and ON (relatched: the pawl in over the cradle's ledge; the handle at ON, within 7
#       degrees of its stop) and every other breaker still ON, never switched off; no gripper within 1 cm of any handle;
#       all still; for 1 s
# 0.5   relatched but its handle left OFF (or not yet still, or a hand on it)
# 0     anything else; and for the rest of the run once any other breaker has been switched off (its handle thrown
#       past the middle), or once the tripped one's handle has been forced past its stop
SCORE_OVER = "end"
RUN_LIMIT_STEPS = 1200
STILL_STEPS = 20
CLEAR_DIST = 10.0 * MM
ON_TOL = math.radians(7.0)
_STATE: dict = {}


@resetkit.on_reset
def _reset_state() -> None:
    _STATE.clear()
    _STATE.update(still=0, switched_off=False)


_reset_state()


def handle_phi(objects, space: int) -> float:
    """The handle's throw (rad) in its breaker's frame: + towards OFF, - towards ON."""
    _, r = breaker_frame(space)
    rel = r.T @ objects[HANDLES[space].name].rotation()
    return math.atan2(rel[2, 1], rel[1, 1])


def pawl_in(objects) -> float:
    """How far (m) the pawl has moved in over the cradle's ledge from where it starts."""
    _, r = breaker_frame(TRIPPED)
    start = to_env(b_to_panel(TRIPPED, _PAWL0))
    return float(-(r.T @ (np.asarray(objects[PAWL.name].pos) - start))[1])


def relatched(objects) -> bool:
    return pawl_in(objects) >= 1.2 * MM


def is_on(objects, space: int) -> bool:
    return handle_phi(objects, space) <= -PHI_MAX + ON_TOL


def hands_clear(objects) -> bool:
    pts = resetkit.finger_points()
    if not len(pts):
        return True
    for s, h in HANDLES.items():
        st = objects[h.name]
        if box_signed_distance(pts, st.to_world((0.0, 0.0, LEVER_TOP / 2)), st.rotation(),
                               np.array([LEVER_W, LEVER_T, LEVER_TOP])).min() < CLEAR_DIST:
            return False
    return True


def _watch(objects) -> None:
    for s in HANDLES:
        if s != TRIPPED and handle_phi(objects, s) > 0.0:
            _STATE["switched_off"] = True


def _update(objects) -> tuple[float, bool]:
    _watch(objects)
    if _STATE["switched_off"]:
        return 0.0, False
    if relatched(objects):
        others_on = all(is_on(objects, s) for s in HANDLES if s != TRIPPED)
        # still: no handle's throw changed by more than 0.2 degrees since the last step (joint-held, the handles'
        # read-back velocities are not physical, so their stillness is judged from their angles)
        phis = {s: handle_phi(objects, s) for s in HANDLES}
        last = _STATE.get("phis", phis)
        still = all(abs(phis[s] - last[s]) < math.radians(0.2) for s in HANDLES)
        _STATE["phis"] = phis
        if is_on(objects, TRIPPED) and others_on and still and hands_clear(objects):
            return 1.0, True
        return 0.5, False
    return 0.0, False


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
# Open loop. The right hand, its closed fingertips towards the panel, comes beside the tripped handle on its ON side
# and pushes its lever firmly all the way to OFF (aimed well past, the soft arm); backs off; comes round to its OFF
# side and pushes it all the way to ON; backs off and goes home.
TIP = 13.0 * MM
RHOME, LHOME = (0.14, -0.22, TOP + 0.17), (0.14, 0.22, TOP + 0.17)
PUSH_Z = PIV_Z + float(os.environ.get("BR_PZ", "2")) * MM  # the fingertips this far out from the mounting face
SIDE = float(os.environ.get("BR_SIDE", "13")) * MM  # the fingertips' middle this far beside the upright lever
PAST = float(os.environ.get("BR_PAST", "60")) * MM  # each push aimed this far past the lever's far stop
CAL = np.array([float(v) for v in os.environ.get("BR_CAL", "0,0,0").split(",")]) * MM
DEMO = os.environ.get("BR_DEMO", "")  # "on": push it straight to ON without resetting it (the shortcut check)


def _at(y_b: float, z_b: float = PUSH_Z) -> tuple[float, float, float]:
    """The TCP with the fingertips at (y, z) in the tripped breaker's frame (x on the lever's middle)."""
    o, r = breaker_frame(TRIPPED)
    p = o + r @ np.array([0.0, y_b, z_b + TIP]) + CAL
    return tuple(float(v) for v in p)


def demo():
    g = resetkit.tilted(0.0, 0.0, 0.0)  # fingertips straight at the wall, the fingers side by side across
    off_reach = -(LEVER_TOP * math.sin(PHI_MAX)) - PAST  # the lever's top at OFF is at y = -14 sin 25
    on_reach = LEVER_TOP * math.sin(PHI_MAX) + PAST
    right = [Waypoint(1.0, RHOME, FINGER_OPEN, "wait", 0.0)]
    pushes = [("on", "push it to ON")] if DEMO == "on" else [("off", "push it to OFF"), ("on", "push it to ON")]
    for side, label in pushes:
        start = SIDE if side == "off" else -SIDE - LEVER_TOP * math.sin(PHI_MAX)
        goal = off_reach if side == "off" else on_reach
        right += [Waypoint(2.5, _at(start, PUSH_Z + 0.04), FINGER_CLOSED, f"beside it ({side})", g),
                  Waypoint(1.0, _at(start), FINGER_CLOSED, "in to the lever", g),
                  Waypoint(1.5, _at(goal), FINGER_CLOSED, label, g),
                  Waypoint(0.8, _at(goal), FINGER_CLOSED, "hold", g),
                  Waypoint(1.0, _at(goal, PUSH_Z + 0.04), FINGER_CLOSED, "back off", g)]
    right += [Waypoint(2.5, RHOME, FINGER_OPEN, "home", 0.0)]
    left = [Waypoint(sum(w.duration for w in right), LHOME, FINGER_OPEN, "wait", 0.0)]
    for plan in (left, right):
        plan.append(Waypoint(2.0, plan[-1].pos, plan[-1].finger, "done", plan[-1].yaw))
    return {"right": right, "left": left}
