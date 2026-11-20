"""On an ESD mat: a flat, empty static-shielding bag (SCS 10035, 3 x 5 in, open top) lies with its closed mouth towards
the robot, a Raspberry Pi Zero 2 W board lies beside it, and a conductive tote stands on the robot's left.

The robot must get the board into the bag (opening the bag's mouth itself) and put the filled bag into the tote.

Run it:  inspect-robots run --embodiment yam_arena -E task=bag_a_circuit_board.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "Put the circuit board into the antistatic bag, then put the bag in the tote." ...
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bagkit  # noqa: E402
from bagkit import Hinge, body, box  # noqa: E402
from yam_arena import station  # noqa: E402
from yam_arena.analysis import box_signed_distance  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

bagkit.install()
SCENE = "station"
TOP = TABLE_TOP[SCENE]
MM = 0.001
SILVER = (0.62, 0.64, 0.66)  # the bag's metallised film
SEAM = (0.50, 0.52, 0.55)
MAT_C = (0.20, 0.36, 0.42)  # a blue-grey dissipative mat
PCB = (0.08, 0.40, 0.18)
CHIP = (0.12, 0.12, 0.13)
STEEL = (0.75, 0.75, 0.72)
TOTE_C = (0.10, 0.10, 0.11)
bagkit.METALLIC.update({SILVER: 0.35, SEAM: 0.4, STEEL: 0.3})


def _env(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


# -- the mat (fixed) ------------------------------------------------------------------------------------------------
MAT_T = 3.0 * MM
MAT_POS = (0.27, -0.04)
MAT_SIZE = (0.30, 0.36)
MAT_TOP = TOP + MAT_T
MAT = body("esd_mat", pos=(MAT_POS[0], MAT_POS[1], TOP + MAT_T / 2),
           shapes=[box((0, 0, 0), (MAT_SIZE[0], MAT_SIZE[1], MAT_T), MAT_C)], friction=0.6)

# -- the bag: SCS 10035, 76 x 127 mm outside, lying along x, its mouth towards the robot (-x) -------------------------
# Its cross-section (y-z) is a closed loop of five strips, each as long as the bag, hinged along the bag's length:
# the bottom film's middle (BC), the bottom film's two edges (BE_L/BE_R, which curl up into walls when the seams are
# pushed in) and the top film's two halves (TL/TR, meeting at the ridge). The side seams are the hinges BE-T.
BAG_L = 127.0 * MM
BAG_W = 76.0 * MM
FILM_T = 2.0 * MM  # WHY 2 mm (2026-11-20): thinner strips tunnel under fingertips (lessons: 1 mm plates tunnel)
HB = 25.0 * MM  # half the bottom film's flat middle
EW = BAG_W / 2 - HB  # each bottom edge strip (13 mm)
HT = BAG_W / 2  # each half of the top film (38 mm)
A0 = math.radians(_env("BG_A0", 3.0))  # the edges' tilt as laid down (a flat bag is never perfectly flat)
BX, BY = 0.30, 0.0  # the bag's centre
MOUTH_X = BX - BAG_L / 2
LIP_H = 4.0 * MM  # the welded bottom seam, a low ridge inside the closed end
M_FILM = {"bc": 0.0015, "be": 0.0008, "t": 0.0015}  # kg; WHY: about 6 g in all, 4x a real bag (lighter is unstable)
Z0 = MAT_TOP + FILM_T / 2  # the bottom film's mid-surface


def _loop(a: float):
    """Cross-section points (y, z above Z0) of the loop with the edges tilted ``a``: (P0, P1, S_L, S_R, ridge)."""
    ys = HB + EW * math.cos(a)
    zs = EW * math.sin(a)
    zr = zs + math.sqrt(max(HT ** 2 - ys ** 2, 0.0))
    return (HB, 0.0), (-HB, 0.0), (ys, zs), (-ys, zs), (0.0, zr)


def _qx(t: float):
    return (math.sin(t / 2), 0.0, 0.0, math.cos(t / 2))


def _strip(name: str, p, q, mass: float, extra=(), lift: float = 0.0):
    """A film strip from cross-section point p to q (y, z), as long as the bag; its box sits ``lift`` (m) above that
    line (upwards in the world when laid flat)."""
    (y0, z0), (y1, z1) = p, q
    ang = math.atan2(z1 - z0, y1 - y0)
    w = math.hypot(y1 - y0, z1 - z0)
    up = 1.0 if math.cos(ang) >= 0 else -1.0
    shapes = [box((0, 0, up * lift), (BAG_L, w, FILM_T), SILVER)] + list(extra)
    return body(name, pos=(BX, BY + (y0 + y1) / 2, Z0 + (z0 + z1) / 2), shapes=shapes, mass=mass, friction=0.4,
                quat=_qx(ang), solver_iterations=64, contact_offset=0.0008), ang


_P0, _P1, _SL, _SR, _R = _loop(A0)
BC, _ = _strip("bag_bottom", _P1, _P0, M_FILM["bc"],  # -y to +y: unturned
               extra=[box((BAG_L / 2 - 2.5 * MM, 0, FILM_T / 2 + LIP_H / 2), (4.0 * MM, 2 * HB, LIP_H), SEAM)])
BE_L, _ = _strip("bag_edge_l", _P0, _SL, M_FILM["be"])
BE_R, _ = _strip("bag_edge_r", _P1, _SR, M_FILM["be"])
# WHY lift (2026-11-20 probe): the loop's lines are the films' mid-surfaces, so a flat loop put the top film inside
# the bottom one and the contacts shook the bag across the mat; the top film's box sits one film thickness higher
T_L, _ = _strip("bag_top_l", _SL, _R, M_FILM["t"], lift=FILM_T)
T_R, _ = _strip("bag_top_r", _SR, _R, M_FILM["t"], lift=FILM_T)
BAG = [BC, BE_L, BE_R, T_L, T_R]

FILM_K = _env("BG_K", 2.0e-5)  # N m/rad per hinge: a 2.8 mil film's bending, as a weak spring (see fidelity)
FILM_D = _env("BG_D", 1.0e-5)


def _local(spec, world):
    r = bagkit.quat_to_matrix(spec.quat)
    return tuple((r.T @ (np.asarray(world, float) - np.asarray(spec.pos, float))).tolist())


def _w(p):
    return (BX, BY + p[0], Z0 + p[1])


WALL_MAX = math.radians(_env("BG_WALL", 95.0))  # WHY (2026-11-20 probe): unlimited, the squeezed edges folded flat
# inwards over the bottom film and narrowed the opening to 24 mm; a real bag's edge film stands up against the fingers
# Hinge angles are about the bottom film's +x, from the laid-down tilt: the left edge (+y) rises with a positive turn,
# the right edge with a negative one. Each may lie flat (down to -A0) or stand up to WALL_MAX.
_LIM = {BE_L.name: (-A0, WALL_MAX - A0), BE_R.name: (A0 - WALL_MAX, A0)}
for _par, _ch, _pt in ((BC, BE_L, _P0), (BC, BE_R, _P1), (BE_L, T_L, _SL), (BE_R, T_R, _SR), (T_L, T_R, _R)):
    _lo, _hi = _LIM.get(_ch.name, (None, None)) if _par is BC else (None, None)
    bagkit.add_hinge(Hinge(_par.name, _ch.name, anchor=_local(_par, _w(_pt)), axis=(1.0, 0.0, 0.0),
                           stiffness=FILM_K, damping=FILM_D, lower=_lo, upper=_hi))
for _s in BAG:
    bagkit.damped(_s.name, 0.5, 0.5)

# -- the board: Raspberry Pi Zero 2 W, 65 x 30 mm, 10 g, components up ------------------------------------------------
PCB_L, PCB_W, PCB_T = 65.0 * MM, 30.0 * MM, 1.2 * MM
BRD_X, BRD_Y = 0.20, -0.14
_bshapes = [box((0, 0, 0), (PCB_L, PCB_W, PCB_T), PCB),
            box((-2 * MM, 0, PCB_T / 2 + 0.6 * MM), (12 * MM, 12 * MM, 1.2 * MM), CHIP),  # the SoC/RAM package
            box((-PCB_L / 2 + 9 * MM, -PCB_W / 2 + 4 * MM, PCB_T / 2 + 1.6 * MM), (11 * MM, 7.5 * MM, 3.2 * MM), STEEL),
            box((PCB_L / 2 - 14 * MM, -PCB_W / 2 + 3 * MM, PCB_T / 2 + 1.3 * MM), (8 * MM, 5.6 * MM, 2.6 * MM), STEEL),
            box((PCB_L / 2 - 26 * MM, -PCB_W / 2 + 3 * MM, PCB_T / 2 + 1.3 * MM), (8 * MM, 5.6 * MM, 2.6 * MM), STEEL),
            box((-PCB_L / 2 + 7 * MM, 3 * MM, PCB_T / 2 + 0.7 * MM), (12 * MM, 12 * MM, 1.4 * MM), STEEL)]  # SD slot
BOARD = body("board", pos=(BRD_X, BRD_Y, MAT_TOP + PCB_T / 2), shapes=_bshapes, mass=0.010, friction=0.4,
             solver_iterations=32, contact_offset=0.0008)

# -- the tote: Quantum DG91035-CON conductive grid container, 276 x 210 x 89 mm outside, 233 x 167 mm inside ----------
TOTE_X, TOTE_Y = 0.31, 0.28
TOTE_IN = (233.0 * MM, 167.0 * MM)
TOTE_H = 89.0 * MM
TOTE_FLOOR = 13.0 * MM  # floor top above the table (inside depth 76 mm)
WALL = 5.0 * MM
_tshapes = [box((0, 0, TOTE_FLOOR / 2), (TOTE_IN[0] + 2 * WALL, TOTE_IN[1] + 2 * WALL, TOTE_FLOOR), TOTE_C)]
for _sx in (-1, 1):
    _tshapes.append(box((_sx * (TOTE_IN[0] / 2 + WALL / 2), 0, TOTE_H / 2), (WALL, TOTE_IN[1] + 2 * WALL, TOTE_H), TOTE_C))
    _tshapes.append(box((0, _sx * (TOTE_IN[1] / 2 + WALL / 2), TOTE_H / 2), (TOTE_IN[0], WALL, TOTE_H), TOTE_C))
TOTE = body("tote", pos=(TOTE_X, TOTE_Y, TOP), shapes=_tshapes, friction=0.5)

OBJECTS = [MAT, *BAG, BOARD, TOTE]
bagkit.register(*OBJECTS)

# -- scoring ----------------------------------------------------------------------------------------------------
# 1.0   the board wholly in the bag (between its films, inside the side seams, its trailing edge at least IN_MIN past
#       the mouth) AND the bag wholly inside the tote (every strip's corners between its inner walls and below its
#       rim), lying on its floor (the bottom film within FLOOR_TOL of it, tilted under FLAT_DEG), no gripper within
#       CLEAR_DIST of the bag or the board, the bag and the board still; held STILL_STEPS (1 s)
# 0.5   the board wholly in the bag, the bag anywhere else (or held, or moving)
# 0.25  the board's centre in the bag, between its films (at least half the board in)
# 0     anything else; and from then on if the board or the bag ever falls off the table (latched)
SCORE_OVER = "end"
RUN_LIMIT_STEPS = 2400
STILL_STEPS = 20  # 1 s at 20 Hz
IN_MIN = 5.0 * MM
SEAM_W = 5.0 * MM  # each side seam's weld; the cavity is 76 - 2 x 5 = 66 mm wide
CAVITY_HALF = BAG_W / 2 - SEAM_W
INNER_END = BAG_L / 2 - 5.0 * MM  # the bottom seam's inner edge (bag frame x)
FLOOR_TOL = 3.0 * MM
FLAT_DEG = 10.0
CLEAR_DIST = 20.0 * MM
STILL_MM = 0.5 * MM  # per control step (1 cm/s): judged by pose change (joint-held bodies report false speeds)
FALL_Z = TOP - 0.02
_STATE: dict = {}


@bagkit.on_reset
def _reset_state() -> None:
    _STATE.clear()
    _STATE.update(still=0, n=0, fell=False, prev=None, calm=0)


_reset_state()


def _corners(spec, st) -> np.ndarray:
    """World corners of a body's bounding box (its shapes' extent in its own frame)."""
    lo, hi = spec.bounds_local()
    pts = [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    return np.array([st.to_world(p) for p in pts])


def _top_bottom_at(objects, y: float) -> float:
    """Height (bag frame z) of the top film's underside above bag-frame ``y``: from the two top strips' lines."""
    bc = objects[BC.name]
    best = None
    for spec in (T_L, T_R):
        st = objects[spec.name]
        w = spec.shapes[0].size[1]
        a = bc.to_local(st.to_world((0.0, -w / 2, 0.0)))
        b = bc.to_local(st.to_world((0.0, w / 2, 0.0)))
        if min(a[1], b[1]) - 1e-4 <= y <= max(a[1], b[1]) + 1e-4:
            f = 0.0 if abs(b[1] - a[1]) < 1e-9 else (y - a[1]) / (b[1] - a[1])
            z = a[2] + f * (b[2] - a[2]) + FILM_T / 2  # the strip's box sits FILM_T over its line
            best = z if best is None else max(best, z)
    return -1.0 if best is None else best


def board_in_bag(objects) -> tuple[bool, bool]:
    """(wholly in, centre in): the board between the films, inside the side seams, past the mouth."""
    bc, bd = objects[BC.name], objects[BOARD.name]
    c = bc.to_local(bd.pos)
    lo, hi = BOARD.bounds_local()
    pcb = np.array([bc.to_local(bd.to_world((x, y, 0.0))) for x in (lo[0], hi[0]) for y in (lo[1], hi[1])])
    floor = FILM_T / 2 - 1.0 * MM  # the board's centre over the bottom film (1 mm allowance for contact softness)

    def between(p) -> bool:
        return floor < p[2] < _top_bottom_at(objects, float(p[1]))

    centre_in = bool(-BAG_L / 2 < c[0] < INNER_END and abs(c[1]) < CAVITY_HALF and between(c))
    whole = bool(centre_in and np.all(pcb[:, 0] > -BAG_L / 2 + IN_MIN) and np.all(pcb[:, 0] < INNER_END + 1 * MM)
                 and np.all(np.abs(pcb[:, 1]) < CAVITY_HALF))
    return whole, centre_in


def bag_in_tote(objects) -> bool:
    """Every strip's corners between the tote's inner walls and below its rim; the bottom film on its floor, flat."""
    for spec in BAG:
        pts = _corners(spec, objects[spec.name])
        if np.any(np.abs(pts[:, 0] - TOTE_X) > TOTE_IN[0] / 2) or np.any(np.abs(pts[:, 1] - TOTE_Y) > TOTE_IN[1] / 2):
            return False
        if np.any(pts[:, 2] > TOP + TOTE_H):
            return False
    bc = objects[BC.name]
    low = _corners(BC, bc)[:, 2].min()
    return bool(low - (TOP + TOTE_FLOOR) < FLOOR_TOL and bc.tilt_deg() < FLAT_DEG)


def hands_clear(objects) -> bool:
    pts = bagkit.finger_points()
    if not len(pts):
        return True
    for spec in (*BAG, BOARD):
        st = objects[spec.name]
        lo, hi = spec.bounds_local()
        if box_signed_distance(pts, st.to_world((lo + hi) / 2), st.rotation(), hi - lo).min() < CLEAR_DIST:
            return False
    return True


def _track(objects) -> bool:
    """Still: the bag's bottom film and the board each moved under STILL_MM since the last control step."""
    now = np.concatenate([np.asarray(objects[s.name].pos, float) for s in (BC, BOARD)])
    prev, _STATE["prev"] = _STATE["prev"], now
    return prev is not None and bool(np.all(np.abs(now - prev).reshape(2, 3).max(axis=1) < STILL_MM))


def _update(objects) -> tuple[float, bool]:
    if any(np.asarray(objects[s.name].pos)[2] < FALL_Z for s in (BC, BOARD)):
        _STATE["fell"] = True
    if _STATE["fell"]:
        return 0.0, False
    whole, centre = board_in_bag(objects)
    if whole:
        if bag_in_tote(objects) and hands_clear(objects) and _STATE["calm"]:
            return 1.0, True
        return 0.5, False
    return (0.25 if centre else 0.0), False


def _step(objects) -> None:
    """Once per control step: stillness, then the full-score count."""
    _STATE["calm"] = _track(objects)
    ok = _update(objects)[1]
    _STATE["still"] = _STATE["still"] + 1 if ok else 0
    _log(objects)


def _log(objects) -> None:
    _STATE["n"] += 1
    if _STATE["n"] % 10 == 0 and os.environ.get("BG_LOG"):
        tl, tr, bc = (objects[s.name] for s in (T_L, T_R, BC))
        b = objects[BOARD.name]
        ridge = (np.asarray(tl.pos)[2] + np.asarray(tr.pos)[2]) / 2 - Z0
        print(f"[bag] n={_STATE['n']} ridge_mid={ridge*1000:.1f}mm bc={np.round(np.asarray(bc.pos)-[BX,BY,Z0],4)} "
              f"rel={np.round(bc.to_local(b.pos)*1000,1)} in={board_in_bag(objects)} tote={bag_in_tote(objects)} "
              f"clear={hands_clear(objects)} still={_STATE['still']} tier={_update(objects)[0]}", flush=True)


def success(objects) -> bool:
    """True only once the full-score state has held for STILL_STEPS control steps in a row (1 s), as score() is 1.0
    only then; yam_arena ends the episode after success() has then held for its own SUCCESS_HOLD_S (1 s)."""
    _step(objects)
    return _STATE["still"] >= STILL_STEPS


def score(objects) -> float:
    tier = _update(objects)[0]
    if tier == 1.0 and _STATE["still"] < STILL_STEPS:
        return 0.5
    return tier


# -- scripted demo (probe) -------------------------------------------------------------------------------------------
TIP = 13.0 * MM
RHOME, LHOME = (0.14, -0.22, TOP + 0.17), (0.14, 0.22, TOP + 0.17)
SQ_X = MOUTH_X + _env("BG_SQX", 45.0) * MM  # where the left hand squeezes the seams
SQ_HALF = _env("BG_SQ", 24.0) * MM
SQ_Y = BY  # fingers commanded to this half-gap
SQ_Z = MAT_TOP + _env("BG_SQZ", 2.0) * MM + TIP  # fingertips this high above the mat


GRIP_P = math.radians(_env("BG_GP", 45.0))  # the right hand's grip, tipped this far down towards +x
GRIP_BACK = 9.0 * MM  # fingertips' centre this far in front of the board's trailing edge
BRD_HALF = PCB_W / 2


def _tcp(tip, pitch=GRIP_P):
    """TCP for fingertips (pad centre) at ``tip`` with the approach axis heading +x, ``pitch`` below horizontal."""
    d = np.array([math.cos(pitch), 0.0, -math.sin(pitch)])
    return tuple((np.asarray(tip, float) - TIP * d).tolist())


CARRY_X = BX - _env("BG_CX", 25.0) * MM  # the left hand takes the filled bag across the board's middle
CARRY_HALF = _env("BG_CH", 13.0) * MM  # through the films, a little inside the board's 15 mm half-width
CARRY_Z = TOP + 0.15
DROP_Z = TOP + TOTE_FLOOR + _env("BG_DZ", 5.0) * MM + TIP  # fingertips this high over the tote's floor


def demo():
    g = bagkit.tilted(0.0, GRIP_P, 0.0)
    tail = BRD_X - PCB_L / 2
    zb = MAT_TOP + PCB_T / 2  # board centre on the mat
    zin = Z0 + FILM_T / 2 + _env("BG_ZIN", 14.0) * MM + PCB_T / 2  # board centre while it slides over the bottom film
    hold = -(BRD_HALF - _env("BG_BSQ", 3.0) * MM)
    t_open = 1.0 + 2.0 + 1.5 + 2.5
    gx0 = tail + GRIP_BACK
    gx_in = MOUTH_X - PCB_L - 5 * MM + GRIP_BACK  # board's front edge 5 mm short of the mouth
    gx_mouth = MOUTH_X - _env("BG_STOP", 4.0) * MM  # the fingers stop just short of the mouth (the roof is low there)
    push_end = MOUTH_X + _env("BG_DEEP", 30.0) * MM  # the closed fingertips push the trailing edge this far in
    pp = math.radians(_env("BG_PP", 20.0))
    gp = bagkit.tilted(0.0, pp, 0.0)
    zt = zin + _env("BG_ZT", -12.0) * MM  # closed fingertips at the board's edge height
    right = [Waypoint(t_open - 4.0, RHOME, FINGER_OPEN, "wait", 0.0),
             Waypoint(2.0, _tcp((gx0, BRD_Y, zb + 0.04)), FINGER_OPEN, "over the board's end", g),
             Waypoint(1.5, _tcp((gx0, BRD_Y, zb)), FINGER_OPEN, "round it", g),
             Waypoint(1.5, _tcp((gx0, BRD_Y, zb)), hold, "grip it", g),
             Waypoint(1.0, _tcp((gx0, BRD_Y, zb + 0.02)), hold, "lift it", g),
             Waypoint(2.0, _tcp((gx_in, BY, zin + 0.01)), hold, "to the mouth", g),
             Waypoint(1.0, _tcp((gx_in, BY, zin)), hold, "down to it", g),
             Waypoint(2.0, _tcp((gx_mouth, BY, zin)), hold, "slide it in", g),
             Waypoint(1.2, _tcp((gx_mouth, BY, zin)), FINGER_OPEN, "let go", g),
             Waypoint(1.0, _tcp((gx_mouth - 0.03, BY, zin + 0.02)), FINGER_OPEN, "back off", g),
             Waypoint(1.8, _tcp((gx_mouth - 0.03, BY, zt + 0.02), pp), 0.0, "close the fingers", gp),
             Waypoint(1.0, _tcp((MOUTH_X - 0.02, BY, zt), pp), 0.0, "behind its edge", gp),
             Waypoint(2.0, _tcp((push_end, BY, zt), pp), 0.0, "push it home", gp),
             Waypoint(1.0, _tcp((push_end - 0.03, BY, zt + 0.015), pp), 0.0, "back out", gp),
             Waypoint(2.0, RHOME, FINGER_OPEN, "home", 0.0)]
    t_r = sum(w.duration for w in right)
    left = [Waypoint(1.0, LHOME, FINGER_OPEN, "wait", 0.0),
            Waypoint(2.0, (SQ_X, SQ_Y, SQ_Z + 0.05), FINGER_OPEN, "over the bag", 0.0),
            Waypoint(1.5, (SQ_X, SQ_Y, SQ_Z), FINGER_OPEN, "beside its seams", 0.0),
            Waypoint(2.5, (SQ_X, SQ_Y, SQ_Z), -SQ_HALF, "squeeze", 0.0),
            Waypoint(t_r - t_open - 2.0, (SQ_X, SQ_Y, SQ_Z), -SQ_HALF, "hold open", 0.0),
            Waypoint(1.5, (SQ_X, SQ_Y, SQ_Z), FINGER_OPEN, "let the bag close", 0.0),
            Waypoint(1.0, (SQ_X, SQ_Y, SQ_Z + 0.03), FINGER_OPEN, "up", 0.0),
            Waypoint(1.0, (CARRY_X, SQ_Y, SQ_Z + 0.03), FINGER_OPEN, "over the board in it", 0.0),
            Waypoint(1.0, (CARRY_X, SQ_Y, SQ_Z), FINGER_OPEN, "beside it", 0.0),
            Waypoint(2.0, (CARRY_X, SQ_Y, SQ_Z), -CARRY_HALF, "grip the filled bag", 0.0),
            Waypoint(1.5, (CARRY_X, SQ_Y, CARRY_Z), -CARRY_HALF, "lift it", 0.0),
            Waypoint(2.5, (TOTE_X, TOTE_Y, CARRY_Z), -CARRY_HALF, "over the tote", 0.0),
            Waypoint(2.0, (TOTE_X, TOTE_Y, DROP_Z), -CARRY_HALF, "down to its floor", 0.0),
            Waypoint(1.8, (TOTE_X, TOTE_Y, DROP_Z), FINGER_OPEN, "let go", 0.0),
            Waypoint(1.5, (TOTE_X, TOTE_Y, CARRY_Z), FINGER_OPEN, "up out of it", 0.0),
            Waypoint(2.0, LHOME, FINGER_OPEN, "home", 0.0),
            Waypoint(1.5, LHOME, FINGER_OPEN, "done", 0.0)]
    if os.environ.get("BG_DEMO") == "noopen":  # a shortcut check: the right hand alone pushes the board at the shut bag
        left = [Waypoint(sum(w.duration for w in left), LHOME, FINGER_OPEN, "wait", 0.0)]
    t_l = sum(w.duration for w in left)
    right.append(Waypoint(t_l - t_r, RHOME, FINGER_OPEN, "done", 0.0))
    return {"right": right, "left": left}
