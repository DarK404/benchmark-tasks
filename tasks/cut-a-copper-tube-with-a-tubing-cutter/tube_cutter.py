"""On the station table: a length of 1/2 in copper tube stands upright in a stand. A screw-feed tubing cutter sits
on it at the cut mark, finger-snug (its wheel just touching the tube), its feed knob pointing to the robot's right.

The robot must cut the top 4 cm off the tube (at the mark) with the cutter and take the offcut off. As the maker says: turn the feed knob
a quarter turn, take the cutter right round the tube, and again, until the tube parts. Fed much more than a quarter
turn between laps, the wheel crushes the tube out of round instead of cutting it (a dented tube end will not take a
fitting). The cutter goes round on the tube only when pushed; the knob turns only when turned.

Run it:  inspect-robots run --embodiment yam_arena -E task=tube_cutter.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "Cut the top 4 cm off the copper tube with the tubing cutter, and take the offcut off." ...
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cutterkit  # noqa: E402
from cutterkit import Hinge, TubeCut, body, box, cyl  # noqa: E402
from yam_arena.analysis import box_signed_distance  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

cutterkit.install()
SCENE = "station"
TOP = TABLE_TOP[SCENE]
MM = 0.001
COPPER = (0.78, 0.45, 0.28)
BRASS = (0.80, 0.66, 0.30)
IRON = (0.22, 0.22, 0.23)
PLY = (0.80, 0.68, 0.50)
RIDGID_GREY = (0.45, 0.47, 0.50)  # the cutter's grey body
RED = (0.72, 0.08, 0.06)  # its red knob
STEEL = (0.75, 0.76, 0.78)
DARK = (0.10, 0.08, 0.07)
cutterkit.METALLIC.update({COPPER: 0.35, BRASS: 0.35, STEEL: 0.3})

# -- sizes (env frame: x forward, y left, z up from the table top) ---------------------------------------------------
TX, TY = 0.24, 0.0  # the tube's axis
# the tube: 1/2 in Type L copper (ASTM B88): 0.625 in (15.88 mm) outside, 0.040 in (1.02 mm) wall, ~424 g/m
TUBE_R = 15.875 * MM / 2
WALL = 1.016 * MM
Z_FOOT = 70.0 * MM  # where it comes out of the stand's compression nut
Z_MARK = 100.0 * MM  # the cut mark (the cutter's wheel)
OFFCUT_L = 40.0 * MM  # above the mark: the tube's top at 140 mm
# the cutter: RIDGID No. 10 screw-feed tubing cutter (32910): about 6 in long, 1.5 in tall, 0.8 lb; feed screw
# 7/16-14 (1.81 mm a turn)
PITCH = 25.4 * MM / 14
CUT_H = 38.0 * MM  # its body's height, centred on the mark
RAIL_Y = 18.0 * MM  # its frame's outer half-width
KNOB_X0, KNOB_X1 = 95.0 * MM, 130.0 * MM  # the knob along the cutter's x, from the tube's axis
KNOB_R = 12.5 * MM
CUTTER_MASS, KNOB_MASS = 0.340, 0.020
YAW0 = -math.pi / 2  # the knob points to the robot's right (-y)
QUARTER = PITCH / 4  # the maker's feed per lap
DENT = PITCH / 2  # the wheel fed over half a turn past the groove under it crushes the tube (our proxy)
CLEAR_DIST = 10.0 * MM


def _stand_shapes() -> list:
    """In the stand's frame (on the table under the tube's axis)."""
    return [
        box((0.0, 0.0, 9.0 * MM), (0.12, 0.12, 18.0 * MM), PLY),  # the plywood block
        cyl((0.0, 0.0, 24.0 * MM), 36.5 * MM, 12.0 * MM, IRON),  # the 1/2 in floor flange
        cyl((0.0, 0.0, 44.0 * MM), 16.0 * MM, 28.0 * MM, BRASS),  # the FIP x compression adapter
        cyl((0.0, 0.0, 64.0 * MM), 14.0 * MM, 12.0 * MM, BRASS),  # its compression nut
        cyl((0.0, 0.0, (Z_FOOT - 6.0 * MM + Z_MARK) / 2), TUBE_R, Z_MARK - Z_FOOT + 6.0 * MM, COPPER),  # the stub
    ]


STAND = body("tube_stand", pos=(TX, TY, TOP), shapes=_stand_shapes(), mass=2.0, friction=0.6, contact_offset=0.0005)
cutterkit.kinematic(STAND.name)


def _cutter_shapes() -> tuple[list, list]:
    """In the cutter's frame: origin on the tube's axis at the mark, x out along the feed screw to the knob."""
    h = CUT_H
    solid = [
        box((-17.0 * MM, 0.0, 0.0), (14.0 * MM, 2 * RAIL_Y, h), RIDGID_GREY),  # the roller end of the frame
        box((42.5 * MM, RAIL_Y - 3.5 * MM, 0.0), (105.0 * MM, 7.0 * MM, h), RIDGID_GREY),  # the side rails
        box((42.5 * MM, -RAIL_Y + 3.5 * MM, 0.0), (105.0 * MM, 7.0 * MM, h), RIDGID_GREY),
        box((88.0 * MM, 0.0, 0.0), (14.0 * MM, 2 * RAIL_Y, 26.0 * MM), RIDGID_GREY),  # the screw boss
    ]
    seen = [
        cyl((-10.0 * MM, 6.0 * MM, 0.0), 4.5 * MM, 16.0 * MM, STEEL),  # the two rollers
        cyl((-10.0 * MM, -6.0 * MM, 0.0), 4.5 * MM, 16.0 * MM, STEEL),
        box((30.0 * MM, 0.0, 0.0), (36.0 * MM, 2 * RAIL_Y - 14.0 * MM, 20.0 * MM), RIDGID_GREY),  # the wheel's slide
        cyl((TUBE_R + 9.0 * MM, 0.0, 0.0), 9.0 * MM, 2.5 * MM, STEEL),  # the cutting wheel, on the mark
        cyl((63.0 * MM, 0.0, 0.0), 3.5 * MM, 50.0 * MM, STEEL, axis="X"),  # the feed screw
    ]
    return solid, seen


_CS, _CV = _cutter_shapes()
_R0 = np.array([[math.cos(YAW0), -math.sin(YAW0)], [math.sin(YAW0), math.cos(YAW0)]])
CUTTER = body("tube_cutter", pos=(TX, TY, TOP + Z_MARK), yaw=YAW0, mass=CUTTER_MASS, friction=0.8,
              shapes=_CS + _CV, visual_only=tuple(range(len(_CS), len(_CS) + len(_CV))), contact_offset=0.0005,
              com=(30.0 * MM, 0.0, 0.0), inertia=(1.2e-4, 6.0e-4, 6.5e-4))
_KC = np.array([TX, TY]) + _R0 @ np.array([(KNOB_X0 + KNOB_X1) / 2, 0.0])
# the feed knob: knurled red plastic on the feed screw, turning on the cutter (a white index line along it)
KNOB = body("feed_knob", pos=(_KC[0], _KC[1], TOP + Z_MARK), yaw=YAW0, mass=KNOB_MASS, friction=1.0,
            shapes=[cyl((0.0, 0.0, 0.0), KNOB_R, KNOB_X1 - KNOB_X0, RED, axis="X"),
                    box((0.0, 0.0, KNOB_R + 0.2 * MM), (KNOB_X1 - KNOB_X0 - 1.0 * MM, 3.0 * MM, 0.6 * MM),
                        (0.95, 0.95, 0.95))],
            visual_only=(1,), contact_offset=0.0005, inertia=(1.0e-5, 1.0e-5, 1.0e-5))
# the offcut: the tube above the mark (it does not collide with the cutter, which rides on the cut round it; once cut
# off it stands on the stub's cut end)
OFFCUT = body("offcut", pos=(TX, TY, TOP + Z_MARK), mass=0.017, friction=0.6, contact_offset=0.0005,
              shapes=[cyl((0.0, 0.0, OFFCUT_L / 2), TUBE_R, OFFCUT_L, COPPER),
                      cyl((0.0, 0.0, OFFCUT_L + 0.2 * MM), TUBE_R - WALL, 0.6 * MM, DARK)],
              visual_only=(1,))
cutterkit.no_collide(OFFCUT.name, CUTTER.name)
OBJECTS = [STAND, CUTTER, KNOB, OFFCUT]
cutterkit.register(*OBJECTS)
# the forces (our estimates): fed a quarter turn a lap, the wheel bears on the tube with about 25 N; the cutter
# then turns against a rolling and cutting resistance of about 0.3 N m (5 N pushed 6 cm out), and the knob against
# the feed screw's friction under that load, about 0.025 N m (T = F p / (2 pi eta), eta 0.3: 2 N at its rim, less
# than the 2.7 N at the knob that would turn the whole cutter, so the knob can be turned without holding it). Both
# are Coulomb frictions on their hinges.
CUT_TORQUE = 0.30
KNOB_TORQUE = 0.025
cutterkit.add_hinge(Hinge(STAND.name, CUTTER.name, anchor=(0.0, 0.0, Z_MARK), axis=(0.0, 0.0, 1.0), damping=20.0,
                          max_torque=CUT_TORQUE))
cutterkit.add_hinge(Hinge(CUTTER.name, KNOB.name, anchor=((KNOB_X0 + KNOB_X1) / 2, 0.0, 0.0), axis=(1.0, 0.0, 0.0),
                          damping=0.5, max_torque=KNOB_TORQUE))
MODEL = cutterkit.add_model(TubeCut(CUTTER.name, KNOB.name, OFFCUT.name, ((TX, TY, TOP + Z_MARK), (0, 0, 0, 1)),
                                    PITCH, WALL, DENT, knob_x=(KNOB_X0, KNOB_X1), knob_r=KNOB_R))

# -- scoring -------------------------------------------------------------------------------------------------------
# 1.0   the tube cut through, the offcut off it and down on the board or the table, no gripper within 1 cm of the
#       cutter or the offcut, both still, for 1 s
# 0.5   cut through (the offcut not yet off and down, or a hand on it, or not still)
# 0.25  a groove at least a quarter turn's feed deep (0.45 mm) all round
# 0     anything else; and, latched for the whole trial, the tube dented: the wheel fed over half a turn past the
#       groove under it
SCORE_OVER = "end"
RUN_LIMIT_STEPS = 3000
STILL_STEPS = 20
GROOVE_025 = QUARTER
_STATE: dict = {}


@cutterkit.on_reset
def _reset_state() -> None:
    _STATE.update(still=0)


_reset_state()


def offcut_off(objects) -> bool:
    """The offcut lifted off the stub and set down: its axis over 3 cm from the tube's, resting low on the board
    or the table (its middle under 6 cm up)."""
    p = np.asarray(objects[OFFCUT.name].pos, float) - np.array([0.0, 0.0, TOP])
    return math.hypot(p[0] - TX, p[1] - TY) > 0.03 and p[2] < 0.06


def hands_clear(objects) -> bool:
    pts = cutterkit.finger_points()
    if not len(pts):
        return True
    for spec, centre, size in ((CUTTER, (53.0 * MM, 0.0, 0.0), (154.0 * MM, 2 * RAIL_Y, CUT_H)),
                               (OFFCUT, (0.0, 0.0, OFFCUT_L / 2), (2 * TUBE_R, 2 * TUBE_R, OFFCUT_L))):
        st = objects[spec.name]
        if box_signed_distance(pts, st.to_world(centre), st.rotation(), np.array(size)).min() < CLEAR_DIST:
            return False
    return True


def _update(objects) -> tuple[float, bool]:
    if MODEL.dented:
        return 0.0, False
    if MODEL.through:
        still = all(objects[s.name].resting() for s in (CUTTER, OFFCUT))
        if offcut_off(objects) and still and hands_clear(objects):
            return 1.0, True
        return 0.5, False
    if MODEL.groove.min() >= GROOVE_025 - 1e-9:
        return 0.25, False
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
# Open loop, from the fixed layout only: the knob is turned with the cutter at its start (knob to the right), and
# each lap brings it back there.
TIP = 13.0 * MM  # the TCP is this far behind the pad tips
RHOME, LHOME = (0.14, -0.22, TOP + 0.17), (0.14, 0.22, TOP + 0.17)
PUSH_X = 60.0 * MM  # the pads push the frame's side this far out from the tube's axis
PUSH_Z = Z_MARK - 8.0 * MM  # the pad tips' height while pushing (the frame is 81 to 119 mm up)
LEAD = math.radians(22.0)  # the pads aim this far ahead of the frame's side (the soft arm trails: about 6 N)
END_LEAD = math.radians(17.0)  # ... and at the end of a push (the cutter then stops on its heading)
STEP = math.radians(10.0)
WAIT = "wait"


def _hand_yaw(yaw: float, off: float = 0.0) -> float:
    """The hand turned so its fingers close along the cutter's y, then ``off`` further (rad), wrapped."""
    y = yaw + off
    return (y + math.pi) % (2 * math.pi) - math.pi


def _push_xy(yaw: float, off: float) -> tuple[float, float]:
    """The closed pads' centre beside the frame's -y side, ``PUSH_X`` out, with the hand turned ``off`` from
    closing along the cutter's y (the pads are 8 mm thick along the closing axis and 15.5 mm wide)."""
    half = (8.0 * MM * abs(math.cos(off)) + 15.5 * MM * abs(math.sin(off))) / 2 + 0.5 * MM
    lx, ly = PUSH_X, -(RAIL_Y + half)
    c, s = math.cos(yaw), math.sin(yaw)
    return TX + c * lx - s * ly, TY + s * lx + c * ly


def push(y0: float, y1: float, off: float, label: str) -> list:
    """Closed fingertips down beside the frame's side behind the cutter (for a turn to the left, seen from above),
    then round the tube with it from cutter heading ``y0`` to ``y1``, then up. ``off`` turns the hand from
    closing along the cutter's y, to keep the wrist's last joint within its +/-120 degrees over the arc."""
    z = TOP + PUSH_Z + TIP
    zup = TOP + Z_MARK + CUT_H / 2 + 0.015 + TIP
    x0, yy0 = _push_xy(y0 - math.radians(8.0), off)
    out = [Waypoint(1.5, (x0, yy0, zup), FINGER_CLOSED, f"over the cutter ({label})", _hand_yaw(y0, off)),
           Waypoint(0.8, (x0, yy0, z), FINGER_CLOSED, "down beside it", _hand_yaw(y0, off))]
    n = max(1, round((y1 - y0) / STEP))
    for k in range(1, n + 1):
        a = y0 + (y1 - y0) * k / n
        x, y = _push_xy(a + (LEAD if k < n else END_LEAD), off)
        out.append(Waypoint(0.35, (x, y, z), FINGER_CLOSED, "push it round", _hand_yaw(a, off)))
    x, y = _push_xy(y1 - math.radians(12.0), off)
    out += [Waypoint(0.4, (x, y, z), FINGER_CLOSED, "back off", _hand_yaw(y1, off)),
            Waypoint(0.5, (x, y, zup), FINGER_CLOSED, "up", _hand_yaw(y1, off))]
    return out


ROLL = 0.25 * 2 * math.pi * KNOB_R  # a quarter turn of the knob's rim
ROLL_LAG = 0.0 * MM  # the fingertip aims this much further (the soft arm trails)
ROLL_FROM = 6.0 * MM  # the stroke starts this far past the knob's top (the arm trails ~5 mm: it lands just past it)
STROKES = 2  # a stroke turns the knob about 40 degrees: two make about the maker's quarter turn
ROLL_TO = 16.0 * MM  # ... and rolls on over its far flank (pressed there it turns the tightening way; pressed on the
# near flank it would turn back)
PRESS_IN = 10.0 * MM  # the fingertip aims this far into the knob's top: about 3 N down (the arm is soft)


def _cutter_point(x: float, y: float = 0.0, yaw: float = YAW0) -> tuple[float, float]:
    """A point on the cutter (its frame's x, y) with the cutter at heading ``yaw`` (default: its start), env frame."""
    c, s = math.cos(yaw), math.sin(yaw)
    return float(TX + c * x - s * y), float(TY + s * x + c * y)


LAP_SHORT = math.radians(2.3)  # a lap pushed round stops this far short of the start heading (seen in the sim)


def turn_knob(label: str, yaw: float = YAW0) -> list:
    """The right hand's closed fingertips press on the knob's top and roll it a quarter turn the tightening way
    (clockwise seen from its end: its top drawn towards the cutter's +y side, away from the robot), then lift off."""
    x0, y0 = _cutter_point((KNOB_X0 + KNOB_X1) / 2, ROLL_FROM, yaw)
    x1, y1 = _cutter_point((KNOB_X0 + KNOB_X1) / 2, ROLL_TO, yaw)
    z = TOP + Z_MARK + KNOB_R - PRESS_IN + TIP
    zup = TOP + Z_MARK + KNOB_R + TIP + 0.02
    yaw = _hand_yaw(YAW0, math.pi)  # fingers close across the knob (8 mm along the roll, 15.5 mm along it)
    out = []
    for k in range(STROKES):
        out += [Waypoint(1.0 if k == 0 else 0.6, (x0, y0, zup), FINGER_CLOSED,
                         f"over the knob ({label})" if k == 0 else "back over it", yaw),
                Waypoint(0.6, (x0, y0, z), FINGER_CLOSED, "down on its top", yaw),
                Waypoint(1.2, (x1, y1, z), FINGER_CLOSED, "roll it a quarter turn" if k == 0 else "and on", yaw),
                Waypoint(0.5, (x1, y1, zup), FINGER_CLOSED, "up off it", yaw)]
    return out


def offcut_off_plan() -> list:
    """Take the offcut by its top, lift it off the stub and stand it on the table to the left, then let go."""
    ztop = TOP + Z_MARK + OFFCUT_L
    zg = ztop - 12.0 * MM + TIP
    put = (TX - 0.06, TY + 0.12)
    pinch = -(TUBE_R - 1.5 * MM)
    wide = pinch - 8.0 * MM
    return [Waypoint(2.0, (TX, TY, zg + 0.02), wide, "over the offcut", 0.0),
            Waypoint(0.8, (TX, TY, zg), wide, "round it", 0.0),
            Waypoint(0.6, (TX, TY, zg), pinch, "take it", 0.0),
            Waypoint(1.0, (TX, TY, zg + 0.028), pinch, "lift it off", 0.0),
            Waypoint(1.5, (*put, zg + 0.028), pinch, "carry it", 0.0),
            Waypoint(1.2, (*put, TOP + OFFCUT_L - 12.0 * MM + TIP + 1.0 * MM), pinch, "stand it down", 0.0),
            Waypoint(0.6, (*put, TOP + OFFCUT_L - 12.0 * MM + TIP + 1.0 * MM), wide, "let go", 0.0),
            Waypoint(0.8, (*put, TOP + 0.10), wide, "up", 0.0),
            Waypoint(1.5, LHOME, FINGER_OPEN, "home", 0.0)]


def _sync(a: list, b: list) -> None:
    """Pad the shorter plan with a wait so both end together."""
    ta, tb = sum(w.duration for w in a), sum(w.duration for w in b)
    short, d = (a, tb - ta) if ta < tb else (b, ta - tb)
    if d > 1e-6:
        short.append(Waypoint(d, short[-1].pos, short[-1].finger, WAIT, short[-1].yaw))


LAPS = 3


def demo():
    """``TC_DEMO=nofeed`` (a shortcut check): push the cutter round without ever turning the knob."""
    nofeed = os.environ.get("TC_DEMO") == "nofeed"
    right = [Waypoint(2.0, RHOME, FINGER_CLOSED, WAIT, 0.0)]  # (the fingers close slowly: the run's guard)
    left = [Waypoint(2.0, LHOME, FINGER_CLOSED, WAIT, 0.0)]
    for lap in range(LAPS):
        if not nofeed:
            right += turn_knob(f"lap {lap + 1}", YAW0 if lap == 0 else YAW0 - LAP_SHORT)
        right += [Waypoint(1.2, RHOME, FINGER_CLOSED, "aside", 0.0)]
        _sync(right, left)
        y0 = YAW0 + 2 * math.pi * lap
        a, b, c = y0 + math.radians(120.0), y0 + math.radians(240.0), y0 + 2 * math.pi
        right += push(y0, a, math.radians(60.0), f"lap {lap + 1}") + [Waypoint(1.2, RHOME, FINGER_CLOSED, "aside", 0.0)]
        _sync(right, left)
        left += push(a, b, math.radians(-120.0), f"lap {lap + 1}") + [Waypoint(1.2, LHOME, FINGER_CLOSED, "aside", 0.0)]
        _sync(right, left)
        right += push(b, c, math.radians(-180.0), f"lap {lap + 1}") + [Waypoint(1.5, RHOME, FINGER_CLOSED, "aside", 0.0)]
        _sync(right, left)
    left += offcut_off_plan()
    _sync(right, left)
    for plan in (left, right):
        plan.append(Waypoint(2.0, plan[-1].pos, plan[-1].finger, "done", plan[-1].yaw))
    return {"right": right, "left": left}
