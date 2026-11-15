"""On the station table: a 12-inch table fan lies on its back on a towel, face up, after cleaning, its rear grille on
and its motor shaft pointing up. Its washed blade lies on the left, its front grille on the right, its spinner on the
table in front of the right hand.

The robot must put the fan back together as its manual says: the blade onto the shaft until the shaft's retaining
pin fits its notch, the spinner screwed onto the shaft anticlockwise (left-hand thread; the shaft turns freely, so
the blade is held still meanwhile), and the front grille on the rear grille with its clips closed.

Run it:  inspect-robots run --embodiment yam_arena -E task=desk_fan.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "Put the fan back together." --scorer task_score ...
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fankit  # noqa: E402
from fankit import Fan, body, box, cyl, prism12, ring  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

fankit.install()
SCENE = "station"
TOP = TABLE_TOP[SCENE]
MM = 0.001
E = os.environ

WHITE = (0.93, 0.93, 0.92)
GREY = (0.55, 0.56, 0.58)
STEEL = (0.75, 0.76, 0.78)
CLEAR = (0.55, 0.72, 0.90)  # the blade: light blue plastic
BLUE = (0.25, 0.45, 0.80)
TOWEL = (0.86, 0.80, 0.70)
DARK = (0.15, 0.15, 0.17)
fankit.METALLIC.update({STEEL: 0.5})

# -- sizes: a Midea FT30-8MB 12-inch table fan; the maker publishes no part dimensions: these are our estimates -----
AXIS = (0.27, 0.0)  # the motor shaft's axis, the fan lying face up (its rim reachable all round but its far point)
HOUSING_R, HOUSING_H = 60.0 * MM, 80.0 * MM  # the motor housing, lying on its back
NUT_R, NUT_TOP = 30.0 * MM, 90.0 * MM  # the rear grille's plastic nut, tight on the housing
SHAFT_R, SHAFT_TOP = 4.0 * MM, 122.0 * MM
PIN_Z = 104.0 * MM  # the shaft's retaining pin
RIM_R, RIM_Z = 168.0 * MM, 105.0 * MM  # the rear grille's rim (the front grille sits on it)
HUB_R, HUB_H = 20.0 * MM, 18.0 * MM  # the blade's hub, with its notch for the pin
BLADE_R, BLADE_W, BLADE_T = 150.0 * MM, 80.0 * MM, 3.0 * MM  # a 300 mm (12-inch) three-blade propeller
SPIN_R, SPIN_H = 20.0 * MM, 15.0 * MM  # the spinner (a knob with a left-hand thread)
DOME_H = 24.0 * MM  # the front grille's rise from its rim to its centre badge
BADGE_R, BADGE_H = 30.0 * MM, 10.0 * MM
CLIP_ANGLES = (45.0, 135.0, 225.0, 315.0)  # the front grille's four clips (the fan turned so none is at its far point)
TIGHT_TURNS = 1.0


def _fan_shapes() -> tuple[list, list]:
    ax, ay = AXIS
    solid = [
        *prism12((ax, ay, HOUSING_H / 2), HOUSING_R, HOUSING_H, WHITE),  # the motor housing on its back
        # the rear grille's rim
        *ring(RIM_Z - 4.0 * MM, RIM_Z, RIM_R - 6.0 * MM, RIM_R, GREY, segments=32, centre=AXIS),
    ]
    seen = [
        cyl((ax, ay, (HOUSING_H + NUT_TOP) / 2), NUT_R, NUT_TOP - HOUSING_H, WHITE),  # the plastic nut
        cyl((ax, ay, (NUT_TOP + SHAFT_TOP) / 2), SHAFT_R, SHAFT_TOP - NUT_TOP, STEEL),  # the motor shaft
        box((ax, ay, PIN_Z), (2 * SHAFT_R + 6.0 * MM, 2.0 * MM, 2.0 * MM), STEEL),  # its retaining pin (along x)
        cyl((ax, ay, (HOUSING_H + RIM_Z) / 2 - 10.0 * MM), RIM_R - 30.0 * MM, 1.0 * MM, GREY),  # the rear grille's mesh
        box((ax + 0.21, ay, 0.02), (0.18, 0.04, 0.04), WHITE),  # the pillar, lying behind the head
        cyl((ax + 0.31, ay, 0.09), 0.09, 0.02, WHITE, axis="X"),  # the base, upright behind it
        box((ax + RIM_R - 3.0 * MM, ay, RIM_Z + 6.0 * MM), (12.0 * MM, 30.0 * MM, 6.0 * MM), GREY),  # the handle (up)
    ]
    return solid, seen


_FS, _FV = _fan_shapes()
TOWEL_BODY = body("towel", pos=(AXIS[0], AXIS[1], TOP), friction=0.8,
                  shapes=[box((0.0, 0.0, 1.0 * MM), (0.40, 0.40, 2.0 * MM), TOWEL)])
FAN = body("fan", pos=(0.0, 0.0, TOP + 2.0 * MM), friction=0.6, shapes=_FS + _FV,
           visual_only=tuple(range(len(_FS), len(_FS) + len(_FV))))

BLADE_XY = (0.32, 0.36)
FRONT_XY = (0.32, -0.36)
SPIN_XY = (0.13, -0.22)  # clear of the rear grille's rim (the gripper body is wider than the spinner)


def _blade_shapes() -> tuple[list, list]:
    solid = [*prism12((0.0, 0.0, HUB_H / 2), HUB_R, HUB_H, CLEAR)]
    seen = [box((HUB_R + 0.5 * MM, 0.0, HUB_H - 3.0 * MM), (1.0 * MM, 3.0 * MM, 6.0 * MM), DARK),  # the notch, at 0
            box((-HUB_R - 0.5 * MM, 0.0, HUB_H - 3.0 * MM), (1.0 * MM, 3.0 * MM, 6.0 * MM), DARK)]
    for k in range(3):
        a = math.radians(90.0 + 120.0 * k)
        r = (HUB_R + BLADE_R) / 2
        solid.append(box((r * math.cos(a), r * math.sin(a), 8.0 * MM), (BLADE_R - HUB_R, BLADE_W, BLADE_T), CLEAR,
                         yaw=a))
    return solid, seen


_BS, _BV = _blade_shapes()
BLADE = body("blade", pos=(*BLADE_XY, TOP + 0.5 * MM), yaw=math.pi, mass=0.06, friction=0.8, shapes=_BS + _BV,
             visual_only=tuple(range(len(_BS), len(_BS) + len(_BV))), inertia=(2e-4, 2e-4, 4e-4),
             solver_iterations=32)
SPINNER = body("spinner", pos=(*SPIN_XY, TOP + 0.5 * MM), mass=0.01, friction=0.9,
               shapes=[*prism12((0.0, 0.0, SPIN_H / 2), SPIN_R, SPIN_H, WHITE),
                       box((0.0, 0.0, SPIN_H + 0.2 * MM), (2 * SPIN_R - 6.0 * MM, 3.0 * MM, 0.4 * MM), DARK)],
               visual_only=(6,), solver_iterations=32)


def _front_shapes() -> tuple[list, list]:
    """The front grille, dome up, its origin on its axis at its rim."""
    solid = [*ring(-3.0 * MM, 3.0 * MM, RIM_R - 6.0 * MM, RIM_R, GREY, segments=32),
             *prism12((0.0, 0.0, DOME_H + BADGE_H / 2), BADGE_R, BADGE_H, BLUE)]
    seen = []
    for k in range(16):  # the wire spokes from the rim up to the badge
        a = 2 * math.pi * k / 16
        r0, r1 = BADGE_R, RIM_R - 6.0 * MM
        rm = (r0 + r1) / 2
        ln = math.hypot(r1 - r0, DOME_H)
        pitch = math.atan2(DOME_H, r1 - r0)
        seen.append(box((rm * math.cos(a), rm * math.sin(a), DOME_H / 2), (ln, 2.0 * MM, 2.0 * MM), GREY, yaw=a))
        _ = pitch
    for rr, zz in ((70.0 * MM, 0.75 * DOME_H), (120.0 * MM, 0.4 * DOME_H)):
        seen += ring(zz - 1.0 * MM, zz + 1.0 * MM, rr - 1.0 * MM, rr + 1.0 * MM, GREY, segments=24)
    return solid, seen


_GS, _GV = _front_shapes()
FRONT = body("front", pos=(*FRONT_XY, TOP + 3.5 * MM), mass=0.12, friction=0.8, shapes=_GS + _GV,
             visual_only=tuple(range(len(_GS), len(_GS) + len(_GV))), inertia=(1e-3, 1e-3, 2e-3),
             solver_iterations=32)
CLIPS = [body(f"clip{k}", pos=(FRONT_XY[0] + RIM_R * math.cos(math.radians(a)),
                              FRONT_XY[1] + RIM_R * math.sin(math.radians(a)), TOP + 3.5 * MM), mass=0.002,
              shapes=[box((3.0 * MM, 0.0, -4.0 * MM), (4.0 * MM, 12.0 * MM, 12.0 * MM), WHITE)],
              visual_only=(0,))
         for k, a in enumerate(CLIP_ANGLES)]

OBJECTS = [TOWEL_BODY, FAN, BLADE, SPINNER, FRONT, *CLIPS]
fankit.register(*OBJECTS)
for c in CLIPS:  # until the grille is on they ride with it, out of the way: the model draws them
    fankit.no_gravity(c.name)

BLADE_SEAT = NUT_TOP + 2.0 * MM  # the hub on the pin, just above the nut
SPIN_SEAT = BLADE_SEAT + HUB_H
FRONT_SEAT = RIM_Z + 2.0 * MM + 3.5 * MM  # its rim (6 mm deep) on the rear's (the fan 2 mm up on the towel)
MODEL = fankit.add_model(Fan(
    "blade", "spinner", "front", [c.name for c in CLIPS], axis=AXIS, shaft_z=TOP + SHAFT_TOP,
    blade_seat=TOP + BLADE_SEAT, blade_r=BLADE_R, hub_r=HUB_R, spinner_seat=TOP + SPIN_SEAT, spinner_r=SPIN_R,
    front_seat=TOP + FRONT_SEAT, clip_r=RIM_R + 3.0 * MM, clip_angles=CLIP_ANGLES, pin_angle=0.0, clip_offset=0.0,
    tight_turns=TIGHT_TURNS))

# -- scoring -------------------------------------------------------------------------------------------------------
# 1.0   the blade on the shaft's pin; the spinner screwed tight; the front grille on with all four clips closed; no
#       part of either gripper within 1 cm of the spinner or the front grille; both resting, held for 1 s
# 0.5   the blade on and the spinner tight, but the front grille not on or a clip open (or not yet still)
# 0.25  the blade on its pin, the spinner not tight
# 0     anything else
SCORE_OVER = "end"
RUN_LIMIT_STEPS = 2400
CLEAR_DIST = 10.0 * MM
STILL_STEPS = 20
_STATE: dict = {}


@fankit.on_reset
def _reset_state() -> None:
    _STATE.update(still=0)


_reset_state()
SPECS = {s.name: s for s in OBJECTS}


def fingers_to(state, spec) -> float:
    pts = fankit.finger_points()
    if not len(pts):
        return math.inf
    lo, hi = spec.bounds_local()
    local = (pts - state.pos) @ state.rotation()
    d = np.maximum(np.maximum(lo - local, local - hi), 0.0)
    return float(np.linalg.norm(d, axis=1).min())


def fingers_to_grille(state) -> float:
    """The nearest finger point's distance from the front grille, taken as a round dome (its rim to its badge)."""
    pts = fankit.finger_points()
    if not len(pts):
        return math.inf
    local = (pts - state.pos) @ state.rotation()
    out_r = np.maximum(np.hypot(local[:, 0], local[:, 1]) - RIM_R, 0.0)
    out_z = np.maximum(np.maximum(-3.0 * MM - local[:, 2], local[:, 2] - (DOME_H + BADGE_H)), 0.0)
    return float(np.hypot(out_r, out_z).min())


def _update(objects) -> tuple[float, bool]:
    m = MODEL
    if not m.blade_on:
        return 0.0, False
    if not m.tight:
        return 0.25, False
    if m.front_on and all(m.clips):
        still = objects["spinner"].resting() and objects["front"].resting()
        gap = min(fingers_to(objects["spinner"], SPECS["spinner"]), fingers_to_grille(objects["front"]))
        if still and gap >= CLEAR_DIST:
            return 1.0, True
    return 0.5, False


def success(objects) -> bool:
    """Put back together, let go (held 1 s by the run). Counts the steps the full-score state has held, for score()."""
    ok = _update(objects)[1]
    _STATE["still"] = _STATE["still"] + 1 if ok else 0
    return ok


def score(objects) -> float:
    """1.0 / 0.5 / 0.25 / 0 as above; 1.0 only once the full-score state has held for 1 s (until then 0.5)."""
    tier = _update(objects)[0]
    if tier == 1.0 and _STATE["still"] < STILL_STEPS:
        return 0.5
    return tier


# -- scripted demo -------------------------------------------------------------------------------------------------
TIP = 13.0 * MM
RHOME, LHOME = (0.16, -0.22, TOP + 0.15), (0.16, 0.22, TOP + 0.15)
SQUEEZE = 1.0 * MM
HOLD_HUB = -(HUB_R - SQUEEZE)
HOLD_SPIN = -(SPIN_R - SQUEEZE)
HOLD_BADGE = -(BADGE_R - SQUEEZE)
OPEN_HUB = -32.0 * MM
DEMO_MODE = E.get("FAN_DEMO", "")  # shortcut checks: "nohold" turns the spinner with nothing holding the blade


def down(deg: float) -> float:
    return math.radians(deg)


def _t(w) -> float:
    return sum(p.duration for p in w)


def wait(t: float, last: Waypoint, label: str) -> Waypoint:
    return Waypoint(max(0.05, t), last.pos, last.finger, label, last.yaw)


def c(z):
    return (AXIS[0], AXIS[1], TOP + z)


def left_blade() -> list:
    """Take the blade by its hub, carry it over and lower it onto the shaft, its notch on the pin."""
    x, y = BLADE_XY
    zg = TOP + 0.5 * MM + 12.0 * MM + TIP
    zs = TOP + BLADE_SEAT + 12.0 * MM + TIP
    return [
        Waypoint(2.5, (x, y, zg + 0.06), OPEN_HUB, "over the blade", 0.0),
        Waypoint(1.5, (x, y, zg), OPEN_HUB, "round its hub", 0.0),
        Waypoint(1.6, (x, y, zg), HOLD_HUB, "take the blade", 0.0),
        Waypoint(1.5, (x, y, zs + 0.04), HOLD_HUB, "lift it", 0.0),
        Waypoint(2.5, (AXIS[0], AXIS[1], zs + 0.04), HOLD_HUB, "over the shaft", 0.0),
        Waypoint(1.5, (AXIS[0], AXIS[1], zs + 0.001), HOLD_HUB, "onto the pin", 0.0),
        Waypoint(1.6, (AXIS[0], AXIS[1], zs + 0.001), OPEN_HUB, "let go of the hub", 0.0),
        Waypoint(1.0, (AXIS[0], AXIS[1], zs + 0.03), OPEN_HUB, "up", 0.0),
    ]


HOLD_AT = math.radians(150.0)  # the blade the left hand holds still (its middle): the one towards the near left


HOLD_R = 140.0 * MM  # where along it: far enough out that the two wrists stay clear of each other
HOLD_YAW = HOLD_AT - math.pi  # the fingers closed across the blade (the same line as HOLD_AT, a yaw the wrist reaches)


def _hold_xyz() -> tuple:
    x, y = AXIS[0] + HOLD_R * math.cos(HOLD_AT), AXIS[1] + HOLD_R * math.sin(HOLD_AT)
    return x, y, TOP + BLADE_SEAT + 8.0 * MM + BLADE_T / 2 + TIP + 1.0 * MM


def left_over_blade() -> list:
    """Straight from the hub to above the blade it will hold, out of the right hand's way."""
    x, y, z = _hold_xyz()
    return [Waypoint(1.6, (x, y, z + 0.04), FINGER_CLOSED, "over a blade", HOLD_YAW)]


def left_hold_blade() -> list:
    x, y, z = _hold_xyz()  # fingers closed, pressing the blade still from above
    return [Waypoint(1.2, (x, y, z), FINGER_CLOSED, "hold the blade still", HOLD_YAW)]


def left_off_blade() -> list:
    x, y, z = _hold_xyz()
    return [Waypoint(1.0, (x, y, z + 0.06), FINGER_CLOSED, "off the blade", HOLD_YAW),
            Waypoint(2.0, LHOME, FINGER_OPEN, "home", 0.0)]


STROKE = 120.0  # degrees per stroke turning the spinner


def right_spinner() -> list:
    x, y = SPIN_XY
    low = 4.0 * MM  # the fingertips this far above the spinner's base
    zg = TOP + 0.5 * MM + low + TIP
    zs = TOP + SPIN_SEAT + TIGHT_TURNS * 1.5 * MM + low + TIP
    out = [
        Waypoint(2.0, (x, y, zg + 0.05), OPEN_HUB, "over the spinner", down(-60.0)),
        Waypoint(1.2, (x, y, zg), OPEN_HUB, "round the spinner", down(-60.0)),
        Waypoint(1.6, (x, y, zg), HOLD_SPIN, "take the spinner", down(-60.0)),
        Waypoint(1.5, (x, y, zs + 0.04), HOLD_SPIN, "lift it", down(-60.0)),
        Waypoint(2.0, (*AXIS, zs + 0.015), HOLD_SPIN, "over the shaft's end", down(-60.0)),
        Waypoint(1.5, (*AXIS, zs), HOLD_SPIN, "onto the thread", down(-60.0)),
    ]
    turns = int(math.ceil(TIGHT_TURNS * 360.0 / STROKE)) + 2  # a spare stroke: a regrip loses a little
    z = zs
    for k in range(turns):
        out.append(Waypoint(2.0, (*AXIS, z), HOLD_SPIN, "screw it on anticlockwise", down(-60.0 + STROKE)))
        out.append(Waypoint(1.6, (*AXIS, z), OPEN_HUB, "let go", down(-60.0 + STROKE)))
        out.append(Waypoint(1.2, (*AXIS, z), OPEN_HUB, "turn the hand back", down(-60.0)))
        z -= STROKE / 360.0 * 1.5 * MM
        out.append(Waypoint(1.6, (*AXIS, z), HOLD_SPIN, "take it again", down(-60.0)))
    out += [Waypoint(1.6, (*AXIS, z), OPEN_HUB, "let go of the spinner", down(-60.0)),
            Waypoint(1.0, (*AXIS, z + 0.012), OPEN_HUB, "up", down(-60.0)),
            Waypoint(1.5, (AXIS[0] - 0.05, AXIS[1] - 0.12, z + 0.012), OPEN_HUB, "aside", down(-60.0))]
    return out


CARRY = SPIN_SEAT + SPIN_H + TIGHT_TURNS * 1.5 * MM + 10.0 * MM - FRONT_SEAT  # its rim 10 mm over the spinner


def right_front() -> list:
    x, y = FRONT_XY
    low = 1.0 * MM  # the fingertips this far above the badge's underside
    zg = TOP + 3.5 * MM + DOME_H + low + TIP
    zs = TOP + FRONT_SEAT + DOME_H + low + TIP
    return [
        Waypoint(2.0, (x, y, zg + 0.05), -40.0 * MM, "over the grille", 0.0),
        Waypoint(1.2, (x, y, zg), -40.0 * MM, "round its badge", 0.0),
        Waypoint(1.6, (x, y, zg), HOLD_BADGE, "take the grille", 0.0),
        Waypoint(2.0, (x, y, zs + CARRY), HOLD_BADGE, "lift it", 0.0),
        Waypoint(2.5, (*AXIS, zs + CARRY), HOLD_BADGE, "over the fan", 0.0),
        Waypoint(1.2, (*AXIS, zs + 0.006), HOLD_BADGE, "down over the spinner", 0.0),
        Waypoint(1.5, (*AXIS, zs + 0.001), HOLD_BADGE, "onto the rear grille", 0.0),
        Waypoint(1.6, (*AXIS, zs + 0.001), -40.0 * MM, "let go of the grille", 0.0),
        Waypoint(1.0, (*AXIS, zs + 0.004), -40.0 * MM, "up", 0.0),
        Waypoint(1.5, (AXIS[0] - 0.06, AXIS[1] - 0.10, zs - 0.01), -40.0 * MM, "aside", 0.0),
    ]


def press_clip(angle_deg: float, label: str) -> list:
    a = math.radians(angle_deg)
    r = RIM_R + 3.0 * MM
    x, y = AXIS[0] + r * math.cos(a), AXIS[1] + r * math.sin(a)
    z = TOP + FRONT_SEAT + TIP
    return [Waypoint(1.6, (x, y, z + 0.015), FINGER_CLOSED, f"over {label}", 0.0),
            Waypoint(1.0, (x, y, z - 0.006), FINGER_CLOSED, f"press {label} shut", 0.0),
            Waypoint(0.8, (x, y, z + 0.015), FINGER_CLOSED, f"off {label}", 0.0)]


def demo():
    left = left_blade()
    right = [wait(_t(left), Waypoint(0.1, RHOME, FINGER_OPEN, "", 0.0), "wait for the blade")]
    sp = right_spinner()
    turn_starts = _t(right) + _t(sp[:6])
    left += left_over_blade()
    if DEMO_MODE == "nohold":
        left += left_off_blade()
    else:
        left.append(wait(turn_starts - _t(left) - _t(left_hold_blade()), left[-1], "wait"))
        left += left_hold_blade()
    right += sp
    if DEMO_MODE != "nohold":
        left.append(wait(_t(right) - _t(left), left[-1], "hold the blade still"))
        left += left_off_blade()
    front = right_front()
    right += front
    # both hands close the clips: the left those on its side, the right those on its own
    left.append(wait(_t(right) - _t(left), left[-1], "wait for the grille"))
    left += press_clip(135.0, "the near-left clip") + press_clip(45.0, "the far-left clip")
    right += press_clip(225.0, "the near-right clip") + press_clip(315.0, "the far-right clip")
    left.append(Waypoint(2.0, LHOME, FINGER_OPEN, "home", 0.0))
    right.append(Waypoint(2.0, RHOME, FINGER_OPEN, "home", 0.0))
    if _t(left) > _t(right):
        right.append(wait(_t(left) - _t(right), right[-1], "done"))
    else:
        left.append(wait(_t(right) - _t(left), left[-1], "done"))
    for plan in (left, right):
        plan.append(wait(2.0, plan[-1], "done"))
    return {"right": right, "left": left}
