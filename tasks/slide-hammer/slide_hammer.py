"""On the station table: a steel pin is stuck in the bore of a fixture block, its eye standing out of the block's face,
and a slide hammer lies on its rack beside it.

The robot must pull the pin out with the slide hammer and put the slide hammer back: hook it into the pin's eye, rest
its handle end on the work support, and sling its weight along the shaft against the end stop, blow after blow, until
the pin comes out. A steady pull, however long, does not move the pin.

Run it:  inspect-robots run --embodiment yam_arena -E task=slide_hammer.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "..." --scorer task_score ...
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hammerkit  # noqa: E402
from hammerkit import SlideHammerPin, Slider, body, box, cyl, ring  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

hammerkit.install()
SCENE = "station"
TOP = TABLE_TOP[SCENE]
MM = 0.001
STEEL = (0.50, 0.51, 0.53)
BRIGHT = (0.72, 0.73, 0.75)  # the pin and the shaft: bright steel
BLUED = (0.16, 0.18, 0.22)  # the fixture block: blued steel
RED = (0.72, 0.12, 0.09)  # the weight's paint
GRIP = (0.10, 0.10, 0.11)
WOOD = (0.55, 0.40, 0.25)
hammerkit.METALLIC.update({STEEL: 0.35, BRIGHT: 0.2, BLUED: 0.4})

# -- the fixture block and its stuck pin ---------------------------------------------------------------------------
EYE_XY = (0.30, -0.03)  # the middle of the pin's eye, before it moves
PULL = (0.0, 1.0, 0.0)  # the pin's axis, out of the block (towards the left arm)
SHAFT_Z = 40.0 * MM  # the slide hammer's shaft axis above the table, hooked on or on its rack
ARM_BELOW = 6.0 * MM  # the hook's arm's underside below the shaft's axis: it rests on the eye
RING_T = 6.0 * MM
RING_TOP = SHAFT_Z - ARM_BELOW
PIN_Z = RING_TOP - RING_T / 2  # the pin's axis
HOLE_R, RING_R = 12.5 * MM, 22.5 * MM  # the eye's hole and outside (an M10 eye: 25 mm hole, 45 mm across)
PIN_R, PIN_OUT, PIN_IN = 5.0 * MM, 15.0 * MM, 40.0 * MM  # the pin's shank: out of the face, in the bore
FACE_Y = EYE_XY[1] - RING_R - PIN_OUT  # the block's face
BLOCK = (60.0 * MM, 60.0 * MM, 64.0 * MM)
TRAVEL = PIN_IN  # pulled this far, the pin is out

BLOCK_BODY = body("block", pos=(EYE_XY[0], FACE_Y - BLOCK[1] / 2, TOP), friction=0.8, shapes=[
    box((0.0, 0.0, BLOCK[2] / 2), BLOCK, BLUED),
    cyl((0.0, BLOCK[1] / 2 + 0.1 * MM, PIN_Z), PIN_R + 1.5 * MM, 0.2 * MM, (0.05, 0.05, 0.06), axis="Y"),  # bore edge
], visual_only=(1,))


def _pin_shapes() -> list:
    """The pin in its own frame: origin at the eye's middle, x along the pull (out of the block), z up."""
    return [
        *ring(-RING_T / 2, RING_T / 2, HOLE_R, RING_R, BRIGHT, segments=16),  # the flat eye
        cyl((-(RING_R + (PIN_OUT + PIN_IN) / 2 - 1.0 * MM), 0.0, 0.0), PIN_R, PIN_OUT + PIN_IN + 2.0 * MM, BRIGHT,
            axis="X"),  # the shank
    ]


_PS = _pin_shapes()
_pull_yaw = math.atan2(PULL[1], PULL[0])
_q = lambda yaw: (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))  # noqa: E731
PIN = body("pin", pos=(*EYE_XY, TOP + PIN_Z), mass=1e-3, inertia=(1e-8, 1e-8, 1e-8), quat=_q(_pull_yaw),
           shapes=_PS, visual_only=tuple(range(len(_PS))))
hammerkit.no_gravity(PIN.name)
# the eye's collider, just inside the drawn one: the hook's peg bears on it, its arm rests on it, and it is solid to
# the fingers; heavy and held in place every substep, so a pull does not move it (only the model's blows do)
EYE = body("eye_solid", pos=(*EYE_XY, TOP + PIN_Z), mass=20.0, inertia=(1e-2, 1e-2, 1e-2), quat=_q(_pull_yaw),
           friction=0.8, shapes=[*ring(-RING_T / 2 + 0.2 * MM, RING_T / 2 - 0.2 * MM, HOLE_R + 0.3 * MM,
                                       RING_R - 0.3 * MM, BRIGHT, segments=12),
                                 box((-(RING_R + PIN_OUT / 2), 0.0, 0.0), (PIN_OUT, 2 * PIN_R - 0.6 * MM,
                                                                          2 * PIN_R - 0.6 * MM), BRIGHT)])
hammerkit.no_gravity(EYE.name)

# -- the slide hammer ----------------------------------------------------------------------------------------------
# Its frame: origin on the hook's peg at the shaft's axis, x along the shaft towards the handle, z up.
PEG_R, PEG_LEN = 4.0 * MM, 18.0 * MM
SHAFT_W = 14.0 * MM
FRONT_GRIP = 40.0 * MM  # the shaft between the hook and the collar, where the steadying hand holds it
COLLAR_X = (125.0 * MM, 137.0 * MM)  # far enough back for both hands (each gripper is ~10 cm across)
STOP_X = (368.0 * MM, 382.0 * MM)
HANDLE_X = (382.0 * MM, 468.0 * MM)
HANDLE_W = 24.0 * MM
WEIGHT_L, WEIGHT_W = 70.0 * MM, 48.0 * MM  # the weight's length and its ends' width
WAIST_L, WAIST_W, END_L = 46.0 * MM, 36.0 * MM, 12.0 * MM  # its waisted grip between the ends
FRONT_AT = COLLAR_X[1] + WEIGHT_L / 2 + 1.0 * MM  # the weight's middle against the front collar (spawned there)
STOP_AT = STOP_X[0] - WEIGHT_L / 2  # ... and against the end stop
HAMMER_YAW = _pull_yaw  # the shaft along the pull, the handle away from the block
RACK_XY = (0.20, EYE_XY[1])  # the hook's peg when the slide hammer lies on its rack
SUPPORT_AT = 457.0 * MM  # the work support and the rack's back cradle, under the handle's end
HANDLE_GRIP = 420.0 * MM  # where the other hand holds the handle
RACK_FRONT = 95.0 * MM  # the rack's front cradle, under the shaft between the steadying hand and the collar


def _hammer_shapes() -> tuple[list, int]:
    hz = 0.0
    solid = [
        box((0.0, 0.0, -(ARM_BELOW + PEG_LEN) / 2), (2 * PEG_R, 2 * PEG_R, PEG_LEN - ARM_BELOW), STEEL),  # the peg
        box((6.0 * MM, 0.0, hz), (24.0 * MM, 10.0 * MM, 2 * ARM_BELOW), STEEL),  # the hook's arm
        box(((COLLAR_X[0] + COLLAR_X[1]) / 2, 0.0, hz), (COLLAR_X[1] - COLLAR_X[0], 26.0 * MM, 26.0 * MM), STEEL),
        box(((12.0 * MM + STOP_X[1]) / 2, 0.0, hz), (STOP_X[1] - 12.0 * MM, SHAFT_W, SHAFT_W), BRIGHT),
        box(((STOP_X[0] + STOP_X[1]) / 2, 0.0, hz), (STOP_X[1] - STOP_X[0], 32.0 * MM, 32.0 * MM), STEEL),
        box(((HANDLE_X[0] + HANDLE_X[1]) / 2, 0.0, hz), (HANDLE_X[1] - HANDLE_X[0], HANDLE_W, HANDLE_W), GRIP),
    ]
    seen = [
        cyl(((12.0 * MM + STOP_X[1]) / 2, 0.0, hz), SHAFT_W / 2 + 0.2 * MM, STOP_X[1] - 12.0 * MM, BRIGHT,
            axis="X"),
        cyl(((STOP_X[0] + STOP_X[1]) / 2, 0.0, hz), 16.2 * MM, STOP_X[1] - STOP_X[0], STEEL, axis="X"),
        cyl(((COLLAR_X[0] + COLLAR_X[1]) / 2, 0.0, hz), 13.2 * MM, COLLAR_X[1] - COLLAR_X[0], STEEL, axis="X"),
        cyl(((HANDLE_X[0] + HANDLE_X[1]) / 2, 0.0, hz), HANDLE_W / 2 + 0.3 * MM, HANDLE_X[1] - HANDLE_X[0], GRIP,
            axis="X"),
        cyl((0.0, 0.0, -ARM_BELOW - (PEG_LEN - ARM_BELOW) / 2), PEG_R + 0.1 * MM, PEG_LEN - ARM_BELOW, STEEL),
    ]
    return solid + seen, len(solid)


_HS, _NHS = _hammer_shapes()


def hammer_pose(peg_xy, z=SHAFT_Z):
    return (*peg_xy, TOP + z)


HAMMER = body("hammer", pos=hammer_pose(RACK_XY), mass=0.25, com=(0.25, 0.0, 0.0), friction=1.0,
              quat=_q(HAMMER_YAW), shapes=_HS, visual_only=tuple(range(_NHS, len(_HS))))
_hx = np.array([math.cos(HAMMER_YAW), math.sin(HAMMER_YAW), 0.0])
# the weight: square in section (as on many dent pullers), a red waisted grip between two raised steel ends, so the
# hand cannot slide along it
WEIGHT = body("weight", pos=tuple(np.array(hammer_pose(RACK_XY)) + FRONT_AT * _hx), mass=0.30, friction=1.0,
              quat=_q(HAMMER_YAW), shapes=[
                  box((0.0, 0.0, 0.0), (WAIST_L, WAIST_W, WAIST_W), RED),
                  box((-(WEIGHT_L - END_L) / 2, 0.0, 0.0), (END_L, WEIGHT_W, WEIGHT_W), STEEL),
                  box(((WEIGHT_L - END_L) / 2, 0.0, 0.0), (END_L, WEIGHT_W, WEIGHT_W), STEEL),
              ])
hammerkit.add_slider(Slider("hammer", "weight", anchor=(FRONT_AT, 0.0, 0.0), axis=(1.0, 0.0, 0.0),
                            lower=-1.0 * MM, upper=STOP_AT - FRONT_AT))


def _cradle(x_world, y_world, top, half_w, shapes):
    """A cradle under the shaft or the handle running along y: a block up to ``top`` with a cheek either side."""
    shapes += [
        box((x_world, y_world, top / 2), (36.0 * MM, 16.0 * MM, top), WOOD),
        box((x_world + half_w + 4.0 * MM, y_world, top + 5.0 * MM), (8.0 * MM, 16.0 * MM, 10.0 * MM), WOOD),
        box((x_world - half_w - 4.0 * MM, y_world, top + 5.0 * MM), (8.0 * MM, 16.0 * MM, 10.0 * MM), WOOD),
    ]


def _y_at(peg_xy, x_local):
    return peg_xy[1] + x_local * _hx[1]


_support, _rack = [], []
_cradle(0.0, _y_at(EYE_XY, SUPPORT_AT), SHAFT_Z - HANDLE_W / 2, HANDLE_W / 2 + 3.0 * MM, _support)
SUPPORT = body("support", pos=(EYE_XY[0], 0.0, TOP), friction=0.8, shapes=_support)
_cradle(0.0, _y_at(RACK_XY, RACK_FRONT), SHAFT_Z - SHAFT_W / 2, SHAFT_W / 2 + 4.0 * MM, _rack)  # the shaft
_cradle(0.0, _y_at(RACK_XY, SUPPORT_AT), SHAFT_Z - HANDLE_W / 2, HANDLE_W / 2 + 4.0 * MM, _rack)
_rack.append(box((0.0, (_y_at(RACK_XY, RACK_FRONT) + _y_at(RACK_XY, SUPPORT_AT)) / 2, 4.0 * MM),
                 (24.0 * MM, SUPPORT_AT - RACK_FRONT, 8.0 * MM), WOOD))  # the rack's base
RACK = body("rack", pos=(RACK_XY[0], 0.0, TOP), friction=0.8, shapes=_rack)

OBJECTS = [BLOCK_BODY, PIN, EYE, SUPPORT, RACK, HAMMER, WEIGHT]
hammerkit.register(*OBJECTS)
MODEL = hammerkit.add_model(SlideHammerPin(
    "hammer", "weight", "pin", EYE.name, (*EYE_XY, TOP + PIN_Z), PULL, TOP + RING_TOP, hole_r=HOLE_R, ring_r=RING_R,
    peg_r=PEG_R, peg_len=PEG_LEN, arm_below=ARM_BELOW, capture=HOLE_R - PEG_R + 3.0 * MM, stop_at=STOP_AT,
    front_at=FRONT_AT - 1.0 * MM, travel=TRAVEL))


# -- scoring -------------------------------------------------------------------------------------------------------
SCORE_OVER = "end"
RUN_LIMIT_STEPS = 2000
STARTED = 5.0 * MM
CLEAR = 10.0 * MM
STILL = 0.01


def fingers_to(state, spec) -> float:
    """Nearest distance (m) from either gripper's hulls to a body's bounding box (in its own frame). inf offline."""
    pts = hammerkit.finger_points()
    if not len(pts):
        return math.inf
    lo, hi = spec.bounds_local()
    local = (pts - state.pos) @ state.rotation()
    d = np.maximum(np.maximum(lo - local, local - hi), 0.0)
    return float(np.linalg.norm(d, axis=1).min())


def on_rack(hammer) -> bool:
    """Lying back on its rack: the shaft and the handle in their cradles (anywhere along them), level, along the rack."""
    off = np.asarray(hammer.pos) - np.array(hammer_pose(RACK_XY))
    along = float(np.dot(off, _hx))
    across = off - along * _hx
    yaw = math.atan2(hammer.rotation()[1, 0], hammer.rotation()[0, 0])
    return bool(abs(along) < 25.0 * MM and np.linalg.norm(across[:2]) < 8.0 * MM and abs(across[2]) < 5.0 * MM
                and hammer.tilt_deg() < 5.0 and abs(math.remainder(yaw - HAMMER_YAW, 2 * math.pi)) < math.radians(10.0))


def _update(objects) -> tuple[float, bool]:
    m = MODEL
    if m.out:
        h, w = objects["hammer"], objects["weight"]
        if on_rack(h) and min(fingers_to(h, HAMMER), fingers_to(w, WEIGHT)) >= CLEAR:
            return 1.0, bool(h.resting(max_speed=STILL) and w.resting(max_speed=STILL))
        return 0.5, False
    if m.advance >= STARTED:
        return 0.25, False
    return 0.0, False


def success(objects) -> bool:
    """The pin pulled out of its bore; the slide hammer back on its rack, released and still for 1 s."""
    return _update(objects)[1]


def score(objects) -> float:
    """1.0 / 0.5 / 0.25 / 0 as in the README."""
    return _update(objects)[0]


# -- scripted demo -------------------------------------------------------------------------------------------------
# Open loop, from the fixed layout. Both hands carry the slide hammer from its rack (the right by the shaft just behind
# the hook, the left by the handle), hook it into the pin's eye and rest its handle on the work support. The right hand
# keeps the shaft steady and in line; the left takes the weight and slings it against the end stop again and again,
# easing it back between blows, until the pin is out. Then the left takes the handle again and both lay the slide
# hammer back on its rack.
RHOME, LHOME = (0.16, -0.22, TOP + 0.15), (0.16, 0.22, TOP + 0.15)
TIP_PAST_TCP = 13.0 * MM
RYAW, LYAW = math.radians(90.0), math.radians(-90.0)  # the fingers closing along x, across the shaft
SHAFT_TCP_Z = SHAFT_Z - SHAFT_W / 2 - 2.0 * MM + TIP_PAST_TCP  # fingertips just under the shaft
HANDLE_TCP_Z = SHAFT_Z - HANDLE_W / 2 - 2.0 * MM + TIP_PAST_TCP
WEIGHT_TCP_Z = SHAFT_Z - WAIST_W / 2 + 4.0 * MM + TIP_PAST_TCP  # fingertips 4 mm above the waist's underside
HOLD_SHAFT = FINGER_OPEN * 4.0 / 94.0
HOLD_HANDLE = FINGER_OPEN * 14.0 / 94.0
HOLD_WEIGHT = FINGER_OPEN * 26.0 / 94.0
THROW_TO = float(os.environ.get("THROW_TO", "0.36"))  # the hand aims 27 mm past the stop, so the weight strikes it fast
THROW_S = float(os.environ.get("THROW_S", "0.35"))
BLOWS = int(os.environ.get("BLOWS", "10"))
FOLLOW = 3.0 * MM  # each throw aims this much further: the slide hammer comes out with the pin
SHIFT = 25.0 * MM  # how far the slide hammer has come out with the pin, the steadying hand going with it
DEMO_MODE = os.environ.get("HAMMER_DEMO", "")  # shortcut checks: "steady" pulls the weight against the stop slowly


def at(peg_xy, x_local, z):
    """A world point on the slide hammer's line, ``x_local`` along the shaft from the peg at ``peg_xy``."""
    return (peg_xy[0] + x_local * _hx[0], peg_xy[1] + x_local * _hx[1], TOP + z)


def _w(dur, pos, finger, label, yaw):
    return Waypoint(dur, tuple(pos), finger, label, yaw)


def _t(w) -> float:
    return sum(p.duration for p in w)


def _carry(peg_from, peg_to, labels, r_at=FRONT_GRIP, l_at=HANDLE_GRIP):
    """Both hands: lift the slide hammer (held at ``r_at`` and ``l_at`` along it) off where its peg is at ``peg_from``,
    carry it over ``peg_to``, steady it, lower it."""
    up = 0.04
    r, l = [], []
    for dur, peg, dz, lab in [(1.2, peg_from, up, labels[0]), (2.0, peg_to, up, labels[1]),
                              (1.0, peg_to, 0.015, "steady"), (1.5, peg_to, 0.0, labels[2])]:
        r.append(_w(dur, at(peg, r_at, SHAFT_TCP_Z + dz), HOLD_SHAFT, lab, RYAW))
        l.append(_w(dur, at(peg, l_at, HANDLE_TCP_Z + dz), HOLD_HANDLE, lab, LYAW))
    return r, l


def demo():
    up = 0.04
    rack, work = RACK_XY, EYE_XY
    # take hold of it on the rack, both hands at once
    right, left = [], []
    for dur, dz, rf, lf, lab in [(1.8, up, FINGER_OPEN, FINGER_OPEN, "over it"), (1.2, 0.0, FINGER_OPEN, FINGER_OPEN, "round it"),
                                 (1.6, 0.0, HOLD_SHAFT, HOLD_HANDLE, "grip")]:
        right.append(_w(dur, at(rack, FRONT_GRIP, SHAFT_TCP_Z + dz), rf, lab, RYAW))
        left.append(_w(dur, at(rack, HANDLE_GRIP, HANDLE_TCP_Z + dz), lf, lab, LYAW))
    r, l = _carry(rack, work, ("lift it off", "over the eye", "hook it in"))
    right += r
    left += l
    # the left hand lets go of the handle (now on the support) and takes the weight; the right holds the shaft
    left += [
        _w(1.6, at(work, HANDLE_GRIP, HANDLE_TCP_Z), FINGER_OPEN, "let go", LYAW),
        _w(1.0, at(work, HANDLE_GRIP, HANDLE_TCP_Z + up), FINGER_OPEN, "up", LYAW),
        _w(1.5, at(work, FRONT_AT, WEIGHT_TCP_Z + up), FINGER_OPEN, "over the weight", LYAW),
        _w(1.2, at(work, FRONT_AT, WEIGHT_TCP_Z), FINGER_OPEN, "round it", LYAW),
        _w(1.6, at(work, FRONT_AT, WEIGHT_TCP_Z), HOLD_WEIGHT, "grip the weight", LYAW),
    ]
    back = FRONT_AT + 20.0 * MM
    for k in range(BLOWS):
        moved = min(k * FOLLOW, SHIFT)  # the slide hammer has come out this far with the pin
        if DEMO_MODE == "steady":  # the same pulls, slowly: the weight laid against the stop and pulled on
            left += [_w(2.0, at(work, THROW_TO + moved, WEIGHT_TCP_Z), HOLD_WEIGHT, f"pull {k + 1}", LYAW),
                     _w(0.6, at(work, THROW_TO + moved, WEIGHT_TCP_Z), HOLD_WEIGHT, f"hold {k + 1}", LYAW)]
        else:
            left += [_w(THROW_S, at(work, THROW_TO + moved, WEIGHT_TCP_Z), HOLD_WEIGHT, f"blow {k + 1}", LYAW),
                     _w(0.2, at(work, THROW_TO + moved, WEIGHT_TCP_Z), HOLD_WEIGHT, f"hold {k + 1}", LYAW)]
        left.append(_w(1.0, at(work, back + moved, WEIGHT_TCP_Z), HOLD_WEIGHT, f"ease back {k + 1}", LYAW))
    left += [
        _w(1.0, at(work, FRONT_AT + SHIFT + 3.0 * MM, WEIGHT_TCP_Z), HOLD_WEIGHT, "weight forward", LYAW),
        _w(1.6, at(work, FRONT_AT + SHIFT + 3.0 * MM, WEIGHT_TCP_Z), FINGER_OPEN, "let go of it", LYAW),
        _w(1.0, at(work, FRONT_AT + SHIFT, WEIGHT_TCP_Z + up), FINGER_OPEN, "up", LYAW),
        _w(1.5, at(work, HANDLE_GRIP + SHIFT, HANDLE_TCP_Z + up), FINGER_OPEN, "over the handle", LYAW),
        _w(1.2, at(work, HANDLE_GRIP + SHIFT, HANDLE_TCP_Z), FINGER_OPEN, "round it", LYAW),
        _w(1.6, at(work, HANDLE_GRIP + SHIFT, HANDLE_TCP_Z), HOLD_HANDLE, "grip", LYAW),
    ]
    right.append(_w(_t(left) - _t(right), at(work, FRONT_GRIP, SHAFT_TCP_Z), HOLD_SHAFT, "steady the shaft", RYAW))
    # both lay it back on its rack (it has come out with the pin, the steadying hand with it)
    out = (work[0] + SHIFT * _hx[0], work[1] + SHIFT * _hx[1])
    r, l = _carry(out, rack, ("lift it out", "over the rack", "lay it on"))
    right += r
    left += l
    for side, hold_z, grip_x, yaw, home in [(right, SHAFT_TCP_Z, FRONT_GRIP, RYAW, RHOME),
                                            (left, HANDLE_TCP_Z, HANDLE_GRIP, LYAW, LHOME)]:
        side += [
            _w(1.6, at(rack, grip_x, hold_z), FINGER_OPEN, "let go", yaw),
            _w(1.0, at(rack, grip_x, hold_z + 0.05), FINGER_OPEN, "up", yaw),
            _w(1.5, home, FINGER_OPEN, "home", 0.0),
        ]
    end = max(_t(right), _t(left)) + 2.5
    right.append(_w(end - _t(right), RHOME, FINGER_OPEN, "done", 0.0))
    left.append(_w(end - _t(left), LHOME, FINGER_OPEN, "done", 0.0))
    return {"right": right, "left": left}
