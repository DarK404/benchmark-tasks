"""On the station table: two freshly glued oak boards lie edge to edge, the glue line still 2 mm open, across two small
sliding bar clamps lying jaws-up on the bench near the boards' ends, 24 cm in front of the arm bases. A third clamp
hangs jaws-down on a wooden rack on the left.

The robot must clamp up the panel: every clamp closed snug so the joint closes, the clamps alternating below and
above so the panel dries flat, and none forced so hard that it squeezes the glue out.

Run it:  inspect-robots run --embodiment yam_arena -E task=glue_panel.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "..." --scorer task_score ...
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gluekit  # noqa: E402
from gluekit import Clamp, Clamps, Slider, body, box, cyl  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

gluekit.install()
SCENE = "station"
TOP = TABLE_TOP[SCENE]
MM = 0.001
OAK = (0.60, 0.40, 0.22)  # red oak, planed
GRAIN_DARK = (0.46, 0.29, 0.15)
GRAIN_LIGHT = (0.68, 0.48, 0.28)
END_GRAIN = (0.48, 0.31, 0.17)
GLUE = (0.95, 0.93, 0.80)
BAR_STEEL = (0.55, 0.56, 0.58)
JAW_RED = (0.72, 0.12, 0.10)
PAD = (0.15, 0.15, 0.16)
WOOD = (0.55, 0.40, 0.25)
gluekit.METALLIC.update({BAR_STEEL: 0.35})

# -- the panel (two boards along y; board A on the robot's side, B beyond) ----------------------------------------
# WHY 28 cm out and one hand at a time (2026-11-10, sim trace): pushing the two lower clamps' jaws at once, 11 cm
# apart and 14 cm in front of the bases, the arms ran into each other and each ended 70 mm off its target.
PANEL_XY = (0.28, 0.0)
BOARD_L, BOARD_W, BOARD_T = 160.0 * MM, 50.0 * MM, 12.0 * MM
GAP0 = 2.0 * MM  # the glue line, open to start
BAR_T = 8.0 * MM  # the clamp bars' thickness: the boards lie on the lower bars
BOARD_Z = BAR_T + BOARD_T / 2


def _board(name: str, side: int):
    x = PANEL_XY[0] + side * (BOARD_W / 2 + GAP0 / 2)
    shapes = [
        box((0.0, 0.0, 0.0), (BOARD_W, BOARD_L, BOARD_T), OAK),
        box((0.0, BOARD_L / 2 + 0.1 * MM, 0.0), (BOARD_W - 0.2 * MM, 0.2 * MM, BOARD_T - 0.2 * MM), END_GRAIN),
        box((0.0, -BOARD_L / 2 - 0.1 * MM, 0.0), (BOARD_W - 0.2 * MM, 0.2 * MM, BOARD_T - 0.2 * MM), END_GRAIN),
        box((-side * (BOARD_W / 2 + 0.1 * MM), 0.0, 0.0), (0.2 * MM, BOARD_L - 1.0 * MM, BOARD_T - 2.0 * MM), GLUE),
    ]
    # the grain: streaks along the board on both faces, a different figure on each board (seen only)
    streaks = [(-19, 1.2, 0), (-11, 0.8, 1), (-6, 2.0, 0), (3, 1.0, 0), (9, 1.6, 1), (17, 0.9, 0)]
    for xs, w, light in streaks:
        xs = xs * side * MM
        for zf in (1, -1):
            shapes.append(box((xs, 0.0, zf * (BOARD_T / 2 + 0.05 * MM)), (w * MM, BOARD_L - 0.4 * MM, 0.1 * MM),
                              GRAIN_LIGHT if light else GRAIN_DARK))
    # WHY 75 g: a 160 x 50 x 12 mm oak board (0.75 g/cm3)
    return body(name, pos=(x, PANEL_XY[1], TOP + BOARD_Z), mass=0.075, friction=0.8, shapes=shapes,
                visual_only=tuple(range(1, len(shapes))))


BOARD_A = _board("board_a", -1)
BOARD_B = _board("board_b", +1)

# -- the clamps (bar frame: x along the bar towards the fixed jaw, origin at the bar's middle, bottom at z 0) ------
BAR_L, BAR_W = 200.0 * MM, 16.0 * MM
# WHY 26 mm (2026-11-10, sim trace): at 18 mm the 16 mm fingertips gripping the sliding jaw reached past its pad and
# pushed the board themselves, and the clamp held nothing once they let go.
JAW_X = 26.0 * MM  # each jaw's thickness along the bar
GRIP_BACK = 4.0 * MM  # the fingers grip the sliding jaw this far behind its middle
# WHY jaws wider than the bar (2026-11-10, sim trace): pushed from behind, the jaw of a clamp lying jaws-down could not
# be reached past the bar's end, which the fingers shoved instead. A hand grips the sliding jaw and slides it; at 24 mm
# against the bar's 16 the fingers close on the jaw only.
JAW_W = 24.0 * MM
JAW_H = 18.0 * MM  # how far a jaw stands off the bar
PAD_T = 4.0 * MM  # the pad: seen only; it squeezes the work through the clamp model's springs
HALF_SPAN = BOARD_W + GAP0 / 2  # from the panel's middle to each board's outer edge (glue line open)
FIXED_X = HALF_SPAN + 0.5 * MM  # the fixed jaw's pad face, from the bar's middle
OPEN = 15.0 * MM  # how far the sliding pad starts from board A's outer edge
SLIDE_X0 = -(HALF_SPAN + OPEN)  # the sliding pad face at the start


def _bar_shapes() -> list:
    """A clamp's bar and fixed jaw, jaws up in the bar's frame (a clamp jaws-down is the same turned over)."""
    zj = BAR_T + JAW_H / 2
    return [
        box((0.0, 0.0, BAR_T / 2), (BAR_L, BAR_W, BAR_T), BAR_STEEL),
        box((FIXED_X + RECESS + (JAW_X - RECESS) / 2, 0.0, zj), (JAW_X - RECESS, JAW_W, JAW_H), JAW_RED),
        box((FIXED_X + PAD_T / 2, 0.0, zj), (PAD_T, JAW_W, JAW_H), PAD),
        box((FIXED_X + (PAD_T + RECESS) / 2, 0.0, zj), (RECESS - PAD_T, JAW_W, JAW_H), JAW_RED),
    ]


def _jaw_shapes() -> list:
    """A sliding jaw, in its own frame (origin at its middle, JAW_Z above the bar's bottom): the part standing off the
    bar collides (what a hand grips); the pad on its +x face and the part wrapping the bar are seen only.

    WHY the wrap is seen only (2026-11-10, sim trace): on a fixed bar the jaw's collider round the bar was not filtered
    from the bar by its joint and the jaw would not slide."""
    up = BAR_T + JAW_H / 2 - JAW_Z  # the standing part's middle, in the jaw's frame
    # WHY its collider stands 1 mm clear of the bar (2026-11-10, sim traces): resting on the fixed bar's top face, the
    # jaw slid under a steady 3 N but not when gripped: the fingers' squeeze pressed it onto the bar and the contact's
    # friction held it against the arm. The joint carries it; the drawn jaw still meets the bar.
    front = JAW_X / 2 - RECESS  # the collider's front face
    return [
        box(((front - JAW_X / 2) / 2, 0.0, up + CLEAR / 2), (front + JAW_X / 2, JAW_W, JAW_H - CLEAR), JAW_RED),
        box(((JAW_X - PAD_T) / 2, 0.0, up), (PAD_T, JAW_W, JAW_H), PAD),
        box((-PAD_T / 2, 0.0, BAR_T / 2 - JAW_Z), (JAW_X - PAD_T, JAW_W, BAR_T + 2.0 * MM), JAW_RED),
        box(((front + JAW_X / 2 - PAD_T) / 2, 0.0, up), (RECESS - PAD_T, JAW_W, JAW_H), JAW_RED),
    ]


CLEAR = 1.0 * MM
# WHY the jaws collide only 8 mm behind their pad faces (2026-11-10, substep trace): 4 mm behind, a pad pressed into
# a board edge let the jaw's collider strike the board as well; the simulator's push-out kicked the light boards at up
# to 0.5 m/s and a released clamp rattled the panel between its jaws. The model's bite stops at 6 mm.
RECESS = 8.0 * MM
JAW_PAD_X = JAW_X / 2  # the sliding jaw's pad face in its frame: its +x face
JAW_Z = (BAR_T + JAW_H) / 2  # the sliding jaw's middle above the bar's bottom (it wraps the bar)
PAD_Z = BAR_T + JAW_H / 2  # the pads' middle above the bar's bottom


def _clamp(name: str, y: float, below: bool, base=None, rest_z=None):
    """A clamp across the panel at ``y`` (below: lying on the bench jaws up; above: jaws down), or hanging on the
    rack at ``base`` (x, y) with its bar's bottom face ``rest_z`` up (jaws down)."""
    if base is None:
        bx, by = PANEL_XY[0], PANEL_XY[1] + y
        z = 0.0 if below else BAR_T + BOARD_T + BAR_T  # jaws-down: the bar's frame turned over, its 'bottom' on top
    else:
        bx, by = base
        z = rest_z + BAR_T
    quat = None if below else (1.0, 0.0, 0.0, 0.0)  # jaws down: turned over about x
    # WHY the lower clamps' bars are fixed (2026-11-10, sim traces): free on the bench, every push shifted them and the
    # boards together, and the clamps lost or gained pressure on their own; a real bar clamp under a glue-up is pinned
    # by its weight and the boards'.
    bar = body(f"{name}_bar", pos=(bx, by, TOP + z), mass=None if below else 0.35, friction=0.8, quat=quat,
               shapes=_bar_shapes(), visual_only=(2, 3), inertia=(8e-5, 1.2e-3, 1.2e-3))
    jz = JAW_Z
    jx = SLIDE_X0 - JAW_PAD_X
    jaw_world_z = z + (jz if below else -jz)
    jaw = body(f"{name}_jaw", pos=(bx + jx, by, TOP + jaw_world_z), mass=0.15, friction=0.5, quat=quat,
               shapes=_jaw_shapes(), visual_only=(1, 2, 3), inertia=(2e-5, 2e-5, 2e-5))
    gluekit.add_slider(Slider(bar.name, jaw.name, anchor=(jx, 0.0, jz), axis=(1.0, 0.0, 0.0),
                              lower=-0.03, upper=0.06))
    return bar, jaw


LOW_Y = 55.0 * MM
RACK_XY = (0.20, 0.20)
RACK_H = 60.0 * MM
CLAMP1 = _clamp("clamp1", -LOW_Y, below=True)
CLAMP2 = _clamp("clamp2", +LOW_Y, below=True)
SPARE = _clamp("spare", 0.0, below=False, base=RACK_XY, rest_z=RACK_H)
RACK = body("rack", pos=(*RACK_XY, TOP), friction=0.6, shapes=[  # two posts under the bar, inside its jaws
    box((-0.03, 0.0, RACK_H / 2), (0.01, 0.03, RACK_H), WOOD),
    box((0.03, 0.0, RACK_H / 2), (0.01, 0.03, RACK_H), WOOD),
    box((0.0, 0.0, 2.5 * MM), (0.09, 0.05, 5.0 * MM), WOOD),
])

OBJECTS = [BOARD_A, BOARD_B, *CLAMP1, *CLAMP2, *SPARE, RACK]
gluekit.register(*OBJECTS)

CLAMPS = [Clamp(bar.name, jaw.name, FIXED_X, JAW_PAD_X, PAD_Z, PAD_Z - JAW_Z) for bar, jaw in (CLAMP1, CLAMP2, SPARE)]
MODEL = gluekit.add_model(Clamps(CLAMPS, [("board_a", -BOARD_W / 2, -1), ("board_b", BOARD_W / 2, +1)],
                                 half_len=BOARD_L / 2, pad_half_y=JAW_W / 2, half_thick=BOARD_T / 2,
                                 k_pad=3000.0))


# -- scoring -------------------------------------------------------------------------------------------------------
SCORE_OVER = "end"
RUN_LIMIT_STEPS = 1600
CONTROL_HZ = 20.0
GAP_OK = 0.2 * MM  # the glue line closed: at most this open where measured
OVERLAP_OK = 0.5 * MM  # ...and not ridden over (the edges' overlap, as the boards' inner faces measure it)
LEVEL_Z = 1.0 * MM  # the boards' faces level with each other
LEVEL_DEG = 2.0
GAP_AT = (-70.0 * MM, 0.0, 70.0 * MM)  # where the glue line is measured along it: both ends and the middle
SNUG = (1.0, 6.0)  # N: a clamp left on the work squeezing it this hard is snug
CRUSH = SNUG[1]  # N: any clamp squeezing harder than this...
CRUSH_S = 0.5  # ...for this long has squeezed the glue out of the joint (latched)
CLEAR = 10.0 * MM  # no part of either gripper this near a clamp or the panel: released
STILL = 0.01  # m/s


def gaps(objects) -> np.ndarray:
    """The glue line's opening (m) at both ends and the middle: from board A's inner face to board B's, at mid-
    thickness, along A's across-the-joint axis (negative: overlapping)."""
    a, b = objects["board_a"], objects["board_b"]
    out = []
    for y in GAP_AT:
        pa = a.to_world((BOARD_W / 2, y, 0.0))
        across = a.rotation()[:, 0]
        pb_local = b.to_local(pa)  # B's inner face at the same place along the joint
        pb = b.to_world((-BOARD_W / 2, pb_local[1], 0.0))
        out.append(float(np.dot(pb - pa, across)))
    return np.array(out)


def level(objects) -> bool:
    a, b = objects["board_a"], objects["board_b"]
    return bool(abs(a.pos[2] - b.pos[2]) <= LEVEL_Z and a.tilt_deg() <= LEVEL_DEG and b.tilt_deg() <= LEVEL_DEG)


def fingers_to(state, spec) -> float:
    """Nearest distance (m) from either gripper's hulls to a body's bounding box (in its own frame). inf offline."""
    pts = gluekit.finger_points()
    if not len(pts):
        return math.inf
    lo, hi = spec.bounds_local()
    local = (pts - state.pos) @ state.rotation()
    d = np.maximum(np.maximum(lo - local, local - hi), 0.0)
    return float(np.linalg.norm(d, axis=1).min())


def clamp_side(objects, c) -> int:
    """-1 for a clamp under the panel, +1 for one on top of it, 0 for one not across it."""
    bar = objects[c.bar]
    a = objects["board_a"]
    if abs(np.dot(bar.rotation()[:, 0], a.rotation()[:, 0])) < 0.9:  # its bar not across the joint
        return 0
    return -1 if bar.pos[2] < a.pos[2] else 1


def snug(c) -> bool:
    """Left on the work (no hand on its jaw), both pads bearing on the panel's outer edges, squeezing snug."""
    return bool(not math.isnan(c.pin) and c.engaged == {"jaw", "fixed"} and SNUG[0] <= c.force <= SNUG[1])


def pattern_ok(objects, clamps) -> bool:
    """Along the joint, the clamps alternate: below, above, below (no two neighbours on the same side)."""
    a = objects["board_a"]
    along = [(float(a.to_local(objects[c.bar].pos)[1]), clamp_side(objects, c)) for c in clamps]
    sides = [side for _, side in sorted(along)]
    return bool(len(sides) == 3 and 0 not in sides and all(s != t for s, t in zip(sides, sides[1:])))


_STATE: dict = {}


def _reset_state() -> None:
    _STATE.update(starved=False, crushing=0)


_reset_state()
gluekit.on_reset(_reset_state)


def _update(objects) -> tuple[float, bool]:
    clamps = MODEL.clamps
    _STATE["crushing"] = _STATE["crushing"] + 1 if max(c.force for c in clamps) > CRUSH else 0
    _STATE["starved"] |= _STATE["crushing"] >= round(CRUSH_S * CONTROL_HZ)
    if _STATE["starved"]:
        return 0.0, False
    g = gaps(objects)
    closed = (g <= GAP_OK) & (g >= -OVERLAP_OK)
    flat = level(objects)
    held = [c for c in clamps if snug(c)]
    if closed.all() and flat and len(held) == 3 and pattern_ok(objects, held):
        parts = [objects[n] for n in ("board_a", "board_b")] + [objects[c.bar] for c in clamps]
        specs = [BOARD_A, BOARD_B] + [s for s in OBJECTS if s.name in {c.bar for c in clamps} | {c.jaw for c in clamps}]
        released = all(fingers_to(objects[s.name], s) >= CLEAR for s in specs)
        still = all(p.resting(max_speed=STILL) for p in parts)
        if released and still:
            return 1.0, True
        return 0.5, False
    if closed.all() and flat and len(held) >= 2:
        return 0.5, False
    if (closed[0] or closed[-1]) and flat and len(held) >= 1:
        return 0.25, False
    return 0.0, False


def success(objects) -> bool:
    """The glue line closed at both ends and the middle, the boards level, three clamps snug and alternating below,
    above, below along the joint, released and still for 1 s; no clamp ever forced (latched)."""
    return _update(objects)[1]


def score(objects) -> float:
    """1.0 / 0.5 / 0.25 / 0 as in the README; a clamp forced past the limit latches 0."""
    return _update(objects)[0]


# -- scripted demo -------------------------------------------------------------------------------------------------
# Open loop, from the fixed layout. One arm at a time near the panel (two together ran into each other): the right
# arm closes the right-hand lower clamp; the left lays the spare clamp jaws-down across the panel's middle and closes
# it, then closes the left-hand lower clamp.
RHOME, LHOME = (0.16, -0.22, TOP + 0.15), (0.16, 0.22, TOP + 0.15)
PUSH_PAST = 26.0 * MM  # the jaw is sent this far past where its pad meets the work (the soft arm pushes ~4 N)
# Shortcut checks (GLUE_DEMO): "nospare" closes the two lower clamps only (the joint closes, the panel is not
# clamped on top); "crush" drives the right-hand clamp 25 mm further, as a heavy hand would.
DEMO_MODE = os.environ.get("GLUE_DEMO", "")
TIP_PAST_TCP = 13.0 * MM
GRIP_OPEN = FINGER_OPEN * 0.45  # 42 mm between the fingertips, round a 24 mm jaw
SPARE_DX = 0.0 * MM  # the spare is laid square over the panel, before anything is clamped (fixed pad 0.5 mm from B)
# WHY the spare goes on first (2026-11-10, sim trace): laid after a lower clamp had pushed the boards against its fixed
# jaw, the spare's fixed jaw either struck board B coming down or, laid clear of it, slid in once its jaw was closed and
# ratcheted itself to 10 N. On the unclamped panel its fixed jaw lines up with the others' (0.5 mm clear).
# WHY ease off before letting go (2026-11-10, sim trace): let go still pushing 12 mm past the jaw, the soft arm
# sprang forward as the fingers opened, dragged the jaw 7 mm on, and the clamp ratcheted itself to 12 N.
# WHY 2 mm behind the contact, not 2 mm past (2026-11-10, substep trace): let go still pushing, the arm sprang forward
# as the fingers opened and dragged the jaw on into the board; pulling back a little, any drag opens the jaw, which
# the ratchet holds.
EASE = PUSH_PAST + 2.0 * MM
TRAVEL = OPEN + 1.0 * MM + PUSH_PAST  # each sliding jaw: to board A's edge (the panel shifts about 1 mm), then on


def slide_jaw(y: float, jaw_x0: float, travel: float, tip_z: float) -> list:
    """Grip a clamp's sliding jaw (its middle at ``jaw_x0`` along x, from the panel's middle) from above, the
    fingertips ``tip_z`` above the table, slide it ``travel`` along the bar, let go and lift away."""
    x0 = PANEL_XY[0] + jaw_x0 - GRIP_BACK
    z = TOP + tip_z + TIP_PAST_TCP
    yy = PANEL_XY[1] + y
    return [
        Waypoint(2.0, (x0, yy, z + 0.05), GRIP_OPEN, "over the jaw", 0.0),
        Waypoint(1.2, (x0, yy, z), GRIP_OPEN, "round it", 0.0),
        Waypoint(1.6, (x0, yy, z), FINGER_CLOSED, "grip the jaw", 0.0),
        Waypoint(2.0, (x0 + travel, yy, z), FINGER_CLOSED, "slide it closed", 0.0),
        Waypoint(1.0, (x0 + travel, yy, z), FINGER_CLOSED, "hold", 0.0),
        Waypoint(1.0, (x0 + travel - EASE, yy, z), FINGER_CLOSED, "ease off", 0.0),
        Waypoint(1.6, (x0 + travel - EASE, yy, z), GRIP_OPEN, "let go", 0.0),
        Waypoint(1.0, (x0 + travel - EASE, yy, z + 0.05), GRIP_OPEN, "up", 0.0),
    ]


def place_spare() -> list:
    """Lift the spare clamp off its rack by the middle of its bar and lay it jaws-down across the panel's middle."""
    rx, ry = RACK_XY
    bar_mid = RACK_H + BAR_T / 2
    on_panel_mid = BAR_T + BOARD_T + BAR_T / 2
    up = 5.0 * MM  # the TCP above the bar's middle: the fingertips round its lower half
    return [
        Waypoint(2.0, (rx, ry, TOP + bar_mid + up + 0.05), FINGER_OPEN, "over the spare clamp", 0.0),
        Waypoint(1.2, (rx, ry, TOP + bar_mid + up), FINGER_OPEN, "down to its bar", 0.0),
        Waypoint(1.6, (rx, ry, TOP + bar_mid + up), FINGER_CLOSED, "grip", 0.0),
        Waypoint(1.2, (rx, ry, TOP + bar_mid + up + 0.03), FINGER_CLOSED, "lift it off", 0.0),
        Waypoint(2.5, (PANEL_XY[0] + SPARE_DX, PANEL_XY[1], TOP + on_panel_mid + up + 0.03), FINGER_CLOSED,
                 "over the panel", 0.0),
        Waypoint(1.0, (PANEL_XY[0] + SPARE_DX, PANEL_XY[1], TOP + on_panel_mid + up + 0.03), FINGER_CLOSED,
                 "steady", 0.0),
        Waypoint(2.0, (PANEL_XY[0] + SPARE_DX, PANEL_XY[1], TOP + on_panel_mid + up + 0.002), FINGER_CLOSED,
                 "lay it on", 0.0),
        Waypoint(1.6, (PANEL_XY[0] + SPARE_DX, PANEL_XY[1], TOP + on_panel_mid + up + 0.002), FINGER_OPEN,
                 "let go", 0.0),
        Waypoint(1.0, (PANEL_XY[0] + SPARE_DX, PANEL_XY[1], TOP + on_panel_mid + up + 0.05), FINGER_OPEN, "up", 0.0),
    ]


def _t(w) -> float:
    return sum(p.duration for p in w)


def demo():
    jaw0 = SLIDE_X0 - JAW_PAD_X  # a sliding jaw's middle, open, from its bar's middle
    low_tip = 10.0 * MM  # fingertips 10 mm up: round a lower clamp's jaw above its bar's foot
    top_tip = 6.0 * MM  # round the spare's jaw, below its bar (it lies jaws-down on the panel)
    # left: lay the spare across the middle and close it; right: close its lower clamp; left: close the other
    left = [] if DEMO_MODE == "nospare" else place_spare() + slide_jaw(0.0, jaw0 + SPARE_DX, TRAVEL, top_tip)
    left.append(Waypoint(1.5, LHOME, FINGER_CLOSED, "clear", 0.0))
    right = [Waypoint(_t(left), RHOME, FINGER_CLOSED, "wait", 0.0)]
    right_travel = TRAVEL + (25.0 * MM if DEMO_MODE == "crush" else 0.0)
    right += slide_jaw(-LOW_Y, jaw0, right_travel, low_tip) + [Waypoint(1.5, RHOME, FINGER_CLOSED, "clear", 0.0)]
    left.append(Waypoint(_t(right) - _t(left), LHOME, FINGER_CLOSED, "wait", 0.0))
    left += slide_jaw(+LOW_Y, jaw0, TRAVEL, low_tip) + [Waypoint(1.5, LHOME, FINGER_CLOSED, "clear", 0.0)]
    right.append(Waypoint(max(0.1, _t(left) - _t(right)), RHOME, FINGER_CLOSED, "wait", 0.0))
    # both clear while the clamps and panel are seen to stay put (success must hold 1 s)
    left.append(Waypoint(2.5, LHOME, FINGER_CLOSED, "done", 0.0))
    right.append(Waypoint(2.5, RHOME, FINGER_CLOSED, "done", 0.0))
    return {"right": right, "left": left}
