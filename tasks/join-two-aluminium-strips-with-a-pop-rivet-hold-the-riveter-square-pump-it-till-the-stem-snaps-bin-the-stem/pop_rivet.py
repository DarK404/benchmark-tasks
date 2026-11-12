"""On the station table: two aluminium strips lie stacked and clamped on a wooden block, a hole drilled through both;
a hand riveter stands nose down in its stand, a holder of blind (pop) rivets stands beside it, and a scrap tray.

The robot must rivet the strips together and put the riveter back: load a rivet into the riveter by its stem, set
it through the hole, hold the riveter square and pressed on the rivet's head while pumping its handles until the
stem snaps, empty the spent stem into the scrap tray, and stand the riveter back in its stand.

Run it:  inspect-robots run --embodiment yam_arena -E task=pop_rivet.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "..." --scorer task_score ...
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rivetkit  # noqa: E402
from rivetkit import Hinge, Riveting, body, box, prism12  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

rivetkit.install()
SCENE = "station"
TOP = TABLE_TOP[SCENE]
MM = 0.001
ALU = (0.78, 0.79, 0.81)
STEEL = (0.55, 0.56, 0.58)
RIVETER_RED = (0.70, 0.10, 0.08)  # the cast head, enamelled
GRIP_BLACK = (0.10, 0.10, 0.11)  # vinyl grips
ZINC = (0.70, 0.72, 0.74)  # the plated nosepiece and jaw case
WOOD = (0.55, 0.40, 0.25)
DARK = (0.08, 0.08, 0.09)
rivetkit.METALLIC.update({ALU: 0.3, STEEL: 0.35, ZINC: 0.25})

# -- the work: two strips on a block, a hole through both -------------------------------------------------------
WORK_XY = (0.27, 0.0)
BLOCK_H = 35.0 * MM
STRIP_T = 1.5 * MM
WORK_TOP = BLOCK_H + 2 * STRIP_T
STRIP = (30.0 * MM, 120.0 * MM)  # x, y
HOLE_R = 2.45 * MM  # a 4.9 mm hole for a 4.8 mm rivet
WORK = body("work", pos=(*WORK_XY, TOP), friction=0.6, shapes=[
    box((0.0, 0.0, BLOCK_H / 2), (0.08, 0.14, BLOCK_H), WOOD),
    box((0.0, 0.0, BLOCK_H + STRIP_T / 2), (*STRIP, STRIP_T), ALU),
    box((0.0, 0.0, BLOCK_H + 1.5 * STRIP_T), (STRIP[0] - 0.4 * MM, STRIP[1] - 0.4 * MM, STRIP_T), ALU),
    *prism12((0.0, 0.0, WORK_TOP + 0.05 * MM), HOLE_R, 0.2 * MM, DARK),  # the hole, seen from above
    box((0.0, 0.0, BLOCK_H - 6.0 * MM), (0.014, 0.014, 12.0 * MM), DARK),  # the bore under it in the block
    box((0.0, 0.05, WORK_TOP + 6.0 * MM), (0.04, 0.02, 12.0 * MM), STEEL),  # a clamp's jaw at the strips' end
    box((0.0, -0.05, WORK_TOP + 6.0 * MM), (0.04, 0.02, 12.0 * MM), STEEL),
], visual_only=(3, 4))

# -- the rivets, stem up in their holder ---------------------------------------------------------------------------
HOLDER_XY = (0.21, -0.05)
HOLDER_H = 20.0 * MM
# A 4.8 x 8 mm dome-head blind rivet, aluminium body, zinc-plated steel mandrel: 9.5 mm head, 8 mm shank, rated grip
# 2.0-3.5 mm (the two 1.5 mm strips make 3.0 mm), for a 4.9 mm hole; mandrel 2.7 mm across, 28 mm long overall.
# WHY 8 mm (2026-11-12, review): a 12 mm shank is the length for a 6-8 mm grip, not this 3 mm stack.
HEAD_T, HEAD_R, SHANK_R, SHANK_L = 1.5 * MM, 4.75 * MM, 2.4 * MM, 8.0 * MM
STEM_R, STEM_L = 1.35 * MM, 28.0 * MM - HEAD_T - SHANK_L  # the stem above the head (the rest runs through the shank)
GRIP_RANGE = (2.0 * MM, 3.5 * MM)
RIVET_DY = (-8.0 * MM, 8.0 * MM)
HOLDER = body("rivet_holder", pos=(*HOLDER_XY, TOP), friction=0.6, shapes=[
    box((0.0, 0.0, HOLDER_H / 2), (0.03, 0.04, HOLDER_H), WOOD)])
HIDDEN_Z = TOP - 0.30


def _rivet(k: int):
    """A 4.8 mm aluminium blind rivet (its frame at the top of its domed head): the head, the hollow shank, and its
    steel stem (mandrel) running up through it, with the ball end that upsets the shank in the shank's end."""
    head = (HOLDER_XY[0], HOLDER_XY[1] + RIVET_DY[k], TOP + HOLDER_H + HEAD_T)
    shapes = [
        *prism12((0.0, 0.0, -HEAD_T + 0.35 * MM), HEAD_R, 0.7 * MM, ALU),  # the domed head, in three steps
        *prism12((0.0, 0.0, -HEAD_T + 0.95 * MM), HEAD_R * 0.8, 0.5 * MM, ALU),
        *prism12((0.0, 0.0, -0.2 * MM), HEAD_R * 0.55, 0.4 * MM, ALU),
        *prism12((0.0, 0.0, -HEAD_T - SHANK_L / 2), SHANK_R, SHANK_L, ALU),
        *prism12((0.0, 0.0, -HEAD_T - SHANK_L - 0.6 * MM), STEM_R * 1.25, 1.2 * MM, STEEL),  # the mandrel's ball
    ]
    rivet = body(f"rivet{k}", pos=head, mass=1e-3, inertia=(1e-8, 1e-8, 1e-8), shapes=shapes,
                 visual_only=tuple(range(len(shapes))))
    stem_shapes = [
        *prism12((0.0, 0.0, STEM_L / 2), STEM_R, STEM_L, STEEL),
        *prism12((0.0, 0.0, 0.4 * MM), STEM_R * 1.12, 0.8 * MM, STEEL),  # the break neck's ring
    ]
    stem = body(f"stem{k}", pos=head, mass=1e-3, inertia=(1e-8, 1e-8, 1e-8), shapes=stem_shapes,
                visual_only=tuple(range(len(stem_shapes))))
    return rivet, stem, head


RIVETS = [_rivet(k) for k in range(2)]
TAIL = body("tail", pos=(*WORK_XY, HIDDEN_Z), mass=1e-3, inertia=(1e-8, 1e-8, 1e-8), shapes=[
    *prism12((0.0, 0.0, -2.5 * MM), 3.6 * MM, 5.0 * MM, ALU)], visual_only=tuple(range(6)))
for _b in [*[r for r, _, _ in RIVETS], *[s for _, s, _ in RIVETS], TAIL]:
    rivetkit.no_gravity(_b.name)

# -- the hand riveter, nose down in its holster --------------------------------------------------------------------
# A plier-type hand riveter. Its frame: origin at the nosepiece's tip, z up the nose. A hex nosepiece on a jaw case,
# a cast head, a fixed handle rising from the head at +x and a sprung handle pivoted on the head at -x, held open by
# its return spring against its stop. Both handles have wide ribbed vinyl grips.
# WHY wide grips (2026-11-10, sim traces): pinched across 18 mm handles, the riveter swung about the line between the
# two pads (nothing but friction within each pad's small contact resists that), 5 to 8 degrees as it was carried.
# The fingertips are 38 mm wide: 30 mm grips give each pad a wide contact, whose friction holds it.
NOSE_L = 25.0 * MM
HEAD_Z = (25.0 * MM, 50.0 * MM)
HINGE_AT = (-6.0 * MM, 0.0, 52.0 * MM)
FIXED_X = 18.0 * MM  # the fixed handle's middle
HANDLE_W = 12.0 * MM
HANDLE_L = 85.0 * MM
GRIP_Y = 30.0 * MM  # the grips' depth (along y)
GRIP_SPAN = (85.0 * MM, 130.0 * MM)  # the grips, up the handles from the nose's tip
OPEN_DEG, CLOSED_DEG = 25.0, -5.0  # the sprung handle from upright (open: its top away from the fixed one)
SPRING = 2.0  # N m/rad, towards 50 degrees: preloaded against the open stop
HOLSTER_XY = (0.18, -0.14)
HOLSTER_H = 40.0 * MM
REST_Z = HOLSTER_H - HEAD_Z[0]  # the nose's tip, the riveter standing in its holster


def _ribs(face_x, out_sign, z0, z1) -> list:
    """The grip's moulded ribs across a handle's outer face (seen only)."""
    return [box((face_x + out_sign * 0.15 * MM, 0.0, z), (0.3 * MM, GRIP_Y - 2.0 * MM, 1.5 * MM), DARK)
            for z in np.arange(z0 + 4.0 * MM, z1 - 2.0 * MM, 5.0 * MM)]


def _riveter_shapes() -> tuple[list, int]:
    hz = (HEAD_Z[0] + HEAD_Z[1]) / 2
    fixed_x0 = FIXED_X - HANDLE_W / 2
    solid = [
        box((0.0, 0.0, hz), (0.028, 0.022, HEAD_Z[1] - HEAD_Z[0]), RIVETER_RED),  # the cast head
        box((FIXED_X - 0.004, 0.0, HEAD_Z[1] + 0.012), (0.020, GRIP_Y, 0.024), RIVETER_RED),  # its arm to the handle
        box((FIXED_X, 0.0, (HEAD_Z[1] + 0.024 + GRIP_SPAN[1] + 0.012) / 2),
            (HANDLE_W, GRIP_Y, GRIP_SPAN[1] + 0.012 - HEAD_Z[1] - 0.024), GRIP_BLACK),  # the fixed handle's grip
    ]
    seen = [
        *_ribs(FIXED_X + HANDLE_W / 2, 1, *GRIP_SPAN),
        *prism12((0.0, 0.0, 0.006), 4.0 * MM, 0.012, ZINC),  # the nosepiece
        *prism12((0.0, 0.0, 0.015), 5.5 * MM, 0.006, ZINC),  # its hex
        *prism12((0.0, 0.0, 0.0215), 7.5 * MM, 0.007, ZINC),  # the jaw case
        *prism12((0.0, 0.0, 0.0002), 1.7 * MM, 0.0004, DARK),  # the nosepiece's bore
        *prism12((HINGE_AT[0], 0.0, HINGE_AT[2]), 3.0 * MM, 0.024, ZINC),  # the pivot pin's head (seen end on)
        box((fixed_x0 - 0.0005, 0.0, HEAD_Z[1] + 0.006), (0.002, 0.012, 0.004), ZINC),  # the handle stop
    ]
    return solid + seen, len(solid)


_RSH, _N_RSOLID = _riveter_shapes()
RIVETER = body("riveter", pos=(*HOLSTER_XY, TOP + REST_Z), mass=0.36, friction=1.0, com=(0.006, 0.0, 0.060),
               shapes=_RSH, visual_only=tuple(range(_N_RSOLID, len(_RSH))))
_open = math.radians(OPEN_DEG)


def _handle_shapes() -> list:
    """The sprung handle, in its own frame (origin on the pivot, z up the handle): a steel lever with a vinyl grip."""
    g0, g1 = GRIP_SPAN[0] - HINGE_AT[2], GRIP_SPAN[1] - HINGE_AT[2]
    return [
        box((0.0, 0.0, g0 / 2), (0.008, 0.012, g0), RIVETER_RED),  # the lever
        box((0.0, 0.0, (g0 + HANDLE_L) / 2), (HANDLE_W, GRIP_Y, HANDLE_L - g0), GRIP_BLACK),  # the grip
        *_ribs(-HANDLE_W / 2, -1, g0, g1),
    ]


_HSH = _handle_shapes()
HANDLE = body("handle", pos=(HOLSTER_XY[0] + HINGE_AT[0], HOLSTER_XY[1], TOP + REST_Z + HINGE_AT[2]), mass=0.06,
              friction=1.0, quat=(0.0, math.sin(-_open / 2), 0.0, math.cos(-_open / 2)), inertia=(4e-5, 4e-5, 2e-6),
              com=(0.0, 0.0, 0.045), shapes=_HSH, visual_only=tuple(range(2, len(_HSH))))
# joint angle 0 = open (its spawn pose); closing turns it positive about y; the spring pulls towards -25 (50 open)
rivetkit.add_hinge(Hinge("riveter", "handle", anchor=HINGE_AT, axis=(0.0, 1.0, 0.0), damping=0.02,
                         stiffness=SPRING, target=math.radians(OPEN_DEG - 50.0), lower=0.0,
                         upper=math.radians(OPEN_DEG - CLOSED_DEG)))
# a holster: a block with a socket for the nose, and a rim round the head so the riveter stands in it
STAND = body("holster", pos=(*HOLSTER_XY, TOP), friction=0.7, shapes=[
    box((0.0, 0.0, HOLSTER_H / 2), (0.06, 0.06, HOLSTER_H), WOOD),
    box((0.0, 0.027, HOLSTER_H + 4.0 * MM), (0.056, 0.006, 8.0 * MM), WOOD),
    box((0.0, -0.027, HOLSTER_H + 4.0 * MM), (0.056, 0.006, 8.0 * MM), WOOD),
    box((0.027, 0.0, HOLSTER_H + 4.0 * MM), (0.006, 0.048, 8.0 * MM), WOOD),
    box((-0.027, 0.0, HOLSTER_H + 4.0 * MM), (0.006, 0.048, 8.0 * MM), WOOD),
    *prism12((0.0, 0.0, HOLSTER_H + 0.1 * MM), 8.0 * MM, 0.2 * MM, DARK),  # the nose socket
], visual_only=tuple(range(5, 11)))

# -- the scrap tray ------------------------------------------------------------------------------------------------
TRAY_XY = (0.32, -0.11)
TRAY = (0.06, 0.045)
TRAY_FLOOR = 2.0 * MM
TRAY_WALL = 15.0 * MM
TRAY_BODY = body("tray", pos=(*TRAY_XY, TOP), friction=0.6, shapes=[
    box((0.0, 0.0, TRAY_FLOOR / 2), (*TRAY, TRAY_FLOOR), STEEL),
    box((TRAY[0] / 2, 0.0, TRAY_WALL / 2), (0.002, TRAY[1], TRAY_WALL), STEEL),
    box((-TRAY[0] / 2, 0.0, TRAY_WALL / 2), (0.002, TRAY[1], TRAY_WALL), STEEL),
    box((0.0, TRAY[1] / 2, TRAY_WALL / 2), (TRAY[0], 0.002, TRAY_WALL), STEEL),
    box((0.0, -TRAY[1] / 2, TRAY_WALL / 2), (TRAY[0], 0.002, TRAY_WALL), STEEL),
])

OBJECTS = [WORK, HOLDER, STAND, TRAY_BODY, RIVETER, HANDLE, *[r for r, _, _ in RIVETS], *[s for _, s, _ in RIVETS],
           TAIL]
rivetkit.register(*OBJECTS)
HOLE_TOL = 1.5 * MM  # the rivet's rounded end finds the 4.9 mm hole within this
RELEASE_DEG = 21.0  # the handles opened this wide (near their stop), the jaws let the spent stem go
MODEL = rivetkit.add_model(Riveting(
    "riveter", "handle", [(r.name, h) for r, _, h in RIVETS], [s.name for _, s, _ in RIVETS], "tail", WORK_XY,
    TOP + WORK_TOP, TOP + BLOCK_H, (WORK_XY[0] - STRIP[0] / 2, WORK_XY[0] + STRIP[0] / 2,
                                    WORK_XY[1] - STRIP[1] / 2, WORK_XY[1] + STRIP[1] / 2),
    HOLE_TOL, TOP + HOLDER_H, (HOLDER_XY[0] - 0.015, HOLDER_XY[0] + 0.015, HOLDER_XY[1] - 0.02, HOLDER_XY[1] + 0.02),
    (TRAY_XY[0] - TRAY[0] / 2, TRAY_XY[0] + TRAY[0] / 2, TRAY_XY[1] - TRAY[1] / 2, TRAY_XY[1] + TRAY[1] / 2),
    TOP + TRAY_FLOOR, TOP, HIDDEN_Z, head_t=HEAD_T, shank=SHANK_L, stem_len=STEM_L,
    release_angle=math.radians(RELEASE_DEG)))


# -- scoring -------------------------------------------------------------------------------------------------------
SCORE_OVER = "end"
RUN_LIMIT_STEPS = 1600
CLEAR = 10.0 * MM
STILL = 0.01


def fingers_to(state, spec) -> float:
    """Nearest distance (m) from either gripper's hulls to a body's bounding box (in its own frame). inf offline."""
    pts = rivetkit.finger_points()
    if not len(pts):
        return math.inf
    lo, hi = spec.bounds_local()
    local = (pts - state.pos) @ state.rotation()
    d = np.maximum(np.maximum(lo - local, local - hi), 0.0)
    return float(np.linalg.norm(d, axis=1).min())


def in_stand(riveter) -> bool:
    """Standing back in its holster: nose down, its head in the holster's rim."""
    off = np.hypot(*(np.asarray(riveter.pos[:2]) - HOLSTER_XY))
    return bool(off < 15.0 * MM and abs(riveter.pos[2] - (TOP + REST_Z)) < 4.0 * MM and riveter.tilt_deg() < 10.0)


def _update(objects) -> tuple[float, bool]:
    m = MODEL
    if m.spoilt:
        return 0.0, False
    if m.set_ok():
        r = objects["riveter"]
        released = min(fingers_to(r, RIVETER), fingers_to(objects["handle"], HANDLE)) >= CLEAR
        if m.stem_place == "tray" and in_stand(r) and released:
            return 1.0, bool(r.resting(max_speed=STILL))
        return 0.5, False
    if m.active is not None and m.state[m.active] == "inserted":
        return 0.25, False
    return 0.0, False


STILL_STEPS = 20  # 1 s at the 20 Hz control rate
_STATE: dict = {}


@rivetkit.on_reset
def _reset_state() -> None:
    _STATE.update(still=0)


_reset_state()


def success(objects) -> bool:
    """The rivet set square and tight through both strips, its spent stem in the scrap tray, the riveter back in its
    stand, released and still (held 1 s by the run); never pulled crooked or loose (latched). It also counts the steps
    the full-score state has held, for score()."""
    ok = _update(objects)[1]
    _STATE["still"] = _STATE["still"] + 1 if ok else 0
    return ok


def score(objects) -> float:
    """1.0 / 0.5 / 0.25 / 0 as in the README; a rivet pulled crooked or loose latches 0. 1.0 only once the full-score
    state has held for 1 s (counted by success(), which the run calls first every step); until then 0.5."""
    tier = _update(objects)[0]
    if tier == 1.0 and _STATE["still"] < STILL_STEPS:
        return 0.5
    return tier


# -- scripted demo -------------------------------------------------------------------------------------------------
# Open loop, from the fixed layout. The right hand takes the riveter by its handles (fingers closing along x across
# them); lowers its nose over a rivet's stem in the holder; carries it to the hole and pushes the rivet in; pumps the
# handles three times, keeping the riveter pressed on the head and still: as only the sprung handle moves, the hand
# follows it with half of each stroke so the fixed handle's finger stays put; over the scrap tray it opens the
# handles to let the stem drop out; then stands the riveter back in its stand.
RHOME, LHOME = (0.16, -0.22, TOP + 0.15), (0.16, 0.22, TOP + 0.15)
YAW = math.pi / 2  # the right fingers close along x
GRIP_Z = 108.0 * MM  # the TCP above the nose's tip, the fingertips 95 mm up the handles
FIXED_OUT = FIXED_X + HANDLE_W / 2  # the fixed handle's outer face (riveter frame)
HOLD_GAP, SQUEEZE_GAP, RELEASE_GAP = 54.0 * MM, 20.0 * MM, 64.0 * MM
# the handles' span at their closed stop, where the fingers meet them when squeezing to 20 mm: the hand is placed by
# where the fingers end up, not by the gap it asks for
# WHY (2026-11-10, sim trace): placed for a 20 mm gap, the fixed handle's finger stood 5 mm off it, the other finger
# shoved the riveter 7 to 15 degrees off square
CLOSED_SPAN = 31.2 * MM
PRESS = 2.0 * MM  # the nose sent this far below what it stands on
DEMO_MODE = os.environ.get("RIVET_DEMO", "")  # shortcut checks: "tilted" pumps with the hand 16 mm to one side,
# the riveter tipped about 8 degrees; "nobin" drops the spent stem on the work


def fingers(gap: float) -> float:
    return FINGER_OPEN * gap / 0.094


# where the held riveter hangs from the planned grip, and how far the arm sags under it (measured in sim 2026-11-10)
HANG = np.array([1.7, -5.4, -2.9]) * MM
# WHY re-measured (2026-11-12): with the 8 mm rivet (was 12) the riveter sits 4 mm lower on the strip and hangs 3.5 mm
# further to +y; from the trace of the first run.
HANG_ON = np.array([0.4, -1.9, -2.9]) * MM  # ...and where it sits, its rivet's end on the strip
# ...and after the squeezes, which slide it 12 mm along between the fingers (it is set down 18 mm off otherwise, on the
# holster's rim, and falls over)
HANG_AFTER = np.array([-10.4, -3.0, -2.6]) * MM
SAG_Z = 8.0 * MM


def tcp(nose, gap: float, held: bool = True, pinned: bool = False, on: bool = False, after: bool = False) -> tuple:
    """The TCP that holds the riveter, its nose's tip at ``nose`` (world), the fingers ``gap`` apart: the fixed
    handle's finger on its outer face. ``held``: allowing for the arm's sag and how it hangs free in the grip;
    ``pinned``: its nose on a rivet, where it cannot hang off square, so the hand goes straight above it."""
    p = np.array([nose[0] + FIXED_OUT - max(gap, CLOSED_SPAN) / 2, nose[1], nose[2] + GRIP_Z])
    if held:
        hang = (np.array([0.0, 0.0, HANG[2]]) if pinned else HANG_ON if on else HANG_AFTER if after else HANG)
        p = p + np.array([0.0, 0.0, SAG_Z]) - hang
    return tuple(p)


def _w(dur, pos, finger, label, yaw=YAW):
    return Waypoint(dur, tuple(pos), finger, label, yaw)


def _t(w) -> float:
    return sum(p.duration for p in w)


def demo():
    stand = np.array([*HOLSTER_XY, TOP + REST_Z])
    head0 = np.array(RIVETS[0][2])
    hole = np.array([*WORK_XY, TOP + WORK_TOP + HEAD_T])  # the nose's tip on a rivet's head in the hole
    lift = np.array([0.0, 0.0, 1.0])
    right = [
        _w(2.0, tcp(stand + 0.04 * lift, 0.094, False), FINGER_OPEN, "over the riveter"),
        _w(1.2, tcp(stand, 0.094, False), FINGER_OPEN, "round its grips"),
        _w(1.6, tcp(stand, HOLD_GAP, False), fingers(HOLD_GAP), "grip"),
        _w(1.2, tcp(stand + 0.035 * lift, HOLD_GAP), fingers(HOLD_GAP), "lift it out"),
        _w(2.0, tcp(head0 + (STEM_L + 0.008) * lift, HOLD_GAP), fingers(HOLD_GAP), "over a rivet"),
        _w(1.0, tcp(head0 + (STEM_L + 0.008) * lift, HOLD_GAP), fingers(HOLD_GAP), "steady"),
        _w(2.5, tcp(head0 - PRESS * lift, HOLD_GAP), fingers(HOLD_GAP), "down over its stem"),
        _w(1.0, tcp(head0 + (STEM_L + 0.008) * lift, HOLD_GAP), fingers(HOLD_GAP), "loaded"),
        _w(2.0, tcp(hole + 0.016 * lift, HOLD_GAP), fingers(HOLD_GAP), "over the hole"),
    ]
    on = hole - PRESS * lift
    lean = np.array([0.0, 0.016, 0.0]) if DEMO_MODE == "tilted" else np.zeros(3)
    # the rivet pushed in where the riveter hangs free; then, its nose pinned on the rivet, the hand goes straight
    # above the nose, squaring it up
    # WHY (2026-11-10, sim traces): held where it hangs free, it was squeezed 5 degrees off square; squared up before
    # the rivet was in, the nose missed the hole
    # finding the hole by feel: the rivet's end set down on the strip 8 mm short of the hole (supported there, the
    # riveter no longer swings in the hand), then slid across until it drops in
    # WHY (2026-11-10, sim traces): lowered straight onto the hole, the riveter swinging in the two-finger grip missed
    # the 4.9 mm hole by up to 7 mm
    beside = hole + np.array([-0.008, 0.0, SHANK_L])
    right += [
        _w(1.5, tcp(beside + 0.002 * lift, HOLD_GAP), fingers(HOLD_GAP), "over beside the hole"),
        _w(1.0, tcp(beside - PRESS * lift, HOLD_GAP, on=True), fingers(HOLD_GAP), "rivet on the strip"),
        _w(2.0, tcp(on, HOLD_GAP, on=True), fingers(HOLD_GAP), "slide it into the hole"),
        _w(0.8, tcp(on, HOLD_GAP, on=True), fingers(HOLD_GAP), "rivet in"),
        _w(1.0, tcp(on, HOLD_GAP, pinned=True), fingers(HOLD_GAP), "square it up"),
    ]
    for k in range(3):
        right += [
            _w(1.2, np.array(tcp(on, SQUEEZE_GAP, pinned=True)) + lean, fingers(SQUEEZE_GAP), f"squeeze {k + 1}"),
            _w(1.2, np.array(tcp(on, HOLD_GAP, pinned=True)) + lean, fingers(HOLD_GAP), f"open {k + 1}"),
        ]
    over_tray = np.array([*TRAY_XY, TOP + 0.030]) if DEMO_MODE != "nobin" else hole + 0.012 * lift
    right += [
        _w(1.0, tcp(hole + 0.012 * lift, HOLD_GAP, after=True), fingers(HOLD_GAP), "lift off"),
        _w(1.5, tcp(over_tray, HOLD_GAP, after=True), fingers(HOLD_GAP), "over the tray"),
        _w(1.2, tcp(over_tray, RELEASE_GAP, after=True), fingers(RELEASE_GAP), "drop the stem"),
        _w(1.0, tcp(over_tray, HOLD_GAP, after=True), fingers(HOLD_GAP), "close"),
        _w(2.0, tcp(stand + 0.035 * lift, HOLD_GAP, after=True), fingers(HOLD_GAP), "over the holster"),
        _w(1.0, tcp(stand + 0.035 * lift, HOLD_GAP, after=True), fingers(HOLD_GAP), "steady"),
        _w(1.5, tcp(stand + 0.0005 * lift, HOLD_GAP, after=True), fingers(HOLD_GAP), "stand it in"),
        _w(1.6, tcp(stand + 0.0005 * lift, 0.094, after=True), FINGER_OPEN, "let go"),
        # straight up clear of the handles' tops, then away, before going home
        # WHY (2026-11-10, sim trace): going home from just above them, the hand swept through the handles and knocked
        # the riveter out of its holster
        _w(1.2, tcp(stand + 0.06 * lift, 0.094, False), FINGER_OPEN, "up"),
        _w(1.0, tcp(stand + np.array([0.0, -0.07, 0.06]), 0.094, False), FINGER_OPEN, "away"),
        _w(1.5, RHOME, FINGER_OPEN, "home", 0.0),
    ]
    end_t = _t(right) + 2.5
    right.append(_w(2.5, RHOME, FINGER_OPEN, "done", 0.0))
    left = [_w(end_t, LHOME, FINGER_OPEN, "wait", 0.0)]
    return {"right": right, "left": left}
