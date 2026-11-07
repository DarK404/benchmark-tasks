"""On the station table: a sealed 450 ml round steel paint tin (80 mm across, 95 mm tall) stands on the midline
25 cm in front of the arm bases. Its lid is pressed into the rim and sits just below the rim's rolled edge, so it
cannot be pinched; between the lid's edge and the rim there is a narrow groove where a blade fits. A flat-blade
screwdriver stands upright, tip down, in a wooden stand 17 cm in front of the right arm, 3 cm in from it.

Run it:  inspect-robots run --embodiment yam_arena -E task=paint_tin.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "Open the paint tin: lever its lid off with the screwdriver and put the lid on the table." \\
           --scorer task_score ...
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tinkit  # noqa: E402
from tinkit import Seal, body, box, cyl, prism12, ring, tilted  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

tinkit.install()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
MM = 0.001

# -- the tin (tin frame: origin at the centre of its base, z up its axis) ------------------------------------------
TIN_XY = (0.25, -0.02)
TIN_R = 40.0 * MM  # outside radius of the body and of the rolled rim (bead): a 450 ml tin, 80 mm across
BEAD_IN = 38.0 * MM  # the bead's inside radius: the outer wall of the lid groove
BEAD_TOP = 95.0 * MM  # the highest point of the tin
CHANNEL_FLOOR = 85.0 * MM  # floor of the groove between the bead and the lid's seat
# WHY these radii (2026-11-07, resized to a larger tin; 80 mm across so the open gripper clears it by 7 mm a side when turning it): an offline 2D sweep of the blade's outline through the stroke (tracking errors of the
# real arm are 1-2 mm) picked the lid, seat, floor and blade reach that leave at least 2 mm everywhere: under the
# lid's edge, above the floor, clear of the seat's side, and 3.8 mm of lift at the end. Tighter fits (a 29 mm seat,
# 31 mm lid) jammed the blade against the seat or the lid's edge in the simulator.
SEAT_R = 26.5 * MM  # the raised seat the lid rests on (12-sided, to its flats); the lid overhangs it, over the groove
SEAT_TOP = 91.0 * MM
TIN_MASS = float(__import__("os").environ.get("TIN_MASS_DEV", "0.60"))  # kg: 450 ml of paint (about 0.55 kg) and the tin
SILVER = (0.74, 0.75, 0.77)
LID_GREY = (0.58, 0.60, 0.63)
BEAD_SEGMENTS = 24

# -- the lid (lid frame: origin at the centre of its underside, z up) ----------------------------------------------
# WHY a 12-sided prism of boxes (2026-11-07, sim trace): as a PhysX custom cylinder, the lid let the blade's end pass
# straight through it from below; box-on-box contacts hold.
LID_R = 30.5 * MM  # to its flats (61 mm across; corners 31.6 mm); overhangs the seat by 4 mm: the blade goes under it
LID_T = 2.0 * MM  # its top is 2 mm below the bead, so there is no edge to pinch
LID_MASS = 0.025

# -- the screwdriver (tool frame: origin at the handle's centre, z along the tool, tip at -z) --------------------
HANDLE_R, HANDLE_L = 12.5 * MM, 90.0 * MM
# WHY the 16 mm blade (2026-11-07, sim trace): with an 8 mm flat, the round 6 mm shaft behind it came down on the
# bead and pressed the whole tool, and the lid, down. A real 6 mm flat-blade screwdriver is forged flat for about
# 15 mm before the round shaft starts; the flat is modelled as one 1.2 mm plate.
SHAFT_R, SHAFT_L = 3.0 * MM, 42.0 * MM
BLADE_W, BLADE_T, BLADE_L = 6.0 * MM, 1.2 * MM, 16.0 * MM  # width (along the tool's y), thickness, length
TIP_Z = -(HANDLE_L / 2 + SHAFT_L + BLADE_L)  # the blade's end in the tool frame
TOOL_MASS = float(__import__("os").environ.get("TIN_TOOL_MASS", "0.12"))  # kg: a 6 x 75 mm flat-blade screwdriver
HANDLE_RED = (0.78, 0.16, 0.12)
STEEL = (0.62, 0.63, 0.66)

# -- the stand (fixed): a beech block with a square hole for the shaft --------------------------------------------
STAND_XY = (0.17, -0.275)
STAND_SIZE = (40.0 * MM, 40.0 * MM, 60.0 * MM)
STAND_HOLE = 8.0 * MM
BEECH = (0.80, 0.66, 0.46)


def _bead() -> list:
    r_mid, width = (TIN_R + BEAD_IN) / 2, TIN_R - BEAD_IN
    chord = 2 * TIN_R * math.tan(math.pi / BEAD_SEGMENTS) + 0.3 * MM  # overlapping, so the ring has no gaps
    h = BEAD_TOP - CHANNEL_FLOOR
    out = []
    for k in range(BEAD_SEGMENTS):
        a = 2 * math.pi * k / BEAD_SEGMENTS
        out.append(box((r_mid * math.cos(a), r_mid * math.sin(a), CHANNEL_FLOOR + h / 2), (chord, width, h), SILVER,
                       yaw=a + math.pi / 2))
    return out


# WHY a plug in a ring (2026-11-07, reviewing the scene in the viewer): a lid resting on a flat seat looked like "a
# cylinder on top of" the tin and slid about 4 mm. A real paint-tin lid has a plug ring pressed down inside a ring in
# the rim: that fit seals it and keeps it centred. Here the lid's panel rests on the rim's inner ring, and its plug
# hangs inside that ring with 1.5 mm to spare. The ring's outer face is where the old seat's was, so the blade's
# tuned path keeps its clearance.
INNER_RING = (23.0 * MM, 27.0 * MM)  # the rim's inner ring the lid sits on (inside, outside radius)
PLUG_RING = (20.0 * MM, 21.5 * MM)  # the lid's plug (inside, outside radius)
PLUG_DEPTH = 3.0 * MM
PAINT = (0.16, 0.42, 0.55)
LABEL = (0.93, 0.91, 0.86)
LABEL_STRIPE = (0.16, 0.42, 0.55)

_tin_collide = [
    cyl((0, 0, CHANNEL_FLOOR / 2), TIN_R, CHANNEL_FLOOR, SILVER),
    *ring(CHANNEL_FLOOR, SEAT_TOP, *INNER_RING, SILVER),
    *_bead(),
]
_tin_look = [
    cyl((0, 0, CHANNEL_FLOOR + 0.1 * MM), INNER_RING[0], 0.2 * MM, PAINT),  # paint seen inside the ring
    cyl((0, 0, 45.0 * MM), TIN_R + 0.3 * MM, 60.0 * MM, LABEL),  # paper label
    cyl((0, 0, 45.0 * MM), TIN_R + 0.4 * MM, 12.0 * MM, LABEL_STRIPE),
]
TIN = body(
    "tin",
    pos=(*TIN_XY, TOP),
    shapes=[*_tin_collide, *_tin_look],
    visual_only=tuple(range(len(_tin_collide), len(_tin_collide) + len(_tin_look))),
    mass=TIN_MASS,
    com=(0.0, 0.0, 0.040),
    friction=0.6,
    contact_offset=float(__import__("os").environ.get("TIN_CO", "2.0")) * MM,
    solver_iterations=int(__import__("os").environ.get("TIN_ITERS", "32")),
)

_lid_collide = [*prism12((0, 0, LID_T / 2), LID_R, LID_T, LID_GREY)] + ([] if __import__("os").environ.get("TIN_NO_PLUG") else ring(-PLUG_DEPTH, 0.0, *PLUG_RING, LID_GREY))
# WHY the rolled edge is only seen (2026-11-07, sim trace): as a 1.5 mm collision ring on top of the lid's edge it
# stood in the blade's way at the end of the stroke (the blade crosses the edge 0.7 mm above the panel), and neither
# pry freed a catch.
_lid_look = ring(LID_T, LID_T + 0.6 * MM, LID_R - 1.5 * MM, LID_R, SILVER)  # the rolled edge, seen only
# The lid's turn in its seat (a dev knob; 0). Turning it 15 degrees, to put a flat rather than a corner of the
# 12-sided lid at the pry, made the pry fail outright in the sim trace (2026-11-07), so it stays at 0.
LID_YAW = math.radians(float(__import__("os").environ.get("TIN_LID_YAW", "0")))
LID = body(
    "lid",
    pos=(*TIN_XY, TOP + SEAT_TOP),
    yaw=LID_YAW,
    shapes=[*_lid_collide, *_lid_look],
    visual_only=tuple(range(len(_lid_collide), len(_lid_collide) + len(_lid_look))),
    mass=LID_MASS,
    com=(0.0, 0.0, 0.6 * MM),
    friction=0.5,
    contact_offset=float(__import__("os").environ.get("TIN_CO", "2.0")) * MM,
    solver_iterations=int(__import__("os").environ.get("TIN_ITERS", "32")),
)

_handle_bottom = HANDLE_L / 2
SCREWDRIVER = body(
    "screwdriver",
    pos=(*STAND_XY, TOP + STAND_SIZE[2] + _handle_bottom),
    shapes=[
        cyl((0, 0, 0), HANDLE_R, HANDLE_L, HANDLE_RED),
        cyl((0, 0, -_handle_bottom - SHAFT_L / 2), SHAFT_R, SHAFT_L, STEEL),
        box((0, 0, -_handle_bottom - SHAFT_L - BLADE_L / 2), (BLADE_T, BLADE_W, BLADE_L), STEEL),
    ],
    mass=TOOL_MASS,
    com=(0.0, 0.0, -0.012),
    friction=0.9,
    contact_offset=float(__import__("os").environ.get("TIN_CO", "2.0")) * MM,
    solver_iterations=int(__import__("os").environ.get("TIN_ITERS", "32")),
)


def _stand_walls() -> list:
    sx, sy, sz = STAND_SIZE
    wall = (sx - STAND_HOLE) / 2
    off = STAND_HOLE / 2 + wall / 2
    return [
        box((off, 0, sz / 2), (wall, sy, sz), BEECH),
        box((-off, 0, sz / 2), (wall, sy, sz), BEECH),
        box((0, off, sz / 2), (STAND_HOLE, wall, sz), BEECH),
        box((0, -off, sz / 2), (STAND_HOLE, wall, sz), BEECH),
    ]


STAND = body("stand", pos=(*STAND_XY, TOP), shapes=_stand_walls(), friction=0.6)

OBJECTS = [STAND, TIN, LID, SCREWDRIVER]
tinkit.register(*OBJECTS)

# WHY 4.5 mm once the lid is free all round (2026-11-07): a real lid levered right out of the rim's channel rides on
# top of it, raised about its plug's depth. Down at 2 mm its top was flush with the bead, so the fingers had to reach
# into the 7.5 mm groove beside it with about 1 mm to spare, and the loose lid wandered 2-4 mm; at 4.5 mm its edge
# stands clear of the bead and the fingers can close on it from above the bead. Partly pried, it stays at 2 mm.
LID_FREE_PROUD = float(__import__("os").environ.get("TIN_FREE_PROUD", "4.5")) * MM
# WHY a centring pull on the free lid (2026-11-07, sim trace): riding up on the rim, its 3 mm plug comes clear of the
# rim's ring, and nothing kept it there: it drifted 16 mm across the bead in a few seconds. A real lid's plug sits in
# the top of the channel and stays put. 100 N/m, at most 0.3 N: about 1/150 of the fingers' squeeze. (A 5.8 mm plug
# that stays in the ring instead stopped either pry freeing a single catch.)
LID_FREE_CENTRE_K = float(__import__("os").environ.get("TIN_FREE_CENTRE_K", "100"))
SEAL = tinkit.add_seal(Seal(lid="lid", tin="tin", points=12, radius=25.0 * MM, lid_z=0.0, tin_z=SEAT_TOP,
                            hold=float(__import__("os").environ.get("TIN_HOLD_DEV", "0.5")), release=float(__import__("os").environ.get("TIN_RELEASE", "0.8")) * MM, pop_tilt=20.0,
                            proud=2.0 * MM, proud_free=LID_FREE_PROUD, centre_k=LID_FREE_CENTRE_K))


# -- scoring -------------------------------------------------------------------------------------------------------
# 1.0   the lid lies flat on the table (its lowest point within 5 mm of it, tilted under 10 degrees), at least 5 mm
#       clear of the tin, no part of either gripper within 20 mm of it, lid and tin still (under 1 cm/s) for 1 s
# 0.5   the lid is off the tin (its centre more than 40 mm, the tin's radius, from the tin's axis), not as for 1.0
# 0.25  the lid is on the tin but loose round at least half its rim (6 of the 12 seal catches freed)
# 0     anything else. Also 0, latched for the rest of the trial: the screwdriver's blade has not been in the groove
#       between the lid and the rim (the lid must be levered, not pulled or knocked off), the tin tilted over 15
#       degrees, the tin moved over 30 mm from its start, the tin or the lid fell below the table top.
SCORE_OVER = "end"
RUN_LIMIT_STEPS = 2400  # task.json run_limit: 2 minutes at 20 Hz; the demo takes about 48 s
LID_CORNER_R = LID_R / math.cos(math.radians(15))  # the 12-gon's corner radius: 31.6 mm
LID_CLEAR = 5.0 * MM  # lid's edge to the tin's side, on the table
LID_ON_TABLE = 5.0 * MM  # lid's lowest point above the table top
LID_FLAT_DEG = 10.0
FINGERS_CLEAR = 20.0 * MM
LID_OFF_R = TIN_R  # lid centre this far from the tin's axis: off the tin
LOOSE_CATCHES = 6
TIN_TILT_LIMIT = 15.0  # deg
TIN_MOVE_LIMIT = 30.0 * MM
FELL = 20.0 * MM  # an object's lowest point this far below the table top: it went over the edge
# The groove: between the lid's edge and the bead's inner wall, below the bead's top (tin frame). WHY 1 mm either
# side: the blade's end is measured at its centreline, and the lid can sit up to 1 mm off centre in the seat.
GROOVE_R = (LID_R - 1.0 * MM, BEAD_IN + 1.0 * MM)
GROOVE_Z = (CHANNEL_FLOOR - 1.0 * MM, BEAD_TOP)

_STATE: dict = {}


@tinkit.on_reset
def _reset_state() -> None:
    _STATE.update(tin0=None, tool_in_groove=False, violation=None)


_reset_state()


def fingers_to_lid(lid) -> float:
    """Nearest distance (m) from either arm's gripper hulls (palm and fingers) to the lid, taken as a disc of the 12-gon's corner
    radius (so it errs on the near side). inf offline or before the first reset."""
    pts = tinkit.finger_points()
    if not len(pts):
        return math.inf
    local = (pts - lid.pos) @ lid.rotation()
    radial = np.maximum(np.hypot(local[:, 0], local[:, 1]) - LID_CORNER_R, 0.0)
    vertical = np.maximum(np.maximum(local[:, 2] - LID_T, -local[:, 2]), 0.0)
    return float(np.hypot(radial, vertical).min())


def blade_in_groove(screwdriver, tin) -> bool:
    tip = tin.to_local(screwdriver.to_world((0.0, 0.0, TIP_Z)))
    r = math.hypot(tip[0], tip[1])
    return bool(GROOVE_R[0] <= r <= GROOVE_R[1] and GROOVE_Z[0] <= tip[2] <= GROOVE_Z[1])


def _update(objects) -> tuple[float, bool]:
    """(score, full score with everything still). Latches the tool use and the violations."""
    lid, tin, tool = objects["lid"], objects["tin"], objects["screwdriver"]
    if _STATE["tin0"] is None:
        _STATE["tin0"] = tin.pos.copy()
    if _STATE["violation"] is None:
        if tin.tilt_deg() > TIN_TILT_LIMIT:
            _STATE["violation"] = "tin tilted"
        elif np.hypot(*(tin.pos[:2] - _STATE["tin0"][:2])) > TIN_MOVE_LIMIT:
            _STATE["violation"] = "tin moved"
        elif min(tin.low[2], lid.low[2]) < TOP - FELL:
            _STATE["violation"] = "fell off the table"
    _STATE["tool_in_groove"] |= blade_in_groove(tool, tin)
    if _STATE["violation"] or not _STATE["tool_in_groove"]:
        return 0.0, False
    off_r = float(np.hypot(*(lid.to_world((0.0, 0.0, LID_T / 2))[:2] - tin.pos[:2])))
    flat = min(lid.tilt_deg(), 180.0 - lid.tilt_deg()) < LID_FLAT_DEG
    on_table = flat and lid.low[2] < TOP + LID_ON_TABLE and off_r >= TIN_R + LID_CORNER_R + LID_CLEAR
    if on_table and fingers_to_lid(lid) >= FINGERS_CLEAR:
        return 1.0, bool(lid.resting() and tin.resting())
    if off_r > LID_OFF_R:
        return 0.5, False
    return (0.25 if int((~SEAL.held).sum()) >= LOOSE_CATCHES else 0.0), False


def success(objects) -> bool:
    """The lid levered off and lying flat on the table clear of the tin, let go, everything still (held 1 s)."""
    return _update(objects)[1]


def score(objects) -> float:
    """1.0 / 0.5 / 0.25 / 0 as in the table above, from the state at the end; violations latch."""
    return _update(objects)[0]


# -- scripted demo -------------------------------------------------------------------------------------------------

GRASP_BELOW_TOP = 25.0 * MM  # the TCP this far below the handle's top end
# WHY (2026-11-07, sim trace): with the fingers closing horizontally across the tool, the lever stroke twists the
# tool about the pinch axis, which a two-pad grip resists only by friction: the tool turned 4 degrees in the grip
# just from its weight (8 mm at the tip). Rolled 90 degrees, the pads close in the lever's plane and take the
# lever torque as normal force.
GRIP_ROLL = math.radians(float(__import__("os").environ.get("TIN_GRIP_ROLL", "90")))
TCP_TO_TIP = HANDLE_L / 2 - GRASP_BELOW_TOP - TIP_Z  # along the tool


# Where the blade's end sits in the TCP frame once the tool is gripped (mm). Nominal: (0, 0, TCP_TO_TIP).
# WHY (2026-11-07, sim trace of the official stack): the handle seats 3.3 mm deeper between the pads than the
# nominal grasp height and leans 2 degrees, so the end sits at about (1.5, 3.6, 119.6) mm. This turns with the hand,
# so it is applied in the TCP frame; PRY_CORRECTION is what is left over in the world frame at the pry pose (arm sag
# under load). The demo stays open loop from the fixed layout: neither is read from the simulator at run time.
TIP_IN_TCP = np.array([float(v) for v in __import__("os").environ.get("TIN_TIP_IN_TCP", "1.5,3.6,119.6").split(",")]) * MM
PRY_CORRECTION = np.array([float(v) for v in __import__("os").environ.get("TIN_PRY_CORR", "3.3,4.5,8.6").split(",")]) * MM


def tool_dir(heading: float, pitch: float) -> np.ndarray:
    return np.array([math.cos(heading) * math.cos(pitch), math.sin(heading) * math.cos(pitch), -math.sin(pitch)])


# How far below the bead's edge the stroke's pivot is planned (a dev knob, 0 by default). WHY 0 (2026-11-07, sim
# trace): once the tool is calibrated (PRY_CORRECTION) and thin contacts get enough solver iterations, the blade
# rides on the bead's edge by itself (underside 78.2-78.7 mm on a 78 mm bead), so no extra press is needed.
PRESS = float(__import__("os").environ.get("TIN_PRESS", "0.0")) * MM


def _pry_geometry(theta: float, shift=(0.0, 0.0)):
    c = np.array([TIN_XY[0] + shift[0], TIN_XY[1] + shift[1], TOP])
    u = np.array([math.cos(theta), math.sin(theta), 0.0])
    heading = theta + math.pi
    fulcrum = c + BEAD_IN * u + np.array([0, 0, BEAD_TOP])
    return heading, fulcrum


# The second pry is made with the arm in a slightly different pose (the tin has moved in the turn), so it has its own
# measured correction (same method, sim trace 2026-11-07).
PRY2_CORRECTION = np.array([float(v) for v in __import__("os").environ.get("TIN_PRY2_CORR", "4.1,4.5,9.0").split(",")]) * MM


def _tcp_for(theta: float, p: float, s: float, press: float = 0.0, shift=(0.0, 0.0)) -> np.ndarray:
    """TCP position that puts the blade's underside on the bead's inner top edge (lowered by ``press``), its end
    ``s`` past that edge, the tool tipped ``p`` below the horizontal, pointing at the tin's centre."""
    heading, fulcrum = _pry_geometry(theta, shift)
    d = tool_dir(heading, p)
    n = np.cross(d, np.array([-math.sin(heading), math.cos(heading), 0.0]))  # normal above the blade
    n = n if n[2] > 0 else -n
    tip = fulcrum - np.array([0.0, 0.0, press]) + s * d + (BLADE_T / 2) * n
    corr = PRY2_CORRECTION if any(shift) else PRY_CORRECTION
    return tip - tinkit.tool_frame(heading, p, GRIP_ROLL) @ TIP_IN_TCP + corr


PRY_REACH = 9.0 * MM  # how far the blade's end reaches past the bead's inner edge


def pry_waypoints(theta: float, p0: float, p1: float, s: float = PRY_REACH, steps: int = 5, shift=(0.0, 0.0)) -> list:
    """Insert the blade at rim angle ``theta`` (rad, from the tin's centre) pointing at the centre, tipped ``p0``
    below the horizontal with its underside on the bead's inner top edge and its end ``s`` past that edge, then
    lever it about that edge down to ``p1``: the end swings in under the lid's edge and up against its underside.

    WHY p0 = 60 degrees: entered at 45, the blade's end met the lid's edge before getting under it. Entered
    steeply, the end stays outside the lid's edge until the stroke carries it in under the edge (2 mm clear in the
    offline sweep) and up against the lid's underside."""
    heading, _ = _pry_geometry(theta)
    pre = _tcp_for(theta, p0, s, shift=shift) - 0.035 * tool_dir(heading, p0)
    wps = [
        Waypoint(1.5, tuple(pre + np.array([0, 0, 0.03])), FINGER_CLOSED, "over the groove", tilted(heading, p0, GRIP_ROLL)),
        Waypoint(1.0, tuple(pre), FINGER_CLOSED, "line up", tilted(heading, p0, GRIP_ROLL)),
        Waypoint(1.2, tuple(_tcp_for(theta, p0, s, shift=shift)), FINGER_CLOSED, "insert", tilted(heading, p0, GRIP_ROLL)),
    ]
    for k in range(1, steps + 1):
        p = p0 + (p1 - p0) * k / steps
        wps.append(Waypoint(0.5, tuple(_tcp_for(theta, p, s, PRESS, shift)), FINGER_CLOSED, f"lever {k}",
                            tilted(heading, p, GRIP_ROLL)))
    return wps


def withdraw_waypoints(theta: float, p0: float, p1: float, s: float = PRY_REACH, shift=(0.0, 0.0)) -> list:
    """Back the blade out: keep the stroke's final tilt and slide the tool straight out, level, until the blade's end
    is clear of the rim, then rise.

    WHY (2026-11-07, sim trace): during the stroke the blade slides 4-6 mm deeper under the lid than planned. Turning
    the tool back up about the bead (even after dropping it 1.5 mm) swung that buried end up from 76 to 93 mm and
    flipped the lid. Sliding out level lets the lid settle back as the end comes out from under it."""
    heading, _ = _pry_geometry(theta)
    g = tilted(heading, p1, GRIP_ROLL)
    end = _tcp_for(theta, p1, s, PRESS, shift)
    out_dir = -np.array([math.cos(heading), math.sin(heading), 0.0])  # level, away from the tin's centre
    out = end + 0.035 * out_dir
    return [
        Waypoint(1.5, tuple(out), FINGER_CLOSED, "slide out", g),
        Waypoint(1.0, tuple(out + np.array([0, 0, 0.05])), FINGER_CLOSED, "clear", g),
    ]


EASE_DOWN = math.radians(float(__import__("os").environ.get("TIN_EASE_DOWN", "4")))


def ease_blade_down(theta: float, p1: float, s: float = PRY_REACH, shift=(0.0, 0.0)) -> Waypoint:
    """After the last lever, turn the tool back up a few degrees about the bead: the blade's end drops about 1 mm,
    out from under the freed lid, which then settles level on its support.

    WHY (2026-11-07, sim trace): propped on the blade, the free lid stayed tilted 2-3 degrees across the fingers, and
    squeezed at its 2 mm edge it turned over between them and slipped out. Drawing the blade back radially instead
    dragged the lid 3-4 mm off the fingers' aim."""
    heading, _ = _pry_geometry(theta)
    p = p1 + EASE_DOWN
    return Waypoint(0.8, tuple(_tcp_for(theta, p, s, PRESS, shift)), FINGER_CLOSED, "ease the blade down",
                    tilted(heading, p, GRIP_ROLL))


# -- the left arm turns the tin ------------------------------------------------------------------------------------

TURN_FROM, TURN_TO = math.radians(-120), math.radians(30)  # grip yaw: the tin turns 150 degrees anticlockwise
TURN_GRIP_Z = 80.0 * MM  # TCP above the table: the pads hold the body just below the rim
# WHY 80 (2026-11-07, sim trace): gripped at mid-height (55 mm), 40 mm of tin stood between the fingers and their
# inner faces near the palm caught the rim and tipped the tin over.
# WHY (2026-11-07, sim trace with the 1.8 s grip): the turn leaves the tin's centre 9.7 mm forward and 4.8 mm left
# of where it started (the gripper closes slightly off its axis). The second pry aims at where the tin ends up; open
# loop. (With the old 0.8 s grip the fingers shoved the tin 15 mm left as they closed.)
TURN_SHIFT = tuple(float(v) * MM for v in __import__("os").environ.get("TIN_TURN_SHIFT", "9.7,4.8").split(","))


# WHY (2026-11-07, sim trace): over the tin the left arm's TCP sits 5.5 mm off in y and 1 mm low; with only 4 mm
# between the open pads and the 86 mm tin, its fingers hit the tin on the way down and knocked it over. Its targets
# over the tin are shifted the other way. Open loop.
LEFT_CORRECTION = np.array([float(v) for v in __import__("os").environ.get("TIN_LEFT_CORR", "0.3,5.5,1.1").split(",")]) * MM


# Measured at the grip itself (TCP 80 mm up, fingers round the tin, before they close): the arm sits further off
# there than 16 cm above the tin, so the turn has its own correction (sim trace 2026-11-07).
TURN_CORRECTION = np.array([float(v) for v in __import__("os").environ.get("TIN_TURN_CORR", "3.2,7.3,2.9").split(",")]) * MM


# WHY the left arm holds the tin throughout (2026-11-07, sim trace): each time it let go after turning the tin, the
# tin was left 5-15 mm somewhere new and the second pry missed. Held, the tin stays where the gripper puts it, as a
# person steadies a tin with one hand while prying with the other. The pads sit 75 and 45 degrees round the rim
# from the pry, clear of the screwdriver.
def _grip_xyz():
    x, y = np.array(TIN_XY) + TURN_CORRECTION[:2]
    return x, y, TOP + TURN_GRIP_Z + TURN_CORRECTION[2]


def grip_tin() -> list:
    x, y, z = _grip_xyz()
    a = tilted(TURN_FROM, math.pi / 2)
    return [
        Waypoint(1.8, (x, y, TOP + 0.16), FINGER_OPEN, "over the tin", a),
        Waypoint(0.5, (x, y, TOP + 0.16), FINGER_OPEN, "line up on the tin", a),
        Waypoint(1.5, (x, y, z), FINGER_OPEN, "down round the tin", a),
        Waypoint(GRIP_S, (x, y, z), FINGER_CLOSED, "grip the tin", a),
    ]


def turn_held_tin() -> list:
    x, y, z = _grip_xyz()
    return [Waypoint(3.0, (x, y, z), FINGER_CLOSED, "turn the tin", tilted(TURN_TO, math.pi / 2))]


def let_go_of_tin() -> list:
    x, y, z = _grip_xyz()
    b = tilted(TURN_TO, math.pi / 2)
    return [
        Waypoint(GRIP_S, (x, y, z), FINGER_OPEN, "let go of the tin", b),
        Waypoint(1.2, (x, y, TOP + 0.16), FINGER_OPEN, "up from the tin", b),
    ]


# -- the left arm lifts the loose lid off and puts it on the table -----------------------------------------------

# WHY pinch while the blade is still in (2026-11-07, sim trace): sliding the blade out from under the loose lid
# dragged it 18 mm across and left it perched on the rim, tilted 7 degrees, where any touch of a finger knocked it
# down. So the left hand takes the lid first, as a person does with the free hand, and the screwdriver comes out
# from under the held lid (see LID_GRIP_AT).
# While the blade is held in, the free lid turns slowly on the rim; when the fingers close it has reached -154
# degrees (the same in every run, 2026-11-07 trace with the 1.8 s grip), so its flats face 24, 54, .. 174 degrees.
# WHY across flats: aimed 11 degrees off a flat, a finger came down on a corner of the 12-sided lid and tipped it
# (sim trace 2026-11-07). The fingers close along world 144/324 degrees (grip yaw 54.2; the left arm reaches yaws up
# to about 61 here). Aimed at the lid's centre while the blade is held in (the centring pull keeps the free lid on
# the tin; world, mm; open loop, from the trace).
PINCH_XY = tuple(float(v) * MM for v in __import__("os").environ.get("TIN_PINCH_XY", "260.3,-13.7").split(","))
LID_PINCH_YAW = math.radians(float(__import__("os").environ.get("TIN_PINCH_YAW", "54.2")))
# Fingertips 0.2 mm above the bead's top: the pads cover the whole edge of the free lid (underside 95.5, top 97.5).
# WHY not higher (2026-11-07, sim trace): with the tips level with its underside, the first finger to touch the
# floating lid tipped it and it slipped out under the other finger.
LID_PINCH_TIPS = float(__import__("os").environ.get("TIN_PINCH_TIPS", "95.2")) * MM  # fingertips above the table
HOVER = 10.0 * MM  # the fingers wait this far above the pinch for 1 s first, so tracking error can be read there
# Measured at the 10 mm hover with the blade held in (sim trace 2026-11-07), from where the fingertips' collision
# hulls actually are (simulated link poses): correction = -(error there). WHY not from the joint readings: forward
# kinematics of the measured joints put the gripper about 5 mm from where the simulated gripper link is here.
PINCH_CORRECTION = np.array([float(v) for v in __import__("os").environ.get("TIN_PINCH_CORR", "-0.25,-1.3,0.1").split(",")]) * MM
LID_DOWN_XY = (0.24, 0.13)  # where the lid is put down: 15 cm left of the tin

# WHY a 70 mm gap: the fingertips (4 mm thick, 15.4 mm wide in the URDF collision hulls) come down above the bead,
# each inner face 4.5 mm outside the lid's flat, so the lid can sit up to 4 mm off the aim. Fully open (94 mm),
# the fingers came down outside the tin and picked it up.
PINCH_GAP = float(__import__("os").environ.get("TIN_PINCH_GAP", "70")) * MM
FINGER_PINCH_OPEN = -PINCH_GAP / 2
# WHY a gentle squeeze (2026-11-07, sim trace): closed fully, the fingers press with their full 50 N, and the lid,
# tilted 2 degrees across them, turned over between the pads and shot out. Commanded to a gap 2 mm narrower than the
# lid's 61 mm across its flats, each finger's servo (5000 N/m) presses about 5 N: still 20 times the lid's weight.
LID_SQUEEZE = float(__import__("os").environ.get("TIN_LID_SQUEEZE", "1.0")) * MM
FINGER_ON_LID = -(LID_R - LID_SQUEEZE)
# Once the lid is clear of the tin the fingers close to 4 mm under its width (about 20 N): held at 5 N between two
# points, it swung about the line between the pads as it was lowered (54 degrees) and slid 7 cm when let go.
FINGER_FIRM_ON_LID = -(LID_R - 4.0 * MM)


def lift_lid_off() -> list:
    x, y = np.array(PINCH_XY) + PINCH_CORRECTION[:2]
    tcp = TOP + LID_PINCH_TIPS + PAD_TIP_PAST_TCP + PINCH_CORRECTION[2]
    g = tilted(LID_PINCH_YAW, math.pi / 2)
    dx, dy = LID_DOWN_XY
    put = TOP + 3.0 * MM + PAD_TIP_PAST_TCP  # fingertips 3 mm above the table: the lid drops 2 mm when let go
    return [
        Waypoint(2.0, (x, y, TOP + 0.16), FINGER_PINCH_OPEN, "over the lid", g),
        Waypoint(1.5, (x, y, tcp + HOVER), FINGER_PINCH_OPEN, "down over the groove", g),
        Waypoint(1.0, (x, y, tcp + HOVER), FINGER_PINCH_OPEN, "line up on the groove", g),
        Waypoint(1.0, (x, y, tcp), FINGER_PINCH_OPEN, "fingers into the groove", g),
        Waypoint(0.8, (x, y, tcp), FINGER_ON_LID, "pinch the lid", g),
        Waypoint(1.5, (x, y, tcp + 0.06), FINGER_ON_LID, "lift the lid out", g),
        Waypoint(0.5, (x, y, tcp + 0.06), FINGER_FIRM_ON_LID, "grip the lid firmly", g),
        Waypoint(2.0, (dx, dy, tcp + 0.06), FINGER_FIRM_ON_LID, "carry the lid", g),
        Waypoint(2.5, (dx, dy, put), FINGER_FIRM_ON_LID, "lower the lid", g),
        Waypoint(0.8, (dx, dy, put), FINGER_OPEN, "let go of the lid", g),
        Waypoint(1.0, (dx, dy, TOP + 0.12), FINGER_OPEN, "up from the lid", g),
    ]


# WHY the blade comes out only once the lid is pinched (2026-11-07, sim trace): during the strokes the blade's end
# works about 6 mm deeper than planned, to 22 mm from the lid's centre, against the outside of the lid's plug ring.
# Lifting the lid then hooked the plug's lower edge under the blade, and the held screwdriver pinned the lid. With
# the lid held by the fingers, the blade slides out without dragging it (loose, the lid was dragged 18 mm).
LID_GRIP_AT = 5  # lift_lid_off's waypoints up to and including "pinch the lid"


def wait(seconds: float, last: Waypoint, label: str = "wait") -> Waypoint:
    return Waypoint(seconds, last.pos, last.finger, label, last.yaw)


def _t(wps) -> float:
    return sum(w.duration for w in wps)


# WHY 1.8 s to open or close the gripper fully (2026-11-07, first `inspect-robots run` of the demo): the run's
# delta-limit guard lets the gripper command change by at most 5% of its range per step (a full stroke in 1 s, or
# 1.5 s along the smooth-step profile). The 0.6-0.8 s strokes were clamped and lagged; the tin came out of the turn
# a little differently, the open-loop pinch missed the lid, and the demo scored 0.25. The dev harness steps the env
# directly, without the guard, so it had not shown this. test_the_demo_passes_the_run_guards checks every step.
GRIP_S = 1.8


def pick_tool() -> list:
    sx, sy = STAND_XY
    grasp_z = TOP + STAND_SIZE[2] + HANDLE_L - GRASP_BELOW_TOP
    down = tilted(0.0, math.pi / 2, GRIP_ROLL)
    return [
        Waypoint(1.5, (sx, sy, grasp_z + 0.05), FINGER_OPEN, "over the handle", down),
        Waypoint(1.2, (sx, sy, grasp_z), FINGER_OPEN, "down to the handle", down),
        Waypoint(GRIP_S, (sx, sy, grasp_z), FINGER_CLOSED, "grip", down),
        Waypoint(1.5, (sx, sy, grasp_z + 0.075), FINGER_CLOSED, "lift out", down),
    ]


def pinch_lid_attempt() -> list:
    """Dev check for a loophole: the left gripper reaches into the groove, pinches the sealed lid across its flats and
    pulls straight up, without the screwdriver."""
    x, y = TIN_XY
    tips_z = TOP + SEAT_TOP + 1.0 * MM  # fingertips 1 mm above the lid's underside, beside its edge
    tcp_z = tips_z + PAD_TIP_PAST_TCP
    yaw = 0.0  # fingers close along world y: across two flats of the lid (flats face 0, 30, 60.. degrees)
    down = tilted(yaw, math.pi / 2)
    return [
        Waypoint(2.0, (x, y, tcp_z + 0.06), FINGER_OPEN, "over the lid", down),
        Waypoint(2.0, (x, y, tcp_z), FINGER_OPEN, "fingers into the groove", down),
        Waypoint(1.0, (x, y, tcp_z), FINGER_CLOSED, "pinch", down),
        Waypoint(2.0, (x, y, tcp_z + 0.06), FINGER_CLOSED, "pull up", down),
        Waypoint(1.0, (x, y, tcp_z + 0.06), FINGER_CLOSED, "hold", down),
    ]


PRY_AT, PRY_P0, PRY_P1 = math.radians(255), math.radians(60), math.radians(float(__import__("os").environ.get("TIN_P1", "10")))


def demo():
    if __import__("os").environ.get("TIN_DEMO") == "pinch":
        return {"left": pinch_lid_attempt()}
    right = pick_tool() + pry_waypoints(PRY_AT, PRY_P0, PRY_P1) + withdraw_waypoints(PRY_AT, PRY_P0, PRY_P1)
    # WHY park (watched in the viewer): waiting with the blade pointed at the tin put it a few cm from the left
    # gripper while that arm turned the tin. Offset found by an offline IK search: the farthest reachable spot with
    # the same grip (blade end 13 cm from the tin's axis).
    park = Waypoint(1.5, tuple(np.array(right[-1].pos) + np.array([0.08, -0.06, 0.0])), FINGER_CLOSED,
                    "park the screwdriver", right[-1].yaw)
    right.append(park)
    # The left arm comes in slowly (in 1 s its wrist spun at 176 deg/s) and waits; it takes hold of the tin only to
    # turn it. WHY not hold the tin during the pries (sim trace 2026-11-07): held still, by the gripper or a holder
    # ring, the tin cannot give the millimetre it gives on the table under the blade, and the blade butts against
    # the lid's edge instead of getting under it.
    left = [Waypoint(2.5, (0.24, 0.12, TOP + 0.20), FINGER_OPEN, "ready", tilted(TURN_FROM, math.pi / 2))]
    left.append(wait(_t(right) - _t(left), left[-1], "wait for the first pry"))
    left += grip_tin() + turn_held_tin() + let_go_of_tin()
    right.append(wait(_t(left) - _t(right), right[-1], "wait for the turn"))
    right += pry_waypoints(PRY_AT, PRY_P0, PRY_P1, shift=TURN_SHIFT)
    right.append(ease_blade_down(PRY_AT, PRY_P1, shift=TURN_SHIFT))
    left.append(wait(_t(right) - _t(left), left[-1], "wait for the second pry"))
    lift = lift_lid_off()
    right.append(wait(_t(left) + _t(lift[:LID_GRIP_AT]) - _t(right), right[-1], "hold the blade in"))
    out = withdraw_waypoints(PRY_AT, PRY_P0, PRY_P1, shift=TURN_SHIFT)
    left += lift[:LID_GRIP_AT] + [wait(_t(out[:1]) + 0.3, lift[LID_GRIP_AT - 1], "hold the lid")] + lift[LID_GRIP_AT:]
    right += out
    right.append(Waypoint(1.5, park.pos, FINGER_CLOSED, "park again", park.yaw))
    if _t(left) > _t(right):
        right.append(wait(_t(left) - _t(right), right[-1], "hold the screwdriver"))
    else:
        left.append(wait(_t(right) - _t(left), left[-1], "done"))
    return {"right": right, "left": left}
