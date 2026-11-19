"""On the station table: a Dell PowerEdge R640 1U rack server (the 8 x 2.5 in drive chassis), unplugged, lies on the
bench with its front towards the robot. Its eight 2.5 in drive bays are in two rows of four; all hold drive carriers
but one, the upper row's second bay from the robot's right. A hot-swap drive carrier with a drive in it lies on the
table to the server's right, its handle open.

The robot must install the carrier in the empty bay as Dell's service manual says: with the handle open, slide the
carrier into the bay until it reaches the backplane, then close the handle to lock it. Closing the handle swings its
hook in behind the bay frame's lip and cams the carrier the last 6 mm home, pushing the drive's connector onto the
backplane's; the handle's free end then snaps under a sprung catch. A carrier only pushed in by hand stops against
the connector, unmated and not held.

Run it:  inspect-robots run --embodiment yam_arena -E task=drive_carrier.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "Install the drive carrier in the server's empty drive bay and lock its handle." ...
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import carrierkit  # noqa: E402
from carrierkit import Hinge, Spring, body, box  # noqa: E402
from yam_arena.analysis import box_signed_distance  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

carrierkit.install()
SCENE = "station"
TOP = TABLE_TOP[SCENE]
MM = 0.001
STEEL = (0.18, 0.19, 0.20)  # the chassis and the bay frames
SILVER = (0.70, 0.71, 0.73)  # the carriers' rails and the drives
BEZEL = (0.12, 0.12, 0.13)  # the carriers' fronts and handles
BLUE = (0.10, 0.35, 0.75)  # the handles' release buttons
COPPER = (0.80, 0.55, 0.25)  # the connectors' contacts
GREEN = (0.10, 0.35, 0.15)  # the backplane board
LID = (0.52, 0.53, 0.55)  # the top cover (grey sheet steel)
carrierkit.METALLIC.update({SILVER: 0.35})

# -- sizes ------------------------------------------------------------------------------------------------------------
# The bay frame: x is measured from the front of a seated carrier (+x into the server), y across the server (env y),
# z up from the table. The seated carrier's front is at env x SF, 1 mm behind the chassis's front (X_FRONT); the
# posts of the bay frame between the bays stand 8.3 mm proud of the chassis, their fronts (X_CAGE) at env x 0.19.
X_CAGE = -9.3 * MM
X_FRONT = -1.0 * MM
# The rack: the server is installed in a 2-post 19 in rack standing on the table, in its second rack unit (44.45 mm
# up), with a 1U unit under it and another directly over it, their fronts and the posts' ears at the cage's front.
RACK_U = 44.45 * MM
RACK_Z = RACK_U  # the server's underside, above the table
UNIT_H = 43.7 * MM  # the neighbouring 1U units' height
POST_Y0, POST_Y1 = 225.0 * MM, 241.0 * MM  # each post's flange, either side of the 450 mm opening
POST_H = 0.30  # the rack section's height (about 6U)
SF = 0.19 - X_CAGE
SRV_W, SRV_H, SRV_D = 0.434, 42.8 * MM, 0.26  # the R640's width and height (Dell); the front 26 cm is modelled
BAY_W, BAY_H, COL_W = 71.0 * MM, 18.0 * MM, 8.0 * MM
PITCH = BAY_W + COL_W
ROW_Z = {"lower": 3.0 * MM, "upper": 23.0 * MM}  # each row's bay floor above the table
SHELF = (21.0 * MM, 23.0 * MM)
COVER_Z0 = 41.0 * MM
COLS = [(k - 1.5) * PITCH for k in range(4)]  # bay centres in y
EMPTY = (1, "upper")  # the empty bay: the upper row's second from the robot's right (y -39.5 mm)

# The carrier: a Dell 2.5 in hot-swap carrier with a drive in it: 70 x 15 mm, 106 mm long, 0.17 kg (our estimates).
# Its frame: origin at the middle of its front face, +x into the bay.
CW, CH, CL = 70.0 * MM, 15.0 * MM, 106.0 * MM
FRONT_T = 6.0 * MM
C_MASS = 0.17
# Its SAS connector: a slot in its rear end holding two sprung contact shoes (top and bottom) that close on the
# backplane's blade. The blade touches the shoes 6 mm before the carrier is home (MATE) and the last 6 mm are its
# mating stroke: up its 15-degree lead (4.5 mm) and 1.5 mm along its flat.
SLOT_X0, SLOT_Y, SLOT_Z = CL - 22.0 * MM, 16.0 * MM, 3.0 * MM
MATE = 6.0 * MM
BLADE_TIP_X = 88.0 * MM  # the blade's tip, in a seated carrier's frame
BACKPLANE_X = CL + 0.4 * MM  # the backplane's face: a carrier pressed home stops against it 0.4 mm on
BLADE_T, BLADE_W = 2.4 * MM, 30.0 * MM
RIDGE_DEG = 15.0
RIDGE_L = (BLADE_T / 2) / math.tan(math.radians(RIDGE_DEG))
TIP_PLATE = 0.2 * MM  # the plates that make its sloped faces: thin, so where they meet at the tip it is only
# 0.4 mm thick, less than the shoes' gap
SHOE_X1 = BLADE_TIP_X + MATE
SHOE_X0, SHOE_T, SHOE_GAP = SHOE_X1 - 5.0 * MM, 1.5 * MM, 0.6 * MM
SHOE_MASS = 0.004
SHOE_K, SHOE_TRAVEL = 5000.0, 2.0 * MM
SHOE_PRELOAD = float(os.environ.get("DC_PRELOAD", "6"))  # N; on the blade's flat each shoe presses 10.5 N
CONTACT_MU = float(os.environ.get("DC_MU", "0.3"))

# The handle: a 62 mm lever hinged on a vertical pin at the carrier front's -y corner; closed it lies across the
# front, open (90 degrees) it points out towards the robot. A hook 8 mm out from the pin on its other side swings,
# as the handle closes, through the frame column's notch and bears on the back of the column's lip, drawing the
# carrier in. A light torsion spring opens it when the catch is released.
H_LEN, H_T, H_H = 62.0 * MM, 6.0 * MM, 10.0 * MM
H_MASS = 0.015
PIVOT = np.array([-5.3 * MM, -CW / 2 + 3.5 * MM, 0.0])  # in the carrier's frame
HOOK_L, HOOK_W, HOOK_H = 8.0 * MM, 2.0 * MM, 6.0 * MM
OPEN_DEG = 90.0
H_SPRING = 0.006  # N m/rad: the handle's opening spring (0.01 N m when shut)
# the frame column on a bay's -y side: a lip at the cage's front, the hook's notch behind it, solid behind that
LIP_X1 = PIVOT[0] - HOOK_W / 2 - 0.1 * MM  # the lip's rear face: the hook comes to rest 0.1 mm off it
NOTCH_X1 = 4.0 * MM
NOTCH_D = 6.0 * MM  # the notch's depth into the column from the bay's side
CHAMFER = 1.5 * MM  # the lead-ins on the frame posts at the bay's mouth (45 degrees)
CHAMFER_Z = 2.5 * MM  # ... and along the chassis front's shelf and ceiling edges
# the catch on the carrier's front: a sprung tab whose sloped nose the closing handle lifts, and which then drops in
# front of the handle's free end
TAB_Y0, TAB_Y1 = PIVOT[1] + 54.0 * MM, PIVOT[1] + 60.0 * MM  # over the handle's last 8 mm
TAB_MASS, TAB_K, TAB_PRELOAD, TAB_TRAVEL = 0.002, 300.0, 0.5, 2.5 * MM
NOSE_X1 = PIVOT[0] - H_T / 2 - 0.3 * MM  # the nose's square rear face, just in front of the shut handle
SLOPE_Z0 = H_H / 2 - 1.0 * MM  # its sloped face ends 1 mm below the handle's top (lifting the tab 1 mm) ...
NOSE_Z0 = SLOPE_Z0 + 0.3 * MM  # ... and its underside is 0.3 mm higher, so the handle's edge cannot catch there
TAB_Z1 = 6.8 * MM
SLOPE_RUN = 5.0 * MM  # the nose's sloped face runs 5 mm forward as it rises 2.8 mm (30 degrees)


def bay_origin(col: int, row: str) -> np.ndarray:
    """A seated carrier's frame origin (the middle of its front face; it rests on the bay floor) for bay (col, row)."""
    return np.array([SF, COLS[col], TOP + RACK_Z + ROW_Z[row] + CH / 2])


def _plate(p0, p1, t, colour, y, w, outward) -> object:
    """A thin box whose ``outward`` face lies on the line p0-p1 in the x-z plane (a sloped face), ``w`` long in y."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    if np.dot(n, outward) < 0:
        n = -n
    c = (p0 + p1) / 2 - n * t / 2
    return box((c[0], y, c[1]), (float(np.linalg.norm(d)), w, t), colour, tilt_y=-math.atan2(d[1], d[0]))


def _plate_xy(p0, p1, t, colour, z, h, outward) -> object:
    """The same in the x-y plane (a vertical sloped face), ``h`` tall."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    if np.dot(n, outward) < 0:
        n = -n
    c = (p0 + p1) / 2 - n * t / 2
    return box((c[0], c[1], z), (float(np.linalg.norm(d)), t, h), colour, yaw=math.atan2(d[1], d[0]))


def _carrier_shapes(connector=True) -> list:
    out = [box((FRONT_T / 2, 0.0, 0.0), (FRONT_T, CW, CH), BEZEL),
           box(((FRONT_T + SLOT_X0) / 2, 0.0, 0.0), (SLOT_X0 - FRONT_T, CW, CH), SILVER)]
    rl, xc = CL - SLOT_X0, (SLOT_X0 + CL) / 2
    if connector:
        for s in (-1.0, 1.0):
            out.append(box((xc, s * (SLOT_Y + CW / 2) / 2, 0.0), (rl, CW / 2 - SLOT_Y, CH), SILVER))
            out.append(box((xc, 0.0, s * (SLOT_Z + CH / 2) / 2), (rl, 2 * SLOT_Y, CH / 2 - SLOT_Z), SILVER))
    else:
        out.append(box((xc, 0.0, 0.0), (rl, CW, CH), SILVER))
    return out


def _server_shapes() -> list:
    """The server's front in the bay frame (its body origin on the table at x SF, y 0)."""
    out = []
    xf, cage = X_FRONT, 2 * PITCH + COL_W / 2
    # the chassis floor, the shelf between the rows, the top cover, the sides beyond the cage, the backplane
    out.append(box(((xf + SRV_D) / 2, 0.0, ROW_Z["lower"] / 2), (SRV_D - xf, SRV_W, ROW_Z["lower"]), STEEL))
    out.append(box(((xf + CHAMFER_Z + BACKPLANE_X) / 2, 0.0, sum(SHELF) / 2), (BACKPLANE_X - xf - CHAMFER_Z, 2 * cage,
                                                                          SHELF[1] - SHELF[0]), STEEL))
    out.append(box(((xf + CHAMFER_Z + SRV_D) / 2, 0.0, (COVER_Z0 + SRV_H) / 2), (SRV_D - xf - CHAMFER_Z, SRV_W,
                                                                               SRV_H - COVER_Z0), LID))
    out.append(box(((xf + SRV_D) / 2, 0.0, (COVER_Z0 + SRV_H) / 2 + 0.6 * MM), (SRV_D - xf, SRV_W,
                                                                               SRV_H - COVER_Z0 - 1.2 * MM), LID))
    # the lead-ins along the upper bays' floor and ceiling at the mouth
    out.append(_plate((xf, SHELF[1] - CHAMFER_Z), (xf + CHAMFER_Z, SHELF[1]), 0.8 * MM, STEEL, 0.0, 2 * cage, (-1, 1)))
    out.append(_plate((xf, COVER_Z0 + CHAMFER_Z), (xf + CHAMFER_Z, COVER_Z0), 0.8 * MM, STEEL, 0.0, 2 * cage, (-1, -1)))
    for s in (-1.0, 1.0):
        out.append(box(((xf + SRV_D) / 2, s * (cage + SRV_W / 2) / 2, (ROW_Z["lower"] + COVER_Z0) / 2),
                       (SRV_D - xf, SRV_W / 2 - cage, COVER_Z0 - ROW_Z["lower"]), STEEL))
    out.append(box((BACKPLANE_X + 3 * MM, 0.0, (ROW_Z["lower"] + COVER_Z0) / 2), (6 * MM, 2 * cage,
                                                                                 COVER_Z0 - ROW_Z["lower"]), GREEN))
    # the frame columns between and beside the bays
    for row in ("lower", "upper"):
        z0 = ROW_Z[row]
        zc = z0 + BAY_H / 2
        for k in range(5):
            yc = (k - 2) * PITCH
            y0, y1 = yc - COL_W / 2, yc + COL_W / 2
            out.append(box(((NOTCH_X1 + BACKPLANE_X) / 2, yc, zc), (BACKPLANE_X - NOTCH_X1, COL_W, BAY_H), STEEL))
            wall = COL_W - NOTCH_D  # the notch is open to the bay on the column's +y side
            out.append(box(((LIP_X1 + NOTCH_X1) / 2, y0 + wall / 2, zc), (NOTCH_X1 - LIP_X1, wall, BAY_H), STEEL))
            # the lip: its front corners chamfered (the bay's lead-in)
            xp = X_CAGE
            out.append(box(((xp + CHAMFER + LIP_X1) / 2, yc, zc), (LIP_X1 - xp - CHAMFER, COL_W, BAY_H), STEEL))
            out.append(box((xp + CHAMFER / 2, yc, zc), (CHAMFER, COL_W - 2 * CHAMFER, BAY_H), STEEL))
            out.append(_plate_xy((xp, y1 - CHAMFER), (xp + CHAMFER, y1), 0.8 * MM, STEEL, zc, BAY_H, (-1, 1)))
            out.append(_plate_xy((xp, y0 + CHAMFER), (xp + CHAMFER, y0), 0.8 * MM, STEEL, zc, BAY_H, (-1, -1)))
    # the filled bays' carriers (fixed, their handles shut) and the empty bay's connector blade
    for row in ("lower", "upper"):
        for col in range(4):
            o = bay_origin(col, row) - np.array([SF, 0.0, TOP + RACK_Z])  # (the server body's frame)
            if (col, row) == EMPTY:
                x0 = BLADE_TIP_X + RIDGE_L
                out.append(box(((x0 + BACKPLANE_X) / 2, o[1], o[2]), (BACKPLANE_X - x0, BLADE_W, BLADE_T), COPPER))
                for s in (-1.0, 1.0):  # its tapered tip
                    out.append(_plate((BLADE_TIP_X, o[2]), (x0, o[2] + s * BLADE_T / 2), TIP_PLATE, COPPER, o[1],
                                      BLADE_W, (-1, s)))
                continue
            for sh in _carrier_shapes(connector=False):
                out.append(box(tuple(np.asarray(sh.centre) + o), sh.size, sh.color))
            out.append(box(tuple(o + np.array([PIVOT[0], PIVOT[1] + H_LEN / 2, 0.0])), (H_T, H_LEN, H_H), BEZEL))
            out.append(box(tuple(o + np.array([PIVOT[0] - H_T / 2 - 0.3 * MM, PIVOT[1] + H_LEN - 10 * MM, 0.0])),
                           (0.6 * MM, 12 * MM, 6 * MM), BLUE))
    return out


SERVER_SHAPES = _server_shapes()
SERVER = body("server", pos=(SF, 0.0, TOP + RACK_Z), shapes=SERVER_SHAPES, mass=17.0, friction=float(os.environ.get("DC_SMU", "0.15")), contact_offset=0.0005,
              visual_only=tuple(i for i, s in enumerate(SERVER_SHAPES) if s.color == BLUE))
carrierkit.kinematic(SERVER.name)  # it lies on the bench; nothing moves it


def _rack_shapes() -> list:
    """The rack round the server (its frame: x from the seated carrier's front, y across, z up from the table): the 1U
    unit under it, the 1U unit over it, the two posts, and every unit's ears on the posts."""
    xf, d = X_CAGE, 0.26
    out = [box(((xf + d) / 2 + xf / 2, 0.0, UNIT_H / 2), (d, SRV_W, UNIT_H), STEEL),  # the unit under it
           box(((xf + d) / 2 + xf / 2, 0.0, 2 * RACK_U + UNIT_H / 2), (d, SRV_W, UNIT_H), LID)]  # the unit over it
    for s in (-1.0, 1.0):
        yc = s * (POST_Y0 + POST_Y1) / 2
        out.append(box((xf + 15 * MM, yc, POST_H / 2), (30 * MM, POST_Y1 - POST_Y0, POST_H), STEEL))  # a post
        out.append(box((xf - 1.0 * MM, s * (SRV_W / 2 + POST_Y1) / 2, 1.5 * RACK_U),
                       (2.0 * MM, POST_Y1 - SRV_W / 2, 3 * RACK_U - 1.0 * MM), STEEL))  # the three units' ears
    return out


RACK = body("rack", pos=(SF, 0.0, TOP), shapes=_rack_shapes(), friction=0.3, contact_offset=0.0005)

# the carrier: on the table to the server's right, front towards the robot, its handle open
C_START = np.array([0.075, -0.13, TOP + CH / 2])  # on the table in front of the rack, between its posts
CARRIER = body("carrier", pos=tuple(C_START), shapes=_carrier_shapes(), mass=C_MASS, friction=0.4,
               contact_offset=0.0005, solver_iterations=32, com=(CL / 2, 0.0, 0.0))


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _shoe(sign: float):
    """One of the connector's contact shoes (sign +1: the top one), at rest on its stop."""
    local = np.array([(SHOE_X0 + SHOE_X1) / 2, 0.0, sign * (SHOE_GAP / 2 + SHOE_T / 2)])
    return body(f"contact_{'a' if sign > 0 else 'b'}", pos=tuple(C_START + local),
                shapes=[box((0.0, 0.0, 0.0), (SHOE_X1 - SHOE_X0, 2 * SLOT_Y - 1.0 * MM, SHOE_T), COPPER)],
                mass=SHOE_MASS, friction=CONTACT_MU, contact_offset=0.0005, solver_iterations=32)


SHOE_A, SHOE_B = _shoe(+1.0), _shoe(-1.0)

# the handle (its frame at the hinge pin, axes as the carrier's when shut), spawned open
_R_OPEN = _rz(math.radians(OPEN_DEG))
_PIVOT_W = C_START + PIVOT
_handle_shapes = [box((0.0, H_LEN / 2, 0.0), (H_T, H_LEN, H_H), BEZEL),
                  box((0.0, -(HOOK_L + HOOK_W / 2) / 2, 0.0), (HOOK_W, HOOK_L + HOOK_W / 2, HOOK_H), BEZEL),
                  box((-H_T / 2 - 0.3 * MM, H_LEN - 10 * MM, 0.0), (0.6 * MM, 12 * MM, 6 * MM), BLUE)]
HANDLE = body("handle", pos=tuple(_PIVOT_W), shapes=_handle_shapes, mass=H_MASS, friction=0.3,
              contact_offset=0.0005, solver_iterations=32, quat=carrierkit.matrix_to_quat_rel(_R_OPEN),
              com=(0.0, H_LEN / 2, 0.0), visual_only=(2,))

# the catch tab, on the carrier (its frame: the carrier's axes, origin over the nose's rear face)
_TAB_LOCAL = np.array([NOSE_X1, (TAB_Y0 + TAB_Y1) / 2, 0.0])
_tw = TAB_Y1 - TAB_Y0
_tab_shapes = [box((-NOSE_X1 / 2, 0.0, (H_H / 2 + 0.3 * MM + TAB_Z1) / 2), (-NOSE_X1, _tw, TAB_Z1 - H_H / 2 - 0.3 * MM),
                   BEZEL),  # over the shut handle's end, back to the carrier's front
               box((-1.0 * MM, 0.0, (NOSE_Z0 + TAB_Z1) / 2), (2.0 * MM, _tw, TAB_Z1 - NOSE_Z0), BEZEL),  # the nose
               _plate((-2.0 * MM - SLOPE_RUN, TAB_Z1), (-2.0 * MM, SLOPE_Z0), 0.8 * MM, BEZEL, 0.0, _tw, (-1, -1))]  # slope
TAB = body("catch_tab", pos=tuple(C_START + _TAB_LOCAL), shapes=_tab_shapes, mass=TAB_MASS, friction=0.3,
           contact_offset=0.0005, solver_iterations=32)

OBJECTS = [SERVER, RACK, CARRIER, SHOE_A, SHOE_B, HANDLE, TAB]
carrierkit.register(*OBJECTS)
_spre = SHOE_PRELOAD / SHOE_K
carrierkit.add_spring(Spring(CARRIER.name, SHOE_A.name, axis="Z", stiffness=SHOE_K, damping=2.0, target=-_spre,
                             lower=0.0, upper=SHOE_TRAVEL))
carrierkit.add_spring(Spring(CARRIER.name, SHOE_B.name, axis="Z", stiffness=SHOE_K, damping=2.0, target=+_spre,
                             lower=-SHOE_TRAVEL, upper=0.0))
carrierkit.add_hinge(Hinge(CARRIER.name, HANDLE.name, anchor=tuple(PIVOT), axis=(0.0, 0.0, 1.0), damping=0.002,
                           stiffness=H_SPRING, target=0.0, lower=-math.radians(OPEN_DEG), upper=0.0))
_tpre = TAB_PRELOAD / TAB_K
carrierkit.add_spring(Spring(CARRIER.name, TAB.name, axis="Z", stiffness=TAB_K, damping=0.3, target=-_tpre,
                             lower=0.0, upper=TAB_TRAVEL))

# -- scoring -------------------------------------------------------------------------------------------------------
# 1.0   the carrier home in the empty bay (within 0.7 mm of home along the bay, so its connector is on the blade's
#       flat: fully mated; within 1.5 mm across it and 2 degrees square), its handle shut (within 3 degrees) under the
#       catch (the tab dropped back within 0.5 mm of rest); no gripper within 1 cm; still; for 1 s
# 0.5   home, the handle not locked (or a hand on it, or not yet still)
# 0.25  in the bay with its connector on the blade, not home
# 0     anything else
SCORE_OVER = "end"
RUN_LIMIT_STEPS = 1200
STILL_STEPS = 20
CLEAR_DIST = 10.0 * MM
SEAT_TOL, SQUARE_DEG, HANDLE_DEG, TAB_TOL = 0.7 * MM, 2.0, 3.0, 0.5 * MM
_STATE: dict = {}


@carrierkit.on_reset
def _reset_state() -> None:
    _STATE.clear()
    _STATE.update(still=0)


_reset_state()


def to_bay(st) -> np.ndarray:
    return np.asarray(st.pos) - bay_origin(*EMPTY)


def _yaw_deg(r) -> float:
    return math.degrees(abs(math.atan2(r[1, 0], r[0, 0])))


def seated(st) -> bool:
    d = to_bay(st)
    return bool(abs(d[0]) <= SEAT_TOL and abs(d[1]) <= 1.5 * MM and abs(d[2]) <= 1.5 * MM
                and _yaw_deg(st.rotation()) <= SQUARE_DEG and st.tilt_deg() <= SQUARE_DEG)


def engaged(st) -> bool:
    """In the bay with the blade's tip past the shoes' rear ends."""
    d = to_bay(st)
    return bool(abs(d[1]) <= 3 * MM and abs(d[2]) <= 3 * MM and -MATE < d[0] <= SEAT_TOL)


def handle_deg(objects) -> float:
    """How far (degrees) the handle stands open from shut across the carrier's front."""
    rel = objects[CARRIER.name].rotation().T @ objects[HANDLE.name].rotation()
    return math.degrees(math.atan2(rel[1, 0], rel[0, 0]))


def tab_lift(objects) -> float:
    c = objects[CARRIER.name]
    return float(c.to_local(objects[TAB.name].pos)[2] - _TAB_LOCAL[2])


def latched(objects) -> bool:
    return abs(handle_deg(objects)) <= HANDLE_DEG and abs(tab_lift(objects)) <= TAB_TOL


def hands_clear(objects) -> bool:
    pts = carrierkit.finger_points()
    if not len(pts):
        return True
    st = objects[CARRIER.name]
    centre = st.to_world((CL / 2 - 8 * MM, 0.0, 0.0))
    return box_signed_distance(pts, centre, st.rotation(), np.array([CL + 16 * MM, CW, CH])).min() >= CLEAR_DIST


def _update(objects) -> tuple[float, bool]:
    st = objects[CARRIER.name]
    if seated(st):
        ang = handle_deg(objects)
        handle_still = abs(ang - _STATE.get("handle", ang)) < 0.2
        _STATE["handle"] = ang
        if latched(objects) and st.resting() and handle_still and hands_clear(objects):
            return 1.0, True
        return 0.5, False
    return (0.25 if engaged(st) else 0.0), False


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
# Open loop, from the fixed layout only. The right hand takes the carrier across its sides (fingers closing along y),
# carries it level in line with the empty bay, slides its rear half in and lets go short of the bay frame; pushes its
# front with the closed fingertips until it stops on the connector; then sweeps the open handle shut by pushing it
# near its middle along its arc: the hook cams the carrier home and the handle's end snaps under the catch.
TIP = 13.0 * MM
RHOME, LHOME = (0.14, -0.22, TOP + 0.17), (0.14, 0.22, TOP + 0.17)
GRIP_X = 48.0 * MM  # the fingers take the carrier this far behind its front, near its centre of mass (53 mm), so
# it is carried level; let go, its front end tips down onto the frame until it is pushed in
GRIP_DZ = float(os.environ.get("DC_GRIPZ", "-2")) * MM  # the pad tips this far below the carrier's mid-height
OPEN_W = -(CW / 2 + 10.0 * MM)
HOLD = -(CW / 2 - 4.0 * MM)
CARRY_Z = TOP + 0.09
CAL = np.array([float(v) for v in os.environ.get("DC_CAL", "0.6,2.6,1.7").split(",")]) * MM
LIFT = 1.2 * MM  # carried this far above the bay floor
YAW_FIX = math.radians(float(os.environ.get("DC_YAW", "0")))  # the hand turned so the carrier, which turns
# about 1 degree in the grip as it is lifted, goes into the bay square
RELEASE_X = float(os.environ.get("DC_REL", "-16")) * MM  # the pad tips' x (bay frame) when it lets go: short of the frame posts
FINGER_R = float(os.environ.get("DC_FR", "8")) * MM  # the closed fingers' face, ahead of the TCP
PUSH_PAST = float(os.environ.get("DC_PAST", "18")) * MM  # the push aims this far past the connector's touch point
PUSH_Y = 5.0 * MM  # across the carrier's front, clear of the open handle and the catch
PUSH_DZ = -4.0 * MM  # the fingertips this far below its mid-height
SWEEP_R = 40.0 * MM  # the handle is pushed this far out from its pin
HANDLE_DZ = -5.0 * MM  # ... with the fingertips this far below its mid-height (it is 10 mm tall)
PRESS = float(os.environ.get("DC_PRESS", "15")) * MM  # ... and the targets lie this far into it (the soft arm)
PUSH_PITCH = math.radians(float(os.environ.get("DC_PITCH", "45")))  # the pushing fingertips tilted this far down:
# straight down, the fingers would strike the front of the unit over the server before the carrier is in
FR_T = float(os.environ.get("DC_FRT", "6")) * MM  # the tilted fingers' leading face, ahead of the fingertip point
DEMO = os.environ.get("DC_DEMO", "")  # "push": push it home by hand and leave the handle open (the shortcut check)


def _bay(p) -> tuple[float, float, float]:
    """A TCP target from a point in the empty bay's frame (x from the seated front, y from the bay's middle, z from
    the seated carrier's mid-height)."""
    o = bay_origin(*EMPTY)
    return tuple(float(v) for v in (o + np.asarray(p, float) + CAL))


def cam_shift(deg: float) -> float:
    """Where the closing handle's hook holds the carrier (x of its front from home, m) at handle angle ``deg``: the
    hook's front face bears on the rear corner of the column's lip (at the bay's edge) and slides along it as the
    handle shuts; never further out than the touch point."""
    a = math.radians(deg)
    edge = -BAY_W / 2  # the lip's rear corner, across the bay
    s = (PIVOT[1] - edge - HOOK_W / 2 * math.sin(a)) / max(math.cos(a), 1e-6)  # how far out along the hook
    if s > HOOK_L + HOOK_W / 2:
        return -MATE  # the hook does not yet reach in behind the lip
    pin_x = LIP_X1 + HOOK_W / 2 * math.cos(a) - s * math.sin(a)
    return float(np.clip(pin_x - PIVOT[0], -MATE, 0.0))


CARRY_PITCH = math.radians(float(os.environ.get("DC_CPITCH", "60")))  # the carrying hand tilted this far down: the
# pads still on the carrier's sides, the fingers leaning back clear of the unit over the server as it goes in
GRIP_C = carrierkit.tilted(0.0, CARRY_PITCH)


def _carry_tcp(point) -> tuple[float, float, float]:
    """The TCP with the tilted pads' tips at ``point`` (env)."""
    a = np.array([math.cos(CARRY_PITCH), 0.0, -math.sin(CARRY_PITCH)])
    return tuple(float(v) for v in np.asarray(point, float) - TIP * a)


def take_and_insert() -> list:
    s = C_START
    at = _carry_tcp((s[0] + GRIP_X, s[1], s[2] + GRIP_DZ))
    o = bay_origin(*EMPTY) + CAL
    lined = _carry_tcp(o + np.array([X_CAGE - 10 * MM - CL + GRIP_X, 0.0, LIFT + GRIP_DZ]))
    inside = _carry_tcp(o + np.array([RELEASE_X, 0.0, LIFT + GRIP_DZ]))
    up = np.array([0.0, 0.0, 0.05])
    return [Waypoint(2.0, tuple(np.asarray(at) + up), OPEN_W, "over the carrier", GRIP_C),
            Waypoint(1.2, at, OPEN_W, "round its sides", GRIP_C),
            Waypoint(1.0, at, HOLD, "take it", GRIP_C),
            Waypoint(1.2, (at[0], at[1], CARRY_Z), HOLD, "lift it", GRIP_C),
            Waypoint(3.5, (lined[0], lined[1], CARRY_Z), HOLD, "over in front of the bay", GRIP_C),
            Waypoint(1.5, lined, HOLD, "in line with the bay", GRIP_C),
            Waypoint(1.0, lined, HOLD, "settle", GRIP_C),
            Waypoint(2.0, inside, HOLD, "slide it in", GRIP_C),
            Waypoint(1.0, inside, OPEN_W, "let go", GRIP_C),
            Waypoint(1.0, tuple(np.asarray(inside) + np.array([-0.03, 0.0, 0.03])), OPEN_W, "back off", GRIP_C)]


GRIP_T = carrierkit.tilted(0.0, PUSH_PITCH)  # fingertips pointing in at the rack, tilted down, closed side by side


def _tilted_tcp(point, push_dir) -> tuple[float, float, float]:
    """The TCP for the tilted closed fingertips pressing at ``point`` (the bay's frame) in direction ``push_dir``."""
    a = np.array([math.cos(PUSH_PITCH), 0.0, -math.sin(PUSH_PITCH)])  # the approach (the fingers' direction)
    p = np.asarray(point, float) - FR_T * np.asarray(push_dir, float) - TIP * a
    return _bay(p)


def push_in() -> list:
    face = -(GRIP_X - RELEASE_X)  # where it was let go (its front's x)
    goal = -MATE + PUSH_PAST if DEMO != "push" else 0.20
    z = PUSH_DZ  # (let go, its front end has tipped down onto the frame; it levels as it goes in)
    fwd = (1.0, 0.0, 0.0)
    return [Waypoint(2.0, _tilted_tcp((face - 0.03, PUSH_Y, 0.03), fwd), FINGER_CLOSED, "in front of it", GRIP_T),
            Waypoint(1.0, _tilted_tcp((face - 0.01, PUSH_Y, z), fwd), FINGER_CLOSED, "down to its front", GRIP_T),
            Waypoint(2.0, _tilted_tcp((goal, PUSH_Y, z), fwd), FINGER_CLOSED, "push it in", GRIP_T),
            Waypoint(0.8, _tilted_tcp((-MATE - 0.03, PUSH_Y, z), fwd), FINGER_CLOSED, "back off", GRIP_T)]


def _on_handle(deg: float, press: float) -> tuple[float, float, float]:
    """The TCP pushing the handle (at angle ``deg`` open) at SWEEP_R out from its pin, from its closing side."""
    a = math.radians(deg)
    pin = PIVOT + np.array([cam_shift(deg), 0.0, 0.0])
    along = np.array([-math.sin(a), math.cos(a), 0.0])
    push = np.array([math.cos(a), math.sin(a), 0.0])  # the direction its pushed point moves as it shuts
    p = pin + SWEEP_R * along - (H_T / 2 - press) * push
    return _tilted_tcp((p[0], p[1], HANDLE_DZ), push)


def close_handle() -> list:
    start = np.asarray(_on_handle(OPEN_DEG, -6 * MM))
    out = [Waypoint(1.5, (start[0], start[1], start[2] + 0.03), FINGER_CLOSED, "beside the handle", GRIP_T),
           Waypoint(1.0, tuple(start), FINGER_CLOSED, "down beside it", GRIP_T)]
    for deg in (80.0, 65.0, 50.0, 35.0, 20.0, 8.0, 0.0, -4.0):
        out.append(Waypoint(1.0, _on_handle(deg, PRESS), FINGER_CLOSED,
                            "press it shut" if deg <= 0 else "close the handle", GRIP_T))
    end = np.asarray(_on_handle(0.0, -10 * MM))
    out += [Waypoint(1.0, tuple(end), FINGER_CLOSED, "off it", GRIP_T),
            Waypoint(1.0, (end[0] - 0.03, end[1], end[2] + 0.03), FINGER_CLOSED, "up", GRIP_T),
            Waypoint(2.0, RHOME, FINGER_OPEN, "home", 0.0)]
    return out


def demo():
    right = [Waypoint(1.0, RHOME, FINGER_OPEN, "wait", 0.0)] + take_and_insert() + push_in()
    right += close_handle() if DEMO != "push" else [Waypoint(2.0, RHOME, FINGER_OPEN, "home", 0.0)]
    left = [Waypoint(sum(w.duration for w in right), LHOME, FINGER_OPEN, "wait", 0.0)]
    for plan in (left, right):
        plan.append(Waypoint(2.0, plan[-1].pos, plan[-1].finger, "done", plan[-1].yaw))
    return {"right": right, "left": left}
