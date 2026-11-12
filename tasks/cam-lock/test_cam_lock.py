"""Checks of cam_lock.py's cam-and-bolt model, scoring and demo, without the simulator. Needs yam_arena (the official
stack):

    python -m pytest -q test_cam_lock.py
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

TASK = load(Path(__file__).with_name("cam_lock.py"))
M = TASK.module
import camkit as ck  # noqa: E402  (the task file put its folder on sys.path)

MM = 0.001
ZERO = np.zeros(3)
I3 = np.eye(3)
R, L = 0, 1


def rz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def shelf_at(gap, dy=0.0, skew=0.0, vel=0.0):
    pos = np.array([M.FACE_X - gap - M.HALF, dy, M.TOP + M.PANEL_T / 2])
    return ck.Frame(pos=pos, rot=rz(skew), vel=np.array([vel, 0.0, 0.0]), spin=ZERO.copy(), com=pos)


def driver_at(tip, yaw=0.0):
    tip = np.asarray(tip, float)
    return ck.Frame(pos=tip, rot=rz(yaw), vel=ZERO.copy(), spin=ZERO.copy(), com=tip + [0, 0, 0.1])


PARKED = {"driver_r": driver_at([*M.HOLDER_XY[0], M.TOP + 0.5 * MM]),
          "driver_l": driver_at([*M.HOLDER_XY[1], M.TOP + 0.5 * MM])}


@pytest.fixture(autouse=True)
def fresh():
    M.MODEL.clear()
    M.MODEL.origin = np.zeros(3)
    M._reset_state()
    yield


def step(gap, drivers=None, **kw):
    acc = {}
    frames = {"shelf": shelf_at(gap, **kw), **PARKED, **(drivers or {})}
    M.MODEL.update(frames, acc)
    return acc


def cam_world(k, gap):
    return np.array(M.cam_at(k, gap))


def turn(k, gap, deg, start_yaw=0.0, n=30):
    """Screwdriver k in cam k, turned by ``deg`` (positive: clockwise seen from above)."""
    name = M.DRIVERS[k].name
    for yaw in np.linspace(start_yaw, start_yaw - deg, n):
        step(gap, {name: driver_at(cam_world(k, gap), yaw)})
    assert M.MODEL.on[name] == k


def open_cams(gap=M.START_GAP):
    turn(R, gap, -90.0)  # the right one a quarter turn anticlockwise
    turn(L, gap, 90.0)  # the left one a quarter turn clockwise
    assert M.MODEL.is_open(R) and M.MODEL.is_open(L)


def push_home():
    for g in np.linspace(M.START_GAP, 0.0, 60):
        step(g)


# -- the bolts and the cams -------------------------------------------------------------------------------------------
def test_the_cams_start_turned_away_from_the_edge():
    step(M.START_GAP)
    assert not M.MODEL.is_open(R) and not M.MODEL.is_open(L)


def test_a_screwdriver_in_the_recess_turns_the_cam_with_it():
    turn(R, M.START_GAP, -90.0)
    assert math.degrees(M.MODEL.phi[R]) == pytest.approx(0.0, abs=1.0)


def test_a_screwdriver_just_off_the_recess_is_guided_in_gently():
    tip = cam_world(R, M.START_GAP) + [3.0 * MM, 0.0, 1.0 * MM]
    acc = step(M.START_GAP, {"driver_r": driver_at(tip)})
    assert M.MODEL.on["driver_r"] is None and -2.0 < acc["driver_r"][0][0] < 0.0


def test_the_holders_mouth_guides_a_blade_into_its_hole():
    tip = np.array([M.HOLDER_XY[0][0] + 4.0 * MM, M.HOLDER_XY[0][1], M.TOP + M.HOLDER_H + 3.0 * MM])
    acc = step(M.START_GAP, {"driver_r": driver_at(tip)})
    assert acc["driver_r"][0][0] < 0.0


def test_square_and_in_line_the_shelf_goes_onto_the_bolts():
    step(M.BOLT_LEN + 2 * MM)
    step(M.BOLT_LEN - 2 * MM)
    assert all(M.MODEL.entered)


def test_out_of_line_its_edge_stands_on_the_bolts_ends():
    step(M.BOLT_LEN + 2 * MM, dy=4.0 * MM)
    acc = step(M.BOLT_LEN - 2 * MM, dy=4.0 * MM)
    assert not any(M.MODEL.entered) and acc["shelf"][0][0] < 0


def test_with_the_cams_open_the_shelf_comes_home_and_the_heads_go_in():
    open_cams()
    push_home()
    assert all(M.MODEL.bolt_in)


def test_with_a_cam_closed_its_bolts_head_meets_the_cams_side():
    turn(L, M.START_GAP, 90.0)  # only the left one opened
    for g in np.linspace(M.START_GAP, M.MODEL.g_closed + 0.5 * MM, 40):
        step(g)
    assert not M.MODEL.bolt_in[R] and M.MODEL.push[R] == 0.0
    step(M.MODEL.g_closed - 1.0 * MM)
    assert M.MODEL.push[R] > 0.0 and not M.MODEL.bolt_in[R]


def test_shoved_hard_onto_a_closed_cam_the_shelf_still_stops_short():
    for g in np.linspace(M.START_GAP, M.MODEL.g_closed, 40):
        step(g)
    step(M.MODEL.g_closed - 3.0 * MM)  # 15 N
    assert not any(M.MODEL.bolt_in) and M.MODEL.push[R] > 10.0


def test_half_a_turn_clockwise_draws_the_joint_shut_and_locks_it():
    open_cams()
    push_home()
    for k in (R, L):
        turn(k, M.MODEL.g_home, 180.0)
    assert all(M.MODEL.locked(k) for k in (R, L))
    assert M.MODEL.target_gap(R) == pytest.approx(0.0, abs=1e-6)


def test_drawn_in_the_cam_pulls_the_shelf_to_the_face():
    open_cams()
    push_home()
    turn(R, M.MODEL.g_home, 180.0)
    acc = step(1.0 * MM)
    assert acc["shelf"][0][0] > 0  # pulled in


def test_with_the_head_in_the_cam_stops_anticlockwise_at_ten_degrees():
    open_cams()
    push_home()
    turn(R, M.MODEL.g_home, -40.0)
    assert math.degrees(M.MODEL.phi[R]) == pytest.approx(-10.0, abs=0.5)


def test_with_no_bolt_in_a_cam_turns_freely():
    turn(R, M.START_GAP, 300.0)
    assert math.degrees(M.MODEL.phi[R]) == pytest.approx(390.0, abs=1.0)


def test_the_cams_are_drawn_turned_with_the_shelf():
    turn(R, M.START_GAP, -90.0)
    pos, quat = M.MODEL.poses["cam_r"]
    assert np.allclose(pos, cam_world(R, M.START_GAP)) and quat[3] == pytest.approx(1.0, abs=1e-3)


# -- scoring, on the task's layout ----------------------------------------------------------------------------------
def states(shelf_gap=0.0, drivers_back=True):
    out = {}
    for spec in M.OBJECTS:
        pos = np.array(spec.pos, float)
        quat = spec.quat if spec.quat else ck.yaw_quat(spec.yaw)
        if spec.name == "shelf":
            pos[0] = M.FACE_X - shelf_gap - M.HALF
        if spec.name.startswith("driver") and not drivers_back:
            pos[:2] += [0.05, 0.0]
        out[spec.name] = ck.body_state(spec, pos, quat, ZERO, ZERO)
    return out


def join():
    open_cams()
    push_home()
    for k in (R, L):
        turn(k, M.MODEL.g_home, 180.0)
    step(0.0)


def test_joined_and_the_screwdrivers_back_succeeds():
    join()
    assert held(states()) == (1.0, True)


def held(st, n=M.STILL_STEPS):
    """success() then score() on the same state ``n`` times, as the run calls them; the last (score, success)."""
    for _ in range(n):
        ok = M.success(st)
        out = (M.score(st), ok)
    return out


def test_joined_with_a_screwdriver_not_back_is_half():
    join()
    assert M.score(states(drivers_back=False)) == 0.5


def test_a_cam_not_locked_is_not_joined():
    open_cams()
    push_home()
    turn(R, M.MODEL.g_home, 180.0)
    turn(L, M.MODEL.g_home, 90.0)
    step(0.0)
    assert M.score(states()) == 0.25


def test_on_the_bolts_but_not_drawn_up_is_a_quarter():
    open_cams()
    push_home()
    assert M.score(states(M.MODEL.g_home)) == 0.25


def test_shoved_onto_closed_cams_scores_nothing():
    for g in np.linspace(M.START_GAP, M.MODEL.g_closed - 3.0 * MM, 40):
        step(g)
    assert M.score(states()) == 0.0 and not M.success(states())


def test_the_cam_reaches_the_bolts_head():
    assert abs(M.CAM_IN - (M.BOLT_LEN - M.HEAD_L / 2)) < M.CAM_R / 2


def test_nothing_done_scores_nothing():
    assert M.score(states(M.START_GAP)) == 0.0


# -- the scripted demo -----------------------------------------------------------------------------------------------
# The per-step limits of inspect-robots' DeltaLimitGuard for the YAM action space, as recorded in a run's
# configuration: six joints, then the gripper (0-1, 5% a step).
GUARD_MAX_DELTA = np.array([0.287979, 0.1832595, 0.1570795, 0.1631885, 0.15708, 0.20944, 0.05])


@pytest.mark.parametrize("arm", ["left", "right"])
def test_the_demo_reaches_every_target_along_its_path(arm):
    """Forward kinematics of the commanded joints against the target, every 50 ms."""
    plan = ArmPlan(M.demo()[arm], station.BASES[arm])
    base = np.array(station.BASES[arm])
    worst = 0.0
    for t in np.arange(0.0, plan.duration, 0.05):
        pos = plan.target(t)[0]
        tcp = ik.tcp_pose(plan.action7(t)[0][:6])
        tcp = np.asarray(tcp[0] if isinstance(tcp, tuple) else tcp[:3, 3]) + base
        worst = max(worst, float(np.linalg.norm(tcp - pos)))
    assert worst < 1.0 * MM


@pytest.mark.parametrize("arm", ["left", "right"])
def test_the_demo_passes_the_run_guards(arm):
    plan = ArmPlan(M.demo()[arm], station.BASES[arm])
    acts = np.array([plan.action7(t)[0] for t in np.arange(0.0, plan.duration, 0.05)])
    assert (np.abs(np.diff(acts, axis=0)) / GUARD_MAX_DELTA).max() < 0.95


def test_the_demo_fits_the_run_limit():
    assert max(ArmPlan(p, station.BASES[a]).duration for a, p in M.demo().items()) * 20 < M.RUN_LIMIT_STEPS


def test_the_demo_opens_the_cams_before_pushing_and_turns_them_half_a_round_after():
    for arm in ("right", "left"):
        labels = [w.label for w in M.demo()[arm]]
        assert labels.index("turn it open") < labels.index("push it on") < labels.index("lock it, 1")
        turns = [w for w in M.demo()[arm] if w.label.startswith("lock it")]
        assert len(turns) == 2  # two quarter turns, clockwise: 180 degrees
