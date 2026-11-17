"""Offline checks of radiator_valve.py's geometry, scoring and demo, without the simulator. Needs yam_arena (the
official stack):

    python -m pytest -q test_radiator_valve.py
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

TASK = load(Path(__file__).with_name("radiator_valve.py"))
M = TASK.module
import valvekit as vk  # noqa: E402  (the task file put its folder on sys.path)

MM = 0.001
ZERO = np.zeros(3)


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    M._reset_state()
    monkeypatch.setattr(vk, "finger_points", lambda robots=None, **kw: np.zeros((0, 3)))
    yield


def _rot(axis, a):
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def head_at(depth=M.COLLAR_L, dx=0.0, dz=0.0, roll=0.0, pitch=0.0, moving=False):
    """The head's state: ``depth`` (m) onto the collar (16 mm seated), moved (dx, dz) off the axis, turned ``roll``
    (rad) about its own axis and pitched ``pitch`` (rad) about x."""
    r = _rot("x", pitch) @ _rot("y", roll)
    origin = M._SEAT + np.array([dx, depth - M.COLLAR_L, dz])
    vel = np.array([0.0, 0.0, 0.05]) if moving else ZERO
    return vk.body_state(M.HEAD, origin, vk.matrix_to_quat_rel(r), vel, ZERO)


def objects(st):
    return {M.HEAD.name: st}


def run(st, steps=M.STILL_STEPS + 2):
    out = None
    for _ in range(steps):
        M.success(objects(st))
        out = M.score(objects(st))
    return out


# -- the parts -------------------------------------------------------------------------------------------------------

def test_the_head_is_an_ra_2000_sized_sensor():
    assert 2 * M.HEAD_R == pytest.approx(49.0 * MM)
    assert M.HEAD_L == pytest.approx(90.0 * MM)
    assert M.HEAD.mass == pytest.approx(0.15)


def test_the_bore_takes_the_collar_with_clearance():
    assert M.RIDGE_R < M.BORE_R and M.CORE_R < M.TOOTH_IN < M.RIDGE_R
    assert M.BORE_D > M.COLLAR_L  # the collar's tip stops short of the bore's bottom


def test_the_teeth_must_spread_over_the_ridge_and_drop_in_behind_it_before_the_shoulder():
    spread = M.RIDGE_R - M.TOOTH_IN
    assert spread == pytest.approx(1.0 * MM)
    assert M.SNAP_TRAVEL > spread
    # seated (the head's frame is then the seat frame), each tooth (2-6 mm in from the base face) lies on the shoulder
    # side of the ridge's square back face, with 0.6 mm to spare: it drops in before the base meets the shoulder
    assert M.TOOTH_Y0 - M.RIDGE_Y1 == pytest.approx(0.6 * MM)


def test_the_teeth_are_sprung_inwards_against_their_stops():
    sa, sb = vk._SPRINGS[M.SNAP_A.name], vk._SPRINGS[M.SNAP_B.name]
    pre = M.SNAP_PRELOAD / M.SNAP_K
    assert sa.target == pytest.approx(-pre) and sb.target == pytest.approx(pre)
    assert (sa.lower, sa.upper) == (0.0, M.SNAP_TRAVEL) and (sb.lower, sb.upper) == (-M.SNAP_TRAVEL, 0.0)


def test_the_snap_takes_a_gentle_push():
    # the force to slide a tooth up the ridge's lead-in face, at the crest (spread 1 mm), both teeth
    n = M.SNAP_PRELOAD + M.SNAP_K * (M.RIDGE_R - M.TOOTH_IN)
    t = math.tan(math.radians(M.LEAD_DEG))
    push = 2 * n * (t + M.SNAP_MU) / (1 - M.SNAP_MU * t)
    assert 4.0 < push < 12.0  # N


# -- scoring ---------------------------------------------------------------------------------------------------------

def test_snapped_home_with_the_pointer_up_is_full_after_a_second():
    st = head_at()
    assert M.snapped(st)
    M.success(objects(st))
    assert M.score(objects(st)) == 0.5
    assert run(st) == 1.0


def test_its_play_behind_the_ridge_still_counts():
    assert run(head_at(depth=M.COLLAR_L - 0.6 * MM)) == 1.0


def test_pointer_turned_away_from_the_top_is_half():
    assert run(head_at(roll=math.radians(15.0))) == 1.0
    M._reset_state()
    assert run(head_at(roll=math.radians(40.0))) == 0.5


def test_a_hand_on_it_is_half(monkeypatch):
    st = head_at()
    pt = st.to_world((0.0, -50 * MM, M.HEAD_R))
    monkeypatch.setattr(vk, "finger_points", lambda robots=None, **kw: np.array([pt]))
    assert run(st) == 0.5


def test_still_moving_is_half():
    assert run(head_at(moving=True)) == 0.5


def test_part_way_on_is_a_quarter():
    st = head_at(depth=10.0 * MM)
    assert not M.snapped(st) and M.started(st)
    assert run(st) == 0.25


def test_drooping_off_the_collar_tip_is_nothing():
    st = head_at(depth=5.5 * MM, dz=-18 * MM, pitch=math.radians(25.0))
    assert run(st) == 0.0


def test_not_on_the_valve_is_nothing():
    st = vk.body_state(M.HEAD, M.HEAD_START, (0.0, 0.0, 0.0, 1.0), ZERO, ZERO)
    assert run(st) == 0.0


def test_skewed_is_not_snapped():
    assert not M.snapped(head_at(pitch=math.radians(5.0)))
    assert not M.snapped(head_at(dx=2.0 * MM))


# -- the demo ------------------------------------------------------------------------------------------------------

PLANS = M.demo()
GUARD = np.array([0.287979, 0.1832595, 0.1570795, 0.1631885, 0.15708, 0.20944, 0.05])


def _samples(arm):
    plan = ArmPlan(PLANS[arm], station.BASES[arm])
    return plan, np.arange(0.0, plan.duration, 0.05)


@pytest.mark.parametrize("arm", ["right", "left"])
def test_the_demo_reaches_every_target_along_its_path(arm):
    plan, ts = _samples(arm)
    base = np.array(station.BASES[arm])
    for t in ts:
        pos = plan.target(t)[0]
        a = plan.action7(t)[0]
        tcp = ik.tcp_pose(a[:6])
        tcp = np.asarray(tcp[0] if isinstance(tcp, tuple) else tcp[:3, 3]) + base
        assert np.linalg.norm(tcp - pos) < 1.0 * MM, (arm, t, plan.target(t)[3])


@pytest.mark.parametrize("arm", ["right", "left"])
def test_the_demo_passes_the_run_guards(arm):
    plan, ts = _samples(arm)
    prev = None
    for t in ts:
        a = plan.action7(t)[0]
        if prev is not None:
            assert (np.abs(a - prev) / GUARD).max() < 0.95, (arm, t, plan.target(t)[3])
        prev = a


def test_the_demo_fits_the_run_limit():
    for arm in ("right", "left"):
        assert sum(w.duration for w in PLANS[arm]) * 20 < M.RUN_LIMIT_STEPS


def test_the_demo_pushes_straight_along_the_valve_axis_and_lets_go():
    w = {x.label: x for x in PLANS["right"]}
    a, b = np.array(w["lined up with the collar"].pos), np.array(w["push it on"].pos)
    d = b - a
    assert abs(d[0]) < 1e-9 and abs(d[2]) < 1e-9 and d[1] > 0
    assert w["let go"].finger == M.OPEN_W


def test_the_hand_stays_clear_of_the_valve():
    # the hand is about 100 mm wide along y: held 55 mm from the head's base, it stops short of the shoulder
    assert -M.GRIP_Y > 50.0 * MM


def test_the_left_hand_stays_home():
    assert {tuple(w.pos) for w in PLANS["left"]} == {M.LHOME}


def test_the_halfway_check_stops_short_of_the_snap(monkeypatch):
    monkeypatch.setenv("RV_DEMO", "halfway")
    w = {x.label: x for x in M.demo()["right"]}
    assert w["push it on"].pos[1] < M._tcp(M._SEAT)[1]
