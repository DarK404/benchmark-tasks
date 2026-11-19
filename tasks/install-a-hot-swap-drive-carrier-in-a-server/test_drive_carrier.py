"""Offline checks of drive_carrier.py's geometry, forces, scoring and demo, without the simulator. Needs yam_arena (the
official stack):

    python -m pytest -q test_drive_carrier.py
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

TASK = load(Path(__file__).with_name("drive_carrier.py"))
M = TASK.module
import carrierkit as ck  # noqa: E402  (the task file put its folder on sys.path)

MM = 0.001
ZERO = np.zeros(3)


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    M._reset_state()
    monkeypatch.setattr(ck, "finger_points", lambda robots=None, **kw: np.zeros((0, 3)))
    yield


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def objects(d=0.0, handle=0.0, tab=0.0, yaw=0.0, moving=False, dy=0.0):
    """The carrier ``d`` (m) out from home in the empty bay, turned ``yaw`` (deg), its handle ``handle`` degrees open
    and its catch tab lifted ``tab`` (m)."""
    rc = _rz(math.radians(yaw))
    pos = M.bay_origin(*M.EMPTY) + np.array([d, dy, 0.0])
    vel = np.array([0.02, 0.0, 0.0]) if moving else ZERO
    carrier = ck.body_state(M.CARRIER, pos, ck.matrix_to_quat_rel(rc), vel, ZERO)
    rh = rc @ _rz(math.radians(handle))
    handle_st = ck.body_state(M.HANDLE, pos + rc @ M.PIVOT, ck.matrix_to_quat_rel(rh), ZERO, ZERO)
    tab_st = ck.body_state(M.TAB, pos + rc @ (M._TAB_LOCAL + np.array([0.0, 0.0, tab])), ck.matrix_to_quat_rel(rc),
                           ZERO, ZERO)
    return {M.CARRIER.name: carrier, M.HANDLE.name: handle_st, M.TAB.name: tab_st}


def run(objs, steps=M.STILL_STEPS + 2):
    out = None
    for _ in range(steps):
        M.success(objs)
        out = M.score(objs)
    return out


# -- the parts -------------------------------------------------------------------------------------------------------

def test_the_server_is_a_1u_r640_with_two_rows_of_four_bays():
    assert M.SRV_W == pytest.approx(0.434) and M.SRV_H == pytest.approx(42.8 * MM)
    assert len(M.COLS) == 4 and set(M.ROW_Z) == {"lower", "upper"}
    assert M.ROW_Z["upper"] + M.BAY_H <= M.COVER_Z0 < M.SRV_H
    assert M.EMPTY == (1, "upper")


def test_the_carrier_fits_its_bay_with_a_little_play():
    assert 0.0 < M.BAY_W - M.CW <= 1.01 * MM
    assert 0.0 < M.BAY_H - M.CH <= 3.01 * MM


def test_the_open_handle_points_out_at_the_robot_and_shuts_across_the_front():
    h = ck._HINGES[M.HANDLE.name]
    assert (h.lower, h.upper) == (pytest.approx(-math.radians(M.OPEN_DEG)), 0.0)  # spawned open, shuts by -90
    assert h.stiffness > 0 and h.target == 0.0  # its spring opens it
    assert np.allclose(M._R_OPEN @ [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0])
    assert M.PIVOT[1] + M.H_LEN < M.CW / 2  # shut, it lies within the carrier's width


def test_the_connector_mates_with_a_force_within_the_sas_limit():
    # the shoes are spread from SHOE_GAP to the blade's thickness; their springs' force there:
    assert M.BLADE_T > M.SHOE_GAP
    f = M.SHOE_PRELOAD + M.SHOE_K * (M.BLADE_T - M.SHOE_GAP) / 2
    a, mu = math.radians(M.RIDGE_DEG), M.CONTACT_MU
    peak = 2 * f * (math.sin(a) + mu * math.cos(a)) / (math.cos(a) - mu * math.sin(a))  # up the lead
    hold = 2 * f * mu  # along the flat
    assert peak < 25.0  # SFF-8482: mating force no more than 25 N
    assert hold >= 5.0  # ... and an unmating force of at least 5 N
    assert peak == pytest.approx(13.0, abs=0.5) and hold == pytest.approx(6.3, abs=0.2)


def test_the_blade_tip_is_thinner_than_the_shoes_gap_and_touches_them_6_mm_before_home():
    tip = 2 * M.TIP_PLATE * math.cos(math.radians(M.RIDGE_DEG))  # where its two sloped plates meet
    assert tip < M.SHOE_GAP
    assert M.SHOE_X1 - M.BLADE_TIP_X == pytest.approx(M.MATE)
    assert M.SHOE_X1 > M.BLADE_TIP_X + M.RIDGE_L  # home, the shoes' ends are on the blade's flat
    assert M.SLOT_X0 < M.BLADE_TIP_X and M.CL < M.BACKPLANE_X


# the cam: the hook against the bay frame, in the bay's mid-height plane
_O = M.bay_origin(*M.EMPTY) - np.array([M.SF, 0.0, M.TOP + M.RACK_Z])
_OBS = []
for _s in M.SERVER_SHAPES:
    _c, _h = np.array(_s.centre), np.array(_s.size) / 2
    if _s.kind == "box" and not (_s.tilt_y or _s.tilt_x or _s.yaw) and abs(_c[2] - _O[2]) <= _h[2]:
        _OBS.append((_c[0] - _h[0], _c[0] + _h[0], _c[1] - _O[1] - _h[1], _c[1] - _O[1] + _h[1]))


def _hook_points(deg, d):
    a = math.radians(deg)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    xs = np.linspace(-M.HOOK_W / 2, M.HOOK_W / 2, 9)
    ys = np.linspace(-(M.HOOK_L + M.HOOK_W / 2), 0.0, 40)
    return np.array([[x, y] for x in xs for y in ys]) @ r.T + np.array([M.PIVOT[0] + d, M.PIVOT[1]])


def _hits(p, eps=0.02 * MM):
    return any(((p[:, 0] > x0 + eps) & (p[:, 0] < x1 - eps) & (p[:, 1] > y0 + eps) & (p[:, 1] < y1 - eps)).any()
               for x0, x1, y0, y1 in _OBS)


def test_closing_the_handle_cams_the_carrier_home_from_the_connector():
    assert M.cam_shift(M.OPEN_DEG) == pytest.approx(-M.MATE)  # open: nothing holds it in yet
    assert abs(M.cam_shift(0.0)) < M.SEAT_TOL  # shut: home
    for deg in np.arange(M.OPEN_DEG, -0.1, -2.5):
        d = M.cam_shift(deg)
        assert not _hits(_hook_points(deg, d + 0.05 * MM)), deg  # the hook clear along the path ...
        if d > -M.MATE + 0.1 * MM:
            assert _hits(_hook_points(deg, d - 0.3 * MM)), deg  # ... bearing on the lip wherever it draws it in


def test_the_open_handle_passes_into_the_bay_and_a_carrier_left_short_blocks_it():
    assert not any(_hits(_hook_points(M.OPEN_DEG, d)) for d in np.arange(-0.12, -M.MATE, 0.002))
    blocked = [deg for deg in np.arange(M.OPEN_DEG, -0.1, -0.5) if _hits(_hook_points(deg, -10.0 * MM))]
    assert blocked and blocked[0] > 45.0  # 10 mm short, the hook strikes the frame's lip


def test_shut_the_hook_holds_it_in():
    assert _hits(_hook_points(0.0, -0.5 * MM))


def test_the_catch_lifts_over_the_handle_and_drops_in_front_of_it():
    overlap = M.H_H / 2 - M.NOSE_Z0
    assert 0.5 * MM <= overlap < M.TAB_TRAVEL
    assert M.SLOPE_Z0 < M.NOSE_Z0  # the slope ends below the nose: no corner to catch the handle's edge
    assert M.NOSE_X1 < M.PIVOT[0] - M.H_T / 2  # the nose's square face just in front of the shut handle
    assert M.PIVOT[1] + M.H_LEN > M.TAB_Y1  # over the handle's free end
    sp = ck._SPRINGS[M.TAB.name]
    assert sp.target < 0 and (sp.lower, sp.upper) == (0.0, M.TAB_TRAVEL)


# -- scoring ---------------------------------------------------------------------------------------------------------

def test_home_and_locked_is_full_after_a_second():
    objs = objects(d=-0.2 * MM)
    assert M.seated(objs[M.CARRIER.name]) and M.latched(objs)
    M.success(objs)
    assert M.score(objs) == 0.5
    assert run(objs) == 1.0


def test_success_ends_the_trial_only_after_the_second_held():
    objs = objects(d=-0.2 * MM)
    assert TASK.success_hold_s == 1.0  # and yam_arena holds it another 1 s before ending the trial
    assert not any(M.success(objs) for _ in range(M.STILL_STEPS - 1))
    assert M.success(objs) and M.score(objs) == 1.0


def test_the_handle_left_open_is_half():
    assert run(objects(handle=10.0)) == 0.5
    assert run(objects(handle=M.OPEN_DEG)) == 0.5


def test_the_catch_not_dropped_is_half():
    assert run(objects(tab=1.0 * MM)) == 0.5


def test_a_hand_on_it_is_half(monkeypatch):
    objs = objects()
    pt = objs[M.CARRIER.name].to_world((-3.0 * MM, 0.0, 0.0))
    monkeypatch.setattr(ck, "finger_points", lambda robots=None, **kw: np.array([pt]))
    assert run(objs) == 0.5


def test_still_moving_is_half():
    assert run(objects(moving=True)) == 0.5


def test_on_the_connector_but_not_home_is_a_quarter():
    for d in (-1.0 * MM, -3.0 * MM, -5.5 * MM):
        M._reset_state()
        assert run(objects(d=d, handle=M.OPEN_DEG)) == 0.25, d


def test_short_of_the_connector_is_nothing():
    assert run(objects(d=-10.0 * MM, handle=0.0)) == 0.0


def test_not_square_is_not_home():
    assert not M.seated(objects(yaw=3.0)[M.CARRIER.name])


def test_where_it_starts_is_nothing():
    c = ck.body_state(M.CARRIER, M.C_START, (0.0, 0.0, 0.0, 1.0), ZERO, ZERO)
    objs = objects()
    objs[M.CARRIER.name] = c
    assert run(objs) == 0.0


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
        assert np.linalg.norm(tcp - pos) < 2.0 * MM, (arm, t, plan.target(t)[3])


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


def test_the_demo_slides_it_in_then_pushes_then_closes_the_handle():
    labels = [w.label for w in PLANS["right"]]
    assert labels.index("slide it in") < labels.index("let go") < labels.index("push it in") \
        < labels.index("close the handle") < labels.index("press it shut")
    for w in PLANS["right"][labels.index("push it in"):labels.index("off it")]:
        assert w.finger == M.FINGER_CLOSED  # pushed with the closed fingertips, nothing held


def test_the_left_hand_stays_home():
    assert {tuple(w.pos) for w in PLANS["left"]} == {M.LHOME}


def test_the_push_check_never_touches_the_handle(monkeypatch):
    monkeypatch.setattr(M, "DEMO", "push")
    labels = [w.label for w in M.demo()["right"]]
    assert "push it in" in labels and "close the handle" not in labels
