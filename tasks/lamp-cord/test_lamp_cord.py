"""Offline checks of lamp_cord.py's scoring and demo, without the simulator. Needs yam_arena (the official stack):

    python -m pytest -q test_lamp_cord.py
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

TASK = load(Path(__file__).with_name("lamp_cord.py"))
M = TASK.module
import lampkit as lk  # noqa: E402  (the task file put its folder on sys.path)

ZERO = np.zeros(3)
MM = 0.001


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    M._reset_state()
    monkeypatch.setattr(lk, "finger_points", lambda robots=None: np.zeros((0, 3)))
    yield


def states(tip_y, moving=False, off_axis=0.0):
    """The cord straight along the pipe's axis (y), its end at ``tip_y``; ``off_axis`` moves it sideways (out of
    the pipe)."""
    out = {}
    for spec in M.OBJECTS:
        pos = np.array(spec.pos, float)
        quat = spec.quat if spec.quat else lk.yaw_quat(spec.yaw)
        out[spec.name] = lk.body_state(spec, pos, quat, ZERO, ZERO)
    for i, c in enumerate(M.CORD):
        p = np.array([M.PIPE_X + off_axis, tip_y - (i + 0.5) * M.LINK, M.TOP + M.PIPE_Z])
        v = np.array([0.0, 0.05, 0.0]) if moving else ZERO
        out[c.name] = lk.body_state(c, p, lk.yaw_quat(-math.pi / 2), v, ZERO)
    return out


def held(st, n=M.STILL_STEPS):
    for _ in range(n):
        ok = M.success(st)
        res = (M.score(st), ok)
    return res


# -- scoring ------------------------------------------------------------------------------------------------------
def test_the_end_read_from_the_states_is_where_it_was_put():
    st = states(M.PIPE_Y1 + 0.080)
    assert M.cord_tip(st)[1] == pytest.approx(M.PIPE_Y1 + 0.080, abs=1e-6)


def test_threaded_through_with_3_in_out_of_the_top_succeeds():
    assert held(states(M.PIPE_Y1 + 0.076)) == (1.0, True)


def test_full_only_after_a_second_held():
    st = states(M.PIPE_Y1 + 0.080)
    assert held(st, M.STILL_STEPS - 1)[0] == 0.5
    assert held(st, 1)[0] == 1.0


@pytest.mark.parametrize("out", [0.030, 0.055, 0.110, 0.150])
def test_through_but_not_about_3_in_out_is_half(out):
    assert held(states(M.PIPE_Y1 + out)) == (0.5, False)


@pytest.mark.parametrize("out", [0.060, 0.100])
def test_the_length_window_is_inclusive(out):
    assert held(states(M.PIPE_Y1 + out))[0] == 1.0


def test_still_moving_is_half():
    assert held(states(M.PIPE_Y1 + 0.080, moving=True)) == (0.5, False)


def test_a_hand_still_on_the_cord_is_half(monkeypatch):
    st = states(M.PIPE_Y1 + 0.080)
    on = np.array([[M.PIPE_X, M.PIPE_Y1 + 0.040, M.TOP + M.PIPE_Z + 0.004]])
    monkeypatch.setattr(lk, "finger_points", lambda robots=None: on)
    assert held(st) == (0.5, False)


def test_half_way_up_the_pipe_is_a_quarter():
    assert held(states(0.02)) == (0.25, False)


def test_just_into_the_pipe_is_nothing():
    assert held(states(M.PIPE_Y0 + 0.05)) == (0.0, False)


def test_untouched_is_nothing():
    assert held({**states(0.0), **{c.name: s for c, s in zip(M.CORD, [
        lk.body_state(c, np.array(c.pos, float), lk.yaw_quat(-math.pi / 2), ZERO, ZERO) for c in M.CORD])}}) \
        == (0.0, False)


def test_a_cord_laid_past_the_lamp_not_through_its_pipe_is_nothing():
    """Its end beyond the top, but the cord beside the lamp, not in the pipe."""
    assert held(states(M.PIPE_Y1 + 0.080, off_axis=0.05)) == (0.0, False)


def test_the_pipe_and_cord_fit():
    """SPT-1 (2.8 x 5.5 mm) passes the 7.6 mm bore: its half-diagonal is inside the bore's radius."""
    assert math.hypot(M.CORD_W / 2, M.CORD_T / 2) < M.PIPE_ID / 2


# -- the demo -----------------------------------------------------------------------------------------------------
GUARD_MAX_DELTA = np.array([0.287979, 0.1832595, 0.1570795, 0.1631885, 0.15708, 0.20944, 0.05])


@pytest.mark.parametrize("arm", ["left", "right"])
def test_the_demo_reaches_every_target_along_its_path(arm):
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


def test_the_left_hand_takes_the_cord_only_after_the_right_has_fed_it():
    d = M.demo()
    lp, rp = ArmPlan(d["left"], station.BASES["left"]), ArmPlan(d["right"], station.BASES["right"])
    take = min(t for t in np.arange(0.0, lp.duration, 0.05) if lp.target(t)[3] == "take it")
    last_feed = max(t for t in np.arange(0.0, rp.duration, 0.05) if rp.target(t)[3] == "feed it in")
    assert take > last_feed


def test_the_right_hand_stays_clear_of_the_lamps_base():
    d = M.demo()
    rp = ArmPlan(d["right"], station.BASES["right"])
    ys = [rp.target(t)[0][1] for t in np.arange(0.0, rp.duration, 0.05) if rp.target(t)[0][2] < M.TOP + 0.08]
    assert max(ys) <= M.HAND_MAX + 1e-6
