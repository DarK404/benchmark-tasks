"""Checks of tap_wrench.py's scoring, its thread model, and its demo, without the simulator. Needs yam_arena (the
official stack):

    python -m pytest -q test_tap_wrench.py
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

TASK = load(Path(__file__).with_name("tap_wrench.py"))
M = TASK.module
import tapkit as tk  # noqa: E402  (the task file put its folder on sys.path)

ZERO = np.zeros(3)


def wrench(moving=False, turns=0.0):
    yaw = math.radians(M.START_YAW - 360.0 * turns)
    v = np.array([0.05, 0.0, 0.0]) if moving else ZERO
    return tk.body_state(M.WRENCH, (*M.TAP_XY, M.TOP + M.BLOCK_TOP), tk.yaw_quat(yaw), v, ZERO)


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    """A new trial: the thread at its start, latches cleared, grippers far away."""
    M.THREAD.clear()
    M._reset_state()
    monkeypatch.setattr(tk, "finger_points", lambda: np.array([[0.0, 0.0, M.TOP + 0.5]]))
    yield


def run(*turns, moving=False):
    """Score the trial frame by frame through the given turn counts; returns the last (score, success)."""
    out = None
    for t in turns:
        M.THREAD.turns = t
        f = {"wrench": wrench(moving, t), "block": None}
        out = (M.score(f), M.success(f))
    return out


# -- the tiers ---------------------------------------------------------------------------------------------------


def test_start_scores_zero():
    assert run(0.0) == (0.0, False)


def test_a_tenth_in_is_a_quarter():
    assert run(0.0, 0.12) == (0.25, False)


def test_a_quarter_in_is_half():
    assert run(0.0, 0.15, 0.26) == (0.5, False)


def test_quarter_in_then_backed_off_and_let_go_is_full():
    assert run(0.0, 0.15, 0.27, 0.20, 0.16) == (1.0, True)


def test_backed_off_but_not_far_enough_is_half():
    assert run(0.0, 0.27, 0.19) == (0.5, False)


def test_backing_off_before_the_quarter_turn_does_not_count():
    assert run(0.0, 0.20, 0.05, 0.26) == (0.5, False)


def test_still_moving_is_not_success_yet():
    assert run(0.0, 0.27, 0.15, moving=True) == (1.0, False)


def test_a_gripper_still_on_the_wrench_is_half(monkeypatch):
    w = wrench(turns=0.15)
    on_handle = w.to_world((0.10, 0.0, M.BAR_UP))
    monkeypatch.setattr(tk, "finger_points", lambda: on_handle[None, :])
    assert run(0.0, 0.27, 0.15) == (0.5, False)


def test_grippers_just_clear_are_full(monkeypatch):
    w = wrench(turns=0.15)
    above = w.to_world((0.10, 0.0, M.BAR_UP + M.HANDLE_R + 20 * 0.001))
    monkeypatch.setattr(tk, "finger_points", lambda: above[None, :])
    assert run(0.0, 0.27, 0.15) == (1.0, True)


# -- violations latch --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("why", ["side load 7.5 N", "tilt 2.4 deg", "axial load 70 N (thread stripped)"])
def test_a_snapped_tap_scores_zero_for_good(why):
    run(0.0, 0.10)
    M.THREAD.snapped = why
    assert run(0.27, 0.15) == (0.0, False)


def test_reset_clears_the_latch():
    M.THREAD.snapped = "side load 7.5 N"
    run(0.1)
    M.THREAD.clear()
    M._reset_state()
    assert run(0.0, 0.27, 0.15) == (1.0, True)


# -- the thread model --------------------------------------------------------------------------------------------


def frame(yaw_deg=0.0, off=(0.0, 0.0, 0.0), tilt_deg=0.0, spin_z=0.0):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    ct, st = math.cos(math.radians(tilt_deg)), math.sin(math.radians(tilt_deg))
    rx = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    pos = np.asarray(off, float)
    return {"w": tk.Frame(pos=pos, rot=rz @ rx, vel=ZERO.copy(), spin=np.array([0.0, 0.0, spin_z]), com=pos)}


def thread(**kw):
    t = tk.Thread("w", origin=(0.0, 0.0, 0.0), **kw)
    t.clear()
    t.update(frame(), {})
    return t


def test_held_below_the_cutting_torque_the_tap_does_not_slip():
    t = thread(torque_in=1.0, k_stick=100.0)
    acc = {}
    t.update(frame(-0.3), acc)  # 0.3 deg clockwise: the spring pushes back 0.52 N m, under 1.0
    assert 0.4 < acc["w"][1][2] < 0.6


def test_past_the_cutting_torque_it_slips_at_exactly_that_torque():
    t = thread(torque_in=1.0, torque_out=0.3)
    acc = {}
    t.update(frame(-30.0), acc)
    assert acc["w"][1][2] == pytest.approx(1.0)
    acc = {}
    t.update(frame(0.0), acc)  # back 30 degrees: backing out takes only 0.3
    assert acc["w"][1][2] == pytest.approx(-0.3)


def test_turning_in_pulls_the_tap_down_one_pitch_per_turn():
    t = thread(pitch=0.0015)
    for yaw in range(-30, -361, -30):  # one full turn clockwise, in steps
        t.update(frame(float(yaw)), {})
    acc = {}
    t.update(frame(-360.0), acc)
    assert t.turns == pytest.approx(1.0)
    assert acc["w"][0][2] == pytest.approx(-t.k_axial * 0.0015, rel=1e-6)  # still at the start height: pulled down


def test_a_steady_side_load_snaps_the_tap():
    t = thread(side_limit=5.0, k_side=5000.0, side_filter=0.1, dt=1 / 240)
    for _ in range(240):  # 1.5 mm off the axis: 7.5 N sideways, held 1 s
        t.update(frame(off=(0.0015, 0.0, 0.0)), {})
    assert t.snapped and t.snapped.startswith("side load")


def test_a_brief_side_knock_does_not_snap_it():
    t = thread(side_limit=5.0, k_side=5000.0, side_filter=0.1, dt=1 / 240)
    for _ in range(5):  # 7.5 N for 20 ms
        t.update(frame(off=(0.0015, 0.0, 0.0)), {})
    for _ in range(240):
        t.update(frame(), {})
    assert t.snapped is None


def test_tilt_snaps_the_tap():
    t = thread(tilt_limit=2.0)
    t.update(frame(tilt_deg=2.5), {})
    assert t.snapped and t.snapped.startswith("tilt")


def test_a_hard_pull_strips_the_thread():
    t = thread(axial_limit=60.0, k_axial=5000.0)
    t.update(frame(off=(0.0, 0.0, 0.02)), {})  # 20 mm up: 100 N
    assert t.snapped and "stripped" in t.snapped


# -- the demo ----------------------------------------------------------------------------------------------------

# The per-step limits of inspect-robots' DeltaLimitGuard for the YAM action space, as recorded in a run's
# configuration: six joints, then the gripper (0-1, 5% a step).
GUARD_MAX_DELTA = np.array([0.287979, 0.1832595, 0.1570795, 0.1631885, 0.15708, 0.20944, 0.05])


@pytest.mark.parametrize("arm", ["left", "right"])
def test_the_demo_is_within_reach_and_passes_the_run_guards(arm):
    plan = ArmPlan(M.demo()[arm], station.BASES[arm])
    acts = np.array([plan.action7(t)[0] for t in np.arange(0.0, plan.duration, 0.05)])
    assert plan.worst_ik_mm < 0.5 and plan.worst_ik_deg < 0.1, (plan.worst_ik_mm, plan.worst_ik_deg)
    assert (np.abs(np.diff(acts, axis=0)) / GUARD_MAX_DELTA).max() < 0.95


def test_the_demo_turns_far_enough_in_and_back_with_margin():
    assert (M.A_START - M.A_IN) / 360.0 >= M.IN_TURNS + 0.02
    assert (M.A_BACK - M.A_IN) / 360.0 >= M.BACK_TURNS + 0.02


def test_the_demo_fits_the_run_limit():
    assert max(ArmPlan(p, station.BASES[a]).duration for a, p in M.demo().items()) * 20 < M.RUN_LIMIT_STEPS
