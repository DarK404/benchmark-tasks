"""Offline checks of christmas_cracker.py's parts, scoring and demo, without the simulator. Needs yam_arena:

    python -m pytest -q test_christmas_cracker.py
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

TASK = load(Path(__file__).with_name("christmas_cracker.py"))
M = TASK.module
import crackerkit as ck  # noqa: E402

MM = 0.001
ZERO = np.zeros(3)


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    M._reset_state()
    monkeypatch.setattr(ck, "finger_points", lambda robots=None, **kw: np.zeros((0, 3)))
    yield


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def objects(dy_left=0.0, dz=0.0, moving=False, lifted=0.0):
    """The cracker lying as spawned, then its left end moved ``dy_left`` outwards (torn off if > TORN), all raised
    ``dz``."""
    out = {}
    v = np.array([0.0, 0.0, 0.05]) if moving else ZERO
    for spec, r in ((M.MIDDLE, M._R_ALONG_Y), (M.ENDS["right"], _rx(math.pi / 2)), (M.ENDS["left"], M._R_ALONG_Y)):
        p = np.array(spec.pos) + [0.0, 0.0, dz]
        if spec is M.ENDS["left"]:
            p = p + [0.0, dy_left, lifted]
        out[spec.name] = ck.body_state(spec, p, ck.matrix_to_quat_rel(r), v, ZERO)
    out[M.GIFT.name] = ck.body_state(M.GIFT, np.array(M.GIFT.pos) + [0.0, 0.0, dz], (0, 0, 0, 1), ZERO, ZERO)
    return out


def run(objs, steps=M.STILL_STEPS + 2):
    out = None
    for _ in range(steps):
        M.success(objs)
        out = M.score(objs)
    return out


# -- the parts -------------------------------------------------------------------------------------------------------

def test_the_cracker_is_a_standard_12_inch_one():
    length = 2 * (M.BODY_HALF + M.NECK_L + M.END_L)
    assert length == pytest.approx(0.30, abs=0.01) and 2 * M.TUBE_R == pytest.approx(0.05)
    assert M.NECK_R < M.TUBE_R  # crimped


def test_each_crimp_is_a_joint_that_tears_past_its_force_and_one_is_weaker():
    for side, e in M.ENDS.items():
        g = ck._GLUES[e.name]
        assert g.parent == M.MIDDLE.name and g.break_force == M.BREAK_F[side] and g.break_torque == M.BREAK_T
    assert M.BREAK_F["right"] < M.BREAK_F["left"]
    # lifted by its ends, the middle's own weight bends a crimp far less than it takes to tear it
    assert (M.M_BODY + M.M_GIFT) * 9.81 * M.BODY_HALF < 0.2 * M.BREAK_T


def test_the_middle_is_a_hollow_tube_round_the_gift():
    lo, hi = M.GIFT.bounds_local()
    assert np.max(np.abs(hi[[0, 2]])) < M.TUBE_R - 1.5 * MM


# -- scoring ---------------------------------------------------------------------------------------------------------

def test_whole_on_the_table_is_nothing():
    assert run(objects()) == 0.0


def test_held_up_whole_is_a_quarter():
    assert run(objects(dz=0.05)) == 0.25


def test_cracked_and_down_is_full_after_a_second():
    objs = objects(dy_left=0.06)
    assert M.cracked(objs)
    M.success(objs)
    assert M.score(objs) == 0.5
    assert run(objs) == 1.0


def test_success_ends_the_trial_only_after_the_second_held():
    objs = objects(dy_left=0.06)
    assert TASK.success_hold_s == 1.0
    assert not any(M.success(objs) for _ in range(M.STILL_STEPS - 1))
    assert M.success(objs) and M.score(objs) == 1.0


def test_cracked_but_still_held_up_is_half():
    assert run(objects(dy_left=0.06, dz=0.05)) == 0.5


def test_cracked_but_moving_is_half():
    assert run(objects(dy_left=0.06, moving=True)) == 0.5


def test_a_hand_near_a_piece_is_half(monkeypatch):
    objs = objects(dy_left=0.06)
    pt = np.array(M.MIDDLE.pos) + [0.0, 0.0, M.TUBE_R + 5 * MM]
    monkeypatch.setattr(ck, "finger_points", lambda robots=None, **kw: np.array([pt]))
    assert run(objs) == 0.5


def test_a_crimp_stretched_but_not_torn_is_not_cracked():
    assert not M.cracked(objects(dy_left=0.5 * M.TORN))


# -- the demo ------------------------------------------------------------------------------------------------------

PLANS = M.demo()
GUARD = np.array([0.287979, 0.1832595, 0.1570795, 0.1631885, 0.15708, 0.20944, 0.05])


@pytest.mark.parametrize("arm", ["right", "left"])
def test_the_demo_reaches_every_target_and_passes_the_guards(arm):
    plan = ArmPlan(PLANS[arm], station.BASES[arm])
    base = np.array(station.BASES[arm])
    prev = None
    for t in np.arange(0.0, plan.duration, 0.05):
        pos = plan.target(t)[0]
        a = plan.action7(t)[0]
        tcp = ik.tcp_pose(a[:6])
        tcp = np.asarray(tcp[0] if isinstance(tcp, tuple) else tcp[:3, 3]) + base
        assert np.linalg.norm(tcp - pos) < 2.0 * MM, (arm, t, plan.target(t)[3])
        if prev is not None:
            assert (np.abs(a - prev) / GUARD).max() < 0.95, (arm, t, plan.target(t)[3])
        prev = a


def test_the_demo_fits_the_run_limit_and_both_hands_pull_together():
    for arm in ("right", "left"):
        assert sum(w.duration for w in PLANS[arm]) * 20 < M.RUN_LIMIT_STEPS
    t = {arm: 0.0 for arm in PLANS}
    starts = {}
    for arm in PLANS:
        for w in PLANS[arm]:
            if w.label == "pull":
                starts[arm] = t[arm]
            t[arm] += w.duration
    assert starts["left"] == starts["right"]


def test_the_onehand_check_leaves_the_left_hand_home(monkeypatch):
    monkeypatch.setenv("XC_DEMO", "onehand")
    assert {tuple(w.pos) for w in M.demo()["left"]} == {M.LHOME}
