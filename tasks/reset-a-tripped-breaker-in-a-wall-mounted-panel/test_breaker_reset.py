"""Offline checks of breaker_reset.py's mechanism geometry, scoring and demo, without the simulator. Needs yam_arena:

    python -m pytest -q test_breaker_reset.py
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

TASK = load(Path(__file__).with_name("breaker_reset.py"))
M = TASK.module
import resetkit as rk  # noqa: E402

MM = 0.001
ZERO = np.zeros(3)
DEG = math.radians


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    M._reset_state()
    monkeypatch.setattr(rk, "finger_points", lambda robots=None, **kw: np.zeros((0, 3)))
    yield


def objects(phis=None, pawl_in=0.0):
    """Every handle at its throw (rad; default: as spawned), the pawl moved ``pawl_in`` in over the ledge."""
    phis = {s: M.phi_start(s) for s in M.HANDLES} | (phis or {})
    out = {}
    for s, h in M.HANDLES.items():
        out[h.name] = rk.body_state(h, np.array(h.pos), rk.matrix_to_quat_rel(M._body_rot(s, phis[s])), ZERO, ZERO)
    _, r = M.breaker_frame(M.TRIPPED)
    p = M.to_env(M.b_to_panel(M.TRIPPED, M._PAWL0)) - r @ np.array([0.0, pawl_in, 0.0])
    out[M.PAWL.name] = rk.body_state(M.PAWL, p, rk.matrix_to_quat_rel(r), ZERO, ZERO)
    return out


def run(objs, steps=M.STILL_STEPS + 2):
    out = None
    for _ in range(steps):
        M.success(objs)
        out = M.score(objs)
    return out


# -- the mechanism (analytic checks of the geometry the simulator runs) ---------------------------------------------

def _notch_lift(phi):
    """How far the pin's leading top corner would rise into the cradle's notch at throw ``phi`` (m): the cradle must
    lift this much to let the handle reach it."""
    y = M.ARM_L * math.sin(phi)
    z_top = -M.ARM_L * math.cos(phi) + M.PIN_H / 2
    lead = abs(y) - M.PIN_H / 2
    slope = math.tan(DEG(M.V_OFF_DEG)) if phi > 0 else math.tan(DEG(M.V_ON_DEG))
    return z_top - (-M.ARM_L + M.PIN_H / 2) + max(lead, 0.0) * slope


def test_pushed_fully_to_off_the_cradle_lifts_past_its_latch():
    assert _notch_lift(M.PHI_MAX) > M.LATCH_D + 0.8 * MM
    assert _notch_lift(M.PHI_MAX) < M.CR_UP  # within the cradle's travel


def test_pushed_to_on_the_cradle_never_reaches_its_latch():
    assert _notch_lift(-M.PHI_MAX) < M.LATCH_D - 0.5 * MM


def test_unlatched_the_notch_centres_the_handle_against_the_toggle():
    # at the ON stop the notch's push on the pin turns the handle back towards OFF, harder than the toggle throws it on
    phi = -M.PHI_MAX
    y, z = M.ARM_L * math.sin(phi), -M.ARM_L * math.cos(phi)
    a = DEG(M.V_ON_DEG)
    n = np.array([math.sin(a), -math.cos(a)])  # the notch's ON side presses down and towards +y
    force = M.CR_F + M.CR_K * _notch_lift(phi)  # the cradle's spring, lifted by the pin
    torque_notch = y * n[1] * force - z * n[0] * force  # about the pin's axis (x)
    plunger = M.PL_K * (M.PL_UP + M.PL_F / M.PL_K - M.ARM_L * (1 - math.cos(phi)))
    torque_toggle = abs(y) * plunger
    assert torque_notch > 2.0 * torque_toggle


def test_the_toggle_throws_the_handle_to_either_end():
    sp = rk._SPRINGS[M.PLUNGERS[1].name]
    assert sp.target > sp.upper  # preloaded up against the arm
    for s, h in M.HANDLES.items():
        hi = rk._HINGES[h.name]
        assert (hi.upper - hi.lower) == pytest.approx(2 * M.PHI_MAX)


def test_the_pawl_and_cradle_springs():
    cr, pw = rk._SPRINGS[M.CRADLE.name], rk._SPRINGS[M.PAWL.name]
    assert cr.target < 0 and (cr.lower, cr.upper) == (0.0, M.CR_UP)  # pressing down on its stop
    assert pw.target < pw.lower  # preloaded in towards the ledge
    assert M.PAWL_TRAVEL > 2.0 * MM  # it can slip in under the ledge


def test_three_breakers_on_and_the_tripped_one_midway():
    for s in M.HANDLES:
        assert M.phi_start(s) == (0.0 if s == M.TRIPPED else -M.PHI_MAX)


# -- scoring ---------------------------------------------------------------------------------------------------------

def test_as_it_starts_is_nothing():
    assert run(objects()) == 0.0


def test_reset_and_on_is_full_after_a_second():
    objs = objects({M.TRIPPED: -M.PHI_MAX + DEG(1)}, pawl_in=2.0 * MM)
    M.success(objs)
    assert M.score(objs) == 0.5
    assert run(objs) == 1.0


def test_success_ends_the_trial_only_after_the_second_held():
    objs = objects({M.TRIPPED: -M.PHI_MAX}, pawl_in=2.0 * MM)
    assert TASK.success_hold_s == 1.0
    assert not any(M.success(objs) for _ in range(M.STILL_STEPS - 1))
    assert M.success(objs) and M.score(objs) == 1.0


def test_relatched_but_left_off_is_half():
    assert run(objects({M.TRIPPED: DEG(15)}, pawl_in=2.0 * MM)) == 0.5


def test_pushed_to_on_without_relatching_is_nothing():
    assert run(objects({M.TRIPPED: -M.PHI_MAX}, pawl_in=0.0)) == 0.0


def test_switching_another_breaker_off_is_zero_for_the_rest_of_the_run():
    other = next(s for s in M.HANDLES if s != M.TRIPPED)
    M.success(objects({other: DEG(20)}))
    assert run(objects({M.TRIPPED: -M.PHI_MAX}, pawl_in=2.0 * MM)) == 0.0  # put back ON: still zero


def test_a_hand_on_a_handle_is_half(monkeypatch):
    objs = objects({M.TRIPPED: -M.PHI_MAX}, pawl_in=2.0 * MM)
    st = objs[M.HANDLES[M.TRIPPED].name]
    pt = st.to_world((0.0, 0.0, M.LEVER_TOP))
    monkeypatch.setattr(rk, "finger_points", lambda robots=None, **kw: np.array([pt]))
    assert run(objs) == 0.5


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


def test_the_demo_pushes_off_before_on_and_fits_the_run_limit():
    labels = [w.label for w in PLANS["right"]]
    assert labels.index("push it to OFF") < labels.index("push it to ON")
    assert sum(w.duration for w in PLANS["right"]) * 20 < M.RUN_LIMIT_STEPS
    assert {tuple(w.pos) for w in PLANS["left"]} == {M.LHOME}


def test_the_on_check_never_pushes_off(monkeypatch):
    monkeypatch.setattr(M, "DEMO", "on")
    labels = [w.label for w in M.demo()["right"]]
    assert "push it to OFF" not in labels and "push it to ON" in labels
