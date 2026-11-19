"""Offline checks of tube_cutter.py's cutting model, scoring and demo, without the simulator. Needs yam_arena (the
official stack):

    python -m pytest -q test_tube_cutter.py
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

TASK = load(Path(__file__).with_name("tube_cutter.py"))
M = TASK.module
import cutterkit as ck  # noqa: E402  (the task file put its folder on sys.path)

ZERO = np.zeros(3)
MM = 0.001


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    M._reset_state()
    M.MODEL.clear()
    monkeypatch.setattr(ck, "finger_points", lambda robots=None, **kw: np.zeros((0, 3)))
    yield


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


class Rig:
    """Drives the model as the simulator would: the cutter's heading, and the knob's angle on it."""

    def __init__(self):
        self.m = M.MODEL
        self.m.knob_x = None  # (no fingertips offline: the model does not hold the knob)
        self.yaw, self.knob = M.YAW0, 0.0
        self.step()

    def step(self):
        rc = _rz(self.yaw)
        frames = {M.CUTTER.name: ck.Frame(pos=ZERO, rot=rc, vel=ZERO, spin=ZERO, com=ZERO),
                  M.KNOB.name: ck.Frame(pos=ZERO, rot=rc @ _rx(self.knob), vel=ZERO, spin=ZERO, com=ZERO),
                  M.OFFCUT.name: ck.Frame(pos=ZERO, rot=np.eye(3), vel=ZERO, spin=ZERO, com=ZERO)}
        self.m.update(frames, {})

    def tighten(self, turns, steps=20):
        for _ in range(steps):
            self.knob -= 2 * math.pi * turns / steps
            self.step()

    def lap(self, degrees=360.0, steps=120):
        for _ in range(steps):
            self.yaw += math.radians(degrees) / steps
            self.step()


@pytest.fixture(autouse=True)
def restore_knob():
    yield
    M.MODEL.knob_x = (M.KNOB_X0, M.KNOB_X1)


def test_the_knob_is_held_where_it_is_while_nothing_touches_it():
    m = M.MODEL
    frames = {M.CUTTER.name: ck.Frame(pos=ZERO, rot=_rz(M.YAW0), vel=ZERO, spin=ZERO, com=ZERO),
              M.KNOB.name: ck.Frame(pos=ZERO, rot=_rz(M.YAW0) @ _rx(-0.3), vel=ZERO, spin=ZERO, com=ZERO),
              M.OFFCUT.name: ck.Frame(pos=ZERO, rot=np.eye(3), vel=ZERO, spin=ZERO, com=ZERO)}
    m.update(frames, {})
    assert M.KNOB.name in m.poses and m.held_k == pytest.approx(-0.3)


def test_the_forces_are_frictions_on_the_two_hinges():
    hinges = {h.child: h for h in ck._HINGES.values()}
    assert hinges[M.CUTTER.name].max_torque == M.CUT_TORQUE
    assert hinges[M.KNOB.name].max_torque == M.KNOB_TORQUE
    # the knob turns before the whole cutter does when its top is rolled
    assert M.KNOB_TORQUE / M.KNOB_R < M.CUT_TORQUE / ((M.KNOB_X0 + M.KNOB_X1) / 2)


def states(offcut=None, moving=False):
    st = {}
    for spec in M.OBJECTS:
        quat = spec.quat if spec.quat else ck.yaw_quat(spec.yaw)
        pos = np.array(spec.pos, float)
        v = np.array([0.0, 0.0, 0.05]) if moving else ZERO
        if spec.name == M.OFFCUT.name and offcut is not None:
            pos = np.array(offcut, float)
        st[spec.name] = ck.body_state(spec, pos, quat, v if spec.name == M.OFFCUT.name else ZERO, ZERO)
    return st


DOWN = (M.TX - 0.06, M.TY + 0.12, M.TOP)  # the offcut stood on the table


def held(st, n=M.STILL_STEPS):
    for _ in range(n):
        ok = M.success(st)
        res = (M.score(st), ok)
    return res


# -- the cutting model --------------------------------------------------------------------------------------------
def test_the_feed_screw_is_7_16_14():
    assert M.PITCH == pytest.approx(1.814 * MM, abs=0.001 * MM)


def test_a_quarter_turn_a_lap_cuts_through_on_the_third_lap():
    r = Rig()
    for lap in range(3):
        r.tighten(0.25)
        assert not r.m.through
        r.lap()
    assert r.m.through and not r.m.dented
    assert r.m.max_bite == pytest.approx(M.QUARTER, abs=0.01 * MM)


def test_each_lap_cuts_the_groove_to_the_wheel_all_round():
    r = Rig()
    r.tighten(0.25)
    r.lap()
    assert r.m.groove.min() == pytest.approx(M.QUARTER, abs=1e-9)


def test_part_of_a_lap_cuts_only_the_part_passed():
    r = Rig()
    r.tighten(0.25)
    r.lap(180.0)
    assert r.m.groove.max() == pytest.approx(M.QUARTER) and r.m.groove.min() == 0.0


def test_turning_the_knob_without_going_round_cuts_nothing():
    r = Rig()
    r.tighten(0.25)
    assert r.m.groove.max() == 0.0


def test_jiggling_the_cutter_in_place_cuts_nothing_so_cranking_still_dents():
    r = Rig()
    for _ in range(3):
        r.tighten(0.25)
        r.lap(1.0, steps=4); r.lap(-1.0, steps=4)
    assert r.m.groove.max() == 0.0 and r.m.dented


def test_a_full_turn_at_once_dents_the_tube_even_without_going_round():
    r = Rig()
    r.tighten(1.0)
    assert r.m.dented


def test_half_a_turn_at_once_is_still_allowed_but_more_dents():
    r = Rig()
    r.tighten(0.49)
    assert not r.m.dented
    r.tighten(0.03)
    assert r.m.dented


def test_turning_either_way_round_cuts():
    r = Rig()
    r.tighten(0.25)
    r.lap(-360.0)
    assert r.m.groove.min() == pytest.approx(M.QUARTER)


def test_loosening_never_cuts():
    r = Rig()
    r.tighten(-0.5)
    r.lap()
    assert r.m.groove.max() == 0.0 and not r.m.dented


def test_the_offcut_is_held_until_cut_through_then_let_go():
    r = Rig()
    r.tighten(0.25)
    r.lap()
    assert M.OFFCUT.name in r.m.poses
    r.tighten(0.25); r.lap(); r.tighten(0.25); r.lap()
    assert r.m.through and M.OFFCUT.name not in r.m.poses


# -- scoring ------------------------------------------------------------------------------------------------------
def _cut(laps=3):
    r = Rig()
    for _ in range(laps):
        r.tighten(0.25)
        r.lap()
    return r


def test_cut_through_with_the_offcut_set_down_succeeds():
    _cut()
    assert held(states(offcut=DOWN)) == (1.0, True)


def test_full_only_after_a_second_held():
    _cut()
    st = states(offcut=DOWN)
    assert held(st, M.STILL_STEPS - 1)[0] == 0.5
    assert held(st, 1)[0] == 1.0


def test_success_ends_the_trial_only_after_the_second_held():
    _cut()
    st = states(offcut=DOWN)
    assert TASK.success_hold_s == 1.0  # and yam_arena holds it another 1 s before ending the trial
    assert not any(M.success(st) for _ in range(M.STILL_STEPS - 1))
    assert M.success(st) and M.score(st) == 1.0


def test_the_instruction_names_the_length_cut_off():
    import json
    task = json.loads(Path(__file__).with_name("task.json").read_text())
    assert "top 4 cm" in task["instruction"]
    assert M.OFFCUT_L == pytest.approx(40.0 * MM)


def test_cut_through_with_the_offcut_still_on_is_half():
    _cut()
    assert held(states()) == (0.5, False)


def test_the_offcut_still_moving_is_half():
    _cut()
    assert held(states(offcut=DOWN, moving=True)) == (0.5, False)


def test_a_hand_on_the_cutter_is_half(monkeypatch):
    _cut()
    on = np.array([[M.TX, M.TY - 0.05, M.TOP + M.Z_MARK]])
    monkeypatch.setattr(ck, "finger_points", lambda robots=None, **kw: on)
    assert held(states(offcut=DOWN)) == (0.5, False)


def test_one_lap_deep_is_a_quarter_and_less_is_nothing():
    _cut(1)
    assert held(states())[0] == 0.25
    M.MODEL.clear()
    r = Rig()
    r.tighten(0.25)
    r.lap(300.0)
    assert held(states())[0] == 0.0


def test_a_dent_scores_nothing_even_once_cut_and_set_down():
    r = Rig()
    r.tighten(0.75)  # three quarter turns at once
    r.lap(); r.tighten(0.25); r.lap()
    assert r.m.through and r.m.dented
    assert held(states(offcut=DOWN)) == (0.0, False)


def test_untouched_is_nothing():
    assert held(states()) == (0.0, False)


def test_the_offcut_left_standing_on_the_stub_is_not_off():
    assert not M.offcut_off(states())
    assert M.offcut_off(states(offcut=DOWN))


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


def test_the_demo_turns_the_knob_a_quarter_turn_once_a_lap_before_going_round():
    d = M.demo()
    labels = [w.label for w in d["right"]]
    assert labels.count("roll it a quarter turn") == M.LAPS and labels.count("and on") == M.LAPS * (M.STROKES - 1)
    assert M.ROLL == pytest.approx(0.25 * 2 * math.pi * M.KNOB_R)


def test_the_hands_keep_apart():
    """Both low: the grippers' centres stay over 8 cm apart."""
    d = M.demo()
    lp, rp = ArmPlan(d["left"], station.BASES["left"]), ArmPlan(d["right"], station.BASES["right"])
    for t in np.arange(0.0, min(lp.duration, rp.duration), 0.05):
        a, b = np.asarray(lp.target(t)[0]), np.asarray(rp.target(t)[0])
        if a[2] < M.TOP + 0.17 and b[2] < M.TOP + 0.17:
            assert np.linalg.norm(a[:2] - b[:2]) > 0.08, t


def test_laps_without_feeding_cut_nothing():
    r = Rig()
    for _ in range(3):
        r.lap()
    assert r.m.groove.max() == 0.0 and not r.m.through
