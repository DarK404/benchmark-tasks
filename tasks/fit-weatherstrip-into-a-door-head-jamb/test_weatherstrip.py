"""Offline checks of weatherstrip.py's scoring and demo, without the simulator. Needs yam_arena (the official stack):

    python -m pytest -q test_weatherstrip.py
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

TASK = load(Path(__file__).with_name("weatherstrip.py"))
M = TASK.module
import weatherkit as wk  # noqa: E402  (the task file put its folder on sys.path)

ZERO = np.zeros(3)
MM = 0.001
Q = wk.yaw_quat(math.pi / 2)


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    M._reset_state()
    monkeypatch.setattr(wk, "finger_points", lambda robots=None: np.zeros((0, 3)))
    yield


def states(depth=None, shift=0.0, out=(), moving=False, laid=True):
    """The strip straight along the kerf, each link's fin ``depth`` into it (default: home, the bulb on the face),
    the whole strip moved ``shift`` along y; links in ``out`` pulled 5 mm out of the kerf. ``laid=False``: the strip
    where it starts, loose in the rabbet."""
    st = {}
    for spec in M.OBJECTS:
        quat = spec.quat if spec.quat else wk.yaw_quat(spec.yaw)
        st[spec.name] = wk.body_state(spec, np.array(spec.pos, float), quat, ZERO, ZERO)
    if not laid:
        return st
    for i, s in enumerate(M.STRIP):
        d = (M.FIN_X if depth is None else depth) - (5.0 * MM if i in out else 0.0)
        x = M.STOP_X + d - M.FIN_X - M.BULB_X / 2
        p = np.array([x, M.JY0 + (i + 0.5) * M.LINK + shift, M.TOP + M.KERF_Z])
        v = np.array([0.0, 0.0, 0.02]) if moving else ZERO
        st[s.name] = wk.body_state(s, p, Q, v, ZERO)
    return st


def held(st, n=M.STILL_STEPS):
    for _ in range(n):
        ok = M.success(st)
        res = (M.score(st), ok)
    return res


# -- geometry -------------------------------------------------------------------------------------------------------
def test_the_fin_fits_the_kerf_and_lies_at_its_height():
    assert M.FIN_Z < M.KERF_W
    assert M.KERF_Z == pytest.approx(M.JT + M.BULB_Z / 2)
    assert M.FIN_X == pytest.approx(M.KERF_D)


def test_the_strip_is_cut_to_the_jambs_length():
    assert M.N_LINKS * M.LINK == pytest.approx(M.JL)
    assert M.ends(states()) == pytest.approx((M.JY0, M.JY1), abs=1e-9)


def test_the_rabbet_floor_is_a_door_thickness_wide():
    assert M.STOP_X - M.JX0 == pytest.approx(0.048, abs=1e-6)  # a 1-3/4 in (44.5 mm) door and its play


def test_a_home_fin_reads_its_full_depth():
    st = states()
    assert M.fin_depth(st, M.STRIP[5]) == pytest.approx(M.KERF_D, abs=1e-9)


def test_the_strip_where_it_starts_is_not_in_the_kerf_and_overhangs_the_right_end():
    st = states(laid=False)
    assert not any(M.seated(st))
    assert all(-0.007 < M.fin_depth(st, s) < -0.005 for s in M.STRIP[1:])  # (link 0 overhangs the end)
    assert M.ends(st) == pytest.approx((M.JY0 - 0.020, M.JY1 - 0.020), abs=1e-9)


# -- scoring ------------------------------------------------------------------------------------------------------
def test_home_all_along_and_flush_succeeds():
    assert held(states()) == (1.0, True)


def test_full_only_after_a_second_held():
    st = states()
    assert held(st, M.STILL_STEPS - 1)[0] == 0.5
    assert held(st, 1)[0] == 1.0


@pytest.mark.parametrize("shift", [-0.0019, 0.0019])
def test_ends_within_2_mm_count_as_flush(shift):
    assert held(states(shift=shift))[0] == 1.0


@pytest.mark.parametrize("shift", [-0.004, 0.004])
def test_an_end_off_by_more_than_2_mm_is_half(shift):
    assert held(states(shift=shift)) == (0.5, False)


def test_one_link_not_home_is_half():
    assert held(states(out=(10,))) == (0.5, False)


def test_ninety_percent_home_is_half_but_less_is_a_quarter():
    assert held(states(out=(1, 2, 3)))[0] == 0.5  # 31 of 34
    assert held(states(out=(1, 2, 3, 4)))[0] == 0.25  # 30 of 34


def test_a_quarter_home_is_a_quarter_and_less_is_nothing():
    assert held(states(out=tuple(range(9, 34))))[0] == 0.25  # 9 of 34
    assert held(states(out=tuple(range(8, 34))))[0] == 0.0  # 8


def test_fins_part_way_in_are_not_home():
    assert held(states(depth=M.SEAT - 0.5 * MM)) == (0.0, False)
    assert held(states(depth=M.SEAT))[0] == 1.0


def test_still_moving_is_half():
    assert held(states(moving=True)) == (0.5, False)


def test_a_hand_still_on_the_strip_is_half(monkeypatch):
    on = np.array([[M.SEATED_X - M.BULB_X / 2 - 3.0 * MM, 0.0, M.TOP + M.JT + 2.0 * MM]])
    monkeypatch.setattr(wk, "finger_points", lambda robots=None: on)
    assert held(states()) == (0.5, False)


def test_a_hand_1_cm_off_is_clear(monkeypatch):
    off = np.array([[M.SEATED_X - M.BULB_X / 2 - 10.5 * MM, 0.0, M.TOP + M.JT + 2.0 * MM]])
    monkeypatch.setattr(wk, "finger_points", lambda robots=None: off)
    assert held(states())[0] == 1.0


def test_untouched_is_nothing():  # the strip loose in the rabbet
    assert held(states(laid=False)) == (0.0, False)


def test_a_strip_laid_on_the_floor_but_not_pressed_is_nothing():
    assert held(states(depth=-2.0 * MM)) == (0.0, False)


# -- the kerf model ------------------------------------------------------------------------------------------------
def _frame(st):
    return wk.Frame(pos=st.pos.copy(), rot=st.rotation(), vel=ZERO, spin=ZERO, com=st.pos.copy())


def test_the_model_holds_a_link_once_its_fin_is_home_and_puts_it_back_if_it_drifts():
    k = wk.Kerf([s.name for s in M.STRIP], M.FIN_TIP, M.STOP_X, M.KERF_Z + M.TOP, M.JY0 - M.FLUSH, M.JY1 + M.FLUSH,
                M.SEAT, home=M.MODEL.home)
    st = states(out=(3,))
    frames = {s.name: _frame(st[s.name]) for s in M.STRIP}
    k.update(frames, {})
    assert M.STRIP[3].name not in k.locked and len(k.locked) == M.N_LINKS - 1
    moved = frames[M.STRIP[5].name]
    moved.pos = moved.pos - np.array([0.003, 0.0, 0.0])  # pulled 3 mm back out
    k.update(frames, {})
    assert M.STRIP[5].name in k.poses
    assert np.allclose(k.poses[M.STRIP[5].name][0], st[M.STRIP[5].name].pos)


def test_the_model_does_not_hold_the_strip_where_it_starts():
    k = wk.Kerf([s.name for s in M.STRIP], M.FIN_TIP, M.STOP_X, M.KERF_Z + M.TOP, M.JY0 - M.FLUSH, M.JY1 + M.FLUSH,
                M.SEAT, home=M.MODEL.home)
    st = states(laid=False)
    k.update({s.name: _frame(st[s.name]) for s in M.STRIP}, {})
    assert not k.locked


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


def test_the_hands_keep_apart():
    """The two grippers' centres stay over 12 cm apart along y whenever both are low (the housings are 6.5 cm
    across along y at these yaws)."""
    d = M.demo()
    lp, rp = ArmPlan(d["left"], station.BASES["left"]), ArmPlan(d["right"], station.BASES["right"])
    for t in np.arange(0.0, min(lp.duration, rp.duration), 0.05):
        a, b = lp.target(t)[0], rp.target(t)[0]
        if a[2] < M.TOP + 0.10 and b[2] < M.TOP + 0.10:
            assert a[1] - b[1] > 0.12, t


def test_the_demo_presses_every_link():
    labels = [w.label for p in M.demo().values() for w in p]
    assert sorted(int(lab.split()[-1]) for lab in labels if lab.startswith("over link")) == list(range(M.N_LINKS))
