"""Offline checks of breaker.py's geometry, scoring and demo, without the simulator. Needs yam_arena (the official
stack):

    python -m pytest -q test_breaker.py
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

TASK = load(Path(__file__).with_name("breaker.py"))
M = TASK.module
import breakerkit as bk  # noqa: E402  (the task file put its folder on sys.path)

MM = 0.001
ZERO = np.zeros(3)


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    M._reset_state()
    monkeypatch.setattr(bk, "finger_points", lambda robots=None, **kw: np.zeros((0, 3)))
    yield


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def breaker_at(space=M.TARGET, dx=0.0, dy=0.0, dz=0.0, tip=0.0, yaw=0.0, moving=False):
    """The breaker's state: seated in ``space``, then moved (dx, dy, dz along its seated frame), tipped ``tip`` (rad)
    line end up about its foot's outer bottom edge, and turned ``yaw`` about the vertical."""
    o, y0 = M.seat_pose(space)
    r0 = M._rot(y0)
    piv = np.array([0.0, -M.BL / 2, 0.0])
    r = r0 @ M._rot(yaw) @ _rx(tip)
    origin = o + r0 @ piv - r @ piv + r0 @ np.array([dx, dy, dz])
    vel = np.array([0.0, 0.0, 0.05]) if moving else ZERO
    return bk.body_state(M.BREAKER, origin, bk.matrix_to_quat_rel(r), vel, ZERO)


def objects(st):
    return {M.BREAKER.name: st}


def run(st, steps=M.STILL_STEPS + 2):
    out = None
    for _ in range(steps):
        M.success(objects(st))
        out = M.score(objects(st))
    return out


# -- the parts -------------------------------------------------------------------------------------------------------

def test_the_breaker_is_a_homeline_hom120():
    assert M.BW == pytest.approx(25.4 * MM)
    assert M.BL == pytest.approx(79.5 * MM)
    assert M.B_BODY + M.B_FACE + M.B_HANDLE == pytest.approx(74.0 * MM, abs=2 * MM)  # 75.7 mm deep overall


def test_the_load_center_is_a_hom48l125gc_with_1_in_spaces():
    assert (M.ENC_L, M.ENC_W, M.ENC_D) == pytest.approx((376 * MM, 226 * MM, 97 * MM))
    assert M.PITCH == pytest.approx(25.4 * MM)
    assert M.ROW_X[1] - M.ROW_X[2] == pytest.approx(M.PITCH)


def test_spaces_1_and_3_hold_breakers_and_2_and_4_are_empty():
    # space 4 is the right-hand column (-y), the row nearer the robot; its line end points at the bus (+y)
    o, yaw = M.seat_pose(4)
    assert o[1] < M.PY and o[0] < M.PX and yaw == 0.0
    o3, yaw3 = M.seat_pose(3)
    assert o3[1] > M.PY and yaw3 == pytest.approx(math.pi)
    n_fittings = 4 * len(M._space_fittings(4))
    housing = len(M._breaker_housing(jaw_slot=False))
    assert len(M.PANEL.shapes) == 9 + n_fittings + 2 * housing


def test_the_jaw_must_spread_to_take_the_stab():
    gap = 2 * M.GAP
    assert gap == pytest.approx(2.0 * MM)
    assert M.STAB_T - gap == pytest.approx(1.2 * MM)  # 0.6 mm a shoe
    assert M.JAW_TRAVEL > (M.STAB_T - gap) / 2 + 2.0 * MM  # room to take an off-centre stab
    # the shoes' springs: preloaded against their stops, pressing inwards
    sa, sb = bk._SPRINGS[M.JAW_A.name], bk._SPRINGS[M.JAW_B.name]
    assert sa.target == pytest.approx(-M.JAW_PRELOAD / M.JAW_K) and sb.target == pytest.approx(M.JAW_PRELOAD / M.JAW_K)
    assert (sa.lower, sa.upper) == (0.0, M.JAW_TRAVEL) and (sb.lower, sb.upper) == (-M.JAW_TRAVEL, 0.0)


def test_the_jaw_grips_the_stab_with_a_spring_force_and_friction():
    spread = (M.STAB_T - 2 * M.GAP) / 2
    grip = M.JAW_PRELOAD + M.JAW_K * spread
    assert 10.0 < grip < 20.0  # N a side
    slide = 2 * M.JAW_MU * grip
    assert 5.0 < slide < 15.0  # N to slide the jaw along the blade
    # the stab's shapes have the jaw's friction
    assert all(bk._SHAPE_FRICTION[M.PANEL.name][i] == M.JAW_MU for i in M.STAB_SHAPES)
    assert M.JAW_A.friction == M.JAW_B.friction == M.JAW_MU


def test_the_stab_fits_in_the_jaw_slot():
    assert M.STAB_TOP < M.SLOT_Z  # its crest stays under the housing
    assert M.SLOT_Y0 < M.STAB_Y0 and M.STAB_Y1 < M.SLOT_Y1
    assert M.STAB_T / 2 < M.SLOT_X
    assert M.STAB_TOP == pytest.approx(M.STAB_H + (M.STAB_T / 2) / math.tan(math.radians(M.RIDGE_DEG)))


def test_the_hook_takes_the_rails_lip_with_clearance():
    assert M.FOOT_Z < M.LIP_Z0 and M.LIP_Z1 < M.HOOK_Z1  # 1 mm under and over the lip
    assert M.HOOK_Z1 - M.LIP_Z1 == pytest.approx(1.0 * MM) and M.LIP_Z0 - M.FOOT_Z == pytest.approx(1.0 * MM)
    assert M.LIP_Y1 < M.HOOK_Y1  # the lip's tip short of the slot's back wall
    assert -M.BL / 2 > M.RAIL_Y1  # the load end clear of the rail


def test_the_line_ends_are_apart_over_the_bus():
    gap = 2 * (M.SEAT_Y - M.BL / 2)
    assert gap == pytest.approx(10.0 * MM)


# -- scoring ---------------------------------------------------------------------------------------------------------

def test_seated_hooked_and_still_is_full_after_a_second():
    st = breaker_at()
    assert M.seated(st) and M.hooked(st) and M.engaged(st)
    M.success(objects(st))
    assert M.score(objects(st)) == 0.5  # not yet held 1 s
    assert run(st) == 1.0


def test_it_may_sit_anywhere_its_play_allows():
    for dy in (-1.0 * MM, 0.0, 2.0 * MM):
        M._reset_state()
        assert run(breaker_at(dy=dy)) == 1.0, dy


def test_a_hand_on_it_is_half(monkeypatch):
    st = breaker_at()
    top = st.to_world((0.0, 30 * MM, M.B_BODY + 2 * MM))
    monkeypatch.setattr(bk, "finger_points", lambda robots=None, **kw: np.array([top]))
    assert run(st) == 0.5


def test_still_moving_is_half():
    assert run(breaker_at(moving=True)) == 0.5


def test_the_jaw_part_way_on_is_half():
    # tipped about the hook so the line end stands 2 mm proud: the jaw on the blade but not home
    tip = math.asin(2.0 * MM / (M.BL / 2 + (M.SHOE_Y0 + M.SHOE_Y1) / 2))
    st = breaker_at(tip=tip)
    line, load = M.heights(st)
    assert line == pytest.approx(2.0 * MM, abs=0.1 * MM) and abs(load) < 0.5 * MM
    assert M.hooked(st) and M.engaged(st) and not M.seated(st)
    assert run(st) == 0.5


def test_resting_on_the_stab_unpushed_is_a_quarter():
    # hooked, its jaw resting on the stab's crest (about 12 degrees; it sits 1.5 mm towards the bus, as in the sim)
    tip = math.asin((M.STAB_TOP - M.SHOE_Z0) / (M.BL / 2 + (M.SHOE_Y0 + M.SHOE_Y1) / 2))
    st = breaker_at(dy=1.5 * MM, tip=tip)
    assert M.hooked(st) and not M.engaged(st)
    assert run(st) == 0.25


def test_pushed_on_without_the_hook_is_nothing():
    # level on the stab but 5 mm towards the bus (the foot not under the lip); the stab stops a real one from
    # going more than 2 mm, so this is also what a breaker set down with its load end on top of the lip looks like
    st = breaker_at(dy=5.0 * MM)
    assert not M.hooked(st)
    assert run(st) == 0.0
    st = breaker_at(dz=M.LIP_Z1 - M.FOOT_Z + 0.5 * MM, tip=0.0)  # load end up on the lip
    assert not M.seated(st)
    assert run(st) <= 0.25


def test_in_the_wrong_space_is_nothing():
    assert run(breaker_at(space=2)) == 0.0


def test_not_square_is_not_full():
    st = breaker_at(yaw=math.radians(3.0))
    assert not M.seated(st)
    assert run(st) < 1.0


def test_where_it_starts_is_nothing():
    st = bk.body_state(M.BREAKER, np.array(M.BREAKER.pos), bk.yaw_quat(M.START_YAW), ZERO, ZERO)
    assert run(st) == 0.0


# -- the demo ------------------------------------------------------------------------------------------------------

PLANS = M.demo()
GUARD = np.array([0.287979, 0.1832595, 0.1570795, 0.1631885, 0.15708, 0.20944, 0.05])  # rad (finger: m) per step


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


def test_the_demo_hooks_before_it_presses_and_lets_go_in_between():
    labels = [w.label for w in PLANS["right"]]
    assert labels.index("hook under the lip") < labels.index("let go") < labels.index("push it home")
    press = PLANS["right"][labels.index("push it home")]
    assert press.finger == M.FINGER_CLOSED  # closed fingertips on the top, nothing held


def test_the_demo_turns_the_breaker_a_quarter_turn_line_end_to_the_bus():
    labels = [w.label for w in PLANS["right"]]
    take = PLANS["right"][labels.index("take it")]
    turned = PLANS["right"][labels.index("hook under the lip")]
    assert float(take.yaw) == 0.0 and float(turned.yaw) == pytest.approx(math.pi / 2)
    # turning +90 degrees takes the line end from +x (as it lies) to +y (at the bus)
    assert (M._rot(math.pi / 2) @ M._rot(M.START_YAW) @ np.array([0.0, 1.0, 0.0]))[1] == pytest.approx(1.0)


def test_the_left_hand_stays_home():
    assert {tuple(w.pos) for w in PLANS["left"]} == {M.LHOME}


def test_the_nohook_check_never_hooks(monkeypatch):
    monkeypatch.setenv("BK_DEMO", "nohook")
    labels = [w.label for w in M.demo()["right"]]
    assert "hook under the lip" not in labels and "onto the stab" in labels
