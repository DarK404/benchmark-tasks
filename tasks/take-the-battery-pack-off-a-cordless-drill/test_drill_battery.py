"""Offline checks of drill_battery.py's parts, scoring and demo, without the simulator. Needs yam_arena:

    python -m pytest -q test_drill_battery.py
"""

from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

TASK = load(Path(__file__).with_name("drill_battery.py"))
M = TASK.module
import drillkit as dk  # noqa: E402

MM = 0.001
ZERO = np.zeros(3)


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    M._reset_state()
    monkeypatch.setattr(dk, "finger_points", lambda robots=None, **kw: np.zeros((0, 3)))
    yield


R = dk.quat_to_matrix(M.Q0)


def objects(slide=0.0, pack_at=None, lift=0.0, drill_shift=(0.0, 0.0, 0.0)):
    """The layout, with the pack slid ``slide`` (m) forwards along the foot, or moved to ``pack_at`` (world, its frame
    origin), and the drill (with a seated pack) raised ``lift``."""
    dshift = np.asarray(drill_shift, float) + [0.0, 0.0, lift]
    out = {M.DRILL.name: dk.body_state(M.DRILL, np.asarray(M.DRILL.pos) + dshift, M.Q0, ZERO, ZERO),
           M.SHOE.name: dk.body_state(M.SHOE, np.asarray(M.SHOE.pos) + dshift, M.Q0, ZERO, ZERO)}
    if pack_at is not None:
        pshift = np.asarray(pack_at, float) - np.asarray(M.PACK.pos)
    else:
        pshift = dshift + R @ np.array([slide, 0.0, 0.0])
    for spec in (M.PACK, M.BUTTONS[1], M.BUTTONS[-1]):
        out[spec.name] = dk.body_state(spec, np.asarray(spec.pos) + pshift, M.Q0, ZERO, ZERO)
    return out


# where the demo leaves the pack: well clear of the drill, lying as laid down, on the table
CLEAR_OF_DRILL = np.asarray(M.PACK.pos) + R @ np.array([0.16, 0.0, 0.0])


def run(objs, steps=M.STILL_STEPS + 2):
    out = None
    for _ in range(steps):
        M.success(objs)
        out = M.score(objs)
    return out


# -- the parts -------------------------------------------------------------------------------------------------------

def test_the_pack_and_drill_are_the_named_products_sizes():
    assert (M.PK_L, M.PK_W, M.PK_H) == pytest.approx((0.118, 0.079, 0.055))  # 48-11-1820: 4.66 x 3.12 x 2.15 in
    assert M.M_PACK == pytest.approx(0.43, abs=0.01) and M.M_DRILL == pytest.approx(1.04, abs=0.01)  # 0.95 lb, 2.3 lb


def test_the_latch_holds_until_a_button_is_pushed_in_past_its_overlap():
    pawl_out, stop_in = M.PAWL_Y[1], M.STOP_Y - M.STOP[1] / 2
    assert pawl_out - stop_in == pytest.approx(M.LATCH_OVERLAP)  # at rest the stop is in the pawl's way
    assert M.LATCH_OVERLAP < M.BTN_TRAVEL  # a full push clears it
    assert M.STOP_X - M.STOP[0] / 2 > M.PAWL_X + 3 * MM - M.TONGUE_X0  # the stop is in front of the pawl


def test_the_rails_hook_under_the_tongues_lips():
    assert M.TONGUE_LOW_W / 2 < M.RAIL_IN < M.RAIL_OUT < M.TONGUE_LIP_W / 2 + 1 * MM
    assert M.RAIL_Z[1] < -M.PK_H + M.BASE_H + 5 * MM + M.PK_H - M.PK_H  # below the lips (z 50 mm on the pack)


def test_the_pack_starts_seated():
    assert M.on_drill(objects()) and abs(M.pack_offset(objects())[0]) < 1e-6


# -- scoring ---------------------------------------------------------------------------------------------------------

def test_the_start_is_nothing():
    assert run(objects()) == 0.0


def test_slid_part_way_along_the_rails_is_a_quarter():
    assert run(objects(slide=M.SLID_MIN + 5 * MM)) == 0.25
    M._reset_state()
    assert run(objects(slide=M.SLID_MIN - 5 * MM)) == 0.0


def test_off_and_down_is_full_after_a_second():
    objs = objects(pack_at=CLEAR_OF_DRILL)
    assert not M.on_drill(objs) and M.pack_down(objs) and M.drill_down(objs)
    M.success(objs)
    assert M.score(objs) == 0.5
    assert run(objs) == 1.0


def test_success_ends_the_trial_only_after_the_second_held():
    objs = objects(pack_at=CLEAR_OF_DRILL)
    assert TASK.success_hold_s == 1.0
    assert not any(M.success(objs) for _ in range(M.STILL_STEPS))
    assert M.success(objs) and M.score(objs) == 1.0


def test_off_but_held_up_is_half():
    assert run(objects(pack_at=CLEAR_OF_DRILL + [0.0, 0.0, 0.05])) == 0.5


def test_off_but_the_drill_still_held_up_is_half():
    objs = objects(pack_at=CLEAR_OF_DRILL)
    objs.update({k: v for k, v in objects(lift=0.05).items() if k in (M.DRILL.name, M.SHOE.name)})
    assert run(objs) == 0.5


def test_the_drill_carried_away_is_half():
    objs = objects(pack_at=CLEAR_OF_DRILL)
    objs.update({k: v for k, v in objects(drill_shift=(0.0, 0.2, 0.0)).items() if k in (M.DRILL.name, M.SHOE.name)})
    assert run(objs) == 0.5


def test_whole_drill_lifted_with_its_pack_is_nothing():
    assert run(objects(lift=0.05)) == 0.0


def test_a_hand_near_the_pack_is_half(monkeypatch):
    objs = objects(pack_at=CLEAR_OF_DRILL)
    pt = np.asarray(objs[M.PACK.name].to_world((0.0, 0.0, M.PK_H / 2))) + [0.0, 0.0, 0.05]
    monkeypatch.setattr(dk, "finger_points", lambda robots=None, **kw: np.array([pt]))
    assert run(objs) == 0.5


def test_moving_is_half():
    a, b = objects(pack_at=CLEAR_OF_DRILL), objects(pack_at=CLEAR_OF_DRILL + [2 * MM, 0.0, 0.0])
    out = None
    for k in range(M.STILL_STEPS + 4):
        objs = a if k % 2 else b
        M.success(objs)
        out = M.score(objs)
    assert out == 0.5


def test_a_fall_off_the_table_latches_zero():
    M.success(objects(pack_at=(0.3, -0.7, M.TOP - 0.3)))
    assert run(objects(pack_at=CLEAR_OF_DRILL)) == 0.0


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


def test_the_demo_fits_the_run_limit_and_holds_the_drill_while_the_pack_slides():
    for arm in ("right", "left"):
        assert sum(w.duration for w in PLANS[arm]) * 20 < M.RUN_LIMIT_STEPS

    def span(arm, label):
        t = 0.0
        for w in PLANS[arm]:
            if w.label == label:
                return t, t + w.duration
            t += w.duration
        raise KeyError(label)

    held_from = span("left", "grip it")[1]
    held_until = span("left", "hold it up")[1]
    slide = span("right", "slide it off")
    assert held_from <= slide[0] and slide[1] <= held_until


def test_the_nosqueeze_check_never_closes_on_the_buttons(monkeypatch):
    monkeypatch.setenv("DB_DEMO", "nosqueeze")
    fingers = {w.finger for w in M.demo()["right"]}
    assert M.B_SQ not in fingers and min(fingers) <= -(M.PK_W / 2 + M.BTN_PROUD - 1 * MM)
