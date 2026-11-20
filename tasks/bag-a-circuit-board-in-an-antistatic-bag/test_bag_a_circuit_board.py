"""Offline checks of bag_a_circuit_board.py's parts, scoring and demo, without the simulator. Needs yam_arena:

    python -m pytest -q test_bag_a_circuit_board.py
"""

from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

TASK = load(Path(__file__).with_name("bag_a_circuit_board.py"))
M = TASK.module
import bagkit as bk  # noqa: E402

MM = 0.001
ZERO = np.zeros(3)
Q0 = (0.0, 0.0, 0.0, 1.0)


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    M._reset_state()
    monkeypatch.setattr(bk, "finger_points", lambda robots=None, **kw: np.zeros((0, 3)))
    yield


def objects(bag_at=None, board_rel=None, board_world=None, top_lift=3.0 * MM):
    """The bag as laid down (shifted so its bottom film's centre is at ``bag_at``, with the top film raised
    ``top_lift`` as over a board); the board at ``board_rel`` (bag frame, m) or ``board_world``, flat, along x."""
    shift = np.zeros(3) if bag_at is None else np.asarray(bag_at, float) - np.asarray(M.BC.pos, float)
    out = {}
    for spec in M.BAG:
        up = np.array([0.0, 0.0, top_lift]) if spec in (M.T_L, M.T_R) else ZERO
        out[spec.name] = bk.body_state(spec, np.asarray(spec.pos) + shift + up, spec.quat, ZERO, ZERO)
    if board_rel is not None:
        p = np.asarray(M.BC.pos) + shift + np.asarray(board_rel, float)
    elif board_world is not None:
        p = np.asarray(board_world, float)
    else:
        p = np.asarray(M.BOARD.pos)
    out[M.BOARD.name] = bk.body_state(M.BOARD, p, Q0, ZERO, ZERO)
    for spec in (M.MAT, M.TOTE):
        out[spec.name] = bk.body_state(spec, spec.pos, Q0, ZERO, ZERO)
    return out


IN_BAG = (-11.5 * MM, 0.0, M.FILM_T / 2 + M.PCB_T / 2)  # where the demo leaves it
FLOOR_BAG = (M.TOTE_X, M.TOTE_Y, M.TOP + M.TOTE_FLOOR + M.FILM_T / 2)


def run(objs, steps=M.STILL_STEPS + 2):
    out = None
    for _ in range(steps):
        M.success(objs)
        out = M.score(objs)
    return out


# -- the parts -------------------------------------------------------------------------------------------------------

def test_the_bag_is_a_3_by_5_inch_bag_and_the_board_fits_its_cavity():
    assert M.BAG_W == pytest.approx(3 * 25.4 * MM, abs=0.5 * MM) and M.BAG_L == pytest.approx(5 * 25.4 * MM, abs=0.5 * MM)
    assert M.PCB_W < 2 * M.CAVITY_HALF and M.PCB_L < M.BAG_L - 2 * M.SEAM_W
    # the films' loop: bottom middle + two edges = top halves (one film width each side)
    assert 2 * M.HB + 2 * M.EW == pytest.approx(2 * M.HT)


def test_the_films_form_one_closed_loop_of_five_hinges():
    pairs = {(h.parent, h.child) for hs in bk._HINGES.values() for h in hs}
    assert pairs == {(M.BC.name, M.BE_L.name), (M.BC.name, M.BE_R.name), (M.BE_L.name, M.T_L.name),
                     (M.BE_R.name, M.T_R.name), (M.T_L.name, M.T_R.name)}
    for spec in (M.BE_L, M.BE_R):  # the edges can stand up as walls (to 95 deg), not fold under the bag
        h = [h for h in bk._HINGES[spec.name] if h.parent == M.BC.name][0]
        assert h.upper - h.lower == pytest.approx(M.WALL_MAX, abs=1e-6)


def test_the_bag_lies_shut_the_mouth_too_low_for_the_board():
    # laid down, the top film's underside over the middle is lower than the board's 5.3 mm components
    objs = objects(top_lift=0.0)
    assert M._top_bottom_at(objs, 0.0) - M.FILM_T / 2 < 4.0 * MM


def test_the_tote_holds_the_bag_and_the_bag_lies_within_reach():
    assert M.BAG_L < M.TOTE_IN[0] and M.BAG_W < M.TOTE_IN[1]
    assert M.TOTE_H - M.TOTE_FLOOR == pytest.approx(76 * MM, abs=1 * MM)


# -- scoring ---------------------------------------------------------------------------------------------------------

def test_the_start_is_nothing():
    assert run(objects()) == 0.0


def test_board_in_the_bag_on_the_mat_is_half():
    objs = objects(board_rel=IN_BAG)
    assert M.board_in_bag(objs) == (True, True)
    assert run(objs) == 0.5


def test_filled_bag_on_the_tote_floor_is_full_after_a_second():
    objs = objects(bag_at=FLOOR_BAG, board_rel=IN_BAG)
    assert M.bag_in_tote(objs)
    M.success(objs)
    assert M.score(objs) == 0.5
    assert run(objs) == 1.0


def test_success_ends_the_trial_only_after_the_second_held():
    objs = objects(bag_at=FLOOR_BAG, board_rel=IN_BAG)
    assert TASK.success_hold_s == 1.0
    # the first step only records the poses (stillness needs two), then STILL_STEPS of full score
    assert not any(M.success(objs) for _ in range(M.STILL_STEPS))
    assert M.success(objs) and M.score(objs) == 1.0


def test_board_half_in_is_a_quarter():
    rel = (-M.BAG_L / 2 + 10 * MM, 0.0, IN_BAG[2])  # centre 10 mm inside the mouth: 22 mm of it outside
    objs = objects(bag_at=FLOOR_BAG, board_rel=rel)
    assert M.board_in_bag(objs) == (False, True)
    assert run(objs) == 0.25


def test_board_just_short_of_the_5_mm_margin_is_only_a_quarter():
    tail_in = M.IN_MIN - 1 * MM
    rel = (-M.BAG_L / 2 + tail_in + M.PCB_L / 2, 0.0, IN_BAG[2])
    assert run(objects(bag_at=FLOOR_BAG, board_rel=rel)) == 0.25
    M._reset_state()
    rel = (-M.BAG_L / 2 + M.IN_MIN + 1 * MM + M.PCB_L / 2, 0.0, IN_BAG[2])
    assert run(objects(bag_at=FLOOR_BAG, board_rel=rel)) == 1.0


def test_board_on_top_of_the_shut_bag_is_nothing():
    on_top = (IN_BAG[0], 0.0, 3 * M.FILM_T + M.PCB_T)  # over the top film, not between the films
    objs = objects(bag_at=FLOOR_BAG, board_rel=on_top, top_lift=0.0)
    assert M.board_in_bag(objs) == (False, False)
    assert run(objs) == 0.0


def test_board_beside_the_bag_is_nothing():
    assert run(objects(bag_at=FLOOR_BAG, board_rel=(0.0, M.BAG_W, IN_BAG[2]))) == 0.0


def test_bag_held_above_the_floor_is_half():
    held = (FLOOR_BAG[0], FLOOR_BAG[1], FLOOR_BAG[2] + 20 * MM)
    assert run(objects(bag_at=held, board_rel=IN_BAG)) == 0.5


def test_bag_over_a_wall_or_outside_the_tote_is_half():
    over = (M.TOTE_X + M.TOTE_IN[0] / 2, M.TOTE_Y, M.TOP + M.TOTE_H + 5 * MM)  # lying across the far wall
    assert run(objects(bag_at=over, board_rel=IN_BAG)) == 0.5
    M._reset_state()
    beside = (M.TOTE_X - M.TOTE_IN[0] / 2 - 0.10, M.TOTE_Y, FLOOR_BAG[2] - M.TOTE_FLOOR)  # on the table by it
    assert run(objects(bag_at=beside, board_rel=IN_BAG)) == 0.5


def test_a_hand_near_the_bag_is_half(monkeypatch):
    objs = objects(bag_at=FLOOR_BAG, board_rel=IN_BAG)
    pt = np.array(FLOOR_BAG) + [0.0, 0.0, 10 * MM]
    monkeypatch.setattr(bk, "finger_points", lambda robots=None, **kw: np.array([pt]))
    assert run(objs) == 0.5


def test_moving_is_half():
    a = objects(bag_at=FLOOR_BAG, board_rel=IN_BAG)
    b = objects(bag_at=np.array(FLOOR_BAG) + [2 * MM, 0.0, 0.0], board_rel=IN_BAG)
    out = None
    for k in range(M.STILL_STEPS + 4):
        objs = a if k % 2 else b
        M.success(objs)
        out = M.score(objs)
    assert out == 0.5


def test_a_fall_off_the_table_latches_zero():
    good = objects(bag_at=FLOOR_BAG, board_rel=IN_BAG)
    fallen = objects(bag_at=FLOOR_BAG, board_world=(0.3, -0.6, M.TOP - 0.3))
    M.success(fallen)
    assert M.score(fallen) == 0.0
    assert run(good) == 0.0  # put back, it still scores 0


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


def test_the_demo_fits_the_run_limit_and_holds_the_bag_open_while_the_board_goes_in():
    for arm in ("right", "left"):
        assert sum(w.duration for w in PLANS[arm]) * 20 < M.RUN_LIMIT_STEPS

    def span(arm, label):
        t = 0.0
        for w in PLANS[arm]:
            if w.label == label:
                return t, t + w.duration
            t += w.duration
        raise KeyError(label)

    open_from, _ = span("left", "squeeze")
    _, open_until = span("left", "hold open")
    slide, _ = span("right", "slide it in")
    _, pushed = span("right", "push it home")
    assert open_from < slide and pushed <= open_until


def test_the_noopen_check_leaves_the_left_hand_home(monkeypatch):
    monkeypatch.setenv("BG_DEMO", "noopen")
    assert {tuple(w.pos) for w in M.demo()["left"]} == {M.LHOME}
