"""Offline checks of roomba_brushes.py's geometry, scoring and demo, without the simulator. Needs yam_arena (the
official stack):

    python -m pytest -q test_roomba_brushes.py
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

TASK = load(Path(__file__).with_name("roomba_brushes.py"))
M = TASK.module
import brushkit as bk  # noqa: E402  (the task file put its folder on sys.path)

MM = 0.001
ZERO = np.zeros(3)


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    M._reset_state()
    monkeypatch.setattr(bk, "finger_points", lambda robots=None, **kw: np.zeros((0, 3)))
    yield


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def brush_at(peg, seat=None, d=(0.0, 0.0, 0.0), tip=0.0, moving=False):
    seat = seat or M.SEAT_OF[peg]
    r = _rx(-tip)
    vel = np.array([0.0, 0.0, 0.05]) if moving else ZERO
    return bk.body_state(M.BRUSHES[peg], M.seat_origin(seat) + np.asarray(d), bk.matrix_to_quat_rel(r), vel, ZERO)


def door_at(deg, catch_back=0.0):
    """The door turned ``deg`` open from shut, and its catches pushed ``catch_back`` (m) back from rest."""
    r = _ry(math.radians(deg))
    door = bk.body_state(M.DOOR, M._HINGE_W, bk.matrix_to_quat_rel(r), ZERO, ZERO)
    out = {M.DOOR.name: door}
    for c, sgn in ((M.CATCH_A, 1.0), (M.CATCH_B, -1.0)):
        local = np.array([M.CATCH_REST_X + catch_back, sgn * M.CATCH_Y, 0.0])
        out[c.name] = bk.body_state(c, M._HINGE_W + r @ local, bk.matrix_to_quat_rel(r), ZERO, ZERO)
    return out


def objects(sq, hx, door_deg=0.0, catch_back=0.0):
    out = {M.BRUSHES["square"].name: sq, M.BRUSHES["hex"].name: hx}
    out.update(door_at(door_deg, catch_back))
    return out


def run(objs, steps=M.STILL_STEPS + 2):
    out = None
    for _ in range(steps):
        M.success(objs)
        out = M.score(objs)
    return out


def seated_both():
    return brush_at("square", d=(0, -3 * MM, 0)), brush_at("hex", d=(0, -3 * MM, 0))


# -- the parts -------------------------------------------------------------------------------------------------------

def _largest_square_in_hex(af):
    return (3 - math.sqrt(3)) * af / math.sqrt(3)


def _hex_min_bounding_square(af):
    return 2 * af / math.sqrt(3) * math.cos(math.radians(15))


def test_each_peg_fits_only_its_own_socket():
    sq_hole = M.SQUARE + 2 * M.HOLE_CLEAR["front"]
    hex_hole = M.HEX_AF + 2 * M.HOLE_CLEAR["rear"]
    assert M.SQUARE < sq_hole and M.HEX_AF < hex_hole  # each goes in its own
    assert _largest_square_in_hex(hex_hole) < M.SQUARE  # the square peg cannot pass the hex socket at any turn
    assert _hex_min_bounding_square(M.HEX_AF) > sq_hole  # nor the hex peg the square one


def test_the_seats_are_side_by_side_in_the_opening_with_their_drives_on_one_side():
    assert M.PK_X0 < M.SEAT_X["front"] - M.BR_R and M.SEAT_X["rear"] + M.BR_R < M.PK_X1
    assert M.SEAT_X["rear"] - M.SEAT_X["front"] > 2 * M.BR_R  # the brushes do not touch
    assert M.PLATE_Y1 < M.BACK_Y < M.PK_Y1  # the drive socket, then its end stop, at +y
    assert M.CRADLE_Y1 < -M.BR_L / 2  # the bearing cradle beyond the brush's body, at -y


def test_the_door_rests_open_past_upright_and_shuts_over_the_opening():
    assert 90.0 < M.OPEN_DEG < 120.0
    assert M.HINGE_X - M.DOOR_L < M.PK_X0  # shut, it reaches past the front edge (over the catch lip)
    h = bk._HINGES[M.DOOR.name]
    assert (h.lower, h.upper) == (pytest.approx(math.radians(-M.OPEN_DEG)), 0.0)


def test_the_catches_hook_under_the_lip_and_must_spring_back_to_pass_it():
    overlap = M.LIP_X1 - M.NOSE_X0  # the nose under the lip
    assert overlap > 1.0 * MM and M.CATCH_TRAVEL > overlap
    assert M.NOSE_Z1 < M.LIP_Z0  # the nose sits below the lip's square underside
    for c in (M.CATCH_A, M.CATCH_B):
        sp = bk._SPRINGS[c.name]
        assert sp.target < 0 and (sp.lower, sp.upper) == (0.0, M.CATCH_TRAVEL)


# -- scoring ---------------------------------------------------------------------------------------------------------

def test_both_seated_and_the_door_latched_is_full_after_a_second():
    sq, hx = seated_both()
    objs = objects(sq, hx)
    assert M.seated(sq, "front") and M.seated(hx, "rear") and M.latched(objs)
    M.success(objs)
    assert M.score(objs) == 0.5
    assert run(objs) == 1.0


def test_the_door_shut_but_its_catches_not_home_is_half():
    sq, hx = seated_both()
    assert run(objects(sq, hx, catch_back=1.5 * MM)) == 0.5


def test_the_door_left_open_is_half():
    sq, hx = seated_both()
    assert run(objects(sq, hx, door_deg=28.0)) == 0.5


def test_a_hand_on_it_is_half(monkeypatch):
    sq, hx = seated_both()
    objs = objects(sq, hx)
    pt = sq.to_world((0.0, 0.0, M.BR_R))
    monkeypatch.setattr(bk, "finger_points", lambda robots=None, **kw: np.array([pt]))
    assert run(objs) == 0.5


def test_a_brush_still_moving_is_half():
    sq, hx = seated_both()
    assert run(objects(brush_at("square", d=(0, -3 * MM, 0), moving=True), hx)) == 0.5


def test_one_brush_seated_is_a_quarter():
    sq, _ = seated_both()
    hx = brush_at("hex", d=(0, -10 * MM, 10 * MM), tip=math.radians(6.0))  # peg short of home, resting tilted
    assert not M.seated(hx, "rear")
    assert run(objects(sq, hx, door_deg=28.0)) == 0.25


def test_axial_play_within_the_peg_engagement_still_counts():
    assert M.seated(brush_at("square", d=(0, -4.9 * MM, 0)), "front")
    assert not M.seated(brush_at("square", d=(0, -6.0 * MM, 0)), "front")


def test_a_brush_in_the_wrong_seat_does_not_count():
    sq = brush_at("square", seat="rear")
    hx = brush_at("hex", seat="front")
    assert run(objects(sq, hx)) == 0.0


def test_nothing_done_is_nothing():
    sq = bk.body_state(M.BRUSHES["square"], M.BRUSH_START["square"], (0.0, 0.0, 0.0, 1.0), ZERO, ZERO)
    hx = bk.body_state(M.BRUSHES["hex"], M.BRUSH_START["hex"], (0.0, 0.0, 0.0, 1.0), ZERO, ZERO)
    assert run(objects(sq, hx, door_deg=M.OPEN_DEG)) == 0.0


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


def test_the_demo_fits_both_brushes_before_shutting_the_door():
    right = [w.label for w in PLANS["right"]]
    assert right.index("peg through the socket") < right.index("push it over") < right.index("press it shut")
    left = PLANS["left"]
    t = 0.0
    for w in left:
        t += w.duration
        if w.label == "let it drop":
            break
    t_push = sum(w.duration for w in PLANS["right"][:right.index("push it over")])
    assert t < t_push  # the left brush is in before the door is pushed over


def test_each_brush_goes_to_the_seat_that_matches_its_peg():
    assert M.SEAT_OF == {"square": "front", "hex": "rear"}


def test_the_swap_check_offers_each_brush_to_the_other_seat(monkeypatch):
    monkeypatch.setenv("RB_DEMO", "swap")
    labels = [w.label for w in M.demo()["right"]]
    assert "over the rear seat" in labels
