"""Checks of stirrup.py's measuring and scoring, and of its demo, without the simulator. Needs yam_arena (the official
stack):

    python -m pytest -q test_stirrup.py
"""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from yam_arena import station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

TASK = load(Path(__file__).with_name("stirrup.py"))
M = TASK.module
import rodkit  # noqa: E402  (the task file put its folder on sys.path)

MM = 0.001
R = M.WRAP_R
SEG = 1.0 * MM  # the synthetic bars are drawn in 1 mm segments: a 90-degree bend is 17 joints of about 5 degrees
# Straight lengths that make the drawing exactly: middle sides' outer lines are 2R longer than their straight part,
# the end sides R longer; each end reaches the other end's side on its centreline.
IDEAL = dict(a=83.0 - R / MM, b=53.0 - 2 * R / MM, c=83.0 - 2 * R / MM, d=53.0 - R / MM)


def bar(a, b, c, d, bends=(-90.0, -90.0, -90.0), lift=0.0, kink=None, z=0.0):
    """A bar drawn from its free end: straights a, b, c, d (mm) and the bends between them (deg, negative is the
    right way), each formed round radius R. ``lift`` raises the free end (mm); ``kink`` = (side, deg) puts a sharp
    bend in the middle of that side."""
    turns = []
    for k, length in enumerate((a, b, c, d)):
        n = max(1, int(round(length * MM / SEG)))
        side = [0.0] * n
        if kink is not None and kink[0] == k:
            side[n // 2] = math.radians(kink[1])
        turns += side
        if k < 3 and bends[k]:
            m = max(1, int(round(abs(math.radians(bends[k])) * R / SEG)))
            turns += [math.radians(bends[k]) / m] * m
    angles = np.array(turns[1:])
    heading = math.pi / 2 + np.concatenate([[0.0], np.cumsum(angles)])
    steps = np.stack([np.cos(heading), np.sin(heading), np.zeros_like(heading)], axis=1) * SEG
    pts = np.vstack([np.zeros(3), np.cumsum(steps, axis=0)])
    pts[:, 2] = z + np.linspace(lift * MM, 0.0, len(pts)) if lift else z
    return pts, angles


def state(pts, angles, moving=False):
    speeds = np.full(len(pts) - 1, 0.05 if moving else 0.0)
    return SimpleNamespace(points=pts, angles=angles, speeds=speeds,
                           resting=lambda: bool(speeds.max() < 0.01))


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    """A new trial: latches cleared, grippers far away."""
    M._reset_state()
    monkeypatch.setattr(rodkit, "finger_points", lambda: np.array([[0.0, 0.0, 1.0]]))
    yield


def run(*bars, moving=False):
    """Score the trial frame by frame through the given bars; returns the last (score, success)."""
    out = None
    for pts, angles in bars:
        f = {"rod": state(pts, angles, moving)}
        out = (M.score(f), M.success(f))
    return out


def ideal(**kw):
    args = dict(IDEAL)
    args.update({k: v for k, v in kw.items() if k in args})
    return bar(**args, **{k: v for k, v in kw.items() if k not in args})


# -- measuring ---------------------------------------------------------------------------------------------------


def test_the_ideal_bar_measures_as_drawn():
    m = M.measure(*ideal())
    assert m["corners"] == pytest.approx([-90.0, -90.0, -90.0], abs=0.5)
    assert m["sides"] == pytest.approx([50 * MM, 80 * MM], abs=1.0 * MM)
    assert max(m["gaps"]) < 1.0 * MM and m["straight"] < 0.5 * MM and m["flat"] == 0.0


def test_a_straight_bar_has_no_bends():
    m = M.measure(*bar(250, 0, 0, 0, bends=(0, 0, 0)))
    assert m["bends"] == [] and m["corners"] == []


def test_a_bend_spread_over_small_joints_is_measured_whole():
    pts, angles = ideal()
    m = M.measure(pts, angles)
    assert m["bends"] == pytest.approx([-90.0, -90.0, -90.0], abs=0.5)


# -- the tiers ---------------------------------------------------------------------------------------------------


def test_straight_scores_zero():
    assert run(bar(250, 0, 0, 0, bends=(0, 0, 0))) == (0.0, False)


def test_one_good_bend_is_a_quarter():
    assert run(bar(80, 170, 0, 0, bends=(-90, 0, 0))) == (0.25, False)


def test_two_good_bends_are_a_quarter():
    assert run(bar(80, 50, 120, 0, bends=(-90, -90, 0))) == (0.25, False)


def test_three_good_bends_with_the_ends_apart_are_half():
    assert run(ideal(a=IDEAL["a"] - 10)) == (0.5, False)


def test_a_side_out_of_size_is_half():
    assert run(ideal(b=IDEAL["b"] + 7)) == (0.5, False)


def test_the_drawing_released_and_still_is_full():
    assert run(ideal()) == (1.0, True)


def test_the_long_side_first_is_the_same_stirrup():
    assert run(ideal(a=53.0 - R / MM, b=IDEAL["c"], c=IDEAL["b"], d=83.0 - R / MM)) == (1.0, True)


def test_within_the_tolerances_is_full():
    assert run(ideal(a=IDEAL["a"] - 4, b=IDEAL["b"] + 4, c=IDEAL["c"] - 4)) == (1.0, True)


def test_a_corner_off_by_more_than_5_degrees_is_not_full():
    score, ok = run(ideal(bends=(-90, -84, -90)))
    assert score < 1.0 and not ok


def test_corners_within_5_degrees_are_good():
    m = M.measure(*ideal(bends=(-86, -94, -90)))
    assert all(M._good(c) for c in m["corners"])


def test_still_moving_is_not_success_yet():
    assert run(ideal(), moving=True) == (1.0, False)


def test_a_gripper_still_on_the_bar_is_half(monkeypatch):
    pts, angles = ideal()
    monkeypatch.setattr(rodkit, "finger_points", lambda: pts[100][None, :] + [0.0, 0.0, 0.004])
    assert run((pts, angles)) == (0.5, False)


def test_a_gripper_just_clear_is_full(monkeypatch):
    pts, angles = ideal()
    monkeypatch.setattr(rodkit, "finger_points", lambda: pts[100][None, :] + [0.0, 0.0, 0.013])
    assert run((pts, angles)) == (1.0, True)


def test_a_bar_not_flat_is_half():
    assert run(ideal(lift=8.0)) == (0.5, False)


def test_a_kinked_side_is_half():
    assert run(ideal(kink=(2, 10.0))) == (0.5, False)


def test_a_fourth_bend_is_not_full():
    score, ok = run(ideal(kink=(0, -45.0)))
    assert score < 1.0 and not ok


# -- violations latch --------------------------------------------------------------------------------------------


def test_bends_made_both_ways_score_zero_for_good():
    assert run(bar(80, 50, 120, 0, bends=(-90, 40, 0))) == (0.0, False)
    assert run(ideal()) == (0.0, False)


def test_a_kink_the_other_way_beside_a_bend_is_not_hidden_in_it():
    pts, angles = ideal()
    i = int(np.argmax(np.abs(angles) > 0)) - 2  # just before the first bend, one straight joint apart from it
    angles = angles.copy()
    angles[i - 6:i] = math.radians(40.0 / 6)
    m = M.measure(pts, angles)
    assert max(m["bends"]) >= 30.0 and min(m["bends"]) <= -30.0


def test_a_small_bend_the_other_way_does_not_latch():
    assert run(bar(80, 50, 120, 0, bends=(-90, 20, 0)), ideal()) == (1.0, True)


def test_the_mirror_image_is_the_same_stirrup():
    assert run(ideal(bends=(90.0, 90.0, 90.0))) == (1.0, True)


def test_reset_clears_the_latch():
    run(bar(80, 50, 120, 0, bends=(-90, 40, 0)))
    M._reset_state()
    assert run(ideal()) == (1.0, True)


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


def test_the_demo_fits_the_run_limit():
    assert max(ArmPlan(p, station.BASES[a]).duration for a, p in M.demo().items()) * 20 < M.RUN_LIMIT_STEPS


def test_the_chalk_marks_are_where_the_demo_bends():
    assert M.MARKS[0] == pytest.approx(M.FIRST)
    assert np.diff(M.MARKS) == pytest.approx(np.array(M.FEEDS))
    assert M.MARKS[-1] < M.ROD.length


def test_the_finished_hoop_clears_the_guide_pins():
    """The first side closes 53 mm (centreline) from the third, which runs round the bending pin's far side; the
    front and stop pins (24 and 30 mm back, 5 mm radius to their flats) stay at least 3 mm inside it."""
    first_side = 53 * MM - R  # its centreline, from the pin's centre along the feed side
    pin_edge = max(M.FRONT[1], M.STOP[1]) - M.PIN[1] + M.STOP_R / math.cos(math.pi / 12)
    assert first_side - M.ROD_T / 2 - pin_edge > 3 * MM
