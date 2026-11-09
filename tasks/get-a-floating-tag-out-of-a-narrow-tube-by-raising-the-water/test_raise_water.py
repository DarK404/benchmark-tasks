"""Checks of raise_water.py's water model, design numbers, scoring and demo, without the simulator. Needs yam_arena (the
official stack):

    python -m pytest -q test_raise_water.py
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

TASK = load(Path(__file__).with_name("raise_water.py"))
M = TASK.module
import waterkit as wk  # noqa: E402  (the task file put its folder on sys.path)

MM = 0.001
ZERO = np.zeros(3)
G = 9.81


def frame(pos, vel=ZERO, spin=ZERO):
    pos = np.asarray(pos, float)
    return wk.Frame(pos=pos, rot=np.eye(3), vel=np.asarray(vel, float), spin=np.asarray(spin, float), com=pos)


def water(*floaters):
    w = wk.Water((0.0, 0.0), 0.02, 0.0, math.pi * 0.02 ** 2 * 0.05, floaters)
    w.clear()
    return w


SLUG = wk.Floater("slug", -0.0175, 0.035, M.SLUG_AREA, M.SLUG_R12, drag=1.5)


# -- the water model ---------------------------------------------------------------------------------------------


def test_with_nothing_in_it_the_level_is_the_water_over_the_bore():
    w = water(SLUG)
    w.update({}, {})
    assert w.level == pytest.approx(0.05)


def test_a_sunk_slug_raises_the_level_by_its_volume_over_the_bore():
    w = water(SLUG)
    w.update({"slug": frame((0, 0, 0.0175))}, {})
    assert w.level - 0.05 == pytest.approx(M.SLUG_AREA * 0.035 / (math.pi * 0.02 ** 2), rel=1e-6)


def test_a_slug_part_way_in_counts_only_what_is_under():
    w = water(SLUG)
    acc = {}
    w.update({"slug": frame((0, 0, 0.06))}, acc)  # bottom at 42.5 mm: under the 50 mm water by a few mm
    under = w.level - 0.0425
    assert 0 < under < 0.035
    assert acc["slug"][0][2] == pytest.approx(1000 * G * M.SLUG_AREA * under, rel=1e-6)


def test_a_body_outside_the_bore_is_not_in_the_water():
    w = water(SLUG)
    acc = {}
    w.update({"slug": frame((0.05, 0, 0.0175))}, acc)
    assert w.level == pytest.approx(0.05) and "slug" not in acc


def test_the_water_damps_a_moving_body_against_its_motion():
    w = water(SLUG)
    acc = {}
    w.update({"slug": frame((0, 0, 0.0175), vel=(0, 0, -0.2))}, acc)
    buoy = 1000 * G * M.SLUG_AREA * 0.035
    assert acc["slug"][0][2] == pytest.approx(buoy + 1.5 * 0.2, rel=1e-6)


def test_the_float_floats_at_its_designed_depth():
    f = M.FLOATERS[0]
    w = wk.Water((0.0, 0.0), M.BORE_R, 0.0, M.AREA * 0.07, [f])
    acc = {}
    level = 0.07 + f.area * M.F_SINK / M.AREA
    w.update({f.name: frame((0, 0, level - M.F_SINK))}, acc)
    assert acc[f.name][0][2] == pytest.approx(M.F_MASS * G, rel=1e-3)
    assert M.F_SINK < M.F_H


# -- the design numbers: what makes the task work ----------------------------------------------------------------


def tag_top_after(n: int) -> float:
    """The tag's top above the table with ``n`` metal slugs under water, the float floating."""
    level = M.FLOOR + (M.AREA * M.WATER0 + n * M.SLUG_AREA * M.SLUG_H + M.F_AREA * M.F_SINK) / M.AREA
    return level - M.F_SINK + M.F_H + M.TAG_H


def test_the_tag_starts_well_out_of_reach():
    assert M.RIM - tag_top_after(0) > 30 * MM


def test_two_slugs_leave_the_tag_below_the_rim():
    assert tag_top_after(2) < M.RIM


def test_three_slugs_lift_the_tag_10_mm_clear_of_the_rim():
    assert tag_top_after(3) - M.RIM > 10 * MM


def test_three_slugs_stacked_are_under_water():
    level = tag_top_after(3) - M.TAG_H - M.F_H + M.F_SINK
    assert M.FLOOR + 3 * M.SLUG_H < level


def test_the_slugs_pass_through_the_float_and_cannot_tumble_in_the_bore():
    assert 2 * M.SLUG_R12 < 2 * M.F_RIN - 2 * MM
    assert math.hypot(2 * M.SLUG_R12, M.SLUG_H) > 2 * M.BORE_R


def test_a_cork_slug_floats_and_a_metal_one_sinks():
    full = 1000.0 * M.SLUG_AREA * M.SLUG_H  # kg of water displaced fully under
    masses = {s.name: s.mass for s in M.SLUGS}
    assert all(m < full for n, m in masses.items() if n.startswith("cork"))
    assert all(m > full for n, m in masses.items() if n.startswith("metal"))


# -- scoring -----------------------------------------------------------------------------------------------------


def float_state(pos, moving=False):
    v = np.array([0.05, 0.0, 0.0]) if moving else ZERO
    return wk.body_state(M.FLOAT, tuple(pos), (0.0, 0.0, 0.0, 1.0), v, ZERO)


IN_TUBE = (*M.TUBE_XY, M.TOP + 0.08)
RAISED = (*M.TUBE_XY, M.TOP + M.RIM - M.F_H - M.TAG_H + 12 * MM)
LIFTED = (*M.TUBE_XY, M.TOP + M.RIM + 20 * MM)
IN_TRAY = (*M.TRAY_XY, M.TOP + 3 * MM)


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    """A new trial: latches cleared, grippers far away."""
    M._reset_state()
    monkeypatch.setattr(wk, "finger_points", lambda: np.array([[0.0, 0.0, M.TOP + 0.5]]))
    yield


def run(*positions, moving=False):
    out = None
    for p in positions:
        f = {"tagfloat": float_state(p, moving)}
        out = (M.score(f), M.success(f))
    return out


def test_the_start_scores_zero():
    assert run(IN_TUBE) == (0.0, False)


def test_the_tag_raised_above_the_rim_is_a_quarter():
    assert run(IN_TUBE, RAISED) == (0.25, False)


def test_raising_latches_even_if_the_water_falls_again():
    assert run(RAISED, IN_TUBE) == (0.25, False)


def test_the_float_out_of_the_tube_is_half():
    assert run(IN_TUBE, RAISED, LIFTED) == (0.5, False)


def test_dropped_on_the_table_after_lifting_is_half():
    assert run(RAISED, LIFTED, (0.30, -0.25, M.TOP)) == (0.5, False)


def test_in_the_tray_released_and_still_is_full():
    assert run(IN_TUBE, RAISED, LIFTED, IN_TRAY) == (1.0, True)


def test_still_moving_in_the_tray_is_not_success_yet():
    assert run(IN_TRAY, moving=True) == (1.0, False)


def test_a_gripper_still_holding_it_over_the_tray_is_not_full(monkeypatch):
    tag = float_state(IN_TRAY).to_world(M.TAG_LOCAL_TOP)
    monkeypatch.setattr(wk, "finger_points", lambda: tag[None, :])
    assert run(LIFTED, IN_TRAY) == (0.5, False)


def test_a_gripper_just_clear_is_full(monkeypatch):
    tag = float_state(IN_TRAY).to_world(M.TAG_LOCAL_TOP)
    monkeypatch.setattr(wk, "finger_points", lambda: (tag + [0.0, 0.0, 0.012])[None, :])
    assert run(LIFTED, IN_TRAY) == (1.0, True)


def test_reset_clears_the_latches():
    run(RAISED, LIFTED)
    M._reset_state()
    assert run(IN_TUBE) == (0.0, False)


# -- the demo ----------------------------------------------------------------------------------------------------

# The per-step limits of inspect-robots' DeltaLimitGuard for the YAM action space, as recorded in a run's
# configuration: six joints, then the gripper (0-1, 5% a step).
GUARD_MAX_DELTA = np.array([0.287979, 0.1832595, 0.1570795, 0.1631885, 0.15708, 0.20944, 0.05])


@pytest.mark.parametrize("arm", ["left", "right"])
def test_the_demo_reaches_every_target_along_its_path(arm):
    """Forward kinematics of the commanded joints against the target, every 50 ms: the arms' joint limits stop them
    pointing straight down much above 200 mm, and the planner's own error figure does not always show it."""
    plan = ArmPlan(M.demo()[arm], station.BASES[arm])
    base = np.array(station.BASES[arm])
    worst = 0.0
    for t in np.arange(0.0, plan.duration, 0.05):
        pos = plan.target(t)[0]
        q = plan.action7(t)[0][:6]
        tcp = ik.tcp_pose(q)
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


def test_the_demo_drops_metal_slugs_only():
    assert all(M.KINDS[i] == "S" for i in M.METAL_SLOTS[:3])
