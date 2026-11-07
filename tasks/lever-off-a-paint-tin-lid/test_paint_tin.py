"""Checks of paint_tin.py's scoring and demo, without the simulator. Needs yam_arena (the official stack):

    python -m pytest -q test_paint_tin.py
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

TASK = load(Path(__file__).with_name("paint_tin.py"))
M = TASK.module
import tinkit  # noqa: E402  (the task file put its folder on sys.path)

MM = 0.001
TIN0 = np.array([*M.TIN_XY, M.TOP])


def _quat(yaw_deg=0.0, tilt_deg=0.0):
    """xyzw of a turn ``yaw_deg`` about the vertical times a tilt ``tilt_deg`` about the body's own x."""
    sz, cz = math.sin(math.radians(yaw_deg) / 2), math.cos(math.radians(yaw_deg) / 2)
    sx, cx = math.sin(math.radians(tilt_deg) / 2), math.cos(math.radians(tilt_deg) / 2)
    return np.array([cz * sx, sz * sx, sz * cx, cz * cx])


def _state(spec, xy, low_z, yaw=0.0, tilt=0.0, moving=False):
    """``spec`` turned and tilted, its bounding box's lowest point at ``low_z`` and its origin over ``xy``."""
    q = _quat(yaw, tilt)
    probe = tinkit.body_state(spec, (0.0, 0.0, 0.0), q, np.zeros(3), np.zeros(3))
    v = np.array([0.05, 0.0, 0.0]) if moving else np.zeros(3)
    return tinkit.body_state(spec, (*xy, low_z - probe.low[2]), q, v, np.zeros(3))


def tin(dxy=(0.0, 0.0), tilt=0.0, low_z=None, moving=False):
    return _state(M.TIN, TIN0[:2] + np.array(dxy), M.TOP if low_z is None else low_z, tilt=tilt, moving=moving)


def lid_on_tin():
    return tinkit.body_state(M.LID, (*M.TIN_XY, M.TOP + M.SEAT_TOP), _quat(), np.zeros(3), np.zeros(3))


def lid_on_table(xy=(0.24, 0.13), tilt=0.0, low_z=None, moving=False):
    return _state(M.LID, xy, M.TOP if low_z is None else low_z, tilt=tilt, moving=moving)


def tool_with_tip_at(tip):
    """The screwdriver upright with its blade's end at ``tip`` (world)."""
    return tinkit.body_state(M.SCREWDRIVER, np.asarray(tip) - np.array([0.0, 0.0, M.TIP_Z]), _quat(),
                             np.zeros(3), np.zeros(3))


def tip_in_groove(angle_deg=255.0):
    r = (M.LID_R + M.BEAD_IN) / 2
    a = math.radians(angle_deg)
    return TIN0 + np.array([r * math.cos(a), r * math.sin(a), M.CHANNEL_FLOOR + 3 * MM])


TOOL_PARKED = tool_with_tip_at(np.array([0.30, -0.30, M.TOP + 0.10]))
TOOL_IN_GROOVE = tool_with_tip_at(tip_in_groove())


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    """A new trial: seal re-armed, latches cleared, fingers far away."""
    M.SEAL.clear()
    M._reset_state()
    monkeypatch.setattr(tinkit, "finger_points", lambda: np.array([[0.0, 0.0, M.TOP + 0.4]]))
    yield


def objs(lid, tin_state=None, tool=TOOL_PARKED):
    return {"lid": lid, "tin": tin() if tin_state is None else tin_state, "screwdriver": tool, "stand": None}


def run(*frames):
    """Score a trial frame by frame; returns (score, success) of the last frame."""
    for f in frames:
        out = (M.score(f), M.success(f))
    return out


def pried(n_free=12):
    M.SEAL.held[:n_free] = False


# -- the tiers ---------------------------------------------------------------------------------------------------


def test_start_scores_zero():
    assert run(objs(lid_on_tin())) == (0.0, False)


def test_lid_set_down_flat_and_let_go_is_full_score():
    pried()
    assert run(objs(lid_on_tin(), tool=TOOL_IN_GROOVE), objs(lid_on_table())) == (1.0, True)


def test_lid_upside_down_on_the_table_counts_as_flat():
    pried()
    assert run(objs(lid_on_tin(), tool=TOOL_IN_GROOVE), objs(lid_on_table(tilt=180.0))) == (1.0, True)


def test_lid_still_moving_is_not_success_yet():
    pried()
    assert run(objs(lid_on_tin(), tool=TOOL_IN_GROOVE), objs(lid_on_table(moving=True))) == (1.0, False)


def test_tin_still_moving_is_not_success_yet():
    pried()
    assert run(objs(lid_on_tin(), tool=TOOL_IN_GROOVE), objs(lid_on_table(), tin(moving=True))) == (1.0, False)


def test_lid_still_in_the_gripper_on_the_table_is_half(monkeypatch):
    pried()
    frames = [objs(lid_on_tin(), tool=TOOL_IN_GROOVE), objs(lid_on_table())]
    lid = frames[1]["lid"]
    edge = lid.to_world((M.LID_R + 2 * MM, 0.0, M.LID_T / 2))  # a fingertip 2 mm off the lid's edge
    monkeypatch.setattr(tinkit, "finger_points", lambda: edge[None, :])
    assert run(*frames) == (0.5, False)


def test_fingers_just_beyond_20_mm_are_clear(monkeypatch):
    pried()
    frames = [objs(lid_on_tin(), tool=TOOL_IN_GROOVE), objs(lid_on_table())]
    above = frames[1]["lid"].to_world((0.0, 0.0, M.LID_T + 21 * MM))
    monkeypatch.setattr(tinkit, "finger_points", lambda: above[None, :])
    assert run(*frames) == (1.0, True)


def test_lid_tilted_on_the_table_is_half():
    pried()
    assert run(objs(lid_on_tin(), tool=TOOL_IN_GROOVE), objs(lid_on_table(tilt=25.0))) == (0.5, False)


def test_lid_propped_against_the_tin_is_half():
    pried()
    xy = TIN0[:2] + np.array([0.0, M.TIN_R + 12 * MM])
    assert run(objs(lid_on_tin(), tool=TOOL_IN_GROOVE), objs(lid_on_table(xy=xy, tilt=70.0))) == (0.5, False)


def test_lid_flat_but_touching_the_tin_is_half():
    pried()
    xy = TIN0[:2] + np.array([0.0, M.TIN_R + M.LID_CORNER_R + 2 * MM])  # 2 mm clear: under the 5 mm asked
    assert run(objs(lid_on_tin(), tool=TOOL_IN_GROOVE), objs(lid_on_table(xy=xy))) == (0.5, False)


def test_lid_held_in_the_air_beside_the_tin_is_half():
    pried()
    assert run(objs(lid_on_tin(), tool=TOOL_IN_GROOVE), objs(lid_on_table(low_z=M.TOP + 0.08))) == (0.5, False)


@pytest.mark.parametrize("n_free,want", [(12, 0.25), (6, 0.25), (5, 0.0), (0, 0.0)])
def test_lid_left_on_the_tin_scores_by_how_loose_it_is(n_free, want):
    pried(n_free)
    assert run(objs(lid_on_tin(), tool=TOOL_IN_GROOVE), objs(lid_on_tin())) == (want, False)


def test_loose_lid_held_over_the_tin_is_a_quarter():
    pried()
    over = tinkit.body_state(M.LID, (*M.TIN_XY, M.TOP + M.BEAD_TOP + 0.03), _quat(), np.zeros(3), np.zeros(3))
    assert run(objs(lid_on_tin(), tool=TOOL_IN_GROOVE), objs(over)) == (0.25, False)


# -- the tool must be used ---------------------------------------------------------------------------------------


def test_lid_taken_off_without_the_blade_in_the_groove_scores_zero():
    pried()
    assert run(objs(lid_on_tin()), objs(lid_on_table())) == (0.0, False)


def test_tool_in_the_groove_latches():
    pried()
    assert run(objs(lid_on_tin(), tool=TOOL_IN_GROOVE), objs(lid_on_tin()), objs(lid_on_table())) == (1.0, True)


@pytest.mark.parametrize("where", ["on the lid", "outside the tin", "above the bead"])
def test_blade_elsewhere_is_not_in_the_groove(where):
    tip = {
        "on the lid": TIN0 + np.array([0.0, 0.0, M.SEAT_TOP + M.LID_T + 1 * MM]),
        "outside the tin": TIN0 + np.array([M.TIN_R + 10 * MM, 0.0, M.CHANNEL_FLOOR]),
        "above the bead": tip_in_groove() + np.array([0.0, 0.0, M.BEAD_TOP - M.CHANNEL_FLOOR]),
    }[where]
    assert not M.blade_in_groove(tool_with_tip_at(tip), tin())


@pytest.mark.parametrize("angle", [0.0, 90.0, 255.0, 300.0])
def test_blade_in_the_groove_anywhere_round_the_rim(angle):
    assert M.blade_in_groove(tool_with_tip_at(tip_in_groove(angle)), tin())


# -- violations latch --------------------------------------------------------------------------------------------


def _full_end():
    return objs(lid_on_table())


def test_tin_tipped_over_and_stood_back_up_scores_zero():
    pried()
    assert run(objs(lid_on_tin(), tool=TOOL_IN_GROOVE), objs(lid_on_tin(), tin(tilt=20.0)), _full_end()) == (0.0, False)


def test_tin_tilted_under_the_limit_is_fine():
    pried()
    assert run(objs(lid_on_tin(), tool=TOOL_IN_GROOVE), objs(lid_on_tin(), tin(tilt=12.0)), _full_end()) == (1.0, True)


def test_tin_pushed_away_and_back_scores_zero():
    pried()
    away = tin(dxy=(0.0, 0.035))
    assert run(objs(lid_on_tin(), tool=TOOL_IN_GROOVE), objs(lid_on_table(), away), _full_end()) == (0.0, False)


def test_tin_moved_under_the_limit_is_fine():
    pried()
    near = tin(dxy=(0.02, 0.0))
    assert run(objs(lid_on_tin(), tool=TOOL_IN_GROOVE), objs(lid_on_table(), near)) == (1.0, True)


def test_tin_moved_from_where_it_settled_not_from_the_nominal_spot():
    pried()
    settled = tin(dxy=(0.025, 0.0))
    later = tin(dxy=(0.05, 0.0))  # 25 mm from where it started the trial
    blade = tool_with_tip_at(tip_in_groove() + np.array([0.025, 0.0, 0.0]))  # in the settled tin's groove
    assert run(objs(lid_on_tin(), settled, blade), objs(lid_on_table(), later)) == (1.0, True)


def test_lid_dropped_off_the_table_scores_zero():
    pried()
    fallen = lid_on_table(xy=(0.6, 0.1), low_z=M.TOP - 0.3)
    assert run(objs(lid_on_tin(), tool=TOOL_IN_GROOVE), objs(fallen), _full_end()) == (0.0, False)


def test_tin_knocked_off_the_table_scores_zero():
    pried()
    gone = tin(dxy=(0.0, 0.0), low_z=M.TOP - 0.3)
    assert run(objs(lid_on_tin(), tool=TOOL_IN_GROOVE), objs(lid_on_table(), gone)) == (0.0, False)


def test_reset_clears_the_latches():
    pried()
    run(objs(lid_on_tin(), tin(tilt=30.0), TOOL_IN_GROOVE))
    assert M.score(_full_end()) == 0.0
    M._reset_state()
    assert run(objs(lid_on_tin(), tool=TOOL_IN_GROOVE), _full_end()) == (1.0, True)


def test_no_fingers_known_offline_counts_as_clear(monkeypatch):
    monkeypatch.setattr(tinkit, "finger_points", lambda: np.zeros((0, 3)))
    assert M.fingers_to_lid(lid_on_table()) == math.inf


# -- the demo ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("arm", ["left", "right"])
def test_the_demo_is_within_reach(arm):
    plan = ArmPlan(M.demo()[arm], station.BASES[arm])
    for t in np.arange(0.0, plan.duration, 0.05):
        plan.action7(t)
    assert plan.worst_ik_mm < 0.5 and plan.worst_ik_deg < 0.1, (plan.worst_ik_mm, plan.worst_ik_deg)


def test_the_demo_fits_the_run_limit():
    plan = ArmPlan(M.demo()["left"], station.BASES["left"])
    assert plan.duration * 20 < M.RUN_LIMIT_STEPS


# The per-step limits of inspect-robots' DeltaLimitGuard for the YAM action space, as recorded in a run's
# configuration (inspect_robots_yam.collision builds the guard chain): six joints, then the gripper (0-1, 5% a step).
GUARD_MAX_DELTA = np.array([0.287979, 0.1832595, 0.1570795, 0.1631885, 0.15708, 0.20944, 0.05])


@pytest.mark.parametrize("arm", ["left", "right"])
def test_the_demo_passes_the_run_guards(arm):
    """No planned step may be clamped by the delta-limit guard: a clamped open-loop demo drifts off its plan."""
    plan = ArmPlan(M.demo()[arm], station.BASES[arm])
    acts = np.array([plan.action7(t)[0] for t in np.arange(0.0, plan.duration, 0.05)])
    worst = (np.abs(np.diff(acts, axis=0)) / GUARD_MAX_DELTA).max()
    assert worst < 0.95, worst
