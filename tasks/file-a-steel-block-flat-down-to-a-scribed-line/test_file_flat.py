"""Checks of file_flat.py's cutting model, scoring and demo, without the simulator. Needs yam_arena (the official
stack):

    python -m pytest -q test_file_flat.py
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

TASK = load(Path(__file__).with_name("file_flat.py"))
M = TASK.module
import filekit as fk  # noqa: E402  (the task file put its folder on sys.path)

MM = 0.001
ZERO = np.zeros(3)
DT = 1.0 / 240.0


def frame(pos, vel=ZERO, pitch_deg=0.0):
    t = math.radians(pitch_deg)
    rot = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(t), -math.sin(t)], [0.0, math.sin(t), math.cos(t)]])
    pos = np.asarray(pos, float)
    return fk.Frame(pos=pos, rot=rot, vel=np.asarray(vel, float), spin=ZERO.copy(), com=pos)


def model(**kw):
    """One 20 x 20 mm section under the middle of a 150 mm blade, its top at 10 mm."""
    args = dict(k=2000.0, damping=0.0, mu=0.5, wear=6e-4)
    args.update(kw)
    m = fk.Filing("file", (-0.075, 0.075), 0.01, [(-0.01, 0.01, -0.01, 0.01)], 0.010, **args)
    m.dt = DT
    return m


# -- the cutting model -------------------------------------------------------------------------------------------


def test_a_file_above_the_top_feels_nothing():
    m, acc = model(), {}
    m.update({"file": frame((0, 0, 0.011))}, acc)
    assert "file" not in acc and m.h[0] == pytest.approx(0.010)


def test_pressed_into_the_top_it_is_pushed_up_by_the_spring():
    m, acc = model(), {}
    m.update({"file": frame((0, 0, 0.009))}, acc)  # 1 mm in
    assert acc["file"][0][2] == pytest.approx(2000.0 * 0.001, rel=1e-6)


def test_sliding_towards_the_tip_cuts_by_pressure_times_distance():
    m = model()
    for _ in range(240):  # 1 s at 50 mm/s, 1 mm in: 2 N
        m.update({"file": frame((0, 0, 0.009), vel=(0, 0.05, 0))}, {})
    assert 0.010 - m.h[0] == pytest.approx(6e-4 * 2.0 * 0.05, rel=0.05)


def test_sliding_back_cuts_nothing_but_drags():
    m, acc = model(), {}
    m.update({"file": frame((0, 0, 0.009), vel=(0, -0.05, 0))}, acc)
    assert m.h[0] == pytest.approx(0.010)
    assert acc["file"][0][1] == pytest.approx(0.5 * 2.0, rel=1e-6)  # friction against the motion


def test_a_section_beyond_the_blade_is_not_cut():
    m = model()
    m.update({"file": frame((0, 0.2, 0.009), vel=(0, 0.05, 0))}, {})
    assert m.h[0] == pytest.approx(0.010)


def test_a_file_rocked_tip_up_bears_harder_under_the_handle_end():
    m = fk.Filing("file", (-0.075, 0.075), 0.01, [(-0.01, 0.01, -0.03, -0.01), (-0.01, 0.01, 0.01, 0.03)], 0.010,
                  k=2000.0, damping=0.0)
    m.update({"file": frame((0, 0, 0.0095), pitch_deg=1.0)}, {})
    assert m.force[0] > m.force[1] > 0.0


def test_the_cut_stops_at_the_solid_part():
    m = model()
    m.floor = 0.0099
    for _ in range(2400):
        m.update({"file": frame((0, 0, 0.005), vel=(0, 0.05, 0))}, {})
    assert m.h[0] == pytest.approx(0.0099)


# -- scoring -----------------------------------------------------------------------------------------------------


def file_state(on_rest=True, moving=False):
    pos = (*M.REST_XY, M.TOP + M.REST_H) if on_rest else (*M.VISE_XY, M.TOP + M.H0 + 0.02)
    v = np.array([0.05, 0.0, 0.0]) if moving else ZERO
    return fk.body_state(M.FILE, pos, (0.0, 0.0, 0.0, 1.0), v, ZERO)


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    """A new trial: the top uncut, latches cleared, grippers far away."""
    M.FILING.clear()
    M._reset_state()
    monkeypatch.setattr(fk, "finger_points", lambda: np.array([[0.0, 0.0, M.TOP + 0.5]]))
    yield


def run(*tops_mm, **kw):
    """Score frame by frame with the sections' heights above the line (mm)."""
    out = None
    for tops in tops_mm:
        M.FILING.h[:] = M.TOP + M.LINE + np.asarray(tops, float) * MM
        f = {"file": file_state(**kw)}
        out = (M.score(f), M.success(f))
    return out


def test_the_start_scores_zero():
    assert run((1.0, 1.0, 1.0, 1.0)) == (0.0, False)


def test_a_third_of_a_millimetre_off_everywhere_is_a_quarter():
    assert run((0.62, 0.64, 0.66, 0.68)) == (0.25, False)


def test_not_yet_cut_everywhere_is_zero():
    assert run((0.5, 0.6, 0.7, 0.8)) == (0.0, False)


def test_within_0_4_mm_is_half():
    assert run((0.35, 0.3, 0.3, 0.25)) == (0.5, False)


def test_to_the_line_flat_with_the_file_on_its_rest_is_full():
    assert run((0.1, 0.09, 0.08, 0.1)) == (1.0, True)


def test_to_the_line_but_not_flat_is_half():
    assert run((0.18, 0.1, 0.05, 0.0)) == (0.5, False)


def test_to_the_line_with_the_file_left_on_the_block_is_half():
    assert run((0.1, 0.1, 0.1, 0.1), on_rest=False) == (0.5, False)


def test_the_file_still_moving_is_not_success_yet():
    assert run((0.1, 0.1, 0.1, 0.1), moving=True) == (1.0, False)


def test_a_gripper_still_on_the_file_is_half(monkeypatch):
    handle = file_state().to_world((0.0, M.HANDLE_Y, M.HANDLE_Z))
    monkeypatch.setattr(fk, "finger_points", lambda: handle[None, :])
    assert run((0.1, 0.1, 0.1, 0.1)) == (0.5, False)


def test_filing_below_the_line_spoils_it_for_good():
    assert run((0.1, 0.0, -0.08, 0.1)) == (0.0, False)
    assert run((0.1, 0.1, 0.1, 0.1)) == (0.0, False)


def test_touching_the_line_is_not_spoilt():
    assert run((0.0, -0.04, 0.0, 0.05)) == (1.0, True)


def test_reset_clears_the_latch():
    run((-0.2, -0.2, -0.2, -0.2))
    M._reset_state()
    assert run((0.1, 0.1, 0.1, 0.1)) == (1.0, True)


# -- the demo ----------------------------------------------------------------------------------------------------

# The per-step limits of inspect-robots' DeltaLimitGuard for the YAM action space, as recorded in a run's
# configuration: six joints, then the gripper (0-1, 5% a step).
GUARD_MAX_DELTA = np.array([0.287979, 0.1832595, 0.1570795, 0.1631885, 0.15708, 0.20944, 0.05])


@pytest.mark.parametrize("arm", ["left", "right"])
def test_the_demo_reaches_every_target_along_its_path(arm):
    """Forward kinematics of the commanded joints against the target, every 50 ms (the arms' joint limits stop them
    pointing straight down much above 200 mm, and the planner's own error figure does not always show it)."""
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


def test_both_hands_stroke_together():
    plans = M.demo()
    starts = {}
    for arm, wps in plans.items():
        t, out = 0.0, []
        for w in wps:
            if w.label == "push stroke":
                out.append(round(t, 3))
            t += w.duration
        starts[arm] = out
    assert starts["left"] == starts["right"] and len(starts["left"]) == 4 * len(M.PUSHES)
