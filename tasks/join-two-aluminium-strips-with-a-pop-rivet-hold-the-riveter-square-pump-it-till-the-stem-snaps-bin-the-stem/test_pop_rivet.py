"""Checks of pop_rivet.py's riveting model, scoring and demo, without the simulator. Needs yam_arena (the official
stack):

    python -m pytest -q test_pop_rivet.py
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

TASK = load(Path(__file__).with_name("pop_rivet.py"))
M = TASK.module
import rivetkit as rk  # noqa: E402  (the task file put its folder on sys.path)

MM = 0.001
ZERO = np.zeros(3)
HOLE = np.array([*M.WORK_XY, M.TOP + M.WORK_TOP])  # the hole's mouth, on the work's top
HEAD0 = np.array(M.RIVETS[0][2])  # rivet 0's head top, in the holder


def ry(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])


def frames(nose, angle_deg=17.0, tilt_deg=0.0):
    """The riveter with its nose's tip at ``nose``, tilted ``tilt_deg`` about y, its sprung handle ``angle_deg`` open."""
    r = ry(tilt_deg)
    nose = np.asarray(nose, float)
    hinge = nose + r @ np.array(M.HINGE_AT)
    return {"riveter": rk.Frame(pos=nose, rot=r, vel=ZERO.copy(), spin=ZERO.copy(), com=nose),
            "handle": rk.Frame(pos=hinge, rot=r @ ry(-angle_deg), vel=ZERO.copy(), spin=ZERO.copy(), com=hinge)}


@pytest.fixture(autouse=True)
def fresh():
    M.MODEL.clear()
    M._reset_state()
    M.MODEL.origin = np.zeros(3)
    yield


def step(nose, angle=17.0, tilt=0.0):
    acc = {}
    M.MODEL.update(frames(nose, angle, tilt), acc)
    return acc


def load_rivet():
    step(HEAD0 + [0, 0, 0.03])
    step(HEAD0 + [0, 0, 0.0005])
    assert M.MODEL.state[0] == "loaded"


def insert_rivet():
    load_rivet()
    step(HOLE + [0, 0, M.HEAD_T + M.SHANK_L + 0.003])
    step(HOLE + [0, 0, M.HEAD_T])
    assert M.MODEL.state[0] == "inserted"


def squeeze(times=3, nose=None, tilt=0.0):
    nose = HOLE + [0, 0, M.HEAD_T - 0.0005] if nose is None else nose
    for _ in range(times):
        for a in np.linspace(17.0, M.CLOSED_DEG, 12):
            step(nose, a, tilt)
        for a in np.linspace(M.CLOSED_DEG, 17.0, 12):
            step(nose, a, tilt)


# -- the riveting model --------------------------------------------------------------------------------------------
def test_lowered_over_a_stem_the_riveter_loads_the_rivet():
    load_rivet()
    assert M.MODEL.active == 0


def test_off_the_stem_the_nose_stands_on_its_tip_and_loads_nothing():
    acc = step(HEAD0 + [0.004, 0.0, M.STEM_L - 0.002])
    assert M.MODEL.state[0] == "holder" and acc["riveter"][0][2] > 0  # pushed up


def test_the_nosepiece_mouth_steers_a_stem_within_3_mm_onto_the_axis():
    acc = step(HEAD0 + [0.002, 0.0, 0.01])
    assert acc["riveter"][0][0] < 0  # drawn back onto the stem's axis


def test_over_the_hole_the_rivet_goes_in():
    insert_rivet()


def test_off_the_hole_the_rivets_end_stands_on_the_strip():
    load_rivet()
    acc = step(HOLE + [0.004, 0.0, M.HEAD_T + M.SHANK_L - 0.002])
    assert M.MODEL.state[0] == "loaded" and acc["riveter"][0][2] > 0


def test_three_squeezes_set_the_rivet_and_snap_the_stem():
    insert_rivet()
    squeeze(3)
    assert M.MODEL.set_ok() and M.MODEL.stem_in_nose and M.MODEL.pull >= M.MODEL.set_pull


def test_two_squeezes_do_not_set_it():
    insert_rivet()
    squeeze(2)
    assert M.MODEL.state[0] == "inserted" and not M.MODEL.spoilt


def test_pulled_with_the_riveter_tilted_the_rivet_is_set_crooked():
    insert_rivet()
    squeeze(1, tilt=8.0)
    assert M.MODEL.spoilt and M.MODEL.spoil_reason == "crooked"


def test_pulled_with_the_nose_off_the_head_the_rivet_is_set_loose():
    insert_rivet()
    squeeze(1, nose=HOLE + [0, 0, M.HEAD_T + 0.0015])  # just off the head, not pressed on it
    assert M.MODEL.spoilt and M.MODEL.spoil_reason == "loose"


def test_lifted_before_any_pull_the_rivet_comes_out_with_the_riveter():
    insert_rivet()
    step(HOLE + [0, 0, M.HEAD_T + 0.01])
    assert M.MODEL.state[0] == "loaded"


def test_lifted_after_a_part_pull_the_rivet_is_loose():
    insert_rivet()
    for a in np.linspace(17.0, 5.0, 6):
        step(HOLE + [0, 0, M.HEAD_T - 0.0005], a)
    step(HOLE + [0, 0, M.HEAD_T + 0.01], 5.0)
    assert M.MODEL.spoilt


def test_the_nose_is_held_on_the_rivet_in_the_hole():
    insert_rivet()
    acc = step(HOLE + [0.001, 0.0, M.HEAD_T - 0.0005])
    assert acc["riveter"][0][0] == pytest.approx(-M.MODEL.k_side * 0.001, rel=0.05)


def test_the_spent_stem_drops_out_when_the_handles_open_wide():
    insert_rivet()
    squeeze(3)
    over = np.array([*M.TRAY_XY, M.TOP + 0.03])
    step(over, 17.0)
    assert M.MODEL.stem_in_nose
    step(over, M.RELEASE_DEG + 1.0)
    assert not M.MODEL.stem_in_nose and M.MODEL.stem_place == "tray"


def test_a_stem_let_go_over_the_work_is_left_on_it():
    insert_rivet()
    squeeze(3)
    step(HOLE + [0.01, 0.0, 0.01], M.RELEASE_DEG + 1.0)
    assert M.MODEL.stem_place == "work"


# -- scoring, on the task's layout ---------------------------------------------------------------------------------
def states(in_holster=True, handle_open=25.0):
    out = {}
    for spec in M.OBJECTS:
        pos = np.array(spec.pos, float)
        quat = spec.quat if spec.quat else rk.yaw_quat(spec.yaw)
        if spec.name in ("riveter", "handle") and not in_holster:
            pos[:2] += np.array([0.05, 0.05])
        out[spec.name] = rk.body_state(spec, pos, quat, ZERO, ZERO)
    return out


def finished(place="tray"):
    insert_rivet()
    squeeze(3)
    over = np.array([*M.TRAY_XY, M.TOP + 0.03]) if place == "tray" else HOLE + [0.01, 0.0, 0.01]
    step(over, M.RELEASE_DEG + 1.0)


def test_set_binned_and_holstered_succeeds():
    finished()
    assert held(states()) == (1.0, True)


def held(st, n=M.STILL_STEPS):
    """success() then score() on the same state ``n`` times, as the run calls them; the last (score, success)."""
    for _ in range(n):
        ok = M.success(st)
        out = (M.score(st), ok)
    return out


def test_full_only_after_a_second_still():
    finished()
    assert held(states(), M.STILL_STEPS - 1)[0] == 0.5
    assert held(states(), 1)[0] == 1.0


def test_the_rivet_is_rated_for_the_two_strips():
    assert M.GRIP_RANGE[0] <= 2 * M.STRIP_T <= M.GRIP_RANGE[1]


def test_the_stem_left_on_the_work_is_half():
    finished("work")
    assert M.score(states()) == 0.5


def test_the_riveter_not_back_in_its_holster_is_half():
    finished()
    assert M.score(states(in_holster=False)) == 0.5


def test_a_rivet_in_the_hole_not_set_is_a_quarter():
    insert_rivet()
    assert M.score(states()) == 0.25


def test_a_rivet_set_crooked_scores_nothing_for_good():
    insert_rivet()
    squeeze(1, tilt=8.0)
    squeeze(2)
    step(np.array([*M.TRAY_XY, M.TOP + 0.03]), M.RELEASE_DEG + 1.0)
    assert M.score(states()) == 0.0 and not M.success(states())


def test_nothing_done_scores_nothing():
    assert M.score(states()) == 0.0


# -- the scripted demo -----------------------------------------------------------------------------------------------
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


def test_the_demo_pumps_three_full_strokes():
    labels = [w.label for w in M.demo()["right"]]
    assert sum(lab.startswith("squeeze") for lab in labels) == 3
