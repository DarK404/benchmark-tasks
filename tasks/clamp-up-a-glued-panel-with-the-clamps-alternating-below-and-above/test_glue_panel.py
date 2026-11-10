"""Checks of glue_panel.py's clamp model, scoring and demo, without the simulator. Needs yam_arena (the official
stack):

    python -m pytest -q test_glue_panel.py
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

TASK = load(Path(__file__).with_name("glue_panel.py"))
M = TASK.module
import gluekit as gk  # noqa: E402  (the task file put its folder on sys.path)

MM = 0.001
ZERO = np.zeros(3)
EYE = np.eye(3)
K = 3000.0


def frame(pos, vel=ZERO, rot=EYE):
    pos = np.asarray(pos, float)
    return gk.Frame(pos=pos, rot=np.asarray(rot, float), vel=np.asarray(vel, float), spin=ZERO.copy(), com=pos)


# -- the clamp model, on one clamp: its bar's middle at the origin, the fixed pad at x +50 mm, the boards 50 mm wide
# and 12 mm thick, their pad-height middles at z 0 --------------------------------------------------------------------
JAW_PAD_X = 13.0 * MM


def model(gripped=False):
    c = gk.Clamp("bar", "jaw", fixed_x=0.050, jaw_pad_x=JAW_PAD_X, fixed_z=0.0, jaw_pad_z=0.0)
    m = gk.Clamps([c], [("a", -0.025, -1), ("b", 0.025, 1)], half_len=0.08, pad_half_y=0.012, half_thick=0.006,
                  k_pad=K, pad_damping=0.0)
    m.clear()
    m._gripped = lambda jaw, fj: gripped
    return m, c


def frames(jaw_pad_at, a_x=-0.025, b_x=0.025, a_z=0.0):
    """The jaw placed so its pad face is at x ``jaw_pad_at``; boards A and B at ``a_x``, ``b_x``."""
    return {"bar": frame((0, 0, 0)), "jaw": frame((jaw_pad_at - JAW_PAD_X, 0, 0)), "a": frame((a_x, 0, a_z)),
            "b": frame((b_x, 0, 0))}


def test_both_pads_on_the_work_squeeze_both_edges_equally():
    m, c = model(gripped=True)
    acc = {}
    m.update(frames(-0.049, b_x=0.0255), acc)  # the jaw pad 1 mm into A, the fixed pad 0.5 mm into B
    fa, fb = acc["a"][0], acc["b"][0]
    assert fa[0] == pytest.approx(K * 1.5 * MM) and fb[0] == pytest.approx(-K * 1.5 * MM)
    assert np.allclose(fa + fb, 0.0)  # no net push on the panel
    assert c.force == pytest.approx(K * 1.5 * MM) and c.engaged == {"jaw", "fixed"}


def test_one_pad_alone_pushes_its_edge_towards_the_other_jaw():
    m, c = model(gripped=True)
    acc = {}
    m.update(frames(-0.049, b_x=0.020), acc)  # B 5 mm short of the fixed pad
    assert acc["a"][0][0] == pytest.approx(K * 1.0 * MM) and "b" not in acc and c.engaged == {"jaw"}


def test_a_pad_lowered_over_a_board_is_not_on_its_edge():
    m, c = model(gripped=True)
    acc = {}
    m.update(frames(-0.040), acc)  # 10 mm inside A's edge without ever having borne on it
    assert "a" not in acc and c.force == 0.0


def test_a_pad_driven_past_its_bite_keeps_pushing_at_its_limit():
    m, c = model(gripped=True)
    m.update(frames(-0.049), {})  # bearing on A's edge
    acc = {}
    m.update(frames(-0.040), acc)  # then driven 10 mm in
    assert acc["a"][0][0] == pytest.approx(K * m.max_bite)


def test_a_pad_above_the_board_does_not_touch_it():
    m, c = model(gripped=True)
    acc = {}
    m.update(frames(-0.049, a_z=0.010), acc)  # A 10 mm below the pad's middle
    assert "a" not in acc


def test_a_bearing_pad_stays_on_till_it_is_clear():
    m, c = model(gripped=True)
    m.update(frames(-0.049, b_x=0.0255), {})
    m.update(frames(-0.049, b_x=0.0248), {})  # B 0.2 mm off the fixed pad: still bearing
    assert c.engaged == {"jaw", "fixed"}
    m.update(frames(-0.049, b_x=0.0240), {})  # 1 mm off: clear
    assert c.engaged == {"jaw"}


def test_a_gripped_jaw_takes_up_the_narrowest_opening_and_is_held_there_when_let_go():
    m, c = model(gripped=True)
    for x in (-0.060, -0.052, -0.049, -0.0485, -0.049, -0.050):  # slid in, pressed, eased back
        m.update(frames(x, b_x=0.0255), {})
    assert c.lock == pytest.approx(0.0985 + m.backlash)
    m._gripped = lambda jaw, fj: False
    m.update(frames(-0.050, b_x=0.0255), {})  # let go 1.5 mm further open than the narrowest
    assert c.pin == pytest.approx(0.0985 + m.backlash)
    pos, _ = m.poses["jaw"]
    assert pos[0] == pytest.approx(0.050 - c.pin - JAW_PAD_X)  # the jaw put back where the clutch holds it
    assert c.force == pytest.approx(K * (1.5 * MM - m.backlash + 0.5 * MM))


def test_a_jaw_never_worked_is_held_where_it_lies():
    m, c = model(gripped=False)
    acc = {}
    m.update(frames(-0.065), acc)
    assert c.pin == pytest.approx(0.115) and "a" not in acc and c.force == 0.0


def test_the_jaws_friction_resists_sliding_up_to_its_limit():
    m, c = model(gripped=True)
    m.update(frames(-0.060), {})
    acc = {}
    m.update(frames(-0.055), acc)  # slid 5 mm along the bar in one substep
    assert abs(acc["jaw"][0][0]) == pytest.approx(m.slide_force)
    assert acc["jaw"][0][0] < 0  # against the closing


# -- scoring, on the task's layout ---------------------------------------------------------------------------------
PANEL = np.array([*M.PANEL_XY, M.TOP])


def states(gap=0.0, spare_y=0.0, board_vel=ZERO, b_tilt_deg=0.0):
    """Every object as it lies at the end of a good glue-up: the joint closed (``gap`` open), the boards on the lower
    bars, the spare across the middle."""
    out = {}
    for s in M.OBJECTS:
        pos = np.array(s.pos, float)
        quat = s.quat if s.quat else gk.yaw_quat(s.yaw)
        if s.name == "board_a":
            pos[0] = M.PANEL_XY[0] - M.BOARD_W / 2 - gap / 2
        if s.name == "board_b":
            pos[0] = M.PANEL_XY[0] + M.BOARD_W / 2 + gap / 2
            t = math.radians(b_tilt_deg) / 2
            quat = (math.sin(t), 0.0, 0.0, math.cos(t))
        if s.name.startswith("spare"):
            pos[:2] = (pos[0] - M.RACK_XY[0] + M.PANEL_XY[0], M.PANEL_XY[1] + spare_y)
            pos[2] = pos[2] - (M.TOP + M.RACK_H + M.BAR_T) + M.TOP + M.BAR_T + M.BOARD_T + M.BAR_T
        vel = board_vel if s.name.startswith("board") else ZERO
        out[s.name] = gk.body_state(s, pos, quat, vel, ZERO)
    return out


def set_clamps(forces=(2.0, 2.0, 2.0), held=(True, True, True)):
    for c, f, h in zip(M.MODEL.clamps, forces, held):
        c.force, c.engaged = f, {"jaw", "fixed"}
        c.pin = 0.099 if h else math.nan


@pytest.fixture(autouse=True)
def fresh():
    M._reset_state()
    M.MODEL.clear()
    yield


def test_the_layout_alternates_below_above_below():
    st = states()
    assert [M.clamp_side(st, c) for c in M.MODEL.clamps] == [-1, -1, 1]
    assert M.pattern_ok(st, M.MODEL.clamps)


def test_a_closed_joint_with_three_snug_alternating_clamps_left_alone_succeeds():
    set_clamps()
    st = states()
    assert np.allclose(M.gaps(st), 0.0, atol=1e-9)
    assert M.score(st) == 1.0 and M.success(st)


def test_not_still_is_not_yet_done():
    set_clamps()
    assert M.score(states(board_vel=np.array([0.02, 0, 0]))) == 0.5
    assert not M.success(states(board_vel=np.array([0.02, 0, 0])))


def test_a_joint_left_open_scores_nothing():
    set_clamps()
    assert M.score(states(gap=0.5 * MM)) == 0.0


def test_a_board_ridden_up_is_not_flat():
    set_clamps()
    assert M.score(states(b_tilt_deg=5.0)) == 0.0


def test_the_spare_left_slack_is_half():
    set_clamps(forces=(2.0, 2.0, 0.5))
    assert M.score(states()) == 0.5


def test_the_spare_not_between_the_others_is_half():
    set_clamps()
    st = states(spare_y=-75.0 * MM)  # beyond the right-hand lower clamp: below, below side by side
    assert not M.pattern_ok(st, M.MODEL.clamps)
    assert M.score(st) == 0.5


def test_a_hand_still_on_a_jaw_does_not_count_that_clamp():
    set_clamps(held=(True, True, False))
    assert M.score(states()) == 0.5


def test_one_clamp_snug_is_a_quarter():
    set_clamps(forces=(2.0, 0.0, 0.0))
    assert M.score(states()) == 0.25


def test_a_clamp_forced_too_hard_starves_the_joint_for_good():
    set_clamps(forces=(2.0, 2.0, 8.0))
    for _ in range(round(M.CRUSH_S * M.CONTROL_HZ)):
        M.score(states())
    set_clamps()  # eased back to snug
    assert M.score(states()) == 0.0 and not M.success(states())


def test_a_moment_over_the_limit_is_not_a_crush():
    set_clamps(forces=(2.0, 2.0, 8.0))
    for _ in range(3):
        M.score(states())
    set_clamps()
    assert M.score(states()) == 1.0


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


def test_one_arm_at_a_time_works_near_the_panel():
    """The arms ran into each other working the lower clamps together: at most one hand below 10 cm at any time."""
    plans = {a: ArmPlan(p, station.BASES[a]) for a, p in M.demo().items()}
    end = max(p.duration for p in plans.values())
    for t in np.arange(0.0, end, 0.05):
        low = [a for a, p in plans.items() if t < p.duration and p.target(t)[0][2] < M.TOP + 0.10]
        assert len(low) <= 1, (t, low)
