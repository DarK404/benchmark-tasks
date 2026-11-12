"""Checks of compression_joint.py's compression-joint model, scoring and demo, without the simulator. Needs yam_arena
(the official stack):

    python -m pytest -q test_compression_joint.py
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

TASK = load(Path(__file__).with_name("compression_joint.py"))
M = TASK.module
import fittingkit as jk  # noqa: E402  (the task file put its folder on sys.path)

MM = 0.001
ZERO = np.zeros(3)
AXIS = np.array([M.AX, M.AY])
STUB = M.TOP + M.STUB
HOME = STUB - M.SOCKET


def rz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def fr(pos, rot=None, vel=None):
    pos = np.asarray(pos, float)
    return jk.Frame(pos=pos, rot=np.eye(3) if rot is None else rot, vel=ZERO.copy() if vel is None else vel,
                    spin=ZERO.copy(), com=pos)


FAR = {"olive": fr([0.27, 0.06, M.TOP + 0.0135]), "nut": fr([0.27, 0.11, M.TOP + 0.016]),
       "spanner": fr([0.27, -0.17, M.TOP + 0.0085], rz(180.0))}


@pytest.fixture(autouse=True)
def fresh():
    M.MODEL.clear()
    M.MODEL.origin = np.zeros(3)
    M._reset_state()
    yield


def step(pipe_z=HOME, **parts):
    acc = {}
    frames = {"pipe": fr([*AXIS, pipe_z]), **FAR, **{k: v for k, v in parts.items()}}
    M.MODEL.update(frames, acc)
    return acc


def ring_at(z, dxy=(0.0, 0.0)):
    return fr([AXIS[0] + dxy[0], AXIS[1] + dxy[1], z])


def drop(name, z_from, z_to, n=30, **others):
    for z in np.linspace(z_from, z_to, n):
        step(**{name: ring_at(z)}, **others)


TOP_OF_PIPE = HOME + M.PIPE_L
OLIVE_SEAT = STUB - M.MODEL.cone + M.OLIVE_LEN / 2
NUT_ON_OLIVE = STUB - 1.0 * MM + M.NUT_LEN / 2


def olive_then_nut():
    drop("olive", TOP_OF_PIPE + 2 * MM, OLIVE_SEAT)
    drop("nut", TOP_OF_PIPE + 4 * MM, NUT_ON_OLIVE - 0.5 * MM, olive=ring_at(OLIVE_SEAT))
    assert M.MODEL.threaded and M.MODEL.olive_in


def spanner_on(psi):
    nz = M.MODEL._nut_z()
    return fr([*AXIS, nz], rz(psi))


def strokes(k, a=-60.0, b=-120.0, n=20):
    others = dict(olive=ring_at(OLIVE_SEAT), nut=ring_at(M.MODEL._nut_z()))
    for _ in range(k):
        for psi in np.linspace(a, b, n):
            step(spanner=spanner_on(psi), **others)
        off = fr(np.array([*AXIS, M.MODEL._nut_z()]) + 0.03 * np.array([math.cos(math.radians(b)),
                                                                           math.sin(math.radians(b)), 0.0]),
                 rz(b))
        step(spanner=off, **others)  # drawn off along its handle
        assert not M.MODEL.engaged


# -- the pipe, the olive and the nut ---------------------------------------------------------------------------------
def test_the_pipe_stood_in_the_socket_on_its_stop_is_home():
    step()
    assert M.MODEL.pipe_home()


def test_a_pipe_short_of_its_stop_is_not_home():
    step(HOME + 3 * MM)
    assert M.MODEL.pipe_in and not M.MODEL.pipe_home()


def test_the_socket_holds_the_pipes_end_on_the_axis():
    acc = {}
    M.MODEL.update({"pipe": fr([AXIS[0] + 2 * MM, AXIS[1], HOME]), **FAR}, acc)
    assert acc["pipe"][0][0] < 0


def test_a_ring_lowered_over_the_pipes_top_is_guided_on_and_rides_down_it():
    acc = step(olive=ring_at(TOP_OF_PIPE + 4 * MM, (3 * MM, 0.0)))
    assert acc["olive"][0][0] < 0
    drop("olive", TOP_OF_PIPE + 2 * MM, OLIVE_SEAT)
    assert M.MODEL.on_pipe["olive"]


def test_the_olive_rests_in_the_stubs_cone():
    drop("olive", TOP_OF_PIPE + 2 * MM, OLIVE_SEAT - 1 * MM)
    acc = step(olive=ring_at(OLIVE_SEAT - 1 * MM))
    assert acc["olive"][0][2] > 0


def test_the_nut_on_the_olive_is_on_the_thread_with_the_olive_in():
    olive_then_nut()


def test_the_nut_dropped_first_leaves_the_olive_out():
    drop("nut", TOP_OF_PIPE + 4 * MM, NUT_ON_OLIVE - 0.5 * MM)
    drop("olive", TOP_OF_PIPE + 2 * MM, NUT_ON_OLIVE + M.NUT_LEN / 2 + M.OLIVE_LEN / 2,
         nut=ring_at(NUT_ON_OLIVE))
    assert M.MODEL.threaded and not M.MODEL.olive_in


# -- the spanner and the thread ------------------------------------------------------------------------------------
def test_the_spanner_on_the_flats_turns_the_nut_clockwise():
    olive_then_nut()
    strokes(1)
    assert M.MODEL.turns() == pytest.approx(60.0 / 360.0, abs=0.01)


def test_off_the_flats_the_spanner_does_not_go_on():
    olive_then_nut()
    step(spanner=spanner_on(-60.0 + 30.0), olive=ring_at(OLIVE_SEAT), nut=ring_at(M.MODEL._nut_z()))
    assert not M.MODEL.engaged


def test_it_is_snug_after_a_turn_and_twelve_strokes_take_it_a_turn_past():
    olive_then_nut()
    strokes(6)
    assert M.MODEL.turns() == pytest.approx(1.0, abs=0.02) and M.MODEL.turns_past() < 0.05
    strokes(6)
    assert 0.75 <= M.MODEL.turns_past() <= 1.25 and M.MODEL.sealed()


def test_turned_past_one_and_a_half_turns_the_olive_is_crushed_for_good():
    olive_then_nut()
    strokes(16)
    assert M.MODEL.crushed and not M.MODEL.sealed()


def test_with_no_olive_the_nut_runs_down_and_stops_loose():
    drop("nut", TOP_OF_PIPE + 4 * MM, NUT_ON_OLIVE - 0.5 * MM)
    strokes(18)
    assert M.MODEL.turns() == pytest.approx(M.MODEL.free_turns, abs=0.02) and not M.MODEL.sealed()


def test_the_nut_is_drawn_turned_and_run_down():
    olive_then_nut()
    strokes(3)
    pos, quat = M.MODEL.poses["nut"]
    assert pos[2] < NUT_ON_OLIVE and quat[2] < 0  # turned clockwise (negative yaw)


# -- scoring, on the task's layout ----------------------------------------------------------------------------------
def states(spanner_back=True):
    out = {}
    for spec in M.OBJECTS:
        pos = np.array(spec.pos, float)
        quat = spec.quat if spec.quat else jk.yaw_quat(spec.yaw)
        if spec.name == "spanner" and not spanner_back:
            pos[:2] += [0.05, 0.05]
        out[spec.name] = jk.body_state(spec, pos, quat, ZERO, ZERO)
    return out


def held(st, n=M.STILL_STEPS):
    """success() then score() on the same state ``n`` times, as the run calls them; the last (score, success)."""
    for _ in range(n):
        ok = M.success(st)
        out = (M.score(st), ok)
    return out


def test_sealed_and_the_spanner_back_succeeds():
    olive_then_nut()
    strokes(12)
    assert held(states()) == (1.0, True)


def test_full_only_after_a_second_still():
    olive_then_nut()
    strokes(12)
    assert held(states(), M.STILL_STEPS - 1)[0] == 0.5
    assert held(states(), 1)[0] == 1.0


def test_sealed_but_the_spanner_not_back_is_half():
    olive_then_nut()
    strokes(12)
    assert M.score(states(spanner_back=False)) == 0.5


def test_assembled_right_but_not_tight_enough_is_a_quarter():
    olive_then_nut()
    strokes(8)
    assert M.MODEL.turns_past() < 0.75 and M.score(states()) == 0.25


def test_no_olive_scores_nothing():
    drop("nut", TOP_OF_PIPE + 4 * MM, NUT_ON_OLIVE - 0.5 * MM)
    strokes(12)
    assert M.score(states()) == 0.0


def test_overtightened_scores_nothing_for_good():
    olive_then_nut()
    strokes(16)
    assert M.MODEL.crushed and held(states()) == (0.0, False)


def test_past_one_and_a_quarter_turns_scores_nothing():
    olive_then_nut()
    strokes(14)
    assert 1.25 < M.MODEL.turns_past() < 1.5 and not M.MODEL.crushed
    assert held(states()) == (0.0, False)


def test_backing_off_an_overtightened_nut_does_not_undo_it():
    olive_then_nut()
    strokes(14)
    strokes(2, a=-120.0, b=-60.0)  # two strokes back
    assert 0.75 <= M.MODEL.turns_past() <= 1.25
    assert held(states()) == (0.0, False)


def test_nothing_done_scores_nothing():
    assert M.score(states()) == 0.0


# -- the scripted demo -----------------------------------------------------------------------------------------------
# The per-step limits of inspect-robots' DeltaLimitGuard for the YAM action space, as recorded in a run's
# configuration: six joints, then the gripper (0-1, 5% a step).
GUARD_MAX_DELTA = np.array([0.287979, 0.1832595, 0.1570795, 0.1631885, 0.15708, 0.20944, 0.05])


@pytest.mark.parametrize("arm", ["left", "right"])
def test_the_demo_reaches_every_target_along_its_path(arm):
    """Forward kinematics of the commanded joints against the target, every 50 ms."""
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


def test_the_demo_drops_the_olive_before_the_nut():
    labels = [w.label for w in M.demo()["left"]]
    assert labels.index("drop the olive") < labels.index("drop the nut")


def test_the_demo_keeps_the_spanner_hand_clear_until_the_left_hand_is_home():
    demo = M.demo()
    t_left_home = sum(w.duration for w in demo["left"][:[w.label for w in demo["left"]].index("home") + 1])
    labels = [w.label for w in demo["right"]]
    t_near = sum(w.duration for w in demo["right"][:labels.index("beside the valve")])
    assert t_near >= t_left_home - 0.5
