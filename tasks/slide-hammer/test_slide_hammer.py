"""Checks of slide_hammer.py's pin-and-blows model, scoring and demo, without the simulator. Needs yam_arena (the
official stack):

    python -m pytest -q test_slide_hammer.py
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

TASK = load(Path(__file__).with_name("slide_hammer.py"))
M = TASK.module
import hammerkit as hk  # noqa: E402  (the task file put its folder on sys.path)

MM = 0.001
ZERO = np.zeros(3)
EYE = np.array([*M.EYE_XY, M.TOP + M.SHAFT_Z])  # the peg's origin hooked in, at the eye's middle
HX = M._hx


def rz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


R0 = rz(math.degrees(M.HAMMER_YAW))


def frames(peg, weight_at=None, speed=0.0, rot=R0):
    """The slide hammer with its peg's origin at ``peg``, its weight ``weight_at`` along the shaft moving at
    ``speed`` (towards the handle) relative to it."""
    peg = np.asarray(peg, float)
    hx = rot[:, 0]
    w = peg + hx * (M.FRONT_AT if weight_at is None else weight_at)
    return {"hammer": hk.Frame(pos=peg, rot=rot, vel=ZERO.copy(), spin=ZERO.copy(), com=peg),
            "weight": hk.Frame(pos=w, rot=rot, vel=hx * speed, spin=ZERO.copy(), com=w)}


@pytest.fixture(autouse=True)
def fresh():
    M.MODEL.clear()
    M.MODEL.origin = np.zeros(3)
    yield


def step(peg, weight_at=None, speed=0.0, rot=R0):
    acc = {}
    M.MODEL.update(frames(peg, weight_at, speed, rot), acc)
    return acc


def hook_in():
    step(EYE + [0, 0, 0.03])
    step(EYE)
    assert M.MODEL.engaged


def blow(v, peg=None):
    """The weight thrown from the front to the end stop, arriving at ``v`` m/s, then eased back."""
    peg = EYE + HX * M.MODEL.shown if peg is None else peg  # the slide hammer comes out with the pin
    for x in np.linspace(M.FRONT_AT + 0.02, M.STOP_AT - 0.02, 6):
        step(peg, x, v)
    step(peg, M.STOP_AT, 0.0)  # stopped dead by the end stop
    step(peg, M.FRONT_AT + 0.02, -0.1)  # eased back


def settle():
    for _ in range(int(0.6 / M.MODEL.dt)):  # the eye catches up with the pin
        step(EYE + HX * M.MODEL.shown, M.FRONT_AT + 0.02, 0.0)


# -- the hook in the eye --------------------------------------------------------------------------------------------
def test_lowered_into_the_eye_the_hook_is_in():
    hook_in()


def test_anywhere_in_the_25_mm_hole_the_hook_is_in():
    step(EYE + [0.009, 0.0, 0.0])
    assert M.MODEL.engaged


def test_on_the_ring_or_beside_it_the_hook_is_not_in():
    step(EYE + [0.016, 0.0, 0.0])
    assert not M.MODEL.engaged
    step(EYE + [0.04, 0.0, 0.0])
    assert not M.MODEL.engaged


def test_just_over_the_holes_edge_the_pegs_round_tip_is_guided_in():
    tip_just_above = M.RING_TOP + 1.0 * MM - (M.SHAFT_Z - M.PEG_LEN)
    acc = step(EYE + [M.MODEL.hole_r - M.MODEL.peg_r + 2.0 * MM, 0.0, tip_just_above])
    assert not M.MODEL.engaged and acc["hammer"][0][0] < 0


def test_hooked_in_the_shaft_is_kept_from_rolling():
    hook_in()
    tilted = rz(math.degrees(M.HAMMER_YAW)) @ np.array([[1, 0, 0], [0, math.cos(0.2), -math.sin(0.2)],
                                                         [0, math.sin(0.2), math.cos(0.2)]])
    acc = step(EYE, rot=tilted)
    assert abs(float(np.dot(acc["hammer"][1], HX))) > 0.05


def test_lifted_out_the_hook_comes_out():
    hook_in()
    step(EYE + [0, 0, 0.03])
    assert not M.MODEL.engaged


# -- the blows ------------------------------------------------------------------------------------------------------
def test_a_blow_pulls_the_pin_by_how_fast_the_weight_strikes():
    hook_in()
    blow(0.7)
    assert M.MODEL.advance == pytest.approx(M.MODEL.gain * (0.7 - M.MODEL.v0)) and M.MODEL.blows == 1


def test_a_slow_arrival_at_the_stop_moves_nothing():
    hook_in()
    blow(0.2)
    assert M.MODEL.advance == 0.0 and M.MODEL.blows == 0


def test_a_steady_pull_on_the_weight_moves_nothing():
    hook_in()
    for x in np.linspace(M.FRONT_AT, M.STOP_AT, 40):
        step(EYE, x, 0.08)
    for _ in range(200):  # held against the stop
        step(EYE, M.STOP_AT, 0.0)
    assert M.MODEL.advance == 0.0


def test_blows_with_the_hook_not_in_move_nothing():
    step(EYE + [0.05, 0.0, 0.03])
    blow(0.8, peg=EYE + [0.05, 0.0, 0.03])
    assert M.MODEL.advance == 0.0


def test_slammed_the_other_way_the_weight_drives_the_pin_back_in():
    hook_in()
    blow(0.7)
    before = M.MODEL.advance
    for x in np.linspace(M.STOP_AT - 0.02, M.FRONT_AT + 0.02, 6):
        step(EYE, x, -0.7)
    step(EYE, M.FRONT_AT - 1.0 * MM, 0.0)
    assert M.MODEL.advance < before


def test_ten_good_blows_pull_the_pin_out_and_it_hangs_on_the_hook():
    hook_in()
    for _ in range(10):
        blow(0.65)
        settle()
    assert M.MODEL.out and not M.MODEL.engaged
    pin = M.MODEL.poses["pin"][0]
    assert pin[2] < EYE[2] and np.linalg.norm(pin[:2] - (EYE + HX * M.MODEL.shown)[:2]) < 0.02
    assert M.MODEL.poses["eye_solid"][0][0] < -0.5  # its collider put away


def test_the_eye_follows_the_pin_gradually():
    hook_in()
    blow(0.8)
    assert 0.0 < M.MODEL.shown < M.MODEL.advance
    settle()
    assert M.MODEL.shown == pytest.approx(M.MODEL.advance)


# -- scoring, on the task's layout ----------------------------------------------------------------------------------
def states(along=0.0, across=0.0):
    out = {}
    for spec in M.OBJECTS:
        pos = np.array(spec.pos, float)
        quat = spec.quat if spec.quat else hk.yaw_quat(spec.yaw)
        if spec.name in ("hammer", "weight"):
            pos += HX * along + np.array([across, 0.0, 0.0])
        out[spec.name] = hk.body_state(spec, pos, quat, ZERO, ZERO)
    return out


def pin_out():
    hook_in()
    for _ in range(10):
        blow(0.65)
        settle()
    assert M.MODEL.out


def test_pin_out_and_the_slide_hammer_back_on_its_rack_succeeds():
    pin_out()
    assert M.score(states()) == 1.0 and M.success(states())


def test_on_its_rack_it_may_lie_anywhere_along_its_cradles():
    pin_out()
    assert M.score(states(along=0.02)) == 1.0


def test_pin_out_but_the_slide_hammer_not_back_is_half():
    pin_out()
    assert M.score(states(across=0.05)) == 0.5 and not M.success(states(across=0.05))


def test_the_pin_started_but_not_out_is_a_quarter():
    hook_in()
    blow(0.8)
    assert M.score(states()) == 0.25


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


def test_the_two_hands_stay_a_gripper_width_apart_on_the_shaft():
    """Each gripper is about 10 cm across its closing direction: the steadying hand and the weight's hand keep
    further apart than that."""
    assert M.FRONT_AT - M.FRONT_GRIP > 0.11


def test_each_throw_aims_past_the_stop_and_follows_the_slide_hammer_out():
    aims = [w.pos[1] for w in M.demo()["left"] if w.label.startswith("blow")]
    stop_y = M.EYE_XY[1] + M.STOP_AT * HX[1]
    assert len(aims) == M.BLOWS and all(a > stop_y + 0.02 for a in aims) and aims == sorted(aims)
