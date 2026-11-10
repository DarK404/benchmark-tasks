"""Checks of ratchet_bolt.py's ratchet-and-bolt model, scoring and demo, without the simulator. Needs yam_arena (the
official stack):

    python -m pytest -q test_ratchet_bolt.py
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

TASK = load(Path(__file__).with_name("ratchet_bolt.py"))
M = TASK.module
import ratchetkit as rk  # noqa: E402  (the task file put its folder on sys.path)

MM = 0.001
ZERO = np.zeros(3)
ON = np.array([*M.BOLT_XY, M.TOP + M.SEAT + M.WASHER_T])  # the socket's mouth bottomed on the bolt's washer
HEAD_TOP = M.TOP + M.HEAD_TOP


def rz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def frames(mouth, yaw_deg=-115.0, lever_deg=-30.0, spin=0.0):
    """The ratchet with its socket's mouth at ``mouth``, its handle pointing ``yaw_deg``, its lever at ``lever_deg``."""
    r = rz(yaw_deg)
    mouth = np.asarray(mouth, float)
    pivot = mouth + r @ np.array(M.LEVER_AT)
    return {"ratchet": rk.Frame(pos=mouth, rot=r, vel=ZERO.copy(), spin=np.array([0.0, 0.0, spin]), com=mouth),
            "lever": rk.Frame(pos=pivot, rot=r @ rz(lever_deg), vel=ZERO.copy(), spin=ZERO.copy(), com=pivot)}


@pytest.fixture(autouse=True)
def fresh():
    M.MODEL.clear()
    M.MODEL.origin = np.zeros(3)
    yield


def step(mouth, yaw=-115.0, lever=-30.0):
    acc = {}
    M.MODEL.update(frames(mouth, yaw, lever), acc)
    return acc


def seat(lever=30.0, yaw=-115.0):
    step(ON + [0, 0, 0.02], yaw, lever)
    step(ON, yaw, lever)
    assert M.MODEL.engaged


def swing(a, b, lever=30.0, n=40):
    for y in np.linspace(a, b, n):
        step(ON, y, lever)


def strokes(k, lever=30.0):
    for _ in range(k):
        swing(-115.0, -65.0, lever)
        swing(-65.0, -115.0, lever)


# -- the direction lever --------------------------------------------------------------------------------------------
def test_the_ratchet_starts_set_to_tighten():
    step(ON + [0.1, 0, 0.05], lever=-30.0)
    assert M.MODEL.setting == "tighten"


def test_the_lever_past_the_middle_sets_it_to_loosen_and_the_detent_holds_it_there():
    acc = step(ON + [0.1, 0, 0.05], lever=5.0)
    assert M.MODEL.setting == "loosen" and acc["lever"][1][2] > 0  # pushed on towards +30 degrees


def test_the_lever_spawns_on_tighten_in_the_task():
    spec = {o.name: o for o in M.OBJECTS}
    r, lv = spec["ratchet"], spec["lever"]
    yaw = lambda q: 2 * math.atan2(q[2], q[3])  # noqa: E731  (x, y, z, w) about z
    rel = math.degrees(math.remainder(yaw(lv.quat) - yaw(r.quat), 2 * math.pi))
    assert rel == pytest.approx(-M.LEVER_DEG)


# -- the socket on the bolt -----------------------------------------------------------------------------------------
def test_lowered_on_the_axis_the_socket_goes_on():
    seat()


def test_within_4_mm_the_sockets_chamfer_steers_it_onto_the_axis():
    acc = step(ON + [0.003, 0, M.HEAD_H + 0.001])
    assert acc["ratchet"][0][0] < 0 and not M.MODEL.engaged


def test_further_off_the_socket_stands_on_the_head():
    acc = step(ON + [0.006, 0, M.HEAD_H - 0.002])
    assert acc["ratchet"][0][2] > 0 and acc["ratchet"][0][0] == 0 and not M.MODEL.engaged


def test_the_socket_bottoms_on_the_washer():
    seat()
    acc = step(ON - [0, 0, 0.002])
    assert acc["ratchet"][0][2] > 0


def test_lifted_off_it_comes_off():
    seat()
    step(ON + [0, 0, M.HEAD_H + 0.003])
    assert not M.MODEL.engaged


def test_the_head_collider_is_moved_away_only_while_the_socket_is_over_it():
    step(ON + [0.1, 0, 0.05])
    assert not M.MODEL.socket_near and M.MODEL.poses["bolt_solid"][0][2] == pytest.approx(M.MODEL.head_z_mid)
    step(ON + [0.002, 0, M.HEAD_H + 0.002])
    assert M.MODEL.socket_near and M.MODEL.poses["bolt_solid"][0][0] < 0


# -- the clutch and the bolt ----------------------------------------------------------------------------------------
def test_on_loosen_a_pull_breaks_the_bolt_loose_and_turns_it_out():
    seat()
    swing(-115.0, -65.0)
    wind_up = math.degrees(M.MODEL.running / M.MODEL.k_tor)  # the spring's wind-up at the end of the pull
    assert M.MODEL.loose and math.degrees(M.MODEL.turned) == pytest.approx(50.0 - wind_up, abs=1.5)


def test_the_bolt_resists_up_to_its_breakaway_torque():
    seat()
    swing(-115.0, -115.0 + 0.5, n=5)  # 0.5 degrees: under breakaway (0.69 degrees of wind-up)
    assert M.MODEL.turned == 0.0 and M.MODEL.torque == pytest.approx(M.MODEL.k_tor * math.radians(0.5), rel=0.02)


def test_the_back_swing_clicks_and_turns_nothing():
    seat()
    swing(-115.0, -65.0)
    t = M.MODEL.turned
    swing(-65.0, -115.0)
    assert M.MODEL.turned == pytest.approx(t) and M.MODEL.torque == 0.0


def test_eight_strokes_bring_the_bolt_a_full_turn_out():
    seat()
    strokes(8)
    assert M.MODEL.turns() >= 1.0 and not M.MODEL.stretched


def test_the_head_is_drawn_turning_and_rising():
    seat()
    strokes(8)
    swing(-115.0, -100.0)  # (the head is re-posed only when it moves)
    pos, quat = M.MODEL.poses["bolt_head"]
    assert pos[2] == pytest.approx(M.MODEL.head_z_mid + M.MODEL.pitch * M.MODEL.turns(), abs=1e-6)


def test_left_on_tighten_the_pull_free_wheels_and_the_back_swing_overtightens():
    seat(lever=-30.0)
    swing(-115.0, -65.0, lever=-30.0)
    assert M.MODEL.turned == 0.0
    swing(-65.0, -115.0, lever=-30.0)
    assert M.MODEL.stretched


# -- scoring, on the task's layout ----------------------------------------------------------------------------------
def states(in_tray=True):
    out = {}
    for spec in M.OBJECTS:
        pos = np.array(spec.pos, float)
        quat = spec.quat if spec.quat else rk.yaw_quat(spec.yaw)
        if spec.name in ("ratchet", "lever") and not in_tray:
            pos[:2] += np.array([0.03, 0.03])
        out[spec.name] = rk.body_state(spec, pos, quat, ZERO, ZERO)
    return out


def test_a_full_turn_out_and_the_ratchet_back_succeeds():
    seat()
    strokes(8)
    assert M.score(states()) == 1.0 and M.success(states())


def test_the_ratchet_not_back_is_half():
    seat()
    strokes(8)
    assert M.score(states(in_tray=False)) == 0.5 and not M.success(states(in_tray=False))


def test_broken_loose_but_under_a_turn_is_a_quarter():
    seat()
    strokes(2)
    assert 30.0 <= math.degrees(M.MODEL.turned) < 360.0 and M.score(states()) == 0.25


def test_a_click_or_two_is_nothing():
    seat()
    swing(-115.0, -95.0)
    assert math.degrees(M.MODEL.turned) < 30.0 and M.score(states()) == 0.0


def test_an_overtightened_bolt_scores_nothing_for_good():
    seat(lever=-30.0)
    swing(-115.0, -65.0, lever=-30.0)
    swing(-65.0, -115.0, lever=-30.0)
    step(ON + [0, 0, 0.03], lever=-30.0)
    seat(lever=30.0)
    strokes(12)
    assert M.MODEL.turns() >= 1.0 and M.score(states()) == 0.0 and not M.success(states())


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


def test_the_demo_flips_the_lever_before_the_ratchet_is_picked_up():
    demo = M.demo()
    t_flip = sum(w.duration for w in demo["left"][:3])
    labels = [w.label for w in demo["right"]]
    t_grip = sum(w.duration for w in demo["right"][:labels.index("grip")])
    assert t_flip < t_grip


def test_the_demo_swings_nine_strokes():
    labels = [w.label for w in M.demo()["right"]]
    assert sum(lab.startswith("pull") for lab in labels) == 9
