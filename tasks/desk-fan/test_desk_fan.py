"""Offline checks of desk_fan.py's model, scoring and demo, without the simulator. Needs yam_arena (the official
stack):

    python -m pytest -q test_desk_fan.py
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

TASK = load(Path(__file__).with_name("desk_fan.py"))
M = TASK.module
import fankit as fk  # noqa: E402  (the task file put its folder on sys.path)

ZERO = np.zeros(3)
MM = 0.001
AX = np.array([*M.AXIS, 0.0])


def rz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def fr(pos, rot=None):
    pos = np.asarray(pos, float)
    return fk.Frame(pos=pos, rot=np.eye(3) if rot is None else rot, vel=ZERO.copy(), spin=ZERO.copy(), com=pos)


def at(z, dx=0.0):
    return AX + np.array([dx, 0.0, M.TOP + z])


FAR = fr([1.0, 1.0, 1.0])
BLADE_ON = at(M.BLADE_SEAT)
SPIN_ON = at(M.SPIN_SEAT + M.TIGHT_TURNS * 1.5 * MM)
FRONT_ON = at(M.FRONT_SEAT)


def spinner_hand(deg):
    """Fake finger hull points: a hand closed across the spinner, its finger axis at ``deg``."""
    a = math.radians(deg)
    d = np.array([math.cos(a), math.sin(a), 0.0])
    return np.array([at(M.SPIN_SEAT) + s * d * (M.SPIN_R + 2.0 * MM) + np.array([0, 0, z]) for s in (-1, 1)
                     for z in (0.004, 0.010, 0.016)])


def blade_hand():
    """A hand closed, pressing a blade still from above, along it."""
    c = at(M.BLADE_SEAT + 8.0 * MM + M.BLADE_T / 2 + 2.0 * MM)
    a = M.HOLD_AT
    along, across = np.array([math.cos(a), math.sin(a), 0.0]), np.array([-math.sin(a), math.cos(a), 0.0])
    return np.array([c + M.HOLD_R * along + s * 0.008 * across + np.array([0.0, 0.0, z])
                     for s in (-1, 1) for z in (0.0, 0.004)])


def clip_hand(angle):
    a = math.radians(angle)
    tip = FRONT_ON + np.array([math.cos(a), math.sin(a), 0.0]) * (M.RIM_R + 3.0 * MM)
    return np.array([tip + np.array([0.0, 0.0, -0.004])])


NONE = np.zeros((0, 3))


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    M.MODEL.clear()
    M.MODEL.origin = np.zeros(3)
    M._reset_state()
    monkeypatch.setattr(fk, "finger_points", lambda robots=None: np.zeros((0, 3)))
    yield


def step(blade=None, spinner=None, front=None, right=NONE, left=NONE):
    m = M.MODEL
    m._tick = 0  # read the fake hands every step
    pts = {"left_robot": left, "right_robot": right}
    fk.finger_points = lambda robots=None, _p=pts: _p[robots[0]] if robots else np.vstack(list(_p.values()))
    m.update({"blade": blade or FAR, "spinner": spinner or FAR, "front": front or FAR}, {})


def seat_blade():
    step(blade=fr(BLADE_ON))
    assert M.MODEL.blade_on


def seat_spinner():
    seat_blade()
    step(spinner=fr(SPIN_ON))
    assert M.MODEL.spinner_on


def turn(deg, hold=True, start=-60.0, steps=None):
    """The right hand turns the spinner ``deg`` (anticlockwise positive) in 10-degree steps, a stroke at a time."""
    left = blade_hand() if hold else NONE
    done = 0.0
    while abs(done) < abs(deg) - 1e-9:
        stroke = math.copysign(min(120.0, abs(deg) - abs(done)), deg)
        for a in np.linspace(start, start + stroke, int(abs(stroke) / 10) + 1):
            step(right=spinner_hand(a), left=left)
        step(left=left)  # let go, turn the hand back
        done += stroke


# -- the model ----------------------------------------------------------------------------------------------------
def test_the_blade_set_on_the_shaft_by_its_notch_seats():
    seat_blade()


def test_the_blade_set_on_the_other_way_round_seats_too():
    step(blade=fr(BLADE_ON, rz(180.0)))
    assert M.MODEL.blade_on and math.degrees(M.MODEL.shaft) == pytest.approx(180.0)


@pytest.mark.parametrize("yaw", [30.0, -30.0, 90.0])
def test_the_blade_off_its_notch_does_not_seat(yaw):
    step(blade=fr(BLADE_ON, rz(yaw)))
    assert not M.MODEL.blade_on


def test_the_blade_off_the_shafts_axis_does_not_seat():
    step(blade=fr(at(M.BLADE_SEAT, dx=0.006)))
    assert not M.MODEL.blade_on


def test_the_spinner_without_the_blade_does_not_go_on():
    step(spinner=fr(SPIN_ON))
    assert not M.MODEL.spinner_on


def test_the_spinner_off_the_shafts_axis_does_not_go_on():
    seat_blade()
    step(spinner=fr(SPIN_ON + np.array([0.006, 0, 0])))
    assert not M.MODEL.spinner_on


def test_screwed_anticlockwise_a_full_turn_with_the_blade_held_it_is_tight():
    seat_spinner()
    turn(370.0)
    assert M.MODEL.tight and M.MODEL.screw == pytest.approx(M.TIGHT_TURNS)


def test_screwed_part_way_it_is_not_tight():
    seat_spinner()
    turn(180.0)
    assert not M.MODEL.tight and 0.4 < M.MODEL.screw < 0.6


def test_turned_with_nothing_holding_the_blade_the_shaft_turns_and_nothing_tightens():
    seat_spinner()
    turn(370.0, hold=False)
    assert not M.MODEL.tight and M.MODEL.screw == 0.0
    assert math.degrees(M.MODEL.shaft) == pytest.approx(370.0, abs=1.0)


def test_turned_clockwise_it_does_not_screw_on():
    seat_spinner()
    turn(-370.0, start=60.0)
    assert not M.MODEL.tight and M.MODEL.screw == 0.0


def test_the_front_grille_needs_the_spinner_tight():
    seat_spinner()
    step(front=fr(FRONT_ON))
    assert not M.MODEL.front_on


def _tight():
    seat_spinner()
    turn(370.0)
    assert M.MODEL.tight


@pytest.mark.parametrize("yaw", [0.0, 90.0, 180.0])
def test_the_front_grille_lined_up_with_the_clips_goes_on(yaw):
    _tight()
    step(front=fr(FRONT_ON, rz(yaw)))
    assert M.MODEL.front_on


def test_the_front_grille_turned_off_the_clips_stays_off():
    _tight()
    step(front=fr(FRONT_ON, rz(45.0)))
    assert not M.MODEL.front_on


def test_a_clip_pressed_shut_closes_only_that_clip():
    _tight()
    step(front=fr(FRONT_ON))
    step(right=clip_hand(225.0))
    assert M.MODEL.clips == [False, False, True, False]


def test_a_finger_off_the_clips_closes_none():
    _tight()
    step(front=fr(FRONT_ON))
    step(right=clip_hand(270.0))
    assert not any(M.MODEL.clips)


# -- scoring ------------------------------------------------------------------------------------------------------
def states(moving=False):
    out = {}
    for spec in M.OBJECTS:
        pos = np.array(spec.pos, float)
        quat = spec.quat if spec.quat else fk.yaw_quat(spec.yaw)
        out[spec.name] = fk.body_state(spec, pos, quat, ZERO, ZERO)
    out["spinner"] = fk.body_state(M.SPECS["spinner"], at(M.SPIN_SEAT), fk.yaw_quat(0.0), ZERO, ZERO)
    v = np.array([0.05, 0.0, 0.0]) if moving else ZERO
    out["front"] = fk.body_state(M.SPECS["front"], FRONT_ON, fk.yaw_quat(0.0), v, ZERO)
    return out


def held(st, n=M.STILL_STEPS):
    for _ in range(n):
        ok = M.success(st)
        out = (M.score(st), ok)
    return out


def assembled(blade=True, tight=True, front=True, clips=4):
    m = M.MODEL
    m.blade_on = blade
    m.spinner_on = m.tight = tight
    m.front_on = front
    m.clips = [k < clips for k in range(4)]


def test_put_back_together_succeeds():
    assembled()
    assert held(states()) == (1.0, True)


def test_full_only_after_a_second_held():
    assembled()
    assert held(states(), M.STILL_STEPS - 1)[0] == 0.5
    assert held(states(), 1)[0] == 1.0


def test_the_grille_still_moving_is_half():
    assembled()
    assert held(states(moving=True)) == (0.5, False)


def test_a_clip_left_open_is_half():
    assembled(clips=3)
    assert held(states()) == (0.5, False)


def test_the_front_grille_not_on_is_half():
    assembled(front=False, clips=0)
    assert held(states()) == (0.5, False)


def test_the_spinner_not_tight_is_a_quarter():
    assembled(tight=False, front=False, clips=0)
    assert held(states()) == (0.25, False)


def test_nothing_done_is_nothing():
    assert held(states()) == (0.0, False)


def test_a_hand_still_on_the_grille_is_half(monkeypatch):
    assembled()
    on = np.array([FRONT_ON + np.array([0.0, 0.0, 0.010])])
    monkeypatch.setattr(fk, "finger_points", lambda robots=None: on)
    assert held(states()) == (0.5, False)


def test_a_hand_beside_the_grille_but_clear_of_its_rim_is_full(monkeypatch):
    assembled()
    beside = np.array([FRONT_ON + np.array([-0.11, 0.173, 0.02])])  # inside its square outline, 3.7 cm off its rim
    monkeypatch.setattr(fk, "finger_points", lambda robots=None: beside)
    assert held(states()) == (1.0, True)


def test_a_hand_on_the_grilles_rim_is_half(monkeypatch):
    assembled()
    on = np.array([FRONT_ON + np.array([0.0, -(M.RIM_R + 0.004), 0.0])])
    monkeypatch.setattr(fk, "finger_points", lambda robots=None: on)
    assert held(states()) == (0.5, False)


# -- the demo -----------------------------------------------------------------------------------------------------
GUARD_MAX_DELTA = np.array([0.287979, 0.1832595, 0.1570795, 0.1631885, 0.15708, 0.20944, 0.05])


@pytest.mark.parametrize("arm", ["left", "right"])
def test_the_demo_reaches_every_target_along_its_path(arm):
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


def test_the_left_hand_holds_the_blade_while_the_right_screws_the_spinner_on():
    d = M.demo()
    lp, rp = ArmPlan(d["left"], station.BASES["left"]), ArmPlan(d["right"], station.BASES["right"])
    turning = [t for t in np.arange(0.0, rp.duration, 0.05) if rp.target(t)[3] == "screw it on anticlockwise"]
    assert turning and all(lp.target(t)[3] == "hold the blade still" for t in turning)


def test_the_hands_stay_clear_of_each_other():
    d = M.demo()
    lp, rp = ArmPlan(d["left"], station.BASES["left"]), ArmPlan(d["right"], station.BASES["right"])
    gap = min(np.linalg.norm(np.asarray(lp.target(t)[0]) - np.asarray(rp.target(t)[0]))
              for t in np.arange(0.0, min(lp.duration, rp.duration), 0.05))
    assert gap > 0.11
