"""Offline checks of nebulizer.py's model, scoring and demo, without the simulator. Needs yam_arena (the official
stack):

    python -m pytest -q test_nebulizer.py
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

TASK = load(Path(__file__).with_name("nebulizer.py"))
M = TASK.module
import nebkit as nk  # noqa: E402  (the task file put its folder on sys.path)

ZERO = np.zeros(3)
MM = 0.001


def rz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def fr(pos, rot=None):
    pos = np.asarray(pos, float)
    return nk.Frame(pos=pos, rot=np.eye(3) if rot is None else rot, vel=ZERO.copy(), spin=ZERO.copy(), com=pos)


TANK_AT = np.array([*M.STAND_XY, M.TOP + M.TANK_Z0])
FAR = {n: fr([1.0, 1.0, 1.0]) for n in ("cap", "mp", "cuff")}


def hands(top_deg=None, tank=False):
    """Fake finger hull points: a hand closed across the top at ``top_deg`` (finger axis), another on the tank."""
    out = {"left_robot": np.zeros((0, 3)), "right_robot": np.zeros((0, 3))}
    if top_deg is not None:
        a = math.radians(top_deg)
        d = np.array([math.cos(a), math.sin(a), 0.0])
        pts = [TANK_AT + s * d * (M.TOP_R + 2.0 * MM) + np.array([0, 0, z]) for s in (-1, 1)
               for z in (0.008, 0.015, 0.022)]
        out["right_robot"] = np.array(pts)
    if tank:
        d = np.array([0.0, 1.0, 0.0])
        pts = [TANK_AT + s * d * (M.TANK_R + 2.0 * MM) + np.array([0, 0, z]) for s in (-1, 1)
               for z in (-0.020, -0.012, -0.006)]
        out["left_robot"] = np.array(pts)
    return out


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    M.MODEL.clear()
    M.MODEL.origin = np.zeros(3)
    M._reset_state()
    monkeypatch.setattr(nk, "finger_points", lambda robots=None: np.zeros((0, 3)))
    yield


def step(top_pos, top_rot=None, pts=None, **parts):
    m = M.MODEL
    m._tick = 0  # read the fake hands every step
    if pts is not None:
        nk.finger_points = lambda robots=None, _p=pts: _p[robots[0]] if robots else np.vstack(list(_p.values()))
    frames = {"tank": fr(TANK_AT), "top": fr(top_pos, top_rot), **FAR, **parts}
    m.update(frames, {})


def seat():
    step(TANK_AT + np.array([0, 0, 0.002]))
    assert M.MODEL.seated


def turn_to(deg, tank=True):
    for a in np.linspace(90.0 + M.LOCK_DEG, 90.0 + M.LOCK_DEG - deg, 12):
        step(TANK_AT, pts=hands(top_deg=a, tank=tank))


# -- the model ----------------------------------------------------------------------------------------------------
def test_the_top_set_on_the_tank_square_and_by_its_tab_seats():
    seat()


@pytest.mark.parametrize("yaw", [30.0, -30.0])
def test_the_top_set_on_off_its_tab_does_not_seat(yaw):
    step(TANK_AT + np.array([0, 0, 0.002]), rz(yaw))
    assert not M.MODEL.seated


def test_the_top_off_the_tanks_axis_does_not_seat():
    step(TANK_AT + np.array([0.006, 0, 0.002]))
    assert not M.MODEL.seated


def test_turned_clockwise_with_the_tank_held_it_clicks_at_the_lock():
    seat()
    turn_to(M.LOCK_DEG + 5.0)
    assert M.MODEL.locked and math.isclose(math.degrees(M.MODEL.turn), -M.LOCK_DEG, abs_tol=1e-6)


def test_turned_part_way_it_is_not_locked():
    seat()
    turn_to(M.LOCK_DEG / 2)
    assert not M.MODEL.locked and -M.LOCK_DEG < math.degrees(M.MODEL.turn) < 0


def test_turned_with_nothing_holding_the_tank_nothing_changes():
    seat()
    turn_to(M.LOCK_DEG + 5.0, tank=False)
    assert not M.MODEL.locked and M.MODEL.turn == 0.0


def test_turned_anticlockwise_it_stays_at_its_notch():
    seat()
    for a in np.linspace(90.0 + M.LOCK_DEG, 90.0 + M.LOCK_DEG + 30.0, 8):
        step(TANK_AT, pts=hands(top_deg=a, tank=True))
    assert M.MODEL.turn == 0.0 and not M.MODEL.locked


def _locked_top_frame():
    m = M.MODEL
    return m._part_frame(fr(TANK_AT), np.array([0.0, 0.0, 0.0005]), m.turn)


def test_the_cap_pressed_on_the_inlet_goes_on():
    seat()
    turn_to(M.LOCK_DEG + 5.0)
    tf = _locked_top_frame()
    step(TANK_AT, cap=fr(tf.world((0.0, 0.0, M.INLET_TOP - 0.002))))
    assert M.MODEL.cap_on


def test_the_cap_held_off_the_inlets_axis_stays_off():
    seat()
    tf = _locked_top_frame()
    step(TANK_AT, cap=fr(tf.world((0.006, 0.0, M.INLET_TOP - 0.002))))
    assert not M.MODEL.cap_on


def _on_spigot(parent_frame, tip, out_dir, over, across=0.0):
    """A frame for a part pushed ``over`` metres onto a spigot (its x axis along it), ``across`` off its axis."""
    ax = parent_frame.rot @ np.asarray(out_dir, float)
    x = ax
    z = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    side = y * across
    return fr(parent_frame.world(tip) - ax * over + side, np.column_stack([x, y, z]))


def test_the_mouthpiece_pushed_onto_the_outlet_goes_on():
    seat()
    turn_to(M.LOCK_DEG + 5.0)
    tf = _locked_top_frame()
    step(TANK_AT, mp=_on_spigot(tf, M.MODEL.outlet, M.MODEL.outlet_dir, 0.008))
    assert M.MODEL.mp_on


def test_the_mouthpiece_short_of_the_outlet_stays_off():
    seat()
    tf = _locked_top_frame()
    step(TANK_AT, mp=_on_spigot(tf, M.MODEL.outlet, M.MODEL.outlet_dir, 0.001))
    assert not M.MODEL.mp_on


def test_the_cuff_pushed_onto_the_nozzle_goes_on():
    step(TANK_AT + np.array([1.0, 0, 0]), cuff=_on_spigot(fr(TANK_AT), M.MODEL.nozzle, M.MODEL.nozzle_dir, 0.006))
    assert M.MODEL.cuff_on


def test_the_cuff_off_the_nozzles_axis_stays_off():
    step(TANK_AT + np.array([1.0, 0, 0]),
         cuff=_on_spigot(fr(TANK_AT), M.MODEL.nozzle, M.MODEL.nozzle_dir, 0.006, across=0.008))
    assert not M.MODEL.cuff_on


def test_the_tube_has_25_links_of_4_cm():
    lengths = np.linalg.norm(np.diff(np.array(M.PATH), axis=0), axis=1)
    assert len(lengths) == 25 and np.allclose(lengths, 0.04, atol=1e-6)


# -- scoring ------------------------------------------------------------------------------------------------------
def states(tank_pos=None, tank_tilt=0.0, moving=False):
    out = {}
    for spec in M.OBJECTS:
        pos = np.array(spec.pos, float)
        quat = spec.quat if spec.quat else nk.yaw_quat(spec.yaw)
        v = np.array([0.05, 0, 0]) if moving else ZERO
        if spec.name == "tank":
            if tank_pos is not None:
                pos = np.asarray(tank_pos, float)
            t = math.radians(tank_tilt)
            quat = (math.sin(t / 2), 0.0, 0.0, math.cos(t / 2))
        out[spec.name] = nk.body_state(spec, pos, quat, v if spec.name == "tank" else ZERO, ZERO)
    return out


def held(st, n=M.STILL_STEPS):
    for _ in range(n):
        ok = M.success(st)
        out = (M.score(st), ok)
    return out


def assembled(cap=True, mp=True, cuff=True, locked=True):
    m = M.MODEL
    m.seated = True
    m.locked = locked
    m.cap_on, m.mp_on, m.cuff_on = cap, mp, cuff


def test_put_together_and_back_in_its_stand_succeeds():
    assembled()
    assert held(states()) == (1.0, True)


def test_full_only_after_a_second_still():
    assembled()
    assert held(states(), M.STILL_STEPS - 1)[0] == 0.5
    assert held(states(), 1)[0] == 1.0


def test_still_moving_is_not_full():
    assembled()
    assert held(states(moving=True)) == (0.5, False)


def test_put_together_but_not_in_its_stand_is_half():
    assembled()
    assert held(states(tank_pos=TANK_AT + np.array([0.0, -0.06, 0.04]))) == (0.5, False)


def test_put_together_but_tipped_in_its_stand_is_half():
    assembled()
    assert held(states(tank_tilt=20.0)) == (0.5, False)


@pytest.mark.parametrize("missing", ["cap", "mp", "cuff"])
def test_a_part_missing_is_a_quarter(missing):
    assembled(**{missing: False})
    assert held(states()) == (0.25, False)


def test_the_top_seated_but_not_turned_to_the_click_is_nothing():
    assembled(locked=False)
    assert held(states()) == (0.0, False)


def test_nothing_done_is_nothing():
    assert held(states()) == (0.0, False)


def test_a_hand_still_on_the_kit_is_half(monkeypatch):
    assembled()
    on = np.array([[*M.STAND_XY, M.TOP + M.TANK_Z0 - 0.010]])
    monkeypatch.setattr(nk, "finger_points", lambda robots=None: on)
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


def test_the_left_hand_holds_the_tank_while_the_right_turns_the_top():
    d = M.demo()
    lp, rp = ArmPlan(d["left"], station.BASES["left"]), ArmPlan(d["right"], station.BASES["right"])
    turning = [t for t in np.arange(0.0, rp.duration, 0.05) if rp.target(t)[3] == "turn it to the click"]
    assert turning and all(lp.target(t)[3] == "hold the tank" for t in turning)
