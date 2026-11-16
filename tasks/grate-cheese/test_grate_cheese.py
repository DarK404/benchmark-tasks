"""Offline checks of grate_cheese.py's grating model, scoring and demo, without the simulator. Needs yam_arena (the
official stack):

    python -m pytest -q test_grate_cheese.py
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

TASK = load(Path(__file__).with_name("grate_cheese.py"))
M = TASK.module
import gratekit as gk  # noqa: E402  (the task file put its folder on sys.path)

ZERO = np.zeros(3)
MM = 0.001
GRATER_AT = np.array([M.DISH_XY[0], M.GRATER_Y, M.TOP + M.GRATER_Z])


def fr(pos, vel=ZERO, rot=None):
    pos = np.asarray(pos, float)
    return gk.Frame(pos=pos, rot=np.eye(3) if rot is None else rot, vel=np.asarray(vel, float), spin=ZERO.copy(),
                    com=pos)


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    M.MODEL.clear()
    M.MODEL.origin = np.zeros(3)
    M._reset_state()
    monkeypatch.setattr(gk, "finger_points", lambda robots=None: np.zeros((0, 3)))
    yield


def stroke(dy, press_mm=2.0, steps=240, y0=None):
    """Slide the cheese's underside along the grating surface, pressed ``press_mm`` into it, by ``dy`` (m; minus is
    towards the grater's tip) over ``steps`` substeps."""
    m = M.MODEL
    y0 = M.GRATER_Y + (0.02 if dy < 0 else -0.02) if y0 is None else y0
    v = dy / (steps * m.dt)
    for k in range(steps):
        y = y0 + dy * k / steps
        acc = {}
        cz = M.TOP + M.GRATER_Z + M.SURF_Z - press_mm * MM + m.depth
        m.update({"cheese": fr([M.DISH_XY[0], y, cz], [0.0, v, 0.0]), "grater": fr(GRATER_AT)}, acc)
    return acc


# -- the model ----------------------------------------------------------------------------------------------------
def test_a_pressed_stroke_away_from_the_handle_grates():
    stroke(-0.05)
    assert M.MODEL.grated > 0.0005


def test_a_stroke_back_towards_the_handle_grates_nothing():
    stroke(+0.05)
    assert M.MODEL.grated == 0.0


def test_a_harder_press_grates_more():
    stroke(-0.05, press_mm=1.0)
    light = M.MODEL.grated
    M.MODEL.clear()
    stroke(-0.05, press_mm=3.0)
    assert M.MODEL.grated > 1.5 * light


def test_the_grater_takes_the_opposite_push():
    acc = stroke(-0.05, steps=10)
    fc, fg = acc["cheese"][0], acc["grater"][0]
    assert np.allclose(fc, -fg) and fc[2] > 0.0


def test_off_the_grating_surface_nothing_grates():
    m = M.MODEL
    for _ in range(100):
        m.update({"cheese": fr([M.DISH_XY[0] + 0.08, M.GRATER_Y, M.TOP + M.GRATER_Z + M.SURF_Z - 0.002],
                               [0.0, -0.03, 0.0]), "grater": fr(GRATER_AT)}, {})
    assert m.grated == 0.0


def test_a_finger_on_the_blades_latches():
    on = np.array([[M.DISH_XY[0], M.GRATER_Y, M.TOP + M.GRATER_Z + M.SURF_Z + 0.001]])
    gk.finger_points = lambda robots=None: on
    stroke(-0.01, steps=8)
    gk.finger_points = lambda robots=None: np.zeros((0, 3))
    stroke(-0.01, steps=8)
    assert M.MODEL.knuckles


def test_a_finger_clear_of_the_blades_does_not_latch():
    up = np.array([[M.DISH_XY[0], M.GRATER_Y, M.TOP + M.GRATER_Z + M.SURF_Z + 0.008]])
    gk.finger_points = lambda robots=None: up
    stroke(-0.01, steps=8)
    assert not M.MODEL.knuckles


def test_the_grater_off_the_dish_latches():
    m = M.MODEL
    m.update({"cheese": fr([1.0, 1.0, 1.0]), "grater": fr(GRATER_AT + np.array([0.0, -0.08, -0.03]))}, {})
    m.update({"cheese": fr([1.0, 1.0, 1.0]), "grater": fr(GRATER_AT)}, {})
    assert m.off_dish


# -- scoring ------------------------------------------------------------------------------------------------------
def states(cheese_on_board=True, moving=False):
    out = {}
    for spec in M.OBJECTS:
        pos = np.array(spec.pos, float)
        quat = spec.quat if spec.quat else gk.yaw_quat(spec.yaw)
        out[spec.name] = gk.body_state(spec, pos, quat, ZERO, ZERO)
    if not cheese_on_board:
        out["cheese"] = gk.body_state(M.CHEESE, np.array([0.30, 0.20, M.TOP]), gk.yaw_quat(0.0), ZERO, ZERO)
    if moving:
        out["cheese"] = gk.body_state(M.CHEESE, np.array(M.CHEESE.pos), gk.yaw_quat(0.0), np.array([0.05, 0, 0]),
                                      ZERO)
    return out


def grated(g):
    M.MODEL.volume = g / M.MODEL.rho


def held(st, n=M.STILL_STEPS):
    for _ in range(n):
        ok = M.success(st)
        res = (M.score(st), ok)
    return res


def test_about_20_g_grated_and_the_cheese_back_on_its_board_succeeds():
    grated(0.020)
    assert held(states()) == (1.0, True)


def test_full_only_after_a_second_held():
    grated(0.020)
    assert held(states(), M.STILL_STEPS - 1)[0] == 0.5
    assert held(states(), 1)[0] == 1.0


@pytest.mark.parametrize("g", [0.006, 0.012, 0.030, 0.060])
def test_grated_but_not_about_20_g_is_half(g):
    grated(g)
    assert held(states()) == (0.5, False)


@pytest.mark.parametrize("g", [0.015, 0.025])
def test_the_amount_window_is_inclusive(g):
    grated(g)
    assert held(states())[0] == 1.0


def test_the_cheese_not_put_back_is_half():
    grated(0.020)
    assert held(states(cheese_on_board=False)) == (0.5, False)


def test_still_moving_is_half():
    grated(0.020)
    assert held(states(moving=True)) == (0.5, False)


def test_a_little_grated_is_a_quarter():
    grated(0.002)
    assert held(states()) == (0.25, False)


def test_nothing_grated_is_nothing():
    assert held(states()) == (0.0, False)


@pytest.mark.parametrize("flag", ["knuckles", "off_dish"])
def test_a_violation_scores_nothing(flag):
    grated(0.020)
    setattr(M.MODEL, flag, True)
    assert held(states()) == (0.0, False)


def test_a_hand_still_on_the_cheese_is_half(monkeypatch):
    grated(0.020)
    on = np.array([[M.BOARD_XY[0], M.BOARD_XY[1], M.TOP + M.BOARD_T + 0.015]])
    monkeypatch.setattr(gk, "finger_points", lambda robots=None: on)
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


def test_the_left_hand_steadies_the_grater_through_every_stroke():
    d = M.demo()
    lp, rp = ArmPlan(d["left"], station.BASES["left"]), ArmPlan(d["right"], station.BASES["right"])
    grating = [t for t in np.arange(0.0, rp.duration, 0.05) if rp.target(t)[3].startswith("grate")]
    assert grating and all(lp.target(t)[3] in ("hold the grater", "steady the grater") for t in grating)
