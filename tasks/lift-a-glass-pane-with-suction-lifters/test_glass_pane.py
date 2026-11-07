"""Checks of glass_pane.py's scoring, its suction and hinge models, and its demo, without the simulator. Needs yam_arena
(the official stack):

    python -m pytest -q test_glass_pane.py
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

TASK = load(Path(__file__).with_name("glass_pane.py"))
M = TASK.module
import glazekit as g  # noqa: E402  (the task file put its folder on sys.path)

MM = 0.001
ZERO = np.zeros(3)


def _quat(yaw_deg=0.0, tilt_deg=0.0):
    """xyzw of a turn ``yaw_deg`` about the vertical times a tilt ``tilt_deg`` about the body's own y."""
    sz, cz = math.sin(math.radians(yaw_deg) / 2), math.cos(math.radians(yaw_deg) / 2)
    sy, cy = math.sin(math.radians(tilt_deg) / 2), math.cos(math.radians(tilt_deg) / 2)
    return np.array([-sz * sy, cz * sy, sz * cy, cz * cy])


def pane(x=None, y=0.0, under=None, tilt=0.0, vz=0.0):
    """The pane, its underside's centre ``under`` above the table (default: in the sash)."""
    x = M.SASH_XY[0] if x is None else x
    under = M.BED_TOP if under is None else under
    return g.body_state(M.PANE, (x, y, M.TOP + under), _quat(0.0, tilt), np.array([0, 0, vz]), ZERO)


def on_bearers(**kw):
    kw.setdefault("x", M.BEARERS_X)
    return pane(under=M.BEARER_TOP, **kw)


def lifter_pair(side, at=None, lever_deg=60.0):
    """A lifter and its lever, the lever turned ``lever_deg`` (0: locked flat)."""
    s = 1.0 if side == "l" else -1.0
    hinge = M.HINGE_L if side == "l" else M.HINGE_R
    yaw = 0.0 if s > 0 else math.pi
    pos = np.array(at if at is not None else (M.LIFTER_X, s * M.LIFTER_START, M.TOP))
    lifter = g.body_state(getattr(M, f"LIFTER_{side.upper()}"), pos, g.yaw_quat(yaw), ZERO, ZERO)
    r = g.quat_to_matrix(g.yaw_quat(yaw)) @ hinge.rotation(lever_deg)
    piv = np.asarray(hinge.pivot)
    lever_pos = pos + g.quat_to_matrix(g.yaw_quat(yaw)) @ (piv - hinge.rotation(lever_deg) @ piv)
    lever = g.body_state(getattr(M, f"LEVER_{side.upper()}"), lever_pos, g.matrix_to_quat(r), ZERO, ZERO)
    return {f"lifter_{side}": lifter, f"lever_{side}": lever}


def objs(p, lever_deg=60.0):
    out = {"pane": p, "sash": None, "bearers": None}
    for side in ("l", "r"):
        out.update(lifter_pair(side, lever_deg=lever_deg))
    return out


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    """A new trial: cups clear, latches cleared, grippers far away."""
    for c in M.CUPS.values():
        c.clear()
    M._reset_state()
    monkeypatch.setattr(g, "finger_points", lambda: np.array([[0.0, 0.0, M.TOP + 0.5]]))
    yield


def seal(*sides, sealed=True):
    for s in sides:
        M.CUPS[s].sealed = sealed
        M.CUPS[s].ever_sealed = M.CUPS[s].ever_sealed or sealed


def unseal_all():
    for c in M.CUPS.values():
        c.sealed = False


def run(*frames):
    """Score frame by frame; each frame is (objects, setup callable or None). Returns the last (score, success)."""
    out = None
    for f, setup in frames:
        if setup:
            setup()
        out = (M.score(f), M.success(f))
    return out


LIFTED = M.BED_TOP + M.LIFT_CLEAR + 10 * MM


def full_trial(end):
    return run((objs(pane()), lambda: seal("l", "r")), (objs(pane(under=LIFTED)), None), (end, unseal_all))


# -- the tiers ---------------------------------------------------------------------------------------------------


def test_start_scores_zero():
    assert run((objs(pane()), None)) == (0.0, False)


def test_one_lifter_locked_is_a_quarter():
    assert run((objs(pane()), lambda: seal("l"))) == (0.25, False)


def test_lift_with_both_lifters_locked_is_half():
    assert run((objs(pane()), lambda: seal("l", "r")), (objs(pane(under=LIFTED)), None)) == (0.5, False)


def test_lift_with_only_one_lifter_does_not_count():
    assert run((objs(pane()), lambda: seal("l")), (objs(pane(under=LIFTED)), None)) == (0.25, False)


def test_pane_laid_on_the_bearers_and_let_go_is_full_score():
    assert full_trial(objs(on_bearers())) == (1.0, True)


def test_one_cup_lift_laid_on_the_bearers_is_only_a_quarter():
    frames = ((objs(pane()), lambda: seal("l")), (objs(pane(under=LIFTED)), None), (objs(on_bearers()), unseal_all))
    assert run(*frames) == (0.25, False)


def test_still_moving_is_not_success_yet():
    assert full_trial(objs(on_bearers(vz=-0.05))) == (1.0, False)


@pytest.mark.parametrize("lever_deg", [0.0, 30.0, 44.0])
def test_levers_not_fully_up_is_half(lever_deg):
    assert full_trial(objs(on_bearers(), lever_deg=lever_deg)) == (0.5, False)


def test_still_sealed_is_half():
    frames = ((objs(pane()), lambda: seal("l", "r")), (objs(pane(under=LIFTED)), None), (objs(on_bearers()), None))
    assert run(*frames) == (0.5, False)


def test_tilted_on_the_bearers_is_half():
    assert full_trial(objs(on_bearers(tilt=5.0))) == (0.5, False)


def test_off_centre_on_the_bearers_is_half():
    assert full_trial(objs(on_bearers(x=M.BEARERS_X + 30 * MM))) == (0.5, False)


def test_not_down_on_the_bearers_is_half():
    assert full_trial(objs(pane(x=M.BEARERS_X, under=M.BEARER_TOP + 10 * MM))) == (0.5, False)


def test_gripper_near_the_glass_is_half(monkeypatch):
    end = objs(on_bearers())
    near = end["pane"].to_world((0.0, 0.0, M.PANE_SIZE[2] + 15 * MM))
    monkeypatch.setattr(g, "finger_points", lambda: near[None, :])
    assert full_trial(end) == (0.5, False)


def test_gripper_still_on_a_lever_is_half(monkeypatch):
    end = objs(on_bearers())
    on_lever = end["lever_l"].to_world((0.0, 0.08, 0.023))
    monkeypatch.setattr(g, "finger_points", lambda: on_lever[None, :])
    assert full_trial(end) == (0.5, False)


def test_gripper_just_clear_of_everything_is_full(monkeypatch):
    end = objs(on_bearers())
    above = end["pane"].to_world((0.0, 0.0, M.PANE_SIZE[2] + 0.20))
    monkeypatch.setattr(g, "finger_points", lambda: above[None, :])
    assert full_trial(end) == (1.0, True)


# -- violations latch --------------------------------------------------------------------------------------------


def test_dropped_then_put_right_scores_zero():
    frames = ((objs(pane()), lambda: seal("l", "r")), (objs(pane(under=LIFTED, vz=-0.5)), None),
              (objs(on_bearers()), unseal_all))
    assert run(*frames) == (0.0, False)


def test_pane_below_its_bed_is_a_drop():
    assert run((objs(pane(under=M.BED_TOP - 10 * MM)), None), (objs(pane()), None)) == (0.0, False)


def test_tilted_in_the_carry_then_laid_flat_scores_zero():
    frames = ((objs(pane()), lambda: seal("l", "r")), (objs(pane(under=LIFTED, tilt=25.0)), None),
              (objs(on_bearers()), unseal_all))
    assert run(*frames) == (0.0, False)


def test_tilt_in_the_sash_is_not_a_carry_violation():
    assert run((objs(pane(tilt=25.0)), None), (objs(pane()), lambda: seal("l"))) == (0.25, False)


def test_gripper_touching_the_glass_latches(monkeypatch):
    start = objs(pane())
    touch = start["pane"].to_world((0.0, 0.0, M.PANE_SIZE[2] + 0.5 * MM))
    monkeypatch.setattr(g, "finger_points", lambda: touch[None, :])
    run((start, None))
    monkeypatch.setattr(g, "finger_points", lambda: np.array([[0.0, 0.0, M.TOP + 0.5]]))
    assert full_trial(objs(on_bearers())) == (0.0, False)


def test_reset_clears_the_latches():
    run((objs(pane(under=M.BED_TOP - 10 * MM)), None))
    assert M.score(objs(pane())) == 0.0
    M._reset_state()
    assert full_trial(objs(on_bearers())) == (1.0, True)


# -- the suction model -------------------------------------------------------------------------------------------


def _frame(state, spec):
    r = state.rotation()
    return g.Frame(pos=np.asarray(state.pos, float), rot=r, vel=ZERO.copy(), spin=ZERO.copy(),
                   com=state.pos + r @ np.asarray(spec.com))


def frames_for(lifter_at, lever_deg, p=None, lifter_tilt=0.0):
    p = p or pane()
    pair = lifter_pair("l", at=lifter_at, lever_deg=lever_deg)
    lif = pair["lifter_l"]
    if lifter_tilt:
        lif = g.body_state(M.LIFTER_L, lif.pos, _quat(0.0, lifter_tilt), ZERO, ZERO)
    return {"pane": _frame(p, M.PANE), "lifter_l": _frame(lif, M.LIFTER_L), "lever_l": _frame(pair["lever_l"], M.LEVER_L)}


def on_glass(dx=0.0, dy=0.12, dz=0.0):
    return (M.SASH_XY[0] + dx, dy, M.TOP + M.BED_TOP + M.PANE_SIZE[2] + dz)


def test_cup_seals_when_the_lever_goes_down_on_the_glass():
    cup = M.CUPS["l"]
    cup.update(frames_for(on_glass(), 60.0), {})
    acc = {}
    cup.update(frames_for(on_glass(), 0.0), acc)
    assert cup.sealed and cup.ever_sealed
    cup.update(frames_for(on_glass(), 60.0), {})
    assert not cup.sealed and cup.ever_sealed


@pytest.mark.parametrize("where", ["tilted", "over the edge", "lifted off", "on the table"])
def test_cup_does_not_seal_off_the_glass(where):
    at, tilt = {
        "tilted": (on_glass(), 5.0),
        "over the edge": (on_glass(dy=M.PANE_SIZE[1] / 2 - M.CUP_R), 0.0),
        "lifted off": (on_glass(dz=5 * MM), 0.0),
        "on the table": ((M.LIFTER_X, M.LIFTER_START, M.TOP), 0.0),
    }[where]
    cup = M.CUPS["l"]
    cup.update(frames_for(at, 60.0, lifter_tilt=tilt), {})
    cup.update(frames_for(at, 0.0, lifter_tilt=tilt), {})
    assert not cup.sealed


def test_lever_shut_elsewhere_must_be_reopened_to_seal():
    cup = M.CUPS["l"]
    cup.update(frames_for((M.LIFTER_X, M.LIFTER_START, M.TOP), 0.0), {})  # shut on the table: no seal
    cup.update(frames_for(on_glass(), 0.0), {})  # slid onto the glass still shut: still none
    assert not cup.sealed
    cup.update(frames_for(on_glass(), 60.0), {})
    cup.update(frames_for(on_glass(), 0.0), {})
    assert cup.sealed


def test_sealed_cup_pulls_the_pane_towards_it_and_pops_when_torn_off():
    cup = M.CUPS["l"]
    cup.update(frames_for(on_glass(), 60.0), {})
    cup.update(frames_for(on_glass(), 0.0), {})
    acc = {}
    cup.update(frames_for(on_glass(dz=1 * MM), 0.0), acc)  # the lifter 1 mm up: springs pull the pane up
    assert acc["pane"][0][2] > 0 and acc["lifter_l"][0][2] < 0 and cup.sealed
    cup.update(frames_for(on_glass(dz=20 * MM), 0.0), {})  # 20 mm: 80 N a spring, over the limit
    assert not cup.sealed and cup.popped == 1


# -- the hinge ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("side", ["l", "r"])
def test_levers_start_open(side):
    hinge, lifter, lever = (M.HINGE_L, M.LIFTER_L, M.LEVER_L) if side == "l" else (M.HINGE_R, M.LIFTER_R, M.LEVER_R)
    angle = hinge.angle(g.quat_to_matrix(g.yaw_quat(lifter.yaw)), g.quat_to_matrix(lever.quat))
    assert abs(angle - M.LEVER_OPEN) < 1e-6


# -- the demo ----------------------------------------------------------------------------------------------------

# The per-step limits of inspect-robots' DeltaLimitGuard for the YAM action space, as recorded in a run's
# configuration: six joints, then the gripper (0-1, 5% a step).
GUARD_MAX_DELTA = np.array([0.287979, 0.1832595, 0.1570795, 0.1631885, 0.15708, 0.20944, 0.05])


@pytest.mark.parametrize("arm", ["left", "right"])
def test_the_demo_is_within_reach_and_passes_the_run_guards(arm):
    plan = ArmPlan(M.demo()[arm], station.BASES[arm])
    acts = np.array([plan.action7(t)[0] for t in np.arange(0.0, plan.duration, 0.05)])
    assert plan.worst_ik_mm < 0.5 and plan.worst_ik_deg < 0.1, (plan.worst_ik_mm, plan.worst_ik_deg)
    assert (np.abs(np.diff(acts, axis=0)) / GUARD_MAX_DELTA).max() < 0.95


def test_the_demo_fits_the_run_limit():
    assert max(ArmPlan(p, station.BASES[a]).duration for a, p in M.demo().items()) * 20 < M.RUN_LIMIT_STEPS
