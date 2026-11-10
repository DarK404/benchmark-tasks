"""Checks of tape_wrap.py's tape model, scoring and demo, without the simulator. Needs yam_arena (the official
stack):

    python -m pytest -q test_tape_wrap.py
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

TASK = load(Path(__file__).with_name("tape_wrap.py"))
M = TASK.module
import tapekit as tk  # noqa: E402  (the task file put its folder on sys.path)

MM = 0.001
ZERO = np.zeros(3)
R_PIPE, Z0, Z1, ROLL_R, WIDTH = 0.0132, 0.082, 0.100, 0.015, 0.012


def rz(angle):
    c, s_ = math.cos(angle), math.sin(angle)
    return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])


def frame(pos, rot=None):
    pos = np.asarray(pos, float)
    return tk.Frame(pos=pos, rot=np.eye(3) if rot is None else rot, vel=ZERO.copy(), spin=ZERO.copy(), com=pos)


@pytest.fixture
def hands(monkeypatch):
    """Gripper hull points the model sees (none by default)."""
    pts = {"p": np.zeros((0, 3))}
    monkeypatch.setattr(tk, "finger_points", lambda robots=tk.ROBOTS: pts["p"])
    return pts


def model(**kw):
    m = tk.TapeWrap("roll", "disp", [f"s{k}" for k in range(28)], "span", "tail", (0.0, 0.0), R_PIPE, Z0, Z1, ROLL_R,
                    WIDTH, -0.3, **kw)
    m.clear()
    return m


class Hand:
    """Carries the dispenser: the roll at a place, turned on its spindle by ``spin`` (rad, anticlockwise from
    above)."""

    def __init__(self):
        self.spin = 0.0

    def at(self, angle_deg, r=0.045, z=0.092):
        a = math.radians(angle_deg)
        pos = (r * math.cos(a), r * math.sin(a), z)
        return {"roll": frame(pos, rz(self.spin)), "disp": frame(pos)}


def step(m, hand, angle_deg, r=0.045, z=0.092, acc=None, free=True):
    """One substep; ``free``: the roll turns as freely as the tape asks (its payout keeps up)."""
    m.update(hand.at(angle_deg, r, z), {} if acc is None else acc)
    if free and m.anchored and not m.torn and m.stretch > 0:
        hand.spin -= m.stretch / ROLL_R  # turned clockwise: paid out


def carry(m, hand, start_deg, turns, r=0.045, z=(0.092, 0.092), steps_per_turn=96, free=True):
    """Carry the roll round the pipe ``turns`` turns (positive: clockwise seen from above), its height going from
    z[0] to z[1]."""
    n = max(1, int(abs(turns) * steps_per_turn))
    for k in range(1, n + 1):
        f = k / n
        step(m, hand, start_deg - 360.0 * turns * f, r, z[0] + (z[1] - z[0]) * f, free=free)


def started(m, hand=None, angle=90.0):
    hand = hand or Hand()
    step(m, hand, angle, R_PIPE + ROLL_R + 0.001)  # touched to the thread
    for r in np.linspace(R_PIPE + ROLL_R + 0.001, 0.045, 33):  # and drawn out to 45 mm, the tape paying out
        step(m, hand, angle, r)
    return m, hand


# -- the tape model ------------------------------------------------------------------------------------------------
def test_touching_the_thread_sticks_the_tapes_end(hands):
    m, h = model(), Hand()
    step(m, h, 90.0)
    assert not m.anchored
    started(m, h)
    assert m.anchored and m.anchor == pytest.approx(math.radians(90.0))


def test_touching_above_the_pipes_end_starts_nothing(hands):
    m, h = model(), Hand()
    step(m, h, 90.0, R_PIPE + ROLL_R + 0.001, z=0.108)
    assert not m.anchored


def test_the_tape_counts_turns_clockwise_from_the_threads_end(hands):
    m, h = started(model())
    carry(m, h, 90.0, 4.0)
    assert m.turns() == pytest.approx(4.0, abs=0.02) and not m.torn


def test_wound_anticlockwise_the_turns_count_negative(hands):
    m, h = started(model())
    carry(m, h, 90.0, -3.0)
    assert m.turns() == pytest.approx(-3.0, abs=0.02)


def test_carried_back_it_unwinds(hands):
    m, h = started(model())
    carry(m, h, 90.0, 3.0)
    carry(m, h, 90.0, -1.0)
    assert m.turns() == pytest.approx(2.0, abs=0.02)
    assert max(abs(a) for a, _ in m.samples) <= 2.0 * 2 * math.pi + math.radians(6)


def test_each_part_is_laid_at_the_rolls_height(hands):
    m, h = started(model())
    carry(m, h, 90.0, 3.0, z=(0.092, 0.086))
    lo = min(b[0] for b in m.bands())
    hi = max(b[1] for b in m.bands())
    assert lo == pytest.approx(0.086 - WIDTH / 2, abs=0.3 * MM) and hi == pytest.approx(0.092 + WIDTH / 2, abs=1e-9)


def test_away_from_the_thread_the_run_lays_no_tape(hands):
    m, h = started(model())
    carry(m, h, 90.0, 1.0)
    carry(m, h, 90.0, 0.5, z=(0.115, 0.115))
    carry(m, h, 90.0, 0.5, z=(0.065, 0.065))
    assert Z0 - WIDTH / 2 <= min(b[0] for b in m.bands()) and max(b[1] for b in m.bands()) <= Z1 + 1e-9


def test_the_tape_pulls_the_roll_round_clockwise_towards_the_pipe(hands):
    m, h = started(model())
    carry(m, h, 90.0, 1.0)
    acc = {}
    step(m, h, 90.0 - 368.0, acc=acc, free=False)  # carried on 8 degrees, the roll held still: the tape stretches
    f, t = acc["roll"]
    roll = h.at(90.0 - 368.0)["roll"].pos
    assert m.tension > 0 and np.dot(f[:2], -roll[:2]) > 0  # towards the pipe's side
    assert t[2] < 0  # turning the roll clockwise, which lets the tape off


def test_a_roll_that_cannot_turn_tears_the_tape_at_once(hands):
    m, h = started(model())
    carry(m, h, 90.0, 0.5, free=False)
    assert m.torn and not m.anchored  # torn before half a turn bound its end: it fell off


def test_a_steady_wrap_pays_out_without_tearing(hands):
    m, h = started(model())
    carry(m, h, 90.0, 4.0)
    assert not m.torn and m.paid > 4 * 2 * math.pi * R_PIPE


def test_yanked_faster_than_the_roll_turns_it_tears_and_the_tail_stands_loose(hands):
    m, h = started(model())
    carry(m, h, 90.0, 4.0)
    for r in np.linspace(0.045, 0.10, 6):  # yanked out: the braked roll does not keep up
        step(m, h, 90.0, r, free=False)
    assert m.torn and m.anchored and not m.tail_down


def test_a_hand_pressing_the_torn_tail_lays_it_down(hands):
    m, h = started(model())
    carry(m, h, 90.0, 4.0)
    for r in np.linspace(0.045, 0.10, 6):
        step(m, h, 90.0, r, free=False)
    hands["p"] = np.array([m.tail_point() + [0.0, 0.003, -0.002]])
    for _ in range(4):  # the model reads the hands every fourth substep
        step(m, h, -90.0, 0.12, 0.15, free=False)
    assert m.tail_down


def test_the_wound_tape_is_drawn_quarter_by_quarter(hands):
    m, h = started(model())
    carry(m, h, 90.0, 1.0)
    on = [k for k in range(28) if m._shown.get(f"s{k}") and m._shown[f"s{k}"][2] > -0.2]
    assert on == [0, 1, 2, 3]


# -- scoring, on the task's layout ---------------------------------------------------------------------------------
def states(on_mat=True):
    out = {}
    for spec in (M.DISPENSER, M.ROLL):
        pos = np.array(spec.pos, float)
        if not on_mat:
            pos[:2] = (M.PIPE_XY[0] - 0.02, M.PIPE_XY[1] + 0.08)
        out[spec.name] = tk.body_state(spec, pos, (0.0, 0.0, 0.0, 1.0), ZERO, ZERO)
    return out


def wrap(turns=4.0, top=97.5 * MM, bottom=79.5 * MM, torn=True, tail_down=True):
    """Set the task's tape model as if the tape had been wound ``turns`` (clockwise positive), its edges at ``top``
    and ``bottom`` above the table."""
    m = M.MODEL
    m.clear()
    m.anchored, m.secured, m.torn, m.tail_down = True, True, torn, tail_down
    m.wound = turns * 2 * math.pi
    n = 64
    zs = np.linspace(top - WIDTH / 2, bottom + WIDTH / 2, n) + M.TOP
    m.samples = [(m.wound * k / (n - 1), float(z)) for k, z in enumerate(zs)]


@pytest.fixture(autouse=True)
def fresh():
    M.MODEL.clear()
    yield


def test_a_proper_wrap_left_finished_succeeds():
    wrap()
    assert M.score(states()) == 1.0 and M.success(states())


def test_wound_the_wrong_way_scores_nothing():
    wrap(turns=-4.0)
    assert M.score(states()) == 0.0


def test_no_tape_scores_nothing():
    assert M.score(states()) == 0.0


def test_the_tail_left_standing_is_half():
    wrap(tail_down=False)
    assert M.score(states()) == 0.5


def test_the_dispenser_not_put_back_is_half():
    wrap()
    assert M.score(states(on_mat=False)) == 0.5


def test_tape_over_the_pipes_end_is_a_quarter():
    wrap(top=99.5 * MM)
    assert M.score(states()) == 0.25


def test_the_thread_not_covered_to_its_end_is_a_quarter():
    wrap(top=95.0 * MM)
    assert M.score(states()) == 0.25


def test_the_thread_not_covered_to_its_shoulder_is_a_quarter():
    wrap(bottom=84.0 * MM)
    assert M.score(states()) == 0.25


@pytest.mark.parametrize("turns, expect", [(2.5, 0.25), (3.0, 1.0), (6.0, 1.0), (6.5, 0.25)])
def test_three_to_six_turns(turns, expect):
    wrap(turns=turns)
    assert M.score(states()) == expect


def test_not_torn_off_is_a_quarter():
    wrap(torn=False, tail_down=False)
    assert M.score(states()) == 0.25


def test_less_than_a_turn_scores_nothing():
    wrap(turns=0.8)
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


def test_one_hand_at_a_time_near_the_pipe():
    """The arms' wrists met over the pipe with both hands at it: never both within 12 cm of its axis."""
    plans = {a: ArmPlan(p, station.BASES[a]) for a, p in M.demo().items()}
    end = max(p.duration for p in plans.values())
    for t in np.arange(0.0, end, 0.05):
        near = [a for a, p in plans.items() if t < p.duration
                and np.hypot(*(np.array(p.target(t)[0][:2]) - M.PIPE_XY)) < 0.12]
        assert len(near) <= 1, (t, near)
