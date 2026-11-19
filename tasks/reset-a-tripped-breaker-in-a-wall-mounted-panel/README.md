# Reset a tripped breaker: push its mid-way handle firmly all the way to OFF, then to ON, leaving the other breakers on

A Square D Homeline 4-space load center (an unpowered training panel, not wired to any supply) is mounted upright on
a plywood wall board standing on the table, as it is installed on a wall: its door open, its dead front (cover) on.
Four Homeline breakers show their handles through the dead front's slots. Three are ON; one has tripped, its handle
standing midway between ON and OFF. The robot must reset the tripped breaker.

Resetting a tripped breaker is one of the commonest jobs in a home: a heater, kettle and toaster on one circuit, and
the lights go out until someone opens the panel door and resets the breaker. Square D's directions (its FAQ, and the
directory label inside its load center doors): a tripped breaker's handle sits about midway between ON and OFF; to
reset it, push the handle all the way to OFF, then all the way back to ON. Pushed straight towards ON from the middle,
it does not stay on; a stiff one must be pressed firmly until it touches OFF.

The task tests finding the one handle that is out of line, and working a mechanism whose state only the right
sequence changes: a firm push all the way to OFF (the handle resists, then gives as the breaker relatches), then a
push back to ON, without disturbing the other breakers' handles beside it.

## Files

| File | What |
|---|---|
| `breaker_reset.py` | The yam_arena task file: the load center on its wall board with its dead front, the four breakers' handles, the tripped breaker's trip mechanism, `success()`, `score()` and the scripted `demo()` |
| `resetkit.py` | Our own helper: rigid bodies built from boxes (turned boxes for sloped faces), hinges with limits, sprung slides (the toggles, the trip cradle and its pawl), kinematic bodies, and hooks into yam_arena's scene and reset |
| `test_breaker_reset.py` | Offline checks of the mechanism's geometry (the latch is reached only by a push to OFF; unlatched, the handle is centred), every scoring tier and the demo's reach and per-step limits |

`resetkit.py` hooks `yam_arena.scene` and the embodiment's reset without modifying yam_arena. The task runs with the
standard `yam_arena` embodiment and the `task_score` scorer. All the code is the task authors' own (MIT licence); no
third-party assets are used.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=breaker_reset.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "One breaker in this unpowered training panel has tripped. Reset it." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/brr-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode. To render the
trial's videos (one MP4 per camera), run `inspect-robots video ~/yam-eval/brr-dream/<log>.json`.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=breaker_reset.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "One breaker in this unpowered training panel has tripped. Reset it." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/brr-demo
```

The demo is open loop: it uses only the fixed layout, never the simulator's internal state. The right hand does it
all, its closed fingertips towards the panel and pressed lightly on the dead front as they slide; the left stays home.

1. **Push it to OFF**: from beside the tripped handle (space 4) on its ON side, push its lever firmly all the way to
   OFF (aimed well past, the soft simulated arm pushes until it gives and the breaker relatches); back off.
2. **Push it to ON**: come round to its OFF side and push it all the way to ON; back off and go home.

The demo reaches 1.0 (success) in 322 steps (16.1 s). No step of it is clamped by the run's delta-limit
guard (the gripper may change by at most 5% of its range per step); `test_breaker_reset.py` checks every step against
it and that every pose along the arm's path is reached.

**Shortcut check.** With `BR_DEMO=on` the demo pushes the tripped handle straight to ON without pushing it to OFF
first: it springs back to the middle and the trial scores 0.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`). The handles'
positions are their throws in their breakers' frames; the stops are 25 degrees either side of the middle.

| Score | Condition | How to check on a real table |
|---|---|---|
| 1.0 | the tripped breaker reset (relatched) and ON (its handle within 7 degrees of the ON stop); every other breaker still ON, never switched off; no part of either gripper within 1 cm of any handle; every handle still; held for 1 s | look: all four handles at ON; the reset one stays ON (a breaker not reset springs back to the middle) |
| 0.5 | relatched but its handle left at OFF (or not yet still, or a hand on it) | look: the handle at OFF |
| 0 | anything else (for example pushed only towards ON, so it springs back to the middle); and for the rest of the run once any other breaker has been switched off (its handle thrown past the middle) | look |

Still means no handle's throw changed by more than 0.2 degrees in a control step (judged from the handles' angles:
hinged to the panel, their read-back velocities are not physical). `score()` gives 1.0 only once the full-score
state has held for 1 s (until then 0.5), and `success()` returns True
only then (yam_arena then ends the trial after its own 1 s success hold). Offline checks of the mechanism, every tier
and the demo: `python -m pytest -q test_breaker_reset.py` (17 tests, no simulator).

## Runtime and seeds

Measured on an L40S with the viewer on, Isaac Sim's start-up included:
- the demo trial (322 steps, 16.1 s of simulated time) took 1.1 minutes of wall-clock time;
- the Dream-1 trial (the full 1,200 steps, 1 minute simulated; it scored 0.0) took 3.4 minutes.

The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

- **The breakers' state is decided by contact and springs, with no rules.** Each handle turns on a hinge (25 degrees
  either side of the middle). Under its pivot an arm reaches 12 mm into the breaker. A sprung toggle plunger
  (`resetkit.Spring`: a PhysX joint with a preloaded linear drive; 2000 N/m, 7 N at mid) pushes up on the arm's tip
  and throws the handle to whichever end it is nearer, as a toggle mechanism does.
- **The trip mechanism** (the tripped breaker): a sprung cradle (5 N, 1000 N/m) presses a notch down on a pin at the
  arm's tip and centres the handle midway. Pushed towards ON, the pin rides the notch's shallow side (15 degrees) and
  the cradle springs it back. Pushed all the way to OFF, the pin rides the steep side (35 degrees) and lifts the
  cradle 3 mm. There a sprung pawl (3 N) slips in under a ledge on the cradle and holds it up: relatched. The notch is
  then clear of the pin, and the handle toggles OFF and ON. A full reset takes about 6 N at the handle.
- **Our estimates:** the mechanism's inner shapes and springs (Square D publishes neither), and the handle's size (8
  mm wide, 16.5 mm through the dead front). The enclosure's and breakers' outside sizes are the maker's.
- **Not modelled:** the contacts and the trip unit itself (nothing is powered); the panel door (it stands open).
- **Geometry.** Rigid bodies built from boxes; the load center and its wall board are fixed.

## Real-world build

- A Square D Homeline HOM48L125GC load center (4 spaces) with its cover (dead front), not wired to any supply, screwed
  upright (main lugs up) to a 19 mm plywood board, 60 cm wide and 50 cm tall, standing on the table and clamped or
  braced. The enclosure's bottom is 1 cm above the table, its back against the board, whose face is at x 42 cm.
- Four Square D HOM120 breakers in spaces 1 to 4. Three are ON; the one in space 4 is tripped before each trial. With
  no supply, trip it with a breaker trip tester, or use a breaker that has been tripped on a test bench.

Positions are in `task.json`.

## Reset between trials

Trip the breaker in space 4 again (with a trip tester), so its handle stands midway; check the other three are ON.
Nothing is consumed.
