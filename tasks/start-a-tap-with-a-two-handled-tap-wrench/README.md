# Start a tap with a two-handled tap wrench: a quarter turn in with both hands, then back it off

An aluminium block is clamped in a machinist's vise. An M10 hand tap stands started square in a hole drilled in its
top, its square end clamped in a bar tap wrench with two long round handles. The robot must turn the tap a quarter
turn in (clockwise seen from above), back it off at least a tenth of a turn to break the chip, and let go, keeping
the tap square as it cuts.

Machinists and fitters cut threads by hand this way, with a hand on each handle and an even push. The instruction
does not say so: turned by one handle, the tap is pushed sideways, leans, and cuts its thread crooked, so the screw
that goes in later stands crooked and the part is scrap. The task tests turning one tool about a fixed axis with both
arms moving together, so that it is turned without being pushed sideways, and a two-step order: in, then back.

## Files

| File | What |
|---|---|
| `tap_wrench.py` | The yam_arena task file: objects, the thread, `success()`, `score()` and the scripted `demo()` |
| `tapkit.py` | Our own helper: rigid bodies built from boxes and cylinders (with settable inertia), the tap-in-a-hole thread model, and hooks into yam_arena's scene and reset |
| `test_tap_wrench.py` | Offline checks of the scorer, the thread model, and the demo's reach and per-step limits |

`tapkit.py` hooks `yam_arena.scene` and the embodiment's reset without modifying yam_arena, so the task runs with the
standard `yam_arena` embodiment and the `task_score` scorer. All the code is the task authors' own (MIT licence); no
third-party assets are used.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=tap_wrench.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Start the tap: turn it a quarter turn in, then back it off at least a tenth of a turn to break the chip, and let go." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/tap-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=tap_wrench.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Start the tap: turn it a quarter turn in, then back it off at least a tenth of a turn to break the chip, and let go." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/tap-demo
```

The demo is open loop: it uses only the fixed layout, never the simulator's internal state.

1. The left arm takes the back-left handle and the right arm the front-right one, 120 mm out from the tap, each
   gripper heading towards the tap with its fingers across the handle.
2. Together they turn the wrench 100 degrees clockwise, each hand moving round the circle, then ease off.
3. Together they turn it back 50 degrees, ease off, let go and lift away.

The demo reaches 1.0 (success) in 273 steps (13.7 s), the thread leaning 0.9 degrees. No step of it is clamped by
the run's delta-limit guard (the gripper may change by at most 5% of its range per step); `test_tap_wrench.py` checks
every step against it.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

| Score | Condition | How to check on a real table |
|---|---|---|
| 0 | the thread cut crooked: with the wrench let go, the tap leans more than 2 degrees off square | after the trial, stand a 50 mm engineer's square on the block's top beside the tap, in two directions at right angles: a gap over 1.4 mm between the square's blade and the tap's shank at 40 mm up |
| 1.0 | the tap was turned a quarter turn in (90 degrees clockwise), then backed off at least a tenth of a turn (36 degrees) from the furthest it went in; no part of either gripper within 1 cm of the wrench; the wrench still (under 1 cm/s) for 1 s | video: the paint mark against the felt-pen line; the square, as above |
| 0.5 | a quarter turn in, not then backed off and let go as for 1.0 (or not yet still for 1 s) | video |
| 0.25 | a tenth of a turn in | video |
| 0 | anything else | |

A crooked thread cannot be put right: the tap's lean changes only while it cuts new thread, so turning it back and
forth in the thread already cut leaves the lean as it is. Backing off counts only after the quarter turn in.
`score()` gives 1.0 only once the wrench has been still, with both grippers clear, for 1 s (until then 0.5).
Two scripted checks (`TAP_DEMO=...` in the demo) show that the shortcuts fail:

- `onehand`: the left arm alone turns the wrench by its handle, the stroke the demo makes with two. The tap leaned
  about 4 degrees as it was pushed round and cut its thread leaning 6.1 degrees: 0. (The two-handed demo leaves it
  at 0.9 degrees.)
- `centre`: the right arm clamps the wrench's round centre with its full squeeze and turns its wrist like a spanner.
  The grip slipped and the tap did not turn: 0.

Offline checks of every tier, latch and edge case, and of the thread model: `python -m pytest -q test_tap_wrench.py`
(25 tests, no simulator).

## Runtime and seeds

Measured on an L40S with the viewer on, Isaac Sim's start-up included: the demo trial (273 steps, 13.7 s of simulated
time) took 1.0 minute of wall-clock time; the Dream-1 trial (the full 1,200 steps, 1 minute simulated) took 3.0
minutes.
The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

- **Geometry.** Rigid bodies made of boxes and cylinders. The handles collide as 9 mm square bars inside the round
  12.8 mm handles that are seen (PhysX's custom cylinders miss thin contacts); the knurling is only painted. The
  tap is seen as it stands: 16 mm of its fluted cutting part above the hole (thread crests drawn as rings, 1.5 mm
  apart, on four lands), its 8 mm shank and its square drive; it collides only through the wrench. The wrench's
  centre is round (28 mm across), as on real adjustable tap wrenches, so it cannot be turned like a nut. The
  vise's castings collide; its jaw plates, screw, handle and mat are only seen. The block is clamped fixed.
- **The thread** is a force model (`tapkit.Thread`) on the tap-and-wrench body. A stiff spring holds it on the
  hole's axis at the block's top (5,000 N/m sideways), and another ties its depth to its turn (1.5 mm per turn, M10).
  Turning it in takes 0.9 N m and backing out 0.3 N m: stick-slip friction, a torsional spring anchored where the tap
  last stopped that slides once those torques are reached. No chips are cut.
- **Leaning, and a crooked thread.** A tap one turn into its thread is held upright only by those few threads, so a
  side push on the wrench leans it: here a spring of 6 N m/rad holds the tap's axis along the thread's. As the tap
  cuts new thread it cuts it along its own axis, so the thread's axis moves towards the tap's: by the share
  (new cut) / (0.1 turn) of the angle between them. Turned by one handle, the whole push (7.5 N, 40 mm above the
  block) leans the tap about 3 to 4 degrees and the thread follows; turned evenly with both, the pushes cancel. The
  constants are chosen, not measured: with them the two-handed demo leaves the thread within 1 degree and the
  one-handed stroke 6 degrees, either side of the 2-degree limit that a square shows plainly (1.4 mm at 40 mm).
  Nothing snaps or strips in the model; a real tap side-loaded this way is more likely to cut crooked than to break.
- **Inertia.** The wrench's moments of inertia are raised (0.01 kg m2 about the axis, 0.002 in tilt) so the stiff
  thread springs stay stable; at the demo's turning rate this costs about 0.03 N m.
- **A fresh thread every trial.** The model cuts at the same torque on every trial, as a fresh block does in the real
  set-up (see the reset below).
- **Why a quarter turn, not a full thread.** Holding opposite handles, each arm on its own side, the two arms can turn
  the wrench together through at most about 130 degrees; going further needs a regrip with the handles pointing front
  and back, where one arm must reach past the other and they collide. A machinist starting a tap also works in short
  strokes, backing off after each.
- Friction: wrench 0.35 (polished steel), block and vise 0.6. No contact-force cap is enforced in simulation.

## Real-world build

A small machinist's vise (75 mm jaws) on a rubber mat, fixed to the table; an aluminium block (60 x 60 x 40 mm) with
an 8.5 mm hole, clamped in it, a fresh one for every trial; an M10 x 1.5 hand tap started square exactly one turn in;
a size-2 adjustable bar tap wrench (280 mm); a paint mark on one handle's end and a felt-pen line on the block's top;
a 50 mm engineer's square for the check after the trial. Positions are in `task.json`.

## Reset between trials

Every trial cuts fresh thread in a fresh block:
1. Before the trials, prepare a batch of blocks. Drill each 8.5 mm in a drill press. Start the tap in it square, the
   drill press's chuck guiding the wrench's square drive: set the wrench with its paint-marked handle pointing
   back-left as in the layout, and turn exactly one full turn in (1.5 mm), so the handles end where they began. Draw
   the felt-pen line under the paint mark, then back the tap out.
2. Between trials, back the tap out of the used block, set that block aside, and clamp a fresh one in the vise at the
   same place. Screw the tap in by hand: it stops where the started thread ends, one turn in, with its handles
   back-left and front-right and the paint mark over the felt-pen line.

The blocks are cheap aluminium offcuts (0.4 kg each); a used block, its thread finished to depth, is a stock tapped block.
