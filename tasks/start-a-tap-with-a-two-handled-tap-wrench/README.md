# Start a tap with a two-handled tap wrench: a quarter turn in with both hands, then back it off

An aluminium block is clamped in a machinist's vise. An M10 hand tap stands started square in a hole drilled in its
top, its square end clamped in a bar tap wrench with two long round handles. The robot must take a handle in each
hand, turn the wrench a quarter turn in (clockwise seen from above) with an even push on both handles, back it off at
least a tenth of a turn to break the chip, and let go.

Machinists and fitters cut threads by hand this way. The push must be even: turned by one handle, the tap is pushed
sideways and snaps in the hole, the classic costly mistake (a broken tap is hard to get out and often scraps the
part). The task tests turning one tool about a fixed axis with both arms moving together, so that it is turned
without being pushed sideways, and a two-step order: in, then back.

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
  --instruction "Start the tap: turn it a quarter turn in with both hands on the wrench, then back it off at least a tenth of a turn to break the chip, and let go." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/tap-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=tap_wrench.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Start the tap: turn it a quarter turn in with both hands on the wrench, then back it off at least a tenth of a turn to break the chip, and let go." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/tap-demo
```

The demo is open loop: it uses only the fixed layout, never the simulator's internal state.

1. The left arm takes the back-left handle and the right arm the front-right one, 120 mm out from the tap, each
   gripper heading towards the tap with its fingers across the handle.
2. Together they turn the wrench 100 degrees clockwise, each hand moving round the circle, then ease off.
3. Together they turn it back 50 degrees, ease off, let go and lift away.

The demo reaches 1.0 (success) in 272 steps (13.6 s). No step of it is clamped by the run's delta-limit guard
(the gripper may change by at most 5% of its range per step); `test_tap_wrench.py` checks every step against it.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

| Score | Condition | How to check on a real table |
|---|---|---|
| 0 | at any point the tap snapped (pushed sideways more than 5 N averaged over 0.1 s, or tilted more than 2 degrees off the hole's axis) or its thread stripped (pushed or pulled along its axis more than 60 N) | a snapped tap; a square against the tap |
| 1.0 | the tap was turned a quarter turn in (90 degrees clockwise), then backed off at least a tenth of a turn (36 degrees) from the furthest it went in; no part of either gripper within 1 cm of the wrench; the wrench still (under 1 cm/s) for 1 s | video: the paint mark against the felt-pen line |
| 0.5 | a quarter turn in, not then backed off as for 1.0 | video |
| 0.25 | a tenth of a turn in | video |
| 0 | anything else | |

The snap latches: a tap side-loaded and then turned on still scores 0. Backing off counts only after the quarter turn
in. Two scripted checks (`TAP_DEMO=...` in the demo) show that the shortcuts fail:

- `onehand`: the left arm alone turns the wrench by its handle, the stroke the demo makes with two. The side load
  reached 8.5 N and the tap snapped: 0.
- `centre`: the right arm clamps the wrench's round centre with its full squeeze and turns its wrist like a spanner.
  The grip slipped and the tap did not turn: 0.

Offline checks of every tier, latch and edge case, and of the thread model: `python -m pytest -q test_tap_wrench.py`
(24 tests, no simulator).

## Runtime and seeds

Measured on an L40S, Isaac Sim's start-up included: the demo trial (272 steps, 13.6 s of simulated time) took
1.0 minute of wall-clock time; the Dream-1 trial (the full 1,200 steps, 1 minute simulated) took 2.9 minutes.
The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

- **Geometry.** Rigid bodies made of boxes and cylinders. The handles collide as 9 mm square bars inside the round
  12.8 mm handles that are seen (PhysX's custom cylinders miss thin contacts); the knurling is only painted. The
  tap is seen as it stands: 16 mm of its fluted cutting part above the hole (thread crests drawn as rings, 1.5 mm
  apart, on four lands), its 8 mm shank and its square drive; it collides only through the wrench. The wrench's
  centre is round (28 mm across), as on real adjustable tap wrenches, so it cannot be turned like a nut. The
  vise's castings collide; its jaw plates, screw, handle and mat are only seen. The block is clamped fixed.
- **The thread** is a force model (`tapkit.Thread`) on the tap-and-wrench body. Stiff springs hold it on the hole's
  axis (5,000 N/m sideways, 30 N m/rad in tilt) and tie its depth to its turn (1.5 mm per turn, M10). Turning it in
  takes 0.9 N m and backing out 0.3 N m: stick-slip friction, a torsional spring anchored where the tap last stopped
  that slides once those torques are reached. No chips are cut and the thread does not wear.
- **Snapping.** A real tap snaps from bending and depends on its size and the hole; here a "snapped" tap is a side
  load over 5 N (averaged over 0.1 s, so a brief knock does not count) or a tilt over 2 degrees. Turned by one handle,
  the whole push (7.5 N) loads the tap sideways; turned evenly with both, the pushes cancel.
- **Inertia.** The wrench's moments of inertia are raised (0.01 kg m2 about the axis, 0.002 in tilt) so the stiff
  thread springs stay stable; at the demo's turning rate this costs about 0.03 N m.
- **Why a quarter turn, not a full thread.** Holding opposite handles, each arm on its own side, the two arms can turn
  the wrench together through at most about 130 degrees; going further needs a regrip with the handles pointing front
  and back, where one arm must reach past the other and they collide. A machinist starting a tap also works in short
  strokes, backing off after each.
- Friction: wrench 0.35 (polished steel), block and vise 0.6. No contact-force cap is enforced in simulation.

## Real-world build

A small machinist's vise (75 mm jaws) on a rubber mat, fixed to the table; an aluminium block (60 x 60 x 40 mm) with
an 8.5 mm hole, clamped in it; an M10 x 1.5 hand tap started square with a few threads cut; a size-2 adjustable bar
tap wrench (280 mm); a paint mark on one handle's end and a felt-pen line on the block's top. Positions are in
`task.json`.

## Reset between trials

Back the tap out until the paint mark is back over the felt-pen line, with the handles back-left and front-right. Each
trial cuts a little more thread, so the torque drops over repeated trials in one hole: clamp a fresh block every five
trials.
