# Clamp up a glued panel with the clamps alternating below and above

Two freshly glued oak boards lie edge to edge, the glue line still 2 mm open, across two small sliding bar clamps
lying jaws-up under the panel's ends. A third clamp hangs jaws-down on a wooden rack beside it. The robot must clamp
up the panel: every clamp closed snug so the joint closes, the clamps alternating below and above so the panel dries
flat, and none forced so hard that it squeezes the glue out.

Edge-gluing boards into a panel is everyday work in cabinet and furniture shops. A glue-up is judged by a standard
every woodworker learns: the joint closed along its whole length, clamps on alternate faces so the pressure does not
bow the panel, and pressure that is snug, not crushing, since an overtightened clamp squeezes the glue out and
starves the joint. The task tests clamping to that standard: where a clamp goes and on which side, how to close a
sliding jaw, and when to stop.

## Files

| File | What |
|---|---|
| `glue_panel.py` | The yam_arena task file: objects, the clamp model's set-up, `success()`, `score()` and the scripted `demo()` |
| `gluekit.py` | Our own helper: rigid bodies built from boxes, sliding jaws on prismatic joints, the clamp model (pads, squeeze, clutch), and hooks into yam_arena's scene and reset |
| `test_glue_panel.py` | Offline checks of the clamp model, every scoring tier and the latch, and the demo's reach and per-step limits |

`gluekit.py` hooks `yam_arena.scene` and the embodiment's reset without modifying yam_arena, so the task runs with
the standard `yam_arena` embodiment and the `task_score` scorer. All the code is the task authors' own (MIT licence);
no third-party assets are used.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=glue_panel.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Clamp up the glued panel: close every clamp snug so the joint closes, with the clamps alternating above and below so it stays flat." \
  --scorer task_score --max-steps 1600 --store-frames --log-dir ~/yam-eval/glue-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode. To render the
trial's videos (one MP4 per camera): `inspect-robots video ~/yam-eval/glue-dream/<log>.json`.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=glue_panel.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Clamp up the glued panel: close every clamp snug so the joint closes, with the clamps alternating above and below so it stays flat." \
  --scorer task_score --max-steps 1600 --store-frames --log-dir ~/yam-eval/glue-demo
```

The demo is open loop: it uses only the fixed layout, never the simulator's internal state. One arm works near the
panel at a time (two together run into each other).

1. The left arm lifts the spare clamp off its rack by the middle of its bar and lays it jaws-down across the
   panel's middle, its sliding jaw towards the robot. It grips that jaw by its sides, slides it along the bar onto
   board A's edge and on, until it presses about 4 N, eases off and lets go. The clamp's clutch holds.
2. The right arm closes the right-hand lower clamp the same way.
3. The left arm closes the left-hand lower clamp.

The joint closes as the first clamp takes up; all three clamps end snug (1.5 to 3 N), alternating below, above,
below. The demo reaches 1.0 (success) in 1,036 steps (52 s). No step of it is clamped by the run's
delta-limit guard (the gripper may change by at most 5% of its range per step); `test_glue_panel.py` checks every
step against it, and checks that every pose along both arms' paths is reached.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`). The glue line is
measured at both ends of the joint and its middle (20 mm from each end, and halfway), from one board's inner face to
the other's. A clamp counts as snug when no hand is on its sliding jaw, both pads bear on the panel's outer edges,
and it squeezes them with 1 to 6 N.

| Score | Condition | How to check on a real bench |
|---|---|---|
| 0 | at any point any clamp squeezed harder than 6 N for 0.5 s: the glue is squeezed out and the joint starved | the force pads' readings |
| 1.0 | the glue line closed (at most 0.2 mm open, not ridden over) at both ends and the middle; the boards level (faces within 1 mm, tilted under 2 degrees); three clamps snug, alternating below, above, below along the joint; no part of either gripper within 1 cm of the clamps or the panel; all still (under 1 cm/s) for 1 s | feeler gauge at the joint; straightedge across it; the force pads; look |
| 0.5 | the glue line closed at both ends and the middle and the boards level, with at least two clamps snug (or done as for 1.0 but not released or not still) | as above |
| 0.25 | the glue line closed at one end at least, the boards level, with at least one clamp snug | feeler gauge |
| 0 | anything else | |

Forcing a clamp latches: a starved joint scores 0 whatever follows. Two scripted checks (`GLUE_DEMO=...` in the
demo) show the other ways of falling short:

- `nospare`: only the two lower clamps are closed; the spare stays on its rack. The joint closes and both clamps are
  snug, but the panel is not clamped on top: 0.5 (the policy stops at step 656).
- `crush`: the right-hand clamp is driven 25 mm further, as a heavy hand would. Held there, it squeezes the panel
  past 6 N for more than 0.5 s, and the joint is starved: 0, which stays 0 although all three clamps end in place.

Offline checks of the clamp model, every tier, the latch and the demo: `python -m pytest -q test_glue_panel.py`
(26 tests, no simulator).

## Runtime and seeds

Measured on an L40S with the viewer on, Isaac Sim's start-up included: the demo trial (1,036 steps, 52 s of
simulated time) took 4.7 minutes of wall-clock time; the Dream-1 trial (the full 1,600 steps, 80 s simulated)
took 7.8 minutes. The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

- **The clamps** are our own model (`gluekit.Clamps`). Each sliding jaw runs on a PhysX prismatic joint along its
  bar. While a gripper is closed on it, it slides against the bar's friction (up to 1.5 N) and the clamp's clutch
  takes up the narrowest opening the hand holds, with 0.1 mm of play. While no gripper holds it, the clutch holds it
  on the bar at that opening. The pads do not touch the boards through the simulator's contacts. With both pads on
  the panel, the clamp squeezes it with 3,000 N/m times how far the pads are past its edges, the same force on both
  edges; one pad alone pushes its edge, and the panel slides until the other edge meets the other pad. So a clamp
  closed past the work keeps its pressure, as a real one does.
- **Pressure** is scaled to the arms: these arms push a few newtons, so "snug" is 1 to 6 N here, where a real
  glue-up of this size takes a few hundred.
- **The glue** is not simulated: no squeeze-out, no slip on wet glue, no curing. Overtightening is scored from the
  clamp force. Cupping, a panel bowing under clamps all on one face, is not simulated either: the alternating
  pattern is scored directly, as a woodworker checks it by eye.
- **The lower clamps' bars are fixed** to the table; a real clamp under a glue-up is held by its own weight and the
  boards'.
- **Geometry.** Rigid bodies built from boxes. The bars and the parts of the jaws standing off them collide; the
  pads, the jaws' front 8 mm and the part of each sliding jaw round its bar are only seen. Boards 0.075 kg each (oak); bar
  clamps 0.35 kg bar and 0.15 kg sliding jaw. Friction 0.8 between the boards and the bars, 0.5 on the jaws.
- **Arms.** The simulated arms are soft (about 0.23 N per mm they are sent past a contact): the demo presses a jaw
  on by sending the hand 26 mm past where its pad meets the work, and eases back before letting go.

## Real-world build

Two oak boards 160 x 50 x 12 mm; three small one-hand sliding bar clamps with 200 mm bars, each sliding jaw carrying
a thin force sensor on its pad; a wooden rack of two posts. A dry clamp-up: no glue is used in real trials (the glue
line is the 2 mm gap and a cream stripe painted on one edge). The two lower clamps are screwed to the table.
Positions are in `task.json`.

## Reset between trials

Open every clamp fully (sliding jaw 15 mm clear of the panel), lay the boards back on the lower clamps with a 2 mm
spacer in the joint and remove it, and hang the spare clamp back on its rack.
