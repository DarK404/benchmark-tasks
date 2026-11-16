# Cut a copper tube through with a screw-feed tubing cutter: a quarter turn a lap, never dented, offcut off

A length of 1/2 in copper tube stands upright in a stand. A screw-feed tubing cutter sits on it at the cut mark,
finger-snug (its wheel just touching), its feed knob pointing to the robot's right. The robot must cut the tube
through with the cutter and take the offcut off.

Cutting copper tube to length with a tubing cutter is daily work for plumbers, pipefitters and steamfitters (O*NET
47-2152) and HVAC mechanics and installers (49-9021). The maker's procedure (RIDGID's tubing cutter manuals): with the
wheel on the mark and just touching, turn the feed screw about a quarter turn, take the cutter right round the tube,
and tighten a quarter turn each lap until the tube parts. Fed much more at once, the wheel crushes the tube out of
round instead of cutting it, and a dented end will not take a fitting.

The task tests tool use with a feed that has to be judged: the knob is turned a little at a time, between laps, and
each lap takes two hands in turn pushing the cutter round the tube (one hand cannot follow it all the way round), so
the robot alternates feeding and going round until the tube parts, then lifts the loose offcut off.

## Files

| File | What |
|---|---|
| `tube_cutter.py` | The yam_arena task file: objects, the cutting model's set-up, `success()`, `score()` and the scripted `demo()` |
| `cutterkit.py` | Our own helper: rigid bodies built from boxes and cylinders, hinges, kinematic bodies, per-shape friction and hidden contact shapes, the cutting model (`TubeCut`), and hooks into yam_arena's scene and reset |
| `test_tube_cutter.py` | Offline checks of the cutting model, every scoring tier, and the demo's reach and per-step limits |

`cutterkit.py` hooks `yam_arena.scene` and the embodiment's reset without modifying yam_arena. The task runs with the
standard `yam_arena` embodiment and the `task_score` scorer. All the code is the task authors' own (MIT licence); no
third-party assets are used.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=tube_cutter.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Cut the copper tube through with the tubing cutter, and take the offcut off." \
  --scorer task_score --max-steps 3000 --store-frames --log-dir ~/yam-eval/tc-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode. To render the
trial's videos (one MP4 per camera), run `inspect-robots video ~/yam-eval/tc-dream/<log>.json`.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=tube_cutter.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Cut the copper tube through with the tubing cutter, and take the offcut off." \
  --scorer task_score --max-steps 3000 --store-frames --log-dir ~/yam-eval/tc-demo
```

The demo is open loop: it uses only the fixed layout, never the simulator's internal state. Three times:

1. **Feed.** With the knob pointing to the right, the right hand presses its closed fingertips on the knob's top and
   rolls it about a quarter turn the tightening way (the top drawn away from the robot), then lifts off.
2. **A lap.** Closed fingertips against the side of the cutter's frame push it round the tube: the right hand a
   third of the way, the left hand the next third, the right hand the last third, back to the start.

The tube parts during the second or third lap. Then the left hand takes the offcut by its top, lifts it off the stub
and stands it on the table. The demo reaches 1.0 (success) in 2,057 steps (102.9 s). No step of it is clamped by
the run's delta-limit guard (the gripper may change by at most 5% of its range per step); `test_tube_cutter.py`
checks every step against it, that every pose along the arms' paths is reached, and that the hands keep apart.

**Shortcut check.** With `TC_DEMO=crank` the demo turns the knob a full turn before the first lap: the wheel is fed
past half a turn into an uncut tube, the dent latches, and it scores 0.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

| Score | Condition | How to check on a real table |
|---|---|---|
| 0 (latched) | at any moment, the wheel fed more than half a turn of the feed screw (0.91 mm) past the groove under it: the tube crushed out of round | calipers: the outside diameter beside the cut differs by more than 0.3 mm between any two directions |
| 1.0 | cut through all round; the offcut off the stub and resting on the board or the table; no part of either gripper within 1 cm of the cutter or the offcut; both still; held for 1 s | look; the offcut lifts free |
| 0.5 | cut through (the offcut still on the stub, or in a hand, or not yet still) | look |
| 0.25 | a groove at least 0.45 mm deep (a quarter turn's feed) all round | knife-edge caliper jaws: the groove-bottom diameter 14.98 mm or less everywhere |
| 0 | anything else | |

`score()` gives 1.0 only once the full-score state has held for 1 s (until then 0.5). Offline checks of the model,
every tier and the demo: `python -m pytest -q test_tube_cutter.py` (29 tests, no simulator).

## Runtime and seeds

Measured on an L40S with the viewer on, Isaac Sim's start-up included:
- the demo trial (2,057 steps, 102.9 s of simulated time) took 5.4 minutes of wall-clock time;
- the Dream-1 trial (the full 3,000 steps, 2.5 minutes simulated) took 8.0 minutes.

The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

- **The cutting** is our own model (`cutterkit.TubeCut`). The cutter turns on the tube on a hinge about its axis,
  against a rolling resistance of 0.25 N m s/rad (our estimate: a few newtons at the frame). The groove is kept in 72
  sectors round the tube: as the wheel passes into each, it cuts it to the wheel's depth, up to the 1.02 mm wall (the
  maker's quarter turn a lap cuts about 0.45 mm a lap). Turning the cutter back and forth within a sector cuts nothing.
  The tube is cut through when every sector is; the offcut (held in place by the model until then) then stands
  loose on the stub's cut end.
- **The feed knob** is part of the cutter's body: it turns as a fingertip pressed on its top rolls across it, without
  slip (for contact it is a slippery box with a flat top at the knob's top, so the fingertip stays on top as it would
  on the turning knob; it is drawn round, turning). The feed screw is 7/16-14: 1.81 mm a turn (a user's figure for
  the RIDGID No. 10; the maker publishes none).
- **The dent** is our proxy: the wheel fed more than half a turn (twice the maker's quarter turn) past the groove
  under it. On a real trial it is judged by the tube's roundness with calipers, as in the scoring table.
- **Not modelled:** the forces of cutting and of turning the knob against the wheel (the real knob stiffens as the
  wheel bites); the cutter's slight wander off the mark; the burr inside the cut end.
- **Geometry.** Rigid bodies built from boxes and cylinders; the stand is fixed. The cutter's body sizes are ours,
  from its published length (about 6 in) and height (1.5 in).

## Real-world build

- A RIDGID No. 10 screw-feed tubing cutter (32910; 1/8 to 1 in; F-158 wheel).
- 1/2 in Type L copper tube (ASTM B88: 0.625 in outside, 0.040 in wall), 7 cm lengths.
- A 1/2 in malleable-iron floor flange screwed to a 12 x 12 x 1.8 cm plywood block, and a 1/2 in FIP x 1/2 in
  compression brass adapter screwed into it, with 1/2 in compression ferrules.

Positions are in `task.json`.

## Reset between trials

Loosen the adapter's nut and take out the stub with its ferrule; push in a new 7 cm length of tube with a new ferrule
so it stands 7 cm out of the nut, tighten the nut, and mark it 3 cm above the nut (10 cm above the table). Open the
cutter, put it on at the mark with the knob to the right, and tighten the knob until the wheel just touches. The tube
and the ferrule are consumed.
