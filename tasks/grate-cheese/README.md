# Grate about 20 g of cheese into a dish on a flat grater, fingers never on the blades

A flat coarse grater lies across a glass dish, its handle to the robot's left; half a block of cheddar sits on a
cutting board to the right. The robot must grate about 20 g of the cheese into the dish.

Grating cheese is one of the commonest home cooking tasks. A flat (paddle) grater is laid across a dish or bowl and
steadied by its handle while the cheese is pushed along the blades, pressing, away from the handle; it cuts on the
push. Two things make it more than moving an object: the cut depends on the pressure and the length of each stroke,
so getting about the amount a recipe asks for means judging the strokes; and the hand holding the cheese comes
closer to the blades as the block wears down, so the stroke has to stay above them (grated knuckles are a familiar
kitchen injury).

## Files

| File | What |
|---|---|
| `grate_cheese.py` | The yam_arena task file: objects, the grating model's set-up, `success()`, `score()` and the scripted `demo()` |
| `gratekit.py` | Our own helper: rigid bodies built from boxes, the grating model (the cheese's underside bearing on the grating surface, cutting on the push stroke), shapes that collide without being drawn, and hooks into yam_arena's scene and reset |
| `test_grate_cheese.py` | Offline checks of the grating model, every scoring tier, and the demo's reach and per-step limits |

`gratekit.py` hooks `yam_arena.scene` and the embodiment's reset without modifying yam_arena. The task runs with the
standard `yam_arena` embodiment and the `task_score` scorer. All the code is the task authors' own (MIT licence); no
third-party assets are used.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=grate_cheese.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Grate about 20 g of the cheese into the dish." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/grate-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode. To render the
trial's videos (one MP4 per camera), run `inspect-robots video ~/yam-eval/grate-dream/<log>.json`.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=grate_cheese.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Grate about 20 g of the cheese into the dish." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/grate-demo
```

The demo is open loop: it uses only the fixed layout, never the simulator's internal state.

1. **Hold the grater.** The left hand takes the grater by its handle and steadies it on the dish.
2. **Take the cheese.** The right hand takes the cheese from the board by its sides, 17 mm up them, and sets it on
   the blades at the handle end.
3. **Grate.** Eight strokes: press, push the block along the blades away from the handle, ease off, come back.
4. **Put it down.** The right hand puts the cheese back on the board and lets go; the left lets go of the grater.

The demo reaches 1.0 (success) in 922 steps (46.1 s). In our development runs of the same demo it grated about
20 g, and the fingers never came nearer the blades than 7 mm. No step of it is clamped by the run's delta-limit
guard (the gripper may change by at most 5% of its range per step); `test_grate_cheese.py` checks every step
against it, that every pose along the arms' paths is reached, and that the left hand steadies the grater through
every stroke.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

| Score | Condition | How to check on a real table |
|---|---|---|
| 1.0 | 15 to 25 g of cheese grated into the dish; the cheese back on its board; the grater still on the dish; no part of either gripper within 1 cm of the cheese; the cheese and the grater still; held for 1 s | weigh the dish before and after on a kitchen scale; look |
| 0.5 | 5 g or more grated, but outside 15 to 25 g, or the cheese not back on the board (or a hand still on it, or not yet still) | the scale; look |
| 0.25 | some grated, under 5 g | the scale |
| 0 | nothing grated; or, at any moment (latched), a finger of either gripper within 3 mm of the blades over the grating surface, or the grater off the dish (its middle more than 5 cm from where it lay, or tipped over 20 degrees) | the scale; the video |

`score()` gives 1.0 only once the full-score state has held for 1 s (until then 0.5). Offline checks of the model,
every tier and the demo: `python -m pytest -q test_grate_cheese.py` (29 tests, no simulator).

## Runtime and seeds

Measured on an L40S with the viewer on, Isaac Sim's start-up included:
- the demo trial (922 steps, 46.1 s of simulated time) took 4.9 minutes of wall-clock time;
- the Dream-1 trial (the full 2,400 steps, 2 minutes simulated) took 7.9 minutes.

The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

- **The grating** is our own model (`gratekit.Grating`). The cheese's underside bears on the grating surface at 25
  points through springs (200 N/m each, capped at 3 N), so the cut never rests on a thin simulator contact; the grater
  takes the opposite forces. Sliding resists with friction 0.5 (full from 5 cm/s up). Pushed towards the grater's tip
  while pressed, the block loses cheese by Archard's wear law: 8e-6 m3 per newton and metre slid, about 2.3 g a stroke
  at 9 N (our estimate for coarse grating, not a measured figure); the return stroke cuts nothing. The grated amount
  is that volume times cheddar's density (1,060 kg/m3).
- **What is drawn.** The block is drawn worn from below (its lower 14 mm is a layer the model raises as it is
  grated); a pile of shreds rises in the dish with the grated amount. The block's full collision box is not drawn:
  on the board and the table the simulator's contacts hold it; on the grater only the model's do (the grating surface
  does not collide, and its frame's rails sit 8 mm below it so the worn block never rests on them; on the real grater
  the frame is nearly flush).
- **Friction.** The grater's steel frame on the glass rim: 0.2. Unsteadied, the grater creeps about a centimetre
  along the rim over the strokes in our simulation; the demo steadies it, as people do.
- **Geometry.** Rigid bodies built from boxes; the dish and the board fixed.
- **Arms.** The simulated arms are soft (about 0.3 N/mm): the demo presses by aiming 30 mm into the grater (about
  9 N) and aims each stroke 12 mm past its ends, as the arm trails the target against the friction.

## Real-world build

- A Microplane Professional coarse paddle grater, item 38000 (13 x 2-5/8 in, stainless steel; grating surface 5-1/4 x
  2-5/16 in), laid across the dish, its handle to the robot's left.
- A 20 cm round glass dish, 45 mm deep (taped down).
- A wooden cutting board, 25 x 15 cm, 12 mm thick (taped down).
- Half of a Tillamook sharp cheddar 8 oz block (5.5 x 2.25 x 1.13 in), cut to 70 x 57 x 29 mm, about 113 g, cold
  from the fridge.

Positions are in `task.json`.

## Reset between trials

Empty the dish and weigh it; put a fresh half block on the board as in the setup (the cheese is consumed: each
trial starts with a full 29 mm thick half block, so an 8 oz block serves two trials); lay the grater back across
the dish.
