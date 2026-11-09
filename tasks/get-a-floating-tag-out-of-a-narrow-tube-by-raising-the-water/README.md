# Get a floating tag out of a narrow tube by raising the water

A clear tube stands fixed on the table with some water in it. A cork ring float with a red tag rides on the water,
too deep in the narrow tube for a gripper to reach. Beside it stand six slugs: four aluminium, two cork. The robot is
told only to take the tag out of the tube and put it in the tray. Dropping aluminium slugs in raises the water and
the float with it; after three, the tag stands clear of the rim and can be picked out. Cork slugs float and hardly
raise the water.

This is the Aesop's fable paradigm, the standard test of causal understanding of water displacement in animals and
children: New Caledonian crows and rooks drop stones into a tube to raise floating food (Bird & Emery 2009; Jelbert
et al. 2014), and children solve it from about five to seven years (Cheke, Loissel & Clayton 2012). It tests whether
a robot understands that sinking objects raise the water, chooses objects that sink over ones that float, and plans
the steps itself rather than following a procedure it is told.

## Files

| File | What |
|---|---|
| `raise_water.py` | The yam_arena task file: objects, the water, `success()`, `score()` and the scripted `demo()` |
| `waterkit.py` | Our own helper: rigid bodies built from boxes (see-through ones drawn as glass), the water-in-a-tube model, and hooks into yam_arena's scene and reset |
| `test_raise_water.py` | Offline checks of the water model, the design numbers, every scoring tier, and the demo's reach and per-step limits |

`waterkit.py` hooks `yam_arena.scene` and the embodiment's reset without modifying yam_arena, so the task runs with
the standard `yam_arena` embodiment and the `task_score` scorer. All the code is the task authors' own (MIT licence);
no third-party assets are used.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=raise_water.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Take the floating tag out of the tube and put it in the tray." \
  --scorer task_score --max-steps 1600 --store-frames --log-dir ~/yam-eval/water-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode. To render the
trial's videos (one MP4 per camera): `inspect-robots video ~/yam-eval/water-dream/<log>.json`.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=raise_water.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Take the floating tag out of the tube and put it in the tray." \
  --scorer task_score --max-steps 1600 --store-frames --log-dir ~/yam-eval/water-demo
```

The demo is open loop: it uses only the fixed layout, never the simulator's internal state.

1. The left arm takes three aluminium slugs from the rack, one at a time, carries each over the tube and lets it drop
   in through the float's ring. The water rises 15 mm with each.
2. The right arm pinches the tag, now 13 mm above the rim, lifts the float out of the tube, lays it in the tray and
   lets go.

The demo reaches 1.0 (success) in 1,196 steps (60 s). No step of it is clamped by the run's delta-limit guard (the
gripper may change by at most 5% of its range per step); `test_raise_water.py` checks every step against it, and
checks that every pose along both arms' paths is reached (the arms cannot point straight down much above 200 mm).

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

| Score | Condition | How to check on a real table |
|---|---|---|
| 1.0 | the float (with its tag) resting in the tray; no part of either gripper within 1 cm of it; still (under 1 cm/s) for 1 s | look |
| 0.5 | the float taken wholly out of the tube at some point, not ending in the tray | video |
| 0.25 | the tag's top raised above the rim at some point | video; the rim is the mark |
| 0 | anything else | |

The 0.5 and 0.25 conditions latch: once reached they count to the end. Two scripted checks (`WATER_DEMO=...` in the
demo) show that the shortcuts fail:

- `cork`: the left arm drops the two cork slugs instead of aluminium ones, then the right arm tries for the tag. The
  corks float (the second rests on the first, above the water); the water rises 6 mm and the tag stays 27 mm below
  the rim: 0.
- `reach`: nothing is dropped; the right arm reaches down into the tube, fingers slightly open, as deep as it goes.
  The fingers reach about 25 mm into the 40 mm bore and the tag is 33 mm down: 0.

The tube is fixed, so it cannot be tipped out, and there is no other tool to fish with. Offline checks of the water
model, the design numbers (two slugs leave the tag below the rim, three lift it 13 mm clear), every tier, the
latches and the demo: `python -m pytest -q test_raise_water.py` (28 tests, no simulator).

## Runtime and seeds

Measured on an L40S with the viewer on, Isaac Sim's start-up included: the demo trial (1,196 steps, 60 s of simulated
time) took 3.8 minutes of wall-clock time; the Dream-1 trial (the full 1,600 steps, 80 s simulated) took 4.6
minutes. The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

- **The water** is our own model (`waterkit.Water`); PhysX has no fluid. The level is the starting water plus the
  volume of everything under the surface, over the bore's area, solved every physics substep. Each body in the tube
  is pushed up by the weight of the water it displaces, at the centre of its submerged part, and damped in
  proportion to how much of it is under. Bodies are taken as upright for their submerged volume (the slugs cannot
  tilt more than about 20 degrees in the bore). There is no splashing, flow or surface tension, and the level
  follows at once. The water is drawn as a blue column whose top is moved to the level.
- **Geometry.** Rigid bodies made of boxes: round parts are 12-sided (the slugs) or 16 to 24 box segments (the tube,
  the ring). The tube and the water are drawn as glass so the cameras see inside.
- **Masses.** Aluminium slugs 51 g, cork slugs 4.6 g, the float 4 g. Friction 0.6 (slugs, rack), 0.3 (tube),
  0.2 (float). No contact-force cap is enforced in simulation.
- **The float** is kept from turning by a rib in the tube wall through a notch in its ring, as gauge floats are
  guided; the tag stays on the robot's side.

## Real-world build

An acrylic tube, 40 mm bore and 48 mm outside, 135 mm deep inside, bonded to an 80 x 80 x 5 mm steel plate fixed to
the table; a 3 mm acrylic rib bonded inside the far wall. A cork ring 39 mm outside, 30 mm hole, 12 mm thick, notched
for the rib, with a red acrylic tag 25 x 12 x 3 mm glued upright on its near side. Four aluminium and two cork slugs,
27 mm across and 35 mm long, in a wooden rack with 30 mm pockets 8 mm deep at 60 mm centres. A grey plastic tray 120 x
80 mm with 15 mm walls. Positions are in `task.json`.

## Reset between trials

Tip the water out, dry the slugs and stand them back in the rack in the same order, put the float back in the tube
with its notch on the rib, and fill the tube to the 71 mm line marked on it.
