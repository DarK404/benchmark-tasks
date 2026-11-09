# Bend a steel bar into a rectangular stirrup on a bending table

A straight 3 mm mild steel bar lies in the channel of a hand bending table, with three chalk marks on it. The robot
must bend it into a rectangular stirrup, 80 x 50 mm inside, with the table's lever: one arm swings the lever to make a
right-angle bend at a mark, and the other feeds the bar along the channel to the next mark. All three bends go the
same way, and the two ends meet at the fourth corner.

Reinforcing iron workers bend stirrups, the rectangular hoops that hold a beam's main bars, this way on site. Each
bend must be at its mark, the right way and by the right amount, allowing for springback, or the hoop does not
close. The task tests making a part to a drawing through a sequence of bends, with one arm feeding and the other
forming.

## Files

| File | What |
|---|---|
| `stirrup.py` | The yam_arena task file: the bending table, lever and bar, the shape measurement, `success()`, `score()` and the scripted `demo()` |
| `rodkit.py` | Our own helper: rigid bodies built from boxes, a hinge, the bar as an articulated chain with plastic joints, the bending-table model, and hooks into yam_arena's scene and reset |
| `test_stirrup.py` | Offline checks of the measurement, every scoring tier and latch, and the demo's reach and per-step limits |

`rodkit.py` hooks `yam_arena.scene` and the embodiment's reset without modifying yam_arena, so the task runs with the
standard `yam_arena` embodiment and the `task_score` scorer. All the code is the task authors' own (MIT licence); no
third-party assets are used.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=stirrup.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Bend the steel bar into a stirrup 80 x 50 mm inside: at each chalk mark, swing the lever to make a right-angle bend, all the same way, and feed the bar along between bends so the ends meet." \
  --scorer task_score --max-steps 1600 --store-frames --log-dir ~/yam-eval/stirrup-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode. To render the
trial's videos (one MP4 per camera): `inspect-robots video ~/yam-eval/stirrup-dream/<log>.json`.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=stirrup.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Bend the steel bar into a stirrup 80 x 50 mm inside: at each chalk mark, swing the lever to make a right-angle bend, all the same way, and feed the bar along between bends so the ends meet." \
  --scorer task_score --max-steps 1600 --store-frames --log-dir ~/yam-eval/stirrup-demo
```

The demo is open loop: it uses only the fixed layout, never the simulator's internal state.

1. The right arm takes the lever's handle from above and swings it about 100 degrees anticlockwise, holds it there
   for a second, and swings it back: the first bend, at the first mark.
2. The left arm's closed fingertips come down behind the bar's feed end and push it 50 mm along the channel, so the
   second mark reaches the pin; the right arm makes the second bend.
3. The left arm pushes the bar on 79 mm; the right arm makes the third bend and lets go.

The demo reaches 1.0 (success) in 900 steps (45 s). No step of it is clamped by the run's delta-limit guard (the
gripper may change by at most 5% of its range per step); `test_stirrup.py` checks every step against it.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`). A corner is a bend
of 20 degrees or more; its angle is read between the two sides it joins, as with a protractor. The sides are measured
to where the sides' lines meet at each corner, less the bar's thickness (inside lengths).

| Score | Condition | How to check on a real table |
|---|---|---|
| 0 | at any point the bar was bent both ways (bends of 30 degrees or more in each direction): scrap | look at the bar |
| 1.0 | exactly three corners, each 90 +-5 degrees, all the same way; the two sides between them 50 and 80 mm inside (+-5 mm, either order); the hoop closed: each end within 5 mm of the line of the side at the other end; every side straight (no point more than 3 mm off its line); the bar flat (within a 5 mm height band); no part of either gripper within 1 cm of the bar; the bar still (under 1 cm/s) for 1 s | lay the bar on the drawing: a protractor, a ruler, a straightedge |
| 0.5 | three corners at 90 +-5 degrees, but not everything else for 1.0 | protractor |
| 0.25 | at least one corner at 90 +-5 degrees | protractor |
| 0 | anything else | |

The latch holds: a bar bent both ways and then straightened scores 0. A stirrup bent wholly the other way is the
same part turned over and scores the same. The chalk marks are guides; the scorer checks the finished shape.

A scripted shortcut check (`STIRRUP_DEMO=nofeed` in the demo) shows that skipping the feed fails: the right arm
swings the lever three times without the bar being fed on. The second bend piles onto the first; the folded leg meets
a guide pin and kinks the bar the other way: 0.

Offline checks of the measurement, every tier, the latch and edge cases, and of the demo:
`python -m pytest -q test_stirrup.py` (29 tests, no simulator).

## Runtime and seeds

Measured on an L40S with the viewer on, Isaac Sim's start-up included: the demo trial (900 steps, 45 s of simulated
time) took about 6 minutes of wall-clock time; the Dream-1 trial (the full 1,600 steps, 80 s simulated) took 11.5
minutes. The bar's plasticity runs in Python on every physics substep, which
makes the simulation slower than real time. The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

- **The bar** is an articulated chain of 51 rigid links, 5 mm long with a 3 mm square section, joined by revolute
  joints about the vertical whose drives are the bar's bending stiffness (E I of 3 mm steel, 0.795 N m2: 159 N m/rad
  a joint). A joint yields for good once its elastic moment passes 0.8 N m (the plastic moment of 3 mm mild steel):
  its rest angle follows. The yield is judged on the joint angles averaged over 20 ms, so a contact jolt does not bend
  the bar. The bar bends only in the plane of the table, and does not collide with itself.
- **The bend** is a model of the bending table (`rodkit.Bender`), not formed by contact: as the lever turns
  anticlockwise, the bar wraps round the pin from the point where it touches it (centreline radius 10.8 mm), and the
  lever is resisted by 0.8 N m while the bend grows. Let off, a bend springs back 3 degrees. While a bend is formed,
  the bar beside it does not yield (on a real table the roller presses it against the pin there). The roller is
  only seen; pushing a bar of short links round the pin by contact kinked it in every variant tried.
- **Friction** is 0.02 between the bar and the plate (a greased table): with more, PhysX's contacts on the bar's many
  short links resisted its sliding far beyond its 14 g weight.
- **Scale.** Site stirrups are 8 to 12 mm rebar, bent on larger tables; this is about 1:3, sized for the arms.
- The guide pins and strips are as on a real table; the strips are 0.8 mm high so the fingertips push the bar's end
  above them.

## Real-world build

A steel plate 160 x 290 x 15 mm fixed to the table; a 16 mm steel bending pin and three 10 mm guide pins, 22 mm high;
two steel strips 3 x 0.8 x 110 mm glued either side of the channel; a lever of 16 x 6 mm red-painted flat bar on the
bending pin, with a 12 mm roller near the pivot and a 20 mm round handle 150 mm out. The bar is 3 mm bright mild steel
(annealed), 255 mm long, with chalk marks 88, 138 and 217 mm from its free end. Positions are in `task.json`.

## Reset between trials

The bar is consumed: lay a fresh straight bar in the channel, its free end 88 mm past the bending pin, and swing the
lever back to its start, the handle towards the robot's right (as in `task.json`).
