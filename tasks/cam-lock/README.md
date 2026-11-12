# Join a shelf to a cabinet side with cam locks: cams open to the edge, push the shelf on, turn them tight

A cabinet side stands upright in an assembly jig with two connecting bolts and two dowels standing out of its face. A
shelf lies flat in front of it with its two cam locks in its top face, both turned away from the edge. A screwdriver
stands in a holder on either side. The robot must join the shelf to the side with its cam locks.

Assembling flat-pack furniture at home is one of the most familiar practical jobs there is: bookcases, wardrobes and
kitchen units arrive as panels, and the cam-and-bolt connector (a cam lock, or minifix) holds most of them together.
The bolt is screwed into one panel. The cam sits in a bore in the other panel's face, and its opening must face the
edge, its arrow pointing at the bolt, before the panels go together. Only then can the bolt's head enter it. Turned
half a round clockwise, the cam's eccentric draws the head in and pulls the joint shut.

"Arrows to the edge" is the step the instructions show first, and the one most often missed: pushed onto a cam that
is still closed, the bolt's head meets the cam's side and the panel will not go on.

The task tests setting a fitting's orientation before mating, then drawing the joint tight with a tool, with two hands:
- each hand turns its own cam open;
- both push the shelf on square;
- each turns its cam half a round in two quarter turns, letting go and regripping between them.

## Files

| File | What |
|---|---|
| `cam_lock.py` | The yam_arena task file: objects, the cam-lock model's set-up, `success()`, `score()` and the scripted `demo()` |
| `camkit.py` | Our own helper: rigid bodies built from boxes, the cam-and-bolt model (the bolts and their holes, the cams, the screwdrivers in the cams' recesses and their holders' holes), and hooks into yam_arena's scene and reset |
| `test_cam_lock.py` | Offline checks of the model, every scoring tier, and the demo's reach and per-step limits |

`camkit.py` hooks `yam_arena.scene` and the embodiment's reset without modifying yam_arena. The task runs with the
standard `yam_arena` embodiment and the `task_score` scorer. All the code is the task authors' own (MIT licence); no
third-party assets are used.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=cam_lock.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Join the shelf to the cabinet side with its two cam locks." \
  --scorer task_score --max-steps 2000 --store-frames --log-dir ~/yam-eval/cam-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode. To render the
trial's videos (one MP4 per camera), run `inspect-robots video ~/yam-eval/cam-dream/<log>.json`.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=cam_lock.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Join the shelf to the cabinet side with its two cam locks." \
  --scorer task_score --max-steps 2000 --store-frames --log-dir ~/yam-eval/cam-demo
```

The demo is open loop: it uses only the fixed layout, never the simulator's internal state. The two hands work at
the same time, each on its own side.

1. **Set the cams.** Each hand takes its screwdriver from its holder, puts it into its cam's cross recess and turns the
   cam's arrow to the shelf's edge: the right cam a quarter turn anticlockwise, the left a quarter turn clockwise. Then
   it stands the screwdriver back in its holder.
2. **Push the shelf on.** With closed fingers on the shelf's near edge, both hands push it 3 cm, square, onto the
   bolts and dowels, until the bolts' heads are in the cams.
3. **Lock the cams.** Each hand takes its screwdriver again and turns its cam clockwise in two quarter turns. Between
   them it lets go (the screwdriver stands in the recess), turns its wrist back and grips again. The joint draws
   shut.
4. **Tidy up.** Both stand their screwdrivers back in their holders and lift clear.

The demo reaches 1.0 (success) in 1,126 steps (56.3 s). No step of the demo is clamped by the run's
delta-limit guard, which lets the gripper change by at most 5% of its range per step. `test_cam_lock.py` checks every
step against it, and checks that every pose along the arms' paths is reached.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

| Score | Condition | How to check on a real bench |
|---|---|---|
| 1.0 | both cams locked (turned at least 120 degrees clockwise past open with the bolt's head in them); the joint shut at both bolts (gap at most 0.5 mm); both screwdrivers standing back in their holders; no part of either gripper within 1 cm of the shelf or the screwdrivers; all still (under 1 cm/s) for 1 s | a feeler gauge at the joint; look |
| 0.5 | the joint locked and shut, but a screwdriver not back, or not yet still for 1 s | as above |
| 0.25 | the shelf on the bolts with both heads in their cams, but not locked and shut | look |
| 0 | anything else | |

`score()` gives 1.0 only once the full-score state has held for 1 s (until then 0.5). Two scripted checks
(`CAM_DEMO=...`) show what falls short:
- `noset`: the same job without turning the cams open. The shelf stops 9 mm short on the cams' sides, and the turns
  that follow lock nothing: 0.
- `force`: the same, but the shelf is shoved on hard (about 12 N at each bolt): it still stops 9 mm short, 0.

Offline checks of the model, every tier and the demo: `python -m pytest -q test_cam_lock.py` (27 tests, no simulator).

## Runtime and seeds

Measured on an L40S with the viewer on, Isaac Sim's start-up included:
- the demo trial (1,126 steps, 56.3 s of simulated time) took 4.0 minutes of wall-clock time;
- the Dream-1 trial (the full 2,000 steps, 100 s simulated) took 6.3 minutes.

The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

- **The connectors** are our own model (`camkit.CamLockJoint`); PhysX has no bolts in holes or eccentric cams.
  - *The bolts.* The bolts and dowels are drawn. The shelf's edge holes take them only when it comes up square and in
    line (within 2.5 mm and 2.5 degrees); otherwise its edge stands on their ends.
  - *Mating.* With a cam open (its arrow within 30 degrees of the edge), the bolt's head goes in and the shelf comes
    up to 1.5 mm of the face. With a cam closed, the head meets the cam's side 9 mm short, and the shelf stops there
    however hard it is pushed. A real board would break out round the cam if forced far harder, with a mallet say;
    these arms cannot push nearly that hard, so nothing breaks in the model, and nothing in the score depends on it.
  - *Locking.* With the head in, turning the cam clockwise draws the joint shut as an eccentric would, fully at 180
    degrees; past 30 degrees the cam holds the head. Its stops are at -10 and 200 degrees. With no head in it, a cam
    turns freely.
  - *The screwdrivers.* A screwdriver's tip within 2 mm of a cam's middle is in its cross recess (it is guided gently
    from 4 mm). From then on the cam turns with it, and it stands upright in the recess when let go. The holders'
    chamfered mouths guide a blade's tip into its 9 mm hole from 6 mm. The cams are drawn by bodies the model turns.
- **Geometry.** Rigid bodies built from boxes: the shelf (0.5 kg, friction 0.4) and the screwdrivers (0.12 kg, round
  handles, friction 1.0); the side, the jig and the holders fixed.
- **Real sizes**: Minifix-type 15 mm cams in 15 mm bores, drilling distance B = 24 mm (the cam's middle 24 mm in from
  the edge) with the connecting bolts made for B = 24 mm, whose heads reach the cam's middle; PH2 × 100 screwdrivers. The side is cut down
  to 12 cm tall so that the hands turning the screwdrivers stay above it.

## Real-world build

- A 16 mm white melamine particleboard side (32 x 12 cm) clamped upright in a jig, with two Minifix-type connecting
  bolts for drilling distance B = 24 mm (M6 thread, in 8 mm holes; about 25 mm out of the face to the head's end) and
  two 8 mm beech dowels in its face.
- A 16 mm melamine shelf (26 x 20 cm) with two 15 mm zinc cam housings in 15 mm bores in its top face, their middles
  24 mm in from the joining edge, and 8 mm edge holes for the bolts and dowels (a standard Minifix drilling jig for
  B = 24/34 sets all of these).
- Two PH2 x 100 screwdrivers in drilled wooden holders.

Positions are in `task.json`.

## Reset between trials

Turn the cams back half a round, pull the shelf off, turn the cams to the starting marks (a quarter turn either way
from open), lay the shelf at its mark, and stand the screwdrivers in their holders.
