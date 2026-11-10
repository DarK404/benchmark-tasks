# File a steel block flat down to a scribed line

A mild steel bar is clamped in a machinist's vise with its top 1 mm above a line scribed round it, and a flat file
lies on a rest beside it. The robot must file the top down to the line, keeping it flat, and put the file back on
its rest.

Filing to a line is the first exercise of every fitting and machining apprenticeship, and a daily job in fitting
parts. A file cuts only on the push stroke; both hands must press evenly, the handle hand and the hand on the tip,
or the file rocks and rounds the work; and filed below the line, the part is scrap. The task tests controlled
material removal with a hand tool: pressure, flatness, and knowing when to stop.

## Files

| File | What |
|---|---|
| `file_flat.py` | The yam_arena task file: objects, the cut, `success()`, `score()` and the scripted `demo()` |
| `filekit.py` | Our own helper: rigid bodies built from boxes, the filing model (the file standing on springs we compute, the block's top lowered as it is filed), and hooks into yam_arena's scene and reset |
| `test_file_flat.py` | Offline checks of the filing model, every scoring tier, and the demo's reach and per-step limits |

`filekit.py` hooks `yam_arena.scene` and the embodiment's reset without modifying yam_arena, so the task runs with
the standard `yam_arena` embodiment and the `task_score` scorer. All the code is the task authors' own (MIT licence);
no third-party assets are used.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=file_flat.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "File the top of the block down to the scribed line, keeping it flat, then put the file back on its rest." \
  --scorer task_score --max-steps 1600 --store-frames --log-dir ~/yam-eval/file-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode. To render the
trial's videos (one MP4 per camera): `inspect-robots video ~/yam-eval/file-dream/<log>.json`.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=file_flat.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "File the top of the block down to the scribed line, keeping it flat, then put the file back on its rest." \
  --scorer task_score --max-steps 1600 --store-frames --log-dir ~/yam-eval/file-demo
```

The demo is open loop: it uses only the fixed layout, never the simulator's internal state.

1. The right arm takes the file by its handle and lays the blade on the block; the left arm pinches the blade near
   its tip, beyond the block's far end.
2. Together they make five push strokes of 80 mm along the block, lifting the file 15 mm for each return: three
   strokes pushing 8 N down, then two lighter finishing strokes of 4 N, as a fitter eases off near the line. The tip
   hand bears a little harder than the handle hand, which keeps the file level.
3. The left arm lets go and the right arm lays the file back on its rest.

The top ends 0.08 to 0.12 mm above the line, flat within 0.04 mm. The demo reaches 1.0 (success) in 801 steps
(40 s). No step of it is clamped by the run's delta-limit guard (the gripper may change by at most 5% of its range
per step); `test_file_flat.py` checks every step against it, and checks that every pose along both arms' paths is
reached.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`). The top is measured
in four sections along its length, each against the scribed line.

| Score | Condition | How to check on a real bench |
|---|---|---|
| 0 | at any point any section filed more than 0.05 mm below the line: the part is spoilt | the line |
| 1.0 | every section at most 0.2 mm above the line and the top flat within 0.15 mm; the file lying back on its rest, no part of either gripper within 1 cm of it, still (under 1 cm/s) for 1 s | straightedge and feeler gauge against the line; look |
| 0.5 | every section at most 0.4 mm above the line (or finished as for 1.0, but the file not back on its rest) | feeler gauge |
| 0.25 | at least 0.3 mm filed off everywhere | feeler gauge |
| 0 | anything else | |

Filing below the line latches: a spoilt part scores 0 whatever follows. Two scripted checks (`FILE_DEMO=...` in the
demo) show that careless filing fails:

- `nostop`: seven full 8 N strokes instead of easing off. After four the top is 0.16 mm above the line, within
  tolerance; the fifth takes it below the line, and the part is spoilt: 0.
- `onehand`: the right arm files alone, pushing the handle down, the tip not held. The file pivots in the grip and
  rides nose-up (about 26 degrees), touching the block only at its near end: 0.05 mm comes off there, 0.

Offline checks of the filing model, every tier, the latch and the demo: `python -m pytest -q test_file_flat.py`
(25 tests, no simulator).

## Runtime and seeds

Measured on an L40S with the viewer on, Isaac Sim's start-up included: the demo trial (801 steps, 40 s of simulated
time) took 2.6 minutes of wall-clock time; the Dream-1 trial (the full 1,600 steps, 80 s simulated) took 4.4
minutes. The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

- **The cut** is our own model (`filekit.Filing`); PhysX cannot remove material. Where the blade's flat bottom lies
  below the block's top (three points across each of four sections), the block pushes the file up with a spring we
  compute (2,000 N/m a section) and resists its sliding with friction 0.5; the file stands on these springs, not on
  the simulator's contacts, so the drawn file can sink about 1 mm into the drawn top. While the file slides towards
  its tip, each section loses height in proportion to the force on it times the distance slid (Archard's wear law,
  0.6 mm per newton-metre); sliding back cuts nothing. A rocked or unevenly pressed file bears harder at one end
  and cuts it lower. No chips, burrs or heat; the file does not wear or clog.
- **Geometry.** Rigid bodies built from boxes. The vise's castings and the bar collide; its jaw plates, screw,
  T-bar handle and mat, the file's round handle, ferrule and teeth, and the scribed line are only seen (the handle
  collides as a 20 mm square bar inside the round one). The bar collides only up to 8 mm below the line; above
  that its top is the model's, drawn as four sections, blued with a bright filed face, moved down as they are cut.
- **Masses and friction.** File 0.14 kg; friction 0.6 between the grippers and the file, the rest and the vise.
- **Arms.** The simulated arms are soft (about 0.23 N per mm they are sent past a contact); the demo presses by
  sending the hands below the contact.

## Real-world build

A small machinist's vise (75 mm jaws) on a rubber mat, fixed to the table; a mild steel bar 60 x 16 mm in section
clamped with its top 86 mm above the table, its faces coated with marking blue and a line scribed round it 1 mm
below its top with a height gauge; a 150 mm
second-cut flat file with a wooden handle; a rest of two wooden blocks. Positions are in `task.json`.

## Reset between trials

The block is consumed: clamp a fresh bar with its top 1 mm above a freshly scribed line (or scribe a used bar 2 mm
lower and clamp it higher), brush off the filings, and lay the file back on its rest.
