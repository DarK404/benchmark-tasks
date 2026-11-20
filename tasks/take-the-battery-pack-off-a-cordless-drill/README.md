# Take the battery pack off a cordless drill: hold the drill, push in both release buttons and slide the pack off

A cordless drill (Milwaukee 3601-20, M18) lies on its side on the table with its battery pack (M18 CP2.0,
48-11-1820) on, as left on a bench. The instruction is "Take the battery pack off the drill."

Taking the battery off a cordless drill, to charge it or to swap in a charged one, is something most people who own a
drill do every time they use it. The pack slides onto rails on the drill's foot and is held by a latch that lets go
only while the release buttons on both of its sides are pushed in. So one hand holds the drill while the other
squeezes both buttons and slides the pack off along the foot; neither hand can do it alone. A parallel gripper can
push both buttons at once, but only by closing across the pack, and the drill must be held against the pull.

## Files

| File | What |
|---|---|
| `drill_battery.py` | The yam_arena task file: the drill, the pack, its two sprung release buttons with their latch pawls, the terminal shoe, `success()`, `score()` and the scripted `demo()` |
| `drillkit.py` | Our own helper: rigid bodies built from boxes, sprung slides, and hooks into yam_arena's scene and reset |
| `test_drill_battery.py` | Offline checks of the parts, every scoring tier and the demo's reach and per-step limits |

`drillkit.py` hooks `yam_arena.scene` and the embodiment's reset without modifying yam_arena. The task runs with the
standard `yam_arena` embodiment and the `task_score` scorer. All the code is the task authors' own (MIT licence); no
third-party assets are used.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=drill_battery.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Take the battery pack off the drill." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/db-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode. To render the
trial's videos (one MP4 per camera), run `inspect-robots video ~/yam-eval/db-dream/<log>.json`.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=drill_battery.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Take the battery pack off the drill." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/db-demo
```

The demo is open loop: it uses only the fixed layout, never the simulator's internal state.

1. **The left hand lifts the drill.** It takes the drill's grip from above, fingers closing across it, and lifts the
   drill and its pack 5 cm. The target leads the lift by 4 cm, as the soft arm sags under 1.5 kg.
2. **The right hand takes the pack by its buttons.** It comes at the pack from the robot's side, nearly level (tipped
   5 deg), with one finger above and one below it, each on a release button. The level approach matters: tipped
   further, the fingers' up-down span is narrower than the 85 mm across the two buttons. The target is corrected by a
   few mm for where the lifted pack hangs, a fixed calibration from the layout.
3. **It squeezes both buttons in and slides the pack off** forwards along the foot, 11 cm, while the left hand holds
   the drill.
4. **The left hand lays the drill back down and lets go.** The right hand lowers the pack until its lower finger is
   just above the table and lets go; the pack drops off the finger and lies on the table. Both hands go home.

The demo reaches 1.0 (success) in 646 steps (32.3 s). No step of it is clamped by the run's delta-limit guard (the gripper may
change by at most 5% of its range per step). `test_drill_battery.py` checks every step against the guard and that
every pose along the arms' paths is reached.

**Shortcut check.** With `DB_DEMO=nosqueeze` the right hand does the same, but its fingers only touch the buttons
without pushing them in. The latch holds, the pack never moves along the rails, and the trial scores 0.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- **On the drill:** the pack still on the foot's rails (slid less than 7.5 cm along them, still in line with them).
- **Down:** lying on the tabletop, any way up (its lowest point within 3 mm of it; for the pack, its housing or a
  button cap).

| Score | Condition | How to check on a real table |
|---|---|---|
| 1.0 | the pack off the drill and down; the drill down within 15 cm of where it lay; no part of either gripper within 2 cm of either; both still (under 1 cm/s); held for 1 s | look: two separate parts lying on the table |
| 0.5 | the pack off the drill, anything else (still held or in the air, moving, the drill held up or carried more than 15 cm) | look; tape measure for the drill |
| 0.25 | the pack slid 1 cm or more along the rails but still on the drill | ruler at the foot |
| 0 | anything else (for example the drill lifted with the pack still on); and 0 from then on if the pack or the drill ever falls off the table (more than 2 cm below its top), even if put back | look |

Stillness is judged from the bodies' pose change between control steps (under 0.5 mm per step at 20 Hz). `score()`
gives 1.0 only once the full-score state has held for 1 s (until then 0.5), and `success()` returns True only then
(yam_arena then ends the trial after its own 1 s success hold). Offline checks of the parts, every tier and the demo:
`python -m pytest -q test_drill_battery.py` (19 tests, no simulator).

## Runtime and seeds

Measured on an L40S with the viewer on, Isaac Sim's start-up included:
- the demo trial (646 steps, 32.3 s of simulated time) took 2.2 minutes of wall-clock time;
- the Dream-1 trial (the full 2,400 steps, 2 minutes simulated; it scored 0.0) took 6.5 minutes.

The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

- **All contacts and forces, no rules.** The drill's foot has two rails that hook under the lips of the pack's tongue
  (3 mm wide, 4 mm deep). The pack can only slide along them, forwards off the foot.
- **The latch.** Each release button is a sprung slide in the pack's side (3 mm proud, 3 mm of travel; 6 N at rest,
  10.5 N fully in). Inside the pack it carries a pawl that rises beside the tongue, just behind a stop on the foot; the
  stop overlaps the resting pawl by 1.5 mm. The pack slides only while both buttons are pushed in 1.5 mm or more.
- **The terminals' grip** is a sprung shoe in the foot that presses on the tongue (6 N); it rubs there (friction 0.45)
  and pulls the rails up under the lips (0.5), about 6 N of sliding drag in all.
- **Our estimates:** the button springs, the shoe's preload and the frictions (no published figures; a pack slides off
  with a light pull once the buttons are in). Sizes and masses are Milwaukee's: the drill 5.7 in long, 2.2 in wide,
  2.3 lb; the pack 4.66 x 3.12 x 2.15 in, 0.95 lb. The inner layout of the foot and the pack's tongue is our
  simplification of the real slide interface.
- **Geometry.** Boxes (the chuck is a cylinder). The drill's handle, motor housing and foot are single boxes.

## Real-world build

- A Milwaukee 3601-20 M18 compact brushless drill/driver (bare tool) and a Milwaukee 48-11-1820 M18 CP2.0 pack.
  Use a spent pack, or one with its cells taken out and the case screwed shut, so no trial handles live cells.
- Lay the drill on its left side on the table, at the marks.

Positions are in `task.json`.

## Reset between trials

Slide the pack back onto the drill until it clicks, and lay the drill on its left side at the marks. Nothing is
consumed.
