# Lever a paint tin's lid off with a screwdriver

A round steel paint tin stands on the table with its lid pressed into the rim. The lid sits 2 mm below the rim's
rolled edge, so there is no edge to take hold of; between the lid and the rolled edge is a narrow groove. A flat-blade
screwdriver stands tip down in a wooden block. The robot must pick up the screwdriver, put its blade into the groove
and under the lid's edge, and lever the lid up by pivoting the blade on the rim. One pry frees only part of the seal,
and one arm can pry only a short arc of the rim, so the other hand turns the tin between pries, as a person does.
Once the lid is loose all round, the robot lifts it off and puts it on the table.

Painters and decorators open tins this way many times a day. The task tests using a hand tool as a lever: placing a
thin blade precisely in a narrow gap, making its own pivot on the rim, and working round a press-fit seal over
several pries with both hands taking turns.

## Files

| File | What |
|---|---|
| `paint_tin.py` | The yam_arena task file: objects, the seal, `success()`, `score()` and the scripted `demo()` |
| `tinkit.py` | Our own helper: rigid bodies built from boxes and cylinders, the press-fit seal force model, and hooks into yam_arena's scene and reset |
| `test_paint_tin.py` | Offline checks of the scorer on synthetic states, and of the demo's reach |

`tinkit.py` hooks `yam_arena.scene` and the embodiment's reset without modifying yam_arena, so the task runs with the
standard `yam_arena` embodiment and the `task_score` scorer. All the code is the task authors' own (MIT licence); no
third-party assets are used.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=paint_tin.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Open the paint tin: lever its lid off with the screwdriver and put the lid on the table." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/paint-tin-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=paint_tin.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Open the paint tin: lever its lid off with the screwdriver and put the lid on the table." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/paint-tin-demo
```

The demo is open loop: it uses only the fixed layout, never the simulator's internal state.

1. The right arm takes the screwdriver from its stand in a pen grip, puts the blade into the groove at the tin's
   back right, under the lid's edge, and levers it from 60 to 10 degrees about the rim. Eight of the 12 catches let go.
2. It draws the blade out and parks the tool. The left arm grips the tin's body, turns it 150 degrees on the table and
   lets go.
3. The right arm pries again at the same spot, which is now the other side of the lid, and frees the rest of the seal.
   It then eases the blade down 4 degrees so the lid settles level on the rim.
4. The left hand pinches the loose lid by two opposite flats, gently at first. The blade is drawn out only once the
   lid is held: pulled out from under a loose lid, it dragged the lid onto the rim.
5. The left hand lifts the lid 6 cm, grips it firmly, carries it 15 cm to the left of the tin, sets it down flat and
   lets go.

The demo reaches 1.0 (success) in 1,030 steps (51.5 s). No step of it is clamped by the run's delta-limit guard
(the gripper may change by at most 5% of its range per step); `test_paint_tin.py` checks every step against it.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

| Score | Condition | How to check on a real table |
|---|---|---|
| 0 | at any point: the tin tilted more than 15 degrees, moved more than 3 cm from its start, or the tin or the lid fell off the table (more than 2 cm below its top); or the screwdriver's blade never went into the groove between the lid and the rim | video |
| 1.0 | the lid lies flat on the table (its lowest point within 5 mm of it, tilted under 10 degrees, either face up), its edge at least 5 mm clear of the tin, no part of either gripper within 2 cm of it, and the lid and the tin still (moving under 1 cm/s) for 1 s | ruler, top photo |
| 0.5 | the lid is off the tin (its centre more than 4 cm from the tin's axis) but not as for 1.0: propped against the tin, tilted on the table, or still held | ruler, photo |
| 0.25 | the lid is still on the tin, but lifted out of the rim round at least half its edge (in simulation: 6 of the 12 seal catches freed) | 1 mm feeler gauge under the lid's edge |
| 0 | anything else | |

The violations latch: a tin tipped over and stood back up, or pushed away and back, still scores 0. The groove check
makes the lid have to be levered, not pulled or knocked off; the blade passing through the groove on its way under
the lid's edge is what it records. A scripted check without the tool (`TIN_DEMO=pinch`: the left gripper reaches into
the groove, closes on the sealed lid and pulls up) lifted and tipped the whole tin by 20 degrees while the lid stayed
sealed: score 0.

Offline checks of every tier, latch and edge case: `python -m pytest -q test_paint_tin.py` (39 tests, no simulator).

## Runtime and seeds

Measured on an L40S. The simulation runs about 3.5 times slower than real time (about 6 control steps a second:
each step has 12 physics substeps with 32 solver iterations for the thin parts, the seal model and three cameras).
The demo trial (1,030 steps, 51.5 s of simulated time) took 3.3 minutes of wall-clock time, about 15 s of it Isaac
Sim's start-up; the Dream-1 trial (the full 2,400 steps, 2 minutes simulated) took 7.0 minutes. The layout is fixed (`-E object_jitter=0`); there are no seeds to choose. The demo reached success at the same
step in a development harness and under `inspect-robots run`.

## Fidelity

- **Geometry.** Rigid bodies made of boxes. The lid, the rim's rings and the rolled edge are 12- and 24-sided prisms,
  because PhysX's custom cylinders missed the blade's thin contacts. The lid's rolled edge is visual only: as a
  collision shape it stood in the blade's way.
- **The press-fit seal** is a force model (`tinkit.Seal`). Twelve catches spaced round the lid each pull it down with
  0.5 N until the lid's edge there has risen 0.8 mm, then let go for good. A real lid needs tens of newtons, which the
  screwdriver's leverage supplies; in simulation a stronger seal could not be pried at all.
- **The freed lid.** Once every catch has let go, the lid rides 4.5 mm up on the rim, as a real lid levered out of its
  channel sits on top of it, and a weak centring pull (100 N/m, at most 0.3 N) stands in for its plug resting in the
  channel. Without the pull, the loose lid slid 16 mm across the rim. All seal forces stop once the lid is more than
  12 mm off the tin's axis.
- **Clearances.** The groove is 7.5 mm wide where a real one is 2-3 mm, so the open-loop blade path has 2 mm margins
  against the arm's tracking error.
- Friction: lid 0.5, tin 0.6, screwdriver 0.9, stand 0.6. No contact-force cap is enforced in simulation.

## Real-world build

A 450-500 ml round steel paint tin with a press-fit lid (empty ones are sold for craft use), filled to about 600 g, a
6 mm flat-blade screwdriver, and a 40 x 40 x 60 mm beech block with an 8 mm hole fixed to the table. Positions are in
`task.json`. Real tins have a narrower groove and a stronger seal than the simulation.

## Reset between trials

Press the lid back into the rim all round with the heel of the hand until it is flush, stand the tin on its marked
circle, and put the screwdriver back tip down in the stand. Nothing is consumed.
