# Pull a Christmas cracker with both hands until it cracks and parts, then put the pieces down

A Christmas cracker (a Tom Smith Traditional 12 in cracker) lies across the table in front of the robot. The robot must pull it,
one end in each hand, until it cracks and comes apart, then put the pieces down on the table.

Pulling crackers is part of Christmas dinner in Britain and much of the Commonwealth: about 150 million are pulled in
the UK each Christmas (a consumer survey widely reported in the UK press). Two people, or one person with both
hands, each take an end and pull until a crimp tears and the snap goes off; whoever keeps the middle gets the hat and
the gift.

The task tests a two-handed pull on one light, hollow object. Each hand must hold its end firmly enough not to slip
while the pull builds to the crimp's tearing force, and both must pull against each other along the cracker's
length: one hand alone only drags it about.

## Files

| File | What |
|---|---|
| `christmas_cracker.py` | The yam_arena task file: the cracker's three pieces, its crimps, the gift inside, `success()`, `score()` and the scripted `demo()` |
| `crackerkit.py` | Our own helper: rigid bodies built from boxes, breakable joints (the crimps), and hooks into yam_arena's scene and reset |
| `test_christmas_cracker.py` | Offline checks of the parts, every scoring tier and the demo's reach and per-step limits |

`crackerkit.py` hooks `yam_arena.scene` and the embodiment's reset without modifying yam_arena. The task runs with
the standard `yam_arena` embodiment and the `task_score` scorer. All the code is the task authors' own (MIT licence);
no third-party assets are used.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=christmas_cracker.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Pull the Christmas cracker with both hands until it cracks, then put the pieces down." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/xc-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode. To render the
trial's videos (one MP4 per camera), run `inspect-robots video ~/yam-eval/xc-dream/<log>.json`.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=christmas_cracker.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Pull the Christmas cracker with both hands until it cracks, then put the pieces down." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/xc-demo
```

The demo is open loop: it uses only the fixed layout, never the simulator's internal state.

1. **Each hand takes one end** from above, the fingers closing firmly across its tube.
2. **Both lift it** together, 6 cm off the table.
3. **Both pull outwards** along it at the same time, aimed 8 cm further out than they are (the soft simulated arms pull
   until the weaker crimp tears and the cracker parts).
4. **Each hand puts its piece down** on the table and lets go, then goes home.

The demo reaches 1.0 (success) in 312 steps (15.6 s). No step of it is clamped by the run's delta-limit
guard (the gripper may change by at most 5% of its range per step); `test_christmas_cracker.py` checks every step
against it and that every pose along the arms' paths is reached.

**Shortcut check.** With `XC_DEMO=onehand` only the right hand grips and pulls: nothing holds the other end, the
cracker is dragged whole, and it scores 0.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

| Score | Condition | How to check on a real table |
|---|---|---|
| 1.0 | cracked (an end torn off the middle at its crimp, the crimp more than 10 mm apart); every piece lying on the tabletop, inside its edges (not held up, not off the table or on the floor); no part of either gripper within 2 cm of any piece; all still; held for 1 s | look: two pieces on the table, a torn crimp |
| 0.5 | cracked, the pieces not yet down and let go (or still moving) | look |
| 0.25 | held up off the table by both ends (both more than 3 cm up), not cracked | look |
| 0 | anything else (for example still whole on the table, or dragged whole by one hand) | look |

`score()` gives 1.0 only once the full-score state has held for 1 s (until then 0.5), and `success()` returns True
only then (yam_arena then ends the trial after its own 1 s success hold). Offline checks of the parts, every tier and
the demo: `python -m pytest -q test_christmas_cracker.py` (17 tests, no simulator).

## Runtime and seeds

Measured on an L40S with the viewer on, Isaac Sim's start-up included:
- the demo trial (312 steps, 15.6 s of simulated time) took 1.3 minutes of wall-clock time;
- the Dream-1 trial (the full 1,200 steps, 1 minute simulated; it scored 0.0) took 3.3 minutes.

The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

- **The crimps tear by force, with no rules.** The cracker is three rigid pieces: the middle (a hollow card tube) and
  the two ends (each a crimped neck and an end tube). Each end is joined to the middle by a PhysX fixed joint that
  breaks when its force passes the crimp's tearing force (20 N on one side, 18 N on the other: one crimp always gives
  first) or its bending moment passes 0.25 N m. The arms must pull that hard through their grip. The gift, a small
  boxed toy, lies loose in the middle.
- **Our estimates:** the tearing forces (no maker or study publishes a cracker's pull-apart force; 18-20 N is a firm
  two-handed tug, the snap strip's friction included), the masses (40 g in all, about what a 12 in cracker weighs),
  and the frictions (0.7 paper, 0.5 the gift). The length is the product's (12 in, 30 cm); its 5 cm
  diameter and 2.4 cm crimps are our estimates. A real trial can check the tearing force by pulling
  one cracker from the same box apart with a luggage scale.
- **Not modelled:** the paper's give (the tubes are rigid, so it cannot be crushed); the snap's bang; the paper hat.
- **Geometry.** Rigid bodies built from boxes (12-sided prisms for the tubes).

## Real-world build

- One cracker from a box of Tom Smith Traditional 12 in Christmas crackers (FSC, plastic-free; all trials from one
  product, ideally one batch), lying across the table at the marked spot. A new cracker each trial (crackers are
  consumed: about £1 each).

Positions are in `task.json`.

## Reset between trials

Clear away the pieces, the hat, the joke and the gift, and lay a new cracker across the table at the marked spot.
