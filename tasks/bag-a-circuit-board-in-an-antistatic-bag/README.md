# Open a flat static-shielding bag, slide a circuit board into it and put the filled bag in a tote

An empty static-shielding bag (SCS 10035, 3 x 5 in, open top) lies flat and shut on an ESD bench mat, its mouth
towards the robot. A Raspberry Pi Zero 2 W board lies beside it, and a conductive tote stands on the robot's left.
The robot must get the board into the bag and put the filled bag in the tote.

Under ESD control programs (ANSI/ESD S20.20), every static-sensitive board that leaves the protected bench is packed in
a static-shielding bag. Electronics assemblers, repair technicians and returns packers bag boards many times a shift.

A new bag lies flat with its films touching, so the mouth is shut: the board's leading edge only rides up onto it. One
hand must open the bag and keep it open: squeezing the bag's two side seams together stands its edges up and arches
the top film into an open mouth. The other hand slides the board in and pushes it home. Then the filled bag, a
floppy, light object, must be picked up and laid in the tote. The task tests two hands working together on a
deformable object: the bag opens only while it is squeezed, and only as far as it is squeezed.

## Files

| File | What |
|---|---|
| `bag_a_circuit_board.py` | The yam_arena task file: the mat, the bag's five-strip film loop, the board, the tote, `success()`, `score()` and the scripted `demo()` |
| `bagkit.py` | Our own helper: rigid bodies built from boxes, hinges (more than one on a body, for the bag's closed loop), and hooks into yam_arena's scene and reset |
| `test_bag_a_circuit_board.py` | Offline checks of the parts, every scoring tier and the demo's reach and per-step limits |

`bagkit.py` hooks `yam_arena.scene` and the embodiment's reset without modifying yam_arena. The task runs with the
standard `yam_arena` embodiment and the `task_score` scorer. All the code is the task authors' own (MIT licence); no
third-party assets are used.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=bag_a_circuit_board.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the circuit board into the antistatic bag, then put the bag in the tote." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/bag-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode. To render the
trial's videos (one MP4 per camera), run `inspect-robots video ~/yam-eval/bag-dream/<log>.json`.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=bag_a_circuit_board.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the circuit board into the antistatic bag, then put the bag in the tote." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/bag-demo
```

The demo is open loop: it uses only the fixed layout, never the simulator's internal state.

1. **The left hand opens the bag.** It comes down with its fingers open either side of the bag, 4.5 cm behind the
   mouth, and closes them on the side seams. The bag's edges stand up and its top film arches into a mouth about 4 cm
   high.
2. **The right hand slides the board in.** It takes the board by its trailing end (fingers tipped 45 deg forwards,
   closing across it), lifts it, lines it up with the bag and slides it in until its fingers reach the mouth.
3. **It pushes the board home.** It lets go, closes its fingers, tips them to 20 deg and pushes the board's trailing
   edge about 2 cm past the mouth.
4. **The left hand lets the bag close,** then takes the filled bag across its middle (the fingers close the seams on
   the board's edges through the films). It lifts the bag, carries it over the tote, lays it on the floor and lets go.

The demo reaches 1.0 (success) in 773 steps (38.7 s). No step of it is clamped by the run's delta-limit guard (the
gripper may change by at most 5% of its range per step). `test_bag_a_circuit_board.py` checks every step against the
guard and that every pose along the arms' paths is reached.

**Shortcut check.** With `BG_DEMO=noopen` the left hand stays home and the right hand does exactly the same. The
board's leading edge rides up onto the shut bag and pushes it 3 cm along the mat; the board never gets between the
films, and the trial scores 0.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`). The board's place
is measured in the bag's own frame (its bottom film's), wherever the bag lies.

| Score | Condition | How to check on a real table |
|---|---|---|
| 1.0 | the board wholly in the bag: between its two films, inside the 5 mm side seams, its trailing edge at least 5 mm past the mouth; **and** the whole bag inside the tote (between its inner walls, below its rim), lying on its floor (bottom film within 3 mm of it, tilted under 10 deg); no part of either gripper within 2 cm of the bag or the board; the bag and the board still (under 1 cm/s); held for 1 s | look through the bag's film or feel it; ruler from the mouth to the board's edge; look into the tote |
| 0.5 | the board wholly in the bag as above, the bag anywhere else (on the mat, still held, over a wall, above the floor, moving) | as above |
| 0.25 | the board's centre between the films and past the mouth (at least half of it in) | ruler |
| 0 | anything else (the board on top of, under or beside the bag); and 0 from then on if the board or the bag ever falls off the table (more than 2 cm below its top), even if put back | look |

"Between the films" is judged against the top film's actual shape: the board's centre must be above the bottom film
and below the top film's underside at that point, so a board lying on the shut bag is outside it. Stillness is
judged from the bodies' pose change between control steps (under 0.5 mm per step at 20 Hz).

`score()` gives 1.0 only once the full-score state has held for 1 s (until then 0.5), and `success()` returns True
only then (yam_arena then ends the trial after its own 1 s success hold). Offline checks of the parts, every tier and
the demo: `python -m pytest -q test_bag_a_circuit_board.py` (21 tests, no simulator).

## Runtime and seeds

Measured on an L40S with the viewer on, Isaac Sim's start-up included:
- the demo trial (773 steps, 38.7 s of simulated time) took 3.3 minutes of wall-clock time;
- the Dream-1 trial (the full 2,400 steps, 2 minutes simulated; it scored 0.0) took 8.3 minutes.

The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

- **All contacts and forces, no rules.** The bag is a closed loop of five film strips, each as long as the bag, hinged
  to one another along the bag's length. They are the bottom film's middle (50 mm), its two edges (13 mm each) and
  the top film's two halves (38 mm each); the side seams are the hinges between edge and top. Each hinge has a weak
  bending spring (2e-5 N m/rad), a 2.8 mil film's stiffness. Pushing the seams towards each other stands the edges
  up (up to 95 deg; they cannot fold under the bag) and arches the top film open, as when a real bag is squeezed at
  its sides. Let go, it sags shut. The board enters only between the films; the bottom seam's weld is a 4 mm ridge
  inside the closed end.
- **Simplifications:** the strips are rigid along the bag's length, so the bag opens along its whole length at once
  (a real bag opens most near where it is squeezed). They are 2 mm thick (thinner plates pass through fingertips in
  PhysX), and the bag weighs 6 g (a real one about 1.5 g: lighter strips are unstable). The film is drawn opaque.
- **Our estimates:** the frictions (film 0.4, mat 0.6, board 0.4, tote 0.5) and the spring. Sizes are the products':
  the bag 3 x 5 in (SCS 10035), the board 65 x 30 x 5.3 mm and 10 g (Raspberry Pi Zero 2 W), the tote 276 x 210 x
  89 mm outside and 233 x 167 x 76 mm inside (Quantum DG91035-CON).
- **Geometry.** Rigid bodies built from boxes; the board has its main packages and connectors as boxes.

## Real-world build

- An SCS 10035 Metal-In static-shielding bag, 3 x 5 in, open top (a pack of 100 is about $8), new and flat.
- A Raspberry Pi Zero 2 W without headers (about $15; any 65 x 30 mm board works).
- A Quantum DG91035-CON conductive grid container, no dividers (sold in packs of 20; one tote is enough).
- A dissipative rubber bench mat, cut to 30 x 36 cm and taped down.

Positions are in `task.json`.

## Reset between trials

Take the board out of the bag. Flatten the bag on the mat at its marks; replace a bag that has creased or lost its
shape (about $0.10). Lay the board at its marks and empty the tote. Nothing is consumed.
