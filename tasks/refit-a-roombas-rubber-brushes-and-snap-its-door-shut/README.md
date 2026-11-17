# Refit a Roomba's two rubber brushes, each peg in its matching drive socket, and snap the cleaning head's door shut

An iRobot Roomba lies upside down on the table, its cleaning head's door open and its two rubber brushes out, lying
on the table either side of it. One brush has a square peg on its drive end, the other a hex peg; both have a grey
bearing cap on their other end. The robot must refit both brushes and close the door.

Robot vacuums are in tens of millions of homes, and their makers tell owners to take the brushes out and clear the
hair from them regularly. iRobot's care steps for the Roomba e, i and j series: remove the brushes, clear the hair
from their square and hex pegs and the grey bearings, then put them back, "match the shape of the multi-surface
brushes pegs with the shape on the Cleaning Head Module", and close the cleaning head's door. A brush whose peg is
not in its socket does not turn; an unlatched door lets the brushes fall out.

The task tests telling the two brushes apart by their pegs and fitting each the way it goes: its peg end first,
tipped up, slid through the drive socket, then let down so its bearing cap drops into its cradle; then swinging the
door over and pressing it until both catches snap.

## Files

| File | What |
|---|---|
| `roomba_brushes.py` | The yam_arena task file: the robot's cleaning head, the brushes, the door and its catches, `success()`, `score()` and the scripted `demo()` |
| `brushkit.py` | Our own helper: rigid bodies built from boxes (turned boxes for the round parts, the hex shapes and the sloped faces), hinges, sprung slides (the catches), kinematic bodies, and hooks into yam_arena's scene and reset |
| `test_roomba_brushes.py` | Offline checks of the parts' geometry (each peg fits only its own socket), every scoring tier, and the demo's reach and per-step limits |

`brushkit.py` hooks `yam_arena.scene` and the embodiment's reset without modifying yam_arena. The task runs with the
standard `yam_arena` embodiment and the `task_score` scorer. All the code is the task authors' own (MIT licence); no
third-party assets are used.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=roomba_brushes.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Refit both brushes in the robot vacuum's cleaning head and close its door." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/rb-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode. To render the
trial's videos (one MP4 per camera), run `inspect-robots video ~/yam-eval/rb-dream/<log>.json`.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=roomba_brushes.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Refit both brushes in the robot vacuum's cleaning head and close its door." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/rb-demo
```

The demo is open loop: it uses only the fixed layout, never the simulator's internal state.

1. **The square-peg brush** (right hand): taken across its middle, lifted, tipped 10 degrees bearing end up, lined up
   with the front seat's square socket, slid in peg first through the socket's sloped mouth, and let go: it drops
   level, its bearing cap into its cradle.
2. **The hex-peg brush** (left hand): the same into the rear seat's hex socket.
3. **The bearing ends** (right hand): each pressed down into its cradle with the closed fingertips.
4. **The door** (right hand): pushed from behind over its upright point; the hand moves out of its swing while it
   falls shut; then its front edge is pressed down until both catches ride over the frame's lip and spring under it.

The demo reaches 1.0 (success) in 1,075 steps (53.8 s). No step of it is clamped by the run's delta-limit guard
(the gripper may change by at most 5% of its range per step); `test_roomba_brushes.py` checks every step against it
and that every pose along the arms' paths is reached.

**Shortcut check.** With `RB_DEMO=swap` the demo offers each brush to the other seat: neither peg goes through the
other's socket, neither brush is seated, and it scores 0.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`). A brush is seated
when its peg is through the matching drive socket and its bearing cap is down in its cradle: within 1.5 mm across its
axis and 2 degrees of seated, and between 5 mm short of the drive's end stop and home along it (its axial play).

| Score | Condition | How to check on a real table |
|---|---|---|
| 1.0 | both brushes seated in the seats that match their pegs; the door shut (within 2 degrees) with both catches sprung home under the frame's lip; no part of either gripper within 1 cm of the brushes or the door; everything still; held for 1 s | each brush turns freely by hand only with its drive; the door does not lift when its edge is pulled gently |
| 0.5 | both brushes seated, the door not latched (or a hand on it, or not yet still) | the door lifts |
| 0.25 | one brush seated | look; the other brush lies tilted or out |
| 0 | anything else (for example a brush in the wrong seat, which its peg does not fit) | look |

`score()` gives 1.0 only once the full-score state has held for 1 s (until then 0.5). Offline checks of the parts,
every tier and the demo: `python -m pytest -q test_roomba_brushes.py` (21 tests, no simulator).

## Runtime and seeds

Measured on an L40S with the viewer on, Isaac Sim's start-up included:
- the demo trial (1,075 steps, 53.8 s of simulated time) took 3.3 minutes of wall-clock time;
- the Dream-1 trial (the full 1,800 steps, 1.5 minutes simulated; it scored 0.0) took 4.8 minutes.

The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

- **Everything is contact, with no rules.** Each drive socket is a plate with a square or hex hole (0.8 / 1.2 mm of
  play round its own peg) and a sloped mouth in front of it; behind it, the drive's end stop. The other brush's peg
  does not fit (the hex peg needs a square hole at least 12.3 mm across, the square socket is 11.6 mm; the largest
  square that passes the hex socket is 9.8 mm, the square peg is 10 mm). The bearing cradles are open-topped notches
  the caps drop into. A seated brush can slide up to about 5 mm along its axis, as a real one can, with its peg still
  through the socket.
- **The door** turns on a hinge along the opening's rear edge and rests open on its stop at 110 degrees. Its two
  catches are sprung hooks (`brushkit.Spring`: a PhysX joint with a preloaded linear drive, 1 N, 1000 N/m): shutting
  the door, each rides down the frame's sloped lip, is pushed back, and springs forward under the lip's square
  underside, which then holds the door shut.
- **Our estimates:** the cleaning head's inner shapes, the peg sizes and the socket play, the door's size and catches
  (iRobot publishes no drawings), the brushes' 60 g, and the frictions (0.8 rubber, 0.5 plastic). The robot's
  outside size is the maker's.
- **Not modelled:** the brushes' flexible rubber fins (drawn on a rigid brush); the side brush; the motors.
- **Geometry.** Rigid bodies built from boxes; the robot is fixed (it lies on the table).

## Real-world build

- An iRobot Roomba i3 (or another e, i or j-series Roomba), upside down on the table, its cleaning head's door open.
- Its two dual multi-surface rubber brushes, taken out and laid beside it.

Positions are in `task.json`.

## Reset between trials

Open the door (press its tab to release the catches and lift it back to its stop), take both brushes out (lift each
bearing end, draw the peg out of its socket) and lay them back on the table at the marked spots, drive ends towards
+y. Nothing is consumed.
