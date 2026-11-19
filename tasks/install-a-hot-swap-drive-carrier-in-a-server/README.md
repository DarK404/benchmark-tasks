# Install a hot-swap drive carrier in a 1U server: slide it in to the connector, then cam it home with its handle

A Dell PowerEdge R640 rack server (the 8 x 2.5 in drive chassis), unplugged, lies on the table with its front towards
the robot. Its eight drive bays, in two rows of four, all hold drive carriers but one. A hot-swap drive carrier with a
drive in it lies on the table beside the server, its handle open. The robot must install it in the empty bay and lock
its handle.

Replacing a failed drive is the commonest hands-on repair in a data center, and data centers are where the compute
for AI is built and kept running. Dell's service manual gives the procedure for a 2.5 in hot-swap carrier: with the
release handle open, slide the carrier into the bay until it contacts the backplane, then close the handle to lock the
carrier in place. Closing the handle is what seats the drive: its hook swings in behind the bay frame and draws the
carrier the last few millimetres onto the backplane's connector. A carrier only pushed in by hand stops against the
connector, unconnected, and the handle left open does not hold it.

The task tests a two-stage insertion in which the last stage is done with a lever: aligning a close-fitting carrier
with its bay and sliding it in square until it stops, then swinging the handle shut along its arc so its cam does
the mating, and pressing it home until its catch snaps.

## Files

| File | What |
|---|---|
| `drive_carrier.py` | The yam_arena task file: the server's front with its bay cage and backplane connector, the carrier with its connector, cam handle and catch, `success()`, `score()` and the scripted `demo()` |
| `carrierkit.py` | Our own helper: rigid bodies built from boxes (turned boxes for sloped faces), hinges with springs and limits, sprung slides (the connector's contacts, the catch), kinematic bodies, and hooks into yam_arena's scene and reset |
| `test_drive_carrier.py` | Offline checks of the parts' geometry and forces (the connector's mating force, the cam's path, the catch), every scoring tier, and the demo's reach and per-step limits |

`carrierkit.py` hooks `yam_arena.scene` and the embodiment's reset without modifying yam_arena. The task runs with
the standard `yam_arena` embodiment and the `task_score` scorer. All the code is the task authors' own (MIT licence);
no third-party assets are used.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=drive_carrier.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Install the drive carrier in the server's empty drive bay and lock its handle." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/dc-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode. To render the
trial's videos (one MP4 per camera), run `inspect-robots video ~/yam-eval/dc-dream/<log>.json`.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=drive_carrier.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Install the drive carrier in the server's empty drive bay and lock its handle." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/dc-demo
```

The demo is open loop: it uses only the fixed layout, never the simulator's internal state. The right hand does it
all; the left stays home.

1. **Take the carrier** across its sides near its middle, lift it and carry it level in line with the empty bay.
2. **Slide it in** until the fingers are just short of the bay frame's posts, and let go (its front end tips down onto
   the frame).
3. **Push it in** with the closed fingertips on its front until it stops against the backplane's connector, 6 mm short
   of home.
4. **Close the handle**: from beside it, push it round its arc near its middle. Its hook swings in behind the frame's
   lip and draws the carrier home onto the connector; at the end the handle's free end lifts the catch, which drops in
   front of it. Then the hand backs off and goes home.

The demo reaches 1.0 (success) in 723 steps (36.2 s). No step of it is clamped by the run's delta-limit guard
(the gripper may change by at most 5% of its range per step); `test_drive_carrier.py` checks every step against it
and that every pose along the arm's path is reached.

**Shortcut check.** With `DC_DEMO=push` the demo pushes the carrier in by hand as hard as it can (aiming 20 cm into
the server) and never touches the handle: the carrier stops on the connector's lead, unmated, and scores 0.25.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`). Measured on the
carrier against its home position in the empty bay (its front 1 mm behind the chassis's front).

| Score | Condition | How to check on a real table |
|---|---|---|
| 1.0 | home in the empty bay (within 0.7 mm along it, so the connector is fully mated; within 1.5 mm across it; square within 2 degrees); its handle shut (within 3 degrees) under its catch (the catch dropped back within 0.5 mm of rest); no part of either gripper within 1 cm of the carrier; the carrier and handle still; held for 1 s | its front flush with its neighbours' (a straightedge across the fronts, a 0.7 mm feeler does not pass); the handle stays shut when its end is pulled gently |
| 0.5 | home, the handle not locked (or a hand on it, or not yet still) | the handle swings open |
| 0.25 | in the bay with its connector on the blade, not home | its front stands proud of its neighbours' by up to 6 mm |
| 0 | anything else (for example short of the connector, or out of the bay) | look |

`score()` gives 1.0 only once the full-score state has held for 1 s (until then 0.5), and `success()` returns True
only then (yam_arena then ends the trial after its own 1 s success hold). Offline checks of the parts, every tier and the demo:
`python -m pytest -q test_drive_carrier.py` (27 tests, no simulator).

## Runtime and seeds

Measured on an L40S with the viewer on, Isaac Sim's start-up included:
- the demo trial (723 steps, 36.2 s of simulated time) took 2.2 minutes of wall-clock time;
- the Dream-1 trial (the full 1,200 steps, 1 minute simulated; it scored 0.0) took 3.3 minutes.

The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

- **Everything is contact and forces, with no rules.** The bay is a cage of collision boxes: floor, shelf and cover
  with 2.5 mm lead-ins at the front, and frame posts between the bays with 1.5 mm lead-ins. Its 71 x 18 mm bays leave
  the 70 x 15 mm carrier 0.5 mm of play a side.
- **The connector.** The backplane's blade (2.4 mm thick, its tip tapered at 15 degrees) enters a slot in the
  carrier's rear between two contact shoes. Each shoe is on a sprung slide (`carrierkit.Spring`: a PhysX joint with a
  preloaded linear drive; 6 N preload, 5000 N/m) and sits 0.6 mm from the other. The blade spreads the shoes to 10.5 N
  each, with friction 0.3. Mating therefore takes up to about 13 N up the lead and 6.3 N along the flat. SFF-8482, the
  SAS drive-connector standard, allows at most 25 N to mate and requires at least 5 N to unmate.
- **The handle** turns on a hinge (`carrierkit.Hinge`) at the carrier front's corner, limited to 90 degrees and opened
  by a light spring (0.01 N m when shut). An 8 mm hook on the pin's far side swings through a notch in the frame post
  and bears on the back of the post's lip. Closing the handle draws the carrier 6 mm in, with a leverage of about 5:1
  at the point the demo pushes. Left 10 mm or more short, the hook strikes the lip and the handle will not shut.
- **The catch** is a sprung tab on the carrier's front (0.5 N preload, 300 N/m). The handle's end rides up its 30-degree
  slope and passes under its nose; the nose then drops in front of the handle's end and holds it shut against the
  handle's spring.
- **Our estimates:** the inner shapes of the bay, carrier, handle, hook, catch and connector (Dell publishes no
  drawings), the carrier's 0.17 kg, the springs, and the frictions (0.15 on the bay's rails, 0.4 carrier, 0.3 handle
  and contacts). The server's outside size is Dell's.
- **Not modelled:** the blue release button (drawn; it is not used to install); the drive's electronics.
- **Geometry.** Rigid bodies built from boxes; the server is fixed (it lies on the table). Only its front 26 cm, the
  drive cage and backplane, is modelled.

## Real-world build

- A Dell PowerEdge R640 with the 8 x 2.5 in drive chassis (434 x 42.8 x 758 mm), unplugged and lying on the table, its
  front frame posts at x 19 cm; its rear stands about 4 cm past the table's far edge, or use a bare R640 front drive
  cage with its backplane.
- Seven Dell 2.5 in hot-swap carriers in its bays, and an eighth with a drive in it for the task (the upper row's
  second bay from the robot's right is left empty).

Positions are in `task.json`.

## Reset between trials

Press the carrier's blue release button so the handle springs open, pull the carrier out by its handle, and lay it
back at the marked spot on the table, front towards the robot, handle open. Nothing is consumed.
