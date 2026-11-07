# Take a glass pane out of a sash with two suction lifters and lay it on bearers

A window sash lies flat on the bench with a glass pane in its rebate. The pane sits 1 mm below the frame's top with
a 2 mm gap all round, so there is no edge to take hold of, and glass edges cut. Beside each arm lies a hand suction
lifter: a rubber cup with a handle across its back and a lever beside the handle. The robot must set each lifter's
cup flat on the glass, fully inside its edges, and turn its lever down to lock it; then lift the pane out with both
arms together, lay it flat on two felt bearers, and lift both levers to unlock the lifters.

Glaziers move every pane this way, and tile setters and worktop fitters use the same lifters on tiles and stone.
The task tests attaching a hand tool to a part the gripper cannot hold, operating the tool's lock, and carrying one
rigid part with two arms moving together.

## Files

| File | What |
|---|---|
| `glass_pane.py` | The yam_arena task file: objects, hinges, suction, `success()`, `score()` and the scripted `demo()` |
| `glazekit.py` | Our own helper: rigid bodies built from boxes and cylinders, hinged levers (PhysX revolute joints) with over-centre springs, the suction model, and hooks into yam_arena's scene and reset |
| `test_glass_pane.py` | Offline checks of the scorer, the suction and hinge models, and the demo's reach and per-step limits |

`glazekit.py` hooks `yam_arena.scene` and the embodiment's reset without modifying yam_arena, so the task runs with
the standard `yam_arena` embodiment and the `task_score` scorer. All the code is the task authors' own (MIT
licence); no third-party assets are used.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=glass_pane.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Take the glass pane out of the window sash with the two suction lifters, lay it flat on the two felt bearers, then unlock the lifters." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/glass-pane-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=glass_pane.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Take the glass pane out of the window sash with the two suction lifters, lay it flat on the two felt bearers, then unlock the lifters." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/glass-pane-demo
```

The demo is open loop: it uses only the fixed layout, never the simulator's internal state. Both arms do the same,
mirrored:

1. Take the lifter by its handle, carry it over the pane and set the cup on the glass, 12 cm from the middle.
2. Let go, push the lever down with the closed fingertips (the cup seals), and take the handle again.
3. Together: lift the pane 5 cm, draw it 22 cm back over the bearers and lay it on them.
4. Let go of the handle, pinch the end of the lever and lift it past its catch: it snaps open and the cup lets go.
   Move away.

The demo reaches 1.0 (success) in 801 steps (40 s). No step of it is clamped by the run's delta-limit guard (the
gripper may change by at most 5% of its range per step); `test_glass_pane.py` checks every step against it.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

| Score | Condition | How to check on a real table |
|---|---|---|
| 0 | at any point: the pane dropped (fell faster than 0.3 m/s, or lower than its felt bed), tilted more than 20 degrees once off the sash, or was touched by a gripper | video |
| 1.0 | after a lift 1 cm clear of the sash made with both lifters locked on it, the pane lies flat on both bearers (tilted under 3 degrees, underside within 3 mm of their tops, centre within 2 cm of the point midway between them), both levers up 45 degrees or more (unlocked), no part of either gripper within 2 cm of the glass or 1 cm of a lifter, all still (under 1 cm/s) for 1 s | ruler, levers, photo |
| 0.5 | the pane was lifted 1 cm clear of the sash with both lifters locked on it, but not as for 1.0 | video |
| 0.25 | a lifter was locked on the pane at some point (its lever down with the cup sealed) | video, levers |
| 0 | anything else | |

The violations latch: a pane dropped or tipped and then put right still scores 0. The lift counts only with both
lifters locked, as the instruction asks: a glazier carries a pane this long with a cup near each end. The pane cannot
be taken without the lifters: in a scripted check (`GLASS_DEMO=edge`: the left gripper aims its fingertips into the
2 mm gap at the pane's end, squeezes and pulls up) the pane did not move, and the touch latched a score of 0.

Offline checks of every tier, latch and edge case, and of the suction rules: `python -m pytest -q test_glass_pane.py`
(35 tests, no simulator).

## Runtime and seeds

Measured on an L40S, Isaac Sim's start-up included: the demo trial (801 steps, 40 s of simulated time) took 3.4
minutes of wall-clock time; the Dream-1 trial (the full 2,400 steps, 2 minutes simulated) took 11.2 minutes. The
simulation runs slower than real time: each control step has 12 physics substeps, with 32 solver iterations for
the thin pane and the lifters, the suction and lever models, and three cameras. The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

- **Geometry.** Rigid bodies made of boxes; the cups and their hubs are 12-sided prisms, because PhysX's custom
  cylinders miss thin contacts.
- **The levers** turn on PhysX revolute joints (0 to 65 degrees) with an over-centre spring (`glazekit.Detent`):
  below 25 degrees it turns the lever shut, above it open, as a real lever's cam does.
- **The suction** is a force model (`glazekit.Suction`). When a lever is turned down while its cup lies flat on the
  glass (tilted under 3 degrees, its face within 3 mm of the glass) with its rim at least 5 mm inside every edge,
  three stiff springs (4,000 N/m) at the cup's face tie it to the points of the glass it sealed on. If any spring
  must pull harder than 40 N, the cup pops off. Turned down anywhere else, it does not seal, and the lever must be
  lifted again first. Real cups hold 30 kg or more; the limit only has to exceed what the arms apply.
- **The glass** is rigid: it does not flex or break.
- Friction: glass 0.5, sash 0.6, bearers 0.7, lifters and levers 0.9. No contact-force cap is enforced in simulation.

## Real-world build

A sash frame (or any flat frame with a 4 cm deep rebate) with felt in the rebate, a 400 x 200 x 4 mm toughened
(safety) glass pane with arrised edges, two single-cup lever suction lifters with 115 mm cups (sold for glaziers and
tile setters), and two felt-topped pine battens fixed to the table. Positions are in `task.json`.

## Reset between trials

Unlock and remove both lifters, lay the pane back on the felt in the rebate, and set the lifters face down at their
marks with their levers up. Nothing is consumed.
