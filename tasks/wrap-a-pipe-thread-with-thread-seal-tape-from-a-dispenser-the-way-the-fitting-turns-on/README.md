# Wrap a pipe thread with thread-seal tape from a dispenser, the way the fitting turns on

A 3/4-inch galvanised pipe nipple stands upright in a floor flange, its male thread at the top, and a handheld
dispenser of PTFE thread-seal tape lies on a mat beside it. The robot must wrap the thread with the tape, ready for a
fitting to be screwed on, and put the dispenser back.

Every taper-threaded pipe joint is sealed this way, and the standard is the same in every trade that does it: the
tape goes on the way the fitting will turn on (clockwise seen from the thread's end), three to six turns over the
thread, the first thread left bare so no shreds of tape get into the pipe, and the torn end pressed down. Wound the
other way, the tape bunches and unwinds as the fitting is screwed on, and the joint leaks: the classic beginner's
mistake. The instruction does not say any of this; the task tests whether the robot knows how a thread is taped and
can do it with the tool: a steady orbit round a small part at the right height and in the right sense, letting the
tape pay out under a light pull, a sharp yank to tear it, and the finish.

## Files

| File | What |
|---|---|
| `tape_wrap.py` | The yam_arena task file: objects, the tape model's set-up, `success()`, `score()` and the scripted `demo()` |
| `tapekit.py` | Our own helper: rigid bodies built from boxes, a braked hinge (the dispenser's spindle), the tape model (wound by the roll's path round the pipe, paid out as the roll turns, pulling and tearing; drawn by bodies it poses), and hooks into yam_arena's scene and reset |
| `test_tape_wrap.py` | Offline checks of the tape model, every scoring tier, and the demo's reach and per-step limits |

`tapekit.py` hooks `yam_arena.scene` and the embodiment's reset without modifying yam_arena, so the task runs with
the standard `yam_arena` embodiment and the `task_score` scorer. All the code is the task authors' own (MIT licence);
no third-party assets are used.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=tape_wrap.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Wrap the pipe's thread with thread-seal tape, ready for the fitting to be screwed on, then put the dispenser back on its mat." \
  --scorer task_score --max-steps 1600 --store-frames --log-dir ~/yam-eval/tape-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode. To render the
trial's videos (one MP4 per camera): `inspect-robots video ~/yam-eval/tape-dream/<log>.json`.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=tape_wrap.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Wrap the pipe's thread with thread-seal tape, ready for the fitting to be screwed on, then put the dispenser back on its mat." \
  --scorer task_score --max-steps 1600 --store-frames --log-dir ~/yam-eval/tape-demo
```

The demo is open loop: it uses only the fixed layout, never the simulator's internal state. One hand works at the
pipe at a time (with both there, the arms' wrists meet above it).

1. The right hand takes the dispenser by its handle, carries it over the pipe's end and touches the roll to the
   thread on the pipe's left side, 2.5 mm below the end, which sticks the tape's end there.
2. It carries the roll round the pipe four turns clockwise (seen from above), 45 mm out, spiralling down so the tape
   covers the thread down to its shoulder; the handle passes over the pipe's end, the first turn binds the tape's
   end, and the roll turns on its spindle as the tape comes off (pulling at 0.2 to 0.6 N).
3. It yanks the roll straight out from the pipe, faster than the braked roll can let the tape off, and the tape
   tears; it lays the dispenser back on its mat.
4. The left hand presses the torn tail down on the wrap.

The tape ends about 2 mm below the pipe's end and 2.5 mm below the thread's shoulder, four turns. The demo reaches
1.0 (success) in 609 steps (30 s). No step of it is clamped by the run's delta-limit guard (the gripper may change
by at most 5% of its range per step); `test_tape_wrap.py` checks every step against it, and checks that every pose
along both arms' paths is reached.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`). Clockwise means
seen from the thread's end (from above).

| Score | Condition | How to check on a real bench |
|---|---|---|
| 1.0 | wound clockwise 3 to 6 turns; covering every part of the thread from 1 mm above its shoulder to 4 mm below the pipe's end; no tape within 1 mm of the end; torn off the roll with its tail pressed down; the dispenser back upright on its mat, no part of either gripper within 1 cm of it, still (under 1 cm/s) for 1 s | count the turns; look at the direction and the end; ruler on the thread |
| 0.5 | wound and torn off as for 1.0, but the tail left standing, or the dispenser not back on its mat or not released | as above; look |
| 0.25 | wound clockwise at least one turn on the thread | look |
| 0 | wound anticlockwise, or less than a turn on the thread | |

Two scripted checks (`TAPE_DEMO=...` in the demo) show that the standard is scored, not just tape on a thread:

- `backwards`: the same wrap, anticlockwise: 0.
- `nopress`: the tail is left standing after the tear: 0.5.

The tape model's own checks (`test_tape_wrap.py`) include that a roll which cannot turn (pinched by its rim, say)
tears the tape in the first turn, and that a steady wrap with the roll free to turn does not.
Offline checks of the tape model, every tier and the demo: `python -m pytest -q test_tape_wrap.py` (33 tests, no
simulator).

## Runtime and seeds

Measured on an L40S with the viewer on, Isaac Sim's start-up included: the demo trial (609 steps, 30 s of simulated
time) took 2.9 minutes of wall-clock time; the Dream-1 trial (the full 1,600 steps, 80 s simulated) took 7.5
minutes. The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

- **The tape** is our own model (`tapekit.TapeWrap`); PhysX has no tape. When the roll touches the thread (its rim
  within 3 mm of it, level with the thread), the tape's end sticks there. From then on the tape is wound by the
  angle the roll travels round the pipe's axis, each part laid at the roll's height while the roll is level with
  the thread; carried back the other way, it unwinds.
- **Paying out and tearing.** The roll turns on the dispenser's spindle (a PhysX hinge with a brake, 0.0025 N m s/rad).
  Tape comes off only as the roll turns. What the wrap and the free run need beyond what has come off stretches the
  tape, which pulls on the roll's rim (300 N/m, along the run) and so turns it against the brake: a steady wrap pays
  out at 0.2 to 0.6 N. Stretched 5 mm (1.5 N) the tape tears, as PTFE tape tears by hand: a yank outruns the braked
  roll and tears it, and a roll that cannot turn tears it at once. Torn before half a turn has bound its end, it
  falls off. The torn tail stands out until a gripper presses it, within 8 mm. The tape does not wrinkle or neck,
  and its pull does not act on the pipe. It is drawn as white quarter-turn strips laid round the thread, the run
  from the pipe to the roll, and the tail.
- **Geometry.** Rigid bodies built from boxes: the pipe, flange, block and mat are fixed; the thread's grooves are
  seen only. The dispenser (60 g) and its roll (20 g, friction 0.8) are loose.
- **Arms.** The simulated arms are soft. The demo grips the handle firmly (the fingers closed to 4 mm on the 16 mm
  handle: held across it, the dispenser can pitch between the pads), and sends the roll 2.7 mm higher than it is
  meant to go, as the arm sags under the dispenser.

## Real-world build

A 3/4-inch BSP galvanised nipple screwed into a floor flange on a wooden block, fixed to the table; a handheld PTFE
tape dispenser with a 12 mm roll (any with a spindle and a handle; a brake spring is standard); a rubber mat.
Positions are in `task.json`.

## Reset between trials

Unwind the tape from the thread (or cut it off and wipe the thread), and lay the dispenser back on its mat; a roll
does about 100 trials.
