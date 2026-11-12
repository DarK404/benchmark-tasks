# Make up a compression joint: pipe home, olive before nut, nut a turn past snug with a spanner

A ball valve's 15 mm compression inlet faces up from its mounting block. A short copper pipe tail stands in a holder,
the olive and the compression nut lie in a parts tray, and a 22 mm open-ended spanner lies in its tray. The robot must
join the pipe to the valve with the compression fitting.

Compression fittings join copper pipe to valves, taps and appliances in every building, and making one up has a
standard every plumber learns:
- **Pipe home.** The pipe goes fully home into the fitting's socket, to its stop.
- **Olive first.** The olive goes on before the nut, so that it sits between the nut and the fitting. A forgotten
  olive, or one put on outside the nut, gives a joint that leaks.
- **The right turn.** The nut is run on and turned about a turn past snug: ¾ to 1¼ turns. Much further, the olive
  and the pipe are crushed, and the joint has to be cut out.

The task tests making up a joint in the only order that seals and tightening it by the right amount, with the two
hands doing different jobs: one places the parts, the other swings the spanner in short strokes, sliding it off and
on again each time.

## Files

| File | What |
|---|---|
| `compression_joint.py` | The yam_arena task file: objects, the joint model's set-up, `success()`, `score()` and the scripted `demo()` |
| `fittingkit.py` | Our own helper: rigid bodies built from boxes, the compression-joint model (the socket, the rings on the pipe, the nut's thread, the spanner on the nut), and hooks into yam_arena's scene and reset |
| `test_compression_joint.py` | Offline checks of the model, every scoring tier and the latch, and the demo's reach and per-step limits |

`fittingkit.py` hooks `yam_arena.scene` and the embodiment's reset without modifying yam_arena. The task runs with the
standard `yam_arena` embodiment and the `task_score` scorer. All the code is the task authors' own (MIT licence); no
third-party assets are used.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=compression_joint.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Join the copper pipe to the valve with the compression fitting." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/joint-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode. To render the
trial's videos (one MP4 per camera), run `inspect-robots video ~/yam-eval/joint-dream/<log>.json`.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=compression_joint.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Join the copper pipe to the valve with the compression fitting." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/joint-demo
```

The demo is open loop: it uses only the fixed layout, never the simulator's internal state.

1. **Pipe in (left hand).** The left hand lifts the pipe tail out of its holder by its upper end and stands it in
   the valve's inlet, home on the socket's stop.
2. **Olive, then nut (left hand).** It takes the olive from the tray and lowers it over the pipe's top; let go, the
   olive slides down into the stub's cone. Then it does the same with the nut, which comes to rest on the olive.
3. **Tighten (right hand).** The right hand has taken the spanner from its tray and waits clear of the valve until
   the left hand has gone home. Then it slides the jaw onto the nut's flats and swings the handle clockwise from -60
   to -120 degrees. It slides the spanner off along its handle, swings back and slides it on again: 12 strokes.
   After 6 strokes (1 turn) the nut is snug on the olive; the next 6 take it about a turn past.
4. **Tidy up.** The spanner goes back in its tray.

The demo reaches 1.0 (success) in 1,913 steps (95.7 s), with the nut about a turn past snug. No step of
the demo is clamped by the run's delta-limit guard, which lets the gripper change by at most 5% of its range per step.
`test_compression_joint.py` checks every step against it, and checks that every pose along the arms' paths is
reached.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

| Score | Condition | How to check on a real bench |
|---|---|---|
| 0 (latched) | at any point the nut was turned more than 1½ turns past snug (the olive and the pipe crushed) | the snug marks, counted on the video (see below) |
| 0 | at any point the nut was turned more than 1¼ turns past snug (the olive over-compressed; backing the nut off does not undo it) | as above |
| 1.0 | the joint sealed: the pipe home (its end within 1 mm of the socket's stop: the pencil mark at the stub), the olive under the nut, and the nut ¾ to 1¼ turns past snug; the spanner back in its tray; no part of either gripper within 1 cm of the spanner, the nut or the pipe; the spanner and the pipe still (under 1 cm/s) for 1 s | the pencil mark, the snug marks, look |
| 0.5 | the joint sealed, but the spanner not back and let go, or not yet still for 1 s | as above |
| 0.25 | the pipe home and the olive under the nut on the thread, the nut under ¾ turn past snug | as above |
| 0 | anything else: no olive, the olive put on outside the nut (the nut first), or the pipe short of its stop | look |

An overtightened joint latches: it scores 0 whatever follows. `score()` gives 1.0 only once the full-score state has
held for 1 s (until then 0.5).

**The snug reference.** The nut has a red paint mark on one flat and on its top face; the valve body has a red
paint line on its top and side. They line up when the nut is snug on a new olive, so the evaluator never touches the
joint during a trial: the turns past snug are the turns of the nut's mark past the body's line, counted on the video
(each 60-degree spanner stroke is a sixth of a turn) and read from the marks' angle at the end. The body's line is set
once, before the trials, for the valve, the nut and the batch of olives used: on a calibration pipe in the valve, run
the nut down onto a new olive and tighten it with a torque wrench and a 22 mm crow's-foot to 2 N m (hand-tight snug),
and paint the line on the body in line with the nut's mark. Repeat with two more new olives: the nut's mark must come
within 15 degrees of the line each time (the thread is single-start, so a new olive of the batch comes snug at the
same angle). In the simulation the nut is snug after one turn on the thread, with its mark on the line. Three scripted checks (`JOINT_DEMO=...`) show what
falls short:
- `noolive`: the same job without the olive. The nut runs down loose and stops: 0.
- `swap`: the nut dropped before the olive. The olive ends up outside the nut: 0.
- `over`: the right order, but 18 strokes. The olive is crushed: 0 (latched).

Offline checks of the model, every tier, the latch and the demo: `python -m pytest -q test_compression_joint.py` (29
tests, no simulator).

## Runtime and seeds

Measured on an L40S with the viewer on, Isaac Sim's start-up included:
- the demo trial (1,913 steps, 95.7 s of simulated time) took 6.6 minutes of wall-clock time;
- the Dream-1 trial (the full 2,400 steps, 120 s simulated) took 6.4 minutes.

The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

- **The joint** is our own model (`fittingkit.CompressionJoint`); PhysX cannot cut a thread or compress an olive.
  - *The socket.* It holds the pipe's end on the axis, and the pipe stands on the stop.
  - *The rings.* The olive and the nut lowered over the pipe's top are guided on, kept centred and level on the pipe,
    and slide down it. The olive comes to rest in the stub's cone; the nut rests on the olive, or on the thread if
    there is no olive.
  - *The thread.* Resting there, the nut is on the thread, and the model poses it from then on. Turned clockwise, it
    runs down 1.8 mm a turn. With the olive under it, it is snug after 1 turn and compresses the olive. Past 1½ turns
    more, the olive is crushed. With no olive under it, it runs down 2.2 turns and stops, loose.
  - *The spanner.* Its jaw slid onto the nut's flats within 3 mm of the axis is on, and turns the nut with it. Slid
    off along its handle, it comes off. Its jaws are drawn but do not collide: the nut is posed by the model once it
    is on the thread, and solid jaws on a posed nut would be thrown off it. The model holds a jaw that is on the nut
    on its axis (it cannot pass through the nut), and lets it go only when it is drawn off along the handle. The
    spanner's handle and head behind the jaw collide.
- **Simplified.** A real nut runs on 3–4 turns by hand before it is snug; here the run-on is 1 turn, all with the
  spanner. No tightening torque is modelled.
- **Geometry.** Rigid bodies built from boxes, at real sizes: a 15 mm pipe, a 22 mm nut, a 22 mm open-ended spanner
  (0.1 kg). The rings' bores are drawn true to size, but their colliders are a little wider, so that they slide down
  the pipe as real ones do.
- **Arms.** The simulated arms are soft. The spanner's jaw end droops a few degrees in the grip, and the demo's hand
  aims 6 mm higher for it.

## Real-world build

- A 15 mm compression ball valve, its inlet facing up, screwed to a wooden mounting block, a red paint line on its
  body at the snug angle (see Scoring), and a red paint mark on one flat and the top face of its nut.
- A torque wrench with a 22 mm crow's-foot, used only to set the snug line before the trials.
- A 15 mm copper pipe tail (60 mm, deburred, pencil-marked at 12 mm) standing in a drilled block.
- The valve's compression nut and a new brass olive, in a tray.
- A 22 mm open-ended spanner, in its tray.

Positions are in `task.json`.

## Reset between trials

Undo the nut and lift the pipe out. Cut the olive off and fit a new one; olives are not reused once compressed.
Stand the pipe in its holder, and put the nut, the olive and the spanner in their trays.
