# Thread a new cord through a table lamp's pipe, leaving about 3 inches out of the top for the socket

A column table lamp being rewired lies on its side on the table, its old cord out and its socket and harp off. The
new lamp cord lies on the table at the lamp's base, its end cut ready. The robot must thread the new cord through
the lamp, leaving about 3 inches out of the top.

Replacing a lamp's cord is one of the commonest household electrical repairs: a frayed, cracked or pet-chewed lamp
cord is a shock and fire hazard, and the fix is a new cord fed through the lamp and wired to the socket. It is done
at home, and by handymen and maintenance staff in homes, hotels and offices. The cord goes in at the bottom of the
lamp's hollow centre pipe (in the base) and is fed up it until it comes out of the threaded top; about 3 inches is
left out there for tying the knot and wiring the socket, and the rest runs out of the base to the plug.

The task tests feeding a limp cord along a narrow bore with two hands: the cord's end has to find a 7.6 mm hole,
the lamp's base stops the hand short of the pipe's mouth so the cord is fed hand over hand (pushed in, let go, taken
again where it hangs, pushed again), and the other hand takes the end as it comes out of the top and draws it out
to length.

## Files

| File | What |
|---|---|
| `lamp_cord.py` | The yam_arena task file: objects, `success()`, `score()` and the scripted `demo()` |
| `lampkit.py` | Our own helper: rigid bodies built from boxes, a bending joint that makes a chain of links behave as a cord with bending stiffness, per-body damping, and hooks into yam_arena's scene and reset |
| `test_lamp_cord.py` | Offline checks of every scoring tier and of the demo's reach and per-step limits |

`lampkit.py` hooks `yam_arena.scene` and the embodiment's reset without modifying yam_arena. The task runs with the
standard `yam_arena` embodiment and the `task_score` scorer. All the code is the task authors' own (MIT licence); no
third-party assets are used.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=lamp_cord.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Thread the new cord through the lamp, leaving about 3 inches out of the top." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/lamp-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode. To render the
trial's videos (one MP4 per camera), run `inspect-robots video ~/yam-eval/lamp-dream/<log>.json`.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=lamp_cord.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Thread the new cord through the lamp, leaving about 3 inches out of the top." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/lamp-demo
```

The demo is open loop: it uses only the fixed layout, never the simulator's internal state.

1. **Into the pipe.** The right hand takes the cord 5 cm behind its end, lifts it to the pipe's height and carries
   the end into the pipe's mouth in the base, 3 cm up it, then lets go: the pipe holds the end.
2. **Up the pipe.** The right hand feeds the cord hand over hand: it takes the cord again where it hangs out of the
   mouth, pushes it 2 cm in, lets go, and again, until the end shows out of the threaded top.
3. **Out to length.** The left hand takes the cord just out of the top and draws it out, so about 3 inches stand
   out, then lets go. Both hands go home.

The demo reaches 1.0 (success) in 1,796 steps (89.8 s). No step of it is clamped by the run's delta-limit guard (the
gripper may change by at most 5% of its range per step); `test_lamp_cord.py` checks every step against it, that every
pose along the arms' paths is reached, and that the right hand stays clear of the lamp's base.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

| Score | Condition | How to check on a real table |
|---|---|---|
| 1.0 | the cord runs through the pipe and its end stands 60 to 100 mm (about 3 in) out of the pipe's top; no part of either gripper within 1 cm of the cord; the cord out of the pipe still; held for 1 s | a rule from the pipe's top to the cord's end |
| 0.5 | the cord through the pipe, its end out of the top, but shorter or longer than that (or a hand still on it, or not yet still) | look; the rule |
| 0.25 | the cord's end at least half way up the pipe, not out of the top | feel or a mark on the cord |
| 0 | anything else (the cord not in the pipe, or only just into it; a cord laid past the lamp, not through it) | look |

`score()` gives 1.0 only once the full-score state has held for 1 s (until then 0.5). Offline checks of every tier
and of the demo: `python -m pytest -q test_lamp_cord.py` (23 tests, no simulator).

## Runtime and seeds

Measured on an L40S with the viewer on, Isaac Sim's start-up included:
- the demo trial (1,796 steps, 89.8 s of simulated time) took 5.4 minutes of wall-clock time;
- the Dream-1 trial (the full 2,400 steps, 2 minutes simulated) took 6.8 minutes.

The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

- **The cord** is 42 links of 12 mm (50 cm of it; the rest of the 8 ft cord is not modelled), 2.8 x 5.5 mm (SPT-1
  18/2), 0.5 g each (about 42 g/m, twice real SPT-1's ~20 g/m, for a steadier simulation), joined by bending
  joints: they cannot stretch or twist, and they bend against a spring of 0.01 N m/rad each (a bending stiffness of
  about 1.2e-4 N m2: held out flat, 10 cm of it droops about 4 cm; our estimate, not a measured figure). The links
  are damped a little so the light chain stays stable. Its end is cut to a point over
  its last 5 mm (3 x 1.8 mm), as is usually done to lead it into the pipe.
- **The pipe** is a 12-sided bore of 7.6 mm across, 306 mm long (the maker gives no bore; 1/8 IP all-thread lamp
  pipe is 3/8 in outside). The lamp is a fixed rigid body; the pipe's top stands 30 mm out of its neck.
- **Friction** 0.3 between the cord and the pipe (PVC on zinc-plated steel; chosen, not measured).
- **Not modelled.** The socket, the harp and the plug: wiring them is the next job, after the cord is through.
- **Arms.** The simulated arms are soft. The YAM gripper is about 10 cm wide across its fingers' closing direction, so
  closing across the cord it stops about 2 cm short of the lamp's base: the cord is fed from there.

## Real-world build

- A column table lamp: a ceramic column 60 mm across and 276 mm long (our sizes), with a Westinghouse 7027200
  12-1/16 in 1/8 IP all-thread lamp pipe (zinc-plated steel, 3/8 in outside) through its axis, standing 30 mm out of
  its neck; socket, harp and old cord off. It lies on its side on the table, taped down.
- New lamp cord: Southwire 49908744 18/2 SPT-1 (2.8 x 5.5 mm), its end cut to a point.

Positions are in `task.json`.

## Reset between trials

Pull the cord back out of the pipe from the base and lay it straight on the table along the pipe's line, its end
3 cm from the base. Nothing is consumed.
