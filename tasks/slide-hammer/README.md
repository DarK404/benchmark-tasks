# Pull a stuck pin with a slide hammer: hook it into the eye, steady the shaft, sling the weight against the stop

A steel eye pin is stuck in the bore of a fixture block, its eye standing out of the block's face. A slide hammer
lies on its rack beside it. The robot must pull the pin out with the slide hammer and put the slide hammer back.

A slide hammer pulls what a steady pull cannot move: stuck pins, axles, bearing races, seals. It has a hook (or an
adaptor) at one end, a heavy weight sliding on its shaft and a stop at the other end. Hooked on, the user steadies the
shaft in line with one hand and slings the weight against the stop with the other. Each blow gives a peak force many
times what the arms could pull, and the part comes out a few millimetres at a time.

The task tests using a tool's momentum where force fails, with two hands that do different jobs:
- hooking the tool on;
- keeping it in line;
- throwing the weight fast enough, again and again, following the tool as it comes out;
- putting the tool back.

Pulling on the weight slowly, however long, does not move the pin.

## Files

| File | What |
|---|---|
| `slide_hammer.py` | The yam_arena task file: objects, the pin model's set-up, `success()`, `score()` and the scripted `demo()` |
| `hammerkit.py` | Our own helper: rigid bodies built from boxes, a PhysX slider (the weight on the shaft), the stuck-pin model (the hook in the eye, the blows, the pin coming out), and hooks into yam_arena's scene and reset |
| `test_slide_hammer.py` | Offline checks of the model, every scoring tier, and the demo's reach and per-step limits |

`hammerkit.py` hooks `yam_arena.scene` and the embodiment's reset without modifying yam_arena. The task runs with the
standard `yam_arena` embodiment and the `task_score` scorer. All the code is the task authors' own (MIT licence); no
third-party assets are used.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=slide_hammer.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Pull the stuck pin out of the block with the slide hammer, then put the slide hammer back on its rack." \
  --scorer task_score --max-steps 2000 --store-frames --log-dir ~/yam-eval/hammer-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode. To render the
trial's videos (one MP4 per camera), run `inspect-robots video ~/yam-eval/hammer-dream/<log>.json`.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=slide_hammer.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Pull the stuck pin out of the block with the slide hammer, then put the slide hammer back on its rack." \
  --scorer task_score --max-steps 2000 --store-frames --log-dir ~/yam-eval/hammer-demo
```

The demo is open loop: it uses only the fixed layout, never the simulator's internal state.

1. **Hook it on.** Both hands lift the slide hammer off its rack: the right by the shaft just behind the hook, the
   left by the handle. They carry it to the work and lower the hook's peg into the pin's eye, with the handle in the
   work support.
2. **Take the weight.** The left hand lets go of the handle and takes the weight by its waisted grip. The right keeps
   the shaft steady and in line.
3. **Pull the pin.** The left hand slings the weight along the shaft against the end stop, aiming a little past it,
   and eases it back: about 1.5 s a blow. The weight strikes at 0.65–0.8 m/s, and each blow pulls the pin about
   5 mm. The slide hammer comes out with the pin, so each throw aims 3 mm further. Nine blows bring the pin out
   (there is one spare); it hangs on the hook.
4. **Put it back.** The left hand takes the handle again, and both hands lay the slide hammer back on its rack.

The demo reaches 1.0 (success) in 964 steps (48.2 s). No step of the demo is clamped by the run's
delta-limit guard, which lets the gripper change by at most 5% of its range per step. `test_slide_hammer.py` checks
every step against it, and checks that every pose along the arms' paths is reached.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

| Score | Condition | How to check on a real bench |
|---|---|---|
| 1.0 | the pin out of its bore (pulled out 40 mm); the slide hammer back on its rack, in both cradles (within 8 mm across them, anywhere along them within 25 mm, level within 5 degrees), no part of either gripper within 1 cm of it or its weight, still (under 1 cm/s) for 1 s | look: the bore is empty |
| 0.5 | the pin out; the slide hammer not back | look |
| 0.25 | the pin started (pulled out 5 mm or more) but still in the bore | a rule against the block's face |
| 0 | anything else | |

A scripted check shows that only blows work: `HAMMER_DEMO=steady` does the same job, but pulls the weight slowly up
to the stop and holds it there, 2.6 s a pull, ten times. It scores 0: the pin never moves.

Offline checks of the model, every tier and the demo: `python -m pytest -q test_slide_hammer.py` (25 tests, no
simulator).

## Runtime and seeds

Measured on an L40S with the viewer on, Isaac Sim's start-up included:
- the demo trial (964 steps, 48.2 s of simulated time) took 3.0 minutes of wall-clock time;
- the Dream-1 trial (the full 2,000 steps, 100 s simulated) took 5.2 minutes.

The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

- **The pin** is our own model (`hammerkit.SlideHammerPin`); PhysX cannot hold a press fit that gives only to blows.
  - *The eye.* Its collider is a heavy body held where the pin is, every physics substep. The hook's peg goes into its
    hole, bears on it and rests on it by ordinary contact, and the fingers can touch it. It does not give to a steady
    pull of any size.
  - *Hooking on.* The peg in the hole keeps the shaft from rolling about its axis. Just over the hole's edge, the
    peg's rounded tip is guided in.
  - *The blows.* The weight slides on the shaft on a PhysX slider. Striking the end stop with the hook in the eye, at
    a speed v relative to the shaft, it pulls the pin out 11 mm per m/s above 0.25 m/s. Striking the front collar
    drives it back in the same way. The eye follows the pin at up to 0.1 m/s.
  - *Out.* Pulled out 40 mm, the pin is free: it is drawn hanging on the hook, and its collider is put away.
- **Masses are scaled to the arms**: a real slide hammer weighs 1–2.5 kg; here the tool is 0.55 kg, its weight
  0.30 kg.
- **Geometry.** Rigid bodies built from boxes; friction 1.0:
  - the slide hammer: 47 cm, with a square-section weight whose waisted grip sits between raised ends;
  - the block, the work support and the rack: fixed.
- **Arms.** The simulated arms are soft. The steadying hand moves out with the slide hammer as the pin comes out,
  and the demo allows for it.

## Real-world build

- A blued-steel fixture block (6 x 6 x 6.4 cm) with a 10 mm reamed bore.
- A 10 mm steel eye pin with a flat 45 mm eye, held in the bore by a friction fit (an O-ring in a groove) that a
  steady pull of the arms cannot overcome.
- A mini slide hammer (about 1 kg, with a hook adaptor).
- A wooden cradle that supports the handle end at the work, and a wooden rack with two cradles.

Positions are in `task.json`.

## Reset between trials

Push the pin back into its bore to its stop, and lay the slide hammer on its rack with its weight forward.
