# Put a table fan back together after washing: blade on its pin, spinner screwed on with the blade held, grille clipped

A 12-inch table fan lies on its back on a towel, face up, after cleaning: its rear grille is on and its motor shaft
points up. Its washed blade lies on the left, its front grille on the right, its spinner on the table in front of
the right hand. The robot must put the fan back together.

Fans gather dust on their blade and grilles, and makers tell owners to take the grilles and blade off and wash them,
then put the fan back together when dry. It is done in homes, offices, hotels and care homes, by housekeepers and
cleaners as part of deep cleaning, and by the fans' owners. Putting it back together follows the maker's manual:

- **The blade goes onto the shaft** until the shaft's retaining pin fits the notch in its hub. Off the pin, the blade
  slips on the shaft.
- **The spinner is screwed onto the shaft anticlockwise** (a left-hand thread, so it tightens as the fan runs)
  until tight. The shaft turns freely, so one hand holds the blade still while the other screws the spinner on;
  turned with nothing holding the blade, the blade and shaft just turn with it.
- **The front grille goes onto the rear grille**, lined up with it, and its clips are closed over the rear grille's
  rim. A clip left open lets the grille come off with the blade turning behind it.

The task tests putting a multi-part appliance back together with two hands: parts that fit only one way (the pin
in the notch, the grille on its clips), a thread that tightens only while the other hand holds the part it would
otherwise turn, and closing four clips round a 34 cm rim.

## Files

| File | What |
|---|---|
| `desk_fan.py` | The yam_arena task file: objects, the model's set-up, `success()`, `score()` and the scripted `demo()` |
| `fankit.py` | Our own helper: rigid bodies built from boxes, the fan model (the blade's notch on the pin, the spinner's thread, the front grille and its clips), and hooks into yam_arena's scene and reset |
| `test_desk_fan.py` | Offline checks of the model, every scoring tier, and the demo's reach and per-step limits |

`fankit.py` hooks `yam_arena.scene` and the embodiment's reset without modifying yam_arena. The task runs with the
standard `yam_arena` embodiment and the `task_score` scorer. All the code is the task authors' own (MIT licence); no
third-party assets are used.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=desk_fan.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the fan back together." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/fan-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode. To render the
trial's videos (one MP4 per camera), run `inspect-robots video ~/yam-eval/fan-dream/<log>.json`.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=desk_fan.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the fan back together." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/fan-demo
```

The demo is open loop: it uses only the fixed layout, never the simulator's internal state.

1. **The blade.** The left hand takes the blade by its hub, carries it over the fan and lowers it onto the shaft,
   its notch on the pin, then lets go.
2. **The spinner.** The left hand moves over the blade that points to the near left. The right hand takes the
   spinner and sets it on the shaft's end; the left presses its blade still, and the right turns the spinner
   anticlockwise a third of a turn at a time (letting go and turning the hand back between strokes) until it is
   tight, then lets go. The left hand goes home.
3. **The front grille.** The right hand takes the front grille by its centre badge, carries it over the fan high
   enough for its rim to clear the spinner, lowers it onto the rear grille with its clips at their places, and lets
   go.
4. **The clips.** The left hand presses the two clips on its side shut, the right hand those on its side.

The demo reaches 1.0 (success) in 1,702 steps (85.1 s). No step of it is clamped by the run's delta-limit guard (the
gripper may change by at most 5% of its range per step); `test_desk_fan.py` checks every step against it, that every
pose along the arms' paths is reached, and that the two hands stay at least 11 cm apart.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

| Score | Condition | How to check on a real table |
|---|---|---|
| 1.0 | the blade on the shaft with the pin in its notch; the spinner screwed down tight; the front grille on the rear grille with all four clips closed; no part of either gripper within 1 cm of the spinner or the front grille; both still (under 1 cm/s); held for 1 s | look: the blade on the pin, try the spinner (does not turn further anticlockwise), each clip over the rim |
| 0.5 | the blade on and the spinner tight, but the front grille not on, a clip open, a hand still on it, or not yet still for 1 s | look |
| 0.25 | the blade on its pin, but the spinner not on or not tight | look; try the spinner |
| 0 | anything else (for example the blade on the shaft but off the pin) | look |

`score()` gives 1.0 only once the full-score state has held for 1 s (until then 0.5). A scripted check
(`FAN_DEMO=nohold`) does the same job but screws the spinner on with nothing holding the blade: the shaft turns with
the spinner, the spinner never tightens, and the fan scores 0.25. Offline checks of the model, every tier and the
demo: `python -m pytest -q test_desk_fan.py` (36 tests, no simulator).

## Runtime and seeds

Measured on an L40S with the viewer on, Isaac Sim's start-up included:
- the demo trial (1,702 steps, 85.1 s of simulated time) took 6.9 minutes of wall-clock time;
- the Dream-1 trial (the full 2,400 steps, 2 minutes simulated) took 7.5 minutes.

The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

- **The fan's fittings** are our own model (`fankit.Fan`).
  - *The blade.* Set on the shaft square, on its axis (within 4 mm) and with its notch within 10 degrees of the pin
    (either way round), it goes on, and from then on we draw it on the shaft.
  - *The spinner.* Set on the shaft's end square and on its axis (within 4 mm), it goes on. Turned anticlockwise by
    a hand closed on it while the other hand holds a blade (fingers on the blade, outside its hub), it screws down
    1.5 mm a turn and is tight after one turn. Turned with nothing holding the blade, the shaft and blade turn with
    it and it does not tighten; turned clockwise, it does not screw on. The hands are told apart by their fingers'
    collision hulls; the turn follows the line between the fingers.
  - *The front grille and its clips.* Set on the rear grille's rim square, on its axis (within 6 mm) and with its
    clips within 10 degrees of their places (any of the four), it sits on, drawn by us; it goes on only once the
    spinner is tight. The grille covers the blade and the spinner, and its mesh is drawn but does not collide, so a
    grille put on first would leave the spinner reachable through it; on a real fan it would have to come off
    again. Each clip closes when a finger presses its end down; we draw it open (tipped out 60 degrees) or closed
    over the rim.
  - *Weights.* The parts once on are drawn by us on the fixed fan: they weigh nothing in the simulation, and their
    speed reads zero (still).
- **Sizes.** Midea publishes no part dimensions for this fan; the sizes here are our estimates for a 12-inch fan: a
  300 mm three-blade propeller with a 40 mm hub (18 mm deep), a spinner 40 mm across and 15 mm tall, a front grille
  336 mm across rising 24 mm to a 60 mm badge; the fan lying on its back stands 105 mm to its rear grille's rim.
  The score depends on what has been put together, not on these sizes.
- **Not modelled.** The motor is not run. The rear grille and its plastic nut are already on and tight.
- **Geometry.** Rigid bodies built from boxes; the fan (lying on its back) and the towel fixed.
- **Arms.** The simulated arms are soft. Reaching this far forward, each arm's wrist stands about 5 cm beyond its
  fingers, so the left hand holds the near-left blade, keeping the two wrists clear of each other.

## Real-world build

- A Midea FT30-8MB 12-inch table fan, switched off and unplugged, laid on its back on a towel, face up, its rear
  grille on with its handle up and its plastic nut tight; its blade, spinner and front grille washed and dried.
- A cotton towel (40 x 40 cm).

Positions are in `task.json`.

## Reset between trials

Open the front grille's clips and lift it off; unscrew the spinner clockwise; lift the blade off the shaft. Lay the
blade on the left and the front grille on the right, as in the setup, and the spinner in front. Nothing is consumed.
