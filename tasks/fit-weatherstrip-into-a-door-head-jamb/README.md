# Fit a kerf-in weatherstrip into a door's head jamb: both ends flush, the fin pressed home all along

An exterior door's head jamb lies on its back on the table, taped down, the kerf in its stop facing the robot. A new
kerf-in weatherstrip, cut to the jamb's length, lies loose on the rabbet floor in front of the kerf, not pressed in,
and slid 2 cm too far to the right. The robot must fit the weatherstrip into the kerf with its ends flush with the
jamb's ends.

Kerf-in weatherstrip (a foam bulb on a barbed fin pushed into a saw kerf in the door stop) is the standard seal on
exterior door frames in the US. It is fitted when a frame is made up or repaired, and replaced when it wears or tears:
a routine door repair for carpenters (O*NET 47-2031) and general maintenance and repair workers (49-9071). The makers'
procedure (for example Endura Products' weatherstrip replacement instructions): cut the strip square to the jamb's
length, put its end flush at the corner, then, starting at that end, press it firmly into the kerf along the whole
length, without stretching it.

The task tests seating a limp 81 cm strip into a 3.2 mm slot along its whole length with its ends registered: once
a fin is in, its barbs hold it where it is, so the strip must be lined up first (here, drawn 2 cm along by its far
end), then pressed home a little at a time working away from one end, the two hands taking a half each. Pressed
out of order or pushed along while being pressed, it bunches and its far end overhangs.

## Files

| File | What |
|---|---|
| `weatherstrip.py` | The yam_arena task file: objects, `success()`, `score()` and the scripted `demo()` |
| `weatherkit.py` | Our own helper: rigid bodies built from boxes, a bending joint that makes a chain of links behave as a limp strip, the kerf model (a fin pushed home is held there), per-body damping, and hooks into yam_arena's scene and reset |
| `test_weatherstrip.py` | Offline checks of every scoring tier, of the kerf model, and of the demo's reach and per-step limits |

`weatherkit.py` hooks `yam_arena.scene` and the embodiment's reset without modifying yam_arena. The task runs with
the standard `yam_arena` embodiment and the `task_score` scorer. All the code is the task authors' own (MIT licence);
no third-party assets are used.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=weatherstrip.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Fit the weatherstrip into the head jamb's kerf, its ends flush with the jamb's ends." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/ws-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode. To render the
trial's videos (one MP4 per camera), run `inspect-robots video ~/yam-eval/ws-dream/<log>.json`.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=weatherstrip.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Fit the weatherstrip into the head jamb's kerf, its ends flush with the jamb's ends." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/ws-demo
```

The demo is open loop: it uses only the fixed layout, never the simulator's internal state.

1. **Line it up.** The left hand presses its closed fingers down on the top of the strip's bulb 5 cm from its
   left-hand end and draws the strip 2 cm to the left, so the right-hand end comes back flush with the jamb's end
   (drawn by its far end, the strip runs straight), then lifts off.
2. **The right half.** Starting at the right-hand end, the right hand puts its closed fingers down on the rabbet
   floor behind the bulb and pushes it towards the stop until the fin is home, backs off, and moves 24 mm along;
   17 presses take it to the middle. The right hand goes home.
3. **The left half.** The left hand presses on from the middle in the same way to the left-hand end, and goes home.

The demo reaches 1.0 (success) in 1,242 steps (62.1 s). No step of it is clamped by the run's delta-limit guard (the
gripper may change by at most 5% of its range per step); `test_weatherstrip.py` checks every step against it, that
every pose along the arms' paths is reached, that the hands keep apart, and that every length of the strip is
pressed.

**Shortcut check.** With `WS_DEMO=noslide` the demo presses the strip in where it lies, without lining it up first:
it ends home all along but with its ends 2 cm off, and scores 0.5.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`). A link is one
23.9 mm length of the strip (34 in all); it is home when its fin is at least 8 mm into the 9.5 mm kerf.

| Score | Condition | How to check on a real table |
|---|---|---|
| 1.0 | every link home; both cut ends within 2 mm of the jamb's ends (no gap, no overhang, over 2 mm); no part of either gripper within 1 cm of the strip; the strip still; held for 1 s | a 1.5 mm feeler leaf must not enter between the bulb's base and the stop's face anywhere; a rule at each end |
| 0.5 | at least 90% of the links (31) home, but not all, or an end off by more than 2 mm (or a hand still on it, or not yet still) | the leaf; the rule |
| 0.25 | at least 25% of the links (9) home | the leaf |
| 0 | anything else (for example the strip pressed against the stop's face with its fin not in the kerf) | the leaf |

`score()` gives 1.0 only once the full-score state has held for 1 s (until then 0.5). Offline checks of every tier,
the kerf model and the demo: `python -m pytest -q test_weatherstrip.py` (29 tests, no simulator).

## Runtime and seeds

Measured on an L40S with the viewer on, Isaac Sim's start-up included:
- the demo trial (1,242 steps, 62.1 s of simulated time) took 4.6 minutes of wall-clock time;
- the Dream-1 trial (the full 2,400 steps, 2 minutes simulated) took 8.4 minutes.

The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

- **The kerf** is our own model (`weatherkit.Kerf`): the kerf is a real 3.2 mm slot in the stop's collision
  geometry, which guides the 1.5 mm fin in; a length of strip whose fin has been pushed at least 8 mm in is then held
  there, fully home and in line with the kerf, as the barbs hold the real strip (it no longer slides along or backs
  out). Partly inserted fins are not held.
- **Not modelled: the insertion force.** In the simulation any push that brings the fin 8 mm in seats it; the real
  strip needs a firm press of the fingers (the makers say "press firmly" and publish no figure; we have not measured
  one). The demo's presses aim 12 mm past home with the soft simulated arms.
- **The strip** is 34 links of 23.9 mm (81.3 cm), a box bulb 15.9 x 12.7 mm with a 9.5 x 1.5 mm fin, 25 g in all
  (our estimate), joined by bending joints: they cannot stretch or twist, and bend against a spring of 0.01 N m/rad
  each (our estimate). The real strip can also be stretched, which the makers warn against; the simulated one
  cannot. The links are damped a little so the light chain stays stable.
- **Friction** 0.5 between the strip and the primed wood (chosen, not measured). The jamb is fixed.
- **Arms.** The simulated arms are soft. Closed, the gripper's two pads are 8 mm thick across the strip, so they fit
  the 16 mm of rabbet floor behind the loose strip's bulb.

## Real-world build

- A Pease Doors exterior head jamb for a 32 in door, SKU TJH324 (primed treated poplar, 4-9/16 in deep, 1-1/4 in
  thick with the stop, 3/4 in without, 32 in long; kerfed for standard 1/8 in weatherstrip), lying on its back,
  fixed with double-sided carpet tape.
- A Pemko Q102BL foam kerf-in weatherstrip (black, 5/8 in reach), cut square to 81.3 cm from a 36 in length
  (Q102BL36 or a longer length).

Positions are in `task.json`.

## Reset between trials

Pull the strip out of the kerf from one end and lay it back loose on the rabbet floor as in the setup (its fin's tip
6 mm from the stop's face, its right-hand end 2 cm past the jamb's end: a pencil tick on the table marks where). The
strip is reused; replace it with a new length when its fin's barbs no longer hold in the kerf.
