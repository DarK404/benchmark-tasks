# Fit a thermostatic head onto a radiator valve: scale pointer at the top, pushed straight on until it snaps home

A small steel panel radiator stands on its floor brackets along the far side of the table, its thermostatic valve at
its right-hand end, the valve's spindle pointing to the robot's right, protective cap off. A new Danfoss RA 2000
snap-on thermostatic head, set to 5, lies on its side in front. The robot must fit the head to the valve with its
scale pointer at the top.

Thermostatic radiator valves are on most radiators in European homes and in many hot-water-heated buildings elsewhere;
their heads come off when a room is decorated or a radiator is changed, and householders put them back. Danfoss's
installation instructions: "Remove the valve cover. Check that the thermostat is set at the factory setting of 5.
Push the thermostat firmly onto the valve until an audible 'click' is heard"; its sensor guide adds "Offer sensor onto
valve ensuring that scale pointer is at top." A head left part way on hangs loose and does not work the valve.

The task tests lining a part up coaxially with a short collar in mid-air (the head's bore takes the collar with 1 mm
of play at its snap teeth), keeping its scale pointer up, and pushing it straight along the valve's axis until its
snap fitting drops in behind the collar's ridge, then letting go without pulling it off.

## Files

| File | What |
|---|---|
| `radiator_valve.py` | The yam_arena task file: the radiator and valve, the head with its sprung snap teeth, `success()`, `score()` and the scripted `demo()` |
| `valvekit.py` | Our own helper: rigid bodies built from boxes (turned boxes for the round parts and the lead-in faces), sprung slides (the snap teeth), per-shape friction, and hooks into yam_arena's scene and reset |
| `test_radiator_valve.py` | Offline checks of the parts' geometry and snap forces, every scoring tier, and the demo's reach and per-step limits |

`valvekit.py` hooks `yam_arena.scene` and the embodiment's reset without modifying yam_arena. The task runs with the
standard `yam_arena` embodiment and the `task_score` scorer. All the code is the task authors' own (MIT licence); no
third-party assets are used.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=radiator_valve.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Fit the thermostatic head onto the radiator valve, its scale pointer at the top." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/rv-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode. To render the
trial's videos (one MP4 per camera), run `inspect-robots video ~/yam-eval/rv-dream/<log>.json`.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=radiator_valve.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Fit the thermostatic head onto the radiator valve, its scale pointer at the top." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/rv-demo
```

The demo is open loop: it uses only the fixed layout, never the simulator's internal state. The right hand does it
all; the left stays home.

1. **Take the head** across its body 55 mm from its base (the hand, about 10 cm wide along the head, then stays clear
   of the valve), fingers closing across it, and lift it.
2. **Line it up** with its bore 25 mm short of the collar's tip, on the valve's axis, pointer up, and pause there.
3. **Push it on** straight along the axis, aimed well past home (the soft simulated arm presses as it goes), until
   the snap teeth ride over the collar's ridge and drop in behind it; ease off.
4. **Let go** and go home.

The demo reaches 1.0 (success) in 355 steps (17.8 s). No step of it is clamped by the run's delta-limit guard
(the gripper may change by at most 5% of its range per step); `test_radiator_valve.py` checks every step against it
and that every pose along the arms' paths is reached.

**Shortcut check.** With `RV_DEMO=halfway` the demo pushes the head only 8 mm onto the collar, short of the snap, and
lets go: the head droops off the collar's tip, and it scores 0.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`). Measured on the head
against its seated place on the valve.

| Score | Condition | How to check on a real table |
|---|---|---|
| 1.0 | snapped home (its base within 1.0 mm of the valve's shoulder: its teeth behind the collar's ridge); on the valve's axis within 1.5 mm and square within 3 degrees; scale pointer within 20 degrees of the top; no part of either gripper within 1 cm of it; still; held for 1 s | a 1 mm feeler does not enter between the head's base and the shoulder; a light pull does not draw it off; the pointer against a square |
| 0.5 | snapped home but the pointer more than 20 degrees from the top; or a hand still on it, or not yet still | the square |
| 0.25 | on the collar with its teeth at or past the ridge's start, not snapped home (part way on) | the feeler enters; a light pull draws it off |
| 0 | anything else (for example drooping off the collar's tip, or not on the valve) | look |

`score()` gives 1.0 only once the full-score state has held for 1 s (until then 0.5). Offline checks of the parts,
every tier and the demo: `python -m pytest -q test_radiator_valve.py` (23 tests, no simulator).

## Runtime and seeds

Measured on an L40S with the viewer on, Isaac Sim's start-up included:
- the demo trial (355 steps, 17.8 s of simulated time) took 1.3 minutes of wall-clock time;
- the Dream-1 trial (the full 1,200 steps, 1 minute simulated; it scored 0.0) took 3.3 minutes.

The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

- **The snap fitting is modelled by contact, not by rules.** The head's bore holds two teeth, one each side, each on a
  sprung slide (`valvekit.Spring`: a PhysX joint with a preloaded linear drive), held 22 mm apart against their stops
  with 3 N each (2000 N/m). The valve's collar has a 12 mm ridge (24 mm across) with a 20-degree lead-in towards its
  tip and a square back: pushed on, the teeth ride up the lead-in, spread 1 mm, slide over the ridge and drop in
  behind it, and the square back then holds the head on. Friction 0.3. By these figures pushing it home takes about
  7 N (Danfoss's data sheet: "applying a gentle pressure"); the spring figures are our estimates.
- **The parts' sizes:** the head's 49 mm diameter and 90 mm length and the valve's outside shape are from Danfoss's
  drawings; the snap connection's inner shapes (collar 20 mm, ridge, teeth, bore) are our estimates; the head's mass
  (150 g) is our estimate.
- **Not modelled:** the head's dial (it stays at 5, as it comes) and its sensing; the valve's pin; the water.
- **Geometry.** Rigid bodies built from boxes; round parts are 12-sided prisms of boxes; the radiator and valve are
  fixed.

## Real-world build

- A Type 11 single steel panel radiator, 300 mm high x 400 mm long, on its floor brackets, not plumbed, with a
  Danfoss RA-N 15 straight thermostatic valve body fitted at its lower right-hand end, spindle horizontal, protective
  cap removed.
- A Danfoss RA 2000 snap-on thermostatic head (RA 2990), set to 5.

Positions are in `task.json`.

## Reset between trials

Take the head off the Danfoss way (turn its base counter-clockwise to release the snap fitting, and draw it off),
check it is still at 5, and lay it back on its side at the marked spot, base towards the valve, scale pointer up.
Nothing is consumed.
