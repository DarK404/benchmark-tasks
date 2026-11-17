# Install a plug-on breaker in a load center: load end hooked on the rail, jaw pressed fully onto the bus stab

A Square D Homeline 4-space load center, not wired to any supply, lies on its back on the table, its dead front
off. Spaces 1 and 3 hold breakers; spaces 2 and 4 are empty. A new Homeline breaker stands beside it. The robot must
install the new breaker in space 4.

Electricians (O*NET 47-2111) install breakers whenever a circuit is added or a breaker is replaced. A plug-on breaker
goes in the maker's way (Square D's Homeline instruction bulletin 48840-249-01: "Insert wire terminal end of circuit
breaker into mounting hook. Push circuit breaker inward until plug-on jaws plug securely on to bus bar connector"):
its load end hooked under the mounting rail at the panel's outer edge, then its line end pushed in until its jaw is
fully on the bus stab. A breaker left part way on its stab makes a poor contact that heats and arcs; one not hooked
on the rail works loose.

The task tests seating a part by force in the right order: turning the breaker so its jaw end faces the bus, hooking
its load end under a 4.5 mm lip blind (tipped up, so the jaw clears the stab), letting it down onto the stab, then
pressing hard enough to spread the jaw's springs over the stab, without pushing it off the rail.

## Files

| File | What |
|---|---|
| `breaker.py` | The yam_arena task file: the load center and the breaker, `success()`, `score()` and the scripted `demo()` |
| `breakerkit.py` | Our own helper: rigid bodies built from boxes, sprung slides (the jaw's shoes), per-shape friction, and hooks into yam_arena's scene and reset |
| `test_breaker.py` | Offline checks of the parts' geometry and forces, every scoring tier, and the demo's reach and per-step limits |

`breakerkit.py` hooks `yam_arena.scene` and the embodiment's reset without modifying yam_arena. The task runs with
the standard `yam_arena` embodiment and the `task_score` scorer. All the code is the task authors' own (MIT licence);
no third-party assets are used.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=breaker.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Install the new breaker in space 4 of the unpowered, unwired load center." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/bk-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode. To render the
trial's videos (one MP4 per camera), run `inspect-robots video ~/yam-eval/bk-dream/<log>.json`.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=breaker.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Install the new breaker in space 4 of the unpowered, unwired load center." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/bk-demo
```

The demo is open loop: it uses only the fixed layout, never the simulator's internal state. The right hand does it
all; the left stays home.

1. **Take the breaker** by the top of its sides, the fingers closing across its width, and lift it over the
   enclosure's wall.
2. **Turn it** a quarter turn so its line end points at the bus, and tip it 18 degrees line end up.
3. **Hook it.** Lower it into space 4 with its load end 11 mm short of the rail, slide it out so its hook slot goes
   under the rail's lip (aimed past home, so the soft arm holds it against the lip), and let go: it settles hooked,
   its jaw resting on the stab's tapered top.
4. **Push it home** with the closed fingertips on the top of its line end, aimed well below (the soft simulated arm
   presses until the jaw is on), then lift off and go home.

The demo reaches 1.0 (success) in 451 steps (22.6 s). No step of it is clamped by the run's delta-limit guard (the
gripper may change by at most 5% of its range per step); `test_breaker.py` checks every step against it and that
every pose along the arms' paths is reached.

**Shortcut check.** With `BK_DEMO=nohook` the demo sets the breaker down level on the stab without hooking its load end
under the rail and presses it on: it scores 0.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`). Measured on the
breaker against its seated place in space 4.

| Score | Condition | How to check on a real table |
|---|---|---|
| 1.0 | in space 4; its jaw fully on the stab and its mounting face within 1.0 mm of the interior at both ends; its load end hooked (the rail's lip at least 1 mm into its hook slot); square (within 2 degrees); no part of either gripper within 1 cm of it; the breaker still; held for 1 s | a 1 mm feeler must not pass under either end; a light lift of the load end does not free it; look |
| 0.5 | hooked on the rail of space 4 with its jaw on the stab's full blade but an end more than 1.0 mm up; or seated with a hand still on it or not yet still | the feeler passes under the line end |
| 0.25 | hooked on the rail of space 4, its jaw not on the stab's blade (resting on its tapered top) | look; the line end lifts freely |
| 0 | anything else (for example the jaw pushed onto the stab with the load end not under the rail's lip, or the breaker in another space) | look; the load end lifts free |

`score()` gives 1.0 only once the full-score state has held for 1 s (until then 0.5). Offline checks of the parts,
every tier and the demo: `python -m pytest -q test_breaker.py` (27 tests, no simulator).

## Runtime and seeds

Measured on an L40S with the viewer on, Isaac Sim's start-up included:
- the demo trial (451 steps, 22.6 s of simulated time) took 1.6 minutes of wall-clock time;
- the Dream-1 trial (the full 1,200 steps, 1 minute simulated; it scored 0.0) took 3.5 minutes.

The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

- **The jaw is modelled by contact, not by rules.** It is two copper shoes in a slot in the breaker's line end, each
  on a sprung slide (`breakerkit.Spring`: a PhysX joint with a preloaded linear drive), held 2.0 mm apart against
  their stops with 12 N each (4000 N/m). The 3.2 mm stab spreads them 0.6 mm each; its top tapers at 15 degrees, which
  steers it in, and the shoes' mouths flare to catch it off centre. Friction 0.3 between shoes and stab. So the jaw
  grips the stab with about 14 N a side. In the simulator a steady 40 N push on the line end leaves the jaw part way
  on and 60 N seats it. The spring's force is our estimate: Square D publishes no seating force, and says to push
  until the jaws plug on securely.
- **The hook and the rail** are collision geometry: the load end's foot under a 4.5 mm high slot, the rail's 2.5 mm
  lip reaching 4.5 mm in, 1 mm clearances. Hooked, the breaker can move about 3 mm along its length (2 mm towards the
  bus, where the stab stops its jaw slot; 1 mm towards the rail, where the lip stops its hook slot) and stays hooked.
- **Our estimates:** the interior's inner shapes (stab, rail, lip and the breaker's jaw slot and hook), the 10 mm
  between the two columns' line ends, the breaker's mass (Square D lists only the boxed weight, 0.28 to 0.32 kg) and
  the frictions (0.5 plastic, 0.4 enclosure). The outside sizes are the maker's.
- **Not modelled:** the handle (the breaker stays OFF; it is installed OFF, as the maker says); the wire connections.
- **Geometry.** Rigid bodies built from boxes; the load center is fixed.

## Real-world build

- A Square D Homeline HOM48L125GC load center (4 spaces, 125 A main lugs), not wired to any supply, lying on its back
  with its dead front off, clamped to the table.
- Two Square D HOM120 breakers seated in spaces 1 and 3; a third HOM120 (OFF) as the new breaker.

Positions are in `task.json`.

## Reset between trials

Remove the breaker the maker's way: lift its line end until the jaw lets go of the stab, then lift its load end off
the rail; stand it back on its back on the table at the marked spot, line end away from the robot, handle OFF.
Nothing is consumed.
