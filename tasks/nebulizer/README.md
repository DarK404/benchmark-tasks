# Put a compressor nebulizer back together after cleaning, ready for the next treatment

A compressor nebulizer's parts lie washed and dried after a treatment. The compressor stands on the right with its air
tube pushed onto its connector, the tube lying coiled in front of it. The kit's medication tank stands in its kit stand;
the inhalation top, the cap and the mouthpiece lie on a drying mat. The robot must put the nebulizer together, ready
for the next treatment.

Nebulizer treatments for asthma, COPD and croup are given every day at home and in care homes, and after each one
the kit is taken apart, washed and dried, then put back together for the next. Home health and personal care aides,
nursing assistants and nurses do it many times a day, and parents and patients do it at home. Putting the kit back
together follows the maker's instructions:

- **The top locks into the tank.** The inhalation top goes into the medication tank with its tab lined up with the
  tank's notch, and is turned clockwise until it clicks. A top not turned to the click comes loose in use.
- **The cap and the mouthpiece go on.** The cap goes on the top's air inlet; the mouthpiece on its outlet.
- **The air tube goes onto the nozzle under the tank.** Without it, no air reaches the kit.
- **The kit stands upright in its stand**, ready to use.

The task tests putting a small multi-part device together with two hands. The top can be turned only while the
other hand holds the tank (turned on its own, the tank turns with it). The tube is pushed onto a nozzle under the kit,
so one hand must hold the kit on its side while the other pushes the tube on.

## Files

| File | What |
|---|---|
| `nebulizer.py` | The yam_arena task file: objects, the model's set-up, `success()`, `score()` and the scripted `demo()` |
| `nebkit.py` | Our own helper: rigid bodies built from boxes, a hose of ball-jointed links, the nebulizer model (the top's seat and click-lock, the cap, mouthpiece and tube cuff pushed on), and hooks into yam_arena's scene and reset |
| `test_nebulizer.py` | Offline checks of the model, every scoring tier, and the demo's reach and per-step limits |

`nebkit.py` hooks `yam_arena.scene` and the embodiment's reset without modifying yam_arena. The task runs with the
standard `yam_arena` embodiment and the `task_score` scorer. All the code is the task authors' own (MIT licence); no
third-party assets are used.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=nebulizer.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the nebulizer together, ready for the next treatment." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/neb-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode. To render the
trial's videos (one MP4 per camera), run `inspect-robots video ~/yam-eval/neb-dream/<log>.json`.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=nebulizer.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the nebulizer together, ready for the next treatment." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/neb-demo
```

The demo is open loop: it uses only the fixed layout, never the simulator's internal state.

1. **The top.** The right hand takes the inhalation top from the mat, fingers across it clear of its outlet, and sets
   it on the tank with its tab at the tank's notch. The left hand comes down beside the tank and holds it; the right
   turns the top clockwise to the click.
2. **The cap and the mouthpiece.** The right hand presses the cap onto the top's air inlet, then pushes the
   mouthpiece onto its outlet.
3. **The tube.** The left hand lifts the kit out of the stand and lays it on its side, the nozzle towards the right
   hand. The right hand takes the tube's free end by its cuff and pushes it onto the nozzle.
4. **Back in the stand.** The left hand stands the kit up and lowers it into the stand, lets go and moves aside.

The demo reaches 1.0 (success) in 1,296 steps (64.8 s). No step of it is clamped by the run's delta-limit
guard (the gripper may change by at most 5% of its range per step); `test_nebulizer.py` checks every step against it,
and that every pose along the arms' paths is reached.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

| Score | Condition | How to check on a real table |
|---|---|---|
| 1.0 | the inhalation top locked in the tank (turned to the click, its tab at the tank's lock mark); the cap on the air inlet; the mouthpiece on the outlet; the tube's cuff on the tank's nozzle; the kit standing upright in its stand (within 10 degrees); no part of either gripper within 1 cm of the kit; the kit still (under 1 cm/s) for 1 s | look: the tab against the lock mark, each part pushed home, the kit in its stand |
| 0.5 | all of that put together, but the kit not back upright in its stand, a hand still on it, or not yet still for 1 s | look |
| 0.25 | the top locked in the tank, but the cap, the mouthpiece or the tube not on | look |
| 0 | anything else (for example the top set in the tank but not turned to the click) | look |

`score()` gives 1.0 only once the full-score state has held for 1 s (until then 0.5). Order does not matter: the score
reads what has been put together. A scripted check (`NEB_DEMO=nohold`) does the same job but turns the top before the
left hand holds the tank: the top never clicks, and the kit scores 0. Offline checks of the model, every tier and the demo:
`python -m pytest -q test_nebulizer.py` (32 tests, no simulator).

## Runtime and seeds

Measured on an L40S with the viewer on, Isaac Sim's start-up included:
- the demo trial (1,296 steps, 64.8 s of simulated time) took 7.5 minutes of wall-clock time;
- the Dream-1 trial (the full 2,400 steps, 2 minutes simulated) took 8.3 minutes.

The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

- **The kit's fittings** are our own model (`nebkit.Nebulizer`).
  - *The top.* Set on the tank's rim, square, on its axis (within 3 mm) and with its tab within 20 degrees of the
    tank's notch, it drops in, and from then on we draw it on the tank. Turned clockwise by a hand closed on it while
    the other hand is closed on the tank, it turns in the tank, and at 40 degrees it clicks and locks. Turned with
    nothing holding the tank, the tank would turn with it: nothing changes. The hands are told apart by their
    fingers' collision hulls; the turn follows the line between the fingers.
  - *The cap, the mouthpiece and the tube's cuff.* Pushed onto their spigot (the air inlet, the outlet, the nozzle)
    square (within 15 degrees), on its axis (within 3 mm for the cap, 5 mm for the others, which a tapered spigot
    guides) and at least 3 mm over it, each goes on and is drawn by us on the part.
  - *Weights.* The parts once on are drawn by us and weigh nothing in the simulation; the tank carries the
    assembled kit's weight (45 g).
- **The air tube** is 25 links of 4 cm joined by ball joints, 6.5 mm across: it hangs and drags like a light tube,
  but does not twist back or resist bending. Its end on the compressor is held there.
- **Sizes.** The compressor is at its published size (142 x 72 x 98 mm). Omron does not publish the kit's dimensions;
  the kit's sizes here are our estimates (a 44 mm tank, a 40 mm top, a 65 mm mouthpiece). The score depends on what
  has been put together, not on these sizes.
- **Not modelled.** No medicine is added: the dose is put in at treatment time, by prescription. No air flows.
- **Geometry.** Rigid bodies built from boxes; the stand, the mat and the compressor fixed.
- **Arms.** The simulated arms are soft; the left hand's level grip is reached only along one line in front of its
  base (x 0.17 m), which is why the stand stands there.

## Real-world build

- An Omron CompAIR NE-C801 compressor (142 x 72 x 98 mm, 270 g), switched off and unplugged, with its PVC air tube
  (Omron NEB-TP-L10E, 100 cm) pushed onto its air connector.
- An Omron V.V.T. Nebulizer Kit N-01 (model NEB-NEST3-81E): the medication tank, the inhalation top with its vaporizer
  head, the cap; and the mouthpiece (Omron NEB-MP-E). Washed and dried, empty.
- A wooden kit stand: a fork whose prongs (55 mm up) take the tank by its shoulder, open underneath for the tube.
- A drying mat (11 x 12 cm).

Positions are in `task.json`.

## Reset between trials

Pull the tube's cuff off the nozzle and lay the tube coiled in front of the compressor. Take the mouthpiece and the
cap off, turn the top anticlockwise and lift it out of the tank, and lay the top, the cap and the mouthpiece on the
mat; stand the tank in its stand with its notch towards the robot. Nothing is consumed.
