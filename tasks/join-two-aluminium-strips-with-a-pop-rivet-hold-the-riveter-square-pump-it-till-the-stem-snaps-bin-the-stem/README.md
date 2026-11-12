# Join two aluminium strips with a pop rivet: hold the riveter square, pump it till the stem snaps, bin the stem

Two aluminium strips lie stacked and clamped on a block, a hole drilled through both. A plier-type hand riveter
stands nose down in its holster, two blind (pop) rivets stand stem up in a holder, and there is a scrap tray. The
robot must rivet the strips together and put the riveter back.

Blind rivets join sheet metal wherever only one side can be reached: ducts, aircraft skins, trailers, enclosures.
Setting one by hand has a standard every sheet-metal worker learns: load the rivet into the riveter by its stem, put
it through the hole, hold the riveter square and pressed on the rivet's head while pumping the handles until the stem
snaps (pulled tilted or lifted off the head, the rivet sets crooked or loose, and has to be drilled out), and bin the
spent stem: left in the tool or on the work, it is foreign-object debris. The task tests working a hand tool to that
standard, with a parallel gripper doing what a hand does: carrying the riveter by its grips, squeezing the handles,
easing them open, and keeping it square on the work.

## Files

| File | What |
|---|---|
| `pop_rivet.py` | The yam_arena task file: objects, the riveting model's set-up, `success()`, `score()` and the scripted `demo()` |
| `rivetkit.py` | Our own helper: rigid bodies built from boxes, sprung hinges (the riveter's handle), the riveting model (loading, inserting, pulling, setting, the spent stem), and hooks into yam_arena's scene and reset |
| `test_pop_rivet.py` | Offline checks of the riveting model, every scoring tier and the latch, and the demo's reach and per-step limits |

`rivetkit.py` hooks `yam_arena.scene` and the embodiment's reset without modifying yam_arena, so the task runs with
the standard `yam_arena` embodiment and the `task_score` scorer. All the code is the task authors' own (MIT licence);
no third-party assets are used.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=pop_rivet.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Rivet the two strips together through the drilled hole with a blind rivet from the holder, then put the riveter back." \
  --scorer task_score --max-steps 1600 --store-frames --log-dir ~/yam-eval/rivet-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode. To render the
trial's videos (one MP4 per camera): `inspect-robots video ~/yam-eval/rivet-dream/<log>.json`.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=pop_rivet.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Rivet the two strips together through the drilled hole with a blind rivet from the holder, then put the riveter back." \
  --scorer task_score --max-steps 1600 --store-frames --log-dir ~/yam-eval/rivet-demo
```

The demo is open loop: it uses only the fixed layout, never the simulator's internal state. The right hand does the
work.

1. It takes the riveter out of its holster by its grips (the fingers closing across both handles), and lowers its
   nose over a rivet's stem in the holder, which loads the rivet.
2. It finds the hole as a person does: the rivet's end set down on the strip 8 mm short of the hole, then slid across
   until it drops in. With the rivet in, the hand moves straight above the nose, squaring the riveter up.
3. It pumps the handles three times, pressing the riveter on the head; only the sprung handle moves, so the hand
   follows half of each stroke and the fixed handle's finger stays put. On the third stroke the stem snaps.
4. Over the scrap tray it eases the handles wide open, and the spent stem drops out; it stands the riveter back in its
   holster and lifts clear of the handles.

The demo reaches 1.0 (success) in 810 steps (40.5 s). The riveter stays within 4.5 degrees of square while
the rivet is pulled. No step of the demo is clamped by the run's delta-limit guard (the gripper may change by at most
5% of its range per step); `test_pop_rivet.py` checks every step against it, and checks that every pose along the arm's
path is reached.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

| Score | Condition | How to check on a real bench |
|---|---|---|
| 0 (latched) | at any point the rivet pulled with the riveter tilted over 6 degrees, or not pressed on its head: set crooked or loose | look at the head; try to turn the strips |
| 1.0 | the rivet set (its stem snapped) through both strips; the spent stem in the scrap tray; the riveter standing back in its holster, no part of either gripper within 1 cm of it, still (under 1 cm/s) for 1 s | look |
| 0.5 | the rivet set, but the stem not in the tray (left in the riveter, on the work or the bench), the riveter not back, or not yet still for 1 s | look |
| 0.25 | a rivet in the hole, not set | look |
| 0 | anything else | |

A crooked or loose rivet latches: it scores 0 whatever follows. `score()` gives 1.0 only once the full-score state has
held for 1 s (until then 0.5). Two scripted checks (`RIVET_DEMO=...` in the demo)
show what falls short:

- `tilted`: the same job, pumping with the hand 16 mm to one side, so the riveter leans about 8 degrees on the rivet:
  0 (set crooked, latched).
- `nobin`: the spent stem let go over the work: 0.5.

Offline checks of the riveting model, every tier, the latch and the demo: `python -m pytest -q test_pop_rivet.py`
(28 tests, no simulator).

## Runtime and seeds

Measured on an L40S with the viewer on, Isaac Sim's start-up included: the demo trial (810 steps, 40.5 s of
simulated time) took 2.6 minutes of wall-clock time; the Dream-1 trial (the full 1,600 steps, 80 s simulated)
took 4.1 minutes. The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

- **The rivet** is our own model (`rivetkit.Riveting`); PhysX cannot pull a stem or upset a rivet's tail. The
  riveter's nose is drawn but does not collide: under it, the model pushes the riveter up from what it stands on (a
  rivet's head, a stem's tip, the holder, the strip). Lowered over a rivet's stem within 3 mm of its axis (the
  nosepiece's countersunk mouth guides it), the riveter loads the rivet. Carried over the hole, the rivet goes in
  within 1.5 mm of the hole's axis (its rounded end finds the hole); elsewhere its end stands on the strip. With the
  stem in the nosepiece and the rivet in the holder or the hole, the nose is held on the rivet's axis (3,000 N/m), so
  pressed on it the riveter can only tip about its nose. Each closing of the handles pulls the stem 5.7 mm per radian
  closed (a full stroke about 2.4 mm); opening them pulls nothing. At 6 mm (the third stroke) the stem snaps; it drops
  out of the nose when the handles open past 21 degrees (their stop is at 25), onto what is under the nose. Lifted off
  before any pull, the riveter takes the rivet back out of the hole.
- **Forces are scaled to the arms**: a real riveter needs a strong squeeze on long handles; here the sprung handle's
  return spring (2 N m/rad, preloaded against its open stop) is what the fingers close against, and it keeps the
  fingers pressed on both grips as the handles open.
- **Geometry.** Rigid bodies built from boxes: the riveter (0.36 kg head and fixed handle, 0.06 kg sprung handle on a
  PhysX hinge) with 3 cm wide grips, friction 1.0; the strips, block, holder, holster and tray fixed. The rivets,
  stems and the set rivet's tail are drawn by bodies the model moves. The rivets are drawn at their real size (8 mm shank,
  18.5 mm of mandrel standing above the head).
- **Arms.** The simulated arms are soft and sag about 8 mm under the riveter; the demo allows for that and for how
  the riveter hangs between the fingers (it hangs a few millimetres off, and shifts as the handles are pumped).

## Real-world build

Two aluminium strips (120 x 30 x 1.5 mm) with a 4.9 mm hole, clamped on a drilled wooden block; a plier-type hand
riveter with a 4.8 mm nosepiece in a wooden holster; two 4.8 x 8 mm dome-head blind rivets in a drilled wooden holder
(aluminium body, zinc-plated steel mandrel; 9.5 mm head, 8 mm shank, mandrel 2.7 mm across and 28 mm long overall;
for a 4.9 mm hole). The 8 mm is the shank's length; the rivet's rated grip range, the material thickness it joins, is
2.0 to 3.5 mm, and the two strips make 3.0 mm. A small steel tray. Positions are in `task.json`.

## Reset between trials

Drill out the set rivet (or use fresh strips), empty the tray, put two rivets in the holder and stand the riveter in
its holster with its handles open.
