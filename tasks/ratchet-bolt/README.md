# Undo a tight bolt with a ratchet that is set to tighten: check its direction lever, flip it, then swing it

A steel bracket on a block has an M10 hex bolt done up tight in it. Beside it, a 3/8" ratchet with a 17 mm socket
lies in a wooden tray, and its direction lever is set to tighten. The robot must undo the bolt with the ratchet and
put the ratchet back.

A ratchet only drives one way. A lever on its head picks which way: the pull turns the socket, and the back-swing
clicks the pawl over its teeth. A mechanic reaching for a ratchet checks that lever first. Used on the wrong setting,
the pulls just click, and the back-swings drive the bolt tighter: a tight bolt is stretched past its yield, and has
to be replaced. The task tests checking and setting a tool's mode before using it, and then using it as a mechanic
does:
- seating the socket square on the head;
- swinging the handle in short arcs, driving on the pull and letting it click back;
- stopping when the bolt is free;
- putting the tool away.

## Files

| File | What |
|---|---|
| `ratchet_bolt.py` | The yam_arena task file: objects, the ratchet-and-bolt model's set-up, `success()`, `score()` and the scripted `demo()` |
| `ratchetkit.py` | Our own helper: rigid bodies built from boxes, sprung hinges (the direction lever), the ratchet-and-bolt model (the lever's detent, the socket on the head, the one-way clutch, the bolt's breakaway and thread), and hooks into yam_arena's scene and reset |
| `test_ratchet_bolt.py` | Offline checks of the model, every scoring tier and the latch, and the demo's reach and per-step limits |

`ratchetkit.py` hooks `yam_arena.scene` and the embodiment's reset without modifying yam_arena. The task runs with the
standard `yam_arena` embodiment and the `task_score` scorer. All the code is the task authors' own (MIT licence); no
third-party assets are used.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=ratchet_bolt.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Undo the bolt with the ratchet, then put the ratchet back in its tray." \
  --scorer task_score --max-steps 1600 --store-frames --log-dir ~/yam-eval/ratchet-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode. To render the
trial's videos (one MP4 per camera), run `inspect-robots video ~/yam-eval/ratchet-dream/<log>.json`.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=ratchet_bolt.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Undo the bolt with the ratchet, then put the ratchet back in its tray." \
  --scorer task_score --max-steps 1600 --store-frames --log-dir ~/yam-eval/ratchet-demo
```

The demo is open loop: it uses only the fixed layout, never the simulator's internal state.

1. **Set the lever (left hand).** With the ratchet still lying in its tray, the left hand pushes the direction lever
   across with its closed fingertips, from tighten to loosen. It pushes the lever just past the middle, and the
   detent snaps it the rest of the way.
2. **Seat the socket (right hand).** The right hand takes the ratchet by its handle, 5 cm from the socket (near its
   balance point). It lifts the ratchet out and lowers the socket onto the bolt's head.
3. **Undo the bolt.** It swings the handle anticlockwise (seen from above) through 50 degree strokes, back-swinging
   between them while the pawl clicks. The first pull breaks the bolt loose. After nine strokes the bolt has turned
   out more than a full turn.
4. **Put the ratchet back.** The hand lifts the socket off, steadies the ratchet over the tray, lays it in its cradle
   and lets go.

The demo reaches 1.0 (success) in 938 steps (46.9 s), with the bolt turned 416 degrees out. No step of the
demo is clamped by the run's delta-limit guard, which lets the gripper change by at most 5% of its range per step;
`test_ratchet_bolt.py` checks every step against it, and checks that every pose along the arms' paths is reached.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

| Score | Condition | How to check on a real bench |
|---|---|---|
| 0 (latched) | at any point the bolt turned more than 20 degrees tighter than it started (stretched) | the paint mark on the head against the mark on the bracket |
| 1.0 | the bolt turned out at least one full turn; the ratchet back in its tray (socket within 8 mm of its place, level within 8 degrees, the handle along the tray within 15 degrees), no part of either gripper within 1 cm of it, still (under 1 cm/s) for 1 s | count the paint mark's turns; look |
| 0.5 | the bolt a full turn out, but the ratchet not back | as above |
| 0.25 | the bolt broken loose (turned out at least 30 degrees) but under a full turn | the paint mark |
| 0 | anything else | |

An overtightened bolt latches: it scores 0 whatever follows. A scripted check (`RATCHET_DEMO=noflip`) shows what
the lever is for. It does the same job without flipping the lever. On tighten, the first pull just clicks and the
first back-swing drives the bolt 36 degrees tighter: 0 (latched).

Offline checks of the model, every tier, the latch and the demo: `python -m pytest -q test_ratchet_bolt.py` (28
tests, no simulator).

## Runtime and seeds

Measured on an L40S with the viewer on, Isaac Sim's start-up included:
- the demo trial (938 steps, 46.9 s of simulated time) took 2.7 minutes of wall-clock time;
- the Dream-1 trial (the full 1,600 steps, 80 s simulated) took 4.3 minutes.

The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

- **The bolt and the clutch** are our own model (`ratchetkit.RatchetBolt`); PhysX has no threads or pawls.
  - *The lever.* The direction lever is a PhysX hinge on the ratchet's head, held at either end of its 60 degree
    travel by a detent. Past the middle, it reads as the other setting.
  - *Seating the socket.* The socket goes onto the head when lowered within 1.5 mm of its axis. Its chamfered mouth
    steers it there from up to 4 mm off; further off, it stands on the head. Once it is on, the socket's mouth is held
    on the bolt's axis (3,000 N/m) and bottoms on the washer, until it is lifted off.
  - *The clutch.* Turned in its driving direction, the socket winds up a stiff spring until the bolt gives: on loosen,
    0.12 N m to break it loose and 0.03 N m once loose; on tighten, 0.15 N m. Turned the other way, it clicks and
    turns nothing.
  - *Drawing.* The head is drawn turning, rising 1.5 mm per turn as the bolt comes out. A collider just inside it
    makes the head solid to the fingers, and a hand cannot turn it. That collider is moved aside while the socket is
    over the head, where the model takes over.
- **Torques are scaled to the arms.** A real M10 bolt takes tens of newton metres. Here the breakaway is about 2.4 N
  at the demo's grip, which the soft arms can give.
- **Geometry.** Rigid bodies built from boxes:
  - the ratchet: 0.28 kg, with a 17 cm handle, a 22 mm socket and a 38 mm head; friction 1.0;
  - its lever: 5 g;
  - the bracket, the block and the tray: fixed.
- **Arms.** The simulated arms are soft. Holding the ratchet out over the tray, the right arm sags about 7 mm; the
  demo aims past that.

## Real-world build

- A steel L-bracket (6 x 7 cm plate, 5 mm thick, with a 4 cm upstand) on a wooden block (8 x 8 x 3 cm).
- An M10 x 1.5 zinc-plated hex bolt (DIN 933: 17 mm across flats, 6.4 mm head) with a 20 mm washer, through the
  plate into a nut, done up tight. A paint mark across its head and onto the plate serves as the witness mark.
- A 3/8" drive ratchet (17 cm, 0.3 kg) with a 17 mm 6-point socket fitted, lying in a wooden tray with a socket
  cradle and a handle rest.

Positions are in `task.json`.

## Reset between trials

Tighten the bolt to its marks (a new bolt if it was overtightened), set the ratchet's lever to tighten and lay it in
its tray.
