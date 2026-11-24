# Door Room — Step 2 of the ladder

Step 2 from **"Training a Brain in a 3D Sandbox"** (2026-11-23): the
raycast door room. Same evolution method as CartPole (step 1), new
world — a room with one door, and raycast "feelers" instead of exact
numbers.

## The world

A 12×12 room, walls on all sides, one door in the east wall
(y 5.5–6.5). The agent starts at a random spot anywhere in the room,
facing a random direction, and gets only **7 raycast distances**
(-90°…+90° from its heading). A ray pointing at the door travels far;
a ray hitting a wall stops short. That contrast is the only clue
where the door is — no coordinates, no "distance to door" number.

Actions: move forward / turn left 30° / turn right 30°.
Reward: +100 for stepping through the door, -0.1 per step, a small
shaping term for getting closer to the door, -0.5 for bumping a wall.

## Run it (verified 2026-11-24 in this VM)

```bash
cd door-room
../.venv/bin/python train.py            # train (~1 min here) and save the brain
../.venv/bin/python train.py --play     # load the saved brain and just play
```

## Verified result

- Training: population 96, stopped at generation 15. The **mean**
  score of the whole population climbed from **-55.5 to about -14**
  — that is the population actually learning, generation by
  generation. The single best brain got lucky early (106 in gen 1,
  on 3 episodes), which is honest to say: with 96 random brains and
  such a strong raycast signal, one of them stumbles on a
  turn-toward-the-long-ray policy fast.
- Fresh check: the saved brain reached the door in **49 of 50**
  fresh single-episode runs (98%, mean score 100.8), from random
  starts anywhere in the room.
- Baseline: 10 random untrained brains reached the door **0%** of
  the time. The trained brain is doing something real, not luck.

## Saving the brain

`train.py` writes two files in this folder:

- `brain.npz` — the weights (179), the brain itself
- `brain.json` — its shape/config, so it can be rebuilt

`--play` loads `brain.npz` and plays with no training code running.
Same §8 point as step 1: saving for **playing** is not saving for
**resuming training** — resuming would also need the population
state, which isn't saved here.

## Try tweaking one thing at a time

- `--generations 60 --population 128` — a longer run
- In `train.py`, remove the shaping term (`(prev_dist - dist) * 2.0`)
  and watch how much slower the population learns — the
  sparse-reward lesson from the report, hands-on.
- Narrow the door (change `DOOR_Y0`/`DOOR_Y1`) and see the win rate fall.
