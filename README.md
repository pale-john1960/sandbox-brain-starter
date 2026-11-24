# Sandbox Brain Starter

Step 1 of the ladder from **"Training a Brain in a 3D Sandbox"** (delivered 2026-11-23):
Python + Gymnasium CartPole — the afternoon warm-up before the raycast
door room, the bicycle, and the walking creature.

## What it is

A tiny brain — a 4 → 16 → 2 neural net (114 weights) — trained by
**evolution**, the second method in the report: a population of 64 random
brains, scored, worst half killed, winners bred and mutated, generation
after generation. No gradient, no torch, no GPU. Pure numpy + gymnasium.

## Run it (verified 2026-11-24 in this VM)

```bash
cd ~/sandbox-brain-starter
.venv/bin/python train.py            # train (~1 s here) and save the brain
.venv/bin/python train.py --play     # load the saved brain and just play
```

Verified result: training hit 500/500 in generation 3 (best 140 → 163 → 500),
and the saved brain then survived 500/500 steps in all 5 fresh play episodes.
CartPole is *supposed* to be the easy one — that's the point of a warm-up.
The files are already here: `train.py`, `brain.npz`, `brain.json`, `.venv/`.

Try tweaking one thing at a time and re-running:
- `--generations 50 --population 100` — a longer, smoother improvement curve
- In `train.py`, change the reward: e.g. score by pole angle instead of
  survival time, and watch it learn a different (worse) trick — the
  reward-hacking lesson from the report, hands-on.

## Saving the brain

`train.py` writes two files:
- `brain.npz` — the weights, the brain itself
- `brain.json` — its shape/config, so it can be rebuilt

`--play` loads `brain.npz` and plays with no training code running.
That is the report's §8 distinction: saving for **playing** is not saving
for **resuming training** — resuming would also need the population state,
which isn't saved here.

## What's next (your call)

Step 2 is done too — see [`door-room/`](door-room/README.md): the same
evolution code in a room with one door, found by 7 raycast feelers.
Verified 2026-11-24: 49/50 fresh runs reached the door (random brains: 0%).

GitHub is done as well — this repo lives at
<https://github.com/pale-john1960/sandbox-brain-starter>, pushed
2026-11-24, first commit `b179b89`.

Two options remain from the original list:

1. **RL branch** — same CartPole, but PPO via Stable-Baselines3, so you
   can compare one-brain-learning vs population-evolution on the same task.
2. **Fruit-fly bridge** — the other report's idea: swap this MLP for a
   small FlyWire subcircuit as the brain and see what changes.
