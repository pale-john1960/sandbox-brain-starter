# Sandbox Brain Starter — Step 2 of the 3D-training ladder
# From the report "Training a Brain in a 3D Sandbox" (2026-11-23):
#   Step 2 = the raycast door room: same evolution code as CartPole
#   (step 1), new world — a room with one door, and raycast "feelers"
#   instead of exact numbers.
#
# The world: a 12x12 room, walls on all sides, one door in the east
# wall (y in [5.5, 6.5]). The agent starts somewhere in the west half,
# facing a random direction. It cannot see the door directly — it only
# gets 7 raycast distances (-90..+90 degrees from its heading), like
# feelers. A ray that points at the door travels far; a ray that hits
# a wall stops short. That contrast is the only clue where the door is.
#
# The brain: 7 raycasts -> 16 hidden (tanh) -> 3 outputs
# (move forward / turn left / turn right). Trained by evolution:
# population, elitism, mutation — nobody learns in its lifetime, the
# population improves. Reward: +100 for reaching the door, -0.1 per
# step (so dawdling costs), plus a small shaping term for getting
# closer to the door. Bumping a wall costs -0.5 and the agent stays put.
#
# Saving the brain: train.py writes brain.npz (weights) + brain.json
# (shape/config) in this folder. --play loads brain.npz and just plays,
# no training code running — the report's §8 inference-vs-resume point.
#
# Run:
#   ../.venv/bin/python train.py            # train and save the brain
#   ../.venv/bin/python train.py --play     # load the saved brain and play
#   ../.venv/bin/python train.py --generations 80 --population 128
#
# No torch, no GPU. Pure numpy. Runs on a normal PC.

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS, HIDDEN, OUTPUTS = 7, 16, 3
SHAPES = [(INPUTS, HIDDEN), (HIDDEN,), (HIDDEN, OUTPUTS), (OUTPUTS,)]

ROOM = 12.0          # room is 12 x 12, walls at 0 and 12
DOOR_Y0, DOOR_Y1 = 5.5, 6.5   # door opening in the east wall
DOOR_X = 11.0
MAX_STEPS = 200
RAY_ANGLES = [-90, -60, -30, 0, 30, 60, 90]  # degrees, relative to heading
MAX_RAY = 16.0
STEP_SIZE = 0.5
TURN = math.radians(30)


def n_params():
    return sum(int(np.prod(s)) for s in SHAPES)


def unpack(theta):
    out, i = [], 0
    for shape in SHAPES:
        n = int(np.prod(shape))
        out.append(theta[i:i + n].reshape(shape))
        i += n
    return out


def act(theta, obs):
    w1, b1, w2, b2 = unpack(theta)
    h = np.tanh(obs @ w1 + b1)
    return int(np.argmax(h @ w2 + b2))


def is_wall(x, y):
    """True if (x, y) is inside a wall. The door gap is not a wall."""
    if x < 0 or x > ROOM or y < 0 or y > ROOM:
        return True
    if x >= DOOR_X and DOOR_Y0 <= y <= DOOR_Y1:
        return False  # the door opening
    return x <= 0.5 or x >= ROOM - 0.5 or y <= 0.5 or y >= ROOM - 0.5


def raycast(x, y, angle):
    """Distance along `angle` until a wall, capped at MAX_RAY."""
    dx, dy = math.cos(angle), math.sin(angle)
    d = 0.2
    while d < MAX_RAY:
        if is_wall(x + dx * d, y + dy * d):
            return d
        d += 0.2
    return MAX_RAY


def observe(x, y, heading):
    return np.array([raycast(x, y, heading + math.radians(a)) / MAX_RAY
                     for a in RAY_ANGLES], dtype=float)


def run_episode(theta, seed, max_steps=MAX_STEPS):
    rng = np.random.default_rng(seed)
    # Start anywhere in the room (not just the west half), so the brain
    # must actually find the door instead of just walking east.
    x = rng.uniform(1.5, 10.0)
    y = rng.uniform(1.5, 10.5)
    heading = rng.uniform(0, 2 * math.pi)
    total, prev_dist = 0.0, math.hypot(DOOR_X - x, 6.0 - y)
    for _ in range(max_steps):
        a = act(theta, observe(x, y, heading))
        if a == 0:  # forward
            nx, ny = x + math.cos(heading) * STEP_SIZE, y + math.sin(heading) * STEP_SIZE
            if is_wall(nx, ny):
                total -= 0.5  # bump
            else:
                x, y = nx, ny
        elif a == 1:
            heading += TURN
        else:
            heading -= TURN
        dist = math.hypot(DOOR_X - x, 6.0 - y)
        total += (prev_dist - dist) * 2.0 - 0.1  # shaping + step cost
        prev_dist = dist
        if x >= DOOR_X and DOOR_Y0 <= y <= DOOR_Y1:
            return total + 100.0, True
    return total, False


def evaluate(theta, episodes=5, seed=0):
    """Mean reward and success rate over `episodes` fixed-seed runs."""
    rewards, wins = [], 0
    for ep in range(episodes):
        r, won = run_episode(theta, seed=seed + ep)
        rewards.append(r)
        wins += won
    return float(np.mean(rewards)), wins / episodes


def train(population=96, generations=60, elite=8, seed=0):
    rng = np.random.default_rng(seed)
    pop = rng.normal(0, 0.5, size=(population, n_params()))
    best_theta, best_score, best_wins = None, -1e9, 0.0
    for gen in range(1, generations + 1):
        scored = [evaluate(t, episodes=3, seed=1000 + gen) for t in pop]
        scores = np.array([s[0] for s in scored])
        order = np.argsort(scores)[::-1]
        pop, scores = pop[order], scores[order]
        wins = scored[order[0]][1]
        if scores[0] > best_score:
            best_score, best_theta, best_wins = float(scores[0]), pop[0].copy(), wins
        print(f"gen {gen:2d}: best {scores[0]:7.1f}  mean {scores.mean():7.1f}  "
              f"best-ever {best_score:7.1f} (wins {wins:.0%})", flush=True)
        # Only stop early after a real curve has had time to form, and
        # only on 3/3 training wins — a single lucky generation is not
        # proof the brain generalises (the fresh 5-episode check decides).
        if gen >= 15 and best_wins >= 1.0 and best_score > 50:
            print("Solved (door found in every training episode) — stopping early.")
            break
        sigma = 0.3 * (1 - gen / (generations + 1)) + 0.05
        children = []
        for _ in range(population - elite):
            a, b = rng.choice(elite, size=2, replace=False)
            child = (pop[a] + pop[b]) / 2 + rng.normal(0, sigma, size=n_params())
            children.append(child)
        pop = np.vstack([pop[:elite], np.array(children)])
    return best_theta, best_score, best_wins


def save(theta, score, wins):
    np.savez(os.path.join(HERE, "brain.npz"), theta=theta)
    with open(os.path.join(HERE, "brain.json"), "w") as f:
        json.dump({"env": "door-room 12x12, 7 raycasts, door east wall y 5.5-6.5",
                   "inputs": INPUTS, "hidden": HIDDEN, "outputs": OUTPUTS,
                   "shapes": [list(s) for s in SHAPES],
                   "train_score_mean_of_5": score, "train_win_rate": wins,
                   "method": "evolution (population, elitism, mutation)",
                   "note": "Weights only — playing, not resuming training."},
                  f, indent=2)


def play(episodes=5):
    data = np.load(os.path.join(HERE, "brain.npz"))
    theta = data["theta"]
    print(f"Loaded brain.npz ({theta.size} weights). Playing {episodes} episodes:")
    for ep in range(episodes):
        r, won = run_episode(theta, seed=9000 + ep)
        print(f"  episode {ep + 1}: {'REACHED THE DOOR' if won else 'did not reach it'}"
              f"  (score {r:.1f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--play", action="store_true", help="load brain.npz and play, no training")
    p.add_argument("--population", type=int, default=96)
    p.add_argument("--generations", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.play:
        play()
    else:
        theta, score, wins = train(population=args.population,
                                   generations=args.generations, seed=args.seed)
        final_score, final_wins = evaluate(theta, episodes=5, seed=42)
        print(f"\nBest brain: train score {score:.1f} (wins {wins:.0%}), "
              f"fresh 5-episode check: {final_score:.1f}, wins {final_wins:.0%}")
        save(theta, final_score, final_wins)
        print("Saved: brain.npz (weights) + brain.json (shape/config)")
        print("Try it: ../.venv/bin/python train.py --play  (from door-room/)")
