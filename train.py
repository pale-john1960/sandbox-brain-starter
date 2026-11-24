# Sandbox Brain Starter — Step 1 of the 3D-training ladder
# From the report "Training a Brain in a 3D Sandbox" (2026-11-23):
#   Step 1 = Python + Gymnasium CartPole, an afternoon.
#   This is the Evolution branch of that report: a POPULATION of brains,
#   scored, worst half killed, winners bred/mutated. Nobody learns in its
#   lifetime — the population improves, generation by generation.
#
# The brain is a tiny MLP: 4 inputs (cart position, cart velocity,
# pole angle, pole angular velocity) -> 16 hidden (tanh) -> 2 outputs
# (push left / push right). Its "brain" is just its weights.
#
# Saving the brain: train.py writes brain.npz (weights) + brain.json
# (shape/config). play.py loads brain.npz and just PLAYS — no training
# code running. That is the inference-vs-resume distinction from §8 of
# the report: resuming training would also need the population state,
# which is not saved here on purpose.
#
# Run:
#   .venv/bin/python train.py            # trains, saves brain.npz
#   .venv/bin/python train.py --play     # loads brain.npz, plays 5 episodes
#   .venv/bin/python train.py --generations 50 --population 100   # longer run
#
# No torch, no GPU. Pure numpy + gymnasium. Runs on a normal PC.

import argparse
import json
import sys

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    sys.exit("gymnasium is not installed. Run: .venv/bin/pip install gymnasium numpy")

ENV_ID = "CartPole-v1"
INPUTS, HIDDEN, OUTPUTS = 4, 16, 2
SHAPES = [(INPUTS, HIDDEN), (HIDDEN,), (HIDDEN, OUTPUTS), (OUTPUTS,)]


def n_params():
    return sum(int(np.prod(s)) for s in SHAPES)


def unpack(theta):
    """Flat weight vector -> list of weight/bias arrays."""
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


def evaluate(theta, episodes=3, seed=0, render=False):
    """Mean total reward over `episodes` runs. CartPole caps at 500."""
    total = 0.0
    for ep in range(episodes):
        env = gym.make(ENV_ID)
        obs, _ = env.reset(seed=seed + ep)
        done, reward_sum = False, 0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(act(theta, obs))
            reward_sum += reward
            done = terminated or truncated
        env.close()
        total += reward_sum
    return total / episodes


def train(population=64, generations=30, elite=8, seed=0):
    rng = np.random.default_rng(seed)
    pop = rng.normal(0, 0.5, size=(population, n_params()))
    best_theta, best_score = None, -1.0
    for gen in range(1, generations + 1):
        scores = np.array([evaluate(t, episodes=2, seed=1000 + gen) for t in pop])
        order = np.argsort(scores)[::-1]
        pop, scores = pop[order], scores[order]
        if scores[0] > best_score:
            best_score, best_theta = float(scores[0]), pop[0].copy()
        print(f"gen {gen:2d}: best {scores[0]:6.1f}  mean {scores.mean():6.1f}  "
              f"best-ever {best_score:6.1f}", flush=True)
        if best_score >= 500:
            print("Solved (500/500) — stopping early.")
            break
        # Breed: children = average of two random elites + mutation.
        # Sigma shrinks as the population improves, like cooling down.
        sigma = 0.3 * (1 - gen / (generations + 1)) + 0.05
        children = []
        for _ in range(population - elite):
            a, b = rng.choice(elite, size=2, replace=False)
            child = (pop[a] + pop[b]) / 2 + rng.normal(0, sigma, size=n_params())
            children.append(child)
        pop = np.vstack([pop[:elite], np.array(children)])
    return best_theta, best_score


def save(theta, score):
    np.savez("brain.npz", theta=theta)
    with open("brain.json", "w") as f:
        json.dump({"env": ENV_ID, "inputs": INPUTS, "hidden": HIDDEN,
                   "outputs": OUTPUTS, "shapes": [list(s) for s in SHAPES],
                   "train_score_mean_of_3": score,
                   "method": "evolution (population, elitism, mutation)",
                   "note": "Weights only — playing, not resuming training."},
                  f, indent=2)


def play(episodes=5):
    data = np.load("brain.npz")
    theta = data["theta"]
    print(f"Loaded brain.npz ({theta.size} weights). Playing {episodes} episodes:")
    for ep in range(episodes):
        env = gym.make(ENV_ID)
        obs, _ = env.reset(seed=9000 + ep)
        done, total = False, 0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(act(theta, obs))
            total += reward
            done = terminated or truncated
        env.close()
        print(f"  episode {ep + 1}: survived {int(total)}/500 steps "
              f"({'SOLVED' if total >= 500 else 'fell over'})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--play", action="store_true", help="load brain.npz and play, no training")
    p.add_argument("--population", type=int, default=64)
    p.add_argument("--generations", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.play:
        play()
    else:
        theta, score = train(population=args.population,
                             generations=args.generations, seed=args.seed)
        final = evaluate(theta, episodes=5, seed=42)
        print(f"\nBest brain: train score {score:.1f}, fresh 5-episode check: {final:.1f}/500")
        save(theta, final)
        print("Saved: brain.npz (weights) + brain.json (shape/config)")
        print("Try it: .venv/bin/python train.py --play")
