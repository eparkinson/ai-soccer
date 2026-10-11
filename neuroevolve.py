"""
Neuroevolution: evolution strategies (OpenAI-ES) on a PPOBrain network's weights.

No gradients: each generation perturbs all the policy weights with --pairs mirrored
random directions (+eps and -eps), plays every perturbed network against the same
panel on the same kick-offs, and moves the weights towards the directions that scored
better (fitness ranks, not raw scores, for robustness to noise):

    theta += lr * sum_i rank_i * eps_i / (pairs * sigma)

Fitness is points per game plus half the goal difference per game against the panel.
The current weights are written to runs/league/neuro/best.npz (a normal PPOBrain
file), which league.py plays as the entrant "Neuro-ES".

    poetry run python neuroevolve.py --workers 3
"""

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from aisoccer.brains.PPOBrain import ACT_DIM, OBS_DIM, PPOBrain
from aisoccer.brainspec import load_brain, panel_from_dir
from aisoccer.ppo import PPO
from aisoccer.game import Game

OUT_DIR = Path(os.environ.get("LEAGUE_DIR", "runs/league")) / "neuro"
HISTORY_DIR = Path("aisoccer/brains/weights/history")
PANEL = ["champion", "DefendersAndAttackers", "StrategicPlanner", "brain:aisoccer.tactics.v1.TacticsV1"]


def log(message):
    print(message, flush=True)
    with open(OUT_DIR / "progress.log", "a") as f:
        f.write(message + "\n")


def champion_spec():
    champions = list(HISTORY_DIR.glob("*champ-*.*"))
    return str(max(champions, key=lambda p: int(p.stem.split("-")[-1])))


def flatten(params):
    return np.concatenate([p.ravel() for p in params])


def unflatten(vector, shapes):
    out, k = [], 0
    for shape in shapes:
        n = int(np.prod(shape))
        out.append(vector[k : k + n].reshape(shape))
        k += n
    return out


def play(task):
    candidates_file, k, shapes, log_std, opponent, es_blue, seed = task
    vector = np.load(candidates_file, mmap_mode="r")[k]  # shared, not copied per task
    weights = {"policy": unflatten(np.array(vector), shapes), "log_std": log_std}
    a = PPOBrain("ES", weights=weights)
    b = load_brain(opponent)
    blue, red = (a, b) if es_blue else (b, a)
    score = Game(blue, red, quiet_mode=True, seed=seed).play()
    return (score["blue"], score["red"]) if es_blue else (score["red"], score["blue"])


def fitness(results):
    results = np.asarray(results, dtype=float)
    gd = results[:, 0] - results[:, 1]
    points = np.where(gd > 0, 3.0, np.where(gd == 0, 1.0, 0.0))
    return points.mean() + 0.5 * gd.mean()


def main():
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--pairs", type=int, default=16, help="mirrored perturbations per generation")
    parser.add_argument("--games", type=int, default=6, help="per panel opponent per perturbation")
    parser.add_argument("--sigma", type=float, default=0.02, help="perturbation size")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--start", default=None, help="network to start from (default: newest champion)")
    parser.add_argument("--random-init", action="store_true", help="start from a random network")
    parser.add_argument("--panel-dir", help="play against brains from this folder (e.g. the league field)")
    parser.add_argument("--panel-size", type=int, default=4)
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="folder for state, log and best.npz")
    args = parser.parse_args()

    OUT_DIR = args.out
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state_file = OUT_DIR / "state.npz"
    rng = np.random.default_rng(int.from_bytes(os.urandom(4), "little"))
    if state_file.exists():
        state = np.load(state_file, allow_pickle=True)
        theta, shapes = state["theta"], [tuple(s) for s in state["shapes"]]
        log_std, generation = state["log_std"], int(state["generation"])
        m, v = state["m"], state["v"]
        log(f"Neuro-ES resumed at generation {generation}")
    else:
        if args.random_init:
            start = "random"
            weights = PPO(OBS_DIM, ACT_DIM, seed=int(rng.integers(2**31))).weights()
        else:
            start = args.start or champion_spec()
            weights = PPOBrain.load_weights(start)
        shapes = [p.shape for p in weights["policy"]]
        theta, log_std, generation = flatten(weights["policy"]), weights["log_std"], 0
        m, v = np.zeros_like(theta), np.zeros_like(theta)
        log(f"Neuro-ES started from {Path(start).stem}: {theta.size} weights")

    with Pool(args.workers) as workers:
        while True:
            generation += 1
            if args.panel_dir:
                panel = panel_from_dir(args.panel_dir, args.panel_size, rng)
            else:
                panel = [champion_spec() if p == "champion" else p for p in PANEL]
            seeds = rng.integers(2**31, size=args.games)
            eps = rng.standard_normal((args.pairs, theta.size))
            candidates = np.concatenate([theta + args.sigma * eps, theta - args.sigma * eps])
            candidates_file = OUT_DIR / "candidates.npy"
            np.save(candidates_file, candidates)
            tasks = [
                (str(candidates_file), k, shapes, log_std, opp, g % 2 == 0, int(seeds[g]))
                for k in range(len(candidates))
                for opp in panel
                for g in range(args.games)
            ]
            results = workers.map(play, tasks, chunksize=4)
            n = len(panel) * args.games
            scores = np.array([fitness(results[k * n : (k + 1) * n]) for k in range(len(candidates))])

            # Centred ranks in [-0.5, 0.5]; mirrored pairs share kick-offs, so their
            # difference is a low-noise estimate of the slope along each direction.
            ranks = np.empty(len(scores))
            ranks[np.argsort(scores)] = np.arange(len(scores))
            ranks = ranks / (len(scores) - 1) - 0.5
            plus, minus = ranks[: args.pairs], ranks[args.pairs :]
            gradient = ((plus - minus)[:, None] * eps).sum(axis=0) / (2 * args.pairs * args.sigma)

            # Adam ascent.
            m = 0.9 * m + 0.1 * gradient
            v = 0.999 * v + 0.001 * gradient**2
            theta = theta + args.lr * (m / (1 - 0.9**generation)) / (
                np.sqrt(v / (1 - 0.999**generation)) + 1e-8
            )

            weights = {"policy": unflatten(theta, shapes), "log_std": log_std}
            tmp = OUT_DIR / "best.tmp.npz"
            PPOBrain.save_weights(tmp, weights)
            os.replace(tmp, OUT_DIR / "best.npz")
            np.savez(
                state_file, theta=theta, shapes=np.array(shapes, dtype=object), log_std=log_std,
                generation=generation, m=m, v=v,
            )
            (OUT_DIR / "best.json").write_text(
                json.dumps({"generation": generation, "fitness": float(scores.mean())})
            )
            log(
                f"generation {generation:4d}: population fitness mean {scores.mean():.2f}, "
                f"best {scores.max():.2f}, worst {scores.min():.2f}"
            )


if __name__ == "__main__":
    main()
