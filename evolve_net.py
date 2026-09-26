"""
Genetic algorithm on neural network weights ("deep GA" neuroevolution).

The genome is every weight of a PPOBrain policy network, starting random: no
hand-written behaviour at all. Each generation every network plays the same panel
(brains from --panel-dir, e.g. the league field) on the same kick-offs. The best
--parents survive as parents; the next generation is the single best network
unchanged plus mutated copies of random parents (Gaussian noise of size --sigma on
every weight). Fitness is points per game plus half the goal difference per game.

The best network is written to <out>/best.npz (a normal PPOBrain file), which
league.py plays as the entrant "GA-net".

    poetry run python evolve_net.py --workers 2 --panel-dir runs/league2/field
"""

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from aisoccer.brains.PPOBrain import ACT_DIM, OBS_DIM, PPOBrain
from aisoccer.brainspec import load_brain, panel_from_dir
from aisoccer.game import Game
from aisoccer.ppo import PPO
from neuroevolve import flatten, unflatten

OUT_DIR = Path(os.environ.get("LEAGUE_DIR", "runs/league")) / "ga_net"


def log(message):
    print(message, flush=True)
    with open(OUT_DIR / "progress.log", "a") as f:
        f.write(message + "\n")


def play(task):
    population_file, k, shapes, log_std, opponent, ga_blue, seed = task
    vector = np.array(np.load(population_file, mmap_mode="r")[k])
    a = PPOBrain("GA-net", weights={"policy": unflatten(vector, shapes), "log_std": log_std})
    b = load_brain(opponent)
    blue, red = (a, b) if ga_blue else (b, a)
    score = Game(blue, red, quiet_mode=True, seed=seed).play()
    return (score["blue"], score["red"]) if ga_blue else (score["red"], score["blue"])


def main():
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--parents", type=int, default=6)
    parser.add_argument("--sigma", type=float, default=0.05, help="mutation size")
    parser.add_argument("--games", type=int, default=4, help="per panel opponent")
    parser.add_argument("--panel-dir", required=True)
    parser.add_argument("--panel-size", type=int, default=4)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    OUT_DIR = args.out
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state_file = OUT_DIR / "state.npz"
    rng = np.random.default_rng(int.from_bytes(os.urandom(4), "little"))

    if state_file.exists():
        state = np.load(state_file, allow_pickle=True)
        population, shapes = state["population"], [tuple(s) for s in state["shapes"]]
        log_std, generation = state["log_std"], int(state["generation"])
        log(f"GA-net resumed at generation {generation}")
    else:
        networks = [PPO(OBS_DIM, ACT_DIM, seed=int(rng.integers(2**31))).weights() for _ in range(args.population)]
        shapes = [p.shape for p in networks[0]["policy"]]
        population = np.stack([flatten(n["policy"]) for n in networks])
        log_std, generation = networks[0]["log_std"], 0
        log(f"GA-net started: {args.population} random networks of {population.shape[1]} weights")

    with Pool(args.workers) as workers:
        while True:
            generation += 1
            panel = panel_from_dir(args.panel_dir, args.panel_size, rng)
            seeds = rng.integers(2**31, size=args.games)
            population_file = OUT_DIR / "population.npy"
            np.save(population_file, population)
            tasks = [
                (str(population_file), k, shapes, log_std, opp, g % 2 == 0, int(seeds[g]))
                for k in range(len(population))
                for opp in panel
                for g in range(args.games)
            ]
            results = np.array(workers.map(play, tasks, chunksize=4), dtype=float)
            results = results.reshape(len(population), len(panel) * args.games, 2)
            gd = results[:, :, 0] - results[:, :, 1]
            points = np.where(gd > 0, 3.0, np.where(gd == 0, 1.0, 0.0))
            fitness = points.mean(axis=1) + 0.5 * gd.mean(axis=1)

            order = np.argsort(-fitness)
            best = population[order[0]]
            tmp = OUT_DIR / "best.tmp.npz"
            PPOBrain.save_weights(tmp, {"policy": unflatten(best, shapes), "log_std": log_std})
            os.replace(tmp, OUT_DIR / "best.npz")
            (OUT_DIR / "best.json").write_text(json.dumps({"generation": generation, "fitness": float(fitness[order[0]])}))
            log(
                f"generation {generation:4d}: best fitness {fitness[order[0]]:.2f}, "
                f"median {np.median(fitness):.2f}"
            )

            parents = population[order[: args.parents]]
            children = parents[rng.integers(len(parents), size=len(population) - 1)]
            children = children + args.sigma * rng.standard_normal(children.shape)
            population = np.concatenate([best[None, :], children])
            np.savez(state_file, population=population, shapes=np.array(shapes, dtype=object),
                     log_std=log_std, generation=generation)


if __name__ == "__main__":
    main()
