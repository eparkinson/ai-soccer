"""
Genetic algorithm for GeneticBrain (see docs/genetic_algorithm_learning.md).

Each generation every individual plays the same panel of opponents on the same
kick-offs: DefendersAndAttackers, StrategicPlanner, BehindAndTowards and the current
PPO champion. Fitness is points per game plus half the goal difference per game. The
best --elite individuals survive unchanged; the rest of the next generation is bred by
tournament selection, uniform crossover and Gaussian mutation.

The best individual so far is written to runs/league/ga/best.json, and plays in every
league round as the entrant "GA". league.py starts, pauses and stops this process.

    poetry run python evolve.py --workers 3
"""

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

from aisoccer.brains.GeneticBrain import CHROMOSOME_LENGTH, DEFENSIVE_SEED, GeneticBrain
from aisoccer.game import Game
from aisoccer.brainspec import panel_from_dir
from league import HISTORY_DIR, LEAGUE_DIR, make_brain

GA_DIR = LEAGUE_DIR / "ga"
LOG_FILE = GA_DIR / "progress.log"
PANEL = ["DefendersAndAttackers", "StrategicPlanner", "BehindAndTowards", "champion"]


def log(message=""):
    print(message, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")


def champion_path():
    champions = sorted(
        HISTORY_DIR.glob("PPO-champ-*.npz"), key=lambda p: int(p.stem.split("-")[-1])
    )
    return str(champions[-1])


def play(task):
    """(chromosome, opponent spec, GA plays blue, seed) -> (goals for, goals against)."""
    chromosome, opponent, ga_blue, seed = task
    a, b = GeneticBrain("GA", chromosome), make_brain(opponent)
    blue, red = (a, b) if ga_blue else (b, a)
    score = Game(blue, red, quiet_mode=True, seed=seed).play()
    return (score["blue"], score["red"]) if ga_blue else (score["red"], score["blue"])


def evaluate(workers, population, games, seed, panel=None):
    """Fitness of every individual, on shared kick-offs against the same panel."""
    rng = np.random.default_rng(seed)
    panel = panel or [champion_path() if p == "champion" else p for p in PANEL]
    seeds = rng.integers(2**31, size=games)
    tasks = [
        (ind, opp, g % 2 == 0, int(seeds[g]))
        for ind in population
        for opp in panel
        for g in range(games)
    ]
    results = np.array(workers.map(play, tasks, chunksize=8), dtype=float)
    results = results.reshape(len(population), len(panel) * games, 2)
    gd = results[:, :, 0] - results[:, :, 1]
    points = np.where(gd > 0, 3.0, np.where(gd == 0, 1.0, 0.0))
    return points.mean(axis=1) + 0.5 * gd.mean(axis=1)


def breed(population, fitness, rng, elite, mutation_rate, mutation_sd):
    order = np.argsort(-fitness)
    children = [population[i].copy() for i in order[:elite]]

    def pick():  # tournament selection
        contenders = rng.choice(len(population), size=3, replace=False)
        return population[contenders[np.argmax(fitness[contenders])]]

    while len(children) < len(population):
        a, b = pick(), pick()
        child = np.where(rng.random(CHROMOSOME_LENGTH) < 0.5, a, b)  # uniform crossover
        mutate = rng.random(CHROMOSOME_LENGTH) < mutation_rate
        child = np.clip(
            child + mutate * rng.normal(0, mutation_sd, CHROMOSOME_LENGTH), 0, 1
        )
        children.append(child)
    return children


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--panel-dir", help="play against brains from this folder (e.g. the league field)")
    parser.add_argument("--panel-size", type=int, default=4)
    parser.add_argument("--random-init", action="store_true", help="start from random chromosomes")
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--games", type=int, default=12, help="per panel opponent")
    parser.add_argument("--elite", type=int, default=4)
    parser.add_argument("--mutation-rate", type=float, default=0.2)
    parser.add_argument("--mutation-sd", type=float, default=0.08)
    args = parser.parse_args()

    GA_DIR.mkdir(parents=True, exist_ok=True)
    state_file = GA_DIR / "population.json"
    rng = np.random.default_rng(int.from_bytes(os.urandom(4), "little"))
    if state_file.exists():
        state = json.loads(state_file.read_text())
        population = [np.array(c) for c in state["population"]]
        generation = state["generation"]
        log(f"GA resumed at generation {generation}")
    else:
        half = 0 if args.random_init else args.population // 2
        seeded = [
            np.clip(DEFENSIVE_SEED + rng.normal(0, 0.1, CHROMOSOME_LENGTH), 0, 1)
            for _ in range(half - 1)
        ]
        population = ([] if args.random_init else [DEFENSIVE_SEED.copy()]) + seeded
        population += [
            rng.random(CHROMOSOME_LENGTH)
            for _ in range(args.population - len(population))
        ]
        generation = 0
        log(f"GA started: {args.population} individuals, half near the defensive seed")

    best_file = GA_DIR / "best.json"
    with Pool(args.workers) as workers:
        while True:
            generation += 1
            panel = panel_from_dir(args.panel_dir, args.panel_size, rng) if args.panel_dir else None
            fitness = evaluate(workers, population, args.games, seed=generation, panel=panel)
            best = int(np.argmax(fitness))
            tmp = GA_DIR / "best.tmp.json"
            GeneticBrain.save(
                tmp,
                population[best],
                generation=generation,
                fitness=float(fitness[best]),
            )
            os.replace(tmp, best_file)
            log(
                f"generation {generation:4d}: best fitness {fitness[best]:.2f}, "
                f"median {np.median(fitness):.2f}"
            )
            population = breed(
                population,
                fitness,
                rng,
                args.elite,
                args.mutation_rate,
                args.mutation_sd,
            )
            state_file.write_text(
                json.dumps(
                    {
                        "generation": generation,
                        "population": [list(map(float, c)) for c in population],
                    }
                )
            )


if __name__ == "__main__":
    main()
