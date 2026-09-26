"""
Genetic algorithm over a tactics brain's team strategy vector (see STRATEGY in
aisoccer/tactics/v1.py). --brain chooses the version.

Each generation every strategy plays the same panel on the same kick-offs:
DefendersAndAttackers, StrategicPlanner, the first PPOBrain and the current league
champion. Fitness is points per game plus half the goal difference per game. The best
--elite survive; the rest are bred by tournament selection, uniform crossover and
Gaussian mutation.

The best strategy so far is written to runs/league/tactics_ga/best.json (a brain spec,
see aisoccer/brainspec.py). league.py runs this as the process behind the entrant
"Tactics-GA".

    poetry run python tune_tactics.py --workers 3
"""

import argparse
import importlib
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from aisoccer.brainspec import load_brain, load_class, panel_from_dir
from aisoccer.game import Game

OUT_DIR = Path(os.environ.get("LEAGUE_DIR", "runs/league")) / "tactics_ga"
HISTORY_DIR = Path("aisoccer/brains/weights/history")
PANEL = [
    "DefendersAndAttackers",
    "StrategicPlanner",
    str(HISTORY_DIR / "PPO-first-it230.npz"),
    "champion",
]


def log(message):
    print(message, flush=True)
    with open(OUT_DIR / "progress.log", "a") as f:
        f.write(message + "\n")


def champion_spec():
    """The newest league champion (any kind of brain)."""
    champions = [p for p in HISTORY_DIR.glob("*champ-*.*")]
    return str(max(champions, key=lambda p: int(p.stem.split("-")[-1])))


BRAIN = "aisoccer.tactics.v1.TacticsV1"


def strategy_keys(brain):
    return importlib.import_module(brain.rsplit(".", 1)[0]).STRATEGY


def as_dict(vector, brain=BRAIN):
    return {k: float(v) for k, v in zip(strategy_keys(brain), vector)}


def play(task):
    brain, vector, opponent, tactics_blue, seed = task
    a = load_class(brain)(name="Tactics", strategy=as_dict(vector, brain))
    b = load_brain(opponent)
    blue, red = (a, b) if tactics_blue else (b, a)
    score = Game(blue, red, quiet_mode=True, seed=seed).play()
    return (
        (score["blue"], score["red"]) if tactics_blue else (score["red"], score["blue"])
    )


def evaluate(workers, brain, population, games, seed, panel=None):
    panel = panel or [champion_spec() if p == "champion" else p for p in PANEL]
    seeds = np.random.default_rng(seed).integers(2**31, size=games)
    tasks = [
        (brain, v, o, g % 2 == 0, int(seeds[g]))
        for v in population
        for o in panel
        for g in range(games)
    ]
    results = np.array(workers.map(play, tasks, chunksize=4), dtype=float)
    results = results.reshape(len(population), len(panel) * games, 2)
    gd = results[:, :, 0] - results[:, :, 1]
    points = np.where(gd > 0, 3.0, np.where(gd == 0, 1.0, 0.0))
    return points.mean(axis=1) + 0.5 * gd.mean(axis=1)


def breed(population, fitness, rng, elite, rate, sd):
    order = np.argsort(-fitness)
    children = [population[i].copy() for i in order[:elite]]

    def pick():
        contenders = rng.choice(len(population), size=3, replace=False)
        return population[contenders[np.argmax(fitness[contenders])]]

    while len(children) < len(population):
        a, b = pick(), pick()
        child = np.where(rng.random(len(a)) < 0.5, a, b)
        mutate = rng.random(len(a)) < rate
        children.append(np.clip(child + mutate * rng.normal(0, sd, len(a)), 0, 1))
    return children


def main():
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--brain", default=BRAIN, help="tactics brain class to tune")
    parser.add_argument("--panel-dir", help="play against brains from this folder (e.g. the league field)")
    parser.add_argument("--panel-size", type=int, default=4)
    parser.add_argument("--random-init", action="store_true", help="start from random strategy vectors")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--games", type=int, default=12, help="per panel opponent")
    parser.add_argument("--elite", type=int, default=3)
    parser.add_argument("--mutation-rate", type=float, default=0.25)
    parser.add_argument("--mutation-sd", type=float, default=0.1)
    args = parser.parse_args()

    OUT_DIR = args.out
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state_file = OUT_DIR / "population.json"
    module = importlib.import_module(args.brain.rsplit(".", 1)[0])
    STRATEGY, DEFAULT_STRATEGY = module.STRATEGY, module.DEFAULT_STRATEGY
    rng = np.random.default_rng(int.from_bytes(os.urandom(4), "little"))
    default = np.array([DEFAULT_STRATEGY[k] for k in STRATEGY])
    if state_file.exists():
        state = json.loads(state_file.read_text())
        population = [np.array(v) for v in state["population"]]
        generation = state["generation"]
        log(f"Tactics GA resumed at generation {generation}")
    else:
        if args.random_init:
            population = [rng.random(len(default)) for _ in range(args.population)]
            # Skill genes (TacticsV2 and later) start near 0: skills not yet discovered.
            for k in getattr(module, "SKILLS", []):
                for vector in population:
                    vector[STRATEGY.index(k)] = rng.random() * 0.1
        else:
            population = [default] + [
                np.clip(default + rng.normal(0, 0.15, len(default)), 0, 1)
                for _ in range(args.population - 1)
            ]
        generation = 0
        log(f"Tactics GA started: {args.population} strategies around the default")

    with Pool(args.workers) as workers:
        while True:
            generation += 1
            panel = panel_from_dir(args.panel_dir, args.panel_size, rng) if args.panel_dir else None
            fitness = evaluate(workers, args.brain, population, args.games, seed=generation, panel=panel)
            best = int(np.argmax(fitness))
            spec = {
                "class": args.brain,
                "kwargs": {"strategy": as_dict(population[best], args.brain)},
                "generation": generation,
                "fitness": float(fitness[best]),
            }
            tmp = OUT_DIR / "best.tmp.json"
            tmp.write_text(json.dumps(spec, indent=1))
            os.replace(tmp, OUT_DIR / "best.json")
            log(
                f"generation {generation:4d}: best fitness {fitness[best]:.2f}, "
                f"median {np.median(fitness):.2f}; best strategy "
                + ", ".join(
                    f"{k} {v:.2f}"
                    for k, v in as_dict(population[best], args.brain).items()
                )
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
                        "population": [list(map(float, v)) for v in population],
                    }
                )
            )


if __name__ == "__main__":
    main()
