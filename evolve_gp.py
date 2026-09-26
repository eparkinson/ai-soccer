"""
Genetic programming for GPBrain (aisoccer/gp/v1.py): evolves the program that tells
every player where to go, starting from random programs.

Standard tree GP, strongly typed (points, numbers and yes/no values only combine where
they fit):

- the first generation is random programs, ramped half-and-half (depths 2 to 6),
- each generation every program plays the same panel (brains from --panel-dir, e.g.
  the league field) on the same kick-offs; fitness is points per game (3 a win, 1 a
  draw) plus half the goal difference per game,
- selection is by tournament on fitness minus --parsimony x program size (pressure
  against bloat), and no program may exceed --max-size nodes or --max-depth,
- the best --elite programs survive unchanged; the rest are children by subtree
  crossover, subtree mutation or point mutation (a function swapped for another of
  the same shape, a terminal for another of the same type, or a constant nudged).

Outputs in --out: progress.log (best fitness and program size each generation),
population.json (the resumable state), best.json (a brain spec for
aisoccer/brainspec.py load_brain) and best.txt (the best program as readable rules).

    poetry run python evolve_gp.py --workers 2 --panel-dir runs/league2/field --out runs/league2/gp
"""

import argparse
import copy
import json
import os
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from aisoccer.brainspec import load_brain, panel_from_dir
from aisoccer.game import Game
from aisoccer.gp.v1 import FUNCTIONS, TERMINALS, GPBrain, depth, node_type, size, to_text

BRAIN = "aisoccer.gp.v1.GPBrain"
OUT_DIR = Path(os.environ.get("LEAGUE_DIR", "runs/league")) / "gp"

FUNCS_BY_TYPE = {t: [f for f, s in FUNCTIONS.items() if s[0] == t] for t in "vnb"}
TERMS_BY_TYPE = {t: [n for n, tt in TERMINALS.items() if tt == t] for t in "vnb"}
SAME_SHAPE = {
    f: [g for g, s in FUNCTIONS.items() if s[0] == FUNCTIONS[f][0] and s[1] == FUNCTIONS[f][1]]
    for f in FUNCTIONS
}


def log(out, message):
    print(message, flush=True)
    with open(out / "progress.log", "a") as f:
        f.write(message + "\n")


# --- random programs and genetic operators ---


def random_constant(rng):
    if rng.random() < 0.4:
        return round(rng.uniform(-2, 2), 2)
    return float(round(rng.uniform(-400, 1800), -1))


def random_terminal(rng, t):
    if t == "n" and rng.random() < 0.7:
        return random_constant(rng)
    return rng.choice(TERMS_BY_TYPE[t])


def random_tree(rng, t, max_depth, full):
    """A random program of type t: 'full' fills every branch to max_depth, 'grow' may stop early."""
    if max_depth <= 1 or (not full and rng.random() < 0.3):
        return random_terminal(rng, t)
    name = rng.choice(FUNCS_BY_TYPE[t])
    return [name] + [random_tree(rng, a, max_depth - 1, full) for a in FUNCTIONS[name][1]]


def ramped(rng, count, min_depth=2, max_depth=6, max_size=60):
    programs = []
    depths = list(range(min_depth, max_depth + 1))
    while len(programs) < count:
        d = depths[len(programs) % len(depths)]
        p = random_tree(rng, "v", d, full=len(programs) % 2 == 0)
        if size(p) <= max_size:
            programs.append(p)
    return programs


def paths(node, path=()):
    """Every node's path (child indices from the root) and type."""
    yield path, node_type(node)
    if isinstance(node, list):
        for i, c in enumerate(node[1:], start=1):
            yield from paths(c, path + (i,))


def get(node, path):
    for i in path:
        node = node[i]
    return node


def replace(node, path, new):
    if not path:
        return new
    node = copy.deepcopy(node)
    parent = get(node, path[:-1])
    parent[path[-1]] = new
    return node


def crossover(rng, a, b):
    """a with one subtree replaced by a subtree of b of the same type."""
    donors = list(paths(b))
    types = {t for _, t in donors}
    path, t = rng.choice([(p, t) for p, t in paths(a) if t in types])  # the roots are both points
    donors = [p for p, tt in donors if tt == t]
    return replace(a, path, copy.deepcopy(get(b, rng.choice(donors))))


def subtree_mutation(rng, a):
    path, t = rng.choice(list(paths(a)))
    return replace(a, path, random_tree(rng, t, rng.randint(1, 4), full=False))


def point_mutation(rng, a, rate=0.15):
    a = copy.deepcopy(a)
    all_paths = list(paths(a))
    chosen = [p for p in all_paths if rng.random() < rate] or [rng.choice(all_paths)]
    for path, t in chosen:
        node = get(a, path)
        if isinstance(node, list):
            node[0] = rng.choice(SAME_SHAPE[node[0]])
        elif isinstance(node, (int, float)):
            c = float(node)
            if abs(c) <= 2:
                new = round(c + rng.gauss(0, 0.3), 2)
            else:
                new = float(round(c + rng.gauss(0, 0.2 * abs(c) + 20)))
            a = replace(a, path, new)
        else:
            a = replace(a, path, random_terminal(rng, t))
    return a


def breed(rng, population, score, args):
    """The next generation: elites, then children from tournaments on `score`."""
    order = sorted(range(len(population)), key=lambda i: -score[i])
    children = [population[i] for i in order[: args.elite]]

    def pick():
        contenders = rng.sample(range(len(population)), args.tournament)
        return population[max(contenders, key=lambda i: score[i])]

    while len(children) < len(population):
        parent = pick()
        for _ in range(10):
            r = rng.random()
            if r < args.crossover:
                child = crossover(rng, parent, pick())
            elif r < args.crossover + (1 - args.crossover) / 2:
                child = subtree_mutation(rng, parent)
            else:
                child = point_mutation(rng, parent)
            if size(child) <= args.max_size and depth(child) <= args.max_depth:
                break
        else:
            child = copy.deepcopy(parent)
        children.append(child)
    return children


# --- fitness ---


def play(task):
    program, opponent, gp_blue, seed, ticks = task
    a = GPBrain("GP", program=program)
    b = load_brain(opponent)
    blue, red = (a, b) if gp_blue else (b, a)
    score = Game(blue, red, game_length=ticks, quiet_mode=True, seed=seed).play()
    return (score["blue"], score["red"]) if gp_blue else (score["red"], score["blue"])


def evaluate(workers, population, panel, seeds, ticks):
    """Points per game + 0.5 x goal difference per game, on shared kick-offs."""
    tasks = [
        (p, opp, g % 2 == 0, int(seeds[g]), ticks)
        for p in population
        for opp in panel
        for g in range(len(seeds))
    ]
    results = workers.map(play, tasks, chunksize=4) if workers is not None else list(map(play, tasks))
    results = np.array(results, dtype=float)
    results = results.reshape(len(population), len(panel) * len(seeds), 2)
    gd = results[:, :, 0] - results[:, :, 1]
    points = np.where(gd > 0, 3.0, np.where(gd == 0, 1.0, 0.0))
    return points.mean(axis=1) + 0.5 * gd.mean(axis=1)


def write_best(out, program, generation, fitness, panel):
    spec = {
        "class": BRAIN,
        "kwargs": {"program": program},
        "generation": generation,
        "fitness": float(fitness),
        "size": size(program),
    }
    tmp = out / "best.tmp.json"
    tmp.write_text(json.dumps(spec))
    os.replace(tmp, out / "best.json")
    text = (
        f"# GPBrain program: generation {generation}, fitness {fitness:.2f}, "
        f"{size(program)} nodes\n# panel: {', '.join(Path(p).stem for p in panel)}\n"
        "# Every player runs to the point this returns (see aisoccer/gp/v1.py for the words).\n"
        + to_text(program)
        + "\n"
    )
    tmp = out / "best.tmp.txt"
    tmp.write_text(text)
    os.replace(tmp, out / "best.txt")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--games", type=int, default=4, help="per panel opponent (even: both sides)")
    parser.add_argument("--panel-dir", help="play against brains from this folder (e.g. the league field)")
    parser.add_argument("--panel-size", type=int, default=4)
    parser.add_argument("--panel", nargs="*", help="brain specs to play instead of --panel-dir")
    parser.add_argument("--random-init", action="store_true", help="accepted; GP always starts from random programs")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--ticks", type=int, default=1800, help="game length for fitness games")
    parser.add_argument("--elite", type=int, default=2)
    parser.add_argument("--tournament", type=int, default=4)
    parser.add_argument("--crossover", type=float, default=0.7, help="share of children by crossover; the rest mutate")
    parser.add_argument("--max-size", type=int, default=60, help="most nodes in a program")
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--parsimony", type=float, default=0.004, help="fitness cost per node, for selection")
    parser.add_argument("--generations", type=int, default=0, help="stop after this many (0: run forever)")
    parser.add_argument("--seed", type=int, help="random seed (default: random)")
    return parser.parse_args(argv)


def run(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    state_file = out / "population.json"
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    if state_file.exists():
        state = json.loads(state_file.read_text())
        population, generation = state["population"], state["generation"]
        log(out, f"GP resumed at generation {generation}")
    else:
        population, generation = ramped(rng, args.population, max_size=args.max_size), 0
        log(out, f"GP started: {args.population} random programs (ramped half-and-half)")
    if not (args.panel or args.panel_dir):
        raise SystemExit("give --panel-dir or --panel")

    workers = Pool(args.workers) if args.workers > 1 else None
    try:
        stop = generation + args.generations if args.generations else None
        while stop is None or generation < stop:
            generation += 1
            panel = args.panel or panel_from_dir(args.panel_dir, args.panel_size, nprng)
            seeds = nprng.integers(2**31, size=args.games)
            fitness = evaluate(workers, population, panel, seeds, args.ticks)
            sizes = np.array([size(p) for p in population])
            score = fitness - args.parsimony * sizes
            best = int(np.argmax(score))
            write_best(out, population[best], generation, fitness[best], panel)
            log(
                out,
                f"generation {generation:4d}: best fitness {fitness[best]:.2f} "
                f"(size {sizes[best]}), median {np.median(fitness):.2f}, "
                f"mean size {sizes.mean():.1f}",
            )
            population = breed(rng, population, score, args)
            tmp = out / "population.tmp.json"
            tmp.write_text(json.dumps({"generation": generation, "population": population}))
            os.replace(tmp, state_file)
    finally:
        if workers is not None:
            workers.close()
            workers.join()


if __name__ == "__main__":
    run(parse_args())
