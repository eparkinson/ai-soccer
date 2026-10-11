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

--vocab v2 evolves programs in GPBrain v2's larger vocabulary (aisoccer/gp/v2.py: v1 plus
intercept, strike(aim), ball_in(n), front_mate, bank_top, bank_bottom, save_point); the default, v1,
is unchanged. --seed-programs puts given programs (from best.json or population.json
files) into the first generation in place of some random ones, e.g. v1's best into a v2 run.

Training against the whole population (all opt-in; without them nothing changes):

- --panel all: every generation, every program plays every brain in --panel-dir (the
  league field), not --panel-size random ones; --exclude leaves out named brains (the
  evolver's own entry) and their copies, and exact copies of one brain (e.g. a champion
  that is also an entrant) count once. The files are snapshotted into <out>/panel at the start of each
  generation, so every program meets the same versions even if the league replaces a file,
- --ticks-per-opponent N: N ticks against each opponent (default 3600 with --panel all,
  else --games games of --ticks), split into an even number of games of --ticks or less;
  the games come in pairs on one kick-off seed with the colours swapped,
- --fitness gd: goal difference per 1000 ticks against each opponent, averaged over the
  opponents with equal weight (the default, points, is points per game plus half the
  goal difference per game).
- --weight-by family --families runs/league3/roster.json (with --fitness gd): the goal
  differences are weighted per strategy family instead of per opponent, so a family with
  many brains in the field counts no more than one with few: every family present gets
  an equal share, split equally among its brains (see family_weights). The roster's
  "families" ({brain: family}) and the league's state.json beside it (champion_names:
  {champ-N: the brain it was promoted from}) are re-read every generation.

Every program, the carried-over elites too, is evaluated afresh each generation on that
generation's panel and seeds (common random numbers), so fitness values compare within a
generation and never mix across fitness measures (population.json holds no fitness).

--vocab v3 (aisoccer/gp/v3.py) evolves role-based programs: a list of roles, each with a
BY expression (who fills it) and a TARGET (where its players go). Besides the ordinary
operators, --structural of the children come from a role-level operator (add, delete,
duplicate-and-mutate, move, take a role from the other parent, cross two roles; see
v3.structural). v1 and v2 seed programs are translated (v3.from_v2: identical play),
duplicates are dropped, and if a seed file is a best.json, it is written (translated) as
<out>/best.json at generation 0, so the entrant starts as the seeded best.

Less noisy selection (opt-in, with --fitness gd; without them nothing changes):

- --reeval-top K --reeval-reps R: after every program has played the panel once, the K
  best (distinct programs) play it R more times on fresh seeds (common to the K), and
  elites and tournament winners are chosen on the mean of all their evaluations; the K
  re-evaluated rank above every other program. Identical programs pool their evaluations.
  The log shows the best's mean fitness and how many evaluations it rests on (not the
  maximum of single evaluations, which carries a winner's curse).
- --reeval-memory M: a program that survives into later generations (an elite, or an
  identical child) also counts its last M evaluations from earlier generations
  (population.json keeps them).
- --xg-weight W (--xg-model runs/league3/xg.json): fitness adds W x the expected-goals
  difference per 1000 ticks (aisoccer/gp/xgame.py), weighted over opponents like the
  goal difference: a denser signal that separates programs in 0-0 games.

Outputs in --out: progress.log (best fitness and program size each generation),
population.json (the resumable state), best.json (a brain spec for
aisoccer/brainspec.py load_brain) and best.txt (the best program as readable rules).

    poetry run python evolve_gp.py --workers 2 --panel-dir runs/league2/field --out runs/league2/gp
"""

import argparse
import copy
import importlib
import json
import os
import random
import re
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from aisoccer.balance import Record, tilt
from aisoccer.brainspec import load_brain, panel_from_dir
from aisoccer.game import Game
from aisoccer.gp.v1 import FUNCTIONS, TERMINALS, GPBrain, depth, size

BRAIN = "aisoccer.gp.v1.GPBrain"
OUT_DIR = Path(os.environ.get("LEAGUE_DIR", "runs/league")) / "gp"

FUNCS_BY_TYPE = {t: [f for f, s in FUNCTIONS.items() if s[0] == t] for t in "vnb"}
TERMS_BY_TYPE = {t: [n for n, tt in TERMINALS.items() if tt == t] for t in "vnb"}
SAME_SHAPE = {
    f: [g for g, s in FUNCTIONS.items() if s[0] == FUNCTIONS[f][0] and s[1] == FUNCTIONS[f][1]]
    for f in FUNCTIONS
}
VOCABS = {"v1": "aisoccer.gp.v1.GPBrain", "v2": "aisoccer.gp.v2.GPBrainV2", "v3": "aisoccer.gp.v3.GPBrainV3"}
GENE_TYPES = ("v", "n", "b")  # the node types ordinary operators work on (v3's roles are not)


class Vocab:
    """One GP language (--vocab): its words, typing, printing and brain class."""

    def __init__(self, name):
        module, _, cls = VOCABS[name].rpartition(".")
        mod = importlib.import_module(module)
        self.name, self.brain, self.module, self.cls = name, VOCABS[name], module, getattr(mod, cls)
        self.functions, self.terminals = mod.FUNCTIONS, mod.TERMINALS
        self.node_type, self.validate, self.to_text = mod.node_type, mod.validate, mod.to_text
        # optional (v3): word weights, a random-program maker, role-level operators, v2 translation
        fw, tw = getattr(mod, "FUNCTION_WEIGHTS", {}), getattr(mod, "TERMINAL_WEIGHTS", {})
        self.random_program = getattr(mod, "random_program", None)
        self.structural = getattr(mod, "structural", None)
        self.convert = getattr(mod, "convert", None)
        self.funcs_by_type = {
            t: [f for f, s in self.functions.items() if s[0] == t for _ in range(fw.get(f, 1))] for t in "vnb"
        }
        self.terms_by_type = {
            t: [n for n, tt in self.terminals.items() if tt == t for _ in range(tw.get(n, 1))] for t in "vnb"
        }
        self.same_shape = {
            f: [g for g, s in self.functions.items() if s[0] == self.functions[f][0] and s[1] == self.functions[f][1]]
            for f in self.functions
        }


_VOCAB_CACHE = {}


def vocab(name="v1"):
    if name not in _VOCAB_CACHE:
        _VOCAB_CACHE[name] = Vocab(name)
    return _VOCAB_CACHE[name]


V1 = vocab("v1")


def log(out, message):
    print(message, flush=True)
    with open(out / "progress.log", "a") as f:
        f.write(message + "\n")


# --- random programs and genetic operators ---


def random_constant(rng):
    if rng.random() < 0.4:
        return round(rng.uniform(-2, 2), 2)
    return float(round(rng.uniform(-400, 1800), -1))


def random_terminal(rng, t, words=V1):
    if t == "n" and rng.random() < 0.7:
        return random_constant(rng)
    return rng.choice(words.terms_by_type[t])


def random_tree(rng, t, max_depth, full, words=V1):
    """A random program of type t: 'full' fills every branch to max_depth, 'grow' may stop early."""
    if max_depth <= 1 or (not full and rng.random() < 0.3):
        return random_terminal(rng, t, words)
    name = rng.choice(words.funcs_by_type[t])
    return [name] + [random_tree(rng, a, max_depth - 1, full, words) for a in words.functions[name][1]]


def ramped(rng, count, min_depth=2, max_depth=6, max_size=60, words=V1):
    programs = []
    depths = list(range(min_depth, max_depth + 1))
    while len(programs) < count:
        d = depths[len(programs) % len(depths)]
        full = len(programs) % 2 == 0
        if words.random_program is not None:
            p = words.random_program(rng, d, full, lambda t, dd, f: random_tree(rng, t, dd, f, words))
        else:
            p = random_tree(rng, "v", d, full=full, words=words)
        if size(p) <= max_size:
            programs.append(p)
    return programs


def paths(node, path=(), words=V1):
    """Every node's path (child indices from the root) and type."""
    yield path, words.node_type(node)
    if isinstance(node, list):
        for i, c in enumerate(node[1:], start=1):
            yield from paths(c, path + (i,), words)


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


def crossover(rng, a, b, words=V1):
    """a with one subtree replaced by a subtree of b of the same type."""
    donors = list(paths(b, words=words))
    types = {t for _, t in donors}
    path, t = rng.choice(
        [(p, t) for p, t in paths(a, words=words) if t in types and t in GENE_TYPES]
    )  # the roots are both points (v3: both programs share point, number and yes/no nodes)
    donors = [p for p, tt in donors if tt == t]
    return replace(a, path, copy.deepcopy(get(b, rng.choice(donors))))


def subtree_mutation(rng, a, words=V1):
    path, t = rng.choice([(p, t) for p, t in paths(a, words=words) if t in GENE_TYPES])
    return replace(a, path, random_tree(rng, t, rng.randint(1, 4), full=False, words=words))


def point_mutation(rng, a, rate=0.15, words=V1):
    a = copy.deepcopy(a)
    all_paths = [(p, t) for p, t in paths(a, words=words) if t in GENE_TYPES]
    chosen = [p for p in all_paths if rng.random() < rate] or [rng.choice(all_paths)]
    for path, t in chosen:
        node = get(a, path)
        if isinstance(node, list):
            node[0] = rng.choice(words.same_shape[node[0]])
        elif isinstance(node, (int, float)):
            c = float(node)
            if abs(c) <= 2:
                new = round(c + rng.gauss(0, 0.3), 2)
            else:
                new = float(round(c + rng.gauss(0, 0.2 * abs(c) + 20)))
            a = replace(a, path, new)
        else:
            a = replace(a, path, random_terminal(rng, t, words))
    return a


def breed(rng, population, score, args):
    """The next generation: elites, then children from tournaments on `score`."""
    words = vocab(getattr(args, "vocab", "v1"))
    order = sorted(range(len(population)), key=lambda i: -score[i])
    children = [population[i] for i in order[: args.elite]]

    def pick():
        contenders = rng.sample(range(len(population)), args.tournament)
        return population[max(contenders, key=lambda i: score[i])]

    def mutate(program):
        if rng.random() < 0.5:
            return subtree_mutation(rng, program, words)
        return point_mutation(rng, program, words=words)

    while len(children) < len(population):
        parent = pick()
        for _ in range(10):
            if words.structural is not None and rng.random() < getattr(args, "structural", 0.2):
                child, _ = words.structural(
                    rng, parent, pick(), lambda t, d: random_tree(rng, t, d, False, words), mutate,
                    lambda a, b: crossover(rng, a, b, words),
                )
                if size(child) <= args.max_size and depth(child) <= args.max_depth:
                    break
                continue
            r = rng.random()
            if r < args.crossover:
                child = crossover(rng, parent, pick(), words)
            elif r < args.crossover + (1 - args.crossover) / 2:
                child = subtree_mutation(rng, parent, words)
            else:
                child = point_mutation(rng, parent, words=words)
            if size(child) <= args.max_size and depth(child) <= args.max_depth:
                break
        else:
            child = copy.deepcopy(parent)
        children.append(child)
    return children


# --- fitness ---


def play(task):
    program, opponent, gp_blue, seed, ticks = task[:5]
    brain = vocab(task[5]).cls if len(task) > 5 else GPBrain
    a = brain("GP", program=program)
    b = load_brain(opponent)
    blue, red = (a, b) if gp_blue else (b, a)
    score = Game(blue, red, game_length=ticks, quiet_mode=True, seed=seed).play()
    return (score["blue"], score["red"]) if gp_blue else (score["red"], score["blue"])


def play_xg(task):
    """play, also returning expected goals: (goals for, against, xG for, xG against) for the GP side."""
    from aisoccer.gp.xgame import play_xg as xgame

    program, opponent, gp_blue, seed, ticks, name, model = task
    a = vocab(name).cls("GP", program=program)
    b = load_brain(opponent)
    score, xg = xgame(*((a, b) if gp_blue else (b, a)), ticks, seed, model)
    me, them = ("blue", "red") if gp_blue else ("red", "blue")
    return score[me], score[them], xg[me], xg[them]


def evaluate(workers, population, panel, seeds, ticks, words=V1):
    """Points per game + 0.5 x goal difference per game, on shared kick-offs."""
    tasks = [
        (p, opp, g % 2 == 0, int(seeds[g]), ticks, words.name)
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


FITNESS_LABELS = {
    "points": "points per game + 0.5 x goal difference per game",
    "gd": "goal difference per 1000 ticks, averaged over opponents (equal weight each)",
}
FAMILY_LABEL = "goal difference per 1000 ticks, averaged over families (equal weight each, split within a family)"


def game_plan(ticks_per_opponent, max_ticks):
    """(games, ticks per game): an even number of games of at most max_ticks, about ticks_per_opponent in all."""
    games = max(2, -(-ticks_per_opponent // max_ticks))
    games += games % 2
    return games, ticks_per_opponent // games


def spec_key(data, suffix):
    """What makes two brain files the same brain: the class and arguments, or the weights."""
    if suffix == ".json":
        spec = json.loads(data)
        return json.dumps({k: spec[k] for k in ("class", "kwargs", "chromosome") if k in spec}, sort_keys=True)
    return data


def field_panel(directory, exclude=(), snapshot=None):
    """
    Every brain file in a directory (the league field) except the excluded names and
    their copies, with exact duplicates counted once (the first by name is kept). With
    `snapshot`, the files are copied there first (atomically) and the copies returned, so
    a whole generation plays the same versions while the league updates the directory.
    """
    brains = []
    for path in sorted(Path(directory).glob("*")):
        if path.suffix in (".npz", ".json") and ".tmp" not in path.name:
            try:
                data = path.read_bytes()
            except FileNotFoundError:  # removed by the league between listing and reading
                continue
            brains.append((path, data, spec_key(data, path.suffix)))
    seen = {key for path, _, key in brains if path.stem in exclude}
    chosen = []
    for path, data, key in brains:
        if key in seen:
            continue
        seen.add(key)
        if snapshot is not None:
            snapshot = Path(snapshot)
            snapshot.mkdir(parents=True, exist_ok=True)
            tmp = snapshot / f"{path.stem}.tmp{path.suffix}"
            tmp.write_bytes(data)
            os.replace(tmp, snapshot / path.name)
            path = snapshot / path.name
        chosen.append(str(path))
    if snapshot is not None:
        names = {Path(p).name for p in chosen}
        for old in Path(snapshot).glob("*"):
            if old.name not in names:
                old.unlink(missing_ok=True)
    return chosen


# --- weighting opponents by family (--weight-by family) ---

CHAMPION = re.compile(r"(^|-)champ-\d+$")  # champ-N.json, PPO-champ-N.npz
CLASS_FAMILIES = [  # a champion's family from its class, when the league has no record of its origin
    ("aisoccer.gp.", "GP"),
    ("aisoccer.llm.", "LLM"),
    ("aisoccer.tactics.v1.", "LLM"),
    ("aisoccer.brains.PPOBrain.", "PPO"),
    ("aisoccer.tactics.learned.", "PPO"),
]
UNKNOWN_CHAMPIONS = "champions"


def read_league(roster_file, attempts=3):
    """The roster's families ({brain: family}) and state.json's champion_names beside it."""
    roster_file = Path(roster_file)
    for attempt in range(attempts):
        try:
            families = json.loads(roster_file.read_text()).get("families") or {}
            state_file = roster_file.with_name("state.json")
            names = (json.loads(state_file.read_text()).get("champion_names") or {}) if state_file.exists() else {}
            return families, names
        except (OSError, ValueError):  # mid-write: try again shortly
            if attempt == attempts - 1:
                raise
            time.sleep(1.0)


def class_family(path):
    """A champion file's family from its brain class: a .npz is a PPOBrain."""
    path = Path(path)
    if path.suffix == ".npz":
        return "PPO"
    try:
        cls = json.loads(path.read_text()).get("class", "")
    except (OSError, ValueError):
        return UNKNOWN_CHAMPIONS
    return next((family for prefix, family in CLASS_FAMILIES if cls.startswith(prefix)), UNKNOWN_CHAMPIONS)


def brain_family(path, families, champion_names):
    """
    The family of a panel brain, by file stem: the roster's family for the name; for a
    champion snapshot, the family of the brain it was promoted from (champion_names), else
    inferred from its class (GP / LLM / PPO, or "champions"); any other brain is a family
    of its own.
    """
    name = Path(path).stem
    if name in families:
        return families[name]
    if name in champion_names:
        original = champion_names[name]
        return families.get(original, original)
    if CHAMPION.search(name):
        return class_family(path)
    return name


def family_weights(panel, families, champion_names):
    """
    Weights over the panel (summing to 1): every family present weighs the same, split
    equally among its brains. Returns (weights, {family: [brain names]}).
    """
    members = {}
    for path in panel:
        members.setdefault(brain_family(path, families, champion_names), []).append(Path(path).stem)
    weights = np.array(
        [1.0 / (len(members) * len(members[brain_family(p, families, champion_names)])) for p in panel]
    )
    return weights, members


def describe_weights(panel, weights, members):
    weight = {Path(p).stem: w for p, w in zip(panel, weights)}
    return "; ".join(
        f"{family} {sum(weight[n] for n in names):.3f} ({', '.join(f'{n} {weight[n]:.3f}' for n in names)})"
        for family, names in sorted(members.items())
    )


def evaluate_panel(workers, population, panel, seeds, ticks, words=V1, measure="gd"):
    """
    Every program against every opponent on the same games: seeds[o] holds one kick-off
    seed per pair of games against opponent o, played once with each colour. Returns the
    fitness (measure "gd": goal difference per 1000 ticks per opponent, averaged over
    opponents; "points": points per game + 0.5 x goal difference per game) and the goal
    difference per 1000 ticks against each opponent (programs x opponents).
    """
    seeds = np.asarray(seeds).reshape(len(panel), -1)
    games = 2 * seeds.shape[1]
    tasks = [
        (p, opp, g % 2 == 0, int(seeds[o, g // 2]), ticks, words.name)
        for p in population
        for o, opp in enumerate(panel)
        for g in range(games)
    ]
    results = workers.map(play, tasks, chunksize=4) if workers is not None else list(map(play, tasks))
    results = np.array(results, dtype=float).reshape(len(population), len(panel), games, 2)
    gd = results[..., 0] - results[..., 1]
    per_opponent = gd.sum(axis=2) * 1000.0 / (games * ticks)
    if measure == "gd":
        return per_opponent.mean(axis=1), per_opponent
    points = np.where(gd > 0, 3.0, np.where(gd == 0, 1.0, 0.0))
    return (points + 0.5 * gd).mean(axis=(1, 2)), per_opponent


def evaluate_seeded(workers, programs, panel, seeds, ticks, words, xg_model=None):
    """
    Every program against every opponent, for each repetition r on its own seeds:
    seeds[r, o] holds one kick-off seed per pair of games (one per colour) against
    opponent o, the same for every program. Returns the goal difference per 1000 ticks
    (repetitions x programs x opponents) and, with an xG model path (or "" for the
    hand-made chance), the xG difference per 1000 ticks likewise (else None).
    """
    seeds = np.asarray(seeds)
    reps, _, pairs = seeds.shape
    games = 2 * pairs
    extra = () if xg_model is None else (xg_model,)
    tasks = [
        (p, opp, g % 2 == 0, int(seeds[r, o, g // 2]), ticks, words.name) + extra
        for r in range(reps)
        for p in programs
        for o, opp in enumerate(panel)
        for g in range(games)
    ]
    fn = play if xg_model is None else play_xg
    results = workers.map(fn, tasks, chunksize=4) if workers is not None else list(map(fn, tasks))
    results = np.array(results, dtype=float).reshape(reps, len(programs), len(panel), games, -1)
    scale = 1000.0 / (games * ticks)
    gd = (results[..., 0] - results[..., 1]).sum(axis=3) * scale
    xgd = None if xg_model is None else (results[..., 2] - results[..., 3]).sum(axis=3) * scale
    return gd, xgd


REEVALUATED = 1e6  # added to the selection score of re-evaluated programs: they rank first


def denoised_generation(workers, population, panel, seeds, nprng, game_ticks, words, weights, args, history):
    """
    One generation's evaluation with --reeval-top / --reeval-memory / --xg-weight: every
    program plays the panel once (seeds), the best args.reeval_top distinct programs
    args.reeval_reps more times; identical programs pool their evaluations, and each also
    counts its last args.reeval_memory evaluations from earlier generations (history:
    {program json: [fitness, ...]}).

    Returns (fitness: each program's mean fitness, selection score, evaluations: each
    program's current evaluations, per-opponent goal difference means {key: array}, xG
    difference means {key: float} or None, screen: the single first evaluation of each).
    """
    xg = args.xg_model if args.xg_weight else None
    if xg is None and args.xg_weight:
        xg = ""
    gd, xgd = evaluate_seeded(workers, population, panel, seeds[None], game_ticks, words, xg)
    keys = [json.dumps(p) for p in population]
    evals, opp, xgs = {}, {}, {}

    def note(key, gd_row, xg_row):
        f = float(gd_row @ weights)
        x = None if xg_row is None else float(xg_row @ weights)
        f += args.xg_weight * x if x is not None else 0.0
        evals.setdefault(key, []).append(f)
        opp.setdefault(key, []).append(gd_row)
        if x is not None:
            xgs.setdefault(key, []).append(x)
        return f

    screen = np.array([note(key, gd[0, i], None if xgd is None else xgd[0, i]) for i, key in enumerate(keys)])
    sizes = np.array([size(p) for p in population])

    def mean(key):
        values = evals[key] + (history.get(key, [])[-args.reeval_memory:] if args.reeval_memory else [])
        return float(np.mean(values))

    top = []
    if args.reeval_top:
        order = sorted(range(len(population)), key=lambda i: -(mean(keys[i]) - args.parsimony * sizes[i]))
        for i in order:
            if keys[i] not in {keys[j] for j in top}:
                top.append(i)
            if len(top) == args.reeval_top:
                break
        more = nprng.integers(2**31, size=(args.reeval_reps, len(panel), seeds.shape[1]))
        gd_r, xgd_r = evaluate_seeded(workers, [population[i] for i in top], panel, more, game_ticks, words, xg)
        for r in range(args.reeval_reps):
            for j, i in enumerate(top):
                note(keys[i], gd_r[r, j], None if xgd_r is None else xgd_r[r, j])
    fitness = np.array([mean(k) for k in keys])
    chosen = {keys[i] for i in top}
    score = fitness - args.parsimony * sizes + np.array([REEVALUATED if k in chosen else 0.0 for k in keys])
    opp_means = {k: np.mean(v, axis=0) for k, v in opp.items()}
    xg_means = {k: float(np.mean(v)) for k, v in xgs.items()} if xgs else None
    return fitness, score, evals, opp_means, xg_means, screen


def write_initial_best(out, program, words, source, data):
    """<out>/best.json and best.txt at generation 0: the seeded best, as translated."""
    spec = {"class": words.brain, "kwargs": {"program": program}, "generation": 0, "size": size(program),
            "seeded_from": str(source)}
    if "fitness" in data:
        spec["fitness"] = float(data["fitness"])
    tmp = out / "best.tmp.json"
    tmp.write_text(json.dumps(spec))
    os.replace(tmp, out / "best.json")
    tmp = out / "best.tmp.txt"
    tmp.write_text(
        f"# {words.cls.__name__} program: generation 0, the seeded best from {source} "
        f"(generation {data.get('generation')}), translated; {size(program)} nodes\n"
        + words.to_text(program) + "\n"
    )
    os.replace(tmp, out / "best.txt")


def write_best(out, program, generation, fitness, panel, words=V1, measure=None, weight_by=None, extra=None):
    spec = {
        "class": words.brain,
        "kwargs": {"program": program},
        "generation": generation,
        "fitness": float(fitness),
        "size": size(program),
    }
    if measure is not None:
        spec["fitness_measure"] = measure
    if weight_by is not None:
        spec["fitness_weight_by"] = weight_by
    if extra:
        spec.update(extra)
    tmp = out / "best.tmp.json"
    tmp.write_text(json.dumps(spec))
    os.replace(tmp, out / "best.json")
    text = (
        f"# {words.cls.__name__} program: generation {generation}, "
        f"{'fitness' if measure in (None, 'points') else 'goal difference per 1000 ticks'} {fitness:.2f}, "
        f"{size(program)} nodes\n# panel: {', '.join(Path(p).stem for p in panel)}\n"
        f"# Every player runs to the point this returns (see {words.module.replace('.', '/')}.py for the words).\n"
        + words.to_text(program)
        + "\n"
    )
    tmp = out / "best.tmp.txt"
    tmp.write_text(text)
    os.replace(tmp, out / "best.txt")


def seed_programs(files, words):
    """
    The programs in best.json files (kwargs.program) and population.json files, validated.
    A language that translates older programs (v3) gets them translated, duplicates dropped.
    """
    programs = []
    for f in files:
        data = json.loads(Path(f).read_text())
        found = data["population"] if "population" in data else [data["kwargs"]["program"]]
        if words.convert is not None:
            found = [words.convert(p) for p in found]
        for p in found:
            words.validate(p)
        programs += found
    if words.convert is not None:
        unique = {}
        for p in programs:
            unique.setdefault(json.dumps(p), p)
        programs = list(unique.values())
    return programs


def seed_labels(files):
    """The seed files, with the generation each was saved at, for the log."""
    labels = []
    for f in files:
        data = json.loads(Path(f).read_text())
        labels.append(f"{f} (generation {data['generation']})" if "generation" in data else str(f))
    return ", ".join(labels)


ALL_PANEL_TICKS = 3600  # ticks per opponent with --panel all: 2 games of 1800, one per colour


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--games", type=int, default=4, help="per panel opponent (even: both sides)")
    parser.add_argument("--panel-dir", help="play against brains from this folder (e.g. the league field)")
    parser.add_argument("--panel-size", type=int, default=4)
    parser.add_argument(
        "--panel", nargs="*",
        help="brain specs to play instead of --panel-dir; 'all': every brain in --panel-dir every generation",
    )
    parser.add_argument(
        "--exclude", nargs="*", default=[], help="with --panel all: brain names to leave out (the own entry)"
    )
    parser.add_argument(
        "--ticks-per-opponent", type=int,
        help="ticks against each opponent, in an even number of games of at most --ticks "
        "(default: 3600 with --panel all, else --games games of --ticks)",
    )
    parser.add_argument(
        "--fitness", choices=sorted(FITNESS_LABELS), default="points",
        help="points: points per game + 0.5 x goal difference per game; "
        "gd: goal difference per 1000 ticks, averaged over opponents",
    )
    parser.add_argument(
        "--weight-by", choices=["opponent", "family"], default="opponent",
        help="with --fitness gd: opponent (each opponent weighs the same) or family (each strategy family "
        "in --families weighs the same, split equally among its brains)",
    )
    parser.add_argument(
        "--balance", type=float, default=None, metavar="TARGET",
        help="with --fitness gd and --reeval-top: tilt the opponent weights each generation so that the best "
        "program's recent goal difference against the weighted panel is TARGET per 1000 ticks (about level, "
        "a small edge), still meeting weaker and stronger brains (aisoccer/balance.py)",
    )
    parser.add_argument("--balance-floor", type=float, default=0.2, help="share of the weights kept on the base weights")
    parser.add_argument("--balance-memory", type=float, default=10, help="generations the best's record fades over")
    parser.add_argument(
        "--families", help="with --weight-by family: the league's roster.json (its 'families'; champion "
        "names come from state.json beside it), re-read every generation",
    )
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
    parser.add_argument("--vocab", choices=sorted(VOCABS), default="v1", help="GP language (aisoccer/gp/<vocab>.py)")
    parser.add_argument(
        "--seed-programs", nargs="*", default=[],
        help="best.json / population.json files whose programs start the first generation (the rest random)",
    )
    parser.add_argument(
        "--structural", type=float, default=0.2,
        help="v3: share of children made by a role-level operator (add, delete, duplicate, move, take, combine)",
    )
    parser.add_argument(
        "--reeval-top", type=int, default=0,
        help="with --fitness gd: re-evaluate the best K programs each generation and select on their mean (0: off)",
    )
    parser.add_argument("--reeval-reps", type=int, default=3, help="with --reeval-top: extra evaluations each")
    parser.add_argument(
        "--reeval-memory", type=int, default=0,
        help="with --fitness gd: also count a surviving program's last M evaluations from earlier generations",
    )
    parser.add_argument(
        "--xg-weight", type=float, default=0.0,
        help="with --fitness gd: add this x the expected-goals difference per 1000 ticks to the fitness",
    )
    parser.add_argument("--xg-model", help="with --xg-weight: the xG model (fit_xg.py), e.g. runs/league3/xg.json")
    args = parser.parse_args(argv)
    args.panel_all = args.panel == ["all"]
    if args.ticks_per_opponent is None and args.panel_all:
        args.ticks_per_opponent = ALL_PANEL_TICKS
    return args


def run(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    state_file = out / "population.json"
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    words = vocab(args.vocab)
    if state_file.exists():
        state = json.loads(state_file.read_text())
        if state.get("vocab", "v1") != args.vocab:
            raise SystemExit(f"{state_file} holds {state.get('vocab', 'v1')} programs, not --vocab {args.vocab}")
        population, generation = state["population"], state["generation"]
        log(out, f"GP resumed at generation {generation}")
        if state.get("fitness", "points") != args.fitness:
            log(
                out,
                f"fitness changed from {state.get('fitness', 'points')} to {args.fitness}: every program "
                f"is evaluated afresh (population.json holds programs only, no fitness)",
            )
    elif args.seed_programs:
        seeded = seed_programs(args.seed_programs, words)[: args.population]
        population, generation = seeded + ramped(
            rng, args.population - len(seeded), max_size=args.max_size, words=words
        ), 0
        log(
            out,
            f"GP ({args.vocab}) started: {len(seeded)} programs seeded from {seed_labels(args.seed_programs)}, "
            f"{args.population - len(seeded)} random programs (ramped half-and-half)",
        )
        if words.convert is not None:
            log(out, f"  seeds translated to {args.vocab} (same play), duplicates dropped; "
                     f"population[0:{len(seeded)}] are seeds, the rest random")
            best_files = [f for f in args.seed_programs if "kwargs" in json.loads(Path(f).read_text())]
            if best_files and not (out / "best.json").exists():
                data = json.loads(Path(best_files[0]).read_text())
                write_initial_best(out, words.convert(data["kwargs"]["program"]), words, best_files[0], data)
                log(out, f"  best.json at generation 0: {best_files[0]}'s program, translated")
    else:
        population, generation = ramped(rng, args.population, max_size=args.max_size, words=words), 0
        log(out, f"GP started: {args.population} random programs (ramped half-and-half)")
    if not (args.panel or args.panel_dir):
        raise SystemExit("give --panel-dir or --panel")
    if args.panel_all and not args.panel_dir:
        raise SystemExit("--panel all needs --panel-dir")
    by_family = args.weight_by == "family"
    if by_family and (args.fitness != "gd" or not args.families):
        raise SystemExit("--weight-by family needs --fitness gd and --families <roster.json>")
    if by_family:
        league = read_league(args.families)  # fail now, not after a generation of games
    legacy = not (args.panel_all or args.ticks_per_opponent or args.fitness != "points")
    denoise = bool(args.reeval_top or args.reeval_memory or args.xg_weight)
    if denoise and (legacy or args.fitness != "gd"):
        raise SystemExit("--reeval-top, --reeval-memory and --xg-weight need --fitness gd")
    history = state.get("history", {}) if denoise and state_file.exists() else {}
    if args.balance is not None and not denoise:
        raise SystemExit("--balance needs --reeval-top (or --reeval-memory)")
    record = Record(state.get("balance", {}) if state_file.exists() else {}, args.balance_memory, args.balance or 0.0)
    if not legacy:
        games, game_ticks = game_plan(args.ticks_per_opponent or args.games * args.ticks, args.ticks)
        who = (
            f"every brain in {args.panel_dir}"
            + (f" except {', '.join(args.exclude)}" if args.exclude else "")
            + " (exact duplicates once)"
            if args.panel_all
            else ", ".join(args.panel) if args.panel else f"{args.panel_size} random brains from {args.panel_dir}"
        )
        label = FAMILY_LABEL if by_family else FITNESS_LABELS[args.fitness]
        log(
            out,
            f"fitness: {label}; panel: {who}; {games * game_ticks} ticks per opponent "
            f"({games} games of {game_ticks} ticks, colours alternated, same seeds for every program)",
        )
        if by_family:
            log(out, f"families from {args.families} (champion names from state.json beside it), every generation")
        if state_file.exists() and state.get("weight_by", "opponent") != args.weight_by:
            log(out, f"opponent weighting changed from {state.get('weight_by', 'opponent')} to {args.weight_by}")
        if denoise:
            log(
                out,
                f"selection: the best {args.reeval_top} programs re-evaluated {args.reeval_reps} more times "
                f"on fresh seeds, selected on the mean" if args.reeval_top else "selection: one evaluation each",
            )
            if args.reeval_memory:
                log(out, f"  surviving programs also count their last {args.reeval_memory} evaluations")
            if args.xg_weight:
                log(out, f"  fitness + {args.xg_weight} x expected-goals difference per 1000 ticks "
                         f"(xG model {args.xg_model or 'hand-made'})")

    workers = Pool(args.workers) if args.workers > 1 else None
    shown = None
    try:
        stop = generation + args.generations if args.generations else None
        while stop is None or generation < stop:
            generation += 1
            if legacy:
                panel = args.panel or panel_from_dir(args.panel_dir, args.panel_size, nprng)
                seeds = nprng.integers(2**31, size=args.games)
                fitness = evaluate(workers, population, panel, seeds, args.ticks, words)
            else:
                if args.panel_all:
                    panel = field_panel(args.panel_dir, set(args.exclude), snapshot=out / "panel")
                else:
                    panel = args.panel or panel_from_dir(args.panel_dir, args.panel_size, nprng)
                seeds = nprng.integers(2**31, size=(len(panel), games // 2))
                if denoise:
                    if by_family:
                        try:
                            league = read_league(args.families)
                        except (OSError, ValueError) as e:
                            log(out, f"  could not read {args.families} ({e}): families as last generation")
                        weights, members = family_weights(panel, *league)
                        text = describe_weights(panel, weights, members)
                        if text != shown:
                            log(out, f"  weights by family (from generation {generation}): {text}")
                            shown = text
                    else:
                        weights = np.full(len(panel), 1.0 / len(panel))
                    if args.balance is not None:
                        stems = [Path(o).stem for o in panel]
                        recent = record.values(stems)
                        weights = tilt(weights, recent, args.balance, args.balance_floor)
                        log(
                            out,
                            f"  balance: best's recent goal difference vs the weighted panel {weights @ recent:+.2f} "
                            "per 1000 ticks; weights "
                            + ", ".join(f"{n} {w:.2f} ({v:+.1f})" for n, w, v in sorted(
                                zip(stems, weights, recent), key=lambda r: -r[1])),
                        )
                    fitness, score, evals, opp, xgs, screen = denoised_generation(
                        workers, population, panel, seeds, nprng, game_ticks, words, weights, args, history
                    )
                    sizes = np.array([size(p) for p in population])
                    best = int(np.argmax(score))
                    key = json.dumps(population[best])
                    values = evals[key] + (history.get(key, [])[-args.reeval_memory:] if args.reeval_memory else [])
                    n = len(values)
                    spread = np.std(values, ddof=1) / np.sqrt(n) if n > 1 else float("nan")
                    write_best(
                        out, population[best], generation, fitness[best], panel, words, args.fitness,
                        args.weight_by if by_family else None, {"evaluations": n},
                    )
                    log(
                        out,
                        f"generation {generation:4d}: best fitness {fitness[best]:.2f} ± {spread:.2f} "
                        f"(size {sizes[best]}, mean of {n} evaluations), screen max {screen.max():.2f}, "
                        f"median {np.median(screen):.2f}, mean size {sizes.mean():.1f}",
                    )
                    log(
                        out,
                        f"  best's goal difference per 1000 ticks vs {len(panel)} brains (mean of this generation's "
                        f"{len(evals[key])}): "
                        + ", ".join(f"{Path(o).stem} {v:+.2f}" for o, v in zip(panel, opp[key])),
                    )
                    if xgs is not None:
                        log(out, f"  best's expected-goals difference per 1000 ticks: {xgs[key]:+.2f}")
                    if args.balance is not None:
                        for o, v in zip(panel, opp[key]):
                            record.add(Path(o).stem, v)
                    past = {k: (history.get(k, []) + v)[-args.reeval_memory:] for k, v in evals.items()}
                    population = breed(rng, population, score, args)
                    history = {k: past[k] for k in (json.dumps(p) for p in population) if k in past}
                    state = {"generation": generation, "population": population, "vocab": args.vocab,
                             "fitness": args.fitness}
                    if by_family:
                        state["weight_by"] = args.weight_by
                    if args.reeval_memory:
                        state["history"] = history
                    if args.balance is not None:
                        state["balance"] = record.data
                    tmp = out / "population.tmp.json"
                    tmp.write_text(json.dumps(state))
                    os.replace(tmp, state_file)
                    continue
                fitness, per_opponent = evaluate_panel(
                    workers, population, panel, seeds, game_ticks, words, args.fitness
                )
                if by_family:
                    try:
                        league = read_league(args.families)
                    except (OSError, ValueError) as e:
                        log(out, f"  could not read {args.families} ({e}): families as last generation")
                    weights, members = family_weights(panel, *league)
                    fitness = per_opponent @ weights
                    text = describe_weights(panel, weights, members)
                    if text != shown:
                        log(out, f"  weights by family (from generation {generation}): {text}")
                        shown = text
            sizes = np.array([size(p) for p in population])
            score = fitness - args.parsimony * sizes
            best = int(np.argmax(score))
            write_best(
                out, population[best], generation, fitness[best], panel, words, None if legacy else args.fitness,
                args.weight_by if by_family else None,
            )
            log(
                out,
                f"generation {generation:4d}: best fitness {fitness[best]:.2f} "
                f"(size {sizes[best]}), median {np.median(fitness):.2f}, "
                f"mean size {sizes.mean():.1f}",
            )
            if not legacy:
                log(
                    out,
                    f"  best's goal difference per 1000 ticks vs {len(panel)} brains: "
                    + ", ".join(f"{Path(o).stem} {v:+.2f}" for o, v in zip(panel, per_opponent[best])),
                )
            population = breed(rng, population, score, args)
            tmp = out / "population.tmp.json"
            state = {"generation": generation, "population": population}
            if args.vocab != "v1":
                state["vocab"] = args.vocab
            if args.fitness != "points":
                state["fitness"] = args.fitness
            if by_family:
                state["weight_by"] = args.weight_by
            tmp.write_text(json.dumps(state))
            os.replace(tmp, state_file)
    finally:
        if workers is not None:
            workers.close()
            workers.join()


if __name__ == "__main__":
    run(parse_args())
