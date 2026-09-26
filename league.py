"""
League training: many different approaches trained at once, judged by a shared league.

Entrants (see LEARNERS, plus the coach and the GA):

- PPO learners (train_ppo.py processes), each a different bet: steady settings, an
  explorer, a statistics-shaped reward, a bigger network distilled from the champion,
  a long horizon, goals-only reward, and a fresh start from the cloned policy.
- The coach (coach.py): a mixed team, each player role taken from the best brain for
  that role.
- The GA (evolve.py): an evolved, interpretable GeneticBrain.
- The current champion.

Every --round-minutes the coordinator pauses all of them and plays a league round on
every CPU core:

1. A Swiss tournament (--swiss-rounds rounds, --swiss-games games per pairing) of the
   entrants and the fixed field: the heuristic brains and every saved PPOBrain version.
2. A head to head of every entrant against the champion (--champion-games games, with
   team statistics recorded), the common yardstick for promotion and replacement.

Then it:

- promotes an entrant to champion if it beats the champion head to head significantly
  and scores at least as many Swiss points per game; the champion is saved to
  aisoccer/brains/weights/history/ and PPOBrain.npz and joins the fixed field,
- replaces a learner (population-based training) only if it has had at least
  --min-age iterations since its last change, is significantly behind the leader
  against the champion, and has not improved for two rounds. It adopts the leader's
  networks with mutated settings, but keeps what defines its approach (its "keep"
  settings), so every approach always has a version in the league,
- shares every entrant with the learners as sparring partners,
- runs the full verdict tournament (ppo_tournament.py) when a new champion also tops
  the Swiss, and stops if the verdict agrees.

    poetry run python league.py --hours 8
    tail -f runs/league/progress.log
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from aisoccer.brains.PPOBrain import DEFAULT_REWARD, PPOBrain
from aisoccer.brainspec import HEURISTICS, load_brain
from aisoccer.game import Game, GameResult
from aisoccer.stats import play_with_stats
from aisoccer.tournament import Tournament

LEAGUE_DIR = Path(os.environ.get("LEAGUE_DIR", "runs/league"))
LOG_FILE = LEAGUE_DIR / "progress.log"
# Where champions are kept. A second league (e.g. one that starts from zero) sets
# LEAGUE_HISTORY to keep its own champions.
HISTORY_DIR = Path(os.environ.get("LEAGUE_HISTORY", PPOBrain.WEIGHTS_FILE.parent / "history"))
LEARNING_ONLY = False  # set by --learning-only: no heuristic brains anywhere


BASE = {
    "lr": 1e-4,
    "min_std": 0.25,
    "target_kl": 0.01,
    "self_play": 0.15,
    "snapshots": 0.1,
    "pool_share": 0.2,
    "reward": dict(DEFAULT_REWARD),
}
CHAMPION_START = "champion"  # start a learner from the current champion's networks
A_START = LEAGUE_DIR / "A" / "policy.npz"

# Each learner is a different bet. "keep" lists what defines its approach: replacement
# keeps these settings (mutated, for reward terms) instead of copying the leader's.
LEARNERS = {
    "A": {
        "about": "steady: the settings that produced the current best",
        "config": BASE,
    },
    "B": {"about": "steady, mutated by population-based training", "config": BASE},
    "C": {
        "about": "explorer: more noise, faster learning, more varied opponents",
        "config": {
            **BASE,
            "lr": 2e-4,
            "min_std": 0.35,
            "target_kl": 0.015,
            "self_play": 0.2,
            "snapshots": 0.15,
            "pool_share": 0.3,
        },
    },
    "S": {
        # Version 2 of the statistics approach. Version 1 (learner B at the start) also
        # cut the control reward and lost ground; this keeps A's reward and adds small
        # terms for the strongest within-matchup predictors of winning (analyse_stats.py).
        "about": "statistics: A's reward plus shot and spread terms",
        "config": {**BASE, "reward": {**DEFAULT_REWARD, "shot": 0.1, "spread": 0.05}},
        "keep": ["reward.shot", "reward.spread"],
        "start": A_START,
        "value_warmup": 5,
    },
    "D": {
        "about": "bigger network (256x256), distilled from the champion then PPO",
        "config": BASE,
        "args": [
            "--hidden",
            "256",
            "256",
            "--bc-games",
            "120",
            "--bc-epochs",
            "25",
            "--dagger-rounds",
            "2",
            "--dagger-games",
            "50",
            "--dagger-epochs",
            "10",
        ],
        "teacher": CHAMPION_START,
        "value_warmup": 10,
        "architecture": "256x256",
    },
    "E": {
        "about": "long horizon: discount 0.998, GAE lambda 0.97",
        "config": {**BASE, "gamma": 0.998, "lam": 0.97},
        "keep": ["gamma", "lam"],
        "start": A_START,
        "value_warmup": 10,
    },
    "F": {
        "about": "goals only: no reward shaping",
        "config": {
            **BASE,
            "reward": {"goal": 2.0, "progress": 0.0, "control": 0.0, "chase": 0.0},
        },
        "keep": ["reward"],
        "start": A_START,
        "value_warmup": 10,
    },
    "G": {
        "about": "fresh start from the behaviour-cloned policy, for diversity",
        "config": BASE,
        "start": Path("runs/ppo/it0010.npz"),
        "value_warmup": 5,
    },
}

SWISS_ROUNDS = 4  # about log2(brains): enough to separate a clear leader
# Games are 36000 ticks (MATCH_TICKS; rounds 1-7 played 9000): four times the old length,
# so each count below is a quarter of what it was for the same amount of play.
SWISS_GAMES = 4  # games per Swiss pairing: 16 per brain
CHAMPION_GAMES = 5  # stage 1: every entrant against the champion
CONFIRM_GAMES = 16  # stage 2: extra games for the most promising entrants
CONFIRM_TOP = 3
# Waiting entrants: sparring partners for the learners once at most SPAR_GD goals per game
# better than the champion; full entry once "not significantly better" in two consecutive
# rounds: 95% CI of the goal difference includes 0 and the mean is at most ENTRY_GD.
# Both are goals per 9000 ticks of play (see per_9000), whatever MATCH_TICKS is.
SPAR_GD = 2.0
ENTRY_GD = 0.5
# Promotion pools the head to heads of the last PROMOTE_ROUNDS rounds (see pooled_record).
PROMOTE_ROUNDS = 3
TRAINING_NICE = 10  # training's CPU priority below the league's games (0)
EVAL_BATCH = 4  # league games without statistics are played this many at a time (VecGame)
SHOWCASE_CANDIDATES = 5  # recorded games per round, of which the most typical is kept
# Every PPOBrain beats these by 1.5-2.5 goals a game, so they cannot separate the
# candidates: they stay out of league rounds (the final verdict still includes them).
UNINFORMATIVE = {
    "RandomWalk",
    "SimpleBrain",
    "AdaptiveChaser",
    "LearningBrain",
    "PPO-scratch",
}


def log(message=""):
    print(message, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")


# ---------------------------------------------------------------- playing games


def make_brain(spec):
    return load_brain(spec)


# Length of a league match in ticks (5 standard games: 90 seconds of video at playback
# speed). Fewer, longer matches carry the same information as many short games (the
# scoring rate does not depend on game length) but far fewer end as draws. Worker
# processes read it from the environment, so league.py sets it there.
MATCH_TICKS = int(os.environ.get("LEAGUE_MATCH_TICKS", 36000))


def per_9000(goals_per_game):
    """Goals per game expressed per 9000 ticks of play, the unit the thresholds use."""
    return goals_per_game * 9000 / MATCH_TICKS


def t95(n):
    """Two-sided 95% t critical value for n samples (close approximation, no scipy)."""
    df = max(n - 1, 1)
    return 1.96 + 2.4 / df if df > 2 else {1: 12.71, 2: 4.30}[df]


def play(task):
    """(entrant spec, opponent spec, entrant plays blue, seed[, with stats]) -> result."""
    entrant, opponent, entrant_blue, seed = task[:4]
    with_stats = len(task) > 4 and task[4]
    a, b = make_brain(entrant), make_brain(opponent)
    blue, red = (a, b) if entrant_blue else (b, a)
    game = Game(blue, red, game_length=MATCH_TICKS, quiet_mode=True, seed=seed)
    me, them = ("blue", "red") if entrant_blue else ("red", "blue")
    if with_stats:
        score, stats = play_with_stats(game)
        return score[me], score[them], stats[me], stats[them]
    score = game.play()
    return score[me], score[them]


def play_batch(tasks):
    """
    Several games without statistics, played together in one VecGame (the batched
    simulator: the same games as play() bit for bit, much cheaper for PPOBrains).
    """
    from aisoccer.vecgame import VecGame

    blue, red = [], []
    for entrant, opponent, entrant_blue, _ in (t[:4] for t in tasks):
        a, b = make_brain(entrant), make_brain(opponent)
        blue.append(a if entrant_blue else b)
        red.append(b if entrant_blue else a)
    scores = VecGame(blue, red, [t[3] for t in tasks], game_length=MATCH_TICKS).play()
    return [
        (s["blue"], s["red"]) if t[2] else (s["red"], s["blue"]) for t, s in zip(tasks, scores)
    ]


def play_all(workers, tasks):
    """
    play() for every task, in order. Games without statistics go in VecGame batches of
    EVAL_BATCH; games with statistics are played one at a time.
    """
    plain = [i for i, t in enumerate(tasks) if not (len(t) > 4 and t[4])]
    stats = [i for i, t in enumerate(tasks) if len(t) > 4 and t[4]]
    batches = [plain[k : k + EVAL_BATCH] for k in range(0, len(plain), EVAL_BATCH)]  # noqa: E203
    items = [("batch", [tasks[i] for i in b]) for b in batches] + [("one", tasks[i]) for i in stats]
    order = [i for b in batches for i in b] + stats
    results = [r for out in workers.map(play_item, items, chunksize=1) for r in out]
    ordered = [None] * len(tasks)
    for i, r in zip(order, results):
        ordered[i] = r
    return ordered


def play_item(item):
    kind, payload = item
    return play_batch(payload) if kind == "batch" else [play(payload)]


def pooled_record(rounds):
    """
    Goal difference per game over several rounds' head to heads with the champion,
    each round given as {"n", "gd", "sd"}: the pooled mean with a 95% CI, and whether
    the entrant was ahead in every round.
    """
    n = sum(r["n"] for r in rounds)
    mean = sum(r["n"] * r["gd"] for r in rounds) / n
    within = sum((r["n"] - 1) * r["sd"] ** 2 for r in rounds)
    between = sum(r["n"] * (r["gd"] - mean) ** 2 for r in rounds)
    sd = np.sqrt((within + between) / max(n - 1, 1))
    return {
        "n": n,
        "gd": mean,
        "gd_ci": t95(n) * sd / np.sqrt(n),
        "ahead_every_round": all(r["gd"] > 0 for r in rounds),
    }


def round_summary(r):
    """{"n", "gd", "sd"} of a head to head record (sd from its CI half-width)."""
    n = int(r["n"])
    return {"n": n, "gd": float(r["gd"]), "sd": float(r["gd_ci"]) * np.sqrt(n) / t95(n) if n > 1 else 0.0}


def backfill_history(champion, round_number):
    """Head to head summaries from the saved rounds against the same champion."""
    history = {}
    for number in range(max(round_number - PROMOTE_ROUNDS + 1, 1), round_number):
        path = LEAGUE_DIR / "rounds" / f"round-{number:03d}.json"
        if not path.exists():
            continue
        saved = json.loads(path.read_text())
        if saved.get("champion") != champion or saved.get("match_ticks") != MATCH_TICKS:
            history = {}  # a different champion or game length: start again
            continue
        for name, r in saved.get("h2h", {}).items():
            history.setdefault(name, []).append(round_summary(r))
    return history


def record(games):
    """Points and goal difference per match, with 95% CI half-widths (t-distribution)."""
    games = np.array([g[:2] for g in games], dtype=float)  # (goals for, against)
    gd = games[:, 0] - games[:, 1]
    points = np.where(gd > 0, 3.0, np.where(gd == 0, 1.0, 0.0))
    n = len(games)
    return {
        "n": n,
        "wins": int((gd > 0).sum()),
        "draws": int((gd == 0).sum()),
        "losses": int((gd < 0).sum()),
        "points": points.mean(),
        "points_ci": t95(n) * points.std(ddof=1) / np.sqrt(n),
        "gd": gd.mean(),
        "gd_ci": t95(n) * gd.std(ddof=1) / np.sqrt(n),
    }


def same_brain(a, b):
    """True if two brain files play identically (same policy for every role)."""
    if Path(a).suffix != ".npz" or Path(b).suffix != ".npz":
        return (
            Path(a).read_bytes() == Path(b).read_bytes() if Path(a).exists() else a == b
        )
    wa, wb = PPOBrain.load_weights(a), PPOBrain.load_weights(b)

    def roles(w):
        return w.get("role_policies") or [w["policy"]] * 5

    return all(
        np.shape(x) == np.shape(y) and np.array_equal(x, y)
        for ra, rb in zip(roles(wa), roles(wb))
        for x, y in zip(ra, rb)
    )


# ---------------------------------------------------------------- processes


class Process:
    """A background process in its own process group, so it can be paused as a whole."""

    def __init__(self, cmd, log_path, threads="2"):
        self.process = subprocess.Popen(
            ["nice", "-n", str(TRAINING_NICE)] + list(map(str, cmd)),  # the league's games come first
            stdout=open(log_path, "a"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=dict(os.environ, OPENBLAS_NUM_THREADS=threads),
        )
        self.started = time.time()

    def signal(self, sig):
        if self.process.poll() is None:
            os.killpg(self.process.pid, sig)

    def exited(self):
        return self.process.poll() is not None

    def cpu_seconds(self):
        """CPU used so far by the whole process group (workers included, dead ones too)."""
        return group_cpu_seconds(self.process.pid)


def group_cpu_seconds(pgid):
    total = 0.0
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            stat = open(f"/proc/{entry.name}/stat").read()
        except OSError:
            continue
        fields = stat[stat.rindex(")") + 2 :].split()  # noqa: E203
        if int(fields[2]) == pgid:
            total += sum(int(v) for v in fields[11:15])  # user, system, and reaped children
    return total / os.sysconf("SC_CLK_TCK")


def account_cpu(state, processes, round_number, round_seconds):
    """
    Add each brain's CPU since the last round to state["cpu"] (seconds, all rounds).
    The first time, CPU used before accounting started is estimated from the current
    process's rate over the earlier rounds (marked in state["cpu_estimated_to"]).
    """
    cpu, seen = state.setdefault("cpu", {}), state.setdefault("cpu_seen", {})
    first = "cpu_estimated_to" not in state
    for name, process in processes.items():
        now = process.cpu_seconds()
        pid = process.process.pid
        last = seen.get(name, {})
        delta = now - last["cpu"] if last.get("pid") == pid else now
        if first:
            age = max(time.time() - process.started, 1.0)
            earlier = max(round_number - 1, 0) * round_seconds
            delta += now / age * earlier
        cpu[name] = cpu.get(name, 0.0) + max(delta, 0.0)
        seen[name] = {"pid": pid, "cpu": now}
    if first:
        state["cpu_estimated_to"] = round_number - 1


class Learner(Process):
    def __init__(self, name, spec, champion, args, saved=None):
        """:param saved: this learner's entry from state.json, to carry on after a restart."""
        self.name = name
        self.about = spec["about"]
        self.keep = spec.get("keep", [])
        self.architecture = spec.get("architecture", "128x128")
        self.dir = LEAGUE_DIR / name
        self.pool_dir = LEAGUE_DIR / "pool" / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.config = json.loads(
            json.dumps(saved["config"] if saved else spec["config"])
        )
        (self.dir / "config.json").write_text(json.dumps(self.config, indent=1))
        cmd = [
            sys.executable,
            "train_ppo.py",
            "--run-dir",
            str(self.dir),
            "--config",
            str(self.dir / "config.json"),
            "--pool-dir",
            str(self.pool_dir),
            "--workers",
            str(args.workers_per_learner),
            "--games",
            str(args.learner_games),
            # Batched simulator: each worker plays its share of the games together
            # (bit-identical to playing them one by one, about 2.6x less game CPU).
            "--vector-games",
            str(-(-args.learner_games // args.workers_per_learner)),
            "--iterations",
            "1000000",
            "--eval-every",
            "0",
            "--seed",
            str(zlib.crc32(name.encode()) % 100000),
        ] + spec.get("args", [])
        if (self.dir / "latest.npz").exists():
            cmd += ["--resume", "--value-warmup", "0"]
        elif spec.get("random") or champion is None:
            # Start from a random network: no cloning, no starting weights.
            cmd += ["--bc-games", "0", "--value-warmup", "0"]
        else:
            if spec.get("teacher") == CHAMPION_START:
                cmd += ["--bc-teacher", f"file:{champion}"]
            else:
                start = spec.get("start", champion)
                start = start if Path(start).exists() else champion
                copy = LEAGUE_DIR / f"start-{name}.npz"
                shutil.copy(start, copy)
                cmd += ["--init-weights", str(copy)]
            cmd += ["--value-warmup", str(spec.get("value_warmup", 0))]
        # One BLAS thread each: with many learners updating at once, more oversubscribes.
        super().__init__(cmd, self.dir / "stdout.log", threads="1")
        self.changed_at = (
            saved["changed_at"] if saved else None
        )  # last start or adoption
        self.ratings: list[float] = saved.get("ratings", []) if saved else []

    def state(self):
        return {
            "config": self.config,
            "changed_at": self.changed_at,
            "ratings": self.ratings,
        }

    def iteration(self):
        path = self.dir / "latest.npz"
        if not path.exists():
            return 0
        return int(np.load(path, allow_pickle=True)["state"].item()["iteration"])

    def adopt(self, weights_path, config):
        self.config = config
        (self.dir / "adopt.json").write_text(json.dumps(config))
        shutil.copy(weights_path, self.dir / "adopt.npz")  # the learner picks it up
        self.changed_at = self.iteration()
        self.ratings = []


def mutate(config, rng):
    """Perturb a learner's settings for population-based training."""
    new = json.loads(json.dumps(config))
    new["lr"] = float(np.clip(config["lr"] * rng.choice([0.5, 2.0]), 3e-5, 5e-4))
    new["min_std"] = float(
        np.clip(config["min_std"] * rng.choice([0.8, 1.25]), 0.15, 0.5)
    )
    new["target_kl"] = float(
        np.clip(config["target_kl"] * rng.choice([0.8, 1.25]), 0.005, 0.03)
    )
    for key in ("self_play", "snapshots", "pool_share"):
        new[key] = float(np.clip(config[key] + rng.choice([-0.05, 0.05]), 0.05, 0.4))
    for key in new["reward"]:
        if key != "goal":
            new["reward"][key] = float(config["reward"][key] * rng.choice([0.8, 1.25]))
    return round_floats(new)


def keep_approach(new, own, keep, rng):
    """Restore the settings that define a learner's approach after copying the leader's."""
    for key in keep:
        if key == "reward":
            new["reward"] = dict(own["reward"])
        elif key.startswith("reward."):
            term = key.split(".", 1)[1]
            if term in own["reward"]:
                new["reward"][term] = float(
                    own["reward"][term] * rng.choice([0.8, 1.25])
                )
        elif key in own:
            new[key] = own[key]
    return round_floats(new)


def round_floats(value):
    if isinstance(value, float):
        return float(f"{value:.4g}")
    if isinstance(value, dict):
        return {k: round_floats(v) for k, v in value.items()}
    return value


# ---------------------------------------------------------------- the league round


def fixed_field(entrant_specs):
    """The heuristic brains and saved brain versions, minus any entrant's own file."""
    field = {} if LEARNING_ONLY else {name: name for name in HEURISTICS if name not in UNINFORMATIVE}
    for path in sorted(
        list(HISTORY_DIR.glob("*.npz")) + list(HISTORY_DIR.glob("*champ-*.json"))
    ):
        if str(path) not in entrant_specs and path.stem not in UNINFORMATIVE:
            field[path.stem] = str(path)
    return field


def swiss_pairings(tournament, played, had_bye, rng):
    """
    One Swiss round: sort by points (ties broken at random), give the bye (if the
    number of brains is odd) to the lowest-ranked brain that has not had one, then pair
    top-down, each brain with the highest-ranked opponent it has not met yet.
    """
    table = sorted(tournament.tournament_scores.table.values(), key=lambda r: (-r["points"], rng.random()))
    order = [row["number"] for row in table]
    bye = None
    if len(order) % 2:
        bye = next((n for n in reversed(order) if n not in had_bye), order[-1])
        order.remove(bye)
        had_bye.add(bye)
    pairings = []
    while order:
        a = order.pop(0)
        b = next((n for n in order if tuple(sorted((a, n))) not in played), order[0])
        order.remove(b)
        pairings.append(tuple(sorted((a, b))))
    return pairings, bye


def swiss(workers, brains, rounds, games, seed):
    """
    Swiss tournament: brains is {name: spec}. Pairs by points with no repeat pairings
    where possible and rotating byes (swiss_pairings), and plays each round's games in
    parallel. Returns the Tournament, whose table and per-game results hold the outcome.
    """
    names = list(brains)
    tournament = Tournament(
        [SimpleNamespace(name=n) for n in names], legs=games, rounds=rounds, seed=seed
    )
    rng = np.random.default_rng(seed)
    played: set[tuple] = set()
    had_bye: set[int] = set()
    for _ in range(rounds):
        pairings, _bye = swiss_pairings(tournament, played, had_bye, rng)
        fixtures = tournament.fixtures(pairings)
        tasks = [(brains[names[b]], brains[names[r]], True, s) for b, r, s in fixtures]
        for (blue, red, _), (blue_goals, red_goals) in zip(
            fixtures, play_all(workers, tasks)
        ):
            tournament.tournament_scores.process((blue, red), (blue_goals, red_goals))
            tournament.results.append((blue, red, blue_goals, red_goals))
        played.update(pairings)
    return tournament


def swiss_standings(tournament):
    """Per brain: games, record and points per game with a 95% CI."""
    games: dict[int, list] = {}
    for blue, red, bg, rg in tournament.results:
        games.setdefault(blue, []).append((bg, rg))
        games.setdefault(red, []).append((rg, bg))
    rows = []
    for number, results in games.items():
        rows.append((tournament.brains[number].name, record(results)))
    return sorted(rows, key=lambda row: -row[1]["points"])


def against_champion(workers, entrants, champion, games, seed):
    """
    Every entrant plays the champion on the same kick-offs, with team statistics
    (stage 1). The CONFIRM_TOP best then play CONFIRM_GAMES more (stage 2), so a real
    edge of about 0.1 goals per game can be confirmed where it matters.
    """

    def run(names, n, seed, stats=True):
        seeds = np.random.default_rng(seed).integers(2**31, size=n)
        tasks = [
            (entrants[name], str(champion), g % 2 == 0, int(seeds[g]), stats)
            for name in names
            for g in range(n)
        ]
        results = play_all(workers, tasks)
        chunks = np.array_split(np.arange(len(results)), len(names)) if names else []
        return {name: [results[i] for i in idx] for name, idx in zip(names, chunks)}

    names = [n for n, spec in entrants.items() if not same_brain(spec, str(champion))]
    games_by = run(names, games, seed)
    stage1 = {n: record(g)["gd"] for n, g in games_by.items()}
    top = sorted(stage1, key=lambda n: -stage1[n])[:CONFIRM_TOP]
    if CONFIRM_GAMES:
        # Stage 2 without statistics (style comes from stage 1), so it can be batched.
        for name, more in run(top, CONFIRM_GAMES, seed + 10**6, stats=False).items():
            games_by[name] = games_by[name] + more
    return {
        n: {"record": record(g), "style": [r[2] for r in g if len(r) > 2]}
        for n, g in games_by.items()
    }


STYLE_STATS = ["control", "own_half", "final_third", "shots", "spread", "passes"]


def log_round(number, standings, entrants, h2h, iterations, notes, cpu=None):
    log(f"\nLEAGUE ROUND {number}  ({time.strftime('%H:%M')})")
    log(
        f"  Swiss tournament, {SWISS_ROUNDS} rounds of {SWISS_GAMES} matches of {MATCH_TICKS} ticks (* = entrant):"
    )
    log(
        f"    {'#':>2}  {'BRAIN':<24} {'GAMES':>5}  {'W-D-L':<12} {'GD/GAME':>13}  {'PTS/GAME':>12}  {'CPU-H':>6}"
    )
    for rank, (name, r) in enumerate(standings, 1):
        mark = "*" if name in entrants else " "
        wdl = f"{r['wins']}-{r['draws']}-{r['losses']}"
        log(
            f"    {rank:>2}{mark} {name:<24} {r['n']:>5}  {wdl:<12} {r['gd']:+.2f} ±{r['gd_ci']:.2f}"
            f"  {r['points']:>5.2f} ±{r['points_ci']:.2f}"
            + (f"  {cpu[name] / 3600:6.1f}" if cpu and name in cpu else "")
        )
    log(
        f"  Head to head with the champion ({CHAMPION_GAMES} matches, +{CONFIRM_GAMES} for the top"
        f" {CONFIRM_TOP}; + = entrant stronger), and style:"
    )
    log(
        f"    {'ENTRANT':<8} {'GAMES':>5}  {'W-D-L':<11} {'GD/GAME':>13} "
        + "".join(f"{s:>12}" for s in STYLE_STATS)
    )
    for name, result in sorted(h2h.items(), key=lambda kv: -kv[1]["record"]["gd"]):
        r = result["record"]
        style = "".join(
            f"{np.mean([g[s] for g in result['style']]):>12.2f}" for s in STYLE_STATS
        )
        wdl = f"{r['wins']}-{r['draws']}-{r['losses']}"
        log(
            f"    {name:<8} {r['n']:>5}  {wdl:<11} {r['gd']:+.2f} ±{r['gd_ci']:.2f} "
            + style
        )
    for note in notes:
        log(f"  {note}")


def save_round(
    number, tournament, standings, entrants, h2h, champion, notes, cpu=None, qualifying=None, champion_cpu=None
):
    """
    Save a round's results as JSON (rounds/round-NNN.json) for the dashboard and the
    blog: every Swiss match (a pairing's games, as totals), the standings and the
    head to heads with the champion.
    """
    matches = {}
    for blue, red, blue_goals, red_goals in tournament.results:
        a, b = tournament.brains[blue].name, tournament.brains[red].name
        key = tuple(sorted((a, b)))
        m = matches.setdefault(key, {"a": key[0], "b": key[1], "games": 0, "goals_a": 0, "goals_b": 0,
                                     "wins_a": 0, "wins_b": 0, "draws": 0})
        goals = {a: blue_goals, b: red_goals}
        m["games"] += 1
        m["goals_a"] += goals[key[0]]
        m["goals_b"] += goals[key[1]]
        if goals[key[0]] > goals[key[1]]:
            m["wins_a"] += 1
        elif goals[key[0]] < goals[key[1]]:
            m["wins_b"] += 1
        else:
            m["draws"] += 1
    data = {
        "round": number,
        "time": time.strftime("%H:%M"),
        "champion": champion.stem if champion else None,
        "entrants": sorted(entrants),
        "matches": list(matches.values()),
        "standings": [{"name": name, **{k: float(v) for k, v in r.items()}} for name, r in standings],
        "h2h": {n: {k: float(v) for k, v in r["record"].items()} for n, r in h2h.items()},
        "notes": notes,
        "match_ticks": MATCH_TICKS,
        # Compute so far per brain (CPU-hours; see account_cpu), and the waiting
        # brains' head to heads with the champion.
        "cpu_hours": {n: v / 3600 for n, v in (cpu or {}).items()},
        "champion_cpu_hours": {n: v / 3600 for n, v in (champion_cpu or {}).items()},
        "qualifying": qualifying or {},
    }
    rounds = LEAGUE_DIR / "rounds"
    rounds.mkdir(exist_ok=True)
    (rounds / f"round-{number:03d}.json").write_text(json.dumps(data, indent=1))


class Verdict:
    """
    The full verdict tournament (ppo_tournament.py, 300 games per pairing) on a new
    champion, run in the background so the league keeps going while it plays.
    """

    def __init__(self, champion):
        self.champion = champion.stem
        self.path = LEAGUE_DIR / f"verdict-{self.champion}.log"
        self.process = subprocess.Popen(
            [
                sys.executable,
                "ppo_tournament.py",
                "--legs",
                "300",
                "--champion",
                str(champion),
            ],
            stdout=open(self.path, "w"),
            stderr=subprocess.STDOUT,
            env=dict(os.environ, OPENBLAS_NUM_THREADS="1"),
            preexec_fn=lambda: os.nice(5),
        )
        self.reported = False

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()

    def result(self):
        """None while running; otherwise (passed, head-to-head text)."""
        if self.process.poll() is None:
            return None
        head_to_head = self.path.read_text().split("HEAD TO HEAD", 1)[-1]
        verdicts = re.findall(
            r"(weaker than|stronger than|no significant)", head_to_head
        )
        return (
            bool(verdicts) and all(v == "weaker than" for v in verdicts),
            head_to_head,
        )


def champion_number(path):
    return int(Path(path).stem.split("-")[-1])


def newest_champion():
    """The newest champion: PPO-champ-N.npz (a PPOBrain) or champ-N.json (any brain)."""
    return max(HISTORY_DIR.glob("*champ-*.*"), key=champion_number, default=None)


def promote(spec):
    """
    Save an entrant as the next champion. A PPOBrain becomes PPO-champ-N.npz (policy
    only) and PPOBrain.npz; any other brain becomes champ-N.json.
    """
    number = max((champion_number(p) for p in HISTORY_DIR.glob("*champ-*.*")), default=-1) + 1
    if str(spec).endswith(".npz"):
        champion = HISTORY_DIR / f"PPO-champ-{number}.npz"
        weights = PPOBrain.load_weights(spec)
        PPOBrain.save_weights(
            champion,
            {
                "policy": weights["policy"],
                "log_std": weights["log_std"],
                "role_policies": weights.get("role_policies"),
            },
        )
        shutil.copy(spec, PPOBrain.WEIGHTS_FILE)
    else:
        champion = HISTORY_DIR / f"champ-{number}.json"
        if str(spec).endswith(".json"):
            shutil.copy(spec, champion)
        else:
            champion.write_text(json.dumps({"class": str(spec).removeprefix("brain:")}))
    return champion


# ---------------------------------------------------------------- main loop


def record_showcase(task):
    """
    Play one game between two brains and record every tick, for replays and videos.
    Returns names, specs, positions (ticks x 11 x 2: blue players, red players, ball)
    and the score after every tick.
    """
    blue_name, blue_spec, red_name, red_spec, seed, length = task
    game = Game(
        load_brain(blue_spec, blue_name), load_brain(red_spec, red_name),
        game_length=length, quiet_mode=True, seed=seed,
    )
    positions, scores = [], []
    while True:
        if game.tick() == GameResult.end:
            break
        bodies = [p.body.position for team in game.teams for p in team.players]
        positions.append(np.array(bodies + [game.ball.body.position], dtype=np.float32))
        scores.append((game.score["blue"], game.score["red"]))
    return {
        "names": [blue_name, red_name],
        "specs": [str(blue_spec), str(red_spec)],
        "positions": np.array(positions),
        "scores": np.array(scores, dtype=np.int16),
    }


def seed_field(field_dir):
    """Generation-0 opponents for the evolving processes: untrained, random brains."""
    from aisoccer.brains.GeneticBrain import CHROMOSOME_LENGTH, GeneticBrain
    from aisoccer.brains.PPOBrain import ACT_DIM, OBS_DIM
    from aisoccer.ppo import PPO
    from aisoccer.tactics.v1 import STRATEGY

    field_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12345)
    PPOBrain.save_weights(field_dir / "random-ppo.npz", PPO(OBS_DIM, ACT_DIM, seed=1).weights())
    GeneticBrain.save(field_dir / "random-ga.json", rng.random(CHROMOSOME_LENGTH))
    strategy = {k: float(v) for k, v in zip(STRATEGY, rng.random(len(STRATEGY)))}
    (field_dir / "random-tactics.json").write_text(
        json.dumps({"class": "aisoccer.tactics.v1.TacticsV1", "kwargs": {"strategy": strategy}})
    )


def default_roster():
    """The roster written on first start; edit runs/league/roster.json to change it."""
    learners = {}
    for name, spec in LEARNERS.items():
        entry = json.loads(json.dumps(spec, default=str))
        learners[name] = entry
    return {
        "learners": learners,
        "processes": {
            "Coach": {
                "about": "mixed team: each player role from the best brain for it",
                "cmd": ["coach.py", "--workers", "2"],
                "entrant": str(LEAGUE_DIR / "coach" / "team.npz"),
            },
            "GA": {
                "about": "genetic algorithm evolving GeneticBrain chromosomes",
                "cmd": ["evolve.py", "--workers", "2"],
                "entrant": str(LEAGUE_DIR / "ga" / "best.json"),
            },
            "Tactics-v1-GA": {
                "about": "genetic algorithm tuning TacticsV1's team strategy vector",
                "cmd": [
                    "tune_tactics.py",
                    "--workers",
                    "3",
                    "--brain",
                    "aisoccer.tactics.v1.TacticsV1",
                ],
                "entrant": str(LEAGUE_DIR / "tactics_ga" / "best.json"),
            },
        },
        "fixed_entrants": {
            "Tactics-v1": "brain:aisoccer.tactics.v1.TacticsV1",
        },
        "settings": {"workers_per_learner": 2, "learner_games": 24},
    }


class Roster:
    """
    The league's entrants, re-read from roster.json at the start of every round so
    entrants can be added, removed or changed without stopping the league:

    - "learners": PPO learners (train_ppo.py processes), see LEARNERS,
    - "processes": other processes that keep improving a brain file, which plays as
      the entrant (e.g. the coach and the GAs): {"cmd": [...], "entrant": path},
    - "fixed_entrants": brains that play as they are: {name: brain spec},
    - "settings": workers per learner and games per learner iteration.

    A changed learner or process is restarted (learners resume from their networks).
    """

    def __init__(self, path, champion):
        self.path = path
        self.champion = champion
        if not path.exists():
            path.write_text(json.dumps(default_roster(), indent=1))
        self.data = json.loads(path.read_text())
        self.learners: dict[str, Learner] = {}
        self.processes: dict[str, Process] = {}
        self.started: dict[str, str] = {}  # name -> the entry it was started with

    def reload(self):
        try:
            self.data = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            log(
                f"  ROSTER: could not read {self.path} ({error}); keeping the previous roster"
            )

    def reconcile(self, state):
        """Start new entries, stop removed ones, restart changed ones. Returns notes."""
        notes = []
        settings = SimpleNamespace(**self.data.get("settings", {}))
        wanted = {
            **{("learner", n): e for n, e in self.data.get("learners", {}).items()},
            **{("process", n): e for n, e in self.data.get("processes", {}).items()},
            # Waiting entrants keep improving but only join the league once they qualify
            # (those with only a brain spec, e.g. LLMBrain versions, have no process).
            **{("process", n): e for n, e in self.data.get("waiting", {}).items() if "cmd" in e},
        }
        running = {("learner", n) for n in self.learners} | {
            ("process", n) for n in self.processes
        }
        for kind, name in running - set(wanted):
            self.stop(kind, name)
            notes.append(f"ROSTER: stopped {name} (removed from the roster)")
        for (kind, name), entry in wanted.items():
            key = json.dumps(entry, sort_keys=True)
            running_now = self.learners.get(name) or self.processes.get(name)
            if self.started.get(name) == key and not (running_now and running_now.exited()):
                continue
            restarting = name in self.started
            if restarting:
                self.stop(kind, name)
            if kind == "learner":
                saved = state.get("learners", {}).get(name)
                if restarting and saved and "config" in entry:
                    saved = {**saved, "config": entry["config"]}
                self.learners[name] = Learner(
                    name, entry, self.champion, settings, saved
                )
            else:
                cmd = [sys.executable] + entry["cmd"]
                self.processes[name] = Process(cmd, LEAGUE_DIR / f"{name}.stdout.log")
            self.started[name] = key
            notes.append(f"ROSTER: {'restarted' if restarting else 'started'} {name}")
        return notes

    def stop(self, kind, name):
        process = (
            self.learners.pop(name, None)
            if kind == "learner"
            else self.processes.pop(name, None)
        )
        if process:
            process.signal(signal.SIGCONT)
            process.signal(signal.SIGTERM)
        self.started.pop(name, None)

    def update(self, change):
        """Change the roster file safely: re-read it (it may have been edited since), apply
        `change` to the data, and write it back atomically."""
        self.reload()
        change(self.data)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1))
        os.replace(tmp, self.path)

    def everyone(self):
        return list(self.learners.values()) + list(self.processes.values())

    def entrant_sources(self):
        """name -> brain spec (a file to snapshot, or a spec string) for every entrant."""
        sources = {n: str(l.dir / "policy.npz") for n, l in self.learners.items()}
        for name, entry in self.data.get("processes", {}).items():
            if name in self.processes and entry.get("entrant"):
                sources[name] = entry["entrant"]
        sources.update(self.data.get("fixed_entrants", {}))
        return sources


def main():
    global SWISS_ROUNDS, SWISS_GAMES, CHAMPION_GAMES, CONFIRM_GAMES, LEARNING_ONLY, MATCH_TICKS
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--hours", type=float, default=0.0, help="0: run until stopped")
    parser.add_argument("--round-minutes", type=float, default=45.0)
    parser.add_argument(
        "--min-age", type=int, default=60, help="iterations before replacing"
    )
    parser.add_argument("--swiss-rounds", type=int, default=SWISS_ROUNDS)
    parser.add_argument("--swiss-games", type=int, default=SWISS_GAMES)
    parser.add_argument("--champion-games", type=int, default=CHAMPION_GAMES)
    parser.add_argument("--confirm-games", type=int, default=CONFIRM_GAMES)
    parser.add_argument(
        "--start",
        type=Path,
        default=PPOBrain.WEIGHTS_FILE,
        help="first champion, if there is none yet",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="log promotions, do not save them"
    )
    parser.add_argument(
        "--learning-only", action="store_true",
        help="no heuristic brains anywhere and no starting champion: round 1's Swiss winner is crowned",
    )
    parser.add_argument("--no-verdict", action="store_true", help="skip verdict tournaments")
    parser.add_argument("--record-every", type=int, default=1, help="record a showcase game every N rounds")
    parser.add_argument("--showcase-ticks", type=int, default=9000)
    parser.add_argument("--match-ticks", type=int, default=MATCH_TICKS, help="length of a league match")
    parser.add_argument(
        "--retire-after", type=int, default=8, help="no retirements before this round"
    )
    args = parser.parse_args()
    LEARNING_ONLY = args.learning_only
    MATCH_TICKS = args.match_ticks
    os.environ["LEAGUE_MATCH_TICKS"] = str(MATCH_TICKS)  # for the worker processes
    SWISS_ROUNDS, SWISS_GAMES = args.swiss_rounds, args.swiss_games
    CHAMPION_GAMES, CONFIRM_GAMES = args.champion_games, args.confirm_games

    LEAGUE_DIR.mkdir(parents=True, exist_ok=True)
    # Stop cleanly (learners included) on kill / SIGTERM, not only on Ctrl-C.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    rng = np.random.default_rng(0)

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    field_dir = LEAGUE_DIR / "field"
    if LEARNING_ONLY and not field_dir.exists():
        seed_field(field_dir)
    if not LEARNING_ONLY and not list(HISTORY_DIR.glob("*champ-*.*")):
        weights = PPOBrain.load_weights(args.start)
        PPOBrain.save_weights(
            HISTORY_DIR / "PPO-champ-0.npz",
            {"policy": weights["policy"], "log_std": weights["log_std"]},
        )
    champion = newest_champion()

    state_file = LEAGUE_DIR / "state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    roster = Roster(LEAGUE_DIR / "roster.json", champion)
    for note in roster.reconcile(state):
        log(note)
    log(
        f"League started (roster: {roster.path}); champion "
        f"{champion.stem if champion else 'none yet: round 1 crowns the Swiss winner'}"
    )

    # A learning-only league opens with a recorded game between two untrained brains:
    # the "round 0" everything else is compared with.
    replays = LEAGUE_DIR / "replays"
    if LEARNING_ONLY and args.record_every and not (replays / "round-000.npz").exists():
        replays.mkdir(exist_ok=True)
        replay = record_showcase(
            ("random-ppo", str(field_dir / "random-ppo.npz"), "random-tactics",
             str(field_dir / "random-tactics.json"), 900, args.showcase_ticks)
        )
        np.savez_compressed(replays / "round-000.npz", round=0, **replay)
        final = replay["scores"][-1]
        log(f"SHOWCASE recorded (round 0, untrained): random-ppo {final[0]} - {final[1]} random-tactics")

    deadline = time.time() + args.hours * 3600 if args.hours else float("inf")
    round_number = state.get("round", 0)
    pending = state.get("pending")  # entrant awaiting confirmation for promotion
    verdict = None
    verdict_log = LEAGUE_DIR / f"verdict-{champion.stem if champion else 'none'}.log"
    if champion and not args.dry_run and not args.no_verdict and not (
        verdict_log.exists() and "HEAD TO HEAD" in verdict_log.read_text()
    ):
        verdict = Verdict(champion)
        log(f"Verdict tournament on {champion.stem} started in the background")
    try:
        while time.time() < deadline:
            time.sleep(args.round_minutes * 60)
            round_number += 1

            if verdict and not verdict.reported and verdict.result() is not None:
                passed, head_to_head = verdict.result()
                verdict.reported = True
                log(f"\nVERDICT on {verdict.champion} (300 games per pairing):")
                log(head_to_head.rstrip())
                log(
                    "  SUCCESS: significantly stronger than every brain."
                    if passed
                    else "  Not yet significantly stronger than every brain."
                )

            for name, process in {**roster.learners, **roster.processes}.items():
                if process.exited():
                    log(
                        f"{name} exited ({process.process.returncode}); see its stdout.log"
                    )

            roster.reload()  # pick up roster edits made since the last round
            # Training keeps running while the round's games are played: the games use
            # frozen snapshots, and training runs at a lower priority (Process: nice), so
            # the games get the CPU they need and training fills whatever is left.
            everyone = roster.everyone()
            account_cpu(
                state, {**roster.learners, **roster.processes}, round_number, args.round_minutes * 60
            )
            try:
                notes = []
                learners = roster.learners
                iterations = {n: l.iteration() for n, l in learners.items()}
                for name, learner in learners.items():
                    if learner.changed_at is None:
                        learner.changed_at = iterations[name]

                # Snapshot every entrant (spec strings such as "brain:..." play as they are).
                snapshots = LEAGUE_DIR / "snapshots"
                snapshots.mkdir(exist_ok=True)
                entrants = {}
                for name, source in roster.entrant_sources().items():
                    if Path(source).suffix in (".npz", ".json"):
                        if not Path(source).exists():
                            continue
                        target = (
                            snapshots
                            / f"{name}-r{round_number:03d}{Path(source).suffix}"
                        )
                        shutil.copy(source, target)
                        entrants[name] = str(target)
                    else:
                        entrants[name] = source
                if champion:
                    entrants[champion.stem] = str(champion)

                # A fresh pool each round, so improved brain code plays straight away.
                with Pool(os.cpu_count()) as workers:
                    field = {**entrants, **fixed_field(set(entrants.values()))}
                    order = list(field)
                    rng.shuffle(order)
                    tournament = swiss(
                        workers,
                        {n: field[n] for n in order},
                        SWISS_ROUNDS,
                        SWISS_GAMES,
                        round_number,
                    )
                    standings = swiss_standings(tournament)
                    swiss_points = {name: r["points"] for name, r in standings}
                    h2h = (
                        against_champion(
                            workers,
                            {n: s for n, s in entrants.items() if n != champion.stem},
                            champion,
                            CHAMPION_GAMES,
                            seed=round_number,
                        )
                        if champion
                        else {}
                    )
                    # Showcase: the top two of the Swiss play several recorded games, and the
                    # most typical one is kept: its goal difference is closest to the pair's
                    # usual margin (their Swiss match this round if they met, otherwise the
                    # median of the candidates). A freak result would misrepresent them.
                    if args.record_every and round_number % args.record_every == 0:
                        (a, _), (b, _) = standings[0], standings[1]
                        showcase_games = workers.map(
                            record_showcase,
                            [
                                (a, field[a], b, field[b], 900 + round_number * 10 + k, args.showcase_ticks)
                                if k % 2 == 0
                                else (b, field[b], a, field[a], 900 + round_number * 10 + k, args.showcase_ticks)
                                for k in range(SHOWCASE_CANDIDATES)
                            ],
                        )

                        def margin(replay):  # goal difference from a's side
                            blue, red = replay["scores"][-1]
                            return (blue - red) if replay["names"][0] == a else (red - blue)

                        margins = [int(margin(c)) for c in showcase_games]
                        totals = [int(sum(c["scores"][-1])) for c in showcase_games]
                        met = [  # their Swiss games this round, from a's side
                            (bg - rg) if tournament.brains[blue].name == a else (rg - bg)
                            for blue, red, bg, rg in tournament.results
                            if {tournament.brains[blue].name, tournament.brains[red].name} == {a, b}
                        ]
                        usual = (
                            float(np.mean(met)) * args.showcase_ticks / MATCH_TICKS
                            if met
                            else float(np.median(margins))
                        )
                        typical_total = float(np.median(totals))
                        pick = min(
                            range(len(showcase_games)),
                            key=lambda k: (abs(margins[k] - usual), abs(totals[k] - typical_total)),
                        )
                        replay = showcase_games[pick]
                        replays = LEAGUE_DIR / "replays"
                        replays.mkdir(exist_ok=True)
                        np.savez_compressed(
                            replays / f"round-{round_number:03d}.npz",
                            round=round_number,
                            showcase_games=np.array(margins),
                            usual_margin=usual,
                            **replay,
                        )
                        final = replay["scores"][-1]
                        notes.append(
                            f"SHOWCASE recorded: {replay['names'][0]} {final[0]} - {final[1]} {replay['names'][1]} "
                            f"(the most typical of {len(showcase_games)} games: margins {margins} for {a}, "
                            f"usual margin {usual:+.1f}; replays/round-{round_number:03d}.npz)"
                        )

                # Promotion: over the last PROMOTE_ROUNDS rounds' head to heads together,
                # significantly better than the champion; ahead of it in each of those
                # rounds (so one lucky round cannot carry it); and at least as high as the
                # champion in this round's Swiss (so a specialist that only beats the
                # champion is not promoted). Pooling uses all the evidence: a brain that is
                # really better gets through, while luck rarely lasts three rounds.
                if "h2h_history" not in state and champion is not None:
                    state["h2h_history"] = backfill_history(champion.stem, round_number)
                history = state.setdefault("h2h_history", {})
                for name in list(history):
                    if name not in h2h:
                        history.pop(name)  # missed a round (retired, replaced): start again
                for name, r in h2h.items():
                    history[name] = (history.get(name, []) + [round_summary(r["record"])])[-PROMOTE_ROUNDS:]
                pooled = {n: pooled_record(rounds) for n, rounds in history.items()}
                candidates = [
                    n
                    for n, p in pooled.items()
                    if len(history[n]) == PROMOTE_ROUNDS
                    and p["gd"] - p["gd_ci"] > 0
                    and p["ahead_every_round"]
                    and swiss_points.get(n, 0) >= swiss_points.get(champion.stem, 0)
                ]
                best = max(candidates, key=lambda n: pooled[n]["gd"], default=None)
                new_champion = best
                if champion is None:
                    # The first champion: the best entrant of the first round's Swiss.
                    new_champion = next((n for n, _ in standings if n in entrants), None)
                if new_champion and champion is not None:
                    p = pooled[new_champion]
                    notes.append(
                        f"PROMOTED on {PROMOTE_ROUNDS} rounds pooled: {new_champion} GD {p['gd']:+.2f} "
                        f"±{p['gd_ci']:.2f} over {p['n']} games, ahead in every round"
                    )
                on_course = [
                    n for n, p in pooled.items() if p["ahead_every_round"] and p["gd"] > 0 and n != new_champion
                ]
                for n in sorted(on_course, key=lambda n: -pooled[n]["gd"]):
                    p = pooled[n]
                    notes.append(
                        f"ON COURSE: {n} ahead of the champion in {len(history[n])} of {PROMOTE_ROUNDS} rounds, "
                        f"pooled GD {p['gd']:+.2f} ±{p['gd_ci']:.2f} over {p['n']} games"
                    )
                pending = max(on_course, key=lambda n: pooled[n]["gd"], default=None)
                if new_champion:
                    state["h2h_history"] = {}  # the next champion is judged afresh

                # Replacement, rated against the champion (the same opponent for all).
                ratings = {n: h2h[n]["record"] for n in learners if n in h2h}
                for name, r in ratings.items():
                    learners[name].ratings.append(r["gd"])
                for architecture in {
                    learner.architecture for learner in learners.values()
                }:
                    group = [
                        n for n in ratings if learners[n].architecture == architecture
                    ]
                    if len(group) < 2:
                        continue
                    leader = max(group, key=lambda n: ratings[n]["gd"])
                    for name in group:
                        learner = learners[name]
                        if name == leader:
                            continue
                        age = iterations[name] - learner.changed_at
                        gap = ratings[leader]["gd"] - ratings[name]["gd"]
                        significant = gap > np.hypot(
                            ratings[leader]["gd_ci"], ratings[name]["gd_ci"]
                        )
                        r = learner.ratings
                        stalled = (
                            len(r) >= 3
                            and r[-1] <= max(r[:-1])
                            and r[-2] <= max(r[:-2])
                        )
                        if age >= args.min_age and significant and stalled:
                            config = keep_approach(
                                mutate(learners[leader].config, rng),
                                learner.config,
                                learner.keep,
                                rng,
                            )
                            learner.adopt(entrants[leader], config)
                            state.get("h2h_history", {}).pop(name, None)  # a new network: start again
                            notes.append(
                                f"REPLACED learner {name} (age {age}, {gap:.2f} goals/game behind "
                                f"{leader} vs the champion, not improving): adopts {leader}'s "
                                f"networks with {config}"
                            )
                            break

                for label_, path, keys in (
                    (
                        "Coach team (roles 0-4)",
                        LEAGUE_DIR / "coach" / "team.json",
                        None,
                    ),
                    ("GA", LEAGUE_DIR / "ga" / "best.json", ("generation", "fitness")),
                    ("Neuro-ES", LEAGUE_DIR / "neuro" / "best.json", ("generation", "fitness")),
                    (
                        "Tactics-v1-GA",
                        LEAGUE_DIR / "tactics_ga" / "best.json",
                        ("generation", "fitness"),
                    ),
                ):
                    if path.exists():
                        info = json.loads(path.read_text())
                        if keys:
                            info = {k: info.get(k) for k in keys}
                        notes.append(f"{label_}: {info}")

                # Qualifying: waiting entrants (brains with hand-written structure, which
                # keep improving in the background) spar with the learners once within
                # SPAR_GD of the champion, and join the league once on par (ENTRY_GD).
                waiting = {
                    n: e for n, e in roster.data.get("waiting", {}).items()
                    if e.get("spec") or (e.get("entrant") and Path(e["entrant"]).exists())
                }
                if champion and waiting:
                    snapshots = LEAGUE_DIR / "snapshots"
                    specs = {}
                    for name, entry in waiting.items():
                        if entry.get("spec"):
                            specs[name] = entry["spec"]
                            continue
                        target = snapshots / f"{name}-r{round_number:03d}{Path(entry['entrant']).suffix}"
                        shutil.copy(entry["entrant"], target)
                        specs[name] = str(target)
                    with Pool(os.cpu_count()) as workers:
                        qualifying = against_champion(
                            workers, specs, champion, CHAMPION_GAMES, seed=round_number + 5000
                        )
                    sparring, on_par = {}, state.get("on_par", {})
                    state["qualifying"] = {
                        n: {k: float(v) for k, v in q["record"].items()} for n, q in qualifying.items()
                    }
                    for name, result in qualifying.items():
                        r, entry = result["record"], roster.data["waiting"][name]
                        # "not_better" (default): joins only once not significantly better
                        # (see SPAR_GD/ENTRY_GD), so it cannot walk in and dominate.
                        # "not_worse": joins once not significantly worse.
                        rule = entry.get("admit", "not_better")
                        if rule == "not_worse":
                            ok, streak = r["gd"] + r["gd_ci"] >= 0, 1
                        else:
                            level = r["gd"] - r["gd_ci"] <= 0 and per_9000(r["gd"]) <= ENTRY_GD
                            streak = on_par.get(name, 0) + 1 if level else 0
                            ok = streak >= 2
                        on_par[name] = streak
                        if ok:

                            def admit(data, name=name, entry=entry):
                                data.get("waiting", {}).pop(name, None)
                                if entry.get("spec"):
                                    data.setdefault("fixed_entrants", {})[name] = entry["spec"]
                                else:
                                    data.setdefault("processes", {})[name] = entry

                            roster.update(admit)
                            on_par.pop(name, None)
                            notes.append(
                                f"ADMITTED {name}: GD {r['gd']:+.2f} ±{r['gd_ci']:.2f} against the champion "
                                "("
                                + ("not significantly better, two rounds running" if rule == "not_better" else "on par")
                                + "); "
                                "it joins the league next round"
                            )
                            continue
                        spar = rule == "not_better" and per_9000(r["gd"]) <= SPAR_GD
                        if spar:
                            sparring[name] = specs[name]
                        notes.append(
                            f"WAITING {name}: GD {r['gd']:+.2f} ±{r['gd_ci']:.2f} against the champion "
                            + (f"(on par {streak} of 2 rounds; " if streak else "(joins once on par; ")
                            + ("SPARRING with the learners)" if spar else f"spars at +{SPAR_GD:g}/9000 ticks or less)")
                        )
                    state["on_par"] = on_par
                    state["sparring"] = sparring
                else:
                    state["sparring"] = {}

                # Retirement: the tournament decides which versions drop out. A fixed or
                # process entrant in the bottom third of the Swiss for 3 rounds in a row
                # is retired (learners have their own replacement rule; champions stay).
                ranks = state.setdefault("ranks", {})
                places = {
                    name: k / len(standings) for k, (name, _) in enumerate(standings)
                }
                retirable = set(roster.data.get("fixed_entrants", {})) | set(
                    roster.data.get("processes", {})
                )
                for name in retirable & set(places) if round_number >= args.retire_after else ():
                    ranks[name] = (ranks.get(name, []) + [places[name]])[-3:]
                    if len(ranks[name]) == 3 and min(ranks[name]) >= 2 / 3:

                        def retire(data, name=name):
                            retired = data.setdefault("retired", {})
                            for section in ("fixed_entrants", "processes"):
                                if name in data.get(section, {}):
                                    retired[name] = {"from": section, "entry": data[section].pop(name)}

                        roster.update(retire)
                        ranks.pop(name)
                        notes.append(
                            f"RETIRED {name}: bottom third of the Swiss for 3 rounds in a row"
                        )

                log_round(
                    round_number, standings, entrants, h2h, iterations, notes,
                    cpu={**state.get("cpu", {}), **state.get("champion_cpu", {})},
                )
                save_round(
                    round_number, tournament, standings, entrants, h2h, champion, notes,
                    cpu=state.get("cpu"), qualifying=state.get("qualifying"),
                    champion_cpu=state.get("champion_cpu"),
                )

                if new_champion and args.dry_run:
                    log(f"  (dry run) would promote {new_champion}")
                elif new_champion:
                    champion = promote(entrants[new_champion])
                    roster.champion = champion
                    pending = None
                    for learner in learners.values():
                        learner.ratings = []  # ratings were against the old champion
                    log(f"  NEW CHAMPION: {new_champion} -> {champion.stem}")
                    # A champion's compute: what its brain had used when it was crowned.
                    state.setdefault("champion_cpu", {})[champion.stem] = state.get("cpu", {}).get(new_champion, 0.0)
                    if verdict:
                        verdict.stop()  # its champion has been superseded
                    if not args.no_verdict:
                        verdict = Verdict(champion)
                        log(f"  Verdict tournament on {champion.stem} started in the background")

                # Sparring partners for the learners: every other entrant and every champion.
                for name, learner in learners.items():
                    for old in learner.pool_dir.glob("*"):
                        old.unlink()
                    for other, spec in entrants.items():
                        if other != name and Path(spec).suffix in (".npz", ".json"):
                            shutil.copy(
                                spec, learner.pool_dir / f"{other}{Path(spec).suffix}"
                            )
                    for champ in HISTORY_DIR.glob("*champ-*.*"):
                        shutil.copy(champ, learner.pool_dir / champ.name)
                    # Waiting entrants within SPAR_GD of the champion: training opponents
                    # only (not in the Swiss, cannot become champion).
                    for other, spec in state.get("sparring", {}).items():
                        target = learner.pool_dir / f"spar-{other}.json"
                        if Path(spec).suffix in (".npz", ".json"):
                            shutil.copy(spec, target.with_suffix(Path(spec).suffix))
                        elif spec.startswith("brain:"):
                            target.write_text(json.dumps({"class": spec.removeprefix("brain:")}))

                # The field the evolving processes play against: every entrant's latest
                # version and every champion (in a learning-only league, nothing else).
                if LEARNING_ONLY:
                    for old in field_dir.glob("*"):
                        old.unlink()
                    for other, spec in entrants.items():
                        if Path(spec).suffix in (".npz", ".json"):
                            shutil.copy(spec, field_dir / f"{other}{Path(spec).suffix}")
                    for champ in HISTORY_DIR.glob("*champ-*.*"):
                        shutil.copy(champ, field_dir / champ.name)

                # Roster changes take effect now, between rounds.
                roster.reload()
                for note in roster.reconcile(state):
                    log(f"  {note}")

                state = {
                    "round": round_number,
                    "pending": pending,
                    "h2h_history": state.get("h2h_history", {}),
                    "ranks": state.get("ranks", {}),
                    "on_par": state.get("on_par", {}),
                    "sparring": state.get("sparring", {}),
                    "cpu": state.get("cpu", {}),
                    "cpu_seen": state.get("cpu_seen", {}),
                    "cpu_estimated_to": state.get("cpu_estimated_to"),
                    "champion_cpu": state.get("champion_cpu", {}),
                    "learners": {
                        **state.get("learners", {}),
                        **{n: l.state() for n, l in roster.learners.items()},
                    },
                }
                state_file.write_text(json.dumps(state, indent=1))
            finally:
                for process in everyone:
                    process.signal(signal.SIGCONT)
    finally:
        if verdict:
            verdict.stop()
        for process in roster.everyone():
            process.signal(signal.SIGTERM)
        log(f"League stopped. Champion: {champion.stem if champion else 'none'}")


if __name__ == "__main__":
    main()
