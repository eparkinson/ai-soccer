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
from aisoccer.constants import Constants
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
# What the title race rates (roster setting "title_rating"): "gd", goal difference per game
# against an average opponent (least squares), or "points", points per game (3 a win, 1 a
# draw, per game) with goal difference per game only breaking ties. Under "gd" a blowout of a
# much weaker brain counts in full, so the brain that runs up the biggest scores against the
# weakest one can lead without winning the matches that matter; under "points" a win is a
# win. Points need a full round robin to be fair (everyone meets the same opponents).
TITLE_RATING = "gd"
TIEBREAK = 0.001  # points per game per goal of goal difference per game: ties only
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


# The league's tunable rules: a roster "settings" key for each module setting. They are
# read again at the start of every round (apply_settings), so rule changes need no
# restart; the round's game workers get the same values (Pool initializer).
TUNABLE = {
    "match_ticks": "MATCH_TICKS",
    "swiss_rounds": "SWISS_ROUNDS",
    "swiss_games": "SWISS_GAMES",
    "champion_games": "CHAMPION_GAMES",
    "confirm_games": "CONFIRM_GAMES",
    "confirm_top": "CONFIRM_TOP",
    "spar_gd": "SPAR_GD",
    "entry_gd": "ENTRY_GD",
    "promote_rounds": "PROMOTE_ROUNDS",
    "title_rating": "TITLE_RATING",
    "eval_batch": "EVAL_BATCH",
    "showcase_candidates": "SHOWCASE_CANDIDATES",
    "round_robin_max": "ROUND_ROBIN_MAX",
    "past_champions": "PAST_CHAMPIONS",
    "swiss_seeding": "SWISS_SEEDING",
    "training_field": "TRAINING_FIELD",
}


def tunables():
    return {key: globals()[name] for key, name in TUNABLE.items()}


def apply_settings(settings):
    """Set the tunable rules from a settings dict; returns a description of the changes."""
    changes = []
    for key, name in TUNABLE.items():
        if key in settings:
            value = type(globals()[name])(settings[key])
            if value != globals()[name]:
                changes.append(f"{key} {globals()[name]} -> {value}")
                globals()[name] = value
    return changes


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


def game_points(gd):
    """Points for a game with goal difference gd: 3 a win, 1 a draw, 0 a loss."""
    return 3.0 if gd > 0 else 1.0 if gd == 0 else 0.0


def match_list(tournament):
    """
    A round's Swiss as matches: [name a, name b, mean goal difference per game (a - b),
    games, a's points per game, b's points per game].
    """
    matches = {}
    for blue, red, bg, rg in tournament.results:
        a, b = tournament.brains[blue].name, tournament.brains[red].name
        key = tuple(sorted((a, b)))
        gd = (bg - rg) if a == key[0] else (rg - bg)
        m = matches.setdefault(key, [0.0, 0, 0.0, 0.0])
        m[0] += gd
        m[1] += 1
        m[2] += game_points(gd)
        m[3] += game_points(-gd)
    return [[a, b, total / n, n, pa / n, pb / n] for (a, b), (total, n, pa, pb) in matches.items()]


def points_ratings(matches, rng, resamples=400):
    """
    The "points" title rating: each brain's points per game over the matches (ties broken
    by goal difference per game), with 95% intervals from resampling whole matches. Same
    shape as tournament_ratings: ({name: {"gd": rating, "gd_ci", "matches", "points",
    "goal_difference"}}, resampled ratings, names); "gd" holds the rating so seeding and
    the notes work unchanged.
    """
    names = sorted({m[0] for m in matches} | {m[1] for m in matches})
    index = {n: k for k, n in enumerate(names)}

    def rate(sample):
        pts, gd, games = np.zeros(len(names)), np.zeros(len(names)), np.zeros(len(names))
        for a, b, g, n, pa, pb in sample:
            for name, p, d in ((a, pa, g), (b, pb, -g)):
                k = index[name]
                pts[k] += p * n
                gd[k] += d * n
                games[k] += n
        games = np.maximum(games, 1)
        return pts / games, gd / games

    ppg, gdpg = rate(matches)
    draws = np.array([
        sum(x * w for x, w in zip(rate([matches[k] for k in rng.integers(len(matches), size=len(matches))]), (1.0, TIEBREAK)))
        for _ in range(resamples)
    ])
    score = ppg + TIEBREAK * gdpg
    counts = {n: sum(n in m[:2] for m in matches) for n in names}
    out = {
        n: {"gd": float(score[k]), "gd_ci": float(1.96 * draws[:, k].std()), "matches": counts[n],
            "points": float(ppg[k]), "goal_difference": float(gdpg[k])}
        for k, n in enumerate(names)
    }
    return out, draws, names


def fit_ratings(matches, names):
    """
    Tournament ratings: goal difference per game against an average opponent, fitted by
    weighted least squares over the matches (each match weighted by its games), so a
    brain's rating allows for who it played. Ratings sum to zero.
    """
    index = {n: k for k, n in enumerate(names)}
    rows, target = [], []
    for a, b, gd, n in matches:
        if a in index and b in index:
            row = np.zeros(len(names))
            row[index[a]], row[index[b]] = 1.0, -1.0
            rows.append(row * np.sqrt(n))
            target.append(gd * np.sqrt(n))
    rows.append(np.ones(len(names)) * 10.0)
    target.append(0.0)
    return np.linalg.lstsq(np.array(rows), np.array(target), rcond=None)[0]


def tournament_ratings(matches, rng, resamples=400):
    """
    Ratings with 95% intervals (resampling whole matches) from a list of matches.
    Returns ({name: {"gd", "gd_ci", "matches"}}, resampled ratings array, names).
    """
    names = sorted({m[0] for m in matches} | {m[1] for m in matches})
    rating = fit_ratings(matches, names)
    draws = np.array([
        fit_ratings([matches[k] for k in rng.integers(len(matches), size=len(matches))], names)
        for _ in range(resamples)
    ])
    counts = {n: sum(n in m[:2] for m in matches) for n in names}
    out = {
        n: {"gd": float(rating[k]), "gd_ci": float(1.96 * draws[:, k].std()), "matches": counts[n]}
        for k, n in enumerate(names)
    }
    return out, draws, names


def backfill_matches(round_number):
    """The Swiss matches of the rounds before this one, from the saved round files."""
    history = {}
    for number in range(max(round_number - PROMOTE_ROUNDS + 1, 1), round_number):
        path = LEAGUE_DIR / "rounds" / f"round-{number:03d}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved.get("match_ticks") == MATCH_TICKS:
                history[str(number)] = [
                    [m["a"], m["b"], (m["goals_a"] - m["goals_b"]) / m["games"], m["games"],
                     (3 * m["wins_a"] + m["draws"]) / m["games"], (3 * m["wins_b"] + m["draws"]) / m["games"]]
                    for m in saved["matches"]
                ]
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


def canonical_spec(spec):
    """A non-network brain spec in one form: a .json naming just a class is "brain:<class>"."""
    spec = str(spec)
    if spec.endswith(".json") and Path(spec).exists():
        data = json.loads(Path(spec).read_text())
        if set(data) == {"class"} or (set(data) == {"class", "kwargs"} and not data["kwargs"]):
            return "brain:" + data["class"]
        return Path(spec).read_text()
    return spec


def same_brain(a, b):
    """True if two brains play identically (same policy for every role, or the same spec)."""
    if Path(a).suffix != ".npz" or Path(b).suffix != ".npz":
        return canonical_spec(a) == canonical_spec(b)
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
        # BLAS threads = the learner's workers: its update runs while its game workers wait, so
        # the same cores stay busy (one thread left most of them idle during every update).
        workers = str(cmd[cmd.index("--workers") + 1]) if "--workers" in cmd else "1"
        super().__init__(cmd, self.dir / "stdout.log", threads=workers)
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


STALE: dict = {}  # directory -> files no longer wanted at its last sync


def sync_dir(directory, wanted):
    """
    Make `directory` hold the wanted files ({file name: source path, or a dict written as
    JSON}) without ever leaving a reader looking for a missing file: each file is replaced
    atomically (a temporary copy, then os.replace), and files that are no longer wanted
    are only deleted at the next sync, so a process that listed the folder a moment ago
    can still open what it listed.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, source in wanted.items():
        tmp = directory / f"{Path(name).stem}.tmp{Path(name).suffix}"
        if isinstance(source, dict):
            tmp.write_text(json.dumps(source))
        else:
            shutil.copy(source, tmp)
        os.replace(tmp, directory / name)
    stale = {p.name for p in directory.iterdir() if p.name not in wanted and ".tmp" not in p.name}
    for name in stale & STALE.get(str(directory), set()):
        (directory / name).unlink(missing_ok=True)
    STALE[str(directory)] = stale - STALE.get(str(directory), set())


# What the training opponent folders (the field for the evolving processes, and each
# learner's pool) hold, set each round. "all" (the default, as before): every entrant with
# a brain file and every champion ever crowned. "tournament": exactly the brains of the
# round's tournament (entrants, including fixed "brain:" entrants written as JSON specs,
# the holder under its champion file name, and the past champions in the field, even one
# rested for an even field), so training aims at the population it is judged against.
TRAINING_FIELD = "all"
TRAINING_FIELDS = ("all", "tournament")


def brain_file(name, spec):
    """
    (file name, source) for a training opponent folder: the brain's file, or a JSON spec
    (a dict, written by sync_dir) that load_brain accepts for a "brain:" class or a
    heuristic. None for a spec no file can stand for (e.g. "explore:").
    """
    spec = str(spec)
    if spec.startswith("brain:"):
        return f"{name}.json", {"class": spec.removeprefix("brain:")}
    if spec in HEURISTICS:
        cls = HEURISTICS[spec]
        return f"{name}.json", {"class": f"{cls.__module__}.{cls.__qualname__}"}
    suffix = Path(spec).suffix
    if suffix in (".npz", ".json") and not spec.startswith("explore:"):
        return f"{name}{suffix}", spec
    return None


def tournament_files(field, champion=None, champion_alias=None, exclude=()):
    """
    {file name: source} for every brain of a tournament field ({name: spec}) but the
    excluded names. The holder is written under its champion file name (champ-N.json,
    PPO-champ-N.npz) even when an identical entrant (champion_alias) stood for it, so
    train_ppo's current_champion finds it; excluding the alias excludes the holder too.
    """
    files = {}
    for name, spec in field.items():
        if name in exclude:
            continue
        if champion is not None and name == champion_alias:
            files[Path(champion).name] = str(champion)
            continue
        entry = brain_file(name, spec)
        if entry:
            files[entry[0]] = entry[1]
    return files


def fixed_field(entrant_specs):
    """The heuristic brains and saved brain versions, minus any entrant's own file."""
    field = {} if LEARNING_ONLY else {name: name for name in HEURISTICS if name not in UNINFORMATIVE}
    for path in sorted(
        list(HISTORY_DIR.glob("*.npz")) + list(HISTORY_DIR.glob("*champ-*.json"))
    ):
        if str(path) not in entrant_specs and path.stem not in UNINFORMATIVE:
            field[path.stem] = str(path)
    return field


def past_field(entrants):
    """
    The brains that play in the tournament besides the entrants ({name: spec}): the most
    recent PAST_CHAMPIONS past champions (and, outside a learning-only league, the
    heuristic brains and saved versions), leaving out any that is the same brain as an
    entrant.
    """
    past = {
        n: spec for n, spec in fixed_field(set(entrants.values())).items()
        if not any(same_brain(spec, e) for e in entrants.values())
    }
    champs = sorted(
        (n for n in past if re.search(r"champ-(\d+)$", n)),
        key=lambda n: int(re.search(r"champ-(\d+)$", n).group(1)),
    )
    for n in champs[: max(len(champs) - PAST_CHAMPIONS, 0)]:
        past.pop(n)  # only the most recent past champions play
    return past


def even_field(field, entrants):
    """
    An even number of brains, so nobody sits out a Swiss round: if the field is odd,
    rest the oldest past champion (a reference point in the Swiss, not a competitor).
    Returns the field and a note, or the field unchanged if it is already even (or has
    no past champion to rest).
    """
    if len(field) % 2 == 0:
        return field, None
    past = [n for n in field if n not in entrants and re.search(r"champ-(\d+)$", n)]
    if not past:
        return field, None
    oldest = min(past, key=lambda n: int(re.search(r"champ-(\d+)$", n).group(1)))
    note = f"SWISS: {oldest} rested this round (even field, no byes)"
    return {n: spec for n, spec in field.items() if n != oldest}, note


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


ROUND_ROBIN_MAX = 8  # fields up to this size play everyone; bigger ones play a Swiss
PAST_CHAMPIONS = 99  # past champions kept in the tournament (the most recent ones)
# How a Swiss is paired. "random": by points, ties broken at random, so the first wave is
# a random draw (swiss_pairings). "rating": seeded by rating, Dutch system: the first
# wave pairs seed 1 with seed n/2+1, 2 with n/2+2, ...; later waves pair within score
# groups, ties broken by seed (swiss_seeds, swiss_pairings_seeded).
SWISS_SEEDING = "random"
SEEDINGS = ("random", "rating")


def seed_order(names, ratings):
    """names from the top seed down: by rating, ties (e.g. unrated brains) in the given order."""
    return sorted(names, key=lambda n: -ratings[n])


def swiss_seeds(names, state, round_number):
    """
    Seeds for a rating-seeded Swiss: {name: rating}, {name: where the rating came from}.
    The rating is the title-race rating the previous round left in state (goal difference
    per game against an average opponent over the last PROMOTE_ROUNDS rounds: several
    rounds of games, so less noisy than one round's). A brain missing from it (a past
    champion rested for a while, a champion snapshot whose games were played under its
    entrant name) takes the latest rating under its own name, then under the name it was
    promoted from (state["champion_names"]), from the saved rounds. A brain never rated,
    a new entrant, gets the median of the field's ratings: its strength is unknown, and
    the median neither favours nor penalises it and, unlike the mean, is not dragged by
    one very weak or very strong brain. Its first result then places it.
    """
    sources = [("title race", state.get("title_race", {}))]
    for number in range(round_number - 1, max(round_number - 11, 0), -1):
        path = LEAGUE_DIR / "rounds" / f"round-{number:03d}.json"
        if path.exists():
            try:
                saved = json.loads(path.read_text())
            except ValueError:
                continue
            if saved.get("match_ticks") == MATCH_TICKS and saved.get("title_race"):
                sources.append((f"round {number}", saved["title_race"]))
    originals = state.get("champion_names", {})
    ratings, where = {}, {}
    for name in names:
        candidates = [name] + ([originals[name]] if name in originals else [])
        for label, race in sources:
            found = next((c for c in candidates if c in race), None)
            if found:
                ratings[name] = float(race[found]["gd"])
                where[name] = label + ("" if found == name else f", as {found}")
                break
    median = float(np.median(list(ratings.values()))) if ratings else 0.0
    for name in names:
        if name not in ratings:
            ratings[name], where[name] = median, "unrated: field median"
    return ratings, where


def dutch_preferences(a, rest, points):
    """
    a's possible opponents, best first. a is the highest-ranked unpaired brain and rest
    the other unpaired brains in rank order. In a's score group (the brains on its
    points) a heads the top half and meets the head of the bottom half: pairing the
    group's top brain this way each time gives 1 v m/2+1, 2 v m/2+2, ... for a group of m
    (odd m: the last brain is left over and floats down). Next come the rest of the
    bottom half, then the top half from its lowest up, then lower score groups from the
    top down (a floats down, or a group's floater meets the next group's top brain).
    """
    group = [n for n in rest if points[n] == points[a]]
    if not group:
        return list(rest)
    split = (len(group) + 1) // 2 - 1  # group[split] heads the bottom half of [a] + group
    below = [n for n in rest if points[n] != points[a]]
    return group[split:] + group[:split][::-1] + below


def dutch_pairing(order, points, played):
    """
    Pair brains (order: rank order, even length) by dutch_preferences with no pairing in
    played, backtracking when a choice leaves the rest unpairable. None if impossible.
    """
    if not order:
        return []
    a, rest = order[0], order[1:]
    for b in dutch_preferences(a, rest, points):
        if tuple(sorted((a, b))) in played:
            continue
        tail = dutch_pairing([n for n in rest if n != b], points, played)
        if tail is not None:
            return [(a, b)] + tail
    return None


def swiss_pairings_seeded(tournament, played, had_bye, seed_rank):
    """
    One Swiss round, Dutch system (SWISS_SEEDING "rating"). Rank by points, ties broken by
    seed (seed_rank: brain number -> 0 for the top seed), so the first round is ranked by
    seed alone and is one score group: 1 v n/2+1, 2 v n/2+2, ... The bye, if the number
    of brains is odd, goes to the lowest-ranked brain that has not had one. Then pair
    within score groups (dutch_preferences), avoiding repeat pairings; if every pairing
    would repeat one, pair as if nothing had been played.
    """
    table = tournament.tournament_scores.table
    order = sorted(table, key=lambda n: (-table[n]["points"], seed_rank[n]))
    points = {n: table[n]["points"] for n in order}
    bye = None
    if len(order) % 2:
        bye = next((n for n in reversed(order) if n not in had_bye), order[-1])
        order.remove(bye)
        had_bye.add(bye)
    pairs = dutch_pairing(order, points, played)
    if pairs is None:
        pairs = dutch_pairing(order, points, set())
    return [tuple(sorted(p)) for p in pairs], bye


def auto_rounds(n):
    """Rounds for a field of n: a full round robin up to ROUND_ROBIN_MAX, else a Swiss."""
    if n <= ROUND_ROBIN_MAX:
        return n - 1 if n % 2 == 0 else n
    return int(np.ceil(np.log2(n))) + 1


def round_robin_schedule(n):
    """The circle method: n - 1 rounds (n if odd, with a bye) in which every pair meets once."""
    players = list(range(n)) + ([None] if n % 2 else [])
    m = len(players)
    schedule = []
    for _ in range(m - 1):
        schedule.append([
            tuple(sorted((players[k], players[m - 1 - k])))
            for k in range(m // 2)
            if players[k] is not None and players[m - 1 - k] is not None
        ])
        players = [players[0], players[-1]] + players[1:-1]
    return schedule


def swiss(workers, brains, rounds, games, seed, seeds=None):
    """
    The league's tournament: brains is {name: spec}. With rounds = 0 the size of the
    field decides (auto_rounds): up to ROUND_ROBIN_MAX brains everyone plays everyone
    (round_robin_schedule); bigger fields play a Swiss, pairing by points with no repeat
    pairings where possible and rotating byes: with SWISS_SEEDING "rating" and seeds
    ({name: rating}, see swiss_seeds) seeded by rating (swiss_pairings_seeded), otherwise
    ties broken at random (swiss_pairings). Each round's games are played in parallel.
    Returns the Tournament, whose table and per-game results hold the outcome.
    """
    names = list(brains)
    rounds = rounds or auto_rounds(len(names))
    robin = round_robin_schedule(len(names)) if len(names) <= ROUND_ROBIN_MAX else None
    if robin:
        rounds = len(robin)
    tournament = Tournament(
        [SimpleNamespace(name=n) for n in names], legs=games, rounds=rounds, seed=seed
    )
    rng = np.random.default_rng(seed)
    played: set[tuple] = set()
    had_bye: set[int] = set()
    seed_rank = None
    if SWISS_SEEDING == "rating" and seeds is not None:
        seed_rank = {names.index(n): r for r, n in enumerate(seed_order(names, seeds))}
    for k in range(rounds):
        if robin:
            pairings = robin[k]
        elif seed_rank is not None:
            pairings, _bye = swiss_pairings_seeded(tournament, played, had_bye, seed_rank)
        else:
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
    if not h2h:
        for note in notes:
            log(f"  {note}")
        return
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
    number, tournament, standings, entrants, h2h, champion, notes, cpu=None, qualifying=None, champion_cpu=None,
    title_race=None,
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
        "title_race": title_race or {},
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
        "speed_scale": np.array(Constants.SPEED_SCALE),  # for render_replay.py's pace
    }


def seed_field(field_dir):
    """Generation-0 opponents for the evolving processes: two untrained networks."""
    from aisoccer.brains.PPOBrain import ACT_DIM, OBS_DIM
    from aisoccer.ppo import PPO

    field_dir.mkdir(parents=True, exist_ok=True)
    PPOBrain.save_weights(field_dir / "random-ppo.npz", PPO(OBS_DIM, ACT_DIM, seed=1).weights())
    PPOBrain.save_weights(field_dir / "random-ppo-2.npz", PPO(OBS_DIM, ACT_DIM, seed=2).weights())


# ---------------------------------------------------------------- core allocation
#
# Optional (roster settings "core_allocation": "cost_gap"; default off): give the training
# strategies that lag on compute cost more cores, so the gap closes faster. At every round
# boundary each strategy family's cost (roster "families": {brain: family}, a family's cost
# is its members' own costs added up, like the LLM line's lineage) is compared with the
# most expensive strategy; "training_workers" workers are shared out: one per brain, the
# rest in proportion to the families' gaps, then equally within a family. A brain is only
# restarted with its new count if that differs from what it runs with by at least
# "core_min_change" workers or "core_min_change_share" of its count. Brains outside the
# families keep their configured workers. The counts live in memory (and state.json
# "cores"), never in roster.json.

#
# "core_allocation": "cost_target" instead aims every training brain at the same budget
# ("cost_target_usd", default 20): each brain in the families gets one worker (or its
# "core_floor"), the rest in proportion to how far its own cost (with lineage) is below
# the target, so they reach it at about the same time. A brain that reaches the target is frozen: its current
# brain file is copied to <league>/frozen/ and it becomes a fixed entrant, so it keeps
# playing but stops training (and costing).

CORE_MODES = ("off", "cost_gap", "cost_target")
DEFAULT_COST_TARGET = 20.0


def configured_workers(kind, entry, settings):
    """The workers a learner (settings' workers_per_learner) or process (its cmd's --workers) is configured with."""
    if kind == "learner":
        return int(settings.get("workers_per_learner", 2))
    cmd = [str(c) for c in entry.get("cmd", [])]
    if "--workers" in cmd[:-1]:
        return int(cmd[cmd.index("--workers") + 1])
    return None  # the process's own default


def with_workers(cmd, workers):
    """A process command with its --workers set to `workers` (added if missing)."""
    cmd = [str(c) for c in cmd]
    if "--workers" in cmd[:-1]:
        cmd[cmd.index("--workers") + 1] = str(workers)
    else:
        cmd += ["--workers", str(workers)]
    return cmd


def strategy_costs(families, own_usd, lineage_usd):
    """
    Dollar cost per strategy: a family's members' own costs added up (retired members
    included; at least any member's lineage, in case earlier versions are not listed),
    and every brain outside the families with its lineage (a version's bill plus its
    predecessors', so the LLM line counts once, as its newest version).
    Returns ({family: cost}, {strategy: cost} for every strategy).
    """
    family_cost = {f: 0.0 for f in set(families.values())}
    for name, usd in own_usd.items():
        if name in families:
            family_cost[families[name]] += usd
    for name, usd in lineage_usd.items():
        if name in families:
            family_cost[families[name]] = max(family_cost[families[name]], usd)
    strategies = dict(family_cost)
    for name, usd in lineage_usd.items():
        if name not in families and name not in strategies:
            strategies[name] = usd
    return family_cost, strategies


def split_budget(budget, gaps, sizes):
    """
    Workers per family: one per brain first, the rest in proportion to the families' cost
    gaps (in proportion to size if no family lags), rounded by largest remainder.
    """
    names = sorted(f for f in sizes if sizes[f] > 0)
    out = {f: sizes[f] for f in names}
    spare = max(int(budget) - sum(out.values()), 0)
    weights = {f: max(float(gaps.get(f, 0.0)), 0.0) for f in names}
    if sum(weights.values()) <= 0:
        weights = {f: float(sizes[f]) for f in names}
    total = sum(weights.values())
    if not names or total <= 0:
        return out
    exact = {f: spare * weights[f] / total for f in names}
    for f in names:
        out[f] += int(exact[f])
    left = spare - sum(int(exact[f]) for f in names)
    for f in sorted(names, key=lambda f: (-(exact[f] - int(exact[f])), -weights[f], f))[:left]:
        out[f] += 1
    return out


def split_family(workers, members, own_usd, weights=None):
    """
    A family's workers shared equally by its brains; spare ones go to the cheapest brains
    first. With `weights` (settings "core_weights", e.g. {"PPO-goals": 2}) a brain's share
    is proportional to its weight (default 1), at least 1 worker each.
    """
    if weights and any(n in weights for n in members):
        w = {n: float(weights.get(n, 1.0)) for n in members}
        rest = workers - len(members)
        exact = {n: rest * w[n] / sum(w.values()) for n in members}
        out = {n: 1 + int(exact[n]) for n in members}
        left = workers - sum(out.values())
        for n in sorted(members, key=lambda n: (-(exact[n] - int(exact[n])), -w[n], n))[:max(left, 0)]:
            out[n] += 1
        return out
    base, extra = divmod(workers, len(members))
    order = sorted(members, key=lambda n: (own_usd.get(n, 0.0), n))
    return {n: base + (1 if k < extra else 0) for k, n in enumerate(order)}


def settle(current, target, min_change=2, min_share=0.25):
    """Hysteresis: move to the target only if it differs by min_change workers or min_share of the current count."""
    if current is None:
        return target
    diff = abs(target - current)
    if diff and (diff >= min_change or diff >= min_share * current):
        return target
    return current


def plan_cores(budget, families, own_usd, lineage_usd, running, weights=None):
    """
    The cost-gap allocation. `running`: the brains that can be given workers (running
    learners and processes in a family). Returns ({brain: workers}, {family: report}).
    """
    family_cost, strategies = strategy_costs(families, own_usd, lineage_usd)
    reference = max(strategies, key=lambda s: (strategies[s], s)) if strategies else None
    top = strategies.get(reference, 0.0)
    members = {}
    for name in sorted(running):
        members.setdefault(families[name], []).append(name)
    gaps = {f: top - family_cost[f] for f in members}
    totals = split_budget(budget, gaps, {f: len(m) for f, m in members.items()})
    targets, report = {}, {}
    for f, names in members.items():
        share = split_family(totals[f], names, own_usd, weights)
        targets.update(share)
        report[f] = {"cost": family_cost[f], "gap": gaps[f], "workers": totals[f], "brains": share}
    return targets, {"reference": reference, "reference_cost": top, "families": report}


def plan_target_cores(budget, lineage_usd, running, target, floors=None):
    """
    The cost-target allocation. Returns ({brain: workers} for the brains still below the
    target, [brains at or above it, to freeze], {brain: gap in dollars}). `floors`
    (settings "core_floor", e.g. {"GP-v3": 6}): a brain's minimum workers while it trains;
    the others share what is left.
    """
    gaps = {n: target - lineage_usd.get(n, 0.0) for n in running}
    live = sorted(n for n in running if gaps[n] > 0)
    frozen = sorted(n for n in running if gaps[n] <= 0)
    floors = {n: int(f) for n, f in (floors or {}).items() if n in live}
    fixed = {}
    while True:  # brains whose share falls below their floor get the floor; the rest re-split
        rest = [n for n in live if n not in fixed]
        left = budget - sum(fixed.values())
        workers = split_budget(left, {n: gaps[n] for n in rest}, {n: 1 for n in rest}) if rest else {}
        low = [n for n in rest if workers[n] < floors.get(n, 0)]
        if not low:
            return {**workers, **fixed}, frozen, gaps
        fixed.update({n: floors[n] for n in low})


def target_note(target, budget, lineage_usd, gaps, workers, held):
    """The round note for the cost-target allocation."""
    parts = [
        f"{n} ${lineage_usd.get(n, 0.0):.2f} (gap ${gaps[n]:.2f}) -> {workers[n]}"
        + (" (kept)" if n in held else "")
        for n in sorted(workers, key=lambda n: -gaps[n])
    ]
    return f"CORES (cost target ${target:.2f} per brain, {budget} training workers): " + "; ".join(parts)


def cores_note(plan, workers, budget, held):
    """The round note: each family's cost, gap and workers, and each brain's count."""
    parts = []
    for f, r in sorted(plan["families"].items(), key=lambda kv: -kv[1]["gap"]):
        brains = ", ".join(
            f"{n} {workers[n]}" + (f" (target {t}, kept)" if n in held else "")
            for n, t in sorted(r["brains"].items())
        )
        parts.append(
            f"{f} family ${r['cost']:.2f} (gap ${r['gap']:.2f}) -> {r['workers']} workers: {brains}"
        )
    return (
        f"CORES (cost gap, {budget} training workers; most expensive strategy {plan['reference']} "
        f"${plan['reference_cost']:.2f}): " + "; ".join(parts)
    )


def brain_usd(cpu_seconds):
    """({brain: own cost}, {brain: cost with lineage}) in dollars, as league_table.py's Compute column."""
    import leaguefiles

    results = {"cpu_hours": {n: s / 3600 for n, s in cpu_seconds.items()}}
    return leaguefiles.compute_usd(results, lineage=False), leaguefiles.compute_usd(results, lineage=True)


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
        # name -> workers set by the core allocation (only where that differs from the
        # configured count); applied by reconcile, see allocate_cores.
        self.workers: dict[str, int] = {}

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
            workers = self.workers.get(name)
            key = json.dumps(entry if workers is None else {"entry": entry, "workers": workers}, sort_keys=True)
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
                args = settings  # the core allocation's count, if it set one
                if workers is not None:
                    args = SimpleNamespace(**{**vars(settings), "workers_per_learner": workers})
                self.learners[name] = Learner(name, entry, self.champion, args, saved)
            else:
                cmd = [sys.executable] + (entry["cmd"] if workers is None else with_workers(entry["cmd"], workers))
                self.processes[name] = Process(cmd, LEAGUE_DIR / f"{name}.stdout.log")
            self.started[name] = key
            notes.append(
                f"ROSTER: {'restarted' if restarting else 'started'} {name}"
                + ("" if workers is None else f" ({workers} workers, core allocation)")
            )
        return notes

    def wanted_workers(self):
        """name -> (kind, configured workers) for every learner and process the roster runs."""
        settings = self.data.get("settings", {})
        out = {
            n: ("learner", configured_workers("learner", e, settings)) for n, e in self.data.get("learners", {}).items()
        }
        for section in ("processes", "waiting"):
            for n, e in self.data.get(section, {}).items():
                if "cmd" in e:
                    out[n] = ("process", configured_workers("process", e, settings))
        return out

    def allocate_cores(self, state):
        """
        The cost-gap core allocation (see plan_cores), once per round: sets self.workers,
        which the next reconcile applies (restarting only the brains whose count changes).
        Returns notes. With "core_allocation" off, every brain goes back to its configured
        workers.
        """
        settings = self.data.get("settings", {})
        mode = settings.get("core_allocation", "off") or "off"
        if mode not in CORE_MODES:
            return [f"CORES: unknown core_allocation {mode!r}; workers unchanged"]
        if mode == "off":
            if not self.workers:
                return []
            self.workers = {}
            return ["CORES: allocation off; every brain goes back to its configured workers"]
        try:
            families = self.data.get("families") or settings.get("families") or {}
            budget = int(settings.get("training_workers", 0))
            min_change = int(settings.get("core_min_change", 2))
            min_share = float(settings.get("core_min_change_share", 0.25))
            wanted = self.wanted_workers()
            running = [n for n in wanted if n in families]
            if not running or budget <= 0:
                self.workers = {}
                return ["CORES: nothing to allocate (set training_workers and families)"]
            own, lineage = brain_usd(state.get("cpu", {}))
            if mode == "cost_target":
                return self.allocate_to_target(settings, budget, lineage, running, wanted, min_change, min_share)
            targets, plan = plan_cores(
                budget, families, own, lineage, running, settings.get("core_weights") or None
            )
            workers, held = {}, set()
            for name, target in targets.items():
                configured = wanted[name][1]
                current = self.workers.get(name, configured)
                workers[name] = settle(current, target, min_change, min_share)
                if workers[name] != target:
                    held.add(name)
            self.workers = {n: w for n, w in workers.items() if w != wanted[n][1]}
            return [cores_note(plan, workers, budget, held)]
        except Exception as error:  # never let the allocation stop the league
            return [f"CORES: allocation failed ({error!r}); workers unchanged"]

    def allocate_to_target(self, settings, budget, lineage, running, wanted, min_change, min_share):
        """The cost-target allocation (see plan_target_cores): freezes brains at the target, shares out the rest."""
        target = float(settings.get("cost_target_usd", DEFAULT_COST_TARGET))
        targets, frozen, gaps = plan_target_cores(budget, lineage, running, target, settings.get("core_floor"))
        notes = []
        for name in frozen:
            notes.append(self.freeze(name, lineage.get(name, 0.0), target))
        workers, held = {}, set()
        for name, count in targets.items():
            current = self.workers.get(name, wanted[name][1])
            workers[name] = settle(current, count, min_change, min_share)
            if workers[name] != count:
                held.add(name)
        self.workers = {n: w for n, w in workers.items() if w != wanted[n][1]}
        if workers:
            notes.append(target_note(target, budget, lineage, gaps, workers, held))
        else:
            notes.append(f"CORES: every training brain has reached the ${target:.2f} target; nothing left to train")
        return notes

    def freeze(self, name, usd, target):
        """
        Turn a training brain into a fixed entrant with a copy of its current brain file
        (the reconcile then stops its training). Its roster entry moves to "frozen".
        """
        if name in self.data.get("learners", {}):
            section, source = "learners", LEAGUE_DIR / name / "policy.npz"
        else:
            section, source = "processes", Path(self.data["processes"][name]["entrant"])
        frozen_dir = LEAGUE_DIR / "frozen"
        frozen_dir.mkdir(exist_ok=True)
        copy = frozen_dir / f"{name}{source.suffix}"
        shutil.copy2(source, copy)

        def change(data):
            entry = data.get(section, {}).pop(name, None)
            if entry is not None:
                data.setdefault("frozen", {})[name] = {
                    "from": section, "entry": entry, "cost_usd": round(usd, 2), "file": str(copy),
                }
            data.setdefault("fixed_entrants", {})[name] = str(copy)

        self.update(change)
        self.workers.pop(name, None)
        return f"FROZEN {name}: ${usd:.2f} reached the ${target:.2f} compute target; it keeps playing as {copy} but stops training"

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
    if roster.data.get("settings", {}).get("core_allocation", "off") in ("cost_gap", "cost_target"):
        # Start with the last round's core allocation, so nothing restarts again next round.
        roster.workers = {n: int(w) for n, w in state.get("cores", {}).items()}
        if roster.data["settings"]["core_allocation"] == "cost_target":
            # Switching to (or restarting in) cost_target mode applies its plan at once.
            for note in roster.allocate_cores(state):
                log(note)
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
        # Two untrained networks, moving on the exploration noise every learner starts with.
        replay = record_showcase(
            ("untrained-1", "explore:" + str(field_dir / "random-ppo.npz"), "untrained-2",
             "explore:" + str(field_dir / "random-ppo-2.npz"), 900, args.showcase_ticks)
        )
        np.savez_compressed(replays / "round-000.npz", round=0, **replay)
        final = replay["scores"][-1]
        log(f"SHOWCASE recorded (round 0, untrained): untrained-1 {final[0]} - {final[1]} untrained-2")

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
            time.sleep(roster.data.get("settings", {}).get("round_minutes", args.round_minutes) * 60)
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
            for change in apply_settings(roster.data.get("settings", {})):
                log(f"  SETTINGS: {change}")
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
                # The champion plays as itself, unless it is identical to a live entrant
                # (a brain that does not change, like an LLMBrain version): then that entrant
                # stands for it, in the Swiss and in the title race.
                champion_alias = None
                if champion:
                    champion_alias = next(
                        (n for n, spec in entrants.items() if same_brain(spec, str(champion))), None
                    )
                    if champion_alias is None:
                        entrants[champion.stem] = str(champion)

                # A fresh pool each round, so improved brain code plays straight away.
                with Pool(os.cpu_count(), initializer=apply_settings, initargs=(tunables(),)) as workers:
                    past = past_field(entrants)
                    field = {**entrants, **past}
                    tournament_field = dict(field)  # with any rested past champion
                    field, rested = even_field(field, entrants)
                    if rested:
                        notes.append(rested)
                    order = list(field)
                    rng.shuffle(order)
                    seeds = None
                    if SWISS_SEEDING not in SEEDINGS:
                        notes.append(f"SWISS: unknown swiss_seeding {SWISS_SEEDING!r}, paired at random")
                    elif SWISS_SEEDING == "rating" and len(order) > ROUND_ROBIN_MAX:
                        seeds, seeded_from = swiss_seeds(order, state, round_number)
                        notes.append(
                            "SWISS seeds (title-race rating): "
                            + ", ".join(
                                f"{n} {k + 1} ({seeds[n]:+.1f}"
                                + ("" if seeded_from[n] == "title race" else f"; {seeded_from[n]}")
                                + ")"
                                for k, n in enumerate(seed_order(order, seeds))
                            )
                        )
                    tournament = swiss(
                        workers,
                        {n: field[n] for n in order},
                        SWISS_ROUNDS,
                        SWISS_GAMES,
                        round_number,
                        seeds,
                    )
                    standings = swiss_standings(tournament)
                    h2h = {}  # the title is decided by the tournament (see the title race below)
                    round_matches = match_list(tournament)
                    round_ratings, _, _ = tournament_ratings([m[:4] for m in round_matches], np.random.default_rng(round_number))
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

                # The title race: a rating from every Swiss match of the last PROMOTE_ROUNDS
                # rounds (goal difference per game against an average opponent, allowing for
                # who each brain played). A brain takes the title when it has played all those
                # rounds, tops the rating, and is above the holder in at least 97.5% of
                # resamples of the matches. A tournament judges a brain against the whole
                # field, which a head to head against one brain cannot.
                history = state.setdefault("swiss_matches", backfill_matches(round_number))
                if TITLE_RATING == "points" and any(len(m) < 6 for rnd in history.values() for m in rnd):
                    # Matches saved before points were kept: take them again from the round files.
                    history.clear()
                    history.update(backfill_matches(round_number))
                history[str(round_number)] = round_matches
                for key in sorted(history, key=int)[:-PROMOTE_ROUNDS]:
                    history.pop(key)
                window = [history[k] for k in sorted(history, key=int)]
                pooled = [m for rnd in window for m in rnd]
                if TITLE_RATING == "points":
                    title, draws, rated = points_ratings(pooled, np.random.default_rng(round_number + 7))
                else:
                    title, draws, rated = tournament_ratings(
                        [m[:4] for m in pooled], np.random.default_rng(round_number + 7)
                    )
                everywhere = set.intersection(*({m[0] for m in rnd} | {m[1] for m in rnd} for rnd in window))
                eligible = [
                    n for n in rated
                    if n in entrants and n in everywhere
                    and (champion is None or n not in (champion.stem, champion_alias))
                ]
                new_champion = None
                leader = max(eligible, key=lambda n: title[n]["gd"], default=None)
                holder = (champion_alias or champion.stem) if champion is not None else None
                confidence = None
                if leader and holder in rated and len(window) == PROMOTE_ROUNDS:
                    k, h = rated.index(leader), rated.index(holder)
                    confidence = float(np.mean(draws[:, k] > draws[:, h]))
                    if confidence >= 0.975 and title[leader]["gd"] > title[holder]["gd"]:
                        new_champion = leader
                if champion is None:
                    # The first champion: the best entrant of the first round's Swiss.
                    new_champion = next((n for n, _ in standings if n in entrants), None)
                top = sorted(rated, key=lambda n: -title[n]["gd"])[:6]
                notes.append(
                    (
                        f"TITLE RACE over rounds {', '.join(sorted(history, key=int))} (rating = points per game, "
                        "3 a win and 1 a draw, goal difference per game breaking ties): "
                        + ", ".join(
                            f"{n} {title[n]['points']:.2f} ±{title[n]['gd_ci']:.2f} (gd {title[n]['goal_difference']:+.1f})"
                            for n in top
                        )
                        + (f"; holder {holder} {title[holder]['points']:.2f}" if holder in title else "")
                    )
                    if TITLE_RATING == "points" else
                    f"TITLE RACE over rounds {', '.join(sorted(history, key=int))} (rating = goal difference per game "
                    "against an average opponent): "
                    + ", ".join(f"{n} {title[n]['gd']:+.2f} ±{title[n]['gd_ci']:.2f}" for n in top)
                    + (f"; holder {holder} {title[holder]['gd']:+.2f}" if holder in title else "")
                )
                if new_champion and champion is not None:
                    notes.append(
                        f"PROMOTED on the {PROMOTE_ROUNDS}-round tournament rating: {new_champion} "
                        f"{title[new_champion]['gd']:+.2f} above holder {holder} {title[holder]['gd']:+.2f} "
                        f"in {confidence:.1%} of resamples"
                    )
                elif leader and confidence is not None:
                    notes.append(
                        f"ON COURSE: {leader} leads the title race, above the holder in {confidence:.1%} of "
                        "resamples (needs 97.5%)"
                    )
                pending = leader
                state["title_race"] = {n: title[n] for n in rated}

                # Replacement, rated by this round's tournament rating.
                ratings = {n: round_ratings[n] for n in learners if n in round_ratings}
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
                            notes.append(
                                f"REPLACED learner {name} (age {age}, {gap:.2f} goals/game behind "
                                f"{leader} in the tournament rating, not improving): adopts {leader}'s "
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
                    with Pool(os.cpu_count(), initializer=apply_settings, initargs=(tunables(),)) as workers:
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

                # Core allocation (off by default): applied by the reconcile below.
                notes += roster.allocate_cores(state)

                log_round(
                    round_number, standings, entrants, h2h, iterations, notes,
                    cpu={**state.get("cpu", {}), **state.get("champion_cpu", {})},
                )
                save_round(
                    round_number, tournament, standings, entrants, h2h, champion, notes,
                    cpu=state.get("cpu"), qualifying=state.get("qualifying"),
                    champion_cpu=state.get("champion_cpu"),
                    title_race=state.get("title_race"),
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
                    # The name it played under, for its Swiss seed until it has a rating of its own.
                    state.setdefault("champion_names", {})[champion.stem] = new_champion
                    # A champion's compute: what its brain had used when it was crowned.
                    state.setdefault("champion_cpu", {})[champion.stem] = state.get("cpu", {}).get(new_champion, 0.0)
                    if verdict:
                        verdict.stop()  # its champion has been superseded
                    if not args.no_verdict:
                        verdict = Verdict(champion)
                        log(f"  Verdict tournament on {champion.stem} started in the background")

                # Sparring partners for the learners: every other entrant and every champion
                # ("all"), or every other brain of this round's tournament ("tournament";
                # a champion crowned just now stands under its new champion file name).
                training_field = TRAINING_FIELD
                if training_field not in TRAINING_FIELDS:
                    log(f"  TRAINING FIELD: unknown training_field {training_field!r}; using 'all'")
                    training_field = "all"
                promoted = new_champion and not args.dry_run
                holder, holder_alias = champion, (new_champion if promoted else champion_alias)
                for name, learner in learners.items():
                    wanted = {}
                    if training_field == "tournament":
                        wanted = tournament_files(tournament_field, holder, holder_alias, exclude={name})
                    else:
                        for other, spec in entrants.items():
                            if other != name and Path(spec).suffix in (".npz", ".json"):
                                wanted[f"{other}{Path(spec).suffix}"] = spec
                        for champ in HISTORY_DIR.glob("*champ-*.*"):
                            wanted[champ.name] = champ
                    # Waiting entrants within SPAR_GD of the champion: training opponents
                    # only (not in the Swiss, cannot become champion).
                    for other, spec in state.get("sparring", {}).items():
                        if Path(spec).suffix in (".npz", ".json"):
                            wanted[f"spar-{other}{Path(spec).suffix}"] = spec
                        elif spec.startswith("brain:"):
                            wanted[f"spar-{other}.json"] = {"class": spec.removeprefix("brain:")}
                    sync_dir(learner.pool_dir, wanted)

                # The field the evolving processes play against: every entrant's latest
                # version and every champion (in a learning-only league, nothing else).
                # With training_field "tournament": exactly this round's tournament brains.
                if LEARNING_ONLY and training_field == "tournament":
                    sync_dir(field_dir, tournament_files(tournament_field, holder, holder_alias))
                elif LEARNING_ONLY:
                    wanted = {
                        f"{other}{Path(spec).suffix}": spec
                        for other, spec in entrants.items() if Path(spec).suffix in (".npz", ".json")
                    }
                    wanted.update({champ.name: champ for champ in HISTORY_DIR.glob("*champ-*.*")})
                    sync_dir(field_dir, wanted)

                # Roster changes take effect now, between rounds.
                roster.reload()
                for note in roster.reconcile(state):
                    log(f"  {note}")

                state = {
                    "round": round_number,
                    "pending": pending,
                    "swiss_matches": state.get("swiss_matches", {}),
                    "title_race": state.get("title_race", {}),
                    "ranks": state.get("ranks", {}),
                    "on_par": state.get("on_par", {}),
                    "sparring": state.get("sparring", {}),
                    "cpu": state.get("cpu", {}),
                    "cpu_seen": state.get("cpu_seen", {}),
                    "cpu_estimated_to": state.get("cpu_estimated_to"),
                    "champion_cpu": state.get("champion_cpu", {}),
                    "champion_names": state.get("champion_names", {}),
                    **({"cores": roster.workers} if roster.workers else {}),
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
