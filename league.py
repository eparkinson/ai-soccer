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
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from aisoccer.brains.AdaptiveChaser import AdaptiveChaser
from aisoccer.brains.BehindAndTowards import BehindAndTowards
from aisoccer.brains.DefendersAndAttackers import DefendersAndAttackers
from aisoccer.brains.GeneticBrain import GeneticBrain
from aisoccer.brains.PPOBrain import DEFAULT_REWARD, PPOBrain
from aisoccer.brains.RandomWalk import RandomWalk
from aisoccer.brains.SimpleBrain import SimpleBrain
from aisoccer.brains.StrategicPlanner import StrategicPlanner
from aisoccer.game import Game
from aisoccer.stats import play_with_stats
from aisoccer.tournament import Tournament

LEAGUE_DIR = Path(os.environ.get("LEAGUE_DIR", "runs/league"))
LOG_FILE = LEAGUE_DIR / "progress.log"
HISTORY_DIR = PPOBrain.WEIGHTS_FILE.parent / "history"

HEURISTICS = {
    "DefendersAndAttackers": DefendersAndAttackers,
    "BehindAndTowards": BehindAndTowards,
    "StrategicPlanner": StrategicPlanner,
    "AdaptiveChaser": AdaptiveChaser,
    "SimpleBrain": SimpleBrain,
    "RandomWalk": RandomWalk,
}

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

SWISS_ROUNDS = 5
SWISS_GAMES = 96
CHAMPION_GAMES = 96  # stage 1: every entrant against the champion
CONFIRM_GAMES = 320  # stage 2: extra games for the most promising entrants
CONFIRM_TOP = 3
# Every PPOBrain beats these by 1.5-2.5 goals a game, so they cannot separate the
# candidates: they stay out of league rounds (the final verdict still includes them).
UNINFORMATIVE = {"RandomWalk", "SimpleBrain", "AdaptiveChaser", "PPO-scratch"}


def log(message=""):
    print(message, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")


# ---------------------------------------------------------------- playing games


def make_brain(spec):
    if spec in HEURISTICS:
        return HEURISTICS[spec]()
    path = Path(spec)
    if path.suffix == ".json":
        return GeneticBrain(path.stem, GeneticBrain.load(path))
    return PPOBrain(path.stem, weights=PPOBrain.load_weights(path))


def play(task):
    """(entrant spec, opponent spec, entrant plays blue, seed[, with stats]) -> result."""
    entrant, opponent, entrant_blue, seed = task[:4]
    with_stats = len(task) > 4 and task[4]
    a, b = make_brain(entrant), make_brain(opponent)
    blue, red = (a, b) if entrant_blue else (b, a)
    game = Game(blue, red, quiet_mode=True, seed=seed)
    me, them = ("blue", "red") if entrant_blue else ("red", "blue")
    if with_stats:
        score, stats = play_with_stats(game)
        return score[me], score[them], stats[me], stats[them]
    score = game.play()
    return score[me], score[them]


def record(games):
    """Points and goal difference per game, with 95% CI half-widths."""
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
        "points_ci": 1.96 * points.std(ddof=1) / np.sqrt(n),
        "gd": gd.mean(),
        "gd_ci": 1.96 * gd.std(ddof=1) / np.sqrt(n),
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
            cmd,
            stdout=open(log_path, "a"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=dict(os.environ, OPENBLAS_NUM_THREADS=threads),
        )

    def signal(self, sig):
        if self.process.poll() is None:
            os.killpg(self.process.pid, sig)

    def exited(self):
        return self.process.poll() is not None


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
            "--iterations",
            "1000000",
            "--eval-every",
            "0",
            "--seed",
            str(ord(name)),
        ] + spec.get("args", [])
        if (self.dir / "latest.npz").exists():
            cmd += ["--resume", "--value-warmup", "0"]
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
    """The heuristic brains and saved PPOBrain versions, minus any entrant's own file."""
    field = {name: name for name in HEURISTICS if name not in UNINFORMATIVE}
    for path in sorted(HISTORY_DIR.glob("*.npz")):
        if str(path) not in entrant_specs and path.stem not in UNINFORMATIVE:
            field[path.stem] = str(path)
    return field


def swiss(workers, brains, rounds, games, seed):
    """
    Swiss tournament: brains is {name: spec}. Uses Tournament's Swiss pairing (by points,
    no repeat pairings) and plays each round's games in parallel. Returns the
    Tournament, whose table and per-game results hold the outcome.
    """
    names = list(brains)
    tournament = Tournament(
        [SimpleNamespace(name=n) for n in names], legs=games, rounds=rounds, seed=seed
    )
    banned: list[tuple] = []
    for _ in range(rounds):
        pairings, byes = tournament.calculate_swiss_pairings(banned)
        # The greedy pairing can strand several players (every remaining opponent
        # already met); pair them with each other so at most one sits out.
        left = [
            row["number"] for row in tournament.get_table() if row["number"] in byes
        ]
        while len(left) >= 2:
            pairings.append(tuple(sorted((left.pop(0), left.pop(0)))))
        fixtures = tournament.fixtures(pairings)
        tasks = [(brains[names[b]], brains[names[r]], True, s) for b, r, s in fixtures]
        for (blue, red, _), (blue_goals, red_goals) in zip(
            fixtures, workers.map(play, tasks, chunksize=4)
        ):
            tournament.tournament_scores.process((blue, red), (blue_goals, red_goals))
            tournament.results.append((blue, red, blue_goals, red_goals))
        banned.extend(pairings)
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

    def run(names, n, seed):
        seeds = np.random.default_rng(seed).integers(2**31, size=n)
        tasks = [
            (entrants[name], str(champion), g % 2 == 0, int(seeds[g]), True)
            for name in names
            for g in range(n)
        ]
        results = workers.map(play, tasks, chunksize=4)
        chunks = np.array_split(np.arange(len(results)), len(names)) if names else []
        return {name: [results[i] for i in idx] for name, idx in zip(names, chunks)}

    names = [n for n, spec in entrants.items() if not same_brain(spec, str(champion))]
    games_by = run(names, games, seed)
    stage1 = {n: record(g)["gd"] for n, g in games_by.items()}
    top = sorted(stage1, key=lambda n: -stage1[n])[:CONFIRM_TOP]
    if CONFIRM_GAMES:
        for name, more in run(top, CONFIRM_GAMES, seed + 10**6).items():
            games_by[name] = games_by[name] + more
    return {
        n: {"record": record(g), "style": [r[2] for r in g]}
        for n, g in games_by.items()
    }


STYLE_STATS = ["control", "own_half", "final_third", "shots", "spread", "passes"]


def log_round(number, standings, entrants, h2h, iterations, notes):
    log(f"\nLEAGUE ROUND {number}  ({time.strftime('%H:%M')})")
    log(
        f"  Swiss tournament, {SWISS_ROUNDS} rounds of {SWISS_GAMES} games (* = entrant):"
    )
    log(
        f"    {'#':>2}  {'BRAIN':<24} {'GAMES':>5}  {'W-D-L':<12} {'GD/GAME':>13}  {'PTS/GAME':>12}"
    )
    for rank, (name, r) in enumerate(standings, 1):
        mark = "*" if name in entrants else " "
        wdl = f"{r['wins']}-{r['draws']}-{r['losses']}"
        log(
            f"    {rank:>2}{mark} {name:<24} {r['n']:>5}  {wdl:<12} {r['gd']:+.2f} ±{r['gd_ci']:.2f}"
            f"  {r['points']:>5.2f} ±{r['points_ci']:.2f}"
        )
    log(
        f"  Head to head with the champion ({CHAMPION_GAMES} games, +{CONFIRM_GAMES} for the top"
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


class Verdict:
    """
    The full verdict tournament (ppo_tournament.py, 300 games per pairing) on a new
    champion, run in the background so the league keeps going while it plays.
    """

    def __init__(self, champion):
        self.champion = champion.stem
        self.path = LEAGUE_DIR / f"verdict-{self.champion}.log"
        self.process = subprocess.Popen(
            [sys.executable, "ppo_tournament.py", "--legs", "300"],
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


def newest_champion():
    champions = sorted(
        HISTORY_DIR.glob("PPO-champ-*.npz"), key=lambda p: int(p.stem.split("-")[-1])
    )
    return champions[-1]


def promote(spec):
    """Save an entrant as the next champion (policy only) and as PPOBrain.npz."""
    number = len(list(HISTORY_DIR.glob("PPO-champ-*.npz")))
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
    return champion


# ---------------------------------------------------------------- main loop


def main():
    global SWISS_ROUNDS, SWISS_GAMES, CHAMPION_GAMES, CONFIRM_GAMES
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--round-minutes", type=float, default=45.0)
    parser.add_argument(
        "--min-age", type=int, default=60, help="iterations before replacing"
    )
    parser.add_argument("--workers-per-learner", type=int, default=2)
    parser.add_argument(
        "--learner-games", type=int, default=24, help="games per iteration"
    )
    parser.add_argument("--coach-workers", type=int, default=2)
    parser.add_argument("--ga-workers", type=int, default=2)
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
        "--only", nargs="+", help="run only these learners (for testing)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="log promotions, do not save them"
    )
    args = parser.parse_args()
    SWISS_ROUNDS, SWISS_GAMES, CHAMPION_GAMES = (
        args.swiss_rounds,
        args.swiss_games,
        args.champion_games,
    )
    CONFIRM_GAMES = args.confirm_games

    LEAGUE_DIR.mkdir(parents=True, exist_ok=True)
    # Stop cleanly (learners included) on kill / SIGTERM, not only on Ctrl-C.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    rng = np.random.default_rng(0)

    if not list(HISTORY_DIR.glob("PPO-champ-*.npz")):
        weights = PPOBrain.load_weights(args.start)
        PPOBrain.save_weights(
            HISTORY_DIR / "PPO-champ-0.npz",
            {"policy": weights["policy"], "log_std": weights["log_std"]},
        )
    champion = newest_champion()

    state_file = LEAGUE_DIR / "state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    specs = {n: s for n, s in LEARNERS.items() if not args.only or n in args.only}
    learners = {
        name: Learner(name, spec, champion, args, state.get("learners", {}).get(name))
        for name, spec in specs.items()
    }
    coach = Process(
        [sys.executable, "coach.py", "--workers", str(args.coach_workers)],
        LEAGUE_DIR / "coach.stdout.log",
    )
    ga = Process(
        [sys.executable, "evolve.py", "--workers", str(args.ga_workers)],
        LEAGUE_DIR / "ga.stdout.log",
    )
    helpers = {"Coach": coach, "GA": ga}
    everyone = list(learners.values()) + [coach, ga]

    log(
        f"League started: {len(learners)} learners, the coach and the GA; champion {champion.stem}"
    )
    for learner in learners.values():
        log(f"  {learner.name}: {learner.about}  {learner.config}")

    deadline = time.time() + args.hours * 3600
    round_number = state.get("round", 0)
    verdict = None
    pending = state.get("pending")  # entrant awaiting confirmation for promotion
    verdict_log = LEAGUE_DIR / f"verdict-{champion.stem}.log"
    finished = verdict_log.exists() and "HEAD TO HEAD" in verdict_log.read_text()
    if not args.dry_run and not finished:
        verdict = Verdict(champion)
        log(f"Verdict tournament on {champion.stem} started in the background")
    try:
        with Pool(os.cpu_count()) as workers:
            while time.time() < deadline:
                time.sleep(args.round_minutes * 60)
                for name, process in {**learners, **helpers}.items():
                    if process.exited():
                        log(
                            f"{name} exited ({process.process.returncode}); see its stdout.log"
                        )
                round_number += 1

                if verdict and not verdict.reported and verdict.result() is not None:
                    passed, head_to_head = verdict.result()
                    verdict.reported = True
                    log(f"\nVERDICT on {verdict.champion} (300 games per pairing):")
                    log(head_to_head.rstrip())
                    if passed:
                        log(
                            "\nSUCCESS: the champion is significantly stronger than every brain."
                        )
                        break
                    log(
                        "  Not yet significantly stronger than every brain; the league continues."
                    )

                for process in everyone:
                    process.signal(signal.SIGSTOP)
                try:
                    notes = []
                    iterations = {n: l.iteration() for n, l in learners.items()}
                    for name, learner in learners.items():
                        if learner.changed_at is None:
                            learner.changed_at = iterations[name]

                    # Snapshot every entrant.
                    snapshots = LEAGUE_DIR / "snapshots"
                    snapshots.mkdir(exist_ok=True)
                    sources = {n: l.dir / "policy.npz" for n, l in learners.items()}
                    sources["Coach"] = LEAGUE_DIR / "coach" / "team.npz"
                    sources["GA"] = LEAGUE_DIR / "ga" / "best.json"
                    entrants = {}
                    for name, source in sources.items():
                        if source.exists():
                            target = (
                                snapshots / f"{name}-r{round_number:03d}{source.suffix}"
                            )
                            shutil.copy(source, target)
                            entrants[name] = str(target)
                    entrants[champion.stem] = str(champion)

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
                    h2h = against_champion(
                        workers,
                        {n: s for n, s in entrants.items() if n != champion.stem},
                        champion,
                        CHAMPION_GAMES,
                        seed=round_number,
                    )

                    # Promotion: significantly beats the champion head to head and does
                    # at least as well in the Swiss. The GA is not a PPOBrain, so it is
                    # reported but not promoted.
                    candidates = [
                        n
                        for n, r in h2h.items()
                        if r["record"]["gd"] - r["record"]["gd_ci"] > 0
                        and swiss_points.get(n, 0) >= swiss_points.get(champion.stem, 0)
                    ]
                    if "GA" in candidates:
                        notes.append(
                            "The GA beat the champion significantly (GeneticBrains are not promoted)."
                        )
                        candidates.remove("GA")
                    best = max(
                        candidates, key=lambda n: h2h[n]["record"]["gd"], default=None
                    )
                    # Confirmation: a candidate is promoted only if it beats the same
                    # champion significantly in two consecutive rounds (fresh games
                    # each round). Picking the best of several candidates on one set
                    # of games otherwise promotes luck (see the verdicts).
                    new_champion = None
                    if best and pending == best:
                        new_champion = best
                    elif best:
                        notes.append(
                            f"PENDING: {best} beat the champion significantly; it is promoted "
                            "if it does so again next round"
                        )
                    pending = best

                    # Replacement, rated against the champion (the same opponent for all).
                    ratings = {n: h2h[n]["record"] for n in learners if n in h2h}
                    for name, r in ratings.items():
                        learners[name].ratings.append(r["gd"])
                    for architecture in {
                        learner.architecture for learner in learners.values()
                    }:
                        group = [
                            n
                            for n in ratings
                            if learners[n].architecture == architecture
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
                                    f"{leader} vs the champion, not improving): adopts {leader}'s "
                                    f"networks with {config}"
                                )
                                break

                    coach_team = LEAGUE_DIR / "coach" / "team.json"
                    if coach_team.exists():
                        notes.append(
                            f"Coach team (roles 0-4): {json.loads(coach_team.read_text())}"
                        )
                    ga_best = LEAGUE_DIR / "ga" / "best.json"
                    if ga_best.exists():
                        info = json.loads(ga_best.read_text())
                        notes.append(
                            f"GA: generation {info.get('generation')}, fitness {info.get('fitness', 0):.2f}"
                        )

                    log_round(round_number, standings, entrants, h2h, iterations, notes)

                    if new_champion and args.dry_run:
                        log(f"  (dry run) would promote {new_champion}")
                    elif new_champion:
                        champion = promote(entrants[new_champion])
                        pending = None
                        for learner in learners.values():
                            learner.ratings = (
                                []
                            )  # ratings were against the old champion
                        log(
                            f"  NEW CHAMPION: {new_champion} -> {champion.stem} (and PPOBrain.npz)"
                        )
                        if verdict:
                            verdict.stop()  # its champion has been superseded
                        verdict = Verdict(champion)
                        log(
                            f"  Verdict tournament on {champion.stem} started in the background"
                        )

                    state_file.write_text(
                        json.dumps(
                            {
                                "round": round_number,
                                "pending": pending,
                                "learners": {n: l.state() for n, l in learners.items()},
                            },
                            indent=1,
                        )
                    )

                    # Sparring partners: every other entrant and every champion.
                    for name, learner in learners.items():
                        for old in learner.pool_dir.glob("*"):
                            old.unlink()
                        for other, spec in entrants.items():
                            if other != name:
                                shutil.copy(
                                    spec,
                                    learner.pool_dir / f"{other}{Path(spec).suffix}",
                                )
                        for champ in HISTORY_DIR.glob("PPO-champ-*.npz"):
                            shutil.copy(champ, learner.pool_dir / champ.name)
                finally:
                    for process in everyone:
                        process.signal(signal.SIGCONT)
    finally:
        if verdict:
            verdict.stop()
        for process in everyone:
            process.signal(signal.SIGTERM)
        log(
            f"League stopped. Champion: {champion.stem} (also in {PPOBrain.WEIGHTS_FILE})"
        )


if __name__ == "__main__":
    main()
