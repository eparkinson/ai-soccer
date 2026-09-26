"""
Train PPOBrain: behaviour cloning of DefendersAndAttackers, then PPO.

1. Behaviour cloning. The policy imitates DefendersAndAttackers (the strongest
   heuristic) from its own games, then a few rounds of DAgger: the clone plays and
   DefendersAndAttackers labels the positions it reaches. PPO from scratch plateaus well
   below the heuristics; starting from a clone it starts level and improves from there.
2. Critic warm-up. A few iterations train only the value network, so PPO's first
   policy updates are not driven by an untrained critic.
3. PPO. Each iteration plays a batch of games in parallel, then runs a PPO update on
   the learner's experience. Opponents are a mix of the current policy (self-play, both
   sides' experience used), earlier snapshots from this run, and fixed opponents: the
   heuristic brains and earlier saved PPOBrain versions (aisoccer/brains/weights/history).

Every --eval-every iterations the deterministic policy plays every fixed opponent on
both sides of the field and each opponent's record against it is printed with 95%
confidence intervals. The version with the best worst-case matchup is saved to
aisoccer/brains/weights/PPOBrain.npz.

Progress is printed and appended to runs/ppo/progress.log, so you can watch it with:

    tail -f runs/ppo/progress.log

Usage:

    poetry run python train_ppo.py --iterations 400
    poetry run python train_ppo.py --resume   # continue from runs/ppo/latest.npz
"""

import os

# The PPO update's matmuls are fastest on a handful of BLAS threads; using every core
# is slower. Must be set before numpy is imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from aisoccer.abstractbrain import AbstractBrain  # noqa: E402
from aisoccer.brains.AdaptiveChaser import AdaptiveChaser  # noqa: E402
from aisoccer.brains.BehindAndTowards import BehindAndTowards  # noqa: E402
from aisoccer.brains.DefendersAndAttackers import DefendersAndAttackers  # noqa: E402
from aisoccer.brains.PPOBrain import ACT_DIM, OBS_DIM, PPOBrain, player_features  # noqa
from aisoccer.brains.RandomWalk import RandomWalk  # noqa: E402
from aisoccer.brains.SimpleBrain import SimpleBrain  # noqa: E402
from aisoccer.brains.StrategicPlanner import StrategicPlanner  # noqa: E402
from aisoccer.brainspec import load_brain  # noqa: E402
from aisoccer.constants import Constants  # noqa: E402
from aisoccer.game import Game  # noqa: E402
from aisoccer.ppo import PPO, Adam, gae  # noqa: E402
from aisoccer.vecgame import VecGame  # noqa: E402

HEURISTICS = {
    "DefendersAndAttackers": DefendersAndAttackers,
    "BehindAndTowards": BehindAndTowards,
    "StrategicPlanner": StrategicPlanner,
    "AdaptiveChaser": AdaptiveChaser,
    "SimpleBrain": SimpleBrain,
    "RandomWalk": RandomWalk,
}
# Earlier PPOBrain versions, kept as fixed opponents so each new version has to beat a
# variety of strategies rather than overfit one heuristic. Saved (policy only, with the
# action repeat each was trained with) in aisoccer/brains/weights/history/.
HISTORY_DIR = PPOBrain.WEIGHTS_FILE.parent / "history"
PAST_VERSIONS = ["PPO-scratch", "PPO-clone", "PPO-it80", "PPO-it120", "PPO-first-it230"]

# Every fixed opponent, for evaluation.
OPPONENTS = list(HEURISTICS) + PAST_VERSIONS

# Training games against fixed opponents draw them with these weights. The stronger
# heuristics are the more useful sparring partners, but no single brain dominates.
OPPONENT_WEIGHTS = {
    "DefendersAndAttackers": 5,
    "BehindAndTowards": 3,
    "StrategicPlanner": 3,
    "AdaptiveChaser": 1,
    "SimpleBrain": 1,
    "RandomWalk": 0.5,
    **{name: 1.5 for name in PAST_VERSIONS},
}


def make_opponent(name):
    """A heuristic brain, a saved PPOBrain version, or ("file:<path>") any weights file."""
    if name in HEURISTICS:
        return HEURISTICS[name]()
    if name.startswith("file:"):
        return load_brain(name.removeprefix("file:"))
    return PPOBrain(name, weights=PPOBrain.load_weights(HISTORY_DIR / f"{name}.npz"))


def save_atomic(path, save):
    """Write via a temporary file so readers (e.g. league.py) never see a partial file."""
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.npz")
    save(tmp)
    os.replace(tmp, path)


RUN_DIR = Path("runs/ppo")
LOG_FILE = RUN_DIR / "progress.log"


def log(message=""):
    print(message, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")


def training_brains(task, make=None):
    """The learner and its opponent for a training game, and whether the learner is blue."""
    weights, opponent, opponent_weights, seed, gamma, reward = task
    if opponent.startswith("file:") and not Path(opponent.removeprefix("file:")).exists():
        opponent = "self"  # the league replaced the pool between rounds: play itself instead
    rng = np.random.default_rng(seed)
    learner = PPOBrain("learner", weights=weights, training=True, reward=reward)

    if opponent == "self":
        other = PPOBrain("self", weights=weights, training=True, reward=reward)
    elif opponent == "snapshot":
        other = PPOBrain("snapshot", weights=opponent_weights, deterministic=False)
    else:
        other = (make or make_opponent)(opponent)

    learner_blue = rng.random() < 0.5
    return learner, other, learner_blue


def training_outcome(task, learner, other, learner_blue, score):
    """What a training game returns: (opponent, goals for, against, trajectories)."""
    _, opponent, _, _, gamma, _ = task
    mine, theirs = (
        (score["blue"], score["red"]) if learner_blue else (score["red"], score["blue"])
    )

    trajectories = [learner.finish_trajectory(gamma)]
    if opponent == "self":
        trajectories.append(other.finish_trajectory(gamma))
    return opponent, mine, theirs, trajectories


def play_training_game(task):
    """Play one game in a worker and return the learner's (and self-play twin's) experience."""
    learner, other, learner_blue = training_brains(task)
    blue, red = (learner, other) if learner_blue else (other, learner)
    score = Game(blue, red, game_length=GAME_LENGTH, quiet_mode=True, seed=task[3]).play()
    return training_outcome(task, learner, other, learner_blue, score)


def vectorisable(task):
    """Games whose brains are all PPOBrains can be played together in a VecGame."""
    opponent = task[1]
    return opponent in ("self", "snapshot") or (
        opponent.startswith("file:") and opponent.endswith(".npz")
    )


def play_training_games(tasks, precision="exact"):
    """
    Play several vectorisable training games together in one VecGame and return
    play_training_game's result for each task: game k is the game that
    play_training_game(tasks[k]) plays, bit for bit, with the same trajectories.
    ``precision="float32"`` runs the networks in single precision: several times
    faster, with network outputs within about 1e-7 of float64, so the games start the
    same but drift apart after a few hundred ticks (statistically the same games).
    """
    loaded = {}

    def make(opponent):  # pool files are read once, their networks shared by the games
        if opponent not in loaded:
            loaded[opponent] = PPOBrain.load_weights(opponent.removeprefix("file:"))
        name = Path(opponent.removeprefix("file:")).stem
        return PPOBrain(name, weights=loaded[opponent])

    games = [training_brains(task, make) for task in tasks]
    blue = [lr if lb else o for lr, o, lb in games]
    red = [o if lb else lr for lr, o, lb in games]
    scores = VecGame(blue, red, [task[3] for task in tasks], game_length=GAME_LENGTH, precision=precision).play()
    return [
        training_outcome(task, *game, score)
        for task, game, score in zip(tasks, games, scores)
    ]


def play_work_item(item):
    """A worker's share of an iteration: one game, or ("batch", tasks) for a VecGame."""
    if item[0] == "batch":
        return play_training_games(item[1], item[2])
    return [play_training_game(item[1])]


def play_iteration_games(workers, tasks, vector_games, precision="exact"):
    """
    Play an iteration's training games and return their outcomes in task order. With
    vector_games, games against PPOBrains (self-play, snapshots, .npz pool files) are
    played in VecGame batches of at most that many games, split evenly across batches;
    the rest are played one at a time as before. ``precision`` is the VecGame
    networks' precision (see play_training_games).
    """
    if not vector_games:
        return workers.map(play_training_game, tasks)
    vector = [i for i, t in enumerate(tasks) if vectorisable(t)]
    single = [i for i, t in enumerate(tasks) if not vectorisable(t)]
    n_batches = -(-len(vector) // vector_games)
    batches = [list(b) for b in np.array_split(vector, n_batches)] if vector else []
    items = [("batch", [tasks[i] for i in b], precision) for b in batches]
    items += [("single", tasks[i]) for i in single]
    order = [i for b in batches for i in b] + single
    results = [o for r in workers.map(play_work_item, items, chunksize=1) for o in r]
    outcomes = [None] * len(tasks)
    for i, outcome in zip(order, results):
        outcomes[i] = outcome
    return outcomes


def capped(action):
    """The game caps accelerations at magnitude 1, so that is what to imitate."""
    action = np.asarray(action, dtype=float)
    norms = np.linalg.norm(action, axis=1, keepdims=True)
    return np.where(norms > 1, action / np.maximum(norms, 1), action)


class Demonstrator(AbstractBrain):
    """Plays as the teacher brain and records what it sees and does, for cloning."""

    def __init__(self, teacher):
        super().__init__(name="demonstrator")
        self.teacher = make_opponent(teacher)
        self.ticks = 0
        self.obs: list[np.ndarray] = []
        self.act: list[np.ndarray] = []

    def do_move(self):
        view = (
            self.my_players_pos,
            self.my_players_vel,
            self.opp_players_pos,
            self.opp_players_vel,
            self.ball_pos,
            self.ball_vel,
            self.my_score,
            self.opp_score,
            self.game_time,
        )
        action = np.asarray(self.teacher.move(*view), dtype=float)
        self.ticks += 1
        # Record at the rate PPOBrain makes decisions.
        if self.ticks % PPOBrain.ACTION_REPEAT == 0:
            self.obs.append(player_features(*view))
            self.act.append(capped(action))
        return action


def play_demo_game(task):
    """The teacher vs a random heuristic; returns its (obs, actions) per player."""
    seed, teacher = task
    rng = np.random.default_rng(seed)
    demonstrator = Demonstrator(teacher)
    other = HEURISTICS[str(rng.choice(list(HEURISTICS)))]()
    blue, red = (demonstrator, other) if rng.random() < 0.5 else (other, demonstrator)
    Game(blue, red, quiet_mode=True, seed=seed).play()
    obs = np.array(demonstrator.obs, dtype=np.float32).reshape(-1, OBS_DIM)
    act = np.array(demonstrator.act, dtype=np.float32).reshape(-1, ACT_DIM)
    return obs, act


class DaggerStudent(PPOBrain):
    """
    Plays with the cloned policy but records what the teacher would have done in each
    position it reaches (DAgger), so cloning also learns to recover from the student's
    own mistakes.
    """

    def __init__(self, weights, teacher):
        super().__init__("student", weights=weights)
        self.teacher = make_opponent(teacher)
        self.obs: list[np.ndarray] = []
        self.act: list[np.ndarray] = []

    def decide(self):
        super().decide()
        view = (
            self.my_players_pos,
            self.my_players_vel,
            self.opp_players_pos,
            self.opp_players_vel,
            self.ball_pos,
            self.ball_vel,
            self.my_score,
            self.opp_score,
            self.game_time,
        )
        self.obs.append(player_features(*view))
        self.act.append(capped(self.teacher.move(*view)))


def play_dagger_game(task):
    """The student vs a random heuristic; returns the teacher-labelled (obs, actions)."""
    weights, seed, teacher = task
    rng = np.random.default_rng(seed)
    student = DaggerStudent(weights, teacher)
    other = HEURISTICS[str(rng.choice(list(HEURISTICS)))]()
    blue, red = (student, other) if rng.random() < 0.5 else (other, student)
    Game(blue, red, quiet_mode=True, seed=seed).play()
    obs = np.array(student.obs, dtype=np.float32).reshape(-1, OBS_DIM)
    act = np.array(student.act, dtype=np.float32).reshape(-1, ACT_DIM)
    return obs, act


def play_eval_game(task):
    """Deterministic learner vs a fixed opponent. Returns (opponent, goals for, against)."""
    weights, opponent, learner_blue, seed = task
    learner = PPOBrain("PPOBrain", weights=weights)
    other = make_opponent(opponent)
    blue, red = (learner, other) if learner_blue else (other, learner)
    score = Game(blue, red, quiet_mode=True, seed=seed).play()
    if learner_blue:
        return opponent, score["blue"], score["red"]
    return opponent, score["red"], score["blue"]


def build_batch(trajectories, gamma, lam):
    """Flatten per-player trajectories into one PPO batch, with GAE per player."""
    obs, act, logp, adv, ret = [], [], [], [], []
    for traj in trajectories:
        steps = len(traj["rew"])
        if steps == 0:
            continue
        dones = np.zeros(steps)
        dones[-1] = 1.0  # end of game
        for p in range(traj["obs"].shape[1]):
            a, r = gae(traj["rew"], traj["val"][:, p], dones, 0.0, gamma, lam)
            adv.append(a)
            ret.append(r)
            obs.append(traj["obs"][:, p])
            act.append(traj["act"][:, p])
            logp.append(traj["logp"][:, p])
    return (
        np.concatenate(obs),
        np.concatenate(act),
        np.concatenate(logp),
        np.concatenate(adv),
        np.concatenate(ret),
    )


def results_line(results):
    """Summarise (opponent, for, against) results as W-D-L and goals per opponent."""
    by_opp: dict[str, list] = {}
    for opp, gf, ga in results:
        by_opp.setdefault(opp, []).append((gf, ga))
    parts = []
    for opp, games in sorted(by_opp.items()):
        w = sum(gf > ga for gf, ga in games)
        d = sum(gf == ga for gf, ga in games)
        lo = len(games) - w - d
        gf = sum(g[0] for g in games)
        ga = sum(g[1] for g in games)
        parts.append(f"{opp[:8]} {w}-{d}-{lo} ({gf}:{ga})")
    return "  ".join(parts)


# DefendersAndAttackers is the brain to beat and games against it are mostly low
# scoring draws, so it gets more evaluation games.
EVAL_GAMES_MULTIPLIER = {"DefendersAndAttackers": 3}


def evaluate(pool, weights, games_per_opponent, seed=0):
    """
    Play the deterministic policy against every fixed opponent (the heuristic brains and
    earlier PPOBrain versions), alternating sides, and
    log each brain's record against PPOBrain with 95% confidence intervals. Returns
    PPOBrain's points per game against its hardest opponent.

    Goals are rare (often under one per game) and roughly Poisson, so short evaluations
    are dominated by noise: see the analysis in README.md. The same seeds are used
    every time so that checkpoints are compared on the same kick-offs.
    """
    rng = np.random.default_rng(seed)
    tasks = []
    for opp in OPPONENTS:
        for g in range(games_per_opponent * EVAL_GAMES_MULTIPLIER.get(opp, 1)):
            tasks.append((weights, opp, g % 2 == 0, int(rng.integers(2**31))))
    results = pool.map(play_eval_game, tasks)

    # Rows are each brain's own record against PPOBrain, so a brain stronger than
    # PPOBrain has a positive goal difference and weaker brains a negative one.
    log(
        "  Each brain's record against PPOBrain (95% CIs); positive GD = stronger than PPOBrain:"
    )
    log(
        f"  {'BRAIN':<22} {'N':>4} {'W':>4} {'D':>4} {'L':>4} {'GF':>5} {'GA':>5}"
        f" {'GD/GAME':>13} {'PTS/GAME':>13}"
    )
    ppo_points, ppo_variances = [], []
    for opp in OPPONENTS:
        # (opponent goals, PPOBrain goals) per game
        games = np.array([(ga, gf) for o, gf, ga in results if o == opp], dtype=float)
        n = len(games)
        gd = games[:, 0] - games[:, 1]
        points = np.where(gd > 0, 3.0, np.where(gd == 0, 1.0, 0.0))
        gd_ci = 1.96 * gd.std(ddof=1) / np.sqrt(n)
        pts_ci = 1.96 * points.std(ddof=1) / np.sqrt(n)
        log(
            f"  {opp:<22} {n:>4} {(gd > 0).sum():>4} {(gd == 0).sum():>4} {(gd < 0).sum():>4}"
            f" {int(games[:, 0].sum()):>5} {int(games[:, 1].sum()):>5}"
            f" {gd.mean():>+6.2f} ±{gd_ci:<5.2f} {points.mean():>6.2f} ±{pts_ci:<5.2f}"
        )
        mine = np.where(gd < 0, 3.0, np.where(gd == 0, 1.0, 0.0))
        ppo_points.append(mine.mean())
        ppo_variances.append(mine.var(ddof=1) / n)

    mean = float(np.mean(ppo_points))
    mean_ci = 1.96 * np.sqrt(np.sum(ppo_variances)) / len(ppo_points)
    worst = int(np.argmin(ppo_points))
    log(
        f"  PPOBrain points/game, all opponents equally weighted: {mean:.2f} ±{mean_ci:.2f}"
    )
    log(
        f"  PPOBrain points/game, hardest opponent ({OPPONENTS[worst]}):"
        f" {ppo_points[worst]:.2f} ±{1.96 * np.sqrt(ppo_variances[worst]):.2f}"
    )
    # Being the best brain means doing well against every opponent, so checkpoints are
    # ranked by their worst matchup rather than an average that big wins over weak
    # brains can inflate.
    return float(ppo_points[worst])


CONFIG_KEYS = {
    "action_repeat",
    "game_length",
    "lr",
    "gamma",
    "lam",
    "min_std",
    "target_kl",
    "self_play",
    "snapshots",
    "pool_share",
    "reward",
    "opponent_weights",
}


GAME_LENGTH = Constants.GAME_LENGTH  # ticks per training game (--game-length)


def set_action_repeat(ticks, game_length=None):
    """
    The learner's (and its self-play twin's) decisions last `ticks` ticks; training
    games last `game_length` ticks. Run in the main process and in every worker.
    """
    global GAME_LENGTH
    PPOBrain.ACTION_REPEAT = ticks
    GAME_LENGTH = game_length or Constants.GAME_LENGTH


def apply_config(args, config):
    """Override settings from a config dict (see --config)."""
    unknown = set(config) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    for key, value in config.items():
        setattr(args, key, value)


def load_networks(ppo, weights):
    """
    Copy policy, value and noise from a weights dict into the learner, in place.
    Returns False if the weights have no value network (e.g. a saved champion), in
    which case the learner keeps its fresh critic, which then needs a warm-up.
    """
    ppo.policy.params[:] = [np.array(p, dtype=float) for p in weights["policy"]]
    ppo.log_std[:] = weights["log_std"]
    if "value" not in weights:
        return False
    ppo.value.params[:] = [np.array(p, dtype=float) for p in weights["value"]]
    return True


def adopt(ppo, args, iteration):
    """
    Population-based training hook: if league.py has left adopt.npz (and optionally
    adopt.json) in the run directory, switch to those networks and settings, with fresh
    optimisers. Returns True if it adopted.
    """
    weights_file = RUN_DIR / "adopt.npz"
    if not weights_file.exists():
        return False
    config_file = RUN_DIR / "adopt.json"
    load_networks(ppo, PPOBrain.load_weights(weights_file))
    if config_file.exists():
        apply_config(args, json.loads(config_file.read_text()))
        config_file.unlink()
    weights_file.unlink()
    ppo.min_log_std = float(np.log(args.min_std))
    ppo.target_kl = args.target_kl
    np.maximum(ppo.log_std, ppo.min_log_std, out=ppo.log_std)
    ppo.policy_opt = Adam(ppo.policy.params + [ppo.log_std], lr=args.lr)
    ppo.value_opt = Adam(ppo.value.params, lr=args.lr)
    log(f"ADOPTED new networks and settings at iteration {iteration}: {settings(args)}")
    return True


def settings(args):
    return {k: getattr(args, k) for k in sorted(CONFIG_KEYS - {"opponent_weights"})}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--games", type=int, default=48, help="games per iteration")
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2)
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument(
        "--game-length", type=int, default=Constants.GAME_LENGTH, help="ticks per training game"
    )
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=PPOBrain.ACTION_REPEAT,
        help="ticks each decision is held for (saved with the weights)",
    )
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument(
        "--eval-games",
        type=int,
        default=48,
        help="per opponent (x3 for DefendersAndAttackers)",
    )
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument("--self-play", type=float, default=0.15)
    parser.add_argument("--snapshots", type=float, default=0.1)
    parser.add_argument(
        "--bc-games",
        type=int,
        default=200,
        help="games of DefendersAndAttackers to imitate before PPO (0 to start from scratch)",
    )
    parser.add_argument("--bc-epochs", type=int, default=40)
    parser.add_argument(
        "--bc-teacher",
        default="DefendersAndAttackers",
        help="brain to imitate: a heuristic brain name or file:<weights path>",
    )
    parser.add_argument(
        "--hidden", type=int, nargs="+", default=[128, 128], help="hidden layer sizes"
    )
    parser.add_argument("--dagger-rounds", type=int, default=2)
    parser.add_argument("--dagger-games", type=int, default=100)
    parser.add_argument("--dagger-epochs", type=int, default=15)
    parser.add_argument(
        "--value-warmup",
        type=int,
        default=10,
        help="iterations that train only the critic at the start (after cloning, or after "
        "resuming with a changed reward)",
    )
    parser.add_argument("--init-log-std", type=float, default=-1.0)
    parser.add_argument(
        "--min-std", type=float, default=0.25, help="floor on the exploration noise"
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.01,
        help="stop each update's epochs past this KL",
    )
    parser.add_argument(
        "--resume-weights",
        type=Path,
        help="with --resume, start the networks from this checkpoint (e.g. an earlier "
        "best) instead of the latest",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--init-weights",
        type=Path,
        help="start from this checkpoint (policy, value and noise) instead of cloning",
    )
    parser.add_argument("--run-dir", type=Path, default=Path("runs/ppo"))
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON overriding settings: lr, min_std, target_kl, self_play, snapshots, "
        "pool_share, reward (weights), opponent_weights",
    )
    parser.add_argument(
        "--pool-dir",
        type=Path,
        help="directory of opponent weights files (e.g. other learners in a league); "
        "pool_share of games are played against a random one",
    )
    parser.add_argument("--pool-share", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--vector-games",
        type=int,
        default=0,
        help="play training games against PPOBrains (self-play, snapshots, .npz pool "
        "files) together in VecGame batches of up to this many games (0: one at a time). "
        "Same games for far less CPU; about games/workers keeps every worker busy",
    )
    parser.add_argument(
        "--vector-precision",
        choices=["exact", "float32"],
        default="exact",
        help="with --vector-games: 'exact' plays bit-identical games to the one-at-a-time "
        "path; 'float32' runs the networks in single precision, faster still, its games "
        "the same up to float32 rounding (they drift apart after a few hundred ticks)",
    )
    args = parser.parse_args()
    args.reward = None
    args.opponent_weights = dict(OPPONENT_WEIGHTS)
    if args.config:
        apply_config(args, json.loads(args.config.read_text()))

    global RUN_DIR, LOG_FILE
    RUN_DIR = args.run_dir
    LOG_FILE = RUN_DIR / "progress.log"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    ppo = PPO(
        OBS_DIM,
        ACT_DIM,
        hidden=tuple(args.hidden),
        lr=args.lr,
        init_log_std=args.init_log_std,
        min_log_std=float(np.log(args.min_std)),
        target_kl=args.target_kl,
        seed=args.seed,
    )
    start_iteration = 0
    if args.resume:
        state = np.load(RUN_DIR / "latest.npz", allow_pickle=True)["state"].item()
        ppo.policy.params[:] = state["policy"]
        ppo.value.params[:] = state["value"]
        ppo.log_std[:] = state["log_std"]
        start_iteration = int(state["iteration"])
        log(f"Resumed from iteration {start_iteration}")
        if args.resume_weights:
            weights = PPOBrain.load_weights(args.resume_weights)
            ppo.policy.params[:] = weights["policy"]
            ppo.value.params[:] = weights["value"]
            ppo.log_std[:] = weights["log_std"]
            log(f"  with the networks from {args.resume_weights}")
    if args.init_weights:
        log(f"Starting from the networks in {args.init_weights}")
        if not load_networks(ppo, PPOBrain.load_weights(args.init_weights)):
            args.value_warmup = max(args.value_warmup, 10)
            log(
                f"  (no value network there: training a fresh critic for {args.value_warmup} iterations first)"
            )
    np.maximum(ppo.log_std, ppo.min_log_std, out=ppo.log_std)

    # Re-point the optimisers at the (possibly replaced) parameter arrays.
    ppo.policy_opt.params = ppo.policy.params + [ppo.log_std]
    ppo.value_opt.params = ppo.value.params

    best_score = -1.0

    set_action_repeat(args.action_repeat, args.game_length)
    log(
        f"Training PPOBrain: {args.games} games/iteration on {args.workers} workers, "
        f"obs {OBS_DIM}, action repeat {PPOBrain.ACTION_REPEAT}"
    )
    log(f"Settings: {settings(args)}")

    # Workers are fresh processes: they get the learner's action repeat too.
    with Pool(args.workers, initializer=set_action_repeat, initargs=(args.action_repeat, args.game_length)) as workers:
        best_weights = PPOBrain.load_weights(PPOBrain.WEIGHTS_FILE)
        if args.resume and best_weights is not None:
            log("\nBASELINE: the saved best weights")
            best_score = evaluate(workers, best_weights, args.eval_games, seed=0)
            log()

        cloned = not args.resume and not args.init_weights and args.bc_games > 0
        if cloned:
            log(f"\nBEHAVIOUR CLONING {args.bc_teacher} from {args.bc_games} games")
            demos = workers.map(
                play_demo_game,
                [(s, args.bc_teacher) for s in range(10**6, 10**6 + args.bc_games)],
            )
            obs = np.concatenate([d[0] for d in demos]).astype(float)
            act = np.concatenate([d[1] for d in demos]).astype(float)
            del demos
            losses = ppo.clone(obs, act, epochs=args.bc_epochs)
            log(
                f"  {len(losses)} epochs, action MSE {losses[0]:.4f} -> {losses[-1]:.4f}"
            )
            for r in range(args.dagger_rounds):
                first = 2 * 10**6 + r * args.dagger_games
                tasks = [
                    (ppo.weights(), s, args.bc_teacher)
                    for s in range(first, first + args.dagger_games)
                ]
                new = workers.map(play_dagger_game, tasks)
                obs = np.concatenate([obs] + [d[0].astype(float) for d in new])
                act = np.concatenate([act] + [d[1].astype(float) for d in new])
                del new
                losses = ppo.clone(obs, act, epochs=args.dagger_epochs)
                log(
                    f"  DAgger round {r + 1}: {len(obs)} samples, "
                    f"action MSE {losses[0]:.4f} -> {losses[-1]:.4f}"
                )
            del obs, act
            log("\nEVALUATION of the cloned policy:")
            best_score = evaluate(workers, ppo.weights(), args.eval_games, seed=0)
            PPOBrain.save_weights(PPOBrain.WEIGHTS_FILE, ppo.weights())
            log()

        league = [ppo.weights()]

        for iteration in range(
            start_iteration + 1, start_iteration + args.iterations + 1
        ):
            started = time.time()
            if adopt(ppo, args, iteration):
                league = [ppo.weights()]
            # Linearly decay the learning rate to 10% over the run.
            progress = (iteration - start_iteration - 1) / args.iterations
            ppo.set_lr(args.lr * (1.0 - 0.9 * progress))
            weights = ppo.weights()

            opponent_names = list(args.opponent_weights)
            opponent_p = np.array(
                [args.opponent_weights[n] for n in opponent_names], float
            )
            opponent_p /= max(opponent_p.sum(), 1e-9)
            pool = sorted(args.pool_dir.glob("*")) if args.pool_dir else []
            pool = [
                p
                for p in pool
                if p.suffix in (".npz", ".json") and ".tmp" not in p.name
            ]

            tasks = []
            for _ in range(args.games):
                roll = rng.random()
                seed = int(rng.integers(2**31))
                task_end = (seed, args.gamma, args.reward)
                if roll < args.self_play:
                    tasks.append((weights, "self", None, *task_end))
                elif roll < args.self_play + args.snapshots:
                    snap = league[rng.integers(len(league))]
                    tasks.append((weights, "snapshot", snap, *task_end))
                elif pool and roll < args.self_play + args.snapshots + args.pool_share:
                    opp = "file:" + str(pool[rng.integers(len(pool))])
                    tasks.append((weights, opp, None, *task_end))
                elif not opponent_names:
                    # No fixed opponents (a league that starts from zero): the league
                    # pool if there is one yet, otherwise self-play.
                    if pool:
                        opp = "file:" + str(pool[rng.integers(len(pool))])
                        tasks.append((weights, opp, None, *task_end))
                    else:
                        tasks.append((weights, "self", None, *task_end))
                else:
                    opp = str(rng.choice(opponent_names, p=opponent_p))
                    tasks.append((weights, opp, None, *task_end))

            outcomes = play_iteration_games(
                workers, tasks, args.vector_games, args.vector_precision
            )
            trajectories = [t for o in outcomes for t in o[3]]
            batch = build_batch(trajectories, args.gamma, args.lam)
            played = time.time() - started
            warming_up = iteration - start_iteration <= args.value_warmup
            stats = ppo.update(*batch, train_policy=not warming_up)

            mean_reward = np.mean([t["rew"].sum() for t in trajectories])
            log(
                f"it {iteration:4d} | {time.time() - started:4.1f}s "
                f"(play {played:4.1f}s) | samples {len(batch[0]):6d} | "
                f"ret {mean_reward:+.2f} | std {np.exp(ppo.log_std).mean():.2f} | "
                f"kl {stats['approx_kl']:.4f} clip {stats['clip_frac']:.2f} "
                f"ep {stats['epochs']:.0f} "
                f"vloss {stats['value_loss']:.4f} | "
                + results_line(
                    [(o, gf, ga) for o, gf, ga, _ in outcomes if o != "self"]
                )
            )

            if iteration % args.snapshot_every == 0:
                league.append(ppo.weights())

            state = dict(ppo.weights(), iteration=iteration)
            save_atomic(
                RUN_DIR / "latest.npz",
                lambda p: np.savez(p, state=np.array(state, dtype=object)),
            )
            # The current networks as a plain weights file, e.g. for league.py.
            save_atomic(
                RUN_DIR / "policy.npz",
                lambda p: PPOBrain.save_weights(p, ppo.weights()),
            )

            if args.eval_every and iteration % args.eval_every == 0:
                log(f"\nEVALUATION after iteration {iteration} (deterministic policy):")
                score = evaluate(workers, ppo.weights(), args.eval_games)
                PPOBrain.save_weights(RUN_DIR / f"it{iteration:04d}.npz", ppo.weights())
                if score > best_score:
                    best_score = score
                    PPOBrain.save_weights(PPOBrain.WEIGHTS_FILE, ppo.weights())
                    log(
                        f"  New best ({score:.2f} points/game against the hardest opponent):"
                        f" saved {PPOBrain.WEIGHTS_FILE}"
                    )
                log()


if __name__ == "__main__":
    main()
