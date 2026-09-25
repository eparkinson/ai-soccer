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
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from aisoccer.brains.AdaptiveChaser import AdaptiveChaser  # noqa: E402
from aisoccer.brains.BehindAndTowards import BehindAndTowards  # noqa: E402
from aisoccer.brains.DefendersAndAttackers import DefendersAndAttackers  # noqa: E402
from aisoccer.brains.PPOBrain import ACT_DIM, OBS_DIM, PPOBrain, player_features  # noqa: E402
from aisoccer.brains.RandomWalk import RandomWalk  # noqa: E402
from aisoccer.brains.SimpleBrain import SimpleBrain  # noqa: E402
from aisoccer.brains.StrategicPlanner import StrategicPlanner  # noqa: E402
from aisoccer.game import Game  # noqa: E402
from aisoccer.ppo import PPO, gae  # noqa: E402

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
PAST_VERSIONS = ["PPO-scratch", "PPO-clone", "PPO-it80", "PPO-it120"]

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
    if name in HEURISTICS:
        return HEURISTICS[name]()
    return PPOBrain(name, weights=PPOBrain.load_weights(HISTORY_DIR / f"{name}.npz"))


RUN_DIR = Path("runs/ppo")
LOG_FILE = RUN_DIR / "progress.log"


def log(message=""):
    print(message, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")


def play_training_game(task):
    """Play one game in a worker and return the learner's (and self-play twin's) experience."""
    weights, opponent, opponent_weights, seed, gamma = task
    rng = np.random.default_rng(seed)
    learner = PPOBrain("learner", weights=weights, training=True)

    if opponent == "self":
        other = PPOBrain("self", weights=weights, training=True)
    elif opponent == "snapshot":
        other = PPOBrain("snapshot", weights=opponent_weights, deterministic=False)
    else:
        other = make_opponent(opponent)

    learner_blue = rng.random() < 0.5
    blue, red = (learner, other) if learner_blue else (other, learner)
    score = Game(blue, red, quiet_mode=True, seed=seed).play()
    mine, theirs = (
        (score["blue"], score["red"]) if learner_blue else (score["red"], score["blue"])
    )

    trajectories = [learner.finish_trajectory(gamma)]
    if opponent == "self":
        trajectories.append(other.finish_trajectory(gamma))
    return opponent, mine, theirs, trajectories


class Demonstrator(DefendersAndAttackers):
    """DefendersAndAttackers that records what it sees and does, for behaviour cloning."""

    def __init__(self):
        super().__init__()
        self.ticks = 0
        self.obs: list[np.ndarray] = []
        self.act: list[np.ndarray] = []

    def do_move(self):
        action = np.asarray(super().do_move(), dtype=float)
        self.ticks += 1
        # Record at the rate PPOBrain makes decisions.
        if self.ticks % PPOBrain.ACTION_REPEAT == 0:
            self.obs.append(
                player_features(
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
            )
            # The game caps accelerations at magnitude 1, so that is what to imitate.
            norms = np.linalg.norm(action, axis=1, keepdims=True)
            self.act.append(np.where(norms > 1, action / np.maximum(norms, 1), action))
        return action


def play_demo_game(seed):
    """DefendersAndAttackers vs a random heuristic; returns its (obs, actions) per player."""
    rng = np.random.default_rng(seed)
    teacher = Demonstrator()
    other = HEURISTICS[str(rng.choice(list(HEURISTICS)))]()
    blue, red = (teacher, other) if rng.random() < 0.5 else (other, teacher)
    Game(blue, red, quiet_mode=True, seed=seed).play()
    obs = np.array(teacher.obs, dtype=np.float32).reshape(-1, OBS_DIM)
    act = np.array(teacher.act, dtype=np.float32).reshape(-1, ACT_DIM)
    return obs, act


class DaggerStudent(PPOBrain):
    """
    Plays with the cloned policy but records what DefendersAndAttackers would have done
    in each position it reaches (DAgger), so cloning also learns to recover from the
    student's own mistakes.
    """

    def __init__(self, weights):
        super().__init__("student", weights=weights)
        self.teacher = DefendersAndAttackers()
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
        action = np.asarray(self.teacher.move(*view), dtype=float)
        norms = np.linalg.norm(action, axis=1, keepdims=True)
        self.obs.append(player_features(*view))
        self.act.append(np.where(norms > 1, action / np.maximum(norms, 1), action))


def play_dagger_game(task):
    """The student vs a random heuristic; returns the teacher-labelled (obs, actions)."""
    weights, seed = task
    rng = np.random.default_rng(seed)
    student = DaggerStudent(weights)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--games", type=int, default=48, help="games per iteration")
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2)
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
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
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    ppo = PPO(
        OBS_DIM,
        ACT_DIM,
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
        start_iteration = state["iteration"]
        log(f"Resumed from iteration {start_iteration}")
        if args.resume_weights:
            weights = PPOBrain.load_weights(args.resume_weights)
            ppo.policy.params[:] = weights["policy"]
            ppo.value.params[:] = weights["value"]
            ppo.log_std[:] = weights["log_std"]
            log(f"  with the networks from {args.resume_weights}")
    np.maximum(ppo.log_std, ppo.min_log_std, out=ppo.log_std)

    # Re-point the optimisers at the (possibly replaced) parameter arrays.
    ppo.policy_opt.params = ppo.policy.params + [ppo.log_std]
    ppo.value_opt.params = ppo.value.params

    opponent_names = list(OPPONENT_WEIGHTS)
    opponent_p = np.array([OPPONENT_WEIGHTS[n] for n in opponent_names], float)
    opponent_p /= opponent_p.sum()
    best_score = -1.0

    log(
        f"Training PPOBrain: {args.games} games/iteration on {args.workers} workers, "
        f"obs {OBS_DIM}, action repeat {PPOBrain.ACTION_REPEAT}"
    )

    with Pool(args.workers) as pool:
        best_weights = PPOBrain.load_weights(PPOBrain.WEIGHTS_FILE)
        if args.resume and best_weights is not None:
            log("\nBASELINE: the saved best weights")
            best_score = evaluate(pool, best_weights, args.eval_games, seed=0)
            log()

        cloned = not args.resume and args.bc_games > 0
        if cloned:
            log(f"\nBEHAVIOUR CLONING DefendersAndAttackers from {args.bc_games} games")
            demos = pool.map(play_demo_game, range(10**6, 10**6 + args.bc_games))
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
                    (ppo.weights(), s) for s in range(first, first + args.dagger_games)
                ]
                new = pool.map(play_dagger_game, tasks)
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
            best_score = evaluate(pool, ppo.weights(), args.eval_games, seed=0)
            PPOBrain.save_weights(PPOBrain.WEIGHTS_FILE, ppo.weights())
            log()

        league = [ppo.weights()]

        for iteration in range(
            start_iteration + 1, start_iteration + args.iterations + 1
        ):
            started = time.time()
            # Linearly decay the learning rate to 10% over the run.
            progress = (iteration - start_iteration - 1) / args.iterations
            ppo.set_lr(args.lr * (1.0 - 0.9 * progress))
            weights = ppo.weights()

            tasks = []
            for _ in range(args.games):
                roll = rng.random()
                seed = int(rng.integers(2**31))
                if roll < args.self_play:
                    tasks.append((weights, "self", None, seed, args.gamma))
                elif roll < args.self_play + args.snapshots:
                    snap = league[rng.integers(len(league))]
                    tasks.append((weights, "snapshot", snap, seed, args.gamma))
                else:
                    opp = str(rng.choice(opponent_names, p=opponent_p))
                    tasks.append((weights, opp, None, seed, args.gamma))

            outcomes = pool.map(play_training_game, tasks)
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
            np.savez(RUN_DIR / "latest.npz", state=np.array(state, dtype=object))

            if iteration % args.eval_every == 0:
                log(f"\nEVALUATION after iteration {iteration} (deterministic policy):")
                score = evaluate(pool, ppo.weights(), args.eval_games)
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
