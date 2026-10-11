"""
Train PPO-roles: five policy networks and five value networks, one per player slot.

PPOBrain shares one network between all five players (each player's observation
carries its role, so the network can still play them differently). Here every player
slot (role 0..4: the player's index in the team, i.e. its kick-off position) has its
own policy and its own critic, trained jointly with PPO from random initialisation (no
behaviour cloning, no starting weights). The question: does giving each player its own
brain make positions (a goalkeeper, defenders) emerge?

Everything else is train_ppo.py's: the same PPO algorithm and defaults
(aisoccer/ppo.py), the same shaped reward, game length and opponent mix: self-play
(both teams play the current role networks, and both sides' experience is used),
snapshots of itself, and brains from --panel-dir (e.g. the league's field). Each role's
networks learn from that player's experience only: role r's batch is column r of every
trajectory, with GAE per player as in train_ppo.build_batch.

Output folder (--out):

- progress.log: one line per iteration, including each role's mean position,
- latest.npz: the full training state (networks, optimisers, random generator);
  a restart with the same --out carries on from it,
- snapshots/: earlier versions played as opponents,
- best.npz: the current role networks as a PPOBrain weights file with
  "role_policies" (load it with aisoccer.brainspec.load_brain), written atomically
  every iteration: the league's entrant file.

Usage:

    python train_roles.py --workers 2 --panel-dir runs/league2/field --out runs/league2/ppo_roles
"""

import os

# The five PPO updates run one after the other in the main process. Their matrices are
# small: one BLAS thread takes about 10% longer than two for half the CPU, which is the
# better trade on a machine shared with a league. Must be set before numpy.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from aisoccer.brains.PPOBrain import ACT_DIM, OBS_DIM, PPOBrain  # noqa: E402
from aisoccer.constants import Constants  # noqa: E402
from aisoccer.ppo import PPO  # noqa: E402
from train_ppo import (  # noqa: E402
    build_batch,
    play_iteration_games,
    results_line,
    save_atomic,
    set_action_repeat,
)

NUM = Constants.NUM_PLAYERS

# train_ppo.py's config keys that apply here (see --config).
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

LOG_FILE = None


def log(message=""):
    print(message, flush=True)
    if LOG_FILE is not None:
        with open(LOG_FILE, "a") as f:
            f.write(message + "\n")


def team_weights(roles):
    """
    The five learners as one PPOBrain weights dict for training games: role r's player
    acts with roles[r]'s policy and noise and estimates values with its critic.
    "policy" and "value" (role 0's) are only there because PPOBrain expects them.
    """
    # PPO.weights() is float64 whatever precision the learners train in, so saved
    # files and training games are as they always were.
    weights = [ppo.weights() for ppo in roles]
    return {
        "policy": weights[0]["policy"],
        "value": weights[0]["value"],
        "log_std": np.stack([w["log_std"] for w in weights]),  # (5, 2)
        "role_policies": [w["policy"] for w in weights],
        "role_values": [w["value"] for w in weights],
        "action_repeat": PPOBrain.ACTION_REPEAT,
    }


def snapshot_weights(roles):
    """What a snapshot opponent needs: the role policies and their noise."""
    weights = team_weights(roles)
    return {k: weights[k] for k in ("policy", "log_std", "role_policies", "action_repeat")}


def entrant_weights(roles):
    """
    The entrant file's contents: the role policies, plus one team-wide log_std row (the
    mean over roles) so that any reader of PPOBrain files, which expect a (2,) log_std,
    can use it. Deterministic play (how the league plays it) ignores the noise.
    """
    weights = snapshot_weights(roles)
    weights["log_std"] = weights["log_std"].mean(axis=0)
    return weights


def role_trajectories(trajectories, role):
    """Each trajectory cut down to one player's column, for train_ppo.build_batch."""
    return [
        {
            "obs": t["obs"][:, [role]],
            "act": t["act"][:, [role]],
            "logp": t["logp"][:, [role]],
            "val": t["val"][:, [role]],
            "rew": t["rew"],
        }
        for t in trajectories
        if len(t["rew"])
    ]


def build_role_batches(trajectories, gamma, lam):
    """One PPO batch per role: that player's experience from every trajectory."""
    return [
        build_batch(role_trajectories(trajectories, r), gamma, lam) for r in range(NUM)
    ]


def make_tasks(rng, n_games, weights, league, pool, args):
    """
    An iteration's training games, drawn as train_ppo.py draws them: self-play,
    a snapshot, a brain from the panel directory, or a fixed opponent.
    """
    opponent_names = list(args.opponent_weights)
    opponent_p = np.array([args.opponent_weights[n] for n in opponent_names], float)
    opponent_p /= max(opponent_p.sum(), 1e-9)
    tasks = []
    for _ in range(n_games):
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
            if pool:
                opp = "file:" + str(pool[rng.integers(len(pool))])
                tasks.append((weights, opp, None, *task_end))
            else:
                tasks.append((weights, "self", None, *task_end))
        else:
            opp = str(rng.choice(opponent_names, p=opponent_p))
            tasks.append((weights, opp, None, *task_end))
    return tasks


def panel(directory):
    """Brain files in the panel directory (not half-written temporary files)."""
    if directory is None or not Path(directory).is_dir():
        return []
    return [
        p
        for p in sorted(Path(directory).glob("*"))
        if p.suffix in (".npz", ".json") and ".tmp" not in p.name
    ]


def make_roles(args):
    return [
        PPO(
            OBS_DIM,
            ACT_DIM,
            hidden=tuple(args.hidden),
            lr=args.lr,
            init_log_std=args.init_log_std,
            min_log_std=float(np.log(args.min_std)),
            target_kl=args.target_kl,
            seed=[args.seed, r],
            dtype=np.dtype(args.update_precision),
        )
        for r in range(NUM)
    ]


def optimiser_state(opt):
    return {"m": [m.copy() for m in opt.m], "v": [v.copy() for v in opt.v], "t": opt.t}


def load_optimiser(opt, state):
    for m, saved in zip(opt.m, state["m"]):
        m[...] = saved
    for v, saved in zip(opt.v, state["v"]):
        v[...] = saved
    opt.t = int(state["t"])


def training_state(roles, rng, iteration):
    return {
        "iteration": iteration,
        "rng": rng.bit_generator.state,
        "roles": [
            {
                "policy": [p.copy() for p in ppo.policy.params],
                "value": [p.copy() for p in ppo.value.params],
                "log_std": ppo.log_std.copy(),
                "policy_opt": optimiser_state(ppo.policy_opt),
                "value_opt": optimiser_state(ppo.value_opt),
            }
            for ppo in roles
        ],
    }


def restore(roles, rng, state):
    """Load a training state into the learners in place (the optimisers keep pointing
    at the same arrays). Returns the iteration it was saved after."""
    for ppo, saved in zip(roles, state["roles"]):
        for p, s in zip(ppo.policy.params, saved["policy"]):
            p[...] = s
        for p, s in zip(ppo.value.params, saved["value"]):
            p[...] = s
        ppo.log_std[...] = saved["log_std"]
        load_optimiser(ppo.policy_opt, saved["policy_opt"])
        load_optimiser(ppo.value_opt, saved["value_opt"])
    rng.bit_generator.state = state["rng"]
    return int(state["iteration"])


def save_snapshot(out, roles, iteration):
    path = out / "snapshots" / f"it{iteration:06d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_atomic(path, lambda p: PPOBrain.save_weights(p, snapshot_weights(roles)))


def load_snapshots(out):
    return [
        PPOBrain.load_weights(p)
        for p in sorted((out / "snapshots").glob("it*.npz"))
        if ".tmp" not in p.name
    ]


def role_positions(trajectories):
    """
    Where each role plays: its mean x (-1 at its own goal, +1 at theirs) and its mean
    rank in the race to the ball (0: always the nearest teammate, 1: always the
    furthest), over every recorded decision.
    """
    obs = np.concatenate([t["obs"] for t in trajectories if len(t["rew"])])
    return obs[:, :, 0].mean(axis=0), obs[:, :, 51].mean(axis=0)


def label(opponent):
    """An opponent's name for the log: a panel file's name, or "snapshot"."""
    return Path(opponent.removeprefix("file:")).stem if opponent.startswith("file:") else opponent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--games", type=int, default=36, help="games per iteration")
    parser.add_argument(
        "--iterations", type=int, default=1_000_000, help="iterations to run (this start)"
    )
    parser.add_argument(
        "--panel-dir",
        type=Path,
        help="directory of brain files to train against (e.g. the league's field); "
        "pool_share of games are played against a random one",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/ppo_roles"))
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON overriding settings (train_ppo.py's keys): lr, gamma, lam, min_std, "
        "target_kl, self_play, snapshots, pool_share, reward, opponent_weights, "
        "action_repeat, game_length",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--hidden", type=int, nargs="+", default=[128, 128], help="hidden layer sizes"
    )
    # train_ppo.py's defaults, overridable by --config.
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--init-log-std", type=float, default=-1.0)
    parser.add_argument("--min-std", type=float, default=0.25)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--self-play", type=float, default=0.3)
    parser.add_argument("--snapshots", type=float, default=0.2)
    parser.add_argument("--pool-share", type=float, default=0.5)
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument("--action-repeat", type=int, default=PPOBrain.ACTION_REPEAT)
    parser.add_argument(
        "--game-length", type=int, default=Constants.GAME_LENGTH, help="ticks per training game"
    )
    parser.add_argument(
        "--vector-games",
        type=int,
        help="play games against PPOBrains in VecGame batches of up to this many games "
        "(bit-identical to one at a time; default: games/workers; 0: one at a time)",
    )
    parser.add_argument(
        "--vector-precision", choices=["exact", "float32"], default="exact"
    )
    parser.add_argument(
        "--update-precision",
        choices=["float32", "float64"],
        default="float32",
        help="precision of the PPO updates (saved weights are float64 either way)",
    )
    args = parser.parse_args(argv)
    args.reward = None
    args.opponent_weights = {}  # no fixed heuristic opponents: self, snapshots, panel
    if args.config:
        config = json.loads(args.config.read_text())
        unknown = set(config) - CONFIG_KEYS
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        for key, value in config.items():
            setattr(args, key, value)
    if args.vector_games is None:
        args.vector_games = -(-args.games // max(args.workers, 1))
    return args


def settings(args):
    return {k: getattr(args, k) for k in sorted(CONFIG_KEYS)}


def main(argv=None):
    global LOG_FILE
    args = parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    LOG_FILE = out / "progress.log"

    set_action_repeat(args.action_repeat, args.game_length)
    rng = np.random.default_rng(args.seed)
    roles = make_roles(args)
    state_file = out / "latest.npz"
    start_iteration = 0
    if state_file.exists():
        state = np.load(state_file, allow_pickle=True)["state"].item()
        start_iteration = restore(roles, rng, state)
        log(f"Resumed from iteration {start_iteration}")
    else:
        log(
            f"Training PPO-roles from random: {NUM} policy + {NUM} value networks "
            f"{'x'.join(map(str, args.hidden))}, {args.games} games/iteration on "
            f"{args.workers} workers, obs {OBS_DIM}, action repeat {PPOBrain.ACTION_REPEAT}"
        )
        save_snapshot(out, roles, 0)
    for ppo in roles:  # settings may have changed since the state was saved
        ppo.min_log_std = float(np.log(args.min_std))
        ppo.target_kl = args.target_kl
        ppo.set_lr(args.lr)
        np.maximum(ppo.log_std, ppo.min_log_std, out=ppo.log_std)
    log(f"Settings: {settings(args)}")
    league = load_snapshots(out)
    save_atomic(out / "best.npz", lambda p: PPOBrain.save_weights(p, entrant_weights(roles)))

    with Pool(
        args.workers,
        initializer=set_action_repeat,
        initargs=(args.action_repeat, args.game_length),
    ) as workers:
        for iteration in range(start_iteration + 1, start_iteration + args.iterations + 1):
            started = time.time()
            cpu_started = time.process_time()
            weights = team_weights(roles)
            tasks = make_tasks(rng, args.games, weights, league, panel(args.panel_dir), args)
            outcomes = play_iteration_games(
                workers, tasks, args.vector_games, args.vector_precision
            )
            trajectories = [t for o in outcomes for t in o[3]]
            played = time.time() - started

            batches = build_role_batches(trajectories, args.gamma, args.lam)
            stats = [ppo.update(*batch) for ppo, batch in zip(roles, batches)]

            if iteration % args.snapshot_every == 0:
                save_snapshot(out, roles, iteration)
                league = load_snapshots(out)
            save_atomic(
                state_file,
                lambda p: np.savez(
                    p, state=np.array(training_state(roles, rng, iteration), dtype=object)
                ),
            )
            save_atomic(
                out / "best.npz", lambda p: PPOBrain.save_weights(p, entrant_weights(roles))
            )

            x, rank = role_positions(trajectories)
            mean_reward = np.mean([t["rew"].sum() for t in trajectories])
            std = np.exp(np.stack([ppo.log_std for ppo in roles])).mean(axis=1)
            log(
                f"it {iteration:5d} | {time.time() - started:5.1f}s (play {played:5.1f}s, "
                f"main cpu {time.process_time() - cpu_started:5.1f}s) | "
                f"samples {sum(len(b[0]) for b in batches):6d} | ret {mean_reward:+.2f} | "
                f"std {' '.join(f'{s:.2f}' for s in std)} | "
                f"kl {np.mean([s['approx_kl'] for s in stats]):.4f} "
                f"clip {np.mean([s['clip_frac'] for s in stats]):.2f} "
                f"vloss {np.mean([s['value_loss'] for s in stats]):.4f} | "
                f"x {' '.join(f'{v:+.2f}' for v in x)} | "
                f"ball-rank {' '.join(f'{v:.2f}' for v in rank)} | "
                + results_line(
                    [(label(o), gf, ga) for o, gf, ga, _ in outcomes if o != "self"]
                )
            )


if __name__ == "__main__":
    main()
