"""
Train the PPO-tactician: PPO chooses TacticsV1's team strategy per situation.

TacticsV1 (aisoccer/tactics/v1.py) has expert skills (intercepts, aimed and bank shots,
passes, a keeper, a formation) and plays them by a 15-number strategy (how high the
line sits, how many press, how far out to shoot, ...), always the same one. The
tactician (aisoccer/tactics/learned.py) is TacticsV1 with a small policy network that
picks the strategy every 30 ticks from the game situation. The skills are not learned;
only when to play which strategy.

Start: the policy outputs DEFAULT_STRATEGY in every situation (plain TacticsV1), and
PPO learns to change it where that wins. It does not start from random.

Opponents: self-play (both sides' experience is used), snapshots of itself, and the
brains in --panel-dir (e.g. the league's field), drawn at random. Reward: goals
(+/-2) plus the change in PPOBrain's shaping potential (ball progress, control, chase),
per decision.

Output folder (--out):

- progress.log: one line per iteration: speed, PPO statistics, the strategy the
  policy plays (mean over the situations of the iteration, and how much it varies
  between situations), and results per opponent,
- latest.npz: the full training state; a restart with the same --out carries on,
- policy.npz: the current networks (TacticianBrain.load_weights),
- entrant.json: the current policy as a self-contained brain spec, written atomically
  every iteration: the league's entrant file (a copy of it is a frozen version),
- snapshots/: earlier versions, played as opponents.

Usage:

    python train_tactician.py --workers 3 --panel-dir runs/league3/field --out runs/league3/tactician
"""

import os

# Tiny networks: one BLAS thread each, in the workers and for the update.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from aisoccer.brainspec import load_brain  # noqa: E402
from aisoccer.constants import Constants  # noqa: E402
from aisoccer.game import Game  # noqa: E402
from aisoccer.ppo import PPO, gae  # noqa: E402
from aisoccer.tactics.learned import (  # noqa: E402
    ACT_DIM,
    DECISION_INTERVAL,
    DEFAULT_LOGITS,
    OBS_DIM,
    TacticianBrain,
    sigmoid,
)
from aisoccer.tactics.v1 import STRATEGY  # noqa: E402

CONFIG_KEYS = {
    "lr", "gamma", "lam", "min_std", "target_kl", "epochs", "minibatch",
    "self_play", "snapshots", "reward", "game_length",
}
SHORT = {  # strategy names for the log
    "line": "line", "push": "push", "width": "wid", "press": "press", "shoot_range": "shoot",
    "pass_bias": "pass", "power": "pow", "keeper_depth": "keep", "mark": "mark",
    "opportunistic": "opp", "w_goal": "wG", "w_bank": "wB", "w_pass": "wP", "w_space": "wS",
    "max_angle": "ang",
}

LOG_FILE = None


def log(message=""):
    print(message, flush=True)
    if LOG_FILE is not None:
        with open(LOG_FILE, "a") as f:
            f.write(message + "\n")


# ---------------------------------------------------------------- games (in workers)


def label(opponent):
    return Path(opponent).stem if opponent not in ("self", "snapshot") else opponent


def play_game(task):
    """One training game. Returns (opponent label, goals for, against, trajectories)."""
    weights, opponent, snapshot, seed, gamma, reward, game_length = task
    rng = np.random.default_rng(seed)
    learner = TacticianBrain("learner", weights=weights, training=True, reward=reward)
    other = None
    if opponent not in ("self", "snapshot"):
        try:
            other = load_brain(opponent)
        except Exception:  # e.g. the league replaced the field file between rounds
            opponent = "self"
    if opponent == "self":
        other = TacticianBrain("self", weights=weights, training=True, reward=reward)
    elif opponent == "snapshot":
        other = TacticianBrain("snapshot", weights=snapshot, deterministic=False)
    learner_blue = rng.random() < 0.5
    blue, red = (learner, other) if learner_blue else (other, learner)
    score = Game(blue, red, game_length=game_length, quiet_mode=True, seed=seed).play()
    mine, theirs = (score["blue"], score["red"]) if learner_blue else (score["red"], score["blue"])
    trajectories = [learner.finish_trajectory(gamma)]
    if opponent == "self":
        trajectories.append(other.finish_trajectory(gamma))
    return label(opponent), mine, theirs, trajectories


# ---------------------------------------------------------------- learner


def make_ppo(args):
    ppo = PPO(
        OBS_DIM,
        ACT_DIM,
        hidden=tuple(args.hidden),
        lr=args.lr,
        epochs=args.epochs,
        minibatch=args.minibatch,
        init_log_std=args.init_log_std,
        min_log_std=float(np.log(args.min_std)),
        target_kl=args.target_kl,
        seed=args.seed,
        dtype=np.float64,  # tiny networks: precision costs nothing here
    )
    # Start as plain TacticsV1: output bias = logit(DEFAULT_STRATEGY), output weights
    # tiny (PPO's out_scale 0.01), so the mean strategy is DEFAULT_STRATEGY everywhere.
    ppo.policy.params[-1][:] = DEFAULT_LOGITS
    return ppo


def brain_weights(ppo, interval):
    return {**ppo.weights(), "interval": interval}


def snapshot_weights(ppo, interval):
    w = brain_weights(ppo, interval)
    return {k: w[k] for k in ("policy", "log_std", "interval")}


def build_batch(trajectories, gamma, lam):
    """One PPO batch from team trajectories, with GAE per game."""
    obs, act, logp, adv, ret = [], [], [], [], []
    for t in trajectories:
        n = len(t["rew"])
        if n == 0:
            continue
        dones = np.zeros(n)
        dones[-1] = 1.0
        a, r = gae(t["rew"], t["val"][:, 0], dones, 0.0, gamma, lam)
        obs.append(t["obs"][:, 0])
        act.append(t["act"][:, 0])
        logp.append(t["logp"][:, 0])
        adv.append(a)
        ret.append(r)
    return tuple(np.concatenate(x) for x in (obs, act, logp, adv, ret))


def optimiser_state(opt):
    return {"m": [m.copy() for m in opt.m], "v": [v.copy() for v in opt.v], "t": opt.t}


def load_optimiser(opt, state):
    for m, saved in zip(opt.m, state["m"]):
        m[...] = saved
    for v, saved in zip(opt.v, state["v"]):
        v[...] = saved
    opt.t = int(state["t"])


def training_state(ppo, rng, iteration):
    return {
        "iteration": iteration,
        "rng": rng.bit_generator.state,
        "policy": [p.copy() for p in ppo.policy.params],
        "value": [p.copy() for p in ppo.value.params],
        "log_std": ppo.log_std.copy(),
        "policy_opt": optimiser_state(ppo.policy_opt),
        "value_opt": optimiser_state(ppo.value_opt),
    }


def restore(ppo, rng, state):
    """Load a training state in place (the optimisers keep their arrays). Returns its iteration."""
    for p, s in zip(ppo.policy.params, state["policy"]):
        p[...] = s
    for p, s in zip(ppo.value.params, state["value"]):
        p[...] = s
    ppo.log_std[...] = state["log_std"]
    load_optimiser(ppo.policy_opt, state["policy_opt"])
    load_optimiser(ppo.value_opt, state["value_opt"])
    rng.bit_generator.state = state["rng"]
    return int(state["iteration"])


def save_atomic(path, save):
    """Write via a temporary file so readers (e.g. league.py) never see a partial file."""
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    save(tmp)
    os.replace(tmp, path)


def save_outputs(out, ppo, rng, iteration, interval):
    save_atomic(
        out / "latest.npz",
        lambda p: np.savez(p, state=np.array(training_state(ppo, rng, iteration), dtype=object)),
    )
    save_atomic(out / "policy.npz", lambda p: TacticianBrain.save_weights(p, brain_weights(ppo, interval)))
    TacticianBrain.write_entrant(out / "entrant.json", brain_weights(ppo, interval), iteration=iteration)


def save_snapshot(out, ppo, iteration, interval):
    path = out / "snapshots" / f"it{iteration:06d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_atomic(path, lambda p: TacticianBrain.save_weights(p, snapshot_weights(ppo, interval)))


def load_snapshots(out):
    return [
        TacticianBrain.load_weights(p)
        for p in sorted((out / "snapshots").glob("it*.npz"))
        if ".tmp" not in p.name
    ]


def panel(directory, exclude=()):
    """Brain files in the panel directory (not half-written temporary files), except the
    excluded brain names (file names without the extension, e.g. its own league entry)."""
    if directory is None or not Path(directory).is_dir():
        return []
    return [
        p for p in sorted(Path(directory).glob("*"))
        if p.suffix in (".npz", ".json") and ".tmp" not in p.name and p.stem not in exclude
    ]


def make_tasks(rng, args, weights, snapshots, pool):
    """Self-play, a snapshot of itself, or a random brain from the panel."""
    tasks = []
    for _ in range(args.games):
        roll = rng.random()
        seed = int(rng.integers(2**31))
        if roll < args.self_play or (not pool and roll >= args.self_play + args.snapshots):
            opponent, snap = "self", None
        elif roll < args.self_play + args.snapshots and snapshots:
            opponent, snap = "snapshot", snapshots[rng.integers(len(snapshots))]
        elif pool:
            opponent, snap = str(pool[rng.integers(len(pool))]), None
        else:
            opponent, snap = "self", None
        tasks.append((weights, opponent, snap, seed, args.gamma, args.reward, args.game_length))
    return tasks


def results_line(results):
    """W-D-L and goals for-against per opponent."""
    by_opp: dict[str, list] = {}
    for opp, gf, ga in results:
        by_opp.setdefault(opp, []).append((gf, ga))
    parts = []
    for opp, games in sorted(by_opp.items()):
        w = sum(gf > ga for gf, ga in games)
        d = sum(gf == ga for gf, ga in games)
        gf = sum(g[0] for g in games)
        ga = sum(g[1] for g in games)
        parts.append(f"{opp} {w}-{d}-{len(games) - w - d} ({gf}-{ga})")
    return ", ".join(parts)


def strategy_line(ppo, obs):
    """The strategy the (deterministic) policy plays: mean over this iteration's
    situations, with its spread between situations in brackets."""
    s = sigmoid(ppo.policy(np.asarray(obs, dtype=ppo.dtype)))
    return " ".join(
        f"{SHORT[k]} {m:.2f}({d:.2f})" for k, m, d in zip(STRATEGY, s.mean(axis=0), s.std(axis=0))
    )


# ---------------------------------------------------------------- main


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--games", type=int, default=36, help="games per iteration")
    parser.add_argument("--iterations", type=int, default=1_000_000, help="iterations to run (this start)")
    parser.add_argument(
        "--panel-dir", type=Path,
        help="directory of brain files to train against (e.g. the league's field)",
    )
    parser.add_argument(
        "--exclude", nargs="*", default=[],
        help="brain names in --panel-dir to leave out (e.g. its own league entry, PPO-tactician)",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/tactician"))
    parser.add_argument("--config", type=Path, help=f"JSON overriding settings: {sorted(CONFIG_KEYS)}")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    parser.add_argument(
        "--interval", type=int, default=DECISION_INTERVAL,
        help="ticks between strategy decisions (a multiple of TacticsV1's 10); fixed at the first start",
    )
    parser.add_argument("--game-length", type=int, default=Constants.GAME_LENGTH, help="ticks per training game")
    parser.add_argument("--lr", type=float, default=3e-4)
    # Per decision (30 ticks): 0.96 per decision is about 0.9986 per tick, a horizon of
    # roughly 700 ticks, about what train_ppo.py's 0.995 per 4 ticks gives.
    parser.add_argument("--gamma", type=float, default=0.96)
    parser.add_argument("--lam", type=float, default=0.9)
    parser.add_argument("--init-log-std", type=float, default=-0.7, help="exploration noise (logit space)")
    parser.add_argument("--min-std", type=float, default=0.1)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--self-play", type=float, default=0.15)
    parser.add_argument("--snapshots", type=float, default=0.1)
    parser.add_argument("--snapshot-every", type=int, default=10)
    args = parser.parse_args(argv)
    args.reward = None
    if args.config:
        config = json.loads(args.config.read_text())
        unknown = set(config) - CONFIG_KEYS
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        for key, value in config.items():
            setattr(args, key, value)
    if args.interval % 10:
        parser.error("--interval must be a multiple of 10 (TacticsV1 re-assigns players every 10 ticks)")
    return args


def main(argv=None):
    global LOG_FILE
    args = parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    LOG_FILE = out / "progress.log"

    rng = np.random.default_rng(args.seed)
    ppo = make_ppo(args)
    interval = args.interval
    start_iteration = 0
    if (out / "latest.npz").exists():
        state = np.load(out / "latest.npz", allow_pickle=True)["state"].item()
        start_iteration = restore(ppo, rng, state)
        interval = TacticianBrain.load_weights(out / "policy.npz").get("interval", interval)
        log(f"Resumed from iteration {start_iteration}")
    else:
        log(
            f"Training PPO-tactician: TacticsV1 + a {OBS_DIM}-{'-'.join(map(str, args.hidden))}-{ACT_DIM} "
            f"strategy network deciding every {interval} ticks, starting at DEFAULT_STRATEGY "
            f"(plain TacticsV1); {args.games} games/iteration on {args.workers} workers"
        )
        save_snapshot(out, ppo, 0, interval)
    ppo.min_log_std = float(np.log(args.min_std))
    ppo.target_kl = args.target_kl
    ppo.epochs = args.epochs
    ppo.minibatch = args.minibatch
    ppo.set_lr(args.lr)
    np.maximum(ppo.log_std, ppo.min_log_std, out=ppo.log_std)
    config = {k: getattr(args, k) for k in sorted(CONFIG_KEYS)}
    log(f"Settings: {config}")
    snapshots = load_snapshots(out)
    save_outputs(out, ppo, rng, start_iteration, interval)

    with Pool(args.workers) as workers:
        for iteration in range(start_iteration + 1, start_iteration + args.iterations + 1):
            started = time.time()
            weights = brain_weights(ppo, interval)
            tasks = make_tasks(rng, args, weights, snapshots, panel(args.panel_dir, set(args.exclude)))
            outcomes = workers.map(play_game, tasks, chunksize=1)
            played = time.time() - started
            trajectories = [t for o in outcomes for t in o[3]]
            batch = build_batch(trajectories, args.gamma, args.lam)
            stats = ppo.update(*batch)

            if iteration % args.snapshot_every == 0:
                save_snapshot(out, ppo, iteration, interval)
                snapshots = load_snapshots(out)
            save_outputs(out, ppo, rng, iteration, interval)

            elapsed = time.time() - started
            log(
                f"it {iteration:5d} | {elapsed:5.1f}s (play {played:5.1f}s, "
                f"{3600 * len(tasks) / max(played, 1e-9):5.0f} games/h) | samples {len(batch[0]):5d} | "
                f"ret {np.mean([t['rew'].sum() for t in trajectories]):+.2f} | "
                f"std {np.exp(ppo.log_std).mean():.2f} | kl {stats['approx_kl']:.4f} "
                f"clip {stats['clip_frac']:.2f} ep {stats['epochs']:.0f} vloss {stats['value_loss']:.4f} | "
                f"{strategy_line(ppo, batch[0])} | "
                + results_line([(o, gf, ga) for o, gf, ga, _ in outcomes if o != "self"])
            )


if __name__ == "__main__":
    main()
