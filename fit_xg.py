"""
Fit an xG model: the chance that a ball on target goes in before anyone touches it.

The model gives the "chance" reward term (PPOBrain.chance_potential) its values. This
script plays games among the brains in a pool directory (by default the league field,
only read) with the batched simulator (aisoccer/vecgame.py), and records a shot event
whenever a touch (the ball hit by a player or a post) leaves the ball on target
(PPOBrain.shot_features: moving towards a goal line and crossing it inside the goal
mouth, directly or off one side wall). Each event has the features of the chance
(PPOBrain.XG_FEATURES: distance to the line, where it crosses the mouth, speed towards
the goal, defenders near the path, the nearest defender's distance from the crossing
point), from the attacking team's point of view, and a label: 1 if the ball goes into
that goal before the next touch and within --horizon ticks, else 0. A touch in the tick
the ball goes in does not stop the goal. Events still open when a game ends are dropped.

Then it fits a logistic regression (numpy, Newton's method, a little L2 on the
standardised features) and writes to --out (JSON): the coefficients, the feature
scaling, the data counts and the fit quality (log-loss and Brier score against the base
rate, and calibration by decile, on held-out games: every fifth game is held out, and
the model written is then fitted on all games). The events are saved to a .npz next to
the JSON after every batch of games, so a refit can reuse them (--refit), or more games
can be added (--append).

    nice -n 15 .venv/bin/python fit_xg.py --out runs/league3/xg.json --games 200 --workers 2
    nice -n 15 .venv/bin/python fit_xg.py --out runs/league3/xg.json --refit
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # one BLAS thread per worker

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from aisoccer.brains.PPOBrain import L, XG_FEATURES, shot_features, xg_column  # noqa: E402
from aisoccer.brainspec import load_brain, panel_from_dir, spec_label  # noqa: E402
from aisoccer.constants import Constants  # noqa: E402
from aisoccer.vecgame import BALL, NUM, VecGame  # noqa: E402

# The model's inputs (PPOBrain.xg_column): distance and speed on a log scale fit the
# league's shots better than linear ones (held-out log-loss 0.137 against 0.148, and
# better calibrated at the top), so doubling either changes the odds by one factor.
DEFAULT_FEATURES = ("log_distance", "offset", "log_speed", "blockers", "keeper")
# Units the odds are reported in: per 100 px, per 1 px/tick, per defender, per doubling.
UNITS = {
    "distance": 100.0,
    "offset": 100.0,
    "speed": 1.0,
    "blockers": 1.0,
    "keeper": 100.0,
    "log_distance": np.log(2.0),
    "log_speed": np.log(2.0),
}
UNIT_NAMES = {
    "log_distance": "doubling the distance to the goal line (1 + px)",
    "log_speed": "doubling the speed towards the goal",
    "distance": "100 px further from the goal line",
    "offset": "100 px further from the middle of the goal mouth",
    "speed": "1 px/tick faster towards the goal",
    "blockers": "1 more defender near the path",
    "keeper": "100 px more between the nearest defender and the crossing point",
}


class ShotGame(VecGame):
    """A VecGame that notes, each tick, which games' ball was hit by a player or a post."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.touched = np.zeros(self.n, dtype=bool)

    def collide_swept(self, pos, vel, dt):
        new_vel, shift = VecGame.collide_swept(pos, vel, dt)
        self.touched |= (new_vel[:, BALL] != vel[:, BALL]).any(axis=-1)
        return new_vel, shift

    def collide(self, pos, vel):
        new_vel = VecGame.collide(pos, vel)
        self.touched |= (new_vel[:, BALL] != vel[:, BALL]).any(axis=-1)
        return new_vel


def attacking_views(pos, vel, games):
    """
    For the given games, the ball (position, speed-capped velocity) and the defenders
    as seen by blue attacking the right goal, and by red attacking the left goal
    (mirrored as Game shows red the field).
    """
    ball = pos[games, BALL].copy()
    ball_vel = vel[games, BALL].copy()
    speed = np.hypot(ball_vel[:, 0], ball_vel[:, 1])
    too_fast = speed > Constants.MAX_BALL_VELOCITY  # capped at the start of the next tick
    ball_vel[too_fast] *= (Constants.MAX_BALL_VELOCITY / speed[too_fast])[:, None]
    blue, red = pos[games, :NUM].copy(), pos[games, NUM:2 * NUM].copy()
    ball_m, vel_m = ball.copy(), ball_vel.copy()
    ball_m[:, 0] = L - 1 - ball_m[:, 0]
    vel_m[:, 0] = -vel_m[:, 0]
    blue[..., 0] = L - 1 - blue[..., 0]
    return (ball, ball_vel, red), (ball_m, vel_m, blue)


def play_batch(job):
    """
    Play one VecGame batch and return its shot events: features (n, F), labels (n,),
    game ids (n,), attacking side (n,: 0 blue, 1 red), plus the games' goals and how many
    of them ended a recorded event.
    """
    pairs, seeds, game_ids, ticks, horizon = job
    blue = [load_brain(b) for b, _ in pairs]
    red = [load_brain(r) for _, r in pairs]
    game = ShotGame(blue, red, seeds, game_length=ticks)
    n = game.n
    open_events = [[] for _ in range(n)]  # per game: [side, features, tick]
    features, labels, ids, sides = [], [], [], []
    goals = scored_events = 0

    def close(g, event, label):
        features.append(event[1])
        labels.append(label)
        ids.append(game_ids[g])
        sides.append(event[0])

    while game.ticks < ticks:
        game.touched[:] = False
        game.tick()
        red_in, blue_in = game.goals()  # in a goal now: scored at the next tick's start
        goals += int(red_in.sum() + blue_in.sum())
        for g in range(n):
            still = []
            for event in open_events[g]:
                if blue_in[g] or red_in[g]:
                    hit = blue_in[g] if event[0] == 0 else red_in[g]
                    close(g, event, 1.0 if hit else 0.0)
                    scored_events += int(hit)
                elif game.touched[g] or game.ticks - event[2] > horizon:
                    close(g, event, 0.0)
                else:
                    still.append(event)
            open_events[g] = still
        new = np.flatnonzero(game.touched & ~(red_in | blue_in))
        if len(new):
            for side, view in enumerate(attacking_views(game.pos, game.vel, new)):
                on_target, f = shot_features(*view)
                for i in np.flatnonzero(on_target):
                    open_events[new[i]].append((side, f[i], game.ticks))
    width = len(XG_FEATURES)
    return {
        "features": np.array(features, dtype=float).reshape(-1, width),
        "labels": np.array(labels, dtype=float),
        "game": np.array(ids, dtype=int),
        "side": np.array(sides, dtype=int),
        "goals": goals,
        "scored_events": scored_events,
        "pairs": [(spec_label(b), spec_label(r)) for b, r in pairs],
        "game_ids": list(game_ids),
    }


def sigmoid(z):
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-z))


def fit_logistic(x, y, l2=1.0, iterations=100):
    """
    Logistic regression of y on the columns of x (already standardised) by Newton's
    method, with an L2 penalty l2 * |coef|^2 / 2 (not on the intercept). Returns
    (intercept, coef).
    """
    a = np.hstack([np.ones((len(x), 1)), x])
    penalty = np.full(a.shape[1], float(l2))
    penalty[0] = 0.0
    base = np.clip(y.mean(), 1e-6, 1 - 1e-6)
    w = np.zeros(a.shape[1])
    w[0] = np.log(base / (1 - base))
    for _ in range(iterations):
        p = sigmoid(a @ w)
        grad = a.T @ (p - y) + penalty * w
        hess = a.T @ (a * (p * (1 - p))[:, None]) + np.diag(penalty) + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return float(w[0]), w[1:]


def log_loss(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def calibration(p, y, bins=10):
    """Mean predicted and observed goal rate in each decile of the predictions."""
    order = np.argsort(p, kind="stable")
    rows = []
    for chunk in np.array_split(order, bins):
        if len(chunk):
            rows.append(
                {
                    "n": int(len(chunk)),
                    "predicted": round(float(p[chunk].mean()), 4),
                    "observed": round(float(y[chunk].mean()), 4),
                }
            )
    return rows


def fit_model(features, y, games, l2, names=DEFAULT_FEATURES):
    """
    Fit on all events (shot_features columns) with the given inputs (xg_column names);
    report the quality of a fit without every fifth game on those games.
    """
    x = np.column_stack([xg_column(features, name) for name in names])
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    z = (x - mean) / scale

    test = games % 5 == 0
    quality = {}
    if test.any() and (~test).any() and 0 < y[~test].mean() < 1:
        intercept, coef = fit_logistic(z[~test], y[~test], l2)
        p = sigmoid(intercept + z[test] @ coef)
        base = float(y[~test].mean())
        quality = {
            "held_out_games": int(len(np.unique(games[test]))),
            "held_out_events": int(test.sum()),
            "held_out_goals": int(y[test].sum()),
            "log_loss": round(log_loss(p, y[test]), 5),
            "base_rate_log_loss": round(log_loss(np.full(test.sum(), base), y[test]), 5),
            "brier": round(float(((p - y[test]) ** 2).mean()), 5),
            "base_rate_brier": round(float(((base - y[test]) ** 2).mean()), 5),
            "calibration_by_decile": calibration(p, y[test]),
        }

    intercept, coef = fit_logistic(z, y, l2)
    p = sigmoid(intercept + z @ coef)
    quality["in_sample_log_loss"] = round(log_loss(p, y), 5)
    quality["in_sample_base_rate_log_loss"] = round(log_loss(np.full(len(y), y.mean()), y), 5)
    odds = {
        name: {
            "per": UNIT_NAMES[name],
            "odds_multiplier": round(float(np.exp(c / s * UNITS[name])), 4),
        }
        for name, c, s in zip(names, coef, scale)
    }
    return {
        "features": list(names),
        "mean": [float(v) for v in mean],
        "scale": [float(v) for v in scale],
        "coef": [float(v) for v in coef],
        "intercept": intercept,
        "l2": l2,
        "odds": odds,
        "fit": quality,
    }


def load_events(path):
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def save_events(path, events):
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **events)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="the model (JSON); events go to <out>.npz")
    parser.add_argument("--pool-dir", type=Path, default=Path("runs/league3/field"))
    parser.add_argument(
        "--extra", nargs="*", default=[], help="more brain specs to play, e.g. brain:aisoccer.llm.v8.LLMBrainV8"
    )
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--ticks", type=int, default=Constants.GAME_LENGTH, help="ticks per game")
    parser.add_argument("--batch", type=int, default=10, help="games per VecGame")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=300, help="ticks an event may wait for its goal")
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument(
        "--features",
        nargs="+",
        default=list(DEFAULT_FEATURES),
        help="model inputs: any of distance, offset, speed, blockers, keeper, log_distance, log_speed",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--append", action="store_true", help="add the new games to the saved events")
    parser.add_argument("--refit", action="store_true", help="play no games: refit the saved events")
    args = parser.parse_args()

    events_path = args.out.with_suffix(".npz")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    events = load_events(events_path) if (args.append or args.refit) else None
    if args.refit and events is None:
        raise SystemExit(f"no saved events in {events_path}")
    started = time.time()
    if events is None:
        events = {
            "features": np.zeros((0, len(XG_FEATURES))),
            "labels": np.zeros(0),
            "game": np.zeros(0, dtype=int),
            "side": np.zeros(0, dtype=int),
            "pairs": np.zeros((0, 2), dtype="<U64"),
            "goals": np.zeros(0, dtype=int),
            "scored_events": np.zeros(0, dtype=int),
            "ticks": np.zeros(0, dtype=int),
        }

    if not args.refit:
        specs = panel_from_dir(args.pool_dir, 10**6, np.random.default_rng(0)) + list(args.extra)
        if len(specs) < 2:
            raise SystemExit(f"need at least two brains, found {specs}")
        print(f"{len(specs)} brains: {', '.join(spec_label(s) for s in specs)}", flush=True)
        rng = np.random.default_rng(args.seed)
        first = int(len(events["pairs"]))  # game ids carry on from the saved events
        jobs = []
        for start in range(0, args.games, args.batch):
            ids = list(range(first + start, first + min(start + args.batch, args.games)))
            pairs = [tuple(rng.choice(specs, size=2, replace=False)) for _ in ids]
            seeds = [int(rng.integers(2**31)) for _ in ids]
            jobs.append((pairs, seeds, ids, args.ticks, args.horizon))
        with Pool(args.workers) as pool:
            for done, result in enumerate(pool.imap_unordered(play_batch, jobs), 1):
                for key in ("features", "labels", "game", "side"):
                    events[key] = np.concatenate([events[key], result[key]])
                pairs = np.zeros((len(events["pairs"]) + len(result["pairs"]), 2), dtype="<U64")
                pairs[: len(events["pairs"])] = events["pairs"]
                pairs[len(events["pairs"]):] = result["pairs"]
                events["pairs"] = pairs
                events["goals"] = np.append(events["goals"], result["goals"])
                events["scored_events"] = np.append(events["scored_events"], result["scored_events"])
                events["ticks"] = np.append(events["ticks"], [args.ticks] * len(result["pairs"]))
                save_events(events_path, events)
                print(
                    f"batch {done}/{len(jobs)}: {len(events['labels'])} events, "
                    f"{int(events['labels'].sum())} goals, {time.time() - started:.0f}s",
                    flush=True,
                )

    x, y = events["features"], events["labels"]
    model = fit_model(x, y, events["game"], args.l2, args.features)
    model["data"] = {
        "events": int(len(y)),
        "event_goals": int(y.sum()),
        "games": int(len(events["pairs"])),
        "game_ticks": sorted({int(t) for t in events["ticks"]}),
        "goals_in_games": int(events["goals"].sum()),
        "goals_after_an_event": int(events["scored_events"].sum()),
        "events_by_side": [int((events["side"] == s).sum()) for s in (0, 1)],
        "brains": sorted({str(b) for b in events["pairs"].ravel()}),
        "horizon_ticks": args.horizon,
        "events_file": events_path.name,
    }
    model["seconds"] = round(time.time() - started, 1)
    model["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = args.out.with_name(args.out.stem + ".tmp.json")
    tmp.write_text(json.dumps(model, indent=1))
    os.replace(tmp, args.out)
    print(json.dumps({k: model[k] for k in ("odds", "fit", "data")}, indent=1))


if __name__ == "__main__":
    main()
