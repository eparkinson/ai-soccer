"""
Which team statistics go with winning?

Plays a round robin of the heuristic brains and saved PPOBrain versions while
recording per-game team statistics (see aisoccer/stats.py), then relates each game's
goal difference to the difference in each statistic between the two teams.

Two views, each with 95% confidence intervals:

- Across all games: mixes "strong brains do X" with "doing X wins games".
- Within each matchup (both brains fixed, only the game varies): closer to what
  actually helps a given brain win, which is what a reward term should target.

A multivariate fit on half the games is checked on the other half, to show how much
of the goal difference the statistics explain together.

    poetry run python analyse_stats.py --games 30
"""

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from aisoccer.brains.AdaptiveChaser import AdaptiveChaser
from aisoccer.brains.BehindAndTowards import BehindAndTowards
from aisoccer.brains.DefendersAndAttackers import DefendersAndAttackers
from aisoccer.brains.LearningBrain import LearningBrain
from aisoccer.brains.PPOBrain import PPOBrain
from aisoccer.brains.RandomWalk import RandomWalk
from aisoccer.brains.SimpleBrain import SimpleBrain
from aisoccer.brains.StrategicPlanner import StrategicPlanner
from aisoccer.game import Game
from aisoccer.stats import STAT_NAMES, play_with_stats

HEURISTICS = {
    "DefendersAndAttackers": DefendersAndAttackers,
    "BehindAndTowards": BehindAndTowards,
    "StrategicPlanner": StrategicPlanner,
    "AdaptiveChaser": AdaptiveChaser,
    "SimpleBrain": SimpleBrain,
    "LearningBrain": LearningBrain,
    "RandomWalk": RandomWalk,
}
HISTORY = PPOBrain.WEIGHTS_FILE.parent / "history"


def brain_names():
    return (
        list(HEURISTICS)
        + [p.stem for p in sorted(HISTORY.glob("*.npz"))]
        + ["PPOBrain"]
    )


def make(name):
    if name in HEURISTICS:
        return HEURISTICS[name]()
    if name == "PPOBrain":
        return PPOBrain()
    return PPOBrain(name, weights=PPOBrain.load_weights(HISTORY / f"{name}.npz"))


def play(task):
    blue, red, seed = task
    score, stats = play_with_stats(
        Game(make(blue), make(red), quiet_mode=True, seed=seed)
    )
    return blue, red, score, stats


def correlation_ci(x, y):
    """Pearson r with a 95% confidence interval (Fisher z)."""
    r = float(np.corrcoef(x, y)[0, 1])
    z, se = np.arctanh(np.clip(r, -0.999999, 0.999999)), 1 / np.sqrt(len(x) - 3)
    return r, float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def demean_by(values, groups):
    out = values.astype(float).copy()
    for g in np.unique(groups):
        out[groups == g] -= out[groups == g].mean(axis=0)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--games", type=int, default=30, help="games per pairing")
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("runs/stats/games.csv"))
    parser.add_argument(
        "--from-csv", action="store_true", help="re-analyse --out without playing games"
    )
    args = parser.parse_args()
    if not args.from_csv:
        play_games(args)
    analyse(args.out)


def play_games(args):
    names = brain_names()
    seeds = np.random.SeedSequence(args.seed)
    tasks = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for g in range(args.games):
                blue, red = (names[i], names[j]) if g % 2 == 0 else (names[j], names[i])
                tasks.append((blue, red, int(seeds.spawn(1)[0].generate_state(1)[0])))
    print(f"Playing {len(tasks)} games between {len(names)} brains...", flush=True)
    with Pool(args.processes) as pool:
        results = pool.map(play, tasks, chunksize=4)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["blue", "red", "blue_goals", "red_goals"]
            + [f"{t}_{n}" for t in ("blue", "red") for n in STAT_NAMES]
        )
        for blue, red, score, stats in results:
            writer.writerow(
                [blue, red, score["blue"], score["red"]]
                + [stats[t][n] for t in ("blue", "red") for n in STAT_NAMES]
            )
    print(f"Raw per-game data: {args.out}\n")


def analyse(csv_path):
    # One row per game, from the point of view of the pairing's alphabetically first
    # brain (brains swap sides between games): goal difference and stat differences.
    pairing, gd, diffs = [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            first = min(row["blue"], row["red"])
            me, them = ("blue", "red") if row["blue"] == first else ("red", "blue")
            pairing.append(f"{first} v {max(row['blue'], row['red'])}")
            gd.append(float(row[f"{me}_goals"]) - float(row[f"{them}_goals"]))
            diffs.append(
                [
                    float(row[f"{me}_{n}"]) - float(row[f"{them}_{n}"])
                    for n in STAT_NAMES
                ]
            )
    pairing, gd, diffs = np.array(pairing), np.array(gd), np.array(diffs)

    gd_within = demean_by(gd, pairing)
    diffs_within = demean_by(diffs, pairing)
    rows = []
    for k, name in enumerate(STAT_NAMES):
        across = correlation_ci(diffs[:, k], gd)
        within = correlation_ci(diffs_within[:, k], gd_within)
        rows.append((name, across, within))
    rows.sort(key=lambda r: -abs(r[2][0]))

    print("Correlation of (team stat - opponent stat) with goal difference, 95% CIs.")
    print("Sorted by the within-matchup correlation.\n")
    print(f"  {'STATISTIC':<16} {'ACROSS ALL GAMES':>24}   {'WITHIN MATCHUPS':>24}")
    for name, (r, lo, hi), (rw, lwo, hwi) in rows:
        print(
            f"  {name:<16} {r:>+7.2f} [{lo:+.2f}, {hi:+.2f}]   {rw:>+7.2f} [{lwo:+.2f}, {hwi:+.2f}]"
        )

    # Multivariate within-matchup fit: train on half the games, test on the other half.
    x = (diffs_within - diffs_within.mean(0)) / (diffs_within.std(0) + 1e-12)
    rng = np.random.default_rng(0)
    test = rng.random(len(x)) < 0.5
    coef, *_ = np.linalg.lstsq(x[~test], gd_within[~test], rcond=None)
    pred = x[test] @ coef
    r2 = (
        1
        - ((gd_within[test] - pred) ** 2).sum()
        / ((gd_within[test] - gd_within[test].mean()) ** 2).sum()
    )
    print(
        f"\nAll statistics together explain {100 * r2:.0f}% of the within-matchup goal "
        f"difference on held-out games."
    )
    print("Standardised coefficients (goals per game per standard deviation):")
    for k in np.argsort(-np.abs(coef)):
        print(f"  {STAT_NAMES[k]:<16} {coef[k]:+.3f}")


if __name__ == "__main__":
    main()
