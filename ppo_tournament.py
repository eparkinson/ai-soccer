"""
Round robin of PPOBrain against every other brain, including earlier saved PPOBrain
versions, with enough games per pairing for the result to mean something.

Goals are rare (under one per game between the strongest brains) and arrive roughly
as a Poisson process, so most short fixtures end level and a few games prove nothing.
With 300 games per pairing the 95% confidence interval on a head-to-head goal
difference is about +/-0.1 goals per game. The head-to-head table shows each brain's
record against PPOBrain and marks it significant when that interval excludes zero.

    poetry run python ppo_tournament.py            # 300 games per pairing
    poetry run python ppo_tournament.py --legs 100 # quicker, less certain
"""

import argparse
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
from aisoccer.brainspec import load_brain
from aisoccer.tournament import Tournament


def head_to_head(tournament, a, b):
    """Goal differences from brain a's point of view in every game between a and b."""
    diffs = []
    for blue, red, blue_goals, red_goals in tournament.results:
        if (blue, red) == (a, b):
            diffs.append(blue_goals - red_goals)
        elif (blue, red) == (b, a):
            diffs.append(red_goals - blue_goals)
    return np.array(diffs, dtype=float)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--legs", type=int, default=300, help="games per pairing")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--champion",
        default=str(PPOBrain.WEIGHTS_FILE),
        help="the brain under test, as a brain spec (see aisoccer/brainspec.py)",
    )
    args = parser.parse_args()

    history = PPOBrain.WEIGHTS_FILE.parent / "history"
    champion = Path(args.champion)
    current = load_brain(
        args.champion, name="PPOBrain" if champion == PPOBrain.WEIGHTS_FILE else None
    )
    saved = sorted(list(history.glob("*.npz")) + list(history.glob("*champ-*.json")))
    past_versions = [
        load_brain(path) for path in saved if path.resolve() != champion.resolve()
    ]

    def same_policy(brain):
        """A saved PPOBrain identical to the brain under test (it cannot beat itself)."""
        return (
            isinstance(brain, PPOBrain)
            and isinstance(current, PPOBrain)
            and brain.role_policies is None
            and current.role_policies is None
            and all(
                np.shape(a) == np.shape(b) and np.array_equal(a, b)
                for a, b in zip(brain.policy.params, current.policy.params)
            )
        )

    past_versions = [brain for brain in past_versions if not same_policy(brain)]
    brains = [
        current,
        *past_versions,
        DefendersAndAttackers(),
        BehindAndTowards(),
        StrategicPlanner(),
        AdaptiveChaser(),
        SimpleBrain(),
        LearningBrain(),
        RandomWalk(),
    ]
    tournament = Tournament(brains, legs=args.legs, seed=args.seed)
    tournament.start()

    print()
    print("POINTS PER GAME (95% CI):")
    for row in tournament.get_table():
        games = row["played"]
        wins, draws = row["wins"], row["draws"]
        points = np.array([3.0] * wins + [1.0] * draws + [0.0] * (games - wins - draws))
        ci = 1.96 * points.std(ddof=1) / np.sqrt(games)
        print(f"   {row['name']:<25} {points.mean():.2f} ±{ci:.2f}")

    print()
    ppo = brains[0].name
    print(f"HEAD TO HEAD: each brain's record against {ppo} (95% CI)")
    print(f"   positive goal difference = stronger than {ppo}")
    for other in range(1, len(brains)):
        gd = head_to_head(tournament, other, 0)
        ci = 1.96 * gd.std(ddof=1) / np.sqrt(len(gd))
        wins, draws = int((gd > 0).sum()), int((gd == 0).sum())
        losses = len(gd) - wins - draws
        if gd.mean() - ci > 0:
            verdict = f"stronger than {ppo} (significant)"
        elif gd.mean() + ci < 0:
            verdict = f"weaker than {ppo} (significant)"
        else:
            verdict = "no significant difference"
        print(
            f"   {brains[other].name:<25} {wins:>4}-{draws:>3}-{losses:<4}"
            f" GD/game {gd.mean():+.2f} ±{ci:.2f}  {verdict}"
        )


if __name__ == "__main__":
    main()
