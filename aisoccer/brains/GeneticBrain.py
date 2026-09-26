import json
from pathlib import Path

import numpy as np

from aisoccer.abstractbrain import AbstractBrain
from aisoccer.constants import Constants

L = Constants.FIELD_LENGTH - 1
H = Constants.FIELD_HEIGHT
OPPONENT_GOAL = np.array([L, H / 2])

# One chromosome = GENES_PER_ROLE genes for each of the five player roles, all in [0, 1].
GENES = [
    "home_x",  # where the player waits, as a fraction of the field (from our goal)
    "home_y",
    "follow",  # how far the home position moves up and down the field with the ball
    "track",  # how closely the player tracks the ball's height when waiting
    "engage",  # distance to the ball at which the player goes for it (up to 900 px)
    "offset",  # how far behind the ball it lines up before pushing (up to 120 px)
    "push",  # how hard it drives the ball towards goal once behind it
    "brake",  # damping on its own velocity, to arrive rather than overshoot
]
GENES_PER_ROLE = len(GENES)
CHROMOSOME_LENGTH = GENES_PER_ROLE * Constants.NUM_PLAYERS

# A starting point resembling DefendersAndAttackers: a deep keeper and defenders that
# engage only near them, and an attacker that always goes for the ball.
DEFENSIVE_SEED = np.array(
    [
        [0.06, 0.25, 0.0, 0.6, 0.12, 0.2, 0.5, 0.3],
        [0.10, 0.75, 0.1, 0.4, 0.25, 0.2, 0.5, 0.3],
        [0.33, 0.35, 0.3, 0.3, 0.35, 0.3, 0.6, 0.3],
        [0.33, 0.65, 0.3, 0.3, 0.45, 0.3, 0.6, 0.3],
        [0.55, 0.50, 0.6, 0.5, 1.00, 0.3, 0.8, 0.2],
    ]
).ravel()


class GeneticBrain(AbstractBrain):
    """
    A heuristic brain whose behaviour is set by an evolvable chromosome (see evolve.py
    and docs/genetic_algorithm_learning.md).

    Each player has a home position that shifts with the ball. When the ball is within
    its engagement range it runs to a point behind the ball, then drives the ball
    towards the opponent's goal; otherwise it returns home.
    """

    BEST_FILE = Path(__file__).parent / "weights" / "GeneticBrain.json"

    def __init__(self, name=None, chromosome=None):
        super().__init__(name=name)
        if chromosome is None:
            chromosome = (
                self.load(self.BEST_FILE) if self.BEST_FILE.exists() else DEFENSIVE_SEED
            )
        chromosome = np.clip(np.asarray(chromosome, dtype=float), 0.0, 1.0)
        self.genes = chromosome.reshape(Constants.NUM_PLAYERS, GENES_PER_ROLE)

    @staticmethod
    def load(path):
        return np.array(json.loads(Path(path).read_text())["chromosome"])

    @staticmethod
    def save(path, chromosome, **info):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps({"chromosome": list(map(float, chromosome)), **info})
        )

    def do_move(self):
        ball = np.asarray(self.ball_pos, dtype=float)
        moves = np.zeros((Constants.NUM_PLAYERS, 2))
        for i, (
            home_x,
            home_y,
            follow,
            track,
            engage,
            offset,
            push,
            brake,
        ) in enumerate(self.genes):
            pos = self.my_players_pos[i]
            vel = self.my_players_vel[i]
            if np.linalg.norm(ball - pos) < engage * 900:
                to_goal = OPPONENT_GOAL - ball
                to_goal /= np.linalg.norm(to_goal) + 1e-9
                behind = ball - to_goal * (
                    Constants.BALL_RADIUS + Constants.PLAYER_RADIUS + offset * 120
                )
                if np.linalg.norm(pos - behind) < 20 + offset * 40:
                    target = ball + to_goal * (50 + push * 300)
                else:
                    target = behind
            else:
                x = home_x * L + follow * (ball[0] - L / 2)
                y = (1 - track) * home_y * H + track * ball[1]
                target = np.array([np.clip(x, 0, L), np.clip(y, 0, H)])
            moves[i] = (target - pos) - brake * 8 * vel
        return moves
