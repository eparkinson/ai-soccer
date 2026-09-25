import numpy as np

from aisoccer.abstractbrain import AbstractBrain
from aisoccer.constants import Constants


class AdaptiveChaser(AbstractBrain):
    """Chases the ball while level or behind; falls back to defend its own goal when winning."""

    OWN_GOAL = np.array([0.0, Constants.FIELD_HEIGHT / 2])

    def do_move(self):
        moves = np.zeros_like(self.my_players_pos, dtype=float)

        score_difference = self.my_score - self.opp_score

        for i, player_pos in enumerate(self.my_players_pos):
            if score_difference > 0:  # Winning: Defensive strategy
                moves[i] = self.OWN_GOAL - player_pos
            else:  # Losing or tied: Offensive strategy
                moves[i] = self.ball_pos - player_pos

        return moves
