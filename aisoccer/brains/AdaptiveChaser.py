from aisoccer.abstractbrain import AbstractBrain
import numpy as np

class AdaptiveChaser(AbstractBrain):
    def __init__(self, name=None):
        super().__init__()
        self.name = name  # Store the name if provided

    def do_move(self):
        moves = np.zeros_like(self.my_players_pos)

        # Calculate the score difference (mocked for now, replace with actual game state if available)
        score_difference = self.my_score - self.opp_score

        for i, player_pos in enumerate(self.my_players_pos):
            if score_difference > 0:  # Winning: Defensive strategy
                # Move players closer to the goal to defend
                goal_position = np.array([0, 0])  # Assume goal is at (0, 0)
                moves[i] = goal_position - player_pos
            else:  # Losing or tied: Offensive strategy
                # Chase the ball aggressively
                moves[i] = self.ball_pos - player_pos

        return moves

    def get_score_difference(self):
        # Placeholder for actual score difference logic
        return 0  # Assume tied for now

    def normalize_moves(self, moves):
        for i in range(len(moves)):
            norm = np.linalg.norm(moves[i])
            if norm > 0:
                moves[i] = moves[i] / norm
        return moves
