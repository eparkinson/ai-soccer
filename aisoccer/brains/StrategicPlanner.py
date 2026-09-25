import numpy as np

from aisoccer.abstractbrain import AbstractBrain
from aisoccer.constants import Constants


class StrategicPlanner(AbstractBrain):
    """Fixed roles: goalkeeper, two defenders, a midfielder and an attacker."""

    OWN_GOAL = np.array(
        [Constants.GOAL_DEPTH + Constants.PLAYER_RADIUS, Constants.FIELD_HEIGHT / 2]
    )
    OPPONENT_GOAL = np.array([Constants.FIELD_LENGTH - 1, Constants.FIELD_HEIGHT / 2])
    CENTRE = np.array([(Constants.FIELD_LENGTH - 1) / 2, Constants.FIELD_HEIGHT / 2])

    # Distances (pixels) at which each role engages the ball
    KEEPER_RANGE = 200
    MIDFIELD_RANGE = 150
    ATTACKER_RANGE = Constants.PLAYER_RADIUS + Constants.BALL_RADIUS + 10

    def do_move(self):
        moves = np.zeros_like(self.my_players_pos, dtype=float)

        # Assign roles dynamically based on game state
        for i, player_pos in enumerate(self.my_players_pos):
            if i == 0:  # Goalkeeper
                moves[i] = self.plan_goalkeeper(player_pos)
            elif i in [1, 2]:  # Defenders
                moves[i] = self.plan_defender(player_pos)
            elif i == 3:  # Midfielder
                moves[i] = self.plan_midfielder(player_pos)
            else:  # Attacker
                moves[i] = self.plan_attacker(player_pos)

        return moves

    def plan_goalkeeper(self, player_pos):
        # Stay on the goal line, and come out for the ball when it is close
        if np.linalg.norm(self.ball_pos - self.OWN_GOAL) < self.KEEPER_RANGE:
            return self.ball_pos - player_pos
        target = np.array(
            [
                self.OWN_GOAL[0],
                np.clip(self.ball_pos[1], Constants.GOAL_Y_MIN, Constants.GOAL_Y_MAX),
            ]
        )
        return target - player_pos

    def plan_defender(self, player_pos):
        # Position between the ball and the goal
        intercept_position = (self.ball_pos + self.OWN_GOAL) / 2
        return intercept_position - player_pos

    def plan_midfielder(self, player_pos):
        # Stay near the center and assist
        if np.linalg.norm(self.ball_pos - player_pos) < self.MIDFIELD_RANGE:
            return self.ball_pos - player_pos
        return self.CENTRE - player_pos

    def plan_attacker(self, player_pos):
        # Get behind the ball, then drive it towards the opponent's goal
        if (
            player_pos[0] < self.ball_pos[0]
            and np.linalg.norm(self.ball_pos - player_pos) < self.ATTACKER_RANGE
        ):
            return self.OPPONENT_GOAL - player_pos
        return self.ball_pos - player_pos
