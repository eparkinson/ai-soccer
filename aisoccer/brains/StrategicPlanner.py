import numpy as np

from aisoccer.abstractbrain import AbstractBrain


class StrategicPlanner(AbstractBrain):
    def __init__(self, name=None):
        super().__init__()
        self.name = name  # Store the name if provided

    def do_move(self):
        moves = np.zeros_like(self.my_players_pos)

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
        # Stay near the goal and block shots
        goal_position = np.array([0, 0])  # Assume goal is at (0, 0)
        if (
            np.linalg.norm(self.ball_pos - goal_position) < 5
        ):  # Ball is close to the goal
            return self.ball_pos - player_pos  # Move towards the ball
        return goal_position - player_pos  # Stay near the goal

    def plan_defender(self, player_pos):
        # Position between the ball and the goal
        goal_position = np.array([0, 0])
        intercept_position = (self.ball_pos + goal_position) / 2
        return intercept_position - player_pos

    def plan_midfielder(self, player_pos):
        # Stay near the center and assist
        center_position = np.array([50, 50])  # Assume center of the field
        if np.linalg.norm(self.ball_pos - player_pos) < 10:  # Ball is close
            return self.ball_pos - player_pos  # Move towards the ball
        return center_position - player_pos

    def plan_attacker(self, player_pos):
        # Chase the ball and aim for the goal
        if np.linalg.norm(self.ball_pos - player_pos) < 5:  # Close to the ball
            opponent_goal = np.array([100, 50])  # Assume opponent's goal position
            return opponent_goal - player_pos  # Move towards the goal
        return self.ball_pos - player_pos  # Move towards the ball
