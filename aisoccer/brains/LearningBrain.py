from aisoccer.abstractbrain import AbstractBrain
import random
import numpy as np


class LearningBrain(AbstractBrain):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.q_table = {}  # State-action table
        self.last_state = None
        self.last_action = None

    def decide(self, game_state):
        """
        Stub implementation: Currently behaves like RandomWalk.
        Replace this with learning-based decision-making logic.
        """
        # Randomly choose a direction and speed for now
        direction = random.uniform(0, 360)  # Angle in degrees
        speed = random.uniform(0, 1)  # Speed as a fraction of max speed
        return direction, speed

    def do_move(self, game_state=None):
        """
        Implements a minimal learning mechanism using a state-action table.
        """
        # Simplify the state (e.g., ball position)
        state = tuple(self.ball_pos)

        # Exploration vs. Exploitation
        if state not in self.q_table or random.random() < 0.1:  # 10% exploration
            action = np.random.uniform(-1, 1, (5, 2))  # Random action
        else:
            action = self.q_table[state]  # Best known action

        # Store the state and action for learning
        self.last_state = state
        self.last_action = action

        return action

    def update_q_table(self, reward):
        """
        Update the Q-table based on the reward received.
        """
        if self.last_state is not None and self.last_action is not None:
            if self.last_state not in self.q_table:
                self.q_table[self.last_state] = self.last_action
            # Simple reward-based update (can be replaced with more complex logic)
            self.q_table[self.last_state] += reward * 0.1  # Learning rate

    def on_goal_scored(self, team: str, game_state: dict):
        """
        Reward the brain when its team scores a goal, with higher rewards for faster goals.

        :param team: The team that scored ('red' or 'blue').
        :param game_state: The current game state as a dictionary.
        """
        if team == "blue":  # Assuming this brain is on the blue team
            ticks_elapsed = game_state.get("ticks_elapsed", 1)
            reward = 1.0 / ticks_elapsed  # Higher reward for faster goals
            self.update_q_table(reward=reward)

    def on_goal_conceded(self, team: str, game_state: dict):
        """
        Penalize the brain when its team concedes a goal, with higher penalties for faster concessions.

        :param team: The team that conceded ('red' or 'blue').
        :param game_state: The current game state as a dictionary.
        """
        if team == "blue":  # Assuming this brain is on the blue team
            ticks_elapsed = game_state.get("ticks_elapsed", 1)
            penalty = -1.0 / ticks_elapsed  # Higher penalty for faster concessions
            self.update_q_table(reward=penalty)
