import numpy as np

from aisoccer.abstractbrain import AbstractBrain


class LearningBrain(AbstractBrain):
    """
    Placeholder learning brain: a state -> action table nudged by goal rewards.

    The state is the ball's position snapped to a coarse grid so that states actually
    repeat. This is a starting point rather than a real RL agent.
    """

    GRID_SIZE = 100  # pixels per state cell
    EXPLORATION_RATE = 0.1
    LEARNING_RATE = 0.1

    def __init__(self, name=None):
        super().__init__(name=name)
        self.q_table = {}  # State-action table
        self.last_state = None
        self.last_action = None

    def discretise_state(self):
        return tuple(int(v) for v in np.asarray(self.ball_pos) // self.GRID_SIZE)

    def do_move(self):
        state = self.discretise_state()

        # Exploration vs. Exploitation
        if state not in self.q_table or self.rng.random() < self.EXPLORATION_RATE:
            action = self.rng.uniform(-1, 1, (5, 2))  # Random action
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
                self.q_table[self.last_state] = np.array(self.last_action, dtype=float)
            # Simple reward-based update (can be replaced with more complex logic)
            self.q_table[self.last_state] += reward * self.LEARNING_RATE

    def on_goal_scored(self, team: str, game_state: dict):
        """Reward the brain when its team scores, with higher rewards for faster goals."""
        ticks_elapsed = max(game_state.get("ticks_elapsed", 1), 1)
        self.update_q_table(reward=1.0 / ticks_elapsed)

    def on_goal_conceded(self, team: str, game_state: dict):
        """Penalise the brain when its team concedes, with higher penalties for faster concessions."""
        ticks_elapsed = max(game_state.get("ticks_elapsed", 1), 1)
        self.update_q_table(reward=-1.0 / ticks_elapsed)
