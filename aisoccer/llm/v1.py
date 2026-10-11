"""
LLMBrain v1: deliberately naive.

Every player runs straight at the ball. That is all. It is the kind of first attempt
most people write, and it plays like a pack of five-year-olds: everyone in one clump
around the ball, nobody in goal, and the ball goes wherever the scrum happens to push it.
"""

import numpy as np

from aisoccer.abstractbrain import AbstractBrain


class LLMBrainV1(AbstractBrain):
    def do_move(self):
        return np.asarray(self.ball_pos, dtype=float)[None, :] - self.my_players_pos
