from random import random

from aisoccer.abstractbrain import *
from aisoccer.brains.BaseBrainUtils import *


class BehindAndTowards(BaseBrainUtils):
    def do_move(self) -> np.array:

        actions = arr = [[0]*2]*5
        for i in range(5):
            if self.is_behind_ball(i):
                actions[i] = self.run_towards(i, self.ball_pos)
            else:
                actions[i] = self.run_back(i)

        return np.array(actions)
