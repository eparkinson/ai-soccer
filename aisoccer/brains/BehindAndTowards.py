import numpy as np

from aisoccer.brains.BaseBrainUtils import BaseBrainUtils


class BehindAndTowards(BaseBrainUtils):
    def do_move(self) -> np.array:

        actions = [[0] * 2] * 5
        for i in range(5):
            if self.is_behind_ball(i):
                actions[i] = self.run_towards(i, self.ball_pos)
            else:
                actions[i] = self.run_back(i)

        return np.array(actions)
