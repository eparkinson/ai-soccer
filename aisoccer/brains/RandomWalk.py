from random import random

import numpy as np

from aisoccer.abstractbrain import AbstractBrain


class RandomWalk(AbstractBrain):
    def do_move(self) -> np.array:
        actions = [[0] * 2] * 5

        for i in range(5):
            actions[i] = [random() - 0.5, random() - 0.5]

        return np.array(actions)
