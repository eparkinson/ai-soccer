import numpy as np
from random import random

from abstractbrain import *


class RandomWalk(AbstractBrain):
    def do_move(self) -> np.array:
        result = []
        for i in range(5):
            acc = [(random() - 0.5), (random() - 0.5)]
            result.append(acc)
        return np.array(result)
