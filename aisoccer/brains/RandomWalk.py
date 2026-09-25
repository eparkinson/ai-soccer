import numpy as np

from aisoccer.abstractbrain import AbstractBrain


class RandomWalk(AbstractBrain):
    def do_move(self) -> np.ndarray:
        return self.rng.random((5, 2)) - 0.5
