from random import random

from core.abstractbrain import *


class BehindAndTowards(AbstractBrain):
    def do_move(self) -> np.array:

        result = []
        for i in range(5):
            acceleration = [0, 0]
            if self.is_behind_ball(i):
                acceleration = self.run_towards(i)
            else:
                acceleration = self.run_back(i)
            result.append(acceleration)
        return np.array(result)

    def run_towards(self, player_index):
        result = np.subtract(self.ball_pos, self.my_players_pos[player_index])
        result = result / np.linalg.norm(result)
        result = np.add(result, [0, (random() - 0.5)])
        return result

    def is_behind_ball(self, player_index):
        return self.my_players_pos[player_index][0] < self.ball_pos[0] - 5

    def run_back(self, player_index):
        result = [-5, random()-0.5]
        return result
