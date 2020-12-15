import numpy as np
import physics
from abstractbrain import *


class Team:
    NUM_PLAYERS = 5
    STARTING_POSITIONS = [[[200, 200], [200, 400], [500, 200], [500, 400], [800, 300]],
                          [[1600, 200], [1600, 400], [1300, 200], [1300, 400], [1000, 300]]]

    def __init__(self, brain: AbstractBrain, side):
        self.players = []
        self.brain = brain
        self.side = side

        for i in range(Team.NUM_PLAYERS):
            starting_position = Team.STARTING_POSITIONS[side][i]
            self.players.append(Player(side, starting_position))


    def apply_move(self, move: np.array):
        for i in range(len(self.players)):
            self.players[i].apply_move(move[i])

    def reset(self):
        for i in range(Team.NUM_PLAYERS):
            starting_position = Team.STARTING_POSITIONS[self.side][i]
            self.players[i].body.position = starting_position
            self.players[i].body.velocity = [0.0, 0.0]

    def position_matrix(self):
        result = []
        for p in self.players:
            position = [p.body.position[0], p.body.position[1]]
            result.append(position)
        return np.array(result)


class Player:
    RADIUS = 30

    def __init__(self, side, starting_position):
        self.side = side
        self.body = physics.Body(Player.RADIUS, starting_position)

    def apply_move(self, move: np.array):
        normal_move = move / np.linalg.norm(move)
        self.body.apply_acceleration(normal_move)
