from aisoccer.abstractbrain import *
from aisoccer.constants import *
from aisoccer.physics import *


class Team:

    def __init__(self, brain: AbstractBrain, side):
        self.players = []
        self.brain = brain
        self.side = side

        for i in range(Constants.NUM_PLAYERS):
            starting_position = Constants.STARTING_POSITIONS[side][i]
            self.players.append(Player(side, starting_position))

    def apply_move(self, move: np.array):
        for i in range(len(self.players)):
            self.players[i].apply_move(move[i])

    def reset(self):
        for i in range(Constants.NUM_PLAYERS):
            starting_position = Constants.STARTING_POSITIONS[self.side][i]
            self.players[i].body.position = starting_position
            self.players[i].body.velocity = [0.0, 0.0]

    def position_matrix(self):
        result = []
        for p in self.players:
            position = [p.body.position[0], p.body.position[1]]
            result.append(position)
        return np.array(result)

    def velocity_matrix(self):
        result = []
        for p in self.players:
            velocity = [p.body.velocity[0], p.body.velocity[1]]
            result.append(velocity)
        return np.array(result)


class Player:
    def __init__(self, side, starting_position):
        self.side = side
        self.body = Body(Constants.PLAYER_RADIUS, starting_position)

    def apply_move(self, move: np.array):
        norm = np.linalg.norm(move)
        if norm > 0:
            normal_move = move / norm
        else:
            normal_move = move
        self.body.apply_acceleration(normal_move)
