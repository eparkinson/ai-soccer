import numpy as np

from aisoccer.abstractbrain import AbstractBrain
from aisoccer.constants import Constants
from aisoccer.physics import Body


class Team:

    def __init__(self, brain: AbstractBrain, side):
        self.players = []
        self.brain = brain
        self.side = "blue" if side == 0 else "red"  # Map side to 'blue' or 'red'
        self.original_side = side  # Store the original side value (0 or 1)

        for i in range(Constants.NUM_PLAYERS):
            starting_position = Constants.STARTING_POSITIONS[side][i]
            self.players.append(Player(side, starting_position))

    def apply_move(self, move: np.array):
        self.brain.last_move = []  # type: ignore[attr-defined]
        for i in range(len(self.players)):
            normal_move = self.players[i].apply_move(move[i])
            self.brain.last_move.append(normal_move)  # type: ignore[attr-defined]

    def reset(self):
        for i in range(Constants.NUM_PLAYERS):
            starting_position = Constants.STARTING_POSITIONS[self.original_side][
                i
            ]  # Use original side
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
        if norm > 1:
            normal_move = move / norm
        else:
            normal_move = move

        self.body.apply_acceleration(normal_move)

        return normal_move
