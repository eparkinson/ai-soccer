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

    def apply_move(self, move: np.ndarray) -> np.ndarray:
        """Apply a 5 x 2 acceleration matrix and return the (magnitude-capped) accelerations used."""
        # Guard the physics against brains that return NaN/inf.
        move = np.asarray(move, dtype=float)
        if not np.isfinite(move).all():  # only scrub when needed: same result, much faster
            move = np.nan_to_num(move, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(move, axis=1, keepdims=True)
        normal_move = np.where(norms > 1, move / np.maximum(norms, 1), move)
        rows = self.rows()
        if rows is not None:  # all players in one physics state: one array update
            state = self.players[0].body._state
            state.vel[rows] = state.vel[rows] + normal_move
        else:
            for i, player in enumerate(self.players):
                player.body.apply_acceleration(normal_move[i])
        return normal_move

    def rows(self):
        """The players' rows in their physics state's arrays, or None if not all in one."""
        first = self.players[0].body
        cached = getattr(self, "_rows", None)
        if cached is not None and first._state is cached[0] and first._index == cached[1][0]:
            return cached[1]
        state = first._state
        if state is None or any(p.body._state is not state for p in self.players):
            return None
        rows = np.array([p.body._index for p in self.players])
        self._rows = (state, rows)
        return rows

    def reset(self):
        for i, player in enumerate(self.players):
            player.body.position = np.array(
                Constants.STARTING_POSITIONS[self.original_side][i], dtype=float
            )
            player.body.velocity = np.array([0.0, 0.0])

    def position_matrix(self):
        rows = self.rows()
        if rows is not None:
            return self.players[0].body._state.pos[rows]  # fancy indexing: a copy
        return np.array([p.body.position for p in self.players], dtype=float)

    def velocity_matrix(self):
        rows = self.rows()
        if rows is not None:
            return self.players[0].body._state.vel[rows]
        return np.array([p.body.velocity for p in self.players], dtype=float)


class Player:
    def __init__(self, side, starting_position):
        self.side = side
        self.body = Body(Constants.PLAYER_RADIUS, starting_position)

    def apply_move(self, move: np.ndarray):
        norm = np.linalg.norm(move)
        if norm > 1:
            normal_move = move / norm
        else:
            normal_move = move

        self.body.apply_acceleration(normal_move)

        return normal_move
