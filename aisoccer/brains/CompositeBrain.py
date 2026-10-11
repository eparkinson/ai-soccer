"""
A team assembled from other brains, one per player: player i plays as player i of the
brain given for role i. Any brains can be mixed (networks, LLMBrain, GP programs): each
sees the whole game as usual, and only its own player's move is used.
"""

import numpy as np

from aisoccer.abstractbrain import AbstractBrain
from aisoccer.brainspec import load_brain
from aisoccer.constants import Constants


class CompositeBrain(AbstractBrain):
    def __init__(self, name=None, roles=None):
        """
        roles: a brain spec (see aisoccer/brainspec.py) for each of the five players
        (default: five SimpleBrain players, so the class can be created without arguments).
        """
        self.members = []
        super().__init__(name=name)
        roles = roles if roles is not None else ["SimpleBrain"] * Constants.NUM_PLAYERS
        if len(roles) != Constants.NUM_PLAYERS:
            raise ValueError(f"CompositeBrain needs {Constants.NUM_PLAYERS} role specs")
        self.roles = list(roles)
        # One brain per distinct spec: roles that share a brain share its computation.
        self.members = [load_brain(spec, f"{name or 'Composite'}-{k}") for k, spec in enumerate(dict.fromkeys(roles))]
        index = {spec: k for k, spec in enumerate(dict.fromkeys(roles))}
        self.member_of_role = [index[spec] for spec in roles]
        self.rng = np.random.default_rng()

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, generator):
        """Game.seed_brains sets this; each member gets its own generator derived from it."""
        self._rng = generator
        for member, seed in zip(self.members, generator.integers(2**63, size=len(self.members))):
            member.rng = np.random.default_rng(int(seed))

    def do_move(self):
        state = (
            self.my_players_pos, self.my_players_vel, self.opp_players_pos, self.opp_players_vel,
            self.ball_pos, self.ball_vel, self.my_score, self.opp_score, self.game_time,
        )
        moves = [np.asarray(member.move(*state), dtype=float) for member in self.members]
        return np.array([moves[m][i] for i, m in enumerate(self.member_of_role)])

    def on_goal_scored(self, team, game_state):
        for member in self.members:
            member.on_goal_scored(team, game_state)

    def on_goal_conceded(self, team, game_state):
        for member in self.members:
            member.on_goal_conceded(team, game_state)
