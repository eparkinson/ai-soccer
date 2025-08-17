from abc import ABC, abstractmethod
from typing import final

import numpy as np


class AbstractBrain(ABC):
    def __init__(self, name=None):
        if name is None:
            self.name = type(self).__name__
        else:
            self.name = name

        self.my_players_pos = None
        self.my_players_vel = None
        self.opp_players_pos = None
        self.opp_players_pos = None
        self.ball_pos = None
        self.ball_vel = None
        self.my_score = None
        self.opp_score = None
        self.game_time = None

    @final
    def move(
        self,
        my_players_pos: np.array,
        my_players_vel: np.array,
        opp_players_pos: np.array,
        opp_players_vel: np.array,
        ball_pos: np.array,
        ball_vel: np.array,
        my_score: int,
        opp_score: int,
        game_time: float,
    ) -> np.array:
        """
        my_players_pos - a 5 x 2 matrix where the i'th row is the 2D position vector for friendly player i.
        my_players_vel - a 5 x 2 matrix where the i'th row is the 2D velocity vector for friendly player i.

        opp_players_pos - a 5 x 2 matrix where the i'th row is the 2D position vector for opposing player i.
        opp_players_vel - a 5 x 2 matrix where the i'th row is the 2D velocity vector for opposing player i.

        ball_pos - a 1 x 2 matrix (2D vector) for the ball's position
        ball_vel - a 1 x 2 matrix (2D vector) for the ball's velocity

        my_score - number of goals the friendly team has scored
        opp_score - number of goals the opposing team has scored

        game_time: a float between 0 and 1 indicating the percentage game time elapsed

        OUTPUT - a 5 x 2 matrix where the i'th row is the 2D acceleration vector for friendly player i
        """
        self.my_players_pos = my_players_pos
        self.my_players_vel = my_players_vel
        self.opp_players_pos = opp_players_pos
        self.opp_players_pos = opp_players_vel
        self.ball_pos = ball_pos
        self.ball_vel = ball_vel
        self.my_score = my_score
        self.opp_score = opp_score
        self.game_time = game_time

        game_state = {
            "my_players_pos": my_players_pos,
            "my_players_vel": my_players_vel,
            "opp_players_pos": opp_players_pos,
            "opp_players_vel": opp_players_vel,
            "ball_pos": ball_pos,
            "ball_vel": ball_vel,
            "my_score": my_score,
            "opp_score": opp_score,
            "game_time": game_time,
        }

        return (
            self.do_move(game_state=game_state)
            if "game_state" in self.do_move.__code__.co_varnames
            else self.do_move()
        )

    @abstractmethod
    def do_move(self, game_state=None) -> np.array:
        pass

    def on_goal_scored(self, team: str, game_state: dict):
        """
        Callback triggered when a goal is scored.

        :param team: The team that scored ('red' or 'blue').
        :param game_state: The current game state as a dictionary.
        """
        pass

    def on_goal_conceded(self, team: str, game_state: dict):
        """
        Callback triggered when a goal is conceded.

        :param team: The team that conceded ('red' or 'blue').
        :param game_state: The current game state as a dictionary.
        """
        pass
