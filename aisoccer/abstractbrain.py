from abc import ABC, abstractmethod
from typing import final

import numpy as np


class AbstractBrain(ABC):
    def __init__(self):
        self.name = None
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
    def get_name(self):
        if self.name is None:
            return type(self).__name__
        else:
            return self.name

    @final
    def move(self,
             my_players_pos: np.array,
             my_players_vel: np.array,
             opp_players_pos: np.array,
             app_players_vel: np.array,
             ball_pos: np.array,
             ball_vel: np.array,
             my_score: int,
             opp_score: int,
             game_time: float
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
        self.opp_players_pos = opp_players_pos
        self.ball_pos = ball_pos
        self.ball_vel = ball_vel
        self.my_score = my_score
        self.opp_score = my_score
        self.game_time = game_time

        return self.do_move()

    @abstractmethod
    def do_move(self) -> np.array:
        pass
