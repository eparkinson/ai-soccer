"""
LLMBrain v4: one player stays between the ball and our goal.

What I saw: v2 finished 3rd then 4th of 10 (league 3, rounds 12 and 13), behind GP
and the Coach, who scored 8-9 a match against it. When we conceded, our goal was
never empty, but on average 4.1 of our 5 players were upfield of the ball: the keeper
faced every attack alone. v3 tried to fix this by moving the keeper off its line and
conceded more, so the keeper is not the problem; the lack of anyone else back is.

What I changed: player 1 no longer chases the ball. It stands halfway between the
ball and the middle of our goal. The keeper and the other three chasers are exactly
as in v2.

Why: a player on the straight line from the ball to our goal is in the way of the
most direct shot and of a dribble towards goal, and it is always goal-side of the
ball, so the keeper is no longer alone. It is the smallest change that puts a second
player behind the ball, and it leaves the keeper, which worked, untouched.
"""

import numpy as np

from aisoccer.abstractbrain import AbstractBrain
from aisoccer.constants import Constants as C

KEEPER = 0
SWEEPER = 1
KEEPER_X = C.GOAL_DEPTH + C.PLAYER_RADIUS  # just in front of our goal line
OUR_GOAL = np.array([C.GOAL_DEPTH, C.FIELD_HEIGHT / 2])  # middle of our goal line


class LLMBrainV4(AbstractBrain):
    def do_move(self):
        ball = np.asarray(self.ball_pos, dtype=float)
        targets = np.tile(ball, (len(self.my_players_pos), 1))  # everyone chases the ball
        # except the keeper, who stays on the goal line, level with the ball
        targets[KEEPER] = (KEEPER_X, np.clip(ball[1], C.GOAL_Y_MIN, C.GOAL_Y_MAX))
        # and the sweeper, who stands halfway between the ball and our goal
        targets[SWEEPER] = (ball + OUR_GOAL) / 2
        return targets - self.my_players_pos
