"""
LLMBrain v6: the chasers run at the back of the ball.

What I saw: v4 won league-3 rounds 14 and 15, then finished 3rd and 2nd (rounds 16 and
17) behind GP and the Coach. In round 16 we had the ball 66% of the time but took fewer
shots than our opponents (34 against 45 a match), and ran nearly twice as far. Our three
chasers all run at the same point, the ball itself, from whichever side they happen to be
on, so a chaser coming from upfield knocks the ball back towards our own goal. v3 and v5
pulled defenders forward and conceded more, so the defence (keeper and sweeper) stays.

What I changed: the three chasers no longer run at the middle of the ball. They run at
its back: the point just behind the ball on the line from the middle of the opponents'
goal through the ball. The keeper and the sweeper are exactly as in v4.

Why: a player that reaches the ball from behind pushes it towards the opponents' goal,
so each touch becomes a step towards a shot instead of a push in a random direction. It
changes only where the chasers aim, not who defends.
"""

import numpy as np

from aisoccer.abstractbrain import AbstractBrain
from aisoccer.constants import Constants as C

KEEPER = 0
SWEEPER = 1
CHASERS = [2, 3, 4]
KEEPER_X = C.GOAL_DEPTH + C.PLAYER_RADIUS  # just in front of our goal line
OUR_GOAL = np.array([C.GOAL_DEPTH, C.FIELD_HEIGHT / 2])  # middle of our goal line
THEIR_GOAL = np.array([C.FIELD_LENGTH - 1 - C.GOAL_DEPTH, C.FIELD_HEIGHT / 2])
TOUCHING = C.PLAYER_RADIUS + C.BALL_RADIUS  # distance between centres when touching


class LLMBrainV6(AbstractBrain):
    def do_move(self):
        ball = np.asarray(self.ball_pos, dtype=float)
        targets = np.tile(ball, (len(self.my_players_pos), 1))
        # the keeper stays on the goal line, level with the ball
        targets[KEEPER] = (KEEPER_X, np.clip(ball[1], C.GOAL_Y_MIN, C.GOAL_Y_MAX))
        # the sweeper stands halfway between the ball and our goal
        targets[SWEEPER] = (ball + OUR_GOAL) / 2
        # the chasers run at the back of the ball, so they push it towards their goal
        to_their_goal = THEIR_GOAL - ball
        to_their_goal /= max(np.linalg.norm(to_their_goal), 1.0)
        targets[CHASERS] = ball - to_their_goal * TOUCHING
        return targets - self.my_players_pos
