"""
LLMBrain v5: the sweeper closes down the ball.

What I saw: v4 won league-3 rounds 14 and 15, then finished 3rd in round 16 behind GP
and its champion copy (v4 vs GP 6-0-4, vs the Coach 2-4-4, conceding 6.5 a match to
the Coach). When we conceded, the decisive touch came from a median 348 px out (301 px
for v2), and the nearest of our players was a median 190 px from the ball: that is the
sweeper, standing halfway back. The shooter had the ball to himself.

What I changed: the sweeper no longer stands halfway between the ball and our goal.
It stands right in front of the ball, touching it, on the ball's side facing our goal.
The keeper and the three chasers are exactly as in v4.

Why: the sweeper already stood on the right line, from the ball to our goal; it was just
too far down it. Right against the ball it blocks every direct shot and dribble towards
our goal at the source, instead of one third of the goal from halfway, and when it
touches the ball it pushes it away from our goal. It is the smallest change that puts
someone on the shooter, and it keeps v4's shape of one keeper, one sweeper, three chasers.
"""

import numpy as np

from aisoccer.abstractbrain import AbstractBrain
from aisoccer.constants import Constants as C

KEEPER = 0
SWEEPER = 1
KEEPER_X = C.GOAL_DEPTH + C.PLAYER_RADIUS  # just in front of our goal line
OUR_GOAL = np.array([C.GOAL_DEPTH, C.FIELD_HEIGHT / 2])  # middle of our goal line
TOUCHING = C.PLAYER_RADIUS + C.BALL_RADIUS  # distance between centres when player and ball touch


class LLMBrainV5(AbstractBrain):
    def do_move(self):
        ball = np.asarray(self.ball_pos, dtype=float)
        targets = np.tile(ball, (len(self.my_players_pos), 1))  # everyone chases the ball
        # except the keeper, who stays on the goal line, level with the ball
        targets[KEEPER] = (KEEPER_X, np.clip(ball[1], C.GOAL_Y_MIN, C.GOAL_Y_MAX))
        # and the sweeper, who stands against the ball on the side facing our goal
        towards_goal = OUR_GOAL - ball
        targets[SWEEPER] = ball + towards_goal / max(np.linalg.norm(towards_goal), 1) * TOUCHING
        return targets - self.my_players_pos
