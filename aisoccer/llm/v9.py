"""
LLMBrain v9: the two sweepers stand shoulder to shoulder.

What I saw: v8 finished 4th in league-3 round 21 (25-1-16, +12.9 a game), behind GP (the
champion) and the Coach (v8 vs GP 0-2-8, goals 9-27; vs the Coach 0-3-7, goals 7-24). The
second sweeper did its job of getting players back: when we conceded, 2.1 of our 5 players
were upfield of the ball (v7: 3.1). But the goals still came from long shots (median 459 px
out) with our goal never empty, so our players were back yet not in the way of the shot. Both
sweepers aim at the very same spot, halfway between the ball and the middle of our goal, and
players bounce off each other, so they spend their time knocking each other aside: two
players there cover about the width of one.

What I changed: the two sweepers stand side by side, touching, across the line from the ball
to the middle of our goal, one on each side of it, instead of both on it. Their midpoint is
v8's spot. The keeper, the chasers and the way every player steers are exactly as in v8.

Why: a shot has to pass the sweepers to reach the goal. Standing shoulder to shoulder they
make a wall two players wide in front of the shooter, twice what one blocks, and they stop
pushing each other off the line. It moves nobody forward (v3, v5) and keeps v8's shape.
"""

import numpy as np

from aisoccer.abstractbrain import AbstractBrain
from aisoccer.constants import Constants as C

KEEPER = 0
SWEEPERS = [1, 2]
CHASERS = [3, 4]
KEEPER_X = C.GOAL_DEPTH + C.PLAYER_RADIUS  # just in front of our goal line
OUR_GOAL = np.array([C.GOAL_DEPTH, C.FIELD_HEIGHT / 2])  # middle of our goal line
THEIR_GOAL = np.array([C.FIELD_LENGTH - 1 - C.GOAL_DEPTH, C.FIELD_HEIGHT / 2])
TOUCHING = C.PLAYER_RADIUS + C.BALL_RADIUS  # distance between centres when touching


class LLMBrainV9(AbstractBrain):
    def do_move(self):
        ball = np.asarray(self.ball_pos, dtype=float)
        targets = np.tile(ball, (len(self.my_players_pos), 1))
        # the keeper stays on the goal line, level with the ball
        targets[KEEPER] = (KEEPER_X, np.clip(ball[1], C.GOAL_Y_MIN, C.GOAL_Y_MAX))
        # the two sweepers stand halfway between the ball and our goal, shoulder to
        # shoulder across the line from the ball to our goal (v8: both on the line)
        to_ball = ball - OUR_GOAL
        to_ball /= max(np.linalg.norm(to_ball), 1.0)
        across = np.array([-to_ball[1], to_ball[0]]) * C.PLAYER_RADIUS
        targets[SWEEPERS] = [(ball + OUR_GOAL) / 2 + across, (ball + OUR_GOAL) / 2 - across]
        # the chasers run at the back of the ball, so they push it towards their goal
        to_their_goal = THEIR_GOAL - ball
        to_their_goal /= max(np.linalg.norm(to_their_goal), 1.0)
        targets[CHASERS] = ball - to_their_goal * TOUCHING
        # each player wants to head for its spot at top speed, slowing as it arrives,
        # and changes its velocity towards that (as in v7)
        wanted = targets - self.my_players_pos
        distance = np.linalg.norm(wanted, axis=1, keepdims=True)
        wanted *= np.minimum(1.0, C.MAX_PLAYER_VELOCITY / np.maximum(distance, 1e-9))
        return wanted - self.my_players_vel
