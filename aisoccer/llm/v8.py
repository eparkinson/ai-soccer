"""
LLMBrain v8: a second player joins the sweeper.

What I saw: v7 finished 3rd in league-3 round 20 (29-1-12), behind GP and the Coach
(v7 vs GP 1-2-7, goals 13-33; vs the Coach 2-2-6, goals 10-23). When we conceded, our
goal was never empty, the shot came from a median 399 px out, the nearest of our players
was 233 px from the ball, and 3.1 of our 5 players were upfield of it: the keeper and the
sweeper faced the attack alone while the three chasers were still behind the play. We
had the ball more (58%) and shot more (51 against 36 a match), so the attack is not
short of players. GP chases with one player and keeps the other four together in front
of its goal, and concedes little.

What I changed: player 2 stops chasing and stands with the sweeper, halfway between the
ball and the middle of our goal. The keeper, the two remaining chasers and the way every
player steers are exactly as in v7.

Why: it is v4's idea again, one more player goal-side of the ball, the change that
helped most so far. Two players on the line from the ball to our goal block twice the
width of one, and the two chasers still outnumber GP's single chaser. It keeps the
defenders back, which v3 and v5 showed matters, and moves nobody forward.
"""

import numpy as np

from aisoccer.abstractbrain import AbstractBrain
from aisoccer.constants import Constants as C

KEEPER = 0
SWEEPERS = [1, 2]  # v7 had one sweeper (player 1) and chasers 2, 3 and 4
CHASERS = [3, 4]
KEEPER_X = C.GOAL_DEPTH + C.PLAYER_RADIUS  # just in front of our goal line
OUR_GOAL = np.array([C.GOAL_DEPTH, C.FIELD_HEIGHT / 2])  # middle of our goal line
THEIR_GOAL = np.array([C.FIELD_LENGTH - 1 - C.GOAL_DEPTH, C.FIELD_HEIGHT / 2])
TOUCHING = C.PLAYER_RADIUS + C.BALL_RADIUS  # distance between centres when touching


class LLMBrainV8(AbstractBrain):
    def do_move(self):
        ball = np.asarray(self.ball_pos, dtype=float)
        targets = np.tile(ball, (len(self.my_players_pos), 1))
        # the keeper stays on the goal line, level with the ball
        targets[KEEPER] = (KEEPER_X, np.clip(ball[1], C.GOAL_Y_MIN, C.GOAL_Y_MAX))
        # the two sweepers stand together, halfway between the ball and our goal
        targets[SWEEPERS] = (ball + OUR_GOAL) / 2
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
