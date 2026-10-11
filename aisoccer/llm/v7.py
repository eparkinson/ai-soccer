"""
LLMBrain v7: players stop at their spot instead of running past it.

What I saw: v6 finished 2nd in league-3 round 18, then 4th in round 19, behind GP, its
champion copy and the Coach (v6 vs GP 1-1-8, goals 16-37). Our goal was never empty when
we conceded, yet the shots went in from a median 455 px out. Our players ran 66% further
than theirs (303969 px against 183357 a match): 6.8 px a tick on average against a top
speed of 7.5, so even the keeper and the sweeper, whose spots move slowly, are nearly
always at full speed. Each player accelerates towards its spot and nothing slows it down,
so it runs past, turns, and runs past again: the keeper swings to and fro across its line
and may be going the wrong way when the shot comes. GP's players steer differently: each
one picks the velocity that takes it to its spot, slowing as it arrives.

What I changed: every player now steers like GP's: it wants to move towards its spot at
top speed, or just fast enough to get there this tick when it is close, and accelerates
by the difference between that and how it is moving now. Every spot is exactly as in v6.

Why: a player that stops on its spot is on its spot: the keeper stands still in front of
the shot and the sweeper stays on the line to our goal. It changes how players move, not
where they go, so the shape that v3 and v5 showed matters is kept.
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


class LLMBrainV7(AbstractBrain):
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
        # each player wants to head for its spot at top speed, slowing as it arrives,
        # and changes its velocity towards that (v6 accelerated at the spot and overshot)
        wanted = targets - self.my_players_pos
        distance = np.linalg.norm(wanted, axis=1, keepdims=True)
        wanted *= np.minimum(1.0, C.MAX_PLAYER_VELOCITY / np.maximum(distance, 1e-9))
        return wanted - self.my_players_vel
