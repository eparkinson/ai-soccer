"""
LLMBrain v10: the chasers strike the ball instead of stopping against it.

What I saw: v8 finished 4th in league-3 rounds 21 and 22, behind GP (the champion since
round 18) and the Coach (after round 22: v8 vs GP 0-3-7, goals 8-19; vs the Coach 0-1-9,
goals 2-29). We have the ball more than they do (54%) and pass more, yet they shoot more
(45.6 against 38.9 a match) and we score under one goal a match against GP, almost only
from scrambles right at their goal (median 35 px out). Since v7 every player slows as it
reaches its spot, and a chaser's spot is the back of the ball, touching it: so a chaser
arrives there at almost no speed and nudges the ball along instead of hitting it. Players
weigh 9 times as much as the ball, so a player running into it at top speed (7.5 px a
tick) sends it off at about 13.5, nearly the ball's top speed of 15: a shot. GP's chaser
runs at the ball and does not slow down.

What I changed: the two chasers no longer slow down as they reach the back of the ball;
they run at it at top speed. Where every player goes, and how the keeper and the two
sweepers move, are exactly as in v8.

Why: running through the back of the ball at full speed turns our possession into shots
towards their goal from further out, instead of slow dribbles that end in scrambles. It
changes only how the chasers arrive, so the defence that v3, v5 and v9 showed matters is
left alone.
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


class LLMBrainV10(AbstractBrain):
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
        # each player heads for its spot at top speed and changes its velocity towards
        # that; the keeper and the sweepers slow as they arrive, so they stop on their spot
        wanted = targets - self.my_players_pos
        distance = np.linalg.norm(wanted, axis=1, keepdims=True)
        top_speed = C.MAX_PLAYER_VELOCITY / np.maximum(distance, 1e-9)
        slowing = np.minimum(1.0, top_speed)
        # v10: the chasers never slow down, so they strike the ball instead of nudging it
        slowing[CHASERS] = top_speed[CHASERS]
        wanted *= slowing
        return wanted - self.my_players_vel
