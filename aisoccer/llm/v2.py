"""
LLMBrain v2: one player stays in goal.

What I saw: v1 finished 10th of 12. When it conceded, our goal was empty 98% of the
time and all five players were upfield of the ball; the nearest one was a median
475 px from it. Most goals came from mid-range touches (median 264 px out) into an
open net. Going forward it was fine: more shots than its opponents.

What I changed: player 0 no longer chases the ball. It stands just in front of our
goal line and slides up and down it to stay level with the ball, never leaving the
goal mouth. The other four still run at the ball, exactly as in v1.

Why: an empty net was the one thing present in nearly every goal we let in. A
keeper is the smallest change that puts someone there, and it costs only one of
five chasers, while v1 already out-shot everyone.
"""

import numpy as np

from aisoccer.abstractbrain import AbstractBrain
from aisoccer.constants import Constants as C

KEEPER = 0
KEEPER_X = C.GOAL_DEPTH + C.PLAYER_RADIUS  # just in front of our goal line


class LLMBrainV2(AbstractBrain):
    def do_move(self):
        ball = np.asarray(self.ball_pos, dtype=float)
        targets = np.tile(ball, (len(self.my_players_pos), 1))  # everyone chases the ball
        # except the keeper, who stays on the goal line, level with the ball
        targets[KEEPER] = (KEEPER_X, np.clip(ball[1], C.GOAL_Y_MIN, C.GOAL_Y_MAX))
        return targets - self.my_players_pos
