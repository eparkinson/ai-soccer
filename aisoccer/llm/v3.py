"""
LLMBrain v3: the keeper comes off the line.

What I saw: v2 finished 3rd of 10 in its first round (league 3, round 12). The keeper
worked: our goal was never empty when we conceded. But we still let in 8-9 goals a
match against GP and the Coach, and the decisive touches came from a median 301 px
out. A keeper who stays on the goal line and only slides sideways cannot reach a shot
into a 400 px goal from closer than about 360 px: the ball is simply faster.

What I changed: the keeper no longer stays on the goal line. It stands on the straight
line from the middle of our goal to the ball, halfway to the ball, but never more than
half a goal-width out. The other four still run at the ball, exactly as in v1 and v2.

Why: stepping out towards the ball is what real keepers do to "narrow the angle": the
closer the keeper is to the shooter, the less of the goal is left to aim at, so the
same sideways speed covers more of it. It keeps the one-keeper, four-chaser shape that
already beats everyone except GP and the Coach, and changes only where the keeper stands.
"""

import numpy as np

from aisoccer.abstractbrain import AbstractBrain
from aisoccer.constants import Constants as C

KEEPER = 0


class LLMBrainV3(AbstractBrain):
    def do_move(self):
        ball = np.asarray(self.ball_pos, dtype=float)
        targets = np.tile(ball, (len(self.my_players_pos), 1))  # everyone chases the ball
        targets[KEEPER] = self.keeper_spot(ball)  # except the keeper
        return targets - self.my_players_pos

    @staticmethod
    def keeper_spot(ball):
        """Halfway from the middle of our goal to the ball, at most half a goal-width out."""
        goal = np.array([C.GOAL_DEPTH + C.PLAYER_RADIUS, C.FIELD_HEIGHT / 2])
        to_ball = ball - goal
        distance = np.linalg.norm(to_ball)
        if distance < 1:
            return goal
        step_out = min(distance / 2, C.GOAL_MOUTH_HEIGHT / 2)
        return goal + to_ball / distance * step_out
