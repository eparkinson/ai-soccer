"""
TacticsBrain version 2: the same team structure as v1, but the players' skills have to
be discovered.

v1's skills are expert from the start (ball prediction, lining up kicks, aimed shots,
a goalkeeper), so even a random strategy plays well. v2 adds four skill genes. At 0 a
skill is absent; the genetic algorithm starts them near 0 and has to find them:

- lookahead: how much players believe the ball keeps moving. At 0 they chase where
  the ball is now; at 1 they intercept it where it will be (v1).
- aim: how the ball winner pushes the ball when not using its craft. At 0 it runs
  into the ball from wherever it is and knocks it in a random direction; at 1 it
  gets behind the ball and pushes it towards the opponent's goal.
- craft: how often the ball winner uses v1's full repertoire (lining up, shots at the
  open corner, bank shots, passes) instead of a plain push.
- keeper: above 0.5 the deepest player keeps goal; below, nobody stays back.
"""

import numpy as np

from aisoccer.tactics import skills_v1 as skills
from aisoccer.tactics.v1 import DEFAULT_STRATEGY as V1_DEFAULT
from aisoccer.tactics.v1 import TacticsV1
from aisoccer.tactics.v1 import STRATEGY as V1_STRATEGY

SKILLS = ["lookahead", "aim", "craft", "keeper"]
STRATEGY = V1_STRATEGY + SKILLS
# Defaults: every skill fully discovered, i.e. v1. Training starts them near 0 instead.
DEFAULT_STRATEGY = {**V1_DEFAULT, **{k: 1.0 for k in SKILLS}}


class TacticsV2(TacticsV1):
    """TacticsV1 with skills that start absent and are learned (see the module docstring)."""

    def __init__(self, name=None, strategy=None):
        super().__init__(name=name, strategy={**DEFAULT_STRATEGY, **(strategy or {})})
        self.crafty = False  # does this decision's ball winner use its craft?
        self.push_direction = np.array([1.0, 0.0])

    def assign(self, s, ball_path):
        plan = super().assign(s, ball_path)
        if s["keeper"] < 0.5:  # no keeper yet: the deepest player joins the formation
            keeper = next(i for i, (skill, _) in enumerate(plan) if skill == "keep")
            plan[keeper] = ("hold", self.formation(s, ball_path[0], 1)[0])
        # Fresh choices for the ball winner, kept until the next decision.
        self.crafty = self.rng.random() < s["craft"]
        angle = self.rng.uniform(0, 2 * np.pi)
        to_goal = skills.unit(skills.GOAL_CENTRE - np.asarray(self.ball_pos, float))
        wild = np.array([np.cos(angle), np.sin(angle)])
        self.push_direction = skills.unit(s["aim"] * to_goal + (1 - s["aim"]) * wild)
        return plan

    def ball_action(self, i, s, ball_path):
        if self.crafty:
            return super().ball_action(i, s, ball_path)
        # A plain push: run at the ball, aiming at the point behind it only as far as
        # the aim skill has developed.
        pos, vel = self.my_players_pos[i], self.my_players_vel[i]
        _, ball_then = skills.intercept(pos, vel, ball_path)
        target = np.asarray(ball_then) - self.push_direction * skills.CONTACT * s["aim"]
        return skills.run_through(pos, vel, target - pos)

    def do_move(self):
        # Lookahead: predict the ball with its velocity scaled by the skill, so at 0
        # the "path" is the ball standing still where it is now.
        s = self.choose_strategy()
        vel = np.asarray(self.ball_vel, float)
        original = self.ball_vel
        self.ball_vel = vel * s["lookahead"]
        try:
            return super().do_move()
        finally:
            self.ball_vel = original
