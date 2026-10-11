"""
Per-game team statistics: possession, passing, territory, shots, shape and effort.

Use ``play_with_stats(game)`` in place of ``game.play()``. Every statistic is from the
team's own point of view (attacking left to right), so blue's and red's numbers are
directly comparable.
"""

import numpy as np

from aisoccer.constants import Constants
from aisoccer.game import GameResult

L = Constants.FIELD_LENGTH
TOUCH = Constants.PLAYER_RADIUS + Constants.BALL_RADIUS + 4  # ball within reach
CONTEST = TOUCH + 10  # an opponent this close means the ball is contested
PASS_DISTANCE = 60  # pixels the ball must travel between teammates to count as a pass
FORWARD_PASS = 50  # pixels of progress for a pass to count as forward
SHOT_RANGE = 700  # pixels from the goal line
SHOT_SPEED = 4.0

TEAMS = ("blue", "red")

STAT_NAMES = [
    "possession",  # share of ticks this team has the ball within reach
    "control",  # share of ticks this team's nearest player is closer to the ball
    "turnovers_won",  # clear possession won from the opponent
    "passes",  # ball travels between teammates with no opponent touch
    "forward_passes",
    "own_half",  # share of ticks the ball is in the opponent's half
    "final_third",  # share of ticks the ball is in the opponent's final third
    "shots",  # ball heading into the opponent's goal mouth at speed, in range
    "spread",  # mean distance between teammates (pixels)
    "deepest",  # deepest player's distance from own goal line (pixels)
    "centroid",  # team's mean distance from own goal line (pixels)
    "distance_run",  # total player movement (pixels)
    "touches",
]


class GameStats:
    def __init__(self):
        self.ticks = 0
        self.totals = {team: {name: 0.0 for name in STAT_NAMES} for team in TEAMS}
        self.touches_by_role = {team: np.zeros(Constants.NUM_PLAYERS) for team in TEAMS}
        self.reset_play()

    def reset_play(self):
        """Forget who had the ball, e.g. after a goal and kick-off."""
        self.holder = None  # (team index, player index) of the last touch
        self.holder_ball = np.zeros(2)  # ball position (holder's side) at that touch
        self.shooting = [False, False]

    def update(self, game):
        self.ticks += 1
        ball = game.ball.body.position
        ball_vel = game.ball.body.velocity
        positions = [team.position_matrix() for team in game.teams]
        velocities = [team.velocity_matrix() for team in game.teams]

        dist = [np.linalg.norm(p - ball, axis=1) for p in positions]
        nearest = [d.min() for d in dist]

        for t, team in enumerate(TEAMS):
            own_x = self.own_x(t, positions[t][:, 0])
            ball_x = self.own_x(t, ball[0])
            s = self.totals[team]
            s["control"] += nearest[t] < nearest[1 - t]
            s["own_half"] += ball_x > L / 2
            s["final_third"] += ball_x > 2 * L / 3
            diffs = positions[t][:, None, :] - positions[t][None, :, :]
            pair = np.sqrt((diffs**2).sum(axis=2))
            s["spread"] += pair.sum() / (len(pair) * (len(pair) - 1))
            s["deepest"] += own_x.min()
            s["centroid"] += own_x.mean()
            s["distance_run"] += np.linalg.norm(velocities[t], axis=1).sum()
            self.update_shots(t, ball, ball_vel)

        # Possession: a player has the ball within reach and no opponent is contesting
        # it. Contested balls do not change hands, so scrambles are not counted as a
        # stream of turnovers.
        t = 0 if nearest[0] < nearest[1] else 1
        if nearest[t] <= TOUCH and nearest[1 - t] > CONTEST:
            player = int(dist[t].argmin())
            self.touch(t, player, np.array([self.own_x(t, ball[0]), ball[1]]))

    def touch(self, t, player, ball):
        team = TEAMS[t]
        self.totals[team]["possession"] += 1
        if self.holder != (t, player):
            travelled = np.linalg.norm(ball - self.holder_ball)
            if self.holder is None or self.holder[0] != t:
                if self.holder is not None:
                    self.totals[team]["turnovers_won"] += 1
            elif travelled < PASS_DISTANCE:
                return  # teammates crowding the ball, not a pass
            else:
                self.totals[team]["passes"] += 1
                if ball[0] - self.holder_ball[0] > FORWARD_PASS:
                    self.totals[team]["forward_passes"] += 1
            self.totals[team]["touches"] += 1
            self.touches_by_role[team][player] += 1
            self.holder = (t, player)
        self.holder_ball = ball

    def update_shots(self, t, ball, ball_vel):
        """Count each time the ball starts heading into team t's target goal mouth."""
        ball_x = self.own_x(t, ball[0])
        vx = ball_vel[0] if t == 0 else -ball_vel[0]
        goal_line = L - 1 - Constants.GOAL_DEPTH
        on_target = False
        if (
            vx > 0
            and np.hypot(*ball_vel) >= SHOT_SPEED
            and goal_line - ball_x < SHOT_RANGE
        ):
            y_at_line = ball[1] + ball_vel[1] * (goal_line - ball_x) / vx
            on_target = Constants.GOAL_Y_MIN < y_at_line < Constants.GOAL_Y_MAX
        if on_target and not self.shooting[t]:
            self.totals[TEAMS[t]]["shots"] += 1
        self.shooting[t] = on_target

    @staticmethod
    def own_x(t, x):
        """x measured from team t's own goal line side."""
        return x if t == 0 else L - 1 - np.asarray(x)

    def summary(self):
        """Per-team statistics for the whole game (shares and means per tick)."""
        per_tick = {"possession", "control", "own_half", "final_third"}
        mean = {"spread", "deepest", "centroid"}
        n = max(self.ticks, 1)
        out = {}
        for team in TEAMS:
            s = dict(self.totals[team])
            for name in per_tick | mean:
                s[name] /= n
            out[team] = s
        return out


def play_with_stats(game):
    """Play a whole game like ``game.play()``, returning (score, per-team statistics)."""
    stats = GameStats()
    while True:
        result = game.tick()
        if result == GameResult.end:
            break
        if result in (GameResult.goal_red, GameResult.goal_blue):
            stats.reset_play()
            continue
        stats.update(game)
    return game.score, stats.summary()
