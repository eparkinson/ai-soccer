"""
A Game that also adds up each side's expected goals (xG), for evolve_gp.py's optional
--xg-weight tiebreak: a denser, lower-variance signal than goals, which break 0-0
plateaus only rarely.

A shot is a touch (the ball's velocity changes other than by a pure wall reflection:
a player or a post hit it) that leaves the ball on target (PPOBrain.shot_features:
heading for a goal line and crossing it inside the mouth, directly or off one side
wall). Its xG is the league's xG model's probability (fit_xg.py, e.g.
runs/league3/xg.json; PPOBrain.xg_probability) that it goes in before the next touch.
The sum over a game's shots is that side's xG, which estimates the goals it "should"
have scored from them. Scores are the Game's own: this only watches.
"""

from aisoccer.brains.PPOBrain import L, chance_values, load_xg_model
from aisoccer.game import Game, GameResult

_MODELS = {}


def xg_model(path):
    """The xG model at path (cached per process), or None for the hand-made chance."""
    if path not in _MODELS:
        _MODELS[path] = load_xg_model(path)
    return _MODELS[path]


def is_touch(before, after):
    """The ball's velocity changed other than by a wall reflection (components negated)."""
    if before[0] == after[0] and before[1] == after[1]:
        return False
    return not (abs(before[0]) == abs(after[0]) and abs(before[1]) == abs(after[1]))


class XGGame(Game):
    """A Game that also records xG for blue and red (self.xg = {"blue": .., "red": ..})."""

    def __init__(self, *args, xg_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = xg_model
        self.xg = {"blue": 0.0, "red": 0.0}
        self.shots = {"blue": 0, "red": 0}

    def note_shot(self):
        ball = self.ball.body.position.reshape(1, 2).astype(float)
        vel = self.ball.body.velocity.reshape(1, 2).astype(float)
        blue = self.teams[0].position_matrix()[None].astype(float)
        red = self.teams[1].position_matrix()[None].astype(float)
        # blue attacks the goal on the right; red's view is mirrored so that it does too
        chance = float(chance_values(ball, vel, red, self.model)[0])
        if chance > 0.0:
            self.xg["blue"] += chance
            self.shots["blue"] += 1
            return
        ball_m, vel_m, blue_m = ball.copy(), vel.copy(), blue.copy()
        ball_m[:, 0] = L - 1 - ball_m[:, 0]
        vel_m[:, 0] = -vel_m[:, 0]
        blue_m[..., 0] = L - 1 - blue_m[..., 0]
        chance = float(chance_values(ball_m, vel_m, blue_m, self.model)[0])
        if chance > 0.0:
            self.xg["red"] += chance
            self.shots["red"] += 1

    def play(self):
        before = self.ball.body.velocity.tolist()
        while True:
            status = self.tick()
            if status == GameResult.end:
                break
            after = self.ball.body.velocity.tolist()
            if status == GameResult.nothing and is_touch(before, after):
                self.note_shot()
            before = after
        return self.score


def play_xg(blue, red, ticks, seed, model_path):
    """(score, xg) of one quiet game: score {"blue", "red"} goals, xg {"blue", "red"}."""
    game = XGGame(blue, red, game_length=ticks, quiet_mode=True, seed=seed, xg_model=xg_model(model_path))
    score = game.play()
    return score, dict(game.xg)
