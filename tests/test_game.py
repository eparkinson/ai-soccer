import numpy as np
import pytest

from aisoccer.abstractbrain import AbstractBrain
from aisoccer.brains.BehindAndTowards import BehindAndTowards
from aisoccer.brains.DefendersAndAttackers import DefendersAndAttackers
from aisoccer.brains.RandomWalk import RandomWalk
from aisoccer.brains.SimpleBrain import SimpleBrain
from aisoccer.constants import Constants
from aisoccer.game import Game, GameResult

GAME_LENGTH = 250


class SpyBrain(AbstractBrain):
    """Records what it is told and stands still."""

    def __init__(self, name=None):
        super().__init__(name)
        self.views = []
        self.scored = []
        self.conceded = []

    def do_move(self):
        self.views.append(
            {
                "my_pos": self.my_players_pos.copy(),
                "my_vel": self.my_players_vel.copy(),
                "opp_pos": self.opp_players_pos.copy(),
                "opp_vel": self.opp_players_vel.copy(),
                "ball_pos": self.ball_pos.copy(),
            }
        )
        return np.zeros((5, 2))

    def on_goal_scored(self, team, game_state):
        self.scored.append((team, game_state))

    def on_goal_conceded(self, team, game_state):
        self.conceded.append((team, game_state))


def play_to_end(game):
    while game.tick() != GameResult.end:
        pass
    return game


@pytest.mark.parametrize(
    "blue,red",
    [
        (RandomWalk, RandomWalk),
        (RandomWalk, BehindAndTowards),
        (DefendersAndAttackers, BehindAndTowards),
        (SimpleBrain, RandomWalk),
    ],
)
def test_game_runs_for_game_length(blue, red):
    game = play_to_end(
        Game(blue(), red(), game_length=GAME_LENGTH, quiet_mode=True, seed=1)
    )
    assert game.state.ticks == GAME_LENGTH


def test_same_seed_gives_identical_games():
    def run(seed):
        game = Game(
            RandomWalk(),
            BehindAndTowards(),
            game_length=500,
            quiet_mode=True,
            seed=seed,
        )
        score = dict(game.play())
        return score, game.ball.body.position.copy(), game.teams[0].position_matrix()

    score_a, ball_a, players_a = run(42)
    score_b, ball_b, players_b = run(42)
    _, ball_c, _ = run(43)

    assert score_a == score_b
    np.testing.assert_array_equal(ball_a, ball_b)
    np.testing.assert_array_equal(players_a, players_b)
    assert not np.array_equal(ball_a, ball_c)


def test_brain_receives_opponent_positions_and_velocities():
    blue = SpyBrain()
    game = Game(blue, RandomWalk(), quiet_mode=True, seed=0)
    for _ in range(5):
        game.tick()
    red_team = game.teams[1]
    expected_pos, expected_vel = red_team.position_matrix(), red_team.velocity_matrix()

    game.tick()

    seen = blue.views[-1]
    assert np.abs(expected_vel).sum() > 0
    np.testing.assert_array_equal(seen["opp_pos"], expected_pos)
    np.testing.assert_array_equal(seen["opp_vel"], expected_vel)


def test_both_teams_see_the_same_kickoff_from_their_own_side():
    blue, red = SpyBrain(), SpyBrain()
    Game(blue, red, game_length=1, quiet_mode=True, seed=0).tick()

    np.testing.assert_allclose(blue.views[0]["my_pos"], red.views[0]["my_pos"])
    np.testing.assert_allclose(blue.views[0]["opp_pos"], red.views[0]["opp_pos"])


def place_ball(game, x, y):
    game.ball.body.position = np.array([x, y], dtype=float)
    game.ball.body.velocity = np.array([0.0, 0.0])


@pytest.mark.parametrize("scoring_team", ["red", "blue"])
def test_both_brains_are_told_about_a_goal(scoring_team):
    blue, red = SpyBrain(), SpyBrain()
    game = Game(blue, red, quiet_mode=True, seed=0)
    for _ in range(10):
        game.tick()

    x = 1 if scoring_team == "red" else Constants.FIELD_LENGTH - 2
    place_ball(game, x, Constants.FIELD_HEIGHT / 2)
    expected = GameResult.goal_red if scoring_team == "red" else GameResult.goal_blue
    assert game.tick() == expected

    scorer, conceder = (red, blue) if scoring_team == "red" else (blue, red)
    assert [t for t, _ in scorer.scored] == [scoring_team]
    assert [t for t, _ in conceder.conceded] == [scoring_team]
    assert scorer.conceded == [] and conceder.scored == []
    assert scorer.scored[0][1]["ticks_elapsed"] == 10
    assert game.score[scoring_team] == 1


def test_no_goal_outside_the_goal_mouth():
    game = Game(SpyBrain(), SpyBrain(), quiet_mode=True, seed=0)
    place_ball(game, 1, Constants.GOAL_Y_MIN - 50)
    assert not game.is_red_goal()
    place_ball(game, Constants.FIELD_LENGTH - 2, Constants.GOAL_Y_MAX + 50)
    assert not game.is_blue_goal()


def test_ball_must_fully_cross_the_line():
    game = Game(SpyBrain(), SpyBrain(), quiet_mode=True, seed=0)
    place_ball(game, Constants.GOAL_DEPTH, Constants.FIELD_HEIGHT / 2)
    assert not game.is_red_goal()
    place_ball(
        game,
        Constants.GOAL_DEPTH - Constants.BALL_RADIUS - 1,
        Constants.FIELD_HEIGHT / 2,
    )
    assert game.is_red_goal()


def test_nan_moves_are_ignored():
    class NanBrain(AbstractBrain):
        def do_move(self):
            return np.full((5, 2), np.nan)

    game = play_to_end(
        Game(NanBrain(), NanBrain(), game_length=20, quiet_mode=True, seed=0)
    )
    assert np.isfinite(game.teams[0].position_matrix()).all()


class TestGameRecording:
    GAME_LENGTH = 10

    def make_game(self, record):
        return Game(
            DefendersAndAttackers(),
            BehindAndTowards(),
            game_length=self.GAME_LENGTH,
            quiet_mode=True,
            record_game=record,
            seed=1,
        )

    def test_if_no_recording_then_df_not_initialized(self):
        assert not self.make_game(False).move_df

    def test_if_recording_then_df_initialized(self):
        assert self.make_game(True).move_df

    def test_recording_has_a_row_per_team_per_tick(self):
        game = self.make_game(True)
        game.play()

        expected_length = 2 * self.GAME_LENGTH
        for key, column in game.move_df.items():
            assert len(column) == expected_length, "length of: " + key
        assert game.move_df["team"][:2] == ["blue", "red"]

    def test_save_recording(self, tmp_path):
        game = self.make_game(True)
        game.play()
        save_file = tmp_path / "game.csv"
        game.save_game(save_file)

        loaded = Game.load_game(save_file)
        assert len(loaded["team"]) == 2 * self.GAME_LENGTH
