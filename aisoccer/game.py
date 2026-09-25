import math
from enum import Enum

import numpy as np
import pandas

from aisoccer.constants import Constants
from aisoccer.physics import Body, PhyState
from aisoccer.team import Team


class Game:
    def __init__(
        self,
        blue_brain,
        red_brain,
        game_length=Constants.GAME_LENGTH,
        quiet_mode=False,
        record_game=False,
        seed=None,
    ):
        """
        :param seed: seeds the kick-off and the brains' ``rng``. Two games with the same
            brains and the same seed play out identically. ``None`` gives a random game.
        """
        self.quiet_mode = quiet_mode
        self.game_length = game_length
        self.rng = np.random.default_rng(seed)
        self.teams = [Team(blue_brain, 0), Team(red_brain, 1)]
        self.state = PhyState(
            Constants.FIELD_LENGTH - 1,
            Constants.FIELD_HEIGHT,
            goal_depth=Constants.GOAL_DEPTH,
            goal_y_min=Constants.GOAL_Y_MIN,
            goal_y_max=Constants.GOAL_Y_MAX,
        )
        self.posts = [
            Body(Constants.POST_RADIUS, [x, y], fixed=True)
            for x in (
                Constants.GOAL_DEPTH,
                Constants.FIELD_LENGTH - 1 - Constants.GOAL_DEPTH,
            )
            for y in (Constants.GOAL_Y_MIN, Constants.GOAL_Y_MAX)
        ]
        self.ball = None
        self.move_df = {}
        self.record_game = record_game
        self.score = {"red": 0, "blue": 0}
        self.last_goal_tick = 0  # Track the tick count of the last goal

        self.seed_brains()

        if self.record_game:
            self.init_df()

        self.start()

    def seed_brains(self):
        """Give each brain its own generator derived from the game's, so seeded games are reproducible."""
        for team, child in zip(self.teams, self.rng.spawn(len(self.teams))):
            team.brain.rng = child

    def start(self):
        self.state.clear()
        self.ball = Ball(
            Constants.BALL_RADIUS,
            (Constants.FIELD_LENGTH - 1) / 2,
            Constants.FIELD_HEIGHT / 2,
        )

        for team in self.teams:
            for player in team.players:
                self.state.add_body(player.body)
            team.reset()

        for post in self.posts:
            self.state.add_body(post)

        self.state.add_body(self.ball.body)
        self.ball.body.velocity = self.rng.random(2) - 0.5

    def tick(self):
        if self.game_length != 0 and self.state.ticks >= self.game_length:
            if not self.quiet_mode:
                print("Game Over!")
            return GameResult.end
        elif self.is_red_goal():
            self.goal("red")
            return GameResult.goal_red
        elif self.is_blue_goal():
            self.goal("blue")
            return GameResult.goal_blue
        else:
            self.run_brains()
            self.limit_velocities()
            self.state.tick()
            return GameResult.nothing

    def goal(self, scoring_team):
        ticks_elapsed = self.state.ticks - self.last_goal_tick
        self.last_goal_tick = self.state.ticks
        self.score[scoring_team] += 1
        if not self.quiet_mode:
            print(f"GOAL! {scoring_team.capitalize()}!")
            print(
                "Score: Blue {:2d} / Red {:2d}     (at {:3.2f}%)".format(
                    self.score["blue"],
                    self.score["red"],
                    self.game_time_complete() * 100,
                )
            )
        self.notify_brains_goal(scoring_team, ticks_elapsed)
        self.start()

    def is_red_goal(self):
        """Red scores when the whole ball crosses blue's goal line (left) inside the goal mouth."""
        x, y = self.ball.body.position
        return x < Constants.GOAL_DEPTH - Constants.BALL_RADIUS and self.in_goal_mouth(
            y
        )

    def is_blue_goal(self):
        """Blue scores when the whole ball crosses red's goal line (right) inside the goal mouth."""
        x, y = self.ball.body.position
        goal_line = Constants.FIELD_LENGTH - 1 - Constants.GOAL_DEPTH
        return x > goal_line + Constants.BALL_RADIUS and self.in_goal_mouth(y)

    @staticmethod
    def in_goal_mouth(y):
        return Constants.GOAL_Y_MIN < y < Constants.GOAL_Y_MAX

    def game_time_complete(self):
        if self.game_length == 0:
            return 0.0  # Infinite game, no completion percentage
        return float(self.state.ticks) / float(self.game_length)

    def run_brains(self):
        blue_team = self.teams[0]
        red_team = self.teams[1]

        blue_players_pos = blue_team.position_matrix()
        blue_players_vel = blue_team.velocity_matrix()

        red_players_pos = red_team.position_matrix()
        red_players_vel = red_team.velocity_matrix()

        ball_pos = self.ball.body.position.copy()
        ball_vel = self.ball.body.velocity.copy()

        blue_score = self.score["blue"]
        red_score = self.score["red"]

        game_time = self.game_time_complete()

        blue_view = (
            blue_players_pos,
            blue_players_vel,
            red_players_pos,
            red_players_vel,
            ball_pos,
            ball_vel,
            blue_score,
            red_score,
            game_time,
        )
        # Red sees the field mirrored so that, like blue, it attacks from left to right.
        red_view = (
            flip_pos(red_players_pos),
            flip_vel(red_players_vel),
            flip_pos(blue_players_pos),
            flip_vel(blue_players_vel),
            flip_pos(ball_pos),
            flip_vel(ball_vel),
            red_score,
            blue_score,
            game_time,
        )

        blue_move = blue_team.apply_move(blue_team.brain.move(*blue_view))
        red_move = red_team.apply_move(flip_acc(red_team.brain.move(*red_view)))

        if self.record_game:
            self.record_move("blue", *blue_view, blue_move)
            self.record_move("red", *red_view, flip_acc(red_move))

    def limit_velocities(self):
        clamp_speed(self.ball.body, Constants.MAX_BALL_VELOCITY)
        for t in self.teams:
            for p in t.players:
                clamp_speed(p.body, Constants.MAX_PLAYER_VELOCITY)

    def play(self):
        while True:
            status = self.tick()
            if status == GameResult.end:
                break

        return self.score

    def record_move(
        self,
        team,
        my_players_pos,
        my_players_vel,
        opp_players_pos,
        opp_players_vel,
        ball_pos,
        ball_vel,
        my_score,
        opp_score,
        game_time,
        moves,
    ):
        """
        Record one row per team per tick, from that team's point of view (red rows are
        mirrored exactly as the red brain saw them). ``moves`` are the capped accelerations.
        """
        self.move_df["tick"].append(self.state.ticks)
        self.move_df["team"].append(team)
        self.move_df["bp_x"].append(ball_pos[0])
        self.move_df["bp_y"].append(ball_pos[1])
        self.move_df["bv_x"].append(ball_vel[0])
        self.move_df["bv_y"].append(ball_vel[1])
        self.move_df["ms"].append(my_score)
        self.move_df["os"].append(opp_score)
        self.move_df["gt"].append(game_time)

        for i in range(Constants.NUM_PLAYERS):
            self.move_df["mpp_" + str(i) + "_x"].append(my_players_pos[i][0])
            self.move_df["mpp_" + str(i) + "_y"].append(my_players_pos[i][1])

            self.move_df["mpv_" + str(i) + "_x"].append(my_players_vel[i][0])
            self.move_df["mpv_" + str(i) + "_y"].append(my_players_vel[i][1])

            self.move_df["opp_" + str(i) + "_x"].append(opp_players_pos[i][0])
            self.move_df["opp_" + str(i) + "_y"].append(opp_players_pos[i][1])

            self.move_df["opv_" + str(i) + "_x"].append(opp_players_vel[i][0])
            self.move_df["opv_" + str(i) + "_y"].append(opp_players_vel[i][1])

            self.move_df["m_" + str(i) + "_x"].append(moves[i][0])
            self.move_df["m_" + str(i) + "_y"].append(moves[i][1])

    def init_df(self):
        df: dict[str, list] = {
            "tick": [],
            "team": [],
            "bp_x": [],
            "bp_y": [],
            "bv_x": [],
            "bv_y": [],
            "ms": [],
            "os": [],
            "gt": [],
        }

        for i in range(Constants.NUM_PLAYERS):
            df["mpp_" + str(i) + "_x"] = []
            df["mpp_" + str(i) + "_y"] = []

            df["mpv_" + str(i) + "_x"] = []
            df["mpv_" + str(i) + "_y"] = []

            df["opp_" + str(i) + "_x"] = []
            df["opp_" + str(i) + "_y"] = []

            df["opv_" + str(i) + "_x"] = []
            df["opv_" + str(i) + "_y"] = []

            df["m_" + str(i) + "_x"] = []
            df["m_" + str(i) + "_y"] = []

        self.move_df = df

    def save_game(self, save_file):
        pandas_df = pandas.DataFrame(self.move_df)
        pandas_df.to_csv(save_file, index_label="row")

    @staticmethod
    def load_game(filename):
        panda_df = pandas.read_csv(filename)
        return panda_df.to_dict()

    def notify_brains_goal(self, scoring_team, ticks_elapsed):
        """Tell the scoring brain it scored and the other brain it conceded."""
        game_state = {"ticks_elapsed": ticks_elapsed}
        for team in self.teams:
            if team.side == scoring_team:
                team.brain.on_goal_scored(scoring_team, game_state)
            else:
                team.brain.on_goal_conceded(scoring_team, game_state)


class Ball:
    def __init__(self, radius, x, y):
        self.body = Body(radius, [x, y])


class GameResult(Enum):
    nothing = 0
    goal_red = 1
    goal_blue = 2
    end = 3


def clamp_speed(body, max_speed):
    vx, vy = body.velocity
    speed = math.hypot(vx, vy)
    if speed > max_speed:
        body.velocity = body.velocity * (max_speed / speed)


def flip_pos(positions):
    """Mirror positions left to right: x -> FIELD_LENGTH - 1 - x."""
    result = np.array(positions, dtype=float)
    result[..., 0] = Constants.FIELD_LENGTH - 1 - result[..., 0]
    return result


def flip_vel(velocities):
    """Mirror velocities left to right: vx -> -vx."""
    result = np.array(velocities, dtype=float)
    result[..., 0] = -result[..., 0]
    return result


def flip_acc(accelerations):
    """Mirror accelerations left to right: ax -> -ax."""
    return flip_vel(accelerations)
