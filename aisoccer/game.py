from enum import Enum
from random import random

import pandas

from aisoccer.team import *


class Game:
    def __init__(
        self,
        blue_brain,
        red_brain,
        game_length=Constants.GAME_LENGTH,
        quiet_mode=False,
        record_game=False,
    ):
        self.quiet_mode = quiet_mode
        self.game_length = game_length
        self.teams = [Team(blue_brain, 0), Team(red_brain, 1)]
        self.state = PhyState(Constants.FIELD_LENGTH, Constants.FIELD_HEIGHT)
        self.ball = None
        self.move_df = {}
        self.record_game = record_game
        self.score = {"red": 0, "blue": 0}
        self.last_goal_tick = 0  # Track the tick count of the last goal

        if self.record_game:
            self.init_df()

        self.start()

    def start(self):
        self.state.clear()
        self.ball = Ball(
            Constants.BALL_RADIUS,
            Constants.FIELD_LENGTH / 2,
            Constants.FIELD_HEIGHT / 2,
        )

        for team in self.teams:
            for player in team.players:
                self.state.add_body(player.body)
            team.reset()

        self.state.add_body(self.ball.body)
        self.ball.body.velocity = np.array([random() - 0.5, random() - 0.5])

    def tick(self):
        if self.game_length != 0 and self.state.ticks >= self.game_length:
            if not self.quiet_mode:
                print("Game Over!")
            return GameResult.end
        elif self.is_red_goal():
            ticks_elapsed = self.state.ticks - self.last_goal_tick
            self.last_goal_tick = self.state.ticks
            self.score["red"] += 1
            if not self.quiet_mode:
                print("GOAL! Red!")
                print(
                    "Score: Blue {0:2d} / Red {1:2d}     (at {2:3.2f}%)".format(
                        self.score["blue"],
                        self.score["red"],
                        self.game_time_complete() * 100,
                    )
                )
            self.notify_brains_goal("red", ticks_elapsed)
            self.start()
            return GameResult.goal_red
        elif self.is_blue_goal():
            ticks_elapsed = self.state.ticks - self.last_goal_tick
            self.last_goal_tick = self.state.ticks
            self.score["blue"] += 1
            if not self.quiet_mode:
                print("GOAL! Blue!")
                print(
                    "Score: Blue {0:2d} / Red {1:2d}     (at {2:3.2f}%)".format(
                        self.score["blue"],
                        self.score["red"],
                        self.game_time_complete() * 100,
                    )
                )
            self.notify_brains_goal("blue", ticks_elapsed)
            self.start()
            return GameResult.goal_blue
        else:
            self.run_brains()
            self.limit_velocities()
            self.state.tick()
            return GameResult.nothing

    def is_red_goal(self):
        return self.ball.body.position[0] < Constants.GOAL_WIDTH + Constants.BALL_RADIUS

    def is_blue_goal(self):
        return self.ball.body.position[0] > (Constants.FIELD_LENGTH - 1) - (
            Constants.GOAL_WIDTH + Constants.BALL_RADIUS
        )

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

        ball_pos = self.ball.body.position
        ball_vel = self.ball.body.velocity

        blue_score = self.score["blue"]
        red_score = self.score["red"]

        game_time = self.game_time_complete()

        blue_brain = blue_team.brain
        red_brain = red_team.brain

        blue_move = blue_brain.move(
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
        blue_team.apply_move(blue_move)

        if self.record_game:
            self.record_move(
                blue_players_pos,
                blue_players_vel,
                red_players_pos,
                red_players_vel,
                ball_pos,
                ball_vel,
                blue_score,
                red_score,
                game_time,
                blue_brain.last_move,
            )

        # TODO: translate red positions and velocities
        #       so that both brains think that they are playing from left (0,y) to right (MAX_X,y)
        red_move = red_brain.move(
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
        red_move = flip_acc(red_move)
        red_team.apply_move(red_move)

    def limit_velocities(self):
        ball_velocity = self.ball.body.normal_velocity()
        if ball_velocity > Constants.MAX_BALL_VELOCITY:
            self.ball.body.velocity = np.multiply(
                self.ball.body.velocity, Constants.MAX_BALL_VELOCITY / ball_velocity
            )

        for t in self.teams:
            for p in t.players:
                player_velocity = p.body.normal_velocity()
                if player_velocity > Constants.MAX_PLAYER_VELOCITY:
                    p.body.velocity = np.multiply(
                        p.body.velocity, Constants.MAX_PLAYER_VELOCITY / player_velocity
                    )

    def play(self):
        while True:
            status = self.tick()
            if status == GameResult.end:
                break

        return self.score

    def record_move(
        self,
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
        self.move_df["bp_x"].append(ball_pos[0])
        self.move_df["bp_y"].append(ball_pos[1])
        self.move_df["bv_x"].append(ball_vel[0])
        self.move_df["bv_y"].append(ball_vel[1])
        self.move_df["ms"].append(my_score)
        self.move_df["os"].append(opp_score)
        self.move_df["gt"].append(game_time)

        for i in range(5):
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
        df = {
            "bp_x": [],
            "bp_y": [],
            "bv_x": [],
            "bv_y": [],
            "ms": [],
            "os": [],
            "gt": [],
        }

        for i in range(5):
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
        game_state = {
            "ticks_elapsed": ticks_elapsed,
            # ...other game state details...
        }
        for team in self.teams:
            if not self.quiet_mode:
                print(
                    f"Team side: {team.side}, Scoring team: {scoring_team}"
                )  # Debugging log
            if scoring_team == team.side:  # Correctly match scoring team with team side
                reward = (1.0 / ticks_elapsed) * 1000  # Scale reward
                team.brain.on_goal_scored(scoring_team, game_state)
                if not self.quiet_mode:
                    print(
                        f"Reward received: {reward:.2f} (Scoring team: {scoring_team})"
                    )
                return  # Exit after processing the scoring team
            else:
                penalty = (-1.0 / ticks_elapsed) * 1000  # Scale penalty
                team.brain.on_goal_conceded(scoring_team, game_state)
                if not self.quiet_mode:
                    print(
                        f"Penalty received: {penalty:.2f} (Scoring team: {scoring_team})"
                    )
                break  # Ensure only one penalty is printed


class Ball:
    def __init__(self, radius, x, y):
        self.body = Body(radius, [x, y])


class GameResult(Enum):
    nothing = 0
    goal_red = 1
    goal_blue = 2
    end = 3


def flip_pos(positions):
    result = positions.copy()

    if positions.ndim == 2:
        for i in range(len(positions)):
            result[i][0] = Constants.FIELD_LENGTH - 1 - positions[i][0]
    elif positions.ndim == 1:
        result[0] = Constants.FIELD_LENGTH - 1 - positions[0]

    return result


def flip_vel(velocities):
    result = velocities.copy()

    if velocities.ndim == 2:
        for i in range(len(velocities)):
            result[i][0] = -1 * velocities[i][0]
    elif velocities.ndim == 1:
        result[0] = -1 * velocities[0]

    return result


def flip_acc(accellerations):
    result = accellerations.copy()

    if accellerations.ndim == 2:
        for i in range(len(accellerations)):
            result[i][0] = -1 * accellerations[i][0]

    return result
