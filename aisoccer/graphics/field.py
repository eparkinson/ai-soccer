import pyglet

from aisoccer.constants import Constants
from aisoccer.game import GameResult


class Field(pyglet.window.Window):
    # Constants relating to visual display of games go here.
    UPDATE_FREQUENCY = 0.02
    # Simulation ticks per rendered frame. Raising this speeds up playback
    # without changing the physics step.
    TICKS_PER_FRAME = 2
    BALL_COLOUR = (255, 255, 255)
    FIELD_COLOUR = (96, 96, 96)
    TEAM_COLOURS = [(64, 64, 255), (255, 64, 64)]
    BACKGROUND_COLOUR = (4, 16, 4)
    RED_GOAL_COLOUR = (64, 0, 0)
    BLUE_GOAL_COLOUR = (0, 0, 64)
    POST_COLOUR = (255, 255, 255)
    STATUS_BAR_HEIGHT = 100

    def __init__(self, game):
        super().__init__(
            Constants.FIELD_LENGTH,
            height=Constants.FIELD_HEIGHT + Field.STATUS_BAR_HEIGHT,
        )
        self.game = game
        self.game_over = False

        self.score = "0 - 0"

    def on_draw(self):
        self.clear()

        self.draw_field()
        self.draw_ball()
        self.draw_players()

    def start_game(self):
        pyglet.clock.schedule_interval(self.update, Field.UPDATE_FREQUENCY)
        pyglet.app.run()

    def restart_game(self, game):
        self.game_over = False
        self.game = game

    def update(self, dt):
        for _ in range(Field.TICKS_PER_FRAME):
            if self.game_over:
                return

            result = self.game.tick()

            self.score = "{} - {}".format(
                self.game.score["blue"], self.game.score["red"]
            )

            if result == GameResult.end:
                self.game_over = True
                pyglet.app.exit()
                self.close()

    def draw_field(self):
        field = pyglet.shapes.Rectangle(
            0,
            0,
            Constants.FIELD_LENGTH - 1,
            Constants.FIELD_HEIGHT - 1,
            Field.BACKGROUND_COLOUR,
        )
        field.draw()

        # Goals: the shaded box behind each goal mouth, plus the goal line either side of it
        mouth_height = Constants.GOAL_Y_MAX - Constants.GOAL_Y_MIN
        right_line = Constants.FIELD_LENGTH - 1 - Constants.GOAL_DEPTH
        pyglet.shapes.Rectangle(
            0,
            Constants.GOAL_Y_MIN,
            Constants.GOAL_DEPTH,
            mouth_height,
            Field.BLUE_GOAL_COLOUR,
        ).draw()
        pyglet.shapes.Rectangle(
            right_line,
            Constants.GOAL_Y_MIN,
            Constants.GOAL_DEPTH,
            mouth_height,
            Field.RED_GOAL_COLOUR,
        ).draw()

        for x in (Constants.GOAL_DEPTH, right_line):
            for y_from, y_to in (
                (0, Constants.GOAL_Y_MIN),
                (Constants.GOAL_Y_MAX, Constants.FIELD_HEIGHT),
            ):
                pyglet.shapes.Line(
                    x, y_from, x, y_to, thickness=2, color=Field.FIELD_COLOUR
                ).draw()

        for post in self.game.posts:
            pyglet.shapes.Circle(
                post.position[0], post.position[1], post.radius, color=Field.POST_COLOUR
            ).draw()

        center_dot = pyglet.shapes.Circle(
            Constants.FIELD_LENGTH / 2,
            Constants.FIELD_HEIGHT / 2,
            6,
            color=Field.FIELD_COLOUR,
        )
        center_dot.draw()

        center_line = pyglet.shapes.Line(
            Constants.FIELD_LENGTH / 2,
            0,
            Constants.FIELD_LENGTH / 2,
            Constants.FIELD_HEIGHT,
            color=Field.FIELD_COLOUR,
        )
        # If you want to set the thickness, set center_line.thickness = 4 (if supported by your pyglet version)
        if hasattr(center_line, "thickness"):
            center_line.thickness = 4
        center_line.draw()

        score_board = pyglet.text.Label(
            self.score,
            color=(255, 255, 255, 255),
            font_size=50,
            x=Constants.FIELD_LENGTH // 2,
            y=Constants.FIELD_HEIGHT + 20,
            anchor_x="center",
        )

        score_board.draw()

    def draw_ball(self):
        ball = self.game.ball.body
        circle = pyglet.shapes.Circle(
            ball.position[0], ball.position[1], ball.radius, color=Field.BALL_COLOUR
        )
        circle.draw()

    def draw_players(self):
        for t in self.game.teams:
            for p in t.players:
                circle = pyglet.shapes.Circle(
                    p.body.position[0],
                    p.body.position[1],
                    Constants.PLAYER_RADIUS,
                    color=Field.TEAM_COLOURS[p.side],
                )
                circle.draw()
