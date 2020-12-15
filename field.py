import pyglet
import team
from game import *


class Field(pyglet.window.Window):
    UPDATE_FREQUENCY = 0.05
    BALL_COLOUR = (255, 255, 255)
    FIELD_COLOUR = (96, 96, 96)
    TEAM_COLOURS = [(0, 0, 255), (255, 0, 0)]
    BACKGROUND_COLOUR = (4, 16, 4)
    RED_GOAL_COLOUR = (64, 0, 0)
    BLUE_GOAL_COLOUR = (0, 0, 64)
    STATUS_BAR_HEIGHT = 100

    def __init__(self, game):
        super(Field, self).__init__(game.FIELD_LENGTH, height=game.FIELD_HEIGHT + Field.STATUS_BAR_HEIGHT)
        self.game = game
        self.game_over = False

    def on_draw(self):
        self.clear()

        self.draw_field()
        self.draw_ball()
        self.draw_players()

    def start_game(self):
        pyglet.clock.schedule_interval(self.update, Field.UPDATE_FREQUENCY)
        pyglet.app.run()

    def update(self, dt):
        if not self.game_over:
            result = self.game.tick()
            if result == GameResult.end:
                self.game_over = True


    def draw_field(self):
        field = pyglet.shapes.Rectangle(0, 0, self.game.FIELD_LENGTH - 1, self.game.FIELD_HEIGHT - 1,
                                        Field.BACKGROUND_COLOUR)
        field.draw()

        blue_goal = pyglet.shapes.Rectangle(0, 0, 20, self.game.FIELD_HEIGHT - 1, Field.BLUE_GOAL_COLOUR)
        blue_goal.draw()

        red_goal = pyglet.shapes.Rectangle(self.game.FIELD_LENGTH - 21, 0,
                                           self.game.FIELD_LENGTH - 1, self.game.FIELD_HEIGHT - 1,
                                           Field.RED_GOAL_COLOUR)
        red_goal.draw()

        center_dot = pyglet.shapes.Circle(self.game.FIELD_LENGTH / 2, self.game.FIELD_HEIGHT / 2, 6,
                                          color=Field.FIELD_COLOUR)
        center_dot.draw()

        center_line = pyglet.shapes.Line(self.game.FIELD_LENGTH / 2, 0,
                                         self.game.FIELD_LENGTH / 2, self.game.FIELD_HEIGHT,
                                         width=4, color=Field.FIELD_COLOUR)
        center_line.draw()

    def draw_ball(self):
        ball = self.game.ball.body
        circle = pyglet.shapes.Circle(ball.position[0], ball.position[1], ball.radius, color=Field.BALL_COLOUR)
        circle.draw()

    def draw_players(self):
        for t in self.game.teams:
            for p in t.players:
                circle = pyglet.shapes.Circle(p.body.position[0],
                                              p.body.position[1],
                                              team.Player.RADIUS,
                                              color=Field.TEAM_COLOURS[p.side])
                circle.draw()
