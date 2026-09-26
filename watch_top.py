"""
Live match: the top two brains of the latest league round, playing each other on repeat.

    poetry run python watch_top.py

The window shows both teams' names in their colours, the score, the game clock, a goal
flash, and the running series between the two. After every game it checks the league:
when a new round changes the top two, the new pair takes over and a new series starts.
Sides swap every game.
"""

import argparse
import time

import pyglet

import leaguefiles as lf
from aisoccer.brainspec import load_brain
from aisoccer.constants import Constants
from aisoccer.game import Game, GameResult
from aisoccer.graphics.field import Field

BLUE = (110, 140, 255, 255)
RED = (255, 110, 110, 255)
WHITE = (240, 240, 240, 255)
GREY = (170, 180, 175, 255)
GOLD = (242, 193, 78, 255)


def top_two():
    """(round number, [(name, spec), (name, spec)]) from the latest league round."""
    if not (lf.LEAGUE_DIR / "progress.log").exists() or lf.latest_round() is None:
        field = lf.LEAGUE_DIR / "field"  # before round 1: the untrained round 0 brains
        return 0, [
            ("untrained-1", f"explore:{field / 'untrained-ppo-1.npz'}"),
            ("untrained-2", f"explore:{field / 'untrained-ppo-2.npz'}"),
        ]
    round_ = lf.latest_round()
    names = [row["name"] for row in round_["swiss"][:2]]
    return round_["number"], [(n, lf.brain_spec(n, round_["number"])) for n in names]


class LiveMatch(Field):
    def __init__(self, game_length):
        self.game_length = game_length
        self.series = {}  # name -> wins, plus "draws"
        self.game_number = 0
        self.flash = None  # (text, colour, until)
        self.round_number, self.pair = top_two()
        super().__init__(self.new_game())
        self.set_caption("AI Soccer League: live match of the top two")

    def new_game(self):
        (a_name, a_spec), (b_name, b_spec) = self.pair
        # Swap sides every game.
        if self.game_number % 2:
            (a_name, a_spec), (b_name, b_spec) = (b_name, b_spec), (a_name, a_spec)
        self.names = [a_name, b_name]
        self.game_number += 1
        return Game(
            load_brain(a_spec, a_name),
            load_brain(b_spec, b_name),
            game_length=self.game_length,
            quiet_mode=True,
        )

    def update(self, dt):
        for _ in range(Field.TICKS_PER_FRAME):
            if self.game_over:
                return
            result = self.game.tick()
            if result in (GameResult.goal_blue, GameResult.goal_red):
                scorer = self.names[0] if result == GameResult.goal_blue else self.names[1]
                colour = BLUE if result == GameResult.goal_blue else RED
                self.flash = (f"GOAL!  {scorer}", colour, time.time() + 1.6)
            if result == GameResult.end:
                self.finish_game()
                return

    def finish_game(self):
        self.game_over = True
        blue, red = self.game.score["blue"], self.game.score["red"]
        if blue == red:
            self.series["draws"] = self.series.get("draws", 0) + 1
            text, colour = f"Full time: draw {blue}-{red}", WHITE
        else:
            winner = self.names[0] if blue > red else self.names[1]
            self.series[winner] = self.series.get(winner, 0) + 1
            text, colour = f"Full time: {winner} wins {max(blue, red)}-{min(blue, red)}", GOLD
        self.flash = (text, colour, time.time() + 3.0)
        pyglet.clock.schedule_once(self.next_game, 3.0)

    def next_game(self, dt):
        round_number, pair = top_two()
        if [n for n, _ in pair] != [n for n, _ in self.pair]:
            self.series, self.game_number = {}, 0  # new top two: new series
        self.round_number, self.pair = round_number, pair
        self.restart_game(self.new_game())

    def on_draw(self):
        self.clear()
        self.draw_field()
        self.draw_ball()
        self.draw_players()
        self.draw_overlay()

    def draw_field(self):
        # The base field without its plain score board; the overlay draws the header.
        score = self.score
        self.score = ""
        super().draw_field()
        self.score = score

    def draw_overlay(self):
        top = Constants.FIELD_HEIGHT
        mid = Constants.FIELD_LENGTH // 2
        blue, red = self.game.score["blue"], self.game.score["red"]
        labels = [
            (self.names[0], 36, BLUE, mid - 170, top + 52, "right"),
            (f"{blue}  –  {red}", 44, WHITE, mid, top + 48, "center"),
            (self.names[1], 36, RED, mid + 170, top + 52, "left"),
        ]
        a, b = (n for n, _ in self.pair)
        series = f"Series: {a} {self.series.get(a, 0)}  –  {self.series.get(b, 0)} {b}   ({self.series.get('draws', 0)} drawn)"
        progress = self.game.game_time_complete()
        info = f"Top two of league round {self.round_number}   ·   game {self.game_number}   ·   {100 * progress:.0f}% played"
        labels += [
            (series, 16, GREY, 20, top + 30, "left"),
            (info, 16, GREY, 20, top + 8, "left"),
        ]
        for text, size, colour, x, y, anchor in labels:
            pyglet.text.Label(
                text, font_size=size, color=colour, x=x, y=y, anchor_x=anchor, weight="bold" if size > 30 else "normal"
            ).draw()
        # Clock bar.
        pyglet.shapes.Rectangle(0, top, Constants.FIELD_LENGTH, 6, color=(60, 70, 64)).draw()
        pyglet.shapes.Rectangle(0, top, Constants.FIELD_LENGTH * progress, 6, color=GOLD[:3]).draw()
        if self.flash and time.time() < self.flash[2]:
            text, colour, _ = self.flash
            pyglet.text.Label(
                text, font_size=64, color=colour, weight="bold", x=mid, y=top // 2, anchor_x="center", anchor_y="center"
            ).draw()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--game-length", type=int, default=3600, help="ticks per game")
    args = parser.parse_args()
    window = LiveMatch(args.game_length)
    pyglet.clock.schedule_interval(window.update, Field.UPDATE_FREQUENCY)
    pyglet.app.run()


if __name__ == "__main__":
    main()
