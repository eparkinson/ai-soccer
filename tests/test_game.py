import random
from unittest import TestCase

from aisoccer.brains.BehindAndTowards import BehindAndTowards
from aisoccer.brains.DefendersAndAttackers import DefendersAndAttackers
from aisoccer.brains.RandomWalk import RandomWalk
from aisoccer.game import Game, GameResult


class TestGame(TestCase):
    GAME_LENGTH = 900

    def test_randomwalk_v_randomwalk(self):
        random.seed(1)
        game = Game(RandomWalk(), RandomWalk(), game_length=self.GAME_LENGTH)

        while True:
            status = game.tick()
            if status == GameResult.end:
                break

        self.assertEqual(game.state.ticks, self.GAME_LENGTH)
        self.assertEqual(game.score['blue'], 0)
        self.assertEqual(game.score['red'], 2)

    def test_randomwalk_v_behindandtowards(self):
        random.seed(1)
        game = Game(RandomWalk(), BehindAndTowards(), game_length=self.GAME_LENGTH)

        while True:
            status = game.tick()
            if status == GameResult.end:
                break

        self.assertEqual(game.state.ticks, self.GAME_LENGTH)
        self.assertEqual(game.score['blue'], 0)
        self.assertEqual(game.score['red'], 4)

    def test_defendersandattackers_v_behindandtowards(self):
        random.seed(1)
        game = Game(DefendersAndAttackers(), BehindAndTowards(), game_length=self.GAME_LENGTH)

        while True:
            status = game.tick()
            if status == GameResult.end:
                break

        self.assertEqual(game.state.ticks, self.GAME_LENGTH)
        self.assertEqual(game.score['blue'], 1)
        self.assertEqual(game.score['red'], 0)
