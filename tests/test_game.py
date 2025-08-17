import random
from unittest import TestCase

from aisoccer.brains.BehindAndTowards import BehindAndTowards
from aisoccer.brains.DefendersAndAttackers import DefendersAndAttackers
from aisoccer.brains.RandomWalk import RandomWalk
from aisoccer.game import Game, GameResult


class TestGame(TestCase):
    GAME_LENGTH = 250

    def test_randomwalk_v_randomwalk(self):
        random.seed(1)
        game = Game(RandomWalk(), RandomWalk(), game_length=self.GAME_LENGTH)

        while True:
            status = game.tick()
            if status == GameResult.end:
                break

        self.assertEqual(self.GAME_LENGTH, game.state.ticks)
        self.assertEqual(0, game.score["blue"])
        self.assertEqual(0, game.score["red"])

    def test_randomwalk_v_behindandtowards(self):
        random.seed(1)
        game = Game(RandomWalk(), BehindAndTowards(), game_length=self.GAME_LENGTH)

        while True:
            status = game.tick()
            if status == GameResult.end:
                break

        self.assertEqual(self.GAME_LENGTH, game.state.ticks)

    def test_defendersandattackers_v_behindandtowards(self):
        random.seed(1)
        game = Game(
            DefendersAndAttackers(), BehindAndTowards(), game_length=self.GAME_LENGTH
        )

        while True:
            status = game.tick()
            if status == GameResult.end:
                break

        self.assertEqual(game.state.ticks, self.GAME_LENGTH)

    def test_brain_compatibility(self):
        from aisoccer.brains.SimpleBrain import SimpleBrain

        brains = [
            RandomWalk(),
            BehindAndTowards(),
            DefendersAndAttackers(),
            SimpleBrain(),
        ]
        for brain in brains:
            game = Game(brain, RandomWalk(), game_length=self.GAME_LENGTH)
            while True:
                status = game.tick()
                if status == GameResult.end:
                    break
            self.assertEqual(self.GAME_LENGTH, game.state.ticks)


class TestGameRecording(TestCase):
    GAME_LENGTH = 10

    def test_if_no_recording_then_df_not_initialized(self):
        random.seed(1)
        game = Game(
            DefendersAndAttackers(), BehindAndTowards(), game_length=self.GAME_LENGTH
        )
        self.assertFalse(game.move_df)

    def test_if_recording_then_df_initialized(self):
        random.seed(1)
        game = Game(
            DefendersAndAttackers(),
            BehindAndTowards(),
            game_length=self.GAME_LENGTH,
            record_game=True,
        )
        self.assertTrue(game.move_df)

    def test_recording_format(self):
        random.seed(1)
        game = Game(
            DefendersAndAttackers(),
            BehindAndTowards(),
            game_length=self.GAME_LENGTH,
            record_game=True,
        )

        score = game.play()

        self.assertTrue(game.move_df)

        expected_length = len(game.move_df["m_0_x"])

        for key in game.move_df:
            self.assertEqual(
                len(game.move_df[key]), expected_length, "length of: " + key
            )

    def test_save_recording(self):
        random.seed(1)
        game = Game(
            DefendersAndAttackers(),
            BehindAndTowards(),
            game_length=self.GAME_LENGTH,
            record_game=True,
        )
        score = game.play()
        game.save_game("test.csv")
