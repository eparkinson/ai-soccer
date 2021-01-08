from unittest import TestCase

from aisoccer.brains.BehindAndTowards import BehindAndTowards
from aisoccer.brains.DefendersAndAttackers import DefendersAndAttackers
from aisoccer.brains.RandomWalk import RandomWalk
from aisoccer.tournament import Tournament


class TestTournament(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.round_robin_tourney = Tournament([BehindAndTowards("anne"),
                                              DefendersAndAttackers("bob"),
                                              RandomWalk("charlie")]
                                             , game_length=1000, rounds=0)
        cls.round_robin_tourney.start()

        swiss_brains = []
        for i in range(6):
            swiss_brains.append(BehindAndTowards("BAT-" + str(i)))

        for i in range(6):
            swiss_brains.append(DefendersAndAttackers("DAA-" + str(i)))

        for i in range(6):
            swiss_brains.append(RandomWalk("RW-" + str(i)))

        cls.swiss_tourney = Tournament(swiss_brains, game_length=500, rounds=3)
        cls.swiss_tourney.start()

    def test_round_robin(self):
        for score in self.round_robin_tourney.get_table():
            self.assertEqual(2, score["played"])

    def test_swiss(self):
        for score in self.swiss_tourney.get_table():
            self.assertEqual(3, score["played"])

    def test_sorted_table(self):
        table = self.round_robin_tourney.get_table()
        previous_points = 99999

        for score in self.round_robin_tourney.get_table():
            self.assertTrue(score["points"] <= previous_points)
            previous_points = score["points"]
