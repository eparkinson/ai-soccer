from unittest import TestCase

from aisoccer.brains.BehindAndTowards import BehindAndTowards
from aisoccer.brains.DefendersAndAttackers import DefendersAndAttackers
from aisoccer.brains.RandomWalk import RandomWalk
from aisoccer.tournament import Tournament


class TestTournament(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.round_robin_tourney = Tournament(
            [
                BehindAndTowards("anne"),
                DefendersAndAttackers("bob"),
                RandomWalk("charlie"),
            ],
            game_length=1000,
            rounds=0,
        )
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
        self.round_robin_tourney.get_table()
        previous_points = 99999

        for score in self.round_robin_tourney.get_table():
            self.assertTrue(score["points"] <= previous_points)
            previous_points = score["points"]

    def test_get_scores(self):
        scores = self.round_robin_tourney.get_scores()
        self.assertIsInstance(scores, list)
        self.assertGreater(len(scores), 0)
        for score in scores:
            self.assertIn("name", score)
            self.assertIn("points", score)

    def test_grouped_scores(self):
        scores = self.round_robin_tourney.get_scores()
        grouped_scores = {}
        for score in scores:
            brain_type = score["name"].split("-")[0]
            if brain_type not in grouped_scores:
                grouped_scores[brain_type] = {
                    "P": 0,
                    "W": 0,
                    "L": 0,
                    "GF": 0,
                    "GA": 0,
                    "GD": 0,
                    "POINTS": 0,
                }
            grouped_scores[brain_type]["P"] += score["played"]
            grouped_scores[brain_type]["W"] += score["wins"]
            grouped_scores[brain_type]["L"] += score["losses"]
            grouped_scores[brain_type]["GF"] += score["goals_for"]
            grouped_scores[brain_type]["GA"] += score["goals_against"]
            grouped_scores[brain_type]["GD"] += score["goal_diff"]
            grouped_scores[brain_type]["POINTS"] += score["points"]
        sorted_grouped_scores = sorted(
            grouped_scores.items(), key=lambda x: x[1]["POINTS"], reverse=True
        )
        self.assertGreater(len(sorted_grouped_scores), 0)
