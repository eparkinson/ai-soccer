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
            seed=1,
        )
        cls.round_robin_tourney.start()

        swiss_brains = []
        for i in range(6):
            swiss_brains.append(BehindAndTowards("BAT-" + str(i)))

        for i in range(6):
            swiss_brains.append(DefendersAndAttackers("DAA-" + str(i)))

        for i in range(6):
            swiss_brains.append(RandomWalk("RW-" + str(i)))

        cls.swiss_tourney = Tournament(swiss_brains, game_length=500, rounds=3, seed=1)
        cls.swiss_tourney.start()

    def test_round_robin(self):
        # 2 opponents, home and away
        for score in self.round_robin_tourney.get_table():
            self.assertEqual(4, score["played"])

    def test_swiss(self):
        # 3 rounds, home and away
        for score in self.swiss_tourney.get_table():
            self.assertEqual(6, score["played"])

    def test_same_seed_gives_same_table(self):
        brains = [
            BehindAndTowards("anne"),
            DefendersAndAttackers("bob"),
            RandomWalk("charlie"),
        ]
        tables = []
        for _ in range(2):
            tourney = Tournament(brains, game_length=300, seed=7, processes=1)
            tourney.start()
            tables.append(tourney.get_table())
        self.assertEqual(tables[0], tables[1])

    def test_legs_alternate_sides(self):
        tourney = Tournament([RandomWalk("a"), RandomWalk("b")], legs=3, seed=0)
        fixtures = tourney.fixtures([(0, 1)])
        self.assertEqual(
            [(0, 1), (1, 0), (0, 1)], [(blue, red) for blue, red, _ in fixtures]
        )

    def test_sorted_table(self):
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
