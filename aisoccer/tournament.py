from concurrent.futures.thread import ThreadPoolExecutor

from aisoccer.constants import Constants
from aisoccer.game import Game


class Tournament:
    def __init__(self, brains, game_length=Constants.GAME_LENGTH, first_brain_against_all=False):
        self.first_brain_against_all = first_brain_against_all
        self.scores = {}
        self.brains = brains
        self.game_length = game_length
        self.tournament_scores = TournamentScores(self.brains)

        num_brains = len(self.brains)

        self.pairings = []

        if not first_brain_against_all:
            for i in range(num_brains):
                for j in range(i + 1, num_brains):
                    self.pairings.append((i, j))
        else:
            for j in range(1, num_brains):
                self.pairings.append((0, j))

    def start(self):
        print("Starting tournament")
        print("")

        print("PLAYERS:")
        for b in self.brains:
            print("   " + b.get_name())

        futures = {}

        with ThreadPoolExecutor(max_workers=4) as executor:
            for p in self.pairings:
                futures[p] = executor.submit(self.play, p)

        for p in self.pairings:
            pairing_score = futures[p].result()
            self.scores[p] = pairing_score
            self.tournament_scores.process(p, pairing_score)

        self.print_scores()

    def play(self, pairing):
        blue_brain = self.brains[pairing[0]]
        red_brain = self.brains[pairing[1]]

        game = Game(blue_brain, red_brain, self.game_length, True)
        score = game.play()
        return score['blue'], score['red']

    def print_scores(self):
        print()
        print("SCORES: ")
        for p in self.pairings:
            b0 = self.brains[p[0]]
            b1 = self.brains[p[1]]
            s0 = self.scores[p][0]
            s1 = self.scores[p][1]

            print("   " + b0.get_name() + " v " + b1.get_name() +
                  "  score:  " + str(s0) + "-" + str(s1))

        print()
        print("TOURNAMENT SCORES:")
        print("                        NAME | NUM |     P |     W |     L |    GF |    GA |    GD |  POINTS")

        for key in self.tournament_scores.table:
            ts = self.tournament_scores.table[key]
            print("   {0:25} | {1:3d} | {2:5d} | {3:5d} | {4:5d} | {5:5d} | {6:5d} | {7:5d} | {8:5d}"
                  .format(ts["name"], ts["number"],
                          ts["played"], ts["wins"], ts["losses"],
                          ts["goals_for"], ts["goals_against"], ts["goal_diff"],
                          ts["points"]))


class TournamentScores:
    def __init__(self, brains):
        self.brains = brains
        self.table = {}

        for bnum in range(len(self.brains)):
            team_score = {}

            team_score["number"] = bnum
            team_score["name"] = self.brains[bnum].get_name()
            team_score["played"] = 0
            team_score["wins"] = 0
            team_score["losses"] = 0
            team_score["draws"] = 0
            team_score["goals_for"] = 0
            team_score["goals_against"] = 0
            team_score["goal_diff"] = 0
            team_score["points"] = 0

            self.table[bnum] = team_score

    def process(self, p, pairing_score):
        (side1, side2) = p
        (score1, score2) = pairing_score

        self.table[side1]["played"] += 1
        self.table[side2]["played"] += 1

        self.table[side1]["goals_for"] += score1
        self.table[side2]["goals_for"] += score2

        self.table[side1]["goals_against"] += score2
        self.table[side2]["goals_against"] += score1

        self.table[side1]["goal_diff"] += score1 - score2
        self.table[side2]["goal_diff"] += score2 - score1

        if score1 > score2:
            self.table[side1]["wins"] += 1
            self.table[side2]["losses"] += 1
            self.table[side1]["points"] += 3
        elif score2 > score1:
            self.table[side2]["wins"] += 1
            self.table[side1]["losses"] += 1
            self.table[side2]["points"] += 3
        else:
            self.table[side1]["draws"] += 1
            self.table[side2]["draws"] += 1
            self.table[side1]["points"] += 1
            self.table[side2]["points"] += 1
