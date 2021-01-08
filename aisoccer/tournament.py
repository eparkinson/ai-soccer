from concurrent.futures.thread import ThreadPoolExecutor

from aisoccer.constants import Constants
from aisoccer.game import Game


class Tournament:
    def __init__(self, brains, game_length=Constants.GAME_LENGTH, rounds=0):
        self.rounds = rounds
        self.brains = brains
        self.game_length = game_length
        self.tournament_scores = TournamentScores(self.brains)

    def start(self):
        print("Starting tournament")
        print("")

        print("PLAYERS:")
        for b in self.brains:
            print("   " + b.name)

        num_brains = len(self.brains)

        if self.rounds == 0:  # round robin
            pairings = []
            for i in range(num_brains):
                for j in range(i + 1, num_brains):
                    pairings.append((i, j))
            self.play_pairings(pairings)

            self.print_scores()
        else:  # Swiss
            banned_pairings = []
            round = 1

            while round <= self.rounds:
                round_pairings, round_byes = self.calculate_swiss_pairings(banned_pairings)

                self.play_pairings(round_pairings)
                banned_pairings.extend(round_pairings)

                for player_num in round_byes:
                    self.tournament_scores.table[player_num]["points"] += 1
                    self.tournament_scores.table[player_num]["played"] += 1

                round += 1

                self.print_scores()

    def calculate_swiss_pairings(self, banned_pairings):
        table = self.get_table()
        pairings = []
        byes = set()
        paired = set()

        for player in table:
            player_num = player["number"]

            player_banned_pairings = set()
            for bp in banned_pairings:
                if bp[0] == player_num:
                    player_banned_pairings.add(bp[1])
                elif bp[1] == player_num:
                    player_banned_pairings.add(bp[0])

            if player_num not in paired:
                for opponent in table:
                    opponent_num = opponent["number"]
                    if (opponent_num not in paired) \
                            and (opponent_num not in player_banned_pairings) \
                            and (opponent_num != player_num):
                        pairing = tuple(sorted((player_num, opponent_num)))
                        pairings.append(pairing)
                        paired.add(player_num)
                        paired.add(opponent_num)
                        break

        for player in table:
            player_num = player["number"]
            if player_num not in paired:
                byes.add(player_num)

        return pairings, byes

    def play_pairings(self, pairings):
        futures = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            for p in pairings:
                futures[p] = executor.submit(self.play, p)
        for p in pairings:
            pairing_score = futures[p].result()
            self.tournament_scores.process(p, pairing_score)

    def play(self, pairing):
        blue_brain = self.brains[pairing[0]]
        red_brain = self.brains[pairing[1]]

        game = Game(blue_brain, red_brain, self.game_length, True)
        score = game.play()
        return score['blue'], score['red']

    def get_table(self):
        return self.tournament_scores.get_table()

    def print_scores(self):
        print()
        print("TOURNAMENT SCORES:")
        print("                        NAME | NUM |     P |     W |     L |    GF |    GA |    GD |  POINTS")

        for ts in self.get_table():
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
            team_score["name"] = self.brains[bnum].name
            team_score["played"] = 0
            team_score["wins"] = 0
            team_score["losses"] = 0
            team_score["draws"] = 0
            team_score["goals_for"] = 0
            team_score["goals_against"] = 0
            team_score["goal_diff"] = 0
            team_score["points"] = 0

            self.table[bnum] = team_score

    def get_table(self):
        return sorted(self.table.values(), reverse=True, key=lambda row: row["points"])

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
