from concurrent.futures.thread import ThreadPoolExecutor

from aisoccer.constants import Constants
from aisoccer.game import Game


class Tournament:
    def __init__(self, brains, game_length=Constants.GAME_LENGTH):
        self.scores = {}
        self.brains = brains
        self.game_length = game_length

        num_brains = len(self.brains)

        self.pairings = []

        for i in range(num_brains):
            for j in range(i + 1, num_brains):
                self.pairings.append((i, j))

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
            self.scores[p] = futures[p].result()

        self.print_scores()

    def play(self, pairing):
        blue_brain = self.brains[pairing[0]]
        red_brain = self.brains[pairing[1]]

        game = Game(blue_brain, red_brain, self.game_length, True)
        score = game.play()
        return score['blue'], score['red']

    def print_scores(self):
        print("SCORES: ")
        for p in self.pairings:
            b0 = self.brains[p[0]]
            b1 = self.brains[p[1]]
            s0 = self.scores[p][0]
            s1 = self.scores[p][1]

            print("   " + b0.get_name() + " v " + b1.get_name() +
                  "  score:  " + str(s0) + "-" + str(s1))
