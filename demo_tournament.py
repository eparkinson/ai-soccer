from aisoccer.brains.BehindAndTowards import BehindAndTowards
from aisoccer.brains.DefendersAndAttackers import DefendersAndAttackers
from aisoccer.brains.RandomWalk import RandomWalk
from aisoccer.tournament import *

brains = []

brains.append(BehindAndTowards("BAT-0"))
brains.append(BehindAndTowards("BAT-1"))
brains.append(DefendersAndAttackers("DAA-1"))
brains.append(RandomWalk("RAND-1"))

tourney = Tournament(brains, game_length=2500, rounds=0)
tourney.start()

print("*****************************")
print(tourney.tournament_scores.table[0]["points"])
