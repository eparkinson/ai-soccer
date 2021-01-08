from aisoccer.brains.BehindAndTowards import BehindAndTowards
from aisoccer.brains.DefendersAndAttackers import DefendersAndAttackers
from aisoccer.brains.RandomWalk import RandomWalk
from aisoccer.tournament import *

brains = []
for i in range(6):
    brains.append(BehindAndTowards("BAT-" + str(i)))

for i in range(6):
    brains.append(DefendersAndAttackers("DAA-" + str(i)))

for i in range(6):
    brains.append(RandomWalk("RAND-" + str(i)))

tourney = Tournament(brains, game_length=1000, rounds=9)
tourney.start()
