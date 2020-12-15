from field import *
from brains.RandomWalk import *
from brains.DefendersAndAttackers import *

game = Game(DefendersAndAttackers(), RandomWalk())

while True:
    status = game.tick()
    if status == GameResult.end:
        break
