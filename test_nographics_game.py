from aisoccer.game import Game
from aisoccer.graphics.field import *
from aisoccer.brains.DefendersAndAttackers import *

game = Game(DefendersAndAttackers(), BehindAndTowards())

while True:
    status = game.tick()
    if status == GameResult.end:
        break
