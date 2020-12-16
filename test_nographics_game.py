from aisoccer.game import Game
from aisoccer.graphics.field import *
from aisoccer.brains.DefendersAndAttackers2 import *

game = Game(DefendersAndAttackers2(), BehindAndTowards())

while True:
    status = game.tick()
    if status == GameResult.end:
        break
