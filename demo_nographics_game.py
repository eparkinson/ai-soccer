from aisoccer.brains.DefendersAndAttackers import *

game = Game(DefendersAndAttackers(), BehindAndTowards(), game_length=1800)

while True:
    status = game.tick()
    if status == GameResult.end:
        break
