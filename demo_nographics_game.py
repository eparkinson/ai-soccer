from aisoccer.brains.BehindAndTowards import BehindAndTowards
from aisoccer.brains.DefendersAndAttackers import DefendersAndAttackers
from aisoccer.game import Game, GameResult

game = Game(DefendersAndAttackers(), BehindAndTowards(), game_length=1800)

while True:
    status = game.tick()
    if status == GameResult.end:
        break
