from core.game import Game
from field import *
from brains.DefendersAndAttackers2 import *

game = Game(DefendersAndAttackers2(), BehindAndTowards())
field = Field(game)

field.start_game()
