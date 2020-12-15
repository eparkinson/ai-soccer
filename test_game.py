from field import *
from brains.RandomWalk import *
from brains.BehindAndTowards import *
from brains.DefendersAndAttackers import *

game = Game(DefendersAndAttackers(), BehindAndTowards())
field = Field(game)

field.start_game()
