from aisoccer.game import *
from aisoccer.graphics.field import *
from aisoccer.brains.DefendersAndAttackers2 import *

game = Game(DefendersAndAttackers2(), BehindAndTowards())
field = Field(game)

field.start_game()
