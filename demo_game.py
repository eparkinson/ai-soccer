from aisoccer.brains.DefendersAndAttackers import *
from aisoccer.graphics.field import *

game = Game(DefendersAndAttackers(), BehindAndTowards())
field = Field(game)

field.start_game()
