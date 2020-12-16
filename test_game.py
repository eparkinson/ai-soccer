from aisoccer.game import *
from aisoccer.graphics.field import *
from aisoccer.brains.DefendersAndAttackers import *

game = Game(DefendersAndAttackers(), BehindAndTowards())
field = Field(game)

field.start_game()
