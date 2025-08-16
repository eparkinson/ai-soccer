from aisoccer.brains.DefendersAndAttackers import *
from aisoccer.brains.SimpleBrain import SimpleBrain
from aisoccer.graphics.field import *

game = Game(DefendersAndAttackers(), SimpleBrain())
field = Field(game)

field.start_game()
