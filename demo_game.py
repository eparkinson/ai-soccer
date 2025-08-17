from aisoccer.brains.DefendersAndAttackers import DefendersAndAttackers
from aisoccer.brains.SimpleBrain import SimpleBrain
from aisoccer.game import Game
from aisoccer.graphics.field import Field

game = Game(DefendersAndAttackers(), SimpleBrain())
field = Field(game)

field.start_game()
