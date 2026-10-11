"""Watch PPOBrain (blue) play DefendersAndAttackers (red) with the saved weights."""

from aisoccer.brains.DefendersAndAttackers import DefendersAndAttackers
from aisoccer.brains.PPOBrain import PPOBrain
from aisoccer.game import Game
from aisoccer.graphics.field import Field

game = Game(PPOBrain(), DefendersAndAttackers())
field = Field(game)

field.start_game()
