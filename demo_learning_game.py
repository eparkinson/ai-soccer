from aisoccer.brains.LearningBrain import LearningBrain
from aisoccer.brains.DefendersAndAttackers import *
from aisoccer.brains.RandomWalk import RandomWalk
from aisoccer.brains.SimpleBrain import SimpleBrain

# Run the game without a GUI field
game = Game(LearningBrain(), SimpleBrain(), game_length=0)

while True:
    result = game.tick()
    if result == GameResult.end:
        break
