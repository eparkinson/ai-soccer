from aisoccer.brains.BehindAndTowards import BehindAndTowards
from aisoccer.brains.DefendersAndAttackers import DefendersAndAttackers
from aisoccer.brains.RandomWalk import RandomWalk
from aisoccer.tournament import *

t = Tournament([RandomWalk(), DefendersAndAttackers(), BehindAndTowards()], 18000)

t.start()