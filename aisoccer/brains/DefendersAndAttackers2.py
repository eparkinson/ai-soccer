from aisoccer.brains.BaseBrainUtils import *
from aisoccer.brains.BehindAndTowards import *


class DefendersAndAttackers2(BaseBrainUtils):
    BASE_POS = [[100, 300],  [300, 300], [600, 100], [800, 500]]

    def do_move(self) -> np.array:

        result = []

        for i in range(4):
            if 25 < self.distance_to_ball(i) < (i+1) * 100 :
                if self.is_behind_ball(i):
                    result.append(self.run_towards(i, self.ball_pos))
                else:
                    result.append(self.run_back(i))
            elif self.is_ball_direction_forward():
                target = np.array([0, 0])
                target[0] = self.BASE_POS[i][0]
                target[1] = self.BASE_POS[i][1]
                result.append(self.run_towards(i, target))
            elif self.is_behind_ball(i):
                target = np.array([0, 0])
                target[0] = self.BASE_POS[i][0] - 15
                target[1] = self.calculate_ball_x_intersection(self.BASE_POS[i][0])
                result.append(self.run_towards(i, target))
            else:
                result.append(self.run_back(i))

        for i in range(4, 5):
            acceleration = [0, 0]
            if self.is_behind_ball(i):
                acceleration = self.run_towards(i, self.ball_pos)
            else:
                acceleration = self.run_back(i)
            result.append(acceleration)

        return np.array(result)

