from brains.BehindAndTowards import *

class DefendersAndAttackers(BehindAndTowards):
    CHANNELS = [500, 250, 100]
    def do_move(self) -> np.array:

        result = []

        for i in range(2):
            acceleration = [0, 0]
            channel_x = self.CHANNELS[i-2]

            if self.my_players_pos[i][0] < channel_x:
                acceleration[0] = 1
            else:
                acceleration[0] = -1

            if self.my_players_pos[i][1] < self.ball_pos[1]:
                acceleration[1] = 1
            else:
                acceleration[1] = -1

            result.append(acceleration)

        for i in range(2,5):
            acceleration = [0, 0]
            if self.is_behind_ball(i):
                acceleration = self.run_towards(i)
            else:
                acceleration = self.run_back(i)
            result.append(acceleration)

        return np.array(result)

