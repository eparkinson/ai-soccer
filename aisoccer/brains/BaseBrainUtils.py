from aisoccer.game import *


class BaseBrainUtils(AbstractBrain, ABC):
    def run_towards(self, player_index, position, scale_factor=1):
        acceleration = np.subtract(position, self.my_players_pos[player_index])
        if scale_factor > 1:
            acceleration = np.divide(acceleration, scale_factor)

        return acceleration

    def is_behind_ball(self, player_index):
        return self.my_players_pos[player_index][0] < self.ball_pos[0] - 5

    def run_back(self, player_index):
        result = [-1, 0]
        return result

    def is_ball_direction_forward(self):
        return self.ball_vel[0] > 0

    def calculate_ball_x_intersection(self, x):
        ball_x = self.ball_pos[0]
        ball_y = self.ball_pos[1]
        ball_dx = self.ball_vel[0]
        ball_dy = self.ball_vel[1]

        t = abs((ball_x - x) / ball_dx)
        y = ball_y + t * ball_dy
        y_multiples = int(y // Constants.FIELD_HEIGHT)

        if (y_multiples % 2) == 0:
            y = y % Constants.FIELD_HEIGHT
        else:
            y = Constants.FIELD_HEIGHT - (y % Constants.FIELD_HEIGHT)

        return y

    def distance_to_ball(self, i):
        d = np.linalg.norm(np.subtract(self.ball_pos, self.my_players_pos[i]))
        return d

    def to_list(self):
        result = np.append(self.my_players_pos, self.opp_players_pos)
        result = np.append(result, self.ball_pos)
        result = np.append(result, self.ball_vel)

        return result.tolist()
