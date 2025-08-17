import numpy as np

from aisoccer.abstractbrain import AbstractBrain


class SimpleBrain(AbstractBrain):
    """Minimal example brain.

    Behavior: each friendly player accelerates toward the ball with a small, capped
    acceleration. This is intentionally simple and deterministic so it's easy to
    test and extend.
    """

    def do_move(self, game_state=None) -> np.array:
        # Defensive: ensure inputs are available and shaped correctly
        my_pos = np.asarray(self.my_players_pos, dtype=float)
        # ball_pos may be (2,) or (1,2)
        ball = np.asarray(self.ball_pos, dtype=float).reshape(-1)
        if ball.size >= 2:
            ball_pos = ball[:2]
        else:
            # fallback: keep players still if ball position is malformed
            return np.zeros((5, 2), dtype=float)

        # Compute direction vectors from each player to the ball
        dirs = ball_pos - my_pos  # shape (5,2)
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        unit = dirs / norms

        # Simple constant acceleration toward the ball, capped to a max magnitude
        accel_mag = 1.0
        max_accel = 5.0
        acc = unit * accel_mag
        acc = np.clip(acc, -max_accel, max_accel)

        return acc
