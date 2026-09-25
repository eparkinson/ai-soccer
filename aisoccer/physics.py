import numpy as np


class Body:
    def __init__(self, radius, position, fixed=False):
        self.radius = radius
        self.position = np.array(position, dtype=float)
        self.velocity = np.array([0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0])
        # Fixed bodies (e.g. goal posts) have infinite mass: other bodies bounce off them
        # but they never move.
        self.fixed = fixed

    def normal_velocity(self):
        return np.linalg.norm(self.velocity)

    def apply_acceleration(self, acceleration: np.ndarray):
        self.acceleration = acceleration
        self.velocity = np.add(self.velocity, self.acceleration)

    def kick(self, velocity):
        self.velocity = np.add(self.velocity, velocity)


class PhyState:
    """
    Physics world: circular bodies in a rectangular box.

    The left and right walls can have a gap (the goal mouth) between ``goal_y_min`` and
    ``goal_y_max``. Outside the gap the walls sit ``goal_depth`` in from the field edge;
    inside the gap bodies may travel all the way to the field edge.
    """

    def __init__(self, maxX, maxY, goal_depth=0, goal_y_min=0.0, goal_y_max=0.0):
        self.bodies = []
        self.maxX = maxX
        self.maxY = maxY
        self.goal_depth = goal_depth
        self.goal_y_min = goal_y_min
        self.goal_y_max = goal_y_max
        self.ticks = 0

    def add_body(self, body):
        self.bodies.append(body)

    def tick(self):
        """
        Advance the world one step. All collisions are resolved simultaneously from the
        state at the start of the tick, so the result does not depend on body order.
        """
        self.ticks = self.ticks + 1

        bodies = self.bodies
        pos = np.array([b.position for b in bodies], dtype=float)
        vel = np.array([b.velocity for b in bodies], dtype=float)
        radius = np.array([b.radius for b in bodies], dtype=float)
        fixed = np.array([b.fixed for b in bodies])

        vel = self._collide(pos, vel, radius, fixed)
        vel = self._bounce_walls(pos, vel, radius)
        vel[fixed] = 0.0
        pos = pos + vel

        for i, b in enumerate(bodies):
            b.position = pos[i]
            b.velocity = vel[i]

    @staticmethod
    def _collide(pos, vel, radius, fixed):
        # Pairwise differences: pdiff[i, j] = pos[i] - pos[j]
        pdiff = pos[:, None, :] - pos[None, :, :]
        vdiff = vel[:, None, :] - vel[None, :, :]
        dist_sq = np.einsum("ijk,ijk->ij", pdiff, pdiff)

        overlap = dist_sq <= (radius[:, None] + radius[None, :]) ** 2
        next_pdiff = pdiff + vdiff
        towards = np.einsum("ijk,ijk->ij", next_pdiff, next_pdiff) < dist_sq
        hit = overlap & towards
        np.fill_diagonal(hit, False)
        hit[fixed, :] = False  # fixed bodies are never pushed

        if not hit.any():
            return vel

        mass = radius**2
        # Elastic impulse factor 2*m_j / (m_i + m_j); against a fixed (infinite mass)
        # body this tends to 2, i.e. a perfect reflection.
        factor = np.where(
            fixed[None, :],
            2.0,
            2.0 * mass[None, :] / (mass[:, None] + mass[None, :]),
        )
        safe_dist_sq = np.where(dist_sq == 0, 1.0, dist_sq)
        scale = np.where(
            hit, factor * np.einsum("ijk,ijk->ij", vdiff, pdiff) / safe_dist_sq, 0.0
        )

        return vel - np.einsum("ij,ijk->ik", scale, pdiff)

    def _bounce_walls(self, pos, vel, radius):
        vel = vel.copy()
        x, y = pos[:, 0], pos[:, 1]

        in_mouth = (y > self.goal_y_min) & (y < self.goal_y_max)
        min_x = np.where(in_mouth, 0.0, self.goal_depth) + radius
        max_x = np.where(in_mouth, self.maxX, self.maxX - self.goal_depth) - radius

        flip_x = ((x < min_x) & (vel[:, 0] < 0)) | ((x > max_x) & (vel[:, 0] > 0))
        flip_y = ((y < radius) & (vel[:, 1] < 0)) | (
            (y > self.maxY - radius) & (vel[:, 1] > 0)
        )

        vel[flip_x, 0] = -vel[flip_x, 0]
        vel[flip_y, 1] = -vel[flip_y, 1]
        return vel

    def clear(self):
        self.bodies = []
