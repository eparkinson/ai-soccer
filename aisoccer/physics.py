import numpy as np


class Body:
    """
    A circular body. Once added to a PhyState its position and velocity live in the
    state's arrays (so a tick updates every body at once); reading them returns a copy.
    """

    def __init__(self, radius, position, fixed=False):
        self.radius = radius
        self._state = None  # the PhyState holding this body's position and velocity
        self._index = None
        self._position = np.array(position, dtype=float)
        self._velocity = np.array([0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0])
        # Fixed bodies (e.g. goal posts) have infinite mass: other bodies bounce off them
        # but they never move.
        self.fixed = fixed

    @property
    def position(self):
        if self._state is not None:
            return self._state.pos[self._index].copy()
        return self._position

    @position.setter
    def position(self, value):
        if self._state is not None:
            self._state.pos[self._index] = value
        else:
            self._position = np.array(value, dtype=float)

    @property
    def velocity(self):
        if self._state is not None:
            return self._state.vel[self._index].copy()
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        if self._state is not None:
            self._state.vel[self._index] = value
        else:
            self._velocity = np.array(value, dtype=float)

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
        # Every body's state in one place, updated in place each tick.
        self.pos = np.zeros((0, 2))
        self.vel = np.zeros((0, 2))
        self.radius = np.zeros(0)
        self.fixed = np.zeros(0, dtype=bool)
        self.maxX = maxX
        self.maxY = maxY
        self.goal_depth = goal_depth
        self.goal_y_min = goal_y_min
        self.goal_y_max = goal_y_max
        self.ticks = 0

    def add_body(self, body):
        position, velocity = body.position, body.velocity
        body._state, body._index = self, len(self.bodies)
        self.bodies.append(body)
        self.pos = np.vstack([self.pos, position])
        self.vel = np.vstack([self.vel, velocity])
        self.radius = np.append(self.radius, float(body.radius))
        self.fixed = np.append(self.fixed, bool(body.fixed))

    def tick(self):
        """
        Advance the world one step. All collisions are resolved simultaneously from the
        state at the start of the tick, so the result does not depend on body order.
        """
        self.ticks = self.ticks + 1

        pos, radius, fixed = self.pos, self.radius, self.fixed
        vel = self._collide(pos, self.vel, radius, fixed)
        vel = self._bounce_walls(pos, vel, radius)
        vel[fixed] = 0.0
        self.pos = pos + vel
        self.vel = vel

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
        for body in self.bodies:  # detach: bodies keep their last state
            position, velocity = body.position, body.velocity
            body._state = None
            body.position, body.velocity = position, velocity
        self.bodies = []
        self.pos = np.zeros((0, 2))
        self.vel = np.zeros((0, 2))
        self.radius = np.zeros(0)
        self.fixed = np.zeros(0, dtype=bool)
