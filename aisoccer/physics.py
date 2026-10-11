import numpy as np


def _dot(a, b):
    """Row-wise dot product of (P, 2) arrays."""
    return a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]


def resolve_contacts(pdiff, vdiff, reach_sq, factor, dt):
    """
    Swept elastic collisions for one sub-step of ``dt`` ticks, for candidate ordered
    pairs (i, j), one per row: pdiff = pos_i - pos_j, vdiff = vel_i - vel_j, reach_sq =
    (r_i + r_j)^2, factor = the elastic impulse factor 2 m_j / (m_i + m_j) (2 when j is
    fixed).

    A pair collides if it is approaching (vdiff . pdiff < 0) and either overlaps now or
    touches during the sub-step. The impulse is along the line of centres at the moment
    of first touch (now, if overlapping), so glancing hits deflect as they should.

    Returns (hit, dv, shift): which rows collide and, for those rows, the change to
    subtract from body i's velocity and the correction to add to its position after
    the sub-step. The correction: a body that touches part way through the sub-step
    at time t moved with its old velocity until then, so it ends (v_old - v_new) * t
    from where moving with the new velocity for the whole sub-step would put it.

    PhyState and VecGame both call this, so their collisions are bit-identical.
    """
    hit = _dot(pdiff, vdiff) < 0
    if not hit.any():
        return hit, None, None
    p, v, reach_sq, factor = pdiff[hit], vdiff[hit], reach_sq[hit], factor[hit]
    pv, vv, dist_sq = _dot(p, v), _dot(v, v), _dot(p, p)  # approaching: vv > 0
    overlap = dist_sq <= reach_sq
    # First touch: the smaller root of |p + v t|^2 = reach^2 (none if disc < 0).
    disc = pv * pv - vv * (dist_sq - reach_sq)
    t = (-pv - np.sqrt(np.maximum(disc, 0.0))) / vv
    touch = overlap | ((disc >= 0) & (t <= dt))
    hit[hit] = touch
    if not touch.any():
        return hit, None, None
    p, v, factor = p[touch], v[touch], factor[touch]
    t = np.where(overlap[touch], 0.0, t[touch])
    normal = p + v * t[:, None]  # line of centres at first touch
    nn = _dot(normal, normal)
    scale = factor * _dot(v, normal) / np.where(nn == 0, 1.0, nn)
    return hit, scale[:, None] * normal, (scale * t)[:, None] * normal


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

    def tick(self, substeps=1, stop=None):
        """
        Advance the world one tick.

        With one sub-step (the original physics, kept exactly): resolve collisions and
        wall bounces from the state at the start of the tick, then move every body by
        its velocity. Two bodies collide if they overlap and would be closer after the
        move. At high speed that lets bodies tunnel: a ball can jump deep into a post,
        be judged to be moving away from it, and carry on through.

        With ``substeps`` > 1 the tick is split into equal sub-steps, each resolving
        collisions (swept: see ``_collide_swept``) and wall bounces, then moving every
        body by velocity / substeps. All collisions in a (sub-)step are resolved
        simultaneously, so the result does not depend on body order. ``stop``, if
        given, is called after each sub-step; when it returns True the rest of the tick
        is skipped (Game stops the moment the ball is in a goal, so it cannot bounce
        back out within the tick).
        """
        self.ticks = self.ticks + 1
        radius, fixed = self.radius, self.fixed
        if substeps == 1:
            pos = self.pos
            vel = self._collide(pos, self.vel, radius, fixed)
            vel = self._bounce_walls(pos, vel, radius)
            vel[fixed] = 0.0
            self.pos = pos + vel
            self.vel = vel
            return
        dt = 1.0 / substeps
        for _ in range(substeps):
            pos = self.pos
            vel, shift = self._collide_swept(pos, self.vel, radius, fixed, dt)
            vel = self._bounce_walls(pos, vel, radius)
            vel[fixed] = 0.0
            self.pos = pos + vel * dt + shift
            self.vel = vel
            if stop is not None and stop():
                break

    def _collide_swept(self, pos, vel, radius, fixed, dt):
        """
        Collisions for one sub-step of ``dt`` ticks (see ``resolve_contacts``), all
        resolved simultaneously. Returns the new velocities and position corrections.
        """
        i, j, reach_sq, factor = self._pairs()
        hit, dv, shift = resolve_contacts(
            pos[i] - pos[j], vel[i] - vel[j], reach_sq, factor, dt
        )
        if dv is None:
            return vel, 0.0
        impulse = np.zeros_like(vel)
        correction = np.zeros_like(vel)
        np.add.at(impulse, i[hit], dv)
        np.add.at(correction, i[hit], shift)
        return vel - impulse, correction

    def _pairs(self):
        """Every ordered pair (i, j) that can collide (i not fixed), in row-major order,
        with its (r_i + r_j)^2 and elastic impulse factor (see _collide)."""
        radius, fixed = self.radius, self.fixed  # replaced (not changed) by add_body
        cache = getattr(self, "_pair_cache", None)
        if cache is None or cache[0] is not radius or cache[1] is not fixed:
            can_hit = ~np.eye(len(radius), dtype=bool)
            can_hit[fixed, :] = False  # fixed bodies are never pushed
            i, j = np.nonzero(can_hit)
            mass = radius**2
            factor = np.where(fixed[j], 2.0, 2.0 * mass[j] / (mass[i] + mass[j]))
            self._pair_cache = (radius, fixed, (i, j, (radius[i] + radius[j]) ** 2, factor))
        return self._pair_cache[2]

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
