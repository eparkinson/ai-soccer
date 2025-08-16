import copy
from random import shuffle

import numpy as np
from numba import njit


class Body(object):
    def __init__(self, radius, position):
        self.radius = radius
        self.position = np.array(position)
        self.velocity = np.array([0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0])

    def normal_velocity(self):
        return np.linalg.norm(self.velocity)

    def set_state(self, body):
        self.radius = body.radius
        self.position = body.position
        self.velocity = body.velocity

    @njit
    def apply_acceleration(self, acceleration: np.array):
        self.acceleration = acceleration
        self.velocity = np.add(self.velocity, self.acceleration)

    @njit
    def kick(self, velocity):
        self.velocity = np.add(self.velocity, velocity)

    @njit
    def move(self):
        self.position = np.add(self.position, self.velocity)

    @njit
    def bounce_wall(self, maxX, maxY):
        if self.position[0] < self.radius and self.velocity[0] < 0:
            self.velocity[0] = -self.velocity[0]

        if self.position[0] > maxX - self.radius and self.velocity[0] > 0:
            self.velocity[0] = -self.velocity[0]

        if self.position[1] < self.radius and self.velocity[1] < 0:
            self.velocity[1] = -self.velocity[1]

        if self.position[1] > maxY - self.radius and self.velocity[1] > 0:
            self.velocity[1] = -self.velocity[1]

    def detect_collision(self, thing):
        pdiff = np.subtract(self.position, thing.position)
        dist_squared = np.dot(pdiff, pdiff)
        overlap = dist_squared <= (self.radius + thing.radius) ** 2

        dist2 = np.linalg.norm(np.subtract(
            np.add(self.position, self.velocity),
            np.add(thing.position, thing.velocity)))

        towards = dist2 ** 2 < dist_squared

        return overlap and towards

    def calculate_collision(self, thing2):
        thing1 = copy.deepcopy(self)
        if not thing1.detect_collision(thing2):
            return thing1

        pos1 = thing1.position
        pos2 = thing2.position

        vel1 = thing1.velocity
        vel2 = thing2.velocity

        pdiff = np.subtract(pos1, pos2)
        vdiff = np.subtract(vel1, vel2)

        dot_product = np.dot(vdiff, pdiff)
        norm_squared = np.inner(pdiff, pdiff)

        mass1 = thing1.radius ** 2
        mass2 = thing2.radius ** 2

        new_vel = vel1 - (2*mass2 / (mass1 + mass2)) * (dot_product / norm_squared * pdiff)

        thing1.velocity = new_vel

        return thing1


class PhyState:
    def __init__(self, maxX, maxY):
        self.bodies = []
        self.maxX = maxX
        self.maxY = maxY
        self.ticks = 0

    def add_body(self, body):
        self.bodies.append(body)

    def tick(self):
        self.ticks = self.ticks + 1

        new_bodies = []
        for b1 in self.bodies:
            new_body = b1
            for b2 in self.bodies:
                if b1 != b2:
                    new_body = new_body.calculate_collision(b2)
            new_body.bounce_wall(self.maxX, self.maxY)
            new_bodies.append(new_body)
            new_body.move()

        for i in range(len(self.bodies)):
            self.bodies[i].set_state(new_bodies[i])

        shuffle(self.bodies)

    def clear(self):
        self.bodies = []
