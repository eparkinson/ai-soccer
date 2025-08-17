from random import randint, random

import pyglet

from aisoccer.physics import Body, PhyState

radius = 25
speed = 20

window = pyglet.window.Window(width=2000, height=600)
state = PhyState(2000, 600)


def setup():
    for i in range(50):
        body = Body(radius, [randint(0, 1999), randint(0, 599)])
        body.kick([(random() - 0.5) * speed, (random() - 0.5) * speed])
        state.add_body(body)


def update(dt):
    state.tick()


@window.event
def on_draw():
    window.clear()
    for b in state.bodies:
        circle = pyglet.shapes.Circle(
            b.position[0], b.position[1], b.radius, color=(50, 225, 30)
        )
        circle.draw()


setup()
pyglet.clock.schedule_interval(update, 0.01)
pyglet.app.run()
