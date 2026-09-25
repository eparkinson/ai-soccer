import numpy as np

from aisoccer.physics import Body, PhyState


def make_state(*bodies, **kwargs):
    state = PhyState(1000, 500, **kwargs)
    for b in bodies:
        state.add_body(b)
    return state


def test_equal_mass_head_on_collision_swaps_velocities():
    a = Body(10, [100, 100])
    b = Body(10, [119, 100])
    a.velocity = np.array([2.0, 0.0])
    b.velocity = np.array([-1.0, 0.0])

    make_state(a, b).tick()

    np.testing.assert_allclose(a.velocity, [-1.0, 0.0])
    np.testing.assert_allclose(b.velocity, [2.0, 0.0])


def test_bodies_moving_apart_do_not_collide():
    a = Body(10, [100, 100])
    b = Body(10, [119, 100])
    a.velocity = np.array([-1.0, 0.0])
    b.velocity = np.array([1.0, 0.0])

    make_state(a, b).tick()

    np.testing.assert_allclose(a.velocity, [-1.0, 0.0])
    np.testing.assert_allclose(b.velocity, [1.0, 0.0])


def test_fixed_body_reflects_and_does_not_move():
    post = Body(4, [100, 100], fixed=True)
    ball = Body(8, [111, 100])
    ball.velocity = np.array([-3.0, 0.0])

    make_state(post, ball).tick()

    np.testing.assert_allclose(ball.velocity, [3.0, 0.0])
    np.testing.assert_allclose(post.position, [100, 100])
    np.testing.assert_allclose(post.velocity, [0.0, 0.0])


def test_result_does_not_depend_on_body_order():
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 200, (8, 2)) + 100
    velocities = rng.uniform(-5, 5, (8, 2))

    def run(order):
        bodies = [Body(20, positions[i]) for i in range(8)]
        for i, b in enumerate(bodies):
            b.velocity = velocities[i].copy()
        state = make_state(*[bodies[i] for i in order])
        for _ in range(20):
            state.tick()
        return np.array([b.position for b in bodies])

    np.testing.assert_allclose(run(range(8)), run(reversed(range(8))))


def test_goal_line_is_a_wall_outside_the_mouth():
    ball = Body(8, [27, 100])
    ball.velocity = np.array([-2.0, 0.0])

    make_state(ball, goal_depth=20, goal_y_min=200, goal_y_max=300).tick()

    assert ball.velocity[0] > 0


def test_ball_passes_the_goal_line_inside_the_mouth():
    ball = Body(8, [27, 250])
    ball.velocity = np.array([-2.0, 0.0])

    make_state(ball, goal_depth=20, goal_y_min=200, goal_y_max=300).tick()

    assert ball.velocity[0] < 0
    assert ball.position[0] < 27
