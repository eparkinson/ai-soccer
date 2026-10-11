"""
Game speed (Constants.SPEED_SCALE) and physics sub-steps (Constants.SUBSTEPS).

- SPEED_SCALE = 1 with SUBSTEPS = 1 is exactly the original game.
- With sub-steps, a ball at full speed never passes through a post, a player or a wall,
  and goals are still detected (and only real goals).
- VecGame stays bit-identical to Game at any speed and number of sub-steps.
"""

import hashlib
from contextlib import contextmanager

import numpy as np
import pytest

from aisoccer.brains.BehindAndTowards import BehindAndTowards
from aisoccer.brains.DefendersAndAttackers import DefendersAndAttackers
from aisoccer.brainspec import HEURISTICS
from aisoccer.constants import Constants
from aisoccer.game import Game, GameResult
from aisoccer.physics import Body, PhyState
from aisoccer.team import Team
from aisoccer.vecgame import BALL, VecGame
from tests.test_game import SpyBrain
from tests.test_vecgame import assert_plays_like_game, physics_ticks, trained_weights


@contextmanager
def speed(scale, substeps):
    previous = Constants.set_speed(scale, substeps)
    try:
        yield
    finally:
        Constants.set_speed(*previous)


# ------------------------------------------------ SPEED_SCALE 1, SUBSTEPS 1: the original


# Fingerprints of three seeded 3000-tick games (every position and velocity after every
# physics tick), recorded with the code before SPEED_SCALE and SUBSTEPS existed, on the
# rules with GOAL_MOUTH_HEIGHT = 400. If a brain used here changes, re-record them from
# that code (or drop the case): the test is about the physics and rules, not the brains.
ORIGINAL_GAMES = [
    ("DefendersAndAttackers", "BehindAndTowards", 1,
     "d32a029390eb241638ef23d8102fce50dfec13d78d1958dbc012d9d2acc7e1d0", {"red": 3, "blue": 2}),
    ("StrategicPlanner", "AdaptiveChaser", 2,
     "61aeeca1c2ed22fd13207b9189c97b6117ade695901ceeb23dc7ec172cbe37eb", {"red": 2, "blue": 1}),
    ("BehindAndTowards", "DefendersAndAttackers", 3,
     "3ec9be21c422c9a4fbc5bf360f815977640d6370766f3afbe98b9014357170e1", {"red": 2, "blue": 0}),
]  # fmt: skip


def game_fingerprint(blue, red, seed, ticks=3000):
    game = Game(HEURISTICS[blue](), HEURISTICS[red](), ticks, quiet_mode=True, seed=seed)
    digest = hashlib.sha256()
    last = -1
    while game.tick() != GameResult.end:
        if game.state.ticks != last:
            last = game.state.ticks
            digest.update(game.state.pos.tobytes())
            digest.update(game.state.vel.tobytes())
    return digest.hexdigest(), game.score


@pytest.mark.skipif(
    Constants.GOAL_MOUTH_HEIGHT != 400, reason="recorded with a 400 px goal mouth"
)
@pytest.mark.parametrize("blue,red,seed,expected,score", ORIGINAL_GAMES)
def test_speed_1_and_one_substep_replay_the_original_game_bit_for_bit(
    blue, red, seed, expected, score
):
    with speed(1, 1):
        assert game_fingerprint(blue, red, seed) == (expected, score)


def original_tick(state):
    """PhyState.tick, _collide and _bounce_walls as they were, verbatim."""
    pos, radius, fixed = state.pos, state.radius, state.fixed
    pdiff = pos[:, None, :] - pos[None, :, :]
    vdiff = state.vel[:, None, :] - state.vel[None, :, :]
    dist_sq = np.einsum("ijk,ijk->ij", pdiff, pdiff)
    overlap = dist_sq <= (radius[:, None] + radius[None, :]) ** 2
    next_pdiff = pdiff + vdiff
    towards = np.einsum("ijk,ijk->ij", next_pdiff, next_pdiff) < dist_sq
    hit = overlap & towards
    np.fill_diagonal(hit, False)
    hit[fixed, :] = False
    vel = state.vel
    if hit.any():
        mass = radius**2
        factor = np.where(
            fixed[None, :], 2.0, 2.0 * mass[None, :] / (mass[:, None] + mass[None, :])
        )
        safe_dist_sq = np.where(dist_sq == 0, 1.0, dist_sq)
        scale = np.where(
            hit, factor * np.einsum("ijk,ijk->ij", vdiff, pdiff) / safe_dist_sq, 0.0
        )
        vel = vel - np.einsum("ij,ijk->ik", scale, pdiff)
    vel = vel.copy()
    x, y = pos[:, 0], pos[:, 1]
    in_mouth = (y > state.goal_y_min) & (y < state.goal_y_max)
    min_x = np.where(in_mouth, 0.0, state.goal_depth) + radius
    max_x = np.where(in_mouth, state.maxX, state.maxX - state.goal_depth) - radius
    flip_x = ((x < min_x) & (vel[:, 0] < 0)) | ((x > max_x) & (vel[:, 0] > 0))
    flip_y = ((y < radius) & (vel[:, 1] < 0)) | ((y > state.maxY - radius) & (vel[:, 1] > 0))
    vel[flip_x, 0] = -vel[flip_x, 0]
    vel[flip_y, 1] = -vel[flip_y, 1]
    vel[fixed] = 0.0
    state.pos = pos + vel
    state.vel = vel


def crowded_state(seed):
    """Many bodies in a small box with goal mouths, so there are collisions every tick."""
    rng = np.random.default_rng(seed)
    state = PhyState(300, 200, goal_depth=20, goal_y_min=60, goal_y_max=140)
    for _ in range(12):
        body = Body(float(rng.choice([8, 24])), rng.uniform([30, 30], [270, 170]))
        body.velocity = rng.normal(0, 4, 2)
        state.add_body(body)
    for xy in ([20, 60], [20, 140], [280, 60], [280, 140]):
        state.add_body(Body(4, xy, fixed=True))
    return state


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_one_substep_is_the_original_physics_tick(seed):
    new, old = crowded_state(seed), crowded_state(seed)
    for _ in range(300):
        new.tick(1)
        original_tick(old)
        assert np.array_equal(new.pos, old.pos) and np.array_equal(new.vel, old.vel)


def test_acceleration_cap_scales_with_speed():
    move = np.array([[3.0, 4.0], [0.3, 0.4], [1.2, 0.0], [0.0, 0.0], [-6.0, 8.0]])
    with speed(1, 1):
        team = Team(SpyBrain(), 0)
        used = team.apply_move(move)
        # The original rule, verbatim: cap each acceleration at magnitude 1.
        norms = np.linalg.norm(move, axis=1, keepdims=True)
        assert np.array_equal(used, np.where(norms > 1, move / np.maximum(norms, 1), move))
    with speed(1.5, 2):
        used = Team(SpyBrain(), 0).apply_move(move)
        np.testing.assert_allclose(
            used, [[0.9, 1.2], [0.3, 0.4], [1.2, 0.0], [0.0, 0.0], [-0.9, 1.2]]
        )
        assert Constants.MAX_PLAYER_VELOCITY == 7.5
        assert Constants.MAX_BALL_VELOCITY == 15


# ------------------------------------------------------------------------ tunnelling


def fire_ball(target_radius, fixed, target_speed, speed_, substeps, angle, offset, phase):
    """
    Fire a ball at ``speed_`` px per tick at a target (a post if ``fixed``, else a
    player moving straight at the ball at ``target_speed``), from direction ``angle``
    with perpendicular ``offset`` from a head-on hit; ``phase`` in [0, 1) shifts where
    the sub-step boundaries fall. Returns (deflected, closest sampled centre distance).
    """
    state = PhyState(1e5, 1e5)
    centre = np.array([5000.0, 5000.0])
    u = np.array([np.cos(angle), np.sin(angle)])  # the ball's direction of travel
    n = np.array([-u[1], u[0]])
    target = Body(target_radius, centre, fixed=fixed)
    start = centre - u * (80 + phase * (speed_ + target_speed)) + n * offset
    ball = Body(Constants.BALL_RADIUS, start)
    state.add_body(target)
    state.add_body(ball)
    state.vel[1] = u * speed_
    if not fixed:
        state.vel[0] = -u * target_speed
    closest = [np.inf]

    def sample():
        closest[0] = min(closest[0], np.linalg.norm(state.pos[1] - state.pos[0]))
        return False

    for _ in range(int(240 / speed_) + 2):
        state.tick(substeps, sample if substeps > 1 else None)
        if substeps == 1:
            sample()
    deflected = not np.allclose(state.vel[1], u * speed_)
    return deflected, closest[0]


TARGETS = {
    "post": (Constants.POST_RADIUS, True, 0.0),
    "standing player": (Constants.PLAYER_RADIUS, False, 0.0),
    "charging player": (Constants.PLAYER_RADIUS, False, "max"),
}
ANGLES = np.linspace(0, 2 * np.pi, 16, endpoint=False) + 0.1
PHASES = np.linspace(0, 1, 6, endpoint=False)
OFFSETS = np.linspace(0, 0.995, 12)  # fractions of the contact distance


def tunnel_count(target, speed_, substeps):
    radius, fixed, target_speed = TARGETS[target]
    if target_speed == "max":
        target_speed = Constants.MAX_PLAYER_VELOCITY
    reach = radius + Constants.BALL_RADIUS
    through, closest = 0, np.inf
    for angle in ANGLES:
        for offset in OFFSETS:
            for phase in PHASES:
                hit, near = fire_ball(
                    radius, fixed, target_speed, speed_, substeps, angle, offset * reach, phase
                )
                through += not hit
                closest = min(closest, near)
    return through, closest / reach


@pytest.mark.parametrize("target", list(TARGETS))
def test_ball_at_full_speed_never_passes_through(target):
    """
    Every shot whose path meets the target bounces off it, from 16 directions, 12
    offsets up to a 0.5% graze and 6 sub-step phases, at SPEED_SCALE 1.5 with 2
    sub-steps; and the ball never even overlaps it at a sub-step (swept collisions).
    """
    with speed(1.5, 2):
        through, closest = tunnel_count(target, Constants.MAX_BALL_VELOCITY, 2)
    assert through == 0
    assert closest > 1 - 1e-9


def test_without_substeps_the_fast_ball_does_pass_through_posts():
    """The same test with the original one-step physics at 1.5x fails badly (and at 1x too)."""
    with speed(1.5, 1):
        through, _ = tunnel_count("post", Constants.MAX_BALL_VELOCITY, 1)
    assert through > 0.5 * len(ANGLES) * len(OFFSETS) * len(PHASES)
    with speed(1, 1):
        through, _ = tunnel_count("post", Constants.MAX_BALL_VELOCITY, 1)
    assert through > 0.25 * len(ANGLES) * len(OFFSETS) * len(PHASES)


def test_glancing_hit_on_a_post_reflects_exactly():
    """A single swept collision is exact: the ball leaves as a perfect reflection would."""
    with speed(1.5, 2):
        post = np.array([500.0, 300.0])
        v = np.array([15.0, 0.0])
        for b in (0.0, 3.0, 7.0, 11.0, 11.9):
            for phase in PHASES:
                state = PhyState(1000, 600)
                state.add_body(Body(4, post, fixed=True))
                ball = Body(8, post + [-60 - 15 * phase, b])
                state.add_body(ball)
                state.vel[1] = v
                for _ in range(8):
                    state.tick(2)
                # Analytic: contact where |p - post| = 12, normal there, reflect.
                x_contact = post[0] - np.sqrt(144 - b * b)
                normal = np.array([x_contact, post[1] + b]) - post
                normal /= np.linalg.norm(normal)
                v_out = v - 2 * (v @ normal) * normal
                t_contact = (x_contact - (post[0] - 60 - 15 * phase)) / 15
                expected = np.array([x_contact, post[1] + b]) + v_out * (8 - t_contact)
                np.testing.assert_allclose(state.vel[1], v_out, atol=1e-9)
                np.testing.assert_allclose(state.pos[1], expected, atol=1e-9)


def field_state():
    return PhyState(
        Constants.FIELD_LENGTH - 1,
        Constants.FIELD_HEIGHT,
        goal_depth=Constants.GOAL_DEPTH,
        goal_y_min=Constants.GOAL_Y_MIN,
        goal_y_max=Constants.GOAL_Y_MAX,
    )


def test_ball_at_full_speed_never_crosses_a_wall():
    """
    Side walls, the goal lines outside the mouths and the back of the goals: the ball's
    centre never crosses the wall's line, from many angles and sub-step phases.
    """
    L, H = Constants.FIELD_LENGTH - 1, Constants.FIELD_HEIGHT
    r, depth = Constants.BALL_RADIUS, Constants.GOAL_DEPTH
    # (a point on the wall, its inward normal)
    walls = [
        ([900, 0], [0, 1]),
        ([900, H], [0, -1]),
        ([depth, 60], [1, 0]),  # blue's goal line, beside the mouth
        ([L - depth, H - 60], [-1, 0]),  # red's goal line, beside the mouth
        ([0, H / 2], [1, 0]),  # back of blue's goal
        ([L, H / 2], [-1, 0]),  # back of red's goal
    ]
    with speed(1.5, 2):
        v_max = Constants.MAX_BALL_VELOCITY
        for point, inward in walls:
            point, inward = np.array(point, float), np.array(inward, float)
            along = np.array([-inward[1], inward[0]])
            for angle in np.linspace(-1.0, 1.0, 11):  # to the wall's normal
                direction = -inward * np.cos(angle) + along * np.sin(angle)
                for phase in PHASES:
                    state = field_state()
                    start = point + inward * (r + 40 + phase * v_max)
                    state.add_body(Body(r, start))
                    state.vel[0] = direction * v_max
                    for _ in range(8):
                        state.tick(2)
                        depth_past = -(state.pos[0] - point) @ inward
                        assert depth_past < 0, (point, angle, phase)
                    assert (state.vel[0] @ inward) > 0  # bounced back


# --------------------------------------------------------------------------- goals


def still_game():
    """A game whose players stand still at kick-off, the ball placed by the test."""
    return Game(SpyBrain(), SpyBrain(), game_length=0, quiet_mode=True, seed=0)


def shoot(game, pos, vel, ticks=30):
    game.ball.body.position = np.array(pos, dtype=float)
    game.ball.body.velocity = np.array(vel, dtype=float)
    for _ in range(ticks):
        result = game.tick()
        if result != GameResult.nothing:
            return result
    return GameResult.nothing


def test_shots_at_full_speed_into_the_mouth_score():
    with speed(1.5, 2):
        v = Constants.MAX_BALL_VELOCITY
        for y_line in np.linspace(Constants.GOAL_Y_MIN + 20, Constants.GOAL_Y_MAX - 20, 9):
            for angle in np.linspace(-0.6, 0.6, 7):
                for phase in PHASES:
                    game = still_game()
                    d = np.array([-np.cos(angle), np.sin(angle)])  # towards blue's goal
                    start = np.array([Constants.GOAL_DEPTH, y_line]) - d * (100 + phase * v)
                    assert shoot(game, start, d * v) == GameResult.goal_red
                    game = still_game()
                    mirrored = [Constants.FIELD_LENGTH - 1 - start[0], start[1]]
                    assert shoot(game, mirrored, [-d[0] * v, d[1] * v]) == GameResult.goal_blue


def test_shots_at_the_posts_and_beside_the_mouth_do_not_score():
    with speed(1.5, 2):
        v = Constants.MAX_BALL_VELOCITY
        targets = [Constants.GOAL_Y_MIN, Constants.GOAL_Y_MAX]  # the posts, dead centre
        targets += [Constants.GOAL_Y_MIN - 60, Constants.GOAL_Y_MAX + 60]  # the line
        for y_line in targets:
            for angle in np.linspace(-0.5, 0.5, 5):
                for phase in PHASES:
                    game = still_game()
                    d = np.array([-np.cos(angle), np.sin(angle)])
                    start = np.array([Constants.GOAL_DEPTH, y_line]) - d * (100 + phase * v)
                    assert shoot(game, start, d * v, ticks=12) == GameResult.nothing
                    assert game.ball.body.velocity[0] > 0  # came back out


def test_goal_is_caught_at_the_sub_step_the_ball_goes_in():
    """
    The ball is over the line after the first sub-step and would bounce off the back
    of the goal and out again in the second: the tick stops at the goal instead.
    """
    with speed(1.5, 2):
        start, vel = [14.0, 400.0], [-15.0, 0.0]
        state = field_state()  # the bare physics, no goal check: it comes back out
        state.add_body(Body(Constants.BALL_RADIUS, start))
        state.vel[0] = vel
        state.tick(2)
        assert state.pos[0, 0] > Constants.GOAL_DEPTH - Constants.BALL_RADIUS
        game = still_game()
        assert shoot(game, start, vel, ticks=2) == GameResult.goal_red


# ------------------------------------------------------------------- VecGame == Game


SPEEDS = [(1, 1), (1.5, 1), (1.5, 2), (2, 3)]


@pytest.mark.parametrize("scale,substeps", SPEEDS)
def test_vecgame_matches_game_at_any_speed(scale, substeps):
    with speed(scale, substeps):
        vec = assert_plays_like_game(
            lambda: (DefendersAndAttackers(), BehindAndTowards()), [5, 6], 700
        )
    assert vec.score.sum() > 0  # goals (and so kick-off resets) happened


@pytest.mark.parametrize("scale,substeps", [(1.5, 2), (2, 3)])
def test_vecgame_matches_game_with_ppo_brains(scale, substeps):
    from aisoccer.brains.PPOBrain import PPOBrain

    trained = trained_weights()
    with speed(scale, substeps):
        assert_plays_like_game(
            lambda: (PPOBrain("a", weights=trained), DefendersAndAttackers()), [7], 300
        )


@pytest.mark.parametrize("scale,substeps", [(1.5, 2), (2, 3)])
def test_vecgame_stops_a_game_at_the_goal_like_game(scale, substeps):
    """Balls going in at every sub-step phase, and one that does not: same as Game."""
    with speed(scale, substeps):
        v = Constants.MAX_BALL_VELOCITY
        starts = [[Constants.GOAL_DEPTH + 2 + v * k / 7, 400.0] for k in range(7)]
        starts.append([900.0, 400.0])
        seeds = list(range(len(starts)))
        vec = VecGame(
            [SpyBrain() for _ in seeds], [SpyBrain() for _ in seeds], seeds, game_length=5
        )
        games = [Game(SpyBrain(), SpyBrain(), 5, quiet_mode=True, seed=s) for s in seeds]
        for k, (start, game) in enumerate(zip(starts, games)):
            vec.pos[k, BALL] = start
            vec.vel[k, BALL] = [-v, 0.0]
            game.ball.body.position = start
            game.ball.body.velocity = [-v, 0.0]
        for _ in range(5):
            vec.tick()
            for k, game in enumerate(games):
                physics_ticks(game)
                assert np.array_equal(vec.pos[k], game.state.pos)
                assert np.array_equal(vec.vel[k], game.state.vel)
        assert list(vec.score[:, 1]) == [1] * 7 + [0]


# ------------------------------------------------------------------------- rendering


def test_display_pace_keeps_the_on_screen_speed():
    from aisoccer.graphics.pace import display_pace

    assert display_pace(1) == (2, 50)  # as before: 2 ticks per 0.02 s frame
    for scale in (1, 1.5, 2, 3):
        ticks_per_frame, fps = display_pace(scale)
        # px per second on screen = px per tick * ticks per second: unchanged.
        assert scale * ticks_per_frame * fps == 100
