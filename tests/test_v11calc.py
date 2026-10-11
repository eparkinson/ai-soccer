import numpy as np
import pytest

from aisoccer.brainspec import load_brain
from aisoccer.constants import Constants as C
from aisoccer.game import Game
from aisoccer.llm.v11 import KICK_GAIN
from aisoccer.llm.v11calc import LLMBrainV11Calc, contact_normal, displacement_limit, solve_move

A, V = C.MAX_ACCELERATION, C.MAX_PLAYER_VELOCITY


def follow(p0, v0, u, K):
    """The game's rule: each tick velocity += (u - velocity) capped at A, then position += velocity."""
    p, v = np.array(p0, float), np.array(v0, float)
    for _ in range(K):
        a = u - v
        n = np.hypot(*a)
        if n > A:
            a = a / n * A
        v = v + a
        p = p + v
    return p, v


def test_one_tick_from_rest():
    ux, uy, vx, vy, ok = solve_move(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0)
    assert ok and np.allclose([ux, uy], [1, 0]) and np.allclose([vx, vy], [1, 0])


def test_drifting_needs_no_change():
    v0 = np.array([3.0, -2.0])
    ux, uy, vx, vy, ok = solve_move(0.0, 0.0, *v0, *(7 * v0), 7.0)
    assert ok and np.allclose([ux, uy], v0) and np.allclose([vx, vy], v0)


def test_full_acceleration_limit():
    K = 10
    assert displacement_limit(K) == pytest.approx(A * 55)
    assert not solve_move(0.0, 0.0, 0.0, 0.0, A * 55 + 1, 0.0, float(K))[-1]
    assert solve_move(0.0, 0.0, 0.0, 0.0, A * 6 - 0.1, 0.0, 3.0)[-1]  # 3 ticks at full rate, under top speed
    assert not solve_move(0.0, 0.0, 0.0, 0.0, A * 55 - 1, 0.0, float(K))[-1]  # needs more than top speed
    ok = solve_move(0.0, 0.0, 0.0, 0.0, 0.0, 200.0, 10.0)[-1]
    assert not ok  # 82.5 px is the most in 10 ticks from rest


@pytest.mark.parametrize("seed", range(20))
def test_solved_move_lands_on_target(seed):
    rng = np.random.default_rng(seed)
    p0 = rng.random(2) * 500
    v0 = (rng.random(2) - 0.5) * 2 * V / np.sqrt(2)
    K = int(rng.integers(1, 40))
    X = p0 + K * v0 + (rng.random(2) - 0.5) * displacement_limit(K)
    ux, uy, vKx, vKy, ok = solve_move(*p0, *v0, *X, float(K))
    if not ok:
        return
    p, v = follow(p0, v0, np.array([ux, uy]), K)
    assert np.allclose(p, X, atol=1e-6) and np.allclose(v, [vKx, vKy], atol=1e-9)


def test_contact_normal_ball_at_rest_is_the_aim():
    d = np.array([np.cos(0.5), np.sin(0.5)])
    n = contact_normal(0.0, 0.0, 5.0, 0.0, *d)
    assert np.allclose(n, d)


@pytest.mark.parametrize("seed", range(20))
def test_contact_normal_sends_ball_along_aim(seed):
    rng = np.random.default_rng(seed)
    b = (rng.random(2) - 0.5) * 10
    w = (rng.random(2) - 0.5) * 30
    ang = rng.random() * 2 * np.pi
    d = np.array([np.cos(ang), np.sin(ang)])
    n = np.array(contact_normal(*b, *w, *d))
    assert np.hypot(*n) == pytest.approx(1.0)
    out = b + KICK_GAIN * (w @ n) * n
    q = np.array([-d[1], d[0]])
    s = -(b @ q + 0.5 * KICK_GAIN * (w @ q)) / (0.5 * KICK_GAIN * np.hypot(*w))
    if abs(s) < 0.99 and out @ d > 0:  # d reachable: the ball leaves along it, and we close
        assert abs(out @ q) < 1e-6 * max(1.0, np.hypot(*out))
        assert w @ n >= 0


def test_brain_loads_and_plays():
    brain = load_brain("brain:aisoccer.llm.v11calc.LLMBrainV11Calc")
    assert isinstance(brain, LLMBrainV11Calc)
    game = Game(brain, load_brain("DefendersAndAttackers"), game_length=300, quiet_mode=True, seed=5)
    score = game.play()
    assert set(score) == {"blue", "red"}


def test_lite_loads_and_plays():
    brain = load_brain("brain:aisoccer.llm.v11calc.LLMBrainV11CalcLite")
    game = Game(brain, load_brain("DefendersAndAttackers"), game_length=300, quiet_mode=True, seed=6)
    game.play()
    assert brain._plan is not None


def test_planned_kick_happens_as_planned():
    """Follow a plan open-loop in the real physics: the ball leaves as the plan says."""
    from aisoccer.llm.v11 import predict_ball
    from aisoccer.llm.v11calc import plan_kicks_calc, quick_value, touches_early

    rng = np.random.default_rng(0)
    errors = []
    for _ in range(40):
        game = Game(load_brain("DefendersAndAttackers"), load_brain("DefendersAndAttackers"), game_length=0,
                    quiet_mode=True, seed=1)
        bodies = [p.body for t in game.teams for p in t.players]
        for j, b in enumerate(bodies):  # everyone else out of the way, at the far end
            b.position, b.velocity = np.array([1500.0 + 25 * j, 60.0 + 70 * j]), np.zeros(2)
        me, ball = bodies[1], game.ball.body
        me.position, me.velocity = np.array([400.0, 400.0]) + rng.random(2) * 300, (rng.random(2) - 0.5) * 6
        ball.position, ball.velocity = np.array([500.0, 300.0]) + rng.random(2) * 200, (rng.random(2) - 0.5) * 16
        traj, tvel, _ = predict_ball(ball.position, ball.velocity)
        pl, K, ux, uy, Bx, By, ax, ay, ok = plan_kicks_calc(me.position[None], me.velocity[None], traj, tvel)
        ok &= ~touches_early(me.position[0] + 0 * K, me.position[1] + 0 * K, me.velocity[0] + 0 * K,
                             me.velocity[1] + 0 * K, ux, uy, K, traj)
        if not ok.any() or np.linalg.norm(me.position - ball.position) < 50:
            continue
        i = int(np.argmax(np.where(ok, quick_value(Bx, By, ax, ay, K), -np.inf)))
        u, before = np.array([ux[i], uy[i]]), ball.velocity.copy()
        for _ in range(int(K[i]) + 2):
            a = u - me.velocity
            me.velocity = me.velocity + a * min(1.0, A / max(np.hypot(*a), 1e-9))
            game.limit_velocities()
            game.state.tick(C.SUBSTEPS)
            if np.linalg.norm(ball.velocity - before) > 1e-6 and np.linalg.norm(ball.position - me.position) < 40:
                break
            before = ball.velocity.copy()
        got, planned = ball.velocity, np.array([ax[i], ay[i]])
        errors.append(np.degrees(abs(np.arctan2(planned[0] * got[1] - planned[1] * got[0], planned @ got))))
    assert len(errors) >= 10
    assert np.median(errors) < 1.0
