"""Obs v3 (raw): the brain interface scaled, nothing derived; batched = one at a time."""
import numpy as np

from aisoccer.brains.PPOBrain import OBS_DIMS, RAW_DIM, PPOBrain, batch_raw_features, observation, obs_version_of
from aisoccer.game import Game
from aisoccer.ppo import PPO


def test_raw_rows_are_the_interface():
    rng = np.random.default_rng(0)
    view = (rng.uniform(0, 1000, (5, 2)), rng.normal(0, 3, (5, 2)), rng.uniform(0, 1000, (5, 2)),
            rng.normal(0, 3, (5, 2)), rng.uniform(0, 1000, 2), rng.normal(0, 5, 2), 2, 3, 0.25)
    obs = observation(view, 3)
    assert obs.shape == (5, RAW_DIM) == (5, OBS_DIMS[3])
    assert np.allclose(obs[:, :5], np.eye(5))
    assert np.allclose(obs[:, 5:], obs[0, 5:])  # everything else is the same for every player
    assert obs[0, -3:].tolist() == [0.2, 0.3, 0.25]
    many = batch_raw_features(*(np.stack([v, v]) if np.ndim(v) else np.array([v, v]) for v in view))
    assert np.allclose(many[1], obs)


def test_raw_brain_plays_and_batches():
    ppo = PPO(RAW_DIM, 2, hidden=(64, 64), seed=1)
    w = dict(ppo.weights(), obs_version=3, action_repeat=1)
    assert obs_version_of(w) == 3
    a, b = PPOBrain("a", weights=w, training=True), PPOBrain("b", weights=w)
    assert a.obs_dim == RAW_DIM and a.memory is None
    Game(a, b, game_length=200, quiet_mode=True, seed=3).play()
    # The brain itself must see the raw rows (2026-10-01: it fell back to v1 features here).
    view = (a.my_players_pos, a.my_players_vel, a.opp_players_pos, a.opp_players_vel,
            a.ball_pos, a.ball_vel, a.my_score, a.opp_score, a.game_time)
    assert np.array_equal(a.features(), observation(view, 3))
    assert np.asarray(a.trajectory["obs"]).shape[-1] == RAW_DIM


def test_noise_hold_keeps_a_sample_for_n_decisions():
    ppo = PPO(RAW_DIM, 2, hidden=(64, 64), seed=1)
    w = dict(ppo.weights(), obs_version=3, action_repeat=1)
    old = PPOBrain.NOISE_HOLD
    try:
        PPOBrain.NOISE_HOLD = 4
        b = PPOBrain("b", weights=w, training=True)
        draws = [b.exploration_noise((5, 2)).copy() for _ in range(9)]
        assert all(np.array_equal(draws[0], d) for d in draws[1:4])
        assert not np.array_equal(draws[3], draws[4])
        assert all(np.array_equal(draws[4], d) for d in draws[5:8])
        PPOBrain.NOISE_HOLD = 1
        c = PPOBrain("c", weights=w, training=True)
        assert not np.array_equal(c.exploration_noise((5, 2)), c.exploration_noise((5, 2)))
    finally:
        PPOBrain.NOISE_HOLD = old
