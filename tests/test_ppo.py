import numpy as np

from aisoccer.brains.PPOBrain import OBS_DIM, PPOBrain, player_features
from aisoccer.brains.SimpleBrain import SimpleBrain
from aisoccer.constants import Constants
from aisoccer.game import Game
from aisoccer.ppo import MLP, PPO, gae, gaussian_log_prob


def numerical_grad(f, param, eps=1e-6):
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = param[i]
        param[i] = old + eps
        up = f()
        param[i] = old - eps
        down = f()
        param[i] = old
        grad[i] = (up - down) / (2 * eps)
    return grad


def test_mlp_backward_matches_numerical_gradient():
    rng = np.random.default_rng(0)
    net = MLP([4, 6, 5, 3], rng)
    x = rng.normal(size=(7, 4))
    target = rng.normal(size=(7, 3))

    def loss():
        return 0.5 * ((net(x) - target) ** 2).sum()

    out, acts = net.forward(x)
    grads = net.backward(acts, out - target)
    for param, grad in zip(net.params, grads):
        assert np.allclose(grad, numerical_grad(loss, param), atol=1e-6)


def test_ppo_policy_gradient_matches_numerical_gradient():
    """One PPO step's analytic gradient equals the numerical gradient of the clipped loss."""
    rng = np.random.default_rng(1)
    ppo = PPO(5, 2, hidden=(8,), entropy_coef=0.01, max_grad_norm=1e9, seed=1)
    obs = rng.normal(size=(20, 5))
    actions = rng.normal(size=(20, 2))
    # Old log probs a little off the current ones so some ratios are clipped.
    old = gaussian_log_prob(actions, ppo.policy(obs), ppo.log_std) + rng.normal(
        0, 0.3, 20
    )
    adv = rng.normal(size=20)

    def loss():
        lp = gaussian_log_prob(actions, ppo.policy(obs), ppo.log_std)
        ratio = np.exp(lp - old)
        clipped = np.clip(ratio, 1 - ppo.clip, 1 + ppo.clip)
        return (
            -np.minimum(ratio * adv, clipped * adv).mean()
            - ppo.entropy_coef * ppo.log_std.sum()
        )

    params = ppo.policy.params + [ppo.log_std]
    expected = [numerical_grad(loss, p) for p in params]

    captured = []
    ppo.policy_opt.step = captured.append  # capture the gradients instead of stepping
    ppo._policy_step(
        obs,
        actions,
        old,
        adv,
        {k: [] for k in ("policy_loss", "clip_frac", "approx_kl")},
    )

    for grad, num in zip(captured[0], expected):
        assert np.allclose(grad, num, atol=1e-5)


def test_gae_without_discounting_sums_future_rewards():
    rewards = np.array([1.0, 0.0, 2.0])
    values = np.zeros(3)
    dones = np.array([0.0, 0.0, 1.0])
    adv, ret = gae(rewards, values, dones, 5.0, gamma=1.0, lam=1.0)
    assert np.allclose(ret, [3.0, 2.0, 2.0])  # no bootstrapping past the end


def test_player_features_shape_and_role():
    rng = np.random.default_rng(2)
    obs = player_features(
        rng.uniform(0, 800, (5, 2)),
        rng.normal(size=(5, 2)),
        rng.uniform(0, 800, (5, 2)),
        rng.normal(size=(5, 2)),
        np.array([900.0, 400.0]),
        np.array([1.0, -1.0]),
        1,
        0,
        0.5,
    )
    assert obs.shape == (5, OBS_DIM)
    assert np.all(np.isfinite(obs))
    # Each row carries its player's role as a one-hot.
    assert np.array_equal(obs[:, 46:51], np.eye(5))


def test_training_brain_records_a_trajectory():
    ppo = PPO(OBS_DIM, 2, seed=3)
    learner = PPOBrain("learner", weights=ppo.weights(), training=True)
    Game(learner, SimpleBrain(), game_length=200, quiet_mode=True, seed=4).play()
    traj = learner.finish_trajectory(gamma=0.99)

    steps = len(traj["rew"])
    # A decision every ACTION_REPEAT ticks, plus a fresh one after each goal.
    assert steps >= 200 // PPOBrain.ACTION_REPEAT
    assert traj["obs"].shape == (steps, Constants.NUM_PLAYERS, OBS_DIM)
    assert traj["act"].shape == (steps, Constants.NUM_PLAYERS, 2)
    assert traj["logp"].shape == traj["val"].shape == (steps, Constants.NUM_PLAYERS)
    assert np.all(np.isfinite(traj["rew"]))


def test_clone_fits_demonstrated_actions():
    rng = np.random.default_rng(5)
    obs = rng.normal(size=(2000, 6))
    actions = np.tanh(obs[:, :2] - obs[:, 2:4])  # a smooth target to imitate
    ppo = PPO(6, 2, hidden=(32,), minibatch=256, seed=5)
    losses = ppo.clone(obs, actions, epochs=20, lr=3e-3)
    assert losses[-1] < 0.1 * losses[0]


def test_value_only_update_leaves_policy_unchanged():
    rng = np.random.default_rng(6)
    ppo = PPO(5, 2, hidden=(8,), seed=6)
    before = [p.copy() for p in ppo.policy.params] + [ppo.log_std.copy()]
    obs = rng.normal(size=(64, 5))
    actions = rng.normal(size=(64, 2))
    logp = gaussian_log_prob(actions, ppo.policy(obs), ppo.log_std)
    ppo.update(
        obs, actions, logp, rng.normal(size=64), rng.normal(size=64), train_policy=False
    )
    after = ppo.policy.params + [ppo.log_std]
    assert all(np.array_equal(b, a) for b, a in zip(before, after))


def test_shaping_rewards_ball_control_and_progress():
    from aisoccer.brains.PPOBrain import shaping_potential

    mine = np.array([[900.0, 400.0]] + [[100.0, 100.0]] * 4)
    theirs_far = np.array([[1500.0, 700.0]] * 5)
    theirs_near = np.array([[905.0, 400.0]] + [[1500.0, 700.0]] * 4)
    ball = np.array([910.0, 400.0])

    # Having the ball to ourselves beats contesting it.
    assert shaping_potential(mine, theirs_far, ball) > shaping_potential(
        mine, theirs_near, ball
    )
    # The same situation further up the field is worth more.
    shift = np.array([400.0, 0.0])
    assert shaping_potential(
        mine + shift, theirs_far + shift, ball + shift
    ) > shaping_potential(mine, theirs_far, ball)


def test_update_stops_early_once_policy_moves_past_target_kl():
    rng = np.random.default_rng(7)
    obs = rng.normal(size=(256, 5))
    actions = rng.normal(size=(256, 2))
    adv = rng.normal(size=256)
    ret = rng.normal(size=256)

    def epochs_run(target_kl):
        ppo = PPO(
            5,
            2,
            hidden=(8,),
            lr=1e-2,
            epochs=8,
            minibatch=64,
            target_kl=target_kl,
            seed=7,
        )
        logp = gaussian_log_prob(actions, ppo.policy(obs), ppo.log_std)
        return ppo.update(obs, actions, logp, adv, ret)["epochs"]

    assert epochs_run(None) == 8
    assert epochs_run(1e-6) == 1


def test_exploration_noise_never_drops_below_floor():
    rng = np.random.default_rng(8)
    ppo = PPO(5, 2, hidden=(8,), lr=0.5, min_log_std=np.log(0.25), seed=8)
    obs = rng.normal(size=(64, 5))
    actions = ppo.policy(obs)  # actions at the mean: likelihood says shrink the noise
    logp = gaussian_log_prob(actions, ppo.policy(obs), ppo.log_std)
    for _ in range(5):
        ppo.update(obs, actions, logp, np.ones(64), np.zeros(64))
    assert np.all(np.exp(ppo.log_std) >= 0.25 - 1e-12)


def test_new_potential_terms():
    from aisoccer.brains.PPOBrain import potential_terms

    bunched = np.array([[900.0, 400.0]] * 5) + np.arange(5)[:, None]
    spread = np.array(
        [
            [300.0, 100.0],
            [300.0, 700.0],
            [900.0, 400.0],
            [1300.0, 150.0],
            [1300.0, 650.0],
        ]
    )
    opp = np.array([[100.0, 100.0]] * 5)
    deep = np.array([1600.0, 400.0])

    assert (
        potential_terms(spread, opp, deep)["spread"]
        > potential_terms(bunched, opp, deep)["spread"]
    )
    assert potential_terms(spread, opp, deep)["final_third"] > 0
    assert potential_terms(spread, opp, np.array([900.0, 400.0]))["final_third"] == 0
    # Heading into the goal mouth is a shot; heading wide or away is not.
    assert potential_terms(spread, opp, deep, np.array([8.0, 0.0]))["shot"] > 0
    assert potential_terms(spread, opp, deep, np.array([8.0, 8.0]))["shot"] == 0
    assert potential_terms(spread, opp, deep, np.array([-8.0, 0.0]))["shot"] == 0


def test_mixed_team_uses_each_roles_policy(tmp_path):
    from aisoccer.brains.PPOBrain import OBS_DIM

    rng = np.random.default_rng(9)
    sources = [PPO(OBS_DIM, 2, seed=s).weights() for s in range(5)]
    mixed = {
        "policy": sources[0]["policy"],
        "log_std": sources[0]["log_std"],
        "role_policies": [w["policy"] for w in sources],
    }
    path = tmp_path / "mixed.npz"
    PPOBrain.save_weights(path, mixed)
    brain = PPOBrain("mixed", weights=PPOBrain.load_weights(path))

    view = (
        rng.uniform(0, 800, (5, 2)),
        rng.normal(size=(5, 2)),
        rng.uniform(0, 800, (5, 2)),
        rng.normal(size=(5, 2)),
        np.array([900.0, 400.0]),
        np.array([1.0, 0.0]),
        0,
        0,
        0.3,
    )
    action = brain.move(*view)
    obs = player_features(*view)
    for r in range(5):
        expected = PPOBrain("one", weights=sources[r]).policy(obs[[r]])[0]
        assert np.allclose(action[r], expected)


import aisoccer.brains.PPOBrain as ppobrain  # noqa: E402


def reference_player_features(
    my_pos, my_vel, opp_pos, opp_vel, ball_pos, ball_vel, my_score, opp_score, game_time
):
    """
    One observation row per player, from that player's point of view (a 5 x OBS_DIM
    matrix). All five players share one policy network, so each row carries the player's
    role (its index) and everything relative to the player itself.
    """
    my_pos = np.asarray(my_pos, dtype=float)
    my_vel = np.asarray(my_vel, dtype=float)
    opp_pos = np.asarray(opp_pos, dtype=float)
    opp_vel = np.asarray(opp_vel, dtype=float)
    ball_pos = np.asarray(ball_pos, dtype=float).reshape(2)
    ball_vel = np.asarray(ball_vel, dtype=float).reshape(2)

    def norm_pos(p):
        return np.stack([p[..., 0] / ppobrain.L * 2 - 1, p[..., 1] / ppobrain.H * 2 - 1], axis=-1)

    ball_rel = ball_pos[None, :] - my_pos  # (5, 2)
    ball_dist = np.linalg.norm(ball_rel, axis=1)

    team_rel = my_pos[None, :, :] - my_pos[:, None, :]  # [i, j] = pos j - pos i
    team_rel = np.take_along_axis(team_rel, ppobrain.TEAMMATES[:, :, None], axis=1)
    team_vel = my_vel[ppobrain.TEAMMATES]

    opp_rel = opp_pos[None, :, :] - my_pos[:, None, :]
    order = np.argsort((opp_rel**2).sum(axis=2), axis=1)  # nearest opponent first
    opp_rel = np.take_along_axis(opp_rel, order[:, :, None], axis=1)
    opp_v = opp_vel[order]

    closer_teammates = (ball_dist[None, :] < ball_dist[:, None]).sum(axis=1) / (ppobrain.NUM - 1)
    score_diff = np.clip(my_score - opp_score, -3, 3) / 3.0

    n = my_pos.shape[0]
    return np.concatenate(
        [
            norm_pos(my_pos),
            my_vel / ppobrain.Constants.MAX_PLAYER_VELOCITY,
            ball_rel / ppobrain.REL_SCALE,
            np.broadcast_to(norm_pos(ball_pos), (n, 2)),
            np.broadcast_to(ball_vel / ppobrain.Constants.MAX_BALL_VELOCITY, (n, 2)),
            team_rel.reshape(n, -1) / ppobrain.REL_SCALE,
            team_vel.reshape(n, -1) / ppobrain.Constants.MAX_PLAYER_VELOCITY,
            opp_rel.reshape(n, -1) / ppobrain.REL_SCALE,
            opp_v.reshape(n, -1) / ppobrain.Constants.MAX_PLAYER_VELOCITY,
            ppobrain.ROLES[:n],
            closer_teammates[:, None],
            np.full((n, 1), score_diff),
            np.full((n, 1), game_time),
        ],
        axis=1,
    )



def test_fast_player_features_match_reference_exactly():
    """The optimised features must be bit-identical: every trained network depends on them."""
    rng = np.random.default_rng(10)
    for k in range(3000):
        args = (
            rng.uniform(0, 1800, (5, 2)),
            rng.normal(0, 3, (5, 2)),
            rng.uniform(0, 1800, (5, 2)),
            rng.normal(0, 3, (5, 2)),
            rng.uniform(0, 1800, 2),
            rng.normal(0, 5, 2),
            int(rng.integers(0, 5)),
            int(rng.integers(0, 5)),
            float(rng.random()),
        )
        if k % 3 == 0:  # exact ties in distances, as at kick-off
            args = (np.round(args[0], -2), args[1], np.round(args[2], -2), *args[3:])
        assert np.array_equal(player_features(*args), reference_player_features(*args))
