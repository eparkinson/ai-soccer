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
