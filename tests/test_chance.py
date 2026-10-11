"""The "chance" reward term: the chance-to-score potential and its xG model (fit_xg.py)."""

import json

import numpy as np
import pytest

import fit_xg
import train_ppo
from aisoccer.brains.PPOBrain import (
    PPOBrain,
    XG_FEATURES,
    batch_chance,
    chance_potential,
    chance_values,
    hand_chance,
    load_xg_model,
    potential_terms,
    shaping_potential,
    shot_features,
    xg_probability,
)
from aisoccer.brains.SimpleBrain import SimpleBrain
from aisoccer.constants import Constants
from aisoccer.game import Game
from aisoccer.vecgame import VecGame
from tests.test_vecgame import trained_weights

L, H = Constants.FIELD_LENGTH, Constants.FIELD_HEIGHT
GOAL_LINE = L - 1 - Constants.GOAL_DEPTH
FAR = np.array([[100.0, 100.0], [100.0, 700.0], [300.0, 100.0], [300.0, 700.0], [500.0, 400.0]])

# A made-up xG model with sensible signs: closer, more central, faster, fewer blockers
# and a keeper further away all raise the chance.
MODEL = {
    "features": list(XG_FEATURES),
    "mean": [600.0, 100.0, 8.0, 0.5, 200.0],
    "scale": [400.0, 60.0, 4.0, 0.7, 150.0],
    "coef": [-1.2, -0.3, 0.8, -0.6, 0.5],
    "intercept": -1.5,
}
# The same on fit_xg's default inputs: distance and speed on a log scale.
LOG_MODEL = dict(MODEL, features=["log_distance", "offset", "log_speed", "blockers", "keeper"])
LOG_MODEL.update(mean=[6.0, 100.0, 2.0, 0.5, 200.0], scale=[1.0, 60.0, 0.7, 0.7, 150.0])


def one(ball, vel, defenders=FAR, model=None):
    """The chance (for the goal on the right) of one ball."""
    return chance_values(np.array([ball], float), np.array([vel], float), defenders[None], model)[0]


def random_states(k, seed):
    rng = np.random.default_rng(seed)
    my_pos = rng.uniform((0, 0), (L, H), (k, 5, 2))
    opp_pos = rng.uniform((0, 0), (L, H), (k, 5, 2))
    ball_pos = rng.uniform((0, 0), (L, H), (k, 2))
    ball_vel = rng.normal(0, 6, (k, 2))
    ball_vel[::7] = 0.0
    ball_vel[1::9, 1] = 0.0  # straight at a goal
    return my_pos, opp_pos, ball_pos, ball_vel


@pytest.mark.parametrize("model", [None, MODEL, LOG_MODEL])
def test_chance_is_zero_off_target(model):
    assert one([1500, 400], [-10, 0], model=model) == 0.0  # going away from their goal
    assert one([1500, 400], [0, 10], model=model) == 0.0  # across the field
    assert one([1500, 400], [0, 0], model=model) == 0.0  # still
    assert one([1500, 400], [10, 10], model=model) == 0.0  # wide: meets the line at y 679
    assert one([1000, 100], [10, 0], model=model) == 0.0  # at the wall beside the mouth
    # Off target both ways: our chance minus theirs is 0.
    assert chance_potential(FAR, L - 1 - FAR, [900, 400], [0, 10], model) == 0.0


@pytest.mark.parametrize("model", [None, MODEL, LOG_MODEL])
def test_chance_on_target_grows_closer_and_faster(model):
    far, near = one([900, 400], [12, 0], model=model), one([1500, 400], [12, 0], model=model)
    slow, fast = one([1300, 400], [3, 0], model=model), one([1300, 400], [15, 0], model=model)
    assert 0 < far < near <= 1
    assert 0 < slow < fast <= 1
    xs = np.linspace(200, GOAL_LINE - 10, 30)
    by_distance = [one([x, 450], [10, 0], model=model) for x in xs]
    assert np.all(np.diff(by_distance) > 0)
    by_speed = [one([1200, 350], [v, 0], model=model) for v in np.linspace(0.5, 15, 30)]
    assert np.all(np.diff(by_speed) > 0)


def test_bounce_off_a_side_wall_can_be_on_target():
    # From (1000, 100) aiming down: meets the line at y = 100 - 0.5 * 779 < 8 without the
    # wall, and at 2 * 8 - that = 305.5 after one bounce: inside the mouth (200 .. 600).
    on_target, features = shot_features(np.array([[1000.0, 100.0]]), np.array([[10.0, -5.0]]), FAR[None])
    assert on_target[0]
    assert features[0, 1] == pytest.approx(abs(2 * 8 - (100 - 0.5 * (GOAL_LINE - 1000)) - H / 2))
    # Steeper: after the bounce it still crosses outside the mouth.
    assert one([1000, 100], [10, -12]) == 0.0


def test_blockers_and_keeper():
    ball, vel = [1200.0, 400.0], [12.0, 0.0]
    clear = one(ball, vel)
    in_path = FAR.copy()
    in_path[0] = [1500, 430]  # 30 px off the path: in reach
    behind = FAR.copy()
    behind[0] = [1150, 400]  # behind the ball: does not block
    assert one(ball, vel, in_path) == pytest.approx(clear / 2)  # the hand-made chance halves
    assert one(ball, vel, behind) == clear
    _, f = shot_features(np.array([ball]), np.array([vel]), in_path[None])
    assert f[0, 3] == 1 and f[0, 4] == pytest.approx(np.hypot(GOAL_LINE - 1500, 30))
    # A defender beside the path of a bounced shot blocks it through its mirror image.
    bounced = FAR.copy()
    bounced[0] = [1300, 20]  # near where the ball from (1000, 100) hits the bottom wall
    _, f = shot_features(np.array([[1000.0, 100.0]]), np.array([[10.0, -5.0]]), bounced[None])
    assert f[0, 3] == 1


def test_ours_minus_theirs_is_antisymmetric():
    my_pos, opp_pos, ball_pos, ball_vel = random_states(300, 3)
    ball_pos = np.round(ball_pos)  # whole pixels mirror exactly
    my_pos, opp_pos = np.round(my_pos), np.round(opp_pos)

    def flip(p):
        p = p.copy()
        p[..., 0] = L - 1 - p[..., 0]
        return p

    vel_m = ball_vel.copy()
    vel_m[:, 0] = -vel_m[:, 0]
    for model in (None, MODEL, LOG_MODEL):
        ours = batch_chance(my_pos, opp_pos, ball_pos, ball_vel, model)
        theirs = batch_chance(flip(opp_pos), flip(my_pos), flip(ball_pos), vel_m, model)
        assert (ours != 0).sum() > 30
        np.testing.assert_allclose(ours, -theirs, atol=1e-12)


@pytest.mark.parametrize("model", [None, MODEL, LOG_MODEL])
def test_single_matches_batched_exactly(model):
    my_pos, opp_pos, ball_pos, ball_vel = random_states(1000, 11)
    batched = batch_chance(my_pos, opp_pos, ball_pos, ball_vel, model)
    assert (batched > 0).sum() > 50 and (batched < 0).sum() > 50
    reward = {"goal": 2.0, "progress": 0.3, "shot": 0.1, "chance": 2.0, "xg_model": model}
    for i in range(len(batched)):
        assert batched[i] == chance_potential(my_pos[i], opp_pos[i], ball_pos[i], ball_vel[i], model)
        without = shaping_potential(my_pos[i], opp_pos[i], ball_pos[i], {**reward, "chance": 0.0}, ball_vel[i])
        assert shaping_potential(my_pos[i], opp_pos[i], ball_pos[i], reward, ball_vel[i]) == without + 2.0 * batched[i]


def test_configs_without_chance_are_unchanged(monkeypatch):
    my_pos, opp_pos, ball_pos, ball_vel = random_states(300, 5)
    old_rewards = [
        {"goal": 2.0, "progress": 0.3, "control": 0.15, "chase": 0.1},
        {"goal": 2.0, "progress": 0.3, "control": 0.15, "chase": 0.1, "shot": 0.08, "spread": 0.0625},
        {"goal": 2.0, "progress": 0.0, "control": 0.0, "chase": 0.0, "xg_model": MODEL},  # a model but no weight
    ]
    for reward in old_rewards:
        for i in range(len(ball_pos)):
            terms = potential_terms(my_pos[i], opp_pos[i], ball_pos[i], ball_vel[i])
            expected = sum(reward.get(name, 0.0) * value for name, value in terms.items())
            assert shaping_potential(my_pos[i], opp_pos[i], ball_pos[i], reward, ball_vel[i]) == expected

    # Without a "chance" weight neither path computes the chance at all.
    def boom(*args, **kwargs):
        raise AssertionError("chance computed")

    monkeypatch.setattr("aisoccer.brains.PPOBrain.chance_potential", boom)
    monkeypatch.setattr("aisoccer.vecppo.batch_chance", boom)
    weights = trained_weights()
    brains = [PPOBrain(str(i), weights=weights, training=True, reward=old_rewards[i % 3]) for i in range(4)]
    VecGame(brains[:2], brains[2:], [1, 2], game_length=200).play()
    Game(
        PPOBrain("a", weights=weights, training=True, reward=old_rewards[2]),
        SimpleBrain(),
        game_length=200,
        quiet_mode=True,
        seed=3,
    ).play()


def test_chance_shaping_telescopes_over_a_game():
    """Discounted, the chance rewards of a game sum to gamma^(T-1) phi_last - phi_first."""
    gamma = 0.990025
    reward = {"goal": 2.0, "progress": 0.0, "control": 0.0, "chase": 0.0, "chance": 2.0}
    learner = PPOBrain("learner", weights=trained_weights(), training=True, reward=reward)
    Game(learner, SimpleBrain(), game_length=1800, quiet_mode=True, seed=7).play()
    phi = np.array(learner.trajectory["phi"])
    traj = learner.finish_trajectory(gamma)
    shaping = traj["rew"] - np.array(learner.trajectory["goal"])
    assert (phi != 0).sum() > 5  # the ball was on target now and then
    discount = gamma ** np.arange(len(shaping))
    assert (discount * shaping).sum() == pytest.approx(gamma ** (len(phi) - 1) * phi[-1] - phi[0], abs=1e-9)


def test_training_games_with_chance_match_play_training_game_exactly():
    """With a chance weight (hand-made and xG), the batched games still give the same trajectories."""
    trained = trained_weights()
    snapshot = PPOBrain.load_weights(PPOBrain.WEIGHTS_FILE)
    rewards = [
        {"goal": 2.0, "progress": 0.0, "control": 0.0, "chase": 0.0, "chance": 2.0},
        {"goal": 2.0, "progress": 0.3, "control": 0.0, "chase": 0.0, "chance": 1.0, "xg_model": MODEL},
    ]
    tasks = [
        (trained, opp, snapshot if opp == "snapshot" else None, 300 + i, 0.99, rewards[i % 2])
        for i, opp in enumerate(["self", "snapshot", "snapshot", "self"])
    ]
    batched = train_ppo.play_training_games(tasks)
    nonzero = 0
    for task, got in zip(tasks, batched):
        expected = train_ppo.play_training_game(task)
        assert got[:3] == expected[:3]
        for traj, ref in zip(got[3], expected[3]):
            for key in ref:
                assert np.array_equal(traj[key], ref[key]), key
            nonzero += int((traj["rew"] != 0).sum())
    assert nonzero > 0


def test_train_ppo_accepts_an_xg_model_key():
    class Args:
        pass

    args = Args()
    train_ppo.apply_config(args, {"reward": {"goal": 2.0, "chance": 2.0}, "xg_model": "runs/league3/xg.json"})
    assert args.xg_model == "runs/league3/xg.json"


def synthetic_events(n, seed):
    """Shots whose true chance falls with distance and rises with speed."""
    rng = np.random.default_rng(seed)
    x = np.column_stack(
        [
            rng.uniform(0, 1500, n),  # distance
            rng.uniform(0, 200, n),  # offset
            rng.uniform(0.5, 15, n),  # speed
            rng.integers(0, 4, n).astype(float),  # blockers
            rng.uniform(0, 600, n),  # keeper
        ]
    )
    z = 0.5 - 0.004 * x[:, 0] + 0.25 * x[:, 2] - 0.8 * x[:, 3] + 0.002 * x[:, 4] - 2.0
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    return x, y


def test_fitted_xg_model_is_sensible(tmp_path):
    x, y = synthetic_events(4000, 1)
    model = fit_xg.fit_model(x, y, np.arange(len(y)) // 10, l2=1.0, names=XG_FEATURES)
    fit = model["fit"]
    assert fit["log_loss"] < fit["base_rate_log_loss"]
    assert model["odds"]["distance"]["odds_multiplier"] < 1 < model["odds"]["speed"]["odds_multiplier"]
    assert model["odds"]["blockers"]["odds_multiplier"] < 1
    # Close to the true effects: 0.4 per 100 px of distance, 0.25 per px/tick of speed.
    assert np.log(model["odds"]["distance"]["odds_multiplier"]) == pytest.approx(-0.4, abs=0.1)
    assert np.log(model["odds"]["speed"]["odds_multiplier"]) == pytest.approx(0.25, abs=0.06)

    path = tmp_path / "xg.json"
    path.write_text(json.dumps(model))
    loaded = load_xg_model(path)
    assert load_xg_model(tmp_path / "missing.json") is None
    p = xg_probability(x, loaded)
    assert np.all((p >= 0) & (p <= 1))
    z = (x - np.array(model["mean"])) / np.array(model["scale"])
    np.testing.assert_allclose(p, 1 / (1 + np.exp(-(model["intercept"] + z @ np.array(model["coef"])))))
    # Monotone: closer and faster shots have larger chances.
    base = np.array([[800.0, 50.0, 8.0, 1.0, 200.0]])
    closer, faster = base.copy(), base.copy()
    closer[0, 0], faster[0, 2] = 300.0, 14.0
    assert xg_probability(closer, loaded)[0] > xg_probability(base, loaded)[0] < xg_probability(faster, loaded)[0]
    # fit_xg's default inputs (log distance and speed) fit these too, and load.
    default = fit_xg.fit_model(x, y, np.arange(len(y)) // 10, l2=1.0)
    assert default["features"] == list(fit_xg.DEFAULT_FEATURES)
    assert default["fit"]["log_loss"] < default["fit"]["base_rate_log_loss"]
    path.write_text(json.dumps(default))
    p_default = xg_probability(x, load_xg_model(path))
    assert np.all((p_default >= 0) & (p_default <= 1))
    assert default["odds"]["log_distance"]["odds_multiplier"] < 1 < default["odds"]["log_speed"]["odds_multiplier"]
    # Extreme inputs saturate at 0 and 1.
    extreme = np.array([[0.0, 0.0, 1e6, 0.0, 1e6], [1e6, 1e3, 0.0, 5.0, 0.0]])
    assert xg_probability(extreme, loaded).tolist() == [1.0, 0.0]


def test_hand_chance_is_a_probability():
    _, _, ball_pos, ball_vel = random_states(500, 2)
    on_target, f = shot_features(ball_pos, ball_vel, np.broadcast_to(FAR, (500, 5, 2)))
    p = hand_chance(f[on_target])
    assert np.all((p >= 0) & (p <= 1)) and (p > 0).mean() > 0.9


def test_shot_events_from_a_short_batch():
    result = fit_xg.play_batch(
        ([("SimpleBrain", "BehindAndTowards"), ("BehindAndTowards", "SimpleBrain")], [1, 2], [0, 1], 900, 300)
    )
    n = len(result["labels"])
    assert result["features"].shape == (n, len(XG_FEATURES))
    assert set(np.unique(result["labels"])) <= {0.0, 1.0}
    assert result["scored_events"] <= result["goals"]
    assert set(result["game"]) <= {0, 1}
