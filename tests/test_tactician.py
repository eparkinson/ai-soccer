import json

import numpy as np

from aisoccer.brainspec import load_brain
from aisoccer.game import Game
from aisoccer.tactics.learned import (
    ACT_DIM,
    DECISION_INTERVAL,
    OBS_DIM,
    TacticianBrain,
    opponent_style,
    team_features,
)
from aisoccer.tactics.v1 import DEFAULT_STRATEGY, STRATEGY, TacticsV1
from train_tactician import build_batch, make_ppo, parse_args


def random_state(rng):
    pos = lambda: rng.uniform([0, 0], [1000, 600], (5, 2))  # noqa: E731
    vel = lambda: rng.normal(0, 2, (5, 2))  # noqa: E731
    return pos(), vel(), pos(), vel(), rng.uniform([0, 0], [1000, 600]), rng.normal(0, 3, 2)


def trained_like_weights(seed=1):
    """A PPO learner as the trainer makes it, with its policy moved away from the start."""
    ppo = make_ppo(parse_args(["--seed", str(seed)]))
    rng = np.random.default_rng(seed)
    for p in ppo.policy.params:
        p += rng.normal(0, 0.3, p.shape)
    return {**ppo.weights(), "interval": DECISION_INTERVAL}


def test_features_shape():
    rng = np.random.default_rng(0)
    mp, mv, op, ov, b, bv = random_state(rng)
    obs = team_features(mp, mv, op, ov, b, bv, 2, 1, opponent_style(op))
    assert OBS_DIM == 35
    assert obs.shape == (OBS_DIM,)
    assert np.all(np.isfinite(obs))


def test_untrained_plays_default_strategy():
    brain = TacticianBrain()
    rng = np.random.default_rng(0)
    for _ in range(20):
        brain.ticks = 0
        mp, mv, op, ov, b, bv = random_state(rng)
        brain.move(mp, mv, op, ov, b, bv, 0, 0, rng.random())
        for k in STRATEGY:
            target = min(max(DEFAULT_STRATEGY[k], 0.02), 0.98)
            assert abs(brain.strategy[k] - target) < 0.01, k


def test_trainer_start_is_default_strategy():
    ppo = make_ppo(parse_args([]))
    brain = TacticianBrain(weights={**ppo.weights(), "interval": DECISION_INTERVAL})
    mp, mv, op, ov, b, bv = random_state(np.random.default_rng(3))
    brain.move(mp, mv, op, ov, b, bv, 0, 0, 0.0)
    assert np.allclose([brain.strategy[k] for k in STRATEGY],
                       [min(max(DEFAULT_STRATEGY[k], 0.02), 0.98) for k in STRATEGY], atol=0.01)


def test_strategy_in_unit_interval_when_sampling():
    brain = TacticianBrain(weights=trained_like_weights(), deterministic=False)
    brain.log_std = np.full(ACT_DIM, 1.0)
    rng = np.random.default_rng(2)
    for _ in range(50):
        brain.ticks = 0
        mp, mv, op, ov, b, bv = random_state(rng)
        brain.move(mp, mv, op, ov, b, bv, 0, 3, rng.random())
        values = np.array([brain.strategy[k] for k in STRATEGY])
        assert np.all((values >= 0) & (values <= 1))


def test_decides_every_interval():
    brain = TacticianBrain(weights=trained_like_weights(), deterministic=False)
    rng = np.random.default_rng(4)
    mp, mv, op, ov, b, bv = random_state(rng)
    seen = []
    for tick in range(2 * DECISION_INTERVAL):
        brain.move(mp, mv, op, ov, b, bv, 0, 0, 0.0)
        seen.append(dict(brain.strategy))
    assert seen[0] == seen[DECISION_INTERVAL - 1]
    assert seen[0] != seen[DECISION_INTERVAL]


def test_npz_round_trip(tmp_path):
    weights = trained_like_weights()
    TacticianBrain.save_weights(tmp_path / "p.npz", weights)
    loaded = TacticianBrain.load_weights(tmp_path / "p.npz")
    for a, b in zip(weights["policy"] + weights["value"], loaded["policy"] + loaded["value"]):
        assert np.array_equal(a, b)
    assert loaded["interval"] == DECISION_INTERVAL
    assert np.array_equal(loaded["log_std"], weights["log_std"])


def test_entrant_json_round_trip(tmp_path):
    weights = trained_like_weights()
    path = tmp_path / "entrant.json"
    TacticianBrain.write_entrant(path, weights, iteration=7)
    assert json.loads(path.read_text())["iteration"] == 7
    brain = load_brain(str(path))
    direct = TacticianBrain(weights=weights)
    assert isinstance(brain, TacticianBrain)
    assert brain.name == "entrant"
    mp, mv, op, ov, b, bv = random_state(np.random.default_rng(5))
    brain.move(mp, mv, op, ov, b, bv, 1, 0, 0.3)
    direct.move(mp, mv, op, ov, b, bv, 1, 0, 0.3)
    assert brain.strategy == direct.strategy


def test_training_game_and_batch():
    weights = trained_like_weights()
    learner = TacticianBrain("learner", weights=weights, training=True)
    score = Game(learner, TacticsV1(), game_length=300, quiet_mode=True, seed=0).play()
    traj = learner.finish_trajectory(0.96)
    decisions = len(traj["rew"])
    assert decisions >= 300 // DECISION_INTERVAL
    assert traj["obs"].shape == (decisions, 1, OBS_DIM)
    assert traj["act"].shape == (decisions, 1, ACT_DIM)
    obs, act, logp, adv, ret = build_batch([traj], 0.96, 0.9)
    assert obs.shape == (decisions, OBS_DIM) and act.shape == (decisions, ACT_DIM)
    assert np.all(np.isfinite(adv)) and np.all(np.isfinite(ret))
    assert sum(score.values()) >= 0
