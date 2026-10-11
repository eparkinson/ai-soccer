"""Obs v2: opt-in extra PPO features, recorded in the weights file, the same in Game and VecGame."""

import argparse

import numpy as np
import pytest

import train_ppo
from aisoccer.brains import PPOBrain as ppobrain
from aisoccer.brains.PPOBrain import (
    OBS_DIM,
    OBS_DIMS,
    OBS_V2_NAMES,
    ObsMemory,
    PPOBrain,
    ScoreTrend,
    extra_features,
    obs_version_of,
    observation,
    player_features,
    time_left,
    widen_inputs,
)
from aisoccer.brains.SimpleBrain import SimpleBrain
from aisoccer.constants import Constants
from aisoccer.game import Game
from aisoccer.ppo import PPO
from tests.test_vecgame import assert_plays_like_game, trained_weights

L, H = Constants.FIELD_LENGTH, Constants.FIELD_HEIGHT
V2 = OBS_DIMS[2]


def v2_weights(seed=0):
    """The trained v1 network widened to v2, with random weights on the new inputs so they matter."""
    weights = widen_inputs(trained_weights(), 2)
    rng = np.random.default_rng(seed)

    def perturb(params):
        w = params[0].copy()
        w[OBS_DIM:] = rng.normal(0.0, 0.3, w[OBS_DIM:].shape)
        return [w] + params[1:]

    for key in ("policy", "value"):
        weights[key] = perturb(weights[key])
    if weights.get("role_policies"):  # the saved brain may be a mixed team
        weights["role_policies"] = [perturb(p) for p in weights["role_policies"]]
    return weights


def random_views(rng, k):
    views = [
        rng.uniform(0, L, (k, 5, 2)),
        rng.normal(0, 4, (k, 5, 2)),
        rng.uniform(0, L, (k, 5, 2)),
        rng.normal(0, 4, (k, 5, 2)),
        rng.uniform(0, L, (k, 2)),
        rng.normal(0, 8, (k, 2)),
    ]
    for a in (views[0], views[2]):
        a[..., 1] *= H / L
    views[4][:, 1] *= H / L
    for i, cap in ((1, Constants.MAX_PLAYER_VELOCITY), (3, Constants.MAX_PLAYER_VELOCITY),
                   (5, Constants.MAX_BALL_VELOCITY)):  # the game's speed caps
        speed = np.linalg.norm(views[i], axis=-1, keepdims=True)
        views[i] = np.where(speed > cap, views[i] / np.maximum(speed, 1e-9) * cap, views[i])
    views[5][::5] = 0.0  # a still ball
    views[5][::7, 0] = 0.0  # a ball moving straight up or down
    views[0][::11, 0] = views[4][::11]  # a player exactly on the ball
    return views


def test_names_and_dims():
    assert OBS_DIMS == {1: 54, 2: 54 + len(OBS_V2_NAMES), 3: 52} and len(OBS_V2_NAMES) == 26
    assert OBS_DIM == 54  # the default is unchanged


def test_old_weights_load_and_save_as_v1(tmp_path):
    for path in (train_ppo.HISTORY_DIR / "PPO-champ-4.npz", train_ppo.HISTORY_DIR / "PPO-clone.npz"):
        weights = PPOBrain.load_weights(path)
        if weights is None:
            continue
        assert "obs_version" not in weights and obs_version_of(weights) == 1
        brain = PPOBrain("old", weights=weights)
        assert brain.obs_version == 1 and brain.obs_dim == 54 and brain.memory is None
        out = tmp_path / "copy.npz"
        PPOBrain.save_weights(out, weights)
        assert "obs_version" not in np.load(out).files  # v1 files are written as before


def test_v1_brain_features_are_player_features():
    brain = PPOBrain("v1", weights=trained_weights())
    rng = np.random.default_rng(1)
    view = [v[0] for v in random_views(rng, 1)] + [2, 1, 0.3]
    brain.move(*view)
    assert np.array_equal(brain.last_obs, player_features(*view))


def test_v2_network_saves_and_loads_its_version(tmp_path):
    weights = v2_weights()
    assert obs_version_of(weights) == 2
    path = tmp_path / "v2.npz"
    PPOBrain.save_weights(path, weights)
    assert int(np.load(path)["obs_version"]) == 2
    loaded = PPOBrain.load_weights(path)
    assert loaded["obs_version"] == 2
    brain = PPOBrain("v2", weights=loaded)
    assert brain.obs_version == 2 and brain.obs_dim == V2
    # A dict that lost the key (e.g. league.promote saves policy and log_std only) is
    # recognised by its input width, and saved with the version again.
    bare = {"policy": weights["policy"], "log_std": weights["log_std"]}
    assert obs_version_of(bare) == 2
    PPOBrain.save_weights(tmp_path / "bare.npz", bare)
    assert int(np.load(tmp_path / "bare.npz")["obs_version"]) == 2
    with pytest.raises(ValueError):
        obs_version_of(dict(bare, obs_version=1))
    # A fresh v2 learner's networks are recognised too.
    assert obs_version_of(PPO(V2, 2, seed=0).weights()) == 2


def test_widened_network_plays_as_before():
    weights = trained_weights()
    wide = widen_inputs(weights, 2)
    assert wide["policy"][0].shape == (V2, weights["policy"][0].shape[1])
    rng = np.random.default_rng(2)
    view = [v[0] for v in random_views(rng, 1)] + [0, 0, 0.5]
    old, new = PPOBrain("old", weights=weights), PPOBrain("new", weights=wide)
    a, b = old.move(*view), new.move(*view)
    assert new.last_obs.shape == (5, V2)
    assert np.allclose(a, b, rtol=0, atol=1e-12)


def test_features_in_range_and_finite():
    rng = np.random.default_rng(3)
    k = 4000
    views = random_views(rng, k)
    left = rng.uniform(0, 1, k)
    trend = rng.integers(-6, 7, k)
    extra = extra_features(*views, left, trend)
    assert extra.shape == (k, 5, 26) and np.isfinite(extra).all()
    lo, hi = extra.min(axis=(0, 1)), extra.max(axis=(0, 1))
    for name, a, b in zip(OBS_V2_NAMES, lo, hi):
        assert -1.0 - 1e-12 <= a and b <= 1.2, (name, a, b)  # distances / L reach ~1.05
    # The indicators and the heading-dependent features are exercised both ways.
    col = {n: i for i, n in enumerate(OBS_V2_NAMES)}
    for name in ("ball_to_their_mouth", "ball_to_our_mouth"):
        assert set(np.unique(extra[..., col[name]])) == {0.0, 1.0}


def test_single_view_features_equal_batched_exactly():
    """Each view alone (as PPOBrain computes it) gives exactly its row of the batch."""
    rng = np.random.default_rng(4)
    k = 1500
    views = random_views(rng, k)
    left = time_left(rng.integers(0, 40000, k), rng.random(k))
    trend = rng.integers(-5, 6, k)
    batch = extra_features(*views, left, trend)
    for i in range(k):
        one = extra_features(*(v[[i]] for v in views), left[[i]], trend[[i]])
        assert np.array_equal(one[0], batch[i])


def test_time_left():
    for length in (300, 1800, 36000):
        ticks = np.arange(length)
        game_time = np.array([float(t) / float(length) for t in ticks])  # as Game computes it
        got = time_left(ticks, game_time)
        expected = np.minimum(length - ticks, ppobrain.TIME_WINDOW) / ppobrain.TIME_WINDOW
        expected[0] = 1.0  # tick 0: game_time 0 says nothing about the length
        assert np.array_equal(got, expected)
    assert time_left(np.array([500]), np.array([0.0]))[0] == 1.0  # an endless game


def test_score_trend_window():
    trend = ScoreTrend(2)
    w = ppobrain.TIME_WINDOW
    seats = np.array([0, 1])
    assert list(trend.update(seats, 0, [0, 0])) == [0, 0]
    assert list(trend.update(seats, 10, [1, 0])) == [1, 0]
    assert list(trend.update(seats, 20, [2, -1])) == [2, -1]
    assert list(trend.update(seats, 10 + w - 1, [2, -1])) == [2, -1]
    assert list(trend.update(seats, 10 + w, [2, -1])) == [1, -1]  # the first goal has left the window
    assert list(trend.update(seats[:1], 20 + w, [3])) == [1]  # a new goal while the second leaves
    assert list(trend.update(seats, 40 + w, [3, -1])) == [1, 0]
    assert list(trend.update(seats, 20 + 2 * w, [3, -1])) == [0, 0]


def test_memory_restarts_with_a_new_game():
    memory = ObsMemory()
    for t in range(5):
        memory.start_tick(t / 10)
        memory.end_tick()
    assert memory.tick == 5
    memory.start_tick(0.0)
    assert memory.tick == 0


def y_mirror_map():
    """For each feature: (source column, sign) of its value in the y-mirrored view."""
    odd = {1, 3, 5, 7, 9} | set(range(11, 18, 2)) | set(range(19, 26, 2)) | set(range(27, 36, 2))
    odd |= set(range(37, 46, 2))
    col = {n: OBS_DIM + i for i, n in enumerate(OBS_V2_NAMES)}
    odd |= {col[n] for n in ("goal_dir_y", "own_goal_dir_y", "ball_goal_dir_y", "ball_cross_y", "behind_ball_sin")}
    source = list(range(V2))
    source[col["post_lo_sin"]], source[col["post_hi_sin"]] = col["post_hi_sin"], col["post_lo_sin"]
    sign = np.array([-1.0 if i in odd or i in (col["post_lo_sin"], col["post_hi_sin"]) else 1.0 for i in range(V2)])
    return np.array(source), sign


def test_y_mirrored_view_gives_mirrored_features():
    rng = np.random.default_rng(5)
    source, sign = y_mirror_map()
    views = random_views(rng, 300)
    for i in range(300):
        view = [v[i] for v in views] + [int(rng.integers(3)), int(rng.integers(3)), 0.4]
        flipped = [view[0].copy(), view[1].copy(), view[2].copy(), view[3].copy(), view[4].copy(), view[5].copy()]
        for a in (flipped[0], flipped[2], flipped[4]):
            a[..., 1] = H - a[..., 1]
        for a in (flipped[1], flipped[3], flipped[5]):
            a[..., 1] = -a[..., 1]
        flipped += view[6:]
        memory_a, memory_b = ObsMemory(), ObsMemory()
        a = observation(view, 2, memory_a)
        b = observation(flipped, 2, memory_b)
        assert np.allclose(b, a[:, source] * sign, rtol=0, atol=1e-9)


def test_red_sees_the_mirrored_game():
    """
    Red's features in a game are blue's features in the mirror-image game (teams swapped,
    x mirrored), so both teams read their own goals and directions correctly.
    """
    features = {}

    class Spy(PPOBrain):
        def decide(self):
            super().decide()
            features.setdefault(self.name, []).append(self.last_obs)

    rng = np.random.default_rng(6)
    pos = rng.uniform(100, 700, (10, 2))
    vel = rng.normal(0, 3, (10, 2))
    ball_pos, ball_vel = np.array([700.0, 300.0]), np.array([4.0, -1.0])
    weights = v2_weights(1)
    for mirrored in (False, True):
        p, v, bp, bv = pos.copy(), vel.copy(), ball_pos.copy(), ball_vel.copy()
        if mirrored:  # the teams swap sides, and x flips
            p, v = np.concatenate([p[5:], p[:5]]), np.concatenate([v[5:], v[:5]])
            p[:, 0], v[:, 0] = L - 1 - p[:, 0], -v[:, 0]
            bp[0], bv[0] = L - 1 - bp[0], -bv[0]
        game = Game(Spy("blue", weights=weights), Spy("red", weights=weights), 50, True, seed=0)
        for i, player in enumerate(game.teams[0].players + game.teams[1].players):
            player.body.position, player.body.velocity = p[i], v[i]
        game.ball.body.position, game.ball.body.velocity = bp, bv
        game.tick()
    blue_plain, red_mirrored = features["blue"][0], features["red"][1]
    assert np.allclose(blue_plain, red_mirrored, rtol=0, atol=1e-9)
    col = {n: OBS_DIM + i for i, n in enumerate(OBS_V2_NAMES)}
    # In the plain game the ball (x 700) is nearer red's target (x 0) than blue's (x ~1780).
    red_plain = features["red"][0]
    assert blue_plain[0, col["ball_goal_dist"]] > red_plain[0, col["ball_goal_dist"]]
    assert np.isclose(blue_plain[0, col["ball_goal_dist"]], np.hypot(ppobrain.GOAL_X - 700, 100) / L)


def test_v2_games_match_game_tick_by_tick(monkeypatch):
    """
    v2 networks in VecGame play Game's games bit for bit. A short score-trend window
    makes goals leave it within the game, and time_left falls during it.
    """
    monkeypatch.setattr(ppobrain, "TIME_WINDOW", 200)
    v2 = v2_weights(2)
    # Against v1, v2 and an untrained v2 network, in one batch. make() is called once
    # per VecGame game, then once per Game game: game k gets others[k] both times.
    others = [trained_weights(), v2, PPO(V2, 2, seed=7).weights(), v2]
    games = []

    def make():
        games.append(None)
        return PPOBrain("a", weights=v2), PPOBrain("b", weights=others[len(games) % 4])

    vec = assert_plays_like_game(make, [40, 41, 42, 43], 900)
    assert vec.score.sum() > 0


def test_v2_training_trajectories_match_exactly(tmp_path, monkeypatch):
    """Self-play, a snapshot and v1 / v2 pool files: VecGame trajectories are play_training_game's."""
    monkeypatch.setattr(ppobrain, "TIME_WINDOW", 150)
    v2 = v2_weights(3)
    v1_file, v2_file = tmp_path / "v1.npz", tmp_path / "v2.npz"
    PPOBrain.save_weights(v1_file, trained_weights())
    PPOBrain.save_weights(v2_file, v2_weights(4))
    opponents = ["self", "snapshot", f"file:{v1_file}", f"file:{v2_file}"]
    old = train_ppo.GAME_LENGTH
    train_ppo.GAME_LENGTH = 600
    try:
        tasks = [(v2, opp, v2 if opp == "snapshot" else None, 300 + i, 0.995, None)
                 for i, opp in enumerate(opponents)]
        batched = train_ppo.play_training_games(tasks)
        goals = 0
        for task, got in zip(tasks, batched):
            expected = train_ppo.play_training_game(task)
            assert got[:3] == expected[:3]
            goals += got[1] + got[2]
            for traj, ref in zip(got[3], expected[3]):
                assert traj["obs"].shape[-1] == V2
                for key in ref:
                    assert np.array_equal(traj[key], ref[key]), key
    finally:
        train_ppo.GAME_LENGTH = old
    assert goals > 0


def test_v1_learner_against_v2_pool_file_keeps_v1_trajectories(tmp_path):
    v2_file = tmp_path / "v2.npz"
    PPOBrain.save_weights(v2_file, v2_weights(5))
    v1 = trained_weights()
    task = (v1, f"file:{v2_file}", None, 7, 0.995, None)
    old = train_ppo.GAME_LENGTH
    train_ppo.GAME_LENGTH = 200
    try:
        got = train_ppo.play_training_games([task])[0]
        expected = train_ppo.play_training_game(task)
    finally:
        train_ppo.GAME_LENGTH = old
    assert got[3][0]["obs"].shape[-1] == OBS_DIM
    for key in expected[3][0]:
        assert np.array_equal(got[3][0][key], expected[3][0][key])


def test_imitation_records_v2_observations(monkeypatch):
    monkeypatch.setattr(PPOBrain, "ACTION_REPEAT", 2)
    demonstrator = train_ppo.Demonstrator("SimpleBrain", obs_version=2)
    Game(demonstrator, SimpleBrain(), game_length=20, quiet_mode=True, seed=3).play()
    obs = np.array(demonstrator.obs)
    assert obs.shape == (10, 5, V2)
    assert np.array_equal(obs[..., :OBS_DIM].shape, (10, 5, OBS_DIM))
    student = train_ppo.DaggerStudent(dict(v2_weights(6), action_repeat=2), "SimpleBrain")
    Game(student, SimpleBrain(), game_length=10, quiet_mode=True, seed=4).play()
    assert np.array(student.obs).shape == (5, 5, V2)
    # The demo and DAgger game functions reshape to the version's width.
    obs, act = train_ppo.play_demo_game((1, "SimpleBrain", 2))
    assert obs.shape[1] == V2 and len(obs) == len(act)
    obs, act = train_ppo.play_demo_game((1, "SimpleBrain"))
    assert obs.shape[1] == OBS_DIM


def test_learner_follows_the_loaded_networks_version(tmp_path, monkeypatch):
    """--resume / --init-weights with another version's networks: the learner goes with them."""
    monkeypatch.setattr(train_ppo, "LOG_FILE", tmp_path / "progress.log")
    args = argparse.Namespace(obs_version=1, lr=1e-3)
    ppo = PPO(OBS_DIM, 2, seed=0)
    v2 = PPO(V2, 2, seed=1).weights()
    assert not train_ppo.load_networks(ppo, {"policy": v2["policy"], "log_std": v2["log_std"]})
    train_ppo.follow_obs_version(ppo, args, "v2.npz")
    assert args.obs_version == 2 and ppo.value.params[0].shape[0] == V2  # a fresh critic to match
    ppo.sync_dtype()
    rng = np.random.default_rng(0)
    n = 256
    stats = ppo.update(rng.normal(size=(n, V2)), rng.normal(size=(n, 2)), np.zeros(n), rng.normal(size=n),
                       rng.normal(size=n))
    assert np.isfinite(stats["value_loss"])
