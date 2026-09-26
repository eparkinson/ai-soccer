import numpy as np
import pytest

import train_ppo
from aisoccer.brains.BehindAndTowards import BehindAndTowards
from aisoccer.brains.DefendersAndAttackers import DefendersAndAttackers
from aisoccer.brains.GeneticBrain import CHROMOSOME_LENGTH
from aisoccer.brains.PPOBrain import (
    OBS_DIM,
    PPOBrain,
    potential_terms,
    shaping_potential,
)
from aisoccer.brains.SimpleBrain import SimpleBrain
from aisoccer.game import Game
from aisoccer.ppo import PPO
from aisoccer.vecgame import VecGame
from aisoccer.vecppo import TERMS, batch_player_features, batch_potential_terms
from tests.test_ppo import reference_player_features


def trained_weights():
    """The saved PPOBrain (it scores goals, unlike a random network) with a fresh critic."""
    weights = PPOBrain.load_weights(PPOBrain.WEIGHTS_FILE)
    if weights is None:
        pytest.skip("no saved PPOBrain weights")
    return dict(weights, value=PPO(OBS_DIM, 2, seed=1).weights()["value"])


def physics_ticks(game):
    """Step a Game to its next physics tick (a goal takes a tick() of its own in Game)."""
    before = game.state.ticks
    while game.state.ticks == before:
        game.tick()


def assert_plays_like_game(make_brains, seeds, ticks, precision="exact"):
    """Every VecGame game has Game's positions and velocities after every tick, and its score."""
    pairs = [make_brains() for _ in seeds]
    vec = VecGame(
        [b for b, _ in pairs],
        [r for _, r in pairs],
        seeds,
        game_length=ticks,
        precision=precision,
    )
    games = [
        Game(*make_brains(), game_length=ticks, quiet_mode=True, seed=s) for s in seeds
    ]
    for _ in range(ticks):
        vec.tick()
        for k, game in enumerate(games):
            physics_ticks(game)
            assert np.array_equal(
                vec.pos[k], game.state.pos
            ), f"game {k} at tick {vec.ticks}"
            assert np.array_equal(
                vec.vel[k], game.state.vel
            ), f"game {k} at tick {vec.ticks}"
    vec.finish()
    for game in games:  # a goal on the last tick is not counted, as in Game
        assert game.tick().name == "end"
    assert vec.scores() == [g.score for g in games]
    return vec


def test_batch_player_features_match_reference_exactly():
    rng = np.random.default_rng(10)
    k = 2000
    args = [
        rng.uniform(0, 1800, (k, 5, 2)),
        rng.normal(0, 3, (k, 5, 2)),
        rng.uniform(0, 1800, (k, 5, 2)),
        rng.normal(0, 3, (k, 5, 2)),
        rng.uniform(0, 1800, (k, 2)),
        rng.normal(0, 5, (k, 2)),
        rng.integers(0, 5, k),
        rng.integers(0, 5, k),
        rng.random(k),
    ]
    for a in (args[0], args[2]):  # exact ties in distances, as at kick-off
        a[::3] = np.round(a[::3], -2)
    obs = batch_player_features(*args)
    for i in range(k):
        assert np.array_equal(obs[i], reference_player_features(*(a[i] for a in args)))


def test_batch_potential_matches_shaping_potential_exactly():
    rng = np.random.default_rng(11)
    k = 1000
    my_pos = rng.uniform(0, 1800, (k, 5, 2))
    opp_pos = rng.uniform(0, 1800, (k, 5, 2))
    ball_pos = rng.uniform(0, 1800, (k, 2))
    ball_vel = rng.normal(0, 6, (k, 2))
    ball_vel[::4, 0] = np.abs(ball_vel[::4, 0])  # plenty of shots on goal
    ball_vel[::7] = 0.0
    reward = {
        "goal": 2.0,
        "progress": 0.3,
        "control": 0.15,
        "chase": 0.1,
        "shot": 0.1,
        "spread": 0.05,
    }
    terms = batch_potential_terms(my_pos, opp_pos, ball_pos, ball_vel)
    assert (terms["shot"] > 0).sum() > 50
    phi = 0
    for name in TERMS:
        phi = phi + reward.get(name, 0.0) * terms[name]
    for i in range(k):
        expected = potential_terms(my_pos[i], opp_pos[i], ball_pos[i], ball_vel[i])
        assert list(expected) == list(TERMS)
        for name in TERMS:
            assert terms[name][i] == expected[name]
        assert phi[i] == shaping_potential(
            my_pos[i], opp_pos[i], ball_pos[i], reward, ball_vel[i]
        )


def test_heuristic_games_match_game_tick_by_tick():
    """Deterministic heuristics: VecGame game k is Game(seed_k), bit for bit, for a whole game."""
    vec = assert_plays_like_game(
        lambda: (DefendersAndAttackers(), BehindAndTowards()), [0, 1, 2, 3], 1800
    )
    assert vec.score.sum() > 0  # goals and kick-off resets were exercised


def test_ppo_games_match_game_tick_by_tick():
    """Fixed PPO networks against each other, deterministic: bit for bit, goals included."""
    trained = trained_weights()
    other = PPO(OBS_DIM, 2, seed=2).weights()

    def brains():
        return PPOBrain("trained", weights=trained), PPOBrain("random", weights=other)

    vec = assert_plays_like_game(brains, [10, 11, 12], 1800)
    assert vec.score.sum() > 0


def test_mixed_brains_and_float32_networks_play_the_same_game():
    """
    A heuristic against a PPOBrain in one VecGame. With float32 networks the games
    start out the same, drifting apart only slowly (by tick 100 by well under a pixel).
    """
    trained = trained_weights()
    assert_plays_like_game(
        lambda: (SimpleBrain(), PPOBrain("p", weights=trained)), [20, 21], 400
    )

    seeds = [30, 31, 32]
    vec = VecGame(
        [PPOBrain("p", weights=trained) for _ in seeds],
        [BehindAndTowards() for _ in seeds],
        seeds,
        game_length=100,
        precision="float32",
    )
    games = [
        Game(PPOBrain("p", weights=trained), BehindAndTowards(), 100, True, seed=s)
        for s in seeds
    ]
    for _ in range(100):
        vec.tick()
        for k, game in enumerate(games):
            physics_ticks(game)
            assert np.allclose(vec.pos[k], game.state.pos, rtol=0, atol=1e-2)


def pool_files(tmp_path):
    """A PPOBrain weights file, a smaller network and a mixed team, as in a league pool."""
    trained = trained_weights()
    small = PPO(OBS_DIM, 2, hidden=(32, 16), seed=3).weights()
    mixed = dict(
        small,
        role_policies=[
            PPO(OBS_DIM, 2, hidden=(32, 16), seed=s).weights()["policy"]
            for s in range(5)
        ],
    )
    files = []
    for name, weights in [("champ", trained), ("small", small), ("mixed", mixed)]:
        path = tmp_path / f"{name}.npz"
        PPOBrain.save_weights(path, weights)
        files.append("file:" + str(path))
    return files


def test_training_games_match_play_training_game_exactly(tmp_path):
    """
    Self-play, snapshots and PPO pool files in one VecGame batch give exactly the
    trajectories and results of playing each game with play_training_game.
    """
    trained = trained_weights()
    snapshot = PPOBrain.load_weights(PPOBrain.WEIGHTS_FILE)
    reward = {
        "goal": 2.0,
        "progress": 0.3,
        "control": 0.15,
        "chase": 0.1,
        "shot": 0.1,
        "spread": 0.05,
    }
    opponents = ["self", "snapshot", *pool_files(tmp_path)]
    tasks = [
        (
            trained,
            opp,
            snapshot if opp == "snapshot" else None,
            100 + i,
            0.995,
            reward if i % 2 else None,
        )
        for i, opp in enumerate(opponents * 2)
    ]
    assert all(train_ppo.vectorisable(t) for t in tasks)
    batched = train_ppo.play_training_games(tasks)
    goals = 0
    for task, got in zip(tasks, batched):
        expected = train_ppo.play_training_game(task)
        assert got[:3] == expected[:3]
        goals += got[1] + got[2]
        assert len(got[3]) == len(expected[3]) == (2 if task[1] == "self" else 1)
        for traj, ref in zip(got[3], expected[3]):
            assert traj.keys() == ref.keys()
            for key in ref:
                assert traj[key].shape == ref[key].shape
                assert np.array_equal(traj[key], ref[key]), key
    assert goals > 0  # the goal rewards and post-goal decisions were exercised


def test_iteration_keeps_task_order_and_falls_back_for_other_opponents(tmp_path):
    class InlinePool:
        def map(self, f, items, chunksize=None):
            return [f(item) for item in items]

    weights = PPO(OBS_DIM, 2, seed=4).weights()
    genetic = tmp_path / "GA.json"
    genetic.write_text('{"chromosome": ' + str([0.5] * CHROMOSOME_LENGTH) + "}")
    opponents = ["SimpleBrain", "self", "file:" + str(genetic), "snapshot", "self"]
    tasks = [
        (weights, opp, weights, 200 + i, 0.99, None) for i, opp in enumerate(opponents)
    ]
    assert [train_ppo.vectorisable(t) for t in tasks] == [
        False,
        True,
        False,
        True,
        True,
    ]
    for vector_games in (1, 2, 8):
        outcomes = train_ppo.play_iteration_games(InlinePool(), tasks, vector_games)
        assert [o[0] for o in outcomes] == opponents
    old = train_ppo.play_iteration_games(InlinePool(), tasks, 0)
    for got, ref in zip(outcomes, old):
        assert got[:3] == ref[:3]
        for traj, expected in zip(got[3], ref[3]):
            assert all(np.array_equal(traj[k], expected[k]) for k in expected)


def test_goal_statistics_match_game_with_float32_networks():
    """
    With float32 networks games drift apart from Game's after a while (chaos amplifies
    1e-7 differences), so compare what matters over many games: goals, wins and draws.
    """
    trained = trained_weights()
    other = PPO(OBS_DIM, 2, seed=5).weights()
    seeds = list(range(1000, 1200))
    ticks = 450

    def brains():
        return PPOBrain("a", weights=trained, deterministic=False), PPOBrain(
            "b", weights=other, deterministic=False
        )

    pairs = [brains() for _ in seeds]
    vec = VecGame(
        [b for b, _ in pairs],
        [r for _, r in pairs],
        seeds,
        game_length=ticks,
        precision="float32",
    )
    new = np.array([[s["blue"], s["red"]] for s in vec.play()])
    old = np.array(
        [
            [g["blue"], g["red"]]
            for g in (Game(*brains(), ticks, True, seed=s).play() for s in seeds)
        ]
    )

    def summary(scores):
        diff = scores[:, 0] - scores[:, 1]
        return scores.sum(axis=1), (diff > 0).astype(float), (diff == 0).astype(float)

    for a, b in zip(summary(new), summary(old)):
        spread = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        assert abs(a.mean() - b.mean()) < 3 * spread + 1e-9
    assert new.sum() > 20
