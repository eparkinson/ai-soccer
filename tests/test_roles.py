import numpy as np
import pytest

import train_ppo
import train_roles
from aisoccer.brains.PPOBrain import OBS_DIM, PPOBrain, player_features
from aisoccer.brainspec import load_brain
from aisoccer.brains.SimpleBrain import SimpleBrain
from aisoccer.game import Game
from aisoccer.ppo import gaussian_log_prob

NUM = 5


@pytest.fixture
def short_games():
    """Training games of 300 ticks for the test; restores train_ppo's settings after."""
    train_ppo.set_action_repeat(2, 300)
    yield 300
    train_ppo.set_action_repeat(2, None)


def roles(seed=0, hidden=(16, 16)):
    args = train_roles.parse_args(["--seed", str(seed), "--hidden", *map(str, hidden)])
    learners = train_roles.make_roles(args)
    for r, ppo in enumerate(learners):
        ppo.log_std[:] = -1.0 + 0.1 * r  # every role explores differently
        ppo.policy.params[-2] *= 30  # a random policy that actually moves
    return learners


def test_role_trajectories_route_each_players_experience_to_its_role():
    steps = 7
    traj = {
        "obs": np.tile(np.arange(NUM)[None, :, None], (steps, 1, OBS_DIM)).astype(float),
        "act": np.tile(np.arange(NUM)[None, :, None], (steps, 1, 2)).astype(float),
        "logp": np.tile(np.arange(NUM)[None, :], (steps, 1)).astype(float),
        "val": np.tile(np.arange(NUM)[None, :], (steps, 1)) * 0.1,
        "rew": np.linspace(-1, 1, steps),
    }
    batches = train_roles.build_role_batches([traj, traj], 0.99, 0.95)
    assert len(batches) == NUM
    for r, (obs, act, logp, adv, ret) in enumerate(batches):
        assert obs.shape == (2 * steps, OBS_DIM) and np.all(obs == r)
        assert np.all(act == r) and np.all(logp == r)
        # GAE with this player's own value column, exactly as build_batch does it
        single = {k: v[:, [r]] if k != "rew" else v for k, v in traj.items()}
        expected = train_ppo.build_batch([single, single], 0.99, 0.95)
        assert np.array_equal(adv, expected[3]) and np.array_equal(ret, expected[4])


def test_training_brain_records_each_roles_value_and_noise():
    learners = roles()
    for r, ppo in enumerate(learners):  # role r's critic says r everywhere
        for p in ppo.value.params:
            p[...] = 0.0
        ppo.value.params[-1][...] = r
    weights = train_roles.team_weights(learners)
    brain = PPOBrain("learner", weights=weights, training=True)
    Game(brain, SimpleBrain(), game_length=40, quiet_mode=True, seed=3).play()
    traj = brain.finish_trajectory(0.99)
    assert np.array_equal(traj["val"], np.tile(np.arange(NUM, dtype=float), (20, 1)))
    # each player's actions come from its own role policy and noise
    log_std = weights["log_std"]
    for t in range(len(traj["obs"])):
        mean = np.stack(
            [ppo.policy(traj["obs"][t, [r]])[0] for r, ppo in enumerate(learners)]
        )
        expected = [
            gaussian_log_prob(traj["act"][t, r], mean[r], log_std[r]) for r in range(NUM)
        ]
        assert np.allclose(traj["logp"][t], expected)


def test_role_training_games_in_vecgame_match_one_at_a_time(tmp_path, short_games):
    """The batched simulator plays role learners bit-identically to Game."""
    learners = roles(seed=1)
    weights = train_roles.team_weights(learners)
    snapshot = train_roles.snapshot_weights(roles(seed=2))
    entrant = tmp_path / "entrant.npz"
    PPOBrain.save_weights(entrant, train_roles.entrant_weights(roles(seed=3)))
    shared = tmp_path / "shared.npz"
    PPOBrain.save_weights(shared, PPOBrain.load_weights(PPOBrain.WEIGHTS_FILE))
    opponents = ["self", "snapshot", "file:" + str(entrant), "file:" + str(shared)]
    tasks = [
        (weights, opp, snapshot if opp == "snapshot" else None, 50 + i, 0.995, None)
        for i, opp in enumerate(opponents)
    ]
    batched = train_ppo.play_training_games(tasks)
    for task, got in zip(tasks, batched):
        expected = train_ppo.play_training_game(task)
        assert got[:3] == expected[:3]
        for traj, ref in zip(got[3], expected[3]):
            for key in ref:
                assert np.array_equal(traj[key], ref[key]), key


def test_short_training_run_resumes_and_its_entrant_plays(tmp_path, short_games):
    panel = tmp_path / "field"
    panel.mkdir()
    PPOBrain.save_weights(panel / "Other.npz", PPOBrain.load_weights(PPOBrain.WEIGHTS_FILE))
    (panel / "half-written.tmp.npz").write_bytes(b"")
    out = tmp_path / "roles"
    argv = [
        "--workers", "1", "--games", "4", "--iterations", "1", "--hidden", "16", "16",
        "--panel-dir", str(panel), "--out", str(out), "--game-length", "300",
        "--snapshot-every", "2",
    ]
    train_roles.main(argv)
    assert (out / "best.npz").exists() and (out / "latest.npz").exists()
    state = np.load(out / "latest.npz", allow_pickle=True)["state"].item()
    assert state["iteration"] == 1

    # The entrant is a PPOBrain with five different role policies and plays a game.
    brain = load_brain(str(out / "best.npz"))
    assert isinstance(brain, PPOBrain) and len(brain.role_policies) == NUM
    assert brain.log_std.shape == (2,)
    obs = np.zeros((1, OBS_DIM))
    outputs = [net(obs)[0] for net in brain.role_policies]
    assert not np.allclose(outputs[0], outputs[1])
    score = Game(brain, SimpleBrain(), game_length=200, quiet_mode=True, seed=1).play()
    assert set(score) == {"blue", "red"}

    # A restart carries on from the saved state.
    learners = roles(hidden=(16, 16))
    rng = np.random.default_rng(123)
    assert train_roles.restore(learners, rng, state) == 1
    for ppo, saved in zip(learners, state["roles"]):
        assert all(np.array_equal(p, s) for p, s in zip(ppo.policy.params, saved["policy"]))
        assert ppo.policy_opt.t == saved["policy_opt"]["t"] > 0
        assert ppo.policy_opt.params[0] is ppo.policy.params[0]  # optimiser still attached
    train_roles.main(argv)
    log = (out / "progress.log").read_text()
    assert "Resumed from iteration 1" in log and "it     2 |" in log
    state = np.load(out / "latest.npz", allow_pickle=True)["state"].item()
    assert state["iteration"] == 2
    assert len(list((out / "snapshots").glob("it*.npz"))) == 2  # iterations 0 and 2
