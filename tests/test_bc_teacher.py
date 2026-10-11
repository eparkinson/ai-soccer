"""--bc-teacher: any brain spec can be imitated, and the old teacher names mean what they did."""

import numpy as np
import pytest

import train_ppo
from aisoccer.brains.DefendersAndAttackers import DefendersAndAttackers
from aisoccer.brains.PPOBrain import OBS_DIM, PPOBrain
from aisoccer.brains.SimpleBrain import SimpleBrain
from aisoccer.constants import Constants
from aisoccer.game import Game
from aisoccer.llm.v11 import LLMBrainV11
from train_ppo import DaggerStudent, Demonstrator, make_opponent, make_teacher

V11 = "brain:aisoccer.llm.v11.LLMBrainV11"


def same_weights(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a.policy.params, b.policy.params)) and np.array_equal(
        a.log_std, b.log_std
    )


@pytest.mark.parametrize("name", list(train_ppo.HEURISTICS))
def test_heuristic_teachers_are_unchanged(name):
    assert type(make_teacher(name)) is type(make_opponent(name)) is train_ppo.HEURISTICS[name]


@pytest.mark.parametrize("name", ["PPO-clone", "PPO-it80"])
def test_saved_version_teachers_are_unchanged(name):
    teacher, opponent = make_teacher(name), make_opponent(name)
    assert type(teacher) is PPOBrain and same_weights(teacher, opponent)


def test_file_teacher_is_unchanged(tmp_path):
    path = tmp_path / "w.npz"
    PPOBrain.save_weights(path, PPOBrain.load_weights(train_ppo.HISTORY_DIR / "PPO-clone.npz"))
    teacher, opponent = make_teacher(f"file:{path}"), make_opponent(f"file:{path}")
    assert type(teacher) is PPOBrain and same_weights(teacher, opponent)


def test_brain_spec_teachers(tmp_path):
    assert isinstance(make_teacher(V11), LLMBrainV11)
    assert isinstance(make_teacher("brain:aisoccer.brains.DefendersAndAttackers.DefendersAndAttackers"),
                      DefendersAndAttackers)
    path = tmp_path / "w.npz"
    PPOBrain.save_weights(path, PPOBrain.load_weights(train_ppo.HISTORY_DIR / "PPO-clone.npz"))
    assert same_weights(make_teacher(str(path)), make_opponent(f"file:{path}"))  # a bare .npz path
    explore = make_teacher(f"explore:{path}")
    assert type(explore) is PPOBrain and not explore.deterministic


def test_demonstrator_imitates_a_brain_spec(monkeypatch):
    """The Demonstrator plays as v11 and records v11's moves, capped as the game caps them."""
    monkeypatch.setattr(PPOBrain, "ACTION_REPEAT", 1)
    demonstrator = Demonstrator(V11)
    assert isinstance(demonstrator.teacher, LLMBrainV11)
    twin = LLMBrainV11()  # fed the same views, a second v11 must make the same moves
    moves = []
    original = demonstrator.teacher.move

    def move(*view):
        action = original(*view)
        moves.append(train_ppo.capped(twin.move(*view)))
        return action

    demonstrator.teacher.move = move
    Game(demonstrator, SimpleBrain(), game_length=12, quiet_mode=True, seed=3).play()
    act = np.array(demonstrator.act)
    assert np.array(demonstrator.obs).shape == (12, 5, OBS_DIM)
    assert np.allclose(act, np.array(moves))
    assert (np.linalg.norm(act, axis=2) <= Constants.MAX_ACCELERATION + 1e-9).all()
    assert np.abs(act).sum() > 0


def test_dagger_student_is_labelled_by_a_brain_spec():
    weights = PPOBrain.load_weights(train_ppo.HISTORY_DIR / "PPO-clone.npz")
    weights["action_repeat"] = 2
    student = DaggerStudent(weights, V11)
    assert isinstance(student.teacher, LLMBrainV11)
    Game(student, SimpleBrain(), game_length=10, quiet_mode=True, seed=4).play()
    assert len(student.obs) == len(student.act) == 5  # one label per decision, every 2 ticks
    assert (np.linalg.norm(np.array(student.act), axis=2) <= Constants.MAX_ACCELERATION + 1e-9).all()
