import json
import random

import numpy as np
import pytest

import evolve_gp
from aisoccer.brainspec import load_brain
from aisoccer.game import Game
from aisoccer.gp.v1 import (
    FUNCTIONS,
    TERMINALS,
    GPBrain,
    compile_program,
    interpret_targets,
    size,
    to_text,
    validate,
)


def random_programs(count, seed=0, max_size=60):
    return evolve_gp.ramped(random.Random(seed), count, max_size=max_size)


def random_state(brain, rng, spread=1.0):
    brain.my_players_pos = rng.random((5, 2)) * [1799, 800] * spread
    brain.my_players_vel = rng.normal(0, 3, (5, 2))
    brain.opp_players_pos = rng.random((5, 2)) * [1799, 800] * spread
    brain.opp_players_vel = rng.normal(0, 3, (5, 2))
    brain.ball_pos = rng.random(2) * [1799, 800] * spread
    brain.ball_vel = rng.normal(0, 5, 2)


class RecordingGP(GPBrain):
    def do_move(self):
        move = super().do_move()
        assert move.shape == (5, 2) and np.isfinite(move).all()
        return move


def test_random_programs_play_full_games():
    for k, program in enumerate(random_programs(8, seed=3)):
        a = RecordingGP("a", program=program)
        b = RecordingGP("b", program=random_programs(8, seed=4)[k])
        score = Game(a, b, game_length=600, quiet_mode=True, seed=k).play()
        assert set(score) == {"blue", "red"}


def test_every_word_is_used_and_programs_are_well_typed():
    programs = random_programs(200, seed=1)
    words = set()
    for p in programs:
        validate(p)
        words |= set(json.dumps(p).replace("[", " ").replace("]", " ").replace(",", " ").replace('"', " ").split())
    assert set(FUNCTIONS) <= words
    assert set(TERMINALS) <= words


def test_compiled_equals_interpreted():
    rng = np.random.default_rng(0)
    for program in random_programs(300, seed=2):
        brain = GPBrain(program=program)
        for spread in (1.0, 0.0, 3.0):  # ordinary, all at one point (ties), off the field
            random_state(brain, rng, spread)
            compiled = np.array(brain.targets())
            reference = interpret_targets(program, brain)
            assert np.isfinite(compiled).all()
            np.testing.assert_allclose(compiled, reference, rtol=1e-9, atol=1e-6, err_msg=to_text(program))


def test_extreme_values_stay_finite():
    program = ["scale", ["unit", ["sub_v", "ball", "ball"]], ["mul", ["mul", 1e300, 1e300], 1e300]]
    brain = GPBrain(program=program)
    random_state(brain, np.random.default_rng(1))
    assert np.isfinite(brain.do_move()).all()
    program = ["behind", "me", "me", ["mul", -1e300, 1e300]]
    brain = GPBrain(program=program)
    random_state(brain, np.random.default_rng(1))
    assert np.isfinite(brain.do_move()).all()


def test_serialisation_round_trips(tmp_path):
    for program in random_programs(20, seed=5):
        path = tmp_path / "p.json"
        path.write_text(json.dumps({"class": "aisoccer.gp.v1.GPBrain", "kwargs": {"program": program}}))
        brain = load_brain(str(path))
        assert isinstance(brain, GPBrain) and brain.program == program
        assert to_text(brain.program) == to_text(program)


def test_invalid_programs_are_rejected():
    for bad in [["add", "ball", 1], "nothing", ["if_v", 1.0, "ball", "me"], 3.0, ["unit"]]:
        with pytest.raises(ValueError):
            validate(bad)


def test_printing_is_readable():
    program = [
        "if_v",
        "nearest_to_ball",
        ["behind", "ball", "opp_goal", 30],
        ["if_v", ["lt", "role", 1.5], ["add_v", "own_goal", ["point", 60, ["sub", "role", 0.5]]], "home"],
    ]
    assert to_text(program) == (
        "def target():\n"
        "    if nearest_to_ball:\n"
        "        return behind(ball, opp_goal, 30)\n"
        "    if role < 1.5:\n"
        "        return own_goal + point(60, role - 0.5)\n"
        "    return home"
    )
    assert to_text(["sub_v", "ball", ["sub_v", "me", "opp"]]) == "def target():\n    return ball - (me - opp)"
    # every random program prints and stays short
    for p in random_programs(100, seed=6):
        text = to_text(p)
        assert text.startswith("def target():") and "[" not in text
        assert len(text.splitlines()) <= size(p) + 1


def test_operators_respect_types_and_limits():
    rng = random.Random(0)
    pop = random_programs(30, seed=7)
    args = evolve_gp.parse_args(["--panel", "SimpleBrain", "--max-size", "40", "--max-depth", "8"])
    children = evolve_gp.breed(rng, pop, [rng.random() for _ in pop], args)
    assert len(children) == len(pop)
    for child in children:
        validate(child)
        compile_program(child)
    for child in children[args.elite:]:
        assert size(child) <= 40 or child in pop


def test_smoke_evolution_and_best_json_loads(tmp_path):
    args = evolve_gp.parse_args(
        ["--workers", "1", "--population", "6", "--games", "2", "--generations", "2",
         "--panel", "SimpleBrain", "--ticks", "300", "--out", str(tmp_path), "--seed", "1"]
    )
    evolve_gp.run(args)
    log = (tmp_path / "progress.log").read_text()
    assert "generation    2" in log
    brain = load_brain(str(tmp_path / "best.json"))
    assert isinstance(brain, GPBrain)
    assert (tmp_path / "best.txt").read_text().count("def target():") == 1
    # resumes where it stopped
    args.generations = 1
    evolve_gp.run(args)
    assert "resumed at generation 2" in (tmp_path / "progress.log").read_text()
    assert json.loads((tmp_path / "population.json").read_text())["generation"] == 3
