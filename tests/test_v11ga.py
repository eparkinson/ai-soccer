"""LLM-v11-GA (aisoccer/llm/v11ga.py) and its tuner (tune_v11.py)."""

import json
import pickle
import re
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pytest

import tune_v11
from aisoccer.brainspec import load_brain
from aisoccer.game import Game
from aisoccer.llm import v11
from aisoccer.llm.v11 import LLMBrainV11
from aisoccer.llm.v11ga import PARAMS, LLMBrainV11GA, check_params
from aisoccer.tactics.v1 import TacticsV1

HEURISTIC = "aisoccer.brains.{0}.{0}"


def recorded(brain):
    """The brain, with every action it takes kept in brain.actions."""
    brain.actions = []
    inner = brain.do_move

    def do_move():
        acc = inner()
        brain.actions.append(np.array(acc, copy=True))
        return acc

    brain.do_move = do_move
    return brain


def game_actions(brain, seed, ticks=400, opponent=None):
    brain = recorded(brain)
    other = recorded(opponent or TacticsV1())
    score = Game(brain, other, game_length=ticks, quiet_mode=True, seed=seed).play()
    return np.array(brain.actions), np.array(other.actions), score


def states(seed=5, ticks=300):
    """Game situations (as LLM-v11 sees them) from a game of v11 against v11."""
    seen = []

    class Watch(LLMBrainV11):
        def do_move(self):
            seen.append(
                [np.array(x, copy=True) for x in (self.my_players_pos, self.my_players_vel, self.opp_players_pos,
                                                  self.opp_players_vel, self.ball_pos, self.ball_vel)]
            )
            return super().do_move()

    Game(Watch(), LLMBrainV11(), game_length=ticks, quiet_mode=True, seed=seed).play()
    return seen


def act(brain, situation):
    return brain.move(*situation, 0, 0, 0.5)


# --- the brain ---


@pytest.mark.parametrize("seed", [1, 2])
def test_empty_params_play_exactly_as_v11(seed):
    mine, theirs, score = game_actions(LLMBrainV11(), seed, opponent=LLMBrainV11())
    ga_mine, ga_theirs, ga_score = game_actions(LLMBrainV11GA(), seed, opponent=LLMBrainV11())
    assert score == ga_score
    assert np.array_equal(mine, ga_mine) and np.array_equal(theirs, ga_theirs)


def test_v11s_own_values_given_explicitly_play_exactly_as_v11():
    """Derived values (KEEPER_X, POACHER_SPOT, TICKS, _UNIT_MOVES) are rebuilt as v11 builds them."""
    mine, _, _ = game_actions(LLMBrainV11(), 3)
    ga, _, _ = game_actions(LLMBrainV11GA(params=dict(PARAMS)), 3)
    assert np.array_equal(mine, ga)


def test_overrides_stay_with_their_instance():
    before = {k: getattr(v11, k) for k in ("PLAN_TICKS", "DIRECTIONS", "LOOK_AHEAD", "GOOD_KICK", "KEEPER_X")}
    moves_before, ticks_before = v11._UNIT_MOVES.copy(), v11.TICKS.copy()
    a = LLMBrainV11GA(params={"PLAN_TICKS": 30, "DIRECTIONS": 16, "LOOK_AHEAD": 60, "GOOD_KICK": 0.6})
    b = LLMBrainV11GA(params={"KEEPER_OFFSET": 40, "LINE_GAP": 120, "POACHERS": 2, "OWN_GOAL_RISK": 1.0})
    plain, ga_plain = LLMBrainV11(), LLMBrainV11GA()
    alone_a = LLMBrainV11GA(params=dict(a.params))
    alone_b = LLMBrainV11GA(params=dict(b.params))
    assert a.constant("_UNIT_MOVES").shape == (33, 2) and b.constant("_UNIT_MOVES").shape == (49, 2)
    assert a.constant("TICKS").shape == (61,) and b.constant("KEEPER_X") == v11.KEEPER_X - 6 + 40
    differ_a = differ_b = 0
    for s in states():  # interleaved on the same situations
        ra, rb, rp, rg = act(a, s), act(b, s), act(plain, s), act(ga_plain, s)
        assert np.array_equal(rp, rg)
        differ_a += not np.array_equal(ra, rp)
        differ_b += not np.array_equal(rb, rp)
    assert differ_a > 0 and differ_b > 0
    # each plays as it would on its own, whatever the others do in between
    for brain in (a, b, alone_a, alone_b):
        brain.chaser = None
    for s in states(seed=6):
        ra, rb = act(a, s), act(b, s)
        act(plain, s)
        assert np.array_equal(ra, act(alone_a, s)) and np.array_equal(rb, act(alone_b, s))
    assert {k: getattr(v11, k) for k in before} == before
    assert np.array_equal(v11._UNIT_MOVES, moves_before) and np.array_equal(v11.TICKS, ticks_before)


def test_isolated_instances_match_fresh_ones_game_by_game():
    params = {"PLAN_TICKS": 32, "BEHIND": 60, "ON_TARGET_OPEN": 2.0}
    first, _, _ = game_actions(LLMBrainV11GA(params=params), 4, opponent=LLMBrainV11GA(params={"DIRECTIONS": 16}))
    again, _, _ = game_actions(LLMBrainV11GA(params=params), 4, opponent=LLMBrainV11GA(params={"DIRECTIONS": 16}))
    plain, _, _ = game_actions(LLMBrainV11(), 4, opponent=LLMBrainV11GA(params={"DIRECTIONS": 16}))
    assert np.array_equal(first, again)
    assert not np.array_equal(first[: len(plain)], plain[: len(first)])


def test_spec_file_loads(tmp_path):
    spec = tmp_path / "LLM-v11-GA.json"
    params = {"GOOD_KICK": 0.5, "PLAN_TICKS": 36.4}
    spec.write_text(json.dumps({"class": "aisoccer.llm.v11ga.LLMBrainV11GA", "kwargs": {"params": params}}))
    brain = load_brain(spec)
    assert isinstance(brain, LLMBrainV11GA) and brain.name == "LLM-v11-GA"
    assert brain.constant("GOOD_KICK") == 0.5 and brain.constant("PLAN_TICKS") == 36
    assert v11.GOOD_KICK == 0.3 and v11.PLAN_TICKS == 40


def test_pickles():
    brain = LLMBrainV11GA(params={"DIRECTIONS": 20})
    copy = pickle.loads(pickle.dumps(brain))
    s = states(ticks=50)[-1]
    assert np.array_equal(act(brain, s), act(copy, s))


def test_bad_params_are_refused():
    with pytest.raises(ValueError):
        check_params({"NOT_A_PARAM": 1})
    with pytest.raises(ValueError):
        check_params({"LOOK_AHEAD": 40, "PLAN_TICKS": 40})
    with pytest.raises(ValueError):
        check_params({"POACHERS": 3})


# --- the tuner ---


def test_cmaes_climbs_a_simple_hill():
    rng = np.random.default_rng(0)
    target = np.array([0.8, 0.2, 0.6])
    es = tune_v11.CMAES([0.3, 0.3, 0.3], 0.2, 8)
    for _ in range(40):
        xs = es.ask(rng)
        es.tell(xs, -((xs - target) ** 2).sum(axis=1))
    assert np.abs(es.mean - target).max() < 0.05
    again = tune_v11.CMAES.from_state(json.loads(json.dumps(es.state())))
    assert np.array_equal(again.C, es.C) and again.sigma == es.sigma


def test_encoding_round_trips_v11s_values():
    names = list(tune_v11.TUNED)
    start = {k: PARAMS[k] for k in names}
    assert tune_v11.decode(tune_v11.encode(start, names), names) == pytest.approx(start)
    assert (tune_v11.reflect(np.array([-0.1, 0.5, 1.2])) == pytest.approx([0.1, 0.5, 0.8]))
    for k in names:
        lo, hi = tune_v11.TUNED[k]
        assert lo <= PARAMS[k] <= hi, k


def fake_play(task):
    """A stand-in for a game: a deterministic score from the task, with no physics."""
    params, opponent, blue_side, seed, ticks = task
    h = zlib.crc32(json.dumps([params, Path(opponent).name, blue_side, seed, ticks]).encode())
    return h % 4, (h // 4) % 3


def make_field(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("SimpleBrain", "RandomWalk"):
        (directory / f"{name}.json").write_text(json.dumps({"class": HEURISTIC.format(name)}))
    (directory / "LLM-v11-GA.json").write_text(json.dumps({"class": tune_v11.BRAIN, "kwargs": {"params": {}}}))
    return directory


def tune(tmp_path, out, *extra):
    args = tune_v11.parse_args(
        ["--out", str(out), "--panel-dir", str(tmp_path / "field"), "--lam", "4", "--seed", "7",
         "--ticks-per-opponent", "200", "--ticks", "100", "--exit-when-done", "--reference-every", "2", *extra]
    )
    tune_v11.run(args)


def test_tuner_runs_resumes_and_writes_an_entrant(tmp_path, monkeypatch):
    monkeypatch.setattr(tune_v11, "play", fake_play)
    make_field(tmp_path / "field")
    roster = tmp_path / "roster.json"
    roster.write_text(json.dumps({"families": {"SimpleBrain": "S", "RandomWalk": "S", "LLM-v11": "LLM"}}))
    straight, split = tmp_path / "straight", tmp_path / "split"
    tune(tmp_path, straight, "--generations", "2", "--families", str(roster))
    tune(tmp_path, split, "--generations", "1", "--families", str(roster))
    tune(tmp_path, split, "--generations", "1", "--families", str(roster))  # resumed
    a, b = (json.loads((d / "state.json").read_text()) for d in (straight, split))
    assert a["generation"] == b["generation"] == 2
    assert a["cma"] == b["cma"] and a["rng"] == b["rng"]
    assert json.loads((straight / "best.json").read_text()) == json.loads((split / "best.json").read_text())
    panel = sorted(p.name for p in (split / "panel").glob("*"))
    assert panel == ["LLM-v11.json", "RandomWalk.json", "SimpleBrain.json"]  # own entry out, v11 in
    text = (split / "progress.log").read_text()
    assert "resumed after generation 1" in text and "LLM-v11's own numbers" in text
    assert "weights by family" in text and "LLM 0.500" in text
    brain = load_brain(split / "best.json")
    assert isinstance(brain, LLMBrainV11GA) and set(brain.params) == set(tune_v11.TUNED)
    assert json.loads((split / "cpu.json").read_text())["seconds"] > 0


def test_tuner_stops_at_the_cpu_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(tune_v11, "play", fake_play)
    make_field(tmp_path / "field")
    out = tmp_path / "out"
    out.mkdir()
    (out / "cpu.json").write_text(json.dumps({"seconds": 3600.0}))
    tune(tmp_path, out, "--cpu-hours-budget", "0.5")
    assert not (out / "state.json").exists() and not (out / "best.json").exists()
    assert re.search(r"CPU budget: 1\.\d\d of 0\.5 hours used; idle", (out / "progress.log").read_text())


def test_tuner_plays_real_games(tmp_path):
    """One tiny generation of real (short) games."""
    make_field(tmp_path / "field")
    out = tmp_path / "out"
    tune(tmp_path, out, "--generations", "1", "--lam", "2", "--ticks-per-opponent", "60", "--ticks", "30")
    state = json.loads((out / "state.json").read_text())
    assert state["generation"] == 1 and state["last_generation_cpu"] > 0


def burn(seconds):
    start = time.process_time()
    while time.process_time() - start < seconds:
        pass
    return seconds


def test_cpu_of_live_pool_workers_is_counted():
    """The pool's workers are the forkserver's children, not ours: they count while alive and after."""
    before = tune_v11.own_cpu_seconds()
    with Pool(2) as pool:
        results = pool.imap(burn, [1.0, 1.0])
        next(results), next(results)
        live = tune_v11.own_cpu_seconds()
    assert live - before > 1.8
    assert tune_v11.own_cpu_seconds() >= live - 0.05
