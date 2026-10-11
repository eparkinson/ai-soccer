"""evolve_gp.py's fitness options: --panel all, --ticks-per-opponent, --fitness gd."""

import hashlib
import json
import zlib
from pathlib import Path

import numpy as np
import pytest

import evolve_gp
from aisoccer.brainspec import load_brain
from aisoccer.gp.v1 import GPBrain

HEURISTIC = "aisoccer.brains.{0}.{0}"
FIELD = ["BehindAndTowards", "DefendersAndAttackers", "SimpleBrain", "StrategicPlanner"]


def fake_play(task):
    """A stand-in for a game: a deterministic score from the task, with no physics."""
    program, opponent, gp_blue, seed, ticks = task[:5]
    h = zlib.crc32(json.dumps([program, Path(opponent).name, gp_blue, seed, ticks]).encode())
    return h % 4, (h // 4) % 3


def make_field(directory, extra=()):
    directory.mkdir(parents=True, exist_ok=True)
    for name in FIELD:
        (directory / f"{name}.json").write_text(json.dumps({"class": HEURISTIC.format(name)}))
    for name, spec in extra:
        (directory / name).write_text(json.dumps(spec))
    return directory


def run_log(out):
    return [line for line in (out / "progress.log").read_text().splitlines() if line.startswith("generation")]


# --- defaults unchanged ---

# The old evolve_gp.py (before --panel all / --fitness) on these runs, with fake_play for
# games: its progress.log lines and hashes of its population.json and best.json.
GOLDEN_PANEL = (
    [
        "generation    1: best fitness 3.12 (size 7), median 2.22, mean size 13.7",
        "generation    2: best fitness 3.25 (size 7), median 1.88, mean size 7.4",
        "generation    3: best fitness 3.06 (size 5), median 2.00, mean size 6.7",
    ],
    ["87b0f0c64d47a484", "bf7cb5488cb2707f"],
)
GOLDEN_DIR = (
    [
        "generation    1: best fitness 3.12 (size 8), median 2.44, mean size 13.7",
        "generation    2: best fitness 2.75 (size 8), median 1.81, mean size 8.6",
        "generation    3: best fitness 2.69 (size 8), median 1.88, mean size 6.7",
    ],
    ["6ac95ef1a4bddc3e", "6a810dcb3f8f982a"],
)


def legacy_run(tmp_path, monkeypatch, panel_args):
    monkeypatch.setattr(evolve_gp, "play", fake_play)
    out = tmp_path / "out"
    args = evolve_gp.parse_args(
        ["--workers", "1", "--population", "10", "--games", "4", "--generations", "3", "--ticks", "300",
         "--out", str(out), "--seed", "3"] + panel_args
    )
    evolve_gp.run(args)
    return run_log(out), [
        hashlib.sha256((out / name).read_bytes()).hexdigest()[:16] for name in ("population.json", "best.json")
    ]


def test_default_options_match_the_old_code_with_panel(tmp_path, monkeypatch):
    assert legacy_run(tmp_path, monkeypatch, ["--panel", "SimpleBrain", "RandomWalk"]) == GOLDEN_PANEL


def test_default_options_match_the_old_code_with_panel_dir(tmp_path, monkeypatch):
    field = make_field(tmp_path / "field")
    got = legacy_run(tmp_path, monkeypatch, ["--panel-dir", str(field), "--panel-size", "2"])
    assert got == GOLDEN_DIR


def test_default_state_and_best_have_no_new_keys(tmp_path, monkeypatch):
    legacy_run(tmp_path, monkeypatch, ["--panel", "SimpleBrain"])
    assert set(json.loads((tmp_path / "out" / "population.json").read_text())) == {"generation", "population"}
    assert "fitness_measure" not in json.loads((tmp_path / "out" / "best.json").read_text())
    assert "fitness:" not in (tmp_path / "out" / "progress.log").read_text()


def test_option_defaults():
    args = evolve_gp.parse_args(["--panel-dir", "x"])
    assert (args.fitness, args.ticks_per_opponent, args.panel_all, args.exclude) == ("points", None, False, [])
    args = evolve_gp.parse_args(["--panel", "all", "--panel-dir", "x"])
    assert args.panel_all and args.ticks_per_opponent == evolve_gp.ALL_PANEL_TICKS == 3600


# --- the panel ---


def test_panel_all_is_every_field_brain_but_the_excluded_and_duplicates(tmp_path):
    own = {"class": "aisoccer.gp.v1.GPBrain", "kwargs": {"program": "ball"}, "generation": 3}
    field = make_field(
        tmp_path / "field",
        extra=[
            ("GP.json", own),
            ("champ-9.json", {"class": HEURISTIC.format("StrategicPlanner")}),  # a copy of StrategicPlanner
            ("champ-8.json", {**own, "generation": 1}),  # the same program saved earlier: a copy of GP
            ("half.tmp.json", {"class": "not.a.Brain"}),  # the league's temporary file
        ],
    )
    (field / "notes.txt").write_text("not a brain")
    names = [Path(p).stem for p in evolve_gp.field_panel(field)]
    assert names == ["BehindAndTowards", "DefendersAndAttackers", "GP", "SimpleBrain", "StrategicPlanner"]
    names = [Path(p).stem for p in evolve_gp.field_panel(field, exclude={"GP"})]
    assert names == ["BehindAndTowards", "DefendersAndAttackers", "SimpleBrain", "StrategicPlanner"]


def test_panel_all_snapshots_the_field(tmp_path):
    field = make_field(tmp_path / "field")
    snap = tmp_path / "snap"
    panel = evolve_gp.field_panel(field, snapshot=snap)
    assert [Path(p).parent for p in panel] == [snap] * len(FIELD)
    assert all(Path(p).read_text() == (field / Path(p).name).read_text() for p in panel)
    (field / "SimpleBrain.json").unlink()
    panel = evolve_gp.field_panel(field, snapshot=snap)
    assert sorted(p.name for p in snap.iterdir()) == sorted(Path(p).name for p in panel)
    assert "SimpleBrain.json" not in [p.name for p in snap.iterdir()]


@pytest.mark.parametrize(
    "total, longest, plan", [(3600, 1800, (2, 1800)), (5400, 1800, (4, 1350)), (7200, 1800, (4, 1800)),
                             (1000, 1800, (2, 500)), (6000, 1800, (4, 1500))]
)
def test_game_plan_is_an_even_number_of_games_of_at_most_ticks(total, longest, plan):
    assert evolve_gp.game_plan(total, longest) == plan


# --- the games and the fitness ---


def recorded_evaluation(monkeypatch, scores, population, panel, seeds, ticks, measure="gd"):
    tasks = []

    def play(task):
        tasks.append(task)
        return scores(task)

    monkeypatch.setattr(evolve_gp, "play", play)
    fitness, per_opponent = evolve_gp.evaluate_panel(None, population, panel, seeds, ticks, measure=measure)
    return tasks, fitness, per_opponent


def test_equal_ticks_alternated_colours_and_common_seeds(monkeypatch):
    population = [["scale", "ball", 0.5], "ball", "home"]
    panel = ["A.json", "B.npz", "C.json"]
    seeds = np.arange(6).reshape(3, 2) + 100  # 2 seeds per opponent: 4 games each
    tasks, _, _ = recorded_evaluation(monkeypatch, lambda t: (0, 0), population, panel, seeds, 900)
    assert len(tasks) == len(population) * len(panel) * 4
    per_program = {}
    for program, opponent, gp_blue, seed, ticks, _ in tasks:
        per_program.setdefault(json.dumps(program), []).append((opponent, gp_blue, seed, ticks))
    games = list(per_program.values())
    assert all(g == games[0] for g in games)  # every program plays exactly the same games
    for o, opponent in enumerate(panel):
        mine = [g for g in games[0] if g[0] == opponent]
        assert sum(g[3] for g in mine) == 3600  # the same ticks against every opponent
        assert [g[1] for g in mine] == [True, False, True, False]  # colours alternate
        assert [g[2] for g in mine] == [100 + 2 * o] * 2 + [101 + 2 * o] * 2  # one seed per pair of colours


def test_fitness_is_goal_difference_per_1000_ticks_averaged_over_opponents(monkeypatch):
    # program 0 wins 3-0 every game against A and loses 0-1 against B; program 1 draws everything
    def scores(task):
        if task[0] == "home":
            return 1, 1
        return (3, 0) if task[1] == "A" else (0, 1)

    population, panel = ["ball", "home"], ["A", "B"]
    tasks, fitness, per_opponent = recorded_evaluation(
        monkeypatch, scores, population, panel, [[1], [2]], 500
    )
    # against A: +3 per 500-tick game = +6 per 1000 ticks; against B: -2 per 1000 ticks
    np.testing.assert_allclose(per_opponent, [[6.0, -2.0], [0.0, 0.0]])
    np.testing.assert_allclose(fitness, [2.0, 0.0])
    # the points measure on the same games: per game, points + half the goal difference
    _, points, _ = recorded_evaluation(monkeypatch, scores, population, panel, [[1], [2]], 500, measure="points")
    np.testing.assert_allclose(points, [(3 + 1.5 + 0 - 0.5) / 2, 1.0])


def test_opponents_weigh_equally_whatever_their_scoring(monkeypatch):
    # a blowout against one opponent counts as much per opponent as a narrow result against another
    def scores(task):
        return (10, 0) if task[1] == "weak" else (0, 1)

    _, fitness, _ = recorded_evaluation(monkeypatch, scores, ["ball"], ["weak", "strong"], [[1], [2]], 1000)
    np.testing.assert_allclose(fitness, [(10 - 1) / 2])


# --- a run ---


def new_run(tmp_path, field, extra=(), generations=2):
    out = tmp_path / "out"
    args = evolve_gp.parse_args(
        ["--workers", "1", "--population", "8", "--generations", str(generations), "--ticks", "300",
         "--ticks-per-opponent", "600", "--panel", "all", "--panel-dir", str(field), "--exclude", "GP",
         "--fitness", "gd", "--out", str(out), "--seed", "2"] + list(extra)
    )
    evolve_gp.run(args)
    return out


def test_every_program_elites_too_plays_the_whole_field_each_generation(tmp_path, monkeypatch):
    field = make_field(tmp_path / "field", extra=[("GP.json", {"class": "aisoccer.gp.v1.GPBrain",
                                                               "kwargs": {"program": "ball"}})])
    calls = []
    real = evolve_gp.evaluate_panel

    def spy(workers, population, panel, seeds, ticks, words, measure):
        calls.append((copy_of(population), [Path(p).name for p in panel], np.array(seeds), ticks, measure))
        return real(workers, population, panel, seeds, ticks, words, measure)

    def copy_of(population):
        return json.loads(json.dumps(population))

    monkeypatch.setattr(evolve_gp, "play", fake_play)
    monkeypatch.setattr(evolve_gp, "evaluate_panel", spy)
    out = new_run(tmp_path, field, generations=3)
    assert len(calls) == 3
    for population, panel, seeds, ticks, measure in calls:
        assert len(population) == 8 and measure == "gd" and ticks == 300
        assert panel == [f"{name}.json" for name in FIELD]  # every field brain but GP itself
        assert seeds.shape == (len(FIELD), 1)
    # the elites carried into generations 2 and 3 are in the populations evaluated there
    for before, after in zip(calls, calls[1:]):
        assert all(p in before[0] for p in after[0][: evolve_gp.parse_args(["--panel", "x"]).elite])
        assert after[2].tolist() != before[2].tolist()  # new seeds each generation
    log = (out / "progress.log").read_text()
    assert "fitness: goal difference per 1000 ticks" in log and "except GP" in log
    assert "600 ticks per opponent (2 games of 300 ticks" in log
    assert log.count("best's goal difference per 1000 ticks vs 4 brains") == 3
    state = json.loads((out / "population.json").read_text())
    assert state["fitness"] == "gd" and state["generation"] == 3
    assert json.loads((out / "best.json").read_text())["fitness_measure"] == "gd"


def test_a_resumed_population_is_evaluated_afresh_under_the_new_fitness(tmp_path, monkeypatch):
    monkeypatch.setattr(evolve_gp, "play", fake_play)
    field = make_field(tmp_path / "field")
    out = tmp_path / "out"
    evolve_gp.run(evolve_gp.parse_args(
        ["--workers", "1", "--population", "8", "--games", "2", "--generations", "1", "--ticks", "300",
         "--panel-dir", str(field), "--out", str(out), "--seed", "1"]
    ))
    old = json.loads((out / "population.json").read_text())
    evaluated = []
    real = evolve_gp.evaluate_panel
    monkeypatch.setattr(
        evolve_gp, "evaluate_panel", lambda *a: evaluated.append(a[1]) or real(*a)
    )
    new_run(tmp_path, field, generations=1)
    assert evaluated == [old["population"]]
    log = (out / "progress.log").read_text()
    assert "resumed at generation 1" in log and "fitness changed from points to gd" in log
    assert json.loads((out / "population.json").read_text())["generation"] == 2


def test_real_games_smoke(tmp_path):
    field = make_field(tmp_path / "field")
    out = new_run(tmp_path, field, extra=["--population", "4"], generations=1)
    spec = json.loads((out / "best.json").read_text())
    assert spec["fitness_measure"] == "gd" and np.isfinite(spec["fitness"])
    assert isinstance(load_brain(str(out / "best.json")), GPBrain)
    assert sorted(p.name for p in (out / "panel").iterdir()) == sorted(f"{n}.json" for n in FIELD)


# --- weighting opponents by family (--weight-by family) ---

FAMILIES = {"GP": "GP", "GP-v2": "GP", "PPO-a": "PPO", "PPO-b": "PPO", "PPO-c": "PPO", "LLM-v11": "LLM",
            "Tactics": "LLM"}


def league_files(directory, families=FAMILIES, champion_names=None):
    directory.mkdir(parents=True, exist_ok=True)
    roster = directory / "roster.json"
    roster.write_text(json.dumps({"processes": {}, "families": families}))
    if champion_names is not None:
        (directory / "state.json").write_text(json.dumps({"round": 3, "champion_names": champion_names}))
    return roster


def test_weight_options_default_off():
    args = evolve_gp.parse_args(["--panel", "all", "--panel-dir", "x"])
    assert (args.weight_by, args.families) == ("opponent", None)


def test_default_gd_run_has_no_family_traces(tmp_path, monkeypatch):
    monkeypatch.setattr(evolve_gp, "play", fake_play)
    out = new_run(tmp_path, make_field(tmp_path / "field"), generations=1)
    assert "weight_by" not in json.loads((out / "population.json").read_text())
    assert "fitness_weight_by" not in json.loads((out / "best.json").read_text())
    assert "family" not in (out / "progress.log").read_text()


def test_family_weights_sum_to_one_and_are_equal_per_family():
    panel = ["f/GP-v2.json", "f/PPO-a.npz", "f/PPO-b.npz", "f/PPO-c.npz", "f/Tactics.json", "f/Odd.json"]
    weights, members = evolve_gp.family_weights(panel, FAMILIES, {})
    assert members == {"GP": ["GP-v2"], "PPO": ["PPO-a", "PPO-b", "PPO-c"], "LLM": ["Tactics"], "Odd": ["Odd"]}
    assert weights.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(weights, [0.25, 0.25 / 3, 0.25 / 3, 0.25 / 3, 0.25, 0.25])
    family_total = {}
    for path, w in zip(panel, weights):
        family = evolve_gp.brain_family(path, FAMILIES, {})
        family_total[family] = family_total.get(family, 0.0) + w
    np.testing.assert_allclose(list(family_total.values()), [0.25] * 4)


def test_champions_map_to_the_family_they_were_promoted_from(tmp_path):
    field = tmp_path / "field"
    field.mkdir()
    specs = {
        "champ-8.json": {"class": "aisoccer.llm.v11.LLMBrainV11"},  # recorded: promoted from LLM-v11
        "champ-7.json": {"class": "aisoccer.gp.v1.GPBrain", "kwargs": {"program": "ball"}},  # recorded as PPO-a
        "champ-9.json": {"class": "aisoccer.brains.SimpleBrain.SimpleBrain"},  # recorded, origin not in families
        "champ-5.json": {"class": "aisoccer.gp.v1.GPBrain", "kwargs": {"program": "ball"}},
        "champ-4.json": {"class": "aisoccer.gp.v2.GPBrainV2", "kwargs": {"program": "ball"}},
        "champ-3.json": {"class": "aisoccer.tactics.v1.TacticsV1"},
        "champ-2.json": {"class": "aisoccer.llm.v5.LLMBrainV5"},
        "champ-1.json": {"class": "aisoccer.brains.SimpleBrain.SimpleBrain"},
        "champ-6.json": {"class": "aisoccer.tactics.learned.TacticianBrain"},
    }
    for name, spec in specs.items():
        (field / name).write_text(json.dumps(spec))
    (field / "PPO-champ-0.npz").write_bytes(b"weights")
    names = {"champ-8": "LLM-v11", "champ-7": "PPO-a", "champ-9": "Newcomer"}
    got = {p.stem: evolve_gp.brain_family(p, FAMILIES, names) for p in sorted(field.iterdir())}
    assert got == {
        "champ-8": "LLM", "champ-7": "PPO", "champ-9": "Newcomer",  # via champion_names
        "champ-5": "GP", "champ-4": "GP", "champ-3": "LLM", "champ-2": "LLM", "champ-6": "PPO",  # via the class
        "PPO-champ-0": "PPO",  # a .npz is a PPOBrain
        "champ-1": "champions",  # no record and an unfamiliar class
    }
    # a name in the roster's families wins; any other brain is a family of its own
    assert evolve_gp.brain_family("x/GP.json", FAMILIES, names) == "GP"
    assert evolve_gp.brain_family("x/Coach.json", FAMILIES, names) == "Coach"


def test_read_league_takes_champion_names_from_the_state_beside_the_roster(tmp_path):
    roster = league_files(tmp_path / "league")
    assert evolve_gp.read_league(roster) == (FAMILIES, {})  # no state.json yet
    roster = league_files(tmp_path / "league", champion_names={"champ-8": "LLM-v11"})
    assert evolve_gp.read_league(roster) == (FAMILIES, {"champ-8": "LLM-v11"})


def test_family_weighted_run_excludes_itself_and_weights_by_family(tmp_path, monkeypatch):
    monkeypatch.setattr(evolve_gp, "play", fake_play)
    field = make_field(tmp_path / "field", extra=[
        ("GP.json", {"class": "aisoccer.gp.v1.GPBrain", "kwargs": {"program": "ball"}}),
        ("GP-v2.json", {"class": "aisoccer.gp.v2.GPBrainV2", "kwargs": {"program": "home"}}),
        ("Tactics.json", {"class": "aisoccer.tactics.v1.TacticsV1"}),
        ("champ-8.json", {"class": "aisoccer.llm.v11.LLMBrainV11"}),
        ("champ-5.json", {"class": "aisoccer.gp.v1.GPBrain", "kwargs": {"program": "opp"}}),
    ])
    roster = league_files(tmp_path / "league", champion_names={"champ-8": "LLM-v11"})
    calls = []
    real = evolve_gp.evaluate_panel

    def spy(*a):
        result = real(*a)
        calls.append(([Path(p).stem for p in a[2]], result[1]))
        return result

    monkeypatch.setattr(evolve_gp, "evaluate_panel", spy)
    out = new_run(tmp_path, field, extra=["--weight-by", "family", "--families", str(roster)], generations=2)
    panel, per_opponent = calls[-1]
    assert "GP" not in panel and len(panel) == len(FIELD) + 4
    family = {"GP-v2": "GP", "champ-5": "GP", "Tactics": "LLM", "champ-8": "LLM"}
    weights = np.array([
        1 / 6 / (2 if name in family else 1) for name in panel  # 6 families: GP, LLM and 4 heuristics
    ])
    assert weights.sum() == pytest.approx(1.0)
    best = json.loads((out / "best.json").read_text())
    assert best["fitness_weight_by"] == "family" and best["fitness_measure"] == "gd"
    assert max(per_opponent @ weights) >= best["fitness"] >= min(per_opponent @ weights)
    assert any(best["fitness"] == pytest.approx(f) for f in per_opponent @ weights)
    log = (out / "progress.log").read_text()
    assert "averaged over families" in log and "except GP" in log
    assert log.count("weights by family") == 1  # logged when they change, here only once
    assert "GP 0.167 (GP-v2 0.083, champ-5 0.083)" in log and "LLM 0.167 (Tactics 0.083, champ-8 0.083)" in log
    assert log.count("best's goal difference per 1000 ticks vs 8 brains") == 2  # per-opponent log kept
    assert json.loads((out / "population.json").read_text())["weight_by"] == "family"


def test_family_weighting_needs_gd_and_a_roster(tmp_path):
    field = make_field(tmp_path / "field")
    with pytest.raises(SystemExit):
        new_run(tmp_path, field, extra=["--weight-by", "family"], generations=1)
    roster = league_files(tmp_path / "league")
    with pytest.raises(SystemExit):
        new_run(tmp_path, field, extra=["--weight-by", "family", "--families", str(roster), "--fitness", "points"])
