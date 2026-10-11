"""GPBrain v3 (aisoccer/gp/v3.py) and evolve_gp.py's v3, re-evaluation and xG options."""

import hashlib
import json
import random
import zlib
from pathlib import Path

import numpy as np
import pytest

import evolve_gp
from aisoccer.brainspec import load_brain
from aisoccer.constants import Constants
from aisoccer.game import Game
from aisoccer.gp import v2, v3
from aisoccer.gp.v2 import GPBrainV2
from aisoccer.gp.v3 import GPBrainV3, from_v2
from aisoccer.gp.xgame import XGGame, is_touch

V3 = evolve_gp.vocab("v3")
RUNS = Path(__file__).resolve().parent.parent / "runs"
STATE = ("my_players_pos", "my_players_vel", "opp_players_pos", "opp_players_vel", "ball_pos", "ball_vel",
         "my_score", "opp_score", "game_time")
# striker (nearest the ball) hits at goal, keeper (deepest of the rest) at save_point, the rest in graded depths
FORMATION = [
    "roles",
    ["role", "ball_dist", ["hit", "opp_goal"]],
    ["role", ["x", "me"], "save_point"],
    ["role", 0.0, ["point", ["sub", ["x", "ball"], ["add", 200.0, ["mul", 150.0, "rank"]]], "lane_y"]],
]
EVERY_NEW_WORD = [
    "roles",
    ["role", ["add", "reach", ["mul", "goal_dist", "time_left"]], ["if_v", "we_have_ball", "pass", "clear"]],
    ["role", ["sub", "lane", "score_diff"], ["if_v", "shot_coming", "save_point", ["mark", "threat"]]],
    ["role", "rank", ["between", "open_goal", ["hit", "open_goal"], ["mul", "lane_y", 0.001]]],
    ["role", 0.0, ["behind", ["strike", "threat"], ["ball_in", "reach"], ["add", "lane", "ball_dist"]]],
]


def v3_programs(count, seed=0):
    return evolve_gp.ramped(random.Random(seed), count, max_size=120, words=V3)


def random_state(brain, rng, spread=1.0):
    brain.my_players_pos = rng.random((5, 2)) * [1799, 800] * spread
    brain.my_players_vel = rng.normal(0, 3, (5, 2))
    brain.opp_players_pos = rng.random((5, 2)) * [1799, 800] * spread
    brain.opp_players_vel = rng.normal(0, 3, (5, 2))
    brain.ball_pos = rng.random(2) * [1799, 800] * spread
    brain.ball_vel = rng.normal(0, 5, 2)
    brain.my_score, brain.opp_score = int(rng.integers(0, 6)), int(rng.integers(0, 6))
    brain.game_time = float(rng.random())
    brain._tick = int(rng.integers(0, 36000))


def shot_state(brain, rng):
    random_state(brain, rng)
    brain.ball_pos = np.array([300.0 + 300 * rng.random(), 250.0 + 300 * rng.random()])
    brain.ball_vel = np.array([-10.0, rng.normal(0, 2)])


def copy_state(a, b):
    for k in STATE + ("_tick",):
        setattr(b, k, getattr(a, k))


def live_v2_programs():
    programs = []
    for f in sorted(RUNS.glob("*/gp_v2/*.json")):
        try:
            d = json.loads(f.read_text())
        except ValueError:  # being rewritten by the live run
            continue
        programs += d.get("population", []) + ([d["kwargs"]["program"]] if "kwargs" in d else [])
    return programs


# --- the language ---


def test_every_v3_word_appears_in_random_programs_and_they_are_well_typed():
    words = set()
    for p in v3_programs(300, seed=1):
        v3.validate(p)
        assert 2 <= len(p) - 1 <= 5
        words |= set(json.dumps(p).replace("[", " ").replace("]", " ").replace(",", " ").replace('"', " ").split())
    assert set(v3.FUNCTIONS) <= words and set(v3.TERMINALS) <= words


def test_v3_compiled_equals_interpreted():
    rng = np.random.default_rng(0)
    programs = v3_programs(150, seed=2) + [FORMATION, EVERY_NEW_WORD, ["roles"] + EVERY_NEW_WORD[1:] * 2]
    for program in programs:
        brain = GPBrainV3(program=program)
        for spread in (1.0, 0.0, 3.0, "shot"):
            if spread == "shot":
                shot_state(brain, rng)
            else:
                random_state(brain, rng, spread)
            got = np.array(brain.targets())
            assert np.isfinite(got).all()
            np.testing.assert_allclose(got, v3.interpret_targets(program, brain), rtol=1e-9, atol=1e-6)


def test_validate_rejects_bad_programs():
    for bad in ("ball", ["roles"], ["roles"] + [["role", 0.0, "ball"]] * 9, ["roles", ["role", "ball", "ball"]],
                ["roles", ["role", 0.0, "rank"]], ["roles", ["role", 0.0]]):
        with pytest.raises(ValueError):
            v3.validate(bad)


def test_readable_text_has_one_function_per_role():
    text = v3.to_text(FORMATION)
    assert "def role_1():  # 1 player, by: ball_dist" in text
    assert "def role_2():  # 1 player, by: x(me)" in text
    assert "def role_3():  # everyone left" in text and "return point(x(ball) - (200 + 150 * rank), lane_y)" in text
    dormant = v3.to_text(["roles"] + [["role", 0.0, "ball"]] * 6)
    assert "def role_6():  # dormant" in dormant


# --- roles ---


def formation_brain():
    brain = GPBrainV3(program=FORMATION)
    brain.my_players_pos = np.array([[900.0, 400.0], [100.0, 380.0], [500.0, 100.0], [300.0, 700.0], [700.0, 500.0]])
    brain.my_players_vel = np.zeros((5, 2))
    brain.opp_players_pos = np.array([[1500.0, 400.0], [1300.0, 200.0], [1300.0, 600.0], [1100.0, 300.0],
                                      [1000.0, 500.0]])
    brain.opp_players_vel = np.zeros((5, 2))
    brain.ball_pos, brain.ball_vel = np.array([950.0, 420.0]), np.zeros(2)
    brain.my_score = brain.opp_score = 0
    brain.game_time = 0.1
    return brain


def test_roles_fill_in_order_and_give_distinct_jobs():
    brain = formation_brain()
    names = v3.terminals_used(FORMATION) | {"me"}
    job = v3.assign(FORMATION, v3.terminal_arrays(brain, names))
    assert job == [0, 1, 2, 2, 2]  # nearest the ball strikes, the deepest other keeps, three fill the field
    t = np.array(brain.targets())
    assert t[1][0] == pytest.approx(v2.KEEPER_X)
    field = {i: t[i] for i in (2, 3, 4)}  # by depth: 3 (x 300) rank 0, 2 (x 500) rank 1, 4 (x 700) rank 2
    assert field[3][0] == pytest.approx(950 - 200) and field[2][0] == pytest.approx(950 - 350)
    assert field[4][0] == pytest.approx(950 - 500)
    assert sorted(p[1] for p in field.values()) == pytest.approx([200.0, 400.0, 600.0])  # lanes by height
    assert len({tuple(np.round(p, 3)) for p in t}) == 5


def test_rank_inside_by_is_over_the_whole_team_and_the_last_by_is_unused():
    brain = formation_brain()
    by_rank = ["roles", ["role", "rank", "own_goal"], ["role", ["sub", 0.0, "rank"], "opp_goal"],
               ["role", "me", "ball"]]
    with pytest.raises(ValueError):
        v3.validate(by_rank)  # a BY is a number
    by_rank[3][1] = ["mul", 1e9, "rank"]
    names = v3.terminals_used(by_rank) | {"me"}
    job = v3.assign(by_rank, v3.terminal_arrays(brain, names))
    assert job == [1, 0, 2, 2, 2]  # deepest of the team (player 1), then the most advanced (player 0)
    assert "rank" not in v3.terminals_used(["roles", ["role", "rank", "ball"]])  # a lone role's BY is never read


def test_roles_past_the_fifth_are_dormant():
    base = ["roles"] + [["role", ["y", "me"], ["point", float(100 * k), 300.0]] for k in range(5)]
    brain = formation_brain()
    t5 = np.array(GPBrainV3(program=base)._fn(v3.python_terminals(brain, GPBrainV3(program=base)._names)))
    more = base + [["role", "reach", "clear"], ["role", 0.0, "pass"]]
    b7 = GPBrainV3(program=more)
    np.testing.assert_array_equal(np.array(b7._fn(v3.python_terminals(brain, b7._names))), t5)
    assert "clear" not in b7._names and "reach" not in b7._names


# --- v2 programs translate exactly ---


def test_v2_programs_translate_to_bit_identical_targets():
    rng = np.random.default_rng(3)
    programs = live_v2_programs() + evolve_gp.ramped(random.Random(4), 80, words=evolve_gp.vocab("v2"))
    programs.append(["if_v", ["and", "nearest_to_ball", ["not", "nearest_to_ball"]], "ball", "save_point"])
    for program in programs:
        a, b = GPBrainV2(program=program), GPBrainV3(program=from_v2(program))
        assert GPBrainV3(program=program).program == b.program  # v2 programs are translated on load
        for k in range(60):
            if k % 4 == 3:
                shot_state(a, rng)
            else:
                random_state(a, rng, (1.0, 0.0, 3.0)[k % 3])
            copy_state(a, b)
            assert a.targets() == b.targets()


class Recorder:
    def __init__(self, cls):
        class Recording(cls):
            def do_move(inner):
                move = super().do_move()
                self.moves.append(move.copy())
                return move

        self.moves = []
        self.cls = Recording


def test_v2_best_translated_plays_identical_games_for_both_colours():
    best = RUNS / "league3" / "gp_v2" / "best.json"
    program = json.loads(best.read_text())["kwargs"]["program"] if best.exists() else [
        "behind", ["if_v", "nearest_to_ball", ["strike", "opp_goal"], "save_point"], "intercept", "role"]
    for blue in (True, False):
        results = []
        for cls in (GPBrainV2, GPBrainV3):
            rec = Recorder(cls)
            gp, other = rec.cls("gp", program=program), load_brain("SimpleBrain")
            score = Game(*((gp, other) if blue else (other, gp)), game_length=600, quiet_mode=True, seed=11).play()
            results.append((score, np.array(rec.moves)))
        assert results[0][0] == results[1][0]
        np.testing.assert_array_equal(results[0][1], results[1][1])


# --- the new words ---


def test_hit_runs_through_within_60_degrees_and_detours_at_full_speed():
    bx, by, ax, ay = 900.0, 400.0, 1779.0, 400.0
    c = np.array([bx - v2.CONTACT, by])
    # 50 degrees off the aim line: strike lines up first, hit runs straight through the contact point
    p = c - 150.0 * np.array([np.cos(np.radians(50)), np.sin(np.radians(50))])
    w = (c - p) / np.linalg.norm(c - p)
    np.testing.assert_allclose(v3.hit_point(bx, by, *p, ax, ay), c + w * v3.HIT_THROUGH)
    assert np.linalg.norm(np.subtract(v2.strike_point(bx, by, *p, ax, ay), c + w * v2.THROUGH)) > 1.0
    # wider, near the line-up point: strike stops there, hit heads the same way but at least HIT_RUN px on
    p = np.array([bx - v2.CONTACT - 40.0, by - 100.0])  # 68 degrees off: both head for the line-up point
    s, h = np.array(v2.strike_point(bx, by, *p, ax, ay)), np.array(v3.hit_point(bx, by, *p, ax, ay))
    assert np.linalg.norm(s - p) < v3.HIT_RUN and np.linalg.norm(h - p) == pytest.approx(v3.HIT_RUN)
    np.testing.assert_allclose((h - p) / np.linalg.norm(h - p), (s - p) / np.linalg.norm(s - p))


def test_clear_goes_forward_into_open_space_and_pass_finds_the_open_forward_mate():
    brain = formation_brain()
    # opponents wall off straight ahead: the clear goes up or down the field, never back
    brain.opp_players_pos = np.array([[1100.0, 420.0], [1100.0, 360.0], [1100.0, 480.0], [1200.0, 420.0],
                                      [1300.0, 420.0]])
    T = v3.python_terminals(brain, ["clear", "pass", "intercept", "open_goal"])
    kx, ky = T["intercept"][0]
    cx, cy = T["clear"][0]
    assert abs(cy - ky) > 1.0  # not straight ahead into the wall
    # player 0 passes to the furthest-forward open mate: player 4 (x 700) is behind the wall? lane clear
    mates = brain.my_players_pos
    px, py = T["pass"][0]
    assert (px, py) != pytest.approx((cx, cy))
    gx, gy = T["open_goal"][0]
    assert gx == pytest.approx(v3.OPP_GOAL[0]) and Constants.GOAL_Y_MIN < gy < Constants.GOAL_Y_MAX
    assert gy != pytest.approx(420.0, abs=40)  # away from the blocked middle
    assert mates.shape == (5, 2)


def test_state_words_threat_mark_between_shot_coming_we_have_ball_score_and_time():
    brain = formation_brain()
    T = v3.python_terminals(brain, ["threat", "we_have_ball", "shot_coming", "score_diff", "time_left"])
    assert T["threat"][0] == (1000.0, 500.0)  # ball in their half: the opponent nearest my goal
    assert T["we_have_ball"][0] is True
    assert T["shot_coming"][0] is False
    brain.ball_pos, brain.ball_vel = np.array([400.0, 400.0]), np.array([-10.0, 1.0])
    brain.opp_players_pos[4] = [420.0, 410.0]
    T = v3.python_terminals(brain, ["threat", "we_have_ball", "shot_coming"])
    assert T["threat"][0] == (420.0, 410.0) and T["we_have_ball"][0] is False and T["shot_coming"][0] is True
    brain.ball_vel = np.array([-10.0, 6.0])  # crosses the goal line at y 400 + 6 x 38 = 628: wide
    assert v3.python_terminals(brain, ["shot_coming"])["shot_coming"][0] is False
    assert v3._mark(500.0, 400.0) == pytest.approx((500.0 - v3.MARK_GAP, 400.0))
    assert v3._between(0.0, 0.0, 100.0, 50.0, 1.7) == (100.0, 50.0) and v3._between(0, 0, 100, 50, 0.5) == (50, 25)
    brain.my_score, brain.opp_score = 7, 1
    assert v3.python_terminals(brain, ["score_diff"])["score_diff"][0] == 1.0
    brain.my_score, brain.opp_score = 1, 2
    assert v3.python_terminals(brain, ["score_diff"])["score_diff"][0] == pytest.approx(-1 / 3)
    for tick, game_time, want in ((0, 0.0, 1.0), (100, 100 / 36000, 1.0), (35100, 35100 / 36000, 0.5),
                                  (900, 0.5, 0.5), (1799, 1799 / 1800, 1 / 1800)):
        assert v3.time_left(tick, game_time) == pytest.approx(want)


def test_time_left_counts_the_brain_s_own_ticks_in_a_game():
    seen = []

    class Watch(GPBrainV3):
        def targets(inner):
            seen.append(v3.python_terminals(inner, ["time_left"])["time_left"][0])
            return super().targets()

    Game(Watch("w", program=["roles", ["role", 0.0, "ball"]]), load_brain("SimpleBrain"), game_length=2000,
         quiet_mode=True, seed=1).play()
    assert len(seen) >= 1990 and seen[0] == 1.0 and seen[150] == 1.0
    assert seen[1100] == pytest.approx((2000 - 1100) / 1800) and min(seen) < 0.01


def first_kick(word, blue, me_xy, ball_xy=(900.0, 300.0), ticks=200, mate=(100.0, 700.0)):
    """
    One GP player at me_xy (its own frame) with a still ball at ball_xy, the others out of
    the way: (ticks to the first touch, the ball's direction then in degrees, its speed).
    """
    gp = GPBrainV3("gp", program=["roles", ["role", "ball_dist", word], ["role", 0.0, "me"]])
    still = v2.GPBrainV2("still", program="me")
    game = Game(*((gp, still) if blue else (still, gp)), game_length=ticks, quiet_mode=True, seed=1)
    fx = (lambda x: x) if blue else (lambda x: Constants.FIELD_LENGTH - 1 - x)
    mine = [me_xy, mate, (160.0, 700.0), (220.0, 700.0), (280.0, 700.0)]
    theirs = [(1700.0, 60.0), (1700.0, 120.0), (1640.0, 60.0), (1640.0, 120.0), (1580.0, 60.0)]
    for team, spots in ((game.teams[0 if blue else 1], mine), (game.teams[1 if blue else 0], theirs)):
        for player, (x, y) in zip(team.players, spots):
            player.body.position = np.array([fx(x), y])
            player.body.velocity = np.zeros(2)
    game.ball.body.position = np.array([fx(ball_xy[0]), ball_xy[1]])
    game.ball.body.velocity = np.zeros(2)
    for t in range(ticks):
        game.tick()
        vx, vy = game.ball.body.velocity
        if np.hypot(vx, vy) > 0.5:
            return t, float(np.degrees(np.arctan2(vy, vx if blue else -vx))), float(np.hypot(vx, vy))
    return None


def test_hit_kicks_where_it_aims_sooner_than_strike_for_both_colours():
    to_goal = float(np.degrees(np.arctan2(400.0 - 300.0, 1779.0 - 900.0)))
    for me in ((700.0, 250.0), (800.0, 100.0), (880.0, 150.0), (1000.0, 300.0), (850.0, 500.0)):
        hit = [first_kick(["hit", "opp_goal"], blue, me) for blue in (True, False)]
        strike = [first_kick(["strike", "opp_goal"], blue, me) for blue in (True, False)]
        assert hit[0] == pytest.approx(hit[1]) and strike[0] == pytest.approx(strike[1])  # either colour
        assert hit[0][1] == pytest.approx(to_goal, abs=2.0) and hit[0][2] > 5.0
        assert hit[0][0] <= strike[0][0]


def test_clear_kicks_forward_and_pass_finds_the_open_forward_mate_for_both_colours():
    kicks = [first_kick("clear", blue, (800.0, 250.0)) for blue in (True, False)]
    assert kicks[0] == pytest.approx(kicks[1]) and abs(kicks[0][1]) <= 71.0, kicks  # forward, not back
    to_mate = float(np.degrees(np.arctan2(650.0 - 300.0, 1300.0 - 900.0)))
    kicks = [first_kick("pass", blue, (800.0, 250.0), mate=(1300.0, 650.0)) for blue in (True, False)]
    assert kicks[0] == pytest.approx(kicks[1]) and kicks[0][1] == pytest.approx(to_mate, abs=3.0), kicks


def test_clear_scores_against_a_team_that_stands_still_for_both_colours():
    program = ["roles", ["role", "ball_dist", "clear"], ["role", 0.0, "home"]]
    for blue in (True, False):
        gp, still = GPBrainV3("gp", program=program), v2.GPBrainV2("still", program="me")
        score = Game(*((gp, still) if blue else (still, gp)), game_length=1200, quiet_mode=True, seed=3).play()
        me, them = ("blue", "red") if blue else ("red", "blue")
        assert score[me] >= 3 and score[them] == 0, (blue, score)


def test_threat_and_mark_keep_a_marker_goal_side_for_both_colours():
    for blue in (True, False):
        marker = GPBrainV3("m", program=["roles", ["role", "goal_dist", ["mark", "threat"]], ["role", 0.0, "home"]])
        runner = v2.GPBrainV2("r", program=["strike", "opp_goal"])
        game = Game(*((marker, runner) if blue else (runner, marker)), game_length=300, quiet_mode=True, seed=2)
        game.play()
        assert np.isfinite(game.teams[0 if blue else 1].position_matrix()).all()


# --- role-level operators ---


def test_role_operators_keep_programs_valid_and_compilable():
    rng = random.Random(5)
    programs = v3_programs(40, seed=6) + [FORMATION, EVERY_NEW_WORD]
    words = V3
    seen = set()
    for n in range(600):
        a, b = rng.choice(programs), rng.choice(programs)
        mutate = (lambda p: evolve_gp.subtree_mutation(rng, p, words)) if n % 2 else (
            lambda p: evolve_gp.point_mutation(rng, p, words=words))
        child, op = v3.structural(rng, a, b, lambda t, d: evolve_gp.random_tree(rng, t, d, False, words), mutate,
                                  lambda p, q: evolve_gp.crossover(rng, p, q, words))
        seen.add(op)
        v3.validate(child)
        assert 1 <= len(child) - 1 <= v3.MAX_ROLES
        brain = GPBrainV3(program=child)
        random_state(brain, np.random.default_rng(n))
        assert np.isfinite(np.array(brain.targets())).all()
        if op == "add":
            assert len(child) == len(a) + 1
        elif op == "delete":
            assert len(child) == len(a) - 1
        elif op == "move":
            assert sorted(json.dumps(r) for r in child[1:]) == sorted(json.dumps(r) for r in a[1:])
        elif op == "take":
            assert any(r in b[1:] for r in child[1:])
        if len(programs) < 200:
            programs.append(child)
    assert seen == set(v3.role_operators())
    full = ["roles"] + [["role", 0.0, "ball"]] * v3.MAX_ROLES
    for _ in range(50):
        child, op = v3.structural(rng, full, full, lambda t, d: "ball", lambda p: p, lambda p, q: p)
        assert op in ("delete", "move", "take") and len(child) - 1 <= v3.MAX_ROLES
    one = ["roles", ["role", 0.0, "ball"]]
    for _ in range(50):
        child, op = v3.structural(rng, one, one, lambda t, d: "ball", lambda p: p, lambda p, q: p)
        assert len(child) - 1 >= 1


def test_ordinary_operators_never_touch_the_role_structure():
    rng = random.Random(7)
    programs = v3_programs(30, seed=8)
    for _ in range(300):
        a, b = rng.choice(programs), rng.choice(programs)
        for child in (evolve_gp.crossover(rng, a, b, V3), evolve_gp.subtree_mutation(rng, a, V3),
                      evolve_gp.point_mutation(rng, a, words=V3)):
            v3.validate(child)
            assert len(child) == len(a)


def test_breeding_v3_makes_valid_children_within_the_limits():
    rng = random.Random(9)
    population = v3_programs(20, seed=10)
    args = evolve_gp.parse_args(["--vocab", "v3", "--panel", "x", "--max-size", "120", "--max-depth", "14"])
    for _ in range(5):
        population = evolve_gp.breed(rng, population, list(np.arange(len(population), dtype=float)), args)
        for p in population:
            v3.validate(p)
            assert evolve_gp.size(p) <= 120 and evolve_gp.depth(p) <= 14 or p in population[:2]
            GPBrainV3(program=p)


# --- evolve_gp: defaults unchanged ---


def fake_play(task):
    """A stand-in for a game: a deterministic score from the task, with no physics."""
    program, opponent, gp_blue, seed, ticks = task[:5]
    h = zlib.crc32(json.dumps([program, Path(opponent).name, gp_blue, seed, ticks]).encode())
    return h % 4, (h // 4) % 3


def fake_play_xg(task):
    gf, ga = fake_play(task)
    h = zlib.crc32(json.dumps([task[0], task[3]]).encode())
    return gf, ga, (h % 100) / 50.0, (h // 100 % 100) / 50.0


HEURISTIC = "aisoccer.brains.{0}.{0}"


def league_dir(tmp_path):
    field = tmp_path / "field"
    field.mkdir()
    for n in ["BehindAndTowards", "DefendersAndAttackers", "SimpleBrain", "StrategicPlanner"]:
        (field / f"{n}.json").write_text(json.dumps({"class": HEURISTIC.format(n)}))
    (field / "GP-v2.json").write_text(json.dumps({"class": "aisoccer.gp.v2.GPBrainV2", "kwargs": {"program": "ball"}}))
    (field / "champ-0.json").write_text(json.dumps({"class": "aisoccer.gp.v1.GPBrain", "kwargs": {"program": "me"}}))
    roster = tmp_path / "roster.json"
    roster.write_text(json.dumps({"families": {"SimpleBrain": "H", "StrategicPlanner": "H", "GP-v2": "GP"}}))
    (tmp_path / "state.json").write_text(json.dumps({"champion_names": {}}))
    return field, roster


def gp_v2_style_args(tmp_path, extra=(), vocab="v2", seeds=None, out="out"):
    field, roster = league_dir(tmp_path)
    if seeds is None:
        seeds = [tmp_path / "seed.json"]
        seeds[0].write_text(json.dumps({"class": "aisoccer.gp.v1.GPBrain",
                                        "kwargs": {"program": ["behind", "ball", "own_goal", -30.0]}}))
    return evolve_gp.parse_args(
        ["--vocab", vocab, "--seed-programs", *map(str, seeds), "--workers", "1", "--panel", "all",
         "--exclude", "GP-v2", "--ticks-per-opponent", "600", "--fitness", "gd", "--parsimony", "0.0015",
         "--panel-dir", str(field), "--out", str(tmp_path / out), "--crossover", "0.5", "--weight-by", "family",
         "--families", str(roster), "--population", "10", "--generations", "3", "--seed", "3", "--ticks", "300",
         *extra]
    )


# GP-v2's live configuration (v2, seeded, --panel all, gd, family weights, crossover 0.5) with
# fake games, run by evolve_gp.py as it was before v3: its log lines and file hashes.
GOLDEN_V2 = (
    [
        "generation    1: best fitness 2.71 (size 2), median 1.56, mean size 9.8",
        "generation    2: best fitness 3.54 (size 5), median 2.50, mean size 8.8",
        "generation    3: best fitness 3.54 (size 25), median 1.67, mean size 7.5",
    ],
    ["589cab147a3f6783", "05a341b5315e5feb", "7924e74e8a0f18bd"],
)


def test_gp_v2_configuration_is_bit_identical_to_before(tmp_path, monkeypatch):
    monkeypatch.setattr(evolve_gp, "play", fake_play)
    args = gp_v2_style_args(tmp_path)
    evolve_gp.run(args)
    out = tmp_path / "out"
    log = [line for line in (out / "progress.log").read_text().splitlines() if line.startswith("generation")]
    names = ("population.json", "best.json", "best.txt")
    hashes = [hashlib.sha256((out / n).read_bytes()).hexdigest()[:16] for n in names]
    assert (log, hashes) == GOLDEN_V2
    assert "history" not in json.loads((out / "population.json").read_text())


def test_new_option_defaults_are_off():
    args = evolve_gp.parse_args(["--panel", "x"])
    assert (args.reeval_top, args.reeval_memory, args.xg_weight, args.xg_model) == (0, 0, 0.0, None)
    assert evolve_gp.vocab("v2").structural is None and evolve_gp.vocab("v2").convert is None
    assert evolve_gp.vocab("v1").terms_by_type == evolve_gp.TERMS_BY_TYPE


def test_denoising_needs_gd(tmp_path):
    with pytest.raises(SystemExit):
        evolve_gp.run(evolve_gp.parse_args(["--panel", "SimpleBrain", "--out", str(tmp_path), "--reeval-top", "4"]))


# --- evolve_gp: v3 seeded from GP-v2 ---


def v2_seed_files(tmp_path):
    rng = random.Random(11)
    pop = evolve_gp.ramped(rng, 6, words=evolve_gp.vocab("v2"))
    pop.append(["if_v", "nearest_to_ball", ["strike", "opp_goal"], ["behind", "intercept", "own_goal", -200.0]])
    best = tmp_path / "v2best.json"
    best.write_text(json.dumps({"class": "aisoccer.gp.v2.GPBrainV2", "kwargs": {"program": pop[-1]},
                                "generation": 393, "fitness": 1.03}))
    popf = tmp_path / "v2pop.json"
    popf.write_text(json.dumps({"generation": 393, "population": pop, "vocab": "v2"}))
    return best, popf, pop


class Stop(Exception):
    pass


def test_v3_starts_from_the_whole_translated_v2_population_and_its_best(tmp_path, monkeypatch):
    best, popf, pop = v2_seed_files(tmp_path)

    def stop(*a, **k):
        raise Stop

    monkeypatch.setattr(evolve_gp, "evaluate_panel", stop)
    args = gp_v2_style_args(tmp_path, ["--max-size", "120", "--max-depth", "14"], vocab="v3", seeds=[best, popf])
    with pytest.raises(Stop):
        evolve_gp.run(args)
    out = tmp_path / "out"
    spec = json.loads((out / "best.json").read_text())
    assert spec["generation"] == 0 and spec["kwargs"]["program"] == from_v2(pop[-1])
    assert spec["class"] == "aisoccer.gp.v3.GPBrainV3" and isinstance(load_brain(str(out / "best.json")), GPBrainV3)
    log = (out / "progress.log").read_text()
    assert "7 programs seeded" in log and "3 random programs" in log and "translated" in log  # best == pop[-1]


def test_v3_run_with_reevaluation_memory_xg_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(evolve_gp, "play", fake_play)
    monkeypatch.setattr(evolve_gp, "play_xg", fake_play_xg)
    calls = []
    real = evolve_gp.evaluate_seeded

    def counting(workers, programs, panel, seeds, *a, **k):
        calls.append((len(programs), np.asarray(seeds).shape[0]))
        return real(workers, programs, panel, seeds, *a, **k)

    monkeypatch.setattr(evolve_gp, "evaluate_seeded", counting)
    best, popf, pop = v2_seed_files(tmp_path)
    extra = ["--max-size", "120", "--max-depth", "14", "--reeval-top", "4", "--reeval-reps", "2",
             "--reeval-memory", "6", "--xg-weight", "0.1", "--xg-model", "none.json"]
    args = gp_v2_style_args(tmp_path, extra, vocab="v3", seeds=[best, popf])
    evolve_gp.run(args)
    assert calls == [(10, 1), (4, 2)] * 3
    out = tmp_path / "out"
    lines = (out / "progress.log").read_text().splitlines()
    gens = [line for line in lines if line.startswith("generation")]
    assert len(gens) == 3 and all("mean of" in g and "evaluations" in g and "screen max" in g for g in gens)
    assert any("expected-goals difference" in line for line in lines)
    spec = json.loads((out / "best.json").read_text())
    assert spec["evaluations"] >= 3 and spec["generation"] == 3
    state = json.loads((out / "population.json").read_text())
    assert state["vocab"] == "v3" and set(state["history"]) <= {json.dumps(p) for p in state["population"]}
    assert state["history"] and all(1 <= len(v) <= 6 for v in state["history"].values())
    # elites survive with their evaluations: resume and keep counting them
    args.generations = 1
    evolve_gp.run(args)
    lines = (out / "progress.log").read_text().splitlines()
    assert any(line.startswith("GP resumed at generation 3") for line in lines)
    assert json.loads((out / "best.json").read_text())["generation"] == 4


def test_selection_ranks_the_reevaluated_first_on_their_mean_and_pools_copies(monkeypatch):
    population = [["roles", ["role", float(k), "ball"]] for k in range(6)]
    population.append(population[0])  # an identical copy pools its evaluations
    screen = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 5.0])  # fitness per program in the screen
    reeval = {0: [0.0, 0.0], 1: [4.0, 4.0], 2: [3.0, 3.0]}  # program -> its two re-evaluations

    def fake(workers, programs, panel, seeds, ticks, words, xg=None):
        reps = np.asarray(seeds).shape[0]
        if reps == 1:
            gd = screen[None, :, None].repeat(len(panel), axis=2)
        else:
            idx = [population.index(p) for p in programs]
            gd = np.array([[reeval[i][r] for i in idx] for r in range(reps)])[..., None].repeat(len(panel), axis=2)
        return gd, None

    monkeypatch.setattr(evolve_gp, "evaluate_seeded", fake)
    args = evolve_gp.parse_args(["--panel", "x", "--fitness", "gd", "--reeval-top", "3", "--reeval-reps", "2",
                                 "--parsimony", "0"])
    panel = ["a", "b"]
    fitness, score, evals, opp, xgs, first = evolve_gp.denoised_generation(
        None, population, panel, np.zeros((2, 1)), np.random.default_rng(0), 300, V3, np.array([0.5, 0.5]), args, {}
    )
    k0 = json.dumps(population[0])
    assert evals[k0] == [5.0, 5.0, 0.0, 0.0] and fitness[0] == fitness[6] == 2.5
    assert list(fitness[:3]) == [2.5, 4.0, 3.0] and list(first) == list(screen)
    order = np.argsort(-score, kind="stable")
    assert list(order[:4]) == [1, 2, 0, 6]  # re-evaluated first (by their mean), then the rest by the screen
    args.reeval_memory = 4
    fitness, *_ = evolve_gp.denoised_generation(
        None, population, panel, np.zeros((2, 1)), np.random.default_rng(0), 300, V3, np.array([0.5, 0.5]), args,
        {k0: [10.0, 10.0]},
    )
    assert fitness[0] == pytest.approx(np.mean([5.0, 5.0, 0.0, 0.0, 10.0, 10.0]))


def test_real_v3_games_smoke_best_json_loads(tmp_path):
    field = tmp_path / "field"
    field.mkdir()
    for n in ("SimpleBrain", "BehindAndTowards"):
        (field / f"{n}.json").write_text(json.dumps({"class": HEURISTIC.format(n)}))
    out = tmp_path / "gp"
    args = evolve_gp.parse_args(
        ["--vocab", "v3", "--workers", "1", "--population", "4", "--generations", "1", "--panel", "all",
         "--panel-dir", str(field), "--ticks-per-opponent", "200", "--ticks", "100", "--fitness", "gd",
         "--reeval-top", "2", "--reeval-reps", "1", "--xg-weight", "0.1", "--out", str(out), "--seed", "2",
         "--max-size", "120"]
    )
    evolve_gp.run(args)
    brain = load_brain(str(out / "best.json"))
    assert isinstance(brain, GPBrainV3)
    assert "def role_1" in (out / "best.txt").read_text()


# --- the xG game ---


def test_xg_game_plays_the_same_game_and_counts_only_touches():
    assert not is_touch([3.0, -2.0], [-3.0, -2.0]) and not is_touch([3.0, -2.0], [3.0, 2.0])
    assert not is_touch([3.0, 2.0], [3.0, 2.0]) and is_touch([3.0, 2.0], [4.0, 1.0])
    program = ["roles", ["role", "ball_dist", ["hit", "open_goal"]], ["role", 0.0, "home"]]
    for blue in (True, False):
        pair = (GPBrainV3("a", program=program), load_brain("SimpleBrain"))
        pair = pair if blue else pair[::-1]
        plain = Game(*pair, game_length=900, quiet_mode=True, seed=4).play()
        pair = (GPBrainV3("a", program=program), load_brain("SimpleBrain"))
        pair = pair if blue else pair[::-1]
        game = XGGame(*pair, game_length=900, quiet_mode=True, seed=4, xg_model=None)
        assert game.play() == plain
        me = "blue" if blue else "red"
        assert game.xg[me] > 0 and game.shots[me] > 0 and min(game.xg.values()) >= 0
