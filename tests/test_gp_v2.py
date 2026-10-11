import json
import math
import random
from pathlib import Path

import numpy as np
import pytest

import evolve_gp
from aisoccer.brainspec import load_brain
from aisoccer.constants import Constants
from aisoccer.game import Game, flip_pos, flip_vel
from aisoccer.gp import v1, v2
from aisoccer.gp.v2 import (
    CONTACT,
    HORIZON,
    GPBrainV2,
    front_mates,
    intercept_points,
    predict_path,
    reach_times,
    save_point,
    strike_point,
)
from aisoccer.tactics.skills_v1 import predict_ball

V2 = evolve_gp.vocab("v2")
# League 3's GP best at generation 85 (runs/league3/gp/best.json)
LIVE_V1 = ["behind", ["behind", ["if_v", "nearest_to_ball", "ball", "own_goal"], "ball", -202.0], "ball_vel", "role"]


def xy(path):
    """A list of points as (xs, ys)."""
    return tuple(list(c) for c in zip(*path))


def v2_programs(count, seed=0):
    return evolve_gp.ramped(random.Random(seed), count, words=V2)


def random_state(brain, rng, spread=1.0):
    brain.my_players_pos = rng.random((5, 2)) * [1799, 800] * spread
    brain.my_players_vel = rng.normal(0, 3, (5, 2))
    brain.opp_players_pos = rng.random((5, 2)) * [1799, 800] * spread
    brain.opp_players_vel = rng.normal(0, 3, (5, 2))
    brain.ball_pos = rng.random(2) * [1799, 800] * spread
    brain.ball_vel = rng.normal(0, 5, 2)


class Recorder:
    def __init__(self, cls):
        class Recording(cls):
            def do_move(inner):
                move = super().do_move()
                self.moves.append(move.copy())
                return move

        self.moves = []
        self.cls = Recording


# --- v1 programs are v2 programs and play identically ---


def test_v1_programs_load_and_play_identically_in_v2():
    rng = np.random.default_rng(0)
    programs = [LIVE_V1] + evolve_gp.ramped(random.Random(1), 60)
    for program in programs:
        v2.validate(program)
        a, b = v1.GPBrain(program=program), GPBrainV2(program=program)
        assert v2.to_text(program) == v1.to_text(program)
        for spread in (1.0, 0.0, 3.0):
            random_state(a, rng, spread)
            for k in ("my_players_pos", "my_players_vel", "opp_players_pos", "opp_players_vel", "ball_pos", "ball_vel"):
                setattr(b, k, getattr(a, k))
            np.testing.assert_array_equal(a.do_move(), b.do_move())
    rec1, rec2 = Recorder(v1.GPBrain), Recorder(GPBrainV2)
    s1 = Game(rec1.cls("a", program=LIVE_V1), v1.GPBrain("b", program="ball"), game_length=400, quiet_mode=True,
              seed=3).play()
    s2 = Game(rec2.cls("a", program=LIVE_V1), v1.GPBrain("b", program="ball"), game_length=400, quiet_mode=True,
              seed=3).play()
    assert s1 == s2
    np.testing.assert_array_equal(np.array(rec1.moves), np.array(rec2.moves))


def test_v1_best_json_seeds_a_v2_run(tmp_path):
    seed = tmp_path / "v1best.json"
    seed.write_text(json.dumps({"class": "aisoccer.gp.v1.GPBrain", "kwargs": {"program": LIVE_V1}}))
    out = tmp_path / "gp"
    args = evolve_gp.parse_args(
        ["--workers", "1", "--population", "6", "--games", "2", "--generations", "2", "--panel", "SimpleBrain",
         "--ticks", "200", "--out", str(out), "--seed", "1", "--vocab", "v2", "--seed-programs", str(seed)]
    )
    evolve_gp.run(args)
    log = (out / "progress.log").read_text()
    assert "1 programs seeded" in log and "generation    2" in log
    spec = json.loads((out / "best.json").read_text())
    assert spec["class"] == "aisoccer.gp.v2.GPBrainV2"
    assert isinstance(load_brain(str(out / "best.json")), GPBrainV2)
    assert "aisoccer/gp/v2.py" in (out / "best.txt").read_text()
    assert json.loads((out / "population.json").read_text())["vocab"] == "v2"
    # a v2 state is not resumed as v1
    args.vocab, args.generations = "v1", 1
    try:
        evolve_gp.run(args)
        raise AssertionError("resumed v2 programs as v1")
    except SystemExit:
        pass


# --- the language ---


def test_every_v2_word_is_used_and_programs_are_well_typed():
    words = set()
    for p in v2_programs(300, seed=1):
        v2.validate(p)
        words |= set(json.dumps(p).replace("[", " ").replace("]", " ").replace(",", " ").replace('"', " ").split())
    assert set(v2.FUNCTIONS) <= words and set(v2.TERMINALS) <= words


def test_v2_compiled_equals_interpreted():
    rng = np.random.default_rng(0)
    programs = v2_programs(200, seed=2) + [
        ["strike", "opp_goal"], ["ball_in", ["mul", "role", 20]], "intercept", "save_point",
        ["if_v", "ball_in_own_half", ["behind", "save_point", "ball", 10.0], ["strike", "save_point"]],
    ]
    for program in programs:
        brain = GPBrainV2(program=program)
        for spread in (1.0, 0.0, 3.0, "shot"):
            if spread == "shot":  # a ball fired at my goal, so save_point follows the shot
                random_state(brain, rng)
                brain.ball_vel = (np.array([20.0, rng.uniform(150, 650)]) - brain.ball_pos) / rng.uniform(20, 100)
            else:
                random_state(brain, rng, spread)
            compiled = np.array(brain.targets())
            reference = v2.interpret_targets(program, brain)
            assert np.isfinite(compiled).all()
            np.testing.assert_allclose(compiled, reference, rtol=1e-9, atol=1e-6, err_msg=v2.to_text(program))


def test_v2_printing_and_breeding():
    program = ["if_v", "nearest_to_ball", ["strike", "bank_top"], ["ball_in", 30.0]]
    assert v2.to_text(program) == (
        "def target():\n    if nearest_to_ball:\n        return strike(bank_top)\n    return ball_in(30)"
    )
    rng = random.Random(0)
    pop = v2_programs(30, seed=7)
    args = evolve_gp.parse_args(["--panel", "SimpleBrain", "--vocab", "v2"])
    for child in evolve_gp.breed(rng, pop, [rng.random() for _ in pop], args):
        v2.validate(child)
        v2.compile_program(child)


def test_random_v2_programs_play_games():
    for k, program in enumerate(v2_programs(4, seed=3)):
        score = Game(GPBrainV2("a", program=program), GPBrainV2("b", program=["strike", "opp_goal"]),
                     game_length=300, quiet_mode=True, seed=k).play()
        assert set(score) == {"blue", "red"}


# --- the new words ---


def test_predicted_path_matches_skills_v1():
    """
    The same until the first bounce, then within a tick's travel per bounce (see
    predict_xy), up to a goal (skills_v1 lets the ball on into the goal).
    """
    rng = np.random.default_rng(4)
    for _ in range(100):
        pos, vel = rng.random(2) * [1700, 700] + 50, rng.normal(0, 6, 2)
        path = np.array(predict_path(*pos.tolist(), *vel.tolist()))[1:]
        ref = predict_ball(pos, vel, HORIZON)
        assert len(path) == HORIZON and np.allclose(path[0], pos + vel)
        steps = np.diff(np.vstack([pos, ref]), axis=0)
        flips = np.cumsum(np.any(np.sign(steps) != np.sign(vel), axis=1))
        mouth = (ref[:, 0] < v2._X_LO) | (ref[:, 0] > v2._X_HI)
        keep = np.cumsum(mouth) == 0
        outside = (ref[:, 0] < v2._X_LO) | (ref[:, 0] > v2._X_HI) | (ref[:, 1] < v2._Y_LO) | (ref[:, 1] > v2._Y_HI)
        before = np.cumsum(outside) == 0  # no bounce yet
        np.testing.assert_allclose(path[before], ref[before], atol=1e-6)
        tolerance = 2 * np.abs(vel) * np.maximum(flips, 1)[:, None] + 1e-9
        assert (np.abs(path - ref)[keep] <= tolerance[keep]).all()


def test_intercept_of_a_stationary_ball_is_the_ball():
    me = np.array([[100.0, 100.0], [900.0, 420.0], [1700.0, 700.0], [880.0, 400.0], [0.0, 0.0]])
    path = predict_path(900.0, 400.0, 0.0, 0.0)
    assert intercept_points(*xy(path), me, np.zeros((5, 2))) == [(900.0, 400.0)] * 5


def test_intercept_of_a_moving_ball_is_the_first_reachable_point():
    rng = np.random.default_rng(5)
    for _ in range(30):
        me, vel = rng.random((5, 2)) * [1799, 800], rng.normal(0, 3, (5, 2))
        path = predict_path(*(rng.random(2) * [1799, 800]).tolist(), *rng.normal(0, 8, 2).tolist())
        meet = intercept_points(*xy(path), me, vel)
        late = reach_times(me, vel, *xy(path)) - np.arange(len(path))
        for i in range(5):
            k = path.index(meet[i])
            if (late[i] <= 0).any():
                assert late[i, k] <= 0 and (late[i, :k] > 0).all()
            else:
                assert late[i, k] == late[i].min()


def _run(pos, vel, target):
    """One tick of the GP controller on a lone player, as the game applies it."""
    top, acc = Constants.MAX_PLAYER_VELOCITY, Constants.MAX_ACCELERATION
    d = target - pos
    n = np.linalg.norm(d)
    desired = d * top / n if n > top else d
    a = desired - vel
    na = np.linalg.norm(a)
    vel = vel + (a * acc / na if na > acc else a)
    speed = np.linalg.norm(vel)
    return vel * top / speed if speed > top else vel


def _first_touch(pos, vel, ball, ball_vel, target_of, ticks=300):
    """Tick and (player, ball) positions and player velocity when the player first touches the ball."""
    for t in range(ticks):
        vel = _run(pos, vel, target_of(pos, vel, ball, ball_vel))
        rel, rv = pos - ball, vel - ball_vel
        a, b, c = rv @ rv, 2 * rel @ rv, rel @ rel - CONTACT**2
        disc = b * b - 4 * a * c
        if a > 0 and disc >= 0:
            s = (-b - math.sqrt(disc)) / (2 * a)
            if 0 <= s <= 1:
                return t, pos + vel * s, ball + ball_vel * s, vel
        pos, ball = pos + vel, ball + ball_vel
    return None


def test_running_at_intercept_meets_a_moving_ball_about_when_predicted():
    def target_of(pos, vel, ball, ball_vel):
        path = predict_path(*ball.tolist(), *ball_vel.tolist())
        return np.array(intercept_points(*xy(path), pos[None], vel[None])[0])

    cases = [
        ((300, 400), (900, 400), (-5, 0)),
        ((300, 400), (600, 300), (3, 1)),
        ((200, 700), (1500, 100), (-9, 7)),
        ((900, 100), (500, 600), (4, -3)),
    ]
    for me, ball, ball_vel in cases:
        me, ball, ball_vel = (np.array(v, dtype=float) for v in (me, ball, ball_vel))
        path = predict_path(*ball.tolist(), *ball_vel.tolist())
        predicted = path.index(intercept_points(*xy(path), me[None], np.zeros((1, 2)))[0])
        assert reach_times(me[None], np.zeros((1, 2)), *xy(path))[0, predicted] <= predicted  # catchable
        touch = _first_touch(me, np.zeros(2), ball, ball_vel, target_of)
        assert touch is not None and touch[0] <= predicted + 3, (touch, predicted)


def test_strike_point_geometry():
    b, aim = (900.0, 400.0), (1779.0, 400.0)
    # straight behind: run through the contact point (868, 400)
    assert strike_point(*b, 700.0, 400.0, *aim) == (868.0 + 30.0, 400.0)
    # behind at an angle: on the line from me through the contact point, past it
    x, y = strike_point(*b, 800.0, 460.0, *aim)
    cross = (868 - 800) * (y - 460) - (400 - 460) * (x - 800)
    assert abs(cross) < 1e-6 and x > 868
    # behind but far off the line: line up behind the contact point first
    assert strike_point(*b, 850.0, 600.0, *aim) == (808.0, 400.0)
    # in front: go round the ball on my side, to beside-and-behind it
    assert strike_point(*b, 1000.0, 420.0, *aim) == (900.0 - 47.0, 400.0 + 72.0)
    assert strike_point(*b, 1000.0, 380.0, *aim) == (900.0 - 47.0, 400.0 - 72.0)
    # no direction: the ball itself
    assert strike_point(*b, 0.0, 0.0, *b) == b


def test_striking_a_still_ball_sends_it_towards_the_aim_at_speed():
    aim = np.array([1779.0, 400.0])

    def target_of(pos, vel, ball, ball_vel):
        return np.array(strike_point(*ball.tolist(), *pos.tolist(), *aim.tolist()))

    ball = np.array([900.0, 400.0])
    starts = [(700, 400), (780, 480), (850, 650), (1100, 420), (1000, 300), (960, 400), (600, 100), (900, 700)]
    for start in starts:
        touch = _first_touch(np.array(start, dtype=float), np.zeros(2), ball, np.zeros(2), target_of)
        assert touch is not None, start
        _, p, bl, vel = touch
        n = (bl - p) / np.linalg.norm(bl - p)
        d = (aim - bl) / np.linalg.norm(aim - bl)
        assert n @ d > math.cos(math.radians(12)), (start, n)  # the ball leaves along n
        assert vel @ n > 0.6 * Constants.MAX_PLAYER_VELOCITY, (start, vel @ n)  # hit hard, not tapped


def test_striking_a_moving_ball_sends_it_roughly_towards_the_aim():
    aim = np.array([1779.0, 400.0])

    def target_of(pos, vel, ball, ball_vel):
        path = predict_path(*ball.tolist(), *ball_vel.tolist())
        meet = intercept_points(*xy(path), pos[None], vel[None])[0]
        return np.array(strike_point(*meet, *pos.tolist(), *aim.tolist()))

    cases = [((600, 400), (-3, 1)), ((600, 200), (2, 2)), ((1100, 500), (-2, -1)), ((800, 700), (0, -3))]
    for start, ball_vel in cases:
        ball = np.array([900.0, 400.0])
        touch = _first_touch(np.array(start, dtype=float), np.zeros(2), ball, np.array(ball_vel, float), target_of)
        assert touch is not None, start
        _, p, bl, vel = touch
        n = (bl - p) / np.linalg.norm(bl - p)
        kicked = np.array(ball_vel) + 1.8 * ((vel - ball_vel) @ n) * n  # the game's kick (KICK_FACTOR ~1.8)
        d = (aim - bl) / np.linalg.norm(aim - bl)
        assert kicked @ d > math.cos(math.radians(25)) * np.linalg.norm(kicked), (start, ball_vel, kicked)


def test_bank_points_bounce_into_the_goal():
    for bank in (v2.BANK_TOP, v2.BANK_BOTTOM):
        ball = np.array([1300.0, 450.0])
        d = np.array(bank) - ball
        path = np.array(predict_path(*ball.tolist(), *(d / np.linalg.norm(d) * 10).tolist()))
        k = np.argmax(path[:, 0] >= 1700)  # after the bounce off the wall
        (x0, y0), (x1, y1) = path[k - 1], path[k]
        assert np.sign(y1 - y0) != np.sign(d[1])
        y_at_goal = y1 + (v1.OPP_GOAL[0] - x1) * (y1 - y0) / (x1 - x0)
        assert abs(y_at_goal - 400) < 1e-6


def test_ball_in_and_front_mate():
    path = predict_path(100.0, 100.0, 3.0, 0.0)
    assert v2.ball_in(xy(path), 10.7) == (130.0, 100.0)
    assert v2.ball_in(xy(path), -5) == path[0] and v2.ball_in(xy(path), 1e9) == path[HORIZON]
    me = [(100.0, 400.0), (1700.0, 400.0), (1500.0, 400.0), (300.0, 400.0), (900.0, 400.0)]
    assert front_mates(me) == [me[1], me[2], me[1], me[1], me[1]]


# --- save_point ---

KX = v2.KEEPER_X  # 48: the goal line (20) + a player radius (24) + 4


def test_save_point_on_a_straight_shot_is_on_the_keeper_line_in_the_ball_s_path():
    assert KX == Constants.GOAL_DEPTH + Constants.PLAYER_RADIUS + 4
    assert save_point(600.0, 300.0, -10.0, 0.0) == (KX, 300.0)
    assert save_point(600.0, 470.0, -5.0, 0.0) == (KX, 470.0)  # arrives within HORIZON ticks


def test_save_point_on_an_angled_shot_is_where_it_crosses_the_keeper_line_not_the_goal_line():
    ball, into = np.array([300.0, 740.0]), np.array([20.0, 250.0])  # into the goal near the top post
    vel = (into - ball) / 40
    x, y = save_point(*ball.tolist(), *vel.tolist())
    slope = vel[1] / vel[0]
    assert x == KX and y == pytest.approx(ball[1] + slope * (KX - ball[0]))
    assert y - 250.0 > 45  # the goal-line crossing would leave the keeper well out of the ball's way


def test_save_point_follows_a_shot_bounced_off_a_side_wall():
    ball, vel = (600.0, 700.0), (-10.0, 6.0)
    x, y = save_point(*ball, *vel)
    # the predicted path: bottom wall (y = 792) at t = 92 / 6, then up; x = 48 at t = 55.2
    assert x == KX and y == pytest.approx(792 - 6 * (55.2 - 92 / 6))
    path = np.array(predict_path(*ball, *vel))
    k = np.argmax(path[:, 0] < KX)  # the same crossing, read off predict_path
    (x0, y0), (x1, y1) = path[k - 1], path[k]
    assert y == pytest.approx(y0 + (y1 - y0) * (KX - x0) / (x1 - x0))
    assert abs(y - 400) < 200 and abs(save_point(600.0, 700.0, -10.0, 0.0)[1] - y) > 100


def test_save_point_cuts_the_angle_when_no_shot_is_coming():
    angle = 400 + 200 * (KX - 20) / (600 - 20)  # on the line from (20, 400) to the ball
    for vel in [(5.0, 1.0), (0.0, 0.0), (0.0, -8.0), (-0.1, 0.0)]:  # away, still, sideways, too slow to arrive
        assert save_point(600.0, 600.0, *vel) == (KX, pytest.approx(angle))
    # heading our way but missing the goal by a lot (it would hit the goal line at y = 40)
    assert save_point(600.0, 300.0, -10.0, -260 / 58) == (KX, pytest.approx(400 - 100 * (KX - 20) / 580))


def test_save_point_stays_inside_the_posts():
    lo, hi = v2.KEEPER_Y_MIN, v2.KEEPER_Y_MAX
    assert lo == Constants.GOAL_Y_MIN + Constants.PLAYER_RADIUS and hi == Constants.GOAL_Y_MAX - Constants.PLAYER_RADIUS
    # the keeper still blocks a ball that would squeeze in past the post
    assert lo - Constants.PLAYER_RADIUS - Constants.BALL_RADIUS < Constants.GOAL_Y_MIN
    assert save_point(60.0, 790.0, 0.0, 0.0) == (KX, hi)  # angle from a tight spot by the byline
    assert save_point(30.0, 100.0, 0.0, 0.0) == (KX, lo)  # ball behind the keeper line
    assert save_point(900.0, 380.0, -10.0, -170 / 85.2) == (KX, lo)  # a shot just inside the top post
    assert save_point(900.0, 420.0, -10.0, 170 / 85.2) == (KX, hi)  # and the bottom one
    rng = np.random.default_rng(6)
    for _ in range(2000):
        x, y = save_point(*(rng.random(2) * [1799, 800]).tolist(), *rng.normal(0, 8, 2).tolist())
        assert x == KX and lo <= y <= hi


def test_save_point_is_the_same_for_every_player_and_in_the_brain_s_own_frame():
    brain = GPBrainV2(program="save_point")
    random_state(brain, np.random.default_rng(7))
    brain.ball_pos, brain.ball_vel = np.array([700.0, 350.0]), np.array([-10.0, 1.0])
    assert brain.targets() == [save_point(700.0, 350.0, -10.0, 1.0)] * 5
    # red: the same shot at red's goal, as the game shows it to red (mirrored), gives the same point
    real_pos, real_vel = np.array([1799 - 700.0, 350.0]), np.array([10.0, 1.0])
    brain.ball_pos, brain.ball_vel = flip_pos(real_pos), flip_vel(real_vel)
    assert brain.targets() == [save_point(700.0, 350.0, -10.0, 1.0)] * 5


def test_save_point_keepers_guard_their_own_goal_for_both_colours():
    for keeper_side in (0, 1):
        keeper, striker = GPBrainV2("keep", program="save_point"), GPBrainV2("shoot", program=["strike", "opp_goal"])
        game = Game(*((keeper, striker) if keeper_side == 0 else (striker, keeper)), game_length=250,
                    quiet_mode=True, seed=5)
        game.play()
        xs = game.teams[keeper_side].position_matrix()[:, 0]
        goal_x = KX if keeper_side == 0 else Constants.FIELD_LENGTH - 1 - KX
        assert np.abs(xs - goal_x).mean() < 60, (keeper_side, xs)


def _live_v2_programs():
    runs = Path(__file__).resolve().parent.parent / "runs"
    programs = []
    for f in sorted(runs.glob("*/gp_v2/*.json")):
        try:
            d = json.loads(f.read_text())
        except ValueError:  # being rewritten by the live run
            continue
        programs += d.get("population", []) + ([d["kwargs"]["program"]] if "kwargs" in d else [])
    return programs


def test_live_v2_programs_still_load_and_compile_the_same_as_they_interpret():
    """The live GP-v2 programs (runs/*/gp_v2, read only; nothing to check without them), whatever words they use."""
    programs = _live_v2_programs()
    if not programs:
        pytest.skip("no runs/*/gp_v2 programs here")
    rng = np.random.default_rng(8)
    for program in programs:
        brain = GPBrainV2(program=program)
        for spread in (1.0, 0.0, 3.0):
            random_state(brain, rng, spread)
            np.testing.assert_allclose(np.array(brain.targets()), v2.interpret_targets(program, brain),
                                       rtol=1e-9, atol=1e-6)
