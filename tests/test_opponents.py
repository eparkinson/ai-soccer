"""Opponent choice for PPO training games: the default mix, the champion share and the ladder."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import train_ppo
from train_ppo import Ladder, choose_opponent, current_champion

POOL = [
    Path("pool/Coach.json"),
    Path("pool/GP.json"),
    Path("pool/PPO-champ-2.npz"),
    Path("pool/PPO-shaped.npz"),
    Path("pool/champ-0.json"),
    Path("pool/champ-6.json"),
]
SNAPSHOTS = ["snap-a", "snap-b", "snap-c"]


def settings(**overrides):
    base = dict(self_play=0.3, snapshots=0.2, pool_share=0.3, champion_share=0.0, ladder=None)
    return SimpleNamespace(**{**base, **overrides})


def old_draw(rng, args, pool, league, opponent_names, opponent_p):
    """train_ppo.py's opponent draw before champion_share and the ladder, verbatim."""
    tasks = []
    for _ in range(40):
        roll = rng.random()
        seed = int(rng.integers(2**31))
        if roll < args.self_play:
            tasks.append(("self", None, seed))
        elif roll < args.self_play + args.snapshots:
            snap = league[rng.integers(len(league))]
            tasks.append(("snapshot", snap, seed))
        elif pool and roll < args.self_play + args.snapshots + args.pool_share:
            tasks.append(("file:" + str(pool[rng.integers(len(pool))]), None, seed))
        elif not opponent_names:
            if pool:
                tasks.append(("file:" + str(pool[rng.integers(len(pool))]), None, seed))
            else:
                tasks.append(("self", None, seed))
        else:
            tasks.append((str(rng.choice(opponent_names, p=opponent_p)), None, seed))
    return tasks


def new_draw(rng, args, pool, league, opponent_names, opponent_p, ladder=None, champion=None):
    tasks = []
    for _ in range(40):
        roll = rng.random()
        seed = int(rng.integers(2**31))
        opp, snap = choose_opponent(roll, rng, args, pool, league, opponent_names, opponent_p, ladder, champion)
        tasks.append((opp, snap, seed))
    return tasks


@pytest.mark.parametrize(
    "args, pool, fixed",
    [
        (settings(), POOL, []),  # a league learner (no fixed opponents)
        (settings(self_play=0.35, snapshots=0.25, pool_share=0.4), POOL, []),
        (settings(), [], []),  # empty pool: self-play instead
        (settings(self_play=0.15, snapshots=0.1, pool_share=0.2), POOL, ["SimpleBrain", "RandomWalk"]),
        (settings(self_play=0.15, snapshots=0.1, pool_share=0.2), [], ["SimpleBrain", "RandomWalk"]),
    ],
)
def test_default_settings_draw_exactly_as_before(args, pool, fixed):
    p = np.array([2.0, 1.0][: len(fixed)])
    p /= max(p.sum(), 1e-9)
    old = old_draw(np.random.default_rng(5), args, pool, SNAPSHOTS, fixed, p)
    # train_ppo.py does not look the champion up with champion_share 0; it would not matter.
    new = new_draw(np.random.default_rng(5), args, pool, SNAPSHOTS, fixed, p, champion=current_champion(pool))
    assert new == old


def test_current_champion_is_the_highest_numbered():
    assert current_champion(POOL) == Path("pool/champ-6.json")
    assert current_champion([Path("x/PPO-champ-12.npz"), Path("x/champ-9.json")]) == Path("x/PPO-champ-12.npz")
    assert current_champion([Path("x/GP.json"), Path("x/spar-champ-x.json")]) is None
    assert current_champion([]) is None


def test_champion_share_plays_the_champion():
    args = settings(champion_share=0.7, self_play=0.1, snapshots=0.1, pool_share=0.1)
    champion = current_champion(POOL)
    games = new_draw(np.random.default_rng(1), args, POOL, SNAPSHOTS, [], None, champion=champion)
    games += new_draw(np.random.default_rng(2), args, POOL, SNAPSHOTS, [], None, champion=champion)
    share = np.mean([opp == f"file:{champion}" for opp, _, _ in games])
    assert 0.55 < share < 0.85
    kinds = {opp if opp in ("self", "snapshot") else "file" for opp, _, _ in games}
    assert kinds == {"self", "snapshot", "file"}  # the rest of the mix still plays

    only = settings(champion_share=1.0)
    games = new_draw(np.random.default_rng(3), only, POOL, SNAPSHOTS, [], None, champion=champion)
    assert all(opp == f"file:{champion}" for opp, _, _ in games)


def test_without_a_champion_in_the_pool_the_normal_mix_plays():
    args = settings(champion_share=1.0)
    games = new_draw(np.random.default_rng(4), args, POOL[:4], SNAPSHOTS, [], None, champion=None)
    assert {opp for opp, _, _ in games} >= {"self", "snapshot"}


def test_ladder_prefers_opponents_beaten_about_half_the_time():
    ladder = Ladder(memory=40, floor=0.1)
    for _ in range(30):
        ladder.add("champ-0.json", 3, 0)  # always beaten
        ladder.add("GP.json", 0, 4)  # always loses
    for k in range(30):
        ladder.add("champ-6.json", *((1, 0) if k % 2 else (0, 1)))  # half and half
    assert ladder.expected("new.npz") == 0.5  # not played yet
    assert ladder.expected("champ-0.json") > 0.9
    assert ladder.expected("GP.json") < 0.1
    assert abs(ladder.expected("champ-6.json") - 0.5) < 0.05
    w = ladder.weights(["champ-0.json", "GP.json", "champ-6.json", "new.npz"])
    assert w[0] < 0.3 and w[1] < 0.3 and w[2] > 0.95 and w[3] == 1.0
    assert np.all(w >= 0.1)  # everyone is still played now and then

    pool = [Path("p/champ-0.json"), Path("p/GP.json"), Path("p/champ-6.json")]
    rng = np.random.default_rng(0)
    picks = [ladder.choose(pool, rng).name for _ in range(2000)]
    counts = {n: picks.count(n) for n in ("champ-0.json", "GP.json", "champ-6.json")}
    assert counts["champ-6.json"] > 2 * max(counts["champ-0.json"], counts["GP.json"])
    assert min(counts.values()) > 50


def test_ladder_draws_count_as_half():
    ladder = Ladder()
    for _ in range(20):
        ladder.add("Coach.json", 0, 0)
    assert ladder.expected("Coach.json") == pytest.approx(0.5)


def test_ladder_forgets_old_results():
    ladder = Ladder(memory=10)
    for _ in range(50):
        ladder.add("GP.json", 0, 3)
    assert ladder.expected("GP.json") < 0.1
    for _ in range(30):
        ladder.add("GP.json", 2, 1)
    assert ladder.expected("GP.json") > 0.85  # the recent wins count, not the old losses


def test_ladder_draws_pool_games_in_choose_opponent():
    ladder = Ladder(floor=0.01)
    for _ in range(40):
        for name in ("Coach.json", "GP.json", "PPO-champ-2.npz", "PPO-shaped.npz", "champ-0.json"):
            ladder.add(name, 0, 5)
    args = settings(self_play=0.0, snapshots=0.0, pool_share=1.0, ladder=True)
    games = new_draw(np.random.default_rng(6), args, POOL, SNAPSHOTS, [], None, ladder=ladder)
    share = np.mean([opp == "file:pool/champ-6.json" for opp, _, _ in games])
    assert share > 0.5  # the one opponent not yet known to be out of reach (the rest weigh ~0.14 each)


def test_ladder_is_kept_across_restarts(tmp_path):
    ladder = Ladder(tmp_path / "ladder.json", memory=20)
    ladder.add("GP.json", 1, 0)
    ladder.add("GP.json", 0, 0)
    ladder.save()
    again = Ladder(tmp_path / "ladder.json", memory=20)
    assert again.expected("GP.json") == ladder.expected("GP.json")
    assert "GP" in again.summary([tmp_path / "GP.json", tmp_path / "Coach.json"])


def test_ladder_config():
    assert Ladder.from_config(None) is None
    assert Ladder.from_config(False) is None
    assert (Ladder.from_config(True).memory, Ladder.from_config(True).floor) == (40, 0.1)
    assert Ladder.from_config({"memory": 20}).memory == 20
    with pytest.raises(ValueError):
        Ladder.from_config({"memroy": 20})


def test_config_accepts_the_new_keys():
    args = SimpleNamespace()
    train_ppo.apply_config(args, {"champion_share": 0.7, "ladder": {"floor": 0.2}})
    assert args.champion_share == 0.7 and args.ladder == {"floor": 0.2}
