"""Balanced training populations (aisoccer/balance.py)."""

import numpy as np

from aisoccer.balance import Record, tilt


def test_tilt_hits_the_target_when_reachable():
    gd = np.array([3.0, 1.0, 0.2, -0.5, -2.0])
    w = tilt(np.ones(5), gd, target=0.1, floor=0.2)
    core = (w - 0.2 / 5) / 0.8  # the tilted part, without the floor
    assert abs(w.sum() - 1) < 1e-9 and abs(core @ gd - 0.1) < 1e-6
    # The floor keeps every opponent at least floor x its base share.
    assert (w >= 0.2 / 5 - 1e-12).all()
    # Most weight near the target, least on the blowouts.
    assert w[2] > w[0] and w[3] > w[0]


def test_tilt_goes_as_far_as_it_can():
    # Beats everybody: most weight on the hardest opponent.
    w = tilt(np.ones(3), np.array([5.0, 2.0, 0.5]), target=0.1)
    assert w.argmax() == 2 and w[2] > 0.8
    # Loses to everybody: most weight on the easiest.
    w = tilt(np.ones(3), np.array([-5.0, -2.0, -0.5]), target=0.1)
    assert w.argmax() == 2 and w[2] > 0.8


def test_tilt_keeps_base_weights_when_already_balanced():
    base = np.array([0.5, 0.25, 0.25])
    gd = np.array([0.1, 0.1, 0.1])
    assert np.allclose(tilt(base, gd, target=0.1), base)


def test_record_fades_and_starts_at_the_prior():
    r = Record(memory=10, prior=0.1)
    assert r.value("new") == 0.1
    for _ in range(50):
        r.add("easy", 3.0)
    assert 2.5 < r.value("easy") < 3.0
    r.add("easy", -3.0)
    assert r.value("easy") < 2.5  # recent results count more


def test_ppo_balance_draws_pool_opponents_near_level(tmp_path):
    import train_ppo

    b = train_ppo.Balance.from_config({"target": 0.1, "floor": 0.2, "memory": 20}, tmp_path / "balance.json")
    ticks = train_ppo.GAME_LENGTH
    for _ in range(30):
        b.add("weak.json", 10, 0)   # blowouts
        b.add("level.npz", 1, 1)    # even games
        b.add("strong.json", 0, 3)  # heavy losses
    pool = [tmp_path / "weak.json", tmp_path / "level.npz", tmp_path / "strong.json"]
    w = b.weights([p.name for p in pool])
    # Beyond the floor, on average level (+0.1 per 1000 ticks): the blowout gets the least weight.
    values = b.record.values([p.name for p in pool])
    core = (w - 0.2 / 3) / 0.8
    assert abs(core @ values - 0.1) < 1e-6 and w.argmin() == 0 and (w > 0.2 / 3 - 1e-9).all()
    b.save()
    again = train_ppo.Balance.from_config(True, tmp_path / "balance.json")
    assert np.allclose(again.weights([p.name for p in pool]), w)
    assert abs(again.record.value("weak.json") - 10 * 1000 / ticks) < 1.0
    assert again.summary(pool).startswith("weighted ")
    assert train_ppo.Balance.from_config(None) is None


def test_a_single_strong_opponent_does_not_take_everything():
    # League 3, round 103: PPO-v2 lost only to LLM-v11 (-0.17), was about level with GP-v3
    # (+0.20) and far ahead of the old champions. The floor on the blowouts must not push
    # nearly all games onto LLM-v11.
    gd = np.array([-0.17, 0.19, 0.20, 0.40, 0.52, 0.66, 0.81, 1.56, 3.30, 4.13, 5.68])
    w = tilt(np.ones(len(gd)), gd, target=0.1, floor=0.2)
    assert w[0] < 0.5 and w[2] > 0.1  # LLM-v11 most, GP-v3 still plenty
