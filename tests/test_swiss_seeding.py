"""Swiss seeding in league.py: the default "random" pairing and the rating-seeded Dutch system."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

import league
from aisoccer.tournament import Tournament


def old_swiss_pairings(tournament, played, had_bye, rng):
    """league.swiss_pairings as it was before seeding was added (the reference)."""
    table = sorted(tournament.tournament_scores.table.values(), key=lambda r: (-r["points"], rng.random()))
    order = [row["number"] for row in table]
    bye = None
    if len(order) % 2:
        bye = next((n for n in reversed(order) if n not in had_bye), order[-1])
        order.remove(bye)
        had_bye.add(bye)
    pairings = []
    while order:
        a = order.pop(0)
        b = next((n for n in order if tuple(sorted((a, n))) not in played), order[0])
        order.remove(b)
        pairings.append(tuple(sorted((a, b))))
    return pairings, bye


def make_tournament(n, points=None):
    tournament = Tournament([SimpleNamespace(name=f"b{k}") for k in range(n)], legs=4, rounds=4, seed=0)
    for k, p in enumerate(points or [0] * n):
        tournament.tournament_scores.table[k]["points"] = p
    return tournament


@pytest.fixture
def seeding(monkeypatch):
    def set_mode(mode):
        monkeypatch.setattr(league, "SWISS_SEEDING", mode)

    return set_mode


@pytest.fixture
def fake_games(monkeypatch):
    """play_all without playing: goals drawn from each game's seed, stronger brains (lower b-number) scoring more."""

    def play_all(workers, tasks):
        out = []
        for entrant, opponent, _, seed in tasks:
            rng = np.random.default_rng(seed)
            strength = {entrant: 3.0, opponent: 3.0}
            strength[entrant] += (20 - int(entrant[1:])) * 0.1
            strength[opponent] += (20 - int(opponent[1:])) * 0.1
            out.append((int(rng.poisson(strength[entrant])), int(rng.poisson(strength[opponent]))))
        return out

    monkeypatch.setattr(league, "play_all", play_all)


# ---------------------------------------------------------------- "random": unchanged


def test_random_mode_matches_old_pairings():
    for trial in range(300):
        setup = np.random.default_rng(trial)
        n = int(setup.integers(3, 16))
        tournament = make_tournament(n, [int(p) for p in setup.integers(0, 13, size=n)])
        pairs = [tuple(sorted(p)) for p in setup.choice(n, size=(int(setup.integers(0, n)), 2)) if p[0] != p[1]]
        byes = {int(b) for b in setup.choice(n, size=int(setup.integers(0, 3)))}
        new_bye, old_bye = set(byes), set(byes)
        new_rng, old_rng = np.random.default_rng(trial), np.random.default_rng(trial)
        assert league.swiss_pairings(tournament, set(pairs), new_bye, new_rng) == old_swiss_pairings(
            tournament, set(pairs), old_bye, old_rng
        )
        assert new_bye == old_bye
        assert new_rng.random() == old_rng.random()  # the same random draws were used


def test_random_mode_swiss_ignores_seeds(seeding, fake_games):
    seeding("random")
    brains = {f"b{k}": f"b{k}" for k in range(12)}
    seeds = {f"b{k}": float(k) for k in range(12)}
    for seed in range(5):
        with_seeds = league.swiss(None, brains, 4, 4, seed, seeds)
        without = league.swiss(None, brains, 4, 4, seed)
        assert with_seeds.results == without.results


def test_unknown_mode_pairs_at_random(seeding, fake_games):
    seeding("ratings")  # a typo: behaves as "random"
    brains = {f"b{k}": f"b{k}" for k in range(10)}
    seeding_typo = league.swiss(None, brains, 4, 4, 3, {n: 0.0 for n in brains}).results
    seeding("random")
    assert seeding_typo == league.swiss(None, brains, 4, 4, 3).results


def test_setting_is_tunable():
    before = league.SWISS_SEEDING
    try:
        assert league.apply_settings({"swiss_seeding": "rating"}) == ["swiss_seeding random -> rating"]
        assert league.tunables()["swiss_seeding"] == "rating"
    finally:
        league.SWISS_SEEDING = before


# ---------------------------------------------------------------- "rating": Dutch system


@pytest.mark.parametrize("n", [10, 12, 14])
def test_first_wave_pairs_top_half_with_bottom_half(n):
    tournament = make_tournament(n)
    seed_rank = {k: int(r) for k, r in enumerate(np.random.default_rng(n).permutation(n))}  # not in number order
    by_seed = {r: k for k, r in seed_rank.items()}
    pairs, bye = league.swiss_pairings_seeded(tournament, set(), set(), seed_rank)
    assert bye is None
    assert pairs == [tuple(sorted((by_seed[i], by_seed[i + n // 2]))) for i in range(n // 2)]


def test_first_wave_odd_field_bye_to_lowest_seed():
    tournament = make_tournament(11)
    had_bye = set()
    pairs, bye = league.swiss_pairings_seeded(tournament, set(), had_bye, {k: k for k in range(11)})
    assert bye == 10 and had_bye == {10}
    assert pairs == [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]


def test_later_wave_pairs_within_score_groups():
    # seeds 0..11; groups: {0,1,2,3} on 9, {4,5} on 6, {6,7,8,9} on 3, {10,11} on 0
    points = [9, 9, 9, 9, 6, 6, 3, 3, 3, 3, 0, 0]
    pairs, _ = league.swiss_pairings_seeded(make_tournament(12, points), set(), set(), {k: k for k in range(12)})
    assert pairs == [(0, 2), (1, 3), (4, 5), (6, 8), (7, 9), (10, 11)]


def test_ties_broken_by_seed_not_number():
    points = [3, 3, 3, 3, 0, 0]
    seed_rank = {0: 3, 1: 2, 2: 1, 3: 0, 4: 5, 5: 4}  # among the leaders, 3 is the top seed
    pairs, _ = league.swiss_pairings_seeded(make_tournament(6, points), set(), set(), seed_rank)
    assert pairs == [(1, 3), (0, 2), (4, 5)]


def test_odd_score_group_floats_its_lowest_down():
    # {0,1,2} on 9: 0 v 1, 2 floats down to meet the top of {3,4,5} on 6; 4 v 5.
    points = [9, 9, 9, 6, 6, 6]
    pairs, _ = league.swiss_pairings_seeded(make_tournament(6, points), set(), set(), {k: k for k in range(6)})
    assert pairs == [(0, 1), (2, 3), (4, 5)]


def test_repeat_avoided_by_next_bottom_half_brain():
    points = [9, 9, 9, 9, 0, 0]
    played = {(0, 2)}  # 0 already met 2: Dutch transposition, 0 v 3 and 1 v 2
    pairs, _ = league.swiss_pairings_seeded(make_tournament(6, points), played, set(), {k: k for k in range(6)})
    assert pairs == [(0, 3), (1, 2), (4, 5)]


def test_backtracks_when_greedy_choice_strands_a_pair():
    # 0 v 1 would leave 2 and 3, who have met: the pairing backtracks to 0 v 2, 1 v 3.
    played = {(2, 3)}
    points = [6, 6, 3, 3]
    pairs, _ = league.swiss_pairings_seeded(make_tournament(4, points), played, set(), {k: k for k in range(4)})
    assert pairs == [(0, 2), (1, 3)]


def test_repeats_allowed_only_when_unavoidable():
    played = {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
    pairs, _ = league.swiss_pairings_seeded(make_tournament(4), played, set(), {k: k for k in range(4)})
    assert pairs == [(0, 2), (1, 3)]  # pure Dutch, as if nothing had been played


@pytest.mark.parametrize("n", [9, 11, 12, 13, 14, 15])
def test_full_seeded_swiss_has_no_repeats_and_rotates_byes(seeding, fake_games, n):
    seeding("rating")
    for seed in range(5):
        names = [f"b{k}" for k in np.random.default_rng(seed).permutation(n)]
        seeds = {name: -float(name[1:]) for name in names}  # b0 the top seed
        tournament = league.swiss(None, {name: name for name in names}, 4, 4, seed, seeds)
        meetings = {}
        games = {k: 0 for k in range(n)}
        for blue, red, _, _ in tournament.results:
            meetings[tuple(sorted((blue, red)))] = meetings.get(tuple(sorted((blue, red))), 0) + 1
            games[blue] += 1
            games[red] += 1
        assert set(meetings.values()) == {4}  # each pairing played once (4 legs), never repeated
        if n % 2:
            assert sorted(games.values()).count(12) == 4  # four different byes
        else:
            assert set(games.values()) == {16}
        # the first wave was seeded: b0 (seed 1) met seed n//2 + 1 (odd n: the last seed had the bye)
        blue, red = tournament.results[0][:2]
        assert {tournament.brains[blue].name, tournament.brains[red].name} == {"b0", f"b{n // 2}"}

# ---------------------------------------------------------------- seeds


def test_seeds_title_race_then_saved_rounds_then_median(tmp_path, monkeypatch):
    monkeypatch.setattr(league, "LEAGUE_DIR", tmp_path)
    (tmp_path / "rounds").mkdir()
    ticks = league.MATCH_TICKS
    (tmp_path / "rounds" / "round-008.json").write_text(
        json.dumps({"match_ticks": ticks, "title_race": {"champ-1": {"gd": 7.0}, "Old": {"gd": 1.0}}})
    )
    (tmp_path / "rounds" / "round-009.json").write_text(
        json.dumps({"match_ticks": ticks // 4, "title_race": {"Old": {"gd": 99.0}}})  # other units: ignored
    )
    state = {
        "title_race": {"A": {"gd": 10.0}, "B": {"gd": -4.0}, "C": {"gd": 2.0}, "Tactics": {"gd": 20.0}},
        "champion_names": {"champ-9": "Tactics"},
    }
    names = ["New", "A", "champ-9", "B", "champ-1", "C", "Old"]
    ratings, where = league.swiss_seeds(names, state, 10)
    assert ratings["A"] == 10.0 and where["A"] == "title race"
    assert ratings["champ-9"] == 20.0 and where["champ-9"] == "title race, as Tactics"
    assert ratings["champ-1"] == 7.0 and where["champ-1"] == "round 8"
    assert ratings["Old"] == 1.0
    assert ratings["New"] == float(np.median([10.0, 20.0, -4.0, 7.0, 2.0, 1.0]))
    assert where["New"] == "unrated: field median"
    assert league.seed_order(names, ratings)[:2] == ["champ-9", "A"]


def test_seeds_with_no_ratings_keep_the_given_order(tmp_path, monkeypatch):
    monkeypatch.setattr(league, "LEAGUE_DIR", tmp_path)
    names = ["x", "y", "z"]
    ratings, _ = league.swiss_seeds(names, {}, 1)
    assert league.seed_order(names, ratings) == names
