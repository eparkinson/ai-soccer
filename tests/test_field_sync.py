"""Training opponent folders (league field and learner pools) with training_field "tournament"."""

import json

import pytest

import evolve_gp
import league
import train_ppo
import train_tactician
from aisoccer.brains.PPOBrain import ACT_DIM, OBS_DIM, PPOBrain
from aisoccer.brainspec import load_brain
from aisoccer.llm.v11 import LLMBrainV11
from aisoccer.ppo import PPO
from aisoccer.tactics.v1 import TacticsV1

TACTICS = "aisoccer.tactics.v1.TacticsV1"


def gp(k):
    return {"class": "aisoccer.gp.v1.GPBrain", "kwargs": {"program": ["behind", "ball", "ball", float(10 * k)]}}


def network(path, seed):
    PPOBrain.save_weights(path, PPO(OBS_DIM, ACT_DIM, seed=seed).weights())
    return str(path)


@pytest.fixture
def league3(tmp_path, monkeypatch):
    """A league 3 like state: 8 champions (champ-7 is Tactics, the holder), 7 entrants."""
    history = tmp_path / "champions"
    history.mkdir()
    for k in (0, 1, 3, 4, 5, 6):
        (history / f"champ-{k}.json").write_text(json.dumps(gp(k)))
    network(history / "PPO-champ-2.npz", 2)
    (history / "champ-7.json").write_text(json.dumps({"class": TACTICS}))
    monkeypatch.setattr(league, "HISTORY_DIR", history)
    monkeypatch.setattr(league, "LEARNING_ONLY", True)
    monkeypatch.setattr(league, "PAST_CHAMPIONS", 2)
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    (snaps / "GP-r034.json").write_text(json.dumps(gp(99)))
    entrants = {
        "PPO-a": network(snaps / "PPO-a-r034.npz", 11),
        "PPO-b": network(snaps / "PPO-b-r034.npz", 12),
        "PPO-c": network(snaps / "PPO-c-r034.npz", 13),
        "GP": str(snaps / "GP-r034.json"),
        "LLM-v8": "brain:aisoccer.llm.v8.LLMBrainV8",
        "LLM-v11": "brain:aisoccer.llm.v11.LLMBrainV11",
        "Tactics": "brain:" + TACTICS,
    }
    champion = history / "champ-7.json"
    # As league.main: an identical entrant stands for the champion, then the past
    # champions join, and an odd field rests the oldest of them.
    alias = next((n for n, s in entrants.items() if league.same_brain(s, str(champion))), None)
    field = {**entrants, **league.past_field(entrants)}
    tournament_field = dict(field)
    played, rested = league.even_field(field, entrants)
    return dict(
        tmp=tmp_path, entrants=entrants, champion=champion, alias=alias,
        field=tournament_field, played=played, rested=rested,
    )


FIELD = {
    "PPO-a.npz", "PPO-b.npz", "PPO-c.npz", "GP.json", "LLM-v8.json", "LLM-v11.json",
    "champ-7.json", "champ-6.json", "champ-5.json",
}


def test_field_is_the_tournament(league3):
    assert league3["alias"] == "Tactics"
    assert "champ-5" in league3["rested"] and "champ-5" not in league3["played"]
    files = league.tournament_files(league3["field"], league3["champion"], league3["alias"])
    assert set(files) == FIELD  # the rested past champion is included
    # One file per tournament brain (the holder under its champion name, not twice).
    assert len(files) == len(league3["field"])
    for old in ("champ-0.json", "champ-1.json", "PPO-champ-2.npz", "champ-3.json", "champ-4.json"):
        assert old not in files


def test_fixed_entrants_are_json_specs_that_load(league3):
    field_dir = league3["tmp"] / "field"
    league.sync_dir(field_dir, league.tournament_files(league3["field"], league3["champion"], league3["alias"]))
    assert {p.name for p in field_dir.iterdir()} == FIELD
    assert json.loads((field_dir / "LLM-v11.json").read_text()) == {"class": "aisoccer.llm.v11.LLMBrainV11"}
    assert isinstance(load_brain(field_dir / "LLM-v11.json"), LLMBrainV11)
    assert isinstance(load_brain(field_dir / "champ-7.json"), TacticsV1)
    for path in field_dir.iterdir():
        assert load_brain(path) is not None
        assert train_ppo.make_opponent(f"file:{path}") is not None


def test_learner_pool_leaves_itself_out(league3):
    args = (league3["field"], league3["champion"], league3["alias"])
    assert set(league.tournament_files(*args, exclude={"PPO-a"})) == FIELD - {"PPO-a.npz"}
    # The holder's alias is the holder: its pool has neither.
    assert set(league.tournament_files(*args, exclude={"Tactics"})) == FIELD - {"champ-7.json"}
    pool = [league3["tmp"] / n for n in league.tournament_files(*args, exclude={"PPO-a"})]
    assert train_ppo.current_champion(pool).name == "champ-7.json"


def test_consumers_read_the_field(league3):
    field_dir = league3["tmp"] / "field"
    league.sync_dir(field_dir, league.tournament_files(league3["field"], league3["champion"], league3["alias"]))
    panel = evolve_gp.field_panel(field_dir, {"GP"})
    assert {p.rsplit("/", 1)[-1] for p in panel} == FIELD - {"GP.json"}
    assert {p.name for p in train_tactician.panel(field_dir)} == FIELD
    assert {p.name for p in train_tactician.panel(field_dir, {"LLM-v8"})} == FIELD - {"LLM-v8.json"}


def test_new_champion_under_its_champion_name(league3):
    # LLM-v11 crowned this round (champ-8): it stands under champ-8.json, Tactics as itself.
    crowned = league3["champion"].with_name("champ-8.json")
    crowned.write_text(json.dumps({"class": "aisoccer.llm.v11.LLMBrainV11"}))
    files = league.tournament_files(league3["field"], crowned, "LLM-v11")
    assert set(files) == FIELD - {"LLM-v11.json", "champ-7.json"} | {"champ-8.json", "Tactics.json"}
    assert files["Tactics.json"] == {"class": TACTICS}
    assert train_ppo.current_champion(list(files)) == "champ-8.json"


def test_stale_files_go_one_sync_later(league3):
    field_dir = league3["tmp"] / "field"
    old = {n: league3["champion"] for n in ("champ-0.json", "champ-7.json")}
    league.sync_dir(field_dir, old)
    league.sync_dir(field_dir, {"champ-7.json": league3["champion"]})
    assert (field_dir / "champ-0.json").exists()  # a reader may just have listed it
    league.sync_dir(field_dir, {"champ-7.json": league3["champion"]})
    assert not (field_dir / "champ-0.json").exists()


def test_brain_file_kinds(tmp_path):
    name, spec = league.brain_file("DnA", "DefendersAndAttackers")
    assert name == "DnA.json"
    (tmp_path / name).write_text(json.dumps(spec))
    assert type(load_brain(tmp_path / name)).__name__ == "DefendersAndAttackers"
    assert league.brain_file("x", "explore:runs/random-ppo.npz") is None
    assert league.brain_file("P", "runs/x/P-r001.npz") == ("P.npz", "runs/x/P-r001.npz")


def test_setting_default_is_the_old_behaviour():
    assert league.TRAINING_FIELD == "all"
    assert league.TUNABLE["training_field"] == "TRAINING_FIELD"
    saved = league.TRAINING_FIELD
    try:
        assert league.apply_settings({"training_field": "tournament"}) == ["training_field all -> tournament"]
    finally:
        league.TRAINING_FIELD = saved


def test_same_numbers_as_before(league3):
    # past_field is the inline code it replaced: the most recent PAST_CHAMPIONS, not ones equal to an entrant.
    assert set(league.past_field(league3["entrants"])) == {"champ-5", "champ-6"}
