"""Cost-gap core allocation in league.py: lagging strategy families get more workers."""

import json
import sys

import pytest

import league


def test_split_budget_proportional_to_gap_after_one_per_brain():
    # 10 workers, one each first, the 8 spare split 3:1 by gap.
    assert league.split_budget(10, {"A": 30.0, "B": 10.0}, {"A": 1, "B": 1}) == {"A": 7, "B": 3}
    # Largest remainder keeps the total exact.
    out = league.split_budget(24, {"GP": 20.6, "PPO": 19.7}, {"GP": 2, "PPO": 5})
    assert sum(out.values()) == 24
    assert out == {"GP": 11, "PPO": 13}


def test_split_budget_minimum_one_per_brain():
    # A family with no gap (the most expensive) still gets one worker per brain.
    assert league.split_budget(10, {"A": 0.0, "B": 5.0}, {"A": 3, "B": 1}) == {"A": 3, "B": 7}
    # A budget below one per brain: everyone still gets one.
    assert league.split_budget(2, {"A": 1.0, "B": 1.0}, {"A": 2, "B": 2}) == {"A": 2, "B": 2}
    # Nobody lags: spare workers in proportion to family size.
    assert league.split_budget(12, {"A": 0.0, "B": 0.0}, {"A": 1, "B": 3}) == {"A": 3, "B": 9}
    # A family without running brains gets nothing.
    assert league.split_budget(6, {"A": 5.0, "B": 5.0}, {"A": 0, "B": 2}) == {"B": 6}


def test_split_family_equal_spare_to_cheapest():
    own = {"a": 3.0, "b": 1.0, "c": 2.0}
    assert league.split_family(7, ["a", "b", "c"], own) == {"a": 2, "b": 3, "c": 2}
    assert league.split_family(8, ["a", "b", "c"], own) == {"a": 2, "b": 3, "c": 3}
    assert league.split_family(9, ["a", "b", "c"], own) == {"a": 3, "b": 3, "c": 3}


def test_settle_hysteresis():
    assert league.settle(5, 6) == 5  # 1 worker, 20%: kept
    assert league.settle(5, 4) == 5
    assert league.settle(5, 7) == 7  # 2 workers
    assert league.settle(3, 4) == 4  # 33%
    assert league.settle(8, 9) == 8
    assert league.settle(8, 9, min_change=1) == 9
    assert league.settle(4, 4) == 4
    assert league.settle(None, 4) == 4  # count unknown: take the target


def test_strategy_costs_family_totals_and_lineage():
    families = {"GP": "GP", "GP-v2": "GP", "PPO-goals": "PPO", "PPO-roles": "PPO"}
    own = {"GP": 1.5, "GP-v2": 0.1, "PPO-goals": 1.2, "PPO-roles": 0.7, "LLM-v8": 1.9, "LLM-v11": 17.6, "Coach": 0.6}
    lineage = {**own, "LLM-v8": 4.7, "LLM-v11": 22.3}
    family, strategies = league.strategy_costs(families, own, lineage)
    assert family == pytest.approx({"GP": 1.6, "PPO": 1.9})  # PPO-roles (retired) counts for its family
    assert strategies["LLM-v11"] == pytest.approx(22.3)  # the LLM line through its lineage
    assert "PPO-goals" not in strategies and "Coach" in strategies
    # A family is the sum of its members' own costs, and at least any member's lineage
    # (earlier versions not listed in the families).
    family, _ = league.strategy_costs({"LLM-v11": "LLM", "Tactics": "LLM"}, own, {**lineage, "Tactics": 2.0})
    assert family["LLM"] == pytest.approx(22.3)
    family, _ = league.strategy_costs(
        {"LLM-v8": "LLM", "LLM-v11": "LLM", "Tactics": "LLM"}, {**own, "Tactics": 2.0}, {**lineage, "Tactics": 2.0}
    )
    assert family["LLM"] == pytest.approx(22.3)  # 1.9 + 17.6 + 2.0 = 21.5 < v11's line


def test_plan_cores_current_numbers():
    families = {
        "GP": "GP", "GP-v2": "GP", "PPO-goals": "PPO", "PPO-shaped": "PPO", "PPO-exploiter": "PPO",
        "PPO-chance": "PPO", "PPO-tactician": "PPO",
    }
    own = {"GP": 1.49, "GP-v2": 0.13, "PPO-goals": 1.163, "PPO-shaped": 1.175, "PPO-exploiter": 0.116,
           "PPO-chance": 0.047, "PPO-tactician": 0.127, "LLM-v11": 17.615, "Coach": 0.596}
    lineage = {**own, "LLM-v11": 22.296}
    targets, plan = league.plan_cores(24, families, own, lineage, list(families))
    assert plan["reference"] == "LLM-v11"
    assert sum(targets.values()) == 24
    assert min(targets.values()) >= 1
    gp = plan["families"]["GP"]
    assert gp["gap"] == pytest.approx(22.296 - 1.62)
    assert targets["GP"] + targets["GP-v2"] == gp["workers"]
    # Equal within a family (spare one to the cheapest).
    ppo = [targets[n] for n in families if families[n] == "PPO"]
    assert max(ppo) - min(ppo) <= 1
    # Per brain, the smaller family gets more.
    assert targets["GP"] > max(ppo)


# ------------------------------------------------------------- the coordinator's Roster


class FakeProcess:
    def __init__(self, cmd, log_path, threads="2"):
        self.cmd = [str(c) for c in cmd]
        STARTED.append(self)

    def signal(self, sig):
        pass

    def exited(self):
        return False


class FakeLearner(FakeProcess):
    def __init__(self, name, spec, champion, args, saved=None):
        self.name, self.workers = name, args.workers_per_learner
        self.vector = -(-args.learner_games // args.workers_per_learner)
        STARTED.append(self)

    def state(self):
        return {}


STARTED: list = []

ROSTER = {
    "learners": {"PPO-goals": {"about": "x", "config": {}}, "PPO-shaped": {"about": "y", "config": {}}},
    "processes": {
        "GP": {"cmd": ["evolve_gp.py", "--workers", "3", "--games", "6"], "entrant": "gp.json"},
        "Coach": {"cmd": ["coach.py", "--workers", "3"], "entrant": "coach.json"},
    },
    "fixed_entrants": {"LLM-v11": "brain:x"},
    "settings": {"workers_per_learner": 3, "learner_games": 72},
}
FAMILIES = {"GP": "GP", "PPO-goals": "PPO", "PPO-shaped": "PPO"}
OWN = {"GP": 1.0, "PPO-goals": 1.0, "PPO-shaped": 1.0, "Coach": 0.5, "LLM-v11": 17.6}
LINEAGE = {**OWN, "LLM-v11": 22.3}


@pytest.fixture
def roster(tmp_path, monkeypatch):
    monkeypatch.setattr(league, "Process", FakeProcess)
    monkeypatch.setattr(league, "Learner", FakeLearner)
    monkeypatch.setattr(league, "LEAGUE_DIR", tmp_path)
    monkeypatch.setattr(league, "brain_usd", lambda cpu: (dict(OWN), dict(LINEAGE)))
    STARTED.clear()

    def make(data):
        path = tmp_path / "roster.json"
        path.write_text(json.dumps(data))
        return league.Roster(path, None)

    return make


def started():
    out = {p.name if isinstance(p, FakeLearner) else p.cmd[1]: p for p in STARTED}
    STARTED.clear()
    return out


def test_off_is_unchanged(roster):
    r = roster(ROSTER)
    r.reconcile({})
    first = started()
    assert first["evolve_gp.py"].cmd == [sys.executable, "evolve_gp.py", "--workers", "3", "--games", "6"]
    assert first["coach.py"].cmd == [sys.executable, "coach.py", "--workers", "3"]
    assert first["PPO-goals"].workers == 3 and first["PPO-goals"].vector == 24
    # The same keys as before the core allocation existed: nothing restarts.
    assert r.started["GP"] == json.dumps(ROSTER["processes"]["GP"], sort_keys=True)
    assert r.allocate_cores({}) == []
    assert r.workers == {}
    assert r.reconcile({}) == [] and not STARTED
    # Families alone (mode off) change nothing either.
    r.data = {**ROSTER, "families": FAMILIES, "settings": {**ROSTER["settings"], "training_workers": 12}}
    assert r.allocate_cores({}) == [] and r.reconcile({}) == []


def test_cost_gap_restarts_only_changed_brains(roster):
    data = {
        **ROSTER,
        "families": FAMILIES,
        "settings": {**ROSTER["settings"], "core_allocation": "cost_gap", "training_workers": 12},
    }
    r = roster(data)
    r.reconcile({})
    started()
    notes = r.allocate_cores({"cpu": {}})
    # GP gap 21.3, PPO gap 21.3: spare 9 split 4.5/4.5 -> GP 1+5 (tie to the larger remainder), PPO 2+4.
    assert notes[0].startswith("CORES (cost gap, 12 training workers; most expensive strategy LLM-v11 $22.30)")
    assert "GP family $1.00 (gap $21.30)" in notes[0]
    assert r.workers["GP"] + r.workers.get("PPO-goals", 3) + r.workers.get("PPO-shaped", 3) == 12
    assert "Coach" not in r.workers  # outside the families: configured workers
    changed = r.reconcile({})
    now = started()
    assert "coach.py" not in now
    assert now["evolve_gp.py"].cmd[:4] == [sys.executable, "evolve_gp.py", "--workers", str(r.workers["GP"])]
    assert all("core allocation" in n for n in changed)
    # Same costs next round: no restarts.
    r.allocate_cores({"cpu": {}})
    assert r.reconcile({}) == [] and not STARTED
    # Back off: configured workers again, one restart each.
    r.data["settings"] = {**r.data["settings"], "core_allocation": "off"}
    assert r.allocate_cores({}) == ["CORES: allocation off; every brain goes back to its configured workers"]
    r.reconcile({})
    assert started()["evolve_gp.py"].cmd[3] == "3"


def test_hysteresis_holds_small_changes(roster, monkeypatch):
    data = {
        **ROSTER,
        "families": FAMILIES,
        "settings": {**ROSTER["settings"], "core_allocation": "cost_gap", "training_workers": 9},
    }
    r = roster(data)
    r.workers = {"GP": 5}  # running with 5; the target is 5 +- 1
    monkeypatch.setattr(league, "brain_usd", lambda cpu: ({**OWN, "GP": 3.0}, {**LINEAGE, "GP": 3.0}))
    targets, _ = league.plan_cores(9, FAMILIES, {**OWN, "GP": 3.0}, {**LINEAGE, "GP": 3.0}, list(FAMILIES))
    assert targets["GP"] == 4
    note = r.allocate_cores({})[0]
    assert r.workers["GP"] == 5 and "GP 5 (target 4, kept)" in note


def test_failure_leaves_workers(roster, monkeypatch):
    data = {**ROSTER, "families": FAMILIES, "settings": {**ROSTER["settings"], "core_allocation": "cost_gap",
                                                         "training_workers": 12}}
    r = roster(data)
    r.workers = {"GP": 6}

    def broken(cpu):
        raise OSError("no ledger")

    monkeypatch.setattr(league, "brain_usd", broken)
    assert r.allocate_cores({})[0].startswith("CORES: allocation failed")
    assert r.workers == {"GP": 6}


def test_with_and_configured_workers():
    assert league.with_workers(["x.py", "--workers", "3", "--out", "d"], 7) == ["x.py", "--workers", "7", "--out", "d"]
    assert league.with_workers(["x.py"], 2) == ["x.py", "--workers", "2"]
    assert league.configured_workers("process", {"cmd": ["x.py", "--workers", "4"]}, {}) == 4
    assert league.configured_workers("process", {"cmd": ["x.py"]}, {}) is None
    assert league.configured_workers("learner", {}, {"workers_per_learner": 3}) == 3


def test_plan_target_cores_shares_by_gap_and_freezes_at_target():
    lineage = {"GP-v3": 18.73, "PPO-goals": 8.77, "PPO-v2": 4.53}
    workers, frozen, gaps = league.plan_target_cores(24, lineage, ["GP-v3", "PPO-goals", "PPO-v2"], 20.0)
    assert frozen == [] and sum(workers.values()) == 24
    # One each, the 21 spare by gap (1.27 : 11.23 : 15.47): the PPOs get almost all of it.
    assert workers == {"GP-v3": 2, "PPO-goals": 9, "PPO-v2": 13}
    workers, frozen, gaps = league.plan_target_cores(24, {**lineage, "GP-v3": 20.01}, list(lineage), 20.0)
    assert frozen == ["GP-v3"] and set(workers) == {"PPO-goals", "PPO-v2"} and sum(workers.values()) == 24
    workers, frozen, _ = league.plan_target_cores(24, {n: 21.0 for n in lineage}, list(lineage), 20.0)
    assert workers == {} and frozen == sorted(lineage)
    # A floor keeps a brain training at a minimum; the others share the rest by gap.
    workers, _, _ = league.plan_target_cores(24, lineage, list(lineage), 25.0, {"GP-v3": 6})
    assert workers["GP-v3"] == 6 and sum(workers.values()) == 24
    assert workers["PPO-v2"] > workers["PPO-goals"] > 1
    # A floor below the brain's own share changes nothing.
    assert league.plan_target_cores(24, lineage, list(lineage), 20.0, {"PPO-v2": 2})[0] == {"GP-v3": 2, "PPO-goals": 9, "PPO-v2": 13}


def test_cost_target_freezes_a_brain_into_a_fixed_entrant(roster, monkeypatch, tmp_path):
    (tmp_path / "gp.json").write_text('{"program": 1}')
    (tmp_path / "PPO-goals").mkdir()
    (tmp_path / "PPO-goals" / "policy.npz").write_bytes(b"weights")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        league, "brain_usd", lambda cpu: (dict(OWN), {**LINEAGE, "GP": 20.5, "PPO-goals": 20.0, "PPO-shaped": 10.0})
    )
    data = {
        **ROSTER,
        "families": FAMILIES,
        "settings": {**ROSTER["settings"], "core_allocation": "cost_target", "cost_target_usd": 20, "training_workers": 12},
    }
    r = roster(data)
    r.reconcile({})
    started()
    notes = r.allocate_cores({"cpu": {}})
    assert notes[0].startswith("FROZEN GP: $20.50 reached the $20.00 compute target")
    assert notes[1].startswith("FROZEN PPO-goals")
    assert notes[2] == "CORES (cost target $20.00 per brain, 12 training workers): PPO-shaped $10.00 (gap $10.00) -> 12"
    saved = json.loads((tmp_path / "roster.json").read_text())
    assert saved["fixed_entrants"]["GP"].endswith("frozen/gp.json".replace("gp", "GP"))
    assert saved["fixed_entrants"]["PPO-goals"].endswith("frozen/PPO-goals.npz")
    assert (tmp_path / "frozen" / "PPO-goals.npz").read_bytes() == b"weights"
    assert "GP" not in saved["processes"] and "PPO-goals" not in saved["learners"]
    assert saved["frozen"]["GP"]["from"] == "processes" and saved["frozen"]["GP"]["cost_usd"] == 20.5
    changes = r.reconcile({})
    assert "ROSTER: stopped GP (removed from the roster)" in changes
    assert "ROSTER: stopped PPO-goals (removed from the roster)" in changes
    assert r.workers == {"PPO-shaped": 12}
