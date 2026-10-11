"""
A fixed yardstick for the league: every round's Swiss winner and the champion of the
time play DefendersAndAttackers (DAA), a strong hand-written brain that nothing in the
league trains against. Fitness is the average goal difference per 9000 ticks against
DAA (DAA itself scores 0; positive is better than DAA), over the same kick-offs for
every brain, with a 95% CI. League 1's final champion is measured once as a reference.

Watches the league and measures new rounds as they are saved; results go to
<league>/benchmark.json for the dashboard. Runs at low priority.

    LEAGUE_DIR=runs/league2 nice -n 10 poetry run python benchmark.py
"""

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import leaguefiles as lf
from aisoccer.brainspec import load_brain
from aisoccer.constants import Constants
from aisoccer.vecgame import VecGame

BENCHMARK = "DefendersAndAttackers"
TICKS = 9000
# The rules a measurement was taken under: results only carry over between identical rules.
RULES = f"goal{Constants.GOAL_MOUTH_HEIGHT}-speed{Constants.SPEED_SCALE:g}x{Constants.SUBSTEPS}"
REFERENCES = {}  # no outside reference line
HIDDEN = {3, 4} if "league2" in str(lf.LEAGUE_DIR) else set()  # league 2 rounds left out of its blog


def play_batch(tasks):
    """(spec, seed) games against DAA in one VecGame; the brain plays blue on even seeds."""
    blue, red = [], []
    for spec, seed in tasks:
        brain, daa = load_brain(spec, "brain"), load_brain(BENCHMARK)
        blue.append(brain if seed % 2 == 0 else daa)
        red.append(daa if seed % 2 == 0 else brain)
    scores = VecGame(blue, red, [seed for _, seed in tasks], game_length=TICKS).play()
    return [
        (s["blue"] - s["red"]) if seed % 2 == 0 else (s["red"] - s["blue"])
        for (_, seed), s in zip(tasks, scores)
    ]


def measure(workers, spec, games, batch=6):
    tasks = [(spec, seed) for seed in range(games)]
    chunks = [tasks[k : k + batch] for k in range(0, games, batch)]  # noqa: E203
    gd = np.array([g for out in workers.map(play_batch, chunks, chunksize=1) for g in out], float)
    return {"gd": float(gd.mean()), "gd_ci": float(1.98 * gd.std(ddof=1) / np.sqrt(len(gd))), "n": len(gd)}


def snapshot(name, number):
    """The file a brain played round `number` with (league snapshots), or None."""
    for suffix in (".npz", ".json"):
        path = lf.LEAGUE_DIR / "snapshots" / f"{name}-r{number:03d}{suffix}"
        if path.exists():
            return str(path)
    return None


def roster_spec(name):
    """A spec-only entrant's brain spec (e.g. "brain:aisoccer.llm.v2.LLMBrainV2") from the roster."""
    roster = json.loads((lf.LEAGUE_DIR / "roster.json").read_text())
    if name in roster.get("fixed_entrants", {}):
        return roster["fixed_entrants"][name]
    return roster.get("retired", {}).get(name, {}).get("spec")


def champion_file(name):
    matches = list(lf.HISTORY_DIR.glob(f"{name}.*")) if name else []
    return str(matches[0]) if matches else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--games", type=int, default=120)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--every", type=float, default=60, help="seconds between checks")
    args = parser.parse_args()
    out_file = lf.LEAGUE_DIR / "benchmark.json"
    data = json.loads(out_file.read_text()) if out_file.exists() else {"specs": {}, "rounds": {}, "references": {}}

    def save():
        tmp = out_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        os.replace(tmp, out_file)

    with Pool(args.workers) as workers:

        def fitness(spec):
            key = f"{spec}@{RULES}"
            if key not in data["specs"]:
                data["specs"][key] = dict(measure(workers, spec, args.games), rules=RULES)
                save()
            return data["specs"][key]

        while True:
            for label, spec in REFERENCES.items():
                if label not in data["references"] and Path(spec).exists():
                    data["references"][label] = fitness(spec)
                    save()
            rounds = {
                r["round"]: r
                for r in (json.loads(f.read_text()) for f in sorted((lf.LEAGUE_DIR / "rounds").glob("round-*.json")))
            }
            for number in sorted(rounds):
                if number in HIDDEN or str(number) in data["rounds"]:
                    continue
                results = rounds[number]
                winner = results["standings"][0]["name"]
                winner_spec = snapshot(winner, number) or champion_file(winner) or roster_spec(winner)
                champ_spec = champion_file(results["champion"])
                entry = {"winner": winner, "champion": results["champion"], "rules": RULES}
                if winner_spec:
                    entry["winner_fitness"] = fitness(winner_spec)
                if champ_spec:
                    entry["champion_fitness"] = fitness(champ_spec)
                data["rounds"][str(number)] = entry
                save()
                print(f"round {number}: {winner} {entry.get('winner_fitness')}, "
                      f"champion {results['champion']} {entry.get('champion_fitness')}", flush=True)
            time.sleep(args.every)


if __name__ == "__main__":
    main()
