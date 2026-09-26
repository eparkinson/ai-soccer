"""
The coach: builds a mixed PPOBrain team, one trained brain per player role.

Every player role (0-4) can use the policy of a different trained brain: the league
learners, the champions and earlier saved PPOBrain versions. The coach starts from the
current champion and keeps proposing swaps ("play role 3 with learner C's policy"). A
swap is kept only if the new team beats the current team head to head over --games
games (by more than one standard error, on shared kick-offs).

The current team is written to runs/league/coach/team.npz, a normal PPOBrain weights
file, and plays in every league round as the entrant "Coach". league.py starts, pauses
and stops this process alongside the learners.

    poetry run python coach.py --workers 4
"""

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from aisoccer.brains.PPOBrain import PPOBrain
from league import HISTORY_DIR, LEAGUE_DIR, play, record

COACH_DIR = LEAGUE_DIR / "coach"
LOG_FILE = COACH_DIR / "progress.log"


def log(message=""):
    print(message, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")


def sources():
    """Brains whose role policies the coach can use: label -> weights path."""
    found = {}
    for path in sorted(LEAGUE_DIR.glob("*/policy.npz")):
        found[f"learner {path.parent.name}"] = path
    for path in sorted(HISTORY_DIR.glob("*.npz")):
        weights = PPOBrain.load_weights(path)
        if (
            weights.get("action_repeat", 2) == PPOBrain.ACTION_REPEAT
            and "role_policies" not in weights
        ):
            found[path.stem] = path
    return found


def build(team):
    """team: list of (label, policy params) per role -> PPOBrain weights dict."""
    return {
        "policy": team[0][1],
        "log_std": np.full(2, -1.0),
        "role_policies": [params for _, params in team],
    }


def save_team(team, path):
    """Write atomically: league.py may copy the team file at any time."""
    tmp = Path(path).with_name(Path(path).stem + ".tmp.npz")
    PPOBrain.save_weights(tmp, build(team))
    os.replace(tmp, path)


def head_to_head(workers, candidate_path, current_path, games, seed):
    seeds = np.random.default_rng(seed).integers(2**31, size=games)
    tasks = [
        (str(candidate_path), str(current_path), g % 2 == 0, int(seeds[g]))
        for g in range(games)
    ]
    return record(workers.map(play, tasks, chunksize=4))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--games", type=int, default=200, help="games per proposed swap"
    )
    parser.add_argument(
        "--start", type=Path, help="team to start from (default: newest champion)"
    )
    args = parser.parse_args()

    COACH_DIR.mkdir(parents=True, exist_ok=True)
    team_file = COACH_DIR / "team.npz"
    labels_file = COACH_DIR / "team.json"
    rng = np.random.default_rng(int(time.time()))

    if team_file.exists() and labels_file.exists():
        weights = PPOBrain.load_weights(team_file)
        labels = json.loads(labels_file.read_text())
        team = list(zip(labels, weights["role_policies"]))
        log(f"Coach resumed with team {labels}")
    else:
        champions = sorted(
            HISTORY_DIR.glob("PPO-champ-*.npz"),
            key=lambda p: int(p.stem.split("-")[-1]),
        )
        start = args.start or (champions[-1] if champions else None)
        while start is None:  # a league starting from zero: wait for a learner's network
            found = sources()
            start = Path(next(iter(found.values()))) if found else None
            if start is None:
                time.sleep(30)
        policy = PPOBrain.load_weights(start)["policy"]
        start_label = f"learner {start.parent.name}" if start.stem == "policy" else start.stem
        team = [(start_label, policy)] * 5
        save_team(team, team_file)
        labels_file.write_text(json.dumps([label for label, _ in team]))
        log(f"Coach started from {start_label} in every role")

    proposal = 0
    with Pool(args.workers) as workers:
        while True:
            available = sources()
            role = int(rng.integers(5))
            options = [s for s in available if s != team[role][0]]
            if not options:
                time.sleep(60)
                continue
            source = options[int(rng.integers(len(options)))]
            params = PPOBrain.load_weights(available[source])["policy"]
            candidate = list(team)
            candidate[role] = (source, params)
            candidate_file = COACH_DIR / "candidate.npz"
            save_team(candidate, candidate_file)

            proposal += 1
            result = head_to_head(
                workers, candidate_file, team_file, args.games, seed=proposal
            )
            se = result["gd_ci"] / 1.96
            kept = result["gd"] - se > 0
            log(
                f"proposal {proposal:3d}: role {role} <- {source:<16} vs current team "
                f"GD {result['gd']:+.2f} ±{result['gd_ci']:.2f}  {'KEPT' if kept else 'rejected'}"
            )
            if kept:
                team = candidate
                save_team(team, team_file)
                labels_file.write_text(json.dumps([label for label, _ in team]))
                log(f"  team now: {[label for label, _ in team]}")


if __name__ == "__main__":
    main()
