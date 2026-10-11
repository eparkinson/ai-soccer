"""
The coach: picks a team of five players from the league's recent players, of any kind.

Role i of the team plays as player i of the brain chosen for that role (a network, an
LLMBrain version, a GP program...; see aisoccer/brains/CompositeBrain.py). Roles name
brains, not frozen copies: every new league round, each role switches to its brain's
latest snapshot, so the team keeps up as its players improve.

Players are chosen by statistics. Every new round the coach scouts: each recent brain
plays --scout-games recorded games (against the champion and other recent brains), and
every player is rated by the value of its touches (aisoccer/player_value.py: how much
each touch changes the team's chance of scoring next). The ratings go to
player_ratings.json. The coach then proposes, role by role, the best-rated player for
that role, biggest expected gain first; a swap is kept only if the new team beats the
current team head to head over --games games (by more than one standard error, on
shared kick-offs), since a player's value can depend on the teammates it is used to.

The team is written to runs/league/coach/team-spec.json (a brain spec) and plays in every
league round as the entrant "Coach"; team.json lists the role brains for the league log.

    poetry run python coach.py --workers 2
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from aisoccer import player_value
from aisoccer.brainspec import load_brain
from aisoccer.game import Game
from league import HISTORY_DIR, LEAGUE_DIR, record, record_showcase

COACH_DIR = LEAGUE_DIR / "coach"
LOG_FILE = COACH_DIR / "progress.log"
STATE_FILE = COACH_DIR / "state.json"
SPEC_FILE = COACH_DIR / "team-spec.json"
TICKS = 9000
TOP = 8  # swaps are drawn from the latest round's top brains


def log(message=""):
    print(message, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")


def recent_players():
    """The latest round: its number and {name: spec} for its players, best first."""
    rounds = sorted((LEAGUE_DIR / "rounds").glob("round-*.json"))
    if not rounds:
        return None, {}
    results = json.loads(rounds[-1].read_text())
    number = results["round"]
    roster = json.loads((LEAGUE_DIR / "roster.json").read_text())
    players = {}
    for row in results["standings"]:
        name = row["name"]
        if name == "Coach" or "champ-" in name:
            continue  # not itself, and past champions are old snapshots, not recent players
        snaps = sorted((LEAGUE_DIR / "snapshots").glob(f"{name}-r{number:03d}.*"))
        spec = (  # a snapshot, or a spec-only brain's spec (also if it was retired since)
            str(snaps[0]) if snaps
            else roster.get("fixed_entrants", {}).get(name) or roster.get("retired", {}).get(name, {}).get("spec")
        )
        if spec:
            players[name] = spec
    return number, players


def team_spec(roles, players):
    return {"class": "aisoccer.brains.CompositeBrain.CompositeBrain", "kwargs": {"roles": [players[n] for n in roles]}}


def write_team(roles, players, path=SPEC_FILE):
    """Write atomically: league.py may copy the team file at any time."""
    tmp = path.with_name(path.stem + ".tmp.json")
    tmp.write_text(json.dumps(team_spec(roles, players)))
    os.replace(tmp, path)
    if path == SPEC_FILE:
        (COACH_DIR / "team.json").write_text(json.dumps(roles))


def play(task):
    candidate, current, candidate_blue, seed = task
    a, b = load_brain(candidate, "candidate"), load_brain(current, "current")
    blue, red = (a, b) if candidate_blue else (b, a)
    score = Game(blue, red, game_length=TICKS, quiet_mode=True, seed=seed).play()
    return (score["blue"], score["red"]) if candidate_blue else (score["red"], score["blue"])


def scout(workers, players, number, games, rng):
    """Record games of every recent brain and rate each brain's five players by role."""
    champions = sorted(HISTORY_DIR.glob("*champ-*.*"), key=lambda p: int(p.stem.split("-")[-1]))
    champion = str(champions[-1]) if champions else None
    names = list(players)
    tasks = []
    for name in names:
        for g in range(games):
            if champion and g < games // 2:
                other_name, other = "champion", champion
            else:
                other_name = names[int(rng.integers(len(names)))]
                other = players[other_name]
            seed = int(rng.integers(2**31))
            blue = g % 2 == 0
            tasks.append(
                (name, players[name], other_name, other, seed, TICKS) if blue
                else (other_name, other, name, players[name], seed, TICKS)
            )
    recorded = workers.map(record_showcase, tasks, chunksize=1)
    value_map = player_value.fit_value_map([(r["positions"], r["scores"]) for r in recorded])
    rated = [
        (r["names"][k], side, r["positions"], r["scores"])
        for r in recorded for k, side in enumerate(("blue", "red"))
        if r["names"][k] in players
    ]
    ratings = player_value.ratings(rated, value_map)
    out = {
        "round": number,
        "games": len(recorded),
        "value_map": value_map.round(3).tolist(),
        "ratings": {n: [round(float(v), 3) for v in r] for n, r in ratings.items()},
        # Football statistics per role per 9000 ticks: goals, assists, shots on target,
        # saves, own goals (for the reports; selection uses the touch value above).
        "stats": player_value.event_stats(rated),
        # Where each player usually is: its mean distance up the pitch from its own goal (0-1).
        "position": average_positions(rated),
    }
    (COACH_DIR / "player_ratings.json").write_text(json.dumps(out, indent=1))
    return ratings


def average_positions(rated):
    sums, counts = {}, {}
    for name, side, positions, _ in rated:
        x = player_value.perspective(positions, side)[:, :5, 0].mean(axis=0) / player_value.L
        sums[name] = sums.get(name, 0) + x
        counts[name] = counts.get(name, 0) + 1
    return {n: [round(float(v), 2) for v in sums[n] / counts[n]] for n in sums}


def proposals(roles, ratings):
    """Swaps worth testing: the best-rated player for each role, biggest expected gain first."""
    out = []
    for role in range(5):
        best = max(ratings, key=lambda n: ratings[n][role])
        current = ratings.get(roles[role], [0.0] * 5)[role]
        gain = ratings[best][role] - current
        if best != roles[role] and gain > 0:
            out.append((gain, role, best))
    return sorted(out, reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--games", type=int, default=48, help="games per proposed swap")
    parser.add_argument("--scout-games", type=int, default=8, help="recorded games per brain per round")
    args = parser.parse_args()
    COACH_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(time.time()))

    number, players = recent_players()
    while not players:  # before the first saved round
        time.sleep(60)
        number, players = recent_players()
    state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    if state.get("roles") and all(n in players for n in state["roles"]):
        roles, proposal = state["roles"], state.get("proposal", 0)
        log(f"Coach resumed with team {roles}")
    else:
        roles, proposal = [next(iter(players))] * 5, state.get("proposal", 0)
        log(f"Coach started from round {number}'s top brain in every role: {roles}")
    write_team(roles, players)

    def save_state():
        STATE_FILE.write_text(json.dumps({"roles": roles, "proposal": proposal, "round": number}))

    save_state()
    with Pool(args.workers) as workers:
        scouted = None
        queue = []
        while True:
            latest, players = recent_players()
            if latest != number:  # a new round: every role moves to its brain's latest snapshot
                number = latest
                best = next(iter(players))
                gone = [n for n in roles if n not in players]
                roles = [n if n in players else best for n in roles]
                write_team(roles, players)
                log(f"round {number}: team refreshed to the latest snapshots"
                    + (f" (replaced {gone} with {best})" if gone else "") + f": {roles}")
                save_state()
            if scouted != number:
                ratings = scout(workers, players, number, args.scout_games, rng)
                scouted = number
                best_per_role = [max(ratings, key=lambda n: ratings[n][r]) for r in range(5)]
                log(f"round {number} scouting: best player per role "
                    + ", ".join(f"{r}: {n} {ratings[n][r]:+.2f}" for r, n in enumerate(best_per_role)))
                queue = proposals(roles, ratings)
            if not queue:
                time.sleep(60)  # nothing worth testing until the next round's scouting
                continue
            gain, role, source = queue.pop(0)
            candidate = list(roles)
            candidate[role] = source
            candidate_file = COACH_DIR / "candidate.json"
            write_team(candidate, players, candidate_file)
            proposal += 1
            seeds = np.random.default_rng(proposal).integers(2**31, size=args.games)
            tasks = [(str(candidate_file), str(SPEC_FILE), g % 2 == 0, int(seeds[g])) for g in range(args.games)]
            result = record(workers.map(play, tasks, chunksize=2))
            kept = result["gd"] - result["gd_ci"] / 1.96 > 0
            log(f"proposal {proposal:4d}: role {role} <- {source:<16} (rating gain {gain:+.2f}) vs current team "
                f"GD {result['gd']:+.2f} ±{result['gd_ci']:.2f}  {'KEPT' if kept else 'rejected'}")
            if kept:
                roles = candidate
                write_team(roles, players)
                log(f"  team now: {roles}")
                queue = proposals(roles, ratings)
            save_state()


if __name__ == "__main__":
    main()
