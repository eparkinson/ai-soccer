"""
Diagnostics for improving LLMBrain: how the current version plays against the
league's leaders, and how every goal happened.

    LEAGUE_DIR=runs/league2 poetry run python llm_report.py brain:aisoccer.llm.v1.LLMBrainV1

Plays --matches full-length matches against each of the top --opponents brains of the
latest round (plus the champion), with team statistics, and records a few matches in
full to analyse every goal: where the ball came from, whether our goal was empty, how
far away our nearest defender was and how many of our players were caught upfield.
Writes a report to <league>/llm/report-<name>.txt.
"""

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import leaguefiles as lf
from aisoccer.brainspec import load_brain
from aisoccer.constants import Constants
from aisoccer.game import Game, GameResult
from aisoccer.stats import play_with_stats

L, H = Constants.FIELD_LENGTH - 1, Constants.FIELD_HEIGHT
MATCH = 9000


def play(task):
    """(ours, theirs, we play blue, seed, record) -> result dict."""
    ours, theirs, we_blue, seed, record = task
    a, b = load_brain(ours, "LLM"), load_brain(theirs)
    blue, red = (a, b) if we_blue else (b, a)
    game = Game(blue, red, game_length=MATCH, quiet_mode=True, seed=seed)
    me, them = ("blue", "red") if we_blue else ("red", "blue")
    if not record:
        score, stats = play_with_stats(game)
        return {"gf": score[me], "ga": score[them], "stats": stats[me], "their_stats": stats[them]}
    # Record positions to analyse every goal (players 0-4 blue, 5-9 red, 10 ball).
    frames, goals = [], []
    while True:
        result = game.tick()
        if result == GameResult.end:
            break
        if result in (GameResult.goal_blue, GameResult.goal_red):
            scorer = "blue" if result == GameResult.goal_blue else "red"
            goals.append((len(frames), scorer))
        bodies = [p.body.position for team in game.teams for p in team.players]
        frames.append(np.array(bodies + [game.ball.body.position]))
    frames = np.array(frames)
    ours_idx = slice(0, 5) if we_blue else slice(5, 10)
    own_goal_x = 0.0 if we_blue else L
    events = []
    for tick, scorer in goals:
        before = frames[max(tick - 40, 0)]  # about when the decisive touch happened
        ball_then = before[10]
        our_players = before[ours_idx]
        from_goal = lambda x: abs(x - own_goal_x)  # noqa: E731
        conceded = scorer != me
        attacking_goal_x = L - own_goal_x
        events.append(
            {
                "conceded": conceded,
                "shot_distance": float(abs(ball_then[0] - (own_goal_x if conceded else attacking_goal_x))),
                "shot_height": float(ball_then[1]),
                "goal_empty": bool(np.min(np.abs(our_players[:, 0] - own_goal_x)) > 200) if conceded else None,
                "nearest_defender": float(np.min(np.linalg.norm(our_players - ball_then, axis=1))),
                "players_caught_upfield": int(np.sum(from_goal(our_players[:, 0]) > from_goal(ball_then[0]))),
            }
        )
    score = game.score
    return {"gf": score[me], "ga": score[them], "events": events}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("brain", help="brain spec of the version to analyse")
    parser.add_argument("--opponents", type=int, default=3)
    parser.add_argument("--matches", type=int, default=8, help="per opponent")
    parser.add_argument("--recorded", type=int, default=2, help="recorded matches per opponent")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    results = lf.round_results()
    opponents = []
    if results:
        table = lf.football_table(results)
        names = [row["name"] for row in table if not row["name"].startswith("LLM")][: args.opponents]
        opponents = [(n, lf.brain_spec(n, results["round"])) for n in names]
    champion = lf.champion()
    if champion and all(spec != str(champion) for _, spec in opponents):
        opponents.append((champion.stem, str(champion)))

    tasks, keys = [], []
    for name, spec in opponents:
        for g in range(args.matches + args.recorded):
            tasks.append((args.brain, spec, g % 2 == 0, 7000 + g, g >= args.matches))
            keys.append(name)
    with Pool(args.workers) as pool:
        outcomes = pool.map(play, tasks)

    lines = [f"LLMBrain diagnostics: {args.brain}", ""]
    all_stats, events = [], []
    for name, _ in opponents:
        mine = [o for k, o in zip(keys, outcomes) if k == name]
        gf, ga = sum(o["gf"] for o in mine), sum(o["ga"] for o in mine)
        wins = sum(o["gf"] > o["ga"] for o in mine)
        draws = sum(o["gf"] == o["ga"] for o in mine)
        lines.append(
            f"vs {name:<16} W-D-L {wins}-{draws}-{len(mine) - wins - draws}  "
            f"goals {gf}-{ga} ({gf / len(mine):.1f}-{ga / len(mine):.1f} per match)"
        )
        all_stats += [(o["stats"], o["their_stats"]) for o in mine if "stats" in o]
        events += [e for o in mine for e in o.get("events", [])]
    lines.append("")
    if all_stats:
        keys_ = ["control", "possession", "own_half", "final_third", "shots", "spread", "passes", "turnovers_won", "distance_run"]
        lines.append("Team statistics per match (ours / theirs):")
        for k in keys_:
            ours = np.mean([s[0][k] for s in all_stats])
            theirs = np.mean([s[1][k] for s in all_stats])
            lines.append(f"  {k:<14} {ours:10.2f} / {theirs:10.2f}")
        lines.append("")
    for label, conceded in (("Goals conceded", True), ("Goals scored", False)):
        chosen = [e for e in events if e["conceded"] == conceded]
        if not chosen:
            lines.append(f"{label}: none in the recorded matches")
            continue
        lines.append(f"{label} ({len(chosen)} in the recorded matches):")
        lines.append(f"  decisive touch, distance from goal: median {np.median([e['shot_distance'] for e in chosen]):.0f} px")
        if conceded:
            lines.append(f"  our goal empty: {np.mean([e['goal_empty'] for e in chosen]):.0%}")
        lines.append(f"  nearest of our players to the ball: median {np.median([e['nearest_defender'] for e in chosen]):.0f} px")
        lines.append(
            f"  our players further upfield than the ball: mean {np.mean([e['players_caught_upfield'] for e in chosen]):.1f} of 5"
        )
    report = "\n".join(lines)
    out = lf.LEAGUE_DIR / "llm"
    out.mkdir(parents=True, exist_ok=True)
    name = args.brain.rsplit(".", 1)[-1]
    (out / f"report-{name}.txt").write_text(report)
    (out / f"report-{name}.json").write_text(json.dumps({"brain": args.brain, "events": events}, default=float))
    print(report)


if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
